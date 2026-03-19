"""
Utilities for the parentness + per-parent count experiment.

This experiment lives in `unmerge/count_test_parentness/` and is designed to be
consistent with the `unmerge/` refactor:
- Use the same HLT simulation knobs (threshold, merge radius, smearing, efficiency loss).
- Use the same 7D engineered features for token inputs.
- Provide robust experiment I/O (config + checkpoints + plots).

Labels:
- group_size[j,i] is the merged cluster size for HLT token i (constructed during merge step).
- parent_gt[j,i] = 1 if token i survives HLT and has group_size>1.
- k_true[j,i]    = group_size-1 (missing count contributed by this merged parent) if parent_gt=1 else 0.
  (Optionally, use group_size instead; controlled by config.)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# -----------------------------
# Simple experiment I/O helpers
# -----------------------------


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_config(config: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return p


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, p.as_posix())
    return p


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    p = Path(path)
    payload = torch.load(p.as_posix(), map_location=map_location)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state, strict=bool(strict))
    return payload if isinstance(payload, dict) else {"state_dict": state}


# -----------------------------
# Preprocessing (HLT effects + features)
# -----------------------------


def wrap_dphi(dphi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def apply_hlt_effects_with_groups(
    const: np.ndarray,
    mask: np.ndarray,
    config: dict,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply HLT effects and return merge groups.

    Returns:
      hlt: [N,S,4]
      hlt_mask: [N,S] bool
      off_mask: [N,S] bool (offline threshold)
      group_size: [N,S] int16, size of merged group for each surviving token (0 for masked tokens)
    """
    np.random.seed(int(seed))
    cfg = config["hlt_effects"]
    n_jets, S, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    # Offline reference mask (before HLT threshold). Uses original pt.
    pt_thr_off = float(cfg["pt_threshold_offline"])
    off_mask = hlt_mask & (hlt[:, :, 0] >= pt_thr_off)

    # HLT threshold
    pt_thr_hlt = float(cfg["pt_threshold_hlt"])
    below = (hlt[:, :, 0] < pt_thr_hlt) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0

    # IMPORTANT: do NOT store python list-of-lists merge groups here.
    # It's extremely memory-heavy for large N,S. For count-only supervision we only need group sizes.
    group_size = np.zeros((n_jets, S), dtype=np.int16)

    if bool(cfg["merge_enabled"]) and float(cfg["merge_radius"]) > 0:
        r = float(cfg["merge_radius"])
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            for ii in valid_idx:
                group_size[jet_idx, int(ii)] = 1

            if len(valid_idx) >= 2:
                to_remove = set()
                for a in range(len(valid_idx)):
                    i = int(valid_idx[a])
                    if i in to_remove:
                        continue
                    for b in range(a + 1, len(valid_idx)):
                        j = int(valid_idx[b])
                        if j in to_remove:
                            continue
                        deta = float(hlt[jet_idx, i, 1] - hlt[jet_idx, j, 1])
                        dphi = float(wrap_dphi(hlt[jet_idx, i, 2] - hlt[jet_idx, j, 2]))
                        dR = math.sqrt(deta * deta + dphi * dphi)
                        if dR < r:
                            pt_i = float(hlt[jet_idx, i, 0])
                            pt_j = float(hlt[jet_idx, j, 0])
                            pt_sum = pt_i + pt_j
                            if pt_sum < 1e-6:
                                continue
                            # i absorbs j
                            group_size[jet_idx, i] = int(group_size[jet_idx, i] + group_size[jet_idx, j])
                            group_size[jet_idx, j] = 0
                            # pT-weighted merge
                            w_i, w_j = pt_i / pt_sum, pt_j / pt_sum
                            hlt[jet_idx, i, 0] = pt_sum
                            hlt[jet_idx, i, 1] = w_i * hlt[jet_idx, i, 1] + w_j * hlt[jet_idx, j, 1]
                            phi_i, phi_j = float(hlt[jet_idx, i, 2]), float(hlt[jet_idx, j, 2])
                            hlt[jet_idx, i, 2] = math.atan2(
                                w_i * math.sin(phi_i) + w_j * math.sin(phi_j),
                                w_i * math.cos(phi_i) + w_j * math.cos(phi_j),
                            )
                            hlt[jet_idx, i, 3] = float(hlt[jet_idx, i, 3]) + float(hlt[jet_idx, j, 3])
                            to_remove.add(j)
                for idx in to_remove:
                    hlt_mask[jet_idx, idx] = False
                    hlt[jet_idx, idx] = 0
    else:
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            for ii in valid_idx:
                group_size[jet_idx, int(ii)] = 1

    # Smearing
    valid = hlt_mask.copy()
    pt_noise = np.clip(
        np.random.normal(1.0, float(cfg.get("pt_resolution", 0.0)), (n_jets, S)), 0.5, 1.5
    )
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)
    eta_noise = np.random.normal(0, float(cfg.get("eta_resolution", 0.0)), (n_jets, S))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)
    phi_noise = np.random.normal(0, float(cfg.get("phi_resolution", 0.0)), (n_jets, S))
    new_phi = hlt[:, :, 2] + phi_noise
    hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)
    # Recompute E from (pt, eta) (massless)
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)

    # Efficiency loss (randomly drop surviving tokens)
    eff = float(cfg.get("efficiency_loss", 0.0))
    if eff > 0:
        lost = (np.random.random((n_jets, S)) < eff) & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0
        group_size[lost] = 0

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    return hlt, hlt_mask, off_mask, group_size


def compute_features(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute 7 relative features (mirrors baseline)."""
    pt = np.maximum(const[:, :, 0], 1e-8)
    eta = np.clip(const[:, :, 1], -5, 5)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], 1e-8)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    m = mask.astype(float)
    jet_px = (px * m).sum(axis=1, keepdims=True)
    jet_py = (py * m).sum(axis=1, keepdims=True)
    jet_pz = (pz * m).sum(axis=1, keepdims=True)
    jet_E = (E * m).sum(axis=1, keepdims=True)
    jet_pt = np.sqrt(jet_px**2 + jet_py**2) + 1e-8
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)
    dEta = eta - jet_eta
    dPhi = np.arctan2(np.sin(phi - jet_phi), np.cos(phi - jet_phi))
    feats = np.stack(
        [
            dEta,
            dPhi,
            np.log(pt + 1e-8),
            np.log(E + 1e-8),
            np.log(pt / jet_pt + 1e-8),
            np.log(E / (jet_E + 1e-8) + 1e-8),
            np.sqrt(dEta**2 + dPhi**2),
        ],
        axis=-1,
    )
    feats = np.clip(np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), -20, 20)
    feats[~mask] = 0
    return feats.astype(np.float32)


def get_stats(feat: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.zeros(feat.shape[-1], dtype=np.float64)
    stds = np.zeros(feat.shape[-1], dtype=np.float64)
    for i in range(feat.shape[-1]):
        vals = feat[idx][:, :, i][mask[idx]]
        means[i] = float(np.nanmean(vals))
        stds[i] = float(np.nanstd(vals) + 1e-8)
    return means.astype(np.float32), stds.astype(np.float32)


def standardize(
    feat: np.ndarray,
    mask: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    clip: float = 10.0,
) -> np.ndarray:
    out = (feat - means[None, None, :]) / stds[None, None, :]
    out = np.clip(out, -float(clip), float(clip))
    out = np.nan_to_num(out, 0.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


# -----------------------------
# Targets + dataset
# -----------------------------


def build_parent_targets(
    hlt_mask: np.ndarray,
    group_size: np.ndarray,
    *,
    count_kind: str = "missing",  # "missing" (group_size-1) or "group_size"
    max_k: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-token parentness and count targets.

    Returns:
      parent_gt: [N,S] float32 (0/1)
      k_true:    [N,S] float32 (>=0)
      jet_sum:   [N]   float32 (sum over tokens of k_true)
    """
    N, S = hlt_mask.shape
    parent_gt = np.zeros((N, S), dtype=np.float32)
    k_true = np.zeros((N, S), dtype=np.float32)
    jet_sum = np.zeros((N,), dtype=np.float32)

    kind = str(count_kind).lower()
    # group_size already encodes merge multiplicity for surviving tokens (0 for masked tokens).
    gs = group_size.astype(np.int32, copy=False)
    parent = (hlt_mask & (gs > 1))
    parent_gt[parent] = 1.0
    if kind in ("missing", "delta", "deltan"):
        kk = np.maximum(gs - 1, 0)
    else:
        kk = np.maximum(gs, 0)
    if max_k is not None:
        kk = np.minimum(kk, int(max_k))
    k_true = kk.astype(np.float32)
    k_true[~hlt_mask] = 0.0
    jet_sum = k_true.sum(axis=1).astype(np.float32)
    return parent_gt, k_true, jet_sum


class JetParentCountDataset(Dataset):
    """
    One sample = one jet (token sequence).

    Returns:
      - hlt: [S,D]
      - mask_hlt: [S]
      - parent_gt: [S]
      - k_true: [S]
      - jet_sum_true: scalar
    """

    def __init__(
        self,
        jet_indices: Sequence[int],
        feat_hlt_std: np.ndarray,
        mask_hlt: np.ndarray,
        parent_gt: np.ndarray,
        k_true: np.ndarray,
        jet_sum_true: np.ndarray,
    ):
        self.jet_idx = np.asarray(jet_indices, dtype=np.int64)
        self.hlt = torch.tensor(feat_hlt_std, dtype=torch.float32)
        self.mask = torch.tensor(mask_hlt, dtype=torch.bool)
        self.parent_gt = torch.tensor(parent_gt, dtype=torch.float32)
        self.k_true = torch.tensor(k_true, dtype=torch.float32)
        self.jet_sum_true = torch.tensor(jet_sum_true, dtype=torch.float32)

    def __len__(self):
        return int(self.jet_idx.shape[0])

    def __getitem__(self, i):
        j = int(self.jet_idx[i])
        return {
            "hlt": self.hlt[j],
            "mask_hlt": self.mask[j],
            "parent_gt": self.parent_gt[j],
            "k_true": self.k_true[j],
            "jet_sum_true": self.jet_sum_true[j],
        }


# -----------------------------
# Training / evaluation helpers
# -----------------------------


@dataclass
class TrainCfg:
    batch_size: int = 256
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 3
    patience: int = 8
    min_delta: float = 0.0
    grad_clip: float = 1.0

    warmup_parent_epochs: int = 3  # only parentness loss
    thr_parent: float = 0.5

    # Loss weights
    w_parent: float = 1.0
    w_k: float = 1.0
    w_sum: float = 0.2

    # Count loss tuning
    huber_delta: float = 1.0
    neg_k_weight: float = 0.2  # weight on non-parent tokens for count regression


def get_scheduler(opt, warmup: int, total: int):
    def lr_lambda(ep):
        if ep < int(warmup):
            return float(ep + 1) / float(max(1, warmup))
        return 0.5 * (1.0 + np.cos(np.pi * (ep - warmup) / float(max(1, total - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, *, pos_weight: float = 1.0
) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return logits.new_tensor(0.0)
    logits_v = logits[mask]
    targets_v = targets[mask]
    pw = logits.new_tensor(float(pos_weight))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    return loss_fn(logits_v, targets_v)


def _masked_weighted_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return pred.new_tensor(0.0)
    pred_v = pred[mask]
    target_v = target[mask]
    w_v = torch.clamp(w[mask], min=0.0)
    loss = F.smooth_l1_loss(pred_v, target_v, reduction="none", beta=float(delta))
    return (loss * w_v).sum() / (w_v.sum() + 1e-8)


@torch.no_grad()
def eval_parentness_and_count(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    pos_weight: float,
    thr_parent: float,
    huber_delta: float,
    neg_k_weight: float,
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    tot_parent_loss = 0.0
    tot_k_loss = 0.0
    tot_sum_mae = 0.0
    n_batches = 0
    n_jets = 0
    n_parent_tokens = 0
    mae_k_on_pos_sum = 0.0
    mae_k_on_pos_n = 0

    for batch in loader:
        x = batch["hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        y_parent = batch["parent_gt"].to(device)
        y_k = batch["k_true"].to(device)
        y_sum = batch["jet_sum_true"].to(device)

        out = model(x, m)
        parent_logit = out.parent_logit
        k_pred = out.k_pred

        parent_loss = _masked_bce_with_logits(parent_logit, y_parent, m, pos_weight=float(pos_weight))
        tot_parent_loss += float(parent_loss.item())

        # Count loss: weight non-parent tokens lightly (helps prevent arbitrary k on negatives).
        w = torch.where(y_parent > 0.5, torch.ones_like(y_parent), y_parent.new_full(y_parent.shape, float(neg_k_weight)))
        k_loss = _masked_weighted_huber(k_pred, y_k, m, w, delta=float(huber_delta))
        tot_k_loss += float(k_loss.item())

        prob = torch.sigmoid(parent_logit)
        pred_parent = (prob > float(thr_parent)) & m
        gt_parent = (y_parent > 0.5) & m
        tp += int((pred_parent & gt_parent).sum().item())
        fp += int((pred_parent & (~gt_parent)).sum().item())
        fn += int(((~pred_parent) & gt_parent).sum().item())

        pred_sum = k_pred.sum(dim=1)
        tot_sum_mae += float((pred_sum - y_sum).abs().mean().item())

        # Token-level MAE on positive (merged) parents only.
        pos_mask = gt_parent
        if int(pos_mask.sum().item()) > 0:
            mae_k_on_pos_sum += float((k_pred[pos_mask] - y_k[pos_mask]).abs().mean().item())
            mae_k_on_pos_n += 1

        n_batches += 1
        n_jets += int(x.shape[0])
        n_parent_tokens += int(gt_parent.sum().item())

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    return {
        "parent_loss": tot_parent_loss / max(1, n_batches),
        "k_loss": tot_k_loss / max(1, n_batches),
        "precision": float(prec),
        "recall": float(rec),
        "sum_mae": tot_sum_mae / max(1, n_batches),
        "mae_k_pos": (mae_k_on_pos_sum / max(1, mae_k_on_pos_n)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "n_pos_tokens": float(n_parent_tokens),
        "n_jets": float(n_jets),
    }


def train_parent_count(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainCfg,
    *,
    parent_pos_weight: float,
    ckpt_path: Optional[str | Path] = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    sched = get_scheduler(opt, int(cfg.warmup_epochs), int(cfg.epochs))

    best = float("inf")
    best_state = None
    no_improve = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_parent_loss": [],
        "val_k_loss": [],
        "val_sum_mae": [],
        "val_prec": [],
        "val_rec": [],
    }

    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        running = 0.0
        n = 0

        for batch in train_loader:
            x = batch["hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            y_parent = batch["parent_gt"].to(device)
            y_k = batch["k_true"].to(device)
            y_sum = batch["jet_sum_true"].to(device)

            opt.zero_grad(set_to_none=True)
            out = model(x, m)

            parent_loss = _masked_bce_with_logits(
                out.parent_logit, y_parent, m, pos_weight=float(parent_pos_weight)
            )

            loss = float(cfg.w_parent) * parent_loss

            if ep > int(cfg.warmup_parent_epochs):
                # Count loss
                w = torch.where(
                    y_parent > 0.5,
                    torch.ones_like(y_parent),
                    y_parent.new_full(y_parent.shape, float(cfg.neg_k_weight)),
                )
                k_loss = _masked_weighted_huber(out.k_pred, y_k, m, w, delta=float(cfg.huber_delta))
                # Sum constraint (per-jet)
                sum_pred = out.k_pred.sum(dim=1)
                sum_loss = F.smooth_l1_loss(sum_pred, y_sum, reduction="mean", beta=float(cfg.huber_delta))
                loss = loss + float(cfg.w_k) * k_loss + float(cfg.w_sum) * sum_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()

            running += float(loss.item())
            n += 1

        sched.step()

        val = eval_parentness_and_count(
            model,
            val_loader,
            device,
            pos_weight=float(parent_pos_weight),
            thr_parent=float(cfg.thr_parent),
            huber_delta=float(cfg.huber_delta),
            neg_k_weight=float(cfg.neg_k_weight),
        )

        # Joint validation scalar: parent_loss + k_loss + sum_mae (proxy)
        val_scalar = float(val["parent_loss"] + val["k_loss"] + val["sum_mae"])
        improved = val_scalar < (best - float(cfg.min_delta))
        if improved:
            best = val_scalar
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if ckpt_path is not None:
                save_checkpoint(model, ckpt_path, extra={"best_val": best, "epoch": ep, "val": val})
        else:
            no_improve += 1

        history["train_loss"].append(float(running / max(1, n)))
        history["val_parent_loss"].append(float(val["parent_loss"]))
        history["val_k_loss"].append(float(val["k_loss"]))
        history["val_sum_mae"].append(float(val["sum_mae"]))
        history["val_prec"].append(float(val["precision"]))
        history["val_rec"].append(float(val["recall"]))

        print(
            f"[Parent+Count] Ep {ep:03d}: train_loss={history['train_loss'][-1]:.4f} "
            f"val_parent={val['parent_loss']:.4f} val_k={val['k_loss']:.4f} val_sum_mae={val['sum_mae']:.3f} "
            f"prec={val['precision']:.3f} rec={val['recall']:.3f} no_improve={no_improve}"
        )

        if ep >= int(cfg.warmup_parent_epochs) and no_improve >= int(cfg.patience):
            print(f"[Parent+Count] Early stopping at epoch {ep} (best_scalar={best:.4f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_scalar": best, "history": history}


# -----------------------------
# Plotting helpers
# -----------------------------


@torch.no_grad()
def predict_on_loader(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_sum_pred = []
    all_sum_true = []
    all_parent_prob = []
    for batch in loader:
        x = batch["hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        y_sum = batch["jet_sum_true"].to(device)
        out = model(x, m)
        sum_pred = (out.k_pred * m.to(out.k_pred.dtype)).sum(dim=1)
        parent_prob = torch.sigmoid(out.parent_logit) * m.to(out.parent_logit.dtype)
        all_sum_pred.append(sum_pred.detach().cpu().numpy())
        all_sum_true.append(y_sum.detach().cpu().numpy())
        all_parent_prob.append(parent_prob.detach().cpu().numpy())
    sp = np.concatenate(all_sum_pred, axis=0).astype(np.float64)
    st = np.concatenate(all_sum_true, axis=0).astype(np.float64)
    pp = np.concatenate(all_parent_prob, axis=0).astype(np.float64)
    return sp, st, pp


def plot_sum_predictions(
    pred_sum: np.ndarray,
    true_sum: np.ndarray,
    *,
    title: str = "Jet-level sum(k) (test)",
    bins: int = 60,
    save_path: str | Path | None = None,
    dpi: int = 160,
):
    import matplotlib.pyplot as plt  # type: ignore

    pred_sum = np.asarray(pred_sum, dtype=np.float64)
    true_sum = np.asarray(true_sum, dtype=np.float64)
    err = pred_sum - true_sum

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.suptitle(title)

    ax = axes[0]
    ax.hist(true_sum, bins=int(bins), alpha=0.75, label="True")
    ax.hist(pred_sum, bins=int(bins), alpha=0.65, label="Pred")
    ax.set_xlabel("sum(k)")
    ax.set_ylabel("Jets")
    ax.set_title("Distribution")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.hist(err, bins=int(bins), alpha=0.8)
    ax.set_xlabel("Pred - True")
    ax.set_ylabel("Jets")
    ax.set_title("Error")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
        print(f"Saved figure: {p}")
    plt.show()


# -----------------------------
# Extra diagnostics (requested)
# -----------------------------


@torch.no_grad()
def collect_token_outputs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """
    Collect token-level outputs/targets for diagnostics.

    Returns numpy arrays:
      - parent_prob: [N,S] float32 in [0,1]
      - parent_gt:   [N,S] float32 (0/1)
      - mask:        [N,S] bool
      - k_pred:      [N,S] float32 >=0
      - k_true:      [N,S] float32 >=0
    """
    model.eval()
    parent_prob_l = []
    parent_gt_l = []
    mask_l = []
    k_pred_l = []
    k_true_l = []
    nb = 0
    for batch in loader:
        x = batch["hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        y_parent = batch["parent_gt"].to(device)
        y_k = batch["k_true"].to(device)
        out = model(x, m)
        pp = torch.sigmoid(out.parent_logit) * m.to(out.parent_logit.dtype)
        parent_prob_l.append(pp.detach().cpu().numpy().astype(np.float32))
        parent_gt_l.append(y_parent.detach().cpu().numpy().astype(np.float32))
        mask_l.append(m.detach().cpu().numpy().astype(np.bool_))
        k_pred_l.append((out.k_pred * m.to(out.k_pred.dtype)).detach().cpu().numpy().astype(np.float32))
        k_true_l.append((y_k * m.to(y_k.dtype)).detach().cpu().numpy().astype(np.float32))
        nb += 1
        if max_batches is not None and nb >= int(max_batches):
            break
    return {
        "parent_prob": np.concatenate(parent_prob_l, axis=0),
        "parent_gt": np.concatenate(parent_gt_l, axis=0),
        "mask": np.concatenate(mask_l, axis=0),
        "k_pred": np.concatenate(k_pred_l, axis=0),
        "k_true": np.concatenate(k_true_l, axis=0),
    }


def sweep_parent_thresholds(
    parent_prob: np.ndarray,
    parent_gt: np.ndarray,
    mask: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute precision/recall for a sweep of parentness thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    pp = np.asarray(parent_prob, dtype=np.float64)
    gt = (np.asarray(parent_gt, dtype=np.float64) > 0.5)
    m = np.asarray(mask, dtype=bool)

    prec = np.zeros_like(thresholds, dtype=np.float64)
    rec = np.zeros_like(thresholds, dtype=np.float64)
    tp = np.zeros_like(thresholds, dtype=np.float64)
    fp = np.zeros_like(thresholds, dtype=np.float64)
    fn = np.zeros_like(thresholds, dtype=np.float64)

    # Flatten valid tokens once for speed/memory.
    pp_v = pp[m]
    gt_v = gt[m]

    for i, thr in enumerate(thresholds):
        pred = (pp_v > float(thr))
        tpi = float(np.sum(pred & gt_v))
        fpi = float(np.sum(pred & (~gt_v)))
        fni = float(np.sum((~pred) & gt_v))
        tp[i], fp[i], fn[i] = tpi, fpi, fni
        prec[i] = tpi / max(1.0, (tpi + fpi))
        rec[i] = tpi / max(1.0, (tpi + fni))

    return {"thr": thresholds, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def plot_parent_pr_sweep(
    sweep: dict[str, np.ndarray],
    *,
    title: str = "Parentness precision/recall sweep",
    save_path: str | Path | None = None,
    dpi: int = 160,
):
    import matplotlib.pyplot as plt  # type: ignore

    thr = sweep["thr"]
    prec = sweep["precision"]
    rec = sweep["recall"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    fig.suptitle(title)

    ax = axes[0]
    ax.plot(thr, prec, marker="o", label="precision")
    ax.plot(thr, rec, marker="o", label="recall")
    ax.set_xlabel("thr_parent")
    ax.set_ylabel("metric")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title("vs threshold")

    ax = axes[1]
    ax.plot(rec, prec, marker="o")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.set_title("PR curve (sweep)")

    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
        print(f"Saved figure: {p}")
    plt.show()


def plot_k_pos_diagnostics(
    k_pred: np.ndarray,
    k_true: np.ndarray,
    parent_gt: np.ndarray,
    mask: np.ndarray,
    *,
    title: str = "k diagnostics on GT parents",
    max_points: int = 200_000,
    seed: int = 42,
    save_path: str | Path | None = None,
    dpi: int = 160,
):
    """Scatter + error hist for k on GT-positive parent tokens only."""
    import matplotlib.pyplot as plt  # type: ignore

    kp = np.asarray(k_pred, dtype=np.float64)
    kt = np.asarray(k_true, dtype=np.float64)
    gt = (np.asarray(parent_gt, dtype=np.float64) > 0.5)
    m = np.asarray(mask, dtype=bool)

    sel = m & gt
    kp_v = kp[sel]
    kt_v = kt[sel]
    n = int(kp_v.size)
    if n == 0:
        print("No positive parent tokens found for k diagnostics.")
        return

    if n > int(max_points):
        rng = np.random.default_rng(int(seed))
        take = rng.choice(n, size=int(max_points), replace=False)
        kp_v = kp_v[take]
        kt_v = kt_v[take]

    err = kp_v - kt_v

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    fig.suptitle(title)

    ax = axes[0]
    ax.scatter(kt_v, kp_v, s=4, alpha=0.15)
    mx = float(max(kt_v.max(), kp_v.max(), 1.0))
    ax.plot([0, mx], [0, mx], "k--", linewidth=1)
    ax.set_xlabel("k_true")
    ax.set_ylabel("k_pred")
    ax.set_title(f"Scatter (n={len(kp_v):,} sampled)")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.hist(err, bins=80, alpha=0.85)
    ax.set_xlabel("k_pred - k_true")
    ax.set_ylabel("Tokens")
    ax.set_title(f"Error (mean={err.mean():.3f}, std={err.std():.3f})")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
        print(f"Saved figure: {p}")
    plt.show()


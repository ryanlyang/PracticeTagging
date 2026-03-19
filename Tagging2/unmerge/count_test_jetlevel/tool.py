"""
Utilities for the count-only unmerger experiment.

This module intentionally keeps preprocessing and experiment I/O consistent with:
`unmerge/unmerge.ipynb` and `unmerge/tool.py`.

Count-only objective:
- Predict jet-level total missing count ΔN = N_off - N_hlt (or similar).
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# -----------------------------
# Feature distribution plotting
# -----------------------------

FEAT_NAMES_7 = [
    "dEta",
    "dPhi",
    "log_pt",
    "log_E",
    "log_pt/jet_pt",
    "log_E/jet_E",
    "dR",
]


def _sample_valid_values_1d(
    feat: np.ndarray,
    mask: np.ndarray,
    jet_idx: np.ndarray,
    dim: int,
    *,
    max_vals: int = 200_000,
    seed: int = 42,
) -> np.ndarray:
    """Sample valid (masked) values for one feature dim from [N,S,D] feature tensors."""
    vals = feat[jet_idx, :, dim][mask[jet_idx]]
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size > int(max_vals):
        rng = np.random.default_rng(int(seed))
        take = rng.choice(vals.size, size=int(max_vals), replace=False)
        vals = vals[take]
    return vals


def plot_feat_dists(
    feat_off: np.ndarray,
    mask_off: np.ndarray,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    *,
    jet_idx: np.ndarray,
    title: str = "Feature distributions",
    feat_names: list[str] | None = None,
    bins: int = 120,
    max_vals: int = 200_000,
    clip: tuple[float, float] | None = None,
    seed: int = 42,
    save_path: str | Path | None = None,
    dpi: int = 160,
):
    """Visualize feature distributions for OFF vs HLT (same as baseline notebook)."""
    import matplotlib.pyplot as plt  # type: ignore

    feat_names = FEAT_NAMES_7 if feat_names is None else feat_names
    n_dim = int(feat_off.shape[-1])
    if n_dim != len(feat_names):
        raise ValueError(f"feat dim={n_dim} but len(feat_names)={len(feat_names)}")

    fig, axes = plt.subplots(2, n_dim, figsize=(3.2 * n_dim, 6), sharey=False)
    fig.suptitle(title)

    for d in range(n_dim):
        v_off = _sample_valid_values_1d(
            feat_off, mask_off, jet_idx, d, max_vals=max_vals, seed=seed
        )
        v_hlt = _sample_valid_values_1d(
            feat_hlt, mask_hlt, jet_idx, d, max_vals=max_vals, seed=seed
        )

        if clip is not None:
            lo, hi = float(clip[0]), float(clip[1])
            v_off = np.clip(v_off, lo, hi)
            v_hlt = np.clip(v_hlt, lo, hi)

        ax0 = axes[0, d]
        ax1 = axes[1, d]
        ax0.hist(v_off, bins=int(bins), alpha=0.85)
        ax1.hist(v_hlt, bins=int(bins), alpha=0.85)
        ax0.set_title(f"OFF: {feat_names[d]}")
        ax1.set_title(f"HLT: {feat_names[d]}")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
        print(f"Saved figure: {p}")
    plt.show()


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
# Preprocessing (mirrors baseline)
# -----------------------------


def wrap_dphi(dphi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def apply_hlt_effects(const: np.ndarray, mask: np.ndarray, config: dict, seed: int = 42):
    """Apply realistic HLT effects (mirrors `unmerge/tool.py`)."""
    np.random.seed(int(seed))
    cfg = config["hlt_effects"]
    n_jets, max_part, _ = const.shape
    hlt = const.copy()
    hlt_mask = mask.copy()

    # Effect 1: Higher pT threshold
    below_threshold = (hlt[:, :, 0] < float(cfg["pt_threshold_hlt"])) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0

    # Effect 2: Cluster merging
    if bool(cfg["merge_enabled"]) and float(cfg["merge_radius"]) > 0:
        r = float(cfg["merge_radius"])
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            if len(valid_idx) < 2:
                continue
            to_remove = set()
            for i in range(len(valid_idx)):
                idx_i = int(valid_idx[i])
                if idx_i in to_remove:
                    continue
                for j in range(i + 1, len(valid_idx)):
                    idx_j = int(valid_idx[j])
                    if idx_j in to_remove:
                        continue
                    deta = float(hlt[jet_idx, idx_i, 1] - hlt[jet_idx, idx_j, 1])
                    dphi = float(wrap_dphi(hlt[jet_idx, idx_i, 2] - hlt[jet_idx, idx_j, 2]))
                    dR = math.sqrt(deta * deta + dphi * dphi)
                    if dR < r:
                        pt_i, pt_j = float(hlt[jet_idx, idx_i, 0]), float(hlt[jet_idx, idx_j, 0])
                        pt_sum = pt_i + pt_j
                        if pt_sum < 1e-6:
                            continue
                        w_i, w_j = pt_i / pt_sum, pt_j / pt_sum
                        hlt[jet_idx, idx_i, 0] = pt_sum
                        hlt[jet_idx, idx_i, 1] = w_i * hlt[jet_idx, idx_i, 1] + w_j * hlt[jet_idx, idx_j, 1]
                        phi_i, phi_j = float(hlt[jet_idx, idx_i, 2]), float(hlt[jet_idx, idx_j, 2])
                        hlt[jet_idx, idx_i, 2] = math.atan2(
                            w_i * math.sin(phi_i) + w_j * math.sin(phi_j),
                            w_i * math.cos(phi_i) + w_j * math.cos(phi_j),
                        )
                        hlt[jet_idx, idx_i, 3] = float(hlt[jet_idx, idx_i, 3]) + float(hlt[jet_idx, idx_j, 3])
                        to_remove.add(idx_j)
            for idx in to_remove:
                hlt_mask[jet_idx, idx] = False
                hlt[jet_idx, idx] = 0

    # Effect 3: Resolution smearing
    valid = hlt_mask.copy()
    pt_noise = np.clip(
        np.random.normal(1.0, float(cfg["pt_resolution"]), (n_jets, max_part)), 0.5, 1.5
    )
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)
    eta_noise = np.random.normal(0, float(cfg["eta_resolution"]), (n_jets, max_part))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)
    phi_noise = np.random.normal(0, float(cfg["phi_resolution"]), (n_jets, max_part))
    new_phi = hlt[:, :, 2] + phi_noise
    hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)

    # Effect 4: Random efficiency loss
    eff = float(cfg.get("efficiency_loss", 0.0))
    if eff > 0:
        lost = (np.random.random((n_jets, max_part)) < eff) & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    return hlt, hlt_mask


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
# Count dataset + training
# -----------------------------


class JetCountDataset(Dataset):
    """Dataset for jet-level count prediction.

    - x_hlt: [N,S,D]
    - mask_hlt: [N,S]
    - y: [N] (ΔN or N_off)
    - weights: [N] optional (kept for consistency; can be 1s)
    """

    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ):
        self.hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)
        if weights is None:
            weights = np.ones(len(y), dtype=np.float32)
        self.w = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i):
        return {
            "hlt": self.hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "y": self.y[i],
            "w": self.w[i],
        }


@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 3
    patience: int = 8
    min_delta: float = 0.0
    grad_clip: float = 1.0


def get_scheduler(opt, warmup: int, total: int):
    def lr_lambda(ep):
        if ep < int(warmup):
            return float(ep + 1) / float(max(1, warmup))
        return 0.5 * (1.0 + np.cos(np.pi * (ep - warmup) / float(max(1, total - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _weighted_huber(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    # Smooth L1 (Huber) per-sample
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=float(delta))
    w = torch.clamp(w, min=0.0)
    return (loss * w).sum() / (w.sum() + 1e-8)


def train_count_epoch(
    model: torch.nn.Module,
    loader,
    opt,
    device,
    *,
    huber_delta: float = 1.0,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        y = batch["y"].to(device)
        w = batch["w"].to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(x, m)
        loss = _weighted_huber(pred, y, w, delta=float(huber_delta))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = int(y.shape[0])
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


@torch.no_grad()
def eval_count(
    model: torch.nn.Module,
    loader,
    device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    for batch in loader:
        x = batch["hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        y = batch["y"].to(device)
        pred = model(x, m)
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    p = np.concatenate(preds, axis=0).astype(np.float64)
    t = np.concatenate(trues, axis=0).astype(np.float64)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    bias = float(np.mean(p - t))
    return mae, rmse, bias, p, t


def plot_count_predictions(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    title: str = "Count prediction (test)",
    bins: int = 60,
    save_path: str | Path | None = None,
    dpi: int = 160,
):
    """Plot pred/true histograms and error distribution."""
    import matplotlib.pyplot as plt  # type: ignore

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.suptitle(title)

    ax = axes[0]
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    ax.hist(true, bins=int(bins), alpha=0.75, label="True")
    ax.hist(pred, bins=int(bins), alpha=0.65, label="Pred")
    ax.set_xlabel("ΔN")
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


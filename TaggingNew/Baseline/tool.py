"""
Utilities for the *baseline KD pipeline* extracted from `Previous/Flowmatching.ipynb`.

This module intentionally stays close to the notebook logic:
- HLT effects simulation (threshold + merge + smearing + efficiency)
- 7 engineered features in jet-axis frame
- standardization using OFFLINE TRAIN distribution
- training: teacher (offline), baseline (HLT), student+KD (HLT) with optional attn distillation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score, roc_curve


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
    percentiles=(0.1, 1, 5, 25, 50, 75, 95, 99, 99.9),
    save_path: str | Path | None = None,
    dpi: int = 160,
):
    """Visualize feature distributions for OFF vs HLT.

    Args:
        feat_off/feat_hlt: [N,S,D] feature arrays
        mask_off/mask_hlt: [N,S] boolean masks
        jet_idx: indices of jets to consider (e.g. train_idx)
        clip: optional (lo, hi) to clip before plotting/stats
    """
    # Soft dependency to keep `tool.py` usable without plotting installed
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

        # Print quick stats (p50 is always printed)
        if v_off.size > 0:
            ps = np.percentile(v_off, percentiles)
            print(
                f"[OFF] {feat_names[d]}: mean={v_off.mean():.4g} std={v_off.std():.4g} p50={np.percentile(v_off, 50):.4g}"
            )
            _ = ps  # keep for debugging if needed
        if v_hlt.size > 0:
            ps = np.percentile(v_hlt, percentiles)
            print(
                f"[HLT] {feat_names[d]}: mean={v_hlt.mean():.4g} std={v_hlt.std():.4g} p50={np.percentile(v_hlt, 50):.4g}"
            )
            _ = ps

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
    """Save a config dict as JSON."""
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
    """Save a model checkpoint (state_dict + optional metadata)."""
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
    """Load a model checkpoint into `model` and return the full payload."""
    p = Path(path)
    payload = torch.load(p.as_posix(), map_location=map_location)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state, strict=bool(strict))
    return payload if isinstance(payload, dict) else {"state_dict": state}


def wrap_dphi(dphi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def apply_hlt_effects(const: np.ndarray, mask: np.ndarray, config: dict, seed: int = 42):
    """Apply realistic HLT effects (mirrors Flowmatching.ipynb).

    Effects:
    1) higher pT threshold
    2) ΔR merge with pT-weighted eta/phi, E sum
    3) resolution smearing
    4) random efficiency loss
    """
    # Note: do NOT call np.random.seed(...) here, otherwise you will pollute the global RNG state
    # and affect external randomness (e.g. DataLoader shuffling, other augmentations/samplers).
    # We use a local RandomState to keep reproducibility without changing the global random stream.
    rs = np.random.RandomState(int(seed))
    cfg = config["hlt_effects"]
    n_jets, max_part, _ = const.shape
    hlt = const.copy()
    hlt_mask = mask.copy()

    # Effect 1: Higher pT threshold
    threshold_enabled = bool(cfg.get("threshold_enabled", True))
    if threshold_enabled:
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
    smear_enabled = bool(cfg.get("smear_enabled", True))
    if smear_enabled:
        valid = hlt_mask.copy()
        pt_noise = np.clip(
            rs.normal(1.0, float(cfg["pt_resolution"]), (n_jets, max_part)), 0.5, 1.5
        )
        hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)
        eta_noise = rs.normal(0, float(cfg["eta_resolution"]), (n_jets, max_part))
        hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)
        phi_noise = rs.normal(0, float(cfg["phi_resolution"]), (n_jets, max_part))
        new_phi = hlt[:, :, 2] + phi_noise
        hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)
        hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)

    # Effect 4: Random efficiency loss
    eff = float(cfg.get("efficiency_loss", 0.0))
    if eff > 0:
        lost = (rs.random_sample((n_jets, max_part)) < eff) & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    return hlt, hlt_mask


def compute_features(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute 7 relative features (mirrors Flowmatching.ipynb)."""
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


def standardize(feat: np.ndarray, mask: np.ndarray, means: np.ndarray, stds: np.ndarray, clip: float = 10.0) -> np.ndarray:
    out = (feat - means[None, None, :]) / stds[None, None, :]
    out = np.clip(out, -float(clip), float(clip))
    out = np.nan_to_num(out, 0.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


class JetDataset(Dataset):
    def __init__(self, feat_off: np.ndarray, feat_hlt: np.ndarray, labels: np.ndarray, masks_off: np.ndarray, masks_hlt: np.ndarray, weights: np.ndarray):
        self.off = torch.tensor(feat_off, dtype=torch.float32)
        self.hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mask_off = torch.tensor(masks_off, dtype=torch.bool)
        self.mask_hlt = torch.tensor(masks_hlt, dtype=torch.bool)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, i):
        return {
            "off": self.off[i],
            "hlt": self.hlt[i],
            "mask_off": self.mask_off[i],
            "mask_hlt": self.mask_hlt[i],
            "label": self.labels[i],
            "weight": self.weights[i],
        }


@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 3
    patience: int = 15
    grad_clip: float = 1.0
    min_delta: float = 0.0
    min_epochs: int = 0


def get_scheduler(opt, warmup: int, total: int):
    def lr_lambda(ep):
        if ep < int(warmup):
            return float(ep + 1) / float(max(1, warmup))
        return 0.5 * (1.0 + np.cos(np.pi * (ep - warmup) / float(max(1, total - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_standard(model, loader, opt, device, feat_key: str, mask_key: str):
    model.train()
    preds, labs = [], []
    total_loss = 0.0
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x, m).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss.item()) * int(y.shape[0])
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
    return total_loss / max(1, len(preds)), float(roc_auc_score(labs, preds))


@torch.no_grad()
def evaluate(model, loader, device, feat_key: str, mask_key: str):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        logits = model(x, m).squeeze(-1)
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
    return float(roc_auc_score(labs, preds)), np.asarray(preds), np.asarray(labs)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    s_soft = torch.sigmoid(student_logits / float(T))
    t_soft = torch.sigmoid(teacher_logits / float(T))
    return F.binary_cross_entropy(s_soft, t_soft) * (float(T) ** 2)


def attn_loss(s_attn: torch.Tensor, t_attn: torch.Tensor, s_mask: torch.Tensor, t_mask: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    s_valid = s_attn * s_mask.float()
    t_valid = t_attn * t_mask.float()
    s_ent = -(s_valid * torch.log(s_valid + eps)).sum(dim=1)
    t_ent = -(t_valid * torch.log(t_valid + eps)).sum(dim=1)
    return F.mse_loss(s_ent, t_ent) + F.mse_loss(s_valid.max(dim=1)[0], t_valid.max(dim=1)[0])


def train_kd(student, teacher, loader, opt, device, cfg: dict):
    student.train()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"]["alpha_attn"])

    preds, labs = [], []
    total_loss = 0.0

    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        x_off = batch["off"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)

        with torch.no_grad():
            t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
            t_logits = t_logits.squeeze(-1)

        opt.zero_grad(set_to_none=True)
        s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
        s_logits = s_logits.squeeze(-1)

        loss_kd = kd_loss(s_logits, t_logits, T)
        loss_hard = F.binary_cross_entropy_with_logits(s_logits, y, weight=w)
        loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off)

        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += float(loss.item()) * int(y.shape[0])
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / max(1, len(preds)), float(roc_auc_score(labs, preds))


def compute_roc(y: np.ndarray, p: np.ndarray):
    fpr, tpr, _ = roc_curve(y, p)
    auc = float(roc_auc_score(y, p))
    return fpr, tpr, auc

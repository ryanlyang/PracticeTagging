"""
Utilities for the *baseline KD pipeline* extracted from `Previous/Flowmatching.ipynb`.

This module intentionally stays close to the notebook logic:
- HLT effects simulation (threshold + merge + smearing + efficiency)
- 7 engineered features in jet-axis frame
- standardization using OFFLINE TRAIN distribution
- training: teacher (offline), baseline (HLT), student+KD (HLT) with optional attn distillation
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Any, Optional

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


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


@dataclass
class PreparedSmearBaselineData:
    constituents_raw: np.ndarray
    masks_raw: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    constituents_off: np.ndarray
    masks_off: np.ndarray
    features_off: np.ndarray
    features_off_std: np.ndarray
    constituents_hlt_eval: np.ndarray
    masks_hlt_eval: np.ndarray
    features_hlt_eval: np.ndarray
    features_hlt_eval_std: np.ndarray
    feat_means_off: np.ndarray
    feat_stds_off: np.ndarray


def set_random_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def apply_offline_threshold(const: np.ndarray, mask: np.ndarray, config: dict):
    pt_thr_off = float(config["hlt_effects"]["pt_threshold_offline"])
    masks_off = mask & (const[:, :, 0] >= pt_thr_off)
    constituents_off = const.copy()
    constituents_off[~masks_off] = 0
    return constituents_off.astype(np.float32), masks_off.astype(bool)


def build_standardized_hlt_features(
    const: np.ndarray,
    mask: np.ndarray,
    config: dict,
    feat_means_off: np.ndarray,
    feat_stds_off: np.ndarray,
    *,
    seed: int,
    clip: float = 10.0,
):
    constituents_hlt, masks_hlt = apply_hlt_effects(const, mask, config, seed=int(seed))
    features_hlt = compute_features(constituents_hlt, masks_hlt)
    features_hlt_std = standardize(
        features_hlt,
        masks_hlt,
        feat_means_off,
        feat_stds_off,
        clip=float(clip),
    )
    return constituents_hlt, masks_hlt, features_hlt, features_hlt_std


def prepare_smear_baseline_data(
    constituents_raw: np.ndarray,
    masks_raw: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: dict,
    *,
    split_seed: int = 42,
    eval_hlt_seed: int = 42,
    clip: float = 10.0,
) -> PreparedSmearBaselineData:
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.3,
        random_state=int(split_seed),
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=int(split_seed),
        stratify=labels[temp_idx],
    )

    constituents_off, masks_off = apply_offline_threshold(constituents_raw, masks_raw, config)
    features_off = compute_features(constituents_off, masks_off)
    feat_means_off, feat_stds_off = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(
        features_off,
        masks_off,
        feat_means_off,
        feat_stds_off,
        clip=float(clip),
    )

    constituents_hlt_eval, masks_hlt_eval, features_hlt_eval, features_hlt_eval_std = build_standardized_hlt_features(
        constituents_raw,
        masks_raw,
        config,
        feat_means_off,
        feat_stds_off,
        seed=int(eval_hlt_seed),
        clip=float(clip),
    )

    return PreparedSmearBaselineData(
        constituents_raw=np.asarray(constituents_raw, dtype=np.float32),
        masks_raw=np.asarray(masks_raw, dtype=bool),
        labels=np.asarray(labels, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float32),
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
        constituents_off=constituents_off,
        masks_off=masks_off,
        features_off=features_off,
        features_off_std=features_off_std,
        constituents_hlt_eval=np.asarray(constituents_hlt_eval, dtype=np.float32),
        masks_hlt_eval=np.asarray(masks_hlt_eval, dtype=bool),
        features_hlt_eval=np.asarray(features_hlt_eval, dtype=np.float32),
        features_hlt_eval_std=np.asarray(features_hlt_eval_std, dtype=np.float32),
        feat_means_off=np.asarray(feat_means_off, dtype=np.float32),
        feat_stds_off=np.asarray(feat_stds_off, dtype=np.float32),
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    seed: Optional[int] = None,
):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(drop_last),
        generator=generator,
    )


def make_offline_dataset(data: PreparedSmearBaselineData, idx: np.ndarray) -> JetDataset:
    idx = np.asarray(idx, dtype=np.int64)
    return JetDataset(
        data.features_off_std[idx],
        data.features_hlt_eval_std[idx],
        data.labels[idx],
        data.masks_off[idx],
        data.masks_hlt_eval[idx],
        data.weights[idx],
    )


def make_hlt_eval_dataset(data: PreparedSmearBaselineData, idx: np.ndarray) -> JetDataset:
    idx = np.asarray(idx, dtype=np.int64)
    return JetDataset(
        data.features_off_std[idx],
        data.features_hlt_eval_std[idx],
        data.labels[idx],
        data.masks_off[idx],
        data.masks_hlt_eval[idx],
        data.weights[idx],
    )


def make_train_hlt_dataset_for_epoch(
    data: PreparedSmearBaselineData,
    config: dict,
    *,
    epoch_seed: int,
    clip: float = 10.0,
) -> JetDataset:
    idx = data.train_idx
    _, masks_hlt_train, _, features_hlt_train_std = build_standardized_hlt_features(
        data.constituents_raw[idx],
        data.masks_raw[idx],
        config,
        data.feat_means_off,
        data.feat_stds_off,
        seed=int(epoch_seed),
        clip=float(clip),
    )
    return JetDataset(
        data.features_off_std[idx],
        features_hlt_train_std,
        data.labels[idx],
        data.masks_off[idx],
        masks_hlt_train,
        data.weights[idx],
    )


@torch.no_grad()
def evaluate_with_metadata(model, loader, device, feat_key: str, mask_key: str):
    model.eval()
    preds, labs, weights = [], [], []
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)
        logits = model(x, m).squeeze(-1)
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
    auc = float(roc_auc_score(labs, preds))
    return auc, np.asarray(preds), np.asarray(labs), np.asarray(weights)


def save_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return p


def save_prediction_bundle(
    path: str | Path,
    *,
    preds: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p.as_posix(),
        preds=np.asarray(preds, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
    )
    return p


def train_standard_full(
    model,
    data: PreparedSmearBaselineData,
    config: dict,
    device,
    *,
    feat_key: str,
    mask_key: str,
    val_loader,
    train_loader_builder: Callable[[int], Any],
    ckpt_path: str | Path,
    history_path: str | Path,
    extra_ckpt: Optional[dict[str, Any]] = None,
    log_prefix: str,
):
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["lr"])
    wd = float(config["training"]["weight_decay"])
    warm = int(config["training"].get("warmup_epochs", 0))
    pat = int(config["training"]["patience"])
    min_delta = float(config["training"].get("min_delta", 1e-4))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = get_scheduler(opt, warm, epochs)
    best_auc = -float("inf")
    best_state = None
    no_imp = 0
    history_rows: list[dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        train_loader = train_loader_builder(ep)
        train_loss, train_auc = train_standard(model, train_loader, opt, device, feat_key, mask_key)
        val_auc, _, _, _ = evaluate_with_metadata(model, val_loader, device, feat_key, mask_key)
        sch.step()

        history_rows.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "val_auc": float(val_auc),
                "lr": float(sch.get_last_lr()[0]),
            }
        )

        if val_auc > best_auc + min_delta:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        print(
            f"{log_prefix} ep={ep:03d} train_loss={train_loss:.6f} train_auc={train_auc:.6f} "
            f"val_auc={val_auc:.6f} best={best_auc:.6f} no_imp={no_imp}"
        )
        if no_imp >= pat:
            print(f"{log_prefix} Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_rows_csv(history_rows, history_path)
    extra = {"best_val_auc": float(best_auc), "history_path": str(history_path)}
    if extra_ckpt is not None:
        extra.update(extra_ckpt)
    save_checkpoint(model, ckpt_path, extra=extra)
    return {
        "best_val_auc": float(best_auc),
        "history_rows": history_rows,
        "history_path": str(history_path),
        "ckpt_path": str(ckpt_path),
    }


def train_kd_full(
    student,
    teacher,
    data: PreparedSmearBaselineData,
    config: dict,
    device,
    *,
    val_loader,
    train_loader_builder: Callable[[int], Any],
    ckpt_path: str | Path,
    history_path: str | Path,
    extra_ckpt: Optional[dict[str, Any]] = None,
    log_prefix: str,
):
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["lr"])
    wd = float(config["training"]["weight_decay"])
    warm = int(config["training"].get("warmup_epochs", 0))
    pat = int(config["training"]["patience"])
    min_delta = float(config["training"].get("min_delta", 1e-4))
    kd_cfg = {"kd": config["kd"]}

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)
    sch = get_scheduler(opt, warm, epochs)
    best_auc = -float("inf")
    best_state = None
    no_imp = 0
    history_rows: list[dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        train_loader = train_loader_builder(ep)
        train_loss, train_auc = train_kd(student, teacher, train_loader, opt, device, kd_cfg)
        val_auc, _, _, _ = evaluate_with_metadata(student, val_loader, device, "hlt", "mask_hlt")
        sch.step()

        history_rows.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "val_auc": float(val_auc),
                "lr": float(sch.get_last_lr()[0]),
            }
        )

        if val_auc > best_auc + min_delta:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        print(
            f"{log_prefix} ep={ep:03d} train_loss={train_loss:.6f} train_auc={train_auc:.6f} "
            f"val_auc={val_auc:.6f} best={best_auc:.6f} no_imp={no_imp}"
        )
        if no_imp >= pat:
            print(f"{log_prefix} Early stopping")
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    save_rows_csv(history_rows, history_path)
    extra = {"best_val_auc": float(best_auc), "history_path": str(history_path)}
    if extra_ckpt is not None:
        extra.update(extra_ckpt)
    save_checkpoint(student, ckpt_path, extra=extra)
    return {
        "best_val_auc": float(best_auc),
        "history_rows": history_rows,
        "history_path": str(history_path),
        "ckpt_path": str(ckpt_path),
    }


def run_smear_baseline_repeat(
    *,
    model_cls: Callable[..., torch.nn.Module],
    model_kwargs: dict[str, Any],
    data: PreparedSmearBaselineData,
    config: dict,
    device,
    repeat_idx: int,
    repeat_seed: int,
    out_dir: str | Path,
) -> dict[str, Any]:
    set_random_seed(int(repeat_seed))

    repeat_dir = ensure_dir(Path(out_dir) / f"repeat_{int(repeat_idx):02d}_seed_{int(repeat_seed)}")
    ckpt_dir = ensure_dir(repeat_dir / "ckpts")
    pred_dir = ensure_dir(repeat_dir / "predictions")
    history_dir = ensure_dir(repeat_dir / "history")

    batch_size = int(config["training"]["batch_size"])
    clip = float(config["training"].get("feature_clip", 10.0))

    teacher_train_ds = make_offline_dataset(data, data.train_idx)
    teacher_val_ds = make_offline_dataset(data, data.val_idx)
    teacher_test_ds = make_offline_dataset(data, data.test_idx)
    hlt_train_fixed_ds = make_hlt_eval_dataset(data, data.train_idx)
    hlt_val_ds = make_hlt_eval_dataset(data, data.val_idx)
    hlt_test_ds = make_hlt_eval_dataset(data, data.test_idx)

    teacher_val_loader = make_loader(teacher_val_ds, batch_size=batch_size, shuffle=False)
    teacher_test_loader = make_loader(teacher_test_ds, batch_size=batch_size, shuffle=False)
    hlt_val_loader = make_loader(hlt_val_ds, batch_size=batch_size, shuffle=False)
    hlt_test_loader = make_loader(hlt_test_ds, batch_size=batch_size, shuffle=False)

    def make_teacher_train_loader(epoch: int):
        return make_loader(
            teacher_train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            seed=int(repeat_seed) * 10_000 + int(epoch),
        )

    def make_hlt_train_loader(epoch: int):
        epoch_seed = int(repeat_seed) * 10_000 + int(epoch)
        if not bool(config.get("resmear_each_epoch", True)):
            return make_loader(
                hlt_train_fixed_ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                seed=epoch_seed,
            )
        epoch_ds = make_train_hlt_dataset_for_epoch(
            data,
            config,
            epoch_seed=epoch_seed,
            clip=clip,
        )
        return make_loader(
            epoch_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            seed=epoch_seed,
        )

    teacher = model_cls(**model_kwargs).to(device)
    teacher_fit = train_standard_full(
        teacher,
        data,
        config,
        device,
        feat_key="off",
        mask_key="mask_off",
        val_loader=teacher_val_loader,
        train_loader_builder=make_teacher_train_loader,
        ckpt_path=ckpt_dir / "offline_teacher.pt",
        history_path=history_dir / "offline_teacher_history.csv",
        extra_ckpt={"repeat_idx": int(repeat_idx), "repeat_seed": int(repeat_seed), "model_key": "offline"},
        log_prefix=f"[Repeat {repeat_idx + 1}][Offline]",
    )
    auc_off, preds_off, labels_off, weights_off = evaluate_with_metadata(
        teacher,
        teacher_test_loader,
        device,
        "off",
        "mask_off",
    )
    pred_path_off = save_prediction_bundle(
        pred_dir / "offline_teacher_test_preds.npz",
        preds=preds_off,
        labels=labels_off,
        weights=weights_off,
    )

    baseline = model_cls(**model_kwargs).to(device)
    baseline_fit = train_standard_full(
        baseline,
        data,
        config,
        device,
        feat_key="hlt",
        mask_key="mask_hlt",
        val_loader=hlt_val_loader,
        train_loader_builder=make_hlt_train_loader,
        ckpt_path=ckpt_dir / "hlt_baseline.pt",
        history_path=history_dir / "hlt_baseline_history.csv",
        extra_ckpt={"repeat_idx": int(repeat_idx), "repeat_seed": int(repeat_seed), "model_key": "hlt"},
        log_prefix=f"[Repeat {repeat_idx + 1}][HLT]",
    )
    auc_hlt, preds_hlt, labels_hlt, weights_hlt = evaluate_with_metadata(
        baseline,
        hlt_test_loader,
        device,
        "hlt",
        "mask_hlt",
    )
    pred_path_hlt = save_prediction_bundle(
        pred_dir / "hlt_baseline_test_preds.npz",
        preds=preds_hlt,
        labels=labels_hlt,
        weights=weights_hlt,
    )

    student = model_cls(**model_kwargs).to(device)
    kd_fit = train_kd_full(
        student,
        teacher,
        data,
        config,
        device,
        val_loader=hlt_val_loader,
        train_loader_builder=make_hlt_train_loader,
        ckpt_path=ckpt_dir / "hlt_kd_student.pt",
        history_path=history_dir / "hlt_kd_student_history.csv",
        extra_ckpt={"repeat_idx": int(repeat_idx), "repeat_seed": int(repeat_seed), "model_key": "hlt_kd"},
        log_prefix=f"[Repeat {repeat_idx + 1}][HLT+KD]",
    )
    auc_kd, preds_kd, labels_kd, weights_kd = evaluate_with_metadata(
        student,
        hlt_test_loader,
        device,
        "hlt",
        "mask_hlt",
    )
    pred_path_kd = save_prediction_bundle(
        pred_dir / "hlt_kd_student_test_preds.npz",
        preds=preds_kd,
        labels=labels_kd,
        weights=weights_kd,
    )

    return {
        "repeat_idx": int(repeat_idx),
        "repeat_seed": int(repeat_seed),
        "repeat_dir": str(repeat_dir),
        "models": {
            "offline": {
                "display_name": "Offline teacher",
                "auc": float(auc_off),
                "preds": preds_off,
                "labels": labels_off,
                "weights": weights_off,
                "prediction_path": str(pred_path_off),
                **teacher_fit,
            },
            "hlt": {
                "display_name": "HLT baseline",
                "auc": float(auc_hlt),
                "preds": preds_hlt,
                "labels": labels_hlt,
                "weights": weights_hlt,
                "prediction_path": str(pred_path_hlt),
                **baseline_fit,
            },
            "hlt_kd": {
                "display_name": "HLT + KD",
                "auc": float(auc_kd),
                "preds": preds_kd,
                "labels": labels_kd,
                "weights": weights_kd,
                "prediction_path": str(pred_path_kd),
                **kd_fit,
            },
        },
    }


def build_repeat_metric_rows(repeat_results: list[dict[str, Any]]):
    detail_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = {}
    for repeat_result in repeat_results:
        for model_key, model_result in repeat_result["models"].items():
            grouped.setdefault(model_key, []).append(float(model_result["auc"]))
            detail_rows.append(
                {
                    "repeat_idx": int(repeat_result["repeat_idx"]),
                    "repeat_seed": int(repeat_result["repeat_seed"]),
                    "model_key": str(model_key),
                    "model_name": str(model_result["display_name"]),
                    "test_auc": float(model_result["auc"]),
                    "best_val_auc": float(model_result["best_val_auc"]),
                    "ckpt_path": str(model_result["ckpt_path"]),
                    "prediction_path": str(model_result["prediction_path"]),
                    "history_path": str(model_result["history_path"]),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    if repeat_results:
        exemplar = repeat_results[0]["models"]
        for model_key, auc_values in grouped.items():
            model_name = exemplar[model_key]["display_name"]
            auc_arr = np.asarray(auc_values, dtype=np.float64)
            summary_rows.append(
                {
                    "model_key": str(model_key),
                    "model_name": str(model_name),
                    "num_repeats": int(auc_arr.size),
                    "auc_mean": float(np.mean(auc_arr)),
                    "auc_std": float(np.std(auc_arr, ddof=0)),
                }
            )
    return detail_rows, summary_rows


def plot_repeat_mean_roc(
    repeat_results: list[dict[str, Any]],
    save_path: str | Path,
    *,
    model_order: Optional[list[str]] = None,
    model_colors: Optional[dict[str, str]] = None,
    dpi: int = 160,
):
    import matplotlib.pyplot as plt  # type: ignore

    if not repeat_results:
        raise ValueError("repeat_results is empty")

    grid = np.linspace(0.0, 1.0, 500)
    model_order = model_order or ["offline", "hlt", "hlt_kd"]
    model_colors = model_colors or {
        "offline": "#4C78A8",
        "hlt": "#F58518",
        "hlt_kd": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    for model_key in model_order:
        curves = []
        aucs = []
        display_name = None
        for repeat_result in repeat_results:
            if model_key not in repeat_result["models"]:
                continue
            model_result = repeat_result["models"][model_key]
            display_name = str(model_result["display_name"])
            fpr, tpr, auc = compute_roc(model_result["labels"], model_result["preds"])
            interp_tpr = np.interp(grid, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            curves.append(interp_tpr)
            aucs.append(float(auc))
        if not curves:
            continue
        curve_arr = np.asarray(curves, dtype=np.float64)
        mean_tpr = curve_arr.mean(axis=0)
        std_tpr = curve_arr.std(axis=0, ddof=0)
        mean_auc = float(np.mean(np.asarray(aucs, dtype=np.float64)))
        std_auc = float(np.std(np.asarray(aucs, dtype=np.float64), ddof=0))
        color = model_colors.get(model_key, None)
        ax.plot(
            grid,
            mean_tpr,
            lw=2.0,
            color=color,
            label=f"{display_name} (AUC={mean_auc:.6f}±{std_auc:.6f})",
        )
        ax.fill_between(
            grid,
            np.clip(mean_tpr - std_tpr, 0.0, 1.0),
            np.clip(mean_tpr + std_tpr, 0.0, 1.0),
            color=color,
            alpha=0.16,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Mean ROC over repeats")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
    print(f"Saved figure: {p}")
    plt.show()
    return p

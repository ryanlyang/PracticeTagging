"""
Joint unsmear utilities.

This directory keeps only the helper functions needed for joint training.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset


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


def save_rows_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Save a list of dict rows as CSV."""
    p = Path(path)
    ensure_dir(p.parent)
    rows = list(rows)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    _write_csv_rows(p, fieldnames, rows)
    return p


def save_prediction_bundle(
    path: str | Path,
    *,
    preds: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> Path:
    """Save predictions / labels / weights into a compressed NPZ bundle."""
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(
        p,
        preds=np.asarray(preds, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
    )
    return p


def wrap_dphi_np(dphi: np.ndarray) -> np.ndarray:
    """Wrap an angular difference into (-pi, pi]."""
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def wrap_dphi_torch(dphi: torch.Tensor) -> torch.Tensor:
    """Wrap an angular difference into (-pi, pi]."""
    return torch.atan2(torch.sin(dphi), torch.cos(dphi))


@dataclass
class HLTEffectsCfg:
    """HLT smearing config used for joint training."""

    pt_threshold_offline: float = 0.5
    pt_threshold_hlt: float = 0.5
    pt_resolution: float = 0.10
    eta_resolution: float = 0.03
    phi_resolution: float = 0.03


def apply_hlt_effects_pair(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: HLTEffectsCfg,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build `(pre_smear, post_smear, post_mask)`."""
    rs = np.random.RandomState(int(seed))
    hlt = const.copy()
    hlt_mask = mask.copy()

    pt_thr_off = float(cfg.pt_threshold_offline)
    hlt_mask = hlt_mask & (hlt[:, :, 0] >= pt_thr_off)
    hlt[~hlt_mask] = 0.0

    pt_thr_hlt = float(cfg.pt_threshold_hlt)
    below = (hlt[:, :, 0] < pt_thr_hlt) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0.0

    pre = hlt.copy()

    valid = hlt_mask.copy()
    n_jets, max_part, _ = hlt.shape
    pt_noise = rs.normal(1.0, float(cfg.pt_resolution), size=(n_jets, max_part))
    pt_noise = np.clip(pt_noise, 0.5, 1.5)
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0.0)

    eta_noise = rs.normal(0.0, float(cfg.eta_resolution), size=(n_jets, max_part))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5.0, 5.0), 0.0)

    phi_noise = rs.normal(0.0, float(cfg.phi_resolution), size=(n_jets, max_part))
    new_phi = hlt[:, :, 2] + phi_noise
    hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0.0)

    # Recompute E with a massless approximation.
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5.0, 5.0)), 0.0)

    pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    pre[~hlt_mask] = 0.0
    hlt[~hlt_mask] = 0.0
    return pre.astype(np.float32), hlt.astype(np.float32), hlt_mask.astype(bool)


def compute_jet_axis(const: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the jet axis."""
    pt = np.maximum(const[:, :, 0], 1e-8)
    eta = np.clip(const[:, :, 1], -5.0, 5.0)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], 1e-8)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    m = mask.astype(np.float32)
    jet_px = (px * m).sum(axis=1, keepdims=True)
    jet_py = (py * m).sum(axis=1, keepdims=True)
    jet_pz = (pz * m).sum(axis=1, keepdims=True)
    jet_E = (E * m).sum(axis=1, keepdims=True)

    jet_pt = np.sqrt(jet_px**2 + jet_py**2) + 1e-8
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)
    return {
        "jet_px": jet_px,
        "jet_py": jet_py,
        "jet_pz": jet_pz,
        "jet_E": jet_E,
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_phi": jet_phi,
    }


def compute_features_with_axis(
    const: np.ndarray,
    mask: np.ndarray,
    axis: Dict[str, np.ndarray],
    *,
    kind: str = "7d",
) -> np.ndarray:
    """Compute engineered features."""
    pt = np.maximum(const[:, :, 0], 1e-8)
    eta = np.clip(const[:, :, 1], -5.0, 5.0)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], 1e-8)

    jet_eta = axis["jet_eta"]
    jet_phi = axis["jet_phi"]
    jet_pt = axis["jet_pt"]
    jet_E = axis["jet_E"]

    dEta = eta - jet_eta
    dPhi = wrap_dphi_np(phi - jet_phi)
    log_pt = np.log(pt + 1e-8)
    log_E = np.log(E + 1e-8)
    log_pt_rel = np.log(pt / jet_pt + 1e-8)
    log_E_rel = np.log(E / (jet_E + 1e-8) + 1e-8)
    dR = np.sqrt(dEta**2 + dPhi**2)

    k = str(kind).lower()
    if k == "3d":
        feats = np.stack([dEta, dPhi, log_pt], axis=-1)
    elif k == "4d":
        feats = np.stack([dEta, dPhi, log_pt, log_E], axis=-1)
    elif k == "7d":
        feats = np.stack([dEta, dPhi, log_pt, log_E, log_pt_rel, log_E_rel, dR], axis=-1)
    else:
        raise ValueError(f"Unknown feature kind: {kind}")

    feats = np.clip(np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    feats[~mask] = 0.0
    return feats.astype(np.float32)


def get_feat_names(kind: str) -> list[str]:
    k = str(kind).lower()
    if k == "3d":
        return ["dEta", "dPhi", "log_pt"]
    if k == "4d":
        return ["dEta", "dPhi", "log_pt", "log_E"]
    if k == "7d":
        return ["dEta", "dPhi", "log_pt", "log_E", "log_pt_rel", "log_E_rel", "dR"]
    raise ValueError(f"Unknown feature kind: {kind}")


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
    *,
    clip: float = 10.0,
) -> np.ndarray:
    out = (feat - means[None, None, :]) / stds[None, None, :]
    out = np.clip(out, -float(clip), float(clip))
    out = np.nan_to_num(out, 0.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def build_unsmear_epoch_arrays(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: HLTEffectsCfg,
    *,
    feature_kind: str,
    means: np.ndarray,
    stds: np.ndarray,
    seed: int = 42,
    clip: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regenerate unsmear training inputs for one epoch."""
    pre_const, post_const, post_mask = apply_hlt_effects_pair(const, mask, cfg, seed=int(seed))

    pre_const = np.asarray(pre_const, dtype=np.float32)
    post_const = np.asarray(post_const, dtype=np.float32)
    post_mask = np.asarray(post_mask, dtype=bool)
    pre_const[~post_mask] = 0.0
    post_const[~post_mask] = 0.0

    axis_pre = compute_jet_axis(pre_const, post_mask)
    axis_post = compute_jet_axis(post_const, post_mask)
    feat_pre = compute_features_with_axis(pre_const, post_mask, axis_pre, kind=feature_kind)
    feat_post = compute_features_with_axis(post_const, post_mask, axis_post, kind=feature_kind)

    x_std = standardize(feat_post, post_mask, means, stds, clip=float(clip))
    y_std = standardize(feat_pre, post_mask, means, stds, clip=float(clip))
    return x_std.astype(np.float32), y_std.astype(np.float32), post_mask.astype(bool)


class JetDataset(Dataset):
    def __init__(
        self,
        feat_off: np.ndarray,
        feat_hlt: np.ndarray,
        labels: np.ndarray,
        masks_off: np.ndarray,
        masks_hlt: np.ndarray,
        weights: np.ndarray,
    ):
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


class JointJetDataset(Dataset):
    """Joint-training dataset for the shared-encoder model."""

    def __init__(
        self,
        x_hlt: np.ndarray,
        y_off: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ):
        self.x = torch.tensor(x_hlt, dtype=torch.float32)
        self.y_unsmear = torch.tensor(y_off, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, i):
        return {
            "x": self.x[i],
            "y_unsmear": self.y_unsmear[i],
            "mask": self.mask[i],
            "label": self.labels[i],
            "weight": self.weights[i],
        }


def masked_smooth_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute SmoothL1 over valid tokens only."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    diff = F.smooth_l1_loss(pred, tgt, reduction="none")
    num = (diff * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def masked_smooth_l1_wrap_dphi(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    dphi_idx: int,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Use a wrap-aware residual for the dPhi dimension."""
    diff = pred - tgt
    scale = (
        dphi_scale
        if isinstance(dphi_scale, torch.Tensor)
        else torch.tensor(float(dphi_scale), device=pred.device, dtype=pred.dtype)
    )
    diff_phi = wrap_dphi_torch(diff[..., int(dphi_idx)] * scale) / scale
    diff = diff.clone()
    diff[..., int(dphi_idx)] = diff_phi
    per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    m = mask.to(pred.dtype).unsqueeze(-1)
    num = (per * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def _weighted_mean(values: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Optionally apply a weighted mean over the batch dimension."""
    if weight is None:
        return values.mean()
    w = weight.to(dtype=values.dtype)
    return (values * w).sum() / w.sum().clamp_min(1e-12)


def _batch_weight_total(weight: Optional[torch.Tensor], batch_size: int) -> float:
    if weight is None:
        return float(batch_size)
    return max(float(weight.detach().sum().item()), 1e-12)


def _maybe_sample_weight(
    sample_weight: Optional[torch.Tensor],
    enabled: bool,
) -> Optional[torch.Tensor]:
    return sample_weight if bool(enabled) else None


def _loss_denominator(
    sample_weight: Optional[torch.Tensor],
    batch_size: int,
    *,
    use_sample_weight: bool,
) -> float:
    if bool(use_sample_weight):
        return _batch_weight_total(sample_weight, batch_size)
    return float(batch_size)


def _auc_scores(
    labels: Sequence[float] | np.ndarray,
    preds: Sequence[float] | np.ndarray,
    sample_weight: Optional[Sequence[float] | np.ndarray] = None,
    *,
    use_sample_weight: bool,
) -> tuple[float, float]:
    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    auc = float(roc_auc_score(labels_np, preds_np))
    auc_weighted = auc
    if bool(use_sample_weight) and sample_weight is not None:
        auc_weighted = float(roc_auc_score(labels_np, preds_np, sample_weight=np.asarray(sample_weight, dtype=np.float64)))
    return auc, auc_weighted


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-event BCE followed by an event-weighted mean."""
    per_event = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _weighted_mean(per_event, sample_weight)


def _per_jet_masked_smooth_l1(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked SmoothL1 for each jet."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    diff = F.smooth_l1_loss(pred, tgt, reduction="none")
    num = (diff * m).sum(dim=(1, 2))
    den = mask.to(pred.dtype).sum(dim=1).clamp_min(1.0)
    return num / den


def _per_jet_masked_smooth_l1_per_feature(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked SmoothL1 for each jet and feature."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    diff = F.smooth_l1_loss(pred, tgt, reduction="none")
    num = (diff * m).sum(dim=1)
    den = mask.to(pred.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    return num / den


def _per_jet_masked_smooth_l1_wrap_dphi(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    dphi_idx: int,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return wrap-aware masked SmoothL1 for each jet."""
    diff = pred - tgt
    scale = (
        dphi_scale
        if isinstance(dphi_scale, torch.Tensor)
        else torch.tensor(float(dphi_scale), device=pred.device, dtype=pred.dtype)
    )
    diff_phi = wrap_dphi_torch(diff[..., int(dphi_idx)] * scale) / scale
    diff = diff.clone()
    diff[..., int(dphi_idx)] = diff_phi
    per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    m = mask.to(pred.dtype).unsqueeze(-1)
    num = (per * m).sum(dim=(1, 2))
    den = mask.to(pred.dtype).sum(dim=1).clamp_min(1.0)
    return num / den


def _per_jet_masked_smooth_l1_wrap_dphi_per_feature(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    dphi_idx: int,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return wrap-aware masked SmoothL1 for each jet and feature."""
    diff = pred - tgt
    scale = (
        dphi_scale
        if isinstance(dphi_scale, torch.Tensor)
        else torch.tensor(float(dphi_scale), device=pred.device, dtype=pred.dtype)
    )
    diff_phi = wrap_dphi_torch(diff[..., int(dphi_idx)] * scale) / scale
    diff = diff.clone()
    diff[..., int(dphi_idx)] = diff_phi
    per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    m = mask.to(pred.dtype).unsqueeze(-1)
    num = (per * m).sum(dim=1)
    den = mask.to(pred.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    return num / den


def _resolve_feature_loss_weights(
    feat_names: Sequence[str],
    feature_loss_weights: Optional[Sequence[float] | np.ndarray],
) -> np.ndarray:
    """Validate and normalize per-feature regression weights."""
    if feature_loss_weights is None:
        return np.ones(len(feat_names), dtype=np.float32)
    arr = np.asarray(feature_loss_weights, dtype=np.float32).reshape(-1)
    if arr.shape[0] != len(feat_names):
        raise ValueError(
            f"Expected {len(feat_names)} feature weights, got {arr.shape[0]} for features {list(feat_names)}"
        )
    if np.any(arr < 0.0):
        raise ValueError("Feature loss weights must be non-negative.")
    return arr


def get_scheduler(opt, warmup: int, total: int):
    def lr_lambda(ep):
        if ep < int(warmup):
            return float(ep + 1) / float(max(1, warmup))
        return 0.5 * (1.0 + np.cos(np.pi * (ep - warmup) / float(max(1, total - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_standard(
    model,
    loader,
    opt,
    device,
    feat_key: str,
    mask_key: str,
    *,
    use_sample_weight_for_all_losses: bool = True,
):
    model.train()
    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_den = 0.0
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x, m).squeeze(-1)
        loss = weighted_bce_with_logits(logits, y, sample_weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        den = _batch_weight_total(w, int(y.shape[0]))
        total_loss += float(loss.item()) * den
        total_den += den
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
    auc, auc_weighted = _auc_scores(
        labs,
        preds,
        np.asarray(weights, dtype=np.float64),
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return total_loss / max(total_den, 1e-12), auc, auc_weighted


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    feat_key: str,
    mask_key: str,
    *,
    use_sample_weight_for_all_losses: bool = True,
):
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
    preds_np = np.asarray(preds)
    labs_np = np.asarray(labs)
    weights_np = np.asarray(weights, dtype=np.float64)
    auc, auc_weighted = _auc_scores(
        labs_np,
        preds_np,
        weights_np,
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return (auc, auc_weighted, preds_np, labs_np, weights_np)


@torch.no_grad()
def eval_standard_model(
    model,
    loader,
    device,
    feat_key: str,
    mask_key: str,
    *,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, float | np.ndarray]:
    """Evaluate a standard classifier and return loss/AUC summaries."""
    model.eval()
    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_den = 0.0
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)
        logits = model(x, m).squeeze(-1)
        loss = weighted_bce_with_logits(logits, y, sample_weight=w)
        den = _batch_weight_total(w, int(y.shape[0]))
        total_loss += float(loss.item()) * den
        total_den += den
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())

    preds_np = np.asarray(preds)
    labs_np = np.asarray(labs)
    weights_np = np.asarray(weights, dtype=np.float64)
    auc, auc_weighted = _auc_scores(
        labs_np,
        preds_np,
        weights_np,
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return {
        "loss": total_loss / max(total_den, 1e-12),
        "auc": auc,
        "auc_weighted": auc_weighted,
        "preds": preds_np,
        "labels": labs_np,
        "weights": weights_np,
    }


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    s_soft = torch.sigmoid(student_logits / float(T))
    t_soft = torch.sigmoid(teacher_logits / float(T))
    per_event = F.binary_cross_entropy(s_soft, t_soft, reduction="none")
    return _weighted_mean(per_event, sample_weight) * (float(T) ** 2)


def attn_loss(
    s_attn: torch.Tensor,
    t_attn: torch.Tensor,
    s_mask: torch.Tensor,
    t_mask: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Align the shape of the pooling-attention distribution."""
    eps = 1e-8
    s_valid = s_attn * s_mask.float()
    t_valid = t_attn * t_mask.float()
    s_ent = -(s_valid * torch.log(s_valid + eps)).sum(dim=1)
    t_ent = -(t_valid * torch.log(t_valid + eps)).sum(dim=1)
    per_event = (s_ent - t_ent) ** 2 + (s_valid.max(dim=1)[0] - t_valid.max(dim=1)[0]) ** 2
    return _weighted_mean(per_event, sample_weight)


def train_kd(
    student,
    teacher,
    loader,
    opt,
    device,
    cfg: dict,
    *,
    use_sample_weight_for_all_losses: bool = True,
):
    student.train()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"].get("alpha_attn", 0.0))

    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_den = 0.0
    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        x_off = batch["off"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)

        with torch.no_grad():
            if a_attn > 0.0:
                t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
                t_logits = t_logits.squeeze(-1)
            else:
                t_logits = teacher(x_off, m_off).squeeze(-1)

        opt.zero_grad(set_to_none=True)
        if a_attn > 0.0:
            s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
            s_logits = s_logits.squeeze(-1)
        else:
            s_logits = student(x_hlt, m_hlt).squeeze(-1)
        aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
        loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=aux_weight)
        loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
        loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
        if a_attn > 0.0:
            loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=aux_weight)
        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        total_loss += float(loss.item()) * den
        total_den += den
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
    auc, auc_weighted = _auc_scores(
        labs,
        preds,
        np.asarray(weights, dtype=np.float64),
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return total_loss / max(total_den, 1e-12), auc, auc_weighted


def train_kd_detailed(
    student,
    teacher,
    loader,
    opt,
    device,
    cfg: dict,
    *,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, float]:
    """Train one KD epoch and return total/hard/kd/attn terms."""
    student.train()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"].get("alpha_attn", 0.0))

    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_hard = 0.0
    total_kd = 0.0
    total_attn = 0.0
    total_mix_den = 0.0
    total_hard_den = 0.0
    total_aux_den = 0.0
    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        x_off = batch["off"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)

        with torch.no_grad():
            if a_attn > 0.0:
                t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
                t_logits = t_logits.squeeze(-1)
            else:
                t_logits = teacher(x_off, m_off).squeeze(-1)

        opt.zero_grad(set_to_none=True)
        if a_attn > 0.0:
            s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
            s_logits = s_logits.squeeze(-1)
        else:
            s_logits = student(x_hlt, m_hlt).squeeze(-1)
        aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
        loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=aux_weight)
        loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
        loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
        if a_attn > 0.0:
            loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=aux_weight)
        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        hard_den = _batch_weight_total(w, int(y.shape[0]))
        aux_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        mix_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        total_loss += float(loss.item()) * mix_den
        total_hard += float(loss_hard.item()) * hard_den
        total_kd += float(loss_kd.item()) * aux_den
        total_attn += float(loss_a.item()) * aux_den
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
        total_mix_den += mix_den
        total_hard_den += hard_den
        total_aux_den += aux_den
    auc, auc_weighted = _auc_scores(
        labs,
        preds,
        np.asarray(weights, dtype=np.float64),
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return {
        "total": total_loss / max(total_mix_den, 1e-12),
        "hard": total_hard / max(total_hard_den, 1e-12),
        "kd": total_kd / max(total_aux_den, 1e-12),
        "attn": total_attn / max(total_aux_den, 1e-12),
        "auc": auc,
        "auc_weighted": auc_weighted,
    }


@torch.no_grad()
def eval_kd_student(
    student,
    teacher,
    loader,
    device,
    cfg: dict,
    *,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, float | np.ndarray]:
    """Evaluate a KD student and return total/hard/kd/attn terms."""
    student.eval()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"].get("alpha_attn", 0.0))

    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_hard = 0.0
    total_kd = 0.0
    total_attn = 0.0
    total_mix_den = 0.0
    total_hard_den = 0.0
    total_aux_den = 0.0
    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        x_off = batch["off"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)

        if a_attn > 0.0:
            t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
            t_logits = t_logits.squeeze(-1)
            s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
            s_logits = s_logits.squeeze(-1)
        else:
            t_logits = teacher(x_off, m_off).squeeze(-1)
            s_logits = student(x_hlt, m_hlt).squeeze(-1)
        aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
        loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=aux_weight)
        loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
        loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
        if a_attn > 0.0:
            loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=aux_weight)
        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a

        hard_den = _batch_weight_total(w, int(y.shape[0]))
        aux_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        mix_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        total_loss += float(loss.item()) * mix_den
        total_hard += float(loss_hard.item()) * hard_den
        total_kd += float(loss_kd.item()) * aux_den
        total_attn += float(loss_a.item()) * aux_den
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
        total_mix_den += mix_den
        total_hard_den += hard_den
        total_aux_den += aux_den

    preds_np = np.asarray(preds)
    labs_np = np.asarray(labs)
    weights_np = np.asarray(weights, dtype=np.float64)
    auc, auc_weighted = _auc_scores(
        labs_np,
        preds_np,
        weights_np,
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return {
        "total": total_loss / max(total_mix_den, 1e-12),
        "hard": total_hard / max(total_hard_den, 1e-12),
        "kd": total_kd / max(total_aux_den, 1e-12),
        "attn": total_attn / max(total_aux_den, 1e-12),
        "auc": auc,
        "auc_weighted": auc_weighted,
        "preds": preds_np,
        "labels": labs_np,
        "weights": weights_np,
    }


def compute_roc(
    y: np.ndarray,
    p: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    roc_kwargs = {}
    auc_weighted = float(roc_auc_score(y, p))
    if sample_weight is not None:
        roc_kwargs["sample_weight"] = sample_weight
        auc_weighted = float(roc_auc_score(y, p, sample_weight=sample_weight))
    fpr, tpr, _ = roc_curve(y, p, **roc_kwargs)
    auc = float(roc_auc_score(y, p))
    return fpr, tpr, auc, auc_weighted


def fpr_at_target_tpr(tpr: np.ndarray, fpr: np.ndarray, target_tpr: float) -> float:
    """Interpolate the FPR at a target TPR."""
    tpr = np.asarray(tpr, dtype=np.float64)
    fpr = np.asarray(fpr, dtype=np.float64)
    order = np.argsort(tpr)
    tpr_sorted = tpr[order]
    fpr_sorted = fpr[order]
    tpr_unique, unique_idx = np.unique(tpr_sorted, return_index=True)
    fpr_unique = fpr_sorted[unique_idx]
    return float(np.interp(float(target_tpr), tpr_unique, fpr_unique))


def resolve_early_stop_metric_name(metric_name: str) -> str:
    """Validate and normalize the early-stopping metric name."""
    normalized = str(metric_name).strip()
    if normalized not in {"val_auc", "val_auc_weighted"}:
        raise ValueError(f"Unsupported early_stop_metric: {metric_name}")
    return normalized


def select_early_stop_score(
    metric_name: str,
    *,
    val_auc: float,
    val_auc_weighted: float,
) -> float:
    """Return the validation score used for early stopping."""
    normalized = resolve_early_stop_metric_name(metric_name)
    if normalized == "val_auc":
        return float(val_auc)
    return float(val_auc_weighted)


def build_epoch_train_arrays(
    train_const_raw: np.ndarray,
    train_mask_raw: np.ndarray,
    cfg: HLTEffectsCfg,
    *,
    feature_kind: str,
    means: np.ndarray,
    stds: np.ndarray,
    seed: int,
    epoch: int,
    fixed_x: np.ndarray,
    fixed_y: np.ndarray,
    fixed_mask: np.ndarray,
    seed_stride: int = 1,
    resmear_each_epoch: bool = False,
    clip: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regenerate training inputs per epoch, or return fixed arrays if disabled."""
    if not bool(resmear_each_epoch):
        return (
            np.asarray(fixed_x, dtype=np.float32),
            np.asarray(fixed_y, dtype=np.float32),
            np.asarray(fixed_mask, dtype=bool),
        )

    epoch_seed = int(seed + (int(epoch) - 1) * int(seed_stride))
    return build_unsmear_epoch_arrays(
        train_const_raw,
        train_mask_raw,
        cfg,
        feature_kind=str(feature_kind),
        means=np.asarray(means, dtype=np.float32),
        stds=np.asarray(stds, dtype=np.float32),
        seed=epoch_seed,
        clip=float(clip),
    )


def make_epoch_hlt_train_loader(
    *,
    epoch: int,
    batch_size: int,
    feat_off_train: np.ndarray,
    off_mask_train: np.ndarray,
    labels_train: np.ndarray,
    weights_train: np.ndarray,
    train_const_raw: np.ndarray,
    train_mask_raw: np.ndarray,
    cfg: HLTEffectsCfg,
    feature_kind: str,
    means: np.ndarray,
    stds: np.ndarray,
    seed: int,
    fixed_feat_hlt_train: np.ndarray,
    fixed_hlt_mask_train: np.ndarray,
    seed_stride: int = 1,
    resmear_each_epoch: bool = False,
    clip: float = 10.0,
) -> DataLoader:
    """Build the HLT training loader for the current epoch."""
    x_ep, _y_ep, m_ep = build_epoch_train_arrays(
        train_const_raw,
        train_mask_raw,
        cfg,
        feature_kind=feature_kind,
        means=means,
        stds=stds,
        seed=seed,
        epoch=epoch,
        fixed_x=fixed_feat_hlt_train,
        fixed_y=feat_off_train,
        fixed_mask=fixed_hlt_mask_train,
        seed_stride=seed_stride,
        resmear_each_epoch=resmear_each_epoch,
        clip=clip,
    )
    ds = JetDataset(feat_off_train, x_ep, labels_train, off_mask_train, m_ep, weights_train)
    return DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)


def make_epoch_joint_train_loader(
    *,
    epoch: int,
    batch_size: int,
    labels_train: np.ndarray,
    weights_train: np.ndarray,
    train_const_raw: np.ndarray,
    train_mask_raw: np.ndarray,
    cfg: HLTEffectsCfg,
    feature_kind: str,
    means: np.ndarray,
    stds: np.ndarray,
    seed: int,
    fixed_x_train: np.ndarray,
    fixed_y_train: np.ndarray,
    fixed_mask_train: np.ndarray,
    seed_stride: int = 1,
    resmear_each_epoch: bool = False,
    clip: float = 10.0,
) -> DataLoader:
    """Build the joint-training loader for the current epoch."""
    x_ep, y_ep, m_ep = build_epoch_train_arrays(
        train_const_raw,
        train_mask_raw,
        cfg,
        feature_kind=feature_kind,
        means=means,
        stds=stds,
        seed=seed,
        epoch=epoch,
        fixed_x=fixed_x_train,
        fixed_y=fixed_y_train,
        fixed_mask=fixed_mask_train,
        seed_stride=seed_stride,
        resmear_each_epoch=resmear_each_epoch,
        clip=clip,
    )
    ds = JointJetDataset(x_ep, y_ep, m_ep, labels_train, weights_train)
    return DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)


def regression_loss_terms(
    mu: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    *,
    feat_names: list[str],
    feat_means: Optional[np.ndarray] = None,
    feat_stds: np.ndarray,
    sample_weight: Optional[torch.Tensor] = None,
    feature_loss_weights: Optional[Sequence[float] | np.ndarray] = None,
    phys_consistency_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Unsmear regression loss terms for the joint model."""
    idx_map = {n: i for i, n in enumerate(list(feat_names))}
    dphi_idx = idx_map.get("dPhi", None)
    deta_idx = idx_map.get("dEta", None)
    dr_idx = idx_map.get("dR", None)
    feat_std_arr = np.asarray(feat_stds, dtype=np.float32)
    feat_mean_arr = (
        np.zeros(len(feat_names), dtype=np.float32)
        if feat_means is None
        else np.asarray(feat_means, dtype=np.float32)
    )
    dphi_scale = float(feat_std_arr[int(dphi_idx)]) if dphi_idx is not None else 1.0
    feature_weight_arr = _resolve_feature_loss_weights(feat_names, feature_loss_weights)
    feature_weight_tensor = torch.as_tensor(feature_weight_arr, device=mu.device, dtype=mu.dtype)
    feat_std_tensor = torch.as_tensor(feat_std_arr, device=mu.device, dtype=mu.dtype)
    feat_mean_tensor = torch.as_tensor(feat_mean_arr, device=mu.device, dtype=mu.dtype)
    phys_weight = float(phys_consistency_weight)
    if phys_weight < 0.0:
        raise ValueError("phys_consistency_weight must be non-negative.")

    if dphi_idx is not None:
        base_per_jet = _per_jet_masked_smooth_l1_wrap_dphi(
            mu,
            y,
            m,
            dphi_idx=int(dphi_idx),
            dphi_scale=dphi_scale,
        )
        base_per_jet_by_feature = _per_jet_masked_smooth_l1_wrap_dphi_per_feature(
            mu,
            y,
            m,
            dphi_idx=int(dphi_idx),
            dphi_scale=dphi_scale,
        )
    else:
        base_per_jet = _per_jet_masked_smooth_l1(mu, y, m)
        base_per_jet_by_feature = _per_jet_masked_smooth_l1_per_feature(mu, y, m)
    base_unweighted = _weighted_mean(base_per_jet, sample_weight)

    feature_losses: dict[str, torch.Tensor] = {}
    weighted_feature_losses: dict[str, torch.Tensor] = {}
    for feat_idx, feat_name in enumerate(list(feat_names)):
        feat_loss = _weighted_mean(base_per_jet_by_feature[:, feat_idx], sample_weight)
        feature_losses[str(feat_name)] = feat_loss
        weighted_feature_losses[str(feat_name)] = feat_loss * feature_weight_tensor[feat_idx]
    base = sum(weighted_feature_losses.values(), torch.zeros((), device=mu.device, dtype=mu.dtype))

    cons_raw = torch.zeros((), device=mu.device, dtype=mu.dtype)
    if (dr_idx is not None) and (deta_idx is not None) and (dphi_idx is not None):
        deta_raw = mu[..., int(deta_idx)] * feat_std_tensor[int(deta_idx)] + feat_mean_tensor[int(deta_idx)]
        # 物理空间中的 dPhi 需要再次 wrap，避免跨越 pi 边界时把等价角度当成大残差。
        dphi_raw = wrap_dphi_torch(
            mu[..., int(dphi_idx)] * feat_std_tensor[int(dphi_idx)] + feat_mean_tensor[int(dphi_idx)]
        )
        dR_pred_raw = mu[..., int(dr_idx)] * feat_std_tensor[int(dr_idx)] + feat_mean_tensor[int(dr_idx)]
        dR_cons_raw = torch.sqrt(deta_raw**2 + dphi_raw**2 + 1e-12)
        cons_per_jet = _per_jet_masked_smooth_l1(
            dR_pred_raw.unsqueeze(-1),
            dR_cons_raw.unsqueeze(-1),
            m,
        )
        cons_raw = _weighted_mean(cons_per_jet, sample_weight)
    cons = cons_raw * phys_weight

    return {
        "total": base + cons,
        "base": base,
        "base_unweighted": base_unweighted,
        "phys": cons,
        "dr_cons_raw": cons_raw,
        "feature_losses": feature_losses,
        "weighted_feature_losses": weighted_feature_losses,
        "feature_loss_weights": {
            str(feat_name): float(feature_weight_arr[idx]) for idx, feat_name in enumerate(list(feat_names))
        },
    }


def make_opt(
    model,
    *,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    epochs: int,
):
    """Create the optimizer and scheduler."""
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))
    return opt, sch


def _flatten_grad_list(grads: list[Optional[torch.Tensor]], params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten a list of gradients into a single vector."""
    parts = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros_like(p, memory_format=torch.contiguous_format).reshape(-1))
        else:
            parts.append(g.detach().reshape(-1))
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts, dim=0)


def get_shared_grad_groups(model) -> dict[str, dict[str, Any]]:
    """Return parameter groups for different parts of the shared trunk."""
    groups: dict[str, dict[str, Any]] = {}

    input_proj_params = list(model.input_proj.parameters()) if hasattr(model, "input_proj") else []
    trunk = None
    trunk_name = None
    if hasattr(model, "encoder"):
        trunk = model.encoder
        trunk_name = "encoder"
    elif hasattr(model, "transformer"):
        trunk = model.transformer
        trunk_name = "transformer"

    if trunk is None or not hasattr(trunk, "layers"):
        raise ValueError("Model does not expose encoder/transformer layers for gradient probing.")

    layers = list(trunk.layers)
    if not layers:
        raise ValueError("Shared trunk has no layers.")

    middle_idx = 1 if len(layers) > 1 else 0
    last_idx = len(layers) - 1

    all_params = list(input_proj_params)
    for layer in layers:
        all_params.extend(list(layer.parameters()))

    groups["shared_all"] = {
        "params": all_params,
        "module": f"input_proj + {trunk_name}.layers[*]",
    }
    groups["input_proj"] = {
        "params": input_proj_params,
        "module": "input_proj",
    }
    groups["layer_1"] = {
        "params": list(layers[middle_idx].parameters()),
        "module": f"{trunk_name}.layers.{middle_idx}",
    }
    groups["layer_last"] = {
        "params": list(layers[last_idx].parameters()),
        "module": f"{trunk_name}.layers.{last_idx}",
    }
    return groups


def _grad_norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vec).item())


def _grad_cosine(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    na = torch.linalg.vector_norm(vec_a)
    nb = torch.linalg.vector_norm(vec_b)
    if float(na.item()) < 1e-12 or float(nb.item()) < 1e-12:
        return float("nan")
    return float(torch.dot(vec_a, vec_b).item() / (na.item() * nb.item()))


def gradient_probe_from_losses(
    model,
    loss_map: dict[str, Optional[torch.Tensor]],
) -> dict[str, Any]:
    """Measure shared-trunk gradient norms and cosines from several loss tensors."""
    groups = get_shared_grad_groups(model)
    active_losses = {k: v for k, v in loss_map.items() if v is not None}

    grad_vectors: dict[str, dict[str, torch.Tensor]] = {}
    norm_rows = []
    for loss_name, loss_tensor in active_losses.items():
        grad_vectors[loss_name] = {}
        for group_name, info in groups.items():
            params = list(info["params"])
            if not params:
                vec = torch.zeros(1, device=loss_tensor.device, dtype=loss_tensor.dtype)
            else:
                grads = torch.autograd.grad(
                    loss_tensor,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                vec = _flatten_grad_list(list(grads), params)
            grad_vectors[loss_name][group_name] = vec
            norm_rows.append(
                {
                    "group": group_name,
                    "group_module": info["module"],
                    "loss_component": loss_name,
                    "grad_norm": _grad_norm(vec),
                }
            )

    cosine_rows = []
    loss_names = list(active_losses.keys())
    for i in range(len(loss_names)):
        for j in range(i + 1, len(loss_names)):
            a = loss_names[i]
            b = loss_names[j]
            pair_name = f"{a}_vs_{b}"
            for group_name, info in groups.items():
                cosine_rows.append(
                    {
                        "group": group_name,
                        "group_module": info["module"],
                        "pair": pair_name,
                        "cosine": _grad_cosine(
                            grad_vectors[a][group_name],
                            grad_vectors[b][group_name],
                        ),
                    }
                )

    return {
        "norm_rows": norm_rows,
        "cosine_rows": cosine_rows,
        "group_modules": {k: v["module"] for k, v in groups.items()},
    }


def feature_gradient_probe_from_regression_terms(
    model,
    reg_terms: dict[str, Any],
) -> dict[str, Any]:
    """Build feature-level gradient probes from per-feature regression losses."""
    feature_losses = {
        str(name): loss for name, loss in dict(reg_terms.get("feature_losses", {})).items() if loss is not None
    }
    feature_weights = {
        str(name): float(weight) for name, weight in dict(reg_terms.get("feature_loss_weights", {})).items()
    }
    diag = gradient_probe_from_losses(model, feature_losses)
    diag["scalar_rows"] = [
        {
            "loss_component": str(name),
            "scalar_loss": float(loss.item()),
            "feature_weight": float(feature_weights.get(str(name), 1.0)),
        }
        for name, loss in feature_losses.items()
    ]
    diag["norm_rows"] = [
        {
            **row,
            "feature_weight": float(feature_weights.get(str(row["loss_component"]), 1.0)),
        }
        for row in diag.get("norm_rows", [])
    ]
    return diag


def make_even_interval_batch_indices(total_batches: int, sample_count: int) -> list[int]:
    total = int(total_batches)
    count = int(sample_count)
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))

    raw = np.linspace(0, total - 1, num=count)
    picked = np.clip(np.round(raw).astype(int), 0, total - 1).tolist()

    out: list[int] = []
    seen: set[int] = set()
    for idx in picked:
        ii = int(idx)
        if ii not in seen:
            out.append(ii)
            seen.add(ii)

    if len(out) < count:
        for idx in range(total):
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
            if len(out) >= count:
                break
        out = sorted(out[:count])
    return out


def _gradient_probe_output_paths(prefix: str | Path) -> dict[str, Path]:
    p = Path(prefix)
    ensure_dir(p.parent)
    return {
        "scalar": p.parent / f"{p.name}_scalar_losses.csv",
        "norm": p.parent / f"{p.name}_grad_norms.csv",
        "cos": p.parent / f"{p.name}_grad_cosines.csv",
        "feature_scalar": p.parent / f"{p.name}_feature_scalar_losses.csv",
        "feature_norm": p.parent / f"{p.name}_feature_grad_norms.csv",
        "feature_cos": p.parent / f"{p.name}_feature_grad_cosines.csv",
        "meta": p.parent / f"{p.name}_meta.json",
    }


def _append_gradient_probe_rows(
    *,
    scalar_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    cosine_rows: list[dict[str, Any]],
    diag: dict[str, Any],
    model_name: str,
    split: str,
    epoch: int,
    batch_idx: int,
    sample_idx: int,
    total_batches: int,
):
    base_row = {
        "model": str(model_name),
        "split": str(split),
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "sample_idx": int(sample_idx),
        "total_batches": int(total_batches),
        "batch_fraction": float((int(batch_idx) + 1) / max(1, int(total_batches))),
    }
    scalar_row_list = diag.get("scalar_rows", None)
    if scalar_row_list is not None:
        for row in scalar_row_list:
            scalar_rows.append(
                {
                    **base_row,
                    **row,
                    "loss_component": str(row["loss_component"]),
                    "scalar_loss": float(row["scalar_loss"]),
                }
            )
    else:
        for loss_name, loss_value in diag.get("scalar_losses", {}).items():
            scalar_rows.append(
                {
                    **base_row,
                    "loss_component": str(loss_name),
                    "scalar_loss": float(loss_value),
                }
            )
    for row in diag.get("norm_rows", []):
        norm_rows.append(
            {
                **base_row,
                **row,
                "grad_norm": float(row["grad_norm"]),
            }
        )
    for row in diag.get("cosine_rows", []):
        cosine_rows.append(
            {
                **base_row,
                **row,
                "cosine": float(row["cosine"]),
            }
        )


def _format_epoch_mean_grad_norm_summary(
    norm_rows: Sequence[dict[str, Any]],
    *,
    epoch: int,
    split: str,
    loss_order: Sequence[str],
    loss_weights: Optional[dict[str, float]] = None,
    group: str = "shared_all",
    label: Optional[str] = None,
) -> str:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in norm_rows:
        if int(row.get("epoch", -1)) != int(epoch):
            continue
        if str(row.get("split", "")) != str(split):
            continue
        if str(row.get("group", "")) != str(group):
            continue
        loss_name = str(row.get("loss_component", ""))
        if loss_name not in loss_order:
            continue
        weight = 1.0 if loss_weights is None else float(loss_weights.get(loss_name, 1.0))
        sums[loss_name] = sums.get(loss_name, 0.0) + float(row["grad_norm"]) * weight
        counts[loss_name] = counts.get(loss_name, 0) + 1

    parts = []
    for loss_name in loss_order:
        if counts.get(loss_name, 0) <= 0:
            continue
        parts.append(f"{loss_name}={sums[loss_name] / max(counts[loss_name], 1):.4f}")
    if not parts:
        return ""
    summary_label = str(label) if label is not None else str(split)
    return f"{summary_label}_{group}_mean_grad_norm[{', '.join(parts)}]"


def _write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]):
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_gradient_probe_tables(
    output_prefix: str | Path,
    *,
    scalar_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    cosine_rows: list[dict[str, Any]],
    feature_scalar_rows: Optional[list[dict[str, Any]]] = None,
    feature_norm_rows: Optional[list[dict[str, Any]]] = None,
    feature_cosine_rows: Optional[list[dict[str, Any]]] = None,
    extra_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    paths = _gradient_probe_output_paths(output_prefix)
    scalar_fields = [
        "model",
        "split",
        "epoch",
        "batch_idx",
        "sample_idx",
        "total_batches",
        "batch_fraction",
        "loss_component",
        "scalar_loss",
    ]
    norm_fields = [
        "model",
        "split",
        "epoch",
        "batch_idx",
        "sample_idx",
        "total_batches",
        "batch_fraction",
        "group",
        "group_module",
        "loss_component",
        "grad_norm",
    ]
    cos_fields = [
        "model",
        "split",
        "epoch",
        "batch_idx",
        "sample_idx",
        "total_batches",
        "batch_fraction",
        "group",
        "group_module",
        "pair",
        "cosine",
    ]
    feature_scalar_rows = list(feature_scalar_rows or [])
    feature_norm_rows = list(feature_norm_rows or [])
    feature_cosine_rows = list(feature_cosine_rows or [])

    def _fieldnames(base_fields: list[str], rows: list[dict[str, Any]]) -> list[str]:
        extra = []
        extra_seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in base_fields and key not in extra_seen:
                    extra.append(str(key))
                    extra_seen.add(str(key))
        return list(base_fields) + extra

    scalar_fields = _fieldnames(scalar_fields, list(scalar_rows) + feature_scalar_rows)
    norm_fields = _fieldnames(norm_fields, list(norm_rows) + feature_norm_rows)
    cos_fields = _fieldnames(cos_fields, list(cosine_rows) + feature_cosine_rows)
    _write_csv_rows(paths["scalar"], scalar_fields, list(scalar_rows))
    _write_csv_rows(paths["norm"], norm_fields, list(norm_rows))
    _write_csv_rows(paths["cos"], cos_fields, list(cosine_rows))
    if feature_scalar_rows:
        _write_csv_rows(paths["feature_scalar"], scalar_fields, feature_scalar_rows)
    if feature_norm_rows:
        _write_csv_rows(paths["feature_norm"], norm_fields, feature_norm_rows)
    if feature_cosine_rows:
        _write_csv_rows(paths["feature_cos"], cos_fields, feature_cosine_rows)
    meta = dict(extra_meta or {})
    meta["output_prefix"] = str(Path(output_prefix))
    paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def save_epoch_metrics_table(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        fieldnames = ["epoch"]
        _write_csv_rows(p, fieldnames, [])
        return p

    preferred = [
        "model",
        "epoch",
        "early_stop_metric",
        "best_stop_score",
        "train_loss",
        "train_auc",
        "train_auc_weighted",
        "train_total",
        "train_hard",
        "train_kd",
        "train_attn",
        "train_joint",
        "train_uns",
        "train_phys",
        "train_cls",
        "val_loss",
        "val_total",
        "val_hard",
        "val_kd",
        "val_attn",
        "val_joint",
        "val_uns",
        "val_phys",
        "val_cls",
        "val_auc",
        "val_auc_weighted",
        "best_auc",
        "best_auc_weighted",
        "no_imp",
        "is_best",
        "stopped_after_epoch",
    ]
    present = []
    row_keys = set()
    for row in rows:
        row_keys.update(row.keys())
    for key in preferred:
        if key in row_keys:
            present.append(key)
    for key in sorted(row_keys):
        if key not in present:
            present.append(key)
    _write_csv_rows(p, present, rows)
    return p


def collect_loader_gradient_probes(
    *,
    loader,
    sample_count: int,
    probe_fn: Callable[[dict[str, Any]], dict[str, Any]],
    model_name: str,
    split: str,
    epoch: int,
) -> dict[str, list[dict[str, Any]]]:
    total_batches = len(loader)
    picked = make_even_interval_batch_indices(total_batches, int(sample_count))
    picked_set = set(picked)
    picked_rank = {idx: rank for rank, idx in enumerate(picked)}
    scalar_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    feature_scalar_rows: list[dict[str, Any]] = []
    feature_norm_rows: list[dict[str, Any]] = []
    feature_cosine_rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx not in picked_set:
            continue
        diag = probe_fn(batch)
        _append_gradient_probe_rows(
            scalar_rows=scalar_rows,
            norm_rows=norm_rows,
            cosine_rows=cosine_rows,
            diag=diag,
            model_name=model_name,
            split=split,
            epoch=int(epoch),
            batch_idx=int(batch_idx),
            sample_idx=int(picked_rank[batch_idx]),
            total_batches=int(total_batches),
        )
        feature_diag = diag.get("feature_probe", None)
        if feature_diag is not None:
            _append_gradient_probe_rows(
                scalar_rows=feature_scalar_rows,
                norm_rows=feature_norm_rows,
                cosine_rows=feature_cosine_rows,
                diag=feature_diag,
                model_name=model_name,
                split=split,
                epoch=int(epoch),
                batch_idx=int(batch_idx),
                sample_idx=int(picked_rank[batch_idx]),
                total_batches=int(total_batches),
            )
    return {
        "scalar_rows": scalar_rows,
        "norm_rows": norm_rows,
        "cosine_rows": cosine_rows,
        "feature_scalar_rows": feature_scalar_rows,
        "feature_norm_rows": feature_norm_rows,
        "feature_cosine_rows": feature_cosine_rows,
    }


@torch.no_grad()
def clone_batch_to_cpu(batch: dict[str, Any]) -> dict[str, Any]:
    """Clone a batch to CPU for fixed gradient probing."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    return out


def probe_hlt_kd_gradients(
    student,
    teacher,
    batch: dict[str, Any],
    *,
    device,
    kd_temperature: float,
    kd_alpha: float,
    kd_alpha_attn: float = 0.0,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, Any]:
    """Collect shared-trunk gradient diagnostics for the HLT+KD student."""
    student.eval()
    teacher.eval()

    x_hlt = batch["hlt"].to(device)
    x_off = batch["off"].to(device)
    m_hlt = batch["mask_hlt"].to(device)
    m_off = batch["mask_off"].to(device)
    y = batch["label"].to(device)
    w = batch["weight"].to(device)

    if float(kd_alpha_attn) > 0.0:
        with torch.no_grad():
            teacher_logits, teacher_attn = teacher(x_off, m_off, return_attention=True)
            teacher_logits = teacher_logits.squeeze(-1)
        student_logits, student_attn = student(x_hlt, m_hlt, return_attention=True)
        student_logits = student_logits.squeeze(-1)
    else:
        with torch.no_grad():
            teacher_logits = teacher(x_off, m_off).squeeze(-1)
        student_logits = student(x_hlt, m_hlt).squeeze(-1)

    aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
    hard_loss = weighted_bce_with_logits(student_logits, y, sample_weight=w)
    kd_loss_val = kd_loss(student_logits, teacher_logits, float(kd_temperature), sample_weight=aux_weight)
    attn_loss_val: Optional[torch.Tensor] = None
    if float(kd_alpha_attn) > 0.0:
        attn_loss_val = attn_loss(student_attn, teacher_attn, m_hlt, m_off, sample_weight=aux_weight)
    total_loss = (
        (1.0 - float(kd_alpha)) * hard_loss
        + float(kd_alpha) * kd_loss_val
        + float(kd_alpha_attn) * (torch.zeros_like(hard_loss) if attn_loss_val is None else attn_loss_val)
    )

    out = gradient_probe_from_losses(
        student,
        {
            "hard": hard_loss,
            "kd": kd_loss_val,
            "attn": attn_loss_val,
            "total": total_loss,
        },
    )
    out["scalar_losses"] = {
        "hard": float(hard_loss.item()),
        "kd": float(kd_loss_val.item()),
        "attn": float("nan") if attn_loss_val is None else float(attn_loss_val.item()),
        "total": float(total_loss.item()),
    }
    return out


def probe_joint_gradients(
    model,
    batch: dict[str, Any],
    *,
    device,
    feat_names: list[str],
    feat_means: Optional[np.ndarray] = None,
    feat_stds: np.ndarray,
    feature_loss_weights: Optional[Sequence[float] | np.ndarray] = None,
    joint_phys_weight: float = 0.0,
    teacher=None,
    use_kd: bool = False,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.0,
    kd_alpha_attn: float = 0.0,
    joint_unsmear_weight: float = 1.0,
    joint_cls_weight: float = 1.0,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, Any]:
    """Collect shared-trunk gradient diagnostics for a joint model."""
    model.eval()
    if teacher is not None:
        teacher.eval()

    x = batch["x"].to(device)
    y_uns = batch["y_unsmear"].to(device)
    m = batch["mask"].to(device)
    y_cls = batch["label"].to(device)
    w = batch["weight"].to(device)

    kd_attn_enabled = bool(use_kd) and (teacher is not None) and (float(kd_alpha_attn) > 0.0)
    if kd_attn_enabled:
        reco, logits, student_attn = model(x, m, return_attention=True)
    else:
        reco, logits = model(x, m)
    aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
    reg_terms = regression_loss_terms(
        reco,
        y_uns,
        m,
        feat_names=feat_names,
        feat_means=feat_means,
        feat_stds=feat_stds,
        sample_weight=aux_weight,
        feature_loss_weights=feature_loss_weights,
        phys_consistency_weight=joint_phys_weight,
    )
    hard_loss = weighted_bce_with_logits(logits.squeeze(-1), y_cls, sample_weight=w)

    kd_loss_val: Optional[torch.Tensor] = None
    attn_loss_val: Optional[torch.Tensor] = None
    cls_loss = hard_loss
    if bool(use_kd) and (teacher is not None):
        with torch.no_grad():
            if kd_attn_enabled:
                teacher_logits, teacher_attn = teacher(y_uns, m, return_attention=True)
                teacher_logits = teacher_logits.squeeze(-1)
            else:
                teacher_logits = teacher(y_uns, m).squeeze(-1)
        kd_loss_val = kd_loss(
            logits.squeeze(-1),
            teacher_logits,
            float(kd_temperature),
            sample_weight=aux_weight,
        )
        if kd_attn_enabled:
            attn_loss_val = attn_loss(student_attn, teacher_attn, m, m, sample_weight=aux_weight)
        cls_loss = (
            (1.0 - float(kd_alpha)) * hard_loss
            + float(kd_alpha) * kd_loss_val
            + float(kd_alpha_attn) * (torch.zeros_like(hard_loss) if attn_loss_val is None else attn_loss_val)
        )

    total_loss = float(joint_unsmear_weight) * reg_terms["total"] + float(joint_cls_weight) * cls_loss
    out = gradient_probe_from_losses(
        model,
        {
            "unsmear": reg_terms["total"],
            "phys": reg_terms["phys"],
            "hard": hard_loss,
            "kd": kd_loss_val,
            "attn": attn_loss_val,
            "total": total_loss,
        },
    )
    out["scalar_losses"] = {
        "unsmear": float(reg_terms["total"].item()),
        "phys": float(reg_terms["phys"].item()),
        "hard": float(hard_loss.item()),
        "kd": float("nan") if kd_loss_val is None else float(kd_loss_val.item()),
        "attn": float("nan") if attn_loss_val is None else float(attn_loss_val.item()),
        "total": float(total_loss.item()),
    }
    out["feature_probe"] = feature_gradient_probe_from_regression_terms(model, reg_terms)
    return out


def train_or_load_standard_model(
    name: str,
    model,
    ckpt_path: str | Path,
    train_loader,
    val_loader,
    *,
    device,
    feat_key: str,
    mask_key: str,
    allow_load: bool,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    epochs: int,
    patience: int,
    early_stop_metric: str = "val_auc",
    use_sample_weight_for_all_losses: bool = True,
    train_loader_factory: Optional[Callable[[int], DataLoader]] = None,
    epoch_metrics_path: str | Path | None = None,
):
    """Train or load a standard classification model."""
    early_stop_metric = resolve_early_stop_metric_name(early_stop_metric)
    if bool(allow_load) and Path(ckpt_path).is_file():
        load_checkpoint(model, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
        if epoch_metrics_path is not None and not Path(epoch_metrics_path).is_file():
            print(f"[{name}] Epoch-metrics table not found for the loaded checkpoint. Rerun training with loading disabled to regenerate it.")
        return model

    opt, sch = make_opt(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_epochs=int(warmup_epochs),
        epochs=int(epochs),
    )
    best_auc, best_auc_weighted, best_stop_score, best_state, no_imp = 0.0, 0.0, float("-inf"), None, 0
    metrics_rows: list[dict[str, Any]] = []
    completed_epochs = 0
    for ep in range(1, int(epochs) + 1):
        epoch_train_loader = train_loader_factory(ep) if train_loader_factory is not None else train_loader
        loss, train_auc, train_auc_weighted = train_standard(
            model,
            epoch_train_loader,
            opt,
            device,
            feat_key,
            mask_key,
            use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        )
        sch.step()
        val_res = eval_standard_model(
            model,
            val_loader,
            device,
            feat_key,
            mask_key,
            use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        )
        val_loss = float(val_res["loss"])
        val_auc = float(val_res["auc"])
        val_auc_weighted = float(val_res["auc_weighted"])
        stop_score = select_early_stop_score(
            early_stop_metric,
            val_auc=float(val_auc),
            val_auc_weighted=float(val_auc_weighted),
        )
        improved = bool(stop_score > best_stop_score + 1e-4)
        if stop_score > best_stop_score + 1e-4:
            best_auc = float(val_auc)
            best_auc_weighted = float(val_auc_weighted)
            best_stop_score = float(stop_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        metrics_rows.append(
            {
                "model": str(name),
                "epoch": int(ep),
                "early_stop_metric": str(early_stop_metric),
                "best_stop_score": float(best_stop_score),
                "train_loss": float(loss),
                "train_auc": float(train_auc),
                "train_auc_weighted": float(train_auc_weighted),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "val_auc_weighted": float(val_auc_weighted),
                "best_auc": float(best_auc),
                "best_auc_weighted": float(best_auc_weighted),
                "no_imp": int(no_imp),
                "is_best": int(improved),
            }
        )
        completed_epochs = int(ep)
        if ep == 1 or ep % 2 == 0:
            print(
                f"[{name}] ep={ep:03d} train_loss={loss:.5f} "
                f"train_auc={train_auc:.5f} train_auc_w={train_auc_weighted:.5f} "
                f"val_loss={val_loss:.5f} val_auc={val_auc:.5f} val_auc_w={val_auc_weighted:.5f} "
                f"monitor={early_stop_metric} best_monitor={best_stop_score:.5f} no_imp={no_imp}"
            )
        if no_imp >= int(patience):
            print(f"[{name}] Early stopping")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if metrics_rows:
        metrics_rows[-1]["stopped_after_epoch"] = int(completed_epochs)
    if epoch_metrics_path is not None:
        save_epoch_metrics_table(epoch_metrics_path, metrics_rows)
    save_checkpoint(
        model,
        ckpt_path,
        extra={
            "best_val_auc": float(best_auc),
            "best_val_auc_weighted": float(best_auc_weighted),
            "early_stop_metric": str(early_stop_metric),
            "best_stop_score": float(best_stop_score),
            "use_sample_weight_for_all_losses": bool(use_sample_weight_for_all_losses),
        },
    )
    print(f"Saved checkpoint: {ckpt_path}")
    return model


def train_or_load_kd_standard_model(
    name: str,
    student,
    teacher,
    ckpt_path: str | Path,
    train_loader,
    val_loader,
    *,
    device,
    allow_load: bool,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    epochs: int,
    patience: int,
    early_stop_metric: str = "val_auc",
    use_sample_weight_for_all_losses: bool = True,
    kd_temperature: float,
    kd_alpha: float,
    kd_alpha_attn: float,
    train_loader_factory: Optional[Callable[[int], DataLoader]] = None,
    grad_probe_cfg: Optional[dict[str, Any]] = None,
    epoch_metrics_path: str | Path | None = None,
):
    """Train or load the KD baseline classifier."""
    early_stop_metric = resolve_early_stop_metric_name(early_stop_metric)
    probe_cfg = dict(grad_probe_cfg or {})
    probe_prefix = probe_cfg.get("output_prefix", None)
    probe_name = str(probe_cfg.get("model_name", name))
    train_probe_batches = int(probe_cfg.get("train_batches_per_epoch", 0))
    val_probe_batches = int(probe_cfg.get("val_batches_per_epoch", 0))
    probe_enabled = probe_prefix is not None and (train_probe_batches > 0 or val_probe_batches > 0)
    if bool(allow_load) and Path(ckpt_path).is_file():
        load_checkpoint(student, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
        if epoch_metrics_path is not None and not Path(epoch_metrics_path).is_file():
            print(f"[{name}] Epoch-metrics table not found for the loaded checkpoint. Rerun training with loading disabled to regenerate it.")
        if probe_enabled:
            probe_paths = _gradient_probe_output_paths(probe_prefix)
            if not (probe_paths["scalar"].is_file() and probe_paths["norm"].is_file() and probe_paths["cos"].is_file()):
                print(f"[{name}] Gradient probe tables not found for the loaded checkpoint. Rerun training with loading disabled to regenerate them.")
        return student

    opt, sch = make_opt(
        student,
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_epochs=int(warmup_epochs),
        epochs=int(epochs),
    )
    best_auc, best_auc_weighted, best_stop_score, best_state, no_imp = 0.0, 0.0, float("-inf"), None, 0
    kd_cfg = {
        "kd": {
            "temperature": float(kd_temperature),
            "alpha_kd": float(kd_alpha),
            "alpha_attn": float(kd_alpha_attn),
        }
    }
    probe_scalar_rows: list[dict[str, Any]] = []
    probe_norm_rows: list[dict[str, Any]] = []
    probe_cosine_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    completed_epochs = 0
    for ep in range(1, int(epochs) + 1):
        epoch_train_loader = train_loader_factory(ep) if train_loader_factory is not None else train_loader
        train_probe_idx = make_even_interval_batch_indices(len(epoch_train_loader), train_probe_batches) if probe_enabled else []
        train_probe_set = set(train_probe_idx)
        train_probe_rank = {idx: rank for rank, idx in enumerate(train_probe_idx)}
        student.train()
        teacher.eval()
        T = float(kd_cfg["kd"]["temperature"])
        a_kd = float(kd_cfg["kd"]["alpha_kd"])
        a_attn = float(kd_cfg["kd"].get("alpha_attn", 0.0))
        preds, labs, weights = [], [], []
        total_loss = 0.0
        total_hard = 0.0
        total_kd = 0.0
        total_attn = 0.0
        total_mix_den = 0.0
        total_hard_den = 0.0
        total_aux_den = 0.0
        for batch_idx, batch in enumerate(epoch_train_loader):
            x_hlt = batch["hlt"].to(device)
            x_off = batch["off"].to(device)
            m_hlt = batch["mask_hlt"].to(device)
            m_off = batch["mask_off"].to(device)
            y = batch["label"].to(device)
            w = batch["weight"].to(device)

            with torch.no_grad():
                if a_attn > 0.0:
                    t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
                    t_logits = t_logits.squeeze(-1)
                else:
                    t_logits = teacher(x_off, m_off).squeeze(-1)

            opt.zero_grad(set_to_none=True)
            if a_attn > 0.0:
                s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
                s_logits = s_logits.squeeze(-1)
            else:
                s_logits = student(x_hlt, m_hlt).squeeze(-1)
            aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
            loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=aux_weight)
            loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
            loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
            if a_attn > 0.0:
                loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=aux_weight)
            loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a

            if batch_idx in train_probe_set:
                diag = gradient_probe_from_losses(
                    student,
                    {
                        "hard": loss_hard,
                        "kd": loss_kd,
                        "attn": loss_a if a_attn > 0.0 else None,
                        "total": loss,
                    },
                )
                diag["scalar_losses"] = {
                    "hard": float(loss_hard.item()),
                    "kd": float(loss_kd.item()),
                    "attn": float(loss_a.item()) if a_attn > 0.0 else float("nan"),
                    "total": float(loss.item()),
                }
                _append_gradient_probe_rows(
                    scalar_rows=probe_scalar_rows,
                    norm_rows=probe_norm_rows,
                    cosine_rows=probe_cosine_rows,
                    diag=diag,
                    model_name=probe_name,
                    split="train",
                    epoch=int(ep),
                    batch_idx=int(batch_idx),
                    sample_idx=int(train_probe_rank[batch_idx]),
                    total_batches=int(len(epoch_train_loader)),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            hard_den = _batch_weight_total(w, int(y.shape[0]))
            aux_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
            mix_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
            total_loss += float(loss.item()) * mix_den
            total_hard += float(loss_hard.item()) * hard_den
            total_kd += float(loss_kd.item()) * aux_den
            total_attn += float(loss_a.item()) * aux_den
            preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
            labs.extend(y.detach().cpu().numpy().flatten())
            weights.extend(w.detach().cpu().numpy().flatten())
            total_mix_den += mix_den
            total_hard_den += hard_den
            total_aux_den += aux_den
        train_auc, train_auc_weighted = _auc_scores(
            labs,
            preds,
            np.asarray(weights, dtype=np.float64),
            use_sample_weight=use_sample_weight_for_all_losses,
        )
        train_res = {
            "total": total_loss / max(total_mix_den, 1e-12),
            "hard": total_hard / max(total_hard_den, 1e-12),
            "kd": total_kd / max(total_aux_den, 1e-12),
            "attn": total_attn / max(total_aux_den, 1e-12),
            "auc": float(train_auc),
            "auc_weighted": float(train_auc_weighted),
        }
        sch.step()
        val_res = eval_kd_student(
            student,
            teacher,
            val_loader,
            device,
            kd_cfg,
            use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        )
        if probe_enabled and val_probe_batches > 0:
            val_probe_rows = collect_loader_gradient_probes(
                loader=val_loader,
                sample_count=val_probe_batches,
                probe_fn=lambda batch: probe_hlt_kd_gradients(
                    student,
                    teacher,
                    batch,
                    device=device,
                    kd_temperature=float(kd_temperature),
                    kd_alpha=float(kd_alpha),
                    kd_alpha_attn=float(kd_alpha_attn),
                    use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
                ),
                model_name=probe_name,
                split="val",
                epoch=int(ep),
            )
            probe_scalar_rows.extend(val_probe_rows["scalar_rows"])
            probe_norm_rows.extend(val_probe_rows["norm_rows"])
            probe_cosine_rows.extend(val_probe_rows["cosine_rows"])
        kd_grad_loss_weights = {
            "hard": float(1.0 - kd_alpha),
            "kd": float(kd_alpha),
            "attn": float(kd_alpha_attn),
        }
        train_grad_norm_summary = _format_epoch_mean_grad_norm_summary(
            probe_norm_rows,
            epoch=int(ep),
            split="train",
            loss_order=["hard", "kd", "attn"],
            loss_weights=kd_grad_loss_weights,
            label="train",
        )
        val_grad_norm_summary = _format_epoch_mean_grad_norm_summary(
            probe_norm_rows,
            epoch=int(ep),
            split="val",
            loss_order=["hard", "kd", "attn"],
            loss_weights=kd_grad_loss_weights,
            label="val",
        )
        grad_norm_suffix = ""
        grad_norm_parts = [part for part in [train_grad_norm_summary, val_grad_norm_summary] if part]
        if grad_norm_parts:
            grad_norm_suffix = " " + " ".join(grad_norm_parts)
        val_auc = float(val_res["auc"])
        stop_score = select_early_stop_score(
            early_stop_metric,
            val_auc=float(val_auc),
            val_auc_weighted=float(val_res["auc_weighted"]),
        )
        improved = bool(stop_score > best_stop_score + 1e-4)
        if stop_score > best_stop_score + 1e-4:
            best_auc = float(val_auc)
            best_auc_weighted = float(val_res["auc_weighted"])
            best_stop_score = float(stop_score)
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        metrics_rows.append(
            {
                "model": str(name),
                "epoch": int(ep),
                "early_stop_metric": str(early_stop_metric),
                "best_stop_score": float(best_stop_score),
                "train_total": float(train_res["total"]),
                "train_hard": float(train_res["hard"]),
                "train_kd": float(train_res["kd"]),
                "train_attn": float(train_res["attn"]),
                "train_auc": float(train_res["auc"]),
                "train_auc_weighted": float(train_res["auc_weighted"]),
                "val_total": float(val_res["total"]),
                "val_hard": float(val_res["hard"]),
                "val_kd": float(val_res["kd"]),
                "val_attn": float(val_res["attn"]),
                "val_auc": float(val_auc),
                "val_auc_weighted": float(val_res["auc_weighted"]),
                "best_auc": float(best_auc),
                "best_auc_weighted": float(best_auc_weighted),
                "no_imp": int(no_imp),
                "is_best": int(improved),
            }
        )
        completed_epochs = int(ep)
        if ep == 1 or ep % 2 == 0:
            print(
                f"[{name}] ep={ep:03d} train_total={train_res['total']:.5f} "
                f"train_hard={train_res['hard']:.5f} train_kd={train_res['kd']:.5f} train_attn={train_res['attn']:.5f} "
                f"train_auc={train_res['auc']:.5f} train_auc_w={train_res['auc_weighted']:.5f} "
                f"val_total={val_res['total']:.5f} val_hard={val_res['hard']:.5f} "
                f"val_kd={val_res['kd']:.5f} val_attn={val_res['attn']:.5f} "
                f"val_auc={val_auc:.5f} val_auc_w={val_res['auc_weighted']:.5f} "
                f"monitor={early_stop_metric} best_monitor={best_stop_score:.5f} no_imp={no_imp}"
                f"{grad_norm_suffix}"
            )
        if no_imp >= int(patience):
            print(f"[{name}] Early stopping")
            break
    if best_state is not None:
        student.load_state_dict(best_state)
    if metrics_rows:
        metrics_rows[-1]["stopped_after_epoch"] = int(completed_epochs)
    if epoch_metrics_path is not None:
        save_epoch_metrics_table(epoch_metrics_path, metrics_rows)
    if probe_enabled:
        save_gradient_probe_tables(
            probe_prefix,
            scalar_rows=probe_scalar_rows,
            norm_rows=probe_norm_rows,
            cosine_rows=probe_cosine_rows,
            extra_meta={
                "model_name": probe_name,
                "train_batches_per_epoch": int(train_probe_batches),
                "val_batches_per_epoch": int(val_probe_batches),
                "epochs": int(completed_epochs),
                "kind": "hlt_kd",
                "use_sample_weight_for_all_losses": bool(use_sample_weight_for_all_losses),
            },
        )
    save_checkpoint(
        student,
        ckpt_path,
        extra={
            "best_val_auc": float(best_auc),
            "best_val_auc_weighted": float(best_auc_weighted),
            "early_stop_metric": str(early_stop_metric),
            "best_stop_score": float(best_stop_score),
            "kd_enabled": True,
            "use_sample_weight_for_all_losses": bool(use_sample_weight_for_all_losses),
        },
    )
    print(f"Saved checkpoint: {ckpt_path}")
    return student


@torch.no_grad()
def eval_joint_model(
    model,
    loader,
    *,
    device,
    feat_names: list[str],
    feat_means: Optional[np.ndarray] = None,
    feat_stds: np.ndarray,
    feature_loss_weights: Optional[Sequence[float] | np.ndarray] = None,
    joint_phys_weight: float = 0.0,
    joint_unsmear_weight: float,
    joint_cls_weight: float,
    teacher=None,
    use_kd: bool = False,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.0,
    kd_alpha_attn: float = 0.0,
    use_sample_weight_for_all_losses: bool = True,
):
    """Evaluate a joint model."""
    model.eval()
    if teacher is not None:
        teacher.eval()
    kd_enabled = bool(use_kd) and (teacher is not None)

    sums = {
        "joint_total": 0.0,
        "unsmear_total": 0.0,
        "phys_total": 0.0,
        "cls_hard_total": 0.0,
        "cls_kd_total": 0.0,
        "cls_attn_total": 0.0,
        "cls_total": 0.0,
    }
    preds, labs, weights = [], [], []
    total_mix_den = 0.0
    total_aux_den = 0.0
    total_hard_den = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y_uns = batch["y_unsmear"].to(device)
        m = batch["mask"].to(device)
        y_cls = batch["label"].to(device)
        w = batch["weight"].to(device)

        kd_attn_enabled = kd_enabled and (float(kd_alpha_attn) > 0.0)
        if kd_attn_enabled:
            reco, logits, s_attn = model(x, m, return_attention=True)
        else:
            reco, logits = model(x, m)
        aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
        reg_terms = regression_loss_terms(
            reco,
            y_uns,
            m,
            feat_names=feat_names,
            feat_means=feat_means,
            feat_stds=feat_stds,
            sample_weight=aux_weight,
            feature_loss_weights=feature_loss_weights,
            phys_consistency_weight=joint_phys_weight,
        )
        hard_loss = weighted_bce_with_logits(logits.squeeze(-1), y_cls, sample_weight=w)

        kd_loss_val = torch.zeros((), device=device, dtype=hard_loss.dtype)
        attn_loss_val = torch.zeros((), device=device, dtype=hard_loss.dtype)
        cls_loss = hard_loss
        if kd_enabled:
            if kd_attn_enabled:
                teacher_logits, t_attn = teacher(y_uns, m, return_attention=True)
                teacher_logits = teacher_logits.squeeze(-1)
            else:
                teacher_logits = teacher(y_uns, m).squeeze(-1)
            kd_loss_val = kd_loss(
                logits.squeeze(-1),
                teacher_logits,
                float(kd_temperature),
                sample_weight=aux_weight,
            )
            if kd_attn_enabled:
                attn_loss_val = attn_loss(s_attn, t_attn, m, m, sample_weight=aux_weight)
            cls_loss = (
                (1.0 - float(kd_alpha)) * hard_loss
                + float(kd_alpha) * kd_loss_val
                + float(kd_alpha_attn) * attn_loss_val
            )

        joint_loss = float(joint_unsmear_weight) * reg_terms["total"] + float(joint_cls_weight) * cls_loss

        mix_den = _loss_denominator(w, int(x.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        aux_den = _loss_denominator(w, int(x.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        hard_den = _batch_weight_total(w, int(x.shape[0]))
        sums["joint_total"] += float(joint_loss.item()) * mix_den
        sums["unsmear_total"] += float(reg_terms["total"].item()) * aux_den
        sums["phys_total"] += float(reg_terms["phys"].item()) * aux_den
        sums["cls_hard_total"] += float(hard_loss.item()) * hard_den
        sums["cls_kd_total"] += float(kd_loss_val.item()) * aux_den
        sums["cls_attn_total"] += float(attn_loss_val.item()) * aux_den
        sums["cls_total"] += float(cls_loss.item()) * mix_den
        preds.extend(torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy().flatten())
        labs.extend(y_cls.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
        total_mix_den += mix_den
        total_aux_den += aux_den
        total_hard_den += hard_den

    preds_np = np.asarray(preds)
    labs_np = np.asarray(labs)
    weights_np = np.asarray(weights, dtype=np.float64)
    auc, auc_weighted = _auc_scores(
        labs_np,
        preds_np,
        weights_np,
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    out = {
        "joint_total": sums["joint_total"] / max(total_mix_den, 1e-12),
        "unsmear_total": sums["unsmear_total"] / max(total_aux_den, 1e-12),
        "phys_total": sums["phys_total"] / max(total_aux_den, 1e-12),
        "cls_hard_total": sums["cls_hard_total"] / max(total_hard_den, 1e-12),
        "cls_kd_total": sums["cls_kd_total"] / max(total_aux_den, 1e-12),
        "cls_attn_total": sums["cls_attn_total"] / max(total_aux_den, 1e-12),
        "cls_total": sums["cls_total"] / max(total_mix_den, 1e-12),
    }
    out["auc"] = auc
    out["auc_weighted"] = auc_weighted
    out["preds"] = preds_np
    out["labels"] = labs_np
    out["weights"] = weights_np
    return out


def train_or_load_joint_model(
    name: str,
    model,
    ckpt_path: str | Path,
    train_loader,
    val_loader,
    *,
    device,
    feat_names: list[str],
    feat_means: Optional[np.ndarray] = None,
    feat_stds: np.ndarray,
    feature_loss_weights: Optional[Sequence[float] | np.ndarray] = None,
    joint_phys_weight: float = 0.0,
    joint_unsmear_weight: float,
    joint_cls_weight: float,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    epochs: int,
    patience: int,
    early_stop_metric: str = "val_auc",
    use_sample_weight_for_all_losses: bool = True,
    teacher=None,
    use_kd: bool = False,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.0,
    kd_alpha_attn: float = 0.0,
    allow_load: bool = False,
    train_loader_factory: Optional[Callable[[int], DataLoader]] = None,
    grad_probe_cfg: Optional[dict[str, Any]] = None,
    epoch_metrics_path: str | Path | None = None,
):
    """Train or load a joint model."""
    early_stop_metric = resolve_early_stop_metric_name(early_stop_metric)
    probe_cfg = dict(grad_probe_cfg or {})
    probe_prefix = probe_cfg.get("output_prefix", None)
    probe_name = str(probe_cfg.get("model_name", name))
    train_probe_batches = int(probe_cfg.get("train_batches_per_epoch", 0))
    val_probe_batches = int(probe_cfg.get("val_batches_per_epoch", 0))
    probe_enabled = probe_prefix is not None and (train_probe_batches > 0 or val_probe_batches > 0)
    if bool(allow_load) and Path(ckpt_path).is_file():
        load_checkpoint(model, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
        if epoch_metrics_path is not None and not Path(epoch_metrics_path).is_file():
            print(f"[{name}] Epoch-metrics table not found for the loaded checkpoint. Rerun training with loading disabled to regenerate it.")
        if probe_enabled:
            probe_paths = _gradient_probe_output_paths(probe_prefix)
            if not (probe_paths["scalar"].is_file() and probe_paths["norm"].is_file() and probe_paths["cos"].is_file()):
                print(f"[{name}] Gradient probe tables not found for the loaded checkpoint. Rerun training with loading disabled to regenerate them.")
            if not (
                probe_paths["feature_scalar"].is_file()
                and probe_paths["feature_norm"].is_file()
                and probe_paths["feature_cos"].is_file()
            ):
                print(
                    f"[{name}] Feature-level gradient probe tables not found for the loaded checkpoint. "
                    "Rerun training with loading disabled to regenerate them."
                )
        return model

    kd_enabled = bool(use_kd) and (teacher is not None)
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    opt, sch = make_opt(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_epochs=int(warmup_epochs),
        epochs=int(epochs),
    )
    best_auc, best_auc_weighted, best_stop_score, best_state, no_imp = 0.0, 0.0, float("-inf"), None, 0
    probe_scalar_rows: list[dict[str, Any]] = []
    probe_norm_rows: list[dict[str, Any]] = []
    probe_cosine_rows: list[dict[str, Any]] = []
    probe_feature_scalar_rows: list[dict[str, Any]] = []
    probe_feature_norm_rows: list[dict[str, Any]] = []
    probe_feature_cosine_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    completed_epochs = 0
    for ep in range(1, int(epochs) + 1):
        model.train()
        epoch_train_loader = train_loader_factory(ep) if train_loader_factory is not None else train_loader
        train_probe_idx = make_even_interval_batch_indices(len(epoch_train_loader), train_probe_batches) if probe_enabled else []
        train_probe_set = set(train_probe_idx)
        train_probe_rank = {idx: rank for rank, idx in enumerate(train_probe_idx)}
        train_preds, train_labs, train_weights = [], [], []
        tot_joint, tot_uns, tot_phys, tot_cls, tot_hard, tot_kd, tot_attn = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        total_mix_den, total_aux_den, total_hard_den = 0.0, 0.0, 0.0
        for batch_idx, batch in enumerate(epoch_train_loader):
            x = batch["x"].to(device)
            y_uns = batch["y_unsmear"].to(device)
            m = batch["mask"].to(device)
            y_cls = batch["label"].to(device)
            w = batch["weight"].to(device)

            opt.zero_grad(set_to_none=True)
            kd_attn_enabled = kd_enabled and (float(kd_alpha_attn) > 0.0)
            if kd_attn_enabled:
                reco, logits, s_attn = model(x, m, return_attention=True)
            else:
                reco, logits = model(x, m)
            aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
            reg_terms = regression_loss_terms(
                reco,
                y_uns,
                m,
                feat_names=feat_names,
                feat_means=feat_means,
                feat_stds=feat_stds,
                sample_weight=aux_weight,
                feature_loss_weights=feature_loss_weights,
                phys_consistency_weight=joint_phys_weight,
            )
            hard_loss = weighted_bce_with_logits(logits.squeeze(-1), y_cls, sample_weight=w)

            kd_loss_val = torch.zeros((), device=device, dtype=hard_loss.dtype)
            attn_loss_val = torch.zeros((), device=device, dtype=hard_loss.dtype)
            cls_loss = hard_loss
            if kd_enabled:
                with torch.no_grad():
                    if kd_attn_enabled:
                        teacher_logits, t_attn = teacher(y_uns, m, return_attention=True)
                        teacher_logits = teacher_logits.squeeze(-1)
                    else:
                        teacher_logits = teacher(y_uns, m).squeeze(-1)
                kd_loss_val = kd_loss(
                    logits.squeeze(-1),
                    teacher_logits,
                    float(kd_temperature),
                    sample_weight=aux_weight,
                )
                if kd_attn_enabled:
                    attn_loss_val = attn_loss(s_attn, t_attn, m, m, sample_weight=aux_weight)
                cls_loss = (
                    (1.0 - float(kd_alpha)) * hard_loss
                    + float(kd_alpha) * kd_loss_val
                    + float(kd_alpha_attn) * attn_loss_val
                )

            joint_loss = float(joint_unsmear_weight) * reg_terms["total"] + float(joint_cls_weight) * cls_loss
            if batch_idx in train_probe_set:
                diag = gradient_probe_from_losses(
                    model,
                    {
                        "unsmear": reg_terms["total"],
                        "phys": reg_terms["phys"],
                        "hard": hard_loss,
                        "kd": kd_loss_val if kd_enabled else None,
                        "attn": attn_loss_val if (kd_enabled and float(kd_alpha_attn) > 0.0) else None,
                        "total": joint_loss,
                    },
                )
                diag["scalar_losses"] = {
                    "unsmear": float(reg_terms["total"].item()),
                    "phys": float(reg_terms["phys"].item()),
                    "hard": float(hard_loss.item()),
                    "kd": float(kd_loss_val.item()) if kd_enabled else float("nan"),
                    "attn": float(attn_loss_val.item()) if (kd_enabled and float(kd_alpha_attn) > 0.0) else float("nan"),
                    "total": float(joint_loss.item()),
                }
                diag["feature_probe"] = feature_gradient_probe_from_regression_terms(model, reg_terms)
                _append_gradient_probe_rows(
                    scalar_rows=probe_scalar_rows,
                    norm_rows=probe_norm_rows,
                    cosine_rows=probe_cosine_rows,
                    diag=diag,
                    model_name=probe_name,
                    split="train",
                    epoch=int(ep),
                    batch_idx=int(batch_idx),
                    sample_idx=int(train_probe_rank[batch_idx]),
                    total_batches=int(len(epoch_train_loader)),
                )
                _append_gradient_probe_rows(
                    scalar_rows=probe_feature_scalar_rows,
                    norm_rows=probe_feature_norm_rows,
                    cosine_rows=probe_feature_cosine_rows,
                    diag=diag["feature_probe"],
                    model_name=probe_name,
                    split="train",
                    epoch=int(ep),
                    batch_idx=int(batch_idx),
                    sample_idx=int(train_probe_rank[batch_idx]),
                    total_batches=int(len(epoch_train_loader)),
                )
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            mix_den = _loss_denominator(w, int(x.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
            aux_den = _loss_denominator(w, int(x.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
            hard_den = _batch_weight_total(w, int(x.shape[0]))
            tot_joint += float(joint_loss.item()) * mix_den
            tot_uns += float(reg_terms["total"].item()) * aux_den
            tot_phys += float(reg_terms["phys"].item()) * aux_den
            tot_cls += float(cls_loss.item()) * mix_den
            tot_hard += float(hard_loss.item()) * hard_den
            tot_kd += float(kd_loss_val.item()) * aux_den
            tot_attn += float(attn_loss_val.item()) * aux_den
            train_preds.extend(torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy().flatten())
            train_labs.extend(y_cls.detach().cpu().numpy().flatten())
            train_weights.extend(w.detach().cpu().numpy().flatten())
            total_mix_den += mix_den
            total_aux_den += aux_den
            total_hard_den += hard_den

        sch.step()
        val_res = eval_joint_model(
            model,
            val_loader,
            device=device,
            feat_names=feat_names,
            feat_means=feat_means,
            feat_stds=feat_stds,
            feature_loss_weights=feature_loss_weights,
            joint_phys_weight=joint_phys_weight,
            joint_unsmear_weight=float(joint_unsmear_weight),
            joint_cls_weight=float(joint_cls_weight),
            teacher=teacher,
            use_kd=kd_enabled,
            kd_temperature=float(kd_temperature),
            kd_alpha=float(kd_alpha),
            kd_alpha_attn=float(kd_alpha_attn),
            use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        )
        if probe_enabled and val_probe_batches > 0:
            val_probe_rows = collect_loader_gradient_probes(
                loader=val_loader,
                sample_count=val_probe_batches,
                probe_fn=lambda batch: probe_joint_gradients(
                    model,
                    batch,
                    device=device,
                    feat_names=feat_names,
                    feat_means=feat_means,
                    feat_stds=feat_stds,
                    feature_loss_weights=feature_loss_weights,
                    joint_phys_weight=joint_phys_weight,
                    teacher=teacher,
                    use_kd=kd_enabled,
                    kd_temperature=float(kd_temperature),
                    kd_alpha=float(kd_alpha),
                    kd_alpha_attn=float(kd_alpha_attn),
                    joint_unsmear_weight=float(joint_unsmear_weight),
                    joint_cls_weight=float(joint_cls_weight),
                    use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
                ),
                model_name=probe_name,
                split="val",
                epoch=int(ep),
            )
            probe_scalar_rows.extend(val_probe_rows["scalar_rows"])
            probe_norm_rows.extend(val_probe_rows["norm_rows"])
            probe_cosine_rows.extend(val_probe_rows["cosine_rows"])
            probe_feature_scalar_rows.extend(val_probe_rows["feature_scalar_rows"])
            probe_feature_norm_rows.extend(val_probe_rows["feature_norm_rows"])
            probe_feature_cosine_rows.extend(val_probe_rows["feature_cosine_rows"])
        joint_grad_loss_weights = {
            "unsmear": float(joint_unsmear_weight),
            "phys": float(joint_unsmear_weight),
            "hard": float(joint_cls_weight) * (float(1.0 - kd_alpha) if kd_enabled else 1.0),
            "kd": float(joint_cls_weight) * float(kd_alpha),
            "attn": float(joint_cls_weight) * float(kd_alpha_attn),
        }
        joint_loss_order = ["unsmear", "phys", "hard"]
        if kd_enabled:
            joint_loss_order.extend(["kd", "attn"])
        train_grad_norm_summary = _format_epoch_mean_grad_norm_summary(
            probe_norm_rows,
            epoch=int(ep),
            split="train",
            loss_order=joint_loss_order,
            loss_weights=joint_grad_loss_weights,
            label="train",
        )
        val_grad_norm_summary = _format_epoch_mean_grad_norm_summary(
            probe_norm_rows,
            epoch=int(ep),
            split="val",
            loss_order=joint_loss_order,
            loss_weights=joint_grad_loss_weights,
            label="val",
        )
        grad_norm_suffix = ""
        grad_norm_parts = [part for part in [train_grad_norm_summary, val_grad_norm_summary] if part]
        if grad_norm_parts:
            grad_norm_suffix = " " + " ".join(grad_norm_parts)
        train_joint = tot_joint / max(total_mix_den, 1e-12)
        train_uns = tot_uns / max(total_aux_den, 1e-12)
        train_phys = tot_phys / max(total_aux_den, 1e-12)
        train_cls = tot_cls / max(total_mix_den, 1e-12)
        train_hard = tot_hard / max(total_hard_den, 1e-12)
        train_kd = tot_kd / max(total_aux_den, 1e-12)
        train_attn = tot_attn / max(total_aux_den, 1e-12)
        train_auc, train_auc_weighted = _auc_scores(
            train_labs,
            train_preds,
            np.asarray(train_weights, dtype=np.float64),
            use_sample_weight=use_sample_weight_for_all_losses,
        )
        val_auc = float(val_res["auc"])
        stop_score = select_early_stop_score(
            early_stop_metric,
            val_auc=float(val_auc),
            val_auc_weighted=float(val_res["auc_weighted"]),
        )
        improved = bool(stop_score > best_stop_score + 1e-4)
        if stop_score > best_stop_score + 1e-4:
            best_auc = val_auc
            best_auc_weighted = float(val_res["auc_weighted"])
            best_stop_score = float(stop_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        metrics_rows.append(
            {
                "model": str(name),
                "epoch": int(ep),
                "early_stop_metric": str(early_stop_metric),
                "best_stop_score": float(best_stop_score),
                "train_joint": float(train_joint),
                "train_uns": float(train_uns),
                "train_phys": float(train_phys),
                "train_cls": float(train_cls),
                "train_hard": float(train_hard),
                "train_kd": float(train_kd),
                "train_attn": float(train_attn),
                "train_auc": float(train_auc),
                "train_auc_weighted": float(train_auc_weighted),
                "val_joint": float(val_res["joint_total"]),
                "val_uns": float(val_res["unsmear_total"]),
                "val_phys": float(val_res["phys_total"]),
                "val_cls": float(val_res["cls_total"]),
                "val_hard": float(val_res["cls_hard_total"]),
                "val_kd": float(val_res["cls_kd_total"]),
                "val_attn": float(val_res["cls_attn_total"]),
                "val_auc": float(val_auc),
                "val_auc_weighted": float(val_res["auc_weighted"]),
                "best_auc": float(best_auc),
                "best_auc_weighted": float(best_auc_weighted),
                "no_imp": int(no_imp),
                "is_best": int(improved),
            }
        )
        completed_epochs = int(ep)
        if ep == 1 or ep % 2 == 0:
            print(
                f"[{name}] ep={ep:03d} train_joint={train_joint:.5f} train_uns={train_uns:.5f} train_phys={train_phys:.5f} "
                f"train_cls={train_cls:.5f} train_hard={train_hard:.5f} train_kd={train_kd:.5f} train_attn={train_attn:.5f} "
                f"train_auc={train_auc:.5f} train_auc_w={train_auc_weighted:.5f} "
                f"val_joint={val_res['joint_total']:.5f} val_uns={val_res['unsmear_total']:.5f} val_phys={val_res['phys_total']:.5f} "
                f"val_cls={val_res['cls_total']:.5f} val_hard={val_res['cls_hard_total']:.5f} "
                f"val_kd={val_res['cls_kd_total']:.5f} val_attn={val_res['cls_attn_total']:.5f} "
                f"val_auc={val_auc:.5f} val_auc_w={val_res['auc_weighted']:.5f} "
                f"monitor={early_stop_metric} best_monitor={best_stop_score:.5f} no_imp={no_imp}"
                f"{grad_norm_suffix}"
            )
        if no_imp >= int(patience):
            print(f"[{name}] Early stopping")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if metrics_rows:
        metrics_rows[-1]["stopped_after_epoch"] = int(completed_epochs)
    if epoch_metrics_path is not None:
        save_epoch_metrics_table(epoch_metrics_path, metrics_rows)
    if probe_enabled:
        save_gradient_probe_tables(
            probe_prefix,
            scalar_rows=probe_scalar_rows,
            norm_rows=probe_norm_rows,
            cosine_rows=probe_cosine_rows,
            feature_scalar_rows=probe_feature_scalar_rows,
            feature_norm_rows=probe_feature_norm_rows,
            feature_cosine_rows=probe_feature_cosine_rows,
            extra_meta={
                "model_name": probe_name,
                "train_batches_per_epoch": int(train_probe_batches),
                "val_batches_per_epoch": int(val_probe_batches),
                "epochs": int(completed_epochs),
                "kind": "joint",
                "kd_enabled": bool(kd_enabled),
                "joint_phys_weight": float(joint_phys_weight),
                "feature_loss_weights": _resolve_feature_loss_weights(feat_names, feature_loss_weights).tolist(),
                "feature_names": list(feat_names),
                "use_sample_weight_for_all_losses": bool(use_sample_weight_for_all_losses),
            },
        )
    save_checkpoint(
        model,
        ckpt_path,
        extra={
            "best_val_auc": float(best_auc),
            "best_val_auc_weighted": float(best_auc_weighted),
            "early_stop_metric": str(early_stop_metric),
            "best_stop_score": float(best_stop_score),
            "kd_enabled": bool(kd_enabled),
            "joint_phys_weight": float(joint_phys_weight),
            "use_sample_weight_for_all_losses": bool(use_sample_weight_for_all_losses),
        },
    )
    print(f"Saved checkpoint: {ckpt_path}")
    return model

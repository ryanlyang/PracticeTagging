"""
Joint unsmear utilities.

This directory keeps only the helper functions needed for joint training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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
        loss = weighted_bce_with_logits(logits, y, sample_weight=w)
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


def train_kd(student, teacher, loader, opt, device, cfg: dict):
    student.train()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"].get("alpha_attn", 0.0))

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
        loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=w)
        loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
        loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
        if a_attn > 0.0:
            loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=w)
        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += float(loss.item()) * int(y.shape[0])
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
    return total_loss / max(1, len(preds)), float(roc_auc_score(labs, preds))


def train_kd_detailed(student, teacher, loader, opt, device, cfg: dict) -> dict[str, float]:
    """Train one KD epoch and return total/hard/kd/attn terms."""
    student.train()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"].get("alpha_attn", 0.0))

    preds, labs = [], []
    total_loss = 0.0
    total_hard = 0.0
    total_kd = 0.0
    total_attn = 0.0
    n = 0
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
        loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=w)
        loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
        loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
        if a_attn > 0.0:
            loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=w)
        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total_hard += float(loss_hard.item()) * bs
        total_kd += float(loss_kd.item()) * bs
        total_attn += float(loss_a.item()) * bs
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        n += bs
    return {
        "total": total_loss / max(1, n),
        "hard": total_hard / max(1, n),
        "kd": total_kd / max(1, n),
        "attn": total_attn / max(1, n),
        "auc": float(roc_auc_score(labs, preds)),
    }


@torch.no_grad()
def eval_kd_student(student, teacher, loader, device, cfg: dict) -> dict[str, float | np.ndarray]:
    """Evaluate a KD student and return total/hard/kd/attn terms."""
    student.eval()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"].get("alpha_attn", 0.0))

    preds, labs = [], []
    total_loss = 0.0
    total_hard = 0.0
    total_kd = 0.0
    total_attn = 0.0
    n = 0
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
        loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=w)
        loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
        loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
        if a_attn > 0.0:
            loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=w)
        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total_hard += float(loss_hard.item()) * bs
        total_kd += float(loss_kd.item()) * bs
        total_attn += float(loss_a.item()) * bs
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        n += bs

    preds_np = np.asarray(preds)
    labs_np = np.asarray(labs)
    return {
        "total": total_loss / max(1, n),
        "hard": total_hard / max(1, n),
        "kd": total_kd / max(1, n),
        "attn": total_attn / max(1, n),
        "auc": float(roc_auc_score(labs_np, preds_np)),
        "preds": preds_np,
        "labels": labs_np,
    }


def compute_roc(y: np.ndarray, p: np.ndarray):
    fpr, tpr, _ = roc_curve(y, p)
    auc = float(roc_auc_score(y, p))
    return fpr, tpr, auc


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
    feat_stds: np.ndarray,
    sample_weight: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Unsmear regression loss terms for the joint model."""
    idx_map = {n: i for i, n in enumerate(list(feat_names))}
    dphi_idx = idx_map.get("dPhi", None)
    deta_idx = idx_map.get("dEta", None)
    dr_idx = idx_map.get("dR", None)
    dphi_scale = float(np.asarray(feat_stds, dtype=np.float32)[int(dphi_idx)]) if dphi_idx is not None else 1.0

    if dphi_idx is not None:
        base_per_jet = _per_jet_masked_smooth_l1_wrap_dphi(
            mu,
            y,
            m,
            dphi_idx=int(dphi_idx),
            dphi_scale=dphi_scale,
        )
    else:
        base_per_jet = _per_jet_masked_smooth_l1(mu, y, m)
    base = _weighted_mean(base_per_jet, sample_weight)

    cons_raw = torch.zeros((), device=mu.device, dtype=mu.dtype)
    if (dr_idx is not None) and (deta_idx is not None) and (dphi_idx is not None):
        dR_cons = torch.sqrt(mu[..., int(deta_idx)] ** 2 + mu[..., int(dphi_idx)] ** 2 + 1e-12)
        dR_pred = mu[..., int(dr_idx)]
        cons_per_jet = _per_jet_masked_smooth_l1(dR_pred.unsqueeze(-1), dR_cons.unsqueeze(-1), m)
        cons_raw = _weighted_mean(cons_per_jet, sample_weight)

    return {
        "total": base,
        "base": base,
        "dr_cons_raw": cons_raw,
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

    hard_loss = weighted_bce_with_logits(student_logits, y, sample_weight=w)
    kd_loss_val = kd_loss(student_logits, teacher_logits, float(kd_temperature), sample_weight=w)
    attn_loss_val: Optional[torch.Tensor] = None
    if float(kd_alpha_attn) > 0.0:
        attn_loss_val = attn_loss(student_attn, teacher_attn, m_hlt, m_off, sample_weight=w)
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
    feat_stds: np.ndarray,
    teacher=None,
    use_kd: bool = False,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.0,
    kd_alpha_attn: float = 0.0,
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
    reg_terms = regression_loss_terms(
        reco,
        y_uns,
        m,
        feat_names=feat_names,
        feat_stds=feat_stds,
        sample_weight=w,
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
            sample_weight=w,
        )
        if kd_attn_enabled:
            attn_loss_val = attn_loss(student_attn, teacher_attn, m, m, sample_weight=w)
        cls_loss = (
            (1.0 - float(kd_alpha)) * hard_loss
            + float(kd_alpha) * kd_loss_val
            + float(kd_alpha_attn) * (torch.zeros_like(hard_loss) if attn_loss_val is None else attn_loss_val)
        )

    total_loss = reg_terms["total"] + cls_loss
    out = gradient_probe_from_losses(
        model,
        {
            "unsmear": reg_terms["total"],
            "hard": hard_loss,
            "kd": kd_loss_val,
            "attn": attn_loss_val,
            "total": total_loss,
        },
    )
    out["scalar_losses"] = {
        "unsmear": float(reg_terms["total"].item()),
        "hard": float(hard_loss.item()),
        "kd": float("nan") if kd_loss_val is None else float(kd_loss_val.item()),
        "attn": float("nan") if attn_loss_val is None else float(attn_loss_val.item()),
        "total": float(total_loss.item()),
    }
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
    train_loader_factory: Optional[Callable[[int], DataLoader]] = None,
):
    """Train or load a standard classification model."""
    if bool(allow_load) and Path(ckpt_path).is_file():
        load_checkpoint(model, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
        return model

    opt, sch = make_opt(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_epochs=int(warmup_epochs),
        epochs=int(epochs),
    )
    best_auc, best_state, no_imp = 0.0, None, 0
    for ep in range(1, int(epochs) + 1):
        epoch_train_loader = train_loader_factory(ep) if train_loader_factory is not None else train_loader
        loss, _ = train_standard(model, epoch_train_loader, opt, device, feat_key, mask_key)
        sch.step()
        val_auc, _, _ = evaluate(model, val_loader, device, feat_key, mask_key)
        if val_auc > best_auc + 1e-4:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep == 1 or ep % 2 == 0:
            print(f"[{name}] ep={ep:03d} train_loss={loss:.5f} val_auc={val_auc:.5f} best={best_auc:.5f} no_imp={no_imp}")
        if no_imp >= int(patience):
            print(f"[{name}] Early stopping")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, ckpt_path, extra={"best_val_auc": float(best_auc)})
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
    kd_temperature: float,
    kd_alpha: float,
    kd_alpha_attn: float,
    train_loader_factory: Optional[Callable[[int], DataLoader]] = None,
):
    """Train or load the KD baseline classifier."""
    if bool(allow_load) and Path(ckpt_path).is_file():
        load_checkpoint(student, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
        return student

    opt, sch = make_opt(
        student,
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_epochs=int(warmup_epochs),
        epochs=int(epochs),
    )
    best_auc, best_state, no_imp = 0.0, None, 0
    kd_cfg = {
        "kd": {
            "temperature": float(kd_temperature),
            "alpha_kd": float(kd_alpha),
            "alpha_attn": float(kd_alpha_attn),
        }
    }
    for ep in range(1, int(epochs) + 1):
        epoch_train_loader = train_loader_factory(ep) if train_loader_factory is not None else train_loader
        train_res = train_kd_detailed(student, teacher, epoch_train_loader, opt, device, kd_cfg)
        sch.step()
        val_res = eval_kd_student(student, teacher, val_loader, device, kd_cfg)
        val_auc = float(val_res["auc"])
        if val_auc > best_auc + 1e-4:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep == 1 or ep % 2 == 0:
            print(
                f"[{name}] ep={ep:03d} train_total={train_res['total']:.5f} "
                f"train_hard={train_res['hard']:.5f} train_kd={train_res['kd']:.5f} train_attn={train_res['attn']:.5f} "
                f"val_total={val_res['total']:.5f} val_hard={val_res['hard']:.5f} "
                f"val_kd={val_res['kd']:.5f} val_attn={val_res['attn']:.5f} "
                f"val_auc={val_auc:.5f} best={best_auc:.5f} no_imp={no_imp}"
            )
        if no_imp >= int(patience):
            print(f"[{name}] Early stopping")
            break
    if best_state is not None:
        student.load_state_dict(best_state)
    save_checkpoint(student, ckpt_path, extra={"best_val_auc": float(best_auc), "kd_enabled": True})
    print(f"Saved checkpoint: {ckpt_path}")
    return student


@torch.no_grad()
def eval_joint_model(
    model,
    loader,
    *,
    device,
    feat_names: list[str],
    feat_stds: np.ndarray,
    joint_unsmear_weight: float,
    joint_cls_weight: float,
    teacher=None,
    use_kd: bool = False,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.0,
    kd_alpha_attn: float = 0.0,
):
    """Evaluate a joint model."""
    model.eval()
    if teacher is not None:
        teacher.eval()
    kd_enabled = bool(use_kd) and (teacher is not None)

    sums = {
        "joint_total": 0.0,
        "unsmear_total": 0.0,
        "cls_hard_total": 0.0,
        "cls_kd_total": 0.0,
        "cls_attn_total": 0.0,
        "cls_total": 0.0,
    }
    preds, labs = [], []
    n = 0
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
        reg_terms = regression_loss_terms(
            reco,
            y_uns,
            m,
            feat_names=feat_names,
            feat_stds=feat_stds,
            sample_weight=w,
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
                sample_weight=w,
            )
            if kd_attn_enabled:
                attn_loss_val = attn_loss(s_attn, t_attn, m, m, sample_weight=w)
            cls_loss = (
                (1.0 - float(kd_alpha)) * hard_loss
                + float(kd_alpha) * kd_loss_val
                + float(kd_alpha_attn) * attn_loss_val
            )

        joint_loss = float(joint_unsmear_weight) * reg_terms["total"] + float(joint_cls_weight) * cls_loss

        bs = int(x.shape[0])
        sums["joint_total"] += float(joint_loss.item()) * bs
        sums["unsmear_total"] += float(reg_terms["total"].item()) * bs
        sums["cls_hard_total"] += float(hard_loss.item()) * bs
        sums["cls_kd_total"] += float(kd_loss_val.item()) * bs
        sums["cls_attn_total"] += float(attn_loss_val.item()) * bs
        sums["cls_total"] += float(cls_loss.item()) * bs
        preds.extend(torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy().flatten())
        labs.extend(y_cls.detach().cpu().numpy().flatten())
        n += bs

    auc = float(compute_roc(np.asarray(labs), np.asarray(preds))[2])
    out = {k: v / max(1, n) for k, v in sums.items()}
    out["auc"] = auc
    out["preds"] = np.asarray(preds)
    out["labels"] = np.asarray(labs)
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
    feat_stds: np.ndarray,
    joint_unsmear_weight: float,
    joint_cls_weight: float,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    epochs: int,
    patience: int,
    teacher=None,
    use_kd: bool = False,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.0,
    kd_alpha_attn: float = 0.0,
    allow_load: bool = False,
    train_loader_factory: Optional[Callable[[int], DataLoader]] = None,
):
    """Train or load a joint model."""
    if bool(allow_load) and Path(ckpt_path).is_file():
        load_checkpoint(model, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
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
    best_auc, best_state, no_imp = 0.0, None, 0
    for ep in range(1, int(epochs) + 1):
        model.train()
        epoch_train_loader = train_loader_factory(ep) if train_loader_factory is not None else train_loader
        tot_joint, tot_uns, tot_cls, tot_hard, tot_kd, tot_attn, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for batch in epoch_train_loader:
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
            reg_terms = regression_loss_terms(
                reco,
                y_uns,
                m,
                feat_names=feat_names,
                feat_stds=feat_stds,
                sample_weight=w,
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
                    sample_weight=w,
                )
                if kd_attn_enabled:
                    attn_loss_val = attn_loss(s_attn, t_attn, m, m, sample_weight=w)
                cls_loss = (
                    (1.0 - float(kd_alpha)) * hard_loss
                    + float(kd_alpha) * kd_loss_val
                    + float(kd_alpha_attn) * attn_loss_val
                )

            joint_loss = float(joint_unsmear_weight) * reg_terms["total"] + float(joint_cls_weight) * cls_loss
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = int(x.shape[0])
            tot_joint += float(joint_loss.item()) * bs
            tot_uns += float(reg_terms["total"].item()) * bs
            tot_cls += float(cls_loss.item()) * bs
            tot_hard += float(hard_loss.item()) * bs
            tot_kd += float(kd_loss_val.item()) * bs
            tot_attn += float(attn_loss_val.item()) * bs
            n += bs

        sch.step()
        val_res = eval_joint_model(
            model,
            val_loader,
            device=device,
            feat_names=feat_names,
            feat_stds=feat_stds,
            joint_unsmear_weight=float(joint_unsmear_weight),
            joint_cls_weight=float(joint_cls_weight),
            teacher=teacher,
            use_kd=kd_enabled,
            kd_temperature=float(kd_temperature),
            kd_alpha=float(kd_alpha),
            kd_alpha_attn=float(kd_alpha_attn),
        )
        train_joint = tot_joint / max(1, n)
        train_uns = tot_uns / max(1, n)
        train_cls = tot_cls / max(1, n)
        train_hard = tot_hard / max(1, n)
        train_kd = tot_kd / max(1, n)
        train_attn = tot_attn / max(1, n)
        val_auc = float(val_res["auc"])
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep == 1 or ep % 2 == 0:
            print(
                f"[{name}] ep={ep:03d} train_joint={train_joint:.5f} train_uns={train_uns:.5f} "
                f"train_cls={train_cls:.5f} train_hard={train_hard:.5f} train_kd={train_kd:.5f} train_attn={train_attn:.5f} "
                f"val_joint={val_res['joint_total']:.5f} val_uns={val_res['unsmear_total']:.5f} "
                f"val_cls={val_res['cls_total']:.5f} val_hard={val_res['cls_hard_total']:.5f} "
                f"val_kd={val_res['cls_kd_total']:.5f} val_attn={val_res['cls_attn_total']:.5f} "
                f"val_auc={val_auc:.5f} best={best_auc:.5f} no_imp={no_imp}"
            )
        if no_imp >= int(patience):
            print(f"[{name}] Early stopping")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, ckpt_path, extra={"best_val_auc": float(best_auc), "kd_enabled": bool(kd_enabled)})
    print(f"Saved checkpoint: {ckpt_path}")
    return model

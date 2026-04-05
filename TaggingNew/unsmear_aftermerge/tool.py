"""
Unsmear utilities for our workflow.

Goals:
- Build supervised pairs where the input is the smeared HLT view and the target is the merged pre-smear view.
- Keep denoising at token level by default, without attempting to recover the child-level degrees of freedom lost in merging.

Notes:
- To make supervision more stable, features are computed in the jet-axis frame and always use the *post-smear* jet axis
  (which is observable at inference time), avoiding extra mismatch from inconsistent input/target axes.
- Efficiency loss is supported: training and supervision are computed only on surviving tokens (`mask=True`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score, roc_curve


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def wrap_dphi_np(dphi: np.ndarray) -> np.ndarray:
    """Wrap an angular difference into (-pi, pi]."""
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def wrap_dphi_torch(dphi: torch.Tensor) -> torch.Tensor:
    """Wrap an angular difference into (-pi, pi] for torch loss computation."""
    return torch.atan2(torch.sin(dphi), torch.cos(dphi))


@dataclass
class HLTEffectsCfg:
    """HLT effects config keeping only the fields needed for unsmear."""

    pt_threshold_offline: float = 0.5
    pt_threshold_hlt: float = 0.5
    merge_enabled: bool = True
    merge_radius: float = 0.01
    # smear
    pt_resolution: float = 0.10
    eta_resolution: float = 0.03
    phi_resolution: float = 0.03
    # efficiency
    efficiency_loss: float = 0.0


def apply_hlt_effects_pair(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: HLTEffectsCfg,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate paired `(pre_smear, post_smear)` data while keeping the efficiency-loss mask consistent.

    Args:
      const: [N,S,4] (pt,eta,phi,E)
      mask:  [N,S] bool
    Returns:
      pre_smear_const:  [N,S,4] after merging and before smearing, with the same effloss mask applied
      post_smear_const: [N,S,4] after merging and smearing, with effloss applied
      post_mask:        [N,S] bool survive mask used for training/evaluation
      origin_counts:    [N,S] int32 number of original tokens merged into each token (singleton=1, merged>1)
    """
    rs = np.random.RandomState(int(seed))
    hlt = const.copy()
    hlt_mask = mask.copy()

    # Effect 0: offline threshold (shared starting point)
    pt_thr_off = float(cfg.pt_threshold_offline)
    hlt_mask = hlt_mask & (hlt[:, :, 0] >= pt_thr_off)
    hlt[~hlt_mask] = 0.0

    # Effect 1: HLT threshold (optional extra threshold)
    pt_thr_hlt = float(cfg.pt_threshold_hlt)
    below = (hlt[:, :, 0] < pt_thr_hlt) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0.0

    # Track origin counts (singleton=1 at start)
    origin_counts = hlt_mask.astype(np.int32)

    # Effect 2: merge (DeltaR < r)
    if bool(cfg.merge_enabled) and float(cfg.merge_radius) > 0.0:
        r = float(cfg.merge_radius)
        n_jets, max_part, _ = hlt.shape
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            if valid_idx.size < 2:
                continue
            to_remove = set()
            for ii in range(len(valid_idx)):
                i = int(valid_idx[ii])
                if i in to_remove:
                    continue
                for jj in range(ii + 1, len(valid_idx)):
                    j = int(valid_idx[jj])
                    if j in to_remove:
                        continue
                    deta = float(hlt[jet_idx, i, 1] - hlt[jet_idx, j, 1])
                    dphi = float(wrap_dphi_np(hlt[jet_idx, i, 2] - hlt[jet_idx, j, 2]))
                    dR = math.sqrt(deta * deta + dphi * dphi)
                    if dR < r:
                        pt_i = float(hlt[jet_idx, i, 0])
                        pt_j = float(hlt[jet_idx, j, 0])
                        pt_sum = pt_i + pt_j
                        if pt_sum < 1e-6:
                            continue
                        w_i = pt_i / pt_sum
                        w_j = pt_j / pt_sum
                        # Merge into i and remove j.
                        hlt[jet_idx, i, 0] = pt_sum
                        hlt[jet_idx, i, 1] = w_i * float(hlt[jet_idx, i, 1]) + w_j * float(
                            hlt[jet_idx, j, 1]
                        )
                        phi_i = float(hlt[jet_idx, i, 2])
                        phi_j = float(hlt[jet_idx, j, 2])
                        hlt[jet_idx, i, 2] = math.atan2(
                            w_i * math.sin(phi_i) + w_j * math.sin(phi_j),
                            w_i * math.cos(phi_i) + w_j * math.cos(phi_j),
                        )
                        hlt[jet_idx, i, 3] = float(hlt[jet_idx, i, 3]) + float(hlt[jet_idx, j, 3])
                        origin_counts[jet_idx, i] += origin_counts[jet_idx, j]
                        to_remove.add(j)
            for j in to_remove:
                hlt_mask[jet_idx, j] = False
                hlt[jet_idx, j] = 0.0
                origin_counts[jet_idx, j] = 0

    # Pre-smear snapshot before smearing; do not apply effloss yet.
    pre = hlt.copy()

    # Effect 3: smear
    valid = hlt_mask.copy()
    n_jets, max_part, _ = hlt.shape
    # pt smearing is multiplicative around 1.
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

    # Effect 4: efficiency loss (apply to post, then reuse the same mask for pre).
    eff = float(cfg.efficiency_loss)
    if eff > 0.0:
        lost = (rs.random_sample((n_jets, max_part)) < eff) & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0.0
        pre[lost] = 0.0
        origin_counts[lost] = 0

    pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    pre[~hlt_mask] = 0.0
    hlt[~hlt_mask] = 0.0
    return pre.astype(np.float32), hlt.astype(np.float32), hlt_mask.astype(bool), origin_counts.astype(np.int32)


def compute_jet_axis(const: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the jet axis from the summed token four-vectors."""
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
    jet_eta = 0.5 * np.log(
        np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8)
    )
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
    const: np.ndarray, mask: np.ndarray, axis: Dict[str, np.ndarray], *, kind: str = "7d"
) -> np.ndarray:
    """Compute engineered features with an externally provided axis.

    kind:
      - 3d: dEta, dPhi, log_pt
      - 4d: dEta, dPhi, log_pt, log_E
      - 7d: dEta, dPhi, log_pt, log_E, log_pt_rel, log_E_rel, dR
    """
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


def get_stats(feat: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.zeros(feat.shape[-1], dtype=np.float64)
    stds = np.zeros(feat.shape[-1], dtype=np.float64)
    for i in range(feat.shape[-1]):
        vals = feat[idx][:, :, i][mask[idx]]
        means[i] = float(np.nanmean(vals))
        stds[i] = float(np.nanstd(vals) + 1e-8)
    return means.astype(np.float32), stds.astype(np.float32)


def standardize(
    feat: np.ndarray, mask: np.ndarray, means: np.ndarray, stds: np.ndarray, *, clip: float = 10.0
) -> np.ndarray:
    out = (feat - means[None, None, :]) / stds[None, None, :]
    out = np.clip(out, -float(clip), float(clip))
    out = np.nan_to_num(out, 0.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


class UnsmearJetDataset(Dataset):
    """Jet-level dataset: predict per-token pre-smear features from post-smear features."""

    def __init__(
        self,
        x_post: np.ndarray,
        y_pre: np.ndarray,
        mask: np.ndarray,
        origin_counts: Optional[np.ndarray] = None,
    ):
        self.x_post = torch.tensor(x_post, dtype=torch.float32)
        self.y_pre = torch.tensor(y_pre, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.origin_counts = (
            torch.tensor(origin_counts, dtype=torch.int32) if origin_counts is not None else None
        )

    def __len__(self) -> int:
        return int(self.x_post.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        out = {"x": self.x_post[i], "y": self.y_pre[i], "mask": self.mask[i]}
        if self.origin_counts is not None:
            out["origin_counts"] = self.origin_counts[i]
        return out


def masked_smooth_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute SmoothL1 on `[B,S,D]` only where `mask=True`."""
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
    """SmoothL1 with a wrap-aware residual on the dPhi dimension.

    If dPhi is trained in standardized space, pass `dphi_scale=std(dPhi)`
    so wrapping happens in angle space: `wrap((pred-tgt)*std)/std`.
    """
    diff = pred - tgt
    scale = dphi_scale if isinstance(dphi_scale, torch.Tensor) else torch.tensor(float(dphi_scale), device=pred.device, dtype=pred.dtype)
    diff_phi = wrap_dphi_torch(diff[..., int(dphi_idx)] * scale) / scale
    diff = diff.clone()
    diff[..., int(dphi_idx)] = diff_phi
    # smooth_l1 on residual
    per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    m = mask.to(pred.dtype).unsqueeze(-1)
    num = (per * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MSE on `[B,S,D]` only where `mask=True`."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    diff = (pred - tgt) ** 2
    num = (diff * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def masked_gaussian_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    active_dim_mask: Optional[torch.Tensor] = None,
    log_var_clip: float = 6.0,
) -> torch.Tensor:
    """Diagonal Gaussian NLL for heteroscedastic regression.

    Args:
      mu, log_var, tgt: [B,S,D]
      mask: [B,S] bool
      active_dim_mask: [D] bool/0-1 mask controlling which dimensions use uncertainty;
        disabled dimensions are treated as `log_var=0` (equivalent to fixed variance).
    """
    log_var = torch.clamp(log_var, min=-float(log_var_clip), max=float(log_var_clip))
    if active_dim_mask is not None:
        adm = active_dim_mask.to(dtype=mu.dtype, device=mu.device).view(1, 1, -1)
        log_var = log_var * adm  # Disabled dimensions => 0

    diff2 = (tgt - mu) ** 2
    inv_var = torch.exp(-log_var)
    per = 0.5 * (diff2 * inv_var + log_var)
    m = mask.to(mu.dtype).unsqueeze(-1)
    num = (per * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def masked_gaussian_nll_wrap_dphi(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    dphi_idx: int,
    dphi_scale: float | torch.Tensor = 1.0,
    active_dim_mask: Optional[torch.Tensor] = None,
    log_var_clip: float = 6.0,
) -> torch.Tensor:
    """Diagonal Gaussian NLL with a wrap-aware residual on dPhi.

    Same note as above: in standardized space, pass `dphi_scale=std(dPhi)`.
    """
    log_var = torch.clamp(log_var, min=-float(log_var_clip), max=float(log_var_clip))
    if active_dim_mask is not None:
        adm = active_dim_mask.to(dtype=mu.dtype, device=mu.device).view(1, 1, -1)
        log_var = log_var * adm

    err = tgt - mu
    scale = dphi_scale if isinstance(dphi_scale, torch.Tensor) else torch.tensor(float(dphi_scale), device=mu.device, dtype=mu.dtype)
    err_phi = wrap_dphi_torch(err[..., int(dphi_idx)] * scale) / scale
    err = err.clone()
    err[..., int(dphi_idx)] = err_phi

    diff2 = err ** 2
    inv_var = torch.exp(-log_var)
    per = 0.5 * (diff2 * inv_var + log_var)
    m = mask.to(mu.dtype).unsqueeze(-1)
    num = (per * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def get_feat_names(kind: str) -> list[str]:
    k = str(kind).lower()
    if k == "3d":
        return ["dEta", "dPhi", "log_pt"]
    if k == "4d":
        return ["dEta", "dPhi", "log_pt", "log_E"]
    if k == "7d":
        return ["dEta", "dPhi", "log_pt", "log_E", "log_pt_rel", "log_E_rel", "dR"]
    raise ValueError(f"Unknown feature kind: {kind}")


def feats_to_7d(
    feat: np.ndarray,
    mask: np.ndarray,
    axis: Dict[str, np.ndarray],
    *,
    kind: str,
) -> np.ndarray:
    """Expand 3D/4D features into 7D engineered features in raw feature space.

    Mainly used when unsmear predicts only 3D/4D features but the downstream model expects 7D inputs.

    Args:
      feat: [N,S,Dk] raw feature values (not standardized)
      mask: [N,S] bool
      axis: jet-axis dict (usually the post-smear axis; contains `jet_eta/jet_phi/jet_pt/jet_E`)
      kind: '3d'/'4d'/'7d', describing the semantics of the input features
    Returns:
      out: [N,S,7] raw 7D engineered features
    """
    k = str(kind).lower()
    if k == "7d":
        out = np.asarray(feat, dtype=np.float32)
        out[~mask] = 0.0
        return out

    feat = np.asarray(feat, dtype=np.float32)
    dEta = feat[..., 0]
    dPhi = feat[..., 1]
    log_pt = feat[..., 2]

    jet_eta = np.asarray(axis["jet_eta"], dtype=np.float32)  # [N,1]
    jet_phi = np.asarray(axis["jet_phi"], dtype=np.float32)  # [N,1]
    jet_pt = np.asarray(axis["jet_pt"], dtype=np.float32)    # [N,1]
    jet_E = np.asarray(axis["jet_E"], dtype=np.float32)      # [N,1]

    pt = np.exp(np.clip(log_pt, -20.0, 20.0))
    dR = np.sqrt(dEta**2 + dPhi**2)
    log_pt_rel = np.log(pt / (jet_pt + 1e-8) + 1e-8)

    if k == "4d":
        log_E = feat[..., 3]
        E = np.exp(np.clip(log_E, -20.0, 20.0))
    elif k == "3d":
        # 3D features do not include energy, so fill E with a massless approximation using absolute eta.
        eta = dEta + jet_eta
        E = pt * np.cosh(np.clip(eta, -5.0, 5.0))
        log_E = np.log(E + 1e-8)
    else:
        raise ValueError(f"Unknown feature kind: {kind}")

    log_E_rel = np.log(E / (jet_E + 1e-8) + 1e-8)

    out = np.stack([dEta, dPhi, log_pt, log_E, log_pt_rel, log_E_rel, dR], axis=-1)
    out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def fm_make_bridge(x_post: torch.Tensor, x_pre: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flow Matching bridge:
      x_t = (1-t)*x_post + t*x_pre
      v*  = x_pre - x_post

    Args:
      x_post, x_pre: [B,S,D]
      t: [B] in [0,1]
    Returns:
      x_t: [B,S,D]
      v:   [B,S,D]
    """
    tt = t.view(-1, 1, 1).to(dtype=x_post.dtype, device=x_post.device)
    x_t = (1.0 - tt) * x_post + tt * x_pre
    v = x_pre - x_post
    return x_t, v


@torch.no_grad()
def fm_sample_euler(
    model,
    *,
    x0: torch.Tensor,
    cond: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 20,
) -> torch.Tensor:
    """
    Euler integration from t=0 -> 1:
      x_{k+1} = x_k + (1/steps) * v_theta(x_k, t_k; cond)
    """
    x = x0
    B = x.shape[0]
    for k in range(int(steps)):
        t = torch.full((B,), float(k) / float(max(1, steps)), device=x.device, dtype=x.dtype)
        v = model(x, cond, mask, t)
        dt = 1.0 / float(max(1, steps))
        x = x + dt * v
        # Keep padding tokens at 0.
        x = x * mask.to(x.dtype).unsqueeze(-1)
    return x


@torch.no_grad()
def fm_sample_heun(
    model,
    *,
    x0: torch.Tensor,
    cond: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 20,
) -> torch.Tensor:
    """Heun's method (RK2) integration from t=0 -> 1.

    This is often much more stable than Euler for the same number of steps.
    """
    x = x0
    B = x.shape[0]
    dt = 1.0 / float(max(1, steps))
    for k in range(int(steps)):
        t0 = float(k) / float(max(1, steps))
        t1 = float(k + 1) / float(max(1, steps))
        t_vec0 = torch.full((B,), t0, device=x.device, dtype=x.dtype)
        t_vec1 = torch.full((B,), t1, device=x.device, dtype=x.dtype)
        v0 = model(x, cond, mask, t_vec0)
        x_euler = x + dt * v0
        x_euler = x_euler * mask.to(x.dtype).unsqueeze(-1)
        v1 = model(x_euler, cond, mask, t_vec1)
        x = x + 0.5 * dt * (v0 + v1)
        x = x * mask.to(x.dtype).unsqueeze(-1)
    return x



# -----------------------------
# Downstream tagger utilities copied from `tagging/Baseline/tool.py`

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

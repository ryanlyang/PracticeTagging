"""
Unsmear utilities for the pure unsmear setup.

Goal:
- Build supervised pairs where the input is the smeared HLT view and the target
  is the corresponding unsmeared view.

Notes:
- Features are computed in the jet-axis frame.
- This pure setup does not include merge or efficiency-loss effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score, roc_curve


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
    """Wrap an angular difference into (-pi, pi] for torch loss computation."""
    return torch.atan2(torch.sin(dphi), torch.cos(dphi))


@dataclass
class HLTEffectsCfg:
    """HLT effects config for the pure unsmear setup."""

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
    """
    Build pure unsmear pairs `(pre_smear, post_smear)`.

    Args:
      const: [N,S,4] (pt,eta,phi,E)
      mask:  [N,S] bool
    Returns:
      pre_smear_const:  [N,S,4] after thresholding, before smearing
      post_smear_const: [N,S,4] after thresholding and smearing
      post_mask:        [N,S] bool mask used for training/evaluation
    """
    rs = np.random.RandomState(int(seed))
    hlt = const.copy()
    hlt_mask = mask.copy()

    # Apply the offline threshold first so offline/HLT start from the same token set.
    pt_thr_off = float(cfg.pt_threshold_offline)
    hlt_mask = hlt_mask & (hlt[:, :, 0] >= pt_thr_off)
    hlt[~hlt_mask] = 0.0

    # Apply the HLT threshold on top; in the pure setup we usually keep it equal to offline.
    pt_thr_hlt = float(cfg.pt_threshold_hlt)
    below = (hlt[:, :, 0] < pt_thr_hlt) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0.0

    # Snapshot before smearing.
    pre = hlt.copy()

    # Apply smearing only.
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
    # Recompute E using a massless approximation after smearing.
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5.0, 5.0)), 0.0)

    pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    pre[~hlt_mask] = 0.0
    hlt[~hlt_mask] = 0.0
    return pre.astype(np.float32), hlt.astype(np.float32), hlt_mask.astype(bool)


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
    """Regenerate unsmear training inputs for one epoch.

    Notes:
    - The target still uses features from the unsmeared view in its own axis frame.
    - The input is re-smeared and the post-smear axis is recomputed, so the jet axis can change each epoch.
    - All returned arrays are already standardized with the given means/stds and can be fed directly into `UnsmearJetDataset`.
    """
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


class UnsmearJetDataset(Dataset):
    """Jet-level dataset: predict per-token target features from smeared inputs."""

    def __init__(
        self,
        x_post: np.ndarray,
        y_pre: np.ndarray,
        mask: np.ndarray,
    ):
        self.x_post = torch.tensor(x_post, dtype=torch.float32)
        self.y_pre = torch.tensor(y_pre, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.x_post.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x_post[i], "y": self.y_pre[i], "mask": self.mask[i]}


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


def fm_make_bridge(
    x_post: torch.Tensor,
    x_pre: torch.Tensor,
    t: torch.Tensor,
    *,
    dphi_idx: int | None = None,
    dphi_mean: float | torch.Tensor = 0.0,
    dphi_scale: float | torch.Tensor = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flow Matching bridge:
      x_t = (1-t)*x_post + t*x_pre
      v*  = x_pre - x_post

    If `dphi_idx` is provided, that dimension is treated as an angular variable
    in standardized space. We then build the bridge along the shortest wrapped
    path in angle space rather than by direct linear interpolation in the
    standardized coordinate.

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

    if dphi_idx is not None:
        idx = int(dphi_idx)
        mean = (
            dphi_mean
            if isinstance(dphi_mean, torch.Tensor)
            else torch.tensor(float(dphi_mean), device=x_post.device, dtype=x_post.dtype)
        )
        scale = (
            dphi_scale
            if isinstance(dphi_scale, torch.Tensor)
            else torch.tensor(float(dphi_scale), device=x_post.device, dtype=x_post.dtype)
        )
        phi_post = x_post[..., idx] * scale + mean
        phi_pre = x_pre[..., idx] * scale + mean
        delta_phi = wrap_dphi_torch(phi_pre - phi_post)
        phi_t = wrap_dphi_torch(phi_post + tt.squeeze(-1) * delta_phi)

        x_t = x_t.clone()
        v = v.clone()
        x_t[..., idx] = (phi_t - mean) / scale
        v[..., idx] = delta_phi / scale

    return x_t, v


@torch.no_grad()
def fm_sample_euler(
    model,
    *,
    x0: torch.Tensor,
    cond: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 20,
    dphi_idx: int | None = None,
    dphi_mean: float | torch.Tensor = 0.0,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Euler integration from t=0 -> 1:
      x_{k+1} = x_k + (1/steps) * v_theta(x_k, t_k; cond)
    """
    x = x0
    B = x.shape[0]
    mean = None
    scale = None
    if dphi_idx is not None:
        mean = (
            dphi_mean
            if isinstance(dphi_mean, torch.Tensor)
            else torch.tensor(float(dphi_mean), device=x.device, dtype=x.dtype)
        )
        scale = (
            dphi_scale
            if isinstance(dphi_scale, torch.Tensor)
            else torch.tensor(float(dphi_scale), device=x.device, dtype=x.dtype)
        )
    for k in range(int(steps)):
        t = torch.full((B,), float(k) / float(max(1, steps)), device=x.device, dtype=x.dtype)
        v = model(x, cond, mask, t)
        dt = 1.0 / float(max(1, steps))
        x = x + dt * v
        if dphi_idx is not None:
            x = x.clone()
            x[..., int(dphi_idx)] = (wrap_dphi_torch(x[..., int(dphi_idx)] * scale + mean) - mean) / scale
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
    dphi_idx: int | None = None,
    dphi_mean: float | torch.Tensor = 0.0,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Heun's method (RK2) integration from t=0 -> 1.

    This is often much more stable than Euler for the same number of steps.
    """
    x = x0
    B = x.shape[0]
    dt = 1.0 / float(max(1, steps))
    mean = None
    scale = None
    if dphi_idx is not None:
        mean = (
            dphi_mean
            if isinstance(dphi_mean, torch.Tensor)
            else torch.tensor(float(dphi_mean), device=x.device, dtype=x.dtype)
        )
        scale = (
            dphi_scale
            if isinstance(dphi_scale, torch.Tensor)
            else torch.tensor(float(dphi_scale), device=x.device, dtype=x.dtype)
        )
    for k in range(int(steps)):
        t0 = float(k) / float(max(1, steps))
        t1 = float(k + 1) / float(max(1, steps))
        t_vec0 = torch.full((B,), t0, device=x.device, dtype=x.dtype)
        t_vec1 = torch.full((B,), t1, device=x.device, dtype=x.dtype)
        v0 = model(x, cond, mask, t_vec0)
        x_euler = x + dt * v0
        if dphi_idx is not None:
            x_euler = x_euler.clone()
            x_euler[..., int(dphi_idx)] = (
                wrap_dphi_torch(x_euler[..., int(dphi_idx)] * scale + mean) - mean
            ) / scale
        x_euler = x_euler * mask.to(x.dtype).unsqueeze(-1)
        v1 = model(x_euler, cond, mask, t_vec1)
        x = x + 0.5 * dt * (v0 + v1)
        if dphi_idx is not None:
            x = x.clone()
            x[..., int(dphi_idx)] = (wrap_dphi_torch(x[..., int(dphi_idx)] * scale + mean) - mean) / scale
        x = x * mask.to(x.dtype).unsqueeze(-1)
    return x



# -----------------------------
# Downstream tagger utilities
# -----------------------------

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

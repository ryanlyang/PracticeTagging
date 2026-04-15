#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Encoder-decoder continuous autoregressive reconstructor + dual-view top tagging.

Key ideas:
- HLT tokens are encoded with a transformer encoder.
- Offline constituents are decoded autoregressively in a continuous token space.
- Decoder uses pointer-style copy from encoded HLT tokens plus learned residual edits.
- Loss = autoregressive Huber + set-level Chamfer + EOS/count supervision.
- Inference uses beam over sequence length hypotheses (EOS trajectories) and a
  confidence-weighted soft view to build reconstructed jets.

This script intentionally keeps the same high-level plumbing as the prior m2-style
pipeline: data loading, pseudo-HLT generation, split handling, teacher/baseline
training, reconstruction, dual-view training, and standard evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY_HUNGARIAN = True
except Exception:
    linear_sum_assignment = None  # type: ignore
    _HAS_SCIPY_HUNGARIAN = False

from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
    fpr_at_target_tpr,
    plot_budget_diagnostics,
    plot_constituent_count_diagnostics,
    plot_roc,
    train_dual_view_classifier,
    train_single_view_classifier,
)
from unmerge_correct_hlt import (
    RANDOM_SEED,
    DualViewCrossAttnClassifier,
    DualViewJetDataset,
    JetDataset,
    ParticleTransformer,
    build_pt_edges,
    compute_features,
    compute_jet_pt,
    eval_classifier,
    eval_classifier_dual,
    get_scheduler,
    get_stats,
    jet_response_resolution,
    load_raw_constituents_from_h5,
    plot_response_resolution,
    train_classifier,
    train_classifier_dual,
    standardize,
)


# ----------------------------- Reproducibility ----------------------------- #
def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------- Defaults ------------------------------------ #
MODEL_CFG = {
    "embed_dim": 160,
    "num_heads": 8,
    "num_layers": 6,
    "ff_dim": 640,
    "dropout": 0.1,
}

CLS_TRAIN_CFG = {
    "batch_size": 512,
    "epochs": 70,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 4,
    "patience": 18,
}

RECO_CFG = {
    "embed_dim": 384,
    "num_heads": 8,
    "num_enc_layers": 6,
    "num_dec_layers": 6,
    "ff_dim": 1024,
    "dropout": 0.1,
    "max_decode_tokens": 100,
}

RECO_TRAIN_CFG = {
    "batch_size": 128,
    "epochs": 120,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "patience": 25,
    "min_epochs": 40,
}

LOSS_CFG = {
    "w_ar": 1.0,
    "w_set": 0.85,
    "w_eos": 0.20,
    "w_count": 0.20,
    "w_ptr_entropy": 0.002,
    "w_angle": 0.0,
    "w_jetpt": 0.0,
    "w_jete": 0.0,
    "w_4vec": 0.0,
    "set_loss_mode": "chamfer",
    "set_unmatched_penalty": 0.35,
    "huber_delta": 0.12,
    "scale_sensitive_weighting": False,
    "scale_pt_power": 0.6,
    "scale_weight_cap": 4.0,
    "angle_pt_power": 0.6,
    "physics_warmup_epochs": 12,
    "w_best_set": 2.5,
    "w_diversity": 0.08,
    "selector_ce": 0.6,
    "selector_rank": 0.2,
    "selector_rank_margin": 0.25,
    "winner_mode": "tag",
    "winner_hybrid_alpha": 1.0,
    "winner_hybrid_beta": 0.5,
}

MULTIHYP_CFG = {
    "num_hypotheses": 1,
    "joint_epochs": 14,
    "joint_lr": 1.2e-4,
    "joint_patience": 6,
}

TOKEN_DIM_WEIGHTS = torch.tensor([1.0, 0.35, 0.25, 0.25, 1.0], dtype=torch.float32)


# ----------------------------- Utilities ----------------------------------- #
def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def sort_constituents_by_pt_np(const: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort each jet's tokens by descending pT (masked tokens go last)."""
    pt = np.where(mask, const[:, :, 0], -1e9)
    order = np.argsort(-pt, axis=1)
    c_sorted = np.take_along_axis(const, order[:, :, None], axis=1).astype(np.float32)
    m_sorted = np.take_along_axis(mask, order, axis=1)
    c_sorted[~m_sorted] = 0.0
    return c_sorted, m_sorted, order


def reorder_features_np(feat: np.ndarray, order: np.ndarray, mask_sorted: np.ndarray) -> np.ndarray:
    out = np.take_along_axis(feat, order[:, :, None], axis=1).astype(np.float32)
    out[~mask_sorted] = 0.0
    return out


def const_to_token_np(const: np.ndarray) -> np.ndarray:
    """[pt,eta,phi,E] -> [log_pt, eta, sin_phi, cos_phi, log_E]."""
    pt = np.clip(const[..., 0], 1e-8, None)
    eta = np.clip(const[..., 1], -5.0, 5.0)
    phi = const[..., 2]
    e = np.clip(const[..., 3], 1e-8, None)
    tok = np.stack(
        [
            np.log(pt),
            eta,
            np.sin(phi),
            np.cos(phi),
            np.log(e),
        ],
        axis=-1,
    ).astype(np.float32)
    tok = np.nan_to_num(tok, nan=0.0, posinf=0.0, neginf=0.0)
    return tok


def token_to_const_torch(tok: torch.Tensor) -> torch.Tensor:
    """[log_pt, eta, sin_phi, cos_phi, log_E] -> [pt, eta, phi, E]."""
    log_pt = tok[..., 0]
    eta = torch.clamp(tok[..., 1], -5.0, 5.0)
    sin_phi = tok[..., 2]
    cos_phi = tok[..., 3]
    norm = torch.sqrt(sin_phi.pow(2) + cos_phi.pow(2) + 1e-8)
    sin_phi = sin_phi / norm
    cos_phi = cos_phi / norm
    phi = torch.atan2(sin_phi, cos_phi)
    log_e = tok[..., 4]

    pt = torch.exp(torch.clamp(log_pt, min=-20.0, max=20.0))
    e = torch.exp(torch.clamp(log_e, min=-20.0, max=20.0))
    e_floor = pt * torch.cosh(eta)
    e = torch.maximum(e, e_floor)

    out = torch.stack([pt, eta, phi, e], dim=-1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def const_to_token_torch(const: torch.Tensor) -> torch.Tensor:
    pt = torch.clamp(const[..., 0], min=1e-8)
    eta = torch.clamp(const[..., 1], min=-5.0, max=5.0)
    phi = const[..., 2]
    e = torch.clamp(const[..., 3], min=1e-8)
    tok = torch.stack(
        [
            torch.log(pt),
            eta,
            torch.sin(phi),
            torch.cos(phi),
            torch.log(e),
        ],
        dim=-1,
    )
    tok = torch.nan_to_num(tok, nan=0.0, posinf=0.0, neginf=0.0)
    return tok


def compute_features_torch(const: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Torch version of unmerge_correct_hlt.compute_features."""
    pt = torch.clamp(const[..., 0], min=1e-8)
    eta = torch.clamp(const[..., 1], min=-5.0, max=5.0)
    phi = const[..., 2]
    E = torch.clamp(const[..., 3], min=1e-8)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    m = mask.float()
    jet_px = (px * m).sum(dim=1, keepdim=True)
    jet_py = (py * m).sum(dim=1, keepdim=True)
    jet_pz = (pz * m).sum(dim=1, keepdim=True)
    jet_E = (E * m).sum(dim=1, keepdim=True)

    jet_pt = torch.sqrt(jet_px.pow(2) + jet_py.pow(2) + 1e-8)
    jet_p = torch.sqrt(jet_px.pow(2) + jet_py.pow(2) + jet_pz.pow(2) + 1e-8)
    frac = torch.clamp((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), min=1e-8, max=1e8)
    jet_eta = 0.5 * torch.log(frac)
    jet_phi = torch.atan2(jet_py, jet_px)

    delta_eta = eta - jet_eta
    delta_phi = torch.atan2(torch.sin(phi - jet_phi), torch.cos(phi - jet_phi))

    log_pt = torch.log(pt + 1e-8)
    log_E = torch.log(E + 1e-8)
    log_pt_rel = torch.log(pt / (jet_pt + 1e-8) + 1e-8)
    log_E_rel = torch.log(E / (jet_E + 1e-8) + 1e-8)
    delta_r = torch.sqrt(delta_eta.pow(2) + delta_phi.pow(2) + 1e-8)

    feat = torch.stack([delta_eta, delta_phi, log_pt, log_E, log_pt_rel, log_E_rel, delta_r], dim=-1)
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = torch.clamp(feat, min=-20.0, max=20.0)
    feat = torch.where(mask.unsqueeze(-1), feat, torch.zeros_like(feat))
    return feat


def standardize_features_torch(
    feat: torch.Tensor,
    mask: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> torch.Tensor:
    out = torch.clamp((feat - means.view(1, 1, -1)) / stds.view(1, 1, -1), min=-10.0, max=10.0)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.where(mask.unsqueeze(-1), out, torch.zeros_like(out))
    return out


def compute_jet_pt_torch(const: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pt = torch.clamp(const[..., 0], min=0.0)
    phi = const[..., 2]
    w = mask.float()
    px = (pt * torch.cos(phi) * w).sum(dim=1)
    py = (pt * torch.sin(phi) * w).sum(dim=1)
    return torch.sqrt(px.pow(2) + py.pow(2) + 1e-8)


def gather_hypothesis(x: torch.Tensor, winner_idx: torch.Tensor) -> torch.Tensor:
    """Gather x[:,k,...] using winner_idx[B]. x is [B,K,...]."""
    bsz = int(x.shape[0])
    gather_idx = winner_idx.view(bsz, 1, *([1] * (x.ndim - 2))).expand(bsz, 1, *x.shape[2:])
    return torch.gather(x, dim=1, index=gather_idx).squeeze(1)


def huber_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
    scale_sensitive_weighting: bool = False,
    scale_pt_power: float = 0.6,
    scale_weight_cap: float = 4.0,
) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    d = float(max(delta, 1e-6))
    loss = torch.where(abs_diff <= d, 0.5 * abs_diff.pow(2) / d, abs_diff - 0.5 * d)
    # Slightly prioritize energy/pt dimensions.
    w = TOKEN_DIM_WEIGHTS.to(loss.device).view(1, 1, -1)
    loss = (loss * w).sum(dim=-1)
    m = mask.float()

    if bool(scale_sensitive_weighting):
        pt = torch.exp(torch.clamp(target[..., 0], min=-20.0, max=20.0))
        pt_ref = (pt * m).sum(dim=1, keepdim=True) / (m.sum(dim=1, keepdim=True) + 1e-6)
        w_pt = torch.pow(torch.clamp(pt / (pt_ref + 1e-6), min=0.1), float(scale_pt_power))
        w_pt = torch.clamp(w_pt, min=0.25, max=float(max(scale_weight_cap, 1.0)))
        loss = loss * w_pt

    return (loss * m).sum() / (m.sum() + 1e-6)


def wrapped_dphi(phi_pred: torch.Tensor, phi_true: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phi_pred - phi_true), torch.cos(phi_pred - phi_true))


def angle_loss_weighted(
    pred_tok: torch.Tensor,
    tgt_tok: torch.Tensor,
    mask: torch.Tensor,
    angle_pt_power: float = 0.6,
    huber_delta: float = 0.10,
) -> torch.Tensor:
    pred_const = token_to_const_torch(pred_tok)
    tgt_const = token_to_const_torch(tgt_tok)

    dphi = wrapped_dphi(pred_const[..., 2], tgt_const[..., 2]).abs()
    d = float(max(huber_delta, 1e-6))
    l = torch.where(dphi <= d, 0.5 * dphi.pow(2) / d, dphi - 0.5 * d)

    m = mask.float()
    pt = torch.clamp(tgt_const[..., 0], min=1e-6)
    pt_ref = (pt * m).sum(dim=1, keepdim=True) / (m.sum(dim=1, keepdim=True) + 1e-6)
    w = torch.pow(torch.clamp(pt / (pt_ref + 1e-6), min=0.1), float(angle_pt_power))
    w = torch.clamp(w, min=0.25, max=6.0)

    return (l * w * m).sum() / (m.sum() + 1e-6)


def fourvec_sums(const: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = mask.float()
    pt = const[..., 0]
    eta = const[..., 1]
    phi = const[..., 2]
    E = const[..., 3]
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return (
        (px * m).sum(dim=1),
        (py * m).sum(dim=1),
        (pz * m).sum(dim=1),
        (E * m).sum(dim=1),
    )


def jet_global_losses(
    pred_tok: torch.Tensor,
    tgt_tok: torch.Tensor,
    pred_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-6
    pred_const = token_to_const_torch(pred_tok)
    tgt_const = token_to_const_torch(tgt_tok)

    pred_pt = (pred_const[..., 0] * pred_mask.float()).sum(dim=1)
    tgt_pt = (tgt_const[..., 0] * tgt_mask.float()).sum(dim=1)
    loss_jetpt = F.smooth_l1_loss(torch.log(pred_pt + eps), torch.log(tgt_pt + eps))

    pred_E = (pred_const[..., 3] * pred_mask.float()).sum(dim=1)
    tgt_E = (tgt_const[..., 3] * tgt_mask.float()).sum(dim=1)
    loss_jete = F.smooth_l1_loss(torch.log(pred_E + eps), torch.log(tgt_E + eps))

    ppx, ppy, ppz, pE = fourvec_sums(pred_const, pred_mask)
    tpx, tpy, tpz, tE = fourvec_sums(tgt_const, tgt_mask)
    norm = (tpx.abs() + tpy.abs() + tpz.abs() + tE.abs() + 1.0)
    loss_4vec = (((ppx - tpx).abs() + (ppy - tpy).abs() + (ppz - tpz).abs() + (pE - tE).abs()) / norm).mean()

    return loss_jetpt, loss_jete, loss_4vec


def chamfer_token_loss(
    pred_tok: torch.Tensor,
    tgt_tok: torch.Tensor,
    pred_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> torch.Tensor:
    w = TOKEN_DIM_WEIGHTS.to(pred_tok.device).view(1, 1, -1)
    p = pred_tok * w
    t = tgt_tok * w
    cost = torch.cdist(p, t, p=1)
    big = torch.full_like(cost, 1e4)

    cost_p = torch.where(tgt_mask.unsqueeze(1), cost, big)
    p2t = cost_p.min(dim=2).values
    p2t = (p2t * pred_mask.float()).sum(dim=1) / (pred_mask.float().sum(dim=1) + 1e-6)

    cost_t = torch.where(pred_mask.unsqueeze(2), cost, big)
    t2p = cost_t.min(dim=1).values
    t2p = (t2p * tgt_mask.float()).sum(dim=1) / (tgt_mask.float().sum(dim=1) + 1e-6)

    return (p2t + t2p).mean()


def hungarian_token_loss(
    pred_tok: torch.Tensor,
    tgt_tok: torch.Tensor,
    pred_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    unmatched_penalty: float = 0.35,
) -> torch.Tensor:
    if (not _HAS_SCIPY_HUNGARIAN) or (linear_sum_assignment is None):
        raise RuntimeError(
            "set_loss_mode='hungarian' requires scipy.optimize.linear_sum_assignment, "
            "but SciPy is unavailable in this environment."
        )

    w = TOKEN_DIM_WEIGHTS.to(pred_tok.device).view(1, 1, -1)
    p = pred_tok * w
    t = tgt_tok * w
    bsz = int(pred_tok.shape[0])
    up = float(max(unmatched_penalty, 0.0))
    losses: List[torch.Tensor] = []

    for bi in range(bsz):
        p_idx = torch.nonzero(pred_mask[bi], as_tuple=False).squeeze(-1)
        t_idx = torch.nonzero(tgt_mask[bi], as_tuple=False).squeeze(-1)
        n_p = int(p_idx.numel())
        n_t = int(t_idx.numel())

        if n_p == 0 and n_t == 0:
            losses.append(torch.zeros((), device=pred_tok.device, dtype=pred_tok.dtype))
            continue
        if n_p == 0:
            losses.append(torch.full((), up * float(n_t), device=pred_tok.device, dtype=pred_tok.dtype))
            continue
        if n_t == 0:
            losses.append(torch.full((), up * float(n_p), device=pred_tok.device, dtype=pred_tok.dtype))
            continue

        p_sel = p[bi, p_idx, :]  # [n_p, d]
        t_sel = t[bi, t_idx, :]  # [n_t, d]
        cost = torch.cdist(p_sel, t_sel, p=1)  # [n_p, n_t]
        c_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(c_np)  # type: ignore[misc]
        row_t = torch.as_tensor(row_ind, device=pred_tok.device, dtype=torch.long)
        col_t = torch.as_tensor(col_ind, device=pred_tok.device, dtype=torch.long)

        matched = cost[row_t, col_t]
        loss_i = matched.mean()

        if n_p > row_t.numel():
            used_rows = torch.zeros(n_p, dtype=torch.bool, device=pred_tok.device)
            used_rows[row_t] = True
            extra_rows = torch.nonzero(~used_rows, as_tuple=False).squeeze(-1)
            if extra_rows.numel() > 0:
                loss_i = loss_i + up * cost[extra_rows, :].min(dim=1).values.mean()

        if n_t > col_t.numel():
            used_cols = torch.zeros(n_t, dtype=torch.bool, device=pred_tok.device)
            used_cols[col_t] = True
            extra_cols = torch.nonzero(~used_cols, as_tuple=False).squeeze(-1)
            if extra_cols.numel() > 0:
                loss_i = loss_i + up * cost[:, extra_cols].min(dim=0).values.mean()

        losses.append(loss_i)

    return torch.stack(losses, dim=0).mean()


def build_fixed_split_indices(
    labels: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_all = np.arange(labels.shape[0])
    n_total = int(n_train + n_val + n_test)
    if n_total > labels.shape[0]:
        raise ValueError(
            f"Requested split train+val+test={n_total} exceeds available jets={labels.shape[0]}"
        )

    # Sample n_total first (stratified), then split exactly to train/val/test.
    if n_total < labels.shape[0]:
        idx_sel, _ = train_test_split(
            idx_all,
            train_size=n_total,
            random_state=int(seed),
            stratify=labels,
        )
    else:
        idx_sel = idx_all

    labels_sel = labels[idx_sel]
    idx_train, idx_tmp = train_test_split(
        idx_sel,
        train_size=n_train,
        random_state=int(seed),
        stratify=labels_sel,
    )

    labels_tmp = labels[idx_tmp]
    idx_val, idx_test = train_test_split(
        idx_tmp,
        train_size=n_val,
        test_size=n_test,
        random_state=int(seed),
        stratify=labels_tmp,
    )
    return idx_train, idx_val, idx_test


def train_single_view_classifier_by_auc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    sch = get_scheduler(opt, train_cfg["warmup_epochs"], train_cfg["epochs"])
    best_val_auc = -1.0
    best_state = None
    no_improve = 0

    for ep in range(int(train_cfg["epochs"])):
        _, tr_auc = train_classifier(model, train_loader, opt, device)
        va_auc, va_preds, va_labs = eval_classifier(model, val_loader, device)
        va_fpr, va_tpr, _ = roc_curve(va_labs, va_preds)
        va_fpr50 = fpr_at_target_tpr(va_fpr, va_tpr, 0.50)
        sch.step()

        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, "
                f"val_fpr50={va_fpr50:.6f}, best_auc={best_val_auc:.4f}"
            )
        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_dual_view_classifier_by_auc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    sch = get_scheduler(opt, train_cfg["warmup_epochs"], train_cfg["epochs"])
    best_val_auc = -1.0
    best_state = None
    no_improve = 0

    for ep in range(int(train_cfg["epochs"])):
        _, tr_auc = train_classifier_dual(model, train_loader, opt, device)
        va_auc, va_preds, va_labs = eval_classifier_dual(model, val_loader, device)
        va_fpr, va_tpr, _ = roc_curve(va_labs, va_preds)
        va_fpr50 = fpr_at_target_tpr(va_fpr, va_tpr, 0.50)
        sch.step()

        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, "
                f"val_fpr50={va_fpr50:.6f}, best_auc={best_val_auc:.4f}"
            )
        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------------------- Datasets ------------------------------------ #
class RecoSeqDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        tgt_tok: np.ndarray,
        tgt_mask: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.tgt_tok = torch.tensor(tgt_tok, dtype=torch.float32)
        self.tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "tgt_tok": self.tgt_tok[i],
            "tgt_mask": self.tgt_mask[i],
            "label": self.labels[i],
        }


class RecoInputDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray, const_hlt: np.ndarray):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
        }


# ----------------------------- Model --------------------------------------- #
class HLT2OfflineSeq2Seq(nn.Module):
    def __init__(
        self,
        input_dim_hlt: int = 7,
        token_dim: int = 5,
        embed_dim: int = 384,
        num_heads: int = 8,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_hlt_tokens: int = 100,
        max_decode_tokens: int = 100,
        use_coord_residual_param: bool = False,
        num_hypotheses: int = 1,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.max_decode_tokens = int(max_decode_tokens)
        self.use_coord_residual_param = bool(use_coord_residual_param)
        self.num_hypotheses = int(max(num_hypotheses, 1))

        self.enc_in = nn.Sequential(
            nn.Linear(input_dim_hlt, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hlt_pos = nn.Parameter(torch.zeros(1, max_hlt_tokens, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        self.dec_in = nn.Sequential(
            nn.Linear(token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dec_pos = nn.Parameter(torch.zeros(1, max_decode_tokens + 1, embed_dim))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4 if self.use_coord_residual_param else token_dim),
        )
        self.gate_head = nn.Linear(embed_dim, 1)
        self.stop_head = nn.Linear(embed_dim, 1)
        self.count_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.hyp_embed = nn.Embedding(self.num_hypotheses, embed_dim)

        self.bos_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        nn.init.normal_(self.hlt_pos, std=0.02)
        nn.init.normal_(self.dec_pos, std=0.02)
        nn.init.normal_(self.bos_token, std=0.02)
        nn.init.normal_(self.hyp_embed.weight, std=0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

    def encode(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = feat_hlt.shape
        x = self.enc_in(feat_hlt) + self.hlt_pos[:, :L, :]
        mem = self.encoder(x, src_key_padding_mask=~mask_hlt)

        hlt_tok = const_to_token_torch(const_hlt)

        m = mask_hlt.float()
        pooled = (mem * m.unsqueeze(-1)).sum(dim=1) / (m.sum(dim=1, keepdim=True) + 1e-6)
        count_pred = F.softplus(self.count_head(pooled).squeeze(-1))
        return mem, hlt_tok, count_pred

    def _predict_from_hidden(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        mask_hlt: torch.Tensor,
        hlt_tok: torch.Tensor,
        hyp_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # h: [B, T, D]
        if self.num_hypotheses > 1:
            hv = self.hyp_embed.weight[int(hyp_idx)].view(1, 1, -1)
            h = h + hv
        q = self.q_proj(h)
        logits = torch.matmul(q, mem.transpose(1, 2)) / math.sqrt(float(q.shape[-1]))
        logits = logits.masked_fill((~mask_hlt).unsqueeze(1), -1e9)
        attn = torch.softmax(logits, dim=-1)

        base_tok = torch.matmul(attn, hlt_tok)
        delta = self.delta_head(h)
        gate = torch.sigmoid(self.gate_head(h))

        if self.use_coord_residual_param:
            # Coordinate-aware copy/edit: residuals in physics space.
            base_const = token_to_const_torch(base_tok)
            deta = delta[..., 0]
            dphi = delta[..., 1]
            dlogpt = delta[..., 2]
            dloge = delta[..., 3]
            g = gate.squeeze(-1)

            eta = torch.clamp(base_const[..., 1] + g * deta, min=-5.0, max=5.0)
            phi = torch.atan2(torch.sin(base_const[..., 2] + g * dphi), torch.cos(base_const[..., 2] + g * dphi))
            pt = base_const[..., 0] * torch.exp(torch.clamp(g * dlogpt, min=-3.0, max=3.0))
            e = base_const[..., 3] * torch.exp(torch.clamp(g * dloge, min=-3.0, max=3.0))
            e_floor = pt * torch.cosh(eta)
            e = torch.maximum(e, e_floor)
            pred_const = torch.stack([pt, eta, phi, e], dim=-1)
            pred_tok = const_to_token_torch(pred_const)
        else:
            pred_tok = base_tok + gate * delta
            # keep sin/cos channel normalized (avoid in-place autograd edits)
            sn = pred_tok[..., 2]
            cs = pred_tok[..., 3]
            norm = torch.sqrt(sn.pow(2) + cs.pow(2) + 1e-8)
            pred_tok = torch.cat(
                [pred_tok[..., :2], (sn / norm).unsqueeze(-1), (cs / norm).unsqueeze(-1), pred_tok[..., 4:]],
                dim=-1,
            )

        stop_logits = self.stop_head(h).squeeze(-1)
        return pred_tok, stop_logits, attn, gate.squeeze(-1)

    def forward_teacher(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        tgt_tok: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = tgt_tok.shape
        mem, hlt_tok, count_pred = self.encode(feat_hlt, mask_hlt, const_hlt)

        bos = self.bos_token.expand(B, 1, self.token_dim)
        dec_in_tok = torch.cat([bos, tgt_tok[:, :-1, :]], dim=1)
        dec_in = self.dec_in(dec_in_tok) + self.dec_pos[:, :T, :]

        h = self.decoder(
            dec_in,
            mem,
            tgt_mask=self._causal_mask(T, dec_in.device),
            memory_key_padding_mask=~mask_hlt,
        )
        pred_tok, stop_logits, attn, gate = self._predict_from_hidden(h, mem, mask_hlt, hlt_tok)
        return {
            "pred_tok": pred_tok,
            "stop_logits": stop_logits,
            "count_pred": count_pred,
            "attn": attn,
            "gate": gate,
        }

    def forward_teacher_multi(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        tgt_tok: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = tgt_tok.shape
        mem, hlt_tok, count_pred = self.encode(feat_hlt, mask_hlt, const_hlt)

        bos = self.bos_token.expand(B, 1, self.token_dim)
        dec_in_tok = torch.cat([bos, tgt_tok[:, :-1, :]], dim=1)
        dec_in = self.dec_in(dec_in_tok) + self.dec_pos[:, :T, :]
        h = self.decoder(
            dec_in,
            mem,
            tgt_mask=self._causal_mask(T, dec_in.device),
            memory_key_padding_mask=~mask_hlt,
        )

        pred_toks: List[torch.Tensor] = []
        stop_logits_list: List[torch.Tensor] = []
        attns: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        for k in range(self.num_hypotheses):
            p, s, a, g = self._predict_from_hidden(h, mem, mask_hlt, hlt_tok, hyp_idx=k)
            pred_toks.append(p)
            stop_logits_list.append(s)
            attns.append(a)
            gates.append(g)

        return {
            "pred_tok": torch.stack(pred_toks, dim=1),       # [B,K,T,D]
            "stop_logits": torch.stack(stop_logits_list, dim=1),  # [B,K,T]
            "count_pred": count_pred,                        # [B]
            "attn": torch.stack(attns, dim=1),               # [B,K,T,L]
            "gate": torch.stack(gates, dim=1),               # [B,K,T]
        }

    @torch.no_grad()
    def decode_greedy(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        max_steps: int,
    ) -> Dict[str, torch.Tensor]:
        B = feat_hlt.shape[0]
        mem, hlt_tok, count_pred = self.encode(feat_hlt, mask_hlt, const_hlt)

        prev_tok = self.bos_token.expand(B, 1, self.token_dim)
        pred_seq = []
        stop_seq = []

        for t in range(int(max_steps)):
            d_in = self.dec_in(prev_tok) + self.dec_pos[:, : prev_tok.shape[1], :]
            h = self.decoder(
                d_in,
                mem,
                tgt_mask=self._causal_mask(prev_tok.shape[1], d_in.device),
                memory_key_padding_mask=~mask_hlt,
            )
            h_last = h[:, -1:, :]
            pred_tok, stop_logits, _attn, _gate = self._predict_from_hidden(h_last, mem, mask_hlt, hlt_tok)
            pred_seq.append(pred_tok[:, 0, :])
            stop_seq.append(stop_logits[:, 0])
            prev_tok = torch.cat([prev_tok, pred_tok], dim=1)

        pred_tok_full = torch.stack(pred_seq, dim=1)
        stop_logits_full = torch.stack(stop_seq, dim=1)
        stop_probs = torch.sigmoid(stop_logits_full)
        return {
            "pred_tok": pred_tok_full,
            "stop_probs": stop_probs,
            "count_pred": count_pred,
        }

    @torch.no_grad()
    def decode_greedy_multi(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        max_steps: int,
    ) -> Dict[str, torch.Tensor]:
        B = feat_hlt.shape[0]
        mem, hlt_tok, count_pred = self.encode(feat_hlt, mask_hlt, const_hlt)

        all_pred: List[torch.Tensor] = []
        all_stop: List[torch.Tensor] = []
        for k in range(self.num_hypotheses):
            prev_tok = self.bos_token.expand(B, 1, self.token_dim)
            pred_seq = []
            stop_seq = []
            for _t in range(int(max_steps)):
                d_in = self.dec_in(prev_tok) + self.dec_pos[:, : prev_tok.shape[1], :]
                h = self.decoder(
                    d_in,
                    mem,
                    tgt_mask=self._causal_mask(prev_tok.shape[1], d_in.device),
                    memory_key_padding_mask=~mask_hlt,
                )
                h_last = h[:, -1:, :]
                pred_tok, stop_logits, _attn, _gate = self._predict_from_hidden(
                    h_last, mem, mask_hlt, hlt_tok, hyp_idx=k
                )
                pred_seq.append(pred_tok[:, 0, :])
                stop_seq.append(stop_logits[:, 0])
                prev_tok = torch.cat([prev_tok, pred_tok], dim=1)
            pred_tok_full = torch.stack(pred_seq, dim=1)
            stop_probs = torch.sigmoid(torch.stack(stop_seq, dim=1))
            all_pred.append(pred_tok_full)
            all_stop.append(stop_probs)

        return {
            "pred_tok": torch.stack(all_pred, dim=1),    # [B,K,T,D]
            "stop_probs": torch.stack(all_stop, dim=1),  # [B,K,T]
            "count_pred": count_pred,                    # [B]
        }


# ----------------------------- Reco train/eval ----------------------------- #
class HypothesisSelector(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_bkf: torch.Tensor) -> torch.Tensor:
        # feat_bkf: [B,K,F]
        return self.net(feat_bkf).squeeze(-1)


def selector_rank_loss(sel_logits: torch.Tensor, winner_idx: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    # Encourage winner score > others by margin. Avoid inplace writes on autograd tensors.
    bsz, _k = sel_logits.shape
    w = sel_logits.gather(1, winner_idx.view(-1, 1))  # [B,1]
    diff = float(margin) - (w - sel_logits)  # [B,K]
    rank_raw = F.relu(diff)

    keep = torch.ones_like(rank_raw, dtype=torch.bool)
    keep = keep.scatter(1, winner_idx.view(-1, 1), False)

    rank = rank_raw * keep.to(rank_raw.dtype)
    denom = keep.to(rank_raw.dtype).sum().clamp_min(1.0)
    return rank.sum() / denom


def chamfer_token_loss_vec(
    pred_tok: torch.Tensor,
    tgt_tok: torch.Tensor,
    pred_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> torch.Tensor:
    vals = []
    for bi in range(int(pred_tok.shape[0])):
        vals.append(chamfer_token_loss(pred_tok[bi:bi+1], tgt_tok[bi:bi+1], pred_mask[bi:bi+1], tgt_mask[bi:bi+1]))
    return torch.stack(vals, dim=0)


def hungarian_token_loss_vec(
    pred_tok: torch.Tensor,
    tgt_tok: torch.Tensor,
    pred_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    unmatched_penalty: float = 0.35,
) -> torch.Tensor:
    vals = []
    for bi in range(int(pred_tok.shape[0])):
        vals.append(
            hungarian_token_loss(
                pred_tok[bi:bi+1],
                tgt_tok[bi:bi+1],
                pred_mask[bi:bi+1],
                tgt_mask[bi:bi+1],
                unmatched_penalty=unmatched_penalty,
            )
        )
    return torch.stack(vals, dim=0)


@torch.no_grad()
def build_selector_features_torch(
    pred_tok_bktd: torch.Tensor,
    pred_mask_bt: torch.Tensor,
    hlt_const: torch.Tensor,
    hlt_mask: torch.Tensor,
    teacher: nn.Module,
    feat_means_t: torch.Tensor,
    feat_stds_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns feat [B,K,F], teacher_logits [B,K]
    bsz, k, t, _ = pred_tok_bktd.shape
    feats_all = []
    logits_all = []
    hlt_n = hlt_mask.float().sum(dim=1)
    hlt_pt = compute_jet_pt_torch(hlt_const, hlt_mask)
    base_mask = pred_mask_bt

    for hk in range(k):
        p_tok = pred_tok_bktd[:, hk, :, :]
        p_const = token_to_const_torch(p_tok)
        p_feat = compute_features_torch(p_const, base_mask)
        p_feat_std = standardize_features_torch(p_feat, base_mask, feat_means_t, feat_stds_t)
        p_logit = teacher(p_feat_std, base_mask).squeeze(-1)
        p_prob = torch.sigmoid(p_logit)
        p_entropy = -(p_prob * torch.log(p_prob.clamp(min=1e-8)) + (1.0 - p_prob) * torch.log((1.0 - p_prob).clamp(min=1e-8)))
        p_n = base_mask.float().sum(dim=1)
        p_pt = compute_jet_pt_torch(p_const, base_mask)
        pt_ratio_hlt = p_pt / (hlt_pt + 1e-8)

        feat_k = torch.stack(
            [
                p_logit,
                p_prob,
                p_entropy,
                p_n / 100.0,
                (p_n - hlt_n).abs() / 100.0,
                pt_ratio_hlt,
                (pt_ratio_hlt - 1.0).abs(),
            ],
            dim=-1,
        )
        feats_all.append(feat_k)
        logits_all.append(p_logit)

    return torch.stack(feats_all, dim=1), torch.stack(logits_all, dim=1)


def compute_reco_losses_multihyp(
    out_multi: Dict[str, torch.Tensor],
    tgt_tok: torch.Tensor,
    tgt_mask: torch.Tensor,
    labels: torch.Tensor,
    hlt_const: torch.Tensor,
    hlt_mask: torch.Tensor,
    teacher: Optional[nn.Module],
    feat_means_t: Optional[torch.Tensor],
    feat_stds_t: Optional[torch.Tensor],
    loss_cfg: Dict,
    physics_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    pred_tok_bktd = out_multi["pred_tok"]
    stop_logits_bkt = out_multi["stop_logits"]
    attn_bktl = out_multi["attn"]
    gate_bkt = out_multi["gate"]
    count_pred = out_multi["count_pred"]

    bsz, k, t, _ = pred_tok_bktd.shape
    device = pred_tok_bktd.device

    tgt_count = tgt_mask.float().sum(dim=1)
    steps = torch.arange(t, device=device).unsqueeze(0)
    pred_mask_bt = steps < tgt_count.long().unsqueeze(1)

    set_mode = str(loss_cfg.get("set_loss_mode", "chamfer")).strip().lower()
    set_vecs = []
    tag_vecs = []
    for hk in range(k):
        p_tok = pred_tok_bktd[:, hk, :, :]
        if set_mode == "hungarian":
            s_vec = hungarian_token_loss_vec(
                p_tok,
                tgt_tok,
                pred_mask_bt,
                tgt_mask,
                unmatched_penalty=float(loss_cfg.get("set_unmatched_penalty", 0.35)),
            )
        else:
            s_vec = chamfer_token_loss_vec(p_tok, tgt_tok, pred_mask_bt, tgt_mask)
        set_vecs.append(s_vec)

    set_mat = torch.stack(set_vecs, dim=1)  # [B,K]

    selector_feat = None
    teacher_logits = None
    if teacher is not None and feat_means_t is not None and feat_stds_t is not None:
        selector_feat, teacher_logits = build_selector_features_torch(
            pred_tok_bktd, pred_mask_bt, hlt_const, hlt_mask, teacher, feat_means_t, feat_stds_t
        )
        lab = labels.float().view(-1, 1).expand(-1, k)
        tag_mat = F.binary_cross_entropy_with_logits(teacher_logits, lab, reduction="none")
    else:
        tag_mat = set_mat.detach()

    mode = str(loss_cfg.get("winner_mode", "tag")).strip().lower()
    if mode == "reco":
        winner_score = set_mat.detach()
    elif mode == "hybrid":
        a = float(loss_cfg.get("winner_hybrid_alpha", 1.0))
        b = float(loss_cfg.get("winner_hybrid_beta", 0.5))
        winner_score = (a * tag_mat + b * set_mat).detach()
    else:
        winner_score = tag_mat.detach()

    winner_idx = torch.argmin(winner_score, dim=1)  # [B]

    out_sel = {
        "pred_tok": gather_hypothesis(pred_tok_bktd, winner_idx),
        "stop_logits": gather_hypothesis(stop_logits_bkt, winner_idx),
        "count_pred": count_pred,
        "attn": gather_hypothesis(attn_bktl, winner_idx),
        "gate": gather_hypothesis(gate_bkt, winner_idx),
    }

    base_losses = compute_reco_losses(out_sel, tgt_tok, tgt_mask, loss_cfg, physics_scale=physics_scale)

    # Strong "at least one hypothesis is very good" objective.
    best_set = set_mat.min(dim=1).values.mean()
    total = base_losses["total"] - float(loss_cfg["w_set"]) * base_losses["set"] + float(loss_cfg.get("w_best_set", 2.5)) * best_set

    # Encourage hypotheses to avoid collapse.
    div = torch.zeros((), device=device, dtype=pred_tok_bktd.dtype)
    n_pairs = 0
    m = pred_mask_bt.float().unsqueeze(1).unsqueeze(-1)  # [B,1,T,1]
    for i in range(k):
        for j in range(i + 1, k):
            d = (pred_tok_bktd[:, i, :, :] - pred_tok_bktd[:, j, :, :]).abs()
            d = (d * m[:, 0]).sum(dim=(1, 2)) / (m[:, 0].sum(dim=(1, 2)) + 1e-6)
            # minimizing exp(-d) pushes d up.
            div = div + torch.exp(-d).mean()
            n_pairs += 1
    if n_pairs > 0:
        div = div / float(n_pairs)
        total = total + float(loss_cfg.get("w_diversity", 0.08)) * div

    out = dict(base_losses)
    out["total"] = total
    out["best_set"] = best_set
    out["diversity"] = div
    out["winner_idx"] = winner_idx
    out["set_mat"] = set_mat
    out["tag_mat"] = tag_mat
    out["selector_feat"] = selector_feat
    return out


# ----------------------------- Reco train/eval ----------------------------- #
def compute_reco_losses(
    out: Dict[str, torch.Tensor],
    tgt_tok: torch.Tensor,
    tgt_mask: torch.Tensor,
    loss_cfg: Dict,
    physics_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    B, T, _ = tgt_tok.shape
    device = tgt_tok.device

    pred_tok = out["pred_tok"]
    stop_logits = out["stop_logits"]
    count_pred = out["count_pred"]
    attn = out["attn"]

    tgt_count = tgt_mask.float().sum(dim=1)

    loss_ar = huber_masked(
        pred_tok,
        tgt_tok,
        tgt_mask,
        delta=float(loss_cfg["huber_delta"]),
        scale_sensitive_weighting=bool(loss_cfg.get("scale_sensitive_weighting", False)),
        scale_pt_power=float(loss_cfg.get("scale_pt_power", 0.6)),
        scale_weight_cap=float(loss_cfg.get("scale_weight_cap", 4.0)),
    )

    steps = torch.arange(T, device=device).unsqueeze(0)
    stop_target = (steps >= tgt_count.long().unsqueeze(1)).float()
    loss_eos = F.binary_cross_entropy_with_logits(stop_logits, stop_target)

    pred_mask_for_set = steps < tgt_count.long().unsqueeze(1)
    set_mode = str(loss_cfg.get("set_loss_mode", "chamfer")).strip().lower()
    if set_mode == "chamfer":
        loss_set = chamfer_token_loss(pred_tok, tgt_tok, pred_mask_for_set, tgt_mask)
    elif set_mode == "hungarian":
        loss_set = hungarian_token_loss(
            pred_tok,
            tgt_tok,
            pred_mask_for_set,
            tgt_mask,
            unmatched_penalty=float(loss_cfg.get("set_unmatched_penalty", 0.35)),
        )
    else:
        raise ValueError(f"Unsupported set_loss_mode='{set_mode}'. Use 'chamfer' or 'hungarian'.")

    loss_count = F.smooth_l1_loss(count_pred, tgt_count)

    # Lower entropy encourages edit-from-copy behavior.
    ptr_entropy = -(attn * torch.log(attn.clamp(min=1e-8))).sum(dim=-1).mean()

    # Geometry + global consistency auxiliaries.
    loss_angle = angle_loss_weighted(
        pred_tok,
        tgt_tok,
        pred_mask_for_set,
        angle_pt_power=float(loss_cfg.get("angle_pt_power", 0.6)),
        huber_delta=max(float(loss_cfg.get("huber_delta", 0.12)) * 0.8, 0.05),
    )
    loss_jetpt, loss_jete, loss_4vec = jet_global_losses(
        pred_tok,
        tgt_tok,
        pred_mask_for_set,
        tgt_mask,
    )

    ps = float(max(min(physics_scale, 1.0), 0.0))

    total = (
        float(loss_cfg["w_ar"]) * loss_ar
        + float(loss_cfg["w_set"]) * loss_set
        + float(loss_cfg["w_eos"]) * loss_eos
        + float(loss_cfg["w_count"]) * loss_count
        + float(loss_cfg["w_ptr_entropy"]) * ptr_entropy
        + ps * float(loss_cfg.get("w_angle", 0.0)) * loss_angle
        + ps * float(loss_cfg.get("w_jetpt", 0.0)) * loss_jetpt
        + ps * float(loss_cfg.get("w_jete", 0.0)) * loss_jete
        + ps * float(loss_cfg.get("w_4vec", 0.0)) * loss_4vec
    )

    return {
        "total": total,
        "ar": loss_ar,
        "set": loss_set,
        "eos": loss_eos,
        "count": loss_count,
        "ptr_entropy": ptr_entropy,
        "angle": loss_angle,
        "jetpt": loss_jetpt,
        "jete": loss_jete,
        "fourvec": loss_4vec,
    }


@dataclass
class RecoValMetrics:
    val_total: float
    val_ar: float
    val_set: float
    val_eos: float
    val_count: float
    val_angle: float
    val_jetpt: float
    val_jete: float
    val_fourvec: float


def train_reconstructor_seq2seq(
    model: HLT2OfflineSeq2Seq,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
    teacher: Optional[nn.Module] = None,
    feat_means_t: Optional[torch.Tensor] = None,
    feat_stds_t: Optional[torch.Tensor] = None,
) -> Tuple[HLT2OfflineSeq2Seq, Dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_state = None
    best_val = float("inf")
    best_metrics = None
    no_improve = 0
    warmup_epochs = int(max(loss_cfg.get("physics_warmup_epochs", 0), 0))

    for ep in range(int(train_cfg["epochs"])):
        physics_scale = 1.0 if warmup_epochs <= 0 else min(1.0, float(ep + 1) / float(warmup_epochs))
        model.train()
        tr_tot = tr_ar = tr_set = tr_eos = tr_cnt = 0.0
        tr_ang = tr_jpt = tr_je = tr_4v = 0.0
        tr_best = tr_div = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            tgt_tok = batch["tgt_tok"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            labels = batch["label"].to(device)

            if model.num_hypotheses > 1:
                outm = model.forward_teacher_multi(feat_hlt, mask_hlt, const_hlt, tgt_tok)
                losses = compute_reco_losses_multihyp(
                    outm,
                    tgt_tok,
                    tgt_mask,
                    labels,
                    const_hlt,
                    mask_hlt,
                    teacher,
                    feat_means_t,
                    feat_stds_t,
                    loss_cfg,
                    physics_scale=physics_scale,
                )
            else:
                out = model.forward_teacher(feat_hlt, mask_hlt, const_hlt, tgt_tok)
                losses = compute_reco_losses(out, tgt_tok, tgt_mask, loss_cfg, physics_scale=physics_scale)

            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bsz = feat_hlt.shape[0]
            n_tr += bsz
            tr_tot += float(losses["total"].detach().item()) * bsz
            tr_ar += float(losses["ar"].detach().item()) * bsz
            tr_set += float(losses["set"].detach().item()) * bsz
            tr_eos += float(losses["eos"].detach().item()) * bsz
            tr_cnt += float(losses["count"].detach().item()) * bsz
            tr_ang += float(losses["angle"].detach().item()) * bsz
            tr_jpt += float(losses["jetpt"].detach().item()) * bsz
            tr_je += float(losses["jete"].detach().item()) * bsz
            tr_4v += float(losses["fourvec"].detach().item()) * bsz
            tr_best += float(losses.get("best_set", losses["set"]).detach().item()) * bsz
            tr_div += float(losses.get("diversity", torch.zeros((), device=device)).detach().item()) * bsz

        model.eval()
        va_tot = va_ar = va_set = va_eos = va_cnt = 0.0
        va_ang = va_jpt = va_je = va_4v = 0.0
        va_best = va_div = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                tgt_tok = batch["tgt_tok"].to(device)
                tgt_mask = batch["tgt_mask"].to(device)
                labels = batch["label"].to(device)

                if model.num_hypotheses > 1:
                    outm = model.forward_teacher_multi(feat_hlt, mask_hlt, const_hlt, tgt_tok)
                    losses = compute_reco_losses_multihyp(
                        outm,
                        tgt_tok,
                        tgt_mask,
                        labels,
                        const_hlt,
                        mask_hlt,
                        teacher,
                        feat_means_t,
                        feat_stds_t,
                        loss_cfg,
                        physics_scale=physics_scale,
                    )
                else:
                    out = model.forward_teacher(feat_hlt, mask_hlt, const_hlt, tgt_tok)
                    losses = compute_reco_losses(out, tgt_tok, tgt_mask, loss_cfg, physics_scale=physics_scale)

                bsz = feat_hlt.shape[0]
                n_va += bsz
                va_tot += float(losses["total"].detach().item()) * bsz
                va_ar += float(losses["ar"].detach().item()) * bsz
                va_set += float(losses["set"].detach().item()) * bsz
                va_eos += float(losses["eos"].detach().item()) * bsz
                va_cnt += float(losses["count"].detach().item()) * bsz
                va_ang += float(losses["angle"].detach().item()) * bsz
                va_jpt += float(losses["jetpt"].detach().item()) * bsz
                va_je += float(losses["jete"].detach().item()) * bsz
                va_4v += float(losses["fourvec"].detach().item()) * bsz
                va_best += float(losses.get("best_set", losses["set"]).detach().item()) * bsz
                va_div += float(losses.get("diversity", torch.zeros((), device=device)).detach().item()) * bsz

        tr_tot /= max(n_tr, 1)
        tr_ar /= max(n_tr, 1)
        tr_set /= max(n_tr, 1)
        tr_eos /= max(n_tr, 1)
        tr_cnt /= max(n_tr, 1)
        tr_ang /= max(n_tr, 1)
        tr_jpt /= max(n_tr, 1)
        tr_je /= max(n_tr, 1)
        tr_4v /= max(n_tr, 1)
        tr_best /= max(n_tr, 1)
        tr_div /= max(n_tr, 1)

        va_tot /= max(n_va, 1)
        va_ar /= max(n_va, 1)
        va_set /= max(n_va, 1)
        va_eos /= max(n_va, 1)
        va_cnt /= max(n_va, 1)
        va_ang /= max(n_va, 1)
        va_jpt /= max(n_va, 1)
        va_je /= max(n_va, 1)
        va_4v /= max(n_va, 1)
        va_best /= max(n_va, 1)
        va_div /= max(n_va, 1)

        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"Reco ep {ep+1:03d} | "
                f"phys={physics_scale:.3f} | "
                f"train total={tr_tot:.5f} ar={tr_ar:.5f} set={tr_set:.5f} eos={tr_eos:.5f} cnt={tr_cnt:.5f} "
                f"ang={tr_ang:.5f} jpt={tr_jpt:.5f} je={tr_je:.5f} j4={tr_4v:.5f} bset={tr_best:.5f} div={tr_div:.5f} | "
                f"val total={va_tot:.5f} ar={va_ar:.5f} set={va_set:.5f} eos={va_eos:.5f} cnt={va_cnt:.5f} "
                f"ang={va_ang:.5f} jpt={va_jpt:.5f} je={va_je:.5f} j4={va_4v:.5f} bset={va_best:.5f} div={va_div:.5f}"
            )

        if va_tot < best_val:
            best_val = float(va_tot)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = RecoValMetrics(
                val_total=float(va_tot),
                val_ar=float(va_ar),
                val_set=float(va_set),
                val_eos=float(va_eos),
                val_count=float(va_cnt),
                val_angle=float(va_ang),
                val_jetpt=float(va_jpt),
                val_jete=float(va_je),
                val_fourvec=float(va_4v),
            )
            no_improve = 0
        else:
            no_improve += 1

        if ep + 1 >= int(train_cfg["min_epochs"]) and no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping reconstructor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_metrics is None:
        best_metrics = RecoValMetrics(
            val_total=float("nan"),
            val_ar=float("nan"),
            val_set=float("nan"),
            val_eos=float("nan"),
            val_count=float("nan"),
            val_angle=float("nan"),
            val_jetpt=float("nan"),
            val_jete=float("nan"),
            val_fourvec=float("nan"),
        )

    return model, {
        "val_total": best_metrics.val_total,
        "val_ar": best_metrics.val_ar,
        "val_set": best_metrics.val_set,
        "val_eos": best_metrics.val_eos,
        "val_count": best_metrics.val_count,
        "val_angle": best_metrics.val_angle,
        "val_jetpt": best_metrics.val_jetpt,
        "val_jete": best_metrics.val_jete,
        "val_fourvec": best_metrics.val_fourvec,
    }


@torch.no_grad()
def reconstruct_dataset_seq2seq(
    model: HLT2OfflineSeq2Seq,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    max_constits: int,
    device: torch.device,
    batch_size: int,
    beam_size: int = 4,
    beam_len_sigma: float = 1.5,
    beam_temperature: float = 1.0,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()
    ds = RecoInputDataset(feat_hlt, mask_hlt, const_hlt)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    n_jets = int(feat_hlt.shape[0])
    T = int(max_constits)

    reco_const = np.zeros((n_jets, T, 4), dtype=np.float32)
    reco_mask = np.zeros((n_jets, T), dtype=bool)
    reco_merge_flag = np.zeros((n_jets, T), dtype=np.float32)  # reused as confidence mask
    reco_eff_flag = np.zeros((n_jets, T), dtype=np.float32)

    created_merge_count = np.zeros((n_jets,), dtype=np.int32)
    created_eff_count = np.zeros((n_jets,), dtype=np.int32)
    pred_budget_total = np.zeros((n_jets,), dtype=np.float32)
    pred_budget_merge = np.zeros((n_jets,), dtype=np.float32)
    pred_budget_eff = np.zeros((n_jets,), dtype=np.float32)

    offset = 0
    eps = 1e-8

    for batch in dl:
        feat = batch["feat_hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        c = batch["const_hlt"].to(device)
        bsz = int(feat.shape[0])

        dec = model.decode_greedy(feat, m, c, max_steps=T)
        pred_tok = dec["pred_tok"]  # [B,T,5]
        stop_probs = dec["stop_probs"]  # [B,T]
        count_pred = dec["count_pred"]  # [B]

        pred_const = token_to_const_torch(pred_tok).detach().cpu().numpy().astype(np.float32)
        stop_np = stop_probs.detach().cpu().numpy().astype(np.float32)
        count_np = count_pred.detach().cpu().numpy().astype(np.float32)
        hlt_count_np = m.detach().cpu().numpy().sum(axis=1).astype(np.int32)

        for i in range(bsz):
            cp = float(count_np[i])
            center = int(np.rint(cp))
            cand = set()
            span = max(int(beam_size), 2)
            for d in range(-span, span + 1):
                cand.add(int(np.clip(center + d, 0, T)))
            cand = sorted(cand)

            lp = []
            s = np.clip(stop_np[i], 1e-6, 1.0 - 1e-6)
            for L in cand:
                cont = float(np.log(1.0 - s[:L] + eps).sum()) if L > 0 else 0.0
                stop_term = float(np.log(s[L] + eps)) if L < T else 0.0
                count_prior = -0.5 * ((float(L) - cp) / max(float(beam_len_sigma), 1e-3)) ** 2
                lp.append(cont + stop_term + count_prior)
            lp = np.asarray(lp, dtype=np.float64)

            k = min(int(beam_size), len(cand))
            top_idx = np.argsort(-lp)[:k]
            top_lens = np.asarray([cand[j] for j in top_idx], dtype=np.int32)
            top_lp = lp[top_idx]
            top_lp = top_lp / max(float(beam_temperature), 1e-3)
            top_w = np.exp(top_lp - np.max(top_lp))
            top_w = top_w / max(top_w.sum(), 1e-12)

            steps = np.arange(T, dtype=np.int32)
            active_prob = np.zeros((T,), dtype=np.float32)
            for ww, LL in zip(top_w, top_lens):
                active_prob += float(ww) * (steps < int(LL)).astype(np.float32)

            soft_len = float(np.sum(top_w * top_lens.astype(np.float64)))
            final_len = int(np.clip(np.rint(soft_len), 0, T))

            pred_i = pred_const[i]
            if final_len > 0:
                # Keep AR order but enforce pT-descending for classifier consistency.
                pt = pred_i[:final_len, 0]
                ord_i = np.argsort(-pt)
                pred_i[:final_len] = pred_i[:final_len][ord_i]
                active_prob[:final_len] = active_prob[:final_len][ord_i]

            idx0 = offset + i
            reco_const[idx0] = pred_i
            reco_mask[idx0, :final_len] = True
            reco_merge_flag[idx0] = active_prob

            reco_n = final_len
            hlt_n = int(hlt_count_np[i])
            created = max(reco_n - hlt_n, 0)
            created_merge_count[idx0] = int(created)
            created_eff_count[idx0] = 0
            pred_budget_total[idx0] = float(created)
            pred_budget_merge[idx0] = float(created)
            pred_budget_eff[idx0] = 0.0

        offset += bsz

    reco_const = np.nan_to_num(reco_const, nan=0.0, posinf=0.0, neginf=0.0)
    reco_const[~reco_mask] = 0.0
    return (
        reco_const,
        reco_mask,
        reco_merge_flag,
        reco_eff_flag,
        created_merge_count,
        created_eff_count,
        pred_budget_total,
        pred_budget_merge,
        pred_budget_eff,
    )


@torch.no_grad()
def reconstruct_dataset_seq2seq_multihyp(
    model: HLT2OfflineSeq2Seq,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    max_constits: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ds = RecoInputDataset(feat_hlt, mask_hlt, const_hlt)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    n_jets = int(feat_hlt.shape[0])
    t = int(max_constits)
    k = int(model.num_hypotheses)

    out_const = np.zeros((n_jets, k, t, 4), dtype=np.float32)
    out_mask = np.zeros((n_jets, k, t), dtype=bool)
    out_conf = np.zeros((n_jets, k, t), dtype=np.float32)

    offset = 0
    for batch in dl:
        feat = batch["feat_hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        c = batch["const_hlt"].to(device)
        bsz = int(feat.shape[0])

        decm = model.decode_greedy_multi(feat, m, c, max_steps=t)
        pred_tok = decm["pred_tok"]      # [B,K,T,5]
        stop_probs = decm["stop_probs"]  # [B,K,T]
        count_pred = decm["count_pred"]  # [B]

        pred_const = token_to_const_torch(pred_tok).detach().cpu().numpy().astype(np.float32)
        stop_np = stop_probs.detach().cpu().numpy().astype(np.float32)
        count_np = count_pred.detach().cpu().numpy().astype(np.float32)

        for i in range(bsz):
            cp = int(np.clip(np.rint(count_np[i]), 0, t))
            for hk in range(k):
                s = stop_np[i, hk]
                stop_pos = np.where(s > 0.5)[0]
                if stop_pos.size > 0:
                    L = int(np.clip(stop_pos[0], 0, t))
                else:
                    L = cp
                L = int(np.clip(L, 0, t))
                arr = pred_const[i, hk]
                if L > 0:
                    ord_i = np.argsort(-arr[:L, 0])
                    arr[:L] = arr[:L][ord_i]
                idx0 = offset + i
                out_const[idx0, hk] = arr
                out_mask[idx0, hk, :L] = True
                out_conf[idx0, hk, :L] = 1.0

        offset += bsz

    out_const = np.nan_to_num(out_const, nan=0.0, posinf=0.0, neginf=0.0)
    out_const[~out_mask] = 0.0
    return out_const, out_mask, out_conf


@torch.no_grad()
def build_selector_features_and_targets(
    hyp_const_bkt4: np.ndarray,
    hyp_mask_bkt: np.ndarray,
    hlt_const_bt4: np.ndarray,
    hlt_mask_bt: np.ndarray,
    off_const_bt4: np.ndarray,
    off_mask_bt: np.ndarray,
    labels_b: np.ndarray,
    teacher: nn.Module,
    feat_means: np.ndarray,
    feat_stds: np.ndarray,
    device: torch.device,
    winner_mode: str,
    winner_alpha: float,
    winner_beta: float,
    set_loss_mode: str,
    set_unmatched_penalty: float,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    n, k, t, _ = hyp_const_bkt4.shape
    feat_out = np.zeros((n, k, 7), dtype=np.float32)
    winner_out = np.zeros((n,), dtype=np.int64)

    means_t = torch.tensor(feat_means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(feat_stds, dtype=torch.float32, device=device)

    teacher.eval()
    for st in range(0, n, int(batch_size)):
        ed = min(st + int(batch_size), n)
        b = ed - st

        hlt_const = torch.tensor(hlt_const_bt4[st:ed], dtype=torch.float32, device=device)
        hlt_mask = torch.tensor(hlt_mask_bt[st:ed], dtype=torch.bool, device=device)
        off_const = torch.tensor(off_const_bt4[st:ed], dtype=torch.float32, device=device)
        off_mask = torch.tensor(off_mask_bt[st:ed], dtype=torch.bool, device=device)
        lab = torch.tensor(labels_b[st:ed], dtype=torch.float32, device=device)

        hlt_n = hlt_mask.float().sum(dim=1)
        hlt_pt = compute_jet_pt_torch(hlt_const, hlt_mask)
        tag_cols = []
        set_cols = []
        feat_cols = []
        for hk in range(k):
            p_const = torch.tensor(hyp_const_bkt4[st:ed, hk], dtype=torch.float32, device=device)
            p_mask = torch.tensor(hyp_mask_bkt[st:ed, hk], dtype=torch.bool, device=device)
            p_tok = const_to_token_torch(p_const)
            t_tok = const_to_token_torch(off_const)

            if str(set_loss_mode).lower() == "hungarian":
                s_vec = hungarian_token_loss_vec(
                    p_tok,
                    t_tok,
                    p_mask,
                    off_mask,
                    unmatched_penalty=float(set_unmatched_penalty),
                )
            else:
                s_vec = chamfer_token_loss_vec(p_tok, t_tok, p_mask, off_mask)
            set_cols.append(s_vec)

            p_feat = compute_features_torch(p_const, p_mask)
            p_feat_std = standardize_features_torch(p_feat, p_mask, means_t, stds_t)
            p_logit = teacher(p_feat_std, p_mask).squeeze(-1)
            p_prob = torch.sigmoid(p_logit)
            p_entropy = -(p_prob * torch.log(p_prob.clamp(min=1e-8)) + (1.0 - p_prob) * torch.log((1.0 - p_prob).clamp(min=1e-8)))
            tag_loss = F.binary_cross_entropy_with_logits(p_logit, lab, reduction="none")
            tag_cols.append(tag_loss)

            p_n = p_mask.float().sum(dim=1)
            p_pt = compute_jet_pt_torch(p_const, p_mask)
            pt_ratio_hlt = p_pt / (hlt_pt + 1e-8)
            f_k = torch.stack(
                [
                    p_logit,
                    p_prob,
                    p_entropy,
                    p_n / 100.0,
                    (p_n - hlt_n).abs() / 100.0,
                    pt_ratio_hlt,
                    (pt_ratio_hlt - 1.0).abs(),
                ],
                dim=-1,
            )
            feat_cols.append(f_k)

        set_mat = torch.stack(set_cols, dim=1)
        tag_mat = torch.stack(tag_cols, dim=1)
        feat_mat = torch.stack(feat_cols, dim=1)

        wm = str(winner_mode).strip().lower()
        if wm == "reco":
            score = set_mat
        elif wm == "hybrid":
            score = float(winner_alpha) * tag_mat + float(winner_beta) * set_mat
        else:
            score = tag_mat
        win = torch.argmin(score, dim=1)

        feat_out[st:ed] = feat_mat.detach().cpu().numpy().astype(np.float32)
        winner_out[st:ed] = win.detach().cpu().numpy().astype(np.int64)

    return feat_out, winner_out


def train_selector_model(
    selector: HypothesisSelector,
    feat_tr: np.ndarray,
    y_tr: np.ndarray,
    feat_va: np.ndarray,
    y_va: np.ndarray,
    device: torch.device,
    epochs: int = 45,
    lr: float = 2e-3,
    batch_size: int = 512,
    patience: int = 8,
    rank_weight: float = 0.2,
    rank_margin: float = 0.25,
) -> HypothesisSelector:
    xtr = torch.tensor(feat_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    xva = torch.tensor(feat_va, dtype=torch.float32)
    yva = torch.tensor(y_va, dtype=torch.long)

    tr_dl = DataLoader(torch.utils.data.TensorDataset(xtr, ytr), batch_size=int(batch_size), shuffle=True)
    va_dl = DataLoader(torch.utils.data.TensorDataset(xva, yva), batch_size=int(batch_size), shuffle=False)

    selector = selector.to(device)
    opt = torch.optim.AdamW(selector.parameters(), lr=float(lr), weight_decay=1e-4)
    best_state = None
    best_acc = -1.0
    no_improve = 0
    for ep in range(int(epochs)):
        selector.train()
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = selector(xb)
            loss_ce = F.cross_entropy(logits, yb)
            loss_rank = selector_rank_loss(logits, yb, margin=float(rank_margin))
            loss = loss_ce + float(rank_weight) * loss_rank
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), max_norm=1.0)
            opt.step()

        selector.eval()
        n_ok = 0
        n_tot = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = torch.argmax(selector(xb), dim=1)
                n_ok += int((pred == yb).sum().item())
                n_tot += int(yb.numel())
        va_acc = float(n_ok / max(n_tot, 1))
        if (ep + 1) % 5 == 0:
            print(f"Selector ep {ep+1}: val_acc={va_acc:.4f}, best={best_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(patience):
            print(f"Early stopping selector at epoch {ep+1}")
            break

    if best_state is not None:
        selector.load_state_dict(best_state)
    return selector


@torch.no_grad()
def selector_predict_indices(selector: HypothesisSelector, feat_bkf: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    x = torch.tensor(feat_bkf, dtype=torch.float32)
    dl = DataLoader(x, batch_size=int(batch_size), shuffle=False)
    out = []
    selector.eval()
    for xb in dl:
        xb = xb.to(device)
        pred = torch.argmax(selector(xb), dim=1)
        out.append(pred.detach().cpu().numpy().astype(np.int64))
    return np.concatenate(out, axis=0)


def gather_selected_view(
    hyp_const: np.ndarray,
    hyp_mask: np.ndarray,
    hyp_conf: np.ndarray,
    winner_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, _k, t, _d = hyp_const.shape
    const = np.zeros((n, t, 4), dtype=np.float32)
    mask = np.zeros((n, t), dtype=bool)
    conf = np.zeros((n, t), dtype=np.float32)
    for i in range(n):
        k = int(winner_idx[i])
        const[i] = hyp_const[i, k]
        mask[i] = hyp_mask[i, k]
        conf[i] = hyp_conf[i, k]
    return const, mask, conf


# ----------------------------- Main ---------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=250000)
    parser.add_argument("--n_train_split", type=int, default=100000)
    parser.add_argument("--n_val_split", type=int, default=50000)
    parser.add_argument("--n_test_split", type=int, default=100000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_joint_seq2seq"))
    parser.add_argument("--run_name", type=str, default="seq2seq_default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--roc_fpr_min", type=float, default=1e-4)

    # Reconstructor settings
    parser.add_argument("--reco_batch_size", type=int, default=RECO_TRAIN_CFG["batch_size"])
    parser.add_argument("--reco_epochs", type=int, default=RECO_TRAIN_CFG["epochs"])
    parser.add_argument("--reco_lr", type=float, default=RECO_TRAIN_CFG["lr"])
    parser.add_argument("--reco_patience", type=int, default=RECO_TRAIN_CFG["patience"])
    parser.add_argument("--reco_min_epochs", type=int, default=RECO_TRAIN_CFG["min_epochs"])
    parser.add_argument("--reco_huber_delta", type=float, default=LOSS_CFG["huber_delta"])
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--beam_len_sigma", type=float, default=1.5)
    parser.add_argument("--beam_temperature", type=float, default=1.0)
    parser.add_argument("--use_coord_residual_param", action="store_true")
    parser.add_argument("--num_hypotheses", type=int, default=MULTIHYP_CFG["num_hypotheses"])
    parser.add_argument("--joint_epochs", type=int, default=MULTIHYP_CFG["joint_epochs"])
    parser.add_argument("--joint_lr", type=float, default=MULTIHYP_CFG["joint_lr"])
    parser.add_argument("--joint_patience", type=int, default=MULTIHYP_CFG["joint_patience"])

    # Loss weights
    parser.add_argument("--loss_w_ar", type=float, default=LOSS_CFG["w_ar"])
    parser.add_argument("--loss_w_set", type=float, default=LOSS_CFG["w_set"])
    parser.add_argument("--loss_w_eos", type=float, default=LOSS_CFG["w_eos"])
    parser.add_argument("--loss_w_count", type=float, default=LOSS_CFG["w_count"])
    parser.add_argument("--loss_w_ptr_entropy", type=float, default=LOSS_CFG["w_ptr_entropy"])
    parser.add_argument("--set_loss_mode", type=str, default=LOSS_CFG["set_loss_mode"], choices=["chamfer", "hungarian"])
    parser.add_argument("--set_unmatched_penalty", type=float, default=LOSS_CFG["set_unmatched_penalty"])
    parser.add_argument("--loss_w_angle", type=float, default=LOSS_CFG["w_angle"])
    parser.add_argument("--loss_w_jetpt", type=float, default=LOSS_CFG["w_jetpt"])
    parser.add_argument("--loss_w_jete", type=float, default=LOSS_CFG["w_jete"])
    parser.add_argument("--loss_w_4vec", type=float, default=LOSS_CFG["w_4vec"])
    parser.add_argument("--loss_w_best_set", type=float, default=LOSS_CFG["w_best_set"])
    parser.add_argument("--loss_w_diversity", type=float, default=LOSS_CFG["w_diversity"])
    parser.add_argument("--winner_mode", type=str, default=LOSS_CFG["winner_mode"], choices=["tag", "reco", "hybrid"])
    parser.add_argument("--winner_hybrid_alpha", type=float, default=LOSS_CFG["winner_hybrid_alpha"])
    parser.add_argument("--winner_hybrid_beta", type=float, default=LOSS_CFG["winner_hybrid_beta"])
    parser.add_argument("--enable_scale_sensitive_weighting", action="store_true")
    parser.add_argument("--scale_pt_power", type=float, default=LOSS_CFG["scale_pt_power"])
    parser.add_argument("--scale_weight_cap", type=float, default=LOSS_CFG["scale_weight_cap"])
    parser.add_argument("--angle_pt_power", type=float, default=LOSS_CFG["angle_pt_power"])
    parser.add_argument("--physics_warmup_epochs", type=int, default=LOSS_CFG["physics_warmup_epochs"])

    parser.add_argument("--skip_save_models", action="store_true")
    parser.add_argument("--save_fusion_scores", action="store_true")
    parser.add_argument("--response_n_bins", type=int, default=18)
    parser.add_argument("--response_min_count", type=int, default=300)
    parser.add_argument("--selector_epochs", type=int, default=45)
    parser.add_argument("--selector_lr", type=float, default=2e-3)
    parser.add_argument("--selector_patience", type=int, default=8)
    parser.add_argument("--selector_rank_weight", type=float, default=LOSS_CFG["selector_rank"])
    parser.add_argument("--selector_rank_margin", type=float, default=LOSS_CFG["selector_rank_margin"])

    # Pseudo-HLT knobs (kept from prior plumbing)
    parser.add_argument("--merge_radius", type=float, default=BASE_CONFIG["hlt_effects"]["merge_radius"])
    parser.add_argument("--pt_threshold_hlt", type=float, default=BASE_CONFIG["hlt_effects"]["pt_threshold_hlt"])
    parser.add_argument("--pt_threshold_offline", type=float, default=BASE_CONFIG["hlt_effects"]["pt_threshold_offline"])
    parser.add_argument("--eff_plateau_barrel", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    parser.add_argument("--eff_plateau_endcap", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    parser.add_argument("--smear_a", type=float, default=BASE_CONFIG["hlt_effects"]["smear_a"])
    parser.add_argument("--smear_b", type=float, default=BASE_CONFIG["hlt_effects"]["smear_b"])
    parser.add_argument("--smear_c", type=float, default=BASE_CONFIG["hlt_effects"]["smear_c"])

    args = parser.parse_args()

    set_seed(int(args.seed))

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    # Build local configs.
    hlt_cfg = _deepcopy_cfg()
    hlt_cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    hlt_cfg["hlt_effects"]["pt_threshold_hlt"] = float(args.pt_threshold_hlt)
    hlt_cfg["hlt_effects"]["pt_threshold_offline"] = float(args.pt_threshold_offline)
    hlt_cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    hlt_cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    hlt_cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    hlt_cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    hlt_cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    reco_train_cfg = dict(RECO_TRAIN_CFG)
    reco_train_cfg["batch_size"] = int(args.reco_batch_size)
    reco_train_cfg["epochs"] = int(args.reco_epochs)
    reco_train_cfg["lr"] = float(args.reco_lr)
    reco_train_cfg["patience"] = int(args.reco_patience)
    reco_train_cfg["min_epochs"] = int(args.reco_min_epochs)

    loss_cfg = {
        "w_ar": float(args.loss_w_ar),
        "w_set": float(args.loss_w_set),
        "w_eos": float(args.loss_w_eos),
        "w_count": float(args.loss_w_count),
        "w_ptr_entropy": float(args.loss_w_ptr_entropy),
        "set_loss_mode": str(args.set_loss_mode).strip().lower(),
        "set_unmatched_penalty": float(args.set_unmatched_penalty),
        "w_angle": float(args.loss_w_angle),
        "w_jetpt": float(args.loss_w_jetpt),
        "w_jete": float(args.loss_w_jete),
        "w_4vec": float(args.loss_w_4vec),
        "w_best_set": float(args.loss_w_best_set),
        "w_diversity": float(args.loss_w_diversity),
        "winner_mode": str(args.winner_mode).strip().lower(),
        "winner_hybrid_alpha": float(args.winner_hybrid_alpha),
        "winner_hybrid_beta": float(args.winner_hybrid_beta),
        "selector_ce": 0.6,
        "selector_rank": float(args.selector_rank_weight),
        "selector_rank_margin": float(args.selector_rank_margin),
        "huber_delta": float(args.reco_huber_delta),
        "scale_sensitive_weighting": bool(args.enable_scale_sensitive_weighting),
        "scale_pt_power": float(args.scale_pt_power),
        "scale_weight_cap": float(args.scale_weight_cap),
        "angle_pt_power": float(args.angle_pt_power),
        "physics_warmup_epochs": int(args.physics_warmup_epochs),
    }

    # ----------------------------- Load data -------------------------------- #
    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = int(args.offset_jets + args.n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=int(args.max_constits),
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets for offset={args.offset_jets} + n_train_jets={args.n_train_jets}. "
            f"Loaded={all_const_full.shape[0]}"
        )

    const_raw = all_const_full[int(args.offset_jets): int(args.offset_jets + args.n_train_jets)]
    labels = all_labels_full[int(args.offset_jets): int(args.offset_jets + args.n_train_jets)].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(hlt_cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy().astype(np.float32)
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        hlt_cfg,
        seed=int(args.seed),
    )
    budget_merge_true = budget_truth["merge_lost_per_jet"].astype(np.float32)
    budget_eff_true = budget_truth["eff_lost_per_jet"].astype(np.float32)

    print("Computing features...")
    features_off = compute_features(const_off, masks_off)
    features_hlt = compute_features(hlt_const, hlt_mask)

    # Sort HLT and offline constituents by pT for sequence modeling.
    hlt_const_sort, hlt_mask_sort, hlt_order = sort_constituents_by_pt_np(hlt_const, hlt_mask)
    const_off_sort, masks_off_sort, _off_order = sort_constituents_by_pt_np(const_off, masks_off)
    features_hlt_sort = reorder_features_np(features_hlt, hlt_order, hlt_mask_sort)

    train_idx, val_idx, test_idx = build_fixed_split_indices(
        labels=labels,
        n_train=int(args.n_train_split),
        n_val=int(args.n_val_split),
        n_test=int(args.n_test_split),
        seed=int(args.seed),
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = standardize(features_hlt_sort, hlt_mask_sort, feat_means, feat_stds)

    # Save split/stats for reproducibility and downstream tooling.
    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        feat_means=feat_means.astype(np.float32),
        feat_stds=feat_stds.astype(np.float32),
    )

    with open(save_root / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # ----------------------------- Teacher ---------------------------------- #
    print("\n" + "=" * 72)
    print("STEP 1: TEACHER (Offline)")
    print("=" * 72)

    bs_cls = int(CLS_TRAIN_CFG["batch_size"])

    ds_tr_off = JetDataset(features_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    ds_va_off = JetDataset(features_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    ds_te_off = JetDataset(features_off_std[test_idx], masks_off[test_idx], labels[test_idx])

    dl_tr_off = DataLoader(ds_tr_off, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_off = DataLoader(ds_va_off, batch_size=bs_cls, shuffle=False)
    dl_te_off = DataLoader(ds_te_off, batch_size=bs_cls, shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **MODEL_CFG).to(device)
    teacher = train_single_view_classifier_by_auc(teacher, dl_tr_off, dl_va_off, device, CLS_TRAIN_CFG, name="Teacher")
    auc_teacher, preds_teacher, labs_test = eval_classifier(teacher, dl_te_off, device)

    # ----------------------------- Baseline HLT ----------------------------- #
    print("\n" + "=" * 72)
    print("STEP 2: BASELINE (HLT)")
    print("=" * 72)

    ds_tr_hlt = JetDataset(features_hlt_std[train_idx], hlt_mask_sort[train_idx], labels[train_idx])
    ds_va_hlt = JetDataset(features_hlt_std[val_idx], hlt_mask_sort[val_idx], labels[val_idx])
    ds_te_hlt = JetDataset(features_hlt_std[test_idx], hlt_mask_sort[test_idx], labels[test_idx])

    dl_tr_hlt = DataLoader(ds_tr_hlt, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_hlt = DataLoader(ds_va_hlt, batch_size=bs_cls, shuffle=False)
    dl_te_hlt = DataLoader(ds_te_hlt, batch_size=bs_cls, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **MODEL_CFG).to(device)
    baseline = train_single_view_classifier_by_auc(baseline, dl_tr_hlt, dl_va_hlt, device, CLS_TRAIN_CFG, name="HLT")
    auc_hlt, preds_hlt, _ = eval_classifier(baseline, dl_te_hlt, device)

    # ----------------------------- Reconstructor ---------------------------- #
    print("\n" + "=" * 72)
    print("STEP 3: SEQ2SEQ RECONSTRUCTOR (Continuous AR + Beam Length)")
    print("=" * 72)

    tgt_tok_all = const_to_token_np(const_off_sort)
    tgt_mask_all = masks_off_sort.astype(bool)

    ds_tr_reco = RecoSeqDataset(
        feat_hlt=features_hlt_std[train_idx],
        mask_hlt=hlt_mask_sort[train_idx],
        const_hlt=hlt_const_sort[train_idx],
        tgt_tok=tgt_tok_all[train_idx],
        tgt_mask=tgt_mask_all[train_idx],
        labels=labels[train_idx],
    )
    ds_va_reco = RecoSeqDataset(
        feat_hlt=features_hlt_std[val_idx],
        mask_hlt=hlt_mask_sort[val_idx],
        const_hlt=hlt_const_sort[val_idx],
        tgt_tok=tgt_tok_all[val_idx],
        tgt_mask=tgt_mask_all[val_idx],
        labels=labels[val_idx],
    )

    dl_tr_reco = DataLoader(
        ds_tr_reco,
        batch_size=int(reco_train_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_va_reco = DataLoader(
        ds_va_reco,
        batch_size=int(reco_train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    reco = HLT2OfflineSeq2Seq(
        input_dim_hlt=7,
        token_dim=5,
        embed_dim=int(RECO_CFG["embed_dim"]),
        num_heads=int(RECO_CFG["num_heads"]),
        num_enc_layers=int(RECO_CFG["num_enc_layers"]),
        num_dec_layers=int(RECO_CFG["num_dec_layers"]),
        ff_dim=int(RECO_CFG["ff_dim"]),
        dropout=float(RECO_CFG["dropout"]),
        max_hlt_tokens=int(args.max_constits),
        max_decode_tokens=int(args.max_constits),
        use_coord_residual_param=bool(args.use_coord_residual_param),
        num_hypotheses=int(max(args.num_hypotheses, 1)),
    ).to(device)

    teacher.eval()
    feat_means_t = torch.tensor(feat_means, dtype=torch.float32, device=device)
    feat_stds_t = torch.tensor(feat_stds, dtype=torch.float32, device=device)
    reco, reco_val_metrics = train_reconstructor_seq2seq(
        reco,
        dl_tr_reco,
        dl_va_reco,
        device,
        reco_train_cfg,
        loss_cfg,
        teacher=teacher if reco.num_hypotheses > 1 else None,
        feat_means_t=feat_means_t if reco.num_hypotheses > 1 else None,
        feat_stds_t=feat_stds_t if reco.num_hypotheses > 1 else None,
    )

    print("Best reconstructor val metrics:")
    for k, v in reco_val_metrics.items():
        print(f"  {k}: {v:.6f}")

    if reco.num_hypotheses > 1 and int(args.joint_epochs) > 0:
        print("\n" + "=" * 72)
        print("STEP 3B: JOINT-LIKE REFINEMENT (Tag-Aware Winner Guidance)")
        print("=" * 72)
        joint_cfg = dict(reco_train_cfg)
        joint_cfg["epochs"] = int(args.joint_epochs)
        joint_cfg["lr"] = float(args.joint_lr)
        joint_cfg["patience"] = int(args.joint_patience)
        joint_cfg["min_epochs"] = max(4, int(args.joint_epochs // 2))
        reco, reco_val_metrics_joint = train_reconstructor_seq2seq(
            reco,
            dl_tr_reco,
            dl_va_reco,
            device,
            joint_cfg,
            loss_cfg,
            teacher=teacher,
            feat_means_t=feat_means_t,
            feat_stds_t=feat_stds_t,
        )
        print("Best joint-like reconstructor val metrics:")
        for k, v in reco_val_metrics_joint.items():
            print(f"  {k}: {v:.6f}")
        for k, v in reco_val_metrics_joint.items():
            reco_val_metrics[f"joint_{k}"] = float(v)

    print("Building reconstructed dataset...")
    selector = None
    selector_feat_all = None
    selector_target_all = None
    winner_pred_all = None
    hyp_const_all = None
    hyp_mask_all = None
    hyp_conf_all = None
    if reco.num_hypotheses > 1:
        print("\n" + "=" * 72)
        print("STEP 4: MULTI-HYP SELECTOR (K-Way)")
        print("=" * 72)
        hyp_const_all, hyp_mask_all, hyp_conf_all = reconstruct_dataset_seq2seq_multihyp(
            model=reco,
            feat_hlt=features_hlt_std,
            mask_hlt=hlt_mask_sort,
            const_hlt=hlt_const_sort,
            max_constits=int(args.max_constits),
            device=device,
            batch_size=int(reco_train_cfg["batch_size"]),
        )
        selector_feat_all, selector_target_all = build_selector_features_and_targets(
            hyp_const_bkt4=hyp_const_all,
            hyp_mask_bkt=hyp_mask_all,
            hlt_const_bt4=hlt_const_sort,
            hlt_mask_bt=hlt_mask_sort,
            off_const_bt4=const_off_sort,
            off_mask_bt=masks_off_sort,
            labels_b=labels.astype(np.float32),
            teacher=teacher,
            feat_means=feat_means,
            feat_stds=feat_stds,
            device=device,
            winner_mode=str(args.winner_mode),
            winner_alpha=float(args.winner_hybrid_alpha),
            winner_beta=float(args.winner_hybrid_beta),
            set_loss_mode=str(args.set_loss_mode),
            set_unmatched_penalty=float(args.set_unmatched_penalty),
            batch_size=max(128, int(reco_train_cfg["batch_size"])),
        )
        selector = HypothesisSelector(feat_dim=int(selector_feat_all.shape[-1]), hidden=64)
        selector = train_selector_model(
            selector,
            selector_feat_all[train_idx],
            selector_target_all[train_idx],
            selector_feat_all[val_idx],
            selector_target_all[val_idx],
            device=device,
            epochs=int(args.selector_epochs),
            lr=float(args.selector_lr),
            batch_size=512,
            patience=int(args.selector_patience),
            rank_weight=float(args.selector_rank_weight),
            rank_margin=float(args.selector_rank_margin),
        )
        winner_pred_all = selector_predict_indices(selector, selector_feat_all, device=device, batch_size=1024)
        reco_const, reco_mask, reco_merge_flag = gather_selected_view(hyp_const_all, hyp_mask_all, hyp_conf_all, winner_pred_all)
        reco_eff_flag = np.zeros_like(reco_merge_flag, dtype=np.float32)
        hlt_count_all = hlt_mask_sort.sum(axis=1).astype(np.int32)
        reco_count_all = reco_mask.sum(axis=1).astype(np.int32)
        created_merge_count = np.maximum(reco_count_all - hlt_count_all, 0).astype(np.int32)
        created_eff_count = np.zeros_like(created_merge_count, dtype=np.int32)
        pred_budget_total = created_merge_count.astype(np.float32)
        pred_budget_merge = created_merge_count.astype(np.float32)
        pred_budget_eff = np.zeros_like(pred_budget_total, dtype=np.float32)
    else:
        (
            reco_const,
            reco_mask,
            reco_merge_flag,
            reco_eff_flag,
            created_merge_count,
            created_eff_count,
            pred_budget_total,
            pred_budget_merge,
            pred_budget_eff,
        ) = reconstruct_dataset_seq2seq(
            model=reco,
            feat_hlt=features_hlt_std,
            mask_hlt=hlt_mask_sort,
            const_hlt=hlt_const_sort,
            max_constits=int(args.max_constits),
            device=device,
            batch_size=int(reco_train_cfg["batch_size"]),
            beam_size=int(args.beam_size),
            beam_len_sigma=float(args.beam_len_sigma),
            beam_temperature=float(args.beam_temperature),
        )

    features_reco = compute_features(reco_const, reco_mask)
    features_reco_std = standardize(features_reco, reco_mask, feat_means, feat_stds)
    # Keep soft-view confidence channel + placeholder efficiency flag channel.
    features_reco_flag = np.concatenate(
        [features_reco_std, reco_merge_flag[..., None], reco_eff_flag[..., None]],
        axis=-1,
    ).astype(np.float32)

    # ----------------------------- Taggers on reconstructed view ------------ #
    print("\n" + "=" * 72)
    print("STEP 5: TAGGERS ON RECONSTRUCTED VIEW")
    print("=" * 72)

    ds_tr_reco_cls = JetDataset(features_reco_std[train_idx], reco_mask[train_idx], labels[train_idx])
    ds_va_reco_cls = JetDataset(features_reco_std[val_idx], reco_mask[val_idx], labels[val_idx])
    ds_te_reco_cls = JetDataset(features_reco_std[test_idx], reco_mask[test_idx], labels[test_idx])

    dl_tr_reco_cls = DataLoader(ds_tr_reco_cls, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_reco_cls = DataLoader(ds_va_reco_cls, batch_size=bs_cls, shuffle=False)
    dl_te_reco_cls = DataLoader(ds_te_reco_cls, batch_size=bs_cls, shuffle=False)

    unmerge = ParticleTransformer(input_dim=7, **MODEL_CFG).to(device)
    unmerge = train_single_view_classifier_by_auc(unmerge, dl_tr_reco_cls, dl_va_reco_cls, device, CLS_TRAIN_CFG, name="RecoOnly")
    auc_unmerge, preds_unmerge, _ = eval_classifier(unmerge, dl_te_reco_cls, device)

    # Dual-view (HLT + reconstructed)
    ds_tr_dual = DualViewJetDataset(
        features_hlt_std[train_idx],
        hlt_mask_sort[train_idx],
        features_reco_std[train_idx],
        reco_mask[train_idx],
        labels[train_idx],
    )
    ds_va_dual = DualViewJetDataset(
        features_hlt_std[val_idx],
        hlt_mask_sort[val_idx],
        features_reco_std[val_idx],
        reco_mask[val_idx],
        labels[val_idx],
    )
    ds_te_dual = DualViewJetDataset(
        features_hlt_std[test_idx],
        hlt_mask_sort[test_idx],
        features_reco_std[test_idx],
        reco_mask[test_idx],
        labels[test_idx],
    )

    dl_tr_dual = DataLoader(ds_tr_dual, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_dual = DataLoader(ds_va_dual, batch_size=bs_cls, shuffle=False)
    dl_te_dual = DataLoader(ds_te_dual, batch_size=bs_cls, shuffle=False)

    dual = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **MODEL_CFG).to(device)
    dual = train_dual_view_classifier_by_auc(dual, dl_tr_dual, dl_va_dual, device, CLS_TRAIN_CFG, name="DualView")
    auc_dual, preds_dual, _ = eval_classifier_dual(dual, dl_te_dual, device)

    # Dual-view with confidence channels.
    ds_tr_dual_f = DualViewJetDataset(
        features_hlt_std[train_idx],
        hlt_mask_sort[train_idx],
        features_reco_flag[train_idx],
        reco_mask[train_idx],
        labels[train_idx],
    )
    ds_va_dual_f = DualViewJetDataset(
        features_hlt_std[val_idx],
        hlt_mask_sort[val_idx],
        features_reco_flag[val_idx],
        reco_mask[val_idx],
        labels[val_idx],
    )
    ds_te_dual_f = DualViewJetDataset(
        features_hlt_std[test_idx],
        hlt_mask_sort[test_idx],
        features_reco_flag[test_idx],
        reco_mask[test_idx],
        labels[test_idx],
    )

    dl_tr_dual_f = DataLoader(ds_tr_dual_f, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_dual_f = DataLoader(ds_va_dual_f, batch_size=bs_cls, shuffle=False)
    dl_te_dual_f = DataLoader(ds_te_dual_f, batch_size=bs_cls, shuffle=False)

    dual_flag = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=9, **MODEL_CFG).to(device)
    dual_flag = train_dual_view_classifier_by_auc(dual_flag, dl_tr_dual_f, dl_va_dual_f, device, CLS_TRAIN_CFG, name="DualView+Conf")
    auc_dual_flag, preds_dual_flag, _ = eval_classifier_dual(dual_flag, dl_te_dual_f, device)

    selector_val_acc = float("nan")
    selector_test_acc = float("nan")
    if selector is not None and selector_target_all is not None and winner_pred_all is not None:
        selector_val_acc = float((winner_pred_all[val_idx] == selector_target_all[val_idx]).mean())
        selector_test_acc = float((winner_pred_all[test_idx] == selector_target_all[test_idx]).mean())

    # ----------------------------- Final metrics ---------------------------- #
    print("\n" + "=" * 72)
    print("FINAL TEST EVALUATION")
    print("=" * 72)
    print(f"Teacher (Offline) AUC: {auc_teacher:.6f}")
    print(f"Baseline (HLT)   AUC: {auc_hlt:.6f}")
    print(f"RecoOnly         AUC: {auc_unmerge:.6f}")
    print(f"DualView         AUC: {auc_dual:.6f}")
    print(f"DualView+Conf    AUC: {auc_dual_flag:.6f}")
    if selector is not None:
        print(f"Selector acc (val/test): {selector_val_acc:.4f} / {selector_test_acc:.4f}")

    fpr_t, tpr_t, _ = roc_curve(labs_test, preds_teacher)
    fpr_h, tpr_h, _ = roc_curve(labs_test, preds_hlt)
    fpr_r, tpr_r, _ = roc_curve(labs_test, preds_unmerge)
    fpr_d, tpr_d, _ = roc_curve(labs_test, preds_dual)
    fpr_df, tpr_df, _ = roc_curve(labs_test, preds_dual_flag)

    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.30)
    fpr30_hlt = fpr_at_target_tpr(fpr_h, tpr_h, 0.30)
    fpr30_reco = fpr_at_target_tpr(fpr_r, tpr_r, 0.30)
    fpr30_dual = fpr_at_target_tpr(fpr_d, tpr_d, 0.30)
    fpr30_dual_flag = fpr_at_target_tpr(fpr_df, tpr_df, 0.30)

    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_hlt = fpr_at_target_tpr(fpr_h, tpr_h, 0.50)
    fpr50_reco = fpr_at_target_tpr(fpr_r, tpr_r, 0.50)
    fpr50_dual = fpr_at_target_tpr(fpr_d, tpr_d, 0.50)
    fpr50_dual_flag = fpr_at_target_tpr(fpr_df, tpr_df, 0.50)

    print("\nFPR@30")
    print(
        f"  Teacher/HLT/RecoOnly/Dual/Dual+Conf: "
        f"{fpr30_teacher:.6f} / {fpr30_hlt:.6f} / {fpr30_reco:.6f} / {fpr30_dual:.6f} / {fpr30_dual_flag:.6f}"
    )
    print("FPR@50")
    print(
        f"  Teacher/HLT/RecoOnly/Dual/Dual+Conf: "
        f"{fpr50_teacher:.6f} / {fpr50_hlt:.6f} / {fpr50_reco:.6f} / {fpr50_dual:.6f} / {fpr50_dual_flag:.6f}"
    )

    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_h, fpr_h, "--", f"HLT (AUC={auc_hlt:.3f})", "steelblue"),
            (tpr_r, fpr_r, ":", f"RecoOnly (AUC={auc_unmerge:.3f})", "forestgreen"),
            (tpr_d, fpr_d, "-.", f"DualView (AUC={auc_dual:.3f})", "darkorange"),
            (tpr_df, fpr_df, "-", f"DualView+Conf (AUC={auc_dual_flag:.3f})", "slateblue"),
        ],
        save_root / "results_all.png",
        args.roc_fpr_min,
    )

    # Count/budget diagnostics (same files as prior pipeline).
    count_summary = plot_constituent_count_diagnostics(
        save_root=save_root,
        mask_off=masks_off_sort,
        hlt_mask=hlt_mask_sort,
        reco_mask=reco_mask,
        created_merge_count=created_merge_count,
        created_eff_count=created_eff_count,
        hlt_stats=hlt_stats,
    )

    budget_summary = plot_budget_diagnostics(
        save_root=save_root,
        true_merge=budget_merge_true[test_idx],
        true_eff=budget_eff_true[test_idx],
        pred_merge=pred_budget_merge[test_idx],
        pred_eff=pred_budget_eff[test_idx],
    )

    # Jet pT response diagnostics on test split.
    pt_off_test = compute_jet_pt(const_off_sort[test_idx], masks_off_sort[test_idx])
    pt_hlt_test = compute_jet_pt(hlt_const_sort[test_idx], hlt_mask_sort[test_idx])
    pt_reco_test = compute_jet_pt(reco_const[test_idx], reco_mask[test_idx])
    pt_edges = build_pt_edges(pt_off_test, int(args.response_n_bins))
    response_hlt = jet_response_resolution(pt_off_test, pt_hlt_test, pt_edges, int(args.response_min_count))
    response_reco = jet_response_resolution(pt_off_test, pt_reco_test, pt_edges, int(args.response_min_count))
    plot_response_resolution(
        response_hlt,
        response_reco,
        "HLT",
        "Reco",
        save_root / "jet_response_resolution_hlt_vs_reco.png",
    )
    ratio_hlt = pt_hlt_test / np.clip(pt_off_test, 1e-8, None)
    ratio_reco = pt_reco_test / np.clip(pt_off_test, 1e-8, None)
    valid_hlt = np.isfinite(ratio_hlt)
    valid_reco = np.isfinite(ratio_reco)
    hlt_resp_mean = float(np.mean(ratio_hlt[valid_hlt])) if np.any(valid_hlt) else float("nan")
    hlt_resp_std = float(np.std(ratio_hlt[valid_hlt])) if np.any(valid_hlt) else float("nan")
    reco_resp_mean = float(np.mean(ratio_reco[valid_reco])) if np.any(valid_reco) else float("nan")
    reco_resp_std = float(np.std(ratio_reco[valid_reco])) if np.any(valid_reco) else float("nan")
    print("\nJet pT response (test, vs offline truth):")
    print(
        f"  HLT  mean/std: {hlt_resp_mean:.6f} / {hlt_resp_std:.6f} | "
        f"Reco mean/std: {reco_resp_mean:.6f} / {reco_resp_std:.6f}"
    )

    # Optional fusion scores (teacher/hlt/joint(dual)).
    if bool(args.save_fusion_scores):
        auc_teacher_val, preds_teacher_val, labs_val = eval_classifier(teacher, dl_va_off, device)
        auc_hlt_val, preds_hlt_val, labs_hlt_val = eval_classifier(baseline, dl_va_hlt, device)
        auc_dual_val, preds_dual_val, labs_dual_val = eval_classifier_dual(dual, dl_va_dual, device)

        assert np.array_equal(labs_val.astype(np.float32), labs_hlt_val.astype(np.float32))
        assert np.array_equal(labs_val.astype(np.float32), labs_dual_val.astype(np.float32))
        assert np.array_equal(labs_test.astype(np.float32), labels[test_idx].astype(np.float32))

        np.savez_compressed(
            save_root / "fusion_scores_val_test.npz",
            labels_val=labs_val.astype(np.float32),
            labels_test=labs_test.astype(np.float32),
            preds_teacher_val=np.asarray(preds_teacher_val, dtype=np.float64),
            preds_teacher_test=np.asarray(preds_teacher, dtype=np.float64),
            preds_hlt_val=np.asarray(preds_hlt_val, dtype=np.float64),
            preds_hlt_test=np.asarray(preds_hlt, dtype=np.float64),
            preds_joint_val=np.asarray(preds_dual_val, dtype=np.float64),
            preds_joint_test=np.asarray(preds_dual, dtype=np.float64),
            auc_teacher_val=float(auc_teacher_val),
            auc_teacher_test=float(auc_teacher),
            auc_hlt_val=float(auc_hlt_val),
            auc_hlt_test=float(auc_hlt),
            auc_joint_val=float(auc_dual_val),
            auc_joint_test=float(auc_dual),
            fpr50_joint_val=float(fpr_at_target_tpr(*roc_curve(labs_val, preds_dual_val)[:2], 0.50)),
            fpr50_joint_test=float(fpr50_dual),
            hlt_nconst_val=np.asarray(hlt_mask_sort[val_idx].sum(axis=1), dtype=np.int32),
            hlt_nconst_test=np.asarray(hlt_mask_sort[test_idx].sum(axis=1), dtype=np.int32),
            hlt_jet_pt_val=np.asarray(compute_jet_pt(hlt_const_sort[val_idx], hlt_mask_sort[val_idx]), dtype=np.float64),
            hlt_jet_pt_test=np.asarray(compute_jet_pt(hlt_const_sort[test_idx], hlt_mask_sort[test_idx]), dtype=np.float64),
            off_jet_pt_val=np.asarray(compute_jet_pt(const_off_sort[val_idx], masks_off_sort[val_idx]), dtype=np.float64),
            off_jet_pt_test=np.asarray(compute_jet_pt(const_off_sort[test_idx], masks_off_sort[test_idx]), dtype=np.float64),
        )
        print(f"Saved fusion score arrays to: {save_root / 'fusion_scores_val_test.npz'}")

    np.savez_compressed(
        save_root / "results.npz",
        auc_teacher=float(auc_teacher),
        auc_hlt=float(auc_hlt),
        auc_reco=float(auc_unmerge),
        auc_dual=float(auc_dual),
        auc_dual_flag=float(auc_dual_flag),
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_hlt=fpr_h,
        tpr_hlt=tpr_h,
        fpr_reco=fpr_r,
        tpr_reco=tpr_r,
        fpr_dual=fpr_d,
        tpr_dual=tpr_d,
        fpr_dual_flag=fpr_df,
        tpr_dual_flag=tpr_df,
        fpr30_teacher=float(fpr30_teacher),
        fpr30_hlt=float(fpr30_hlt),
        fpr30_reco=float(fpr30_reco),
        fpr30_dual=float(fpr30_dual),
        fpr30_dual_flag=float(fpr30_dual_flag),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_hlt=float(fpr50_hlt),
        fpr50_reco=float(fpr50_reco),
        fpr50_dual=float(fpr50_dual),
        fpr50_dual_flag=float(fpr50_dual_flag),
        pt_off_test=pt_off_test.astype(np.float64),
        pt_hlt_test=pt_hlt_test.astype(np.float64),
        pt_reco_test=pt_reco_test.astype(np.float64),
        hlt_response_mean=float(hlt_resp_mean),
        hlt_response_std=float(hlt_resp_std),
        reco_response_mean=float(reco_resp_mean),
        reco_response_std=float(reco_resp_std),
        selector_val_acc=float(selector_val_acc),
        selector_test_acc=float(selector_test_acc),
    )

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": hlt_cfg["hlt_effects"], "stats": hlt_stats}, f, indent=2)

    with open(save_root / "reconstructor_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(reco_val_metrics, f, indent=2)

    with open(save_root / "constituent_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(count_summary, f, indent=2)

    with open(save_root / "budget_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(budget_summary, f, indent=2)

    with open(save_root / "jet_response_resolution_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "edges": [float(x) for x in pt_edges.tolist()],
                "hlt_records": response_hlt,
                "reco_records": response_reco,
                "hlt_response_mean": hlt_resp_mean,
                "hlt_response_std": hlt_resp_std,
                "reco_response_mean": reco_resp_mean,
                "reco_response_std": reco_resp_std,
            },
            f,
            indent=2,
        )

    if selector is not None and selector_target_all is not None and winner_pred_all is not None:
        with open(save_root / "selector_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "selector_val_acc": float(selector_val_acc),
                    "selector_test_acc": float(selector_test_acc),
                },
                f,
                indent=2,
            )
        np.savez_compressed(
            save_root / "selector_outputs.npz",
            selector_feat_all=selector_feat_all.astype(np.float32),
            selector_target_all=selector_target_all.astype(np.int64),
            selector_pred_all=winner_pred_all.astype(np.int64),
        )

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": float(auc_teacher)}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": float(auc_hlt)}, save_root / "baseline.pt")
        torch.save({"model": reco.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor.pt")
        torch.save({"model": unmerge.state_dict(), "auc": float(auc_unmerge)}, save_root / "reco_only_classifier.pt")
        torch.save({"model": dual.state_dict(), "auc": float(auc_dual)}, save_root / "dual_view_classifier.pt")
        torch.save({"model": dual_flag.state_dict(), "auc": float(auc_dual_flag)}, save_root / "dual_view_conf_classifier.pt")
        if selector is not None:
            torch.save({"model": selector.state_dict()}, save_root / "hypothesis_selector.pt")

    np.savez_compressed(
        save_root / "reconstructed_dataset.npz",
        const_off=const_off_sort.astype(np.float32),
        mask_off=masks_off_sort.astype(bool),
        hlt_const=hlt_const_sort.astype(np.float32),
        hlt_mask=hlt_mask_sort.astype(bool),
        reco_const=reco_const.astype(np.float32),
        reco_mask=reco_mask.astype(bool),
        reco_merge_flag=reco_merge_flag.astype(np.float32),
        reco_eff_flag=reco_eff_flag.astype(np.float32),
        created_merge_count=created_merge_count.astype(np.int32),
        created_eff_count=created_eff_count.astype(np.int32),
        budget_merge_true=budget_merge_true.astype(np.float32),
        budget_eff_true=budget_eff_true.astype(np.float32),
        budget_total_pred=pred_budget_total.astype(np.float32),
        budget_merge_pred=pred_budget_merge.astype(np.float32),
        budget_eff_pred=pred_budget_eff.astype(np.float32),
        labels=labels.astype(np.int64),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )

    if hyp_const_all is not None and hyp_mask_all is not None and hyp_conf_all is not None:
        np.savez_compressed(
            save_root / "reconstructed_multihyp_dataset.npz",
            hyp_const=hyp_const_all.astype(np.float32),
            hyp_mask=hyp_mask_all.astype(bool),
            hyp_conf=hyp_conf_all.astype(np.float32),
            winner_idx=winner_pred_all.astype(np.int64) if winner_pred_all is not None else np.zeros((labels.shape[0],), dtype=np.int64),
        )

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-driven Offline Reconstructor + Dual-View tagger joint training.

Pipeline:
1) Build pseudo-HLT from offline jets (same realistic generator family).
2) Train teacher (offline) and baseline (HLT).
3) Stage A: pretrain reconstructor with teacher-on-reco KD + phys + budget-hinge (no chamfer-driven selection).
4) Stage B: pretrain dual-view classifier with reconstructor frozen.
5) Stage C: joint finetune reconstructor + dual-view classifier.
6) Select checkpoints by lowest val FPR@50% TPR.

Notes:
- Uses soft candidate view from reconstructor outputs for differentiable joint training.
- Includes a differentiable low-FPR surrogate targeting TPR=0.5 behavior.
- Variant: non-privileged budgeting with unmerge-only correction (Stage-A teacher-KD objective).
  * Supervises added-constituent target via true_added = (offline_count - hlt_count).
  * No privileged merge/eff split supervision.
  * Efficiency-generation branch is hard-disabled in outputs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from tqdm import tqdm

import offline_reconstructor_no_gt_local30kv2 as reco_base
from unmerge_correct_hlt import (
    RANDOM_SEED,
    load_raw_constituents_from_h5,
    compute_features,
    compute_jet_pt,
    build_pt_edges,
    jet_response_resolution,
    plot_response_resolution,
    get_stats,
    standardize,
    ParticleTransformer,
    DualViewCrossAttnClassifier,
    DualViewKDDataset,
    JetDataset,
    get_scheduler,
    eval_classifier,
    eval_classifier_dual,
    train_classifier,
)

from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
    OfflineReconstructor,    compute_reconstruction_losses,    reconstruct_dataset,
    plot_roc,
    fpr_at_target_tpr,
    plot_constituent_count_diagnostics,    train_dual_kd_student,
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


set_seed(RANDOM_SEED)


def _deepcopy_config() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _clamp_target_scale(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _weighted_batch_mean(vec: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return vec.mean()
    denom = sample_weight.sum().clamp(min=1e-6)
    return (vec * sample_weight).sum() / denom



def wrap_phi_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def _configure_smear_only_hlt(cfg: Dict) -> None:
    """Force strict smear-only HLT degradation (count-preserving by construction)."""
    h = cfg["hlt_effects"]
    h["merge_enabled"] = False
    h["merge_radius"] = 0.0
    h["pt_threshold_hlt"] = 0.0
    h["post_smear_pt_threshold"] = 0.0
    h["eff_plateau_barrel"] = 1.0
    h["eff_plateau_endcap"] = 1.0
    h["eff_floor"] = 1.0
    h["eff_ceil"] = 1.0
    h["eff_density_alpha"] = 0.0
    h["eff_quality_min"] = 1.0
    h["eff_quality_max"] = 1.0
    h["jet_quality_sigma"] = 0.0
    h["reassign_prob_base"] = 0.0
    h["reassign_density_coeff"] = 0.0
    h["reassign_prob_max"] = 0.0
    h["reassign_strength_min"] = 0.0
    h["reassign_strength_max"] = 0.0


def _scale_smearing_widths(cfg: Dict, smear_scale: float) -> None:
    """Scale all smearing width knobs by a global factor."""
    s = float(smear_scale)
    if s <= 0.0:
        raise ValueError("--smear_scale must be > 0.")
    h = cfg["hlt_effects"]
    keys = [
        "smear_a",
        "smear_b",
        "smear_c",
        "smear_sigma_min",
        "smear_sigma_max",
        "eta_smear_const",
        "eta_smear_inv_sqrt",
        "phi_smear_const",
        "phi_smear_inv_sqrt",
        "tail_sigma_scale",
        "tail_sigma_add",
    ]
    for k in keys:
        h[k] = float(h[k]) * s


def _apply_calibrated_smearing(
    cfg: Dict,
    use_pt_gev: bool,
    core_scale: float,
    angle_scale: float,
    tail_base: float,
    tail_sigma_mult: float,
    tail_prob_max: float,
) -> None:
    """
    Calibrate smearing strength in a physically interpretable way.

    - If use_pt_gev=True, convert pT-dependent resolution terms to GeV behavior
      while data remains stored in MeV-like units.
    - core_scale multiplies relative pT response width.
    - angle_scale multiplies eta/phi smearing widths.
    - tail knobs control non-Gaussian tail frequency/width.
    """
    h = cfg["hlt_effects"]
    unit_scale = 1000.0 if bool(use_pt_gev) else 1.0
    sqrt_unit = float(np.sqrt(unit_scale))

    core = float(core_scale)
    if core <= 0.0:
        raise ValueError("--smear_core_scale must be > 0")
    ang = float(angle_scale)
    if ang <= 0.0:
        raise ValueError("--smear_angle_scale must be > 0")

    h["smear_a"] = float(h["smear_a"]) * sqrt_unit * core
    h["smear_b"] = float(h["smear_b"]) * core
    h["smear_c"] = float(h["smear_c"]) * unit_scale * core

    h["eta_smear_const"] = float(h["eta_smear_const"]) * ang
    h["phi_smear_const"] = float(h["phi_smear_const"]) * ang
    h["eta_smear_inv_sqrt"] = float(h["eta_smear_inv_sqrt"]) * sqrt_unit * ang
    h["phi_smear_inv_sqrt"] = float(h["phi_smear_inv_sqrt"]) * sqrt_unit * ang

    ts = float(tail_sigma_mult)
    if ts <= 0.0:
        raise ValueError("--smear_tail_sigma_mult must be > 0")
    h["tail_sigma_scale"] = float(h["tail_sigma_scale"]) * ts
    h["tail_sigma_add"] = float(h["tail_sigma_add"]) * ts

    if float(tail_base) >= 0.0:
        h["tail_base"] = float(tail_base)
    if float(tail_prob_max) >= 0.0:
        h["tail_prob_max"] = float(tail_prob_max)


def _flatten_token_residuals(
    const_off: np.ndarray,
    const_other: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    valid = mask.astype(bool)
    dpt = const_other[..., 0] - const_off[..., 0]
    deta = const_other[..., 1] - const_off[..., 1]
    dphi = wrap_phi_np(const_other[..., 2] - const_off[..., 2])
    pt_ref = const_off[..., 0]
    eta_ref = np.abs(const_off[..., 1])
    return {
        "dpt": dpt[valid].astype(np.float64),
        "deta": deta[valid].astype(np.float64),
        "dphi": dphi[valid].astype(np.float64),
        "pt_ref": pt_ref[valid].astype(np.float64),
        "abs_eta_ref": eta_ref[valid].astype(np.float64),
    }


def _residual_metric_pack(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "mae": float(np.mean(np.abs(x))),
        "rmse": float(np.sqrt(np.mean(np.square(x)))),
    }


def _safe_improvement(old: float, new: float) -> float:
    if (not np.isfinite(old)) or abs(old) < 1e-12:
        return float("nan")
    return float((old - new) / old)


def _summarize_constituent_residuals(
    const_off: np.ndarray,
    const_hlt: np.ndarray,
    const_reco: np.ndarray,
    mask_common: np.ndarray,
) -> Dict[str, object]:
    hlt = _flatten_token_residuals(const_off, const_hlt, mask_common)
    reco = _flatten_token_residuals(const_off, const_reco, mask_common)

    out: Dict[str, object] = {
        "n_tokens": int(hlt["dpt"].shape[0]),
    }

    for key in ("dpt", "deta", "dphi"):
        mh = _residual_metric_pack(hlt[key])
        mr = _residual_metric_pack(reco[key])
        improved = np.abs(reco[key]) < np.abs(hlt[key]) if hlt[key].size > 0 else np.array([], dtype=bool)
        out[key] = {
            "hlt_vs_offline": mh,
            "reco_vs_offline": mr,
            "improvement": {
                "mean_abs_reduction_frac": _safe_improvement(abs(mh["mean"]), abs(mr["mean"])),
                "std_reduction_frac": _safe_improvement(mh["std"], mr["std"]),
                "mae_reduction_frac": _safe_improvement(mh["mae"], mr["mae"]),
                "rmse_reduction_frac": _safe_improvement(mh["rmse"], mr["rmse"]),
                "fraction_tokens_improved": float(np.mean(improved)) if improved.size > 0 else float("nan"),
            },
        }

    pt_edges = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, np.inf], dtype=np.float64)
    eta_edges = np.array([0.0, 0.8, 1.37, 1.8, 2.5, 3.2, 5.0], dtype=np.float64)

    def _bin_mae(var_key: str, ref_key: str, edges: np.ndarray) -> list[dict[str, float]]:
        ref = hlt[ref_key]
        xh = np.abs(hlt[var_key])
        xr = np.abs(reco[var_key])
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (ref >= lo) & (ref < hi)
            n = int(np.sum(sel))
            if n > 0:
                mae_h = float(np.mean(xh[sel]))
                mae_r = float(np.mean(xr[sel]))
                imp = _safe_improvement(mae_h, mae_r)
            else:
                mae_h = float("nan")
                mae_r = float("nan")
                imp = float("nan")
            rows.append({"low": float(lo), "high": float(hi), "n": n, "mae_hlt": mae_h, "mae_reco": mae_r, "mae_reduction_frac": imp})
        return rows

    out["binning"] = {
        "by_pt": {
            "dpt": _bin_mae("dpt", "pt_ref", pt_edges),
            "deta": _bin_mae("deta", "pt_ref", pt_edges),
            "dphi": _bin_mae("dphi", "pt_ref", pt_edges),
        },
        "by_abs_eta": {
            "dpt": _bin_mae("dpt", "abs_eta_ref", eta_edges),
            "deta": _bin_mae("deta", "abs_eta_ref", eta_edges),
            "dphi": _bin_mae("dphi", "abs_eta_ref", eta_edges),
        },
    }
    return out


def _plot_constituent_residual_distributions(
    const_off: np.ndarray,
    const_hlt: np.ndarray,
    const_reco: np.ndarray,
    mask_common: np.ndarray,
    out_path: Path,
) -> None:
    hlt = _flatten_token_residuals(const_off, const_hlt, mask_common)
    reco = _flatten_token_residuals(const_off, const_reco, mask_common)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    specs = [
        ("dpt", r"$\\Delta p_T$ (other - offline)"),
        ("deta", r"$\\Delta \\eta$ (other - offline)"),
        ("dphi", r"$\\Delta \\phi$ (other - offline)"),
    ]
    for ax, (k, xlab) in zip(axes, specs):
        h = hlt[k]
        r = reco[k]
        if (h.size + r.size) > 0:
            lim = float(np.nanpercentile(np.concatenate([h, r]), 99.0))
        else:
            lim = 1.0
        lim = max(float(lim), 1e-6)
        bins = np.linspace(-lim, lim, 120)
        ax.hist(h, bins=bins, density=True, histtype="step", linewidth=1.6, label="HLT vs offline", color="steelblue")
        ax.hist(r, bins=bins, density=True, histtype="step", linewidth=1.6, label="Reco vs offline", color="darkorange")
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Constituent Residual Distributions (Test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_constituent_mae_by_bin(summary: Dict[str, object], out_path: Path, axis_key: str, title: str) -> None:
    bins = summary["binning"][axis_key]  # type: ignore[index]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=False)
    keys = ["dpt", "deta", "dphi"]
    labels = [r"|$\\Delta p_T$| MAE", r"|$\\Delta \\eta$| MAE", r"|$\\Delta \\phi$| MAE"]
    for ax, k, yl in zip(axes, keys, labels):
        rows = bins[k]  # type: ignore[index]
        xs, ys_h, ys_r = [], [], []
        for row in rows:
            lo = float(row["low"])
            hi = float(row["high"])
            x = 0.5 * (lo + (hi if np.isfinite(hi) else lo + max(1.0, lo)))
            xs.append(x)
            ys_h.append(float(row["mae_hlt"]))
            ys_r.append(float(row["mae_reco"]))
        ax.plot(xs, ys_h, marker="o", label="HLT vs offline", color="steelblue")
        ax.plot(xs, ys_r, marker="s", label="Reco vs offline", color="darkorange")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25)
    axes[0].set_xlabel("Bin center")
    axes[1].set_xlabel("Bin center")
    axes[2].set_xlabel("Bin center")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def _predict_reco_tokens_ordered(
    model: OfflineReconstructor,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = feat_hlt.shape[0]
    out_const = np.zeros_like(const_hlt, dtype=np.float32)
    out_mask = mask_hlt.copy().astype(bool)
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        x = torch.tensor(feat_hlt[start:end], dtype=torch.float32, device=device)
        m = torch.tensor(mask_hlt[start:end], dtype=torch.bool, device=device)
        c = torch.tensor(const_hlt[start:end], dtype=torch.float32, device=device)
        reco_out = model(x, m, c, stage_scale=1.0)
        L = x.shape[1]
        tok = reco_out["cand_tokens"][:, :L, :].detach().cpu().numpy().astype(np.float32)
        out_const[start:end] = tok
    out_const[~out_mask] = 0.0
    return out_const, out_mask
def predict_single_view_scores(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = JetDataset(feat, mask, labels)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    _, preds, labs = eval_classifier(model, dl, device)
    return preds.astype(np.float32), labs.astype(np.float32)


def prob_threshold_at_target_tpr(pos_probs: np.ndarray, target_tpr: float) -> float:
    if pos_probs.size == 0:
        return 0.5
    q = float(np.clip(1.0 - float(target_tpr), 0.0, 1.0))
    return float(np.quantile(pos_probs, q))


def build_discrepancy_weights(
    y_train: np.ndarray,
    p_teacher_train: np.ndarray,
    p_baseline_train: np.ndarray,
    p_teacher_val: np.ndarray,
    p_baseline_val: np.ndarray,
    y_val: np.ndarray,
    target_tpr: float,
    tau: float,
    lambda_disc: float,
    max_mult: float,
    include_pos: bool,
    pos_scale: float,
    normalize_mean_one: bool,
    teacher_conf_min: float,
    correctness_tau: float,
    use_teacher_hard_correct_gate: bool,
    use_teacher_conf_gate: bool,
    use_teacher_better_gate: bool,
    weight_mode: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    tau = max(float(tau), 1e-6)
    correctness_tau = max(float(correctness_tau), 1e-6)
    target_tpr = float(np.clip(target_tpr, 0.0, 1.0))
    max_mult = max(float(max_mult), 1.0)
    lambda_disc = max(float(lambda_disc), 0.0)
    teacher_conf_min = float(np.clip(teacher_conf_min, 0.0, 1.0))

    t_off = prob_threshold_at_target_tpr(p_teacher_val[y_val == 1], target_tpr)
    t_hlt = prob_threshold_at_target_tpr(p_baseline_val[y_val == 1], target_tpr)

    y_train = y_train.astype(np.int64)
    p_teacher_true = np.where(y_train == 1, p_teacher_train, 1.0 - p_teacher_train).astype(np.float32)
    p_hlt_true = np.where(y_train == 1, p_baseline_train, 1.0 - p_baseline_train).astype(np.float32)

    teacher_hard_correct = ((p_teacher_train >= 0.5).astype(np.int64) == y_train).astype(np.float32)
    gate = np.ones_like(p_teacher_true, dtype=np.float32)
    if bool(use_teacher_hard_correct_gate):
        gate *= teacher_hard_correct
    if bool(use_teacher_conf_gate):
        gate *= _sigmoid_np((p_teacher_true - teacher_conf_min) / correctness_tau).astype(np.float32)
    if bool(use_teacher_better_gate):
        gate *= _sigmoid_np((p_teacher_true - p_hlt_true) / correctness_tau).astype(np.float32)

    mode = str(weight_mode).strip().lower()
    if mode not in ("tail_disagreement", "smooth_delta"):
        raise ValueError(f"Unsupported --disc_weight_mode: {weight_mode}")

    if mode == "smooth_delta":
        r_neg = (
            np.maximum(p_baseline_train - p_teacher_train, 0.0).astype(np.float32)
            * (y_train == 0).astype(np.float32)
            * gate
        ).astype(np.float32)
        r = r_neg.astype(np.float32, copy=True)
        r_pos = np.zeros_like(r, dtype=np.float32)
        if bool(include_pos):
            r_pos = (
                np.maximum(p_teacher_train - p_baseline_train, 0.0).astype(np.float32)
                * (y_train == 1).astype(np.float32)
                * gate
            ).astype(np.float32)
            r += float(pos_scale) * r_pos
    else:
        r_neg = (
            _sigmoid_np((p_baseline_train - t_hlt) / tau)
            * _sigmoid_np((t_off - p_teacher_train) / tau)
            * (y_train == 0).astype(np.float32)
            * gate
        ).astype(np.float32)
        r = r_neg.astype(np.float32, copy=True)
        r_pos = np.zeros_like(r, dtype=np.float32)
        if bool(include_pos):
            r_pos = (
                _sigmoid_np((p_teacher_train - t_off) / tau)
                * _sigmoid_np((t_hlt - p_baseline_train) / tau)
                * (y_train == 1).astype(np.float32)
                * gate
            ).astype(np.float32)
            r += float(pos_scale) * r_pos

    w = (1.0 + float(lambda_disc) * r).astype(np.float32)
    w = np.clip(w, 1.0, float(max_mult)).astype(np.float32)
    if bool(normalize_mean_one):
        m = float(np.mean(w))
        if m > 0:
            w = (w / m).astype(np.float32)

    summary = {
        "weight_mode": mode,
        "target_tpr": float(target_tpr),
        "tau": float(tau),
        "lambda_disc": float(lambda_disc),
        "max_mult": float(max_mult),
        "include_pos": bool(include_pos),
        "pos_scale": float(pos_scale),
        "normalize_mean_one": bool(normalize_mean_one),
        "teacher_conf_min": float(teacher_conf_min),
        "correctness_tau": float(correctness_tau),
        "use_teacher_hard_correct_gate": bool(use_teacher_hard_correct_gate),
        "use_teacher_conf_gate": bool(use_teacher_conf_gate),
        "use_teacher_better_gate": bool(use_teacher_better_gate),
        "t_off_val": float(t_off),
        "t_hlt_val": float(t_hlt),
        "teacher_hard_correct_rate": float(np.mean(teacher_hard_correct)),
        "mean_r_neg": float(np.mean(r_neg)),
        "mean_r_pos": float(np.mean(r_pos)),
        "mean_weight": float(np.mean(w)),
        "p95_weight": float(np.percentile(w, 95.0)),
        "max_weight": float(np.max(w)),
        "fraction_w_gt_1p1": float(np.mean(w > 1.1)),
        "fraction_w_gt_1p5": float(np.mean(w > 1.5)),
    }
    return w, summary


def compute_reconstruction_losses_weighted(
    out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    loss_cfg: Dict,
    sample_weight: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    eps = 1e-8
    sw = None
    if sample_weight is not None:
        sw = sample_weight.float().clamp(min=0.0)

    pred = out["cand_tokens"]
    w = out["cand_weights"].clamp(0.0, 1.0)

    cost = reco_base._token_cost_matrix(pred, const_off)
    valid_tgt = mask_off.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))

    pred_to_tgt = cost.min(dim=2).values
    loss_pred_to_tgt = (w * pred_to_tgt).sum(dim=1) / (w.sum(dim=1) + eps)

    penalty = float(loss_cfg["unselected_penalty"]) * (1.0 - w).unsqueeze(2)
    tgt_to_pred = (cost + penalty).min(dim=1).values
    tgt_mask_f = mask_off.float()
    loss_tgt_to_pred = (tgt_to_pred * tgt_mask_f).sum(dim=1) / (tgt_mask_f.sum(dim=1) + eps)
    loss_set_vec = loss_pred_to_tgt + loss_tgt_to_pred

    pred_px, pred_py, pred_pz, pred_E = reco_base._weighted_fourvec_sums(pred, w)
    true_px, true_py, true_pz, true_E = reco_base._weighted_fourvec_sums(const_off, mask_off.float())

    norm = true_px.abs() + true_py.abs() + true_pz.abs() + true_E.abs() + 1.0
    loss_phys_vec = (
        (pred_px - true_px).abs()
        + (pred_py - true_py).abs()
        + (pred_pz - true_pz).abs()
        + (pred_E - true_E).abs()
    ) / norm
    pred_pt = torch.sqrt(pred_px.pow(2) + pred_py.pow(2) + eps)
    true_pt = torch.sqrt(true_px.pow(2) + true_py.pow(2) + eps)
    pt_ratio = pred_pt / (true_pt + eps)
    loss_pt_ratio_vec = F.smooth_l1_loss(pt_ratio, torch.ones_like(pt_ratio), reduction="none")
    e_ratio = pred_E / (true_E + eps)
    loss_e_ratio_vec = F.smooth_l1_loss(e_ratio, torch.ones_like(e_ratio), reduction="none")

    true_count = mask_off.float().sum(dim=1)
    hlt_count = mask_hlt.float().sum(dim=1)
    true_added = (true_count - hlt_count).clamp(min=0.0)

    pred_count = w.sum(dim=1)
    pred_added = out["child_weight"].sum(dim=1) + out["gen_weight"].sum(dim=1)
    pred_added_merge = out["child_weight"].sum(dim=1)
    pred_added_eff = out["gen_weight"].sum(dim=1)

    budget_total = out["budget_total"]
    budget_merge = out["budget_merge"]
    budget_eff = out["budget_eff"]
    budget_true_total = budget_merge_true + budget_eff_true

    loss_budget_vec = (
        F.smooth_l1_loss(pred_count, true_count, reduction="none")
        + F.smooth_l1_loss(budget_total, true_count, reduction="none")
        + F.smooth_l1_loss(pred_added, true_added, reduction="none")
        + F.smooth_l1_loss(budget_merge + budget_eff, true_added, reduction="none")
        + F.smooth_l1_loss(budget_merge, budget_merge_true, reduction="none")
        + F.smooth_l1_loss(budget_eff, budget_eff_true, reduction="none")
        + F.smooth_l1_loss(pred_added_merge, budget_merge_true, reduction="none")
        + F.smooth_l1_loss(pred_added_eff, budget_eff_true, reduction="none")
        + F.smooth_l1_loss(budget_merge + budget_eff, budget_true_total, reduction="none")
    )

    loss_sparse_vec = out["child_weight"].mean(dim=1) + out["gen_weight"].mean(dim=1)

    split_delta = out["split_delta"]
    split_step = torch.sqrt(split_delta[..., 1].pow(2) + split_delta[..., 2].pow(2) + eps)
    split_w = out["child_weight"].reshape(split_step.shape)
    loss_local_split_vec = (split_w * split_step).sum(dim=(1, 2)) / (split_w.sum(dim=(1, 2)) + eps)

    gen_tokens = out["gen_tokens"]
    gen_w = out["gen_weight"]
    h_eta = const_hlt[:, :, 1]
    h_phi = const_hlt[:, :, 2]
    g_eta = gen_tokens[:, :, 1].unsqueeze(2)
    g_phi = gen_tokens[:, :, 2].unsqueeze(2)
    d_eta = g_eta - h_eta.unsqueeze(1)
    d_phi = torch.atan2(torch.sin(g_phi - h_phi.unsqueeze(1)), torch.cos(g_phi - h_phi.unsqueeze(1)))
    dR = torch.sqrt(d_eta.pow(2) + d_phi.pow(2) + eps)
    dR = torch.where(mask_hlt.unsqueeze(1), dR, torch.full_like(dR, 1e4))
    nearest = dR.min(dim=2).values
    excess = F.relu(nearest - float(loss_cfg["gen_local_radius"]))
    loss_local_gen_vec = (gen_w * excess).sum(dim=1) / (gen_w.sum(dim=1) + eps)
    loss_local_vec = loss_local_split_vec + loss_local_gen_vec

    total_vec = (
        float(loss_cfg["w_set"]) * loss_set_vec
        + float(loss_cfg["w_phys"]) * loss_phys_vec
        + float(loss_cfg["w_pt_ratio"]) * loss_pt_ratio_vec
        + float(loss_cfg["w_e_ratio"]) * loss_e_ratio_vec
        + float(loss_cfg["w_budget"]) * loss_budget_vec
        + float(loss_cfg["w_sparse"]) * loss_sparse_vec
        + float(loss_cfg["w_local"]) * loss_local_vec
    )

    return {
        "total": _weighted_batch_mean(total_vec, sw),
        "set": _weighted_batch_mean(loss_set_vec, sw),
        "phys": _weighted_batch_mean(loss_phys_vec, sw),
        "pt_ratio": _weighted_batch_mean(loss_pt_ratio_vec, sw),
        "e_ratio": _weighted_batch_mean(loss_e_ratio_vec, sw),
        "budget": _weighted_batch_mean(loss_budget_vec, sw),
        "sparse": _weighted_batch_mean(loss_sparse_vec, sw),
        "local": _weighted_batch_mean(loss_local_vec, sw),
    }



def enforce_unsmear_only_output(reco_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Hard-disable split/generation branches and enforce token-count preservation.

    This converts the base reconstructor outputs into a strict unsmear-only view:
    - keep only per-token corrected slots (first L tokens),
    - zero all split/gen outputs and branch flags,
    - force candidate weights to 1.0 for valid HLT tokens and 0 otherwise.
    """
    out = dict(reco_out)
    L = int(out["action_prob"].shape[1])

    tok_valid = (out["cand_weights"][:, :L] > 0.0).float()

    # Count-preserving candidate weights.
    cw = torch.zeros_like(out["cand_weights"])
    cw[:, :L] = tok_valid
    out["cand_weights"] = cw

    # Disable split/gen branches.
    out["child_weight"] = torch.zeros_like(out["child_weight"])
    out["gen_weight"] = torch.zeros_like(out["gen_weight"])
    out["budget_merge"] = torch.zeros_like(out["budget_merge"])
    out["budget_eff"] = torch.zeros_like(out["budget_eff"])
    out["budget_total"] = tok_valid.sum(dim=1)

    if "cand_merge_flags" in out:
        out["cand_merge_flags"] = torch.zeros_like(out["cand_merge_flags"])
    if "cand_eff_flags" in out:
        out["cand_eff_flags"] = torch.zeros_like(out["cand_eff_flags"])

    return out
def wrap_reconstructor_unsmear_only(model: OfflineReconstructor) -> OfflineReconstructor:
    """
    Wrap reconstructor forward so all downstream calls (including reconstruct_dataset)
    see unsmear-only outputs.
    """
    base_forward = model.forward

    def _wrapped_forward(*args, **kwargs):
        return enforce_unsmear_only_output(base_forward(*args, **kwargs))

    model.forward = _wrapped_forward  # type: ignore[method-assign]
    return model


def train_single_view_classifier_auc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> nn.Module:
    """Train single-view top-tagger and select checkpoint by best val AUC."""
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    best_val_auc = float("-inf")
    fpr50_at_best = float("nan")
    best_state = None
    no_improve = 0

    for ep in tqdm(range(int(train_cfg["epochs"])), desc=name):
        _, tr_auc = train_classifier(model, train_loader, opt, device)
        va_auc, va_preds, va_labs = eval_classifier(model, val_loader, device)
        va_fpr, va_tpr, _ = roc_curve(va_labs, va_preds)
        va_fpr50 = fpr_at_target_tpr(va_fpr, va_tpr, 0.50)
        sch.step()

        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            fpr50_at_best = float(va_fpr50)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, "
                f"val_fpr50={va_fpr50:.6f}, best_auc={best_val_auc:.4f}, "
                f"fpr50@best={fpr50_at_best:.6f}"
            )
        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class JointDualDataset(Dataset):
    def __init__(
        self,
        feat_hlt_reco: np.ndarray,
        feat_hlt_dual: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
        labels: np.ndarray,
        sample_weight_cls: np.ndarray | None = None,
        sample_weight_reco: np.ndarray | None = None,
    ):
        self.feat_hlt_reco = torch.tensor(feat_hlt_reco, dtype=torch.float32)
        self.feat_hlt_dual = torch.tensor(feat_hlt_dual, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        n = labels.shape[0]
        if sample_weight_cls is None:
            sw_cls = np.ones((n,), dtype=np.float32)
        else:
            sw_cls = np.asarray(sample_weight_cls, dtype=np.float32)
            if sw_cls.shape[0] != n:
                raise ValueError(f"sample_weight_cls length mismatch: {sw_cls.shape[0]} vs {n}")
        if sample_weight_reco is None:
            sw_reco = np.ones((n,), dtype=np.float32)
        else:
            sw_reco = np.asarray(sample_weight_reco, dtype=np.float32)
            if sw_reco.shape[0] != n:
                raise ValueError(f"sample_weight_reco length mismatch: {sw_reco.shape[0]} vs {n}")
        self.sample_weight_cls = torch.tensor(sw_cls, dtype=torch.float32)
        self.sample_weight_reco = torch.tensor(sw_reco, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt_reco.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt_reco": self.feat_hlt_reco[i],
            "feat_hlt_dual": self.feat_hlt_dual[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
            "label": self.labels[i],
            "sample_weight_cls": self.sample_weight_cls[i],
            "sample_weight_reco": self.sample_weight_reco[i],
        }


class WeightedReconstructionDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        labels: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
        sample_weight_reco: np.ndarray | None = None,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)
        n = feat_hlt.shape[0]
        if sample_weight_reco is None:
            sw = np.ones((n,), dtype=np.float32)
        else:
            sw = np.asarray(sample_weight_reco, dtype=np.float32)
            if sw.shape[0] != n:
                raise ValueError(f"sample_weight_reco length mismatch: {sw.shape[0]} vs {n}")
        self.sample_weight_reco = torch.tensor(sw, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "label": self.labels[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
            "sample_weight_reco": self.sample_weight_reco[i],
        }


def compute_features_torch(const: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Torch equivalent of compute_features(...) from unmerge_correct_hlt.py.
    const: [B, L, 4] => [pt, eta, phi, E]
    mask:  [B, L] bool
    returns: [B, L, 7]
    """
    eps = 1e-8
    pt = const[..., 0].clamp(min=eps)
    eta = const[..., 1].clamp(min=-5.0, max=5.0)
    phi = const[..., 2]
    E = const[..., 3].clamp(min=eps)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    w = mask.float()
    jet_px = (px * w).sum(dim=1, keepdim=True)
    jet_py = (py * w).sum(dim=1, keepdim=True)
    jet_pz = (pz * w).sum(dim=1, keepdim=True)
    jet_E = (E * w).sum(dim=1, keepdim=True)

    jet_pt = torch.sqrt(jet_px.pow(2) + jet_py.pow(2) + eps)
    jet_p = torch.sqrt(jet_px.pow(2) + jet_py.pow(2) + jet_pz.pow(2) + eps)
    ratio = (jet_p + jet_pz) / (jet_p - jet_pz + eps)
    ratio = torch.clamp(ratio, min=1e-8, max=1e8)
    jet_eta = 0.5 * torch.log(ratio)
    jet_phi = torch.atan2(jet_py, jet_px)

    delta_eta = eta - jet_eta
    delta_phi = torch.atan2(torch.sin(phi - jet_phi), torch.cos(phi - jet_phi))

    log_pt = torch.log(pt + eps)
    log_E = torch.log(E + eps)
    log_pt_rel = torch.log(pt / (jet_pt + eps) + eps)
    log_E_rel = torch.log(E / (jet_E + eps) + eps)
    delta_R = torch.sqrt(delta_eta.pow(2) + delta_phi.pow(2) + eps)

    feat = torch.stack(
        [delta_eta, delta_phi, log_pt, log_E, log_pt_rel, log_E_rel, delta_R],
        dim=-1,
    )
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = torch.clamp(feat, min=-20.0, max=20.0)
    feat = feat * w.unsqueeze(-1)
    return feat


def build_soft_corrected_view(
    reco_out: Dict[str, torch.Tensor],
    weight_floor: float = 1e-4,
    scale_features_by_weight: bool = True,
    include_flags: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds a differentiable fixed-length corrected view.
    The DualViewCrossAttnClassifier expects both views to share the same sequence length.
    We therefore map reconstructor outputs back to L token slots (L = HLT token count):
      - corrected token kinematics from tok branch
      - parent-level split mass summary
      - per-token share of efficiency budget

    Output feature dims:
      - default: 7 base kinematic features + 3 channels
        [tok_weight, parent_added_weight, eff_share] = 10.
      - with include_flags: +2 channels
        [merge_flag, eff_flag] => 12.
    """
    reco_out = enforce_unsmear_only_output(reco_out)
    eps = 1e-8
    L = reco_out["action_prob"].shape[1]
    tok_tokens = reco_out["cand_tokens"][:, :L, :]
    tok_w = reco_out["cand_weights"][:, :L].clamp(0.0, 1.0)
    mask_b = tok_w > float(weight_floor)
    none_valid = ~mask_b.any(dim=1)
    if none_valid.any():
        mask_b = mask_b.clone()
        mask_b[none_valid, 0] = True

    feat7 = compute_features_torch(tok_tokens, mask_b)
    if scale_features_by_weight:
        feat7 = feat7 * tok_w.unsqueeze(-1)

    # Parent-level merge-added mass per token from split branch.
    child_w = reco_out["child_weight"]
    K = max(int(child_w.shape[1] // max(L, 1)), 1)
    parent_added = child_w.reshape(child_w.shape[0], L, K).sum(dim=2).clamp(0.0, 1.0)

    # Distribute efficiency budget as a smooth per-token share signal.
    valid_count = mask_b.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    eff_share = (reco_out["budget_eff"].unsqueeze(1) / valid_count).clamp(0.0, 1.0)
    eff_share = eff_share * mask_b.float()

    extra = torch.stack([tok_w, parent_added, eff_share], dim=-1)
    if include_flags:
        tok_merge_flag = reco_out["cand_merge_flags"][:, :L].clamp(0.0, 1.0)
        tok_eff_flag = reco_out["cand_eff_flags"][:, :L].clamp(0.0, 1.0)
        extra = torch.cat([extra, tok_merge_flag.unsqueeze(-1), tok_eff_flag.unsqueeze(-1)], dim=-1)
    feat_b = torch.cat([feat7, extra], dim=-1)
    feat_b = torch.nan_to_num(feat_b, nan=0.0, posinf=0.0, neginf=0.0)
    feat_b = feat_b * mask_b.unsqueeze(-1).float()
    return feat_b, mask_b


def low_fpr_surrogate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_tpr: float = 0.50,
    tau: float = 0.05,
) -> torch.Tensor:
    """
    Differentiable proxy that targets low FPR around fixed TPR operating point.
    """
    probs = torch.sigmoid(logits)
    pos = probs[labels > 0.5]
    neg = probs[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.zeros((), device=logits.device)

    # TPR=0.5 implies threshold near median positive score.
    q = float(max(0.0, min(1.0, 1.0 - target_tpr)))
    thr = torch.quantile(pos.detach(), q=q)

    # Soft FPR proxy: negatives above threshold.
    neg_term = torch.sigmoid((neg - thr) / max(float(tau), 1e-4)).mean()
    # Keep positive side calibrated around threshold.
    pos_term = torch.sigmoid((thr - pos) / max(float(tau), 1e-4)).mean()
    return neg_term + 0.5 * pos_term


@torch.no_grad()
def eval_joint_model(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    dual_model.eval()
    reconstructor.eval()

    preds = []
    labs = []
    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        feat_hlt_dual = batch["feat_hlt_dual"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = build_soft_corrected_view(
            reco_out,
            weight_floor=corrected_weight_floor,
            scale_features_by_weight=True,
            include_flags=corrected_use_flags,
        )
        logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)
        p = torch.sigmoid(logits)
        preds.append(p.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    labs = np.concatenate(labs) if labs else np.zeros(0, dtype=np.float32)
    if preds.size == 0:
        return float("nan"), preds, labs, float("nan")
    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(labs, preds)
    fpr50 = fpr_at_target_tpr(fpr, tpr, 0.50)
    return float(auc), preds, labs, float(fpr50)


@torch.no_grad()
def eval_joint_model_both_metrics(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool = False,
    weighted_key: Optional[str] = None,
) -> Dict[str, object]:
    dual_model.eval()
    reconstructor.eval()

    preds_list = []
    labs_list = []
    w_list = []
    has_weights = weighted_key is not None

    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        feat_hlt_dual = batch["feat_hlt_dual"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = build_soft_corrected_view(
            reco_out,
            weight_floor=corrected_weight_floor,
            scale_features_by_weight=True,
            include_flags=corrected_use_flags,
        )
        logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)
        p = torch.sigmoid(logits)
        preds_list.append(p.detach().cpu().numpy())
        labs_list.append(y.detach().cpu().numpy())

        if has_weights:
            if weighted_key in batch:
                w_list.append(batch[weighted_key].detach().cpu().numpy())
            else:
                has_weights = False
                w_list = []

    preds = np.concatenate(preds_list) if preds_list else np.zeros(0, dtype=np.float32)
    labs = np.concatenate(labs_list) if labs_list else np.zeros(0, dtype=np.float32)
    if preds.size == 0:
        return {
            "preds": preds,
            "labs": labs,
            "weights": np.zeros(0, dtype=np.float32),
            "auc_unweighted": float("nan"),
            "fpr50_unweighted": float("nan"),
            "auc_weighted": float("nan"),
            "fpr50_weighted": float("nan"),
        }

    auc_unw = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else float("nan")
    fpr_unw, tpr_unw, _ = roc_curve(labs, preds)
    fpr50_unw = fpr_at_target_tpr(fpr_unw, tpr_unw, 0.50)

    weights = np.concatenate(w_list).astype(np.float32) if (has_weights and w_list) else np.zeros(0, dtype=np.float32)
    if weights.size == preds.size and float(np.sum(weights)) > 0.0 and len(np.unique(labs)) > 1:
        auc_w = roc_auc_score(labs, preds, sample_weight=weights)
        fpr_w, tpr_w, _ = roc_curve(labs, preds, sample_weight=weights)
        fpr50_w = fpr_at_target_tpr(fpr_w, tpr_w, 0.50)
    else:
        auc_w = float("nan")
        fpr50_w = float("nan")

    return {
        "preds": preds,
        "labs": labs,
        "weights": weights,
        "auc_unweighted": float(auc_unw),
        "fpr50_unweighted": float(fpr50_unw),
        "auc_weighted": float(auc_w),
        "fpr50_weighted": float(fpr50_w),
    }


@torch.no_grad()
def build_corrected_view_numpy(
    reconstructor: OfflineReconstructor,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    device: torch.device,
    batch_size: int,
    corrected_weight_floor: float,
    corrected_use_flags: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    n_jets, seq_len, _ = feat_hlt.shape
    out_dim_b = 12 if corrected_use_flags else 10
    feat_b = np.zeros((n_jets, seq_len, out_dim_b), dtype=np.float32)
    mask_b = np.zeros((n_jets, seq_len), dtype=bool)

    reconstructor.eval()
    for start in range(0, n_jets, int(batch_size)):
        end = min(start + int(batch_size), n_jets)
        x = torch.tensor(feat_hlt[start:end], dtype=torch.float32, device=device)
        m = torch.tensor(mask_hlt[start:end], dtype=torch.bool, device=device)
        c = torch.tensor(const_hlt[start:end], dtype=torch.float32, device=device)
        reco_out = reconstructor(x, m, c, stage_scale=1.0)
        fb, mb = build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=bool(corrected_use_flags),
        )
        feat_b[start:end] = fb.detach().cpu().numpy()
        mask_b[start:end] = mb.detach().cpu().numpy()
    return feat_b, mask_b


def summarize_soft_corrected_view(
    feat_b: np.ndarray,
    mask_b: np.ndarray,
) -> Dict[str, float]:
    # Extra channels: [tok_weight, parent_added_weight, eff_share] (+ optional merge/eff flags)
    tok_w = feat_b[..., 7]
    parent_added = feat_b[..., 8]
    eff_share = feat_b[..., 9]
    has_flags = feat_b.shape[-1] >= 12
    merge_flag_soft = feat_b[..., 10] if has_flags else np.zeros_like(tok_w)
    eff_flag_soft = feat_b[..., 11] if has_flags else np.zeros_like(tok_w)
    valid = mask_b.astype(bool)
    if not np.any(valid):
        return {
            "mean_tokens_active_per_jet": 0.0,
            "mean_tok_weight_valid": 0.0,
            "mean_parent_added_valid": 0.0,
            "mean_eff_share_valid": 0.0,
            "mean_merge_flag_soft_valid": 0.0,
            "mean_eff_flag_soft_valid": 0.0,
            "p95_tok_weight_valid": 0.0,
            "p95_parent_added_valid": 0.0,
            "p95_eff_share_valid": 0.0,
            "p95_merge_flag_soft_valid": 0.0,
            "p95_eff_flag_soft_valid": 0.0,
        }
    tok_cnt = valid.sum(axis=1).astype(np.float64)
    return {
        "mean_tokens_active_per_jet": float(tok_cnt.mean()),
        "mean_tok_weight_valid": float(tok_w[valid].mean()),
        "mean_parent_added_valid": float(parent_added[valid].mean()),
        "mean_eff_share_valid": float(eff_share[valid].mean()),
        "mean_merge_flag_soft_valid": float(merge_flag_soft[valid].mean()) if has_flags else 0.0,
        "mean_eff_flag_soft_valid": float(eff_flag_soft[valid].mean()) if has_flags else 0.0,
        "p95_tok_weight_valid": float(np.percentile(tok_w[valid], 95.0)),
        "p95_parent_added_valid": float(np.percentile(parent_added[valid], 95.0)),
        "p95_eff_share_valid": float(np.percentile(eff_share[valid], 95.0)),
        "p95_merge_flag_soft_valid": float(np.percentile(merge_flag_soft[valid], 95.0)) if has_flags else 0.0,
        "p95_eff_flag_soft_valid": float(np.percentile(eff_flag_soft[valid], 95.0)) if has_flags else 0.0,
    }


class JetRegressionDataset(Dataset):
    def __init__(self, feat: np.ndarray, mask: np.ndarray, target_log: np.ndarray):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.target_log = torch.tensor(target_log, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat": self.feat[i],
            "mask": self.mask[i],
            "target_log": self.target_log[i],
        }


class JetLevelRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(embed_dim))
        self.head = nn.Sequential(
            nn.Linear(int(embed_dim), int(embed_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(embed_dim), int(output_dim)),
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(feat)
        x = self.encoder(x, src_key_padding_mask=~mask)
        w = mask.float().unsqueeze(-1)
        pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        pooled = self.norm(pooled)
        return self.head(pooled)


def _jet_mass_np(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    eps = 1e-8
    pt = const[..., 0]
    eta = const[..., 1]
    phi = const[..., 2]
    energy = const[..., 3]
    w = mask.astype(np.float32)
    px = (pt * np.cos(phi) * w).sum(axis=1)
    py = (pt * np.sin(phi) * w).sum(axis=1)
    pz = (pt * np.sinh(eta) * w).sum(axis=1)
    e = (energy * w).sum(axis=1)
    p2 = px * px + py * py + pz * pz
    m2 = np.maximum(e * e - p2, eps)
    return np.sqrt(m2).astype(np.float32)


def _tau_n(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, n_axes: int, r0: float) -> float:
    eps = 1e-8
    n = int(pt.shape[0])
    if n == 0:
        return 0.0
    n_axes = max(1, min(int(n_axes), n))
    # Use hardest constituents as simple deterministic axes.
    axis_idx = np.argsort(-pt)[:n_axes]
    a_eta = eta[axis_idx]
    a_phi = phi[axis_idx]
    deta = eta[:, None] - a_eta[None, :]
    dphi = np.arctan2(np.sin(phi[:, None] - a_phi[None, :]), np.cos(phi[:, None] - a_phi[None, :]))
    dr = np.sqrt(deta * deta + dphi * dphi)
    min_dr = dr.min(axis=1)
    d0 = np.sum(pt) * float(r0) + eps
    tau = float(np.sum(pt * min_dr) / d0)
    return tau


def _d2_topk(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, topk: int, beta: float) -> float:
    eps = 1e-8
    if pt.shape[0] < 3:
        return 0.0
    order = np.argsort(-pt)[: min(int(topk), int(pt.shape[0]))]
    pt_k = pt[order]
    eta_k = eta[order]
    phi_k = phi[order]
    m = int(pt_k.shape[0])
    if m < 3:
        return 0.0
    z = pt_k / (np.sum(pt_k) + eps)
    deta = eta_k[:, None] - eta_k[None, :]
    dphi = np.arctan2(np.sin(phi_k[:, None] - phi_k[None, :]), np.cos(phi_k[:, None] - phi_k[None, :]))
    dr = np.sqrt(deta * deta + dphi * dphi) + eps

    e2 = 0.0
    for i in range(m):
        zi = z[i]
        for j in range(i + 1, m):
            e2 += zi * z[j] * (dr[i, j] ** beta)

    e3 = 0.0
    for i in range(m):
        zi = z[i]
        for j in range(i + 1, m):
            zij = zi * z[j]
            dij = dr[i, j] ** beta
            for k in range(j + 1, m):
                e3 += zij * z[k] * dij * (dr[i, k] ** beta) * (dr[j, k] ** beta)
    d2 = float(e3 / ((e2 ** 3) + eps))
    return d2


def _compute_substructure_np(
    const: np.ndarray,
    mask: np.ndarray,
    tau_r0: float = 0.8,
    tau_topk: int = 24,
    d2_topk: int = 10,
    d2_beta: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_jets = int(const.shape[0])
    tau21 = np.zeros(n_jets, dtype=np.float32)
    tau32 = np.zeros(n_jets, dtype=np.float32)
    d2 = np.zeros(n_jets, dtype=np.float32)
    eps = 1e-8
    for j in range(n_jets):
        m = mask[j]
        if not np.any(m):
            continue
        c = const[j, m]
        order = np.argsort(-c[:, 0])[: min(int(tau_topk), c.shape[0])]
        c = c[order]
        pt = c[:, 0].astype(np.float64)
        eta = c[:, 1].astype(np.float64)
        phi = c[:, 2].astype(np.float64)
        t1 = _tau_n(pt, eta, phi, 1, tau_r0)
        t2 = _tau_n(pt, eta, phi, 2, tau_r0)
        t3 = _tau_n(pt, eta, phi, 3, tau_r0)
        tau21[j] = np.float32(t2 / (t1 + eps))
        tau32[j] = np.float32(t3 / (t2 + eps))
        d2[j] = np.float32(_d2_topk(pt, eta, phi, topk=int(d2_topk), beta=float(d2_beta)))
    tau21 = np.nan_to_num(tau21, nan=0.0, posinf=0.0, neginf=0.0)
    tau32 = np.nan_to_num(tau32, nan=0.0, posinf=0.0, neginf=0.0)
    d2 = np.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0)
    tau21 = np.clip(tau21, 0.0, 5.0)
    tau32 = np.clip(tau32, 0.0, 5.0)
    d2 = np.clip(d2, 0.0, 1e3)
    return tau21, tau32, d2


def compute_jet_regression_targets(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Returns transformed target vectors:
      [log_pt, log_E, log_m, tau21, tau32, log1p_d2, log1p_n_off, log1p_n_added]
    for offline target and HLT reference (last channel set to 0 for HLT side).
    """
    eps = 1e-8
    idx = {
        "log_pt": 0,
        "log_e": 1,
        "log_m": 2,
        "tau21": 3,
        "tau32": 4,
        "log1p_d2": 5,
        "log1p_n_off": 6,
        "log1p_n_added": 7,
    }
    n_jets = int(const_off.shape[0])
    off = np.zeros((n_jets, 8), dtype=np.float32)
    hlt = np.zeros((n_jets, 8), dtype=np.float32)

    pt_off = compute_jet_pt(const_off, mask_off).astype(np.float32)
    pt_hlt = compute_jet_pt(const_hlt, mask_hlt).astype(np.float32)
    e_off = (const_off[:, :, 3] * mask_off.astype(np.float32)).sum(axis=1).astype(np.float32)
    e_hlt = (const_hlt[:, :, 3] * mask_hlt.astype(np.float32)).sum(axis=1).astype(np.float32)
    m_off = _jet_mass_np(const_off, mask_off)
    m_hlt = _jet_mass_np(const_hlt, mask_hlt)
    tau21_off, tau32_off, d2_off = _compute_substructure_np(const_off, mask_off)
    tau21_hlt, tau32_hlt, d2_hlt = _compute_substructure_np(const_hlt, mask_hlt)
    n_off = mask_off.sum(axis=1).astype(np.float32)
    n_hlt = mask_hlt.sum(axis=1).astype(np.float32)
    n_added = np.maximum(n_off - n_hlt, 0.0).astype(np.float32)

    off[:, idx["log_pt"]] = np.log(np.clip(pt_off, eps, None))
    off[:, idx["log_e"]] = np.log(np.clip(e_off, eps, None))
    off[:, idx["log_m"]] = np.log(np.clip(m_off, eps, None))
    off[:, idx["tau21"]] = tau21_off
    off[:, idx["tau32"]] = tau32_off
    off[:, idx["log1p_d2"]] = np.log1p(np.clip(d2_off, 0.0, None))
    off[:, idx["log1p_n_off"]] = np.log1p(np.clip(n_off, 0.0, None))
    off[:, idx["log1p_n_added"]] = np.log1p(np.clip(n_added, 0.0, None))

    hlt[:, idx["log_pt"]] = np.log(np.clip(pt_hlt, eps, None))
    hlt[:, idx["log_e"]] = np.log(np.clip(e_hlt, eps, None))
    hlt[:, idx["log_m"]] = np.log(np.clip(m_hlt, eps, None))
    hlt[:, idx["tau21"]] = tau21_hlt
    hlt[:, idx["tau32"]] = tau32_hlt
    hlt[:, idx["log1p_d2"]] = np.log1p(np.clip(d2_hlt, 0.0, None))
    hlt[:, idx["log1p_n_off"]] = np.log1p(np.clip(n_hlt, 0.0, None))
    hlt[:, idx["log1p_n_added"]] = 0.0

    off = np.nan_to_num(off, nan=0.0, posinf=0.0, neginf=0.0)
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    return off.astype(np.float32), hlt.astype(np.float32), idx


def _jet_reg_metric_dict(pred: np.ndarray, true: np.ndarray, idx: Dict[str, int]) -> Dict[str, float]:
    m: Dict[str, float] = {}
    m["mae_log_pt"] = float(np.mean(np.abs(pred[:, idx["log_pt"]] - true[:, idx["log_pt"]])))
    m["mae_log_e"] = float(np.mean(np.abs(pred[:, idx["log_e"]] - true[:, idx["log_e"]])))
    m["mae_log_m"] = float(np.mean(np.abs(pred[:, idx["log_m"]] - true[:, idx["log_m"]])))
    m["mae_tau21"] = float(np.mean(np.abs(pred[:, idx["tau21"]] - true[:, idx["tau21"]])))
    m["mae_tau32"] = float(np.mean(np.abs(pred[:, idx["tau32"]] - true[:, idx["tau32"]])))
    m["mae_log1p_d2"] = float(np.mean(np.abs(pred[:, idx["log1p_d2"]] - true[:, idx["log1p_d2"]])))
    m["mae_log1p_n_off"] = float(np.mean(np.abs(pred[:, idx["log1p_n_off"]] - true[:, idx["log1p_n_off"]])))
    m["mae_log1p_n_added"] = float(np.mean(np.abs(pred[:, idx["log1p_n_added"]] - true[:, idx["log1p_n_added"]])))

    m["mae_pt"] = float(
        np.mean(np.abs(np.exp(pred[:, idx["log_pt"]]) - np.exp(true[:, idx["log_pt"]])))
    )
    m["mae_e"] = float(
        np.mean(np.abs(np.exp(pred[:, idx["log_e"]]) - np.exp(true[:, idx["log_e"]])))
    )
    m["mae_m"] = float(
        np.mean(np.abs(np.exp(pred[:, idx["log_m"]]) - np.exp(true[:, idx["log_m"]])))
    )
    m["mae_d2"] = float(
        np.mean(np.abs(np.expm1(pred[:, idx["log1p_d2"]]) - np.expm1(true[:, idx["log1p_d2"]])))
    )
    m["mae_n_off"] = float(
        np.mean(np.abs(np.expm1(pred[:, idx["log1p_n_off"]]) - np.expm1(true[:, idx["log1p_n_off"]])))
    )
    m["mae_n_added"] = float(
        np.mean(np.abs(np.expm1(pred[:, idx["log1p_n_added"]]) - np.expm1(true[:, idx["log1p_n_added"]])))
    )
    return m


def train_jet_regressor(
    model: JetLevelRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
) -> Tuple[JetLevelRegressor, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))
    best_state = None
    best_val = float("inf")
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for ep in tqdm(range(int(epochs)), desc="JetRegressor"):
        model.train()
        tr = 0.0
        n_tr = 0
        for batch in train_loader:
            feat = batch["feat"].to(device)
            mask = batch["mask"].to(device)
            tgt = batch["target_log"].to(device)
            opt.zero_grad()
            pred = model(feat, mask)
            loss = F.smooth_l1_loss(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = feat.size(0)
            tr += loss.item() * bs
            n_tr += bs
        sch.step()
        tr /= max(n_tr, 1)

        model.eval()
        va = 0.0
        n_va = 0
        all_pred = []
        all_tgt = []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["feat"].to(device)
                mask = batch["mask"].to(device)
                tgt = batch["target_log"].to(device)
                pred = model(feat, mask)
                loss = F.smooth_l1_loss(pred, tgt)
                bs = feat.size(0)
                va += loss.item() * bs
                n_va += bs
                all_pred.append(pred.detach().cpu().numpy())
                all_tgt.append(tgt.detach().cpu().numpy())
        va /= max(n_va, 1)
        pred_val = np.concatenate(all_pred, axis=0) if len(all_pred) > 0 else np.zeros((0, 8), dtype=np.float32)
        tgt_val = np.concatenate(all_tgt, axis=0) if len(all_tgt) > 0 else np.zeros((0, 8), dtype=np.float32)
        # default indices for compact reporting (log_pt/log_e are 0/1 in target layout)
        mae_pt_val = float(np.mean(np.abs(np.exp(pred_val[:, 0]) - np.exp(tgt_val[:, 0])))) if pred_val.size else float("nan")
        mae_e_val = float(np.mean(np.abs(np.exp(pred_val[:, 1]) - np.exp(tgt_val[:, 1])))) if pred_val.size else float("nan")

        if va < best_val:
            best_val = float(va)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "best_val_loss": float(va),
                "best_val_mae_pt": float(mae_pt_val),
                "best_val_mae_e": float(mae_e_val),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"JetReg ep {ep+1}: train_loss={tr:.5f}, val_loss={va:.5f}, "
                f"val_mae_pt={mae_pt_val:.3f}, val_mae_e={mae_e_val:.3f}, best={best_val:.5f}"
            )
        if no_improve >= int(patience):
            print(f"Early stopping JetRegressor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


@torch.no_grad()
def predict_jet_regressor(
    model: JetLevelRegressor,
    feat: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out_dim = int(model.head[-1].out_features)
    out = np.zeros((feat.shape[0], out_dim), dtype=np.float32)
    for start in range(0, feat.shape[0], int(batch_size)):
        end = min(start + int(batch_size), feat.shape[0])
        x = torch.tensor(feat[start:end], dtype=torch.float32, device=device)
        m = torch.tensor(mask[start:end], dtype=torch.bool, device=device)
        pred = model(x, m)
        out[start:end] = pred.detach().cpu().numpy().astype(np.float32)
    return out


def stage_scale_local(epoch: int, cfg: Dict) -> float:
    s1 = int(cfg.get("stage1_epochs", 0))
    s2 = int(cfg.get("stage2_epochs", 0))
    if epoch < s1:
        return 0.35
    if epoch < s2:
        return 0.70
    return 1.0


def _standardize_features_torch(
    feat: torch.Tensor,
    mask: torch.Tensor,
    means_t: torch.Tensor,
    stds_t: torch.Tensor,
) -> torch.Tensor:
    feat_std = (feat - means_t.view(1, 1, -1)) / stds_t.view(1, 1, -1)
    feat_std = torch.nan_to_num(feat_std, nan=0.0, posinf=0.0, neginf=0.0)
    feat_std = feat_std * mask.unsqueeze(-1).float()
    return feat_std


def _build_teacher_reco_features_from_output(
    reco_out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    weight_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reco_out = enforce_unsmear_only_output(reco_out)
    L = int(const_hlt.shape[1])
    tok_tokens = reco_out["cand_tokens"][:, :L, :]
    tok_w = reco_out["cand_weights"][:, :L].clamp(0.0, 1.0)

    # Require both non-trivial token weight and HLT-valid token support.
    mask_b = (tok_w > float(weight_floor)) & mask_hlt
    none_valid = ~mask_b.any(dim=1)
    if none_valid.any():
        # Fall back to HLT mask when reco mask is empty.
        mask_b = torch.where(mask_hlt, mask_hlt, mask_b)
        none_valid = ~mask_b.any(dim=1)
        if none_valid.any():
            mask_b = mask_b.clone()
            mask_b[none_valid, 0] = True

    feat7 = compute_features_torch(tok_tokens, mask_b)
    return feat7, mask_b


def _sorted_edit_budget_vec(
    reco_tokens: torch.Tensor,
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
) -> torch.Tensor:
    # Approximate permutation-robust edit magnitude by sorting both sets by pT.
    # This avoids direct index-aligned penalties that are unstable for set outputs.
    very_low = -1e9
    pt_pred = torch.where(mask_hlt, reco_tokens[..., 0], torch.full_like(reco_tokens[..., 0], very_low))
    pt_hlt = torch.where(mask_hlt, const_hlt[..., 0], torch.full_like(const_hlt[..., 0], very_low))

    idx_pred = torch.argsort(pt_pred, dim=1, descending=True)
    idx_hlt = torch.argsort(pt_hlt, dim=1, descending=True)

    gather4_pred = idx_pred.unsqueeze(-1).expand(-1, -1, reco_tokens.shape[-1])
    gather4_hlt = idx_hlt.unsqueeze(-1).expand(-1, -1, const_hlt.shape[-1])

    pred_sorted = torch.gather(reco_tokens, 1, gather4_pred)
    hlt_sorted = torch.gather(const_hlt, 1, gather4_hlt)
    mask_sorted = torch.gather(mask_hlt, 1, idx_hlt)

    abs_diff_tok = (pred_sorted - hlt_sorted).abs().mean(dim=-1)
    denom = mask_sorted.float().sum(dim=1).clamp(min=1.0)
    mean_edit = (abs_diff_tok * mask_sorted.float()).sum(dim=1) / denom
    return mean_edit


def train_reconstructor_weighted(
    model: OfflineReconstructor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
    apply_reco_weight: bool,
    teacher_model: nn.Module,
    feat_means: np.ndarray,
    feat_stds: np.ndarray,
    kd_temperature: float,
    lambda_kd: float,
    lambda_phys: float,
    lambda_budget_hinge: float,
    budget_eps: float,
    budget_weight_floor: float,
    target_tpr_for_fpr: float,
) -> Tuple[OfflineReconstructor, Dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    kd_temperature = max(float(kd_temperature), 1e-3)
    lambda_kd = float(max(lambda_kd, 0.0))
    lambda_phys = float(max(lambda_phys, 0.0))
    lambda_budget_hinge = float(max(lambda_budget_hinge, 0.0))
    budget_eps = float(max(budget_eps, 0.0))
    budget_weight_floor = float(max(budget_weight_floor, 0.0))

    means_t = torch.tensor(feat_means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(np.clip(feat_stds, 1e-6, None), dtype=torch.float32, device=device)

    teacher_model.eval()
    for p_t in teacher_model.parameters():
        p_t.requires_grad_(False)

    best_state = None
    best_val_auc = float("-inf")
    no_improve = 0
    best_metrics: Dict[str, float] = {}
    min_stop_epoch = int(train_cfg.get("stage2_epochs", 0)) + int(train_cfg.get("min_full_scale_epochs", 5))

    for ep in tqdm(range(int(train_cfg["epochs"])), desc="Reconstructor"):
        model.train()
        sc = stage_scale_local(ep, train_cfg)

        tr_total = tr_kd = tr_phys = tr_budget_hinge = tr_teacher_auc = tr_teacher_fpr50 = 0.0
        n_tr = 0
        tr_probs_all = []
        tr_labels_all = []

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            budget_merge_true = batch["budget_merge_true"].to(device)
            budget_eff_true = batch["budget_eff_true"].to(device)
            labels_batch = batch["label"].to(device)
            sw_reco = batch.get("sample_weight_reco", None)
            if sw_reco is not None:
                sw_reco = sw_reco.to(device)
            sw_for_loss = sw_reco if (bool(apply_reco_weight) and sw_reco is not None) else None

            opt.zero_grad()
            out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=sc)

            # Keep physics regularization from existing recon losses, but remove chamfer-driven total usage.
            if sw_for_loss is not None:
                aux_losses = compute_reconstruction_losses_weighted(
                    out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    budget_merge_true,
                    budget_eff_true,
                    loss_cfg,
                    sample_weight=sw_for_loss,
                )
            else:
                aux_losses = compute_reconstruction_losses(
                    out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    budget_merge_true,
                    budget_eff_true,
                    loss_cfg,
                )
            loss_phys = aux_losses["phys"]

            with torch.no_grad():
                feat_off_raw = compute_features_torch(const_off, mask_off)
                feat_off_std = _standardize_features_torch(feat_off_raw, mask_off, means_t, stds_t)
                logits_teacher_off = teacher_model(feat_off_std, mask_off).view(-1)

            feat_reco_raw, mask_reco = _build_teacher_reco_features_from_output(
                out,
                const_hlt,
                mask_hlt,
                weight_floor=budget_weight_floor,
            )
            feat_reco_std = _standardize_features_torch(feat_reco_raw, mask_reco, means_t, stds_t)
            logits_teacher_reco = teacher_model(feat_reco_std, mask_reco).view(-1)

            target_soft = torch.sigmoid(logits_teacher_off / kd_temperature)
            kd_vec = (
                F.binary_cross_entropy_with_logits(
                    logits_teacher_reco / kd_temperature,
                    target_soft,
                    reduction="none",
                )
                * (kd_temperature * kd_temperature)
            )
            loss_kd = _weighted_batch_mean(kd_vec, sw_for_loss)

            reco_tokens = out["cand_tokens"][:, : const_hlt.shape[1], :]
            mean_edit_vec = _sorted_edit_budget_vec(reco_tokens, const_hlt, mask_hlt)
            budget_hinge_vec = F.relu(mean_edit_vec - budget_eps)
            loss_budget_hinge = _weighted_batch_mean(budget_hinge_vec, sw_for_loss)

            loss_total = (
                lambda_kd * loss_kd
                + lambda_phys * loss_phys
                + lambda_budget_hinge * loss_budget_hinge
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            probs_reco = torch.sigmoid(logits_teacher_reco).detach().cpu().numpy()
            tr_probs_all.append(probs_reco)
            tr_labels_all.append(labels_batch.detach().cpu().numpy().astype(np.int64))

            bs = feat_hlt.size(0)
            tr_total += loss_total.item() * bs
            tr_kd += loss_kd.item() * bs
            tr_phys += loss_phys.item() * bs
            tr_budget_hinge += loss_budget_hinge.item() * bs
            n_tr += bs

        model.eval()
        va_total = va_kd = va_phys = va_budget_hinge = 0.0
        n_va = 0
        va_probs_all = []
        va_labels_all = []

        with torch.no_grad():
            for batch in val_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                const_off = batch["const_off"].to(device)
                mask_off = batch["mask_off"].to(device)
                budget_merge_true = batch["budget_merge_true"].to(device)
                budget_eff_true = batch["budget_eff_true"].to(device)
                labels_batch = batch["label"].to(device)
                sw_reco = batch.get("sample_weight_reco", None)
                if sw_reco is not None:
                    sw_reco = sw_reco.to(device)
                sw_for_loss = sw_reco if (bool(apply_reco_weight) and sw_reco is not None) else None

                out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)

                if sw_for_loss is not None:
                    aux_losses = compute_reconstruction_losses_weighted(
                        out,
                        const_hlt,
                        mask_hlt,
                        const_off,
                        mask_off,
                        budget_merge_true,
                        budget_eff_true,
                        loss_cfg,
                        sample_weight=sw_for_loss,
                    )
                else:
                    aux_losses = compute_reconstruction_losses(
                        out,
                        const_hlt,
                        mask_hlt,
                        const_off,
                        mask_off,
                        budget_merge_true,
                        budget_eff_true,
                        loss_cfg,
                    )
                loss_phys = aux_losses["phys"]

                feat_off_raw = compute_features_torch(const_off, mask_off)
                feat_off_std = _standardize_features_torch(feat_off_raw, mask_off, means_t, stds_t)
                logits_teacher_off = teacher_model(feat_off_std, mask_off).view(-1)

                feat_reco_raw, mask_reco = _build_teacher_reco_features_from_output(
                    out,
                    const_hlt,
                    mask_hlt,
                    weight_floor=budget_weight_floor,
                )
                feat_reco_std = _standardize_features_torch(feat_reco_raw, mask_reco, means_t, stds_t)
                logits_teacher_reco = teacher_model(feat_reco_std, mask_reco).view(-1)

                target_soft = torch.sigmoid(logits_teacher_off / kd_temperature)
                kd_vec = (
                    F.binary_cross_entropy_with_logits(
                        logits_teacher_reco / kd_temperature,
                        target_soft,
                        reduction="none",
                    )
                    * (kd_temperature * kd_temperature)
                )
                loss_kd = _weighted_batch_mean(kd_vec, sw_for_loss)

                reco_tokens = out["cand_tokens"][:, : const_hlt.shape[1], :]
                mean_edit_vec = _sorted_edit_budget_vec(reco_tokens, const_hlt, mask_hlt)
                budget_hinge_vec = F.relu(mean_edit_vec - budget_eps)
                loss_budget_hinge = _weighted_batch_mean(budget_hinge_vec, sw_for_loss)

                loss_total = (
                    lambda_kd * loss_kd
                    + lambda_phys * loss_phys
                    + lambda_budget_hinge * loss_budget_hinge
                )

                probs_reco = torch.sigmoid(logits_teacher_reco).detach().cpu().numpy()
                va_probs_all.append(probs_reco)
                va_labels_all.append(labels_batch.detach().cpu().numpy().astype(np.int64))

                bs = feat_hlt.size(0)
                va_total += loss_total.item() * bs
                va_kd += loss_kd.item() * bs
                va_phys += loss_phys.item() * bs
                va_budget_hinge += loss_budget_hinge.item() * bs
                n_va += bs

        sch.step()

        tr_total /= max(n_tr, 1)
        tr_kd /= max(n_tr, 1)
        tr_phys /= max(n_tr, 1)
        tr_budget_hinge /= max(n_tr, 1)

        va_total /= max(n_va, 1)
        va_kd /= max(n_va, 1)
        va_phys /= max(n_va, 1)
        va_budget_hinge /= max(n_va, 1)

        tr_probs = np.concatenate(tr_probs_all, axis=0) if len(tr_probs_all) > 0 else np.zeros((0,), dtype=np.float32)
        tr_labels = np.concatenate(tr_labels_all, axis=0) if len(tr_labels_all) > 0 else np.zeros((0,), dtype=np.int64)
        va_probs = np.concatenate(va_probs_all, axis=0) if len(va_probs_all) > 0 else np.zeros((0,), dtype=np.float32)
        va_labels = np.concatenate(va_labels_all, axis=0) if len(va_labels_all) > 0 else np.zeros((0,), dtype=np.int64)

        if np.unique(tr_labels).size > 1 and tr_probs.size > 0:
            tr_auc = float(roc_auc_score(tr_labels, tr_probs))
            tr_fpr, tr_tpr, _ = roc_curve(tr_labels, tr_probs)
            tr_fpr50 = float(fpr_at_target_tpr(tr_fpr, tr_tpr, float(target_tpr_for_fpr)))
        else:
            tr_auc, tr_fpr50 = float("nan"), float("nan")

        if np.unique(va_labels).size > 1 and va_probs.size > 0:
            va_auc = float(roc_auc_score(va_labels, va_probs))
            va_fpr, va_tpr, _ = roc_curve(va_labels, va_probs)
            va_fpr50 = float(fpr_at_target_tpr(va_fpr, va_tpr, float(target_tpr_for_fpr)))
        else:
            va_auc, va_fpr50 = float("nan"), float("nan")

        if np.isfinite(va_auc) and (va_auc > best_val_auc):
            best_val_auc = float(va_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_metrics = {
                "selected_metric": "teacher_on_reco_val_auc",
                "selected_val_auc": float(va_auc),
                "selected_val_fpr50": float(va_fpr50),
                "selected_val_total_loss": float(va_total),
                "val_total": float(va_total),
                "val_kd": float(va_kd),
                "val_phys": float(va_phys),
                "val_budget_hinge": float(va_budget_hinge),
                "train_total": float(tr_total),
                "train_kd": float(tr_kd),
                "train_phys": float(tr_phys),
                "train_budget_hinge": float(tr_budget_hinge),
                "train_teacher_auc": float(tr_auc),
                "train_teacher_fpr50": float(tr_fpr50),
                "val_teacher_auc": float(va_auc),
                "val_teacher_fpr50": float(va_fpr50),
            }
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Ep {ep+1}: train_total={tr_total:.4f}, val_total={va_total:.4f}, "
                f"train_teacher_auc={tr_auc:.4f}, val_teacher_auc={va_auc:.4f}, "
                f"val_teacher_fpr50={va_fpr50:.6f}, "
                f"best_teacher_auc={best_val_auc:.4f} | "
                f"kd={va_kd:.4f}, phys={va_phys:.4f}, budget_hinge={va_budget_hinge:.4f}, "
                f"stage_scale={sc:.2f}"
            )
        if (ep + 1) >= min_stop_epoch and no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping reconstructor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def train_joint_dual(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    freeze_reconstructor: bool,
    epochs: int,
    patience: int,
    lr_dual: float,
    lr_reco: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_reco: float,
    lambda_rank: float,
    lambda_cons: float,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    min_epochs: int,
    select_metric: str = "auc",
    apply_cls_weight: bool = False,
    apply_reco_weight: bool = False,
    val_weight_key: Optional[str] = None,
    use_weighted_val_selection: bool = False,
) -> Tuple[OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    for p in reconstructor.parameters():
        p.requires_grad = not freeze_reconstructor

    params = [{"params": dual_model.parameters(), "lr": float(lr_dual)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_dual), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_state_dual_sel = None
    best_state_reco_sel = None
    best_state_dual_auc = None
    best_state_reco_auc = None
    best_state_dual_fpr = None
    best_state_reco_fpr = None

    best_val_fpr50 = float("inf")  # best observed across epochs (selection source)
    best_val_auc = float("-inf")   # best observed across epochs (selection source)
    best_val_fpr50_unw = float("inf")
    best_val_auc_unw = float("-inf")
    best_val_fpr50_w = float("inf")
    best_val_auc_w = float("-inf")
    best_sel_score = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    sel_val_fpr50 = float("nan")
    sel_val_auc = float("nan")
    sel_val_fpr50_unw = float("nan")
    sel_val_auc_unw = float("nan")
    sel_val_fpr50_w = float("nan")
    sel_val_auc_w = float("nan")
    val_metric_source = "weighted" if bool(use_weighted_val_selection) else "unweighted"
    no_improve = 0

    for ep in tqdm(range(int(epochs)), desc=stage_name):
        dual_model.train()
        if freeze_reconstructor:
            reconstructor.eval()
        else:
            reconstructor.train()

        tr_loss = 0.0
        tr_cls = 0.0
        tr_rank = 0.0
        tr_reco = 0.0
        tr_cons = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt_reco = batch["feat_hlt_reco"].to(device)
            feat_hlt_dual = batch["feat_hlt_dual"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            b_merge = batch["budget_merge_true"].to(device)
            b_eff = batch["budget_eff_true"].to(device)
            y = batch["label"].to(device)
            sw_cls = batch.get("sample_weight_cls", None)
            sw_reco = batch.get("sample_weight_reco", None)
            if sw_cls is not None:
                sw_cls = sw_cls.to(device)
            if sw_reco is not None:
                sw_reco = sw_reco.to(device)

            opt.zero_grad()

            if freeze_reconstructor:
                with torch.no_grad():
                    reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            else:
                reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)

            feat_b, mask_b = build_soft_corrected_view(
                reco_out,
                weight_floor=corrected_weight_floor,
                scale_features_by_weight=True,
                include_flags=corrected_use_flags,
            )
            logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)

            if bool(apply_cls_weight) and sw_cls is not None:
                loss_cls_raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                denom = sw_cls.sum().clamp(min=1e-6)
                loss_cls = (loss_cls_raw * sw_cls).sum() / denom
            else:
                loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()

            if float(lambda_reco) > 0.0:
                if bool(apply_reco_weight) and sw_reco is not None:
                    reco_losses = compute_reconstruction_losses_weighted(
                        reco_out,
                        const_hlt,
                        mask_hlt,
                        const_off,
                        mask_off,
                        b_merge,
                        b_eff,
                        BASE_CONFIG["loss"],
                        sample_weight=sw_reco,
                    )
                else:
                    reco_losses = compute_reconstruction_losses(
                        reco_out,
                        const_hlt,
                        mask_hlt,
                        const_off,
                        mask_off,
                        b_merge,
                        b_eff,
                        BASE_CONFIG["loss"],
                    )
                loss_reco = reco_losses["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            loss = (
                loss_cls
                + float(lambda_rank) * loss_rank
                + float(lambda_reco) * loss_reco
                + float(lambda_cons) * loss_cons
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual_model.parameters(), 1.0)
            if not freeze_reconstructor:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = feat_hlt_reco.size(0)
            tr_loss += loss.item() * bs
            tr_cls += loss_cls.item() * bs
            tr_rank += loss_rank.item() * bs
            tr_reco += loss_reco.item() * bs
            tr_cons += loss_cons.item() * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)

        va_pack = eval_joint_model_both_metrics(
            reconstructor=reconstructor,
            dual_model=dual_model,
            loader=val_loader,
            device=device,
            corrected_weight_floor=corrected_weight_floor,
            corrected_use_flags=corrected_use_flags,
            weighted_key=val_weight_key if bool(use_weighted_val_selection) else None,
        )
        va_auc_unw = float(va_pack["auc_unweighted"])
        va_fpr50_unw = float(va_pack["fpr50_unweighted"])
        va_auc_w = float(va_pack["auc_weighted"])
        va_fpr50_w = float(va_pack["fpr50_weighted"])
        has_weighted_val = bool(use_weighted_val_selection) and np.isfinite(va_auc_w) and np.isfinite(va_fpr50_w)
        va_auc = float(va_auc_w) if has_weighted_val else float(va_auc_unw)
        va_fpr50 = float(va_fpr50_w) if has_weighted_val else float(va_fpr50_unw)
        metric_source_epoch = "weighted" if has_weighted_val else "unweighted"

        if np.isfinite(va_fpr50_unw) and float(va_fpr50_unw) < best_val_fpr50_unw:
            best_val_fpr50_unw = float(va_fpr50_unw)
        if np.isfinite(va_auc_unw) and float(va_auc_unw) > best_val_auc_unw:
            best_val_auc_unw = float(va_auc_unw)
        if np.isfinite(va_fpr50_w) and float(va_fpr50_w) < best_val_fpr50_w:
            best_val_fpr50_w = float(va_fpr50_w)
        if np.isfinite(va_auc_w) and float(va_auc_w) > best_val_auc_w:
            best_val_auc_w = float(va_auc_w)

        # Track best by each metric.
        if np.isfinite(va_fpr50) and float(va_fpr50) < best_val_fpr50:
            best_val_fpr50 = float(va_fpr50)
            best_state_dual_fpr = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco_fpr = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state_dual_auc = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco_auc = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}

        if str(select_metric).lower() == "auc":
            improved = np.isfinite(va_auc) and (float(va_auc) > best_sel_score)
            current_score = float(va_auc) if np.isfinite(va_auc) else float("-inf")
        else:
            improved = np.isfinite(va_fpr50) and (float(va_fpr50) < best_sel_score)
            current_score = float(va_fpr50) if np.isfinite(va_fpr50) else float("inf")

        if improved:
            best_sel_score = current_score
            sel_val_fpr50 = float(va_fpr50)
            sel_val_auc = float(va_auc)
            sel_val_fpr50_unw = float(va_fpr50_unw)
            sel_val_auc_unw = float(va_auc_unw)
            sel_val_fpr50_w = float(va_fpr50_w)
            sel_val_auc_w = float(va_auc_w)
            val_metric_source = str(metric_source_epoch)
            best_state_dual_sel = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco_sel = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # Print every epoch for Stage C variants; keep every 5 for earlier stages.
        print_every = 1 if str(stage_name).startswith("StageC") else 5
        if (ep + 1) % print_every == 0:
            print(
                f"{stage_name} ep {ep+1}: train_loss={tr_loss:.4f} "
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}) | "
                f"val_auc_unw={va_auc_unw:.4f}, val_fpr50_unw={va_fpr50_unw:.6f}, "
                f"val_auc_w={va_auc_w:.4f}, val_fpr50_w={va_fpr50_w:.6f}, "
                f"val_metric_source={metric_source_epoch}, "
                f"select={str(select_metric).lower()}, best_sel={best_sel_score:.6f}"
            )

        if (ep + 1) >= int(min_epochs) and no_improve >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_dual_sel is not None:
        dual_model.load_state_dict(best_state_dual_sel)
    if best_state_reco_sel is not None:
        reconstructor.load_state_dict(best_state_reco_sel)

    metrics = {
        "val_metric_source": str(val_metric_source),
        "selection_metric": str(select_metric).lower(),
        "selected_val_fpr50": float(sel_val_fpr50),
        "selected_val_auc": float(sel_val_auc),
        "selected_val_fpr50_unweighted": float(sel_val_fpr50_unw),
        "selected_val_auc_unweighted": float(sel_val_auc_unw),
        "selected_val_fpr50_weighted": float(sel_val_fpr50_w),
        "selected_val_auc_weighted": float(sel_val_auc_w),
        "best_val_fpr50_seen": float(best_val_fpr50),
        "best_val_auc_seen": float(best_val_auc),
        "best_val_fpr50_seen_unweighted": float(best_val_fpr50_unw),
        "best_val_auc_seen_unweighted": float(best_val_auc_unw),
        "best_val_fpr50_seen_weighted": float(best_val_fpr50_w),
        "best_val_auc_seen_weighted": float(best_val_auc_w),
    }
    state_pack = {
        "selected": {"dual": best_state_dual_sel, "reco": best_state_reco_sel},
        "auc": {"dual": best_state_dual_auc, "reco": best_state_reco_auc},
        "fpr50": {"dual": best_state_dual_fpr, "reco": best_state_reco_fpr},
    }
    return reconstructor, dual_model, metrics, state_pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=860000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=100)
    parser.add_argument(
        "--n_train_split",
        type=int,
        default=-1,
        help="If >0 and paired with --n_val_split/--n_test_split, use exact split counts instead of 70/15/15.",
    )
    parser.add_argument("--n_val_split", type=int, default=-1)
    parser.add_argument("--n_test_split", type=int, default=-1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_joint"),
    )
    parser.add_argument("--run_name", type=str, default="joint_default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--skip_save_models", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    # HLT controls
    parser.add_argument("--merge_radius", type=float, default=BASE_CONFIG["hlt_effects"]["merge_radius"])
    parser.add_argument("--eff_plateau_barrel", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    parser.add_argument("--eff_plateau_endcap", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    parser.add_argument("--smear_a", type=float, default=BASE_CONFIG["hlt_effects"]["smear_a"])
    parser.add_argument("--smear_b", type=float, default=BASE_CONFIG["hlt_effects"]["smear_b"])
    parser.add_argument("--smear_c", type=float, default=BASE_CONFIG["hlt_effects"]["smear_c"])
    parser.add_argument("--smear_scale", type=float, default=1.0, help="Global multiplier for all smearing width parameters.")
    parser.add_argument("--smear_use_pt_gev", action="store_true", help="Interpret pT-dependent smearing formulas in GeV (internally rescales coefficients for MeV-like inputs).")
    parser.add_argument("--smear_core_scale", type=float, default=1.0, help="Global multiplier on relative pT/E smearing width.")
    parser.add_argument("--smear_angle_scale", type=float, default=1.0, help="Global multiplier on eta/phi smearing widths.")
    parser.add_argument("--smear_tail_base", type=float, default=-1.0, help="Override tail-base probability; negative keeps default.")
    parser.add_argument("--smear_tail_sigma_mult", type=float, default=1.0, help="Multiplier on non-Gaussian tail width terms.")
    parser.add_argument("--smear_tail_prob_max", type=float, default=-1.0, help="Override tail_prob_max; negative keeps default.")

    # Stage A (reconstructor pretrain)
    parser.add_argument("--stageA_epochs", type=int, default=90)
    parser.add_argument("--stageA_patience", type=int, default=18)
    parser.add_argument("--stageA_kd_temp", type=float, default=2.5)
    parser.add_argument("--stageA_lambda_kd", type=float, default=5.0)
    parser.add_argument("--stageA_lambda_phys", type=float, default=0.05)
    parser.add_argument("--stageA_lambda_budget_hinge", type=float, default=1.0)
    parser.add_argument("--stageA_budget_eps", type=float, default=0.015)
    parser.add_argument("--stageA_budget_weight_floor", type=float, default=1e-4)
    parser.add_argument("--stageA_target_tpr", type=float, default=0.50)

    # Stage B (tagger pretrain, reconstructor frozen)
    parser.add_argument("--stageB_epochs", type=int, default=45)
    parser.add_argument("--stageB_patience", type=int, default=12)
    parser.add_argument("--stageB_min_epochs", type=int, default=12)
    parser.add_argument("--stageB_lr_dual", type=float, default=4e-4)
    parser.add_argument("--stageB_lambda_rank", type=float, default=0.0)
    parser.add_argument("--stageB_lambda_cons", type=float, default=0.0)
    parser.add_argument(
        "--stageB_train_frac",
        type=float,
        default=1.0,
        help="Fraction of train split used in Stage B only (Stage A/C still use full train split).",
    )
    parser.add_argument(
        "--stageB_subset_seed",
        type=int,
        default=-1,
        help="Seed for Stage-B subset sampling; negative means reuse --seed.",
    )

    # Kept for CLI compatibility; this script always selects by val_auc.
    parser.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])

    # Stage C (joint finetune)
    parser.add_argument("--stageC_epochs", type=int, default=65)
    parser.add_argument("--stageC_patience", type=int, default=14)
    parser.add_argument("--stageC_min_epochs", type=int, default=25)
    parser.add_argument("--stageC_lr_dual", type=float, default=2e-4)
    parser.add_argument("--stageC_lr_reco", type=float, default=1e-4)
    parser.add_argument("--lambda_reco", type=float, default=0.35)
    # Stage C rank term is disabled in this variant.
    parser.add_argument("--lambda_rank", type=float, default=0.0)
    parser.add_argument("--lambda_cons", type=float, default=0.06)
    parser.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    parser.add_argument("--use_corrected_flags", action="store_true")
    parser.add_argument("--loss_w_pt_ratio", type=float, default=BASE_CONFIG["loss"]["w_pt_ratio"])
    parser.add_argument("--loss_w_e_ratio", type=float, default=BASE_CONFIG["loss"]["w_e_ratio"])
    parser.add_argument("--loss_w_budget", type=float, default=BASE_CONFIG["loss"]["w_budget"])
    parser.add_argument("--loss_w_sparse", type=float, default=BASE_CONFIG["loss"]["w_sparse"])
    parser.add_argument("--loss_w_local", type=float, default=BASE_CONFIG["loss"]["w_local"])
    parser.add_argument(
        "--added_target_scale",
        type=float,
        default=1.0,
        help="Scale in [0,1] for non-privileged true_added target: target=scale*(offline_count-hlt_count).",
    )
    parser.add_argument("--disc_weight_enable", action="store_true")
    parser.add_argument("--disc_weight_mode", type=str, default="smooth_delta", choices=["tail_disagreement", "smooth_delta"])
    parser.add_argument("--disc_target_tpr", type=float, default=0.50)
    parser.add_argument("--disc_tau", type=float, default=0.05)
    parser.add_argument("--disc_include_pos", action="store_true")
    parser.add_argument("--disc_pos_scale", type=float, default=0.25)
    parser.add_argument("--disc_no_mean_normalize", action="store_true")
    parser.add_argument("--disc_teacher_conf_min", type=float, default=0.65)
    parser.add_argument("--disc_correctness_tau", type=float, default=0.05)
    parser.add_argument("--disc_disable_teacher_hard_correct_gate", action="store_true")
    parser.add_argument("--disc_disable_teacher_conf_gate", action="store_true")
    parser.add_argument("--disc_disable_teacher_better_gate", action="store_true")
    parser.add_argument("--disc_reco_lambda", type=float, default=15.0)
    parser.add_argument("--disc_reco_max_mult", type=float, default=20.0)
    parser.add_argument("--disc_cls_lambda", type=float, default=2.0)
    parser.add_argument("--disc_cls_max_mult", type=float, default=3.0)
    parser.add_argument("--disc_apply_cls_stagec", action="store_true")

    # Reconstructor decode controls (used for diagnostics and KD set build).
    parser.add_argument("--reco_weight_threshold", type=float, default=0.03)
    parser.add_argument("--reco_disable_budget_topk", action="store_true")

    # Response/resolution diagnostics.
    parser.add_argument("--response_n_bins", type=int, default=8)
    parser.add_argument("--response_min_count", type=int, default=30)

    # Stage D (final KD with frozen reconstructor and fixed corrected view)
    parser.add_argument("--disable_final_kd", action="store_true")
    parser.add_argument("--stageD_kd_epochs", type=int, default=-1)
    parser.add_argument("--stageD_kd_patience", type=int, default=-1)
    parser.add_argument("--stageD_kd_lr", type=float, default=-1.0)

    # Optional frozen jet-level regressor -> additional dual-view input channels.
    parser.add_argument("--enable_jet_regressor", action="store_true")
    parser.add_argument("--jet_reg_epochs", type=int, default=40)
    parser.add_argument("--jet_reg_patience", type=int, default=10)
    parser.add_argument("--jet_reg_lr", type=float, default=3e-4)
    parser.add_argument("--jet_reg_weight_decay", type=float, default=1e-5)
    parser.add_argument("--jet_reg_warmup_epochs", type=int, default=3)
    parser.add_argument("--jet_reg_embed_dim", type=int, default=128)
    parser.add_argument("--jet_reg_num_heads", type=int, default=8)
    parser.add_argument("--jet_reg_num_layers", type=int, default=4)
    parser.add_argument("--jet_reg_ff_dim", type=int, default=512)
    parser.add_argument("--jet_reg_dropout", type=float, default=0.1)

    args = parser.parse_args()
    set_seed(int(args.seed))

    if bool(args.use_corrected_flags):
        print("Note: forcing --use_corrected_flags OFF for unsmear-only variant.")
    args.use_corrected_flags = False

    # This variant always selects checkpoints by validation AUC.
    selection_metric = "auc"
    if str(args.selection_metric).lower() != "auc":
        print(f"Note: overriding --selection_metric={args.selection_metric} to 'auc' in this script.")

    cfg = _deepcopy_config()

    # Start from user-provided base smearing knobs, then enforce smear-only behavior.
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)
    _configure_smear_only_hlt(cfg)
    _scale_smearing_widths(cfg, float(args.smear_scale))
    _apply_calibrated_smearing(
        cfg,
        use_pt_gev=bool(args.smear_use_pt_gev),
        core_scale=float(args.smear_core_scale),
        angle_scale=float(args.smear_angle_scale),
        tail_base=float(args.smear_tail_base),
        tail_sigma_mult=float(args.smear_tail_sigma_mult),
        tail_prob_max=float(args.smear_tail_prob_max),
    )

    # Strict Chamfer-only reconstruction objective.
    cfg["loss"]["w_set"] = 1.0
    cfg["loss"]["w_phys"] = 0.0
    cfg["loss"]["w_pt_ratio"] = 0.0
    cfg["loss"]["w_e_ratio"] = 0.0
    cfg["loss"]["w_budget"] = 0.0
    cfg["loss"]["w_sparse"] = 0.0
    cfg["loss"]["w_local"] = 0.0
    cfg["reconstructor_training"]["epochs"] = int(args.stageA_epochs)
    cfg["reconstructor_training"]["patience"] = int(args.stageA_patience)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Save dir: {save_root}")
    hdbg = cfg["hlt_effects"]
    print(
        "Smearing config: "
        f"use_pt_gev={bool(args.smear_use_pt_gev)}, "
        f"core_scale={float(args.smear_core_scale):.3f}, "
        f"angle_scale={float(args.smear_angle_scale):.3f}, "
        f"tail_base={float(hdbg['tail_base']):.4f}, "
        f"tail_prob_max={float(hdbg['tail_prob_max']):.4f}, "
        f"a={float(hdbg['smear_a']):.4f}, b={float(hdbg['smear_b']):.4f}, c={float(hdbg['smear_c']):.4f}"
    )

    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = args.offset_jets + args.n_train_jets
    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}"
        )

    const_raw = all_const_full[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )
    # Smear-only setup: count-preserving targets (no added/lost constituents by construction).
    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    added_target_scale = 0.0
    budget_merge_true_raw = np.zeros_like(true_added_raw, dtype=np.float32)
    budget_eff_true_raw = np.zeros_like(true_added_raw, dtype=np.float32)
    budget_merge_true = np.zeros_like(true_added_raw, dtype=np.float32)
    budget_eff_true = np.zeros_like(true_added_raw, dtype=np.float32)

    count_gap_abs = np.abs(true_count - hlt_count)
    print(
        "Smear-only target setup: "
        f"mean_abs_count_gap={float(count_gap_abs.mean()):.6f}, "
        f"max_abs_count_gap={float(count_gap_abs.max()):.6f}"
    )
    print("Computing features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    n_train_split = int(args.n_train_split)
    n_val_split = int(args.n_val_split)
    n_test_split = int(args.n_test_split)
    custom_split = (n_train_split > 0 and n_val_split > 0 and n_test_split > 0)

    idx = np.arange(len(labels))
    if custom_split:
        total_need = int(n_train_split + n_val_split + n_test_split)
        if total_need > len(idx):
            raise ValueError(
                f"Requested split counts exceed available jets: "
                f"{n_train_split}+{n_val_split}+{n_test_split} > {len(idx)}"
            )
        if total_need < len(idx):
            idx_use, _ = train_test_split(
                idx,
                train_size=total_need,
                random_state=int(args.seed),
                stratify=labels[idx],
            )
        else:
            idx_use = idx
        train_idx, rem_idx = train_test_split(
            idx_use,
            train_size=int(n_train_split),
            random_state=int(args.seed),
            stratify=labels[idx_use],
        )
        val_idx, test_idx = train_test_split(
            rem_idx,
            train_size=int(n_val_split),
            test_size=int(n_test_split),
            random_state=int(args.seed),
            stratify=labels[rem_idx],
        )
    else:
        train_idx, temp_idx = train_test_split(
            idx, test_size=0.30, random_state=int(args.seed), stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=int(args.seed), stratify=labels[temp_idx]
        )
    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} "
        f"(custom_counts={custom_split})"
    )

    stageB_train_frac = float(np.clip(args.stageB_train_frac, 0.0, 1.0))
    if stageB_train_frac <= 0.0:
        raise ValueError("--stageB_train_frac must be > 0.")
    stageB_subset_seed = int(args.seed) if int(args.stageB_subset_seed) < 0 else int(args.stageB_subset_seed)
    if stageB_train_frac < 0.999999:
        n_stageb = max(1, int(round(float(len(train_idx)) * stageB_train_frac)))
        rng_stageb = np.random.default_rng(stageB_subset_seed)
        stageB_train_idx = np.asarray(rng_stageb.choice(train_idx, size=n_stageb, replace=False), dtype=np.int64)
        stageB_train_idx.sort()
    else:
        stageB_train_idx = train_idx.astype(np.int64, copy=True)
    print(
        f"Stage-B train subset: N={len(stageB_train_idx)} / {len(train_idx)} "
        f"(frac={len(stageB_train_idx) / max(1, len(train_idx)):.3f}, seed={stageB_subset_seed})"
    )

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    # Persist exact data setup/splits so Stage-C-only reruns can faithfully reload.
    data_setup = {
        "train_path_arg": str(args.train_path),
        "train_files": [str(p.resolve()) for p in train_files],
        "n_train_jets": int(args.n_train_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "seed": int(args.seed),
        "split": (
            {
                "mode": "custom_counts",
                "n_train_split": int(len(train_idx)),
                "n_val_split": int(len(val_idx)),
                "n_test_split": int(len(test_idx)),
            }
            if custom_split
            else {"mode": "fractions", "train_frac": 0.70, "val_frac": 0.15, "test_frac": 0.15}
        ),
        "hlt_effects": cfg["hlt_effects"],
        "variant": "nopriv_unsmearonly",
        "added_target_scale": float(added_target_scale),
        "mean_true_added_raw": float(true_added_raw.mean()),
        "mean_target_added": float(budget_merge_true.mean()),
                    "smear_scale": float(args.smear_scale),
                    "smear_only_mode": True,
        "stageB_train_frac": float(stageB_train_frac),
        "stageB_subset_seed": int(stageB_subset_seed),
        "stageB_train_count": int(len(stageB_train_idx)),
        "discrepancy_weighting": {
            "enabled": bool(args.disc_weight_enable),
            "weight_mode": str(args.disc_weight_mode),
            "target_tpr": float(args.disc_target_tpr),
            "tau": float(args.disc_tau),
            "include_pos": bool(args.disc_include_pos),
            "pos_scale": float(args.disc_pos_scale),
            "no_mean_normalize": bool(args.disc_no_mean_normalize),
            "teacher_conf_min": float(args.disc_teacher_conf_min),
            "correctness_tau": float(args.disc_correctness_tau),
            "disable_teacher_hard_correct_gate": bool(args.disc_disable_teacher_hard_correct_gate),
            "disable_teacher_conf_gate": bool(args.disc_disable_teacher_conf_gate),
            "disable_teacher_better_gate": bool(args.disc_disable_teacher_better_gate),
            "reco_lambda": float(args.disc_reco_lambda),
            "reco_max_mult": float(args.disc_reco_max_mult),
            "cls_lambda": float(args.disc_cls_lambda),
            "cls_max_mult": float(args.disc_cls_max_mult),
            "apply_cls_stagec": bool(args.disc_apply_cls_stagec),
        },
    }
    with open(save_root / "data_setup.json", "w", encoding="utf-8") as f:
        json.dump(data_setup, f, indent=2)
    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        stageB_train_idx=stageB_train_idx.astype(np.int64),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
    )

    # Teacher / baseline
    print("\n" + "=" * 70)
    print("STEP 1: TEACHER + BASELINE")
    print("=" * 70)
    BS = int(cfg["training"]["batch_size"])

    ds_train_off = JetDataset(feat_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    ds_val_off = JetDataset(feat_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    ds_test_off = JetDataset(feat_off_std[test_idx], masks_off[test_idx], labels[test_idx])
    dl_train_off = DataLoader(ds_train_off, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_off = DataLoader(ds_val_off, batch_size=BS, shuffle=False)
    dl_test_off = DataLoader(ds_test_off, batch_size=BS, shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = train_single_view_classifier_auc(
        teacher, dl_train_off, dl_val_off, device, cfg["training"], name="Teacher"
    )
    auc_teacher, preds_teacher, labs = eval_classifier(teacher, dl_test_off, device)

    ds_train_hlt = JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx])
    ds_val_hlt = JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    ds_test_hlt = JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_train_hlt = DataLoader(ds_train_hlt, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=BS, shuffle=False)
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=BS, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = train_single_view_classifier_auc(
        baseline, dl_train_hlt, dl_val_hlt, device, cfg["training"], name="Baseline"
    )
    auc_baseline, preds_baseline, _ = eval_classifier(baseline, dl_test_hlt, device)

    # Discrepancy weighting vectors (optional), derived from teacher-vs-baseline train/val predictions.
    sample_weight_reco = np.ones((len(train_idx),), dtype=np.float32)
    sample_weight_cls = np.ones((len(train_idx),), dtype=np.float32)
    sample_weight_reco_val = np.ones((len(val_idx),), dtype=np.float32)
    sample_weight_cls_val = np.ones((len(val_idx),), dtype=np.float32)
    discrepancy_reco_summary: Dict[str, float] = {"enabled": False}
    discrepancy_cls_summary: Dict[str, float] = {"enabled": False}
    discrepancy_reco_val_summary: Dict[str, float] = {"enabled": False}
    discrepancy_cls_val_summary: Dict[str, float] = {"enabled": False}
    if bool(args.disc_weight_enable):
        p_teacher_train, y_teacher_train = predict_single_view_scores(
            teacher,
            feat_off_std[train_idx],
            masks_off[train_idx],
            labels[train_idx],
            batch_size=BS,
            num_workers=int(args.num_workers),
            device=device,
        )
        p_teacher_val, y_teacher_val = predict_single_view_scores(
            teacher,
            feat_off_std[val_idx],
            masks_off[val_idx],
            labels[val_idx],
            batch_size=BS,
            num_workers=int(args.num_workers),
            device=device,
        )
        p_baseline_train, y_baseline_train = predict_single_view_scores(
            baseline,
            feat_hlt_std[train_idx],
            hlt_mask[train_idx],
            labels[train_idx],
            batch_size=BS,
            num_workers=int(args.num_workers),
            device=device,
        )
        p_baseline_val, y_baseline_val = predict_single_view_scores(
            baseline,
            feat_hlt_std[val_idx],
            hlt_mask[val_idx],
            labels[val_idx],
            batch_size=BS,
            num_workers=int(args.num_workers),
            device=device,
        )
        if not np.array_equal(y_teacher_train.astype(np.int64), y_baseline_train.astype(np.int64)):
            raise RuntimeError("Teacher/baseline train label mismatch while building discrepancy weights.")
        if not np.array_equal(y_teacher_val.astype(np.int64), y_baseline_val.astype(np.int64)):
            raise RuntimeError("Teacher/baseline val label mismatch while building discrepancy weights.")

        sample_weight_reco, discrepancy_reco_summary = build_discrepancy_weights(
            y_train=y_teacher_train.astype(np.int64),
            p_teacher_train=p_teacher_train,
            p_baseline_train=p_baseline_train,
            p_teacher_val=p_teacher_val,
            p_baseline_val=p_baseline_val,
            y_val=y_teacher_val.astype(np.int64),
            target_tpr=float(args.disc_target_tpr),
            tau=float(args.disc_tau),
            lambda_disc=float(args.disc_reco_lambda),
            max_mult=float(args.disc_reco_max_mult),
            include_pos=bool(args.disc_include_pos),
            pos_scale=float(args.disc_pos_scale),
            normalize_mean_one=(not bool(args.disc_no_mean_normalize)),
            teacher_conf_min=float(args.disc_teacher_conf_min),
            correctness_tau=float(args.disc_correctness_tau),
            use_teacher_hard_correct_gate=(not bool(args.disc_disable_teacher_hard_correct_gate)),
            use_teacher_conf_gate=(not bool(args.disc_disable_teacher_conf_gate)),
            use_teacher_better_gate=(not bool(args.disc_disable_teacher_better_gate)),
            weight_mode=str(args.disc_weight_mode),
        )
        discrepancy_reco_summary["enabled"] = True

        sample_weight_cls, discrepancy_cls_summary = build_discrepancy_weights(
            y_train=y_teacher_train.astype(np.int64),
            p_teacher_train=p_teacher_train,
            p_baseline_train=p_baseline_train,
            p_teacher_val=p_teacher_val,
            p_baseline_val=p_baseline_val,
            y_val=y_teacher_val.astype(np.int64),
            target_tpr=float(args.disc_target_tpr),
            tau=float(args.disc_tau),
            lambda_disc=float(args.disc_cls_lambda),
            max_mult=float(args.disc_cls_max_mult),
            include_pos=bool(args.disc_include_pos),
            pos_scale=float(args.disc_pos_scale),
            normalize_mean_one=(not bool(args.disc_no_mean_normalize)),
            teacher_conf_min=float(args.disc_teacher_conf_min),
            correctness_tau=float(args.disc_correctness_tau),
            use_teacher_hard_correct_gate=(not bool(args.disc_disable_teacher_hard_correct_gate)),
            use_teacher_conf_gate=(not bool(args.disc_disable_teacher_conf_gate)),
            use_teacher_better_gate=(not bool(args.disc_disable_teacher_better_gate)),
            weight_mode=str(args.disc_weight_mode),
        )
        discrepancy_cls_summary["enabled"] = True

        # Build val split weights using the same rule family, for weighted validation model selection.
        sample_weight_reco_val, discrepancy_reco_val_summary = build_discrepancy_weights(
            y_train=y_teacher_val.astype(np.int64),
            p_teacher_train=p_teacher_val,
            p_baseline_train=p_baseline_val,
            p_teacher_val=p_teacher_val,
            p_baseline_val=p_baseline_val,
            y_val=y_teacher_val.astype(np.int64),
            target_tpr=float(args.disc_target_tpr),
            tau=float(args.disc_tau),
            lambda_disc=float(args.disc_reco_lambda),
            max_mult=float(args.disc_reco_max_mult),
            include_pos=bool(args.disc_include_pos),
            pos_scale=float(args.disc_pos_scale),
            normalize_mean_one=(not bool(args.disc_no_mean_normalize)),
            teacher_conf_min=float(args.disc_teacher_conf_min),
            correctness_tau=float(args.disc_correctness_tau),
            use_teacher_hard_correct_gate=(not bool(args.disc_disable_teacher_hard_correct_gate)),
            use_teacher_conf_gate=(not bool(args.disc_disable_teacher_conf_gate)),
            use_teacher_better_gate=(not bool(args.disc_disable_teacher_better_gate)),
            weight_mode=str(args.disc_weight_mode),
        )
        discrepancy_reco_val_summary["enabled"] = True

        sample_weight_cls_val, discrepancy_cls_val_summary = build_discrepancy_weights(
            y_train=y_teacher_val.astype(np.int64),
            p_teacher_train=p_teacher_val,
            p_baseline_train=p_baseline_val,
            p_teacher_val=p_teacher_val,
            p_baseline_val=p_baseline_val,
            y_val=y_teacher_val.astype(np.int64),
            target_tpr=float(args.disc_target_tpr),
            tau=float(args.disc_tau),
            lambda_disc=float(args.disc_cls_lambda),
            max_mult=float(args.disc_cls_max_mult),
            include_pos=bool(args.disc_include_pos),
            pos_scale=float(args.disc_pos_scale),
            normalize_mean_one=(not bool(args.disc_no_mean_normalize)),
            teacher_conf_min=float(args.disc_teacher_conf_min),
            correctness_tau=float(args.disc_correctness_tau),
            use_teacher_hard_correct_gate=(not bool(args.disc_disable_teacher_hard_correct_gate)),
            use_teacher_conf_gate=(not bool(args.disc_disable_teacher_conf_gate)),
            use_teacher_better_gate=(not bool(args.disc_disable_teacher_better_gate)),
            weight_mode=str(args.disc_weight_mode),
        )
        discrepancy_cls_val_summary["enabled"] = True

        np.savez_compressed(
            save_root / "discrepancy_weights_train.npz",
            train_idx=train_idx.astype(np.int64),
            val_idx=val_idx.astype(np.int64),
            sample_weight_reco=sample_weight_reco.astype(np.float32),
            sample_weight_cls=sample_weight_cls.astype(np.float32),
            sample_weight_reco_val=sample_weight_reco_val.astype(np.float32),
            sample_weight_cls_val=sample_weight_cls_val.astype(np.float32),
        )
        print(
            "Discrepancy weights (reco): "
            f"mean={discrepancy_reco_summary.get('mean_weight', float('nan')):.4f}, "
            f"p95={discrepancy_reco_summary.get('p95_weight', float('nan')):.4f}, "
            f"w>1.5={discrepancy_reco_summary.get('fraction_w_gt_1p5', float('nan')):.4f}"
        )
        print(
            "Discrepancy weights (cls): "
            f"mean={discrepancy_cls_summary.get('mean_weight', float('nan')):.4f}, "
            f"p95={discrepancy_cls_summary.get('p95_weight', float('nan')):.4f}, "
            f"w>1.5={discrepancy_cls_summary.get('fraction_w_gt_1p5', float('nan')):.4f}"
        )
        print(
            "Discrepancy weights val (reco/cls): "
            f"reco_mean={discrepancy_reco_val_summary.get('mean_weight', float('nan')):.4f}, "
            f"cls_mean={discrepancy_cls_val_summary.get('mean_weight', float('nan')):.4f}"
        )

    # Optional jet-level regressor to provide frozen global calibration features to dual-view tagger.
    jet_regressor = None
    jet_reg_metrics: Dict[str, object] = {"enabled": bool(args.enable_jet_regressor)}
    feat_hlt_dual = feat_hlt_std.astype(np.float32, copy=True)
    if bool(args.enable_jet_regressor):
        print("\n" + "=" * 70)
        print("STEP 1B: JET-LEVEL REGRESSOR (HLT -> offline global jet targets)")
        print("=" * 70)
        # Targets:
        # [log_pt, log_e, log_m, tau21, tau32, log1p_d2, log1p_n_off, log1p_n_added]
        target_off, target_hlt_ref, target_idx = compute_jet_regression_targets(
            const_off=const_off,
            mask_off=masks_off,
            const_hlt=hlt_const,
            mask_hlt=hlt_mask,
        )
        target_dim = int(target_off.shape[1])

        jet_reg_train_ds = JetRegressionDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], target_off[train_idx])
        jet_reg_val_ds = JetRegressionDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], target_off[val_idx])
        jet_reg_train_loader = DataLoader(
            jet_reg_train_ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
            drop_last=True,
        )
        jet_reg_val_loader = DataLoader(
            jet_reg_val_ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
        )

        jet_regressor = JetLevelRegressor(
            input_dim=7,
            output_dim=target_dim,
            embed_dim=int(args.jet_reg_embed_dim),
            num_heads=int(args.jet_reg_num_heads),
            num_layers=int(args.jet_reg_num_layers),
            ff_dim=int(args.jet_reg_ff_dim),
            dropout=float(args.jet_reg_dropout),
        ).to(device)
        jet_regressor, jet_reg_best = train_jet_regressor(
            model=jet_regressor,
            train_loader=jet_reg_train_loader,
            val_loader=jet_reg_val_loader,
            device=device,
            epochs=int(args.jet_reg_epochs),
            patience=int(args.jet_reg_patience),
            lr=float(args.jet_reg_lr),
            weight_decay=float(args.jet_reg_weight_decay),
            warmup_epochs=int(args.jet_reg_warmup_epochs),
        )
        pred_log_all = predict_jet_regressor(
            model=jet_regressor,
            feat=feat_hlt_std,
            mask=hlt_mask,
            device=device,
            batch_size=int(cfg["training"]["batch_size"]),
        )
        delta_vs_hlt = pred_log_all - target_hlt_ref
        extra_global = np.concatenate([pred_log_all, delta_vs_hlt], axis=-1).astype(np.float32)
        extra_global = np.repeat(extra_global[:, None, :], feat_hlt_std.shape[1], axis=1)
        feat_hlt_dual = np.concatenate([feat_hlt_std, extra_global], axis=-1).astype(np.float32)
        feat_hlt_dual[~hlt_mask] = 0.0

        # Eval metrics on val/test.
        jet_reg_val_pred = pred_log_all[val_idx]
        jet_reg_test_pred = pred_log_all[test_idx]
        jet_reg_val_true = target_off[val_idx]
        jet_reg_test_true = target_off[test_idx]
        val_m = _jet_reg_metric_dict(jet_reg_val_pred, jet_reg_val_true, target_idx)
        test_m = _jet_reg_metric_dict(jet_reg_test_pred, jet_reg_test_true, target_idx)
        jet_reg_metrics = {
            "enabled": True,
            "best": jet_reg_best,
            "target_index": target_idx,
            "val": val_m,
            "test": test_m,
        }
        print(
            "Jet regressor test MAE: "
            f"pT={test_m['mae_pt']:.3f}, E={test_m['mae_e']:.3f}, "
            f"mass={test_m['mae_m']:.3f}, tau21={test_m['mae_tau21']:.4f}, "
            f"tau32={test_m['mae_tau32']:.4f}, n_added={test_m['mae_n_added']:.3f}"
        )

    # Stage A: reconstructor pretrain
    print("\n" + "=" * 70)
    print("STEP 2: STAGE A (RECONSTRUCTOR PRETRAIN)")
    print("=" * 70)
    ds_train_reco = WeightedReconstructionDataset(
        feat_hlt_std[train_idx], hlt_mask[train_idx], hlt_const[train_idx],
        const_off[train_idx], masks_off[train_idx], labels[train_idx],
        budget_merge_true[train_idx], budget_eff_true[train_idx],
        sample_weight_reco=sample_weight_reco,
    )
    ds_val_reco = WeightedReconstructionDataset(
        feat_hlt_std[val_idx], hlt_mask[val_idx], hlt_const[val_idx],
        const_off[val_idx], masks_off[val_idx], labels[val_idx],
        budget_merge_true[val_idx], budget_eff_true[val_idx],
        sample_weight_reco=sample_weight_reco_val,
    )
    dl_train_reco = DataLoader(
        ds_train_reco,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_reco = DataLoader(
        ds_val_reco,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor = wrap_reconstructor_unsmear_only(reconstructor)
    # compute_reconstruction_losses reads BASE_CONFIG["loss"], so sync it to our working config.
    BASE_CONFIG["loss"] = cfg["loss"]
    reconstructor, reco_val_metrics = train_reconstructor_weighted(
        reconstructor,
        dl_train_reco,
        dl_val_reco,
        device,
        cfg["reconstructor_training"],
        cfg["loss"],
        apply_reco_weight=bool(args.disc_weight_enable),
        teacher_model=teacher,
        feat_means=means,
        feat_stds=stds,
        kd_temperature=float(args.stageA_kd_temp),
        lambda_kd=float(args.stageA_lambda_kd),
        lambda_phys=float(args.stageA_lambda_phys),
        lambda_budget_hinge=float(args.stageA_lambda_budget_hinge),
        budget_eps=float(args.stageA_budget_eps),
        budget_weight_floor=float(args.stageA_budget_weight_floor),
        target_tpr_for_fpr=float(args.stageA_target_tpr),
    )


    print("\n" + "=" * 70)
    print("STEP 2B: CORRECTED-VIEW TAGGER (FROZEN STAGE-A RECONSTRUCTOR)")
    print("=" * 70)
    feat_corr_all, mask_corr_all = build_corrected_view_numpy(
        reconstructor=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        device=device,
        batch_size=BS,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=False,
    )

    ds_train_corr = JetDataset(feat_corr_all[train_idx], mask_corr_all[train_idx], labels[train_idx])
    ds_val_corr = JetDataset(feat_corr_all[val_idx], mask_corr_all[val_idx], labels[val_idx])
    ds_test_corr = JetDataset(feat_corr_all[test_idx], mask_corr_all[test_idx], labels[test_idx])
    dl_train_corr = DataLoader(ds_train_corr, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_corr = DataLoader(ds_val_corr, batch_size=BS, shuffle=False)
    dl_test_corr = DataLoader(ds_test_corr, batch_size=BS, shuffle=False)

    corrected_prejoint = ParticleTransformer(input_dim=int(feat_corr_all.shape[-1]), **cfg["model"]).to(device)
    corrected_prejoint = train_single_view_classifier_auc(
        corrected_prejoint,
        dl_train_corr,
        dl_val_corr,
        device,
        cfg["training"],
        name="CorrectedOnly-PreJoint",
    )
    auc_corrected_prejoint, preds_corrected_prejoint, _ = eval_classifier(corrected_prejoint, dl_test_corr, device)
    # Joint datasets
    train_pos = {int(j): i for i, j in enumerate(train_idx.tolist())}
    stageb_w_cls = np.array([sample_weight_cls[train_pos[int(j)]] for j in stageB_train_idx], dtype=np.float32)
    stageb_w_reco = np.array([sample_weight_reco[train_pos[int(j)]] for j in stageB_train_idx], dtype=np.float32)

    ds_train_joint = JointDualDataset(
        feat_hlt_std[train_idx], feat_hlt_dual[train_idx], hlt_mask[train_idx], hlt_const[train_idx],
        const_off[train_idx], masks_off[train_idx],
        budget_merge_true[train_idx], budget_eff_true[train_idx],
        labels[train_idx],
        sample_weight_cls=sample_weight_cls,
        sample_weight_reco=sample_weight_reco,
    )
    ds_train_joint_stageb = JointDualDataset(
        feat_hlt_std[stageB_train_idx], feat_hlt_dual[stageB_train_idx], hlt_mask[stageB_train_idx], hlt_const[stageB_train_idx],
        const_off[stageB_train_idx], masks_off[stageB_train_idx],
        budget_merge_true[stageB_train_idx], budget_eff_true[stageB_train_idx],
        labels[stageB_train_idx],
        sample_weight_cls=stageb_w_cls,
        sample_weight_reco=stageb_w_reco,
    )
    ds_val_joint = JointDualDataset(
        feat_hlt_std[val_idx], feat_hlt_dual[val_idx], hlt_mask[val_idx], hlt_const[val_idx],
        const_off[val_idx], masks_off[val_idx],
        budget_merge_true[val_idx], budget_eff_true[val_idx],
        labels[val_idx],
        sample_weight_cls=sample_weight_cls_val,
        sample_weight_reco=sample_weight_reco_val,
    )
    ds_test_joint = JointDualDataset(
        feat_hlt_std[test_idx], feat_hlt_dual[test_idx], hlt_mask[test_idx], hlt_const[test_idx],
        const_off[test_idx], masks_off[test_idx],
        budget_merge_true[test_idx], budget_eff_true[test_idx],
        labels[test_idx],
        sample_weight_cls=np.ones((len(test_idx),), dtype=np.float32),
        sample_weight_reco=np.ones((len(test_idx),), dtype=np.float32),
    )

    dl_train_joint = DataLoader(
        ds_train_joint, batch_size=BS, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    dl_train_joint_stageb = DataLoader(
        ds_train_joint_stageb, batch_size=BS, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint = DataLoader(
        ds_val_joint, batch_size=BS, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    dl_test_joint = DataLoader(
        ds_test_joint, batch_size=BS, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # Stage B + C: task-driven classifier/reconstructor coupling
    print("\n" + "=" * 70)
    print("STEP 3: STAGE B (DUAL PRETRAIN, FROZEN RECONSTRUCTOR)")
    print("=" * 70)
    dual_input_dim_a = int(feat_hlt_dual.shape[-1])
    dual_input_dim_b = 12 if bool(args.use_corrected_flags) else 10
    dual_joint = DualViewCrossAttnClassifier(input_dim_a=dual_input_dim_a, input_dim_b=dual_input_dim_b, **cfg["model"]).to(device)
    reconstructor, dual_joint, stageB_metrics, stageB_states = train_joint_dual(
        reconstructor=reconstructor,
        dual_model=dual_joint,
        train_loader=dl_train_joint_stageb,
        val_loader=dl_val_joint,
        device=device,
        stage_name="StageB-DualPretrain",
        freeze_reconstructor=True,
        epochs=int(args.stageB_epochs),
        patience=int(args.stageB_patience),
        lr_dual=float(args.stageB_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        lambda_reco=0.0,
        lambda_rank=float(args.stageB_lambda_rank),
        lambda_cons=float(args.stageB_lambda_cons),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
        min_epochs=int(args.stageB_min_epochs),
        select_metric=selection_metric,
        apply_cls_weight=bool(args.disc_weight_enable and float(args.disc_cls_lambda) > 0.0),
        apply_reco_weight=False,
        val_weight_key="sample_weight_cls",
        use_weighted_val_selection=bool(args.disc_weight_enable),
    )

    # Stage B test evaluation + checkpoint snapshot (before Stage C joint finetune).
    auc_stage2, preds_stage2, labs_stage2, _ = eval_joint_model(
        reconstructor,
        dual_joint,
        dl_test_joint,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    assert np.array_equal(labs.astype(np.float32), labs_stage2.astype(np.float32))
    stage2_reco_state = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
    stage2_dual_state = {k: v.detach().cpu().clone() for k, v in dual_joint.state_dict().items()}

    # Also evaluate Stage-B best-val_fpr50 checkpoint on test for direct comparison.
    auc_stage2_fprsel = float("nan")
    preds_stage2_fprsel = None
    if stageB_states.get("fpr50", {}).get("dual") is not None and stageB_states.get("fpr50", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageB_states["fpr50"]["reco"])
        dual_joint.load_state_dict(stageB_states["fpr50"]["dual"])
        auc_stage2_fprsel, preds_stage2_fprsel, labs_stage2_fprsel, _ = eval_joint_model(
            reconstructor,
            dual_joint,
            dl_test_joint,
            device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        assert np.array_equal(labs.astype(np.float32), labs_stage2_fprsel.astype(np.float32))

    # Restore Stage-B selected state before entering Stage C.
    reconstructor.load_state_dict(stage2_reco_state)
    dual_joint.load_state_dict(stage2_dual_state)

    print("\n" + "=" * 70)
    print("STEP 4: STAGE C (JOINT FINETUNE)")
    print("=" * 70)
    reconstructor, dual_joint, stageC_metrics, stageC_states = train_joint_dual(
        reconstructor=reconstructor,
        dual_model=dual_joint,
        train_loader=dl_train_joint,
        val_loader=dl_val_joint,
        device=device,
        stage_name="StageC-Joint",
        freeze_reconstructor=False,
        epochs=int(args.stageC_epochs),
        patience=int(args.stageC_patience),
        lr_dual=float(args.stageC_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        lambda_reco=float(args.lambda_reco),
        lambda_rank=0.0,
        lambda_cons=float(args.lambda_cons),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
        min_epochs=int(args.stageC_min_epochs),
        select_metric=selection_metric,
        apply_cls_weight=bool(args.disc_weight_enable and bool(args.disc_apply_cls_stagec) and float(args.disc_cls_lambda) > 0.0),
        apply_reco_weight=bool(args.disc_weight_enable and float(args.disc_reco_lambda) > 0.0),
        val_weight_key="sample_weight_cls",
        use_weighted_val_selection=bool(args.disc_weight_enable),
    )

    auc_joint, preds_joint, labs_joint, _ = eval_joint_model(
        reconstructor,
        dual_joint,
        dl_test_joint,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    assert np.array_equal(labs.astype(np.float32), labs_joint.astype(np.float32))

    # Evaluate Stage-C best-val_fpr50 checkpoint on test too.
    auc_joint_fprsel = float("nan")
    preds_joint_fprsel = None
    if stageC_states.get("fpr50", {}).get("dual") is not None and stageC_states.get("fpr50", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["fpr50"]["reco"])
        dual_joint.load_state_dict(stageC_states["fpr50"]["dual"])
        auc_joint_fprsel, preds_joint_fprsel, labs_joint_fprsel, _ = eval_joint_model(
            reconstructor,
            dual_joint,
            dl_test_joint,
            device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        assert np.array_equal(labs.astype(np.float32), labs_joint_fprsel.astype(np.float32))

    # Restore Stage-C selected state for downstream diagnostics/KD.
    if stageC_states.get("selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["selected"]["reco"])
    if stageC_states.get("selected", {}).get("dual") is not None:
        dual_joint.load_state_dict(stageC_states["selected"]["dual"])


    # Build order-preserved reconstructed tokens for diagnostics.
    print("\n" + "=" * 70)
    print("STEP 5: RECONSTRUCTION DIAGNOSTICS")
    print("=" * 70)
    reco_const, reco_mask = _predict_reco_tokens_ordered(
        model=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        device=device,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
    )

    created_merge_count = np.zeros((reco_mask.shape[0],), dtype=np.int32)
    created_eff_count = np.zeros((reco_mask.shape[0],), dtype=np.int32)
    pred_budget_total = np.zeros((reco_mask.shape[0],), dtype=np.float32)
    pred_budget_merge = np.zeros((reco_mask.shape[0],), dtype=np.float32)
    pred_budget_eff = np.zeros((reco_mask.shape[0],), dtype=np.float32)

    # Jet pT response/resolution diagnostics (test split).
    pt_truth_test = compute_jet_pt(const_off[test_idx], masks_off[test_idx])
    pt_hlt_test = compute_jet_pt(hlt_const[test_idx], hlt_mask[test_idx])
    pt_reco_test = compute_jet_pt(reco_const[test_idx], reco_mask[test_idx])
    pt_edges = build_pt_edges(pt_truth_test, int(args.response_n_bins))
    rr_hlt = jet_response_resolution(pt_truth_test, pt_hlt_test, pt_edges, int(args.response_min_count))
    rr_reco = jet_response_resolution(pt_truth_test, pt_reco_test, pt_edges, int(args.response_min_count))
    plot_response_resolution(
        rr_hlt,
        rr_reco,
        "HLT (reco)",
        "Unsmear-corrected HLT (reco)",
        save_root / "jet_response_resolution.png",
    )
    rr_hlt_map = {(r["pt_low"], r["pt_high"]): r for r in rr_hlt}
    rr_reco_map = {(r["pt_low"], r["pt_high"]): r for r in rr_reco}
    rr_keys = sorted(set(rr_hlt_map.keys()) & set(rr_reco_map.keys()))
    rr_hlt_common = [rr_hlt_map[k] for k in rr_keys]
    rr_reco_common = [rr_reco_map[k] for k in rr_keys]

    print("\nJet pT response/resolution by truth pT bin (test split):")
    print("  pT_low - pT_high | N | HLT resp | HLT reso | Corrected resp | Corrected reso")
    for h, r in zip(rr_hlt_common, rr_reco_common):
        print(
            f"  {h['pt_low']:.1f} - {h['pt_high']:.1f} | {h['count']:5d} | "
            f"{h['response']:.4f} | {h['resolution']:.4f} | "
            f"{r['response']:.4f} | {r['resolution']:.4f}"
        )

    count_summary = plot_constituent_count_diagnostics(
        save_root=save_root,
        mask_off=masks_off,
        hlt_mask=hlt_mask,
        reco_mask=reco_mask,
        created_merge_count=created_merge_count,
        created_eff_count=created_eff_count,
        hlt_stats=hlt_stats,
    )
    print("\nConstituent-count diagnostics:")
    print(
        f"  Means: offline={count_summary['offline_count_mean']:.3f}, "
        f"hlt={count_summary['hlt_count_mean']:.3f}, reco={count_summary['reco_count_mean']:.3f}"
    )
    print(
        f"  MAE vs offline: hlt={count_summary['hlt_count_mae_vs_offline']:.3f}, "
        f"reco={count_summary['reco_count_mae_vs_offline']:.3f}"
    )

    mask_common_test = masks_off[test_idx] & hlt_mask[test_idx] & reco_mask[test_idx]
    constituent_unsmear_summary = _summarize_constituent_residuals(
        const_off=const_off[test_idx],
        const_hlt=hlt_const[test_idx],
        const_reco=reco_const[test_idx],
        mask_common=mask_common_test,
    )
    _plot_constituent_residual_distributions(
        const_off=const_off[test_idx],
        const_hlt=hlt_const[test_idx],
        const_reco=reco_const[test_idx],
        mask_common=mask_common_test,
        out_path=save_root / "constituent_delta_distributions_test.png",
    )
    _plot_constituent_mae_by_bin(
        constituent_unsmear_summary,
        out_path=save_root / "constituent_mae_vs_pt_test.png",
        axis_key="by_pt",
        title="Constituent MAE vs offline by offline pT bin (test)",
    )
    _plot_constituent_mae_by_bin(
        constituent_unsmear_summary,
        out_path=save_root / "constituent_mae_vs_abseta_test.png",
        axis_key="by_abs_eta",
        title="Constituent MAE vs offline by |offline eta| bin (test)",
    )

    for key, label in (("dpt", "pT"), ("deta", "eta"), ("dphi", "phi")):
        imp = constituent_unsmear_summary[key]["improvement"]
        print(
            f"Residual improvement ({label}) | "
            f"MAE frac={imp['mae_reduction_frac']:.4f}, "
            f"STD frac={imp['std_reduction_frac']:.4f}, "
            f"Token improved frac={imp['fraction_tokens_improved']:.4f}"
        )
    # Build fixed corrected view tensors for final KD stage and additional diagnostics.
    feat_b_all, mask_b_all = build_corrected_view_numpy(
        reconstructor=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        device=device,
        batch_size=BS,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    soft_view_summary_test = summarize_soft_corrected_view(
        feat_b_all[test_idx],
        mask_b_all[test_idx],
    )

    # Stage D: final KD with reconstructor frozen.
    stageD_metrics: Dict[str, object] = {}
    kd_student = None
    auc_joint_kd = float("nan")
    preds_joint_kd = None
    tpr_j_kd = np.array([], dtype=np.float64)
    fpr_j_kd = np.array([], dtype=np.float64)
    fpr30_joint_kd = float("nan")
    fpr50_joint_kd = float("nan")
    if not bool(args.disable_final_kd):
        print("\n" + "=" * 70)
        print("STEP 6: STAGE D (FINAL KD, FROZEN RECONSTRUCTOR)")
        print("=" * 70)
        kd_train_cfg = json.loads(json.dumps(cfg["training"]))
        kd_cfg = json.loads(json.dumps(cfg["kd"]))
        if int(args.stageD_kd_epochs) > 0:
            kd_train_cfg["epochs"] = int(args.stageD_kd_epochs)
        if int(args.stageD_kd_patience) > 0:
            kd_train_cfg["patience"] = int(args.stageD_kd_patience)
        if float(args.stageD_kd_lr) > 0:
            kd_train_cfg["lr"] = float(args.stageD_kd_lr)

        kd_train_ds = DualViewKDDataset(
            feat_hlt_dual[train_idx], hlt_mask[train_idx],
            feat_b_all[train_idx], mask_b_all[train_idx],
            feat_off_std[train_idx], masks_off[train_idx],
            labels[train_idx],
        )
        kd_val_ds = DualViewKDDataset(
            feat_hlt_dual[val_idx], hlt_mask[val_idx],
            feat_b_all[val_idx], mask_b_all[val_idx],
            feat_off_std[val_idx], masks_off[val_idx],
            labels[val_idx],
        )
        kd_test_ds = DualViewKDDataset(
            feat_hlt_dual[test_idx], hlt_mask[test_idx],
            feat_b_all[test_idx], mask_b_all[test_idx],
            feat_off_std[test_idx], masks_off[test_idx],
            labels[test_idx],
        )
        kd_train_loader = DataLoader(kd_train_ds, batch_size=BS, shuffle=True, drop_last=True)
        kd_val_loader = DataLoader(kd_val_ds, batch_size=BS, shuffle=False)
        kd_test_loader = DataLoader(kd_test_ds, batch_size=BS, shuffle=False)

        kd_student = DualViewCrossAttnClassifier(input_dim_a=dual_input_dim_a, input_dim_b=dual_input_dim_b, **cfg["model"]).to(device)
        kd_student.load_state_dict(dual_joint.state_dict())
        kd_student = train_dual_kd_student(
            student=kd_student,
            teacher=teacher,
            kd_train_loader=kd_train_loader,
            kd_val_loader=kd_val_loader,
            device=device,
            train_cfg=kd_train_cfg,
            kd_cfg=kd_cfg,
            name="StageD-Joint+KD",
            run_self_train=bool(kd_cfg.get("self_train", True)),
        )
        auc_joint_kd, preds_joint_kd, labs_joint_kd = eval_classifier_dual(kd_student, kd_test_loader, device)
        assert np.array_equal(labs.astype(np.float32), labs_joint_kd.astype(np.float32))
        fpr_j_kd, tpr_j_kd, _ = roc_curve(labs, preds_joint_kd)
        fpr30_joint_kd = fpr_at_target_tpr(fpr_j_kd, tpr_j_kd, 0.30)
        fpr50_joint_kd = fpr_at_target_tpr(fpr_j_kd, tpr_j_kd, 0.50)
        stageD_metrics = {
            "enabled": 1.0,
            "train_cfg": kd_train_cfg,
            "kd_cfg": kd_cfg,
        }
    else:
        stageD_metrics = {"enabled": 0.0}

    # Final metrics
    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_c, tpr_c, _ = roc_curve(labs, preds_corrected_prejoint)
    fpr_s2, tpr_s2, _ = roc_curve(labs, preds_stage2)
    fpr_j, tpr_j, _ = roc_curve(labs, preds_joint)
    if preds_stage2_fprsel is not None:
        fpr_s2_fprsel, tpr_s2_fprsel, _ = roc_curve(labs, preds_stage2_fprsel)
    else:
        fpr_s2_fprsel, tpr_s2_fprsel = np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if preds_joint_fprsel is not None:
        fpr_j_fprsel, tpr_j_fprsel, _ = roc_curve(labs, preds_joint_fprsel)
    else:
        fpr_j_fprsel, tpr_j_fprsel = np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.30)
    fpr30_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.30)
    fpr30_corrected_prejoint = fpr_at_target_tpr(fpr_c, tpr_c, 0.30)
    fpr30_stage2 = fpr_at_target_tpr(fpr_s2, tpr_s2, 0.30)
    fpr30_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.30)
    fpr30_stage2_fprsel = fpr_at_target_tpr(fpr_s2_fprsel, tpr_s2_fprsel, 0.30) if preds_stage2_fprsel is not None else float("nan")
    fpr30_joint_fprsel = fpr_at_target_tpr(fpr_j_fprsel, tpr_j_fprsel, 0.30) if preds_joint_fprsel is not None else float("nan")
    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.50)
    fpr50_corrected_prejoint = fpr_at_target_tpr(fpr_c, tpr_c, 0.50)
    fpr50_stage2 = fpr_at_target_tpr(fpr_s2, tpr_s2, 0.50)
    fpr50_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.50)
    fpr50_stage2_fprsel = fpr_at_target_tpr(fpr_s2_fprsel, tpr_s2_fprsel, 0.50) if preds_stage2_fprsel is not None else float("nan")
    fpr50_joint_fprsel = fpr_at_target_tpr(fpr_j_fprsel, tpr_j_fprsel, 0.50) if preds_joint_fprsel is not None else float("nan")

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"CorrectedOnly (PreJoint FrozenReco) AUC: {auc_corrected_prejoint:.4f}")
    print(f"Stage2 (PreJoint) AUC: {auc_stage2:.4f}")
    if preds_stage2_fprsel is not None:
        print(f"Stage2 (BestValFPR50) AUC: {auc_stage2_fprsel:.4f}")
    print(f"Joint Dual-View  AUC: {auc_joint:.4f}")
    if preds_joint_fprsel is not None:
        print(f"Joint Dual-View (BestValFPR50) AUC: {auc_joint_fprsel:.4f}")
    if preds_joint_kd is not None:
        print(f"Joint Dual-View+KD AUC: {auc_joint_kd:.4f}")
    print()
    print(
        f"FPR@30 Teacher/Baseline/Stage2/Joint: "
        f"{fpr30_teacher:.6f} / {fpr30_baseline:.6f} / {fpr30_stage2:.6f} / {fpr30_joint:.6f}"
    )
    if preds_stage2_fprsel is not None or preds_joint_fprsel is not None:
        print(
            f"FPR@30 Stage2BestFPR / JointBestFPR: "
            f"{fpr30_stage2_fprsel:.6f} / {fpr30_joint_fprsel:.6f}"
        )
    print(
        f"FPR@50 Teacher/Baseline/Stage2/Joint: "
        f"{fpr50_teacher:.6f} / {fpr50_baseline:.6f} / {fpr50_stage2:.6f} / {fpr50_joint:.6f}"
    )
    if preds_stage2_fprsel is not None or preds_joint_fprsel is not None:
        print(
            f"FPR@50 Stage2BestFPR / JointBestFPR: "
            f"{fpr50_stage2_fprsel:.6f} / {fpr50_joint_fprsel:.6f}"
        )
    if preds_joint_kd is not None:
        print(f"FPR@30 Joint+KD: {fpr30_joint_kd:.6f}")
        print(f"FPR@50 Joint+KD: {fpr50_joint_kd:.6f}")

    plot_lines = [
        (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
        (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
        (tpr_c, fpr_c, "-", f"CorrectedOnly PreJoint (AUC={auc_corrected_prejoint:.3f})", "goldenrod"),
        (tpr_s2, fpr_s2, "-.", f"Stage2 PreJoint (AUC={auc_stage2:.3f})", "darkorange"),
        (tpr_j, fpr_j, "-.", f"Joint Dual (AUC={auc_joint:.3f})", "darkslateblue"),
    ]
    if preds_stage2_fprsel is not None:
        plot_lines.append(
            (tpr_s2_fprsel, fpr_s2_fprsel, ":", f"Stage2 BestValFPR (AUC={auc_stage2_fprsel:.3f})", "peru")
        )
    if preds_joint_fprsel is not None:
        plot_lines.append(
            (tpr_j_fprsel, fpr_j_fprsel, "--", f"Joint BestValFPR (AUC={auc_joint_fprsel:.3f})", "indigo")
        )
    if preds_joint_kd is not None:
        plot_lines.append((tpr_j_kd, fpr_j_kd, ":", f"Joint Dual+KD (AUC={auc_joint_kd:.3f})", "darkgreen"))
    plot_roc(
        plot_lines,
        save_root / "results_teacher_baseline_joint.png",
        min_fpr=1e-4,
    )

    def rr_field(records, key):
        return np.array([r[key] for r in records], dtype=np.float64)

    np.savez(
        save_root / "results.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_corrected_prejoint=auc_corrected_prejoint,
        auc_stage2=auc_stage2,
        auc_stage2_fprsel=auc_stage2_fprsel,
        auc_joint=auc_joint,
        auc_joint_fprsel=auc_joint_fprsel,
        auc_joint_kd=auc_joint_kd,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_corrected_prejoint=fpr_c,
        tpr_corrected_prejoint=tpr_c,
        fpr_stage2=fpr_s2,
        tpr_stage2=tpr_s2,
        fpr_stage2_fprsel=fpr_s2_fprsel,
        tpr_stage2_fprsel=tpr_s2_fprsel,
        fpr_joint=fpr_j,
        tpr_joint=tpr_j,
        fpr_joint_fprsel=fpr_j_fprsel,
        tpr_joint_fprsel=tpr_j_fprsel,
        fpr_joint_kd=fpr_j_kd,
        tpr_joint_kd=tpr_j_kd,
        fpr30_teacher=fpr30_teacher,
        fpr30_baseline=fpr30_baseline,
        fpr30_corrected_prejoint=fpr30_corrected_prejoint,
        fpr30_stage2=fpr30_stage2,
        fpr30_stage2_fprsel=fpr30_stage2_fprsel,
        fpr30_joint=fpr30_joint,
        fpr30_joint_fprsel=fpr30_joint_fprsel,
        fpr30_joint_kd=fpr30_joint_kd,
        fpr50_teacher=fpr50_teacher,
        fpr50_baseline=fpr50_baseline,
        fpr50_corrected_prejoint=fpr50_corrected_prejoint,
        fpr50_stage2=fpr50_stage2,
        fpr50_stage2_fprsel=fpr50_stage2_fprsel,
        fpr50_joint=fpr50_joint,
        fpr50_joint_fprsel=fpr50_joint_fprsel,
        fpr50_joint_kd=fpr50_joint_kd,
        jet_response_pt_low=rr_field(rr_hlt_common, "pt_low"),
        jet_response_pt_high=rr_field(rr_hlt_common, "pt_high"),
        jet_response_count=rr_field(rr_hlt_common, "count"),
        jet_response_hlt_mean=rr_field(rr_hlt_common, "response"),
        jet_response_hlt_std=rr_field(rr_hlt_common, "resolution"),
        jet_response_corrected_mean=rr_field(rr_reco_common, "response"),
        jet_response_corrected_std=rr_field(rr_reco_common, "resolution"),
        added_target_scale=float(added_target_scale),
    )

    with open(save_root / "constituent_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(count_summary, f, indent=2)
    with open(save_root / "constituent_unsmear_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(constituent_unsmear_summary, f, indent=2)
    with open(save_root / "soft_corrected_view_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(soft_view_summary_test, f, indent=2)
    with open(save_root / "jet_regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(jet_reg_metrics, f, indent=2)

    with open(save_root / "joint_stage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": {
                    "mode": "nopriv_unsmearonly",
                    "added_target_scale": float(added_target_scale),
                    "mean_true_added_raw": float(true_added_raw.mean()),
                    "mean_target_added": float(budget_merge_true.mean()),
                    "stageB_train_frac": float(stageB_train_frac),
                    "stageB_subset_seed": int(stageB_subset_seed),
                    "stageB_train_count": int(len(stageB_train_idx)),
                    "train_count": int(len(train_idx)),
                },
                "discrepancy_weighting": {
                    "enabled": bool(args.disc_weight_enable),
                    "mode": str(args.disc_weight_mode),
                    "reco_summary": discrepancy_reco_summary,
                    "cls_summary": discrepancy_cls_summary,
                    "reco_val_summary": discrepancy_reco_val_summary,
                    "cls_val_summary": discrepancy_cls_val_summary,
                    "apply_cls_stagec": bool(args.disc_apply_cls_stagec),
                },
                "jet_regressor": jet_reg_metrics,
                "stageA_reconstructor": reco_val_metrics,
                "stageA_corrected_only": {"auc": float(auc_corrected_prejoint), "fpr30": float(fpr30_corrected_prejoint), "fpr50": float(fpr50_corrected_prejoint)},
                "stageB_joint": stageB_metrics,
                "stageC_joint": stageC_metrics,
                "stageD_kd": stageD_metrics,
                "test_stage2": {
                    "auc_stage2": float(auc_stage2),
                    "auc_stage2_fprsel": float(auc_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr30_stage2": float(fpr30_stage2),
                    "fpr30_stage2_fprsel": float(fpr30_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr50_stage2": float(fpr50_stage2),
                    "fpr50_stage2_fprsel": float(fpr50_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                },
                "test": {
                    "auc_teacher": float(auc_teacher),
                    "auc_baseline": float(auc_baseline),
                    "auc_corrected_prejoint": float(auc_corrected_prejoint),
                    "auc_stage2": float(auc_stage2),
                    "auc_stage2_fprsel": float(auc_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "auc_joint": float(auc_joint),
                    "auc_joint_fprsel": float(auc_joint_fprsel) if preds_joint_fprsel is not None else None,
                    "auc_joint_kd": float(auc_joint_kd) if preds_joint_kd is not None else None,
                    "fpr30_teacher": float(fpr30_teacher),
                    "fpr30_baseline": float(fpr30_baseline),
                    "fpr30_corrected_prejoint": float(fpr30_corrected_prejoint),
                    "fpr30_stage2": float(fpr30_stage2),
                    "fpr30_stage2_fprsel": float(fpr30_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr30_joint": float(fpr30_joint),
                    "fpr30_joint_fprsel": float(fpr30_joint_fprsel) if preds_joint_fprsel is not None else None,
                    "fpr30_joint_kd": float(fpr30_joint_kd) if preds_joint_kd is not None else None,
                    "fpr50_teacher": float(fpr50_teacher),
                    "fpr50_baseline": float(fpr50_baseline),
                    "fpr50_corrected_prejoint": float(fpr50_corrected_prejoint),
                    "fpr50_stage2": float(fpr50_stage2),
                    "fpr50_stage2_fprsel": float(fpr50_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr50_joint": float(fpr50_joint),
                    "fpr50_joint_fprsel": float(fpr50_joint_fprsel) if preds_joint_fprsel is not None else None,
                    "fpr50_joint_kd": float(fpr50_joint_kd) if preds_joint_kd is not None else None,
                },
            },
            f,
            indent=2,
        )

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg["hlt_effects"], "stats": hlt_stats}, f, indent=2)

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": auc_teacher}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": auc_baseline}, save_root / "baseline.pt")
        torch.save({"model": corrected_prejoint.state_dict(), "auc": auc_corrected_prejoint}, save_root / "corrected_only_prejoint.pt")
        if jet_regressor is not None:
            torch.save({"model": jet_regressor.state_dict(), "metrics": jet_reg_metrics}, save_root / "jet_regressor.pt")
        torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor.pt")
        torch.save({"model": stage2_reco_state, "val": reco_val_metrics}, save_root / "offline_reconstructor_stage2.pt")
        torch.save(
            {
                "model": stage2_dual_state,
                "auc": float(auc_stage2),
                "fpr30": float(fpr30_stage2),
                "fpr50": float(fpr50_stage2),
            },
            save_root / "dual_joint_stage2.pt",
        )
        if stageB_states.get("fpr50", {}).get("reco") is not None:
            torch.save({"model": stageB_states["fpr50"]["reco"], "val": reco_val_metrics}, save_root / "offline_reconstructor_stage2_bestfpr50.pt")
        if stageB_states.get("fpr50", {}).get("dual") is not None:
            torch.save(
                {
                    "model": stageB_states["fpr50"]["dual"],
                    "auc": float(auc_stage2_fprsel) if preds_stage2_fprsel is not None else float("nan"),
                    "fpr30": float(fpr30_stage2_fprsel) if preds_stage2_fprsel is not None else float("nan"),
                    "fpr50": float(fpr50_stage2_fprsel) if preds_stage2_fprsel is not None else float("nan"),
                },
                save_root / "dual_joint_stage2_bestfpr50.pt",
            )
        torch.save({"model": dual_joint.state_dict(), "auc": auc_joint}, save_root / "dual_joint.pt")
        if stageC_states.get("fpr50", {}).get("reco") is not None:
            torch.save({"model": stageC_states["fpr50"]["reco"], "val": reco_val_metrics}, save_root / "offline_reconstructor_bestfpr50.pt")
        if stageC_states.get("fpr50", {}).get("dual") is not None:
            torch.save(
                {
                    "model": stageC_states["fpr50"]["dual"],
                    "auc": float(auc_joint_fprsel) if preds_joint_fprsel is not None else float("nan"),
                    "fpr30": float(fpr30_joint_fprsel) if preds_joint_fprsel is not None else float("nan"),
                    "fpr50": float(fpr50_joint_fprsel) if preds_joint_fprsel is not None else float("nan"),
                },
                save_root / "dual_joint_bestfpr50.pt",
            )
        if kd_student is not None:
            torch.save({"model": kd_student.state_dict(), "auc": auc_joint_kd}, save_root / "dual_joint_kd.pt")

    print(f"\nSaved joint results to: {save_root}")


if __name__ == "__main__":
    main()

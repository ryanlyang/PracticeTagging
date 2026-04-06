#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-driven Offline Reconstructor + Dual-View tagger joint training (oracle-study variant).

Pipeline:
1) Build pseudo-HLT from offline jets (same realistic generator family).
2) Train teacher (offline) and baseline (HLT).
3) Stage A: pretrain reconstructor (reconstruction losses).
4) Stage B: pretrain dual-view classifier with reconstructor frozen.
5) Stage C: joint finetune reconstructor + dual-view classifier.
6) Select checkpoints by lowest val FPR@50% TPR.

Notes:
- Uses soft candidate view from reconstructor outputs for differentiable joint training.
- Includes a differentiable low-FPR surrogate targeting TPR=0.5 behavior.
- Variant: optional oracle privileged supervision modes A/B/C/D/E (+B_gen).
- Default behavior remains non-privileged unmerge-only unless oracle mode enables extras.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY_HUNGARIAN = True
except Exception:
    linear_sum_assignment = None  # type: ignore
    _HAS_SCIPY_HUNGARIAN = False

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
    OfflineReconstructor,
    ReconstructionDataset,
    train_reconstructor,
    reconstruct_dataset,
    plot_roc,
    fpr_at_target_tpr,
    plot_constituent_count_diagnostics,
    plot_budget_diagnostics,
    train_dual_kd_student,
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

# ----------------------------- Oracle Study Globals ----------------------------- #
_ORACLE_STUDY_VARIANT: str = "A"
_ORACLE_STUDY_ENABLE_GEN: bool = False
_ORACLE_UNSMEAR_KEEP_THRESH: float = 0.03
_ORACLE_CURRENT_BATCH: Dict[str, torch.Tensor] | None = None
_ORACLE_WARNED_MISSING_BATCH: bool = False


def _deepcopy_config() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _clamp_target_scale(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _load_checkpoint_model_state(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format at {path}")


def _weighted_batch_mean(vec: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return vec.mean()
    denom = sample_weight.sum().clamp(min=1e-6)
    return (vec * sample_weight).sum() / denom


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


def _merge_top2_sources(
    idx_a: np.ndarray,
    pt_a: np.ndarray,
    idx_b: np.ndarray,
    pt_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cand: List[Tuple[int, float]] = []
    for ii, pp in zip(idx_a.tolist(), pt_a.tolist()):
        if int(ii) >= 0 and float(pp) > 0.0:
            cand.append((int(ii), float(pp)))
    for ii, pp in zip(idx_b.tolist(), pt_b.tolist()):
        if int(ii) >= 0 and float(pp) > 0.0:
            cand.append((int(ii), float(pp)))
    if not cand:
        return np.array([-1, -1], dtype=np.int32), np.array([0.0, 0.0], dtype=np.float32)
    # De-duplicate by index, keep max pT for stability.
    best: Dict[int, float] = {}
    for ii, pp in cand:
        if ii not in best or pp > best[ii]:
            best[ii] = pp
    ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:2]
    idx = np.array([kv[0] for kv in ordered] + [-1] * (2 - len(ordered)), dtype=np.int32)
    pt = np.array([kv[1] for kv in ordered] + [0.0] * (2 - len(ordered)), dtype=np.float32)
    return idx, pt


def apply_hlt_effects_realistic_with_oracle_tracking(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: Dict,
    seed: int = RANDOM_SEED,
    unsmear_keep_thresh: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict[str, np.ndarray]]:
    """
    Realistic pseudo-HLT generation with lightweight oracle tracking for study A/B/C/D/E.

    Oracle labels are synthetic-truth from the generation process only and are for training diagnostics:
      - oracle_action: keep=0, unsmear=1, split=2, reassign=3, invalid=-1
      - oracle_tok_target / oracle_tok_valid for non-split actions
      - oracle_split_target[:,:,:2,:] / oracle_split_valid for split top-2 targets
    """
    rs = np.random.RandomState(int(seed))
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    # Track merge ancestry with top-2 source ids (by offline pT) and full source multiplicity.
    src_count = hlt_mask.astype(np.int32)
    src_top2_idx = np.full((n_jets, max_part, 2), -1, dtype=np.int32)
    src_top2_pt = np.zeros((n_jets, max_part, 2), dtype=np.float32)
    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        if valid.size == 0:
            continue
        src_top2_idx[j, valid, 0] = valid.astype(np.int32)
        src_top2_pt[j, valid, 0] = const[j, valid, 0].astype(np.float32)

    reassign_flag = np.zeros((n_jets, max_part), dtype=np.uint8)

    n_initial = int(hlt_mask.sum())
    merge_lost_per_jet = np.zeros(n_jets, dtype=np.float32)
    eff_lost_per_jet = np.zeros(n_jets, dtype=np.float32)

    # 1) Pre-threshold
    pt_threshold = float(hcfg["pt_threshold_hlt"])
    below = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0
    src_count[~hlt_mask] = 0
    src_top2_idx[~hlt_mask] = -1
    src_top2_pt[~hlt_mask] = 0.0
    n_lost_threshold_pre = int(below.sum())

    # 2) Local merging
    n_merged = 0
    merge_radius = float(hcfg["merge_radius"])
    if hcfg["merge_enabled"] and merge_radius > 0:
        for j in range(n_jets):
            valid_idx = np.where(hlt_mask[j])[0]
            if len(valid_idx) < 2:
                continue
            to_remove = set()
            for ii in range(len(valid_idx)):
                a = valid_idx[ii]
                if a in to_remove:
                    continue
                for jj in range(ii + 1, len(valid_idx)):
                    b = valid_idx[jj]
                    if b in to_remove:
                        continue
                    deta = hlt[j, a, 1] - hlt[j, b, 1]
                    dphi = reco_base.wrap_phi_np(hlt[j, a, 2] - hlt[j, b, 2])
                    dR = float(np.sqrt(deta * deta + dphi * dphi))
                    if dR >= merge_radius:
                        continue

                    pt_a = float(hlt[j, a, 0])
                    pt_b = float(hlt[j, b, 0])
                    pt_sum = pt_a + pt_b
                    if pt_sum <= 1e-8:
                        continue
                    wa = pt_a / pt_sum
                    wb = pt_b / pt_sum

                    hlt[j, a, 0] = pt_sum
                    hlt[j, a, 1] = wa * hlt[j, a, 1] + wb * hlt[j, b, 1]
                    phi_a = hlt[j, a, 2]
                    phi_b = hlt[j, b, 2]
                    hlt[j, a, 2] = np.arctan2(
                        wa * np.sin(phi_a) + wb * np.sin(phi_b),
                        wa * np.cos(phi_a) + wb * np.cos(phi_b),
                    )
                    hlt[j, a, 3] = hlt[j, a, 3] + hlt[j, b, 3]

                    src_count[j, a] += src_count[j, b]
                    m_idx, m_pt = _merge_top2_sources(
                        src_top2_idx[j, a],
                        src_top2_pt[j, a],
                        src_top2_idx[j, b],
                        src_top2_pt[j, b],
                    )
                    src_top2_idx[j, a] = m_idx
                    src_top2_pt[j, a] = m_pt

                    to_remove.add(b)
                    n_merged += 1
                    merge_lost_per_jet[j] += 1.0

            for idx in to_remove:
                hlt_mask[j, idx] = False
                hlt[j, idx] = 0
                src_count[j, idx] = 0
                src_top2_idx[j, idx] = -1
                src_top2_pt[j, idx] = 0.0
                reassign_flag[j, idx] = 0

    # 3) Efficiency model
    jet_q = rs.lognormal(mean=0.0, sigma=float(hcfg["jet_quality_sigma"]), size=n_jets).astype(np.float32)
    jet_q = np.clip(jet_q, float(hcfg["jet_quality_min"]), float(hcfg["jet_quality_max"]))

    density = np.zeros((n_jets, max_part), dtype=np.float32)
    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        if len(valid) == 0:
            continue
        density_j = reco_base._compute_local_density_np(
            eta=hlt[j, :, 1],
            phi=hlt[j, :, 2],
            valid_idx=valid,
            radius=float(hcfg["density_radius"]),
        )
        density[j, valid] = density_j

    abs_eta = np.abs(hlt[:, :, 1])
    pt = np.maximum(hlt[:, :, 0], 1e-8)
    eta_plateau = np.where(
        abs_eta < hcfg["eta_break"], hcfg["eff_plateau_barrel"], hcfg["eff_plateau_endcap"]
    ).astype(np.float32)
    pt50 = np.where(
        abs_eta < hcfg["eta_break"], hcfg["eff_pt50_barrel"], hcfg["eff_pt50_endcap"]
    ).astype(np.float32)
    width = np.where(
        abs_eta < hcfg["eta_break"], hcfg["eff_width_barrel"], hcfg["eff_width_endcap"]
    ).astype(np.float32)
    turn_on = 1.0 / (1.0 + np.exp(-(pt - pt50) / np.maximum(width, 1e-6)))
    density_term = np.exp(-float(hcfg["eff_density_alpha"]) * density)
    q_eff = np.clip(jet_q[:, None], float(hcfg["eff_quality_min"]), float(hcfg["eff_quality_max"]))
    eps_eff = eta_plateau * turn_on * density_term * q_eff
    eps_eff = np.clip(eps_eff, float(hcfg["eff_floor"]), float(hcfg["eff_ceil"]))

    u = rs.random_sample((n_jets, max_part))
    lost_eff = (u > eps_eff) & hlt_mask
    hlt_mask[lost_eff] = False
    hlt[lost_eff] = 0
    src_count[lost_eff] = 0
    src_top2_idx[lost_eff] = -1
    src_top2_pt[lost_eff] = 0.0
    reassign_flag[lost_eff] = 0
    eff_lost_per_jet = lost_eff.sum(axis=1).astype(np.float32)
    n_lost_eff = int(lost_eff.sum())

    # 4) Smearing + tails + local reassignment
    n_reassigned = 0
    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        if len(valid) == 0:
            continue

        pt_j = np.maximum(hlt[j, valid, 0], 1e-8)
        eta_j = hlt[j, valid, 1]
        phi_j = hlt[j, valid, 2]
        abs_eta_j = np.abs(eta_j)
        dens_j = density[j, valid]

        eta_scale = 1.0 + float(hcfg["smear_eta_scale"]) * abs_eta_j
        q = float(jet_q[j])

        sigma_rel = np.sqrt(
            (float(hcfg["smear_a"]) / np.sqrt(pt_j)) ** 2
            + float(hcfg["smear_b"]) ** 2
            + (float(hcfg["smear_c"]) / pt_j) ** 2
        )
        sigma_rel = sigma_rel * eta_scale * q
        sigma_rel = np.clip(sigma_rel, float(hcfg["smear_sigma_min"]), float(hcfg["smear_sigma_max"]))

        tail_prob = (
            float(hcfg["tail_base"])
            + float(hcfg["tail_eta_coeff"]) * abs_eta_j
            + float(hcfg["tail_density_coeff"]) * dens_j
        )
        tail_prob = np.clip(tail_prob, 0.0, float(hcfg["tail_prob_max"]))
        is_tail = rs.random_sample(len(valid)) < tail_prob

        ratio = rs.normal(loc=1.0, scale=sigma_rel)
        tail_sigma = float(hcfg["tail_sigma_scale"]) * sigma_rel + float(hcfg["tail_sigma_add"])
        ratio_tail = rs.normal(loc=float(hcfg["tail_mu"]), scale=tail_sigma)
        ratio[is_tail] = ratio_tail[is_tail]
        ratio = np.clip(ratio, float(hcfg["pt_resp_min"]), float(hcfg["pt_resp_max"]))
        pt_new = np.clip(pt_j * ratio, 1e-8, None)

        sigma_eta = (
            float(hcfg["eta_smear_const"]) + float(hcfg["eta_smear_inv_sqrt"]) / np.sqrt(pt_j)
        ) * eta_scale * q
        sigma_phi = (
            float(hcfg["phi_smear_const"]) + float(hcfg["phi_smear_inv_sqrt"]) / np.sqrt(pt_j)
        ) * eta_scale * q

        eta_new = eta_j + rs.normal(loc=0.0, scale=sigma_eta)
        phi_new = reco_base.wrap_phi_np(phi_j + rs.normal(loc=0.0, scale=sigma_phi))

        if float(hcfg["reassign_prob_base"]) > 0.0 and len(valid) > 1:
            p_reassign = float(hcfg["reassign_prob_base"]) + float(hcfg["reassign_density_coeff"]) * dens_j
            p_reassign = np.clip(p_reassign, 0.0, float(hcfg["reassign_prob_max"]))
            do_reassign = rs.random_sample(len(valid)) < p_reassign
            for ii in np.where(do_reassign)[0]:
                deta = eta_new[ii] - eta_new
                dphi = reco_base.wrap_phi_np(phi_new[ii] - phi_new)
                dR = np.sqrt(deta * deta + dphi * dphi)
                dR[ii] = 1e9
                nn = int(np.argmin(dR))
                if dR[nn] > float(hcfg["reassign_radius"]):
                    continue
                lam = rs.uniform(float(hcfg["reassign_strength_min"]), float(hcfg["reassign_strength_max"]))
                eta_new[ii] = (1.0 - lam) * eta_new[ii] + lam * eta_new[nn]
                phi_new[ii] = np.arctan2(
                    (1.0 - lam) * np.sin(phi_new[ii]) + lam * np.sin(phi_new[nn]),
                    (1.0 - lam) * np.cos(phi_new[ii]) + lam * np.cos(phi_new[nn]),
                )
                reassign_flag[j, valid[ii]] = 1
                n_reassigned += 1

        eta_new = np.clip(eta_new, -5.0, 5.0)
        phi_new = reco_base.wrap_phi_np(phi_new)
        e_new = pt_new * np.cosh(eta_new)

        hlt[j, valid, 0] = pt_new
        hlt[j, valid, 1] = eta_new
        hlt[j, valid, 2] = phi_new
        hlt[j, valid, 3] = e_new

    # 5) Optional post-smear threshold
    post_thr = float(hcfg["post_smear_pt_threshold"])
    n_lost_threshold_post = 0
    if post_thr > 0:
        below_post = (hlt[:, :, 0] < post_thr) & hlt_mask
        hlt_mask[below_post] = False
        hlt[below_post] = 0
        src_count[below_post] = 0
        src_top2_idx[below_post] = -1
        src_top2_pt[below_post] = 0.0
        reassign_flag[below_post] = 0
        n_lost_threshold_post = int(below_post.sum())

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0.0
    src_count[~hlt_mask] = 0
    src_top2_idx[~hlt_mask] = -1
    src_top2_pt[~hlt_mask] = 0.0
    reassign_flag[~hlt_mask] = 0

    # Oracle synthetic supervision labels (training only).
    oracle_action = np.full((n_jets, max_part), -1, dtype=np.int8)
    oracle_tok_target = np.zeros((n_jets, max_part, 4), dtype=np.float32)
    oracle_tok_valid = np.zeros((n_jets, max_part), dtype=np.uint8)
    oracle_split_target = np.zeros((n_jets, max_part, 2, 4), dtype=np.float32)
    oracle_split_valid = np.zeros((n_jets, max_part, 2), dtype=np.uint8)

    unsmear_keep_thresh = float(max(unsmear_keep_thresh, 0.0))
    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        for i in valid.tolist():
            c = int(src_count[j, i])
            if c > 1:
                oracle_action[j, i] = 2  # split
                for k in range(2):
                    sidx = int(src_top2_idx[j, i, k])
                    if sidx >= 0 and bool(mask[j, sidx]):
                        oracle_split_target[j, i, k] = const[j, sidx]
                        oracle_split_valid[j, i, k] = 1
                continue

            sidx = int(src_top2_idx[j, i, 0])
            if sidx < 0 or (not bool(mask[j, sidx])):
                oracle_action[j, i] = 0
                continue

            tgt = const[j, sidx]
            oracle_tok_target[j, i] = tgt
            oracle_tok_valid[j, i] = 1

            if int(reassign_flag[j, i]) > 0:
                oracle_action[j, i] = 3
                continue

            h_pt = max(float(hlt[j, i, 0]), 1e-8)
            t_pt = max(float(tgt[0]), 1e-8)
            d_logpt = abs(np.log(h_pt) - np.log(t_pt))
            d_eta = abs(float(hlt[j, i, 1]) - float(tgt[1]))
            d_phi = abs(float(reco_base.wrap_phi_np(float(hlt[j, i, 2]) - float(tgt[2]))))
            score = d_logpt + 0.5 * d_eta + 0.5 * d_phi
            oracle_action[j, i] = 1 if score > unsmear_keep_thresh else 0

    stats = {
        "n_jets": int(n_jets),
        "n_initial": int(n_initial),
        "n_lost_threshold_pre": int(n_lost_threshold_pre),
        "n_merged_pairs": int(n_merged),
        "n_lost_eff": int(n_lost_eff),
        "n_reassigned": int(n_reassigned),
        "n_lost_threshold_post": int(n_lost_threshold_post),
        "n_final": int(hlt_mask.sum()),
        "avg_offline_per_jet": float(mask.sum(axis=1).mean()),
        "avg_hlt_per_jet": float(hlt_mask.sum(axis=1).mean()),
    }
    budget_truth = {
        "merge_lost_per_jet": merge_lost_per_jet.astype(np.float32),
        "eff_lost_per_jet": eff_lost_per_jet.astype(np.float32),
        "total_lost_per_jet": (merge_lost_per_jet + eff_lost_per_jet).astype(np.float32),
        "oracle_action": oracle_action.astype(np.int8),
        "oracle_tok_target": oracle_tok_target.astype(np.float32),
        "oracle_tok_valid": oracle_tok_valid.astype(np.uint8),
        "oracle_split_target": oracle_split_target.astype(np.float32),
        "oracle_split_valid": oracle_split_valid.astype(np.uint8),
        "oracle_src_count": src_count.astype(np.int16),
        "oracle_reassigned": reassign_flag.astype(np.uint8),
    }
    return hlt.astype(np.float32), hlt_mask.astype(bool), stats, budget_truth


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

    def _build_chamfer_target_weights() -> torch.Tensor:
        tgt_mask_f = mask_off.float()
        if not bool(loss_cfg.get("chamfer_topk_target_weighted", False)):
            return tgt_mask_f

        topk_k = int(max(loss_cfg.get("chamfer_topk_k", 0), 0))
        topk_w = float(max(loss_cfg.get("chamfer_topk_weight", 0.0), 0.0))
        if topk_k <= 0 or topk_w <= 0.0:
            return tgt_mask_f

        # Rank only valid offline targets by pT; padded slots are forced to the end.
        tgt_pt = const_off[..., 0].clamp(min=0.0)
        tgt_pt_rank = torch.where(mask_off, tgt_pt, torch.full_like(tgt_pt, -1e9))
        sorted_idx = torch.argsort(tgt_pt_rank, dim=1, descending=True)
        ranks = torch.argsort(sorted_idx, dim=1)
        topk_mask = (ranks < topk_k) & mask_off

        tgt_w = torch.ones_like(tgt_pt)
        tgt_w = torch.where(topk_mask, torch.full_like(tgt_w, topk_w), tgt_w)
        tgt_w = tgt_w * tgt_mask_f

        if bool(loss_cfg.get("chamfer_topk_mean_renorm", True)):
            denom = tgt_mask_f.sum(dim=1, keepdim=True).clamp(min=eps)
            mean_w = tgt_w.sum(dim=1, keepdim=True) / denom
            tgt_w = (tgt_w / mean_w.clamp(min=eps)) * tgt_mask_f
        return tgt_w

    def _loss_set_chamfer_vec() -> torch.Tensor:
        pred_to_tgt = cost.min(dim=2).values
        loss_pred_to_tgt = (w * pred_to_tgt).sum(dim=1) / (w.sum(dim=1) + eps)

        penalty = float(loss_cfg["unselected_penalty"]) * (1.0 - w).unsqueeze(2)
        tgt_to_pred = (cost + penalty).min(dim=1).values
        tgt_weights = _build_chamfer_target_weights()
        loss_tgt_to_pred = (tgt_to_pred * tgt_weights).sum(dim=1) / (tgt_weights.sum(dim=1) + eps)
        return loss_pred_to_tgt + loss_tgt_to_pred

    def _loss_set_sinkhorn_vec(sh_eps: float, sh_iters: int) -> torch.Tensor:
        tgt_mask_f = mask_off.float()
        a = w.clamp(min=0.0)
        a = a / a.sum(dim=1, keepdim=True).clamp(min=eps)
        b = tgt_mask_f
        b = b / b.sum(dim=1, keepdim=True).clamp(min=eps)

        sh_eps = max(float(sh_eps), 1e-4)
        sh_iters = max(int(sh_iters), 1)

        # Stabilized-ish Sinkhorn in primal form.
        k_raw = torch.exp(-cost / sh_eps)
        k_mat = torch.where(valid_tgt, k_raw.clamp(min=1e-12), torch.zeros_like(k_raw))
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(sh_iters):
            kv = torch.bmm(k_mat, v.unsqueeze(-1)).squeeze(-1).clamp(min=eps)
            u = a / kv
            ktu = torch.bmm(k_mat.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp(min=eps)
            v = b / ktu
        t_plan = u.unsqueeze(2) * k_mat * v.unsqueeze(1)
        return (t_plan * cost).sum(dim=(1, 2))

    def _loss_set_hungarian_vec() -> torch.Tensor:
        if not _HAS_SCIPY_HUNGARIAN:
            raise RuntimeError(
                "loss_set_mode='hungarian' requires scipy.optimize.linear_sum_assignment, "
                "but SciPy is unavailable in this environment."
            )
        # Hard assignment (detached matching indices) with differentiable selected costs.
        bsz = int(cost.shape[0])
        loss_list = []
        for bi in range(bsz):
            n_tgt = int(mask_off[bi].sum().item())
            if n_tgt <= 0:
                loss_list.append(torch.zeros((), device=cost.device, dtype=cost.dtype))
                continue
            c_bt = cost[bi, :, :n_tgt]
            c_np = c_bt.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(c_np)  # type: ignore[misc]
            row_t = torch.as_tensor(row_ind, device=cost.device, dtype=torch.long)
            col_t = torch.as_tensor(col_ind, device=cost.device, dtype=torch.long)

            matched_cost = c_bt[row_t, col_t]  # [n_match]
            # Coverage: ensure every target has a matched prediction.
            l_cov = matched_cost.mean()

            # Precision: reward low-cost matched predictions, penalize unmatched predicted mass.
            wb = w[bi]
            matched_mask = torch.zeros_like(wb, dtype=torch.bool)
            matched_mask[row_t] = True
            wb_m = wb[matched_mask]
            l_prec_match = (wb_m * matched_cost).sum() / (wb_m.sum() + eps)
            unmatched_mass = wb[~matched_mask].sum() / (wb.sum() + eps)
            l_fp = float(loss_cfg["unselected_penalty"]) * unmatched_mass
            loss_list.append(l_cov + l_prec_match + l_fp)
        return torch.stack(loss_list, dim=0)

    set_mode = str(loss_cfg.get("set_loss_mode", "chamfer")).strip().lower()
    if set_mode == "chamfer":
        loss_set_vec = _loss_set_chamfer_vec()
    elif set_mode in ("chamfer_sinkhorn", "sinkhorn_chamfer", "hybrid"):
        mix = float(np.clip(float(loss_cfg.get("chamfer_mix", 0.5)), 0.0, 1.0))
        sh_eps = float(loss_cfg.get("sinkhorn_eps", 0.08))
        sh_iters = int(loss_cfg.get("sinkhorn_iters", 25))
        loss_set_vec = mix * _loss_set_chamfer_vec() + (1.0 - mix) * _loss_set_sinkhorn_vec(sh_eps, sh_iters)
    elif set_mode in ("sinkhorn", "ot"):
        # Entropic OT family:
        # - sinkhorn: moderate epsilon
        # - ot: near-OT using a smaller epsilon + more iterations
        if set_mode == "ot":
            sh_eps = float(loss_cfg.get("ot_eps", 0.02))
            sh_iters = int(loss_cfg.get("ot_iters", 50))
        else:
            sh_eps = float(loss_cfg.get("sinkhorn_eps", 0.08))
            sh_iters = int(loss_cfg.get("sinkhorn_iters", 25))
        loss_set_vec = _loss_set_sinkhorn_vec(sh_eps, sh_iters)
    elif set_mode == "hungarian":
        loss_set_vec = _loss_set_hungarian_vec()
    elif set_mode == "combo":
        w_chamfer = max(float(loss_cfg.get("combo_w_chamfer", 0.0)), 0.0)
        w_sinkhorn = max(float(loss_cfg.get("combo_w_sinkhorn", 0.0)), 0.0)
        w_ot = max(float(loss_cfg.get("combo_w_ot", 0.0)), 0.0)
        w_hungarian = max(float(loss_cfg.get("combo_w_hungarian", 0.0)), 0.0)
        w_sum = w_chamfer + w_sinkhorn + w_ot + w_hungarian
        if w_sum <= 0.0:
            raise ValueError("loss_set_mode='combo' requires at least one positive combo weight.")
        w_chamfer /= w_sum
        w_sinkhorn /= w_sum
        w_ot /= w_sum
        w_hungarian /= w_sum

        loss_set_vec = torch.zeros(cost.shape[0], device=cost.device, dtype=cost.dtype)
        if w_chamfer > 0.0:
            loss_set_vec = loss_set_vec + w_chamfer * _loss_set_chamfer_vec()
        if w_sinkhorn > 0.0:
            loss_set_vec = loss_set_vec + w_sinkhorn * _loss_set_sinkhorn_vec(
                float(loss_cfg.get("sinkhorn_eps", 0.08)),
                int(loss_cfg.get("sinkhorn_iters", 25)),
            )
        if w_ot > 0.0:
            loss_set_vec = loss_set_vec + w_ot * _loss_set_sinkhorn_vec(
                float(loss_cfg.get("ot_eps", 0.02)),
                int(loss_cfg.get("ot_iters", 50)),
            )
        if w_hungarian > 0.0:
            loss_set_vec = loss_set_vec + w_hungarian * _loss_set_hungarian_vec()
    else:
        raise ValueError(f"Unsupported set_loss_mode: {set_mode}")

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
    pred_p = torch.sqrt(pred_px.pow(2) + pred_py.pow(2) + pred_pz.pow(2) + eps)
    true_p = torch.sqrt(true_px.pow(2) + true_py.pow(2) + true_pz.pow(2) + eps)
    pred_m2 = torch.clamp(pred_E.pow(2) - pred_p.pow(2), min=eps)
    true_m2 = torch.clamp(true_E.pow(2) - true_p.pow(2), min=eps)
    pred_m = torch.sqrt(pred_m2)
    true_m = torch.sqrt(true_m2)
    m_ratio = pred_m / (true_m + eps)
    loss_m_ratio_vec = F.smooth_l1_loss(m_ratio, torch.ones_like(m_ratio), reduction="none")
    e_ratio = pred_E / (true_E + eps)
    loss_e_ratio_vec = F.smooth_l1_loss(e_ratio, torch.ones_like(e_ratio), reduction="none")

    # Radial profile loss: compare normalized pT profile in ΔR bins around the offline jet axis.
    n_radial_bins = int(max(loss_cfg.get("radial_n_bins", 8), 1))
    radial_max_dr = float(max(loss_cfg.get("radial_max_dr", 1.0), 1e-3))
    axis_eta = torch.asinh(true_pz / (true_pt + eps))
    axis_phi = torch.atan2(true_py, true_px)

    pred_tok_pt = pred[:, :, 0].clamp(min=0.0) * w
    pred_tok_eta = pred[:, :, 1]
    pred_tok_phi = pred[:, :, 2]

    tgt_tok_pt = const_off[:, :, 0].clamp(min=0.0) * mask_off.float()
    tgt_tok_eta = const_off[:, :, 1]
    tgt_tok_phi = const_off[:, :, 2]

    def _delta_r(tok_eta: torch.Tensor, tok_phi: torch.Tensor) -> torch.Tensor:
        d_eta = tok_eta - axis_eta.unsqueeze(1)
        d_phi = torch.atan2(
            torch.sin(tok_phi - axis_phi.unsqueeze(1)),
            torch.cos(tok_phi - axis_phi.unsqueeze(1)),
        )
        return torch.sqrt(d_eta.pow(2) + d_phi.pow(2) + eps)

    dr_pred = _delta_r(pred_tok_eta, pred_tok_phi)
    dr_tgt = _delta_r(tgt_tok_eta, tgt_tok_phi)

    edges = torch.linspace(
        0.0,
        radial_max_dr,
        steps=n_radial_bins + 1,
        device=pred.device,
        dtype=pred.dtype,
    )
    lo = edges[:-1].view(1, 1, -1)
    hi = edges[1:].view(1, 1, -1)

    pred_in = (dr_pred.unsqueeze(-1) >= lo) & (dr_pred.unsqueeze(-1) < hi)
    tgt_in = (dr_tgt.unsqueeze(-1) >= lo) & (dr_tgt.unsqueeze(-1) < hi)
    # Include exact edge points at max ΔR in the final bin.
    pred_in[:, :, -1] = pred_in[:, :, -1] | (dr_pred >= radial_max_dr)
    tgt_in[:, :, -1] = tgt_in[:, :, -1] | (dr_tgt >= radial_max_dr)

    pred_prof = (pred_tok_pt.unsqueeze(-1) * pred_in.float()).sum(dim=1)
    tgt_prof = (tgt_tok_pt.unsqueeze(-1) * tgt_in.float()).sum(dim=1)

    pred_prof = pred_prof / (pred_prof.sum(dim=1, keepdim=True) + eps)
    tgt_prof = tgt_prof / (tgt_prof.sum(dim=1, keepdim=True) + eps)
    loss_radial_profile_vec = F.smooth_l1_loss(pred_prof, tgt_prof, reduction="none").mean(dim=1)

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

    # Asymmetric false-positive mass penalty:
    # penalize predicted mass that is far from any offline target token.
    fp_cost_thresh = float(loss_cfg.get("fp_mass_cost_thresh", 0.80))
    fp_cost_tau = float(max(loss_cfg.get("fp_mass_tau", 0.10), 1e-4))
    min_pred_cost = cost.min(dim=2).values
    unmatched_soft = torch.sigmoid((min_pred_cost - fp_cost_thresh) / fp_cost_tau)
    loss_fp_mass_vec = (w * unmatched_soft).sum(dim=1) / (w.sum(dim=1) + eps)

    # Oracle privileged supervision (study A/B/C/D/E), fed via global current batch.
    bsz = int(pred.shape[0])
    loss_oracle_action_vec = torch.zeros((bsz,), device=pred.device, dtype=pred.dtype)
    loss_oracle_tok_vec = torch.zeros((bsz,), device=pred.device, dtype=pred.dtype)
    loss_oracle_split_vec = torch.zeros((bsz,), device=pred.device, dtype=pred.dtype)
    global _ORACLE_WARNED_MISSING_BATCH
    variant = str(_ORACLE_STUDY_VARIANT).strip().upper()
    if variant not in ("A", "NONE"):
        ob = _ORACLE_CURRENT_BATCH
        if ob is None:
            if not _ORACLE_WARNED_MISSING_BATCH:
                print("[oracle-study] warning: oracle variant active but no oracle batch attached; skipping oracle losses.")
                _ORACLE_WARNED_MISSING_BATCH = True
        else:
            action_t = ob.get("oracle_action", None)
            tok_t = ob.get("oracle_tok_target", None)
            tok_v = ob.get("oracle_tok_valid", None)
            split_t = ob.get("oracle_split_target", None)
            split_v = ob.get("oracle_split_valid", None)
            if (
                action_t is not None
                and tok_t is not None
                and tok_v is not None
                and split_t is not None
                and split_v is not None
            ):
                action_t = action_t.to(pred.device).long()
                tok_t = tok_t.to(pred.device).float()
                tok_v = tok_v.to(pred.device).bool()
                split_t = split_t.to(pred.device).float()
                split_v = split_v.to(pred.device).float()

                L = int(action_t.shape[1])
                K_oracle = int(split_t.shape[2])
                tok_pred = out["cand_tokens"][:, :L, :]
                child_pred = out["cand_tokens"][:, L : L + (L * K_oracle), :].reshape(bsz, L, K_oracle, 4)
                action_prob = out["action_prob"].clamp(min=eps, max=1.0)

                valid_action = (action_t >= 0) & mask_hlt
                ce = F.nll_loss(
                    torch.log(action_prob + eps).permute(0, 2, 1),
                    action_t.clamp(min=0),
                    reduction="none",
                )
                loss_oracle_action_vec = (ce * valid_action.float()).sum(dim=1) / (
                    valid_action.float().sum(dim=1) + eps
                )

                nonsplit_mask = ((action_t == 0) | (action_t == 1) | (action_t == 3)) & tok_v & mask_hlt
                split_parent_mask = (action_t == 2) & mask_hlt

                pred_action = action_prob.argmax(dim=-1)
                if variant == "D":
                    agree = (pred_action == action_t) & (action_t >= 0) & mask_hlt
                    nonsplit_mask = nonsplit_mask & agree
                    split_parent_mask = split_parent_mask & agree

                tok_l1 = F.smooth_l1_loss(tok_pred, tok_t, reduction="none").mean(dim=2)
                loss_oracle_tok_vec = (tok_l1 * nonsplit_mask.float()).sum(dim=1) / (
                    nonsplit_mask.float().sum(dim=1) + eps
                )

                # Split child supervision (top-2 only) with permutation-invariant assignment.
                # split_v: [B,L,2] in {0,1}
                v0 = split_v[:, :, 0]
                v1 = split_v[:, :, 1]
                p0 = child_pred[:, :, 0, :]
                p1 = child_pred[:, :, 1, :]
                t0 = split_t[:, :, 0, :]
                t1 = split_t[:, :, 1, :]

                l00 = F.smooth_l1_loss(p0, t0, reduction="none").mean(dim=2)
                l11 = F.smooth_l1_loss(p1, t1, reduction="none").mean(dim=2)
                l01 = F.smooth_l1_loss(p0, t1, reduction="none").mean(dim=2)
                l10 = F.smooth_l1_loss(p1, t0, reduction="none").mean(dim=2)

                pair_a = v0 * l00 + v1 * l11
                pair_b = v0 * l01 + v1 * l10
                denom_split = (v0 + v1).clamp(min=1.0)
                split_per_parent = torch.minimum(pair_a, pair_b) / denom_split
                loss_oracle_split_vec = (split_per_parent * split_parent_mask.float()).sum(dim=1) / (
                    split_parent_mask.float().sum(dim=1) + eps
                )

    # Variant-specific oracle weights.
    w_oracle_action = float(max(loss_cfg.get("oracle_w_action", 0.0), 0.0))
    w_oracle_tok = float(max(loss_cfg.get("oracle_w_tok", 0.0), 0.0))
    w_oracle_split = float(max(loss_cfg.get("oracle_w_split", 0.0), 0.0))
    if variant in ("C",):
        w_oracle_action *= 3.0
    if variant in ("E",):
        w_oracle_action *= 3.0
        w_oracle_tok *= 2.0
        w_oracle_split *= 2.0

    total_vec = (
        float(loss_cfg["w_set"]) * loss_set_vec
        + float(loss_cfg["w_phys"]) * loss_phys_vec
        + float(loss_cfg["w_pt_ratio"]) * loss_pt_ratio_vec
        + float(loss_cfg.get("w_m_ratio", 0.0)) * loss_m_ratio_vec
        + float(loss_cfg["w_e_ratio"]) * loss_e_ratio_vec
        + float(loss_cfg.get("w_radial_profile", 0.0)) * loss_radial_profile_vec
        + float(loss_cfg["w_budget"]) * loss_budget_vec
        + float(loss_cfg["w_sparse"]) * loss_sparse_vec
        + float(loss_cfg["w_local"]) * loss_local_vec
        + float(loss_cfg.get("w_fp_mass", 0.0)) * loss_fp_mass_vec
        + w_oracle_action * loss_oracle_action_vec
        + w_oracle_tok * loss_oracle_tok_vec
        + w_oracle_split * loss_oracle_split_vec
    )

    return {
        "total": _weighted_batch_mean(total_vec, sw),
        "set": _weighted_batch_mean(loss_set_vec, sw),
        "phys": _weighted_batch_mean(loss_phys_vec, sw),
        "pt_ratio": _weighted_batch_mean(loss_pt_ratio_vec, sw),
        "m_ratio": _weighted_batch_mean(loss_m_ratio_vec, sw),
        "e_ratio": _weighted_batch_mean(loss_e_ratio_vec, sw),
        "radial_profile": _weighted_batch_mean(loss_radial_profile_vec, sw),
        "budget": _weighted_batch_mean(loss_budget_vec, sw),
        "sparse": _weighted_batch_mean(loss_sparse_vec, sw),
        "local": _weighted_batch_mean(loss_local_vec, sw),
        "fp_mass": _weighted_batch_mean(loss_fp_mass_vec, sw),
        "oracle_action": _weighted_batch_mean(loss_oracle_action_vec, sw),
        "oracle_tok": _weighted_batch_mean(loss_oracle_tok_vec, sw),
        "oracle_split": _weighted_batch_mean(loss_oracle_split_vec, sw),
    }


def enforce_unmerge_only_output(reco_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Hard-disable efficiency generation branch at output level.
    This keeps model architecture intact while ensuring no generated-eff constituents
    are used for training/eval/diagnostics in this script variant.
    """
    out = dict(reco_out)
    gen_w = out["gen_weight"] * 0.0
    out["gen_weight"] = gen_w
    out["budget_eff"] = out["budget_eff"] * 0.0

    # Zero eff flags and generated candidate weights in concatenated candidate set.
    if "cand_eff_flags" in out:
        out["cand_eff_flags"] = torch.zeros_like(out["cand_eff_flags"])

    if "cand_weights" in out and gen_w.ndim == 2:
        n_gen = int(gen_w.shape[1])
        if n_gen > 0 and out["cand_weights"].shape[1] >= n_gen:
            cw = out["cand_weights"].clone()
            cw[:, -n_gen:] = 0.0
            out["cand_weights"] = cw
    return out


def wrap_reconstructor_unmerge_only(model: OfflineReconstructor) -> OfflineReconstructor:
    """
    Wrap reconstructor forward so all downstream calls (including reconstruct_dataset)
    see unmerge-only outputs.
    """
    base_forward = model.forward

    def _wrapped_forward(*args, **kwargs):
        return enforce_unmerge_only_output(base_forward(*args, **kwargs))

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
        oracle_action: np.ndarray | None = None,
        oracle_tok_target: np.ndarray | None = None,
        oracle_tok_valid: np.ndarray | None = None,
        oracle_split_target: np.ndarray | None = None,
        oracle_split_valid: np.ndarray | None = None,
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

        if oracle_action is None:
            oracle_action = np.full((n, mask_hlt.shape[1]), -1, dtype=np.int8)
        if oracle_tok_target is None:
            oracle_tok_target = np.zeros((n, mask_hlt.shape[1], 4), dtype=np.float32)
        if oracle_tok_valid is None:
            oracle_tok_valid = np.zeros((n, mask_hlt.shape[1]), dtype=np.uint8)
        if oracle_split_target is None:
            oracle_split_target = np.zeros((n, mask_hlt.shape[1], 2, 4), dtype=np.float32)
        if oracle_split_valid is None:
            oracle_split_valid = np.zeros((n, mask_hlt.shape[1], 2), dtype=np.uint8)

        self.oracle_action = torch.tensor(oracle_action.astype(np.int64), dtype=torch.long)
        self.oracle_tok_target = torch.tensor(oracle_tok_target, dtype=torch.float32)
        self.oracle_tok_valid = torch.tensor(oracle_tok_valid.astype(np.bool_), dtype=torch.bool)
        self.oracle_split_target = torch.tensor(oracle_split_target, dtype=torch.float32)
        self.oracle_split_valid = torch.tensor(oracle_split_valid.astype(np.float32), dtype=torch.float32)

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
            "oracle_action": self.oracle_action[i],
            "oracle_tok_target": self.oracle_tok_target[i],
            "oracle_tok_valid": self.oracle_tok_valid[i],
            "oracle_split_target": self.oracle_split_target[i],
            "oracle_split_valid": self.oracle_split_valid[i],
        }


class WeightedReconstructionDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
        sample_weight_reco: np.ndarray | None = None,
        oracle_action: np.ndarray | None = None,
        oracle_tok_target: np.ndarray | None = None,
        oracle_tok_valid: np.ndarray | None = None,
        oracle_split_target: np.ndarray | None = None,
        oracle_split_valid: np.ndarray | None = None,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)
        n = feat_hlt.shape[0]

        if oracle_action is None:
            oracle_action = np.full((n, mask_hlt.shape[1]), -1, dtype=np.int8)
        if oracle_tok_target is None:
            oracle_tok_target = np.zeros((n, mask_hlt.shape[1], 4), dtype=np.float32)
        if oracle_tok_valid is None:
            oracle_tok_valid = np.zeros((n, mask_hlt.shape[1]), dtype=np.uint8)
        if oracle_split_target is None:
            oracle_split_target = np.zeros((n, mask_hlt.shape[1], 2, 4), dtype=np.float32)
        if oracle_split_valid is None:
            oracle_split_valid = np.zeros((n, mask_hlt.shape[1], 2), dtype=np.uint8)
        self.oracle_action = torch.tensor(oracle_action.astype(np.int64), dtype=torch.long)
        self.oracle_tok_target = torch.tensor(oracle_tok_target, dtype=torch.float32)
        self.oracle_tok_valid = torch.tensor(oracle_tok_valid.astype(np.bool_), dtype=torch.bool)
        self.oracle_split_target = torch.tensor(oracle_split_target, dtype=torch.float32)
        self.oracle_split_valid = torch.tensor(oracle_split_valid.astype(np.float32), dtype=torch.float32)

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
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
            "sample_weight_reco": self.sample_weight_reco[i],
            "oracle_action": self.oracle_action[i],
            "oracle_tok_target": self.oracle_tok_target[i],
            "oracle_tok_valid": self.oracle_tok_valid[i],
            "oracle_split_target": self.oracle_split_target[i],
            "oracle_split_valid": self.oracle_split_valid[i],
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
    if not bool(_ORACLE_STUDY_ENABLE_GEN):
        reco_out = enforce_unmerge_only_output(reco_out)
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


def _apply_oracle_intervention_to_reco_out(
    reco_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    mode: str,
) -> Dict[str, torch.Tensor]:
    mode = str(mode).strip().upper()
    if mode in ("", "NONE", "A"):
        return reco_out

    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in reco_out.items()}
    act_t = batch["oracle_action"].to(out["action_prob"].device).long()
    tok_t = batch["oracle_tok_target"].to(out["action_prob"].device).float()
    tok_v = batch["oracle_tok_valid"].to(out["action_prob"].device).bool()
    split_t = batch["oracle_split_target"].to(out["action_prob"].device).float()
    split_v = batch["oracle_split_valid"].to(out["action_prob"].device).float()
    mask_hlt = batch["mask_hlt"].to(out["action_prob"].device).bool()

    B, L, _ = out["action_prob"].shape
    K = int(split_t.shape[2])
    n_child = int(L * K)

    tok_tokens = out["cand_tokens"][:, :L, :].clone()
    child_tokens = out["cand_tokens"][:, L : L + n_child, :].reshape(B, L, K, 4).clone()
    gen_tokens = out["cand_tokens"][:, L + n_child :, :].clone()

    tok_w = out["cand_weights"][:, :L].clone()
    child_w = out["child_weight"].reshape(B, L, K).clone()
    gen_w = out["gen_weight"].clone()

    pred_action = out["action_prob"].argmax(dim=-1)
    oracle_valid = (act_t >= 0) & mask_hlt
    agree = oracle_valid & (pred_action == act_t)

    if mode == "C":
        # Oracle routing only; keep model parameters.
        nonsplit = oracle_valid & (act_t != 2)
        split_parent = oracle_valid & (act_t == 2)
        tok_w = nonsplit.float()
        child_w = torch.where(split_parent.unsqueeze(-1), child_w, torch.zeros_like(child_w))
    elif mode == "D":
        # Oracle parameters only where routing already agrees.
        nonsplit = agree & ((act_t == 0) | (act_t == 1) | (act_t == 3)) & tok_v
        tok_tokens = torch.where(nonsplit.unsqueeze(-1), tok_t, tok_tokens)
        split_parent = agree & (act_t == 2)
        sv = split_v.bool()
        child_tokens = torch.where((split_parent.unsqueeze(-1).unsqueeze(-1) & sv.unsqueeze(-1)), split_t, child_tokens)
    elif mode == "E":
        # Full oracle action+params intervention.
        nonsplit = oracle_valid & ((act_t == 0) | (act_t == 1) | (act_t == 3)) & tok_v
        split_parent = oracle_valid & (act_t == 2)
        tok_tokens = torch.where(nonsplit.unsqueeze(-1), tok_t, tok_tokens)
        sv = split_v.bool()
        child_tokens = torch.where((split_parent.unsqueeze(-1).unsqueeze(-1) & sv.unsqueeze(-1)), split_t, child_tokens)
        tok_w = nonsplit.float()
        child_w = torch.where(split_parent.unsqueeze(-1), split_v, torch.zeros_like(child_w))
    else:
        return reco_out

    out["child_weight"] = child_w.reshape(B, n_child)
    out["gen_weight"] = gen_w
    out["cand_tokens"] = torch.cat([tok_tokens, child_tokens.reshape(B, n_child, 4), gen_tokens], dim=1)
    out["cand_weights"] = torch.cat([tok_w, out["child_weight"], gen_w], dim=1)
    return out


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
def eval_joint_model_oracle_intervention(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    intervention_mode: str,
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
        reco_out = _apply_oracle_intervention_to_reco_out(reco_out, batch, intervention_mode)
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


def train_reconstructor_weighted(
    model: OfflineReconstructor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
    apply_reco_weight: bool,
    reload_best_at_stage_transition: bool,
) -> Tuple[OfflineReconstructor, Dict[str, float]]:
    global _ORACLE_CURRENT_BATCH
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    best_state = None
    best_val = 1e9
    no_improve = 0
    best_metrics: Dict[str, float] = {}
    min_stop_epoch = int(train_cfg.get("stage2_epochs", 0)) + int(train_cfg.get("min_full_scale_epochs", 5))
    val_metric_source = "weighted" if bool(apply_reco_weight) else "unweighted"

    total_epochs = int(train_cfg["epochs"])
    stage1_end = int(max(int(train_cfg.get("stage1_epochs", 0)), 0))
    stage2_end = int(max(int(train_cfg.get("stage2_epochs", stage1_end)), stage1_end))
    stage1_end = min(stage1_end, total_epochs)
    stage2_end = min(max(stage2_end, stage1_end), total_epochs)

    def _phase_idx(epoch: int) -> int:
        if epoch < stage1_end:
            return 0
        if epoch < stage2_end:
            return 1
        return 2

    def _scheduled_loss_cfg(phase_idx: int) -> Dict:
        cfg_eff = dict(loss_cfg)
        if not bool(train_cfg.get("loss_schedule_enable", False)):
            return cfg_eff
        if phase_idx == 0:
            cfg_eff["w_set"] = float(cfg_eff.get("w_set", 0.0)) * float(
                train_cfg.get("loss_sched_early_set_mult", 1.0)
            )
            cfg_eff["w_budget"] = float(cfg_eff.get("w_budget", 0.0)) * float(
                train_cfg.get("loss_sched_early_budget_mult", 1.0)
            )
        elif phase_idx == 2:
            cfg_eff["w_pt_ratio"] = float(cfg_eff.get("w_pt_ratio", 0.0)) * float(
                train_cfg.get("loss_sched_late_ptratio_mult", 1.0)
            )
            cfg_eff["w_local"] = float(cfg_eff.get("w_local", 0.0)) * float(
                train_cfg.get("loss_sched_late_local_mult", 1.0)
            )
        return cfg_eff

    phase_names = {
        0: "phase_035",
        1: "phase_070",
        2: "phase_100",
    }
    current_phase = _phase_idx(0)
    phase_best: Dict[int, Dict[str, object]] = {}

    for ep in tqdm(range(int(train_cfg["epochs"])), desc="Reconstructor"):
        phase_idx = _phase_idx(ep)
        if ep > 0 and phase_idx != current_phase:
            if bool(reload_best_at_stage_transition):
                prev_best = phase_best.get(current_phase, None)
                if prev_best is not None:
                    state_pack = prev_best["state"]
                    model.load_state_dict(state_pack["model"])
                    opt.load_state_dict(state_pack["opt"])
                    sch.load_state_dict(state_pack["sch"])
                    no_improve = 0
                    print(
                        f"Stage-A transition -> reloaded best from {phase_names[current_phase]} "
                        f"(epoch={int(prev_best['epoch'])+1}, val_total={float(prev_best['val_total']):.4f})"
                    )
                else:
                    print(
                        f"Stage-A transition -> no stored best for {phase_names[current_phase]}, "
                        "continuing with current weights."
                    )
            current_phase = phase_idx

        model.train()
        sc = stage_scale_local(ep, train_cfg)
        loss_cfg_ep = _scheduled_loss_cfg(phase_idx)

        tr_total = tr_set = tr_phys = tr_pt_ratio = tr_m_ratio = tr_e_ratio = tr_radial = tr_budget = tr_sparse = tr_local = tr_fp_mass = 0.0
        n_tr = 0
        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            budget_merge_true = batch["budget_merge_true"].to(device)
            budget_eff_true = batch["budget_eff_true"].to(device)
            sw_reco = batch.get("sample_weight_reco", None)
            if sw_reco is not None:
                sw_reco = sw_reco.to(device)

            opt.zero_grad()
            out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=sc)
            _ORACLE_CURRENT_BATCH = batch
            losses = compute_reconstruction_losses_weighted(
                out,
                const_hlt,
                mask_hlt,
                const_off,
                mask_off,
                budget_merge_true,
                budget_eff_true,
                loss_cfg_ep,
                sample_weight=(sw_reco if (bool(apply_reco_weight) and sw_reco is not None) else None),
            )
            _ORACLE_CURRENT_BATCH = None
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = feat_hlt.size(0)
            tr_total += losses["total"].item() * bs
            tr_set += losses["set"].item() * bs
            tr_phys += losses["phys"].item() * bs
            tr_pt_ratio += losses["pt_ratio"].item() * bs
            tr_m_ratio += losses["m_ratio"].item() * bs
            tr_e_ratio += losses["e_ratio"].item() * bs
            tr_radial += losses["radial_profile"].item() * bs
            tr_budget += losses["budget"].item() * bs
            tr_sparse += losses["sparse"].item() * bs
            tr_local += losses["local"].item() * bs
            tr_fp_mass += losses["fp_mass"].item() * bs
            n_tr += bs

        model.eval()
        va_total_u = va_set_u = va_phys_u = va_pt_ratio_u = va_m_ratio_u = va_e_ratio_u = va_radial_u = va_budget_u = va_sparse_u = va_local_u = va_fp_mass_u = 0.0
        va_total_w = va_set_w = va_phys_w = va_pt_ratio_w = va_m_ratio_w = va_e_ratio_w = va_radial_w = va_budget_w = va_sparse_w = va_local_w = va_fp_mass_w = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                const_off = batch["const_off"].to(device)
                mask_off = batch["mask_off"].to(device)
                budget_merge_true = batch["budget_merge_true"].to(device)
                budget_eff_true = batch["budget_eff_true"].to(device)
                sw_reco = batch.get("sample_weight_reco", None)
                if sw_reco is not None:
                    sw_reco = sw_reco.to(device)

                out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
                _ORACLE_CURRENT_BATCH = batch
                losses_u = compute_reconstruction_losses_weighted(
                    out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    budget_merge_true,
                    budget_eff_true,
                    loss_cfg_ep,
                    sample_weight=None,
                )
                if bool(apply_reco_weight) and sw_reco is not None:
                    losses_w = compute_reconstruction_losses_weighted(
                        out,
                        const_hlt,
                        mask_hlt,
                        const_off,
                        mask_off,
                        budget_merge_true,
                        budget_eff_true,
                        loss_cfg_ep,
                        sample_weight=sw_reco,
                    )
                else:
                    losses_w = losses_u
                _ORACLE_CURRENT_BATCH = None

                bs = feat_hlt.size(0)
                va_total_u += losses_u["total"].item() * bs
                va_set_u += losses_u["set"].item() * bs
                va_phys_u += losses_u["phys"].item() * bs
                va_pt_ratio_u += losses_u["pt_ratio"].item() * bs
                va_m_ratio_u += losses_u["m_ratio"].item() * bs
                va_e_ratio_u += losses_u["e_ratio"].item() * bs
                va_radial_u += losses_u["radial_profile"].item() * bs
                va_budget_u += losses_u["budget"].item() * bs
                va_sparse_u += losses_u["sparse"].item() * bs
                va_local_u += losses_u["local"].item() * bs
                va_fp_mass_u += losses_u["fp_mass"].item() * bs

                va_total_w += losses_w["total"].item() * bs
                va_set_w += losses_w["set"].item() * bs
                va_phys_w += losses_w["phys"].item() * bs
                va_pt_ratio_w += losses_w["pt_ratio"].item() * bs
                va_m_ratio_w += losses_w["m_ratio"].item() * bs
                va_e_ratio_w += losses_w["e_ratio"].item() * bs
                va_radial_w += losses_w["radial_profile"].item() * bs
                va_budget_w += losses_w["budget"].item() * bs
                va_sparse_w += losses_w["sparse"].item() * bs
                va_local_w += losses_w["local"].item() * bs
                va_fp_mass_w += losses_w["fp_mass"].item() * bs
                n_va += bs

        sch.step()
        tr_total /= max(n_tr, 1)
        tr_set /= max(n_tr, 1)
        tr_phys /= max(n_tr, 1)
        tr_pt_ratio /= max(n_tr, 1)
        tr_m_ratio /= max(n_tr, 1)
        tr_e_ratio /= max(n_tr, 1)
        tr_radial /= max(n_tr, 1)
        tr_budget /= max(n_tr, 1)
        tr_sparse /= max(n_tr, 1)
        tr_local /= max(n_tr, 1)
        tr_fp_mass /= max(n_tr, 1)

        va_total_u /= max(n_va, 1)
        va_set_u /= max(n_va, 1)
        va_phys_u /= max(n_va, 1)
        va_pt_ratio_u /= max(n_va, 1)
        va_m_ratio_u /= max(n_va, 1)
        va_e_ratio_u /= max(n_va, 1)
        va_radial_u /= max(n_va, 1)
        va_budget_u /= max(n_va, 1)
        va_sparse_u /= max(n_va, 1)
        va_local_u /= max(n_va, 1)
        va_fp_mass_u /= max(n_va, 1)

        va_total_w /= max(n_va, 1)
        va_set_w /= max(n_va, 1)
        va_phys_w /= max(n_va, 1)
        va_pt_ratio_w /= max(n_va, 1)
        va_m_ratio_w /= max(n_va, 1)
        va_e_ratio_w /= max(n_va, 1)
        va_radial_w /= max(n_va, 1)
        va_budget_w /= max(n_va, 1)
        va_sparse_w /= max(n_va, 1)
        va_local_w /= max(n_va, 1)
        va_fp_mass_w /= max(n_va, 1)

        val_total_sel = float(va_total_w) if bool(apply_reco_weight) else float(va_total_u)

        phase_entry = phase_best.get(phase_idx, None)
        if (phase_entry is None) or (val_total_sel < float(phase_entry["val_total"])):
            phase_best[phase_idx] = {
                "val_total": float(val_total_sel),
                "epoch": int(ep),
                "state": {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "opt": copy.deepcopy(opt.state_dict()),
                    "sch": copy.deepcopy(sch.state_dict()),
                },
            }

        if val_total_sel < best_val:
            best_val = val_total_sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_metrics = {
                "val_metric_source": str(val_metric_source),
                "selected_val_total": float(val_total_sel),
                "val_total": float(val_total_sel),
                "val_total_unweighted": float(va_total_u),
                "val_set_unweighted": float(va_set_u),
                "val_phys_unweighted": float(va_phys_u),
                "val_pt_ratio_unweighted": float(va_pt_ratio_u),
                "val_m_ratio_unweighted": float(va_m_ratio_u),
                "val_e_ratio_unweighted": float(va_e_ratio_u),
                "val_radial_profile_unweighted": float(va_radial_u),
                "val_budget_unweighted": float(va_budget_u),
                "val_sparse_unweighted": float(va_sparse_u),
                "val_local_unweighted": float(va_local_u),
                "val_fp_mass_unweighted": float(va_fp_mass_u),
                "val_total_weighted": float(va_total_w),
                "val_set_weighted": float(va_set_w),
                "val_phys_weighted": float(va_phys_w),
                "val_pt_ratio_weighted": float(va_pt_ratio_w),
                "val_m_ratio_weighted": float(va_m_ratio_w),
                "val_e_ratio_weighted": float(va_e_ratio_w),
                "val_radial_profile_weighted": float(va_radial_w),
                "val_budget_weighted": float(va_budget_w),
                "val_sparse_weighted": float(va_sparse_w),
                "val_local_weighted": float(va_local_w),
                "val_fp_mass_weighted": float(va_fp_mass_w),
            }
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Ep {ep+1}: train_total={tr_total:.4f}, "
                f"val_total_unw={va_total_u:.4f}, val_total_w={va_total_w:.4f}, "
                f"select={val_metric_source}, best_sel={best_val:.4f} | "
                f"set_unw={va_set_u:.4f}, phys_unw={va_phys_u:.4f}, "
                f"budget_unw={va_budget_u:.4f}, fp_mass_unw={va_fp_mass_u:.4f}, "
                f"w_set={float(loss_cfg_ep.get('w_set', 0.0)):.3f}, "
                f"w_budget={float(loss_cfg_ep.get('w_budget', 0.0)):.3f}, "
                f"w_pt={float(loss_cfg_ep.get('w_pt_ratio', 0.0)):.3f}, "
                f"w_local={float(loss_cfg_ep.get('w_local', 0.0)):.3f}, "
                f"stage_scale={sc:.2f}"
            )
        if (ep + 1) >= min_stop_epoch and no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping reconstructor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if len(best_metrics) == 0:
        best_metrics = {}
    best_metrics["stageA_stagewise_best_reload"] = bool(reload_best_at_stage_transition)
    best_metrics["stageA_phase_best"] = {
        phase_names[int(k)]: {
            "epoch": int(v["epoch"]) + 1,
            "val_total": float(v["val_total"]),
        }
        for k, v in sorted(phase_best.items(), key=lambda kv: int(kv[0]))
    }
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
    lambda_delta_cls: float = 0.0,
    delta_tau: float = 0.05,
    delta_lambda_fp: float = 3.0,
    delta_hlt_model: Optional[nn.Module] = None,
    delta_hlt_threshold_prob: float = 0.50,
    delta_warmup_epochs: int = 0,
    progressive_unfreeze: bool = False,
    unfreeze_phase1_epochs: int = 3,
    unfreeze_phase2_epochs: int = 7,
    unfreeze_last_n_encoder_layers: int = 2,
    alternate_freeze: bool = False,
    alternate_reco_only_epochs: int = 5,
    alternate_dual_only_epochs: int = 5,
    lambda_param_anchor: float = 0.0,
    lambda_output_anchor: float = 0.0,
    anchor_decay: float = 1.0,
) -> Tuple[OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    global _ORACLE_CURRENT_BATCH
    def _set_reco_trainability(mode: str) -> int:
        for p in reconstructor.parameters():
            p.requires_grad_(False)

        if mode == "frozen":
            return 0

        if mode == "all":
            for p in reconstructor.parameters():
                p.requires_grad_(True)
            return sum(int(p.numel()) for p in reconstructor.parameters())

        trainable_prefixes: List[str] = [
            "token_norm",
            "action_head",
            "unsmear_head",
            "reassign_head",
            "split_exist_head",
            "split_delta_head",
            "pool_attn",
            "budget_head",
            "gen_attn",
            "gen_norm",
            "gen_head",
            "gen_exist_head",
        ]
        if mode == "lastk":
            n_layers = len(reconstructor.encoder_layers)
            k = int(max(0, min(int(unfreeze_last_n_encoder_layers), n_layers)))
            for idx in range(max(0, n_layers - k), n_layers):
                trainable_prefixes.append(f"encoder_layers.{idx}")

        trainable = 0
        for name, p in reconstructor.named_parameters():
            if any(name.startswith(pref) for pref in trainable_prefixes):
                p.requires_grad_(True)
                trainable += int(p.numel())
        return trainable

    def _set_dual_trainability(trainable: bool) -> int:
        n = 0
        for p in dual_model.parameters():
            p.requires_grad_(bool(trainable))
            if bool(trainable):
                n += int(p.numel())
        return n

    for p in reconstructor.parameters():
        p.requires_grad_(not freeze_reconstructor)
    for p in dual_model.parameters():
        p.requires_grad_(True)

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

    delta_tau = float(max(delta_tau, 1e-6))
    delta_lambda_fp = float(max(delta_lambda_fp, 0.0))
    lambda_delta_cls = float(max(lambda_delta_cls, 0.0))
    use_delta = bool(lambda_delta_cls > 0.0 and delta_hlt_model is not None)
    if use_delta:
        delta_hlt_model.eval()
        for p in delta_hlt_model.parameters():
            p.requires_grad_(False)

    lambda_param_anchor = float(max(lambda_param_anchor, 0.0))
    lambda_output_anchor = float(max(lambda_output_anchor, 0.0))
    anchor_decay = float(max(anchor_decay, 0.0))
    use_anchor = bool(
        (not freeze_reconstructor)
        and (lambda_param_anchor > 0.0 or lambda_output_anchor > 0.0)
    )
    reco_anchor_copy = None
    reco_anchor_state = None
    if use_anchor:
        reco_anchor_copy = copy.deepcopy(reconstructor).to(device)
        reco_anchor_copy.eval()
        for p in reco_anchor_copy.parameters():
            p.requires_grad_(False)
        reco_anchor_state = {
            k: v.detach().clone().to(device)
            for k, v in reconstructor.state_dict().items()
            if torch.is_tensor(v)
        }

    alt_reco_epochs_i = int(max(alternate_reco_only_epochs, 0))
    alt_dual_epochs_i = int(max(alternate_dual_only_epochs, 0))
    alt_cycle_len = int(alt_reco_epochs_i + alt_dual_epochs_i)
    alt_schedule_enabled = bool(
        (not freeze_reconstructor)
        and bool(alternate_freeze)
        and alt_reco_epochs_i > 0
        and alt_dual_epochs_i > 0
        and alt_cycle_len > 0
    )
    progressive_unfreeze_active = bool(progressive_unfreeze) and (not alt_schedule_enabled)
    if alt_schedule_enabled and bool(progressive_unfreeze):
        print(
            "Note: Stage-C alternating freeze is enabled; "
            "ignoring --stageC_progressive_unfreeze schedule."
        )

    for ep in tqdm(range(int(epochs)), desc=stage_name):
        current_reco_frozen = bool(freeze_reconstructor)
        current_dual_frozen = False

        if freeze_reconstructor:
            dual_model.train()
            reconstructor.eval()
            current_unfreeze_mode = "frozen"
            n_trainable_reco = 0
            n_trainable_dual = _set_dual_trainability(True)
        else:
            if alt_schedule_enabled:
                cycle_pos = int(ep % alt_cycle_len)
                if cycle_pos < alt_reco_epochs_i:
                    # Reco-only phase: freeze dual head, optimize reconstructor through fixed dual path.
                    dual_model.eval()
                    n_trainable_dual = _set_dual_trainability(False)
                    reconstructor.train()
                    n_trainable_reco = _set_reco_trainability("all")
                    current_unfreeze_mode = "alt_reco_only"
                    current_reco_frozen = False
                    current_dual_frozen = True
                else:
                    # Dual-only phase: freeze reconstructor, train dual head.
                    dual_model.train()
                    n_trainable_dual = _set_dual_trainability(True)
                    reconstructor.eval()
                    n_trainable_reco = _set_reco_trainability("frozen")
                    current_unfreeze_mode = "alt_dual_only"
                    current_reco_frozen = True
                    current_dual_frozen = False
            else:
                dual_model.train()
                n_trainable_dual = _set_dual_trainability(True)
                reconstructor.train()
                current_reco_frozen = False
                if bool(progressive_unfreeze_active):
                    if (ep + 1) <= int(max(unfreeze_phase1_epochs, 0)):
                        current_unfreeze_mode = "heads"
                    elif (ep + 1) <= int(max(unfreeze_phase2_epochs, 0)):
                        current_unfreeze_mode = "lastk"
                    else:
                        current_unfreeze_mode = "all"
                    n_trainable_reco = _set_reco_trainability(current_unfreeze_mode)
                else:
                    current_unfreeze_mode = "all"
                    n_trainable_reco = _set_reco_trainability("all")

        tr_loss = 0.0
        tr_cls = 0.0
        tr_rank = 0.0
        tr_reco = 0.0
        tr_cons = 0.0
        tr_delta = 0.0
        tr_delta_gain = 0.0
        tr_delta_cost = 0.0
        tr_anchor_param = 0.0
        tr_anchor_out = 0.0
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

            if current_reco_frozen:
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

            loss_delta = torch.zeros((), device=device)
            delta_gain = torch.zeros((), device=device)
            delta_cost = torch.zeros((), device=device)
            if use_delta:
                with torch.no_grad():
                    logits_hlt = delta_hlt_model(feat_hlt_reco, mask_hlt).squeeze(1)
                    p_hlt = torch.sigmoid(logits_hlt)
                    h_soft = torch.sigmoid((p_hlt - float(delta_hlt_threshold_prob)) / float(delta_tau))
                y_f = y.float().view(-1)
                p_joint = torch.sigmoid(logits).view(-1)
                miss_hlt = (1.0 - h_soft).view(-1)
                delta_gain = (y_f * miss_hlt * p_joint).mean()
                delta_cost = ((1.0 - y_f) * miss_hlt * p_joint).mean()
                loss_delta = -delta_gain + float(delta_lambda_fp) * delta_cost
                warm = int(max(delta_warmup_epochs, 0))
                if warm > 0:
                    ramp = min(1.0, float(ep + 1) / float(warm))
                else:
                    ramp = 1.0
            else:
                ramp = 1.0

            if float(lambda_reco) > 0.0:
                _ORACLE_CURRENT_BATCH = batch
                reco_losses = compute_reconstruction_losses_weighted(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_merge,
                    b_eff,
                    BASE_CONFIG["loss"],
                    sample_weight=(sw_reco if (bool(apply_reco_weight) and sw_reco is not None) else None),
                )
                _ORACLE_CURRENT_BATCH = None
                loss_reco = reco_losses["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            if use_anchor:
                if reco_anchor_copy is None or reco_anchor_state is None:
                    raise RuntimeError("Anchor mode enabled but anchor references are missing.")
                with torch.no_grad():
                    out_anchor = reco_anchor_copy(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
                loss_anchor_out = (
                    F.mse_loss(reco_out["cand_weights"], out_anchor["cand_weights"])
                    + F.mse_loss(reco_out["budget_total"], out_anchor["budget_total"])
                    + F.mse_loss(reco_out["budget_merge"], out_anchor["budget_merge"])
                    + F.mse_loss(reco_out["budget_eff"], out_anchor["budget_eff"])
                )
                loss_anchor_param = torch.zeros((), device=device)
                if lambda_param_anchor > 0.0:
                    for name, p in reconstructor.named_parameters():
                        if p.requires_grad:
                            ref = reco_anchor_state[name]
                            loss_anchor_param = loss_anchor_param + F.mse_loss(p, ref)
                anchor_scale = float(anchor_decay) ** float(ep)
            else:
                loss_anchor_out = torch.zeros((), device=device)
                loss_anchor_param = torch.zeros((), device=device)
                anchor_scale = 1.0

            loss = (
                loss_cls
                + float(lambda_rank) * loss_rank
                + float(lambda_reco) * loss_reco
                + float(lambda_cons) * loss_cons
                + float(lambda_delta_cls) * float(ramp) * loss_delta
                + float(anchor_scale) * float(lambda_param_anchor) * loss_anchor_param
                + float(anchor_scale) * float(lambda_output_anchor) * loss_anchor_out
            )
            loss.backward()
            if not current_dual_frozen:
                torch.nn.utils.clip_grad_norm_(dual_model.parameters(), 1.0)
            if not current_reco_frozen:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = feat_hlt_reco.size(0)
            tr_loss += loss.item() * bs
            tr_cls += loss_cls.item() * bs
            tr_rank += loss_rank.item() * bs
            tr_reco += loss_reco.item() * bs
            tr_cons += loss_cons.item() * bs
            tr_delta += loss_delta.item() * bs
            tr_delta_gain += delta_gain.item() * bs
            tr_delta_cost += delta_cost.item() * bs
            tr_anchor_param += loss_anchor_param.item() * bs
            tr_anchor_out += loss_anchor_out.item() * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)
        tr_delta /= max(n_tr, 1)
        tr_delta_gain /= max(n_tr, 1)
        tr_delta_cost /= max(n_tr, 1)
        tr_anchor_param /= max(n_tr, 1)
        tr_anchor_out /= max(n_tr, 1)

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
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}, "
                f"delta={tr_delta:.4f}, d_gain={tr_delta_gain:.4f}, d_cost={tr_delta_cost:.4f}, "
                f"anc_p={tr_anchor_param:.4f}, anc_o={tr_anchor_out:.4f}) | "
                f"val_auc_unw={va_auc_unw:.4f}, val_fpr50_unw={va_fpr50_unw:.6f}, "
                f"val_auc_w={va_auc_w:.4f}, val_fpr50_w={va_fpr50_w:.6f}, "
                f"val_metric_source={metric_source_epoch}, "
                f"select={str(select_metric).lower()}, best_sel={best_sel_score:.6f}, "
                f"unfreeze={current_unfreeze_mode}, reco_trainable={n_trainable_reco}, "
                f"dual_frozen={current_dual_frozen}, dual_trainable={n_trainable_dual}"
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
        "delta_enabled": bool(use_delta),
        "delta_lambda": float(lambda_delta_cls),
        "delta_tau": float(delta_tau),
        "delta_lambda_fp": float(delta_lambda_fp),
        "delta_hlt_threshold_prob": float(delta_hlt_threshold_prob),
        "delta_warmup_epochs": int(max(delta_warmup_epochs, 0)),
        "progressive_unfreeze": bool(progressive_unfreeze_active),
        "unfreeze_phase1_epochs": int(unfreeze_phase1_epochs),
        "unfreeze_phase2_epochs": int(unfreeze_phase2_epochs),
        "unfreeze_last_n_encoder_layers": int(unfreeze_last_n_encoder_layers),
        "alternate_freeze": bool(alt_schedule_enabled),
        "alternate_reco_only_epochs": int(alt_reco_epochs_i),
        "alternate_dual_only_epochs": int(alt_dual_epochs_i),
        "lambda_param_anchor": float(lambda_param_anchor),
        "lambda_output_anchor": float(lambda_output_anchor),
        "anchor_decay": float(anchor_decay),
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
    parser.add_argument("--n_train_jets", type=int, default=50000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=80)
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

    # Stage A (reconstructor pretrain)
    parser.add_argument("--stageA_epochs", type=int, default=90)
    parser.add_argument("--stageA_patience", type=int, default=18)
    parser.add_argument(
        "--disable_stageA_stagewise_best_reload",
        action="store_true",
        help="Disable reloading the best Stage-A validation checkpoint at each stage-scale transition.",
    )
    parser.add_argument(
        "--stageA_loss_schedule",
        action="store_true",
        help="Enable Stage-A phase-scheduled reconstruction loss weights.",
    )
    parser.add_argument(
        "--stageA_loss_sched_early_set_mult",
        type=float,
        default=1.30,
        help="Phase-1 multiplier for set loss weight.",
    )
    parser.add_argument(
        "--stageA_loss_sched_early_budget_mult",
        type=float,
        default=1.30,
        help="Phase-1 multiplier for budget loss weight.",
    )
    parser.add_argument(
        "--stageA_loss_sched_late_ptratio_mult",
        type=float,
        default=1.15,
        help="Phase-3 multiplier for pT-ratio loss weight.",
    )
    parser.add_argument(
        "--stageA_loss_sched_late_local_mult",
        type=float,
        default=1.15,
        help="Phase-3 multiplier for local loss weight.",
    )

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
    parser.add_argument("--oracle_study_variant", type=str, default="A", choices=["A", "B", "C", "D", "E", "B_gen"])
    parser.add_argument("--oracle_unsmear_keep_thresh", type=float, default=0.03)
    parser.add_argument("--oracle_w_action", type=float, default=0.15)
    parser.add_argument("--oracle_w_tok", type=float, default=0.20)
    parser.add_argument("--oracle_w_split", type=float, default=0.20)
    parser.add_argument("--oracle_eval_interventions_for_A", action="store_true")

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
    parser.add_argument("--stageC_lambda_delta", type=float, default=0.0)
    parser.add_argument("--stageC_delta_tau", type=float, default=0.05)
    parser.add_argument("--stageC_delta_lambda_fp", type=float, default=3.0)
    parser.add_argument("--stageC_delta_warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--stageC_init_reco_ckpt",
        type=str,
        default="",
        help="Optional path to reconstructor checkpoint to load immediately before Stage-C.",
    )
    parser.add_argument(
        "--stageC_init_dual_ckpt",
        type=str,
        default="",
        help="Optional path to dual-view tagger checkpoint to load immediately before Stage-C.",
    )
    parser.add_argument(
        "--stageC_progressive_unfreeze",
        action="store_true",
        help="Use staged unfreezing of reconstructor during Stage-C.",
    )
    parser.add_argument("--stageC_unfreeze_phase1_epochs", type=int, default=3)
    parser.add_argument("--stageC_unfreeze_phase2_epochs", type=int, default=7)
    parser.add_argument("--stageC_unfreeze_last_n_encoder_layers", type=int, default=2)
    parser.add_argument(
        "--stageC_alternate_freeze",
        action="store_true",
        help="Alternate Stage-C optimization: reco-only phase then dual-only phase in cycles.",
    )
    parser.add_argument(
        "--stageC_alternate_reco_only_epochs",
        type=int,
        default=5,
        help="Epochs per alternating cycle with dual frozen and reconstructor trainable.",
    )
    parser.add_argument(
        "--stageC_alternate_dual_only_epochs",
        type=int,
        default=5,
        help="Epochs per alternating cycle with reconstructor frozen and dual trainable.",
    )
    parser.add_argument("--stageC_lambda_param_anchor", type=float, default=0.0)
    parser.add_argument("--stageC_lambda_output_anchor", type=float, default=0.0)
    parser.add_argument("--stageC_anchor_decay", type=float, default=0.95)
    parser.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    parser.add_argument("--use_corrected_flags", action="store_true")
    parser.add_argument("--loss_w_phys", type=float, default=BASE_CONFIG["loss"]["w_phys"])
    parser.add_argument("--loss_w_pt_ratio", type=float, default=BASE_CONFIG["loss"]["w_pt_ratio"])
    parser.add_argument("--loss_w_m_ratio", type=float, default=float(BASE_CONFIG["loss"].get("w_m_ratio", 0.0)))
    parser.add_argument("--loss_w_e_ratio", type=float, default=BASE_CONFIG["loss"]["w_e_ratio"])
    parser.add_argument("--loss_w_radial_profile", type=float, default=float(BASE_CONFIG["loss"].get("w_radial_profile", 0.0)))
    parser.add_argument("--loss_radial_n_bins", type=int, default=int(BASE_CONFIG["loss"].get("radial_n_bins", 8)))
    parser.add_argument("--loss_radial_max_dr", type=float, default=float(BASE_CONFIG["loss"].get("radial_max_dr", 1.0)))
    parser.add_argument("--loss_w_fp_mass", type=float, default=float(BASE_CONFIG["loss"].get("w_fp_mass", 0.0)))
    parser.add_argument("--loss_fp_mass_cost_thresh", type=float, default=float(BASE_CONFIG["loss"].get("fp_mass_cost_thresh", 0.80)))
    parser.add_argument("--loss_fp_mass_tau", type=float, default=float(BASE_CONFIG["loss"].get("fp_mass_tau", 0.10)))
    parser.add_argument("--loss_w_budget", type=float, default=BASE_CONFIG["loss"]["w_budget"])
    parser.add_argument("--loss_w_sparse", type=float, default=BASE_CONFIG["loss"]["w_sparse"])
    parser.add_argument("--loss_w_local", type=float, default=BASE_CONFIG["loss"]["w_local"])
    parser.add_argument(
        "--loss_unselected_penalty",
        type=float,
        default=float(BASE_CONFIG["loss"].get("unselected_penalty", 0.0)),
        help="Penalty used in set loss for unmatched predicted mass (chamfer/hungarian paths).",
    )
    parser.add_argument(
        "--loss_gen_local_radius",
        type=float,
        default=float(BASE_CONFIG["loss"].get("gen_local_radius", 0.0)),
        help="Allowed generation locality radius before local penalty starts.",
    )
    parser.add_argument(
        "--loss_set_mode",
        type=str,
        default="chamfer",
        choices=["chamfer", "chamfer_sinkhorn", "sinkhorn", "ot", "hungarian", "combo"],
        help="Set-level reconstruction loss: chamfer (default), chamfer_sinkhorn blend, sinkhorn, near-OT, hungarian, or custom combo.",
    )
    parser.add_argument(
        "--loss_set_chamfer_mix",
        type=float,
        default=0.5,
        help="For loss_set_mode=chamfer_sinkhorn: weight on chamfer term in [0,1].",
    )
    parser.add_argument(
        "--loss_set_chamfer_topk_target_weighted",
        action="store_true",
        help="Upweight top offline-pT targets in chamfer target->pred coverage term.",
    )
    parser.add_argument(
        "--loss_set_chamfer_topk_k",
        type=int,
        default=int(BASE_CONFIG["loss"].get("chamfer_topk_k", 20)),
        help="Number of top offline targets by pT to upweight for chamfer target-side coverage.",
    )
    parser.add_argument(
        "--loss_set_chamfer_topk_weight",
        type=float,
        default=float(BASE_CONFIG["loss"].get("chamfer_topk_weight", 2.5)),
        help="Weight multiplier for top-k offline targets in chamfer target-side coverage.",
    )
    parser.add_argument(
        "--loss_set_chamfer_topk_no_mean_renorm",
        action="store_true",
        help="Disable mean renormalization of chamfer top-k target weights.",
    )
    parser.add_argument("--loss_set_sinkhorn_eps", type=float, default=0.08)
    parser.add_argument("--loss_set_sinkhorn_iters", type=int, default=25)
    parser.add_argument("--loss_set_ot_eps", type=float, default=0.02)
    parser.add_argument("--loss_set_ot_iters", type=int, default=50)
    parser.add_argument("--loss_set_combo_w_chamfer", type=float, default=1.0)
    parser.add_argument("--loss_set_combo_w_sinkhorn", type=float, default=0.0)
    parser.add_argument("--loss_set_combo_w_ot", type=float, default=0.0)
    parser.add_argument("--loss_set_combo_w_hungarian", type=float, default=0.0)
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
    parser.add_argument(
        "--save_fusion_scores",
        action="store_true",
        help="Save val/test score arrays (teacher/HLT/joint) for downstream calibration/fusion analysis.",
    )

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
    global _ORACLE_STUDY_VARIANT, _ORACLE_STUDY_ENABLE_GEN, _ORACLE_UNSMEAR_KEEP_THRESH, _ORACLE_WARNED_MISSING_BATCH
    _ORACLE_STUDY_VARIANT = str(args.oracle_study_variant).strip()
    _ORACLE_STUDY_ENABLE_GEN = str(_ORACLE_STUDY_VARIANT).lower() == "b_gen"
    _ORACLE_UNSMEAR_KEEP_THRESH = float(max(args.oracle_unsmear_keep_thresh, 0.0))
    _ORACLE_WARNED_MISSING_BATCH = False

    # This variant always selects checkpoints by validation AUC.
    selection_metric = "auc"
    if str(args.selection_metric).lower() != "auc":
        print(f"Note: overriding --selection_metric={args.selection_metric} to 'auc' in this script.")

    cfg = _deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)
    cfg["loss"]["w_phys"] = float(args.loss_w_phys)
    cfg["loss"]["w_pt_ratio"] = float(args.loss_w_pt_ratio)
    cfg["loss"]["w_m_ratio"] = float(args.loss_w_m_ratio)
    cfg["loss"]["w_e_ratio"] = float(args.loss_w_e_ratio)
    cfg["loss"]["w_radial_profile"] = float(args.loss_w_radial_profile)
    cfg["loss"]["radial_n_bins"] = int(max(args.loss_radial_n_bins, 1))
    cfg["loss"]["radial_max_dr"] = float(max(args.loss_radial_max_dr, 1e-3))
    cfg["loss"]["w_fp_mass"] = float(args.loss_w_fp_mass)
    cfg["loss"]["fp_mass_cost_thresh"] = float(args.loss_fp_mass_cost_thresh)
    cfg["loss"]["fp_mass_tau"] = float(max(args.loss_fp_mass_tau, 1e-4))
    cfg["loss"]["w_budget"] = float(args.loss_w_budget)
    cfg["loss"]["w_sparse"] = float(args.loss_w_sparse)
    cfg["loss"]["w_local"] = float(args.loss_w_local)
    cfg["loss"]["unselected_penalty"] = float(max(args.loss_unselected_penalty, 0.0))
    cfg["loss"]["gen_local_radius"] = float(max(args.loss_gen_local_radius, 0.0))
    cfg["loss"]["set_loss_mode"] = str(args.loss_set_mode).strip().lower()
    cfg["loss"]["chamfer_mix"] = float(np.clip(args.loss_set_chamfer_mix, 0.0, 1.0))
    cfg["loss"]["chamfer_topk_target_weighted"] = bool(args.loss_set_chamfer_topk_target_weighted)
    cfg["loss"]["chamfer_topk_k"] = int(max(args.loss_set_chamfer_topk_k, 0))
    cfg["loss"]["chamfer_topk_weight"] = float(max(args.loss_set_chamfer_topk_weight, 0.0))
    cfg["loss"]["chamfer_topk_mean_renorm"] = not bool(args.loss_set_chamfer_topk_no_mean_renorm)
    cfg["loss"]["sinkhorn_eps"] = float(max(args.loss_set_sinkhorn_eps, 1e-4))
    cfg["loss"]["sinkhorn_iters"] = int(max(args.loss_set_sinkhorn_iters, 1))
    cfg["loss"]["ot_eps"] = float(max(args.loss_set_ot_eps, 1e-4))
    cfg["loss"]["ot_iters"] = int(max(args.loss_set_ot_iters, 1))
    cfg["loss"]["combo_w_chamfer"] = float(max(args.loss_set_combo_w_chamfer, 0.0))
    cfg["loss"]["combo_w_sinkhorn"] = float(max(args.loss_set_combo_w_sinkhorn, 0.0))
    cfg["loss"]["combo_w_ot"] = float(max(args.loss_set_combo_w_ot, 0.0))
    cfg["loss"]["combo_w_hungarian"] = float(max(args.loss_set_combo_w_hungarian, 0.0))
    cfg["loss"]["oracle_w_action"] = float(max(args.oracle_w_action, 0.0))
    cfg["loss"]["oracle_w_tok"] = float(max(args.oracle_w_tok, 0.0))
    cfg["loss"]["oracle_w_split"] = float(max(args.oracle_w_split, 0.0))

    cfg["reconstructor_training"]["epochs"] = int(args.stageA_epochs)
    cfg["reconstructor_training"]["patience"] = int(args.stageA_patience)
    cfg["reconstructor_training"]["loss_schedule_enable"] = bool(args.stageA_loss_schedule)
    cfg["reconstructor_training"]["loss_sched_early_set_mult"] = float(args.stageA_loss_sched_early_set_mult)
    cfg["reconstructor_training"]["loss_sched_early_budget_mult"] = float(args.stageA_loss_sched_early_budget_mult)
    cfg["reconstructor_training"]["loss_sched_late_ptratio_mult"] = float(args.stageA_loss_sched_late_ptratio_mult)
    cfg["reconstructor_training"]["loss_sched_late_local_mult"] = float(args.stageA_loss_sched_late_local_mult)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

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
    hlt_const, hlt_mask, hlt_stats, budget_truth = apply_hlt_effects_realistic_with_oracle_tracking(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
        unsmear_keep_thresh=float(_ORACLE_UNSMEAR_KEEP_THRESH),
    )
    # Non-privileged target: only supervise total added count from offline-vs-HLT count gap.
    # true_added_raw is used for diagnostics/reporting. Training target can be scaled down.
    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    added_target_scale = _clamp_target_scale(float(args.added_target_scale))
    if bool(_ORACLE_STUDY_ENABLE_GEN):
        budget_merge_true_raw = budget_truth["merge_lost_per_jet"].astype(np.float32)
        budget_eff_true_raw = budget_truth["eff_lost_per_jet"].astype(np.float32)
        budget_merge_true = (added_target_scale * budget_merge_true_raw).astype(np.float32)
        budget_eff_true = (added_target_scale * budget_eff_true_raw).astype(np.float32)
        print(
            f"Oracle-study target setup (generation ON): variant={_ORACLE_STUDY_VARIANT}, "
            f"added_target_scale={added_target_scale:.3f}, "
            f"mean_merge_target={float(budget_merge_true.mean()):.3f}, "
            f"mean_eff_target={float(budget_eff_true.mean()):.3f}"
        )
    else:
        budget_merge_true_raw = true_added_raw.astype(np.float32)
        budget_eff_true_raw = np.zeros_like(true_added_raw, dtype=np.float32)
        budget_merge_true = (added_target_scale * true_added_raw).astype(np.float32)
        budget_eff_true = np.zeros_like(true_added_raw, dtype=np.float32)
        print(
            f"Oracle-study target setup (unmerge-only): variant={_ORACLE_STUDY_VARIANT}, "
            f"added_target_scale={added_target_scale:.3f}, "
            f"mean_true_added_raw={float(true_added_raw.mean()):.3f}, "
            f"mean_target_added={float(budget_merge_true.mean()):.3f}"
        )

    oracle_action = budget_truth["oracle_action"].astype(np.int8)
    oracle_tok_target = budget_truth["oracle_tok_target"].astype(np.float32)
    oracle_tok_valid = budget_truth["oracle_tok_valid"].astype(np.uint8)
    oracle_split_target = budget_truth["oracle_split_target"].astype(np.float32)
    oracle_split_valid = budget_truth["oracle_split_valid"].astype(np.uint8)

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
        "variant": "nopriv_unmergeonly_oracle_study",
        "oracle_study_variant": str(_ORACLE_STUDY_VARIANT),
        "oracle_enable_generation": bool(_ORACLE_STUDY_ENABLE_GEN),
        "oracle_unsmear_keep_thresh": float(_ORACLE_UNSMEAR_KEEP_THRESH),
        "oracle_weights": {
            "w_action": float(cfg["loss"].get("oracle_w_action", 0.0)),
            "w_tok": float(cfg["loss"].get("oracle_w_tok", 0.0)),
            "w_split": float(cfg["loss"].get("oracle_w_split", 0.0)),
        },
        "added_target_scale": float(added_target_scale),
        "mean_true_added_raw": float(true_added_raw.mean()),
        "mean_target_added": float(budget_merge_true.mean()),
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
    auc_baseline_val, preds_baseline_val, labs_baseline_val = eval_classifier(baseline, dl_val_hlt, device)
    hlt_delta_thr_prob = prob_threshold_at_target_tpr(
        np.asarray(preds_baseline_val, dtype=np.float64)[np.asarray(labs_baseline_val, dtype=np.float32) > 0.5],
        target_tpr=0.50,
    )
    print(
        f"StageC weak-delta HLT reference @TPR=0.50: threshold_prob={float(hlt_delta_thr_prob):.6f}, "
        f"val_auc_hlt={float(auc_baseline_val):.4f}"
    )

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
        const_off[train_idx], masks_off[train_idx],
        budget_merge_true[train_idx], budget_eff_true[train_idx],
        sample_weight_reco=sample_weight_reco,
        oracle_action=oracle_action[train_idx],
        oracle_tok_target=oracle_tok_target[train_idx],
        oracle_tok_valid=oracle_tok_valid[train_idx],
        oracle_split_target=oracle_split_target[train_idx],
        oracle_split_valid=oracle_split_valid[train_idx],
    )
    ds_val_reco = WeightedReconstructionDataset(
        feat_hlt_std[val_idx], hlt_mask[val_idx], hlt_const[val_idx],
        const_off[val_idx], masks_off[val_idx],
        budget_merge_true[val_idx], budget_eff_true[val_idx],
        sample_weight_reco=sample_weight_reco_val,
        oracle_action=oracle_action[val_idx],
        oracle_tok_target=oracle_tok_target[val_idx],
        oracle_tok_valid=oracle_tok_valid[val_idx],
        oracle_split_target=oracle_split_target[val_idx],
        oracle_split_valid=oracle_split_valid[val_idx],
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
    if not bool(_ORACLE_STUDY_ENABLE_GEN):
        reconstructor = wrap_reconstructor_unmerge_only(reconstructor)
    else:
        print("Oracle-study note: generation branch enabled for this run.")
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
        reload_best_at_stage_transition=not bool(args.disable_stageA_stagewise_best_reload),
    )

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
        oracle_action=oracle_action[train_idx],
        oracle_tok_target=oracle_tok_target[train_idx],
        oracle_tok_valid=oracle_tok_valid[train_idx],
        oracle_split_target=oracle_split_target[train_idx],
        oracle_split_valid=oracle_split_valid[train_idx],
    )
    ds_train_joint_stageb = JointDualDataset(
        feat_hlt_std[stageB_train_idx], feat_hlt_dual[stageB_train_idx], hlt_mask[stageB_train_idx], hlt_const[stageB_train_idx],
        const_off[stageB_train_idx], masks_off[stageB_train_idx],
        budget_merge_true[stageB_train_idx], budget_eff_true[stageB_train_idx],
        labels[stageB_train_idx],
        sample_weight_cls=stageb_w_cls,
        sample_weight_reco=stageb_w_reco,
        oracle_action=oracle_action[stageB_train_idx],
        oracle_tok_target=oracle_tok_target[stageB_train_idx],
        oracle_tok_valid=oracle_tok_valid[stageB_train_idx],
        oracle_split_target=oracle_split_target[stageB_train_idx],
        oracle_split_valid=oracle_split_valid[stageB_train_idx],
    )
    ds_val_joint = JointDualDataset(
        feat_hlt_std[val_idx], feat_hlt_dual[val_idx], hlt_mask[val_idx], hlt_const[val_idx],
        const_off[val_idx], masks_off[val_idx],
        budget_merge_true[val_idx], budget_eff_true[val_idx],
        labels[val_idx],
        sample_weight_cls=sample_weight_cls_val,
        sample_weight_reco=sample_weight_reco_val,
        oracle_action=oracle_action[val_idx],
        oracle_tok_target=oracle_tok_target[val_idx],
        oracle_tok_valid=oracle_tok_valid[val_idx],
        oracle_split_target=oracle_split_target[val_idx],
        oracle_split_valid=oracle_split_valid[val_idx],
    )
    ds_test_joint = JointDualDataset(
        feat_hlt_std[test_idx], feat_hlt_dual[test_idx], hlt_mask[test_idx], hlt_const[test_idx],
        const_off[test_idx], masks_off[test_idx],
        budget_merge_true[test_idx], budget_eff_true[test_idx],
        labels[test_idx],
        sample_weight_cls=np.ones((len(test_idx),), dtype=np.float32),
        sample_weight_reco=np.ones((len(test_idx),), dtype=np.float32),
        oracle_action=oracle_action[test_idx],
        oracle_tok_target=oracle_tok_target[test_idx],
        oracle_tok_valid=oracle_tok_valid[test_idx],
        oracle_split_target=oracle_split_target[test_idx],
        oracle_split_valid=oracle_split_valid[test_idx],
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
        lambda_delta_cls=0.0,
        delta_tau=float(args.stageC_delta_tau),
        delta_lambda_fp=float(args.stageC_delta_lambda_fp),
        delta_hlt_model=None,
        delta_hlt_threshold_prob=float(hlt_delta_thr_prob),
        delta_warmup_epochs=int(args.stageC_delta_warmup_epochs),
    )

    if str(args.stageC_init_reco_ckpt).strip():
        reco_ckpt_path = str(args.stageC_init_reco_ckpt).strip()
        reconstructor.load_state_dict(_load_checkpoint_model_state(reco_ckpt_path, device))
        print(f"Loaded Stage-C init reconstructor checkpoint: {reco_ckpt_path}")
    if str(args.stageC_init_dual_ckpt).strip():
        dual_ckpt_path = str(args.stageC_init_dual_ckpt).strip()
        dual_joint.load_state_dict(_load_checkpoint_model_state(dual_ckpt_path, device))
        print(f"Loaded Stage-C init dual checkpoint: {dual_ckpt_path}")

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
        lambda_delta_cls=float(args.stageC_lambda_delta),
        delta_tau=float(args.stageC_delta_tau),
        delta_lambda_fp=float(args.stageC_delta_lambda_fp),
        delta_hlt_model=baseline,
        delta_hlt_threshold_prob=float(hlt_delta_thr_prob),
        delta_warmup_epochs=int(args.stageC_delta_warmup_epochs),
        progressive_unfreeze=bool(args.stageC_progressive_unfreeze),
        unfreeze_phase1_epochs=int(args.stageC_unfreeze_phase1_epochs),
        unfreeze_phase2_epochs=int(args.stageC_unfreeze_phase2_epochs),
        unfreeze_last_n_encoder_layers=int(args.stageC_unfreeze_last_n_encoder_layers),
        alternate_freeze=bool(args.stageC_alternate_freeze),
        alternate_reco_only_epochs=int(args.stageC_alternate_reco_only_epochs),
        alternate_dual_only_epochs=int(args.stageC_alternate_dual_only_epochs),
        lambda_param_anchor=float(args.stageC_lambda_param_anchor),
        lambda_output_anchor=float(args.stageC_lambda_output_anchor),
        anchor_decay=float(args.stageC_anchor_decay),
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

    # Build hard reconstructed view for diagnostics.
    print("\n" + "=" * 70)
    print("STEP 5: RECONSTRUCTION DIAGNOSTICS")
    print("=" * 70)
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
    ) = reconstruct_dataset(
        model=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        max_constits=args.max_constits,
        device=device,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        weight_threshold=float(args.reco_weight_threshold),
        use_budget_topk=not bool(args.reco_disable_budget_topk),
    )

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
        "Joint-corrected HLT (reco)",
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
    budget_summary = plot_budget_diagnostics(
        save_root=save_root,
        true_merge=budget_merge_true_raw[test_idx],
        true_eff=budget_eff_true_raw[test_idx],
        pred_merge=pred_budget_merge[test_idx],
        pred_eff=pred_budget_eff[test_idx],
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
    print("\nBudget diagnostics (test split):")
    print(
        f"  MAE: merge={budget_summary['merge_mae']:.3f}, "
        f"eff={budget_summary['eff_mae']:.3f}, total={budget_summary['total_mae']:.3f}"
    )
    print(
        f"  Bias: merge={budget_summary['merge_bias']:.3f}, "
        f"eff={budget_summary['eff_bias']:.3f}, total={budget_summary['total_bias']:.3f}"
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
    fpr30_stage2 = fpr_at_target_tpr(fpr_s2, tpr_s2, 0.30)
    fpr30_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.30)
    fpr30_stage2_fprsel = fpr_at_target_tpr(fpr_s2_fprsel, tpr_s2_fprsel, 0.30) if preds_stage2_fprsel is not None else float("nan")
    fpr30_joint_fprsel = fpr_at_target_tpr(fpr_j_fprsel, tpr_j_fprsel, 0.30) if preds_joint_fprsel is not None else float("nan")
    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.50)
    fpr50_stage2 = fpr_at_target_tpr(fpr_s2, tpr_s2, 0.50)
    fpr50_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.50)
    fpr50_stage2_fprsel = fpr_at_target_tpr(fpr_s2_fprsel, tpr_s2_fprsel, 0.50) if preds_stage2_fprsel is not None else float("nan")
    fpr50_joint_fprsel = fpr_at_target_tpr(fpr_j_fprsel, tpr_j_fprsel, 0.50) if preds_joint_fprsel is not None else float("nan")

    oracle_intervention_metrics: Dict[str, float] = {}
    if bool(args.oracle_eval_interventions_for_A) and str(_ORACLE_STUDY_VARIANT).strip().upper() == "A":
        for m in ("C", "D", "E"):
            auc_i, preds_i, labs_i, fpr50_i = eval_joint_model_oracle_intervention(
                reconstructor,
                dual_joint,
                dl_test_joint,
                device,
                corrected_weight_floor=float(args.corrected_weight_floor),
                corrected_use_flags=bool(args.use_corrected_flags),
                intervention_mode=m,
            )
            if preds_i.size > 0 and len(np.unique(labs_i)) > 1:
                fpr_i, tpr_i, _ = roc_curve(labs_i, preds_i)
                fpr30_i = fpr_at_target_tpr(fpr_i, tpr_i, 0.30)
            else:
                fpr30_i = float("nan")
            oracle_intervention_metrics[f"auc_{m}"] = float(auc_i)
            oracle_intervention_metrics[f"fpr30_{m}"] = float(fpr30_i)
            oracle_intervention_metrics[f"fpr50_{m}"] = float(fpr50_i)

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
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
    if oracle_intervention_metrics:
        print("Oracle intervention diagnostics on A model:")
        print(
            "  AUC[C/D/E]: "
            f"{oracle_intervention_metrics.get('auc_C', float('nan')):.4f} / "
            f"{oracle_intervention_metrics.get('auc_D', float('nan')):.4f} / "
            f"{oracle_intervention_metrics.get('auc_E', float('nan')):.4f}"
        )
        print(
            "  FPR@50[C/D/E]: "
            f"{oracle_intervention_metrics.get('fpr50_C', float('nan')):.6f} / "
            f"{oracle_intervention_metrics.get('fpr50_D', float('nan')):.6f} / "
            f"{oracle_intervention_metrics.get('fpr50_E', float('nan')):.6f}"
        )

    if bool(args.save_fusion_scores):
        print("Saving val/test score arrays for fusion analysis...")
        auc_teacher_val, preds_teacher_val, labs_teacher_val = eval_classifier(teacher, dl_val_off, device)
        auc_baseline_val, preds_baseline_val, labs_baseline_val = eval_classifier(baseline, dl_val_hlt, device)
        auc_joint_val, preds_joint_val, labs_joint_val, fpr50_joint_val = eval_joint_model(
            reconstructor,
            dual_joint,
            dl_val_joint,
            device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        assert np.array_equal(labs_teacher_val.astype(np.float32), labs_baseline_val.astype(np.float32))
        assert np.array_equal(labs_teacher_val.astype(np.float32), labs_joint_val.astype(np.float32))
        assert np.array_equal(labs.astype(np.float32), labs_joint.astype(np.float32))
        np.savez_compressed(
            save_root / "fusion_scores_val_test.npz",
            labels_val=labs_teacher_val.astype(np.float32),
            labels_test=labs.astype(np.float32),
            preds_teacher_val=np.asarray(preds_teacher_val, dtype=np.float64),
            preds_teacher_test=np.asarray(preds_teacher, dtype=np.float64),
            preds_hlt_val=np.asarray(preds_baseline_val, dtype=np.float64),
            preds_hlt_test=np.asarray(preds_baseline, dtype=np.float64),
            preds_joint_val=np.asarray(preds_joint_val, dtype=np.float64),
            preds_joint_test=np.asarray(preds_joint, dtype=np.float64),
            auc_teacher_val=float(auc_teacher_val),
            auc_teacher_test=float(auc_teacher),
            auc_hlt_val=float(auc_baseline_val),
            auc_hlt_test=float(auc_baseline),
            auc_joint_val=float(auc_joint_val),
            auc_joint_test=float(auc_joint),
            fpr50_joint_val=float(fpr50_joint_val),
            fpr50_joint_test=float(fpr50_joint),
            hlt_nconst_val=np.asarray(hlt_mask[val_idx].sum(axis=1), dtype=np.int32),
            hlt_nconst_test=np.asarray(hlt_mask[test_idx].sum(axis=1), dtype=np.int32),
            hlt_jet_pt_val=np.asarray(compute_jet_pt(hlt_const[val_idx], hlt_mask[val_idx]), dtype=np.float64),
            hlt_jet_pt_test=np.asarray(compute_jet_pt(hlt_const[test_idx], hlt_mask[test_idx]), dtype=np.float64),
            off_jet_pt_val=np.asarray(compute_jet_pt(const_off[val_idx], masks_off[val_idx]), dtype=np.float64),
            off_jet_pt_test=np.asarray(compute_jet_pt(const_off[test_idx], masks_off[test_idx]), dtype=np.float64),
        )
        print(f"Saved fusion score arrays to: {save_root / 'fusion_scores_val_test.npz'}")

    plot_lines = [
        (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
        (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
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
        auc_stage2=auc_stage2,
        auc_stage2_fprsel=auc_stage2_fprsel,
        auc_joint=auc_joint,
        auc_joint_fprsel=auc_joint_fprsel,
        auc_joint_kd=auc_joint_kd,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
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
        fpr30_stage2=fpr30_stage2,
        fpr30_stage2_fprsel=fpr30_stage2_fprsel,
        fpr30_joint=fpr30_joint,
        fpr30_joint_fprsel=fpr30_joint_fprsel,
        fpr30_joint_kd=fpr30_joint_kd,
        fpr50_teacher=fpr50_teacher,
        fpr50_baseline=fpr50_baseline,
        fpr50_stage2=fpr50_stage2,
        fpr50_stage2_fprsel=fpr50_stage2_fprsel,
        fpr50_joint=fpr50_joint,
        fpr50_joint_fprsel=fpr50_joint_fprsel,
        fpr50_joint_kd=fpr50_joint_kd,
        oracle_auc_C=float(oracle_intervention_metrics.get("auc_C", float("nan"))),
        oracle_auc_D=float(oracle_intervention_metrics.get("auc_D", float("nan"))),
        oracle_auc_E=float(oracle_intervention_metrics.get("auc_E", float("nan"))),
        oracle_fpr30_C=float(oracle_intervention_metrics.get("fpr30_C", float("nan"))),
        oracle_fpr30_D=float(oracle_intervention_metrics.get("fpr30_D", float("nan"))),
        oracle_fpr30_E=float(oracle_intervention_metrics.get("fpr30_E", float("nan"))),
        oracle_fpr50_C=float(oracle_intervention_metrics.get("fpr50_C", float("nan"))),
        oracle_fpr50_D=float(oracle_intervention_metrics.get("fpr50_D", float("nan"))),
        oracle_fpr50_E=float(oracle_intervention_metrics.get("fpr50_E", float("nan"))),
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
    with open(save_root / "budget_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(budget_summary, f, indent=2)
    with open(save_root / "soft_corrected_view_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(soft_view_summary_test, f, indent=2)
    with open(save_root / "jet_regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(jet_reg_metrics, f, indent=2)

    with open(save_root / "joint_stage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": {
                    "mode": "nopriv_unmergeonly_oracle_study",
                    "oracle_study_variant": str(_ORACLE_STUDY_VARIANT),
                    "oracle_enable_generation": bool(_ORACLE_STUDY_ENABLE_GEN),
                    "oracle_unsmear_keep_thresh": float(_ORACLE_UNSMEAR_KEEP_THRESH),
                    "oracle_weights": {
                        "w_action": float(cfg["loss"].get("oracle_w_action", 0.0)),
                        "w_tok": float(cfg["loss"].get("oracle_w_tok", 0.0)),
                        "w_split": float(cfg["loss"].get("oracle_w_split", 0.0)),
                    },
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
                    "auc_stage2": float(auc_stage2),
                    "auc_stage2_fprsel": float(auc_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "auc_joint": float(auc_joint),
                    "auc_joint_fprsel": float(auc_joint_fprsel) if preds_joint_fprsel is not None else None,
                    "auc_joint_kd": float(auc_joint_kd) if preds_joint_kd is not None else None,
                    "fpr30_teacher": float(fpr30_teacher),
                    "fpr30_baseline": float(fpr30_baseline),
                    "fpr30_stage2": float(fpr30_stage2),
                    "fpr30_stage2_fprsel": float(fpr30_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr30_joint": float(fpr30_joint),
                    "fpr30_joint_fprsel": float(fpr30_joint_fprsel) if preds_joint_fprsel is not None else None,
                    "fpr30_joint_kd": float(fpr30_joint_kd) if preds_joint_kd is not None else None,
                    "fpr50_teacher": float(fpr50_teacher),
                    "fpr50_baseline": float(fpr50_baseline),
                    "fpr50_stage2": float(fpr50_stage2),
                    "fpr50_stage2_fprsel": float(fpr50_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr50_joint": float(fpr50_joint),
                    "fpr50_joint_fprsel": float(fpr50_joint_fprsel) if preds_joint_fprsel is not None else None,
                    "fpr50_joint_kd": float(fpr50_joint_kd) if preds_joint_kd is not None else None,
                    "oracle_intervention_metrics": oracle_intervention_metrics,
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

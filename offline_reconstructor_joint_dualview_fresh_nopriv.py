#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-driven Offline Reconstructor + Dual-View tagger joint training.

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
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from tqdm import tqdm

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
)

from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    wrap_phi_np,
    _compute_local_density_np,
    OfflineReconstructor,
    reconstruct_dataset,
    plot_roc,
    fpr_at_target_tpr,
    plot_constituent_count_diagnostics,
    plot_budget_diagnostics,
    train_dual_kd_student,
)


# ----------------------------- Reproducibility ----------------------------- #
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CORRECTED_VIEW_DIM = 12


def _deepcopy_config() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


class JointDualDataset(Dataset):
    def __init__(
        self,
        feat_hlt_reco: np.ndarray,
        feat_hlt_dual: np.ndarray,
        feat_off: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        budget_total_true: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt_reco = torch.tensor(feat_hlt_reco, dtype=torch.float32)
        self.feat_hlt_dual = torch.tensor(feat_hlt_dual, dtype=torch.float32)
        self.feat_off = torch.tensor(feat_off, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.budget_total_true = torch.tensor(budget_total_true, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt_reco.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt_reco": self.feat_hlt_reco[i],
            "feat_hlt_dual": self.feat_hlt_dual[i],
            "feat_off": self.feat_off[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "budget_total_true": self.budget_total_true[i],
            "label": self.labels[i],
        }


class ReconstructionDatasetNoPriv(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        budget_total_true: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.budget_total_true = torch.tensor(budget_total_true, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "budget_total_true": self.budget_total_true[i],
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds a differentiable fixed-length corrected view.
    The DualViewCrossAttnClassifier expects both views to share the same sequence length.
    We therefore map reconstructor outputs back to L token slots (L = HLT token count):
      - corrected token kinematics from tok branch
      - parent-level split mass summary
      - per-token share of efficiency budget

    Output feature dims:
      7 base kinematic features + 5 channels
      [tok_weight, parent_added_weight, eff_share, merge_flag_soft, eff_flag_soft] = 12.
    """
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

    action_prob = reco_out["action_prob"]
    p_split = action_prob[..., 2].clamp(0.0, 1.0)
    merge_flag_soft = (0.65 * p_split + 0.35 * parent_added).clamp(0.0, 1.0) * mask_b.float()
    eff_flag_soft = eff_share.clamp(0.0, 1.0) * mask_b.float()

    extra = torch.stack([tok_w, parent_added, eff_share, merge_flag_soft, eff_flag_soft], dim=-1)
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
def build_corrected_view_numpy(
    reconstructor: OfflineReconstructor,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    device: torch.device,
    batch_size: int,
    corrected_weight_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_jets, seq_len, _ = feat_hlt.shape
    feat_b = np.zeros((n_jets, seq_len, CORRECTED_VIEW_DIM), dtype=np.float32)
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
        )
        feat_b[start:end] = fb.detach().cpu().numpy()
        mask_b[start:end] = mb.detach().cpu().numpy()
    return feat_b, mask_b


def summarize_soft_corrected_view(
    feat_b: np.ndarray,
    mask_b: np.ndarray,
) -> Dict[str, float]:
    # Extra channels: [tok_weight, parent_added_weight, eff_share, merge_flag_soft, eff_flag_soft]
    tok_w = feat_b[..., 7]
    parent_added = feat_b[..., 8]
    eff_share = feat_b[..., 9]
    merge_flag_soft = feat_b[..., 10]
    eff_flag_soft = feat_b[..., 11]
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
        "mean_merge_flag_soft_valid": float(merge_flag_soft[valid].mean()),
        "mean_eff_flag_soft_valid": float(eff_flag_soft[valid].mean()),
        "p95_tok_weight_valid": float(np.percentile(tok_w[valid], 95.0)),
        "p95_parent_added_valid": float(np.percentile(parent_added[valid], 95.0)),
        "p95_eff_share_valid": float(np.percentile(eff_share[valid], 95.0)),
        "p95_merge_flag_soft_valid": float(np.percentile(merge_flag_soft[valid], 95.0)),
        "p95_eff_flag_soft_valid": float(np.percentile(eff_flag_soft[valid], 95.0)),
    }


def stage_scale(ep: int, cfg: Dict) -> float:
    s1 = int(cfg["stage1_epochs"])
    s2 = int(cfg["stage2_epochs"])
    if ep < s1:
        return 0.35
    if ep < s2:
        return 0.70
    return 1.0


def merge_quota_schedule(ep: int, cfg: Dict) -> Tuple[float, float]:
    """
    Merge-heavy -> balanced curriculum.
    Returns:
      merge_target_share: desired split-branch share in [0, 1]
      quota_strength:     weight multiplier in [0, 1] for quota prior
    """
    s1 = int(cfg["stage1_epochs"])
    s2 = int(cfg["stage2_epochs"])
    merge_start = float(cfg.get("merge_share_start", 0.90))
    merge_mid = float(cfg.get("merge_share_mid", 0.55))
    merge_final = float(cfg.get("merge_share_final", 0.50))
    if ep < s1:
        return merge_start, 1.00
    if ep < s2:
        t = float(ep - s1) / float(max(s2 - s1, 1))
        merge_share = merge_start + t * (merge_mid - merge_start)
        return float(merge_share), float(1.0 - t)
    return merge_final, 0.0


def _topk_mask_variable_k(scores: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Per-row top-k hard mask.
    scores: [B, N]
    k:      [B] int64
    """
    B, N = scores.shape
    out = torch.zeros_like(scores)
    for i in range(B):
        ki = int(k[i].item())
        if ki <= 0:
            continue
        ki = min(ki, N)
        idx = torch.topk(scores[i], k=ki, dim=0, largest=True).indices
        out[i, idx] = 1.0
    return out


def compute_reconstruction_losses_nopriv(
    out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_total_true: torch.Tensor,
    loss_cfg: Dict,
    merge_target_share: float = 0.5,
    quota_strength: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    No-privileged reconstruction loss: uses only jet-level total-added supervision.
    No merge/eff branch-specific truth is consumed.
    """
    eps = 1e-8

    pred = out["cand_tokens"]
    w = out["cand_weights"].clamp(0.0, 1.0)

    # Set-level weighted Chamfer-like loss.
    from offline_reconstructor_no_gt_local30kv2 import _token_cost_matrix, _weighted_fourvec_sums
    cost = _token_cost_matrix(pred, const_off)
    valid_tgt = mask_off.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))

    pred_to_tgt = cost.min(dim=2).values
    loss_pred_to_tgt = (w * pred_to_tgt).sum(dim=1) / (w.sum(dim=1) + eps)

    penalty = float(loss_cfg["unselected_penalty"]) * (1.0 - w).unsqueeze(2)
    tgt_to_pred = (cost + penalty).min(dim=1).values
    tgt_mask_f = mask_off.float()
    loss_tgt_to_pred = (tgt_to_pred * tgt_mask_f).sum(dim=1) / (tgt_mask_f.sum(dim=1) + eps)
    loss_set = (loss_pred_to_tgt + loss_tgt_to_pred).mean()

    # Physics consistency.
    pred_px, pred_py, pred_pz, pred_E = _weighted_fourvec_sums(pred, w)
    true_px, true_py, true_pz, true_E = _weighted_fourvec_sums(const_off, mask_off.float())
    norm = true_px.abs() + true_py.abs() + true_pz.abs() + true_E.abs() + 1.0
    loss_phys = (
        (pred_px - true_px).abs()
        + (pred_py - true_py).abs()
        + (pred_pz - true_pz).abs()
        + (pred_E - true_E).abs()
    ) / norm
    loss_phys = loss_phys.mean()

    pred_pt = torch.sqrt(pred_px.pow(2) + pred_py.pow(2) + eps)
    true_pt = torch.sqrt(true_px.pow(2) + true_py.pow(2) + eps)
    pt_ratio = pred_pt / (true_pt + eps)
    loss_pt_ratio = F.smooth_l1_loss(pt_ratio, torch.ones_like(pt_ratio))
    e_ratio = pred_E / (true_E + eps)
    loss_e_ratio = F.smooth_l1_loss(e_ratio, torch.ones_like(e_ratio))

    # Budget/count losses (total-added only).
    true_count = mask_off.float().sum(dim=1)
    hlt_count = mask_hlt.float().sum(dim=1)
    true_added = (true_count - hlt_count).clamp(min=0.0)
    pred_count = w.sum(dim=1)
    child_w = out["child_weight"].clamp(0.0, 1.0)
    gen_w = out["gen_weight"].clamp(0.0, 1.0)
    pred_added_merge = child_w.sum(dim=1)
    pred_added_eff = gen_w.sum(dim=1)
    pred_added = pred_added_merge + pred_added_eff
    budget_total_pred = (out["budget_merge"] + out["budget_eff"]).clamp(min=0.0)

    loss_budget = (
        F.smooth_l1_loss(pred_count, true_count)
        + F.smooth_l1_loss(out["budget_total"], true_count)
        + F.smooth_l1_loss(pred_added, true_added)
        + F.smooth_l1_loss(budget_total_pred, true_added)
        + F.smooth_l1_loss(budget_total_pred, budget_total_true)
    )

    # Hard allocation consistency: make soft branch totals match hard top-k allocation.
    # This reduces train/infer branch allocation mismatch.
    add_scores = torch.cat([child_w, gen_w], dim=1)  # [B, N_child+N_gen]
    max_added = add_scores.size(1)
    k_added = torch.round(true_added).long().clamp(min=0, max=max_added)
    hard_added_mask = _topk_mask_variable_k(add_scores, k_added)
    n_child = child_w.size(1)
    hard_added_merge = hard_added_mask[:, :n_child].sum(dim=1)
    hard_added_eff = hard_added_mask[:, n_child:].sum(dim=1)

    loss_alloc_hard = (
        F.smooth_l1_loss(pred_added_merge, hard_added_merge.detach())
        + F.smooth_l1_loss(pred_added_eff, hard_added_eff.detach())
    )

    # Merge-first curriculum prior (non-privileged): starts merge-heavy, then opens efficiency.
    share = torch.full_like(true_added, float(np.clip(merge_target_share, 0.0, 1.0)))
    target_merge = share * true_added
    target_eff = (1.0 - share) * true_added
    qs = float(max(0.0, min(1.0, quota_strength)))
    loss_alloc_quota = qs * (
        F.smooth_l1_loss(pred_added_merge, target_merge)
        + F.smooth_l1_loss(pred_added_eff, target_eff)
    )

    # Sparsity regularization.
    loss_sparse = child_w.mean() + gen_w.mean()

    # Locality regularization.
    split_delta = out["split_delta"]
    split_step = torch.sqrt(split_delta[..., 1].pow(2) + split_delta[..., 2].pow(2) + 1e-8)
    split_w = out["child_weight"].reshape(split_step.shape)
    loss_local_split = (split_w * split_step).sum() / (split_w.sum() + eps)

    gen_tokens = out["gen_tokens"]
    gen_w = out["gen_weight"]
    h_eta = const_hlt[:, :, 1]
    h_phi = const_hlt[:, :, 2]
    g_eta = gen_tokens[:, :, 1].unsqueeze(2)
    g_phi = gen_tokens[:, :, 2].unsqueeze(2)
    d_eta = g_eta - h_eta.unsqueeze(1)
    d_phi = torch.atan2(torch.sin(g_phi - h_phi.unsqueeze(1)), torch.cos(g_phi - h_phi.unsqueeze(1)))
    dR = torch.sqrt(d_eta.pow(2) + d_phi.pow(2) + 1e-8)
    dR = torch.where(mask_hlt.unsqueeze(1), dR, torch.full_like(dR, 1e4))
    nearest = dR.min(dim=2).values
    excess = F.relu(nearest - float(loss_cfg["gen_local_radius"]))
    loss_local_gen = (gen_w * excess).sum() / (gen_w.sum() + eps)
    loss_local = loss_local_split + loss_local_gen

    total = (
        float(loss_cfg["w_set"]) * loss_set
        + float(loss_cfg["w_phys"]) * loss_phys
        + float(loss_cfg["w_pt_ratio"]) * loss_pt_ratio
        + float(loss_cfg["w_e_ratio"]) * loss_e_ratio
        + float(loss_cfg["w_budget"]) * loss_budget
        + float(loss_cfg.get("w_alloc_hard", 0.0)) * loss_alloc_hard
        + float(loss_cfg.get("w_alloc_quota", 0.0)) * loss_alloc_quota
        + float(loss_cfg["w_sparse"]) * loss_sparse
        + float(loss_cfg["w_local"]) * loss_local
    )
    return {
        "total": total,
        "set": loss_set,
        "phys": loss_phys,
        "pt_ratio": loss_pt_ratio,
        "e_ratio": loss_e_ratio,
        "budget": loss_budget,
        "alloc_hard": loss_alloc_hard,
        "alloc_quota": loss_alloc_quota,
        "pred_added_merge_mean": pred_added_merge.mean(),
        "pred_added_eff_mean": pred_added_eff.mean(),
        "hard_added_merge_mean": hard_added_merge.float().mean(),
        "hard_added_eff_mean": hard_added_eff.float().mean(),
        "sparse": loss_sparse,
        "local": loss_local,
    }


def train_reconstructor_nopriv(
    model: OfflineReconstructor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
) -> Tuple[OfflineReconstructor, Dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    best_state = None
    best_val = float("inf")
    no_improve = 0
    best_metrics: Dict[str, float] = {}

    for ep in tqdm(range(int(train_cfg["epochs"])), desc="ReconstructorNoPriv"):
        model.train()
        sc = stage_scale(ep, train_cfg)
        merge_share_tgt, quota_strength = merge_quota_schedule(ep, train_cfg)

        tr_total = tr_set = tr_phys = tr_pt_ratio = tr_e_ratio = tr_budget = 0.0
        tr_alloc_hard = tr_alloc_quota = 0.0
        tr_sparse = tr_local = 0.0
        n_tr = 0
        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            budget_total_true = batch["budget_total_true"].to(device)

            opt.zero_grad()
            out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=sc)
            losses = compute_reconstruction_losses_nopriv(
                out,
                const_hlt,
                mask_hlt,
                const_off,
                mask_off,
                budget_total_true,
                loss_cfg,
                merge_target_share=merge_share_tgt,
                quota_strength=quota_strength,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = feat_hlt.size(0)
            tr_total += losses["total"].item() * bs
            tr_set += losses["set"].item() * bs
            tr_phys += losses["phys"].item() * bs
            tr_pt_ratio += losses["pt_ratio"].item() * bs
            tr_e_ratio += losses["e_ratio"].item() * bs
            tr_budget += losses["budget"].item() * bs
            tr_alloc_hard += losses["alloc_hard"].item() * bs
            tr_alloc_quota += losses["alloc_quota"].item() * bs
            tr_sparse += losses["sparse"].item() * bs
            tr_local += losses["local"].item() * bs
            n_tr += bs

        model.eval()
        va_total = va_set = va_phys = va_pt_ratio = va_e_ratio = va_budget = 0.0
        va_alloc_hard = va_alloc_quota = 0.0
        va_sparse = va_local = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                const_off = batch["const_off"].to(device)
                mask_off = batch["mask_off"].to(device)
                budget_total_true = batch["budget_total_true"].to(device)
                out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
                losses = compute_reconstruction_losses_nopriv(
                    out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    budget_total_true,
                    loss_cfg,
                    merge_target_share=merge_share_tgt,
                    quota_strength=quota_strength,
                )
                bs = feat_hlt.size(0)
                va_total += losses["total"].item() * bs
                va_set += losses["set"].item() * bs
                va_phys += losses["phys"].item() * bs
                va_pt_ratio += losses["pt_ratio"].item() * bs
                va_e_ratio += losses["e_ratio"].item() * bs
                va_budget += losses["budget"].item() * bs
                va_alloc_hard += losses["alloc_hard"].item() * bs
                va_alloc_quota += losses["alloc_quota"].item() * bs
                va_sparse += losses["sparse"].item() * bs
                va_local += losses["local"].item() * bs
                n_va += bs

        sch.step()
        tr_total /= max(n_tr, 1)
        va_total /= max(n_va, 1)
        va_set /= max(n_va, 1)
        va_phys /= max(n_va, 1)
        va_pt_ratio /= max(n_va, 1)
        va_e_ratio /= max(n_va, 1)
        va_budget /= max(n_va, 1)
        va_alloc_hard /= max(n_va, 1)
        va_alloc_quota /= max(n_va, 1)
        va_sparse /= max(n_va, 1)
        va_local /= max(n_va, 1)

        if va_total < best_val:
            best_val = va_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_metrics = {
                "val_total": va_total,
                "val_set": va_set,
                "val_phys": va_phys,
                "val_pt_ratio": va_pt_ratio,
                "val_e_ratio": va_e_ratio,
                "val_budget": va_budget,
                "val_alloc_hard": va_alloc_hard,
                "val_alloc_quota": va_alloc_quota,
                "val_sparse": va_sparse,
                "val_local": va_local,
            }
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Ep {ep+1}: train_total={tr_total:.4f}, val_total={va_total:.4f}, best={best_val:.4f} | "
                f"set={va_set:.4f}, phys={va_phys:.4f}, pt_ratio={va_pt_ratio:.4f}, e_ratio={va_e_ratio:.4f}, "
                f"budget={va_budget:.4f}, alloc_h={va_alloc_hard:.4f}, alloc_q={va_alloc_quota:.4f}, "
                f"sparse={va_sparse:.4f}, local={va_local:.4f}, stage_scale={sc:.2f}, "
                f"merge_share_tgt={merge_share_tgt:.2f}, quota_str={quota_strength:.2f}"
            )

        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping reconstructor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def kd_loss_conf_weighted(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t_soft = torch.sigmoid(teacher_logits / temperature)
    s_soft = torch.sigmoid(student_logits / temperature)
    w = (torch.abs(torch.sigmoid(teacher_logits) - 0.5) * 2.0).detach()
    loss = F.binary_cross_entropy(s_soft, t_soft, reduction="none")
    return (loss * w).sum() / (w.sum() + 1e-8) * (temperature ** 2)


def apply_hlt_effects_realistic_with_tracking(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: Dict,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict[str, np.ndarray], np.ndarray]:
    """
    Realistic pseudo-HLT generation with token-level origin-count tracking.
    Tracking is for diagnostics only; training should not consume per-token ancestry.
    """
    rs = np.random.RandomState(int(seed))
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()
    origin_counts = hlt_mask.astype(np.int32)  # 1 for surviving offline token, accumulates under merging.

    n_initial = int(hlt_mask.sum())
    merge_lost_per_jet = np.zeros(n_jets, dtype=np.float32)
    eff_lost_per_jet = np.zeros(n_jets, dtype=np.float32)

    # 1) Pre-threshold
    pt_threshold = float(hcfg["pt_threshold_hlt"])
    below = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0
    origin_counts[~hlt_mask] = 0
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
                    dphi = wrap_phi_np(hlt[j, a, 2] - hlt[j, b, 2])
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
                    origin_counts[j, a] += origin_counts[j, b]
                    to_remove.add(b)
                    n_merged += 1
                    merge_lost_per_jet[j] += 1.0

            for idx in to_remove:
                hlt_mask[j, idx] = False
                hlt[j, idx] = 0
                origin_counts[j, idx] = 0

    # 3) Efficiency model
    jet_q = rs.lognormal(mean=0.0, sigma=float(hcfg["jet_quality_sigma"]), size=n_jets).astype(np.float32)
    jet_q = np.clip(jet_q, float(hcfg["jet_quality_min"]), float(hcfg["jet_quality_max"]))

    density = np.zeros((n_jets, max_part), dtype=np.float32)
    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        if len(valid) == 0:
            continue
        density_j = _compute_local_density_np(
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
    eps = eta_plateau * turn_on * density_term * q_eff
    eps = np.clip(eps, float(hcfg["eff_floor"]), float(hcfg["eff_ceil"]))

    u = rs.random_sample((n_jets, max_part))
    lost_eff = (u > eps) & hlt_mask
    hlt_mask[lost_eff] = False
    hlt[lost_eff] = 0
    origin_counts[lost_eff] = 0
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
        phi_new = wrap_phi_np(phi_j + rs.normal(loc=0.0, scale=sigma_phi))

        if float(hcfg["reassign_prob_base"]) > 0.0 and len(valid) > 1:
            p_reassign = float(hcfg["reassign_prob_base"]) + float(hcfg["reassign_density_coeff"]) * dens_j
            p_reassign = np.clip(p_reassign, 0.0, float(hcfg["reassign_prob_max"]))
            do_reassign = rs.random_sample(len(valid)) < p_reassign
            for ii in np.where(do_reassign)[0]:
                deta = eta_new[ii] - eta_new
                dphi = wrap_phi_np(phi_new[ii] - phi_new)
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
                n_reassigned += 1

        eta_new = np.clip(eta_new, -5.0, 5.0)
        phi_new = wrap_phi_np(phi_new)
        e_new = pt_new * np.cosh(eta_new)

        hlt[j, valid, 0] = pt_new
        hlt[j, valid, 1] = eta_new
        hlt[j, valid, 2] = phi_new
        hlt[j, valid, 3] = e_new

    post_thr = float(hcfg["post_smear_pt_threshold"])
    n_lost_threshold_post = 0
    if post_thr > 0:
        below_post = (hlt[:, :, 0] < post_thr) & hlt_mask
        hlt_mask[below_post] = False
        hlt[below_post] = 0
        origin_counts[below_post] = 0
        n_lost_threshold_post = int(below_post.sum())

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0.0
    origin_counts[~hlt_mask] = 0

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
    }
    return hlt.astype(np.float32), hlt_mask.astype(bool), stats, budget_truth, origin_counts.astype(np.int32)


@torch.no_grad()
def split_usage_diagnostics(
    model: OfflineReconstructor,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    origin_counts: np.ndarray,
    device: torch.device,
    batch_size: int,
    split_exist_threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    n_jets, max_constits = mask_hlt.shape
    K = int(model.max_split_children)
    pred_added_hard = np.zeros((n_jets, max_constits), dtype=np.int32)
    pred_added_soft = np.zeros((n_jets, max_constits), dtype=np.float32)

    for start in range(0, n_jets, int(batch_size)):
        end = min(start + int(batch_size), n_jets)
        x = torch.tensor(feat_hlt[start:end], dtype=torch.float32, device=device)
        m = torch.tensor(mask_hlt[start:end], dtype=torch.bool, device=device)
        c = torch.tensor(const_hlt[start:end], dtype=torch.float32, device=device)
        out = model(x, m, c, stage_scale=1.0)
        child_w = out["child_weight"].detach().cpu().numpy().reshape(end - start, max_constits, K)
        pred_added_soft[start:end] = child_w.sum(axis=2)
        pred_added_hard[start:end] = (child_w > float(split_exist_threshold)).sum(axis=2).astype(np.int32)

    valid = mask_hlt
    true_added = np.maximum(origin_counts - 1, 0).astype(np.int32)
    true_v = true_added[valid]
    pred_h_v = pred_added_hard[valid]
    pred_s_v = pred_added_soft[valid]

    true_clip = np.clip(true_v, 0, K)
    cm = np.zeros((K + 1, K + 1), dtype=np.int64)
    for t, p in zip(true_clip, pred_h_v):
        cm[int(t), int(np.clip(p, 0, K))] += 1

    summary = {
        "max_split_children": int(K),
        "n_tokens_eval": int(true_v.size),
        "true_added_mean": float(true_v.mean()) if true_v.size else 0.0,
        "pred_added_hard_mean": float(pred_h_v.mean()) if pred_h_v.size else 0.0,
        "pred_added_soft_mean": float(pred_s_v.mean()) if pred_s_v.size else 0.0,
        "true_cap_rate_added_gt_k": float((true_v > K).mean()) if true_v.size else 0.0,
        "pred_at_cap_rate": float((pred_h_v == K).mean()) if pred_h_v.size else 0.0,
        "pred_zero_rate": float((pred_h_v == 0).mean()) if pred_h_v.size else 0.0,
        "confusion_added_clipped": cm.tolist(),
    }
    return summary


@torch.no_grad()
def allocation_consistency_diagnostics(
    model: OfflineReconstructor,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    mask_off: np.ndarray,
    budget_total_true: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    n_jets = int(mask_hlt.shape[0])
    child_soft_all = np.zeros(n_jets, dtype=np.float32)
    gen_soft_all = np.zeros(n_jets, dtype=np.float32)
    hard_child_all = np.zeros(n_jets, dtype=np.float32)
    hard_gen_all = np.zeros(n_jets, dtype=np.float32)
    budget_total_pred_all = np.zeros(n_jets, dtype=np.float32)
    budget_merge_pred_all = np.zeros(n_jets, dtype=np.float32)
    budget_eff_pred_all = np.zeros(n_jets, dtype=np.float32)

    true_count = mask_off.sum(axis=1).astype(np.float32)
    hlt_count = mask_hlt.sum(axis=1).astype(np.float32)
    true_added = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)

    for start in range(0, n_jets, int(batch_size)):
        end = min(start + int(batch_size), n_jets)
        x = torch.tensor(feat_hlt[start:end], dtype=torch.float32, device=device)
        m = torch.tensor(mask_hlt[start:end], dtype=torch.bool, device=device)
        c = torch.tensor(const_hlt[start:end], dtype=torch.float32, device=device)
        out = model(x, m, c, stage_scale=1.0)

        child_w = out["child_weight"].clamp(0.0, 1.0)
        gen_w = out["gen_weight"].clamp(0.0, 1.0)
        soft_child = child_w.sum(dim=1)
        soft_gen = gen_w.sum(dim=1)
        soft_added = soft_child + soft_gen

        add_scores = torch.cat([child_w, gen_w], dim=1)
        max_added = add_scores.size(1)
        true_added_b = torch.tensor(true_added[start:end], dtype=torch.float32, device=device)
        k_added = torch.round(true_added_b).long().clamp(min=0, max=max_added)
        hard_mask = _topk_mask_variable_k(add_scores, k_added)
        n_child = child_w.size(1)
        hard_child = hard_mask[:, :n_child].sum(dim=1)
        hard_gen = hard_mask[:, n_child:].sum(dim=1)

        child_soft_all[start:end] = soft_child.detach().cpu().numpy().astype(np.float32)
        gen_soft_all[start:end] = soft_gen.detach().cpu().numpy().astype(np.float32)
        hard_child_all[start:end] = hard_child.detach().cpu().numpy().astype(np.float32)
        hard_gen_all[start:end] = hard_gen.detach().cpu().numpy().astype(np.float32)
        budget_total_pred_all[start:end] = (
            (out["budget_merge"] + out["budget_eff"]).detach().cpu().numpy().astype(np.float32)
        )
        budget_merge_pred_all[start:end] = out["budget_merge"].detach().cpu().numpy().astype(np.float32)
        budget_eff_pred_all[start:end] = out["budget_eff"].detach().cpu().numpy().astype(np.float32)

    soft_added_all = child_soft_all + gen_soft_all
    hard_added_all = hard_child_all + hard_gen_all
    eps = 1e-8
    soft_share = child_soft_all / np.maximum(soft_added_all, eps)
    hard_share = hard_child_all / np.maximum(hard_added_all, eps)
    budget_share = budget_merge_pred_all / np.maximum(budget_merge_pred_all + budget_eff_pred_all, eps)

    max_added_candidates = int(model.max_generated_tokens + model.max_split_children * mask_hlt.shape[1])
    sat_rate = float(np.mean(np.round(true_added).astype(np.int32) > max_added_candidates))
    low = true_added <= 2.0
    mid = (true_added > 2.0) & (true_added <= 6.0)
    high = true_added > 6.0
    def _mae(mask_sel: np.ndarray, arr: np.ndarray, tgt: np.ndarray) -> float:
        if not np.any(mask_sel):
            return float("nan")
        return float(np.mean(np.abs(arr[mask_sel] - tgt[mask_sel])))

    out = {
        "n_jets_eval": int(n_jets),
        "true_added_mean": float(true_added.mean()),
        "soft_added_merge_mean": float(child_soft_all.mean()),
        "soft_added_eff_mean": float(gen_soft_all.mean()),
        "soft_added_total_mean": float(soft_added_all.mean()),
        "hard_added_merge_mean": float(hard_child_all.mean()),
        "hard_added_eff_mean": float(hard_gen_all.mean()),
        "hard_added_total_mean": float(hard_added_all.mean()),
        "soft_vs_hard_added_mae": float(np.mean(np.abs(soft_added_all - hard_added_all))),
        "soft_vs_hard_merge_mae": float(np.mean(np.abs(child_soft_all - hard_child_all))),
        "soft_vs_hard_eff_mae": float(np.mean(np.abs(gen_soft_all - hard_gen_all))),
        "hard_vs_true_added_mae": float(np.mean(np.abs(hard_added_all - true_added))),
        "soft_vs_true_added_mae": float(np.mean(np.abs(soft_added_all - true_added))),
        "budget_total_pred_mean": float(budget_total_pred_all.mean()),
        "budget_total_vs_true_added_mae": float(np.mean(np.abs(budget_total_pred_all - true_added))),
        "budget_total_vs_target_totallost_mae": float(np.mean(np.abs(budget_total_pred_all - budget_total_true))),
        "soft_merge_share_mean": float(np.mean(soft_share)),
        "hard_merge_share_mean": float(np.mean(hard_share)),
        "budget_merge_share_mean": float(np.mean(budget_share)),
        "hard_eff_zero_rate": float(np.mean(hard_gen_all <= 0.0)),
        "hard_merge_at_least_one_rate": float(np.mean(hard_child_all >= 1.0)),
        "added_candidate_saturation_rate": sat_rate,
        "true_added_p95": float(np.percentile(true_added, 95.0)),
        "hard_added_p95": float(np.percentile(hard_added_all, 95.0)),
        "hard_vs_true_added_mae_low_trueadded_le2": _mae(low, hard_added_all, true_added),
        "hard_vs_true_added_mae_mid_trueadded_2to6": _mae(mid, hard_added_all, true_added),
        "hard_vs_true_added_mae_high_trueadded_gt6": _mae(high, hard_added_all, true_added),
        "soft_vs_true_added_mae_low_trueadded_le2": _mae(low, soft_added_all, true_added),
        "soft_vs_true_added_mae_mid_trueadded_2to6": _mae(mid, soft_added_all, true_added),
        "soft_vs_true_added_mae_high_trueadded_gt6": _mae(high, soft_added_all, true_added),
    }
    return out


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
    lambda_kd: float,
    lambda_cons: float,
    corrected_weight_floor: float,
    min_epochs: int,
    kd_teacher: nn.Module | None = None,
    kd_temperature: float = 7.0,
    kd_conf_weighted: bool = True,
) -> Tuple[OfflineReconstructor, nn.Module, Dict[str, float]]:
    for p in reconstructor.parameters():
        p.requires_grad = not freeze_reconstructor

    params = [{"params": dual_model.parameters(), "lr": float(lr_dual)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_dual), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_state_dual = None
    best_state_reco = None
    best_val_fpr50 = float("inf")
    best_val_auc = float("nan")
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
        tr_kd = 0.0
        tr_reco = 0.0
        tr_cons = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt_reco = batch["feat_hlt_reco"].to(device)
            feat_hlt_dual = batch["feat_hlt_dual"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            feat_off = batch["feat_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            b_total = batch["budget_total_true"].to(device)
            y = batch["label"].to(device)

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
            )
            logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)

            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            if kd_teacher is not None and float(lambda_kd) > 0.0:
                with torch.no_grad():
                    t_logits = kd_teacher(feat_off, mask_off).squeeze(1)
                if kd_conf_weighted:
                    loss_kd = kd_loss_conf_weighted(logits, t_logits, float(kd_temperature))
                else:
                    s_soft = torch.sigmoid(logits / float(kd_temperature))
                    t_soft = torch.sigmoid(t_logits / float(kd_temperature))
                    loss_kd = F.binary_cross_entropy(s_soft, t_soft) * (float(kd_temperature) ** 2)
            else:
                loss_kd = torch.zeros((), device=device)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()

            if float(lambda_reco) > 0.0:
                reco_losses = compute_reconstruction_losses_nopriv(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_total,
                    BASE_CONFIG["loss"],
                    merge_target_share=0.5,
                    quota_strength=0.0,
                )
                loss_reco = reco_losses["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            loss = (
                loss_cls
                + float(lambda_rank) * loss_rank
                + float(lambda_kd) * loss_kd
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
            tr_kd += loss_kd.item() * bs
            tr_reco += loss_reco.item() * bs
            tr_cons += loss_cons.item() * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_kd /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)

        va_auc, _, _, va_fpr50 = eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_model,
            loader=val_loader,
            device=device,
            corrected_weight_floor=corrected_weight_floor,
        )

        improved = np.isfinite(va_fpr50) and va_fpr50 < best_val_fpr50
        if improved:
            best_val_fpr50 = float(va_fpr50)
            best_val_auc = float(va_auc)
            best_state_dual = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{stage_name} ep {ep+1}: train_loss={tr_loss:.4f} "
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, kd={tr_kd:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_fpr50={best_val_fpr50:.6f}"
            )

        if (ep + 1) >= int(min_epochs) and no_improve >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_dual is not None:
        dual_model.load_state_dict(best_state_dual)
    if best_state_reco is not None:
        reconstructor.load_state_dict(best_state_reco)

    metrics = {
        "best_val_fpr50": float(best_val_fpr50),
        "best_val_auc": float(best_val_auc),
    }
    return reconstructor, dual_model, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=50000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_joint"),
    )
    parser.add_argument("--run_name", type=str, default="joint_default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--skip_save_models", action="store_true")

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
    parser.add_argument("--w_alloc_hard", type=float, default=0.20)
    parser.add_argument("--w_alloc_quota", type=float, default=0.12)
    parser.add_argument("--curr_merge_share_start", type=float, default=0.90)
    parser.add_argument("--curr_merge_share_mid", type=float, default=0.55)
    parser.add_argument("--curr_merge_share_final", type=float, default=0.50)

    # Stage B (tagger pretrain, reconstructor frozen)
    parser.add_argument("--stageB_epochs", type=int, default=45)
    parser.add_argument("--stageB_patience", type=int, default=12)
    parser.add_argument("--stageB_min_epochs", type=int, default=12)
    parser.add_argument("--stageB_lr_dual", type=float, default=4e-4)

    # Stage C (joint finetune)
    parser.add_argument("--stageC_epochs", type=int, default=65)
    parser.add_argument("--stageC_patience", type=int, default=14)
    parser.add_argument("--stageC_min_epochs", type=int, default=25)
    parser.add_argument("--stageC_lr_dual", type=float, default=2e-4)
    parser.add_argument("--stageC_lr_reco", type=float, default=1e-4)
    parser.add_argument("--lambda_reco", type=float, default=0.35)
    parser.add_argument("--lambda_rank", type=float, default=0.45)
    parser.add_argument("--stageC_enable_kd", action="store_true")
    parser.add_argument("--lambda_kd_stageC", type=float, default=0.20)
    parser.add_argument("--stageC_kd_temperature", type=float, default=7.0)
    parser.add_argument("--stageC_kd_conf_weighted", action="store_true")
    parser.add_argument("--lambda_cons", type=float, default=0.06)
    parser.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    parser.add_argument("--max_split_children", type=int, default=4)
    parser.add_argument("--split_diag_threshold", type=float, default=0.5)

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

    cfg = _deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)
    cfg["reconstructor_model"]["max_split_children"] = int(args.max_split_children)
    cfg["loss"]["w_alloc_hard"] = float(args.w_alloc_hard)
    cfg["loss"]["w_alloc_quota"] = float(args.w_alloc_quota)

    cfg["reconstructor_training"]["epochs"] = int(args.stageA_epochs)
    cfg["reconstructor_training"]["patience"] = int(args.stageA_patience)
    cfg["reconstructor_training"]["merge_share_start"] = float(args.curr_merge_share_start)
    cfg["reconstructor_training"]["merge_share_mid"] = float(args.curr_merge_share_mid)
    cfg["reconstructor_training"]["merge_share_final"] = float(args.curr_merge_share_final)

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

    max_jets_needed = args.offset_jets + 2 * args.n_train_jets
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

    # Two disjoint jet pools:
    #  - Slice A: stages A/B
    #  - Slice C: stage C (fresh jets)
    n_per_slice = int(args.n_train_jets)
    s0 = int(args.offset_jets)
    s1 = s0 + n_per_slice
    s2 = s1 + n_per_slice
    const_raw_a = all_const_full[s0:s1]
    labels_a = all_labels_full[s0:s1].astype(np.int64)
    const_raw_c = all_const_full[s1:s2]
    labels_c = all_labels_full[s1:s2].astype(np.int64)

    raw_mask_a = const_raw_a[:, :, 0] > 0.0
    raw_mask_c = const_raw_c[:, :, 0] > 0.0
    masks_off_a = raw_mask_a & (const_raw_a[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    masks_off_c = raw_mask_c & (const_raw_c[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off_a = const_raw_a.copy()
    const_off_c = const_raw_c.copy()
    const_off_a[~masks_off_a] = 0.0
    const_off_c[~masks_off_c] = 0.0

    print("Generating pseudo-HLT with tracking (diagnostics only)...")
    hlt_const_a, hlt_mask_a, hlt_stats_a, budget_truth_a, origin_counts_a = apply_hlt_effects_realistic_with_tracking(
        const_off_a, masks_off_a, cfg, seed=RANDOM_SEED
    )
    hlt_const_c, hlt_mask_c, hlt_stats_c, budget_truth_c, origin_counts_c = apply_hlt_effects_realistic_with_tracking(
        const_off_c, masks_off_c, cfg, seed=RANDOM_SEED + 1337
    )
    budget_merge_true_a = budget_truth_a["merge_lost_per_jet"].astype(np.float32)
    budget_eff_true_a = budget_truth_a["eff_lost_per_jet"].astype(np.float32)
    budget_total_true_a = np.maximum(masks_off_a.sum(axis=1) - hlt_mask_a.sum(axis=1), 0.0).astype(np.float32)
    budget_merge_true_c = budget_truth_c["merge_lost_per_jet"].astype(np.float32)
    budget_eff_true_c = budget_truth_c["eff_lost_per_jet"].astype(np.float32)
    budget_total_true_c = np.maximum(masks_off_c.sum(axis=1) - hlt_mask_c.sum(axis=1), 0.0).astype(np.float32)

    print("Computing features...")
    feat_off_a = compute_features(const_off_a, masks_off_a)
    feat_hlt_a = compute_features(hlt_const_a, hlt_mask_a)
    feat_off_c = compute_features(const_off_c, masks_off_c)
    feat_hlt_c = compute_features(hlt_const_c, hlt_mask_c)

    idx_a = np.arange(len(labels_a))
    train_idx_a, temp_idx_a = train_test_split(
        idx_a, test_size=0.30, random_state=RANDOM_SEED, stratify=labels_a
    )
    val_idx_a, test_idx_a = train_test_split(
        temp_idx_a, test_size=0.50, random_state=RANDOM_SEED, stratify=labels_a[temp_idx_a]
    )
    idx_c = np.arange(len(labels_c))
    train_idx_c, temp_idx_c = train_test_split(
        idx_c, test_size=0.30, random_state=RANDOM_SEED + 1, stratify=labels_c
    )
    val_idx_c, test_idx_c = train_test_split(
        temp_idx_c, test_size=0.50, random_state=RANDOM_SEED + 1, stratify=labels_c[temp_idx_c]
    )
    print(
        f"Slice A sizes: Train={len(train_idx_a)}, Val={len(val_idx_a)}, Test={len(test_idx_a)} | "
        f"Slice C sizes: Train={len(train_idx_c)}, Val={len(val_idx_c)}, Test={len(test_idx_c)}"
    )

    # Standardize with offline stats from Stage A train split.
    means, stds = get_stats(feat_off_a, masks_off_a, train_idx_a)
    feat_off_std_a = standardize(feat_off_a, masks_off_a, means, stds)
    feat_hlt_std_a = standardize(feat_hlt_a, hlt_mask_a, means, stds)
    feat_off_std_c = standardize(feat_off_c, masks_off_c, means, stds)
    feat_hlt_std_c = standardize(feat_hlt_c, hlt_mask_c, means, stds)

    # Combined test pool = test(A) U test(C), per user request.
    labels_test = np.concatenate([labels_a[test_idx_a], labels_c[test_idx_c]], axis=0)
    feat_off_test_std = np.concatenate([feat_off_std_a[test_idx_a], feat_off_std_c[test_idx_c]], axis=0)
    mask_off_test = np.concatenate([masks_off_a[test_idx_a], masks_off_c[test_idx_c]], axis=0)
    feat_hlt_test_std = np.concatenate([feat_hlt_std_a[test_idx_a], feat_hlt_std_c[test_idx_c]], axis=0)
    hlt_mask_test = np.concatenate([hlt_mask_a[test_idx_a], hlt_mask_c[test_idx_c]], axis=0)
    const_off_test = np.concatenate([const_off_a[test_idx_a], const_off_c[test_idx_c]], axis=0)
    const_hlt_test = np.concatenate([hlt_const_a[test_idx_a], hlt_const_c[test_idx_c]], axis=0)
    budget_merge_test = np.concatenate([budget_merge_true_a[test_idx_a], budget_merge_true_c[test_idx_c]], axis=0)
    budget_eff_test = np.concatenate([budget_eff_true_a[test_idx_a], budget_eff_true_c[test_idx_c]], axis=0)
    budget_total_test = np.concatenate([budget_total_true_a[test_idx_a], budget_total_true_c[test_idx_c]], axis=0)
    origin_counts_test = np.concatenate([origin_counts_a[test_idx_a], origin_counts_c[test_idx_c]], axis=0)

    hlt_stats = {
        "n_jets": int(hlt_stats_a["n_jets"] + hlt_stats_c["n_jets"]),
        "n_initial": int(hlt_stats_a["n_initial"] + hlt_stats_c["n_initial"]),
        "n_lost_threshold_pre": int(hlt_stats_a["n_lost_threshold_pre"] + hlt_stats_c["n_lost_threshold_pre"]),
        "n_merged_pairs": int(hlt_stats_a["n_merged_pairs"] + hlt_stats_c["n_merged_pairs"]),
        "n_lost_eff": int(hlt_stats_a["n_lost_eff"] + hlt_stats_c["n_lost_eff"]),
        "n_reassigned": int(hlt_stats_a["n_reassigned"] + hlt_stats_c["n_reassigned"]),
        "n_lost_threshold_post": int(hlt_stats_a["n_lost_threshold_post"] + hlt_stats_c["n_lost_threshold_post"]),
        "n_final": int(hlt_stats_a["n_final"] + hlt_stats_c["n_final"]),
        "avg_offline_per_jet": float((hlt_stats_a["avg_offline_per_jet"] + hlt_stats_c["avg_offline_per_jet"]) * 0.5),
        "avg_hlt_per_jet": float((hlt_stats_a["avg_hlt_per_jet"] + hlt_stats_c["avg_hlt_per_jet"]) * 0.5),
    }

    # Teacher / baseline
    print("\n" + "=" * 70)
    print("STEP 1: TEACHER + BASELINE")
    print("=" * 70)
    BS = int(cfg["training"]["batch_size"])

    ds_train_off = JetDataset(feat_off_std_a[train_idx_a], masks_off_a[train_idx_a], labels_a[train_idx_a])
    ds_val_off = JetDataset(feat_off_std_a[val_idx_a], masks_off_a[val_idx_a], labels_a[val_idx_a])
    ds_test_off = JetDataset(feat_off_test_std, mask_off_test, labels_test)
    dl_train_off = DataLoader(ds_train_off, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_off = DataLoader(ds_val_off, batch_size=BS, shuffle=False)
    dl_test_off = DataLoader(ds_test_off, batch_size=BS, shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    # Reuse existing training helper from local30kv2 module.
    from offline_reconstructor_no_gt_local30kv2 import train_single_view_classifier  # local import to avoid circular style issues
    teacher = train_single_view_classifier(
        teacher, dl_train_off, dl_val_off, device, cfg["training"], name="Teacher"
    )
    auc_teacher, preds_teacher, labs_teacher = eval_classifier(teacher, dl_test_off, device)
    assert np.array_equal(labels_test.astype(np.float32), labs_teacher.astype(np.float32))

    ds_train_hlt = JetDataset(feat_hlt_std_a[train_idx_a], hlt_mask_a[train_idx_a], labels_a[train_idx_a])
    ds_val_hlt = JetDataset(feat_hlt_std_a[val_idx_a], hlt_mask_a[val_idx_a], labels_a[val_idx_a])
    ds_test_hlt = JetDataset(feat_hlt_test_std, hlt_mask_test, labels_test)
    dl_train_hlt = DataLoader(ds_train_hlt, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=BS, shuffle=False)
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=BS, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = train_single_view_classifier(
        baseline, dl_train_hlt, dl_val_hlt, device, cfg["training"], name="Baseline"
    )
    auc_baseline, preds_baseline, _ = eval_classifier(baseline, dl_test_hlt, device)

    # Optional jet-level regressor to provide frozen global calibration features to dual-view tagger.
    jet_regressor = None
    jet_reg_metrics: Dict[str, object] = {"enabled": bool(args.enable_jet_regressor)}
    feat_hlt_dual_a = feat_hlt_std_a.astype(np.float32, copy=True)
    feat_hlt_dual_c = feat_hlt_std_c.astype(np.float32, copy=True)
    feat_hlt_dual_test = np.concatenate([feat_hlt_dual_a[test_idx_a], feat_hlt_dual_c[test_idx_c]], axis=0)
    if bool(args.enable_jet_regressor):
        print("\n" + "=" * 70)
        print("STEP 1B: JET-LEVEL REGRESSOR (HLT -> offline global jet targets)")
        print("=" * 70)
        # Targets:
        # [log_pt, log_e, log_m, tau21, tau32, log1p_d2, log1p_n_off, log1p_n_added]
        target_off_a, target_hlt_ref_a, target_idx = compute_jet_regression_targets(
            const_off=const_off_a,
            mask_off=masks_off_a,
            const_hlt=hlt_const_a,
            mask_hlt=hlt_mask_a,
        )
        target_off_c, target_hlt_ref_c, _ = compute_jet_regression_targets(
            const_off=const_off_c,
            mask_off=masks_off_c,
            const_hlt=hlt_const_c,
            mask_hlt=hlt_mask_c,
        )
        target_dim = int(target_off_a.shape[1])

        jet_reg_train_ds = JetRegressionDataset(
            feat_hlt_std_a[train_idx_a], hlt_mask_a[train_idx_a], target_off_a[train_idx_a]
        )
        jet_reg_val_ds = JetRegressionDataset(
            feat_hlt_std_a[val_idx_a], hlt_mask_a[val_idx_a], target_off_a[val_idx_a]
        )
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
        pred_log_a = predict_jet_regressor(
            model=jet_regressor,
            feat=feat_hlt_std_a,
            mask=hlt_mask_a,
            device=device,
            batch_size=int(cfg["training"]["batch_size"]),
        )
        pred_log_c = predict_jet_regressor(
            model=jet_regressor,
            feat=feat_hlt_std_c,
            mask=hlt_mask_c,
            device=device,
            batch_size=int(cfg["training"]["batch_size"]),
        )
        delta_vs_hlt_a = pred_log_a - target_hlt_ref_a
        delta_vs_hlt_c = pred_log_c - target_hlt_ref_c

        extra_global_a = np.concatenate([pred_log_a, delta_vs_hlt_a], axis=-1).astype(np.float32)
        extra_global_a = np.repeat(extra_global_a[:, None, :], feat_hlt_std_a.shape[1], axis=1)
        feat_hlt_dual_a = np.concatenate([feat_hlt_std_a, extra_global_a], axis=-1).astype(np.float32)
        feat_hlt_dual_a[~hlt_mask_a] = 0.0

        extra_global_c = np.concatenate([pred_log_c, delta_vs_hlt_c], axis=-1).astype(np.float32)
        extra_global_c = np.repeat(extra_global_c[:, None, :], feat_hlt_std_c.shape[1], axis=1)
        feat_hlt_dual_c = np.concatenate([feat_hlt_std_c, extra_global_c], axis=-1).astype(np.float32)
        feat_hlt_dual_c[~hlt_mask_c] = 0.0

        feat_hlt_dual_test = np.concatenate([feat_hlt_dual_a[test_idx_a], feat_hlt_dual_c[test_idx_c]], axis=0)

        # Eval metrics on val/test.
        jet_reg_val_pred = pred_log_a[val_idx_a]
        jet_reg_test_pred = np.concatenate([pred_log_a[test_idx_a], pred_log_c[test_idx_c]], axis=0)
        jet_reg_val_true = target_off_a[val_idx_a]
        jet_reg_test_true = np.concatenate([target_off_a[test_idx_a], target_off_c[test_idx_c]], axis=0)
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
    ds_train_reco = ReconstructionDatasetNoPriv(
        feat_hlt_std_a[train_idx_a], hlt_mask_a[train_idx_a], hlt_const_a[train_idx_a],
        const_off_a[train_idx_a], masks_off_a[train_idx_a],
        budget_total_true_a[train_idx_a],
    )
    ds_val_reco = ReconstructionDatasetNoPriv(
        feat_hlt_std_a[val_idx_a], hlt_mask_a[val_idx_a], hlt_const_a[val_idx_a],
        const_off_a[val_idx_a], masks_off_a[val_idx_a],
        budget_total_true_a[val_idx_a],
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
    reconstructor, reco_val_metrics = train_reconstructor_nopriv(
        reconstructor,
        dl_train_reco,
        dl_val_reco,
        device,
        cfg["reconstructor_training"],
        cfg["loss"],
    )

    # Stage B datasets (slice A only).
    ds_train_joint_b = JointDualDataset(
        feat_hlt_std_a[train_idx_a], feat_hlt_dual_a[train_idx_a], feat_off_std_a[train_idx_a],
        hlt_mask_a[train_idx_a], hlt_const_a[train_idx_a],
        const_off_a[train_idx_a], masks_off_a[train_idx_a],
        budget_total_true_a[train_idx_a],
        labels_a[train_idx_a],
    )
    ds_val_joint_b = JointDualDataset(
        feat_hlt_std_a[val_idx_a], feat_hlt_dual_a[val_idx_a], feat_off_std_a[val_idx_a],
        hlt_mask_a[val_idx_a], hlt_const_a[val_idx_a],
        const_off_a[val_idx_a], masks_off_a[val_idx_a],
        budget_total_true_a[val_idx_a],
        labels_a[val_idx_a],
    )
    # Stage C datasets (fresh slice C only).
    ds_train_joint_c = JointDualDataset(
        feat_hlt_std_c[train_idx_c], feat_hlt_dual_c[train_idx_c], feat_off_std_c[train_idx_c],
        hlt_mask_c[train_idx_c], hlt_const_c[train_idx_c],
        const_off_c[train_idx_c], masks_off_c[train_idx_c],
        budget_total_true_c[train_idx_c],
        labels_c[train_idx_c],
    )
    ds_val_joint_c = JointDualDataset(
        feat_hlt_std_c[val_idx_c], feat_hlt_dual_c[val_idx_c], feat_off_std_c[val_idx_c],
        hlt_mask_c[val_idx_c], hlt_const_c[val_idx_c],
        const_off_c[val_idx_c], masks_off_c[val_idx_c],
        budget_total_true_c[val_idx_c],
        labels_c[val_idx_c],
    )
    # Final test = A_test + C_test.
    ds_test_joint = JointDualDataset(
        feat_hlt_test_std, feat_hlt_dual_test, feat_off_test_std,
        hlt_mask_test, const_hlt_test,
        const_off_test, mask_off_test,
        budget_total_test,
        labels_test,
    )

    dl_train_joint_b = DataLoader(
        ds_train_joint_b, batch_size=BS, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint_b = DataLoader(
        ds_val_joint_b, batch_size=BS, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    dl_train_joint_c = DataLoader(
        ds_train_joint_c, batch_size=BS, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint_c = DataLoader(
        ds_val_joint_c, batch_size=BS, shuffle=False,
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
    dual_input_dim_a = int(feat_hlt_dual_a.shape[-1])
    dual_joint = DualViewCrossAttnClassifier(
        input_dim_a=dual_input_dim_a,
        input_dim_b=CORRECTED_VIEW_DIM,
        **cfg["model"],
    ).to(device)
    reconstructor, dual_joint, stageB_metrics = train_joint_dual(
        reconstructor=reconstructor,
        dual_model=dual_joint,
        train_loader=dl_train_joint_b,
        val_loader=dl_val_joint_b,
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
        lambda_rank=float(args.lambda_rank),
        lambda_kd=0.0,
        lambda_cons=float(args.lambda_cons),
        corrected_weight_floor=float(args.corrected_weight_floor),
        min_epochs=int(args.stageB_min_epochs),
    )

    print("\n" + "=" * 70)
    print("STEP 4: STAGE C (JOINT FINETUNE)")
    print("=" * 70)
    reconstructor, dual_joint, stageC_metrics = train_joint_dual(
        reconstructor=reconstructor,
        dual_model=dual_joint,
        train_loader=dl_train_joint_c,
        val_loader=dl_val_joint_c,
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
        lambda_rank=float(args.lambda_rank),
        lambda_kd=float(args.lambda_kd_stageC) if bool(args.stageC_enable_kd) else 0.0,
        lambda_cons=float(args.lambda_cons),
        corrected_weight_floor=float(args.corrected_weight_floor),
        min_epochs=int(args.stageC_min_epochs),
        kd_teacher=teacher if bool(args.stageC_enable_kd) else None,
        kd_temperature=float(args.stageC_kd_temperature),
        kd_conf_weighted=bool(args.stageC_kd_conf_weighted),
    )

    auc_joint, preds_joint, labs_joint, _ = eval_joint_model(
        reconstructor, dual_joint, dl_test_joint, device, corrected_weight_floor=float(args.corrected_weight_floor)
    )
    assert np.array_equal(labels_test.astype(np.float32), labs_joint.astype(np.float32))

    # Build hard reconstructed view for diagnostics.
    print("\n" + "=" * 70)
    print("STEP 5: RECONSTRUCTION DIAGNOSTICS")
    print("=" * 70)
    (
        reco_const_a,
        reco_mask_a,
        _,
        _,
        created_merge_count_a,
        created_eff_count_a,
        pred_budget_total_a,
        pred_budget_merge_a,
        pred_budget_eff_a,
    ) = reconstruct_dataset(
        model=reconstructor,
        feat_hlt=feat_hlt_std_a,
        mask_hlt=hlt_mask_a,
        const_hlt=hlt_const_a,
        max_constits=args.max_constits,
        device=device,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        weight_threshold=float(args.reco_weight_threshold),
        use_budget_topk=not bool(args.reco_disable_budget_topk),
    )
    (
        reco_const_c,
        reco_mask_c,
        _,
        _,
        created_merge_count_c,
        created_eff_count_c,
        pred_budget_total_c,
        pred_budget_merge_c,
        pred_budget_eff_c,
    ) = reconstruct_dataset(
        model=reconstructor,
        feat_hlt=feat_hlt_std_c,
        mask_hlt=hlt_mask_c,
        const_hlt=hlt_const_c,
        max_constits=args.max_constits,
        device=device,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        weight_threshold=float(args.reco_weight_threshold),
        use_budget_topk=not bool(args.reco_disable_budget_topk),
    )

    reco_const_test = np.concatenate([reco_const_a[test_idx_a], reco_const_c[test_idx_c]], axis=0)
    reco_mask_test = np.concatenate([reco_mask_a[test_idx_a], reco_mask_c[test_idx_c]], axis=0)
    created_merge_count_test = np.concatenate(
        [created_merge_count_a[test_idx_a], created_merge_count_c[test_idx_c]],
        axis=0,
    )
    created_eff_count_test = np.concatenate(
        [created_eff_count_a[test_idx_a], created_eff_count_c[test_idx_c]],
        axis=0,
    )
    pred_budget_total_test = np.concatenate([pred_budget_total_a[test_idx_a], pred_budget_total_c[test_idx_c]], axis=0)
    pred_budget_merge_test = np.concatenate([pred_budget_merge_a[test_idx_a], pred_budget_merge_c[test_idx_c]], axis=0)
    pred_budget_eff_test = np.concatenate([pred_budget_eff_a[test_idx_a], pred_budget_eff_c[test_idx_c]], axis=0)

    split_diag_test = split_usage_diagnostics(
        model=reconstructor,
        feat_hlt=feat_hlt_test_std,
        mask_hlt=hlt_mask_test,
        const_hlt=const_hlt_test,
        origin_counts=origin_counts_test,
        device=device,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        split_exist_threshold=float(args.split_diag_threshold),
    )
    alloc_diag_test = allocation_consistency_diagnostics(
        model=reconstructor,
        feat_hlt=feat_hlt_test_std,
        mask_hlt=hlt_mask_test,
        const_hlt=const_hlt_test,
        mask_off=mask_off_test,
        budget_total_true=budget_total_test,
        device=device,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
    )

    # Jet pT response/resolution diagnostics (test split).
    pt_truth_test = compute_jet_pt(const_off_test, mask_off_test)
    pt_hlt_test = compute_jet_pt(const_hlt_test, hlt_mask_test)
    pt_reco_test = compute_jet_pt(reco_const_test, reco_mask_test)
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
        mask_off=mask_off_test,
        hlt_mask=hlt_mask_test,
        reco_mask=reco_mask_test,
        created_merge_count=created_merge_count_test,
        created_eff_count=created_eff_count_test,
        hlt_stats=hlt_stats,
    )
    budget_summary = plot_budget_diagnostics(
        save_root=save_root,
        true_merge=budget_merge_test,
        true_eff=budget_eff_test,
        pred_merge=pred_budget_merge_test,
        pred_eff=pred_budget_eff_test,
    )
    budget_total_summary = {
        "true_total_mean": float(budget_total_test.mean()),
        "pred_total_mean": float(pred_budget_total_test.mean()),
        "total_mae": float(np.mean(np.abs(pred_budget_total_test - budget_total_test))),
        "total_bias": float(np.mean(pred_budget_total_test - budget_total_test)),
    }
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
    print("\nSplit-usage diagnostics (test split):")
    print(
        f"  True added mean={split_diag_test['true_added_mean']:.3f}, "
        f"pred hard mean={split_diag_test['pred_added_hard_mean']:.3f}, "
        f"pred soft mean={split_diag_test['pred_added_soft_mean']:.3f}"
    )
    print(
        f"  true>cap rate={split_diag_test['true_cap_rate_added_gt_k']:.3f}, "
        f"pred-at-cap rate={split_diag_test['pred_at_cap_rate']:.3f}, "
        f"pred-zero rate={split_diag_test['pred_zero_rate']:.3f}"
    )
    print("\nAllocation consistency diagnostics (test split):")
    print(
        f"  soft/hard added MAE={alloc_diag_test['soft_vs_hard_added_mae']:.3f}, "
        f"hard/true added MAE={alloc_diag_test['hard_vs_true_added_mae']:.3f}, "
        f"soft/true added MAE={alloc_diag_test['soft_vs_true_added_mae']:.3f}"
    )
    print(
        f"  shares (soft/hard/budget)={alloc_diag_test['soft_merge_share_mean']:.3f} / "
        f"{alloc_diag_test['hard_merge_share_mean']:.3f} / {alloc_diag_test['budget_merge_share_mean']:.3f}, "
        f"hard eff zero rate={alloc_diag_test['hard_eff_zero_rate']:.3f}"
    )

    # Build fixed corrected view tensors for final KD stage and additional diagnostics.
    feat_b_a, mask_b_a = build_corrected_view_numpy(
        reconstructor=reconstructor,
        feat_hlt=feat_hlt_std_a,
        mask_hlt=hlt_mask_a,
        const_hlt=hlt_const_a,
        device=device,
        batch_size=BS,
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    feat_b_c, mask_b_c = build_corrected_view_numpy(
        reconstructor=reconstructor,
        feat_hlt=feat_hlt_std_c,
        mask_hlt=hlt_mask_c,
        const_hlt=hlt_const_c,
        device=device,
        batch_size=BS,
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    feat_b_test = np.concatenate([feat_b_a[test_idx_a], feat_b_c[test_idx_c]], axis=0)
    mask_b_test = np.concatenate([mask_b_a[test_idx_a], mask_b_c[test_idx_c]], axis=0)
    soft_view_summary_test = summarize_soft_corrected_view(
        feat_b_test,
        mask_b_test,
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
            feat_hlt_dual_c[train_idx_c], hlt_mask_c[train_idx_c],
            feat_b_c[train_idx_c], mask_b_c[train_idx_c],
            feat_off_std_c[train_idx_c], masks_off_c[train_idx_c],
            labels_c[train_idx_c],
        )
        kd_val_ds = DualViewKDDataset(
            feat_hlt_dual_c[val_idx_c], hlt_mask_c[val_idx_c],
            feat_b_c[val_idx_c], mask_b_c[val_idx_c],
            feat_off_std_c[val_idx_c], masks_off_c[val_idx_c],
            labels_c[val_idx_c],
        )
        kd_test_ds = DualViewKDDataset(
            feat_hlt_dual_test, hlt_mask_test,
            feat_b_test, mask_b_test,
            feat_off_test_std, mask_off_test,
            labels_test,
        )
        kd_train_loader = DataLoader(kd_train_ds, batch_size=BS, shuffle=True, drop_last=True)
        kd_val_loader = DataLoader(kd_val_ds, batch_size=BS, shuffle=False)
        kd_test_loader = DataLoader(kd_test_ds, batch_size=BS, shuffle=False)

        kd_student = DualViewCrossAttnClassifier(
            input_dim_a=dual_input_dim_a,
            input_dim_b=CORRECTED_VIEW_DIM,
            **cfg["model"],
        ).to(device)
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
        assert np.array_equal(labels_test.astype(np.float32), labs_joint_kd.astype(np.float32))
        fpr_j_kd, tpr_j_kd, _ = roc_curve(labels_test, preds_joint_kd)
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
    fpr_t, tpr_t, _ = roc_curve(labels_test, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labels_test, preds_baseline)
    fpr_j, tpr_j, _ = roc_curve(labels_test, preds_joint)

    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.30)
    fpr30_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.30)
    fpr30_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.30)
    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.50)
    fpr50_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.50)

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"Joint Dual-View  AUC: {auc_joint:.4f}")
    if preds_joint_kd is not None:
        print(f"Joint Dual-View+KD AUC: {auc_joint_kd:.4f}")
    print()
    print(f"FPR@30 Teacher/Baseline/Joint: {fpr30_teacher:.6f} / {fpr30_baseline:.6f} / {fpr30_joint:.6f}")
    print(f"FPR@50 Teacher/Baseline/Joint: {fpr50_teacher:.6f} / {fpr50_baseline:.6f} / {fpr50_joint:.6f}")
    if preds_joint_kd is not None:
        print(f"FPR@30 Joint+KD: {fpr30_joint_kd:.6f}")
        print(f"FPR@50 Joint+KD: {fpr50_joint_kd:.6f}")

    plot_lines = [
        (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
        (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
        (tpr_j, fpr_j, "-.", f"Joint Dual (AUC={auc_joint:.3f})", "darkslateblue"),
    ]
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
        auc_joint=auc_joint,
        auc_joint_kd=auc_joint_kd,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_joint=fpr_j,
        tpr_joint=tpr_j,
        fpr_joint_kd=fpr_j_kd,
        tpr_joint_kd=tpr_j_kd,
        fpr30_teacher=fpr30_teacher,
        fpr30_baseline=fpr30_baseline,
        fpr30_joint=fpr30_joint,
        fpr30_joint_kd=fpr30_joint_kd,
        fpr50_teacher=fpr50_teacher,
        fpr50_baseline=fpr50_baseline,
        fpr50_joint=fpr50_joint,
        fpr50_joint_kd=fpr50_joint_kd,
        jet_response_pt_low=rr_field(rr_hlt_common, "pt_low"),
        jet_response_pt_high=rr_field(rr_hlt_common, "pt_high"),
        jet_response_count=rr_field(rr_hlt_common, "count"),
        jet_response_hlt_mean=rr_field(rr_hlt_common, "response"),
        jet_response_hlt_std=rr_field(rr_hlt_common, "resolution"),
        jet_response_corrected_mean=rr_field(rr_reco_common, "response"),
        jet_response_corrected_std=rr_field(rr_reco_common, "resolution"),
    )

    with open(save_root / "constituent_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(count_summary, f, indent=2)
    with open(save_root / "budget_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(budget_summary, f, indent=2)
    with open(save_root / "budget_total_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(budget_total_summary, f, indent=2)
    with open(save_root / "split_usage_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(split_diag_test, f, indent=2)
    with open(save_root / "allocation_consistency_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(alloc_diag_test, f, indent=2)
    with open(save_root / "soft_corrected_view_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(soft_view_summary_test, f, indent=2)
    with open(save_root / "jet_regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(jet_reg_metrics, f, indent=2)

    with open(save_root / "joint_stage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "jet_regressor": jet_reg_metrics,
                "stageA_reconstructor": reco_val_metrics,
                "stageB_joint": stageB_metrics,
                "stageC_joint": stageC_metrics,
                "stageD_kd": stageD_metrics,
                "split_usage_test": split_diag_test,
                "allocation_consistency_test": alloc_diag_test,
                "budget_total_test": budget_total_summary,
                "test": {
                    "auc_teacher": float(auc_teacher),
                    "auc_baseline": float(auc_baseline),
                    "auc_joint": float(auc_joint),
                    "auc_joint_kd": float(auc_joint_kd) if preds_joint_kd is not None else None,
                    "fpr30_teacher": float(fpr30_teacher),
                    "fpr30_baseline": float(fpr30_baseline),
                    "fpr30_joint": float(fpr30_joint),
                    "fpr30_joint_kd": float(fpr30_joint_kd) if preds_joint_kd is not None else None,
                    "fpr50_teacher": float(fpr50_teacher),
                    "fpr50_baseline": float(fpr50_baseline),
                    "fpr50_joint": float(fpr50_joint),
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
        if jet_regressor is not None:
            torch.save({"model": jet_regressor.state_dict(), "metrics": jet_reg_metrics}, save_root / "jet_regressor.pt")
        torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor.pt")
        torch.save({"model": dual_joint.state_dict(), "auc": auc_joint}, save_root / "dual_joint.pt")
        if kd_student is not None:
            torch.save({"model": kd_student.state_dict(), "auc": auc_joint_kd}, save_root / "dual_joint_kd.pt")

    print(f"\nSaved joint results to: {save_root}")


if __name__ == "__main__":
    main()

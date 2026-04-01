#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jet-latent non-action set2set reconstructor ablation for m2 pipeline.

Design intent:
- No keep/unsmear/split/reassign action routing.
- Input: HLT constituent set.
- Output: offline-like constituent set via direct set generation.
- Keep full m2 A/B/C training pipeline, dual-view classifier, and split counts.
- Use total added-count budget objective (no merge/eff decomposition loss).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_no_gt_local30kv2 as reco_base


def _softplus_pos(x: torch.Tensor, min_val: float = 0.0) -> torch.Tensor:
    return F.softplus(x) + float(min_val)


class OfflineReconstructorJetLatentSet2Set(nn.Module):
    """
    Non-action reconstructor:
    - Per-token direct correction branch (no discrete action competition)
    - Global latent extra-slot branch for added constituents
    - Jointly optimized as weighted set output

    Compatibility notes:
    - Exposes the same output keys consumed by the m2 training/eval stack.
    - Keeps expected module name prefixes so Stage-C progressive unfreeze still works.
    """

    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_split_children: int = 2,  # unused, kept for API compatibility
        max_generated_tokens: int = 48,
    ):
        super().__init__()
        _ = max_split_children
        self.max_generated_tokens = int(max_generated_tokens)
        self.num_heads = int(num_heads)
        self.embed_dim = int(embed_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.relpos_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, num_heads),
        )
        self.encoder_layers = nn.ModuleList(
            [reco_base.RelPosEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.token_norm = nn.LayerNorm(embed_dim)

        # Keep expected names for progressive-unfreeze groupings.
        self.action_head = nn.Linear(embed_dim, 1)  # base token existence logit
        self.unsmear_head = nn.Sequential(          # base token direct correction deltas
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
        )
        self.reassign_head = nn.Sequential(         # extra local angular shifts for base tokens
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),
        )
        self.split_exist_head = nn.Linear(embed_dim, 1)   # extra-to-base assignment prior
        self.split_delta_head = nn.Linear(embed_dim, embed_dim)  # assignment projection

        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.budget_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),  # total_count, added_count, aux
        )

        self.gen_queries = nn.Parameter(torch.randn(1, self.max_generated_tokens, embed_dim) * 0.02)
        self.gen_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.gen_norm = nn.LayerNorm(embed_dim)
        self.gen_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
        )
        self.gen_exist_head = nn.Linear(embed_dim, 1)

    def _build_relpos_bias(self, const_raw: torch.Tensor) -> torch.Tensor:
        eta = const_raw[:, :, 1]
        phi = const_raw[:, :, 2]
        deta = eta[:, :, None] - eta[:, None, :]
        dphi = torch.atan2(
            torch.sin(phi[:, :, None] - phi[:, None, :]),
            torch.cos(phi[:, :, None] - phi[:, None, :]),
        )
        dR = torch.sqrt(deta.pow(2) + dphi.pow(2) + 1e-8)
        rel = torch.stack([deta, dphi, dR], dim=-1)
        bias = self.relpos_mlp(rel)              # [B, L, L, H]
        return bias.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        stage_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        B, L, _ = feat_hlt.shape
        mask_safe = mask_hlt.clone()
        empty = ~mask_safe.any(dim=1)
        if empty.any():
            mask_safe[empty, 0] = True

        x = self.input_proj(feat_hlt)
        rel_bias = self._build_relpos_bias(const_hlt)
        for layer in self.encoder_layers:
            x = layer(x, mask_safe, rel_bias)
        x = self.token_norm(x)

        pt = const_hlt[..., 0].clamp(min=eps)
        eta = const_hlt[..., 1].clamp(min=-5.0, max=5.0)
        phi = const_hlt[..., 2]
        E = const_hlt[..., 3].clamp(min=eps)

        # Base token direct correction (no action simplex).
        base_delta = self.unsmear_head(x)
        base_ang = 0.30 * torch.tanh(self.reassign_head(x))

        d_logpt = float(stage_scale) * 0.70 * torch.tanh(base_delta[..., 0])
        d_eta = float(stage_scale) * (0.50 * torch.tanh(base_delta[..., 1]) + base_ang[..., 0])
        d_phi = float(stage_scale) * (0.50 * torch.tanh(base_delta[..., 2]) + base_ang[..., 1])
        d_logE = float(stage_scale) * 0.70 * torch.tanh(base_delta[..., 3])

        tok_pt = torch.exp(torch.clamp(torch.log(pt) + d_logpt, min=-9.0, max=9.0))
        tok_eta = (eta + d_eta).clamp(min=-5.0, max=5.0)
        tok_phi = reco_base.wrap_phi_t(phi + d_phi)
        tok_E = torch.exp(torch.clamp(torch.log(E) + d_logE, min=-9.0, max=11.0))
        tok_E = torch.maximum(tok_E, tok_pt * torch.cosh(tok_eta))
        tok_tokens = torch.stack([tok_pt, tok_eta, tok_phi, tok_E], dim=-1)

        tok_w = torch.sigmoid(self.action_head(x).squeeze(-1))
        tok_w = tok_w * mask_hlt.float()

        # Jet context.
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=~mask_safe, need_weights=False)
        ctx = pooled.squeeze(1)
        budget_raw = self.budget_head(ctx)
        budget_total = _softplus_pos(budget_raw[:, 0])       # total predicted count
        budget_added = _softplus_pos(budget_raw[:, 1])       # total added-count budget
        budget_aux = _softplus_pos(budget_raw[:, 2])         # aux (kept for compatibility)

        # Extra slots generated from jet latent + token memory.
        gq = self.gen_queries.expand(B, -1, -1)
        gen_dec, _ = self.gen_attn(gq, x, x, key_padding_mask=~mask_safe, need_weights=False)
        gen_dec = self.gen_norm(gen_dec)
        gen_raw = self.gen_head(gen_dec)
        gen_exist = torch.sigmoid(self.gen_exist_head(gen_dec).squeeze(-1)) * float(stage_scale)

        # Jet-centered parameterization for extra slots.
        m = mask_hlt.float()
        n_valid = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        jet_logpt = torch.log((pt * m).sum(dim=1, keepdim=True).clamp(min=eps))
        jet_logE = torch.log((E * m).sum(dim=1, keepdim=True).clamp(min=eps))
        jet_eta = (eta * m).sum(dim=1, keepdim=True) / n_valid
        jet_phi = torch.atan2((torch.sin(phi) * m).sum(dim=1, keepdim=True), (torch.cos(phi) * m).sum(dim=1, keepdim=True))

        ex_pt = torch.exp(torch.clamp(jet_logpt + 0.90 * torch.tanh(gen_raw[..., 0]), min=-9.0, max=9.0))
        ex_eta = (jet_eta + 0.85 * torch.tanh(gen_raw[..., 1])).clamp(min=-5.0, max=5.0)
        ex_phi = reco_base.wrap_phi_t(jet_phi + 0.85 * torch.tanh(gen_raw[..., 2]))
        ex_E = torch.exp(torch.clamp(jet_logE + 0.90 * torch.tanh(gen_raw[..., 3]), min=-9.0, max=11.0))
        ex_E = torch.maximum(ex_E, ex_pt * torch.cosh(ex_eta))
        extra_tokens = torch.stack([ex_pt, ex_eta, ex_phi, ex_E], dim=-1)

        # Calibrate extra mass to added-count budget, then calibrate total count.
        sum_extra = gen_exist.sum(dim=1, keepdim=True).clamp(min=eps)
        scale_extra = (budget_added.unsqueeze(1) / sum_extra).clamp(min=0.25, max=4.0)
        extra_w = (gen_exist * scale_extra).clamp(0.0, 1.0)

        sum_total = (tok_w.sum(dim=1, keepdim=True) + extra_w.sum(dim=1, keepdim=True)).clamp(min=eps)
        scale_total = (budget_total.unsqueeze(1) / sum_total).clamp(min=0.25, max=4.0)
        tok_w = (tok_w * scale_total).clamp(0.0, 1.0)
        extra_w = (extra_w * scale_total).clamp(0.0, 1.0)

        # Extra->base soft assignment for corrected-view augmentation.
        proj_extra = self.split_delta_head(gen_dec)   # [B, G, D]
        assign_logits = torch.einsum("bgd,bld->bgl", proj_extra, x) / math.sqrt(float(self.embed_dim))
        base_prior = self.split_exist_head(x).squeeze(-1)  # [B, L]
        assign_logits = assign_logits + base_prior.unsqueeze(1)
        assign_logits = assign_logits.masked_fill(~mask_safe.unsqueeze(1), -1e4)
        extra_to_base = torch.softmax(assign_logits, dim=-1)

        # Keep a tiny dummy eff branch slot to avoid empty-tensor reductions in Stage-C loss_cons.
        dummy_gen_tokens = torch.zeros((B, 1, 4), dtype=tok_tokens.dtype, device=tok_tokens.device)
        dummy_gen_w = torch.zeros((B, 1), dtype=tok_w.dtype, device=tok_w.device)

        cand_tokens = torch.cat([tok_tokens, extra_tokens, dummy_gen_tokens], dim=1)
        cand_weights = torch.cat([tok_w, extra_w, dummy_gen_w], dim=1)

        tok_merge_flag = torch.zeros_like(tok_w)
        ex_merge_flag = torch.ones_like(extra_w)
        gen_merge_flag = torch.zeros_like(dummy_gen_w)
        cand_merge_flags = torch.cat([tok_merge_flag, ex_merge_flag, gen_merge_flag], dim=1)
        cand_eff_flags = torch.zeros_like(cand_weights)

        # Pseudo action probabilities for compatibility/diagnostics.
        keep = tok_w
        unsmear = torch.zeros_like(keep)
        split = torch.zeros_like(keep)
        reassign = torch.zeros_like(keep)
        action_prob = torch.stack([keep, unsmear, split, reassign], dim=-1)
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True).clamp(min=eps)

        # Placeholder split_delta tensor for compatibility.
        split_delta = torch.zeros((B, L, 1, 3), dtype=tok_tokens.dtype, device=tok_tokens.device)

        return {
            "cand_tokens": cand_tokens,
            "cand_weights": cand_weights,
            "cand_merge_flags": cand_merge_flags,
            "cand_eff_flags": cand_eff_flags,
            "action_prob": action_prob,
            "child_weight": extra_w,            # treat extras as "added merge-like" branch
            "gen_weight": dummy_gen_w,          # disabled eff branch in this ablation
            "budget_total": budget_total,
            "budget_merge": budget_added,       # interpreted as total added-count budget
            "budget_eff": budget_aux * 0.0,     # no eff decomposition objective
            "split_delta": split_delta,
            "gen_tokens": dummy_gen_tokens,
            "tok_tokens": tok_tokens,
            "tok_weights": tok_w,
            "extra_to_base": extra_to_base,
        }


def compute_reconstruction_losses_weighted_set2set(
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
    sw = None if sample_weight is None else sample_weight.float().clamp(min=0.0)

    pred = out["cand_tokens"]
    w = out["cand_weights"].clamp(0.0, 1.0)

    cost = reco_base._token_cost_matrix(pred, const_off)
    valid_tgt = mask_off.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))

    # Chamfer set loss (same family as m2).
    pred_to_tgt = cost.min(dim=2).values
    loss_pred_to_tgt = (w * pred_to_tgt).sum(dim=1) / (w.sum(dim=1) + eps)

    penalty = float(loss_cfg.get("unselected_penalty", 0.0)) * (1.0 - w).unsqueeze(2)
    tgt_to_pred = (cost + penalty).min(dim=1).values
    tgt_w = mask_off.float()
    loss_tgt_to_pred = (tgt_to_pred * tgt_w).sum(dim=1) / (tgt_w.sum(dim=1) + eps)
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

    # Radial profile.
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

    edges = torch.linspace(0.0, radial_max_dr, steps=n_radial_bins + 1, device=pred.device, dtype=pred.dtype)
    lo = edges[:-1].view(1, 1, -1)
    hi = edges[1:].view(1, 1, -1)
    pred_in = (dr_pred.unsqueeze(-1) >= lo) & (dr_pred.unsqueeze(-1) < hi)
    tgt_in = (dr_tgt.unsqueeze(-1) >= lo) & (dr_tgt.unsqueeze(-1) < hi)
    pred_in[:, :, -1] = pred_in[:, :, -1] | (dr_pred >= radial_max_dr)
    tgt_in[:, :, -1] = tgt_in[:, :, -1] | (dr_tgt >= radial_max_dr)
    pred_prof = (pred_tok_pt.unsqueeze(-1) * pred_in.float()).sum(dim=1)
    tgt_prof = (tgt_tok_pt.unsqueeze(-1) * tgt_in.float()).sum(dim=1)
    pred_prof = pred_prof / (pred_prof.sum(dim=1, keepdim=True) + eps)
    tgt_prof = tgt_prof / (tgt_prof.sum(dim=1, keepdim=True) + eps)
    loss_radial_profile_vec = F.smooth_l1_loss(pred_prof, tgt_prof, reduction="none").mean(dim=1)

    # Total added-count budget (requested objective).
    true_count = mask_off.float().sum(dim=1)
    hlt_count = mask_hlt.float().sum(dim=1)
    true_added = (true_count - hlt_count).clamp(min=0.0)

    pred_count = w.sum(dim=1)
    pred_added_from_count = (pred_count - hlt_count).clamp(min=0.0)
    pred_added_head = out["budget_merge"]  # interpreted as total-added head
    pred_total_head = out["budget_total"]

    loss_budget_vec = (
        F.smooth_l1_loss(pred_added_from_count, true_added, reduction="none")
        + F.smooth_l1_loss(pred_added_head, true_added, reduction="none")
        + F.smooth_l1_loss(pred_total_head, true_count, reduction="none")
    )

    # Sparsity: focus on added slots.
    child_w = out["child_weight"]
    if child_w.numel() > 0:
        loss_sparse_vec = child_w.mean(dim=1)
    else:
        loss_sparse_vec = torch.zeros_like(true_added)

    # Locality: penalize predicted tokens far from HLT support.
    h_eta = const_hlt[:, :, 1]
    h_phi = const_hlt[:, :, 2]
    p_eta = pred[:, :, 1].unsqueeze(2)
    p_phi = pred[:, :, 2].unsqueeze(2)
    d_eta = p_eta - h_eta.unsqueeze(1)
    d_phi = torch.atan2(torch.sin(p_phi - h_phi.unsqueeze(1)), torch.cos(p_phi - h_phi.unsqueeze(1)))
    dR = torch.sqrt(d_eta.pow(2) + d_phi.pow(2) + eps)
    dR = torch.where(mask_hlt.unsqueeze(1), dR, torch.full_like(dR, 1e4))
    nearest = dR.min(dim=2).values
    excess = F.relu(nearest - float(loss_cfg.get("gen_local_radius", 0.0)))
    loss_local_vec = (w * excess).sum(dim=1) / (w.sum(dim=1) + eps)

    fp_cost_thresh = float(loss_cfg.get("fp_mass_cost_thresh", 0.80))
    fp_cost_tau = float(max(loss_cfg.get("fp_mass_tau", 0.10), 1e-4))
    min_pred_cost = cost.min(dim=2).values
    unmatched_soft = torch.sigmoid((min_pred_cost - fp_cost_thresh) / fp_cost_tau)
    loss_fp_mass_vec = (w * unmatched_soft).sum(dim=1) / (w.sum(dim=1) + eps)

    total_vec = (
        float(loss_cfg.get("w_set", 1.0)) * loss_set_vec
        + float(loss_cfg.get("w_phys", 0.0)) * loss_phys_vec
        + float(loss_cfg.get("w_pt_ratio", 0.0)) * loss_pt_ratio_vec
        + float(loss_cfg.get("w_m_ratio", 0.0)) * loss_m_ratio_vec
        + float(loss_cfg.get("w_e_ratio", 0.0)) * loss_e_ratio_vec
        + float(loss_cfg.get("w_radial_profile", 0.0)) * loss_radial_profile_vec
        + float(loss_cfg.get("w_budget", 0.0)) * loss_budget_vec
        + float(loss_cfg.get("w_sparse", 0.0)) * loss_sparse_vec
        + float(loss_cfg.get("w_local", 0.0)) * loss_local_vec
        + float(loss_cfg.get("w_fp_mass", 0.0)) * loss_fp_mass_vec
    )

    return {
        "total": base._weighted_batch_mean(total_vec, sw),
        "set": base._weighted_batch_mean(loss_set_vec, sw),
        "phys": base._weighted_batch_mean(loss_phys_vec, sw),
        "pt_ratio": base._weighted_batch_mean(loss_pt_ratio_vec, sw),
        "m_ratio": base._weighted_batch_mean(loss_m_ratio_vec, sw),
        "e_ratio": base._weighted_batch_mean(loss_e_ratio_vec, sw),
        "radial_profile": base._weighted_batch_mean(loss_radial_profile_vec, sw),
        "budget": base._weighted_batch_mean(loss_budget_vec, sw),
        "sparse": base._weighted_batch_mean(loss_sparse_vec, sw),
        "local": base._weighted_batch_mean(loss_local_vec, sw),
        "fp_mass": base._weighted_batch_mean(loss_fp_mass_vec, sw),
    }


def build_soft_corrected_view_set2set(
    reco_out: Dict[str, torch.Tensor],
    weight_floor: float = 1e-4,
    scale_features_by_weight: bool = True,
    include_flags: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    tok_tokens = reco_out.get("tok_tokens", None)
    tok_weights = reco_out.get("tok_weights", None)
    if tok_tokens is None or tok_weights is None:
        # Fallback: treat first L slots as base tokens using action_prob length.
        L = int(reco_out["action_prob"].shape[1])
        tok_tokens = reco_out["cand_tokens"][:, :L, :]
        tok_weights = reco_out["cand_weights"][:, :L]

    mask_b = tok_weights > float(weight_floor)
    none_valid = ~mask_b.any(dim=1)
    if none_valid.any():
        mask_b = mask_b.clone()
        mask_b[none_valid, 0] = True

    feat7 = base.compute_features_torch(tok_tokens, mask_b)
    if bool(scale_features_by_weight):
        feat7 = feat7 * tok_weights.unsqueeze(-1)

    # Per-base-token added support from extra slots.
    parent_added = torch.zeros_like(tok_weights)
    if ("extra_to_base" in reco_out) and ("child_weight" in reco_out):
        assign = reco_out["extra_to_base"]             # [B, G, L]
        extra_w = reco_out["child_weight"]             # [B, G]
        if assign.numel() > 0 and extra_w.numel() > 0:
            parent_added = (assign * extra_w.unsqueeze(-1)).sum(dim=1)
            parent_added = parent_added.clamp(0.0, 1.0)

    eff_share = torch.zeros_like(tok_weights)
    extra = torch.stack([tok_weights, parent_added, eff_share], dim=-1)
    if bool(include_flags):
        merge_flag = torch.zeros_like(tok_weights)
        eff_flag = torch.zeros_like(tok_weights)
        extra = torch.cat([extra, merge_flag.unsqueeze(-1), eff_flag.unsqueeze(-1)], dim=-1)

    feat_b = torch.cat([feat7, extra], dim=-1)
    feat_b = torch.nan_to_num(feat_b, nan=0.0, posinf=0.0, neginf=0.0)
    feat_b = feat_b * mask_b.unsqueeze(-1).float()
    return feat_b, mask_b


def _identity_enforce(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return out


def _identity_wrap(model: nn.Module) -> nn.Module:
    return model


def _patch_base_module() -> None:
    base.OfflineReconstructor = OfflineReconstructorJetLatentSet2Set  # type: ignore[assignment]
    base.compute_reconstruction_losses_weighted = compute_reconstruction_losses_weighted_set2set  # type: ignore[assignment]
    base.build_soft_corrected_view = build_soft_corrected_view_set2set  # type: ignore[assignment]
    base.enforce_unmerge_only_output = _identity_enforce  # type: ignore[assignment]
    base.wrap_reconstructor_unmerge_only = _identity_wrap  # type: ignore[assignment]


def main() -> None:
    _patch_base_module()
    base.main()


if __name__ == "__main__":
    main()

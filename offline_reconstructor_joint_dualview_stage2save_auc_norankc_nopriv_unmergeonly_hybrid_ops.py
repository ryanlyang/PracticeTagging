#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid operation-aware JetClass reconstructor for m2-style training.

Design:
- Shared transformer encoder on full HLT token features.
- Per-token edit branch (unsmear/reassign-like residual correction).
- Per-token split branch (2 children) with parent uplift so split products can
  recover under-measured parent kinematics.
- Jet-latent generator branch for missing tokens (eff/threshold-like losses).
- Set2set training with count/budget + jet-level consistency losses.

This module is independent from older jetlatent-only runners.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_no_gt_local30kv2 as reco_base

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:
    linear_sum_assignment = None  # type: ignore


def _softplus_pos(x: torch.Tensor, min_val: float = 0.0) -> torch.Tensor:
    return F.softplus(x) + float(min_val)


class OfflineReconstructorHybridOps(nn.Module):
    """
    Operation-aware reconstructor with three branches:
    1) token edit branch (constituent-by-constituent correction),
    2) split branch (for merge-product parents),
    3) jet-latent generation branch (for truly missing tokens).

    Compatibility notes:
    - Exposes output keys consumed by current stage-A/B/C pipelines.
    - Keeps expected module names for progressive-unfreeze grouping.
    """

    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_split_children: int = 2,
        max_generated_tokens: int = 48,
    ):
        super().__init__()
        self.max_split_children = int(max(1, max_split_children))
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

        # Edit branch (kept names to stay compatible with existing unfreeze logic).
        self.action_head = nn.Linear(embed_dim, 1)        # token existence prior
        self.unsmear_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
        )
        self.reassign_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),
        )

        # Split branch (merge-product parents).
        self.split_exist_head = nn.Linear(embed_dim, 1)  # parent split-gate
        self.split_uplift_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),  # pt/e uplift logits
        )
        self.split_delta_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.max_split_children * 4),
        )
        self.split_child_exist_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.max_split_children),
        )

        # Jet-level budget heads.
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.budget_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),  # total_count, added_count, aux
        )

        # Generator branch (missing-token recovery).
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
        bias = self.relpos_mlp(rel)
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

        # -------------------------------------------------------------
        # Edit branch
        # -------------------------------------------------------------
        base_delta = self.unsmear_head(x)
        base_ang = 0.28 * torch.tanh(self.reassign_head(x))

        d_logpt = float(stage_scale) * 0.65 * torch.tanh(base_delta[..., 0])
        d_eta = float(stage_scale) * (0.45 * torch.tanh(base_delta[..., 1]) + base_ang[..., 0])
        d_phi = float(stage_scale) * (0.45 * torch.tanh(base_delta[..., 2]) + base_ang[..., 1])
        d_logE = float(stage_scale) * 0.65 * torch.tanh(base_delta[..., 3])

        tok_pt = torch.exp(torch.clamp(torch.log(pt) + d_logpt, min=-9.0, max=9.0))
        tok_eta = (eta + d_eta).clamp(min=-5.0, max=5.0)
        tok_phi = reco_base.wrap_phi_t(phi + d_phi)
        tok_E = torch.exp(torch.clamp(torch.log(E) + d_logE, min=-9.0, max=11.0))
        tok_E = torch.maximum(tok_E, tok_pt * torch.cosh(tok_eta))
        tok_tokens = torch.stack([tok_pt, tok_eta, tok_phi, tok_E], dim=-1)

        tok_exist = torch.sigmoid(self.action_head(x).squeeze(-1)) * mask_hlt.float()

        # -------------------------------------------------------------
        # Split branch (merge-product parents)
        # -------------------------------------------------------------
        p_split = torch.sigmoid(self.split_exist_head(x).squeeze(-1)) * mask_hlt.float()

        # Keep branch weight is reduced when split probability is high.
        tok_w_raw = (tok_exist * (1.0 - p_split)).clamp(0.0, 1.0)
        split_parent_w = (tok_exist * p_split).clamp(0.0, 1.0)

        # Parent uplift allows unmerge products to recover under-measured parent energy/pt.
        uplift_raw = self.split_uplift_head(x)
        uplift_pt = 0.35 * torch.tanh(uplift_raw[..., 0])
        uplift_e = 0.35 * torch.tanh(uplift_raw[..., 1])
        parent_pt = tok_pt * (1.0 + uplift_pt)
        parent_E = tok_E * (1.0 + uplift_e)
        parent_pt = parent_pt.clamp(min=eps)
        parent_E = torch.maximum(parent_E.clamp(min=eps), parent_pt * torch.cosh(tok_eta))

        split_delta = self.split_delta_head(x).view(B, L, self.max_split_children, 4)
        child_exist = torch.sigmoid(self.split_child_exist_head(x))

        c_logpt = 0.45 * torch.tanh(split_delta[..., 0])
        c_eta = 0.35 * torch.tanh(split_delta[..., 1])
        c_phi = 0.35 * torch.tanh(split_delta[..., 2])
        c_logE = 0.45 * torch.tanh(split_delta[..., 3])

        base_pt = parent_pt.unsqueeze(-1)
        base_eta = tok_eta.unsqueeze(-1)
        base_phi = tok_phi.unsqueeze(-1)
        base_E = parent_E.unsqueeze(-1)

        child_pt = torch.exp(torch.clamp(torch.log(base_pt + eps) + c_logpt, min=-9.0, max=9.0))
        child_eta = (base_eta + c_eta).clamp(min=-5.0, max=5.0)
        child_phi = reco_base.wrap_phi_t(base_phi + c_phi)
        child_E = torch.exp(torch.clamp(torch.log(base_E + eps) + c_logE, min=-9.0, max=11.0))
        child_E = torch.maximum(child_E, child_pt * torch.cosh(child_eta))
        split_tokens = torch.stack([child_pt, child_eta, child_phi, child_E], dim=-1)

        child_w_raw = (child_exist * split_parent_w.unsqueeze(-1)).clamp(0.0, 1.0)
        split_w_flat = child_w_raw.reshape(B, L * self.max_split_children)
        split_tok_flat = split_tokens.reshape(B, L * self.max_split_children, 4)
        split_parent_added = child_w_raw.sum(dim=-1)

        # -------------------------------------------------------------
        # Generator branch (efficiency/threshold-like missing tokens)
        # -------------------------------------------------------------
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=~mask_safe, need_weights=False)
        ctx = pooled.squeeze(1)
        budget_raw = self.budget_head(ctx)
        budget_total = _softplus_pos(budget_raw[:, 0])
        budget_added = _softplus_pos(budget_raw[:, 1])
        budget_aux = _softplus_pos(budget_raw[:, 2])

        gq = self.gen_queries.expand(B, -1, -1)
        gen_dec, _ = self.gen_attn(gq, x, x, key_padding_mask=~mask_safe, need_weights=False)
        gen_dec = self.gen_norm(gen_dec)
        gen_raw = self.gen_head(gen_dec)
        gen_exist = torch.sigmoid(self.gen_exist_head(gen_dec).squeeze(-1)) * float(stage_scale)

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
        gen_tokens = torch.stack([ex_pt, ex_eta, ex_phi, ex_E], dim=-1)

        # Count calibration.
        # Reserve part of the added budget for split-added count, then let generator fill the rest.
        split_added_est = (split_w_flat.sum(dim=1) - split_parent_w.sum(dim=1)).clamp(min=0.0)
        gen_target_added = (budget_added - split_added_est).clamp(min=0.0)

        sum_gen = gen_exist.sum(dim=1, keepdim=True).clamp(min=eps)
        gen_scale = (gen_target_added.unsqueeze(1) / sum_gen).clamp(min=0.25, max=4.0)
        gen_w_raw = (gen_exist * gen_scale).clamp(0.0, 1.0)

        # Global total-count calibration.
        pred_count_raw = tok_w_raw.sum(dim=1, keepdim=True) + split_w_flat.sum(dim=1, keepdim=True) + gen_w_raw.sum(dim=1, keepdim=True)
        total_scale = (budget_total.unsqueeze(1) / pred_count_raw.clamp(min=eps)).clamp(min=0.25, max=4.0)
        tok_w = (tok_w_raw * total_scale).clamp(0.0, 1.0)
        split_w = (split_w_flat * total_scale).clamp(0.0, 1.0)
        gen_w = (gen_w_raw * total_scale).clamp(0.0, 1.0)

        # Generator -> base assignment for corrected-view augmentation.
        assign_logits = torch.einsum("bgd,bld->bgl", gen_dec, x) / math.sqrt(float(self.embed_dim))
        assign_logits = assign_logits.masked_fill(~mask_safe.unsqueeze(1), -1e4)
        extra_to_base = torch.softmax(assign_logits, dim=-1)

        # Candidate set.
        cand_tokens = torch.cat([tok_tokens, split_tok_flat, gen_tokens], dim=1)
        cand_weights = torch.cat([tok_w, split_w, gen_w], dim=1)

        tok_merge_flag = torch.zeros_like(tok_w)
        split_merge_flag = torch.ones_like(split_w)
        gen_merge_flag = torch.zeros_like(gen_w)
        cand_merge_flags = torch.cat([tok_merge_flag, split_merge_flag, gen_merge_flag], dim=1)

        tok_eff_flag = torch.zeros_like(tok_w)
        split_eff_flag = torch.zeros_like(split_w)
        gen_eff_flag = torch.ones_like(gen_w)
        cand_eff_flags = torch.cat([tok_eff_flag, split_eff_flag, gen_eff_flag], dim=1)

        # Compatibility action probabilities (4-way simplex).
        keep = tok_w_raw
        unsmear = torch.zeros_like(keep)
        split = split_parent_w
        reassign = torch.zeros_like(keep)
        action_prob = torch.stack([keep, unsmear, split, reassign], dim=-1)
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True).clamp(min=eps)

        # Keep split_delta key compatibility: [B, L, C, 3]
        split_delta_out = split_delta[..., :3]

        return {
            "cand_tokens": cand_tokens,
            "cand_weights": cand_weights,
            "cand_merge_flags": cand_merge_flags,
            "cand_eff_flags": cand_eff_flags,
            "action_prob": action_prob,
            "child_weight": split_w,            # split children branch
            "gen_weight": gen_w,                # generator branch
            "budget_total": budget_total,
            "budget_merge": budget_added,
            "budget_eff": budget_aux * 0.0,
            "split_delta": split_delta_out,
            "gen_tokens": gen_tokens,
            "tok_tokens": tok_tokens,
            "tok_weights": tok_w,
            "extra_to_base": extra_to_base,
            "split_parent_added": split_parent_added,
        }


def compute_reconstruction_losses_weighted_hybrid_ops(
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

    def _loss_set_chamfer_vec() -> torch.Tensor:
        pred_to_tgt = cost.min(dim=2).values
        loss_pred_to_tgt = (w * pred_to_tgt).sum(dim=1) / (w.sum(dim=1) + eps)
        penalty = float(loss_cfg.get("unselected_penalty", 0.0)) * (1.0 - w).unsqueeze(2)
        tgt_to_pred = (cost + penalty).min(dim=1).values
        tgt_w = mask_off.float()
        loss_tgt_to_pred = (tgt_to_pred * tgt_w).sum(dim=1) / (tgt_w.sum(dim=1) + eps)
        return loss_pred_to_tgt + loss_tgt_to_pred

    def _loss_set_hungarian_vec() -> torch.Tensor:
        if linear_sum_assignment is None:
            raise RuntimeError(
                "loss_set_mode='hungarian' requires scipy.optimize.linear_sum_assignment, "
                "but SciPy is unavailable in this environment."
            )
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

            matched_cost = c_bt[row_t, col_t]
            l_cov = matched_cost.mean()

            wb = w[bi]
            matched_mask = torch.zeros_like(wb, dtype=torch.bool)
            matched_mask[row_t] = True
            wb_m = wb[matched_mask]
            l_prec_match = (wb_m * matched_cost).sum() / (wb_m.sum() + eps)
            unmatched_mass = wb[~matched_mask].sum() / (wb.sum() + eps)
            l_fp = float(loss_cfg.get("unselected_penalty", 0.0)) * unmatched_mass
            loss_list.append(l_cov + l_prec_match + l_fp)
        return torch.stack(loss_list, dim=0)

    set_mode = str(loss_cfg.get("set_loss_mode", "chamfer")).strip().lower()
    if set_mode == "chamfer":
        loss_set_vec = _loss_set_chamfer_vec()
    elif set_mode == "hungarian":
        loss_set_vec = _loss_set_hungarian_vec()
    elif set_mode == "combo":
        w_chamfer = max(float(loss_cfg.get("combo_w_chamfer", 0.0)), 0.0)
        w_hungarian = max(float(loss_cfg.get("combo_w_hungarian", 0.0)), 0.0)
        w_sum = w_chamfer + w_hungarian
        if w_sum <= 0.0:
            raise ValueError("loss_set_mode='combo' requires positive combo weights.")
        loss_set_vec = (w_chamfer / w_sum) * _loss_set_chamfer_vec() + (w_hungarian / w_sum) * _loss_set_hungarian_vec()
    else:
        raise ValueError(f"Unsupported set_loss_mode in hybrid ops: {set_mode}")

    pred_px, pred_py, pred_pz, pred_E = reco_base._weighted_fourvec_sums(pred, w)
    true_px, true_py, true_pz, true_E = reco_base._weighted_fourvec_sums(const_off, mask_off.float())

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

    norm = true_px.abs() + true_py.abs() + true_pz.abs() + true_E.abs() + 1.0
    loss_phys_vec = (
        (pred_px - true_px).abs()
        + (pred_py - true_py).abs()
        + (pred_pz - true_pz).abs()
        + (pred_E - true_E).abs()
    ) / norm

    # Radial profile around offline axis.
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
        d_phi = torch.atan2(torch.sin(tok_phi - axis_phi.unsqueeze(1)), torch.cos(tok_phi - axis_phi.unsqueeze(1)))
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

    # Count/budget objectives.
    true_count = mask_off.float().sum(dim=1)
    hlt_count = mask_hlt.float().sum(dim=1)
    true_added = (true_count - hlt_count).clamp(min=0.0)

    pred_count = w.sum(dim=1)
    pred_added_from_count = (pred_count - hlt_count).clamp(min=0.0)
    pred_added_head = out["budget_merge"]
    pred_total_head = out["budget_total"]

    loss_budget_vec = (
        F.smooth_l1_loss(pred_added_from_count, true_added, reduction="none")
        + F.smooth_l1_loss(pred_added_head, true_added, reduction="none")
        + F.smooth_l1_loss(pred_total_head, true_count, reduction="none")
    )

    # Sparsity over all added branches.
    split_w = out.get("child_weight", None)
    gen_w = out.get("gen_weight", None)
    if split_w is not None and gen_w is not None:
        all_added_w = torch.cat([split_w, gen_w], dim=1)
        loss_sparse_vec = all_added_w.mean(dim=1)
    elif split_w is not None:
        loss_sparse_vec = split_w.mean(dim=1)
    elif gen_w is not None:
        loss_sparse_vec = gen_w.mean(dim=1)
    else:
        loss_sparse_vec = torch.zeros_like(true_added)

    # Locality: penalize predictions far from HLT support.
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

    # Optional false-positive soft mass penalty.
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


def build_soft_corrected_view_hybrid_ops(
    reco_out: Dict[str, torch.Tensor],
    weight_floor: float = 1e-4,
    scale_features_by_weight: bool = True,
    include_flags: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    tok_tokens = reco_out.get("tok_tokens", None)
    tok_weights = reco_out.get("tok_weights", None)
    if tok_tokens is None or tok_weights is None:
        # Fallback: infer base token count from action_prob.
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

    # Added support on each parent: split children + generated assignments.
    split_added = reco_out.get("split_parent_added", torch.zeros_like(tok_weights))

    parent_added_gen = torch.zeros_like(tok_weights)
    if ("extra_to_base" in reco_out) and ("gen_weight" in reco_out):
        assign = reco_out["extra_to_base"]
        gen_w = reco_out["gen_weight"]
        if assign.numel() > 0 and gen_w.numel() > 0:
            parent_added_gen = (assign * gen_w.unsqueeze(-1)).sum(dim=1)

    parent_added = (split_added + parent_added_gen).clamp(0.0, 2.0)

    # eff_share kept as smooth budget signal.
    valid_count = mask_b.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    eff_share_scalar = reco_out.get("budget_eff", torch.zeros_like(valid_count.squeeze(-1))).unsqueeze(1)
    eff_share = (eff_share_scalar / valid_count).clamp(0.0, 1.0).expand_as(tok_weights)

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
    base.OfflineReconstructor = OfflineReconstructorHybridOps  # type: ignore[assignment]
    base.compute_reconstruction_losses_weighted = compute_reconstruction_losses_weighted_hybrid_ops  # type: ignore[assignment]
    base.build_soft_corrected_view = build_soft_corrected_view_hybrid_ops  # type: ignore[assignment]
    base.enforce_unmerge_only_output = _identity_enforce  # type: ignore[assignment]
    base.wrap_reconstructor_unmerge_only = _identity_wrap  # type: ignore[assignment]


def main() -> None:
    _patch_base_module()
    base.main()


if __name__ == "__main__":
    main()

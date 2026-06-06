#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple three-operation JetClass reconstructor.

Candidate pool:
  1. unsmeared parent candidate for each HLT constituent,
  2. optional free split children for each HLT constituent,
  3. global generated candidates from jet context.

The split branch is intentionally not locally energy conserving. The model is
controlled with soft set matching, soft budget matching, sparsity, and jet-level
response/axis losses instead of hard operation constraints.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_hybrid_ops as hybrid_ops
import offline_reconstructor_no_gt_local30kv2 as reco_base


SIMPLE3_LOSS_W_AXIS = 0.08


def configure_simple3_ops(loss_w_axis: float | None = None) -> None:
    global SIMPLE3_LOSS_W_AXIS
    if loss_w_axis is not None:
        SIMPLE3_LOSS_W_AXIS = float(max(0.0, loss_w_axis))


def _softplus_pos(x: torch.Tensor, min_val: float = 0.0) -> torch.Tensor:
    return F.softplus(x) + float(min_val)


class OfflineReconstructorSimple3Ops(nn.Module):
    """Less constrained parent/split/generate reconstructor."""

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
        split_suppression: float = 0.65,
        budget_scale_strength: float = 0.35,
    ):
        super().__init__()
        self.max_split_children = int(max(1, max_split_children))
        self.max_generated_tokens = int(max_generated_tokens)
        self.num_heads = int(num_heads)
        self.embed_dim = int(embed_dim)
        self.split_suppression = float(split_suppression)
        self.budget_scale_strength = float(budget_scale_strength)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.relpos_mlp = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, num_heads))
        self.encoder_layers = nn.ModuleList(
            [reco_base.RelPosEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.token_norm = nn.LayerNorm(embed_dim)

        # Unsmear branch. This absorbs small reassignment-like eta/phi motion.
        self.action_head = nn.Linear(embed_dim, 1)
        self.unsmear_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 4))

        # Free split branch. Children are not constrained to sum to the parent.
        self.split_exist_head = nn.Linear(embed_dim, 1)
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

        # Soft budget heads, kept deliberately as regularizers rather than hard constraints.
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.budget_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 3))

        # Global generator branch.
        self.gen_queries = nn.Parameter(torch.randn(1, self.max_generated_tokens, embed_dim) * 0.02)
        self.gen_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.gen_norm = nn.LayerNorm(embed_dim)
        self.gen_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 4))
        self.gen_exist_head = nn.Linear(embed_dim, 1)

    def _build_relpos_bias(self, const_raw: torch.Tensor) -> torch.Tensor:
        eta = const_raw[:, :, 1]
        phi = const_raw[:, :, 2]
        deta = eta[:, :, None] - eta[:, None, :]
        dphi = torch.atan2(torch.sin(phi[:, :, None] - phi[:, None, :]), torch.cos(phi[:, :, None] - phi[:, None, :]))
        dR = torch.sqrt(deta.pow(2) + dphi.pow(2) + 1e-8)
        rel = torch.stack([deta, dphi, dR], dim=-1)
        return self.relpos_mlp(rel).permute(0, 3, 1, 2).contiguous()

    def forward(self, feat_hlt: torch.Tensor, mask_hlt: torch.Tensor, const_hlt: torch.Tensor, stage_scale: float = 1.0) -> Dict[str, torch.Tensor]:
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
        ene = const_hlt[..., 3].clamp(min=eps)

        # Parent/unsmear branch: more flexible eta/phi shifts replace explicit reassign.
        delta = self.unsmear_head(x)
        d_logpt = float(stage_scale) * 0.75 * torch.tanh(delta[..., 0])
        d_eta = float(stage_scale) * 0.80 * torch.tanh(delta[..., 1])
        d_phi = float(stage_scale) * 0.80 * torch.tanh(delta[..., 2])
        d_loge = float(stage_scale) * 0.75 * torch.tanh(delta[..., 3])
        tok_pt = torch.exp(torch.clamp(torch.log(pt) + d_logpt, min=-9.0, max=9.0))
        tok_eta = (eta + d_eta).clamp(min=-5.0, max=5.0)
        tok_phi = reco_base.wrap_phi_t(phi + d_phi)
        tok_E = torch.exp(torch.clamp(torch.log(ene) + d_loge, min=-9.0, max=11.0))
        tok_E = torch.maximum(tok_E, tok_pt * torch.cosh(tok_eta))
        tok_tokens = torch.stack([tok_pt, tok_eta, tok_phi, tok_E], dim=-1)
        tok_exist = torch.sigmoid(self.action_head(x).squeeze(-1)) * mask_hlt.float()

        # Free split branch: two children are proposed for every parent, gated softly.
        p_split = torch.sigmoid(self.split_exist_head(x).squeeze(-1)) * mask_hlt.float()
        tok_w_raw = (tok_exist * (1.0 - self.split_suppression * p_split)).clamp(0.0, 1.0)
        split_parent_w = (tok_exist * p_split).clamp(0.0, 1.0)

        split_delta = self.split_delta_head(x).view(B, L, self.max_split_children, 4)
        child_exist = torch.sigmoid(self.split_child_exist_head(x))
        # Large enough range that split products may exceed parent pT/E when HLT is biased.
        c_logpt = 1.05 * torch.tanh(split_delta[..., 0])
        c_eta = 0.65 * torch.tanh(split_delta[..., 1])
        c_phi = 0.65 * torch.tanh(split_delta[..., 2])
        c_loge = 1.05 * torch.tanh(split_delta[..., 3])
        child_pt = torch.exp(torch.clamp(torch.log(tok_pt.unsqueeze(-1) + eps) + c_logpt, min=-9.0, max=9.0))
        child_eta = (tok_eta.unsqueeze(-1) + c_eta).clamp(min=-5.0, max=5.0)
        child_phi = reco_base.wrap_phi_t(tok_phi.unsqueeze(-1) + c_phi)
        child_E = torch.exp(torch.clamp(torch.log(tok_E.unsqueeze(-1) + eps) + c_loge, min=-9.0, max=11.0))
        child_E = torch.maximum(child_E, child_pt * torch.cosh(child_eta))
        split_tokens = torch.stack([child_pt, child_eta, child_phi, child_E], dim=-1)
        child_w_raw = (child_exist * split_parent_w.unsqueeze(-1)).clamp(0.0, 1.0)
        split_w_flat = child_w_raw.reshape(B, L * self.max_split_children)
        split_tok_flat = split_tokens.reshape(B, L * self.max_split_children, 4)
        split_parent_added = child_w_raw.sum(dim=-1)

        # Jet context and generator.
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
        jet_logE = torch.log((ene * m).sum(dim=1, keepdim=True).clamp(min=eps))
        jet_eta = (eta * m).sum(dim=1, keepdim=True) / n_valid
        jet_phi = torch.atan2((torch.sin(phi) * m).sum(dim=1, keepdim=True), (torch.cos(phi) * m).sum(dim=1, keepdim=True))
        ex_pt = torch.exp(torch.clamp(jet_logpt + 0.95 * torch.tanh(gen_raw[..., 0]), min=-9.0, max=9.0))
        ex_eta = (jet_eta + 0.90 * torch.tanh(gen_raw[..., 1])).clamp(min=-5.0, max=5.0)
        ex_phi = reco_base.wrap_phi_t(jet_phi + 0.90 * torch.tanh(gen_raw[..., 2]))
        ex_E = torch.exp(torch.clamp(jet_logE + 0.95 * torch.tanh(gen_raw[..., 3]), min=-9.0, max=11.0))
        ex_E = torch.maximum(ex_E, ex_pt * torch.cosh(ex_eta))
        gen_tokens = torch.stack([ex_pt, ex_eta, ex_phi, ex_E], dim=-1)

        # Soft budget allocation. Budget guides count but does not hard-normalize the pool.
        split_added_est = (split_w_flat.sum(dim=1) - split_parent_w.sum(dim=1)).clamp(min=0.0)
        gen_target_added = (budget_added - split_added_est).clamp(min=0.0)
        sum_gen = gen_exist.sum(dim=1, keepdim=True).clamp(min=eps)
        gen_scale_raw = (gen_target_added.unsqueeze(1) / sum_gen).clamp(min=0.35, max=3.0)
        gen_scale = 1.0 + self.budget_scale_strength * (gen_scale_raw - 1.0)
        gen_w_raw = (gen_exist * gen_scale).clamp(0.0, 1.0)

        pred_count_raw = tok_w_raw.sum(dim=1, keepdim=True) + split_w_flat.sum(dim=1, keepdim=True) + gen_w_raw.sum(dim=1, keepdim=True)
        total_scale_raw = (budget_total.unsqueeze(1) / pred_count_raw.clamp(min=eps)).clamp(min=0.35, max=3.0)
        total_scale = 1.0 + self.budget_scale_strength * (total_scale_raw - 1.0)
        tok_w = (tok_w_raw * total_scale).clamp(0.0, 1.0)
        split_w = (split_w_flat * total_scale).clamp(0.0, 1.0)
        gen_w = (gen_w_raw * total_scale).clamp(0.0, 1.0)

        assign_logits = torch.einsum("bgd,bld->bgl", gen_dec, x) / math.sqrt(float(self.embed_dim))
        assign_logits = assign_logits.masked_fill(~mask_safe.unsqueeze(1), -1e4)
        extra_to_base = torch.softmax(assign_logits, dim=-1)

        cand_tokens = torch.cat([tok_tokens, split_tok_flat, gen_tokens], dim=1)
        cand_weights = torch.cat([tok_w, split_w, gen_w], dim=1)
        cand_merge_flags = torch.cat([torch.zeros_like(tok_w), torch.ones_like(split_w), torch.zeros_like(gen_w)], dim=1)
        cand_eff_flags = torch.cat([torch.zeros_like(tok_w), torch.zeros_like(split_w), torch.ones_like(gen_w)], dim=1)

        # Four-slot compatibility for V2 attr code: keep/unsmear/split/reassign.
        keep = tok_w_raw
        unsmear = torch.zeros_like(keep)
        split = split_parent_w
        reassign = torch.zeros_like(keep)
        action_prob = torch.stack([keep, unsmear, split, reassign], dim=-1)
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True).clamp(min=eps)

        return {
            "cand_tokens": cand_tokens,
            "cand_weights": cand_weights,
            "cand_merge_flags": cand_merge_flags,
            "cand_eff_flags": cand_eff_flags,
            "action_prob": action_prob,
            "child_weight": split_w,
            "gen_weight": gen_w,
            "budget_total": budget_total,
            "budget_merge": budget_added,
            "budget_eff": budget_aux * 0.0,
            "split_delta": split_delta[..., :3],
            "gen_tokens": gen_tokens,
            "tok_tokens": tok_tokens,
            "tok_weights": tok_w,
            "extra_to_base": extra_to_base,
            "split_parent_added": split_parent_added,
        }


def _axis_loss(out: Dict[str, torch.Tensor], const_off: torch.Tensor, mask_off: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    pred = out["cand_tokens"]
    weights = out["cand_weights"].clamp(0.0, 1.0)
    pred_px, pred_py, pred_pz, _ = reco_base._weighted_fourvec_sums(pred, weights)
    true_px, true_py, true_pz, _ = reco_base._weighted_fourvec_sums(const_off, mask_off.float())
    eps = 1e-8
    pred_pt = torch.sqrt(pred_px.pow(2) + pred_py.pow(2) + eps)
    true_pt = torch.sqrt(true_px.pow(2) + true_py.pow(2) + eps)
    pred_eta = torch.asinh(pred_pz / pred_pt.clamp(min=eps))
    true_eta = torch.asinh(true_pz / true_pt.clamp(min=eps))
    pred_phi = torch.atan2(pred_py, pred_px)
    true_phi = torch.atan2(true_py, true_px)
    deta = pred_eta - true_eta
    dphi = torch.atan2(torch.sin(pred_phi - true_phi), torch.cos(pred_phi - true_phi))
    vec = F.smooth_l1_loss(deta, torch.zeros_like(deta), reduction="none") + F.smooth_l1_loss(dphi, torch.zeros_like(dphi), reduction="none")
    sw = None if sample_weight is None else sample_weight.float().clamp(min=0.0).to(vec.device)
    if sw is None:
        return vec.mean()
    return (vec * sw).sum() / sw.sum().clamp(min=eps)


def compute_reconstruction_losses_weighted_simple3_ops(
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
    losses = hybrid_ops.compute_reconstruction_losses_weighted_hybrid_ops(
        out,
        const_hlt,
        mask_hlt,
        const_off,
        mask_off,
        budget_merge_true,
        budget_eff_true,
        loss_cfg,
        sample_weight=sample_weight,
    )
    loss_axis = _axis_loss(out, const_off, mask_off, sample_weight)
    losses["axis"] = loss_axis
    losses["total"] = losses["total"] + float(SIMPLE3_LOSS_W_AXIS) * loss_axis
    return losses


def build_soft_corrected_view_simple3_ops(
    reco_out: Dict[str, torch.Tensor],
    weight_floor: float = 1e-4,
    scale_features_by_weight: bool = True,
    include_flags: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return hybrid_ops.build_soft_corrected_view_hybrid_ops(
        reco_out,
        weight_floor=weight_floor,
        scale_features_by_weight=scale_features_by_weight,
        include_flags=include_flags,
    )

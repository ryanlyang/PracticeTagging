#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m31: Jet-latent set2set + contextual selector routing + diffusion-lite denoising.

Built as a fork over the m30 wrapper:
- keeps the same base m2 A/B/C pipeline and contextual K-hypothesis selector routing,
- adds iterative denoise refinement on reconstructed token slots,
- uses HLT-conditioned denoise updates over a small number of steps (diffusion-lite),
- keeps Stage-C strategy modes (all_three / selector_dual / selector_only),
- optional "single corrected-view tagger" (no dual branch usage).

Note:
The denoiser is deterministic per forward pass except optional training-time init noise.
Selector is refreshed at Stage-B/Stage-C boundaries from current reconstructor outputs.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_no_gt_local30kv2 as reco_base


def _softplus_pos(x: torch.Tensor, min_val: float = 0.0) -> torch.Tensor:
    return F.softplus(x) + float(min_val)


def _weighted_batch_mean(vec: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return vec.mean()
    sw = sample_weight.float().clamp(min=0.0)
    return (vec * sw).sum() / sw.sum().clamp(min=1e-6)


def _compute_jet_pt_torch(const: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    px = const[..., 0] * torch.cos(const[..., 2]) * mask.float()
    py = const[..., 0] * torch.sin(const[..., 2]) * mask.float()
    return torch.sqrt(px.sum(dim=1).pow(2) + py.sum(dim=1).pow(2) + 1e-8)


def _standardize_features_torch(feat: torch.Tensor, mask: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    x = (feat - means.view(1, 1, -1)) / stds.view(1, 1, -1).clamp(min=1e-6)
    return x * mask.unsqueeze(-1).float()


def _set_loss_chamfer_vec(
    pred_const: torch.Tensor,
    pred_w: torch.Tensor,
    tgt_const: torch.Tensor,
    tgt_mask: torch.Tensor,
    unmatched_penalty: float,
) -> torch.Tensor:
    eps = 1e-8
    cost = reco_base._token_cost_matrix(pred_const, tgt_const)
    valid_tgt = tgt_mask.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))

    p2t = cost.min(dim=2).values
    loss_p2t = (pred_w * p2t).sum(dim=1) / (pred_w.sum(dim=1) + eps)

    penalty = float(unmatched_penalty) * (1.0 - pred_w).unsqueeze(2)
    t2p = (cost + penalty).min(dim=1).values
    tgt_w = tgt_mask.float()
    loss_t2p = (t2p * tgt_w).sum(dim=1) / (tgt_w.sum(dim=1) + eps)
    return loss_p2t + loss_t2p


@dataclass
class M29Options:
    num_hypotheses: int = 6
    winner_mode: str = "hybrid"  # reco/tag/hybrid
    winner_alpha: float = 1.0
    winner_beta: float = 0.6
    w_best_set: float = 2.5
    w_diversity: float = 0.08
    selector_epochs: int = 30
    selector_lr: float = 2e-3
    selector_patience: int = 8
    selector_rank_weight: float = 0.2
    selector_rank_margin: float = 0.25
    selector_weight_floor: float = 1e-4
    selector_hidden: int = 96
    selector_heads: int = 4
    selector_dropout: float = 0.10
    selector_ce_weight: float = 1.0
    selector_kl_weight: float = 0.35
    selector_soft_temp: float = 0.35
    selector_use_soft_targets: bool = True
    selector_use_hlt_logit: bool = True
    selector_train_before_stageb: bool = True
    stagec_mode: str = "all_three"  # all_three/selector_dual/selector_only
    soft_routing_train: bool = True
    routing_temp: float = 1.0
    single_corrected_tagger: bool = False
    denoise_steps: int = 3
    denoise_init_noise_std: float = 0.03
    denoise_delta_scale: float = 0.35
    denoise_step_embed_dim: int = 16


class _M29State:
    def __init__(self) -> None:
        self.opts = M29Options()
        self.teacher: Optional[nn.Module] = None
        self.baseline: Optional[nn.Module] = None
        self.feat_means: Optional[torch.Tensor] = None
        self.feat_stds: Optional[torch.Tensor] = None
        self.selector: Optional["HypothesisSelector"] = None
        self.selector_ready: bool = False


M29_STATE = _M29State()


class HypothesisSelector(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 96, heads: int = 4, dropout: float = 0.10):
        super().__init__()
        hidden = max(int(hidden), 32)
        heads = max(int(heads), 1)
        if hidden % heads != 0:
            heads = 1

        self.in_proj = nn.Linear(feat_dim, hidden)
        self.in_norm = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, dropout=float(dropout), batch_first=True)
        self.attn_drop = nn.Dropout(float(dropout))
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden * 2, hidden),
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, feat_bkf: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(self.in_proj(feat_bkf))
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = x + self.attn_drop(attn_out)
        x = x + self.ff(self.ff_norm(x))
        return self.out(self.out_norm(x)).squeeze(-1)


def selector_rank_loss(sel_logits: torch.Tensor, winner_idx: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    bsz, k = sel_logits.shape
    w = sel_logits.gather(1, winner_idx.view(-1, 1))  # [B,1]
    diff = float(margin) - (w - sel_logits)
    keep = torch.ones_like(sel_logits, dtype=torch.bool)
    keep = keep.scatter(1, winner_idx.view(-1, 1), False)
    rank = F.relu(diff) * keep.float()
    return rank.sum() / keep.float().sum().clamp(min=1.0)


def _gather_k(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    gather_idx = idx.view(b, 1, *([1] * (x.ndim - 2))).expand(b, 1, *x.shape[2:])
    return x.gather(dim=1, index=gather_idx).squeeze(1)


def _extract_hyp_out(out: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    out_h: Dict[str, torch.Tensor] = {}
    k = None
    if "cand_tokens" in out and out["cand_tokens"].ndim >= 4:
        k = int(out["cand_tokens"].shape[1])
    for kk, vv in out.items():
        if not torch.is_tensor(vv):
            out_h[kk] = vv
            continue
        if k is not None and vv.ndim >= 2 and int(vv.shape[1]) == k:
            out_h[kk] = _gather_k(vv, idx)
        else:
            out_h[kk] = vv
    return out_h


def _mix_hyp_out(out: Dict[str, torch.Tensor], probs: torch.Tensor) -> Dict[str, torch.Tensor]:
    out_m: Dict[str, torch.Tensor] = {}
    k = int(probs.shape[1])
    for kk, vv in out.items():
        if not torch.is_tensor(vv):
            out_m[kk] = vv
            continue
        if vv.ndim >= 2 and int(vv.shape[1]) == k:
            w = probs
            while w.ndim < vv.ndim:
                w = w.unsqueeze(-1)
            out_m[kk] = (vv * w).sum(dim=1)
        else:
            out_m[kk] = vv
    return out_m


def _compute_single_losses(
    out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    loss_cfg: Dict,
    sample_weight: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    eps = 1e-8
    sw = None if sample_weight is None else sample_weight.float().clamp(min=0.0)

    pred = out["cand_tokens"]
    w = out["cand_weights"].clamp(0.0, 1.0)

    loss_set_vec = _set_loss_chamfer_vec(
        pred,
        w,
        const_off,
        mask_off,
        unmatched_penalty=float(loss_cfg.get("unselected_penalty", 0.0)),
    )

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

    child_w = out["child_weight"]
    if child_w.numel() > 0:
        loss_sparse_vec = child_w.mean(dim=1)
    else:
        loss_sparse_vec = torch.zeros_like(true_added)

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

    cost = reco_base._token_cost_matrix(pred, const_off)
    valid_tgt = mask_off.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))
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
        "_set_vec": loss_set_vec,
    }


class OfflineReconstructorJetLatentSet2SetK(nn.Module):
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
        num_hypotheses: Optional[int] = None,
    ):
        super().__init__()
        _ = max_split_children
        self.max_generated_tokens = int(max_generated_tokens)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_hypotheses = int(max(1, M29_STATE.opts.num_hypotheses if num_hypotheses is None else num_hypotheses))

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

        self.action_head = nn.Linear(embed_dim, 1)
        self.unsmear_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 4))
        self.reassign_head = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, 2))
        self.split_exist_head = nn.Linear(embed_dim, 1)
        self.split_delta_head = nn.Linear(embed_dim, embed_dim)

        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.budget_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 3))

        self.gen_queries = nn.Parameter(torch.randn(1, self.max_generated_tokens, embed_dim) * 0.02)
        self.gen_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.gen_norm = nn.LayerNorm(embed_dim)
        self.gen_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 4))
        self.gen_exist_head = nn.Linear(embed_dim, 1)

        self.hyp_embed = nn.Embedding(self.num_hypotheses, embed_dim)

        # Diffusion-lite denoiser: iterative HLT-conditioned refinement on token slots.
        self.denoise_steps = int(max(M29_STATE.opts.denoise_steps, 0))
        self.denoise_delta_scale = float(max(M29_STATE.opts.denoise_delta_scale, 0.0))
        self.denoise_init_noise_std = float(max(M29_STATE.opts.denoise_init_noise_std, 0.0))
        self.denoise_step_embed = nn.Embedding(int(max(M29_STATE.opts.denoise_step_embed_dim, 4)), embed_dim)
        self.denoise_token_in = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.denoise_cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=max(1, num_heads // 2), dropout=dropout, batch_first=True)
        self.denoise_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.denoise_norm = nn.LayerNorm(embed_dim)
        self.denoise_head = nn.Linear(embed_dim, 4)

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

    def _tokens_to_denoise_repr(self, tok_tokens: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        pt = tok_tokens[..., 0].clamp(min=eps)
        eta = tok_tokens[..., 1].clamp(min=-5.0, max=5.0)
        phi = tok_tokens[..., 2]
        E = tok_tokens[..., 3].clamp(min=eps)
        return torch.stack([torch.log(pt), eta, torch.sin(phi), torch.cos(phi), torch.log(E)], dim=-1)

    def _denoise_refine_tokens(
        self,
        tok_tokens: torch.Tensor,
        x: torch.Tensor,
        mask_safe: torch.Tensor,
        stage_scale: float,
    ) -> torch.Tensor:
        eps = 1e-8
        n_steps = int(max(self.denoise_steps, 0))
        if n_steps <= 0 or self.denoise_delta_scale <= 0.0:
            return tok_tokens

        cur = tok_tokens
        for s in range(n_steps):
            rep = self._tokens_to_denoise_repr(cur)
            q = self.denoise_token_in(rep)
            step_idx = min(s, int(self.denoise_step_embed.num_embeddings - 1))
            q = q + self.denoise_step_embed.weight[step_idx].view(1, 1, -1)

            ctx, _ = self.denoise_cross_attn(q, x, x, key_padding_mask=~mask_safe, need_weights=False)
            h = self.denoise_norm(q + ctx + self.denoise_ff(ctx))
            d = torch.tanh(self.denoise_head(h))

            scale = float(stage_scale) * float(self.denoise_delta_scale) / float(max(n_steps, 1))
            d_logpt = scale * d[..., 0]
            d_eta = scale * 0.8 * d[..., 1]
            d_phi = scale * 0.8 * d[..., 2]
            d_logE = scale * d[..., 3]

            pt = cur[..., 0].clamp(min=eps)
            eta = cur[..., 1].clamp(min=-5.0, max=5.0)
            phi = cur[..., 2]
            E = cur[..., 3].clamp(min=eps)

            new_pt = torch.exp(torch.clamp(torch.log(pt) + d_logpt, min=-9.0, max=9.0))
            new_eta = (eta + d_eta).clamp(min=-5.0, max=5.0)
            new_phi = reco_base.wrap_phi_t(phi + d_phi)
            new_E = torch.exp(torch.clamp(torch.log(E) + d_logE, min=-9.0, max=11.0))
            new_E = torch.maximum(new_E, new_pt * torch.cosh(new_eta))

            cur = torch.stack([new_pt, new_eta, new_phi, new_E], dim=-1)

        return cur

    def _decode_one(
        self,
        x_base: torch.Tensor,
        mask_safe: torch.Tensor,
        const_hlt: torch.Tensor,
        stage_scale: float,
        hyp_idx: int,
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        bsz, l_tok, _ = x_base.shape
        h_bias = self.hyp_embed.weight[hyp_idx].view(1, 1, -1)
        x = x_base + h_bias

        pt = const_hlt[..., 0].clamp(min=eps)
        eta = const_hlt[..., 1].clamp(min=-5.0, max=5.0)
        phi = const_hlt[..., 2]
        E = const_hlt[..., 3].clamp(min=eps)

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

        # Optional training-time corruption at init, then iterative denoise refinement.
        if self.training and self.denoise_init_noise_std > 0.0:
            z = torch.randn_like(tok_tokens)
            n_logpt = self.denoise_init_noise_std * 0.60 * z[..., 0]
            n_eta = self.denoise_init_noise_std * 0.40 * z[..., 1]
            n_phi = self.denoise_init_noise_std * 0.40 * z[..., 2]
            n_logE = self.denoise_init_noise_std * 0.60 * z[..., 3]
            tok_pt = torch.exp(torch.clamp(torch.log(tok_tokens[..., 0].clamp(min=eps)) + n_logpt, min=-9.0, max=9.0))
            tok_eta = (tok_tokens[..., 1] + n_eta).clamp(min=-5.0, max=5.0)
            tok_phi = reco_base.wrap_phi_t(tok_tokens[..., 2] + n_phi)
            tok_E = torch.exp(torch.clamp(torch.log(tok_tokens[..., 3].clamp(min=eps)) + n_logE, min=-9.0, max=11.0))
            tok_E = torch.maximum(tok_E, tok_pt * torch.cosh(tok_eta))
            tok_tokens = torch.stack([tok_pt, tok_eta, tok_phi, tok_E], dim=-1)

        tok_tokens = self._denoise_refine_tokens(tok_tokens, x, mask_safe, stage_scale)

        tok_w = torch.sigmoid(self.action_head(x).squeeze(-1)) * mask_safe.float()

        q = self.pool_query.expand(bsz, -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=~mask_safe, need_weights=False)
        ctx = pooled.squeeze(1)
        budget_raw = self.budget_head(ctx)
        budget_total = _softplus_pos(budget_raw[:, 0])
        budget_added = _softplus_pos(budget_raw[:, 1])
        budget_aux = _softplus_pos(budget_raw[:, 2])

        gq = self.gen_queries.expand(bsz, -1, -1)
        gen_dec, _ = self.gen_attn(gq, x, x, key_padding_mask=~mask_safe, need_weights=False)
        gen_dec = self.gen_norm(gen_dec)
        gen_raw = self.gen_head(gen_dec)
        gen_exist = torch.sigmoid(self.gen_exist_head(gen_dec).squeeze(-1)) * float(stage_scale)

        m = mask_safe.float()
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

        sum_extra = gen_exist.sum(dim=1, keepdim=True).clamp(min=eps)
        scale_extra = (budget_added.unsqueeze(1) / sum_extra).clamp(min=0.25, max=4.0)
        extra_w = (gen_exist * scale_extra).clamp(0.0, 1.0)

        sum_total = (tok_w.sum(dim=1, keepdim=True) + extra_w.sum(dim=1, keepdim=True)).clamp(min=eps)
        scale_total = (budget_total.unsqueeze(1) / sum_total).clamp(min=0.25, max=4.0)
        tok_w = (tok_w * scale_total).clamp(0.0, 1.0)
        extra_w = (extra_w * scale_total).clamp(0.0, 1.0)

        proj_extra = self.split_delta_head(gen_dec)
        assign_logits = torch.einsum("bgd,bld->bgl", proj_extra, x) / math.sqrt(float(self.embed_dim))
        base_prior = self.split_exist_head(x).squeeze(-1)
        assign_logits = assign_logits + base_prior.unsqueeze(1)
        assign_logits = assign_logits.masked_fill(~mask_safe.unsqueeze(1), -1e4)
        extra_to_base = torch.softmax(assign_logits, dim=-1)

        dummy_gen_tokens = torch.zeros((bsz, 1, 4), dtype=tok_tokens.dtype, device=tok_tokens.device)
        dummy_gen_w = torch.zeros((bsz, 1), dtype=tok_w.dtype, device=tok_w.device)

        cand_tokens = torch.cat([tok_tokens, extra_tokens, dummy_gen_tokens], dim=1)
        cand_weights = torch.cat([tok_w, extra_w, dummy_gen_w], dim=1)

        tok_merge_flag = torch.zeros_like(tok_w)
        ex_merge_flag = torch.ones_like(extra_w)
        gen_merge_flag = torch.zeros_like(dummy_gen_w)
        cand_merge_flags = torch.cat([tok_merge_flag, ex_merge_flag, gen_merge_flag], dim=1)
        cand_eff_flags = torch.zeros_like(cand_weights)

        keep = tok_w
        unsmear = torch.zeros_like(keep)
        split = torch.zeros_like(keep)
        reassign = torch.zeros_like(keep)
        action_prob = torch.stack([keep, unsmear, split, reassign], dim=-1)
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True).clamp(min=eps)

        split_delta = torch.zeros((bsz, l_tok, 1, 3), dtype=tok_tokens.dtype, device=tok_tokens.device)

        return {
            "cand_tokens": cand_tokens,
            "cand_weights": cand_weights,
            "cand_merge_flags": cand_merge_flags,
            "cand_eff_flags": cand_eff_flags,
            "action_prob": action_prob,
            "child_weight": extra_w,
            "gen_weight": dummy_gen_w,
            "budget_total": budget_total,
            "budget_merge": budget_added,
            "budget_eff": budget_aux * 0.0,
            "split_delta": split_delta,
            "gen_tokens": dummy_gen_tokens,
            "tok_tokens": tok_tokens,
            "tok_weights": tok_w,
            "extra_to_base": extra_to_base,
        }

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        stage_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        mask_safe = mask_hlt.clone()
        empty = ~mask_safe.any(dim=1)
        if empty.any():
            mask_safe[empty, 0] = True

        x = self.input_proj(feat_hlt)
        rel_bias = self._build_relpos_bias(const_hlt)
        for layer in self.encoder_layers:
            x = layer(x, mask_safe, rel_bias)
        x = self.token_norm(x)

        if self.num_hypotheses <= 1:
            out = self._decode_one(x, mask_safe, const_hlt, stage_scale, hyp_idx=0)
            out["_src_const_hlt"] = const_hlt
            out["_src_mask_hlt"] = mask_safe
            return out

        outs = [self._decode_one(x, mask_safe, const_hlt, stage_scale, hyp_idx=h) for h in range(self.num_hypotheses)]
        out_m: Dict[str, torch.Tensor] = {}
        for k in outs[0].keys():
            out_m[k] = torch.stack([o[k] for o in outs], dim=1)
        out_m["_src_const_hlt"] = const_hlt
        out_m["_src_mask_hlt"] = mask_safe
        return out_m


@torch.no_grad()
def _build_selector_features_batch(
    reco_out: Dict[str, torch.Tensor],
    hlt_const: torch.Tensor,
    hlt_mask: torch.Tensor,
    labels: Optional[torch.Tensor],
    off_const: Optional[torch.Tensor],
    off_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cand = reco_out["cand_tokens"]
    w = reco_out["cand_weights"]
    if cand.ndim == 3:
        cand = cand.unsqueeze(1)
        w = w.unsqueeze(1)
    bsz, k, _t, _d = cand.shape

    eps = 1e-8
    hlt_n = hlt_mask.float().sum(dim=1)
    hlt_pt = _compute_jet_pt_torch(hlt_const, hlt_mask)

    hlt_px, hlt_py, hlt_pz, hlt_E = reco_base._weighted_fourvec_sums(hlt_const, hlt_mask.float())
    hlt_p = torch.sqrt(hlt_px.pow(2) + hlt_py.pow(2) + hlt_pz.pow(2) + eps)
    hlt_m = torch.sqrt(torch.clamp(hlt_E.pow(2) - hlt_p.pow(2), min=eps))

    feat_cols: List[torch.Tensor] = []
    set_cols: List[torch.Tensor] = []
    tag_cols: List[torch.Tensor] = []

    teacher = M29_STATE.teacher
    means = M29_STATE.feat_means
    stds = M29_STATE.feat_stds

    hlt_logit = torch.zeros((bsz,), device=hlt_const.device, dtype=hlt_const.dtype)
    if teacher is not None and means is not None and stds is not None and bool(M29_STATE.opts.selector_use_hlt_logit):
        hlt_feat = base.compute_features_torch(hlt_const, hlt_mask)
        hlt_feat_std = _standardize_features_torch(
            hlt_feat,
            hlt_mask,
            means.to(hlt_const.device),
            stds.to(hlt_const.device),
        )
        hlt_logit = teacher(hlt_feat_std, hlt_mask).squeeze(-1)
    hlt_prob = torch.sigmoid(hlt_logit)
    hlt_entropy = -(hlt_prob * torch.log(hlt_prob.clamp(min=1e-8)) + (1.0 - hlt_prob) * torch.log((1.0 - hlt_prob).clamp(min=1e-8)))

    for hk in range(k):
        p_const = cand[:, hk, :, :]
        p_w = w[:, hk, :].clamp(0.0, 1.0)
        p_mask = p_w > float(M29_STATE.opts.selector_weight_floor)
        none = ~p_mask.any(dim=1)
        if none.any():
            p_mask = p_mask.clone()
            p_mask[none, 0] = True

        p_feat = base.compute_features_torch(p_const, p_mask)

        if teacher is not None and means is not None and stds is not None:
            p_feat_std = _standardize_features_torch(
                p_feat,
                p_mask,
                means.to(p_feat.device),
                stds.to(p_feat.device),
            )
            p_logit = teacher(p_feat_std, p_mask).squeeze(-1)
        else:
            p_logit = torch.zeros((bsz,), device=p_const.device, dtype=p_const.dtype)

        p_prob = torch.sigmoid(p_logit)
        p_entropy = -(p_prob * torch.log(p_prob.clamp(min=1e-8)) + (1.0 - p_prob) * torch.log((1.0 - p_prob).clamp(min=1e-8)))

        p_n = p_mask.float().sum(dim=1)
        p_pt = _compute_jet_pt_torch(p_const, p_mask)
        pt_ratio_hlt = p_pt / (hlt_pt + eps)

        p_px, p_py, p_pz, p_E = reco_base._weighted_fourvec_sums(p_const, p_w)
        p_p = torch.sqrt(p_px.pow(2) + p_py.pow(2) + p_pz.pow(2) + eps)
        p_m = torch.sqrt(torch.clamp(p_E.pow(2) - p_p.pow(2), min=eps))
        p_m_ratio_hlt = p_m / (hlt_m + eps)

        p_to_hlt = _set_loss_chamfer_vec(
            p_const,
            p_w,
            hlt_const,
            hlt_mask,
            unmatched_penalty=0.0,
        )

        btot = reco_out.get("budget_total")
        badd = reco_out.get("budget_merge")
        baux = reco_out.get("budget_eff")
        if btot is not None and btot.ndim >= 2:
            btot = btot[:, hk]
        if badd is not None and badd.ndim >= 2:
            badd = badd[:, hk]
        if baux is not None and baux.ndim >= 2:
            baux = baux[:, hk]
        if btot is None:
            btot = p_n
        if badd is None:
            badd = (p_n - hlt_n).clamp(min=0.0)
        if baux is None:
            baux = torch.zeros_like(badd)

        feat_k = torch.stack(
            [
                p_logit,
                p_prob,
                p_entropy,
                hlt_logit,
                hlt_prob,
                hlt_entropy,
                p_logit - hlt_logit,
                (p_prob - hlt_prob).abs(),
                p_n / 100.0,
                hlt_n / 100.0,
                (p_n - hlt_n).abs() / 100.0,
                pt_ratio_hlt,
                (pt_ratio_hlt - 1.0).abs(),
                p_m_ratio_hlt,
                (p_m_ratio_hlt - 1.0).abs(),
                p_to_hlt,
                btot / 100.0,
                badd / 100.0,
                baux / 100.0,
            ],
            dim=-1,
        )
        feat_cols.append(feat_k)

        if off_const is not None and off_mask is not None:
            set_vec = _set_loss_chamfer_vec(
                p_const,
                p_w,
                off_const,
                off_mask,
                unmatched_penalty=float(base.BASE_CONFIG["loss"].get("unselected_penalty", 0.0)),
            )
        else:
            set_vec = torch.zeros((bsz,), device=p_const.device, dtype=p_const.dtype)
        set_cols.append(set_vec)

        if labels is not None and teacher is not None and means is not None and stds is not None:
            tag_vec = F.binary_cross_entropy_with_logits(p_logit, labels.float().view(-1), reduction="none")
        else:
            tag_vec = set_vec.detach()
        tag_cols.append(tag_vec)

    feat_mat = torch.stack(feat_cols, dim=1)
    set_mat = torch.stack(set_cols, dim=1)
    tag_mat = torch.stack(tag_cols, dim=1)
    return feat_mat, set_mat, tag_mat


def _score_for_winner(set_mat: torch.Tensor, tag_mat: torch.Tensor) -> torch.Tensor:
    mode = str(M29_STATE.opts.winner_mode).strip().lower()
    if mode == "reco":
        return set_mat
    if mode == "tag":
        return tag_mat
    return float(M29_STATE.opts.winner_alpha) * tag_mat + float(M29_STATE.opts.winner_beta) * set_mat


def _fit_selector_from_loaders(
    reconstructor: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> None:
    if int(M29_STATE.opts.num_hypotheses) <= 1:
        M29_STATE.selector = None
        M29_STATE.selector_ready = False
        return

    reconstructor.eval()

    @torch.no_grad()
    def collect(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feat_all: List[np.ndarray] = []
        tgt_all: List[np.ndarray] = []
        soft_all: List[np.ndarray] = []
        for batch in loader:
            feat_hlt_reco = batch["feat_hlt_reco"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            labels = batch["label"].to(device)

            reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            feat_bkf, set_mat, tag_mat = _build_selector_features_batch(
                reco_out=reco_out,
                hlt_const=const_hlt,
                hlt_mask=mask_hlt,
                labels=labels,
                off_const=const_off,
                off_mask=mask_off,
            )
            score = _score_for_winner(set_mat, tag_mat)
            win = torch.argmin(score, dim=1)
            soft = torch.softmax((-score) / float(max(M29_STATE.opts.selector_soft_temp, 1e-4)), dim=1)

            feat_all.append(feat_bkf.detach().cpu().numpy().astype(np.float32))
            tgt_all.append(win.detach().cpu().numpy().astype(np.int64))
            soft_all.append(soft.detach().cpu().numpy().astype(np.float32))

        if len(feat_all) == 0:
            return (
                np.zeros((0, int(M29_STATE.opts.num_hypotheses), 1), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, int(M29_STATE.opts.num_hypotheses)), dtype=np.float32),
            )
        return (
            np.concatenate(feat_all, axis=0),
            np.concatenate(tgt_all, axis=0),
            np.concatenate(soft_all, axis=0),
        )

    feat_tr, y_tr, p_tr = collect(train_loader)
    feat_va, y_va, _ = collect(val_loader)
    if feat_tr.shape[0] == 0 or feat_va.shape[0] == 0:
        print("m30 selector: empty train/val feature set; skipping selector fit")
        return

    selector = HypothesisSelector(
        feat_dim=int(feat_tr.shape[-1]),
        hidden=int(max(M29_STATE.opts.selector_hidden, 32)),
        heads=int(max(M29_STATE.opts.selector_heads, 1)),
        dropout=float(max(M29_STATE.opts.selector_dropout, 0.0)),
    ).to(device)
    opt = torch.optim.AdamW(selector.parameters(), lr=float(M29_STATE.opts.selector_lr), weight_decay=1e-4)

    xtr = torch.tensor(feat_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    ptr = torch.tensor(p_tr, dtype=torch.float32)
    xva = torch.tensor(feat_va, dtype=torch.float32)
    yva = torch.tensor(y_va, dtype=torch.long)

    use_soft_targets = bool(M29_STATE.opts.selector_use_soft_targets) and float(M29_STATE.opts.selector_kl_weight) > 0.0
    if use_soft_targets:
        tr_ds = TensorDataset(xtr, ytr, ptr)
    else:
        tr_ds = TensorDataset(xtr, ytr)
    tr_dl = DataLoader(tr_ds, batch_size=512, shuffle=True)
    va_dl = DataLoader(TensorDataset(xva, yva), batch_size=512, shuffle=False)

    best_state = None
    best_acc = -1.0
    no_imp = 0
    for ep in range(int(max(M29_STATE.opts.selector_epochs, 1))):
        selector.train()
        for batch in tr_dl:
            if use_soft_targets:
                xb, yb, pb = batch
                pb = pb.to(device)
            else:
                xb, yb = batch
                pb = None
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = selector(xb)
            loss_ce = F.cross_entropy(logits, yb) * float(max(M29_STATE.opts.selector_ce_weight, 0.0))
            loss = loss_ce
            if pb is not None:
                target_prob = pb / pb.sum(dim=1, keepdim=True).clamp(min=1e-8)
                loss_kl = F.kl_div(F.log_softmax(logits, dim=1), target_prob, reduction="batchmean")
                loss = loss + float(max(M29_STATE.opts.selector_kl_weight, 0.0)) * loss_kl
            if float(M29_STATE.opts.selector_rank_weight) > 0.0:
                loss_rank = selector_rank_loss(logits, yb, margin=float(M29_STATE.opts.selector_rank_margin))
                loss = loss + float(M29_STATE.opts.selector_rank_weight) * loss_rank
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
        acc = float(n_ok / max(n_tot, 1))
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= int(max(M29_STATE.opts.selector_patience, 1)):
            break

    if best_state is not None:
        selector.load_state_dict(best_state)
    selector.eval()
    M29_STATE.selector = selector
    M29_STATE.selector_ready = True
    print(
        f"m30 selector trained: val_acc={best_acc:.4f}, Ntr={feat_tr.shape[0]}, "
        f"Nva={feat_va.shape[0]}, feat_dim={feat_tr.shape[-1]}, soft_targets={use_soft_targets}"
    )


def _route_reco_out_for_view(reco_out: Dict[str, torch.Tensor], soft: bool) -> Dict[str, torch.Tensor]:
    cand = reco_out.get("cand_tokens")
    if cand is None or cand.ndim < 4:
        return reco_out

    k = int(cand.shape[1])
    if k <= 1:
        return _extract_hyp_out(reco_out, torch.zeros((cand.shape[0],), dtype=torch.long, device=cand.device))

    selector = M29_STATE.selector
    if selector is None:
        idx = torch.zeros((cand.shape[0],), dtype=torch.long, device=cand.device)
        out_h = _extract_hyp_out(reco_out, idx)
        out_h["winner_idx"] = idx
        return out_h

    hlt_const = reco_out.get("_src_const_hlt")
    hlt_mask = reco_out.get("_src_mask_hlt")
    if hlt_const is None or hlt_mask is None:
        idx = torch.zeros((cand.shape[0],), dtype=torch.long, device=cand.device)
        out_h = _extract_hyp_out(reco_out, idx)
        out_h["winner_idx"] = idx
        return out_h

    feat_bkf, _, _ = _build_selector_features_batch(
        reco_out=reco_out,
        hlt_const=hlt_const,
        hlt_mask=hlt_mask,
        labels=None,
        off_const=None,
        off_mask=None,
    )
    selector.eval()
    logits = selector(feat_bkf)
    probs = torch.softmax(logits / float(max(M29_STATE.opts.routing_temp, 1e-4)), dim=1)
    idx = torch.argmax(logits, dim=1)

    if bool(soft):
        out_m = _mix_hyp_out(reco_out, probs)
        out_m["winner_idx"] = idx
        out_m["router_probs"] = probs
        return out_m

    out_h = _extract_hyp_out(reco_out, idx)
    out_h["winner_idx"] = idx
    out_h["router_probs"] = probs
    return out_h


def compute_reconstruction_losses_weighted_set2set_m29(
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
    _ = budget_merge_true
    _ = budget_eff_true

    cand = out["cand_tokens"]
    if cand.ndim < 4:
        return _compute_single_losses(out, const_hlt, mask_hlt, const_off, mask_off, loss_cfg, sample_weight)

    bsz, k, _t, _d = cand.shape
    set_cols: List[torch.Tensor] = []
    for hk in range(k):
        out_k = _extract_hyp_out(out, torch.full((bsz,), hk, dtype=torch.long, device=cand.device))
        lk = _compute_single_losses(out_k, const_hlt, mask_hlt, const_off, mask_off, loss_cfg, sample_weight=None)
        set_cols.append(lk["_set_vec"])
    set_mat = torch.stack(set_cols, dim=1)
    winner_idx = torch.argmin(set_mat, dim=1)

    out_win = _extract_hyp_out(out, winner_idx)
    base_losses = _compute_single_losses(out_win, const_hlt, mask_hlt, const_off, mask_off, loss_cfg, sample_weight)

    sw = None if sample_weight is None else sample_weight.float().clamp(min=0.0)
    best_set_vec = set_mat.min(dim=1).values
    best_set = _weighted_batch_mean(best_set_vec, sw)

    div = torch.zeros((), device=cand.device, dtype=cand.dtype)
    n_pairs = 0
    w_all = out["cand_weights"].clamp(0.0, 1.0)
    for i in range(k):
        for j in range(i + 1, k):
            wi = w_all[:, i, :]
            wj = w_all[:, j, :]
            m = ((wi > 0.03) | (wj > 0.03)).float()
            d = (cand[:, i, :, :] - cand[:, j, :, :]).abs().sum(dim=-1)
            d = (d * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            div = div + torch.exp(-d).mean()
            n_pairs += 1
    if n_pairs > 0:
        div = div / float(n_pairs)

    total = (
        base_losses["total"]
        - float(loss_cfg.get("w_set", 1.0)) * base_losses["set"]
        + float(M29_STATE.opts.w_best_set) * best_set
        + float(M29_STATE.opts.w_diversity) * div
    )

    out_losses = dict(base_losses)
    out_losses["total"] = total
    out_losses["best_set"] = best_set
    out_losses["diversity"] = div
    out_losses["winner_idx"] = winner_idx
    out_losses["set_mat"] = set_mat
    return out_losses


def build_soft_corrected_view_set2set_m29(
    reco_out: Dict[str, torch.Tensor],
    weight_floor: float = 1e-4,
    scale_features_by_weight: bool = True,
    include_flags: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    use_soft = bool(torch.is_grad_enabled()) and bool(M29_STATE.opts.soft_routing_train)
    routed = _route_reco_out_for_view(reco_out, soft=use_soft)

    tok_tokens = routed.get("tok_tokens", None)
    tok_weights = routed.get("tok_weights", None)
    if tok_tokens is None or tok_weights is None:
        L = int(routed["action_prob"].shape[1])
        tok_tokens = routed["cand_tokens"][:, :L, :]
        tok_weights = routed["cand_weights"][:, :L]

    mask_b = tok_weights > float(weight_floor)
    none_valid = ~mask_b.any(dim=1)
    if none_valid.any():
        mask_b = mask_b.clone()
        mask_b[none_valid, 0] = True

    feat7 = base.compute_features_torch(tok_tokens, mask_b)
    if bool(scale_features_by_weight):
        feat7 = feat7 * tok_weights.unsqueeze(-1)

    parent_added = torch.zeros_like(tok_weights)
    if ("extra_to_base" in routed) and ("child_weight" in routed):
        assign = routed["extra_to_base"]
        extra_w = routed["child_weight"]
        if assign.numel() > 0 and extra_w.numel() > 0:
            parent_added = (assign * extra_w.unsqueeze(-1)).sum(dim=1).clamp(0.0, 1.0)

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


@torch.no_grad()
def reconstruct_dataset_m29(
    model: nn.Module,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    max_constits: int,
    device: torch.device,
    batch_size: int,
    weight_threshold: float = 0.03,
    use_budget_topk: bool = True,
):
    ds = reco_base.ReconstructInputDataset(feat_hlt, mask_hlt, const_hlt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    n = feat_hlt.shape[0]
    reco_const = np.zeros((n, max_constits, 4), dtype=np.float32)
    reco_mask = np.zeros((n, max_constits), dtype=bool)
    reco_merge_flag = np.zeros((n, max_constits), dtype=np.float32)
    reco_eff_flag = np.zeros((n, max_constits), dtype=np.float32)
    created_merge_count = np.zeros(n, dtype=np.int32)
    created_eff_count = np.zeros(n, dtype=np.int32)
    pred_budget_total = np.zeros(n, dtype=np.float32)
    pred_budget_merge = np.zeros(n, dtype=np.float32)
    pred_budget_eff = np.zeros(n, dtype=np.float32)

    offset = 0
    for batch in loader:
        x = batch["feat_hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        c = batch["const_hlt"].to(device)
        out = model(x, m, c, stage_scale=1.0)
        out = _route_reco_out_for_view(out, soft=False)

        cand = out["cand_tokens"].detach().cpu().numpy()
        w = out["cand_weights"].detach().cpu().numpy()
        merge_flags = out["cand_merge_flags"].detach().cpu().numpy()
        eff_flags = out["cand_eff_flags"].detach().cpu().numpy()
        budget_total = out["budget_total"].detach().cpu().numpy()
        budget_merge = out["budget_merge"].detach().cpu().numpy()
        budget_eff = out["budget_eff"].detach().cpu().numpy()
        n_tok = x.shape[1]
        n_child = out["child_weight"].shape[1]
        gen_start = int(n_tok + n_child)

        bsz = cand.shape[0]
        for i in range(bsz):
            order = np.argsort(-w[i])
            if use_budget_topk:
                k_budget = int(np.clip(np.rint(float(budget_total[i])), 1, max_constits))
                picked = [int(idx) for idx in order[:k_budget]]
                if len(picked) == 0:
                    picked = [int(order[0])]
            else:
                picked = []
                for idx in order:
                    if len(picked) >= max_constits:
                        break
                    if w[i, idx] < weight_threshold and len(picked) > 0:
                        break
                    picked.append(int(idx))
                if len(picked) == 0:
                    picked = [int(order[0])]

            nnn = min(len(picked), max_constits)
            sel = picked[:nnn]
            reco_const[offset + i, :nnn] = cand[i, sel]
            reco_mask[offset + i, :nnn] = True
            reco_merge_flag[offset + i, :nnn] = np.clip(merge_flags[i, sel], 0.0, 1.0)
            reco_eff_flag[offset + i, :nnn] = np.clip(eff_flags[i, sel], 0.0, 1.0)
            sel_arr = np.array(sel, dtype=np.int64)
            created_merge_count[offset + i] = int(np.sum((sel_arr >= n_tok) & (sel_arr < gen_start)))
            created_eff_count[offset + i] = int(np.sum(sel_arr >= gen_start))
            pred_budget_total[offset + i] = float(budget_total[i])
            pred_budget_merge[offset + i] = float(budget_merge[i])
            pred_budget_eff[offset + i] = float(budget_eff[i])

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


class SingleViewCorrectedClassifier(nn.Module):
    def __init__(self, input_dim_a: int, input_dim_b: int, **kwargs):
        super().__init__()
        _ = input_dim_a
        self.inner = base.ParticleTransformer(input_dim=int(input_dim_b), **kwargs)

    def forward(self, feat_a, mask_a, feat_b, mask_b):
        _ = feat_a
        _ = mask_a
        return self.inner(feat_b, mask_b)


_ORIG_TRAIN_SINGLE = base.train_single_view_classifier_auc
_ORIG_GET_STATS = base.get_stats
_ORIG_TRAIN_JOINT_DUAL = base.train_joint_dual


def train_single_view_capture_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> nn.Module:
    out = _ORIG_TRAIN_SINGLE(model, train_loader, val_loader, device, train_cfg, name)
    lname = str(name).strip().lower()
    if "teacher" in lname:
        out.eval()
        for p in out.parameters():
            p.requires_grad_(False)
        M29_STATE.teacher = out
        print("m29: captured Teacher model for selector scoring")
    if "hlt" in lname or "baseline" in lname:
        out.eval()
        for p in out.parameters():
            p.requires_grad_(False)
        M29_STATE.baseline = out
    return out


def get_stats_capture(*args, **kwargs):
    means, stds = _ORIG_GET_STATS(*args, **kwargs)
    M29_STATE.feat_means = torch.tensor(np.asarray(means), dtype=torch.float32)
    M29_STATE.feat_stds = torch.tensor(np.asarray(stds), dtype=torch.float32)
    return means, stds


def train_joint_dual_m29(
    reconstructor: nn.Module,
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
):
    stage = str(stage_name)

    if int(M29_STATE.opts.num_hypotheses) > 1:
        need_selector = False
        if bool(M29_STATE.opts.selector_train_before_stageb) and stage.startswith("StageB") and (not M29_STATE.selector_ready):
            need_selector = True
        if stage.startswith("StageC"):
            need_selector = True
        if need_selector:
            _fit_selector_from_loaders(reconstructor, train_loader, val_loader, device)

    mode = str(M29_STATE.opts.stagec_mode).strip().lower()
    if stage.startswith("StageC") and mode == "selector_only":
        print("m29 StageC mode=selector_only: selector refreshed; skipping Stage-C dual/reco parameter updates")
        va_pack = base.eval_joint_model_both_metrics(
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
        state_dual = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
        state_reco = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
        metrics = {
            "val_metric_source": "weighted" if has_weighted_val else "unweighted",
            "selection_metric": str(select_metric).lower(),
            "selected_val_fpr50": float(va_fpr50),
            "selected_val_auc": float(va_auc),
            "selected_val_fpr50_unweighted": float(va_fpr50_unw),
            "selected_val_auc_unweighted": float(va_auc_unw),
            "selected_val_fpr50_weighted": float(va_fpr50_w),
            "selected_val_auc_weighted": float(va_auc_w),
            "best_val_fpr50_seen": float(va_fpr50),
            "best_val_auc_seen": float(va_auc),
            "best_val_fpr50_seen_unweighted": float(va_fpr50_unw),
            "best_val_auc_seen_unweighted": float(va_auc_unw),
            "best_val_fpr50_seen_weighted": float(va_fpr50_w),
            "best_val_auc_seen_weighted": float(va_auc_w),
            "delta_enabled": False,
            "delta_lambda": float(lambda_delta_cls),
            "delta_tau": float(delta_tau),
            "delta_lambda_fp": float(delta_lambda_fp),
            "delta_hlt_threshold_prob": float(delta_hlt_threshold_prob),
            "delta_warmup_epochs": int(max(delta_warmup_epochs, 0)),
            "progressive_unfreeze": False,
            "unfreeze_phase1_epochs": int(unfreeze_phase1_epochs),
            "unfreeze_phase2_epochs": int(unfreeze_phase2_epochs),
            "unfreeze_last_n_encoder_layers": int(unfreeze_last_n_encoder_layers),
            "alternate_freeze": False,
            "alternate_reco_only_epochs": int(alternate_reco_only_epochs),
            "alternate_dual_only_epochs": int(alternate_dual_only_epochs),
            "lambda_param_anchor": float(lambda_param_anchor),
            "lambda_output_anchor": float(lambda_output_anchor),
            "anchor_decay": float(anchor_decay),
        }
        state_pack = {
            "selected": {"dual": state_dual, "reco": state_reco},
            "auc": {"dual": state_dual, "reco": state_reco},
            "fpr50": {"dual": state_dual, "reco": state_reco},
        }
        return reconstructor, dual_model, metrics, state_pack

    if stage.startswith("StageC") and mode == "selector_dual":
        freeze_reconstructor = True

    return _ORIG_TRAIN_JOINT_DUAL(
        reconstructor=reconstructor,
        dual_model=dual_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        stage_name=stage_name,
        freeze_reconstructor=freeze_reconstructor,
        epochs=epochs,
        patience=patience,
        lr_dual=lr_dual,
        lr_reco=lr_reco,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        lambda_reco=lambda_reco,
        lambda_rank=lambda_rank,
        lambda_cons=lambda_cons,
        corrected_weight_floor=corrected_weight_floor,
        corrected_use_flags=corrected_use_flags,
        min_epochs=min_epochs,
        select_metric=select_metric,
        apply_cls_weight=apply_cls_weight,
        apply_reco_weight=apply_reco_weight,
        val_weight_key=val_weight_key,
        use_weighted_val_selection=use_weighted_val_selection,
        lambda_delta_cls=lambda_delta_cls,
        delta_tau=delta_tau,
        delta_lambda_fp=delta_lambda_fp,
        delta_hlt_model=delta_hlt_model,
        delta_hlt_threshold_prob=delta_hlt_threshold_prob,
        delta_warmup_epochs=delta_warmup_epochs,
        progressive_unfreeze=progressive_unfreeze,
        unfreeze_phase1_epochs=unfreeze_phase1_epochs,
        unfreeze_phase2_epochs=unfreeze_phase2_epochs,
        unfreeze_last_n_encoder_layers=unfreeze_last_n_encoder_layers,
        alternate_freeze=alternate_freeze,
        alternate_reco_only_epochs=alternate_reco_only_epochs,
        alternate_dual_only_epochs=alternate_dual_only_epochs,
        lambda_param_anchor=lambda_param_anchor,
        lambda_output_anchor=lambda_output_anchor,
        anchor_decay=anchor_decay,
    )


def _identity_enforce(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return out


def _identity_wrap(model: nn.Module) -> nn.Module:
    return model


def _parse_m29_options() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--m29_num_hypotheses", type=int, default=6)
    p.add_argument("--m29_winner_mode", type=str, default="hybrid", choices=["reco", "tag", "hybrid"])
    p.add_argument("--m29_winner_alpha", type=float, default=1.0)
    p.add_argument("--m29_winner_beta", type=float, default=0.6)
    p.add_argument("--m29_loss_w_best_set", type=float, default=2.5)
    p.add_argument("--m29_loss_w_diversity", type=float, default=0.08)
    p.add_argument("--m29_selector_epochs", type=int, default=30)
    p.add_argument("--m29_selector_lr", type=float, default=2e-3)
    p.add_argument("--m29_selector_patience", type=int, default=8)
    p.add_argument("--m29_selector_rank_weight", type=float, default=0.2)
    p.add_argument("--m29_selector_rank_margin", type=float, default=0.25)
    p.add_argument("--m29_selector_hidden", type=int, default=96)
    p.add_argument("--m29_selector_heads", type=int, default=4)
    p.add_argument("--m29_selector_dropout", type=float, default=0.10)
    p.add_argument("--m29_selector_ce_weight", type=float, default=1.0)
    p.add_argument("--m29_selector_kl_weight", type=float, default=0.35)
    p.add_argument("--m29_selector_soft_temp", type=float, default=0.35)
    p.add_argument("--m29_disable_selector_soft_targets", action="store_true")
    p.add_argument("--m29_disable_selector_hlt_logit", action="store_true")
    p.add_argument("--m29_stagec_mode", type=str, default="all_three", choices=["all_three", "selector_dual", "selector_only"])
    p.add_argument("--m29_no_selector_train_before_stageb", action="store_true")
    p.add_argument("--m29_disable_soft_routing_train", action="store_true")
    p.add_argument("--m29_routing_temp", type=float, default=1.0)
    p.add_argument("--m29_single_corrected_tagger", action="store_true")
    p.add_argument("--m31_denoise_steps", type=int, default=3)
    p.add_argument("--m31_denoise_init_noise_std", type=float, default=0.03)
    p.add_argument("--m31_denoise_delta_scale", type=float, default=0.35)
    p.add_argument("--m31_denoise_step_embed_dim", type=int, default=16)

    opts, rest = p.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest

    M29_STATE.opts = M29Options(
        num_hypotheses=int(max(opts.m29_num_hypotheses, 1)),
        winner_mode=str(opts.m29_winner_mode).strip().lower(),
        winner_alpha=float(opts.m29_winner_alpha),
        winner_beta=float(opts.m29_winner_beta),
        w_best_set=float(opts.m29_loss_w_best_set),
        w_diversity=float(opts.m29_loss_w_diversity),
        selector_epochs=int(max(opts.m29_selector_epochs, 1)),
        selector_lr=float(opts.m29_selector_lr),
        selector_patience=int(max(opts.m29_selector_patience, 1)),
        selector_rank_weight=float(max(opts.m29_selector_rank_weight, 0.0)),
        selector_rank_margin=float(max(opts.m29_selector_rank_margin, 0.0)),
        selector_hidden=int(max(opts.m29_selector_hidden, 16)),
        selector_heads=int(max(opts.m29_selector_heads, 1)),
        selector_dropout=float(max(opts.m29_selector_dropout, 0.0)),
        selector_ce_weight=float(max(opts.m29_selector_ce_weight, 0.0)),
        selector_kl_weight=float(max(opts.m29_selector_kl_weight, 0.0)),
        selector_soft_temp=float(max(opts.m29_selector_soft_temp, 1e-4)),
        selector_use_soft_targets=not bool(opts.m29_disable_selector_soft_targets),
        selector_use_hlt_logit=not bool(opts.m29_disable_selector_hlt_logit),
        selector_train_before_stageb=not bool(opts.m29_no_selector_train_before_stageb),
        stagec_mode=str(opts.m29_stagec_mode).strip().lower(),
        soft_routing_train=not bool(opts.m29_disable_soft_routing_train),
        routing_temp=float(max(opts.m29_routing_temp, 1e-4)),
        single_corrected_tagger=bool(opts.m29_single_corrected_tagger),
        denoise_steps=int(max(opts.m31_denoise_steps, 0)),
        denoise_init_noise_std=float(max(opts.m31_denoise_init_noise_std, 0.0)),
        denoise_delta_scale=float(max(opts.m31_denoise_delta_scale, 0.0)),
        denoise_step_embed_dim=int(max(opts.m31_denoise_step_embed_dim, 4)),
    )


def _patch_base_module() -> None:
    base.OfflineReconstructor = OfflineReconstructorJetLatentSet2SetK  # type: ignore[assignment]
    base.compute_reconstruction_losses_weighted = compute_reconstruction_losses_weighted_set2set_m29  # type: ignore[assignment]
    base.build_soft_corrected_view = build_soft_corrected_view_set2set_m29  # type: ignore[assignment]
    base.enforce_unmerge_only_output = _identity_enforce  # type: ignore[assignment]
    base.wrap_reconstructor_unmerge_only = _identity_wrap  # type: ignore[assignment]
    base.reconstruct_dataset = reconstruct_dataset_m29  # type: ignore[assignment]

    base.train_single_view_classifier_auc = train_single_view_capture_teacher  # type: ignore[assignment]
    base.get_stats = get_stats_capture  # type: ignore[assignment]
    base.train_joint_dual = train_joint_dual_m29  # type: ignore[assignment]

    if bool(M29_STATE.opts.single_corrected_tagger):
        base.DualViewCrossAttnClassifier = SingleViewCorrectedClassifier  # type: ignore[assignment]


def main() -> None:
    _parse_m29_options()
    print(
        "m31 config: "
        f"K={M29_STATE.opts.num_hypotheses}, "
        f"winner_mode={M29_STATE.opts.winner_mode}, "
        f"stageC_mode={M29_STATE.opts.stagec_mode}, "
        f"single_corrected_tagger={M29_STATE.opts.single_corrected_tagger}, "
        f"soft_routing_train={M29_STATE.opts.soft_routing_train}, "
        f"selector_soft_targets={M29_STATE.opts.selector_use_soft_targets}, "
        f"selector_hidden={M29_STATE.opts.selector_hidden}, "
        f"denoise_steps={M29_STATE.opts.denoise_steps}, "
        f"denoise_init_noise_std={M29_STATE.opts.denoise_init_noise_std:.4f}, "
        f"denoise_delta_scale={M29_STATE.opts.denoise_delta_scale:.4f}"
    )
    _patch_base_module()
    base.main()


if __name__ == "__main__":
    main()

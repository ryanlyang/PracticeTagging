#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m33: Deterministic-feasibility candidate search + DualView top tagging.

Pipeline:
1) Train teacher (offline) and baseline (HLT) classifiers.
2) Train class-conditional offline latent autoencoder (realism manifold).
3) Train differentiable Offline->HLT surrogate (D_soft).
4) Train HLT->latent class-conditional proposer (weakly feasibility-regularized).
5) Deterministic chunked search with D_hard acceptance to build candidate pools.
6) Train realism selector (real offline vs generated candidate).
7) Train final DualView heads (NoGate + Gated) using HLT and selected candidates.
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
from tqdm import tqdm

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_no_gt_local30kv2 as reco_base
from unmerge_correct_hlt import (
    ParticleTransformer,
    compute_features,
    get_stats,
    standardize,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fpr_at_target_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float = 0.5) -> float:
    if len(fpr) == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(tpr - target_tpr)))
    return float(fpr[idx])


def _weighted_mean(vec: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    if weight is None:
        return vec.mean()
    w = weight.float().clamp(min=0.0)
    return (vec * w).sum() / w.sum().clamp(min=1e-6)


def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    m = mask.clone()
    empty = ~m.any(dim=1)
    if empty.any():
        m[empty, 0] = True
    return m


def _const_to_token5(const: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    pt = const[..., 0].clamp(min=eps)
    eta = const[..., 1].clamp(min=-5.0, max=5.0)
    phi = const[..., 2]
    e = const[..., 3].clamp(min=eps)
    return torch.stack([torch.log(pt), eta, torch.sin(phi), torch.cos(phi), torch.log(e)], dim=-1)


def _decode_raw_to_const(raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # raw [..., 6]: logpt, eta_raw, sinphi, cosphi, logE, exist_logit
    logpt = torch.clamp(raw[..., 0], min=-9.0, max=9.0)
    eta = 5.0 * torch.tanh(raw[..., 1])
    sinphi = raw[..., 2]
    cosphi = raw[..., 3]
    loge = torch.clamp(raw[..., 4], min=-9.0, max=11.0)
    exist_logit = raw[..., 5]

    pt = torch.exp(logpt)
    phi = torch.atan2(sinphi, cosphi)
    e = torch.exp(loge)
    min_e = pt * torch.cosh(eta)
    e = torch.maximum(e, min_e)

    const = torch.stack([pt, eta, phi, e], dim=-1)
    return const, exist_logit


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


def _count_loss_vec(pred_w: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    pred_count = pred_w.sum(dim=1)
    true_count = tgt_mask.float().sum(dim=1)
    return F.smooth_l1_loss(pred_count, true_count, reduction="none")


def _gaussian_nll_vec(z: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Returns per-sample NLL up to additive constant.
    inv_var = torch.exp(-logvar)
    return 0.5 * (((z - mean) ** 2) * inv_var + logvar).sum(dim=-1)


def _kl_diag_gaussians(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        (logvar_p - logvar_q)
        + (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
    )
    return kl.sum(dim=-1)


def _pairwise_diversity_penalty(z_k: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    # z_k: [B, K, D], lower is better when proposals are diverse (penalty near 0).
    b, k, _ = z_k.shape
    if k <= 1:
        return torch.zeros((b,), device=z_k.device, dtype=z_k.dtype)
    dist = torch.cdist(z_k, z_k, p=2)
    eye = torch.eye(k, device=z_k.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(eye, 1e9)
    sim = torch.exp(-(dist ** 2) / max(float(tau), 1e-6))
    sim = sim.masked_fill(eye, 0.0)
    return sim.sum(dim=(1, 2)) / float(k * (k - 1))


def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return int(z & 0xFFFFFFFFFFFFFFFF)


def _deterministic_jet_seed(base_seed: int, jet_key: int, stream_id: int = 0) -> int:
    x = (int(base_seed) & 0xFFFFFFFFFFFFFFFF) ^ ((int(jet_key) + 0x9E3779B97F4A7C15 * int(stream_id + 1)) & 0xFFFFFFFFFFFFFFFF)
    z = _splitmix64(x)
    # np RandomState expects 32-bit seed.
    return int(z & 0x7FFFFFFF)


def _apply_hlt_effects_deterministic_keyed(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: Dict,
    jet_keys: np.ndarray,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Deterministic Offline->HLT mapping keyed per jet id.
    Same (offline, jet_key, base_seed) -> same HLT, independent of batch order.
    """
    n = int(const.shape[0])
    if n == 0:
        return (
            np.zeros_like(const, dtype=np.float32),
            np.zeros_like(mask, dtype=bool),
            {"n_jets": 0, "n_initial": 0, "n_final": 0, "avg_offline_per_jet": 0.0, "avg_hlt_per_jet": 0.0},
        )

    out_const = np.zeros_like(const, dtype=np.float32)
    out_mask = np.zeros_like(mask, dtype=bool)
    merged = 0
    eff_lost = 0
    pre_thr = 0
    post_thr = 0
    reassigned = 0

    for i in range(n):
        s = _deterministic_jet_seed(base_seed=int(base_seed), jet_key=int(jet_keys[i]), stream_id=0)
        c_i = const[i : i + 1]
        m_i = mask[i : i + 1]
        h_i, hm_i, st_i, _ = reco_base.apply_hlt_effects_realistic_nomap(
            c_i,
            m_i,
            cfg,
            seed=int(s),
        )
        out_const[i] = h_i[0]
        out_mask[i] = hm_i[0]
        merged += int(st_i.get("n_merged_pairs", 0))
        eff_lost += int(st_i.get("n_lost_eff", 0))
        pre_thr += int(st_i.get("n_lost_threshold_pre", 0))
        post_thr += int(st_i.get("n_lost_threshold_post", 0))
        reassigned += int(st_i.get("n_reassigned", 0))

    stats = {
        "n_jets": int(n),
        "n_initial": int(mask.sum()),
        "n_final": int(out_mask.sum()),
        "n_merged_pairs": int(merged),
        "n_lost_eff": int(eff_lost),
        "n_lost_threshold_pre": int(pre_thr),
        "n_lost_threshold_post": int(post_thr),
        "n_reassigned": int(reassigned),
        "avg_offline_per_jet": float(mask.sum(axis=1).mean()),
        "avg_hlt_per_jet": float(out_mask.sum(axis=1).mean()),
    }
    return out_const.astype(np.float32), out_mask.astype(bool), stats


def _jet_mass_vec(const: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.float()
    pt = const[..., 0] * w
    eta = const[..., 1]
    phi = const[..., 2]
    e = const[..., 3] * w
    px = (pt * torch.cos(phi)).sum(dim=1)
    py = (pt * torch.sin(phi)).sum(dim=1)
    pz = (pt * torch.sinh(eta)).sum(dim=1)
    et = e.sum(dim=1)
    m2 = et * et - px * px - py * py - pz * pz
    return torch.sqrt(torch.clamp(m2, min=0.0))


def _residual_fast_vec(
    pred_const: torch.Tensor,
    pred_mask: torch.Tensor,
    tgt_const: torch.Tensor,
    tgt_mask: torch.Tensor,
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    unmatched_penalty: float,
) -> Dict[str, torch.Tensor]:
    pred_w = pred_mask.float()
    r_set = _set_loss_chamfer_vec(
        pred_const=pred_const,
        pred_w=pred_w,
        tgt_const=tgt_const,
        tgt_mask=tgt_mask,
        unmatched_penalty=float(unmatched_penalty),
    )
    pred_count = pred_mask.float().sum(dim=1)
    tgt_count = tgt_mask.float().sum(dim=1)
    r_count = torch.abs(pred_count - tgt_count) / (tgt_count + 1.0)

    pred_pt = (pred_const[..., 0] * pred_mask.float()).sum(dim=1)
    tgt_pt = (tgt_const[..., 0] * tgt_mask.float()).sum(dim=1)
    r_pt = torch.abs(pred_pt - tgt_pt) / (tgt_pt + 1e-6)

    pred_mass = _jet_mass_vec(pred_const, pred_mask)
    tgt_mass = _jet_mass_vec(tgt_const, tgt_mask)
    r_mass = torch.abs(pred_mass - tgt_mass) / (tgt_mass + 1e-6)

    total = (
        float(w_chamfer) * r_set
        + float(w_count) * r_count
        + float(w_pt) * r_pt
        + float(w_mass) * r_mass
    )
    return {
        "total": total,
        "set": r_set,
        "count": r_count,
        "pt": r_pt,
        "mass": r_mass,
    }


class SetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=max(1, int(num_heads // 2)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, input_dim], mask: [B, L]
        b, l, _ = x.shape
        mask_safe = _safe_mask(mask)
        h = self.input_proj(x.view(-1, self.input_dim)).view(b, l, self.embed_dim)
        h = self.encoder(h, src_key_padding_mask=~mask_safe)
        q = self.pool_query.expand(b, -1, -1)
        pooled, _ = self.pool_attn(q, h, h, key_padding_mask=~mask_safe, need_weights=False)
        return self.norm(pooled.squeeze(1))


class SetDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        slots: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        num_classes: int = 2,
        use_class_cond: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.slots = int(slots)
        self.embed_dim = int(embed_dim)
        self.use_class_cond = bool(use_class_cond)

        self.query = nn.Parameter(torch.randn(1, self.slots, self.embed_dim) * 0.02)
        self.lat_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )
        self.class_emb = nn.Embedding(int(num_classes), self.embed_dim) if self.use_class_cond else None
        self.mem_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(num_layers))

        self.out = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.embed_dim, 6),
        )

    def forward(self, z: torch.Tensor, class_idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [B, D]
        b = z.shape[0]
        q = self.query.expand(b, -1, -1)
        cond = self.lat_proj(z).unsqueeze(1)
        if self.class_emb is not None and class_idx is not None:
            cond = cond + self.class_emb(class_idx.long()).unsqueeze(1)
        q = q + cond

        mem = self.mem_proj(z).unsqueeze(1)
        h = self.decoder(q, mem)
        raw = self.out(h)
        return _decode_raw_to_const(raw)


class OfflineLatentAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        slots: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.lat_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim),
        )
        self.decoder = SetDecoder(
            latent_dim=latent_dim,
            slots=slots,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=max(2, num_layers // 2),
            ff_dim=ff_dim,
            dropout=dropout,
            num_classes=2,
            use_class_cond=True,
        )

    def encode(self, const: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = _const_to_token5(const)
        h = self.encoder(x, mask)
        return self.lat_head(h)

    def decode(self, z: torch.Tensor, cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(z, class_idx=cls)


class OfflineToHLTDegrader(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        slots: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.lat_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim),
        )
        self.decoder = SetDecoder(
            latent_dim=latent_dim,
            slots=slots,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=max(2, num_layers // 2),
            ff_dim=ff_dim,
            dropout=dropout,
            num_classes=2,
            use_class_cond=False,
        )

    def forward(self, const_off: torch.Tensor, mask_off: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _const_to_token5(const_off)
        h = self.encoder(x, mask_off)
        z = self.lat_head(h)
        return self.decoder(z, class_idx=None)


class HLTLatentProposer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        num_classes: int = 2,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hlt_encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.class_emb = nn.Embedding(int(num_classes), embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(embed_dim, latent_dim)
        self.logvar_head = nn.Linear(embed_dim, latent_dim)

    def dist(self, const_hlt: torch.Tensor, mask_hlt: torch.Tensor, cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _const_to_token5(const_hlt)
        h = self.hlt_encoder(x, mask_hlt)
        c = self.class_emb(cls.long())
        u = self.mlp(torch.cat([h, c], dim=-1))
        mu = self.mu_head(u)
        logvar = torch.clamp(self.logvar_head(u), min=-7.0, max=5.0)
        return mu, logvar


class FinalFeasibilityClassifier(nn.Module):
    def __init__(
        self,
        feas_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.feas_dim = int(feas_dim)
        self.embed_dim = int(embed_dim)
        self.hlt_encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        # Treat feasibility scores as a special signal pathway with gating.
        self.feas_special = nn.Sequential(
            nn.Linear(self.feas_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feas_gate = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + self.feas_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, const_hlt: torch.Tensor, mask_hlt: torch.Tensor, feas_feat: torch.Tensor) -> torch.Tensor:
        h = self.hlt_encoder(_const_to_token5(const_hlt), mask_hlt)
        s = self.feas_special(feas_feat)
        g = self.feas_gate(s)
        h_mod = h * (1.0 + g)
        x = torch.cat([h_mod, s, feas_feat], dim=-1)
        return self.head(x).squeeze(-1)


class CandidateRealismSelector(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        class_dim: int = 16,
    ):
        super().__init__()
        self.hlt_encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.cand_encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.class_emb = nn.Embedding(2, class_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 3 + class_dim + 1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        const_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_const: torch.Tensor,
        cand_mask: torch.Tensor,
        cand_class: torch.Tensor,
        cand_resid: torch.Tensor,
    ) -> torch.Tensor:
        h = self.hlt_encoder(_const_to_token5(const_hlt), mask_hlt)
        c = self.cand_encoder(_const_to_token5(cand_const), cand_mask)
        ce = self.class_emb(cand_class.long())
        rr = cand_resid.view(-1, 1)
        x = torch.cat([h, c, torch.abs(h - c), ce, rr], dim=-1)
        return self.head(x).squeeze(-1)


class DualViewNoGateClassifier(nn.Module):
    def __init__(
        self,
        cand_feat_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hlt_encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.cand_encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=max(2, num_layers // 2),
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(cand_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        const_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_top_const: torch.Tensor,
        cand_top_mask: torch.Tensor,
        cand_bg_const: torch.Tensor,
        cand_bg_mask: torch.Tensor,
        cand_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.hlt_encoder(_const_to_token5(const_hlt), mask_hlt)
        t = self.cand_encoder(_const_to_token5(cand_top_const), cand_top_mask)
        b = self.cand_encoder(_const_to_token5(cand_bg_const), cand_bg_mask)
        f = self.feat_proj(cand_feat)
        x = torch.cat([h, t, b, f], dim=-1)
        return self.head(x).squeeze(-1)


class DualViewGatedClassifier(nn.Module):
    def __init__(
        self,
        cand_feat_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hlt_encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.cand_encoder = SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=max(2, num_layers // 2),
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(cand_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hlt_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.cand_head = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        const_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_top_const: torch.Tensor,
        cand_top_mask: torch.Tensor,
        cand_bg_const: torch.Tensor,
        cand_bg_mask: torch.Tensor,
        cand_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.hlt_encoder(_const_to_token5(const_hlt), mask_hlt)
        t = self.cand_encoder(_const_to_token5(cand_top_const), cand_top_mask)
        b = self.cand_encoder(_const_to_token5(cand_bg_const), cand_bg_mask)
        f = self.feat_proj(cand_feat)

        logit_h = self.hlt_head(h).squeeze(-1)
        cand_x = torch.cat([t, b, f], dim=-1)
        logit_c = self.cand_head(cand_x).squeeze(-1)
        g = self.gate(torch.cat([h, t, b, f], dim=-1)).squeeze(-1)
        return (1.0 - g) * logit_h + g * logit_c


class OfflineStageDataset(Dataset):
    def __init__(
        self,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.labels = torch.tensor(labels.astype(np.int64), dtype=torch.long)
        n = int(self.labels.shape[0])
        if sample_weight is None:
            sw = np.ones((n,), dtype=np.float32)
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.shape[0] != n:
                raise ValueError(f"sample_weight mismatch: {sw.shape[0]} vs {n}")
        self.sample_weight = torch.tensor(sw, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "label": self.labels[i],
            "sample_weight": self.sample_weight[i],
        }


class PairStageDataset(Dataset):
    def __init__(
        self,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        const_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.labels = torch.tensor(labels.astype(np.int64), dtype=torch.long)
        n = int(self.labels.shape[0])
        if sample_weight is None:
            sw = np.ones((n,), dtype=np.float32)
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.shape[0] != n:
                raise ValueError(f"sample_weight mismatch: {sw.shape[0]} vs {n}")
        self.sample_weight = torch.tensor(sw, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "const_hlt": self.const_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "label": self.labels[i],
            "sample_weight": self.sample_weight[i],
        }


class FinalStageDataset(Dataset):
    def __init__(
        self,
        const_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        feas_feat: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.feas_feat = torch.tensor(feas_feat, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        n = int(self.labels.shape[0])
        if sample_weight is None:
            sw = np.ones((n,), dtype=np.float32)
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.shape[0] != n:
                raise ValueError(f"sample_weight mismatch: {sw.shape[0]} vs {n}")
        self.sample_weight = torch.tensor(sw, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "const_hlt": self.const_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "feas_feat": self.feas_feat[i],
            "label": self.labels[i],
            "sample_weight": self.sample_weight[i],
        }


class SelectorDataset(Dataset):
    def __init__(
        self,
        const_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        cand_const: np.ndarray,
        cand_mask: np.ndarray,
        cand_class: np.ndarray,
        cand_resid: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.cand_const = torch.tensor(cand_const, dtype=torch.float32)
        self.cand_mask = torch.tensor(cand_mask, dtype=torch.bool)
        self.cand_class = torch.tensor(cand_class, dtype=torch.long)
        self.cand_resid = torch.tensor(cand_resid, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        n = int(self.labels.shape[0])
        if sample_weight is None:
            sw = np.ones((n,), dtype=np.float32)
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.shape[0] != n:
                raise ValueError(f"sample_weight mismatch: {sw.shape[0]} vs {n}")
        self.sample_weight = torch.tensor(sw, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "const_hlt": self.const_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "cand_const": self.cand_const[i],
            "cand_mask": self.cand_mask[i],
            "cand_class": self.cand_class[i],
            "cand_resid": self.cand_resid[i],
            "label": self.labels[i],
            "sample_weight": self.sample_weight[i],
        }


class DualViewCandidateDataset(Dataset):
    def __init__(
        self,
        const_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        cand_top_const: np.ndarray,
        cand_top_mask: np.ndarray,
        cand_bg_const: np.ndarray,
        cand_bg_mask: np.ndarray,
        cand_feat: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.cand_top_const = torch.tensor(cand_top_const, dtype=torch.float32)
        self.cand_top_mask = torch.tensor(cand_top_mask, dtype=torch.bool)
        self.cand_bg_const = torch.tensor(cand_bg_const, dtype=torch.float32)
        self.cand_bg_mask = torch.tensor(cand_bg_mask, dtype=torch.bool)
        self.cand_feat = torch.tensor(cand_feat, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        n = int(self.labels.shape[0])
        if sample_weight is None:
            sw = np.ones((n,), dtype=np.float32)
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.shape[0] != n:
                raise ValueError(f"sample_weight mismatch: {sw.shape[0]} vs {n}")
        self.sample_weight = torch.tensor(sw, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "const_hlt": self.const_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "cand_top_const": self.cand_top_const[i],
            "cand_top_mask": self.cand_top_mask[i],
            "cand_bg_const": self.cand_bg_const[i],
            "cand_bg_mask": self.cand_bg_mask[i],
            "cand_feat": self.cand_feat[i],
            "label": self.labels[i],
            "sample_weight": self.sample_weight[i],
        }


@dataclass
class PriorStats:
    mean: torch.Tensor  # [2, D]
    logvar: torch.Tensor  # [2, D]


def _predict_probs(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    p, y = base.predict_single_view_scores(
        model=model,
        feat=feat,
        mask=mask,
        labels=labels,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device=device,
    )
    return p.astype(np.float32), y.astype(np.float32)


def _build_teacher_conf_weights(
    teacher_probs: np.ndarray,
    labels: np.ndarray,
    weight_floor: float,
    hard_correct_gate: bool,
    normalize_mean_one: bool,
) -> np.ndarray:
    pred = (teacher_probs >= 0.5).astype(np.int64)
    y = labels.astype(np.int64)
    correct = (pred == y).astype(np.float32)
    conf = (2.0 * np.abs(teacher_probs - 0.5)).astype(np.float32)

    if hard_correct_gate:
        trust = correct * conf
    else:
        trust = conf
    w = float(weight_floor) + (1.0 - float(weight_floor)) * trust
    w = w.astype(np.float32)
    if normalize_mean_one:
        m = float(np.mean(w))
        if m > 1e-8:
            w = w / m
    return w


def _train_offline_ae(
    model: OfflineLatentAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    loss_w_count: float,
    loss_w_lat_reg: float,
    unmatched_penalty: float,
) -> Tuple[OfflineLatentAE, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    no_imp = 0

    for ep in range(int(epochs)):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        tr_set = 0.0
        tr_cnt = 0.0
        for batch in train_loader:
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            cls = batch["label"].to(device)
            sw = batch["sample_weight"].to(device)

            z = model.encode(const_off, mask_off)
            pred_const, pred_exist_logit = model.decode(z, cls)
            pred_w = torch.sigmoid(pred_exist_logit)

            loss_set_vec = _set_loss_chamfer_vec(
                pred_const,
                pred_w,
                const_off,
                mask_off,
                unmatched_penalty=float(unmatched_penalty),
            )
            loss_cnt_vec = _count_loss_vec(pred_w, mask_off)
            loss_lat_vec = z.pow(2).mean(dim=1)
            total_vec = (
                loss_set_vec
                + float(loss_w_count) * loss_cnt_vec
                + float(loss_w_lat_reg) * loss_lat_vec
            )
            loss = _weighted_mean(total_vec, sw)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bs = int(const_off.shape[0])
            tr_loss += float(loss.detach().item()) * bs
            tr_set += float(_weighted_mean(loss_set_vec.detach(), sw).item()) * bs
            tr_cnt += float(_weighted_mean(loss_cnt_vec.detach(), sw).item()) * bs
            tr_n += bs

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for batch in val_loader:
                const_off = batch["const_off"].to(device)
                mask_off = batch["mask_off"].to(device)
                cls = batch["label"].to(device)
                sw = batch["sample_weight"].to(device)

                z = model.encode(const_off, mask_off)
                pred_const, pred_exist_logit = model.decode(z, cls)
                pred_w = torch.sigmoid(pred_exist_logit)

                loss_set_vec = _set_loss_chamfer_vec(
                    pred_const,
                    pred_w,
                    const_off,
                    mask_off,
                    unmatched_penalty=float(unmatched_penalty),
                )
                loss_cnt_vec = _count_loss_vec(pred_w, mask_off)
                loss_lat_vec = z.pow(2).mean(dim=1)
                total_vec = (
                    loss_set_vec
                    + float(loss_w_count) * loss_cnt_vec
                    + float(loss_w_lat_reg) * loss_lat_vec
                )
                loss = _weighted_mean(total_vec, sw)
                bs = int(const_off.shape[0])
                va_loss += float(loss.item()) * bs
                va_n += bs

        tr = tr_loss / max(1, tr_n)
        trs = tr_set / max(1, tr_n)
        trc = tr_cnt / max(1, tr_n)
        va = va_loss / max(1, va_n)

        if va < best_val:
            best_val = float(va)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"OfflinePrior ep {ep+1:03d}: train={tr:.5f} set={trs:.5f} cnt={trc:.5f} "
                f"val={va:.5f} best={best_val:.5f}@{best_epoch}"
            )

        if no_imp >= int(patience):
            print(f"OfflinePrior early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
    }


def _compute_prior_stats(
    model: OfflineLatentAE,
    loader: DataLoader,
    device: torch.device,
    latent_dim: int,
) -> PriorStats:
    model.eval()
    z_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            z = model.encode(batch["const_off"].to(device), batch["mask_off"].to(device))
            y = batch["label"].to(device)
            z_all.append(z.detach().cpu())
            y_all.append(y.detach().cpu())

    z_cat = torch.cat(z_all, dim=0)
    y_cat = torch.cat(y_all, dim=0)
    means = []
    logvars = []
    for c in [0, 1]:
        m = (y_cat == c)
        if int(m.sum().item()) < 2:
            mu = torch.zeros((latent_dim,), dtype=torch.float32)
            lv = torch.zeros((latent_dim,), dtype=torch.float32)
        else:
            zc = z_cat[m]
            mu = zc.mean(dim=0)
            var = zc.var(dim=0, unbiased=False).clamp(min=1e-4)
            lv = torch.log(var)
        means.append(mu)
        logvars.append(lv)
    mean = torch.stack(means, dim=0).to(device)
    logvar = torch.stack(logvars, dim=0).to(device)
    return PriorStats(mean=mean, logvar=logvar)


def _train_degrader(
    model: OfflineToHLTDegrader,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    loss_w_count: float,
    unmatched_penalty: float,
) -> Tuple[OfflineToHLTDegrader, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    no_imp = 0

    for ep in range(int(epochs)):
        model.train()
        tr_loss = 0.0
        tr_set = 0.0
        tr_cnt = 0.0
        tr_n = 0
        for batch in train_loader:
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)

            pred_hlt, pred_exist_logit = model(const_off, mask_off)
            pred_w = torch.sigmoid(pred_exist_logit)
            loss_set_vec = _set_loss_chamfer_vec(
                pred_hlt,
                pred_w,
                const_hlt,
                mask_hlt,
                unmatched_penalty=float(unmatched_penalty),
            )
            loss_cnt_vec = _count_loss_vec(pred_w, mask_hlt)
            total_vec = loss_set_vec + float(loss_w_count) * loss_cnt_vec
            loss = total_vec.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bs = int(const_off.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_set += float(loss_set_vec.mean().item()) * bs
            tr_cnt += float(loss_cnt_vec.mean().item()) * bs
            tr_n += bs

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for batch in val_loader:
                const_off = batch["const_off"].to(device)
                mask_off = batch["mask_off"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)

                pred_hlt, pred_exist_logit = model(const_off, mask_off)
                pred_w = torch.sigmoid(pred_exist_logit)
                loss_set_vec = _set_loss_chamfer_vec(
                    pred_hlt,
                    pred_w,
                    const_hlt,
                    mask_hlt,
                    unmatched_penalty=float(unmatched_penalty),
                )
                loss_cnt_vec = _count_loss_vec(pred_w, mask_hlt)
                total_vec = loss_set_vec + float(loss_w_count) * loss_cnt_vec
                loss = total_vec.mean()
                bs = int(const_off.shape[0])
                va_loss += float(loss.item()) * bs
                va_n += bs

        tr = tr_loss / max(1, tr_n)
        trs = tr_set / max(1, tr_n)
        trc = tr_cnt / max(1, tr_n)
        va = va_loss / max(1, va_n)

        if va < best_val:
            best_val = float(va)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"Degrader ep {ep+1:03d}: train={tr:.5f} set={trs:.5f} cnt={trc:.5f} "
                f"val={va:.5f} best={best_val:.5f}@{best_epoch}"
            )

        if no_imp >= int(patience):
            print(f"Degrader early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
    }


def _decode_many(
    ae: OfflineLatentAE,
    z_bkd: torch.Tensor,
    cls_bk: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # z_bkd: [B, K, D], cls_bk: [B, K]
    b, k, d = z_bkd.shape
    z_flat = z_bkd.reshape(b * k, d)
    c_flat = cls_bk.reshape(b * k)
    const_flat, exist_logit_flat = ae.decode(z_flat, c_flat)
    w_flat = torch.sigmoid(exist_logit_flat)
    return (
        const_flat.reshape(b, k, const_flat.shape[1], 4),
        exist_logit_flat.reshape(b, k, exist_logit_flat.shape[1]),
        w_flat.reshape(b, k, w_flat.shape[1]),
    )


def _cheap_prescore_matrix(
    z_bkd: torch.Tensor,
    mu_bd: torch.Tensor,
    logvar_bd: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    w_prior: float = 0.70,
    w_q: float = 0.30,
) -> torch.Tensor:
    # Cheap latent prescore (lower is better): blend class-prior plausibility and q-likelihood.
    b, k, d = z_bkd.shape
    zf = z_bkd.reshape(b * k, d)
    muf = mu_bd.unsqueeze(1).expand(-1, k, -1).reshape(b * k, d)
    lvf = logvar_bd.unsqueeze(1).expand(-1, k, -1).reshape(b * k, d)
    if prior_mu.ndim == 1:
        pm = prior_mu.view(1, -1).expand(b * k, -1)
        plv = prior_logvar.view(1, -1).expand(b * k, -1)
    else:
        pm = prior_mu.unsqueeze(1).expand(-1, k, -1).reshape(b * k, d)
        plv = prior_logvar.unsqueeze(1).expand(-1, k, -1).reshape(b * k, d)
    pnll = _gaussian_nll_vec(zf, pm, plv).reshape(b, k)
    qnll = _gaussian_nll_vec(zf, muf, lvf).reshape(b, k)
    return float(w_prior) * pnll + float(w_q) * qnll


def _shortlist_topk_by_score(
    z_bkd: torch.Tensor,
    score_bk: torch.Tensor,
    top_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = int(max(1, min(int(top_m), int(z_bkd.shape[1]))))
    idx = torch.topk(score_bk, k=m, dim=1, largest=False).indices
    z_top = z_bkd.gather(1, idx.unsqueeze(-1).expand(-1, -1, z_bkd.shape[2]))
    score_top = score_bk.gather(1, idx)
    return z_top, score_top, idx


def _decode_degrade_hlt_loss_matrix(
    ae: OfflineLatentAE,
    degrader: OfflineToHLTDegrader,
    z_bmd: torch.Tensor,
    cls_bm: torch.Tensor,
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    pred_exist_threshold: float,
    unmatched_penalty: float,
    detach_threshold_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, m, _ = z_bmd.shape
    off_const_m, _off_exist_logit_m, off_w_m = _decode_many(ae, z_bmd, cls_bm)

    off_const_flat = off_const_m.reshape(b * m, off_const_m.shape[2], 4)
    off_w_flat = off_w_m.reshape(b * m, off_w_m.shape[2])
    off_mask_flat = (off_w_flat.detach() if detach_threshold_mask else off_w_flat) > float(pred_exist_threshold)

    # Avoid in-place edits on graph-view tensors; build the degraded input out-of-place.
    off_for_deg = torch.stack(
        [
            off_const_flat[..., 0] * off_w_flat,
            off_const_flat[..., 1],
            off_const_flat[..., 2],
            off_const_flat[..., 3] * off_w_flat,
        ],
        dim=-1,
    )

    hlt_pred_flat, hlt_pred_exist_logit_flat = degrader(off_for_deg, off_mask_flat)
    hlt_pred_w_flat = torch.sigmoid(hlt_pred_exist_logit_flat)

    ch_rep = const_hlt.unsqueeze(1).expand(-1, m, -1, -1).reshape(b * m, const_hlt.shape[1], 4)
    mh_rep = mask_hlt.unsqueeze(1).expand(-1, m, -1).reshape(b * m, mask_hlt.shape[1])

    hlt_loss_flat = _set_loss_chamfer_vec(
        hlt_pred_flat,
        hlt_pred_w_flat,
        ch_rep,
        mh_rep,
        unmatched_penalty=float(unmatched_penalty),
    )
    hlt_loss = hlt_loss_flat.reshape(b, m)
    off_count = off_w_m.sum(dim=2)
    hlt_count = hlt_pred_w_flat.reshape(b, m, -1).sum(dim=2)
    return hlt_loss, off_count, hlt_count


def _latent_refine_shortlist(
    ae: OfflineLatentAE,
    degrader: OfflineToHLTDegrader,
    z_bmd: torch.Tensor,
    cls_bm: torch.Tensor,
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    pred_exist_threshold: float,
    unmatched_penalty: float,
    steps: int,
    step_size: float,
    max_step_norm: float = 0.20,
) -> torch.Tensor:
    n_steps = int(max(steps, 0))
    if n_steps <= 0:
        return z_bmd
    lr = float(max(step_size, 1e-6))
    z_ref = z_bmd.detach()
    for _ in range(n_steps):
        z_var = z_ref.detach().requires_grad_(True)
        hlt_loss, _, _ = _decode_degrade_hlt_loss_matrix(
            ae=ae,
            degrader=degrader,
            z_bmd=z_var,
            cls_bm=cls_bm,
            const_hlt=const_hlt,
            mask_hlt=mask_hlt,
            pred_exist_threshold=float(pred_exist_threshold),
            unmatched_penalty=float(unmatched_penalty),
            detach_threshold_mask=True,
        )
        loss = hlt_loss.mean()
        grad = torch.autograd.grad(loss, z_var, only_inputs=True, create_graph=False)[0]
        step = -lr * grad
        if float(max_step_norm) > 0.0:
            step_norm = torch.sqrt(step.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
            clip = torch.clamp(float(max_step_norm) / step_norm, max=1.0)
            step = step * clip
        z_ref = (z_var + step).detach()
    return z_ref


def _proposer_objective_batch(
    proposer: HLTLatentProposer,
    ae: OfflineLatentAE,
    degrader: OfflineToHLTDegrader,
    prior: PriorStats,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    k0_samples: int,
    top_m_samples: int,
    wrong_k_samples: int,
    softmin_tau: float,
    diversity_tau: float,
    prescore_w_prior: float,
    prescore_w_q: float,
    lambda_count: float,
    lambda_div: float,
    lambda_margin: float,
    margin: float,
    lambda_kl: float,
    lambda_hlt_cons: float,
    refine_steps: int,
    refine_step_size: float,
    refine_max_step_norm: float,
    pred_exist_threshold: float,
    unmatched_penalty: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    const_off = batch["const_off"].to(device)
    mask_off = batch["mask_off"].to(device)
    const_hlt = batch["const_hlt"].to(device)
    mask_hlt = batch["mask_hlt"].to(device)
    y = batch["label"].to(device)
    sw = batch["sample_weight"].to(device)

    bsz = const_off.shape[0]
    k0 = int(max(k0_samples, 1))
    mu_t, lv_t = proposer.dist(const_hlt, mask_hlt, y)
    eps_t = torch.randn((bsz, k0, mu_t.shape[1]), device=device, dtype=mu_t.dtype)
    z_t0 = mu_t.unsqueeze(1) + torch.exp(0.5 * lv_t).unsqueeze(1) * eps_t
    prior_mu_t = prior.mean[y.long()]
    prior_lv_t = prior.logvar[y.long()]
    cheap_t = _cheap_prescore_matrix(
        z_bkd=z_t0,
        mu_bd=mu_t,
        logvar_bd=lv_t,
        prior_mu=prior_mu_t,
        prior_logvar=prior_lv_t,
        w_prior=float(prescore_w_prior),
        w_q=float(prescore_w_q),
    )
    z_t, cheap_t_top, _ = _shortlist_topk_by_score(
        z_bkd=z_t0,
        score_bk=cheap_t,
        top_m=int(top_m_samples),
    )
    c_t = y.unsqueeze(1).expand(-1, z_t.shape[1])
    if int(max(refine_steps, 0)) > 0:
        with torch.enable_grad():
            z_t = _latent_refine_shortlist(
                ae=ae,
                degrader=degrader,
                z_bmd=z_t,
                cls_bm=c_t,
                const_hlt=const_hlt,
                mask_hlt=mask_hlt,
                pred_exist_threshold=float(pred_exist_threshold),
                unmatched_penalty=float(unmatched_penalty),
                steps=int(refine_steps),
                step_size=float(refine_step_size),
                max_step_norm=float(refine_max_step_norm),
            )
    k = int(z_t.shape[1])
    c_t = y.unsqueeze(1).expand(-1, k)

    off_const_k, _off_exist_logit_k, off_w_k = _decode_many(ae, z_t, c_t)

    off_const_flat = off_const_k.reshape(bsz * k, off_const_k.shape[2], 4)
    off_w_flat = off_w_k.reshape(bsz * k, off_w_k.shape[2])
    tgt_off_flat = const_off.unsqueeze(1).expand(-1, k, -1, -1).reshape(bsz * k, const_off.shape[1], 4)
    tgt_mask_flat = mask_off.unsqueeze(1).expand(-1, k, -1).reshape(bsz * k, mask_off.shape[1])

    off_set_vec_flat = _set_loss_chamfer_vec(
        off_const_flat,
        off_w_flat,
        tgt_off_flat,
        tgt_mask_flat,
        unmatched_penalty=float(unmatched_penalty),
    )
    off_cnt_vec_flat = _count_loss_vec(off_w_flat, tgt_mask_flat)

    off_set_mat = off_set_vec_flat.reshape(bsz, k)
    off_cnt_mat = off_cnt_vec_flat.reshape(bsz, k)

    w_soft = torch.softmax(-off_set_mat / max(float(softmin_tau), 1e-4), dim=1)
    loss_best_vec = (w_soft * off_set_mat).sum(dim=1)
    loss_count_vec = (w_soft * off_cnt_mat).sum(dim=1)

    # HLT consistency through learned degrader on decoded candidates.
    if float(lambda_hlt_cons) > 0.0:
        hlt_set_mat, _, _ = _decode_degrade_hlt_loss_matrix(
            ae=ae,
            degrader=degrader,
            z_bmd=z_t,
            cls_bm=c_t,
            const_hlt=const_hlt,
            mask_hlt=mask_hlt,
            pred_exist_threshold=float(pred_exist_threshold),
            unmatched_penalty=float(unmatched_penalty),
            detach_threshold_mask=True,
        )
        loss_hlt_vec = (w_soft * hlt_set_mat).sum(dim=1)
    else:
        loss_hlt_vec = torch.zeros((bsz,), device=device)

    # Wrong-class contrastive margin.
    kw = int(max(wrong_k_samples, 1))
    y_wrong = 1 - y
    mu_w, lv_w = proposer.dist(const_hlt, mask_hlt, y_wrong)
    eps_w = torch.randn((bsz, kw, mu_w.shape[1]), device=device, dtype=mu_w.dtype)
    z_w = mu_w.unsqueeze(1) + torch.exp(0.5 * lv_w).unsqueeze(1) * eps_w
    c_w = y_wrong.unsqueeze(1).expand(-1, kw)
    off_const_w, _off_exist_logit_w, off_w_w = _decode_many(ae, z_w, c_w)

    off_const_w_flat = off_const_w.reshape(bsz * kw, off_const_w.shape[2], 4)
    off_w_w_flat = off_w_w.reshape(bsz * kw, off_w_w.shape[2])
    tgt_off_w_flat = const_off.unsqueeze(1).expand(-1, kw, -1, -1).reshape(bsz * kw, const_off.shape[1], 4)
    tgt_mask_w_flat = mask_off.unsqueeze(1).expand(-1, kw, -1).reshape(bsz * kw, mask_off.shape[1])

    off_wrong_flat = _set_loss_chamfer_vec(
        off_const_w_flat,
        off_w_w_flat,
        tgt_off_w_flat,
        tgt_mask_w_flat,
        unmatched_penalty=float(unmatched_penalty),
    )
    off_wrong = off_wrong_flat.reshape(bsz, kw)
    wrong_best = torch.min(off_wrong, dim=1).values
    margin_vec = F.relu(float(margin) + loss_best_vec - wrong_best)

    # KL to class prior on q(z|hlt,class)
    mu_p = prior.mean[y.long()]
    lv_p = prior.logvar[y.long()]
    kl_vec = _kl_diag_gaussians(mu_t, lv_t, mu_p, lv_p)

    div_vec = _pairwise_diversity_penalty(z_t, tau=float(diversity_tau))

    total_vec = (
        loss_best_vec
        + float(lambda_count) * loss_count_vec
        + float(lambda_div) * div_vec
        + float(lambda_margin) * margin_vec
        + float(lambda_kl) * kl_vec
        + float(lambda_hlt_cons) * loss_hlt_vec
    )
    loss = _weighted_mean(total_vec, sw)

    logs = {
        "best_set": float(_weighted_mean(loss_best_vec.detach(), sw).item()),
        "prescore": float(_weighted_mean(cheap_t_top[:, 0].detach(), sw).item()),
        "count": float(_weighted_mean(loss_count_vec.detach(), sw).item()),
        "div": float(_weighted_mean(div_vec.detach(), sw).item()),
        "margin": float(_weighted_mean(margin_vec.detach(), sw).item()),
        "kl": float(_weighted_mean(kl_vec.detach(), sw).item()),
        "hlt": float(_weighted_mean(loss_hlt_vec.detach(), sw).item()),
        "total": float(loss.detach().item()),
    }
    return loss, logs


def _train_proposer(
    proposer: HLTLatentProposer,
    ae: OfflineLatentAE,
    degrader: OfflineToHLTDegrader,
    prior: PriorStats,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    k0_samples: int,
    top_m_samples: int,
    wrong_k_samples: int,
    softmin_tau: float,
    diversity_tau: float,
    prescore_w_prior: float,
    prescore_w_q: float,
    lambda_count: float,
    lambda_div: float,
    lambda_margin: float,
    margin: float,
    lambda_kl: float,
    lambda_hlt_cons: float,
    refine_steps: int,
    refine_step_size: float,
    refine_max_step_norm: float,
    pred_exist_threshold: float,
    unmatched_penalty: float,
) -> Tuple[HLTLatentProposer, Dict[str, float]]:
    for p in ae.parameters():
        p.requires_grad_(False)
    for p in degrader.parameters():
        p.requires_grad_(False)
    ae.eval()
    degrader.eval()

    opt = torch.optim.AdamW(proposer.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    no_imp = 0

    for ep in range(int(epochs)):
        proposer.train()
        tr_loss = 0.0
        tr_n = 0
        tr_acc = {"best_set": 0.0, "prescore": 0.0, "count": 0.0, "div": 0.0, "margin": 0.0, "kl": 0.0, "hlt": 0.0}

        for batch in train_loader:
            loss, logs = _proposer_objective_batch(
                proposer=proposer,
                ae=ae,
                degrader=degrader,
                prior=prior,
                batch=batch,
                device=device,
                k0_samples=int(k0_samples),
                top_m_samples=int(top_m_samples),
                wrong_k_samples=int(wrong_k_samples),
                softmin_tau=float(softmin_tau),
                diversity_tau=float(diversity_tau),
                prescore_w_prior=float(prescore_w_prior),
                prescore_w_q=float(prescore_w_q),
                lambda_count=float(lambda_count),
                lambda_div=float(lambda_div),
                lambda_margin=float(lambda_margin),
                margin=float(margin),
                lambda_kl=float(lambda_kl),
                lambda_hlt_cons=float(lambda_hlt_cons),
                refine_steps=int(refine_steps),
                refine_step_size=float(refine_step_size),
                refine_max_step_norm=float(refine_max_step_norm),
                pred_exist_threshold=float(pred_exist_threshold),
                unmatched_penalty=float(unmatched_penalty),
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proposer.parameters(), max_norm=1.0)
            opt.step()

            bs = int(batch["label"].shape[0])
            tr_loss += float(loss.item()) * bs
            tr_n += bs
            for k in tr_acc.keys():
                tr_acc[k] += float(logs[k]) * bs

        proposer.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for batch in val_loader:
                loss, _logs = _proposer_objective_batch(
                    proposer=proposer,
                    ae=ae,
                    degrader=degrader,
                    prior=prior,
                    batch=batch,
                    device=device,
                    k0_samples=int(k0_samples),
                    top_m_samples=int(top_m_samples),
                    wrong_k_samples=int(wrong_k_samples),
                    softmin_tau=float(softmin_tau),
                    diversity_tau=float(diversity_tau),
                    prescore_w_prior=float(prescore_w_prior),
                    prescore_w_q=float(prescore_w_q),
                    lambda_count=float(lambda_count),
                    lambda_div=float(lambda_div),
                    lambda_margin=float(lambda_margin),
                    margin=float(margin),
                    lambda_kl=float(lambda_kl),
                    lambda_hlt_cons=float(lambda_hlt_cons),
                    refine_steps=int(refine_steps),
                    refine_step_size=float(refine_step_size),
                    refine_max_step_norm=float(refine_max_step_norm),
                    pred_exist_threshold=float(pred_exist_threshold),
                    unmatched_penalty=float(unmatched_penalty),
                )
                bs = int(batch["label"].shape[0])
                va_loss += float(loss.item()) * bs
                va_n += bs

        tr = tr_loss / max(1, tr_n)
        va = va_loss / max(1, va_n)

        if va < best_val:
            best_val = float(va)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in proposer.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"Proposer ep {ep+1:03d}: train={tr:.5f} val={va:.5f} best={best_val:.5f}@{best_epoch} | "
                f"best_set={tr_acc['best_set']/max(1,tr_n):.5f} prescore={tr_acc['prescore']/max(1,tr_n):.5f} "
                f"count={tr_acc['count']/max(1,tr_n):.5f} "
                f"div={tr_acc['div']/max(1,tr_n):.5f} margin={tr_acc['margin']/max(1,tr_n):.5f} "
                f"kl={tr_acc['kl']/max(1,tr_n):.5f} hlt={tr_acc['hlt']/max(1,tr_n):.5f}"
            )

        if no_imp >= int(patience):
            print(f"Proposer early stop at ep {ep+1}")
            break

    if best_state is not None:
        proposer.load_state_dict(best_state)

    return proposer, {
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
    }


def _build_feasibility_features(
    proposer: HLTLatentProposer,
    ae: OfflineLatentAE,
    degrader: OfflineToHLTDegrader,
    prior: PriorStats,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    baseline_prob: np.ndarray,
    device: torch.device,
    batch_size: int,
    k0_infer: int,
    top_m_infer: int,
    prescore_w_prior: float,
    prescore_w_q: float,
    latent_refine_steps: int,
    latent_refine_lr: float,
    latent_refine_max_step_norm: float,
    pred_exist_threshold: float,
    unmatched_penalty: float,
) -> np.ndarray:
    proposer.eval()
    ae.eval()
    degrader.eval()

    const_t = torch.tensor(const_hlt, dtype=torch.float32)
    mask_t = torch.tensor(mask_hlt, dtype=torch.bool)

    feats_all: List[np.ndarray] = []
    n = int(const_hlt.shape[0])
    k0 = int(max(k0_infer, 1))
    top_m = int(max(top_m_infer, 1))
    refine_steps = int(max(latent_refine_steps, 0))

    for s in tqdm(range(0, n, int(batch_size)), desc="FeasFeatures", leave=False):
        e = min(n, s + int(batch_size))
        ch = const_t[s:e].to(device)
        mh = mask_t[s:e].to(device)
        b = int(ch.shape[0])

        class_stats = []
        for cls_val in [0, 1]:
            cls = torch.full((b,), int(cls_val), device=device, dtype=torch.long)
            mu, lv = proposer.dist(ch, mh, cls)
            eps = torch.randn((b, k0, mu.shape[1]), device=device, dtype=mu.dtype)
            z0 = mu.unsqueeze(1) + torch.exp(0.5 * lv).unsqueeze(1) * eps
            cheap = _cheap_prescore_matrix(
                z_bkd=z0,
                mu_bd=mu,
                logvar_bd=lv,
                prior_mu=prior.mean[int(cls_val)],
                prior_logvar=prior.logvar[int(cls_val)],
                w_prior=float(prescore_w_prior),
                w_q=float(prescore_w_q),
            )
            z, cheap_top, _ = _shortlist_topk_by_score(
                z_bkd=z0,
                score_bk=cheap,
                top_m=top_m,
            )
            c = cls.unsqueeze(1).expand(-1, z.shape[1])
            if refine_steps > 0:
                with torch.enable_grad():
                    z = _latent_refine_shortlist(
                        ae=ae,
                        degrader=degrader,
                        z_bmd=z,
                        cls_bm=c,
                        const_hlt=ch,
                        mask_hlt=mh,
                        pred_exist_threshold=float(pred_exist_threshold),
                        unmatched_penalty=float(unmatched_penalty),
                        steps=refine_steps,
                        step_size=float(latent_refine_lr),
                        max_step_norm=float(latent_refine_max_step_norm),
                    )

            hlt_loss, off_count, hlt_count = _decode_degrade_hlt_loss_matrix(
                ae=ae,
                degrader=degrader,
                z_bmd=z,
                cls_bm=c,
                const_hlt=ch,
                mask_hlt=mh,
                pred_exist_threshold=float(pred_exist_threshold),
                unmatched_penalty=float(unmatched_penalty),
                detach_threshold_mask=True,
            )
            best_idx = torch.argmin(hlt_loss, dim=1)

            idx = best_idx.view(b, 1)
            best_hlt = hlt_loss.gather(1, idx).squeeze(1)

            best_off_count = off_count.gather(1, idx).squeeze(1)
            best_hlt_count = hlt_count.gather(1, idx).squeeze(1)
            best_prescore = cheap_top.gather(1, idx).squeeze(1)

            k = int(z.shape[1])
            z_flat = z.reshape(b * k, z.shape[2])
            prior_mu = prior.mean[int(cls_val)].view(1, -1)
            prior_lv = prior.logvar[int(cls_val)].view(1, -1)
            z_nll_flat = _gaussian_nll_vec(z_flat, prior_mu, prior_lv)
            z_nll = z_nll_flat.reshape(b, k)
            best_nll = z_nll.gather(1, idx).squeeze(1)

            class_stats.append((best_hlt, best_nll, best_off_count, best_hlt_count, best_prescore))

        # class_stats[0]=bg, class_stats[1]=top
        bg_hlt, bg_nll, bg_offc, bg_hltc, bg_pre = class_stats[0]
        tp_hlt, tp_nll, tp_offc, tp_hltc, tp_pre = class_stats[1]

        delta = bg_hlt - tp_hlt
        ratio = tp_hlt / (bg_hlt + 1e-6)

        bp = torch.tensor(baseline_prob[s:e], device=device, dtype=tp_hlt.dtype)

        feat = torch.stack(
            [
                tp_hlt,
                bg_hlt,
                delta,
                ratio,
                tp_nll,
                bg_nll,
                tp_offc,
                bg_offc,
                tp_hltc,
                bg_hltc,
                tp_pre,
                bg_pre,
                bp,
            ],
            dim=1,
        )
        feats_all.append(feat.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(feats_all, axis=0)


def _train_final_classifier(
    model: FinalFeasibilityClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[FinalFeasibilityClassifier, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = None
    best_val_auc = float("-inf")
    best_epoch = -1
    no_imp = 0

    for ep in range(int(epochs)):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        tr_pred: List[np.ndarray] = []
        tr_lab: List[np.ndarray] = []
        tr_w: List[np.ndarray] = []

        for batch in train_loader:
            ch = batch["const_hlt"].to(device)
            mh = batch["mask_hlt"].to(device)
            ff = batch["feas_feat"].to(device)
            y = batch["label"].to(device)
            sw = batch["sample_weight"].to(device)

            logits = model(ch, mh, ff)
            loss_vec = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = _weighted_mean(loss_vec, sw)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bs = int(y.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_n += bs
            tr_pred.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64))
            tr_lab.append(y.detach().cpu().numpy().astype(np.float64))
            tr_w.append(sw.detach().cpu().numpy().astype(np.float64))

        model.eval()
        va_pred: List[np.ndarray] = []
        va_lab: List[np.ndarray] = []
        va_w: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                ch = batch["const_hlt"].to(device)
                mh = batch["mask_hlt"].to(device)
                ff = batch["feas_feat"].to(device)
                y = batch["label"].to(device)
                sw = batch["sample_weight"].to(device)

                p = torch.sigmoid(model(ch, mh, ff))
                va_pred.append(p.detach().cpu().numpy().astype(np.float64))
                va_lab.append(y.detach().cpu().numpy().astype(np.float64))
                va_w.append(sw.detach().cpu().numpy().astype(np.float64))

        tr_p = np.concatenate(tr_pred, axis=0) if tr_pred else np.array([], dtype=np.float64)
        tr_y = np.concatenate(tr_lab, axis=0) if tr_lab else np.array([], dtype=np.float64)
        tr_w_np = np.concatenate(tr_w, axis=0) if tr_w else None
        va_p = np.concatenate(va_pred, axis=0) if va_pred else np.array([], dtype=np.float64)
        va_y = np.concatenate(va_lab, axis=0) if va_lab else np.array([], dtype=np.float64)
        va_w_np = np.concatenate(va_w, axis=0) if va_w else None

        tr_auc = float(roc_auc_score(tr_y, tr_p, sample_weight=tr_w_np)) if len(np.unique(tr_y)) > 1 else 0.0
        va_auc = float(roc_auc_score(va_y, va_p, sample_weight=va_w_np)) if len(np.unique(va_y)) > 1 else 0.0

        if va_auc > best_val_auc:
            best_val_auc = float(va_auc)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"FinalClf ep {ep+1:03d}: train_loss={tr_loss/max(1,tr_n):.5f} "
                f"train_auc={tr_auc:.4f} val_auc={va_auc:.4f} best_auc={best_val_auc:.4f}@{best_epoch}"
            )

        if no_imp >= int(patience):
            print(f"FinalClf early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_auc": float(best_val_auc),
        "best_epoch": int(best_epoch),
    }


@torch.no_grad()
def _eval_final_classifier(
    model: FinalFeasibilityClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    for batch in loader:
        ch = batch["const_hlt"].to(device)
        mh = batch["mask_hlt"].to(device)
        ff = batch["feas_feat"].to(device)
        y = batch["label"].to(device)
        sw = batch["sample_weight"].to(device)
        p = torch.sigmoid(model(ch, mh, ff))
        preds.append(p.detach().cpu().numpy().astype(np.float64))
        labs.append(y.detach().cpu().numpy().astype(np.float64))
        ws.append(sw.detach().cpu().numpy().astype(np.float64))

    p_np = np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float64)
    y_np = np.concatenate(labs, axis=0) if labs else np.array([], dtype=np.float64)
    w_np = np.concatenate(ws, axis=0) if ws else np.array([], dtype=np.float64)

    auc = float(roc_auc_score(y_np, p_np, sample_weight=w_np)) if len(np.unique(y_np)) > 1 else 0.0
    fpr, tpr, _ = roc_curve(y_np, p_np, sample_weight=w_np)
    fpr50 = fpr_at_target_tpr(fpr, tpr, target_tpr=0.50)
    return auc, float(fpr50), p_np, y_np, w_np


@torch.no_grad()
def _evaluate_hard_residual_matrix(
    cand_const_bmd: torch.Tensor,
    cand_mask_bmd: torch.Tensor,
    tgt_const_bld: torch.Tensor,
    tgt_mask_bl: torch.Tensor,
    jet_keys_b: np.ndarray,
    cfg: Dict,
    base_seed: int,
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    unmatched_penalty: float,
) -> Dict[str, torch.Tensor]:
    device = cand_const_bmd.device
    b, m, l, _ = cand_const_bmd.shape

    cand_const_flat = cand_const_bmd.detach().cpu().numpy().reshape(b * m, l, 4).astype(np.float32)
    cand_mask_flat = cand_mask_bmd.detach().cpu().numpy().reshape(b * m, l).astype(bool)
    keys_flat = np.repeat(jet_keys_b.astype(np.int64), m)

    hlt_pred_np, hlt_mask_np, _ = _apply_hlt_effects_deterministic_keyed(
        const=cand_const_flat,
        mask=cand_mask_flat,
        cfg=cfg,
        jet_keys=keys_flat,
        base_seed=int(base_seed),
    )

    hlt_pred = torch.tensor(hlt_pred_np, dtype=torch.float32, device=device)
    hlt_mask = torch.tensor(hlt_mask_np, dtype=torch.bool, device=device)
    tgt_rep_const = tgt_const_bld.unsqueeze(1).expand(-1, m, -1, -1).reshape(b * m, l, 4)
    tgt_rep_mask = tgt_mask_bl.unsqueeze(1).expand(-1, m, -1).reshape(b * m, l)

    resid = _residual_fast_vec(
        pred_const=hlt_pred,
        pred_mask=hlt_mask,
        tgt_const=tgt_rep_const,
        tgt_mask=tgt_rep_mask,
        w_chamfer=float(w_chamfer),
        w_count=float(w_count),
        w_pt=float(w_pt),
        w_mass=float(w_mass),
        unmatched_penalty=float(unmatched_penalty),
    )
    return {k: v.reshape(b, m) for k, v in resid.items()}


def _chunked_search_candidates_split(
    proposer: HLTLatentProposer,
    ae: OfflineLatentAE,
    degrader: OfflineToHLTDegrader,
    prior: PriorStats,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    jet_keys: np.ndarray,
    cfg: Dict,
    base_seed: int,
    device: torch.device,
    batch_size: int,
    target_k: int,
    chunk_k0: int,
    shortlist_m: int,
    max_rounds: int,
    keep_per_round: int,
    max_pool_size: int,
    eps_total: float,
    eps_count: float,
    pred_exist_threshold: float,
    unmatched_penalty: float,
    prescore_w_prior: float,
    prescore_w_q: float,
    refine_steps: int,
    refine_step_size: float,
    refine_max_step_norm: float,
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    proposer.eval()
    ae.eval()
    degrader.eval()

    n, max_const, _ = const_hlt.shape
    out: Dict[str, Dict[str, np.ndarray]] = {}
    overall_feasible = {}

    const_t = torch.tensor(const_hlt, dtype=torch.float32)
    mask_t = torch.tensor(mask_hlt, dtype=torch.bool)

    for cls_val in [0, 1]:
        cand_const_out = np.zeros((n, target_k, max_const, 4), dtype=np.float32)
        cand_mask_out = np.zeros((n, target_k, max_const), dtype=bool)
        resid_total_out = np.full((n, target_k), np.inf, dtype=np.float32)
        resid_set_out = np.full((n, target_k), np.inf, dtype=np.float32)
        resid_count_out = np.full((n, target_k), np.inf, dtype=np.float32)
        resid_pt_out = np.full((n, target_k), np.inf, dtype=np.float32)
        resid_mass_out = np.full((n, target_k), np.inf, dtype=np.float32)
        feasible_out = np.zeros((n, target_k), dtype=bool)
        feasible_cnt_vec = np.zeros((n,), dtype=np.int32)

        for s in tqdm(range(0, n, int(batch_size)), desc=f"SearchC{cls_val}", leave=False):
            e = min(n, s + int(batch_size))
            b = e - s
            ch = const_t[s:e].to(device)
            mh = mask_t[s:e].to(device)
            keys_b = jet_keys[s:e].astype(np.int64)

            pools: List[List[Dict[str, object]]] = [[] for _ in range(b)]

            for rd in range(int(max_rounds)):
                need_idx = [
                    i for i in range(b)
                    if sum(1 for z in pools[i] if bool(z["feasible"])) < int(target_k)
                ]
                if not need_idx:
                    break

                sub_ch = ch[need_idx]
                sub_mh = mh[need_idx]
                sub_keys = keys_b[need_idx]
                bs_need = int(sub_ch.shape[0])
                cls = torch.full((bs_need,), int(cls_val), device=device, dtype=torch.long)

                mu, lv = proposer.dist(sub_ch, sub_mh, cls)
                eps = torch.randn((bs_need, int(chunk_k0), mu.shape[1]), device=device, dtype=mu.dtype)
                z0 = mu.unsqueeze(1) + torch.exp(0.5 * lv).unsqueeze(1) * eps
                cheap = _cheap_prescore_matrix(
                    z_bkd=z0,
                    mu_bd=mu,
                    logvar_bd=lv,
                    prior_mu=prior.mean[int(cls_val)],
                    prior_logvar=prior.logvar[int(cls_val)],
                    w_prior=float(prescore_w_prior),
                    w_q=float(prescore_w_q),
                )
                z, _cheap_top, _ = _shortlist_topk_by_score(
                    z_bkd=z0,
                    score_bk=cheap,
                    top_m=int(shortlist_m),
                )
                c = cls.unsqueeze(1).expand(-1, z.shape[1])

                if int(refine_steps) > 0:
                    with torch.enable_grad():
                        z = _latent_refine_shortlist(
                            ae=ae,
                            degrader=degrader,
                            z_bmd=z,
                            cls_bm=c,
                            const_hlt=sub_ch,
                            mask_hlt=sub_mh,
                            pred_exist_threshold=float(pred_exist_threshold),
                            unmatched_penalty=float(unmatched_penalty),
                            steps=int(refine_steps),
                            step_size=float(refine_step_size),
                            max_step_norm=float(refine_max_step_norm),
                        )

                off_const, _off_exist_logit, off_w = _decode_many(ae, z, c)
                off_mask = off_w > float(pred_exist_threshold)

                resid = _evaluate_hard_residual_matrix(
                    cand_const_bmd=off_const,
                    cand_mask_bmd=off_mask,
                    tgt_const_bld=sub_ch,
                    tgt_mask_bl=sub_mh,
                    jet_keys_b=sub_keys,
                    cfg=cfg,
                    base_seed=int(base_seed),
                    w_chamfer=float(w_chamfer),
                    w_count=float(w_count),
                    w_pt=float(w_pt),
                    w_mass=float(w_mass),
                    unmatched_penalty=float(unmatched_penalty),
                )
                keep = int(max(1, min(int(keep_per_round), int(off_const.shape[1]))))

                for ii, src_i in enumerate(need_idx):
                    ord_idx = torch.argsort(resid["total"][ii])[:keep]
                    for jj_t in ord_idx.tolist():
                        rt = float(resid["total"][ii, jj_t].item())
                        rc = float(resid["count"][ii, jj_t].item())
                        entry = {
                            "const": off_const[ii, jj_t].detach().cpu().numpy().astype(np.float32),
                            "mask": off_mask[ii, jj_t].detach().cpu().numpy().astype(bool),
                            "res_total": rt,
                            "res_set": float(resid["set"][ii, jj_t].item()),
                            "res_count": rc,
                            "res_pt": float(resid["pt"][ii, jj_t].item()),
                            "res_mass": float(resid["mass"][ii, jj_t].item()),
                            "feasible": bool((rt <= float(eps_total)) and (rc <= float(eps_count))),
                        }
                        pools[src_i].append(entry)
                    if len(pools[src_i]) > int(max_pool_size):
                        pools[src_i].sort(key=lambda z: float(z["res_total"]))
                        pools[src_i] = pools[src_i][: int(max_pool_size)]

            for i in range(b):
                entries = pools[i]
                if not entries:
                    # Emergency fallback.
                    cls = torch.full((1,), int(cls_val), device=device, dtype=torch.long)
                    mu, _lv = proposer.dist(ch[i : i + 1], mh[i : i + 1], cls)
                    z = mu.unsqueeze(1)
                    c = cls.unsqueeze(1)
                    off_const, _off_exist_logit, off_w = _decode_many(ae, z, c)
                    off_mask = off_w > float(pred_exist_threshold)
                    resid = _evaluate_hard_residual_matrix(
                        cand_const_bmd=off_const,
                        cand_mask_bmd=off_mask,
                        tgt_const_bld=ch[i : i + 1],
                        tgt_mask_bl=mh[i : i + 1],
                        jet_keys_b=np.array([keys_b[i]], dtype=np.int64),
                        cfg=cfg,
                        base_seed=int(base_seed),
                        w_chamfer=float(w_chamfer),
                        w_count=float(w_count),
                        w_pt=float(w_pt),
                        w_mass=float(w_mass),
                        unmatched_penalty=float(unmatched_penalty),
                    )
                    entries = [{
                        "const": off_const[0, 0].detach().cpu().numpy().astype(np.float32),
                        "mask": off_mask[0, 0].detach().cpu().numpy().astype(bool),
                        "res_total": float(resid["total"][0, 0].item()),
                        "res_set": float(resid["set"][0, 0].item()),
                        "res_count": float(resid["count"][0, 0].item()),
                        "res_pt": float(resid["pt"][0, 0].item()),
                        "res_mass": float(resid["mass"][0, 0].item()),
                        "feasible": False,
                    }]

                feas = [z for z in entries if bool(z["feasible"])]
                nonf = [z for z in entries if not bool(z["feasible"])]
                feas.sort(key=lambda z: float(z["res_total"]))
                nonf.sort(key=lambda z: float(z["res_total"]))
                selected = (feas + nonf)[: int(target_k)]

                g = s + i
                feasible_cnt_vec[g] = int(len(feas))
                for k, z in enumerate(selected):
                    cand_const_out[g, k] = z["const"]  # type: ignore[index]
                    cand_mask_out[g, k] = z["mask"]  # type: ignore[index]
                    resid_total_out[g, k] = float(z["res_total"])  # type: ignore[index]
                    resid_set_out[g, k] = float(z["res_set"])  # type: ignore[index]
                    resid_count_out[g, k] = float(z["res_count"])  # type: ignore[index]
                    resid_pt_out[g, k] = float(z["res_pt"])  # type: ignore[index]
                    resid_mass_out[g, k] = float(z["res_mass"])  # type: ignore[index]
                    feasible_out[g, k] = bool(z["feasible"])  # type: ignore[index]

        out[f"class{cls_val}"] = {
            "const": cand_const_out,
            "mask": cand_mask_out,
            "res_total": resid_total_out,
            "res_set": resid_set_out,
            "res_count": resid_count_out,
            "res_pt": resid_pt_out,
            "res_mass": resid_mass_out,
            "feasible": feasible_out,
            "feasible_count": feasible_cnt_vec.astype(np.float32),
        }
        overall_feasible[f"class{cls_val}_mean_feasible"] = float(np.mean(np.minimum(feasible_cnt_vec, target_k)))

    out["stats"] = {
        "target_k": int(target_k),
        **overall_feasible,
    }
    return out


def _build_selector_arrays(
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_off: np.ndarray,
    mask_off: np.ndarray,
    pools: Dict[str, Dict[str, np.ndarray]],
    neg_per_class: int,
) -> Dict[str, np.ndarray]:
    rows_hlt = []
    rows_hlt_m = []
    rows_cand = []
    rows_cand_m = []
    rows_cls = []
    rows_resid = []
    rows_y = []

    n = int(const_hlt.shape[0])
    for i in range(n):
        # Positive real offline sample.
        rows_hlt.append(const_hlt[i])
        rows_hlt_m.append(mask_hlt[i])
        rows_cand.append(const_off[i])
        rows_cand_m.append(mask_off[i])
        rows_cls.append(0)
        rows_resid.append(0.0)
        rows_y.append(1.0)

        for cls_val in [0, 1]:
            ckey = f"class{cls_val}"
            cands = pools[ckey]["const"][i]
            cmsk = pools[ckey]["mask"][i]
            cres = pools[ckey]["res_total"][i]
            ord_idx = np.argsort(cres)[: int(max(1, neg_per_class))]
            for j in ord_idx.tolist():
                rows_hlt.append(const_hlt[i])
                rows_hlt_m.append(mask_hlt[i])
                rows_cand.append(cands[j])
                rows_cand_m.append(cmsk[j])
                rows_cls.append(int(cls_val))
                rows_resid.append(float(cres[j]))
                rows_y.append(0.0)

    y = np.asarray(rows_y, dtype=np.float32)
    pos = max(1, int((y > 0.5).sum()))
    neg = max(1, int((y <= 0.5).sum()))
    w_pos = float(0.5 / pos)
    w_neg = float(0.5 / neg)
    sw = np.where(y > 0.5, w_pos, w_neg).astype(np.float32)
    sw = sw / max(1e-8, float(sw.mean()))

    return {
        "const_hlt": np.asarray(rows_hlt, dtype=np.float32),
        "mask_hlt": np.asarray(rows_hlt_m, dtype=bool),
        "cand_const": np.asarray(rows_cand, dtype=np.float32),
        "cand_mask": np.asarray(rows_cand_m, dtype=bool),
        "cand_class": np.asarray(rows_cls, dtype=np.int64),
        "cand_resid": np.asarray(rows_resid, dtype=np.float32),
        "labels": y,
        "sample_weight": sw,
    }


def _train_selector(
    model: CandidateRealismSelector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[CandidateRealismSelector, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_auc = float("-inf")
    best_epoch = 0
    no_imp = 0

    for ep in range(int(epochs)):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for batch in train_loader:
            ch = batch["const_hlt"].to(device)
            mh = batch["mask_hlt"].to(device)
            cc = batch["cand_const"].to(device)
            cm = batch["cand_mask"].to(device)
            cls = batch["cand_class"].to(device)
            rr = batch["cand_resid"].to(device)
            y = batch["label"].to(device)
            sw = batch["sample_weight"].to(device)

            logit = model(ch, mh, cc, cm, cls, rr)
            lv = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            loss = _weighted_mean(lv, sw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            bs = int(y.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_n += bs

        model.eval()
        vp, vy, vw = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ch = batch["const_hlt"].to(device)
                mh = batch["mask_hlt"].to(device)
                cc = batch["cand_const"].to(device)
                cm = batch["cand_mask"].to(device)
                cls = batch["cand_class"].to(device)
                rr = batch["cand_resid"].to(device)
                y = batch["label"].to(device)
                sw = batch["sample_weight"].to(device)
                p = torch.sigmoid(model(ch, mh, cc, cm, cls, rr))
                vp.append(p.detach().cpu().numpy().astype(np.float64))
                vy.append(y.detach().cpu().numpy().astype(np.float64))
                vw.append(sw.detach().cpu().numpy().astype(np.float64))
        vp_np = np.concatenate(vp, axis=0) if vp else np.array([], dtype=np.float64)
        vy_np = np.concatenate(vy, axis=0) if vy else np.array([], dtype=np.float64)
        vw_np = np.concatenate(vw, axis=0) if vw else None
        va_auc = float(roc_auc_score(vy_np, vp_np, sample_weight=vw_np)) if len(np.unique(vy_np)) > 1 else 0.0

        if va_auc > best_auc:
            best_auc = float(va_auc)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"Selector ep {ep+1:03d}: train_loss={tr_loss/max(1,tr_n):.5f} val_auc={va_auc:.4f} best={best_auc:.4f}@{best_epoch}")
        if no_imp >= int(patience):
            print(f"Selector early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_auc": float(best_auc), "best_epoch": int(best_epoch)}


@torch.no_grad()
def _score_selector_candidates(
    selector: CandidateRealismSelector,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    cand_const: np.ndarray,
    cand_mask: np.ndarray,
    cand_resid: np.ndarray,
    cand_class_val: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    selector.eval()
    n, k, l, _ = cand_const.shape
    out = np.zeros((n, k), dtype=np.float32)
    cls = np.full((n * k,), int(cand_class_val), dtype=np.int64)

    hlt_rep = np.repeat(const_hlt[:, None, :, :], k, axis=1).reshape(n * k, l, 4)
    hlt_m_rep = np.repeat(mask_hlt[:, None, :], k, axis=1).reshape(n * k, l)
    cand_flat = cand_const.reshape(n * k, l, 4)
    cand_m_flat = cand_mask.reshape(n * k, l)
    resid_flat = cand_resid.reshape(n * k)

    for s in range(0, n * k, int(batch_size)):
        e = min(n * k, s + int(batch_size))
        ch = torch.tensor(hlt_rep[s:e], dtype=torch.float32, device=device)
        mh = torch.tensor(hlt_m_rep[s:e], dtype=torch.bool, device=device)
        cc = torch.tensor(cand_flat[s:e], dtype=torch.float32, device=device)
        cm = torch.tensor(cand_m_flat[s:e], dtype=torch.bool, device=device)
        cl = torch.tensor(cls[s:e], dtype=torch.long, device=device)
        rr = torch.tensor(resid_flat[s:e], dtype=torch.float32, device=device)
        p = torch.sigmoid(selector(ch, mh, cc, cm, cl, rr)).detach().cpu().numpy().astype(np.float32)
        out.reshape(-1)[s:e] = p
    return out


def _build_dualview_features(
    pools: Dict[str, Dict[str, np.ndarray]],
    sel_score_bg: np.ndarray,
    sel_score_top: np.ndarray,
    baseline_prob: np.ndarray,
    score_alpha: float,
) -> Dict[str, np.ndarray]:
    c0 = pools["class0"]
    c1 = pools["class1"]
    n, k, l, _ = c0["const"].shape

    q_bg = sel_score_bg - float(score_alpha) * c0["res_total"]
    q_tp = sel_score_top - float(score_alpha) * c1["res_total"]
    idx_bg = np.argmax(q_bg, axis=1)
    idx_tp = np.argmax(q_tp, axis=1)

    row = np.arange(n)
    bg_const = c0["const"][row, idx_bg]
    bg_mask = c0["mask"][row, idx_bg]
    tp_const = c1["const"][row, idx_tp]
    tp_mask = c1["mask"][row, idx_tp]

    bg_res = c0["res_total"][row, idx_bg]
    tp_res = c1["res_total"][row, idx_tp]
    bg_set = c0["res_set"][row, idx_bg]
    tp_set = c1["res_set"][row, idx_tp]
    bg_cnt = c0["res_count"][row, idx_bg]
    tp_cnt = c1["res_count"][row, idx_tp]
    bg_pt = c0["res_pt"][row, idx_bg]
    tp_pt = c1["res_pt"][row, idx_tp]
    bg_mass = c0["res_mass"][row, idx_bg]
    tp_mass = c1["res_mass"][row, idx_tp]

    feat = np.stack(
        [
            tp_res,
            bg_res,
            bg_res - tp_res,
            tp_set,
            bg_set,
            tp_cnt,
            bg_cnt,
            tp_pt,
            bg_pt,
            tp_mass,
            bg_mass,
            sel_score_top[row, idx_tp],
            sel_score_bg[row, idx_bg],
            q_tp[row, idx_tp],
            q_bg[row, idx_bg],
            np.mean(c1["feasible"], axis=1),
            np.mean(c0["feasible"], axis=1),
            baseline_prob.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "cand_top_const": tp_const.astype(np.float32),
        "cand_top_mask": tp_mask.astype(bool),
        "cand_bg_const": bg_const.astype(np.float32),
        "cand_bg_mask": bg_mask.astype(bool),
        "cand_feat": feat,
    }


def _forward_dualview(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        const_hlt=batch["const_hlt"],
        mask_hlt=batch["mask_hlt"],
        cand_top_const=batch["cand_top_const"],
        cand_top_mask=batch["cand_top_mask"],
        cand_bg_const=batch["cand_bg_const"],
        cand_bg_mask=batch["cand_bg_mask"],
        cand_feat=batch["cand_feat"],
    )


def _train_dualview_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    name: str,
) -> Tuple[nn.Module, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_auc = float("-inf")
    best_epoch = 0
    no_imp = 0

    for ep in range(int(epochs)):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch["label"]
            sw = batch["sample_weight"]
            logit = _forward_dualview(model, batch)
            lv = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            loss = _weighted_mean(lv, sw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            bs = int(y.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_n += bs

        model.eval()
        vp, vy, vw = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                y = batch["label"]
                sw = batch["sample_weight"]
                p = torch.sigmoid(_forward_dualview(model, batch))
                vp.append(p.detach().cpu().numpy().astype(np.float64))
                vy.append(y.detach().cpu().numpy().astype(np.float64))
                vw.append(sw.detach().cpu().numpy().astype(np.float64))
        vp_np = np.concatenate(vp, axis=0) if vp else np.array([], dtype=np.float64)
        vy_np = np.concatenate(vy, axis=0) if vy else np.array([], dtype=np.float64)
        vw_np = np.concatenate(vw, axis=0) if vw else None
        va_auc = float(roc_auc_score(vy_np, vp_np, sample_weight=vw_np)) if len(np.unique(vy_np)) > 1 else 0.0

        if va_auc > best_auc:
            best_auc = float(va_auc)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"{name} ep {ep+1:03d}: train_loss={tr_loss/max(1,tr_n):.5f} val_auc={va_auc:.4f} best={best_auc:.4f}@{best_epoch}")
        if no_imp >= int(patience):
            print(f"{name} early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_auc": float(best_auc), "best_epoch": int(best_epoch)}


@torch.no_grad()
def _eval_dualview_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pp, yy, ww = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["label"]
        sw = batch["sample_weight"]
        p = torch.sigmoid(_forward_dualview(model, batch))
        pp.append(p.detach().cpu().numpy().astype(np.float64))
        yy.append(y.detach().cpu().numpy().astype(np.float64))
        ww.append(sw.detach().cpu().numpy().astype(np.float64))
    p_np = np.concatenate(pp, axis=0) if pp else np.array([], dtype=np.float64)
    y_np = np.concatenate(yy, axis=0) if yy else np.array([], dtype=np.float64)
    w_np = np.concatenate(ww, axis=0) if ww else np.array([], dtype=np.float64)
    auc = float(roc_auc_score(y_np, p_np, sample_weight=w_np)) if len(np.unique(y_np)) > 1 else 0.0
    fpr, tpr, _ = roc_curve(y_np, p_np, sample_weight=w_np)
    fpr50 = fpr_at_target_tpr(fpr, tpr, target_tpr=0.50)
    return auc, float(fpr50), p_np, y_np, w_np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m33 deterministic-feasibility dualview top tagging")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model33_detfeas_dualview")
    p.add_argument("--run_name", type=str, default="model33_k6_detfeas_dualview_150k75k150k_seed0")

    p.add_argument("--n_train_jets", type=int, default=375000)
    p.add_argument("--n_train_split", type=int, default=100000)
    p.add_argument("--n_val_split", type=int, default=75000)
    p.add_argument("--n_test_split", type=int, default=150000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--use_train_weights", action="store_true")

    # HLT effects
    p.add_argument("--merge_radius", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["merge_radius"]))
    p.add_argument("--eff_plateau_barrel", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"]))
    p.add_argument("--eff_plateau_endcap", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"]))
    p.add_argument("--smear_a", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_a"]))
    p.add_argument("--smear_b", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_b"]))
    p.add_argument("--smear_c", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_c"]))
    p.add_argument("--dhard_seed_offset", type=int, default=1337)

    # Teacher/baseline
    p.add_argument("--cls_epochs", type=int, default=60)
    p.add_argument("--cls_patience", type=int, default=12)
    p.add_argument("--cls_lr", type=float, default=3e-4)
    p.add_argument("--cls_weight_decay", type=float, default=1e-4)
    p.add_argument("--cls_warmup_epochs", type=int, default=3)

    # Confidence weighting for class-sensitive stages
    p.add_argument("--conf_weight_floor", type=float, default=0.10)
    p.add_argument("--disable_conf_hard_correct_gate", action="store_true")
    p.add_argument("--disable_conf_mean_normalize", action="store_true")

    # Latent/architecture
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)

    # Stage 1: offline prior AE
    p.add_argument("--prior_epochs", type=int, default=80)
    p.add_argument("--prior_patience", type=int, default=14)
    p.add_argument("--prior_lr", type=float, default=2e-4)
    p.add_argument("--prior_weight_decay", type=float, default=1e-4)
    p.add_argument("--prior_loss_w_count", type=float, default=0.30)
    p.add_argument("--prior_loss_w_lat_reg", type=float, default=1e-4)

    # Stage 2: offline->HLT degrader
    p.add_argument("--degrader_epochs", type=int, default=70)
    p.add_argument("--degrader_patience", type=int, default=12)
    p.add_argument("--degrader_lr", type=float, default=2e-4)
    p.add_argument("--degrader_weight_decay", type=float, default=1e-4)
    p.add_argument("--degrader_loss_w_count", type=float, default=0.30)

    # Stage 3: proposer
    p.add_argument("--proposer_epochs", type=int, default=90)
    p.add_argument("--proposer_patience", type=int, default=16)
    p.add_argument("--proposer_lr", type=float, default=1.5e-4)
    p.add_argument("--proposer_weight_decay", type=float, default=1e-4)
    p.add_argument("--proposer_k0_train", type=int, default=64)
    p.add_argument("--proposer_top_m_train", type=int, default=8)
    p.add_argument("--proposer_k_train", type=int, default=-1, help="Deprecated alias for --proposer_top_m_train.")
    p.add_argument("--proposer_k_wrong", type=int, default=3)
    p.add_argument("--proposer_prescore_w_prior", type=float, default=0.70)
    p.add_argument("--proposer_prescore_w_q", type=float, default=0.30)
    p.add_argument("--proposer_softmin_tau", type=float, default=0.12)
    p.add_argument("--proposer_diversity_tau", type=float, default=1.0)
    p.add_argument("--proposer_lambda_count", type=float, default=0.25)
    p.add_argument("--proposer_lambda_div", type=float, default=0.08)
    p.add_argument("--proposer_lambda_margin", type=float, default=0.50)
    p.add_argument("--proposer_margin", type=float, default=0.20)
    p.add_argument("--proposer_lambda_kl", type=float, default=0.03)
    p.add_argument("--proposer_lambda_hlt_cons", type=float, default=0.05)
    p.add_argument("--proposer_refine_steps", type=int, default=0)
    p.add_argument("--proposer_refine_lr", type=float, default=0.05)
    p.add_argument("--proposer_refine_max_step_norm", type=float, default=0.20)

    # Stage 4/5: deterministic chunk search + selector + final dualview
    p.add_argument("--k0_infer", type=int, default=64)
    p.add_argument("--top_m_infer", type=int, default=8)
    p.add_argument("--k_infer", type=int, default=-1, help="Deprecated alias for --top_m_infer.")
    p.add_argument("--infer_refine_steps", type=int, default=0)
    p.add_argument("--infer_refine_lr", type=float, default=0.05)
    p.add_argument("--infer_refine_max_step_norm", type=float, default=0.20)
    p.add_argument("--search_batch_size", type=int, default=48)
    p.add_argument("--search_target_k", type=int, default=6)
    p.add_argument("--search_chunk_k0", type=int, default=100)
    p.add_argument("--search_shortlist_m", type=int, default=20)
    p.add_argument("--search_max_rounds", type=int, default=14)
    p.add_argument("--search_keep_per_round", type=int, default=8)
    p.add_argument("--search_max_pool_size", type=int, default=96)
    p.add_argument("--search_eps_total", type=float, default=0.40)
    p.add_argument("--search_eps_count", type=float, default=0.45)
    p.add_argument("--search_w_chamfer", type=float, default=1.0)
    p.add_argument("--search_w_count", type=float, default=0.25)
    p.add_argument("--search_w_pt", type=float, default=0.10)
    p.add_argument("--search_w_mass", type=float, default=0.05)

    p.add_argument("--selector_epochs", type=int, default=40)
    p.add_argument("--selector_patience", type=int, default=8)
    p.add_argument("--selector_lr", type=float, default=2e-4)
    p.add_argument("--selector_weight_decay", type=float, default=1e-4)
    p.add_argument("--selector_neg_per_class", type=int, default=3)
    p.add_argument("--selector_score_alpha", type=float, default=1.25)

    p.add_argument("--dual_epochs", type=int, default=60)
    p.add_argument("--dual_patience", type=int, default=12)
    p.add_argument("--dual_lr", type=float, default=1.5e-4)
    p.add_argument("--dual_weight_decay", type=float, default=1e-4)

    # Set-loss controls
    p.add_argument("--unmatched_penalty", type=float, default=0.0)
    p.add_argument("--pred_exist_threshold", type=float, default=0.08)

    p.add_argument("--save_fusion_scores", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    set_seed(int(args.seed))

    device = torch.device(args.device)
    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    top_m_train = int(args.proposer_k_train) if int(args.proposer_k_train) > 0 else int(args.proposer_top_m_train)
    top_m_infer = int(args.k_infer) if int(args.k_infer) > 0 else int(args.top_m_infer)

    print("=" * 72)
    print("Model-33 Deterministic Feasibility + DualView Pipeline")
    print(f"Run: {save_root}")
    print(
        f"Shortlist config: train K0={int(args.proposer_k0_train)} -> M={top_m_train}; "
        f"infer K0={int(args.k0_infer)} -> M={top_m_infer}; "
        f"search(chunk={int(args.search_chunk_k0)}, rounds={int(args.search_max_rounds)}, targetK={int(args.search_target_k)}); "
        f"refine(train,infer)=({int(args.proposer_refine_steps)},{int(args.infer_refine_steps)})"
    )
    print("=" * 72)

    # Load offline jets.
    train_files = base._parse_h5_path_arg(str(args.train_path))
    max_needed = int(args.offset_jets + args.n_train_jets)
    all_const, all_labels, all_train_w = base.load_raw_constituents_labels_weights_from_h5(
        files=train_files,
        max_jets=max_needed,
        max_constits=int(args.max_constits),
        use_train_weights=bool(args.use_train_weights),
    )
    if all_const.shape[0] < max_needed:
        raise RuntimeError(f"Requested {max_needed} jets but found {all_const.shape[0]}.")

    const_raw = all_const[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)
    train_w = all_train_w[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.float32)

    raw_mask = const_raw[:, :, 0] > 0.0
    cfg = base._deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT (deterministic keyed D_hard)...")
    jet_keys = (np.arange(len(const_off), dtype=np.int64) + int(args.offset_jets)).astype(np.int64)
    const_hlt, mask_hlt, hlt_stats = _apply_hlt_effects_deterministic_keyed(
        const=const_off,
        mask=masks_off,
        cfg=cfg,
        jet_keys=jet_keys,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    print(
        "HLT stats: "
        f"avg_offline={hlt_stats.get('avg_offline_per_jet', float('nan')):.2f}, "
        f"avg_hlt={hlt_stats.get('avg_hlt_per_jet', float('nan')):.2f}, "
        f"merged={hlt_stats.get('n_merged_pairs', 0)}, eff_lost={hlt_stats.get('n_lost_eff', 0)}"
    )

    print("Computing features for teacher/baseline...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(const_hlt, mask_hlt)

    idx_all = np.arange(len(labels))
    total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > len(idx_all):
        raise ValueError(f"Split sum {total_need} exceeds dataset size {len(idx_all)}")
    if total_need < len(idx_all):
        idx_use, _ = train_test_split(
            idx_all,
            train_size=total_need,
            random_state=int(args.seed),
            stratify=labels[idx_all],
        )
    else:
        idx_use = idx_all

    train_idx, rem_idx = train_test_split(
        idx_use,
        train_size=int(args.n_train_split),
        random_state=int(args.seed),
        stratify=labels[idx_use],
    )
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=int(args.n_val_split),
        test_size=int(args.n_test_split),
        random_state=int(args.seed),
        stratify=labels[rem_idx],
    )

    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, mask_hlt, means, stds)

    # STEP 1: teacher and baseline
    print("\n" + "=" * 72)
    print("STEP 1: Teacher + HLT baseline")
    print("=" * 72)

    cls_cfg = {
        "epochs": int(args.cls_epochs),
        "patience": int(args.cls_patience),
        "lr": float(args.cls_lr),
        "weight_decay": float(args.cls_weight_decay),
        "warmup_epochs": int(args.cls_warmup_epochs),
    }

    sw_train = train_w[train_idx] if bool(args.use_train_weights) else np.ones((len(train_idx),), dtype=np.float32)
    sw_val = train_w[val_idx] if bool(args.use_train_weights) else np.ones((len(val_idx),), dtype=np.float32)
    sw_test = train_w[test_idx] if bool(args.use_train_weights) else np.ones((len(test_idx),), dtype=np.float32)

    ds_tr_off = base.WeightedJetDataset(feat_off_std[train_idx], masks_off[train_idx], labels[train_idx], sw_train)
    ds_va_off = base.WeightedJetDataset(feat_off_std[val_idx], masks_off[val_idx], labels[val_idx], sw_val)
    ds_te_off = base.WeightedJetDataset(feat_off_std[test_idx], masks_off[test_idx], labels[test_idx], sw_test)

    dl_tr_off = DataLoader(ds_tr_off, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    dl_va_off = DataLoader(ds_va_off, batch_size=int(args.batch_size), shuffle=False)
    dl_te_off = DataLoader(ds_te_off, batch_size=int(args.batch_size), shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = base.train_single_view_classifier_auc(
        model=teacher,
        train_loader=dl_tr_off,
        val_loader=dl_va_off,
        device=device,
        train_cfg=cls_cfg,
        name="Teacher",
    )
    teacher_auc_test, teacher_p_test, teacher_y_test, teacher_w_test = base._eval_classifier_with_optional_weights(teacher, dl_te_off, device)

    ds_tr_hlt = base.WeightedJetDataset(feat_hlt_std[train_idx], mask_hlt[train_idx], labels[train_idx], sw_train)
    ds_va_hlt = base.WeightedJetDataset(feat_hlt_std[val_idx], mask_hlt[val_idx], labels[val_idx], sw_val)
    ds_te_hlt = base.WeightedJetDataset(feat_hlt_std[test_idx], mask_hlt[test_idx], labels[test_idx], sw_test)

    dl_tr_hlt = DataLoader(ds_tr_hlt, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    dl_va_hlt = DataLoader(ds_va_hlt, batch_size=int(args.batch_size), shuffle=False)
    dl_te_hlt = DataLoader(ds_te_hlt, batch_size=int(args.batch_size), shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = base.train_single_view_classifier_auc(
        model=baseline,
        train_loader=dl_tr_hlt,
        val_loader=dl_va_hlt,
        device=device,
        train_cfg=cls_cfg,
        name="Baseline",
    )
    baseline_auc_test, baseline_p_test, baseline_y_test, baseline_w_test = base._eval_classifier_with_optional_weights(baseline, dl_te_hlt, device)

    print(
        f"Teacher test AUC={teacher_auc_test:.4f} | Baseline test AUC={baseline_auc_test:.4f}"
    )

    # Teacher confidence weights for class-sensitive stages.
    p_teacher_train, y_teacher_train = _predict_probs(
        teacher, feat_off_std[train_idx], masks_off[train_idx], labels[train_idx],
        batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device
    )
    p_teacher_val, y_teacher_val = _predict_probs(
        teacher, feat_off_std[val_idx], masks_off[val_idx], labels[val_idx],
        batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device
    )

    if not np.array_equal(y_teacher_train.astype(np.int64), labels[train_idx].astype(np.int64)):
        raise RuntimeError("Teacher prediction labels mismatch on train split")
    if not np.array_equal(y_teacher_val.astype(np.int64), labels[val_idx].astype(np.int64)):
        raise RuntimeError("Teacher prediction labels mismatch on val split")

    w_conf_train = _build_teacher_conf_weights(
        teacher_probs=p_teacher_train,
        labels=labels[train_idx],
        weight_floor=float(args.conf_weight_floor),
        hard_correct_gate=(not bool(args.disable_conf_hard_correct_gate)),
        normalize_mean_one=(not bool(args.disable_conf_mean_normalize)),
    )
    w_conf_val = _build_teacher_conf_weights(
        teacher_probs=p_teacher_val,
        labels=labels[val_idx],
        weight_floor=float(args.conf_weight_floor),
        hard_correct_gate=(not bool(args.disable_conf_hard_correct_gate)),
        normalize_mean_one=(not bool(args.disable_conf_mean_normalize)),
    )
    print(
        f"Confidence weights: train_mean={w_conf_train.mean():.4f}, val_mean={w_conf_val.mean():.4f}, "
        f"train_p95={np.quantile(w_conf_train, 0.95):.3f}"
    )

    # STEP 2: Offline latent AE + class prior statistics.
    print("\n" + "=" * 72)
    print("STEP 2: Offline latent AE + class priors")
    print("=" * 72)

    ds_prior_tr = OfflineStageDataset(
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        labels=labels[train_idx],
        sample_weight=w_conf_train,
    )
    ds_prior_va = OfflineStageDataset(
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        labels=labels[val_idx],
        sample_weight=w_conf_val,
    )
    dl_prior_tr = DataLoader(
        ds_prior_tr,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_prior_va = DataLoader(
        ds_prior_va,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    ae = OfflineLatentAE(
        latent_dim=int(args.latent_dim),
        slots=int(args.max_constits),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)

    ae, prior_train_metrics = _train_offline_ae(
        model=ae,
        train_loader=dl_prior_tr,
        val_loader=dl_prior_va,
        device=device,
        epochs=int(args.prior_epochs),
        lr=float(args.prior_lr),
        weight_decay=float(args.prior_weight_decay),
        patience=int(args.prior_patience),
        loss_w_count=float(args.prior_loss_w_count),
        loss_w_lat_reg=float(args.prior_loss_w_lat_reg),
        unmatched_penalty=float(args.unmatched_penalty),
    )

    prior_stats = _compute_prior_stats(
        model=ae,
        loader=dl_prior_tr,
        device=device,
        latent_dim=int(args.latent_dim),
    )

    # STEP 3: Offline->HLT degrader.
    print("\n" + "=" * 72)
    print("STEP 3: Offline->HLT degrader")
    print("=" * 72)

    ds_deg_tr = PairStageDataset(
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        labels=labels[train_idx],
        sample_weight=None,
    )
    ds_deg_va = PairStageDataset(
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        labels=labels[val_idx],
        sample_weight=None,
    )
    dl_deg_tr = DataLoader(
        ds_deg_tr,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_deg_va = DataLoader(
        ds_deg_va,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    degrader = OfflineToHLTDegrader(
        latent_dim=int(args.latent_dim),
        slots=int(args.max_constits),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)

    degrader, degrader_metrics = _train_degrader(
        model=degrader,
        train_loader=dl_deg_tr,
        val_loader=dl_deg_va,
        device=device,
        epochs=int(args.degrader_epochs),
        lr=float(args.degrader_lr),
        weight_decay=float(args.degrader_weight_decay),
        patience=int(args.degrader_patience),
        loss_w_count=float(args.degrader_loss_w_count),
        unmatched_penalty=float(args.unmatched_penalty),
    )

    # STEP 4: HLT->latent proposer with best-of-K objective.
    print("\n" + "=" * 72)
    print("STEP 4: HLT->latent proposer (best-of-K)")
    print("=" * 72)

    ds_prop_tr = PairStageDataset(
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        labels=labels[train_idx],
        sample_weight=w_conf_train,
    )
    ds_prop_va = PairStageDataset(
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        labels=labels[val_idx],
        sample_weight=w_conf_val,
    )

    dl_prop_tr = DataLoader(
        ds_prop_tr,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_prop_va = DataLoader(
        ds_prop_va,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    proposer = HLTLatentProposer(
        latent_dim=int(args.latent_dim),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)

    proposer, proposer_metrics = _train_proposer(
        proposer=proposer,
        ae=ae,
        degrader=degrader,
        prior=prior_stats,
        train_loader=dl_prop_tr,
        val_loader=dl_prop_va,
        device=device,
        epochs=int(args.proposer_epochs),
        lr=float(args.proposer_lr),
        weight_decay=float(args.proposer_weight_decay),
        patience=int(args.proposer_patience),
        k0_samples=int(args.proposer_k0_train),
        top_m_samples=int(top_m_train),
        wrong_k_samples=int(args.proposer_k_wrong),
        softmin_tau=float(args.proposer_softmin_tau),
        diversity_tau=float(args.proposer_diversity_tau),
        prescore_w_prior=float(args.proposer_prescore_w_prior),
        prescore_w_q=float(args.proposer_prescore_w_q),
        lambda_count=float(args.proposer_lambda_count),
        lambda_div=float(args.proposer_lambda_div),
        lambda_margin=float(args.proposer_lambda_margin),
        margin=float(args.proposer_margin),
        lambda_kl=float(args.proposer_lambda_kl),
        lambda_hlt_cons=float(args.proposer_lambda_hlt_cons),
        refine_steps=int(args.proposer_refine_steps),
        refine_step_size=float(args.proposer_refine_lr),
        refine_max_step_norm=float(args.proposer_refine_max_step_norm),
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
    )

    # STEP 5: Deterministic chunked feasible search.
    print("\n" + "=" * 72)
    print("STEP 5: Deterministic Chunked Candidate Search (D_hard acceptance)")
    print("=" * 72)

    p_base_train, _ = _predict_probs(
        baseline, feat_hlt_std[train_idx], mask_hlt[train_idx], labels[train_idx],
        batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device
    )
    p_base_val, _ = _predict_probs(
        baseline, feat_hlt_std[val_idx], mask_hlt[val_idx], labels[val_idx],
        batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device
    )
    p_base_test, _ = _predict_probs(
        baseline, feat_hlt_std[test_idx], mask_hlt[test_idx], labels[test_idx],
        batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device
    )

    pools_train = _chunked_search_candidates_split(
        proposer=proposer,
        ae=ae,
        degrader=degrader,
        prior=prior_stats,
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        jet_keys=jet_keys[train_idx],
        cfg=cfg,
        base_seed=int(args.seed + args.dhard_seed_offset),
        device=device,
        batch_size=int(args.search_batch_size),
        target_k=int(args.search_target_k),
        chunk_k0=int(args.search_chunk_k0),
        shortlist_m=int(args.search_shortlist_m),
        max_rounds=int(args.search_max_rounds),
        keep_per_round=int(args.search_keep_per_round),
        max_pool_size=int(args.search_max_pool_size),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
        prescore_w_prior=float(args.proposer_prescore_w_prior),
        prescore_w_q=float(args.proposer_prescore_w_q),
        refine_steps=int(args.infer_refine_steps),
        refine_step_size=float(args.infer_refine_lr),
        refine_max_step_norm=float(args.infer_refine_max_step_norm),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
    )
    pools_val = _chunked_search_candidates_split(
        proposer=proposer,
        ae=ae,
        degrader=degrader,
        prior=prior_stats,
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        jet_keys=jet_keys[val_idx],
        cfg=cfg,
        base_seed=int(args.seed + args.dhard_seed_offset),
        device=device,
        batch_size=int(args.search_batch_size),
        target_k=int(args.search_target_k),
        chunk_k0=int(args.search_chunk_k0),
        shortlist_m=int(args.search_shortlist_m),
        max_rounds=int(args.search_max_rounds),
        keep_per_round=int(args.search_keep_per_round),
        max_pool_size=int(args.search_max_pool_size),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
        prescore_w_prior=float(args.proposer_prescore_w_prior),
        prescore_w_q=float(args.proposer_prescore_w_q),
        refine_steps=int(args.infer_refine_steps),
        refine_step_size=float(args.infer_refine_lr),
        refine_max_step_norm=float(args.infer_refine_max_step_norm),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
    )
    pools_test = _chunked_search_candidates_split(
        proposer=proposer,
        ae=ae,
        degrader=degrader,
        prior=prior_stats,
        const_hlt=const_hlt[test_idx],
        mask_hlt=mask_hlt[test_idx],
        jet_keys=jet_keys[test_idx],
        cfg=cfg,
        base_seed=int(args.seed + args.dhard_seed_offset),
        device=device,
        batch_size=int(args.search_batch_size),
        target_k=int(args.search_target_k),
        chunk_k0=int(args.search_chunk_k0),
        shortlist_m=int(args.search_shortlist_m),
        max_rounds=int(args.search_max_rounds),
        keep_per_round=int(args.search_keep_per_round),
        max_pool_size=int(args.search_max_pool_size),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
        prescore_w_prior=float(args.proposer_prescore_w_prior),
        prescore_w_q=float(args.proposer_prescore_w_q),
        refine_steps=int(args.infer_refine_steps),
        refine_step_size=float(args.infer_refine_lr),
        refine_max_step_norm=float(args.infer_refine_max_step_norm),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
    )

    print(
        "Search stats: "
        f"train(feasC0={pools_train['stats']['class0_mean_feasible']:.2f}, feasC1={pools_train['stats']['class1_mean_feasible']:.2f}) "
        f"val(feasC0={pools_val['stats']['class0_mean_feasible']:.2f}, feasC1={pools_val['stats']['class1_mean_feasible']:.2f})"
    )

    # STEP 6: Realism selector (real offline vs generated feasible candidates).
    print("\n" + "=" * 72)
    print("STEP 6: Train Realism Selector")
    print("=" * 72)

    sel_tr = _build_selector_arrays(
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        pools=pools_train,
        neg_per_class=int(args.selector_neg_per_class),
    )
    sel_va = _build_selector_arrays(
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        pools=pools_val,
        neg_per_class=int(args.selector_neg_per_class),
    )

    ds_sel_tr = SelectorDataset(**sel_tr)
    ds_sel_va = SelectorDataset(**sel_va)
    dl_sel_tr = DataLoader(ds_sel_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_sel_va = DataLoader(ds_sel_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    selector = CandidateRealismSelector(
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers // 2)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    selector, selector_metrics = _train_selector(
        model=selector,
        train_loader=dl_sel_tr,
        val_loader=dl_sel_va,
        device=device,
        epochs=int(args.selector_epochs),
        lr=float(args.selector_lr),
        weight_decay=float(args.selector_weight_decay),
        patience=int(args.selector_patience),
    )

    # Selector scores on candidate pools.
    sel_bg_train = _score_selector_candidates(
        selector, const_hlt[train_idx], mask_hlt[train_idx],
        pools_train["class0"]["const"], pools_train["class0"]["mask"], pools_train["class0"]["res_total"],
        cand_class_val=0, batch_size=int(args.batch_size), device=device
    )
    sel_tp_train = _score_selector_candidates(
        selector, const_hlt[train_idx], mask_hlt[train_idx],
        pools_train["class1"]["const"], pools_train["class1"]["mask"], pools_train["class1"]["res_total"],
        cand_class_val=1, batch_size=int(args.batch_size), device=device
    )
    sel_bg_val = _score_selector_candidates(
        selector, const_hlt[val_idx], mask_hlt[val_idx],
        pools_val["class0"]["const"], pools_val["class0"]["mask"], pools_val["class0"]["res_total"],
        cand_class_val=0, batch_size=int(args.batch_size), device=device
    )
    sel_tp_val = _score_selector_candidates(
        selector, const_hlt[val_idx], mask_hlt[val_idx],
        pools_val["class1"]["const"], pools_val["class1"]["mask"], pools_val["class1"]["res_total"],
        cand_class_val=1, batch_size=int(args.batch_size), device=device
    )
    sel_bg_test = _score_selector_candidates(
        selector, const_hlt[test_idx], mask_hlt[test_idx],
        pools_test["class0"]["const"], pools_test["class0"]["mask"], pools_test["class0"]["res_total"],
        cand_class_val=0, batch_size=int(args.batch_size), device=device
    )
    sel_tp_test = _score_selector_candidates(
        selector, const_hlt[test_idx], mask_hlt[test_idx],
        pools_test["class1"]["const"], pools_test["class1"]["mask"], pools_test["class1"]["res_total"],
        cand_class_val=1, batch_size=int(args.batch_size), device=device
    )

    # Build dualview candidate features.
    dv_tr = _build_dualview_features(
        pools=pools_train,
        sel_score_bg=sel_bg_train,
        sel_score_top=sel_tp_train,
        baseline_prob=p_base_train,
        score_alpha=float(args.selector_score_alpha),
    )
    dv_va = _build_dualview_features(
        pools=pools_val,
        sel_score_bg=sel_bg_val,
        sel_score_top=sel_tp_val,
        baseline_prob=p_base_val,
        score_alpha=float(args.selector_score_alpha),
    )
    dv_te = _build_dualview_features(
        pools=pools_test,
        sel_score_bg=sel_bg_test,
        sel_score_top=sel_tp_test,
        baseline_prob=p_base_test,
        score_alpha=float(args.selector_score_alpha),
    )

    # STEP 7: Final dualview classifiers (no-gate and gated).
    print("\n" + "=" * 72)
    print("STEP 7: Final DualView Classifiers (NoGate + Gated)")
    print("=" * 72)

    ds_dv_tr = DualViewCandidateDataset(
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        cand_top_const=dv_tr["cand_top_const"],
        cand_top_mask=dv_tr["cand_top_mask"],
        cand_bg_const=dv_tr["cand_bg_const"],
        cand_bg_mask=dv_tr["cand_bg_mask"],
        cand_feat=dv_tr["cand_feat"],
        labels=labels[train_idx],
        sample_weight=sw_train,
    )
    ds_dv_va = DualViewCandidateDataset(
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        cand_top_const=dv_va["cand_top_const"],
        cand_top_mask=dv_va["cand_top_mask"],
        cand_bg_const=dv_va["cand_bg_const"],
        cand_bg_mask=dv_va["cand_bg_mask"],
        cand_feat=dv_va["cand_feat"],
        labels=labels[val_idx],
        sample_weight=sw_val,
    )
    ds_dv_te = DualViewCandidateDataset(
        const_hlt=const_hlt[test_idx],
        mask_hlt=mask_hlt[test_idx],
        cand_top_const=dv_te["cand_top_const"],
        cand_top_mask=dv_te["cand_top_mask"],
        cand_bg_const=dv_te["cand_bg_const"],
        cand_bg_mask=dv_te["cand_bg_mask"],
        cand_feat=dv_te["cand_feat"],
        labels=labels[test_idx],
        sample_weight=sw_test,
    )

    dl_dv_tr = DataLoader(ds_dv_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_dv_va = DataLoader(ds_dv_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_dv_te = DataLoader(ds_dv_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    dv_nogate = DualViewNoGateClassifier(
        cand_feat_dim=int(dv_tr["cand_feat"].shape[1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers // 2)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    dv_nogate, dv_nogate_metrics = _train_dualview_model(
        model=dv_nogate,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="DualViewNoGate",
    )

    dv_gated = DualViewGatedClassifier(
        cand_feat_dim=int(dv_tr["cand_feat"].shape[1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers // 2)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    dv_gated, dv_gated_metrics = _train_dualview_model(
        model=dv_gated,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="DualViewGated",
    )

    auc_nog, fpr50_nog, pred_nog, lab_final, w_final = _eval_dualview_model(dv_nogate, dl_dv_te, device)
    auc_gat, fpr50_gat, pred_gat, _lab2, _w2 = _eval_dualview_model(dv_gated, dl_dv_te, device)

    fpr_t, tpr_t, _ = roc_curve(teacher_y_test, teacher_p_test, sample_weight=teacher_w_test)
    fpr_b, tpr_b, _ = roc_curve(baseline_y_test, baseline_p_test, sample_weight=baseline_w_test)
    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.50)

    print("\n" + "=" * 72)
    print("FINAL TEST")
    print("=" * 72)
    print(
        f"Teacher AUC={teacher_auc_test:.4f} FPR50={fpr50_teacher:.6f} | "
        f"HLT baseline AUC={baseline_auc_test:.4f} FPR50={fpr50_baseline:.6f} | "
        f"m33 NoGate AUC={auc_nog:.4f} FPR50={fpr50_nog:.6f} | "
        f"m33 Gated AUC={auc_gat:.4f} FPR50={fpr50_gat:.6f}"
    )

    # Save artifacts.
    torch.save({"model": teacher.state_dict(), "auc_test": float(teacher_auc_test)}, save_root / "teacher.pt")
    torch.save({"model": baseline.state_dict(), "auc_test": float(baseline_auc_test)}, save_root / "baseline_hlt.pt")
    torch.save({"model": ae.state_dict(), "metrics": prior_train_metrics}, save_root / "offline_prior_ae.pt")
    torch.save(
        {
            "prior_mean": prior_stats.mean.detach().cpu(),
            "prior_logvar": prior_stats.logvar.detach().cpu(),
        },
        save_root / "offline_prior_stats.pt",
    )
    torch.save({"model": degrader.state_dict(), "metrics": degrader_metrics}, save_root / "degrader.pt")
    torch.save({"model": proposer.state_dict(), "metrics": proposer_metrics}, save_root / "proposer.pt")
    torch.save({"model": selector.state_dict(), "metrics": selector_metrics}, save_root / "selector.pt")
    torch.save({"model": dv_nogate.state_dict(), "metrics": dv_nogate_metrics}, save_root / "dualview_nogate.pt")
    torch.save({"model": dv_gated.state_dict(), "metrics": dv_gated_metrics}, save_root / "dualview_gated.pt")

    np.savez_compressed(
        save_root / "m33_test_scores.npz",
        labels_test=lab_final.astype(np.float32),
        preds_m33_nogate=pred_nog.astype(np.float32),
        preds_m33_gated=pred_gat.astype(np.float32),
        preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
        preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
        sample_weight=np.asarray(w_final, dtype=np.float32),
        auc_teacher=float(teacher_auc_test),
        auc_hlt=float(baseline_auc_test),
        auc_m33_nogate=float(auc_nog),
        auc_m33_gated=float(auc_gat),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_hlt=float(fpr50_baseline),
        fpr50_m33_nogate=float(fpr50_nog),
        fpr50_m33_gated=float(fpr50_gat),
    )

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "fusion_scores_test.npz",
            labels_test=lab_final.astype(np.float32),
            preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
            preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
            preds_m33_nogate=np.asarray(pred_nog, dtype=np.float32),
            preds_m33_gated=np.asarray(pred_gat, dtype=np.float32),
            sample_weight=np.asarray(w_final, dtype=np.float32),
        )

    report = {
        "model": "m33_detfeas_dualview",
        "seed": int(args.seed),
        "n_train_jets": int(args.n_train_jets),
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "teacher": {
            "auc_test": float(teacher_auc_test),
            "fpr50_test": float(fpr50_teacher),
        },
        "hlt_baseline": {
            "auc_test": float(baseline_auc_test),
            "fpr50_test": float(fpr50_baseline),
        },
        "m33_nogate": {
            "auc_test": float(auc_nog),
            "fpr50_test": float(fpr50_nog),
            "metrics": dv_nogate_metrics,
        },
        "m33_gated": {
            "auc_test": float(auc_gat),
            "fpr50_test": float(fpr50_gat),
            "metrics": dv_gated_metrics,
        },
        "prior_train_metrics": prior_train_metrics,
        "degrader_metrics": degrader_metrics,
        "proposer_metrics": proposer_metrics,
        "selector_metrics": selector_metrics,
        "candidate_search": {
            "train": pools_train["stats"],
            "val": pools_val["stats"],
            "test": pools_test["stats"],
        },
        "teacher_conf_weights": {
            "train_mean": float(w_conf_train.mean()),
            "train_p95": float(np.quantile(w_conf_train, 0.95)),
            "val_mean": float(w_conf_val.mean()),
            "floor": float(args.conf_weight_floor),
            "hard_correct_gate": bool(not args.disable_conf_hard_correct_gate),
            "mean_normalized": bool(not args.disable_conf_mean_normalize),
        },
    }
    with open(save_root / "m33_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
        hlt_avg_count=np.array([float(mask_hlt.mean())], dtype=np.float32),
    )

    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()

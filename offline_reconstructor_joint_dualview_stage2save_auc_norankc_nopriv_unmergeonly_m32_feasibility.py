#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m32: Class-conditional feasibility pipeline for HLT top tagging.

Design goals:
1) Learn an offline latent space + decoder (class-conditioned prior support).
2) Learn offline->HLT degradation to score feasibility under HLT corruption.
3) Learn HLT->latent proposer with best-of-K objective (at least one strong candidate).
4) Build class feasibility signals from degraded candidates.
5) Train final HLT classifier using full HLT view + feasibility signals.

This script is intentionally separate from m29/m30/m31 and does not modify them.
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m32 feasibility-driven HLT top tagging")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model32_feasibility")
    p.add_argument("--run_name", type=str, default="model32_k6_feasibility_150k75k150k_seed0")

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
    p.add_argument("--proposer_lambda_hlt_cons", type=float, default=0.35)
    p.add_argument("--proposer_refine_steps", type=int, default=0)
    p.add_argument("--proposer_refine_lr", type=float, default=0.05)
    p.add_argument("--proposer_refine_max_step_norm", type=float, default=0.20)

    # Stage 4/5: feasibility and final classifier
    p.add_argument("--k0_infer", type=int, default=64)
    p.add_argument("--top_m_infer", type=int, default=8)
    p.add_argument("--k_infer", type=int, default=-1, help="Deprecated alias for --top_m_infer.")
    p.add_argument("--infer_refine_steps", type=int, default=0)
    p.add_argument("--infer_refine_lr", type=float, default=0.05)
    p.add_argument("--infer_refine_max_step_norm", type=float, default=0.20)
    p.add_argument("--final_epochs", type=int, default=70)
    p.add_argument("--final_patience", type=int, default=12)
    p.add_argument("--final_lr", type=float, default=1.5e-4)
    p.add_argument("--final_weight_decay", type=float, default=1e-4)

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
    print("Model-32 Feasibility Pipeline")
    print(f"Run: {save_root}")
    print(
        f"Shortlist config: train K0={int(args.proposer_k0_train)} -> M={top_m_train}; "
        f"infer K0={int(args.k0_infer)} -> M={top_m_infer}; "
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

    print("Generating pseudo-HLT...")
    const_hlt, mask_hlt, hlt_stats, _budget = reco_base.apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
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

    # STEP 5: Build feasibility features and train final classifier.
    print("\n" + "=" * 72)
    print("STEP 5: Feasibility features + final classifier")
    print("=" * 72)

    p_base_train, _ = _predict_probs(
        baseline,
        feat_hlt_std[train_idx],
        mask_hlt[train_idx],
        labels[train_idx],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    p_base_val, _ = _predict_probs(
        baseline,
        feat_hlt_std[val_idx],
        mask_hlt[val_idx],
        labels[val_idx],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    p_base_test, _ = _predict_probs(
        baseline,
        feat_hlt_std[test_idx],
        mask_hlt[test_idx],
        labels[test_idx],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )

    feas_train = _build_feasibility_features(
        proposer=proposer,
        ae=ae,
        degrader=degrader,
        prior=prior_stats,
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        baseline_prob=p_base_train,
        device=device,
        batch_size=int(args.batch_size),
        k0_infer=int(args.k0_infer),
        top_m_infer=int(top_m_infer),
        prescore_w_prior=float(args.proposer_prescore_w_prior),
        prescore_w_q=float(args.proposer_prescore_w_q),
        latent_refine_steps=int(args.infer_refine_steps),
        latent_refine_lr=float(args.infer_refine_lr),
        latent_refine_max_step_norm=float(args.infer_refine_max_step_norm),
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
    )
    feas_val = _build_feasibility_features(
        proposer=proposer,
        ae=ae,
        degrader=degrader,
        prior=prior_stats,
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        baseline_prob=p_base_val,
        device=device,
        batch_size=int(args.batch_size),
        k0_infer=int(args.k0_infer),
        top_m_infer=int(top_m_infer),
        prescore_w_prior=float(args.proposer_prescore_w_prior),
        prescore_w_q=float(args.proposer_prescore_w_q),
        latent_refine_steps=int(args.infer_refine_steps),
        latent_refine_lr=float(args.infer_refine_lr),
        latent_refine_max_step_norm=float(args.infer_refine_max_step_norm),
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
    )
    feas_test = _build_feasibility_features(
        proposer=proposer,
        ae=ae,
        degrader=degrader,
        prior=prior_stats,
        const_hlt=const_hlt[test_idx],
        mask_hlt=mask_hlt[test_idx],
        baseline_prob=p_base_test,
        device=device,
        batch_size=int(args.batch_size),
        k0_infer=int(args.k0_infer),
        top_m_infer=int(top_m_infer),
        prescore_w_prior=float(args.proposer_prescore_w_prior),
        prescore_w_q=float(args.proposer_prescore_w_q),
        latent_refine_steps=int(args.infer_refine_steps),
        latent_refine_lr=float(args.infer_refine_lr),
        latent_refine_max_step_norm=float(args.infer_refine_max_step_norm),
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
    )

    ds_final_tr = FinalStageDataset(
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        feas_feat=feas_train,
        labels=labels[train_idx],
        sample_weight=sw_train,
    )
    ds_final_va = FinalStageDataset(
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        feas_feat=feas_val,
        labels=labels[val_idx],
        sample_weight=sw_val,
    )
    ds_final_te = FinalStageDataset(
        const_hlt=const_hlt[test_idx],
        mask_hlt=mask_hlt[test_idx],
        feas_feat=feas_test,
        labels=labels[test_idx],
        sample_weight=sw_test,
    )

    dl_final_tr = DataLoader(
        ds_final_tr,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_final_va = DataLoader(
        ds_final_va,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_final_te = DataLoader(
        ds_final_te,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    final_model = FinalFeasibilityClassifier(
        feas_dim=int(feas_train.shape[1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers // 2)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)

    final_model, final_metrics = _train_final_classifier(
        model=final_model,
        train_loader=dl_final_tr,
        val_loader=dl_final_va,
        device=device,
        epochs=int(args.final_epochs),
        lr=float(args.final_lr),
        weight_decay=float(args.final_weight_decay),
        patience=int(args.final_patience),
    )

    auc_final, fpr50_final, pred_final, lab_final, w_final = _eval_final_classifier(final_model, dl_final_te, device)

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
        f"m32 feasibility AUC={auc_final:.4f} FPR50={fpr50_final:.6f}"
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
    torch.save({"model": final_model.state_dict(), "metrics": final_metrics}, save_root / "final_feas_classifier.pt")

    np.savez_compressed(
        save_root / "m32_test_scores.npz",
        labels_test=lab_final.astype(np.float32),
        preds_m32=pred_final.astype(np.float32),
        preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
        preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
        sample_weight=np.asarray(w_final, dtype=np.float32),
        auc_teacher=float(teacher_auc_test),
        auc_hlt=float(baseline_auc_test),
        auc_m32=float(auc_final),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_hlt=float(fpr50_baseline),
        fpr50_m32=float(fpr50_final),
    )

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "fusion_scores_test.npz",
            labels_test=lab_final.astype(np.float32),
            preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
            preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
            preds_m32=np.asarray(pred_final, dtype=np.float32),
            sample_weight=np.asarray(w_final, dtype=np.float32),
        )

    report = {
        "model": "m32_feasibility",
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
        "m32": {
            "auc_test": float(auc_final),
            "fpr50_test": float(fpr50_final),
            "final_metrics": final_metrics,
        },
        "prior_train_metrics": prior_train_metrics,
        "degrader_metrics": degrader_metrics,
        "proposer_metrics": proposer_metrics,
        "teacher_conf_weights": {
            "train_mean": float(w_conf_train.mean()),
            "train_p95": float(np.quantile(w_conf_train, 0.95)),
            "val_mean": float(w_conf_val.mean()),
            "floor": float(args.conf_weight_floor),
            "hard_correct_gate": bool(not args.disable_conf_hard_correct_gate),
            "mean_normalized": bool(not args.disable_conf_mean_normalize),
        },
    }
    with open(save_root / "m32_report.json", "w", encoding="utf-8") as f:
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

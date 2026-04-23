#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m38: Seeded m28 completion + deterministic D_hard residual selection + multi-candidate dualview.

Pipeline:
1) Load Offline jets and build deterministic pseudo-HLT.
2) Train Teacher (Offline) and HLT baseline classifiers.
3) Train carryover predictor on HLT tokens (token-level carry likelihood).
4) Train m28-style HLT->Offline completer (set-level objective, multi-hypothesis).
5) For each jet: build diverse seed prefixes, force-prefix decode K candidates,
   run deterministic D_hard on each candidate, score residuals, keep best M.
6) Train final multi-candidate dualview heads (NoGate/Gated).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_detfeas_dualview as m33
import offline_reconstructor_joint_dualview_seq2seq_nexttoken_m28_sinkset_noar as m28
from unmerge_correct_hlt import ParticleTransformer, compute_features, get_stats, standardize


# -----------------------------------------------------------------------------
# Data / utils
# -----------------------------------------------------------------------------


def _const_to_token5_np(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
    tok[~mask] = 0.0
    return tok


def _build_candidate_token12_np(
    off_const: np.ndarray,
    off_mask: np.ndarray,
    hlt_const: np.ndarray,
    hlt_mask: np.ndarray,
) -> np.ndarray:
    off5 = _const_to_token5_np(off_const, off_mask)
    hlt5 = _const_to_token5_np(hlt_const, hlt_mask)
    om = off_mask.astype(np.float32)[..., None]
    hm = hlt_mask.astype(np.float32)[..., None]
    tok = np.concatenate([off5, hlt5, om, hm], axis=-1).astype(np.float32)
    valid = (off_mask | hlt_mask)
    tok[~valid] = 0.0
    return tok


def _fpr50(y_true: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, p, sample_weight=w)
    return float(m33.fpr_at_target_tpr(fpr, tpr, 0.50))


class RecoCarryDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray, carry_tgt: np.ndarray):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.carry_tgt = torch.tensor(carry_tgt, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "carry_tgt": self.carry_tgt[i],
        }


class FeatMaskDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat": self.feat_hlt[i],
            "mask": self.mask_hlt[i],
        }


class DualViewM38Dataset(Dataset):
    """Numpy-backed dataset to avoid duplicating large candidate tensors in memory."""

    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        cand_tokens: np.ndarray,
        cand_masks: np.ndarray,
        cand_meta: np.ndarray,
        summary_feat: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray,
    ):
        self.feat_hlt = feat_hlt.astype(np.float32, copy=False)
        self.mask_hlt = mask_hlt.astype(bool, copy=False)
        self.cand_tokens = cand_tokens.astype(np.float32, copy=False)
        self.cand_masks = cand_masks.astype(bool, copy=False)
        self.cand_meta = cand_meta.astype(np.float32, copy=False)
        self.summary_feat = summary_feat.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.sample_weight = sample_weight.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": torch.from_numpy(self.feat_hlt[i]),
            "mask_hlt": torch.from_numpy(self.mask_hlt[i]),
            "cand_tokens": torch.from_numpy(self.cand_tokens[i]),
            "cand_masks": torch.from_numpy(self.cand_masks[i]),
            "cand_meta": torch.from_numpy(self.cand_meta[i]),
            "summary_feat": torch.from_numpy(self.summary_feat[i]),
            "label": torch.tensor(self.labels[i], dtype=torch.float32),
            "sample_weight": torch.tensor(self.sample_weight[i], dtype=torch.float32),
        }


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class CarryoverTokenPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 384,
        dropout: float = 0.10,
        max_tokens: int = 100,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(embed_dim)),
            nn.LayerNorm(int(embed_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.pos = nn.Parameter(torch.zeros(1, int(max_tokens), int(embed_dim)))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.head = nn.Linear(int(embed_dim), 1)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, feat_hlt: torch.Tensor, mask_hlt: torch.Tensor) -> torch.Tensor:
        b, l, _ = feat_hlt.shape
        x = self.in_proj(feat_hlt) + self.pos[:, :l, :]
        h = self.encoder(x, src_key_padding_mask=~mask_hlt)
        return self.head(h).squeeze(-1)


class MultiCandidateM38NoGate(nn.Module):
    def __init__(
        self,
        cand_meta_dim: int,
        summary_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hlt_encoder = ParticleTransformer(
            input_dim=7,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.cand_encoder = ParticleTransformer(
            input_dim=12,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=max(2, num_layers // 2),
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(int(cand_meta_dim), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(int(summary_dim), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_score = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.cand_value = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
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
        self.embed_dim = int(embed_dim)

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_tokens: torch.Tensor,
        cand_masks: torch.Tensor,
        cand_meta: torch.Tensor,
        summary_feat: torch.Tensor,
    ) -> torch.Tensor:
        _, hlt_z = self.hlt_encoder(feat_hlt, mask_hlt, return_embedding=True)
        b, m, l, d = cand_tokens.shape
        cand_tok = cand_tokens.reshape(b * m, l, d)
        cand_m = cand_masks.reshape(b * m, l)
        _, cand_z_flat = self.cand_encoder(cand_tok, cand_m, return_embedding=True)
        cand_z = cand_z_flat.reshape(b, m, self.embed_dim)

        meta_z = self.meta_proj(cand_meta)
        hz = hlt_z.unsqueeze(1).expand(b, m, self.embed_dim)
        score_in = torch.cat([cand_z, meta_z, hz], dim=-1)
        attn = torch.softmax(self.attn_score(score_in).squeeze(-1), dim=1)

        value = self.cand_value(torch.cat([cand_z, meta_z], dim=-1))
        agg = (attn.unsqueeze(-1) * value).sum(dim=1)
        top1 = value[:, 0, :]
        s = self.summary_proj(summary_feat)

        x = torch.cat([hlt_z, agg, top1, s], dim=-1)
        return self.fuse(x).squeeze(-1)


class MultiCandidateM38Gated(nn.Module):
    def __init__(
        self,
        cand_meta_dim: int,
        summary_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hlt_encoder = ParticleTransformer(
            input_dim=7,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.cand_encoder = ParticleTransformer(
            input_dim=12,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=max(2, num_layers // 2),
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(int(cand_meta_dim), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(int(summary_dim), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_score = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.cand_value = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
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
        self.embed_dim = int(embed_dim)

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_tokens: torch.Tensor,
        cand_masks: torch.Tensor,
        cand_meta: torch.Tensor,
        summary_feat: torch.Tensor,
    ) -> torch.Tensor:
        _, hlt_z = self.hlt_encoder(feat_hlt, mask_hlt, return_embedding=True)
        b, m, l, d = cand_tokens.shape
        cand_tok = cand_tokens.reshape(b * m, l, d)
        cand_m = cand_masks.reshape(b * m, l)
        _, cand_z_flat = self.cand_encoder(cand_tok, cand_m, return_embedding=True)
        cand_z = cand_z_flat.reshape(b, m, self.embed_dim)

        meta_z = self.meta_proj(cand_meta)
        hz = hlt_z.unsqueeze(1).expand(b, m, self.embed_dim)
        score_in = torch.cat([cand_z, meta_z, hz], dim=-1)
        attn = torch.softmax(self.attn_score(score_in).squeeze(-1), dim=1)

        value = self.cand_value(torch.cat([cand_z, meta_z], dim=-1))
        agg = (attn.unsqueeze(-1) * value).sum(dim=1)
        top1 = value[:, 0, :]
        s = self.summary_proj(summary_feat)

        cand_pack = torch.cat([agg, top1, s], dim=-1)
        logit_h = self.hlt_head(hlt_z).squeeze(-1)
        logit_c = self.cand_head(cand_pack).squeeze(-1)
        g = self.gate(torch.cat([hlt_z, cand_pack], dim=-1)).squeeze(-1)
        return (1.0 - g) * logit_h + g * logit_c


# -----------------------------------------------------------------------------
# Training / eval helpers
# -----------------------------------------------------------------------------


def _forward_m38(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        feat_hlt=batch["feat_hlt"],
        mask_hlt=batch["mask_hlt"],
        cand_tokens=batch["cand_tokens"],
        cand_masks=batch["cand_masks"],
        cand_meta=batch["cand_meta"],
        summary_feat=batch["summary_feat"],
    )


def _train_m38_model(
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
            logit = _forward_m38(model, batch)
            lv = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            loss = m33._weighted_mean(lv, sw)
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
                p = torch.sigmoid(_forward_m38(model, batch))
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
            print(
                f"{name} ep {ep+1:03d}: train_loss={tr_loss/max(1,tr_n):.5f} "
                f"val_auc={va_auc:.4f} best={best_auc:.4f}@{best_epoch}"
            )

        if no_imp >= int(patience):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
    }


@torch.no_grad()
def _eval_m38_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pp, yy, ww = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["label"]
        sw = batch["sample_weight"]
        p = torch.sigmoid(_forward_m38(model, batch))
        pp.append(p.detach().cpu().numpy().astype(np.float64))
        yy.append(y.detach().cpu().numpy().astype(np.float64))
        ww.append(sw.detach().cpu().numpy().astype(np.float64))
    p_np = np.concatenate(pp, axis=0)
    y_np = np.concatenate(yy, axis=0)
    w_np = np.concatenate(ww, axis=0)
    auc = float(roc_auc_score(y_np, p_np, sample_weight=w_np)) if len(np.unique(y_np)) > 1 else 0.0
    fpr, tpr, _ = roc_curve(y_np, p_np, sample_weight=w_np)
    fpr50 = float(m33.fpr_at_target_tpr(fpr, tpr, 0.50))
    return auc, fpr50, p_np.astype(np.float32), y_np.astype(np.float32), w_np.astype(np.float32)


def _train_carry_predictor(
    model: CarryoverTokenPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    pos_weight: float,
) -> Tuple[CarryoverTokenPredictor, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    pos_w = torch.tensor(float(max(pos_weight, 1e-3)), dtype=torch.float32, device=device)
    best_auc = float("-inf")
    best_state = None
    best_epoch = 0
    no_imp = 0

    for ep in range(int(epochs)):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for batch in train_loader:
            feat = batch["feat_hlt"].to(device)
            mask = batch["mask_hlt"].to(device)
            tgt = batch["carry_tgt"].to(device)
            logit = model(feat, mask)
            lv = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none", pos_weight=pos_w)
            lv = lv * mask.float()
            loss = lv.sum() / mask.float().sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            bs = int(feat.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_n += bs

        model.eval()
        p_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["feat_hlt"].to(device)
                mask = batch["mask_hlt"].to(device)
                tgt = batch["carry_tgt"].to(device)
                p = torch.sigmoid(model(feat, mask))
                p_all.append(p[mask].detach().cpu().numpy().astype(np.float64))
                y_all.append(tgt[mask].detach().cpu().numpy().astype(np.float64))

        p_np = np.concatenate(p_all, axis=0) if p_all else np.array([], dtype=np.float64)
        y_np = np.concatenate(y_all, axis=0) if y_all else np.array([], dtype=np.float64)
        if len(np.unique(y_np)) > 1:
            va_auc = float(roc_auc_score(y_np, p_np))
        else:
            va_auc = 0.0

        if va_auc > best_auc:
            best_auc = float(va_auc)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"CarryPredictor ep {ep+1:03d}: train_loss={tr_loss/max(1,tr_n):.5f} "
                f"val_auc={va_auc:.4f} best={best_auc:.4f}@{best_epoch}"
            )

        if no_imp >= int(patience):
            print(f"Early stopping CarryPredictor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
    }


@torch.no_grad()
def _predict_carry_probs(
    model: CarryoverTokenPredictor,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    ds = FeatMaskDataset(feat_hlt, mask_hlt)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    out = []
    for batch in dl:
        feat = batch["feat"].to(device)
        mask = batch["mask"].to(device)
        p = torch.sigmoid(model(feat, mask)).detach().cpu().numpy().astype(np.float32)
        p[~mask.detach().cpu().numpy()] = 0.0
        out.append(p)
    return np.concatenate(out, axis=0)


# -----------------------------------------------------------------------------
# Carry target + seeded generation
# -----------------------------------------------------------------------------


def _build_carry_targets_np(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    dist_thresh: float,
    batch_size: int,
) -> np.ndarray:
    """Token-level target: HLT token has a near-identical Offline token in token-5 space."""
    n, l, _ = const_hlt.shape
    off_tok = _const_to_token5_np(const_off, mask_off)
    hlt_tok = _const_to_token5_np(const_hlt, mask_hlt)
    tgt = np.zeros((n, l), dtype=np.float32)

    for s in range(0, n, int(batch_size)):
        e = min(s + int(batch_size), n)
        h = torch.tensor(hlt_tok[s:e], dtype=torch.float32)
        o = torch.tensor(off_tok[s:e], dtype=torch.float32)
        mh = torch.tensor(mask_hlt[s:e], dtype=torch.bool)
        mo = torch.tensor(mask_off[s:e], dtype=torch.bool)

        d = torch.cdist(h, o, p=2)  # [B,L,L]
        valid = mh.unsqueeze(2) & mo.unsqueeze(1)
        d = torch.where(valid, d, torch.full_like(d, 1e6))
        md = d.min(dim=2).values
        t = (md < float(dist_thresh)) & mh
        tgt[s:e] = t.cpu().numpy().astype(np.float32)

    return tgt


def _build_prefix_schedule(k: int, max_prefix: int) -> List[int]:
    if k <= 1:
        return [int(max(0, max_prefix))]
    vals = np.linspace(0, int(max_prefix), int(k))
    out = [int(np.round(v)) for v in vals]
    out = [int(np.clip(v, 0, int(max_prefix))) for v in out]
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _prepare_prefixes_batch(
    carry_probs: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    schedule: List[int],
    seed_temp: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b, l = carry_probs.shape
    k = len(schedule)
    pmax = int(max(1, max(schedule)))

    hlt_tok = _const_to_token5_np(const_hlt, mask_hlt)
    prefix_tok = np.zeros((b, k, pmax, 5), dtype=np.float32)
    prefix_len = np.zeros((b, k), dtype=np.int64)
    prefix_carry_mean = np.zeros((b, k), dtype=np.float32)
    prefix_carry_min = np.zeros((b, k), dtype=np.float32)

    pt = const_hlt[..., 0]
    valid_ids_cache = [np.where(mask_hlt[i])[0] for i in range(b)]

    for i in range(b):
        ids = valid_ids_cache[i]
        if ids.size == 0:
            continue
        base = carry_probs[i, ids]
        for kk, n_pref in enumerate(schedule):
            nn = int(min(max(0, n_pref), ids.size, pmax))
            if nn <= 0:
                prefix_len[i, kk] = 0
                continue
            noise = rng.gumbel(loc=0.0, scale=1.0, size=ids.size).astype(np.float32)
            score = base + float(seed_temp) * (1.0 + 0.05 * kk) * noise
            pick_local = np.argpartition(-score, kth=nn - 1)[:nn]
            pick_ids = ids[pick_local]
            ord_pt = np.argsort(-pt[i, pick_ids])
            pick_ids = pick_ids[ord_pt]

            prefix_len[i, kk] = nn
            prefix_tok[i, kk, :nn] = hlt_tok[i, pick_ids]
            cvals = carry_probs[i, pick_ids]
            prefix_carry_mean[i, kk] = float(np.mean(cvals)) if cvals.size > 0 else 0.0
            prefix_carry_min[i, kk] = float(np.min(cvals)) if cvals.size > 0 else 0.0

    return prefix_tok, prefix_len, prefix_carry_mean, prefix_carry_min


@torch.no_grad()
def _decode_forced_prefix_multi(
    model: m28.HLT2OfflineSeq2Seq,
    feat_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_hlt: torch.Tensor,
    prefix_tok_bkpd: torch.Tensor,
    prefix_len_bk: torch.Tensor,
    max_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forced-prefix autoregressive decoding.

    Returns:
      pred_const_bkt4, pred_mask_bkt, conf_mean_bk, stop_len_bk
    """
    b = int(feat_hlt.shape[0])
    k = int(prefix_tok_bkpd.shape[1])
    t = int(max_steps)

    mem, hlt_tok, count_pred = model.encode(feat_hlt, mask_hlt, const_hlt)
    mem_pad = ~mask_hlt

    all_const = []
    all_mask = []
    all_conf_mean = []
    all_len = []

    for hk in range(k):
        in_tok = model.bos_token.expand(b, 1, model.token_dim)
        n_layers = len(model.decoder.layers)
        layer_cache: List[torch.Tensor | None] = [None] * n_layers

        pred_tok_seq = []
        stop_seq = []
        conf_seq = []

        pref_len = prefix_len_bk[:, hk]
        pref_tok = prefix_tok_bkpd[:, hk]

        for step in range(t):
            x_step = model.dec_in(in_tok) + model.dec_pos[:, step : step + 1, :]
            h_last, layer_cache = model._decoder_step_cached(x_step, mem, mem_pad, layer_cache)
            pred_tok, stop_logits, conf_logits, _attn, _gate = model._predict_from_hidden(
                h_last,
                mem,
                mask_hlt,
                hlt_tok,
                hyp_idx=(hk % int(model.num_hypotheses)),
            )

            next_tok = pred_tok[:, 0, :]
            force_mask = pref_len > step
            if bool(force_mask.any()):
                forced = pref_tok[:, step, :]
                next_tok = torch.where(force_mask.unsqueeze(1), forced, next_tok)

            pred_tok_seq.append(next_tok)
            stop_seq.append(torch.sigmoid(stop_logits[:, 0]))
            conf_seq.append(torch.sigmoid(conf_logits[:, 0]))
            in_tok = next_tok.unsqueeze(1)

        pred_tok_full = torch.stack(pred_tok_seq, dim=1)  # [B,T,5]
        stop_prob = torch.stack(stop_seq, dim=1)          # [B,T]
        conf_prob = torch.stack(conf_seq, dim=1)          # [B,T]

        pred_const = m28.token_to_const_torch(pred_tok_full)

        cp = torch.clamp(torch.round(count_pred), min=0, max=t).long()
        stop_pos = (stop_prob > 0.5).float()
        first_stop = torch.where(
            stop_pos.any(dim=1),
            stop_pos.argmax(dim=1),
            cp,
        )
        out_len = torch.maximum(first_stop.long(), pref_len.long())
        out_len = torch.clamp(out_len, min=0, max=t)

        steps = torch.arange(t, device=feat_hlt.device).view(1, -1)
        out_mask = steps < out_len.unsqueeze(1)

        conf_mean = (conf_prob * out_mask.float()).sum(dim=1) / out_mask.float().sum(dim=1).clamp(min=1.0)

        # sort valid part by pT descending for classifier consistency
        pred_np = pred_const.detach().cpu().numpy().astype(np.float32)
        mask_np = out_mask.detach().cpu().numpy().astype(bool)
        conf_np = conf_mean.detach().cpu().numpy().astype(np.float32)
        len_np = out_len.detach().cpu().numpy().astype(np.int64)
        for i in range(b):
            ll = int(len_np[i])
            if ll > 0:
                ord_i = np.argsort(-pred_np[i, :ll, 0])
                pred_np[i, :ll] = pred_np[i, :ll][ord_i]

        pred_t = torch.tensor(pred_np, dtype=torch.float32, device=feat_hlt.device)
        mask_t = torch.tensor(mask_np, dtype=torch.bool, device=feat_hlt.device)

        all_const.append(pred_t)
        all_mask.append(mask_t)
        all_conf_mean.append(torch.tensor(conf_np, dtype=torch.float32, device=feat_hlt.device))
        all_len.append(torch.tensor(len_np, dtype=torch.float32, device=feat_hlt.device))

    pred_const_bkt4 = torch.stack(all_const, dim=1)
    pred_mask_bkt = torch.stack(all_mask, dim=1)
    conf_mean_bk = torch.stack(all_conf_mean, dim=1)
    stop_len_bk = torch.stack(all_len, dim=1)
    pred_const_bkt4 = torch.nan_to_num(pred_const_bkt4, nan=0.0, posinf=0.0, neginf=0.0)
    pred_const_bkt4 = torch.where(pred_mask_bkt.unsqueeze(-1), pred_const_bkt4, torch.zeros_like(pred_const_bkt4))
    return pred_const_bkt4, pred_mask_bkt, conf_mean_bk, stop_len_bk


@dataclass
class CandidateSplitOutput:
    off_const: np.ndarray
    off_mask: np.ndarray
    hlt_const: np.ndarray
    hlt_mask: np.ndarray
    res_total: np.ndarray
    res_set: np.ndarray
    res_count: np.ndarray
    res_pt: np.ndarray
    res_mass: np.ndarray
    feasible: np.ndarray
    prefix_len: np.ndarray
    prefix_carry_mean: np.ndarray
    prefix_carry_min: np.ndarray
    conf_mean: np.ndarray
    stop_len: np.ndarray
    rank_norm: np.ndarray
    feasible_count_all: np.ndarray
    best_residual_all: np.ndarray


@torch.no_grad()
def _generate_candidates_split(
    reco_model: m28.HLT2OfflineSeq2Seq,
    carry_model: CarryoverTokenPredictor,
    feat_hlt: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    jet_keys: np.ndarray,
    cfg: Dict,
    seed_offset: int,
    candidate_k: int,
    keep_m: int,
    max_prefix: int,
    seed_temp: float,
    eps_total: float,
    eps_count: float,
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> CandidateSplitOutput:
    reco_model.eval()
    carry_model.eval()

    n, t, _ = const_hlt.shape
    k = int(candidate_k)
    m = int(keep_m)

    schedule = _build_prefix_schedule(k=k, max_prefix=max_prefix)
    rng = np.random.default_rng(int(seed))

    off_const_out = np.zeros((n, m, t, 4), dtype=np.float32)
    off_mask_out = np.zeros((n, m, t), dtype=bool)
    hlt_const_out = np.zeros((n, m, t, 4), dtype=np.float32)
    hlt_mask_out = np.zeros((n, m, t), dtype=bool)

    res_total_out = np.full((n, m), np.inf, dtype=np.float32)
    res_set_out = np.full((n, m), np.inf, dtype=np.float32)
    res_count_out = np.full((n, m), np.inf, dtype=np.float32)
    res_pt_out = np.full((n, m), np.inf, dtype=np.float32)
    res_mass_out = np.full((n, m), np.inf, dtype=np.float32)
    feasible_out = np.zeros((n, m), dtype=np.float32)

    prefix_len_out = np.zeros((n, m), dtype=np.float32)
    prefix_mean_out = np.zeros((n, m), dtype=np.float32)
    prefix_min_out = np.zeros((n, m), dtype=np.float32)
    conf_mean_out = np.zeros((n, m), dtype=np.float32)
    stop_len_out = np.zeros((n, m), dtype=np.float32)
    rank_norm_out = np.zeros((n, m), dtype=np.float32)

    feasible_count_all = np.zeros((n,), dtype=np.float32)
    best_res_all = np.full((n,), np.inf, dtype=np.float32)

    ds = FeatMaskDataset(feat_hlt, mask_hlt)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    offset = 0
    for batch in dl:
        feat_b = batch["feat"].to(device)
        mask_b = batch["mask"].to(device)
        bsz = int(feat_b.shape[0])

        const_b_np = const_hlt[offset : offset + bsz]
        mask_b_np = mask_hlt[offset : offset + bsz]
        keys_b = jet_keys[offset : offset + bsz]

        const_b = torch.tensor(const_b_np, dtype=torch.float32, device=device)

        carry_probs_t = torch.sigmoid(carry_model(feat_b, mask_b))
        carry_probs_np = carry_probs_t.detach().cpu().numpy().astype(np.float32)
        carry_probs_np[~mask_b_np] = 0.0

        pref_tok_np, pref_len_np, pref_mean_np, pref_min_np = _prepare_prefixes_batch(
            carry_probs=carry_probs_np,
            const_hlt=const_b_np,
            mask_hlt=mask_b_np,
            schedule=schedule,
            seed_temp=float(seed_temp),
            rng=rng,
        )
        pref_tok_t = torch.tensor(pref_tok_np, dtype=torch.float32, device=device)
        pref_len_t = torch.tensor(pref_len_np, dtype=torch.long, device=device)

        pred_const_bkt4, pred_mask_bkt, conf_mean_bk, stop_len_bk = _decode_forced_prefix_multi(
            model=reco_model,
            feat_hlt=feat_b,
            mask_hlt=mask_b,
            const_hlt=const_b,
            prefix_tok_bkpd=pref_tok_t,
            prefix_len_bk=pref_len_t,
            max_steps=int(t),
        )

        pred_const_np = pred_const_bkt4.detach().cpu().numpy().astype(np.float32)
        pred_mask_np = pred_mask_bkt.detach().cpu().numpy().astype(bool)
        conf_mean_np = conf_mean_bk.detach().cpu().numpy().astype(np.float32)
        stop_len_np = stop_len_bk.detach().cpu().numpy().astype(np.float32)

        flat_const = pred_const_np.reshape(bsz * k, t, 4)
        flat_mask = pred_mask_np.reshape(bsz * k, t)
        flat_keys = np.repeat(keys_b, k).astype(np.int64)

        hlt_pred_const, hlt_pred_mask, _ = m33._apply_hlt_effects_deterministic_keyed(
            const=flat_const,
            mask=flat_mask,
            cfg=cfg,
            jet_keys=flat_keys,
            base_seed=int(seed_offset),
        )

        tgt_const = np.repeat(const_b_np[:, None, :, :], k, axis=1).reshape(bsz * k, t, 4)
        tgt_mask = np.repeat(mask_b_np[:, None, :], k, axis=1).reshape(bsz * k, t)

        resid = m33._residual_fast_vec(
            pred_const=torch.tensor(hlt_pred_const, dtype=torch.float32),
            pred_mask=torch.tensor(hlt_pred_mask, dtype=torch.bool),
            tgt_const=torch.tensor(tgt_const, dtype=torch.float32),
            tgt_mask=torch.tensor(tgt_mask, dtype=torch.bool),
            w_chamfer=float(w_chamfer),
            w_count=float(w_count),
            w_pt=float(w_pt),
            w_mass=float(w_mass),
        )

        r_tot = resid["total"].cpu().numpy().reshape(bsz, k).astype(np.float32)
        r_set = resid["set"].cpu().numpy().reshape(bsz, k).astype(np.float32)
        r_cnt = resid["count"].cpu().numpy().reshape(bsz, k).astype(np.float32)
        r_pt = resid["pt"].cpu().numpy().reshape(bsz, k).astype(np.float32)
        r_mass = resid["mass"].cpu().numpy().reshape(bsz, k).astype(np.float32)

        hlt_pred_const_b = hlt_pred_const.reshape(bsz, k, t, 4)
        hlt_pred_mask_b = hlt_pred_mask.reshape(bsz, k, t)

        feas_all = (r_tot <= float(eps_total)) & (r_cnt <= float(eps_count))
        feasible_count_all[offset : offset + bsz] = feas_all.sum(axis=1).astype(np.float32)
        best_res_all[offset : offset + bsz] = np.min(r_tot, axis=1).astype(np.float32)

        order = np.argsort(r_tot, axis=1)
        pick = order[:, :m]

        for i in range(bsz):
            gi = offset + i
            for j in range(m):
                kj = int(pick[i, j])
                off_const_out[gi, j] = pred_const_np[i, kj]
                off_mask_out[gi, j] = pred_mask_np[i, kj]
                hlt_const_out[gi, j] = hlt_pred_const_b[i, kj]
                hlt_mask_out[gi, j] = hlt_pred_mask_b[i, kj]

                res_total_out[gi, j] = r_tot[i, kj]
                res_set_out[gi, j] = r_set[i, kj]
                res_count_out[gi, j] = r_cnt[i, kj]
                res_pt_out[gi, j] = r_pt[i, kj]
                res_mass_out[gi, j] = r_mass[i, kj]
                feasible_out[gi, j] = float(feas_all[i, kj])

                prefix_len_out[gi, j] = float(pref_len_np[i, kj])
                prefix_mean_out[gi, j] = float(pref_mean_np[i, kj])
                prefix_min_out[gi, j] = float(pref_min_np[i, kj])
                conf_mean_out[gi, j] = float(conf_mean_np[i, kj])
                stop_len_out[gi, j] = float(stop_len_np[i, kj])
                rank_norm_out[gi, j] = float(j) / float(max(1, m - 1))

        offset += bsz

    return CandidateSplitOutput(
        off_const=off_const_out,
        off_mask=off_mask_out,
        hlt_const=hlt_const_out,
        hlt_mask=hlt_mask_out,
        res_total=res_total_out,
        res_set=res_set_out,
        res_count=res_count_out,
        res_pt=res_pt_out,
        res_mass=res_mass_out,
        feasible=feasible_out,
        prefix_len=prefix_len_out,
        prefix_carry_mean=prefix_mean_out,
        prefix_carry_min=prefix_min_out,
        conf_mean=conf_mean_out,
        stop_len=stop_len_out,
        rank_norm=rank_norm_out,
        feasible_count_all=feasible_count_all,
        best_residual_all=best_res_all,
    )


# -----------------------------------------------------------------------------
# Candidate -> model arrays
# -----------------------------------------------------------------------------


def _build_m38_multicandidate_arrays(c: CandidateSplitOutput, max_constits: int, candidate_k: int) -> Dict[str, np.ndarray]:
    n, m, _, _ = c.off_const.shape

    cand_tokens = _build_candidate_token12_np(
        off_const=c.off_const,
        off_mask=c.off_mask,
        hlt_const=c.hlt_const,
        hlt_mask=c.hlt_mask,
    )
    cand_masks = (c.off_mask | c.hlt_mask).astype(bool)
    empty = ~cand_masks.any(axis=2)
    if np.any(empty):
        cand_masks[empty, 0] = True
    cand_tokens[~cand_masks] = 0.0

    prefix_len_norm = c.prefix_len / float(max(1, max_constits))
    stop_len_norm = c.stop_len / float(max(1, max_constits))

    cand_meta = np.stack(
        [
            c.res_total,
            c.res_set,
            c.res_count,
            c.res_pt,
            c.res_mass,
            c.feasible,
            prefix_len_norm,
            c.prefix_carry_mean,
            c.prefix_carry_min,
            c.conf_mean,
            stop_len_norm,
            c.rank_norm,
        ],
        axis=-1,
    ).astype(np.float32)

    best = c.res_total[:, 0]
    second = c.res_total[:, np.minimum(1, m - 1)]
    third = c.res_total[:, np.minimum(2, m - 1)]
    mean = c.res_total.mean(axis=1)
    std = c.res_total.std(axis=1)
    feasible_frac_topm = c.feasible.mean(axis=1)
    feasible_frac_all = np.clip(c.feasible_count_all / float(max(1, candidate_k)), 0.0, 1.0)
    pref_mean = c.prefix_len.mean(axis=1) / float(max(1, max_constits))
    carry_mean = c.prefix_carry_mean.mean(axis=1)
    conf_mean = c.conf_mean.mean(axis=1)
    gap12 = second - best
    gap23 = third - second

    summary_feat = np.stack(
        [
            best,
            second,
            third,
            mean,
            std,
            gap12,
            gap23,
            feasible_frac_topm.astype(np.float32),
            feasible_frac_all.astype(np.float32),
            pref_mean.astype(np.float32),
            carry_mean.astype(np.float32),
            conf_mean.astype(np.float32),
            c.best_residual_all.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "cand_tokens": cand_tokens.astype(np.float32),
        "cand_masks": cand_masks.astype(bool),
        "cand_meta": cand_meta.astype(np.float32),
        "summary_feat": summary_feat.astype(np.float32),
    }


# -----------------------------------------------------------------------------
# Parser / main
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m38 seeded m28 + deterministic residual multicandidate dualview")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model38_seeded_m28_detresid_multicand")
    p.add_argument("--run_name", type=str, default="model38_k6_seeded_m28_detresid_multicand_150k75k300k_seed0")

    p.add_argument("--n_train_jets", type=int, default=525000)
    p.add_argument("--n_train_split", type=int, default=150000)
    p.add_argument("--n_val_split", type=int, default=75000)
    p.add_argument("--n_test_split", type=int, default=300000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--use_train_weights", action="store_true")

    # deterministic D_hard
    p.add_argument("--merge_radius", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["merge_radius"]))
    p.add_argument("--eff_plateau_barrel", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"]))
    p.add_argument("--eff_plateau_endcap", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"]))
    p.add_argument("--smear_a", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_a"]))
    p.add_argument("--smear_b", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_b"]))
    p.add_argument("--smear_c", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_c"]))
    p.add_argument("--dhard_seed_offset", type=int, default=1337)

    # teacher / baseline
    p.add_argument("--cls_epochs", type=int, default=60)
    p.add_argument("--cls_patience", type=int, default=12)
    p.add_argument("--cls_lr", type=float, default=3e-4)
    p.add_argument("--cls_weight_decay", type=float, default=1e-4)
    p.add_argument("--cls_warmup_epochs", type=int, default=3)

    # carry predictor
    p.add_argument("--carry_epochs", type=int, default=24)
    p.add_argument("--carry_patience", type=int, default=6)
    p.add_argument("--carry_lr", type=float, default=2e-4)
    p.add_argument("--carry_weight_decay", type=float, default=1e-4)
    p.add_argument("--carry_embed_dim", type=int, default=128)
    p.add_argument("--carry_num_heads", type=int, default=4)
    p.add_argument("--carry_num_layers", type=int, default=3)
    p.add_argument("--carry_ff_dim", type=int, default=384)
    p.add_argument("--carry_dropout", type=float, default=0.10)
    p.add_argument("--carry_dist_thresh", type=float, default=0.22)

    # m28 completer
    p.add_argument("--reco_batch_size", type=int, default=96)
    p.add_argument("--reco_epochs", type=int, default=140)
    p.add_argument("--reco_patience", type=int, default=20)
    p.add_argument("--reco_min_epochs", type=int, default=35)
    p.add_argument("--reco_lr", type=float, default=2e-4)
    p.add_argument("--reco_weight_decay", type=float, default=1e-5)
    p.add_argument("--reco_embed_dim", type=int, default=384)
    p.add_argument("--reco_num_heads", type=int, default=8)
    p.add_argument("--reco_num_enc_layers", type=int, default=6)
    p.add_argument("--reco_num_dec_layers", type=int, default=6)
    p.add_argument("--reco_ff_dim", type=int, default=1024)
    p.add_argument("--reco_dropout", type=float, default=0.10)
    p.add_argument("--reco_set_loss_mode", type=str, default="hungarian", choices=["chamfer", "hungarian", "sinkhorn"])
    p.add_argument("--reco_loss_w_eos", type=float, default=float(m28.LOSS_CFG["w_eos"]))
    p.add_argument("--reco_loss_w_count", type=float, default=float(m28.LOSS_CFG["w_count"]))
    p.add_argument("--reco_loss_w_jetpt", type=float, default=float(m28.LOSS_CFG["w_jetpt"]))
    p.add_argument("--reco_loss_w_4vec", type=float, default=float(m28.LOSS_CFG["w_4vec"]))

    # seeded candidate generation
    p.add_argument("--seed_candidate_k", type=int, default=6)
    p.add_argument("--seed_keep_m", type=int, default=3)
    p.add_argument("--seed_max_prefix", type=int, default=12)
    p.add_argument("--seed_temp", type=float, default=0.35)
    p.add_argument("--candidate_gen_batch", type=int, default=64)

    # residual acceptance / ranking
    p.add_argument("--search_eps_total", type=float, default=0.60)
    p.add_argument("--search_eps_count", type=float, default=0.30)
    p.add_argument("--search_w_chamfer", type=float, default=1.0)
    p.add_argument("--search_w_count", type=float, default=0.25)
    p.add_argument("--search_w_pt", type=float, default=0.12)
    p.add_argument("--search_w_mass", type=float, default=0.08)

    # final dualview
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--dual_epochs", type=int, default=80)
    p.add_argument("--dual_patience", type=int, default=14)
    p.add_argument("--dual_lr", type=float, default=1.2e-4)
    p.add_argument("--dual_weight_decay", type=float, default=1e-4)

    p.add_argument("--save_fusion_scores", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    m33.set_seed(int(args.seed))

    device = torch.device(args.device)
    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Model-38 Seeded m28 + Deterministic Residual MultiCandidate DualView")
    print(f"Run: {save_root}")
    print(
        f"Split train/val/test = {int(args.n_train_split)}/{int(args.n_val_split)}/{int(args.n_test_split)} | "
        f"K={int(args.seed_candidate_k)} keepM={int(args.seed_keep_m)} max_prefix={int(args.seed_max_prefix)}"
    )
    print("=" * 72)

    # ---------------------------------------------------------------------
    # Data + deterministic pseudo-HLT
    # ---------------------------------------------------------------------
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

    const_raw = all_const[args.offset_jets : args.offset_jets + args.n_train_jets]
    labels = all_labels[args.offset_jets : args.offset_jets + args.n_train_jets].astype(np.int64)
    train_w = all_train_w[args.offset_jets : args.offset_jets + args.n_train_jets].astype(np.float32)

    total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > const_raw.shape[0]:
        raise RuntimeError(f"Requested splits sum to {total_need} but loaded jets are only {const_raw.shape[0]}")

    idx_all = np.arange(len(labels), dtype=np.int64)
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

    cfg = base._deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    raw_mask = const_raw[:, :, 0] > 0.0
    mask_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~mask_off] = 0.0

    print("Generating pseudo-HLT (deterministic keyed D_hard)...")
    jet_keys = (np.arange(len(const_off), dtype=np.int64) + int(args.offset_jets)).astype(np.int64)
    const_hlt, mask_hlt, hlt_stats = m33._apply_hlt_effects_deterministic_keyed(
        const=const_off,
        mask=mask_off,
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
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # standardized features
    print("Computing standardized features for train/val/test...")
    feat_off_tr = compute_features(const_off[train_idx], mask_off[train_idx])
    feat_off_va = compute_features(const_off[val_idx], mask_off[val_idx])
    feat_off_te = compute_features(const_off[test_idx], mask_off[test_idx])
    feat_hlt_tr = compute_features(const_hlt[train_idx], mask_hlt[train_idx])
    feat_hlt_va = compute_features(const_hlt[val_idx], mask_hlt[val_idx])
    feat_hlt_te = compute_features(const_hlt[test_idx], mask_hlt[test_idx])

    means, stds = get_stats(feat_off_tr, mask_off[train_idx], np.arange(feat_off_tr.shape[0], dtype=np.int64))
    feat_off_tr = standardize(feat_off_tr, mask_off[train_idx], means, stds)
    feat_off_va = standardize(feat_off_va, mask_off[val_idx], means, stds)
    feat_off_te = standardize(feat_off_te, mask_off[test_idx], means, stds)
    feat_hlt_tr = standardize(feat_hlt_tr, mask_hlt[train_idx], means, stds)
    feat_hlt_va = standardize(feat_hlt_va, mask_hlt[val_idx], means, stds)
    feat_hlt_te = standardize(feat_hlt_te, mask_hlt[test_idx], means, stds)

    sw_train = train_w[train_idx] if bool(args.use_train_weights) else np.ones((len(train_idx),), dtype=np.float32)
    sw_val = train_w[val_idx] if bool(args.use_train_weights) else np.ones((len(val_idx),), dtype=np.float32)
    sw_test = train_w[test_idx] if bool(args.use_train_weights) else np.ones((len(test_idx),), dtype=np.float32)

    # ---------------------------------------------------------------------
    # STEP 1: Teacher + baseline
    # ---------------------------------------------------------------------
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

    ds_tr_off = base.WeightedJetDataset(feat_off_tr, mask_off[train_idx], labels[train_idx], sw_train)
    ds_va_off = base.WeightedJetDataset(feat_off_va, mask_off[val_idx], labels[val_idx], sw_val)
    ds_te_off = base.WeightedJetDataset(feat_off_te, mask_off[test_idx], labels[test_idx], sw_test)

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

    ds_tr_hlt = base.WeightedJetDataset(feat_hlt_tr, mask_hlt[train_idx], labels[train_idx], sw_train)
    ds_va_hlt = base.WeightedJetDataset(feat_hlt_va, mask_hlt[val_idx], labels[val_idx], sw_val)
    ds_te_hlt = base.WeightedJetDataset(feat_hlt_te, mask_hlt[test_idx], labels[test_idx], sw_test)

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

    print(f"Teacher test AUC={teacher_auc_test:.4f} | Baseline test AUC={baseline_auc_test:.4f}")

    # ---------------------------------------------------------------------
    # STEP 2: Carryover predictor
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2: Carryover token predictor")
    print("=" * 72)

    carry_tgt_tr = _build_carry_targets_np(
        const_off=const_off[train_idx],
        mask_off=mask_off[train_idx],
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        dist_thresh=float(args.carry_dist_thresh),
        batch_size=256,
    )
    carry_tgt_va = _build_carry_targets_np(
        const_off=const_off[val_idx],
        mask_off=mask_off[val_idx],
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        dist_thresh=float(args.carry_dist_thresh),
        batch_size=256,
    )

    pos_frac = float(carry_tgt_tr[mask_hlt[train_idx]].mean()) if np.any(mask_hlt[train_idx]) else 0.5
    pos_w = float((1.0 - pos_frac) / max(pos_frac, 1e-6))

    ds_c_tr = RecoCarryDataset(feat_hlt_tr, mask_hlt[train_idx], carry_tgt_tr)
    ds_c_va = RecoCarryDataset(feat_hlt_va, mask_hlt[val_idx], carry_tgt_va)
    dl_c_tr = DataLoader(ds_c_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_c_va = DataLoader(ds_c_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    carry_model = CarryoverTokenPredictor(
        input_dim=7,
        embed_dim=int(args.carry_embed_dim),
        num_heads=int(args.carry_num_heads),
        num_layers=int(args.carry_num_layers),
        ff_dim=int(args.carry_ff_dim),
        dropout=float(args.carry_dropout),
        max_tokens=int(args.max_constits),
    ).to(device)

    carry_model, carry_metrics = _train_carry_predictor(
        model=carry_model,
        train_loader=dl_c_tr,
        val_loader=dl_c_va,
        device=device,
        epochs=int(args.carry_epochs),
        lr=float(args.carry_lr),
        weight_decay=float(args.carry_weight_decay),
        patience=int(args.carry_patience),
        pos_weight=pos_w,
    )

    # ---------------------------------------------------------------------
    # STEP 3: m28 completer
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 3: Train m28-style completer")
    print("=" * 72)

    tgt_tok_tr = m28.const_to_token_np(const_off[train_idx])
    tgt_tok_va = m28.const_to_token_np(const_off[val_idx])

    ds_reco_tr = m28.RecoSeqDataset(
        feat_hlt=feat_hlt_tr,
        mask_hlt=mask_hlt[train_idx],
        const_hlt=const_hlt[train_idx],
        tgt_tok=tgt_tok_tr,
        tgt_mask=mask_off[train_idx],
        labels=labels[train_idx].astype(np.float32),
    )
    ds_reco_va = m28.RecoSeqDataset(
        feat_hlt=feat_hlt_va,
        mask_hlt=mask_hlt[val_idx],
        const_hlt=const_hlt[val_idx],
        tgt_tok=tgt_tok_va,
        tgt_mask=mask_off[val_idx],
        labels=labels[val_idx].astype(np.float32),
    )
    dl_reco_tr = DataLoader(
        ds_reco_tr,
        batch_size=int(args.reco_batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
    )
    dl_reco_va = DataLoader(
        ds_reco_va,
        batch_size=int(args.reco_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    reco_model = m28.HLT2OfflineSeq2Seq(
        input_dim_hlt=7,
        token_dim=5,
        embed_dim=int(args.reco_embed_dim),
        num_heads=int(args.reco_num_heads),
        num_enc_layers=int(args.reco_num_enc_layers),
        num_dec_layers=int(args.reco_num_dec_layers),
        ff_dim=int(args.reco_ff_dim),
        dropout=float(args.reco_dropout),
        max_hlt_tokens=int(args.max_constits),
        max_decode_tokens=int(args.max_constits),
        use_coord_residual_param=False,
        num_hypotheses=int(args.seed_candidate_k),
    ).to(device)

    reco_train_cfg = {
        "batch_size": int(args.reco_batch_size),
        "epochs": int(args.reco_epochs),
        "lr": float(args.reco_lr),
        "weight_decay": float(args.reco_weight_decay),
        "patience": int(args.reco_patience),
        "min_epochs": int(args.reco_min_epochs),
    }
    reco_loss_cfg = dict(m28.LOSS_CFG)
    reco_loss_cfg["set_loss_mode"] = str(args.reco_set_loss_mode)
    reco_loss_cfg["w_eos"] = float(args.reco_loss_w_eos)
    reco_loss_cfg["w_count"] = float(args.reco_loss_w_count)
    reco_loss_cfg["w_jetpt"] = float(args.reco_loss_w_jetpt)
    reco_loss_cfg["w_4vec"] = float(args.reco_loss_w_4vec)
    reco_loss_cfg["winner_mode"] = "reco"

    reco_model, reco_metrics = m28.train_reconstructor_seq2seq(
        model=reco_model,
        train_loader=dl_reco_tr,
        val_loader=dl_reco_va,
        device=device,
        train_cfg=reco_train_cfg,
        loss_cfg=reco_loss_cfg,
        teacher=None,
        feat_means_t=None,
        feat_stds_t=None,
    )

    # ---------------------------------------------------------------------
    # STEP 4: Seeded candidate generation (train/val)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 4: Seeded candidate generation + deterministic residual ranking")
    print("=" * 72)

    c_tr = _generate_candidates_split(
        reco_model=reco_model,
        carry_model=carry_model,
        feat_hlt=feat_hlt_tr,
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        jet_keys=jet_keys[train_idx],
        cfg=cfg,
        seed_offset=int(args.seed + args.dhard_seed_offset),
        candidate_k=int(args.seed_candidate_k),
        keep_m=int(args.seed_keep_m),
        max_prefix=int(args.seed_max_prefix),
        seed_temp=float(args.seed_temp),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
        batch_size=int(args.candidate_gen_batch),
        device=device,
        seed=int(args.seed) + 101,
    )
    c_va = _generate_candidates_split(
        reco_model=reco_model,
        carry_model=carry_model,
        feat_hlt=feat_hlt_va,
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        jet_keys=jet_keys[val_idx],
        cfg=cfg,
        seed_offset=int(args.seed + args.dhard_seed_offset),
        candidate_k=int(args.seed_candidate_k),
        keep_m=int(args.seed_keep_m),
        max_prefix=int(args.seed_max_prefix),
        seed_temp=float(args.seed_temp),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
        batch_size=int(args.candidate_gen_batch),
        device=device,
        seed=int(args.seed) + 202,
    )

    print(
        "Candidate stats: "
        f"train(feasible_all={np.mean(c_tr.feasible_count_all):.2f}/{int(args.seed_candidate_k)}, bestR={np.mean(c_tr.best_residual_all):.4f}) "
        f"val(feasible_all={np.mean(c_va.feasible_count_all):.2f}/{int(args.seed_candidate_k)}, bestR={np.mean(c_va.best_residual_all):.4f})"
    )

    # ---------------------------------------------------------------------
    # STEP 5: Final dualview heads
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 5: Final Multi-Candidate DualView Classifiers (NoGate + Gated)")
    print("=" * 72)

    mv_tr = _build_m38_multicandidate_arrays(c_tr, max_constits=int(args.max_constits), candidate_k=int(args.seed_candidate_k))
    mv_va = _build_m38_multicandidate_arrays(c_va, max_constits=int(args.max_constits), candidate_k=int(args.seed_candidate_k))

    meta_tr = mv_tr["cand_meta"]
    meta_mean = meta_tr.reshape(-1, meta_tr.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    meta_std = (meta_tr.reshape(-1, meta_tr.shape[-1]).std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    summary_mean = mv_tr["summary_feat"].mean(axis=0, keepdims=True).astype(np.float32)
    summary_std = (mv_tr["summary_feat"].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    mm = meta_mean.reshape(1, 1, -1)
    ms = meta_std.reshape(1, 1, -1)
    sm = summary_mean.reshape(1, -1)
    ss = summary_std.reshape(1, -1)
    mv_tr["cand_meta"] = ((mv_tr["cand_meta"] - mm) / ms).astype(np.float32)
    mv_va["cand_meta"] = ((mv_va["cand_meta"] - mm) / ms).astype(np.float32)
    mv_tr["summary_feat"] = ((mv_tr["summary_feat"] - sm) / ss).astype(np.float32)
    mv_va["summary_feat"] = ((mv_va["summary_feat"] - sm) / ss).astype(np.float32)

    ds_dv_tr = DualViewM38Dataset(
        feat_hlt=feat_hlt_tr,
        mask_hlt=mask_hlt[train_idx],
        cand_tokens=mv_tr["cand_tokens"],
        cand_masks=mv_tr["cand_masks"],
        cand_meta=mv_tr["cand_meta"],
        summary_feat=mv_tr["summary_feat"],
        labels=labels[train_idx],
        sample_weight=sw_train,
    )
    ds_dv_va = DualViewM38Dataset(
        feat_hlt=feat_hlt_va,
        mask_hlt=mask_hlt[val_idx],
        cand_tokens=mv_va["cand_tokens"],
        cand_masks=mv_va["cand_masks"],
        cand_meta=mv_va["cand_meta"],
        summary_feat=mv_va["summary_feat"],
        labels=labels[val_idx],
        sample_weight=sw_val,
    )

    dl_dv_tr = DataLoader(ds_dv_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_dv_va = DataLoader(ds_dv_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    m38_nogate = MultiCandidateM38NoGate(
        cand_meta_dim=int(mv_tr["cand_meta"].shape[-1]),
        summary_dim=int(mv_tr["summary_feat"].shape[-1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    m38_nogate, m38_nogate_metrics = _train_m38_model(
        model=m38_nogate,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="M38NoGate",
    )

    m38_gated = MultiCandidateM38Gated(
        cand_meta_dim=int(mv_tr["cand_meta"].shape[-1]),
        summary_dim=int(mv_tr["summary_feat"].shape[-1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    m38_gated, m38_gated_metrics = _train_m38_model(
        model=m38_gated,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="M38Gated",
    )

    # ---------------------------------------------------------------------
    # STEP 6: Test candidates + final eval
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 6: Test seeded candidates + final evaluation")
    print("=" * 72)

    c_te = _generate_candidates_split(
        reco_model=reco_model,
        carry_model=carry_model,
        feat_hlt=feat_hlt_te,
        const_hlt=const_hlt[test_idx],
        mask_hlt=mask_hlt[test_idx],
        jet_keys=jet_keys[test_idx],
        cfg=cfg,
        seed_offset=int(args.seed + args.dhard_seed_offset),
        candidate_k=int(args.seed_candidate_k),
        keep_m=int(args.seed_keep_m),
        max_prefix=int(args.seed_max_prefix),
        seed_temp=float(args.seed_temp),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
        batch_size=int(args.candidate_gen_batch),
        device=device,
        seed=int(args.seed) + 303,
    )
    mv_te = _build_m38_multicandidate_arrays(c_te, max_constits=int(args.max_constits), candidate_k=int(args.seed_candidate_k))
    mv_te["cand_meta"] = ((mv_te["cand_meta"] - mm) / ms).astype(np.float32)
    mv_te["summary_feat"] = ((mv_te["summary_feat"] - sm) / ss).astype(np.float32)

    ds_dv_te = DualViewM38Dataset(
        feat_hlt=feat_hlt_te,
        mask_hlt=mask_hlt[test_idx],
        cand_tokens=mv_te["cand_tokens"],
        cand_masks=mv_te["cand_masks"],
        cand_meta=mv_te["cand_meta"],
        summary_feat=mv_te["summary_feat"],
        labels=labels[test_idx],
        sample_weight=sw_test,
    )
    dl_dv_te = DataLoader(ds_dv_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    auc_nog, fpr50_nog, pred_nog, lab_final, w_final = _eval_m38_model(m38_nogate, dl_dv_te, device)
    auc_gat, fpr50_gat, pred_gat, _lab2, _w2 = _eval_m38_model(m38_gated, dl_dv_te, device)

    fpr50_teacher = _fpr50(teacher_y_test, teacher_p_test, teacher_w_test)
    fpr50_baseline = _fpr50(baseline_y_test, baseline_p_test, baseline_w_test)

    print("\n" + "=" * 72)
    print("FINAL TEST")
    print("=" * 72)
    print(
        f"Teacher AUC={teacher_auc_test:.4f} FPR50={fpr50_teacher:.6f} | "
        f"HLT baseline AUC={baseline_auc_test:.4f} FPR50={fpr50_baseline:.6f} | "
        f"m38 NoGate AUC={auc_nog:.4f} FPR50={fpr50_nog:.6f} | "
        f"m38 Gated AUC={auc_gat:.4f} FPR50={fpr50_gat:.6f}"
    )

    # save artifacts
    torch.save({"model": teacher.state_dict(), "auc_test": float(teacher_auc_test)}, save_root / "teacher.pt")
    torch.save({"model": baseline.state_dict(), "auc_test": float(baseline_auc_test)}, save_root / "baseline_hlt.pt")
    torch.save({"model": carry_model.state_dict(), "metrics": carry_metrics}, save_root / "carry_predictor.pt")
    torch.save({"model": reco_model.state_dict(), "metrics": reco_metrics}, save_root / "reco_completer_m28style.pt")
    torch.save({"model": m38_nogate.state_dict(), "metrics": m38_nogate_metrics}, save_root / "m38_multicand_nogate.pt")
    torch.save({"model": m38_gated.state_dict(), "metrics": m38_gated_metrics}, save_root / "m38_multicand_gated.pt")

    np.savez_compressed(
        save_root / "m38_test_scores.npz",
        labels_test=lab_final.astype(np.float32),
        preds_m38_nogate=pred_nog.astype(np.float32),
        preds_m38_gated=pred_gat.astype(np.float32),
        preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
        preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
        sample_weight=np.asarray(w_final, dtype=np.float32),
        auc_teacher=float(teacher_auc_test),
        auc_hlt=float(baseline_auc_test),
        auc_m38_nogate=float(auc_nog),
        auc_m38_gated=float(auc_gat),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_hlt=float(fpr50_baseline),
        fpr50_m38_nogate=float(fpr50_nog),
        fpr50_m38_gated=float(fpr50_gat),
    )

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "fusion_scores_test.npz",
            labels_test=lab_final.astype(np.float32),
            preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
            preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
            preds_m38_nogate=np.asarray(pred_nog, dtype=np.float32),
            preds_m38_gated=np.asarray(pred_gat, dtype=np.float32),
            sample_weight=np.asarray(w_final, dtype=np.float32),
        )

    report = {
        "model": "m38_seeded_m28_detresid_multicand",
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
        "carry_predictor": carry_metrics,
        "reco_completer": reco_metrics,
        "candidate_generation": {
            "candidate_k": int(args.seed_candidate_k),
            "keep_m": int(args.seed_keep_m),
            "max_prefix": int(args.seed_max_prefix),
            "eps_total": float(args.search_eps_total),
            "eps_count": float(args.search_eps_count),
            "weights": {
                "chamfer": float(args.search_w_chamfer),
                "count": float(args.search_w_count),
                "pt": float(args.search_w_pt),
                "mass": float(args.search_w_mass),
            },
            "train": {
                "mean_feasible_all": float(np.mean(c_tr.feasible_count_all)),
                "mean_best_residual": float(np.mean(c_tr.best_residual_all)),
            },
            "val": {
                "mean_feasible_all": float(np.mean(c_va.feasible_count_all)),
                "mean_best_residual": float(np.mean(c_va.best_residual_all)),
            },
            "test": {
                "mean_feasible_all": float(np.mean(c_te.feasible_count_all)),
                "mean_best_residual": float(np.mean(c_te.best_residual_all)),
            },
        },
        "m38_nogate": {
            "auc_test": float(auc_nog),
            "fpr50_test": float(fpr50_nog),
            "metrics": m38_nogate_metrics,
        },
        "m38_gated": {
            "auc_test": float(auc_gat),
            "fpr50_test": float(fpr50_gat),
            "metrics": m38_gated_metrics,
        },
    }
    with open(save_root / "m38_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
    )

    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()

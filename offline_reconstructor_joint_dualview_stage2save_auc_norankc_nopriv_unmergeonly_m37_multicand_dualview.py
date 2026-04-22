#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m37: Dictionary retrieval multi-candidate dualview top-tagging pipeline.

Key idea:
- Keep m36 retrieval/indexing stages.
- Replace final dualview head with multi-candidate evidence fusion:
  * HLT query stream
  * shared candidate stream over all retrieved candidates
  * candidate meta/evidence vectors + search-hardness summaries
- Train NoGate and Gated final classifiers on the fused evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_detfeas_dualview as m33
from unmerge_correct_hlt import ParticleTransformer, compute_features, get_stats, standardize


@dataclass
class JetMeta:
    count: np.ndarray
    jet_pt: np.ndarray
    jet_mass: np.ndarray


@dataclass
class RetrievalClassIndex:
    class_val: int
    local_ids: np.ndarray
    nn: NearestNeighbors


class RetrievalEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(embed_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class DualViewM37Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        cand_tokens: np.ndarray,
        cand_masks: np.ndarray,
        cand_meta: np.ndarray,
        cand_class: np.ndarray,
        summary_feat: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.cand_tokens = torch.tensor(cand_tokens, dtype=torch.float32)
        self.cand_masks = torch.tensor(cand_masks, dtype=torch.bool)
        self.cand_meta = torch.tensor(cand_meta, dtype=torch.float32)
        self.cand_class = torch.tensor(cand_class, dtype=torch.float32)
        self.summary_feat = torch.tensor(summary_feat, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        self.sample_weight = torch.tensor(sample_weight.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "cand_tokens": self.cand_tokens[i],
            "cand_masks": self.cand_masks[i],
            "cand_meta": self.cand_meta[i],
            "cand_class": self.cand_class[i],
            "summary_feat": self.summary_feat[i],
            "label": self.labels[i],
            "sample_weight": self.sample_weight[i],
        }


def _const_to_token5_np(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pt = np.clip(const[..., 0], 1e-8, None)
    eta = np.clip(const[..., 1], -5.0, 5.0)
    phi = const[..., 2]
    e = np.clip(const[..., 3], 1e-8, None)
    feat = np.stack(
        [
            np.log(pt),
            eta,
            np.sin(phi),
            np.cos(phi),
            np.log(e),
        ],
        axis=-1,
    ).astype(np.float32)
    feat[~mask] = 0.0
    return feat


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
    return tok.astype(np.float32)


class MultiCandidateNoGate(nn.Module):
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
            nn.Linear(embed_dim * 5, 256),
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

    def _aggregate(
        self,
        hlt_z: torch.Tensor,
        cand_z: torch.Tensor,
        cand_meta: torch.Tensor,
        cand_class: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, m, _ = cand_z.shape
        meta_z = self.meta_proj(cand_meta)
        hz = hlt_z.unsqueeze(1).expand(b, m, self.embed_dim)
        score_in = torch.cat([cand_z, meta_z, hz], dim=-1)
        score = self.attn_score(score_in).squeeze(-1)
        attn = torch.softmax(score, dim=1)

        value = self.cand_value(torch.cat([cand_z, meta_z], dim=-1))
        agg_all = (attn.unsqueeze(-1) * value).sum(dim=1)

        cls_top = cand_class
        cls_bg = 1.0 - cand_class
        w_top = attn * cls_top
        w_bg = attn * cls_bg
        w_top = w_top / w_top.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w_bg = w_bg / w_bg.sum(dim=1, keepdim=True).clamp(min=1e-6)
        agg_top = (w_top.unsqueeze(-1) * value).sum(dim=1)
        agg_bg = (w_bg.unsqueeze(-1) * value).sum(dim=1)
        return agg_all, agg_top, agg_bg

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_tokens: torch.Tensor,
        cand_masks: torch.Tensor,
        cand_meta: torch.Tensor,
        cand_class: torch.Tensor,
        summary_feat: torch.Tensor,
    ) -> torch.Tensor:
        _, hlt_z = self.hlt_encoder(feat_hlt, mask_hlt, return_embedding=True)
        b, m, l, d = cand_tokens.shape
        cand_tok = cand_tokens.reshape(b * m, l, d)
        cand_m = cand_masks.reshape(b * m, l)
        _, cand_z_flat = self.cand_encoder(cand_tok, cand_m, return_embedding=True)
        cand_z = cand_z_flat.reshape(b, m, self.embed_dim)
        agg_all, agg_top, agg_bg = self._aggregate(hlt_z, cand_z, cand_meta, cand_class)
        s = self.summary_proj(summary_feat)
        x = torch.cat([hlt_z, agg_all, agg_top, agg_bg, s], dim=-1)
        return self.fuse(x).squeeze(-1)


class MultiCandidateGated(nn.Module):
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
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 5, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.embed_dim = int(embed_dim)

    def _aggregate(
        self,
        hlt_z: torch.Tensor,
        cand_z: torch.Tensor,
        cand_meta: torch.Tensor,
        cand_class: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, m, _ = cand_z.shape
        meta_z = self.meta_proj(cand_meta)
        hz = hlt_z.unsqueeze(1).expand(b, m, self.embed_dim)
        score_in = torch.cat([cand_z, meta_z, hz], dim=-1)
        score = self.attn_score(score_in).squeeze(-1)
        attn = torch.softmax(score, dim=1)
        value = self.cand_value(torch.cat([cand_z, meta_z], dim=-1))
        agg_all = (attn.unsqueeze(-1) * value).sum(dim=1)

        cls_top = cand_class
        cls_bg = 1.0 - cand_class
        w_top = attn * cls_top
        w_bg = attn * cls_bg
        w_top = w_top / w_top.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w_bg = w_bg / w_bg.sum(dim=1, keepdim=True).clamp(min=1e-6)
        agg_top = (w_top.unsqueeze(-1) * value).sum(dim=1)
        agg_bg = (w_bg.unsqueeze(-1) * value).sum(dim=1)
        return agg_all, agg_top, agg_bg

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_tokens: torch.Tensor,
        cand_masks: torch.Tensor,
        cand_meta: torch.Tensor,
        cand_class: torch.Tensor,
        summary_feat: torch.Tensor,
    ) -> torch.Tensor:
        _, hlt_z = self.hlt_encoder(feat_hlt, mask_hlt, return_embedding=True)
        b, m, l, d = cand_tokens.shape
        cand_tok = cand_tokens.reshape(b * m, l, d)
        cand_m = cand_masks.reshape(b * m, l)
        _, cand_z_flat = self.cand_encoder(cand_tok, cand_m, return_embedding=True)
        cand_z = cand_z_flat.reshape(b, m, self.embed_dim)
        agg_all, agg_top, agg_bg = self._aggregate(hlt_z, cand_z, cand_meta, cand_class)
        s = self.summary_proj(summary_feat)
        cand_pack = torch.cat([agg_all, agg_top, agg_bg, s], dim=-1)
        logit_h = self.hlt_head(hlt_z).squeeze(-1)
        logit_c = self.cand_head(cand_pack).squeeze(-1)
        g = self.gate(torch.cat([hlt_z, cand_pack], dim=-1)).squeeze(-1)
        return (1.0 - g) * logit_h + g * logit_c


def _forward_m37(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        feat_hlt=batch["feat_hlt"],
        mask_hlt=batch["mask_hlt"],
        cand_tokens=batch["cand_tokens"],
        cand_masks=batch["cand_masks"],
        cand_meta=batch["cand_meta"],
        cand_class=batch["cand_class"],
        summary_feat=batch["summary_feat"],
    )


def _train_m37_model(
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
            logit = _forward_m37(model, batch)
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
                p = torch.sigmoid(_forward_m37(model, batch))
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
def _eval_m37_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    p_all, y_all, w_all = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        p = torch.sigmoid(_forward_m37(model, batch))
        p_all.append(p.detach().cpu().numpy().astype(np.float64))
        y_all.append(batch["label"].detach().cpu().numpy().astype(np.float64))
        w_all.append(batch["sample_weight"].detach().cpu().numpy().astype(np.float64))
    p_np = np.concatenate(p_all, axis=0) if p_all else np.array([], dtype=np.float64)
    y_np = np.concatenate(y_all, axis=0) if y_all else np.array([], dtype=np.float64)
    w_np = np.concatenate(w_all, axis=0) if w_all else np.ones_like(y_np, dtype=np.float64)
    auc = float(roc_auc_score(y_np, p_np, sample_weight=w_np)) if len(np.unique(y_np)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(y_np, p_np, sample_weight=w_np)
    fpr50 = float(m33.fpr_at_target_tpr(fpr, tpr, 0.50))
    return auc, fpr50, p_np.astype(np.float32), y_np.astype(np.float32), w_np.astype(np.float32)


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (b + eps)


def _jet_meta_np(const: np.ndarray, mask: np.ndarray) -> JetMeta:
    m = mask.astype(np.float32)
    pt = const[..., 0] * m
    eta = const[..., 1]
    phi = const[..., 2]
    e = const[..., 3] * m

    px = pt * np.cos(phi) * m
    py = pt * np.sin(phi) * m
    pz = pt * np.sinh(eta) * m

    px_sum = px.sum(axis=1)
    py_sum = py.sum(axis=1)
    pz_sum = pz.sum(axis=1)
    e_sum = e.sum(axis=1)

    jet_pt = np.sqrt(np.maximum(px_sum ** 2 + py_sum ** 2, 0.0))
    p2 = px_sum ** 2 + py_sum ** 2 + pz_sum ** 2
    m2 = np.maximum(e_sum ** 2 - p2, 0.0)
    jet_mass = np.sqrt(m2)
    count = mask.sum(axis=1).astype(np.float32)
    return JetMeta(count=count.astype(np.float32), jet_pt=jet_pt.astype(np.float32), jet_mass=jet_mass.astype(np.float32))


def _build_retrieval_desc(
    const: np.ndarray,
    mask: np.ndarray,
    meta: JetMeta,
    max_constits: int,
) -> np.ndarray:
    m = mask.astype(bool)
    pt = np.where(m, const[..., 0], 0.0)
    eta = np.where(m, const[..., 1], 0.0)

    pt_sum = pt.sum(axis=1)
    topk = min(3, pt.shape[1])
    if topk > 0:
        part = np.partition(pt, kth=pt.shape[1] - topk, axis=1)[:, -topk:]
        part = np.sort(part, axis=1)[:, ::-1]
        if topk < 3:
            pad = np.zeros((pt.shape[0], 3 - topk), dtype=np.float32)
            top3 = np.concatenate([part, pad], axis=1)
        else:
            top3 = part
    else:
        top3 = np.zeros((pt.shape[0], 3), dtype=np.float32)

    frac = _safe_div(top3, pt_sum[:, None])
    eta_w = _safe_div((eta * pt).sum(axis=1), pt_sum)
    eta_abs_w = _safe_div((np.abs(eta) * pt).sum(axis=1), pt_sum)

    desc = np.stack(
        [
            meta.count / float(max_constits),
            np.log1p(meta.jet_pt),
            np.log1p(meta.jet_mass),
            frac[:, 0],
            frac[:, 1],
            frac[:, 2],
            eta_w / 5.0,
            eta_abs_w / 5.0,
        ],
        axis=1,
    ).astype(np.float32)
    return desc


def _select_landmark_indices(desc: np.ndarray, n_landmarks: int, seed: int) -> np.ndarray:
    n = int(desc.shape[0])
    k = int(max(1, min(n_landmarks, n)))
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, n))
    ids = [first]
    min_d2 = np.sum((desc - desc[first:first + 1]) ** 2, axis=1)
    min_d2[first] = -1.0
    for _ in range(1, k):
        nxt = int(np.argmax(min_d2))
        ids.append(nxt)
        d2 = np.sum((desc - desc[nxt:nxt + 1]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[ids] = -1.0
    return np.asarray(ids, dtype=np.int64)


def _build_landmark11_desc(
    desc: np.ndarray,
    meta: JetMeta,
    landmark_desc: np.ndarray,
    max_constits: int,
) -> np.ndarray:
    # 11D = 8 landmark-distance channels + count + log(1+pt) + log(1+mass)
    # Distances are computed in standardized physics-descriptor space.
    k = int(landmark_desc.shape[0])
    diff = desc[:, None, :] - landmark_desc[None, :, :]
    d = np.sqrt(np.maximum((diff * diff).sum(axis=2), 0.0)).astype(np.float32)
    if k < 8:
        d = np.concatenate([d, np.zeros((d.shape[0], 8 - k), dtype=np.float32)], axis=1)
    elif k > 8:
        d = d[:, :8]
    c = (meta.count / float(max_constits)).astype(np.float32)[:, None]
    p = np.log1p(meta.jet_pt).astype(np.float32)[:, None]
    m = np.log1p(meta.jet_mass).astype(np.float32)[:, None]
    return np.concatenate([d, c, p, m], axis=1).astype(np.float32)


def _train_retrieval_embedder(
    desc_train: np.ndarray,
    meta_train: JetMeta,
    labels_train: np.ndarray,
    desc_dict: np.ndarray,
    meta_dict: JetMeta,
    dict_labels: np.ndarray,
    max_constits: int,
    w_desc: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    seed: int,
    device: torch.device,
    embed_dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    pool_size: int,
    train_anchors: int,
    lr: float,
    weight_decay: float,
    margin: float,
) -> Tuple[RetrievalEmbedder, Dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    n_all = int(desc_train.shape[0])
    n_use = int(min(max(1, train_anchors), n_all))
    if n_use < n_all:
        use_idx = rng.choice(n_all, size=n_use, replace=False)
    else:
        use_idx = np.arange(n_all, dtype=np.int64)

    dtr = torch.tensor(desc_train[use_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(labels_train[use_idx].astype(np.int64), dtype=torch.long, device=device)
    ctr = torch.tensor(meta_train.count[use_idx], dtype=torch.float32, device=device)
    ptr = torch.tensor(meta_train.jet_pt[use_idx], dtype=torch.float32, device=device)
    mtr = torch.tensor(meta_train.jet_mass[use_idx], dtype=torch.float32, device=device)

    dd = torch.tensor(desc_dict, dtype=torch.float32, device=device)
    cd = torch.tensor(meta_dict.count, dtype=torch.float32, device=device)
    pd = torch.tensor(meta_dict.jet_pt, dtype=torch.float32, device=device)
    md = torch.tensor(meta_dict.jet_mass, dtype=torch.float32, device=device)

    class0 = torch.tensor(np.where(dict_labels.astype(np.int64) == 0)[0], dtype=torch.long, device=device)
    class1 = torch.tensor(np.where(dict_labels.astype(np.int64) == 1)[0], dtype=torch.long, device=device)
    if int(class0.numel()) == 0 or int(class1.numel()) == 0:
        raise RuntimeError("Dictionary must have both classes for learned retrieval embedder.")

    model = RetrievalEmbedder(
        input_dim=int(desc_train.shape[1]),
        hidden_dim=int(hidden_dim),
        embed_dim=int(embed_dim),
        dropout=float(dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    eps = 1e-6
    bs = int(max(16, batch_size))
    psize = int(max(32, pool_size))
    maxc = float(max(1, max_constits))
    loss_hist = []

    for ep in range(int(max(1, epochs))):
        model.train()
        perm = torch.randperm(dtr.shape[0], device=device)
        ep_loss = 0.0
        ep_n = 0

        for s in range(0, int(perm.numel()), bs):
            idx = perm[s:s + bs]
            if idx.numel() == 0:
                continue
            ad = dtr[idx]
            ay = ytr[idx]
            ac = ctr[idx]
            ap = ptr[idx]
            am = mtr[idx]

            b = int(idx.numel())
            cand_idx = torch.empty((b, psize), dtype=torch.long, device=device)
            is_bg = ay == 0
            is_tp = ~is_bg
            if bool(is_bg.any()):
                nbg = int(is_bg.sum().item())
                rid = torch.randint(0, int(class0.numel()), (nbg, psize), device=device)
                cand_idx[is_bg] = class0[rid]
            if bool(is_tp.any()):
                ntp = int(is_tp.sum().item())
                rid = torch.randint(0, int(class1.numel()), (ntp, psize), device=device)
                cand_idx[is_tp] = class1[rid]

            cd_desc = dd[cand_idx]  # [B, P, D]
            rs = torch.linalg.norm(cd_desc - ad.unsqueeze(1), dim=-1)

            cc = cd[cand_idx]
            cp = pd[cand_idx]
            cm = md[cand_idx]
            rc = torch.abs(cc - ac.unsqueeze(1)) / maxc
            rpt = torch.abs(torch.log1p(cp.clamp_min(0.0)) - torch.log1p(ap.clamp_min(0.0).unsqueeze(1)))
            rms = torch.abs(torch.log1p(cm.clamp_min(0.0)) - torch.log1p(am.clamp_min(0.0).unsqueeze(1)))
            rt = float(w_desc) * rs + float(w_count) * rc + float(w_pt) * rpt + float(w_mass) * rms

            pos_j = torch.argmin(rt, dim=1)
            neg_j = torch.argmax(rt, dim=1)
            row = torch.arange(b, device=device)
            pos_desc = cd_desc[row, pos_j]
            neg_desc = cd_desc[row, neg_j]

            z_a = model(ad)
            z_p = model(pos_desc)
            z_n = model(neg_desc)

            loss_trip = F.triplet_margin_loss(z_a, z_p, z_n, margin=float(margin), p=2.0, reduction="mean")
            d_ap = torch.linalg.norm(z_a - z_p, dim=-1)
            d_an = torch.linalg.norm(z_a - z_n, dim=-1)
            loss_rank = F.softplus((d_ap - d_an) / 0.1).mean()
            pos_target = rt[row, pos_j].detach()
            neg_target = rt[row, neg_j].detach()
            pos_target = pos_target / (pos_target.mean() + eps)
            neg_target = neg_target / (neg_target.mean() + eps)
            loss_scale = 0.5 * F.mse_loss(d_ap, pos_target) + 0.5 * F.mse_loss(d_an, neg_target)
            loss = loss_trip + 0.25 * loss_rank + 0.10 * loss_scale

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ep_loss += float(loss.item()) * b
            ep_n += b

        mean_loss = ep_loss / max(1, ep_n)
        loss_hist.append(mean_loss)
        print(f"RetrievalEmbed ep {ep+1:03d}: loss={mean_loss:.6f}")

    metrics = {
        "epochs": int(max(1, epochs)),
        "anchor_count": int(n_use),
        "pool_size": int(psize),
        "loss_final": float(loss_hist[-1]) if loss_hist else float("nan"),
        "loss_best": float(min(loss_hist)) if loss_hist else float("nan"),
    }
    return model, metrics


@torch.no_grad()
def _encode_retrieval_desc(
    model: RetrievalEmbedder,
    desc: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    n = int(desc.shape[0])
    out = np.zeros((n, model.net[-1].out_features), dtype=np.float32)
    bs = int(max(64, batch_size))
    for s in range(0, n, bs):
        e = min(n, s + bs)
        x = torch.tensor(desc[s:e], dtype=torch.float32, device=device)
        z = model(x).detach().cpu().numpy().astype(np.float32)
        out[s:e] = z
    return out


def _topk_smallest_idx(values: np.ndarray, k: int) -> np.ndarray:
    k = int(max(0, min(k, values.shape[0])))
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    if values.shape[0] <= k:
        return np.argsort(values).astype(np.int64)
    part = np.argpartition(values, k - 1)[:k]
    return part[np.argsort(values[part])].astype(np.int64)


def _build_class_index(
    dict_desc: np.ndarray,
    dict_labels: np.ndarray,
    class_val: int,
) -> RetrievalClassIndex:
    local_ids = np.where(dict_labels.astype(np.int64) == int(class_val))[0].astype(np.int64)
    if local_ids.size == 0:
        raise RuntimeError(f"Dictionary has no entries for class {class_val}")
    nn = NearestNeighbors(algorithm="ball_tree", leaf_size=40, metric="euclidean")
    nn.fit(dict_desc[local_ids])
    return RetrievalClassIndex(class_val=int(class_val), local_ids=local_ids, nn=nn)


def _retrieve_pool_for_class(
    query_index_desc: np.ndarray,
    query_score_desc: np.ndarray,
    query_meta: JetMeta,
    query_const_hlt: np.ndarray,
    query_mask_hlt: np.ndarray,
    class_index: RetrievalClassIndex,
    dict_score_desc: np.ndarray,
    dict_const_off: np.ndarray,
    dict_mask_off: np.ndarray,
    dict_const_hlt: np.ndarray,
    dict_mask_hlt: np.ndarray,
    dict_meta: JetMeta,
    target_k: int,
    per_round: int,
    max_rounds: int,
    eps_total: float,
    eps_count: float,
    w_desc: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    max_constits: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    n, l, _ = query_const_hlt.shape
    k_total = int(max(1, per_round * max_rounds))
    k_nn = int(max(1, min(k_total, class_index.local_ids.shape[0])))

    cand_const = np.zeros((n, target_k, l, 4), dtype=np.float32)
    cand_mask = np.zeros((n, target_k, l), dtype=bool)
    cand_hlt_const = np.zeros((n, target_k, l, 4), dtype=np.float32)
    cand_hlt_mask = np.zeros((n, target_k, l), dtype=bool)
    cand_gid = np.full((n, target_k), -1, dtype=np.int64)
    resid_total = np.full((n, target_k), np.inf, dtype=np.float32)
    resid_set = np.full((n, target_k), np.inf, dtype=np.float32)
    resid_count = np.full((n, target_k), np.inf, dtype=np.float32)
    resid_pt = np.full((n, target_k), np.inf, dtype=np.float32)
    resid_mass = np.full((n, target_k), np.inf, dtype=np.float32)
    round_found = np.full((n, target_k), float(max_rounds + 1), dtype=np.float32)
    feasible = np.zeros((n, target_k), dtype=bool)
    feasible_count = np.zeros((n,), dtype=np.float32)

    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        qidx_b = query_index_desc[s:e]
        _dists_b, local_b = class_index.nn.kneighbors(qidx_b, n_neighbors=k_nn, return_distance=True)

        for i in range(e - s):
            local_ids = class_index.local_ids[local_b[i].astype(np.int64)]
            qscore = query_score_desc[s + i]
            cscore = dict_score_desc[local_ids]

            q_cnt = float(query_meta.count[s + i])
            q_pt = float(query_meta.jet_pt[s + i])
            q_mass = float(query_meta.jet_mass[s + i])

            c_cnt = dict_meta.count[local_ids]
            c_pt = dict_meta.jet_pt[local_ids]
            c_mass = dict_meta.jet_mass[local_ids]

            rs = np.sqrt(np.sum((cscore - qscore[None, :]) ** 2, axis=1, dtype=np.float32)).astype(np.float32)
            rc = np.abs(c_cnt - q_cnt) / max(1.0, float(max_constits))
            rpt = np.abs(np.log1p(c_pt) - np.log1p(max(q_pt, 0.0)))
            rms = np.abs(np.log1p(c_mass) - np.log1p(max(q_mass, 0.0)))
            rt = (
                float(w_desc) * rs
                + float(w_count) * rc
                + float(w_pt) * rpt
                + float(w_mass) * rms
            )

            feas_mask = (rt <= float(eps_total)) & (rc <= float(eps_count))
            feasible_count[s + i] = float(feas_mask.sum())

            feas_idx = np.where(feas_mask)[0].astype(np.int64)
            if feas_idx.size > 0:
                ord_feas_local = _topk_smallest_idx(rt[feas_idx], target_k)
                pick_feas = feas_idx[ord_feas_local]
            else:
                pick_feas = np.zeros((0,), dtype=np.int64)

            need = int(target_k - pick_feas.size)
            if need > 0:
                non_idx = np.where(~feas_mask)[0].astype(np.int64)
                if non_idx.size > 0:
                    ord_non_local = _topk_smallest_idx(rt[non_idx], need)
                    pick_non = non_idx[ord_non_local]
                else:
                    pick_non = np.zeros((0,), dtype=np.int64)
                pick = np.concatenate([pick_feas, pick_non], axis=0)
            else:
                pick = pick_feas[:target_k]

            if pick.size == 0:
                pick = np.array([int(np.argmin(rt))], dtype=np.int64)

            if pick.size < target_k:
                pad = np.full((target_k - pick.size,), int(pick[0]), dtype=np.int64)
                pick = np.concatenate([pick, pad], axis=0)

            pick = pick[:target_k]
            gids = local_ids[pick]

            cand_const[s + i] = dict_const_off[gids]
            cand_mask[s + i] = dict_mask_off[gids]
            cand_hlt_const[s + i] = dict_const_hlt[gids]
            cand_hlt_mask[s + i] = dict_mask_hlt[gids]
            cand_gid[s + i] = gids.astype(np.int64)
            resid_total[s + i] = rt[pick]
            resid_set[s + i] = rs[pick]
            resid_count[s + i] = rc[pick]
            resid_pt[s + i] = rpt[pick]
            resid_mass[s + i] = rms[pick]
            round_found[s + i] = (pick // max(1, int(per_round)) + 1).astype(np.float32)
            feasible[s + i] = feas_mask[pick]

    return {
        "const": cand_const,
        "mask": cand_mask,
        "hlt_const": cand_hlt_const,
        "hlt_mask": cand_hlt_mask,
        "gid": cand_gid,
        "res_total": resid_total,
        "res_total_pre": resid_total.copy(),
        "res_set": resid_set,
        "res_set_pre": resid_set.copy(),
        "res_count": resid_count,
        "res_count_pre": resid_count.copy(),
        "res_pt": resid_pt,
        "res_pt_pre": resid_pt.copy(),
        "res_mass": resid_mass,
        "res_mass_pre": resid_mass.copy(),
        "round_found": round_found,
        "feasible": feasible,
        "feasible_count": feasible_count,
    }


def _retrieve_pools_split(
    query_index_desc: np.ndarray,
    query_score_desc: np.ndarray,
    query_meta: JetMeta,
    query_const_hlt: np.ndarray,
    query_mask_hlt: np.ndarray,
    idx_bg: RetrievalClassIndex,
    idx_top: RetrievalClassIndex,
    dict_score_desc: np.ndarray,
    dict_const_off: np.ndarray,
    dict_mask_off: np.ndarray,
    dict_const_hlt: np.ndarray,
    dict_mask_hlt: np.ndarray,
    dict_meta: JetMeta,
    target_k: int,
    per_round: int,
    max_rounds: int,
    eps_total: float,
    eps_count: float,
    w_desc: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    max_constits: int,
    batch_size: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    c0 = _retrieve_pool_for_class(
        query_index_desc=query_index_desc,
        query_score_desc=query_score_desc,
        query_meta=query_meta,
        query_const_hlt=query_const_hlt,
        query_mask_hlt=query_mask_hlt,
        class_index=idx_bg,
        dict_score_desc=dict_score_desc,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_const_hlt=dict_const_hlt,
        dict_mask_hlt=dict_mask_hlt,
        dict_meta=dict_meta,
        target_k=int(target_k),
        per_round=int(per_round),
        max_rounds=int(max_rounds),
        eps_total=float(eps_total),
        eps_count=float(eps_count),
        w_desc=float(w_desc),
        w_count=float(w_count),
        w_pt=float(w_pt),
        w_mass=float(w_mass),
        max_constits=int(max_constits),
        batch_size=int(batch_size),
    )
    c1 = _retrieve_pool_for_class(
        query_index_desc=query_index_desc,
        query_score_desc=query_score_desc,
        query_meta=query_meta,
        query_const_hlt=query_const_hlt,
        query_mask_hlt=query_mask_hlt,
        class_index=idx_top,
        dict_score_desc=dict_score_desc,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_const_hlt=dict_const_hlt,
        dict_mask_hlt=dict_mask_hlt,
        dict_meta=dict_meta,
        target_k=int(target_k),
        per_round=int(per_round),
        max_rounds=int(max_rounds),
        eps_total=float(eps_total),
        eps_count=float(eps_count),
        w_desc=float(w_desc),
        w_count=float(w_count),
        w_pt=float(w_pt),
        w_mass=float(w_mass),
        max_constits=int(max_constits),
        batch_size=int(batch_size),
    )

    return {
        "class0": c0,
        "class1": c1,
        "stats": {
            "target_k": int(target_k),
            "class0_mean_feasible": float(np.mean(np.minimum(c0["feasible_count"], int(target_k)))),
            "class1_mean_feasible": float(np.mean(np.minimum(c1["feasible_count"], int(target_k)))),
        },
    }


def _build_m37_multicandidate_arrays(
    pools: Dict[str, Dict[str, np.ndarray]],
    sel_score_bg: np.ndarray,
    sel_score_top: np.ndarray,
    score_alpha: float,
    max_rounds: int,
) -> Dict[str, np.ndarray]:
    c0 = pools["class0"]
    c1 = pools["class1"]
    n, k, _l, _ = c0["const"].shape

    q_bg = sel_score_bg - float(score_alpha) * c0["res_total"]
    q_tp = sel_score_top - float(score_alpha) * c1["res_total"]

    denom_rank = float(max(1, k - 1))
    rank_bg = np.argsort(np.argsort(c0["res_total"], axis=1), axis=1).astype(np.float32) / denom_rank
    rank_tp = np.argsort(np.argsort(c1["res_total"], axis=1), axis=1).astype(np.float32) / denom_rank
    round_bg = c0["round_found"].astype(np.float32) / float(max(1, max_rounds + 1))
    round_tp = c1["round_found"].astype(np.float32) / float(max(1, max_rounds + 1))

    meta_bg = np.stack(
        [
            c0["res_total"],
            c0["res_set"],
            c0["res_count"],
            c0["res_pt"],
            c0["res_mass"],
            sel_score_bg,
            q_bg,
            c0["feasible"].astype(np.float32),
            round_bg,
            rank_bg,
            np.zeros_like(c0["res_total"], dtype=np.float32),
            np.ones_like(c0["res_total"], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    meta_tp = np.stack(
        [
            c1["res_total"],
            c1["res_set"],
            c1["res_count"],
            c1["res_pt"],
            c1["res_mass"],
            sel_score_top,
            q_tp,
            c1["feasible"].astype(np.float32),
            round_tp,
            rank_tp,
            np.ones_like(c1["res_total"], dtype=np.float32),
            np.zeros_like(c1["res_total"], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    cand_off_const = np.concatenate([c1["const"], c0["const"]], axis=1).astype(np.float32)
    cand_off_mask = np.concatenate([c1["mask"], c0["mask"]], axis=1).astype(bool)
    cand_hlt_const = np.concatenate([c1["hlt_const"], c0["hlt_const"]], axis=1).astype(np.float32)
    cand_hlt_mask = np.concatenate([c1["hlt_mask"], c0["hlt_mask"]], axis=1).astype(bool)
    cand_meta = np.concatenate([meta_tp, meta_bg], axis=1).astype(np.float32)
    cand_class = np.concatenate(
        [np.ones((n, k), dtype=np.float32), np.zeros((n, k), dtype=np.float32)],
        axis=1,
    ).astype(np.float32)

    cand_tokens = _build_candidate_token12_np(
        off_const=cand_off_const,
        off_mask=cand_off_mask,
        hlt_const=cand_hlt_const,
        hlt_mask=cand_hlt_mask,
    )
    cand_masks = (cand_off_mask | cand_hlt_mask).astype(bool)
    empty = ~cand_masks.any(axis=2)
    if np.any(empty):
        cand_masks[empty, 0] = True
    cand_tokens[~cand_masks] = 0.0

    feas_tp = np.minimum(c1["feasible_count"], float(k)) / float(max(1, k))
    feas_bg = np.minimum(c0["feasible_count"], float(k)) / float(max(1, k))
    best_tp = np.min(c1["res_total"], axis=1).astype(np.float32)
    best_bg = np.min(c0["res_total"], axis=1).astype(np.float32)
    mean_tp = np.mean(c1["res_total"], axis=1).astype(np.float32)
    mean_bg = np.mean(c0["res_total"], axis=1).astype(np.float32)
    std_tp = np.std(c1["res_total"], axis=1).astype(np.float32)
    std_bg = np.std(c0["res_total"], axis=1).astype(np.float32)
    qmax_tp = np.max(q_tp, axis=1).astype(np.float32)
    qmax_bg = np.max(q_bg, axis=1).astype(np.float32)
    selmax_tp = np.max(sel_score_top, axis=1).astype(np.float32)
    selmax_bg = np.max(sel_score_bg, axis=1).astype(np.float32)
    roundmin_tp = np.min(c1["round_found"], axis=1).astype(np.float32) / float(max(1, max_rounds + 1))
    roundmin_bg = np.min(c0["round_found"], axis=1).astype(np.float32) / float(max(1, max_rounds + 1))
    spread_all = np.std(np.concatenate([c1["res_total"], c0["res_total"]], axis=1), axis=1).astype(np.float32)

    summary_feat = np.stack(
        [
            best_tp,
            best_bg,
            best_bg - best_tp,
            mean_tp,
            mean_bg,
            std_tp,
            std_bg,
            spread_all,
            feas_tp.astype(np.float32),
            feas_bg.astype(np.float32),
            roundmin_tp,
            roundmin_bg,
            qmax_tp,
            qmax_bg,
            selmax_tp,
            selmax_bg,
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "cand_tokens": cand_tokens.astype(np.float32),
        "cand_masks": cand_masks.astype(bool),
        "cand_meta": cand_meta.astype(np.float32),
        "cand_class": cand_class.astype(np.float32),
        "summary_feat": summary_feat.astype(np.float32),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m37 dictionary-retrieval multicandidate dualview top tagging")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model37_multicand_dualview")
    p.add_argument("--run_name", type=str, default="model37_multicand_dualview_1m150k75k300k_seed0")

    p.add_argument("--n_train_jets", type=int, default=1525000)
    p.add_argument("--n_dict_split", type=int, default=1000000)
    p.add_argument("--n_train_split", type=int, default=150000)
    p.add_argument("--n_val_split", type=int, default=75000)
    p.add_argument("--n_test_split", type=int, default=300000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--use_train_weights", action="store_true")

    # HLT effects (deterministic keyed D_hard)
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

    # Retrieval candidate stage
    p.add_argument("--retrieval_target_k", type=int, default=3)
    p.add_argument("--retrieval_per_round", type=int, default=256)
    p.add_argument("--retrieval_max_rounds", type=int, default=10)
    p.add_argument("--retrieval_batch_size", type=int, default=256)
    p.add_argument("--retrieval_eps_total", type=float, default=0.90)
    p.add_argument("--retrieval_eps_count", type=float, default=0.50)
    p.add_argument("--retrieval_w_desc", type=float, default=1.00)
    p.add_argument("--retrieval_w_count", type=float, default=0.25)
    p.add_argument("--retrieval_w_pt", type=float, default=0.12)
    p.add_argument("--retrieval_w_mass", type=float, default=0.08)
    p.add_argument("--retrieval_descriptor_variant", type=str, default="physics", choices=["physics", "landmark11"])
    p.add_argument("--retrieval_landmarks", type=int, default=8)
    p.add_argument("--retrieval_index_mode", type=str, default="descriptor", choices=["descriptor", "learned"])
    p.add_argument("--retrieval_embed_dim", type=int, default=16)
    p.add_argument("--retrieval_embed_hidden", type=int, default=96)
    p.add_argument("--retrieval_embed_dropout", type=float, default=0.10)
    p.add_argument("--retrieval_embed_epochs", type=int, default=6)
    p.add_argument("--retrieval_embed_batch_size", type=int, default=512)
    p.add_argument("--retrieval_embed_pool_size", type=int, default=256)
    p.add_argument("--retrieval_embed_train_anchors", type=int, default=60000)
    p.add_argument("--retrieval_embed_lr", type=float, default=3e-4)
    p.add_argument("--retrieval_embed_weight_decay", type=float, default=1e-4)
    p.add_argument("--retrieval_embed_margin", type=float, default=0.20)

    # Selector
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--selector_epochs", type=int, default=40)
    p.add_argument("--selector_patience", type=int, default=8)
    p.add_argument("--selector_lr", type=float, default=2e-4)
    p.add_argument("--selector_weight_decay", type=float, default=1e-4)
    p.add_argument("--selector_neg_per_class", type=int, default=3)
    p.add_argument("--selector_score_alpha", type=float, default=1.25)
    p.add_argument("--selector_mode", type=str, default="compat_selector", choices=["compat_selector", "residual_only"])

    # Final dualview classifiers
    p.add_argument("--dual_epochs", type=int, default=60)
    p.add_argument("--dual_patience", type=int, default=12)
    p.add_argument("--dual_lr", type=float, default=1.5e-4)
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
    print("Model-37 Dictionary Retrieval + Multi-Candidate DualView Pipeline")
    print(f"Run: {save_root}")
    print(
        f"Split dict/train/val/test = {int(args.n_dict_split)}/{int(args.n_train_split)}/{int(args.n_val_split)}/{int(args.n_test_split)} | "
        f"retrieve per-round={int(args.retrieval_per_round)} rounds={int(args.retrieval_max_rounds)} targetK={int(args.retrieval_target_k)} | "
        f"desc_variant={str(args.retrieval_descriptor_variant)} index_mode={str(args.retrieval_index_mode)}"
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

    total_need = int(args.n_dict_split + args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > const_raw.shape[0]:
        raise RuntimeError(
            f"Requested splits sum to {total_need} but loaded jets are only {const_raw.shape[0]}"
        )

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
    const_hlt, mask_hlt, hlt_stats = m33._apply_hlt_effects_deterministic_keyed(
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

    # Split: dictionary disjoint from train/val/test.
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

    dict_idx, rem_idx = train_test_split(
        idx_use,
        train_size=int(args.n_dict_split),
        random_state=int(args.seed),
        stratify=labels[idx_use],
    )
    train_idx, rem2_idx = train_test_split(
        rem_idx,
        train_size=int(args.n_train_split),
        random_state=int(args.seed),
        stratify=labels[rem_idx],
    )
    val_idx, test_idx = train_test_split(
        rem2_idx,
        train_size=int(args.n_val_split),
        test_size=int(args.n_test_split),
        random_state=int(args.seed),
        stratify=labels[rem2_idx],
    )

    print(
        f"Split sizes: Dict={len(dict_idx)}, Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )

    # Build features only for train/val/test (dictionary is retrieval only).
    print("Computing standardized features for teacher/baseline (train/val/test only)...")
    feat_off_tr = compute_features(const_off[train_idx], masks_off[train_idx])
    feat_off_va = compute_features(const_off[val_idx], masks_off[val_idx])
    feat_off_te = compute_features(const_off[test_idx], masks_off[test_idx])
    feat_hlt_tr = compute_features(const_hlt[train_idx], mask_hlt[train_idx])
    feat_hlt_va = compute_features(const_hlt[val_idx], mask_hlt[val_idx])
    feat_hlt_te = compute_features(const_hlt[test_idx], mask_hlt[test_idx])

    tr_local_idx = np.arange(feat_off_tr.shape[0], dtype=np.int64)
    means, stds = get_stats(feat_off_tr, masks_off[train_idx], tr_local_idx)

    feat_off_tr = standardize(feat_off_tr, masks_off[train_idx], means, stds)
    feat_off_va = standardize(feat_off_va, masks_off[val_idx], means, stds)
    feat_off_te = standardize(feat_off_te, masks_off[test_idx], means, stds)
    feat_hlt_tr = standardize(feat_hlt_tr, mask_hlt[train_idx], means, stds)
    feat_hlt_va = standardize(feat_hlt_va, mask_hlt[val_idx], means, stds)
    feat_hlt_te = standardize(feat_hlt_te, mask_hlt[test_idx], means, stds)

    # STEP 1: teacher and baseline classifiers.
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

    ds_tr_off = base.WeightedJetDataset(feat_off_tr, masks_off[train_idx], labels[train_idx], sw_train)
    ds_va_off = base.WeightedJetDataset(feat_off_va, masks_off[val_idx], labels[val_idx], sw_val)
    ds_te_off = base.WeightedJetDataset(feat_off_te, masks_off[test_idx], labels[test_idx], sw_test)

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

    # STEP 2: Build dictionary retrieval index (class-conditional true labels).
    print("\n" + "=" * 72)
    print("STEP 2: Dictionary HLT retrieval index")
    print("=" * 72)

    dict_const_off = const_off[dict_idx].astype(np.float32)
    dict_mask_off = masks_off[dict_idx].astype(bool)
    dict_const_hlt = const_hlt[dict_idx].astype(np.float32)
    dict_mask_hlt = mask_hlt[dict_idx].astype(bool)
    dict_labels = labels[dict_idx].astype(np.int64)

    dict_meta = _jet_meta_np(dict_const_hlt, dict_mask_hlt)
    tr_meta = _jet_meta_np(const_hlt[train_idx], mask_hlt[train_idx])
    va_meta = _jet_meta_np(const_hlt[val_idx], mask_hlt[val_idx])
    te_meta = _jet_meta_np(const_hlt[test_idx], mask_hlt[test_idx])

    desc_dict_raw = _build_retrieval_desc(dict_const_hlt, dict_mask_hlt, dict_meta, int(args.max_constits))
    desc_tr_raw = _build_retrieval_desc(const_hlt[train_idx], mask_hlt[train_idx], tr_meta, int(args.max_constits))
    desc_va_raw = _build_retrieval_desc(const_hlt[val_idx], mask_hlt[val_idx], va_meta, int(args.max_constits))
    desc_te_raw = _build_retrieval_desc(const_hlt[test_idx], mask_hlt[test_idx], te_meta, int(args.max_constits))

    desc_mean = desc_dict_raw.mean(axis=0, keepdims=True)
    desc_std = desc_dict_raw.std(axis=0, keepdims=True) + 1e-6

    desc_dict = ((desc_dict_raw - desc_mean) / desc_std).astype(np.float32)
    desc_tr = ((desc_tr_raw - desc_mean) / desc_std).astype(np.float32)
    desc_va = ((desc_va_raw - desc_mean) / desc_std).astype(np.float32)
    desc_te = ((desc_te_raw - desc_mean) / desc_std).astype(np.float32)

    retrieval_variant = str(args.retrieval_descriptor_variant).lower()
    landmark_ids = np.zeros((0,), dtype=np.int64)
    landmark_desc = np.zeros((0, desc_dict.shape[1]), dtype=np.float32)
    landmark_meta: Dict[str, object] = {"variant": retrieval_variant}
    if retrieval_variant == "landmark11":
        n_landmarks = int(max(1, args.retrieval_landmarks))
        landmark_ids = _select_landmark_indices(desc_dict, n_landmarks=n_landmarks, seed=int(args.seed) + 173)
        landmark_desc = desc_dict[landmark_ids].astype(np.float32)
        retrieval_score_dict = _build_landmark11_desc(desc_dict, dict_meta, landmark_desc, int(args.max_constits))
        retrieval_score_tr = _build_landmark11_desc(desc_tr, tr_meta, landmark_desc, int(args.max_constits))
        retrieval_score_va = _build_landmark11_desc(desc_va, va_meta, landmark_desc, int(args.max_constits))
        retrieval_score_te = _build_landmark11_desc(desc_te, te_meta, landmark_desc, int(args.max_constits))
        landmark_meta = {
            "variant": "landmark11",
            "n_landmarks": int(landmark_desc.shape[0]),
            "landmark_ids": landmark_ids.astype(np.int64).tolist(),
        }
        print(f"Landmark11 descriptor enabled: n_landmarks={int(landmark_desc.shape[0])}")
    else:
        retrieval_score_dict = desc_dict
        retrieval_score_tr = desc_tr
        retrieval_score_va = desc_va
        retrieval_score_te = desc_te

    score_mean = retrieval_score_dict.mean(axis=0, keepdims=True).astype(np.float32)
    score_std = (retrieval_score_dict.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    retrieval_embedder = None
    retrieval_embed_metrics: Dict[str, object] = {}
    idx_dict_desc = retrieval_score_dict
    idx_tr_desc = retrieval_score_tr
    idx_va_desc = retrieval_score_va
    idx_te_desc = retrieval_score_te

    if str(args.retrieval_index_mode).lower() == "learned":
        print("Training learned retrieval embedding index...")
        retrieval_embedder, retrieval_embed_metrics = _train_retrieval_embedder(
            desc_train=retrieval_score_tr,
            meta_train=tr_meta,
            labels_train=labels[train_idx],
            desc_dict=retrieval_score_dict,
            meta_dict=dict_meta,
            dict_labels=dict_labels,
            max_constits=int(args.max_constits),
            w_desc=float(args.retrieval_w_desc),
            w_count=float(args.retrieval_w_count),
            w_pt=float(args.retrieval_w_pt),
            w_mass=float(args.retrieval_w_mass),
            seed=int(args.seed),
            device=device,
            embed_dim=int(args.retrieval_embed_dim),
            hidden_dim=int(args.retrieval_embed_hidden),
            dropout=float(args.retrieval_embed_dropout),
            epochs=int(args.retrieval_embed_epochs),
            batch_size=int(args.retrieval_embed_batch_size),
            pool_size=int(args.retrieval_embed_pool_size),
            train_anchors=int(args.retrieval_embed_train_anchors),
            lr=float(args.retrieval_embed_lr),
            weight_decay=float(args.retrieval_embed_weight_decay),
            margin=float(args.retrieval_embed_margin),
        )
        idx_dict_desc = _encode_retrieval_desc(
            retrieval_embedder,
            retrieval_score_dict,
            batch_size=int(args.retrieval_batch_size),
            device=device,
        )
        idx_tr_desc = _encode_retrieval_desc(
            retrieval_embedder,
            retrieval_score_tr,
            batch_size=int(args.retrieval_batch_size),
            device=device,
        )
        idx_va_desc = _encode_retrieval_desc(
            retrieval_embedder,
            retrieval_score_va,
            batch_size=int(args.retrieval_batch_size),
            device=device,
        )
        idx_te_desc = _encode_retrieval_desc(
            retrieval_embedder,
            retrieval_score_te,
            batch_size=int(args.retrieval_batch_size),
            device=device,
        )
    else:
        retrieval_embed_metrics = {"mode": "descriptor", "epochs": 0, "descriptor_variant": retrieval_variant}

    idx_bg = _build_class_index(idx_dict_desc, dict_labels, class_val=0)
    idx_top = _build_class_index(idx_dict_desc, dict_labels, class_val=1)
    print(
        f"Dictionary class sizes: bg={idx_bg.local_ids.shape[0]} top={idx_top.local_ids.shape[0]} "
        f"| index_dim={idx_dict_desc.shape[1]}"
    )

    # STEP 3: Retrieve candidates for train/val.
    print("\n" + "=" * 72)
    print("STEP 3: Retrieve candidates for train/val (no repair)")
    print("=" * 72)

    pools_train = _retrieve_pools_split(
        query_index_desc=idx_tr_desc,
        query_score_desc=retrieval_score_tr,
        query_meta=tr_meta,
        query_const_hlt=const_hlt[train_idx],
        query_mask_hlt=mask_hlt[train_idx],
        idx_bg=idx_bg,
        idx_top=idx_top,
        dict_score_desc=retrieval_score_dict,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_const_hlt=dict_const_hlt,
        dict_mask_hlt=dict_mask_hlt,
        dict_meta=dict_meta,
        target_k=int(args.retrieval_target_k),
        per_round=int(args.retrieval_per_round),
        max_rounds=int(args.retrieval_max_rounds),
        eps_total=float(args.retrieval_eps_total),
        eps_count=float(args.retrieval_eps_count),
        w_desc=float(args.retrieval_w_desc),
        w_count=float(args.retrieval_w_count),
        w_pt=float(args.retrieval_w_pt),
        w_mass=float(args.retrieval_w_mass),
        max_constits=int(args.max_constits),
        batch_size=int(args.retrieval_batch_size),
    )
    pools_val = _retrieve_pools_split(
        query_index_desc=idx_va_desc,
        query_score_desc=retrieval_score_va,
        query_meta=va_meta,
        query_const_hlt=const_hlt[val_idx],
        query_mask_hlt=mask_hlt[val_idx],
        idx_bg=idx_bg,
        idx_top=idx_top,
        dict_score_desc=retrieval_score_dict,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_const_hlt=dict_const_hlt,
        dict_mask_hlt=dict_mask_hlt,
        dict_meta=dict_meta,
        target_k=int(args.retrieval_target_k),
        per_round=int(args.retrieval_per_round),
        max_rounds=int(args.retrieval_max_rounds),
        eps_total=float(args.retrieval_eps_total),
        eps_count=float(args.retrieval_eps_count),
        w_desc=float(args.retrieval_w_desc),
        w_count=float(args.retrieval_w_count),
        w_pt=float(args.retrieval_w_pt),
        w_mass=float(args.retrieval_w_mass),
        max_constits=int(args.max_constits),
        batch_size=int(args.retrieval_batch_size),
    )

    print(
        "Retrieval stats: "
        f"train(feasC0={pools_train['stats']['class0_mean_feasible']:.2f}, feasC1={pools_train['stats']['class1_mean_feasible']:.2f}) "
        f"val(feasC0={pools_val['stats']['class0_mean_feasible']:.2f}, feasC1={pools_val['stats']['class1_mean_feasible']:.2f})"
    )

    # STEP 4: Candidate compatibility selector (optional).
    print("\n" + "=" * 72)
    if str(args.selector_mode).lower() == "compat_selector":
        print("STEP 4: Train HLT-candidate compatibility selector")
    else:
        print("STEP 4: Residual-only candidate quality (selector skipped)")
    print("=" * 72)

    selector = None
    if str(args.selector_mode).lower() == "compat_selector":
        sel_tr = m33._build_selector_arrays(
            const_hlt=const_hlt[train_idx],
            mask_hlt=mask_hlt[train_idx],
            const_off=const_off[train_idx],
            mask_off=masks_off[train_idx],
            pools=pools_train,
            neg_per_class=int(args.selector_neg_per_class),
        )
        sel_va = m33._build_selector_arrays(
            const_hlt=const_hlt[val_idx],
            mask_hlt=mask_hlt[val_idx],
            const_off=const_off[val_idx],
            mask_off=masks_off[val_idx],
            pools=pools_val,
            neg_per_class=int(args.selector_neg_per_class),
        )

        ds_sel_tr = m33.SelectorDataset(**sel_tr)
        ds_sel_va = m33.SelectorDataset(**sel_va)
        dl_sel_tr = DataLoader(ds_sel_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
        dl_sel_va = DataLoader(ds_sel_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

        selector = m33.CandidateRealismSelector(
            embed_dim=int(args.embed_dim),
            num_heads=int(args.num_heads),
            num_layers=max(2, int(args.num_layers // 2)),
            ff_dim=int(args.ff_dim),
            dropout=float(args.dropout),
        ).to(device)
        selector, selector_metrics = m33._train_selector(
            model=selector,
            train_loader=dl_sel_tr,
            val_loader=dl_sel_va,
            device=device,
            epochs=int(args.selector_epochs),
            lr=float(args.selector_lr),
            weight_decay=float(args.selector_weight_decay),
            patience=int(args.selector_patience),
        )

        sel_bg_train = m33._score_selector_candidates(
            selector,
            const_hlt[train_idx],
            mask_hlt[train_idx],
            pools_train["class0"]["const"],
            pools_train["class0"]["mask"],
            pools_train["class0"]["res_total"],
            cand_class_val=0,
            batch_size=int(args.batch_size),
            device=device,
        )
        sel_tp_train = m33._score_selector_candidates(
            selector,
            const_hlt[train_idx],
            mask_hlt[train_idx],
            pools_train["class1"]["const"],
            pools_train["class1"]["mask"],
            pools_train["class1"]["res_total"],
            cand_class_val=1,
            batch_size=int(args.batch_size),
            device=device,
        )
        sel_bg_val = m33._score_selector_candidates(
            selector,
            const_hlt[val_idx],
            mask_hlt[val_idx],
            pools_val["class0"]["const"],
            pools_val["class0"]["mask"],
            pools_val["class0"]["res_total"],
            cand_class_val=0,
            batch_size=int(args.batch_size),
            device=device,
        )
        sel_tp_val = m33._score_selector_candidates(
            selector,
            const_hlt[val_idx],
            mask_hlt[val_idx],
            pools_val["class1"]["const"],
            pools_val["class1"]["mask"],
            pools_val["class1"]["res_total"],
            cand_class_val=1,
            batch_size=int(args.batch_size),
            device=device,
        )
    else:
        selector_metrics = {
            "mode": "residual_only",
            "skipped": 1,
            "best_val_auc": float("nan"),
            "best_epoch": 0,
        }
        sel_bg_train = np.zeros_like(pools_train["class0"]["res_total"], dtype=np.float32)
        sel_tp_train = np.zeros_like(pools_train["class1"]["res_total"], dtype=np.float32)
        sel_bg_val = np.zeros_like(pools_val["class0"]["res_total"], dtype=np.float32)
        sel_tp_val = np.zeros_like(pools_val["class1"]["res_total"], dtype=np.float32)

    # STEP 5: Final multicandidate dualview classifiers.
    print("\n" + "=" * 72)
    print("STEP 5: Final Multi-Candidate DualView Classifiers (NoGate + Gated)")
    print("=" * 72)

    mv_tr = _build_m37_multicandidate_arrays(
        pools=pools_train,
        sel_score_bg=sel_bg_train,
        sel_score_top=sel_tp_train,
        score_alpha=float(args.selector_score_alpha),
        max_rounds=int(args.retrieval_max_rounds),
    )
    mv_va = _build_m37_multicandidate_arrays(
        pools=pools_val,
        sel_score_bg=sel_bg_val,
        sel_score_top=sel_tp_val,
        score_alpha=float(args.selector_score_alpha),
        max_rounds=int(args.retrieval_max_rounds),
    )

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

    ds_dv_tr = DualViewM37Dataset(
        feat_hlt=feat_hlt_tr,
        mask_hlt=mask_hlt[train_idx],
        cand_tokens=mv_tr["cand_tokens"],
        cand_masks=mv_tr["cand_masks"],
        cand_meta=mv_tr["cand_meta"],
        cand_class=mv_tr["cand_class"],
        summary_feat=mv_tr["summary_feat"],
        labels=labels[train_idx],
        sample_weight=sw_train,
    )
    ds_dv_va = DualViewM37Dataset(
        feat_hlt=feat_hlt_va,
        mask_hlt=mask_hlt[val_idx],
        cand_tokens=mv_va["cand_tokens"],
        cand_masks=mv_va["cand_masks"],
        cand_meta=mv_va["cand_meta"],
        cand_class=mv_va["cand_class"],
        summary_feat=mv_va["summary_feat"],
        labels=labels[val_idx],
        sample_weight=sw_val,
    )

    dl_dv_tr = DataLoader(ds_dv_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_dv_va = DataLoader(ds_dv_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    m37_nogate = MultiCandidateNoGate(
        cand_meta_dim=int(mv_tr["cand_meta"].shape[-1]),
        summary_dim=int(mv_tr["summary_feat"].shape[-1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    m37_nogate, m37_nogate_metrics = _train_m37_model(
        model=m37_nogate,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="M37NoGate",
    )

    m37_gated = MultiCandidateGated(
        cand_meta_dim=int(mv_tr["cand_meta"].shape[-1]),
        summary_dim=int(mv_tr["summary_feat"].shape[-1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    m37_gated, m37_gated_metrics = _train_m37_model(
        model=m37_gated,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="M37Gated",
    )

    # STEP 6: Build test pools and evaluate final models.
    print("\n" + "=" * 72)
    print("STEP 6: Test retrieval + final evaluation")
    print("=" * 72)

    pools_test = _retrieve_pools_split(
        query_index_desc=idx_te_desc,
        query_score_desc=retrieval_score_te,
        query_meta=te_meta,
        query_const_hlt=const_hlt[test_idx],
        query_mask_hlt=mask_hlt[test_idx],
        idx_bg=idx_bg,
        idx_top=idx_top,
        dict_score_desc=retrieval_score_dict,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_const_hlt=dict_const_hlt,
        dict_mask_hlt=dict_mask_hlt,
        dict_meta=dict_meta,
        target_k=int(args.retrieval_target_k),
        per_round=int(args.retrieval_per_round),
        max_rounds=int(args.retrieval_max_rounds),
        eps_total=float(args.retrieval_eps_total),
        eps_count=float(args.retrieval_eps_count),
        w_desc=float(args.retrieval_w_desc),
        w_count=float(args.retrieval_w_count),
        w_pt=float(args.retrieval_w_pt),
        w_mass=float(args.retrieval_w_mass),
        max_constits=int(args.max_constits),
        batch_size=int(args.retrieval_batch_size),
    )

    if selector is not None:
        sel_bg_test = m33._score_selector_candidates(
            selector,
            const_hlt[test_idx],
            mask_hlt[test_idx],
            pools_test["class0"]["const"],
            pools_test["class0"]["mask"],
            pools_test["class0"]["res_total"],
            cand_class_val=0,
            batch_size=int(args.batch_size),
            device=device,
        )
        sel_tp_test = m33._score_selector_candidates(
            selector,
            const_hlt[test_idx],
            mask_hlt[test_idx],
            pools_test["class1"]["const"],
            pools_test["class1"]["mask"],
            pools_test["class1"]["res_total"],
            cand_class_val=1,
            batch_size=int(args.batch_size),
            device=device,
        )
    else:
        sel_bg_test = np.zeros_like(pools_test["class0"]["res_total"], dtype=np.float32)
        sel_tp_test = np.zeros_like(pools_test["class1"]["res_total"], dtype=np.float32)

    mv_te = _build_m37_multicandidate_arrays(
        pools=pools_test,
        sel_score_bg=sel_bg_test,
        sel_score_top=sel_tp_test,
        score_alpha=float(args.selector_score_alpha),
        max_rounds=int(args.retrieval_max_rounds),
    )
    mv_te["cand_meta"] = ((mv_te["cand_meta"] - mm) / ms).astype(np.float32)
    mv_te["summary_feat"] = ((mv_te["summary_feat"] - sm) / ss).astype(np.float32)

    ds_dv_te = DualViewM37Dataset(
        feat_hlt=feat_hlt_te,
        mask_hlt=mask_hlt[test_idx],
        cand_tokens=mv_te["cand_tokens"],
        cand_masks=mv_te["cand_masks"],
        cand_meta=mv_te["cand_meta"],
        cand_class=mv_te["cand_class"],
        summary_feat=mv_te["summary_feat"],
        labels=labels[test_idx],
        sample_weight=sw_test,
    )
    dl_dv_te = DataLoader(ds_dv_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    auc_nog, fpr50_nog, pred_nog, lab_final, w_final = _eval_m37_model(m37_nogate, dl_dv_te, device)
    auc_gat, fpr50_gat, pred_gat, _lab2, _w2 = _eval_m37_model(m37_gated, dl_dv_te, device)

    fpr_t, tpr_t, _ = roc_curve(teacher_y_test, teacher_p_test, sample_weight=teacher_w_test)
    fpr_b, tpr_b, _ = roc_curve(baseline_y_test, baseline_p_test, sample_weight=baseline_w_test)
    fpr50_teacher = m33.fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_baseline = m33.fpr_at_target_tpr(fpr_b, tpr_b, 0.50)

    print("\n" + "=" * 72)
    print("FINAL TEST")
    print("=" * 72)
    print(
        f"Teacher AUC={teacher_auc_test:.4f} FPR50={fpr50_teacher:.6f} | "
        f"HLT baseline AUC={baseline_auc_test:.4f} FPR50={fpr50_baseline:.6f} | "
        f"m37 NoGate AUC={auc_nog:.4f} FPR50={fpr50_nog:.6f} | "
        f"m37 Gated AUC={auc_gat:.4f} FPR50={fpr50_gat:.6f}"
    )

    # Save artifacts.
    torch.save({"model": teacher.state_dict(), "auc_test": float(teacher_auc_test)}, save_root / "teacher.pt")
    torch.save({"model": baseline.state_dict(), "auc_test": float(baseline_auc_test)}, save_root / "baseline_hlt.pt")
    if selector is not None:
        torch.save({"model": selector.state_dict(), "metrics": selector_metrics}, save_root / "selector.pt")
    torch.save({"model": m37_nogate.state_dict(), "metrics": m37_nogate_metrics}, save_root / "multicand_nogate.pt")
    torch.save({"model": m37_gated.state_dict(), "metrics": m37_gated_metrics}, save_root / "multicand_gated.pt")
    if retrieval_embedder is not None:
        torch.save(
            {
                "model": retrieval_embedder.state_dict(),
                "metrics": retrieval_embed_metrics,
                "index_mode": str(args.retrieval_index_mode),
                "descriptor_variant": str(args.retrieval_descriptor_variant),
            },
            save_root / "retrieval_embedder.pt",
        )

    np.savez_compressed(
        save_root / "m37_test_scores.npz",
        labels_test=lab_final.astype(np.float32),
        preds_m37_nogate=pred_nog.astype(np.float32),
        preds_m37_gated=pred_gat.astype(np.float32),
        preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
        preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
        sample_weight=np.asarray(w_final, dtype=np.float32),
        auc_teacher=float(teacher_auc_test),
        auc_hlt=float(baseline_auc_test),
        auc_m37_nogate=float(auc_nog),
        auc_m37_gated=float(auc_gat),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_hlt=float(fpr50_baseline),
        fpr50_m37_nogate=float(fpr50_nog),
        fpr50_m37_gated=float(fpr50_gat),
    )

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "fusion_scores_test.npz",
            labels_test=lab_final.astype(np.float32),
            preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
            preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
            preds_m37_nogate=np.asarray(pred_nog, dtype=np.float32),
            preds_m37_gated=np.asarray(pred_gat, dtype=np.float32),
            sample_weight=np.asarray(w_final, dtype=np.float32),
        )

    report = {
        "model": "m37_multicand_dualview",
        "seed": int(args.seed),
        "n_train_jets": int(args.n_train_jets),
        "split": {
            "dictionary": int(len(dict_idx)),
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
        "m37_nogate": {
            "auc_test": float(auc_nog),
            "fpr50_test": float(fpr50_nog),
            "metrics": m37_nogate_metrics,
        },
        "m37_gated": {
            "auc_test": float(auc_gat),
            "fpr50_test": float(fpr50_gat),
            "metrics": m37_gated_metrics,
        },
        "selector_metrics": selector_metrics,
        "retrieval": {
            "index_mode": str(args.retrieval_index_mode),
            "descriptor_variant": str(args.retrieval_descriptor_variant),
            "target_k": int(args.retrieval_target_k),
            "per_round": int(args.retrieval_per_round),
            "max_rounds": int(args.retrieval_max_rounds),
            "eps_total": float(args.retrieval_eps_total),
            "eps_count": float(args.retrieval_eps_count),
            "weights": {
                "desc": float(args.retrieval_w_desc),
                "count": float(args.retrieval_w_count),
                "pt": float(args.retrieval_w_pt),
                "mass": float(args.retrieval_w_mass),
            },
            "train": pools_train["stats"],
            "val": pools_val["stats"],
            "test": pools_test["stats"],
            "dict_class_counts": {
                "bg": int(idx_bg.local_ids.shape[0]),
                "top": int(idx_top.local_ids.shape[0]),
            },
            "embedder_metrics": retrieval_embed_metrics,
            "landmark_meta": landmark_meta,
        },
        "selector_mode": str(args.selector_mode),
    }
    with open(save_root / "m37_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        save_root / "data_splits.npz",
        dict_idx=dict_idx.astype(np.int64),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
        retrieval_desc_mean=desc_mean.astype(np.float32),
        retrieval_desc_std=desc_std.astype(np.float32),
        retrieval_score_mean=score_mean.astype(np.float32),
        retrieval_score_std=score_std.astype(np.float32),
        retrieval_landmark_ids=landmark_ids.astype(np.int64),
        retrieval_landmark_desc=landmark_desc.astype(np.float32),
    )

    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()

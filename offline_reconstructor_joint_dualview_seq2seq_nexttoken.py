#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Encoder-decoder continuous autoregressive reconstructor + dual-view top tagging.

Key ideas:
- HLT tokens are encoded with a transformer encoder.
- Offline constituents are decoded autoregressively in a continuous token space.
- Decoder uses pointer-style copy from encoded HLT tokens plus learned residual edits.
- Loss = autoregressive Huber + set-level Chamfer + EOS/count supervision.
- Inference uses beam over sequence length hypotheses (EOS trajectories) and a
  confidence-weighted soft view to build reconstructed jets.

This script intentionally keeps the same high-level plumbing as the prior m2-style
pipeline: data loading, pseudo-HLT generation, split handling, teacher/baseline
training, reconstruction, dual-view training, and standard evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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

from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
    fpr_at_target_tpr,
    plot_budget_diagnostics,
    plot_constituent_count_diagnostics,
    plot_roc,
    train_dual_view_classifier,
    train_single_view_classifier,
)
from unmerge_correct_hlt import (
    RANDOM_SEED,
    DualViewCrossAttnClassifier,
    DualViewJetDataset,
    JetDataset,
    ParticleTransformer,
    compute_features,
    compute_jet_pt,
    eval_classifier,
    eval_classifier_dual,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
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


# ----------------------------- Defaults ------------------------------------ #
MODEL_CFG = {
    "embed_dim": 160,
    "num_heads": 8,
    "num_layers": 6,
    "ff_dim": 640,
    "dropout": 0.1,
}

CLS_TRAIN_CFG = {
    "batch_size": 512,
    "epochs": 70,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 4,
    "patience": 18,
}

RECO_CFG = {
    "embed_dim": 384,
    "num_heads": 8,
    "num_enc_layers": 6,
    "num_dec_layers": 6,
    "ff_dim": 1024,
    "dropout": 0.1,
    "max_decode_tokens": 100,
}

RECO_TRAIN_CFG = {
    "batch_size": 128,
    "epochs": 120,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "patience": 25,
    "min_epochs": 40,
}

LOSS_CFG = {
    "w_ar": 1.0,
    "w_set": 0.85,
    "w_eos": 0.20,
    "w_count": 0.20,
    "w_ptr_entropy": 0.002,
    "huber_delta": 0.12,
}

TOKEN_DIM_WEIGHTS = torch.tensor([1.0, 0.35, 0.25, 0.25, 1.0], dtype=torch.float32)


# ----------------------------- Utilities ----------------------------------- #
def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def sort_constituents_by_pt_np(const: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort each jet's tokens by descending pT (masked tokens go last)."""
    pt = np.where(mask, const[:, :, 0], -1e9)
    order = np.argsort(-pt, axis=1)
    c_sorted = np.take_along_axis(const, order[:, :, None], axis=1).astype(np.float32)
    m_sorted = np.take_along_axis(mask, order, axis=1)
    c_sorted[~m_sorted] = 0.0
    return c_sorted, m_sorted, order


def reorder_features_np(feat: np.ndarray, order: np.ndarray, mask_sorted: np.ndarray) -> np.ndarray:
    out = np.take_along_axis(feat, order[:, :, None], axis=1).astype(np.float32)
    out[~mask_sorted] = 0.0
    return out


def const_to_token_np(const: np.ndarray) -> np.ndarray:
    """[pt,eta,phi,E] -> [log_pt, eta, sin_phi, cos_phi, log_E]."""
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
    tok = np.nan_to_num(tok, nan=0.0, posinf=0.0, neginf=0.0)
    return tok


def token_to_const_torch(tok: torch.Tensor) -> torch.Tensor:
    """[log_pt, eta, sin_phi, cos_phi, log_E] -> [pt, eta, phi, E]."""
    log_pt = tok[..., 0]
    eta = torch.clamp(tok[..., 1], -5.0, 5.0)
    sin_phi = tok[..., 2]
    cos_phi = tok[..., 3]
    norm = torch.sqrt(sin_phi.pow(2) + cos_phi.pow(2) + 1e-8)
    sin_phi = sin_phi / norm
    cos_phi = cos_phi / norm
    phi = torch.atan2(sin_phi, cos_phi)
    log_e = tok[..., 4]

    pt = torch.exp(torch.clamp(log_pt, min=-20.0, max=20.0))
    e = torch.exp(torch.clamp(log_e, min=-20.0, max=20.0))
    e_floor = pt * torch.cosh(eta)
    e = torch.maximum(e, e_floor)

    out = torch.stack([pt, eta, phi, e], dim=-1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def const_to_token_torch(const: torch.Tensor) -> torch.Tensor:
    pt = torch.clamp(const[..., 0], min=1e-8)
    eta = torch.clamp(const[..., 1], min=-5.0, max=5.0)
    phi = const[..., 2]
    e = torch.clamp(const[..., 3], min=1e-8)
    tok = torch.stack(
        [
            torch.log(pt),
            eta,
            torch.sin(phi),
            torch.cos(phi),
            torch.log(e),
        ],
        dim=-1,
    )
    tok = torch.nan_to_num(tok, nan=0.0, posinf=0.0, neginf=0.0)
    return tok


def huber_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    d = float(max(delta, 1e-6))
    loss = torch.where(abs_diff <= d, 0.5 * abs_diff.pow(2) / d, abs_diff - 0.5 * d)
    # Slightly prioritize energy/pt dimensions.
    w = TOKEN_DIM_WEIGHTS.to(loss.device).view(1, 1, -1)
    loss = (loss * w).sum(dim=-1)
    m = mask.float()
    return (loss * m).sum() / (m.sum() + 1e-6)


def chamfer_token_loss(
    pred_tok: torch.Tensor,
    tgt_tok: torch.Tensor,
    pred_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> torch.Tensor:
    w = TOKEN_DIM_WEIGHTS.to(pred_tok.device).view(1, 1, -1)
    p = pred_tok * w
    t = tgt_tok * w
    cost = torch.cdist(p, t, p=1)
    big = torch.full_like(cost, 1e4)

    cost_p = torch.where(tgt_mask.unsqueeze(1), cost, big)
    p2t = cost_p.min(dim=2).values
    p2t = (p2t * pred_mask.float()).sum(dim=1) / (pred_mask.float().sum(dim=1) + 1e-6)

    cost_t = torch.where(pred_mask.unsqueeze(2), cost, big)
    t2p = cost_t.min(dim=1).values
    t2p = (t2p * tgt_mask.float()).sum(dim=1) / (tgt_mask.float().sum(dim=1) + 1e-6)

    return (p2t + t2p).mean()


def build_fixed_split_indices(
    labels: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_all = np.arange(labels.shape[0])
    n_total = int(n_train + n_val + n_test)
    if n_total > labels.shape[0]:
        raise ValueError(
            f"Requested split train+val+test={n_total} exceeds available jets={labels.shape[0]}"
        )

    # Sample n_total first (stratified), then split exactly to train/val/test.
    if n_total < labels.shape[0]:
        idx_sel, _ = train_test_split(
            idx_all,
            train_size=n_total,
            random_state=int(seed),
            stratify=labels,
        )
    else:
        idx_sel = idx_all

    labels_sel = labels[idx_sel]
    idx_train, idx_tmp = train_test_split(
        idx_sel,
        train_size=n_train,
        random_state=int(seed),
        stratify=labels_sel,
    )

    labels_tmp = labels[idx_tmp]
    idx_val, idx_test = train_test_split(
        idx_tmp,
        train_size=n_val,
        test_size=n_test,
        random_state=int(seed),
        stratify=labels_tmp,
    )
    return idx_train, idx_val, idx_test


# ----------------------------- Datasets ------------------------------------ #
class RecoSeqDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        tgt_tok: np.ndarray,
        tgt_mask: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.tgt_tok = torch.tensor(tgt_tok, dtype=torch.float32)
        self.tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "tgt_tok": self.tgt_tok[i],
            "tgt_mask": self.tgt_mask[i],
        }


class RecoInputDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray, const_hlt: np.ndarray):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
        }


# ----------------------------- Model --------------------------------------- #
class HLT2OfflineSeq2Seq(nn.Module):
    def __init__(
        self,
        input_dim_hlt: int = 7,
        token_dim: int = 5,
        embed_dim: int = 384,
        num_heads: int = 8,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_hlt_tokens: int = 100,
        max_decode_tokens: int = 100,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.max_decode_tokens = int(max_decode_tokens)

        self.enc_in = nn.Sequential(
            nn.Linear(input_dim_hlt, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hlt_pos = nn.Parameter(torch.zeros(1, max_hlt_tokens, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        self.dec_in = nn.Sequential(
            nn.Linear(token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dec_pos = nn.Parameter(torch.zeros(1, max_decode_tokens + 1, embed_dim))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.gate_head = nn.Linear(embed_dim, 1)
        self.stop_head = nn.Linear(embed_dim, 1)
        self.count_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        self.bos_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        nn.init.normal_(self.hlt_pos, std=0.02)
        nn.init.normal_(self.dec_pos, std=0.02)
        nn.init.normal_(self.bos_token, std=0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

    def encode(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = feat_hlt.shape
        x = self.enc_in(feat_hlt) + self.hlt_pos[:, :L, :]
        mem = self.encoder(x, src_key_padding_mask=~mask_hlt)

        hlt_tok = const_to_token_torch(const_hlt)

        m = mask_hlt.float()
        pooled = (mem * m.unsqueeze(-1)).sum(dim=1) / (m.sum(dim=1, keepdim=True) + 1e-6)
        count_pred = F.softplus(self.count_head(pooled).squeeze(-1))
        return mem, hlt_tok, count_pred

    def _predict_from_hidden(
        self,
        h: torch.Tensor,
        mem: torch.Tensor,
        mask_hlt: torch.Tensor,
        hlt_tok: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # h: [B, T, D]
        q = self.q_proj(h)
        logits = torch.matmul(q, mem.transpose(1, 2)) / math.sqrt(float(q.shape[-1]))
        logits = logits.masked_fill((~mask_hlt).unsqueeze(1), -1e9)
        attn = torch.softmax(logits, dim=-1)

        base_tok = torch.matmul(attn, hlt_tok)
        delta = self.delta_head(h)
        gate = torch.sigmoid(self.gate_head(h))
        pred_tok = base_tok + gate * delta

        # keep sin/cos channel normalized (avoid in-place autograd edits)
        sn = pred_tok[..., 2]
        cs = pred_tok[..., 3]
        norm = torch.sqrt(sn.pow(2) + cs.pow(2) + 1e-8)
        pred_tok = torch.cat(
            [pred_tok[..., :2], (sn / norm).unsqueeze(-1), (cs / norm).unsqueeze(-1), pred_tok[..., 4:]],
            dim=-1,
        )

        stop_logits = self.stop_head(h).squeeze(-1)
        return pred_tok, stop_logits, attn, gate.squeeze(-1)

    def forward_teacher(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        tgt_tok: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = tgt_tok.shape
        mem, hlt_tok, count_pred = self.encode(feat_hlt, mask_hlt, const_hlt)

        bos = self.bos_token.expand(B, 1, self.token_dim)
        dec_in_tok = torch.cat([bos, tgt_tok[:, :-1, :]], dim=1)
        dec_in = self.dec_in(dec_in_tok) + self.dec_pos[:, :T, :]

        h = self.decoder(
            dec_in,
            mem,
            tgt_mask=self._causal_mask(T, dec_in.device),
            memory_key_padding_mask=~mask_hlt,
        )
        pred_tok, stop_logits, attn, gate = self._predict_from_hidden(h, mem, mask_hlt, hlt_tok)
        return {
            "pred_tok": pred_tok,
            "stop_logits": stop_logits,
            "count_pred": count_pred,
            "attn": attn,
            "gate": gate,
        }

    @torch.no_grad()
    def decode_greedy(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        max_steps: int,
    ) -> Dict[str, torch.Tensor]:
        B = feat_hlt.shape[0]
        mem, hlt_tok, count_pred = self.encode(feat_hlt, mask_hlt, const_hlt)

        prev_tok = self.bos_token.expand(B, 1, self.token_dim)
        pred_seq = []
        stop_seq = []

        for t in range(int(max_steps)):
            d_in = self.dec_in(prev_tok) + self.dec_pos[:, : prev_tok.shape[1], :]
            h = self.decoder(
                d_in,
                mem,
                tgt_mask=self._causal_mask(prev_tok.shape[1], d_in.device),
                memory_key_padding_mask=~mask_hlt,
            )
            h_last = h[:, -1:, :]
            pred_tok, stop_logits, _attn, _gate = self._predict_from_hidden(h_last, mem, mask_hlt, hlt_tok)
            pred_seq.append(pred_tok[:, 0, :])
            stop_seq.append(stop_logits[:, 0])
            prev_tok = torch.cat([prev_tok, pred_tok], dim=1)

        pred_tok_full = torch.stack(pred_seq, dim=1)
        stop_logits_full = torch.stack(stop_seq, dim=1)
        stop_probs = torch.sigmoid(stop_logits_full)
        return {
            "pred_tok": pred_tok_full,
            "stop_probs": stop_probs,
            "count_pred": count_pred,
        }


# ----------------------------- Reco train/eval ----------------------------- #
def compute_reco_losses(
    out: Dict[str, torch.Tensor],
    tgt_tok: torch.Tensor,
    tgt_mask: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    B, T, _ = tgt_tok.shape
    device = tgt_tok.device

    pred_tok = out["pred_tok"]
    stop_logits = out["stop_logits"]
    count_pred = out["count_pred"]
    attn = out["attn"]

    tgt_count = tgt_mask.float().sum(dim=1)

    loss_ar = huber_masked(pred_tok, tgt_tok, tgt_mask, delta=float(loss_cfg["huber_delta"]))

    steps = torch.arange(T, device=device).unsqueeze(0)
    stop_target = (steps >= tgt_count.long().unsqueeze(1)).float()
    loss_eos = F.binary_cross_entropy_with_logits(stop_logits, stop_target)

    pred_mask_for_set = steps < tgt_count.long().unsqueeze(1)
    loss_set = chamfer_token_loss(pred_tok, tgt_tok, pred_mask_for_set, tgt_mask)

    loss_count = F.smooth_l1_loss(count_pred, tgt_count)

    # Lower entropy encourages edit-from-copy behavior.
    ptr_entropy = -(attn * torch.log(attn.clamp(min=1e-8))).sum(dim=-1).mean()

    total = (
        float(loss_cfg["w_ar"]) * loss_ar
        + float(loss_cfg["w_set"]) * loss_set
        + float(loss_cfg["w_eos"]) * loss_eos
        + float(loss_cfg["w_count"]) * loss_count
        + float(loss_cfg["w_ptr_entropy"]) * ptr_entropy
    )

    return {
        "total": total,
        "ar": loss_ar,
        "set": loss_set,
        "eos": loss_eos,
        "count": loss_count,
        "ptr_entropy": ptr_entropy,
    }


@dataclass
class RecoValMetrics:
    val_total: float
    val_ar: float
    val_set: float
    val_eos: float
    val_count: float


def train_reconstructor_seq2seq(
    model: HLT2OfflineSeq2Seq,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
) -> Tuple[HLT2OfflineSeq2Seq, Dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_state = None
    best_val = float("inf")
    best_metrics = None
    no_improve = 0

    for ep in range(int(train_cfg["epochs"])):
        model.train()
        tr_tot = tr_ar = tr_set = tr_eos = tr_cnt = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            tgt_tok = batch["tgt_tok"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)

            out = model.forward_teacher(feat_hlt, mask_hlt, const_hlt, tgt_tok)
            losses = compute_reco_losses(out, tgt_tok, tgt_mask, loss_cfg)

            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bsz = feat_hlt.shape[0]
            n_tr += bsz
            tr_tot += float(losses["total"].detach().item()) * bsz
            tr_ar += float(losses["ar"].detach().item()) * bsz
            tr_set += float(losses["set"].detach().item()) * bsz
            tr_eos += float(losses["eos"].detach().item()) * bsz
            tr_cnt += float(losses["count"].detach().item()) * bsz

        model.eval()
        va_tot = va_ar = va_set = va_eos = va_cnt = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                tgt_tok = batch["tgt_tok"].to(device)
                tgt_mask = batch["tgt_mask"].to(device)

                out = model.forward_teacher(feat_hlt, mask_hlt, const_hlt, tgt_tok)
                losses = compute_reco_losses(out, tgt_tok, tgt_mask, loss_cfg)

                bsz = feat_hlt.shape[0]
                n_va += bsz
                va_tot += float(losses["total"].detach().item()) * bsz
                va_ar += float(losses["ar"].detach().item()) * bsz
                va_set += float(losses["set"].detach().item()) * bsz
                va_eos += float(losses["eos"].detach().item()) * bsz
                va_cnt += float(losses["count"].detach().item()) * bsz

        tr_tot /= max(n_tr, 1)
        tr_ar /= max(n_tr, 1)
        tr_set /= max(n_tr, 1)
        tr_eos /= max(n_tr, 1)
        tr_cnt /= max(n_tr, 1)

        va_tot /= max(n_va, 1)
        va_ar /= max(n_va, 1)
        va_set /= max(n_va, 1)
        va_eos /= max(n_va, 1)
        va_cnt /= max(n_va, 1)

        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"Reco ep {ep+1:03d} | "
                f"train total={tr_tot:.5f} ar={tr_ar:.5f} set={tr_set:.5f} eos={tr_eos:.5f} cnt={tr_cnt:.5f} | "
                f"val total={va_tot:.5f} ar={va_ar:.5f} set={va_set:.5f} eos={va_eos:.5f} cnt={va_cnt:.5f}"
            )

        if va_tot < best_val:
            best_val = float(va_tot)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = RecoValMetrics(
                val_total=float(va_tot),
                val_ar=float(va_ar),
                val_set=float(va_set),
                val_eos=float(va_eos),
                val_count=float(va_cnt),
            )
            no_improve = 0
        else:
            no_improve += 1

        if ep + 1 >= int(train_cfg["min_epochs"]) and no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping reconstructor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_metrics is None:
        best_metrics = RecoValMetrics(
            val_total=float("nan"),
            val_ar=float("nan"),
            val_set=float("nan"),
            val_eos=float("nan"),
            val_count=float("nan"),
        )

    return model, {
        "val_total": best_metrics.val_total,
        "val_ar": best_metrics.val_ar,
        "val_set": best_metrics.val_set,
        "val_eos": best_metrics.val_eos,
        "val_count": best_metrics.val_count,
    }


@torch.no_grad()
def reconstruct_dataset_seq2seq(
    model: HLT2OfflineSeq2Seq,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    max_constits: int,
    device: torch.device,
    batch_size: int,
    beam_size: int = 4,
    beam_len_sigma: float = 1.5,
    beam_temperature: float = 1.0,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()
    ds = RecoInputDataset(feat_hlt, mask_hlt, const_hlt)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    n_jets = int(feat_hlt.shape[0])
    T = int(max_constits)

    reco_const = np.zeros((n_jets, T, 4), dtype=np.float32)
    reco_mask = np.zeros((n_jets, T), dtype=bool)
    reco_merge_flag = np.zeros((n_jets, T), dtype=np.float32)  # reused as confidence mask
    reco_eff_flag = np.zeros((n_jets, T), dtype=np.float32)

    created_merge_count = np.zeros((n_jets,), dtype=np.int32)
    created_eff_count = np.zeros((n_jets,), dtype=np.int32)
    pred_budget_total = np.zeros((n_jets,), dtype=np.float32)
    pred_budget_merge = np.zeros((n_jets,), dtype=np.float32)
    pred_budget_eff = np.zeros((n_jets,), dtype=np.float32)

    offset = 0
    eps = 1e-8

    for batch in dl:
        feat = batch["feat_hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        c = batch["const_hlt"].to(device)
        bsz = int(feat.shape[0])

        dec = model.decode_greedy(feat, m, c, max_steps=T)
        pred_tok = dec["pred_tok"]  # [B,T,5]
        stop_probs = dec["stop_probs"]  # [B,T]
        count_pred = dec["count_pred"]  # [B]

        pred_const = token_to_const_torch(pred_tok).detach().cpu().numpy().astype(np.float32)
        stop_np = stop_probs.detach().cpu().numpy().astype(np.float32)
        count_np = count_pred.detach().cpu().numpy().astype(np.float32)
        hlt_count_np = m.detach().cpu().numpy().sum(axis=1).astype(np.int32)

        for i in range(bsz):
            cp = float(count_np[i])
            center = int(np.rint(cp))
            cand = set()
            span = max(int(beam_size), 2)
            for d in range(-span, span + 1):
                cand.add(int(np.clip(center + d, 0, T)))
            cand = sorted(cand)

            lp = []
            s = np.clip(stop_np[i], 1e-6, 1.0 - 1e-6)
            for L in cand:
                cont = float(np.log(1.0 - s[:L] + eps).sum()) if L > 0 else 0.0
                stop_term = float(np.log(s[L] + eps)) if L < T else 0.0
                count_prior = -0.5 * ((float(L) - cp) / max(float(beam_len_sigma), 1e-3)) ** 2
                lp.append(cont + stop_term + count_prior)
            lp = np.asarray(lp, dtype=np.float64)

            k = min(int(beam_size), len(cand))
            top_idx = np.argsort(-lp)[:k]
            top_lens = np.asarray([cand[j] for j in top_idx], dtype=np.int32)
            top_lp = lp[top_idx]
            top_lp = top_lp / max(float(beam_temperature), 1e-3)
            top_w = np.exp(top_lp - np.max(top_lp))
            top_w = top_w / max(top_w.sum(), 1e-12)

            steps = np.arange(T, dtype=np.int32)
            active_prob = np.zeros((T,), dtype=np.float32)
            for ww, LL in zip(top_w, top_lens):
                active_prob += float(ww) * (steps < int(LL)).astype(np.float32)

            soft_len = float(np.sum(top_w * top_lens.astype(np.float64)))
            final_len = int(np.clip(np.rint(soft_len), 0, T))

            pred_i = pred_const[i]
            if final_len > 0:
                # Keep AR order but enforce pT-descending for classifier consistency.
                pt = pred_i[:final_len, 0]
                ord_i = np.argsort(-pt)
                pred_i[:final_len] = pred_i[:final_len][ord_i]
                active_prob[:final_len] = active_prob[:final_len][ord_i]

            idx0 = offset + i
            reco_const[idx0] = pred_i
            reco_mask[idx0, :final_len] = True
            reco_merge_flag[idx0] = active_prob

            reco_n = final_len
            hlt_n = int(hlt_count_np[i])
            created = max(reco_n - hlt_n, 0)
            created_merge_count[idx0] = int(created)
            created_eff_count[idx0] = 0
            pred_budget_total[idx0] = float(created)
            pred_budget_merge[idx0] = float(created)
            pred_budget_eff[idx0] = 0.0

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


# ----------------------------- Main ---------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=250000)
    parser.add_argument("--n_train_split", type=int, default=100000)
    parser.add_argument("--n_val_split", type=int, default=50000)
    parser.add_argument("--n_test_split", type=int, default=100000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_joint_seq2seq"))
    parser.add_argument("--run_name", type=str, default="seq2seq_default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--roc_fpr_min", type=float, default=1e-4)

    # Reconstructor settings
    parser.add_argument("--reco_batch_size", type=int, default=RECO_TRAIN_CFG["batch_size"])
    parser.add_argument("--reco_epochs", type=int, default=RECO_TRAIN_CFG["epochs"])
    parser.add_argument("--reco_lr", type=float, default=RECO_TRAIN_CFG["lr"])
    parser.add_argument("--reco_patience", type=int, default=RECO_TRAIN_CFG["patience"])
    parser.add_argument("--reco_min_epochs", type=int, default=RECO_TRAIN_CFG["min_epochs"])
    parser.add_argument("--reco_huber_delta", type=float, default=LOSS_CFG["huber_delta"])
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--beam_len_sigma", type=float, default=1.5)
    parser.add_argument("--beam_temperature", type=float, default=1.0)

    # Loss weights
    parser.add_argument("--loss_w_ar", type=float, default=LOSS_CFG["w_ar"])
    parser.add_argument("--loss_w_set", type=float, default=LOSS_CFG["w_set"])
    parser.add_argument("--loss_w_eos", type=float, default=LOSS_CFG["w_eos"])
    parser.add_argument("--loss_w_count", type=float, default=LOSS_CFG["w_count"])
    parser.add_argument("--loss_w_ptr_entropy", type=float, default=LOSS_CFG["w_ptr_entropy"])

    parser.add_argument("--skip_save_models", action="store_true")
    parser.add_argument("--save_fusion_scores", action="store_true")

    # Pseudo-HLT knobs (kept from prior plumbing)
    parser.add_argument("--merge_radius", type=float, default=BASE_CONFIG["hlt_effects"]["merge_radius"])
    parser.add_argument("--pt_threshold_hlt", type=float, default=BASE_CONFIG["hlt_effects"]["pt_threshold_hlt"])
    parser.add_argument("--pt_threshold_offline", type=float, default=BASE_CONFIG["hlt_effects"]["pt_threshold_offline"])
    parser.add_argument("--eff_plateau_barrel", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    parser.add_argument("--eff_plateau_endcap", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    parser.add_argument("--smear_a", type=float, default=BASE_CONFIG["hlt_effects"]["smear_a"])
    parser.add_argument("--smear_b", type=float, default=BASE_CONFIG["hlt_effects"]["smear_b"])
    parser.add_argument("--smear_c", type=float, default=BASE_CONFIG["hlt_effects"]["smear_c"])

    args = parser.parse_args()

    set_seed(int(args.seed))

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    # Build local configs.
    hlt_cfg = _deepcopy_cfg()
    hlt_cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    hlt_cfg["hlt_effects"]["pt_threshold_hlt"] = float(args.pt_threshold_hlt)
    hlt_cfg["hlt_effects"]["pt_threshold_offline"] = float(args.pt_threshold_offline)
    hlt_cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    hlt_cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    hlt_cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    hlt_cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    hlt_cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    reco_train_cfg = dict(RECO_TRAIN_CFG)
    reco_train_cfg["batch_size"] = int(args.reco_batch_size)
    reco_train_cfg["epochs"] = int(args.reco_epochs)
    reco_train_cfg["lr"] = float(args.reco_lr)
    reco_train_cfg["patience"] = int(args.reco_patience)
    reco_train_cfg["min_epochs"] = int(args.reco_min_epochs)

    loss_cfg = {
        "w_ar": float(args.loss_w_ar),
        "w_set": float(args.loss_w_set),
        "w_eos": float(args.loss_w_eos),
        "w_count": float(args.loss_w_count),
        "w_ptr_entropy": float(args.loss_w_ptr_entropy),
        "huber_delta": float(args.reco_huber_delta),
    }

    # ----------------------------- Load data -------------------------------- #
    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = int(args.offset_jets + args.n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=int(args.max_constits),
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets for offset={args.offset_jets} + n_train_jets={args.n_train_jets}. "
            f"Loaded={all_const_full.shape[0]}"
        )

    const_raw = all_const_full[int(args.offset_jets): int(args.offset_jets + args.n_train_jets)]
    labels = all_labels_full[int(args.offset_jets): int(args.offset_jets + args.n_train_jets)].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(hlt_cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy().astype(np.float32)
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        hlt_cfg,
        seed=int(args.seed),
    )
    budget_merge_true = budget_truth["merge_lost_per_jet"].astype(np.float32)
    budget_eff_true = budget_truth["eff_lost_per_jet"].astype(np.float32)

    print("Computing features...")
    features_off = compute_features(const_off, masks_off)
    features_hlt = compute_features(hlt_const, hlt_mask)

    # Sort HLT and offline constituents by pT for sequence modeling.
    hlt_const_sort, hlt_mask_sort, hlt_order = sort_constituents_by_pt_np(hlt_const, hlt_mask)
    const_off_sort, masks_off_sort, _off_order = sort_constituents_by_pt_np(const_off, masks_off)
    features_hlt_sort = reorder_features_np(features_hlt, hlt_order, hlt_mask_sort)

    train_idx, val_idx, test_idx = build_fixed_split_indices(
        labels=labels,
        n_train=int(args.n_train_split),
        n_val=int(args.n_val_split),
        n_test=int(args.n_test_split),
        seed=int(args.seed),
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = standardize(features_hlt_sort, hlt_mask_sort, feat_means, feat_stds)

    # Save split/stats for reproducibility and downstream tooling.
    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        feat_means=feat_means.astype(np.float32),
        feat_stds=feat_stds.astype(np.float32),
    )

    with open(save_root / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # ----------------------------- Teacher ---------------------------------- #
    print("\n" + "=" * 72)
    print("STEP 1: TEACHER (Offline)")
    print("=" * 72)

    bs_cls = int(CLS_TRAIN_CFG["batch_size"])

    ds_tr_off = JetDataset(features_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    ds_va_off = JetDataset(features_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    ds_te_off = JetDataset(features_off_std[test_idx], masks_off[test_idx], labels[test_idx])

    dl_tr_off = DataLoader(ds_tr_off, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_off = DataLoader(ds_va_off, batch_size=bs_cls, shuffle=False)
    dl_te_off = DataLoader(ds_te_off, batch_size=bs_cls, shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **MODEL_CFG).to(device)
    teacher = train_single_view_classifier(teacher, dl_tr_off, dl_va_off, device, CLS_TRAIN_CFG, name="Teacher")
    auc_teacher, preds_teacher, labs_test = eval_classifier(teacher, dl_te_off, device)

    # ----------------------------- Baseline HLT ----------------------------- #
    print("\n" + "=" * 72)
    print("STEP 2: BASELINE (HLT)")
    print("=" * 72)

    ds_tr_hlt = JetDataset(features_hlt_std[train_idx], hlt_mask_sort[train_idx], labels[train_idx])
    ds_va_hlt = JetDataset(features_hlt_std[val_idx], hlt_mask_sort[val_idx], labels[val_idx])
    ds_te_hlt = JetDataset(features_hlt_std[test_idx], hlt_mask_sort[test_idx], labels[test_idx])

    dl_tr_hlt = DataLoader(ds_tr_hlt, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_hlt = DataLoader(ds_va_hlt, batch_size=bs_cls, shuffle=False)
    dl_te_hlt = DataLoader(ds_te_hlt, batch_size=bs_cls, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **MODEL_CFG).to(device)
    baseline = train_single_view_classifier(baseline, dl_tr_hlt, dl_va_hlt, device, CLS_TRAIN_CFG, name="HLT")
    auc_hlt, preds_hlt, _ = eval_classifier(baseline, dl_te_hlt, device)

    # ----------------------------- Reconstructor ---------------------------- #
    print("\n" + "=" * 72)
    print("STEP 3: SEQ2SEQ RECONSTRUCTOR (Continuous AR + Beam Length)")
    print("=" * 72)

    tgt_tok_all = const_to_token_np(const_off_sort)
    tgt_mask_all = masks_off_sort.astype(bool)

    ds_tr_reco = RecoSeqDataset(
        feat_hlt=features_hlt_std[train_idx],
        mask_hlt=hlt_mask_sort[train_idx],
        const_hlt=hlt_const_sort[train_idx],
        tgt_tok=tgt_tok_all[train_idx],
        tgt_mask=tgt_mask_all[train_idx],
    )
    ds_va_reco = RecoSeqDataset(
        feat_hlt=features_hlt_std[val_idx],
        mask_hlt=hlt_mask_sort[val_idx],
        const_hlt=hlt_const_sort[val_idx],
        tgt_tok=tgt_tok_all[val_idx],
        tgt_mask=tgt_mask_all[val_idx],
    )

    dl_tr_reco = DataLoader(
        ds_tr_reco,
        batch_size=int(reco_train_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_va_reco = DataLoader(
        ds_va_reco,
        batch_size=int(reco_train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    reco = HLT2OfflineSeq2Seq(
        input_dim_hlt=7,
        token_dim=5,
        embed_dim=int(RECO_CFG["embed_dim"]),
        num_heads=int(RECO_CFG["num_heads"]),
        num_enc_layers=int(RECO_CFG["num_enc_layers"]),
        num_dec_layers=int(RECO_CFG["num_dec_layers"]),
        ff_dim=int(RECO_CFG["ff_dim"]),
        dropout=float(RECO_CFG["dropout"]),
        max_hlt_tokens=int(args.max_constits),
        max_decode_tokens=int(args.max_constits),
    ).to(device)

    reco, reco_val_metrics = train_reconstructor_seq2seq(
        reco,
        dl_tr_reco,
        dl_va_reco,
        device,
        reco_train_cfg,
        loss_cfg,
    )

    print("Best reconstructor val metrics:")
    for k, v in reco_val_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("Building reconstructed dataset...")
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
    ) = reconstruct_dataset_seq2seq(
        model=reco,
        feat_hlt=features_hlt_std,
        mask_hlt=hlt_mask_sort,
        const_hlt=hlt_const_sort,
        max_constits=int(args.max_constits),
        device=device,
        batch_size=int(reco_train_cfg["batch_size"]),
        beam_size=int(args.beam_size),
        beam_len_sigma=float(args.beam_len_sigma),
        beam_temperature=float(args.beam_temperature),
    )

    features_reco = compute_features(reco_const, reco_mask)
    features_reco_std = standardize(features_reco, reco_mask, feat_means, feat_stds)
    # Keep soft-view confidence channel + placeholder efficiency flag channel.
    features_reco_flag = np.concatenate(
        [features_reco_std, reco_merge_flag[..., None], reco_eff_flag[..., None]],
        axis=-1,
    ).astype(np.float32)

    # ----------------------------- Taggers on reconstructed view ------------ #
    print("\n" + "=" * 72)
    print("STEP 4: TAGGERS ON RECONSTRUCTED VIEW")
    print("=" * 72)

    ds_tr_reco_cls = JetDataset(features_reco_std[train_idx], reco_mask[train_idx], labels[train_idx])
    ds_va_reco_cls = JetDataset(features_reco_std[val_idx], reco_mask[val_idx], labels[val_idx])
    ds_te_reco_cls = JetDataset(features_reco_std[test_idx], reco_mask[test_idx], labels[test_idx])

    dl_tr_reco_cls = DataLoader(ds_tr_reco_cls, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_reco_cls = DataLoader(ds_va_reco_cls, batch_size=bs_cls, shuffle=False)
    dl_te_reco_cls = DataLoader(ds_te_reco_cls, batch_size=bs_cls, shuffle=False)

    unmerge = ParticleTransformer(input_dim=7, **MODEL_CFG).to(device)
    unmerge = train_single_view_classifier(unmerge, dl_tr_reco_cls, dl_va_reco_cls, device, CLS_TRAIN_CFG, name="RecoOnly")
    auc_unmerge, preds_unmerge, _ = eval_classifier(unmerge, dl_te_reco_cls, device)

    # Dual-view (HLT + reconstructed)
    ds_tr_dual = DualViewJetDataset(
        features_hlt_std[train_idx],
        hlt_mask_sort[train_idx],
        features_reco_std[train_idx],
        reco_mask[train_idx],
        labels[train_idx],
    )
    ds_va_dual = DualViewJetDataset(
        features_hlt_std[val_idx],
        hlt_mask_sort[val_idx],
        features_reco_std[val_idx],
        reco_mask[val_idx],
        labels[val_idx],
    )
    ds_te_dual = DualViewJetDataset(
        features_hlt_std[test_idx],
        hlt_mask_sort[test_idx],
        features_reco_std[test_idx],
        reco_mask[test_idx],
        labels[test_idx],
    )

    dl_tr_dual = DataLoader(ds_tr_dual, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_dual = DataLoader(ds_va_dual, batch_size=bs_cls, shuffle=False)
    dl_te_dual = DataLoader(ds_te_dual, batch_size=bs_cls, shuffle=False)

    dual = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **MODEL_CFG).to(device)
    dual = train_dual_view_classifier(dual, dl_tr_dual, dl_va_dual, device, CLS_TRAIN_CFG, name="DualView")
    auc_dual, preds_dual, _ = eval_classifier_dual(dual, dl_te_dual, device)

    # Dual-view with confidence channels.
    ds_tr_dual_f = DualViewJetDataset(
        features_hlt_std[train_idx],
        hlt_mask_sort[train_idx],
        features_reco_flag[train_idx],
        reco_mask[train_idx],
        labels[train_idx],
    )
    ds_va_dual_f = DualViewJetDataset(
        features_hlt_std[val_idx],
        hlt_mask_sort[val_idx],
        features_reco_flag[val_idx],
        reco_mask[val_idx],
        labels[val_idx],
    )
    ds_te_dual_f = DualViewJetDataset(
        features_hlt_std[test_idx],
        hlt_mask_sort[test_idx],
        features_reco_flag[test_idx],
        reco_mask[test_idx],
        labels[test_idx],
    )

    dl_tr_dual_f = DataLoader(ds_tr_dual_f, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_va_dual_f = DataLoader(ds_va_dual_f, batch_size=bs_cls, shuffle=False)
    dl_te_dual_f = DataLoader(ds_te_dual_f, batch_size=bs_cls, shuffle=False)

    dual_flag = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=9, **MODEL_CFG).to(device)
    dual_flag = train_dual_view_classifier(dual_flag, dl_tr_dual_f, dl_va_dual_f, device, CLS_TRAIN_CFG, name="DualView+Conf")
    auc_dual_flag, preds_dual_flag, _ = eval_classifier_dual(dual_flag, dl_te_dual_f, device)

    # ----------------------------- Final metrics ---------------------------- #
    print("\n" + "=" * 72)
    print("FINAL TEST EVALUATION")
    print("=" * 72)
    print(f"Teacher (Offline) AUC: {auc_teacher:.6f}")
    print(f"Baseline (HLT)   AUC: {auc_hlt:.6f}")
    print(f"RecoOnly         AUC: {auc_unmerge:.6f}")
    print(f"DualView         AUC: {auc_dual:.6f}")
    print(f"DualView+Conf    AUC: {auc_dual_flag:.6f}")

    fpr_t, tpr_t, _ = roc_curve(labs_test, preds_teacher)
    fpr_h, tpr_h, _ = roc_curve(labs_test, preds_hlt)
    fpr_r, tpr_r, _ = roc_curve(labs_test, preds_unmerge)
    fpr_d, tpr_d, _ = roc_curve(labs_test, preds_dual)
    fpr_df, tpr_df, _ = roc_curve(labs_test, preds_dual_flag)

    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.30)
    fpr30_hlt = fpr_at_target_tpr(fpr_h, tpr_h, 0.30)
    fpr30_reco = fpr_at_target_tpr(fpr_r, tpr_r, 0.30)
    fpr30_dual = fpr_at_target_tpr(fpr_d, tpr_d, 0.30)
    fpr30_dual_flag = fpr_at_target_tpr(fpr_df, tpr_df, 0.30)

    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_hlt = fpr_at_target_tpr(fpr_h, tpr_h, 0.50)
    fpr50_reco = fpr_at_target_tpr(fpr_r, tpr_r, 0.50)
    fpr50_dual = fpr_at_target_tpr(fpr_d, tpr_d, 0.50)
    fpr50_dual_flag = fpr_at_target_tpr(fpr_df, tpr_df, 0.50)

    print("\nFPR@30")
    print(
        f"  Teacher/HLT/RecoOnly/Dual/Dual+Conf: "
        f"{fpr30_teacher:.6f} / {fpr30_hlt:.6f} / {fpr30_reco:.6f} / {fpr30_dual:.6f} / {fpr30_dual_flag:.6f}"
    )
    print("FPR@50")
    print(
        f"  Teacher/HLT/RecoOnly/Dual/Dual+Conf: "
        f"{fpr50_teacher:.6f} / {fpr50_hlt:.6f} / {fpr50_reco:.6f} / {fpr50_dual:.6f} / {fpr50_dual_flag:.6f}"
    )

    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_h, fpr_h, "--", f"HLT (AUC={auc_hlt:.3f})", "steelblue"),
            (tpr_r, fpr_r, ":", f"RecoOnly (AUC={auc_unmerge:.3f})", "forestgreen"),
            (tpr_d, fpr_d, "-.", f"DualView (AUC={auc_dual:.3f})", "darkorange"),
            (tpr_df, fpr_df, "-", f"DualView+Conf (AUC={auc_dual_flag:.3f})", "slateblue"),
        ],
        save_root / "results_all.png",
        args.roc_fpr_min,
    )

    # Count/budget diagnostics (same files as prior pipeline).
    count_summary = plot_constituent_count_diagnostics(
        save_root=save_root,
        mask_off=masks_off_sort,
        hlt_mask=hlt_mask_sort,
        reco_mask=reco_mask,
        created_merge_count=created_merge_count,
        created_eff_count=created_eff_count,
        hlt_stats=hlt_stats,
    )

    budget_summary = plot_budget_diagnostics(
        save_root=save_root,
        true_merge=budget_merge_true[test_idx],
        true_eff=budget_eff_true[test_idx],
        pred_merge=pred_budget_merge[test_idx],
        pred_eff=pred_budget_eff[test_idx],
    )

    # Optional fusion scores (teacher/hlt/joint(dual)).
    if bool(args.save_fusion_scores):
        auc_teacher_val, preds_teacher_val, labs_val = eval_classifier(teacher, dl_va_off, device)
        auc_hlt_val, preds_hlt_val, labs_hlt_val = eval_classifier(baseline, dl_va_hlt, device)
        auc_dual_val, preds_dual_val, labs_dual_val = eval_classifier_dual(dual, dl_va_dual, device)

        assert np.array_equal(labs_val.astype(np.float32), labs_hlt_val.astype(np.float32))
        assert np.array_equal(labs_val.astype(np.float32), labs_dual_val.astype(np.float32))
        assert np.array_equal(labs_test.astype(np.float32), labels[test_idx].astype(np.float32))

        np.savez_compressed(
            save_root / "fusion_scores_val_test.npz",
            labels_val=labs_val.astype(np.float32),
            labels_test=labs_test.astype(np.float32),
            preds_teacher_val=np.asarray(preds_teacher_val, dtype=np.float64),
            preds_teacher_test=np.asarray(preds_teacher, dtype=np.float64),
            preds_hlt_val=np.asarray(preds_hlt_val, dtype=np.float64),
            preds_hlt_test=np.asarray(preds_hlt, dtype=np.float64),
            preds_joint_val=np.asarray(preds_dual_val, dtype=np.float64),
            preds_joint_test=np.asarray(preds_dual, dtype=np.float64),
            auc_teacher_val=float(auc_teacher_val),
            auc_teacher_test=float(auc_teacher),
            auc_hlt_val=float(auc_hlt_val),
            auc_hlt_test=float(auc_hlt),
            auc_joint_val=float(auc_dual_val),
            auc_joint_test=float(auc_dual),
            fpr50_joint_val=float(fpr_at_target_tpr(*roc_curve(labs_val, preds_dual_val)[:2], 0.50)),
            fpr50_joint_test=float(fpr50_dual),
            hlt_nconst_val=np.asarray(hlt_mask_sort[val_idx].sum(axis=1), dtype=np.int32),
            hlt_nconst_test=np.asarray(hlt_mask_sort[test_idx].sum(axis=1), dtype=np.int32),
            hlt_jet_pt_val=np.asarray(compute_jet_pt(hlt_const_sort[val_idx], hlt_mask_sort[val_idx]), dtype=np.float64),
            hlt_jet_pt_test=np.asarray(compute_jet_pt(hlt_const_sort[test_idx], hlt_mask_sort[test_idx]), dtype=np.float64),
            off_jet_pt_val=np.asarray(compute_jet_pt(const_off_sort[val_idx], masks_off_sort[val_idx]), dtype=np.float64),
            off_jet_pt_test=np.asarray(compute_jet_pt(const_off_sort[test_idx], masks_off_sort[test_idx]), dtype=np.float64),
        )
        print(f"Saved fusion score arrays to: {save_root / 'fusion_scores_val_test.npz'}")

    np.savez_compressed(
        save_root / "results.npz",
        auc_teacher=float(auc_teacher),
        auc_hlt=float(auc_hlt),
        auc_reco=float(auc_unmerge),
        auc_dual=float(auc_dual),
        auc_dual_flag=float(auc_dual_flag),
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_hlt=fpr_h,
        tpr_hlt=tpr_h,
        fpr_reco=fpr_r,
        tpr_reco=tpr_r,
        fpr_dual=fpr_d,
        tpr_dual=tpr_d,
        fpr_dual_flag=fpr_df,
        tpr_dual_flag=tpr_df,
        fpr30_teacher=float(fpr30_teacher),
        fpr30_hlt=float(fpr30_hlt),
        fpr30_reco=float(fpr30_reco),
        fpr30_dual=float(fpr30_dual),
        fpr30_dual_flag=float(fpr30_dual_flag),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_hlt=float(fpr50_hlt),
        fpr50_reco=float(fpr50_reco),
        fpr50_dual=float(fpr50_dual),
        fpr50_dual_flag=float(fpr50_dual_flag),
    )

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": hlt_cfg["hlt_effects"], "stats": hlt_stats}, f, indent=2)

    with open(save_root / "reconstructor_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(reco_val_metrics, f, indent=2)

    with open(save_root / "constituent_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(count_summary, f, indent=2)

    with open(save_root / "budget_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(budget_summary, f, indent=2)

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": float(auc_teacher)}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": float(auc_hlt)}, save_root / "baseline.pt")
        torch.save({"model": reco.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor.pt")
        torch.save({"model": unmerge.state_dict(), "auc": float(auc_unmerge)}, save_root / "reco_only_classifier.pt")
        torch.save({"model": dual.state_dict(), "auc": float(auc_dual)}, save_root / "dual_view_classifier.pt")
        torch.save({"model": dual_flag.state_dict(), "auc": float(auc_dual_flag)}, save_root / "dual_view_conf_classifier.pt")

    np.savez_compressed(
        save_root / "reconstructed_dataset.npz",
        const_off=const_off_sort.astype(np.float32),
        mask_off=masks_off_sort.astype(bool),
        hlt_const=hlt_const_sort.astype(np.float32),
        hlt_mask=hlt_mask_sort.astype(bool),
        reco_const=reco_const.astype(np.float32),
        reco_mask=reco_mask.astype(bool),
        reco_merge_flag=reco_merge_flag.astype(np.float32),
        reco_eff_flag=reco_eff_flag.astype(np.float32),
        created_merge_count=created_merge_count.astype(np.int32),
        created_eff_count=created_eff_count.astype(np.int32),
        budget_merge_true=budget_merge_true.astype(np.float32),
        budget_eff_true=budget_eff_true.astype(np.float32),
        budget_total_pred=pred_budget_total.astype(np.float32),
        budget_merge_pred=pred_budget_merge.astype(np.float32),
        budget_eff_pred=pred_budget_eff.astype(np.float32),
        labels=labels.astype(np.int64),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()

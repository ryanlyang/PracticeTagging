#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed-K additive predictor with frozen-oracle-logit distillation (K=8 by default).

Pipeline:
1) Load offline jets and generate pseudo-HLT.
2) Train teacher (offline) and baseline (HLT) taggers.
3) Build greedy-oracle target tokens (plus fallback fill) per jet.
4) Train HLT->K-token predictor with Hungarian one-to-one loss.
5) Build augmented view: HLT + predicted K tokens.
6) Train/evaluate top tagger on augmented view and report recovery.

This is an additive-correction setup (does not remove HLT tokens).
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
)
from unmerge_correct_hlt import (
    JetDataset,
    ParticleTransformer,
    compute_features,
    eval_classifier,
    get_scheduler,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
    train_classifier,
)


# ----------------------------- Utilities ----------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fpr_at_target_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    target = float(np.clip(target_tpr, 0.0, 1.0))
    idx = int(np.argmin(np.abs(tpr - target)))
    return float(fpr[idx])


def wrap_phi_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def wrap_phi_t(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _model_logits(model: nn.Module, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = model(feat, mask)
    if isinstance(out, tuple):
        out = out[0]
    return out.squeeze(-1)


# ------------------------- Classifier training ------------------------ #
def train_single_view_classifier_auc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> nn.Module:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(
        opt,
        int(train_cfg["warmup_epochs"]),
        int(train_cfg["epochs"]),
    )

    best_val_auc = float("-inf")
    best_state = None
    no_improve = 0

    for ep in tqdm(range(int(train_cfg["epochs"])), desc=name):
        _, tr_auc = train_classifier(model, train_loader, opt, device)
        va_auc, va_preds, va_labs = eval_classifier(model, val_loader, device)
        va_fpr, va_tpr, _ = roc_curve(va_labs, va_preds)
        va_fpr50 = fpr_at_target_tpr(va_fpr, va_tpr, 0.50)
        sch.step()

        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_auc={best_val_auc:.4f}"
            )
        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -------------------------- Oracle target prep ------------------------- #
def compute_novel_mask(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    dr_match: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    n, t, _ = const_off.shape
    novel = np.zeros((n, t), dtype=bool)
    for s in tqdm(range(0, n, chunk_size), desc="NovelMask"):
        e = min(n, s + chunk_size)
        off_eta = const_off[s:e, :, 1]
        off_phi = const_off[s:e, :, 2]
        hlt_eta = const_hlt[s:e, :, 1]
        hlt_phi = const_hlt[s:e, :, 2]

        deta = off_eta[:, :, None] - hlt_eta[:, None, :]
        dphi = wrap_phi_np(off_phi[:, :, None] - hlt_phi[:, None, :])
        dr = np.sqrt(deta * deta + dphi * dphi)

        valid = mask_off[s:e, :, None] & mask_hlt[s:e, None, :]
        dr = np.where(valid, dr, np.inf)
        min_dr = np.min(dr, axis=2)

        matched = min_dr < float(dr_match)
        novel[s:e] = mask_off[s:e] & (~matched)
    return novel


def compute_ig_scores(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps: int,
) -> np.ndarray:
    model.eval()
    n, t, f = feat.shape
    out = np.zeros((n, t), dtype=np.float32)
    steps = max(1, int(steps))

    for s in tqdm(range(0, n, batch_size), desc="IG"):
        e = min(n, s + batch_size)
        xb = torch.tensor(feat[s:e], dtype=torch.float32, device=device)
        mb = torch.tensor(mask[s:e], dtype=torch.bool, device=device)
        yb = torch.tensor(labels[s:e], dtype=torch.float32, device=device)
        sign = (2.0 * yb - 1.0).view(-1)

        baseline = torch.zeros_like(xb)
        total_grad = torch.zeros_like(xb)

        for st in range(1, steps + 1):
            alpha = float(st) / float(steps)
            z = (baseline + alpha * (xb - baseline)).detach().requires_grad_(True)
            logits = _model_logits(model, z, mb)
            objective = (logits * sign).sum()
            model.zero_grad(set_to_none=True)
            objective.backward()
            total_grad += z.grad.detach()

        ig = (xb - baseline) * (total_grad / float(steps))
        tok = ig.abs().sum(dim=2)
        tok = torch.where(mb, tok, torch.zeros_like(tok))
        out[s:e] = tok.detach().cpu().numpy().astype(np.float32)

    return out


def compute_greedy_insertion_order(
    model: nn.Module,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    feat_off: np.ndarray,
    labels: np.ndarray,
    candidate_pool_idx: np.ndarray,
    k_max: int,
    gain_min: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    n, t, f = feat_hlt.shape
    pool = candidate_pool_idx.shape[1]
    out_order = np.full((n, k_max), -1, dtype=np.int64)

    for s in tqdm(range(0, n, batch_size), desc="Greedy"):
        e = min(n, s + batch_size)
        bsz = e - s
        x_cur = torch.tensor(feat_hlt[s:e], dtype=torch.float32, device=device)
        m_cur = torch.tensor(mask_hlt[s:e], dtype=torch.bool, device=device)
        yb = torch.tensor(labels[s:e], dtype=torch.float32, device=device)
        sign = (2.0 * yb - 1.0).view(-1)

        cand_idx = candidate_pool_idx[s:e].copy()
        cand_avail = cand_idx >= 0

        feat_off_b = feat_off[s:e]
        cand_feat = np.zeros((bsz, pool, f), dtype=np.float32)
        vr, vp = np.where(cand_avail)
        cand_feat[vr, vp] = feat_off_b[vr, cand_idx[vr, vp]]

        for kk in range(k_max):
            empty = (~m_cur).any(dim=1)
            empty_np = empty.detach().cpu().numpy()
            if not np.any(empty_np):
                break

            ins_pos = (~m_cur).to(torch.int64).argmax(dim=1)
            with torch.no_grad():
                base_obj = _model_logits(model, x_cur, m_cur) * sign

            gains = np.full((bsz, pool), -1e9, dtype=np.float32)
            for p in range(pool):
                valid_np = cand_avail[:, p] & empty_np
                if not np.any(valid_np):
                    continue

                rows = np.where(valid_np)[0]
                rows_t = torch.tensor(rows, dtype=torch.long, device=device)
                cols_t = ins_pos[rows_t]

                x_mod = x_cur.clone()
                m_mod = m_cur.clone()
                cf_t = torch.tensor(cand_feat[rows, p], dtype=torch.float32, device=device)
                x_mod[rows_t, cols_t, :] = cf_t
                m_mod[rows_t, cols_t] = True

                with torch.no_grad():
                    obj_mod = _model_logits(model, x_mod, m_mod) * sign
                gain = (obj_mod - base_obj).detach().cpu().numpy()
                gains[rows, p] = gain[rows]

            best_p = np.argmax(gains, axis=1)
            best_gain = gains[np.arange(bsz), best_p]
            choose = empty_np & (best_gain > float(gain_min)) & cand_avail[np.arange(bsz), best_p]
            if not np.any(choose):
                break

            choose_rows = np.where(choose)[0]
            for r in choose_rows:
                p = int(best_p[r])
                tok = int(cand_idx[r, p])
                out_order[s + r, kk] = tok
                pos = int(ins_pos[r].item())
                x_cur[r, pos, :] = torch.tensor(cand_feat[r, p], dtype=torch.float32, device=device)
                m_cur[r, pos] = True
                cand_avail[r, p] = False

    return out_order


def build_fixed_k_target_indices(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    novel_mask: np.ndarray,
    greedy_order: np.ndarray,
    ig_scores: np.ndarray,
    k_fixed: int,
) -> np.ndarray:
    n, t, _ = const_off.shape
    out = np.full((n, k_fixed), -1, dtype=np.int64)

    pt = const_off[:, :, 0]
    all_order = np.argsort(-pt, axis=1)

    # IG fallback ordering on novel tokens.
    ig_order = np.full((n, t), -1, dtype=np.int64)
    for i in range(n):
        idx = np.where(novel_mask[i])[0]
        if idx.size == 0:
            continue
        sc = ig_scores[i, idx]
        ord_idx = idx[np.argsort(-sc)]
        ig_order[i, : ord_idx.size] = ord_idx

    for i in range(n):
        chosen: List[int] = []

        # 1) Greedy picks.
        for tok in greedy_order[i]:
            if tok < 0:
                continue
            ti = int(tok)
            if ti not in chosen:
                chosen.append(ti)
            if len(chosen) >= k_fixed:
                break

        # 2) IG fallback among novel tokens.
        if len(chosen) < k_fixed:
            for tok in ig_order[i]:
                if tok < 0:
                    continue
                ti = int(tok)
                if ti not in chosen:
                    chosen.append(ti)
                if len(chosen) >= k_fixed:
                    break

        # 3) Fill from top-pT valid offline tokens if needed.
        if len(chosen) < k_fixed:
            for tok in all_order[i]:
                ti = int(tok)
                if not bool(mask_off[i, ti]):
                    continue
                if ti not in chosen:
                    chosen.append(ti)
                if len(chosen) >= k_fixed:
                    break

        if len(chosen) == 0:
            chosen = [0] * k_fixed
        while len(chosen) < k_fixed:
            chosen.append(chosen[-1])
        out[i] = np.asarray(chosen[:k_fixed], dtype=np.int64)

    return out


def gather_target_tokens(const_off: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    n, k = target_idx.shape
    out = np.zeros((n, k, 4), dtype=np.float32)
    rows = np.arange(n)[:, None]
    out = const_off[rows, target_idx]
    return out.astype(np.float32)


# -------------------------- Predictor model --------------------------- #
class PredictorDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray, target_const: np.ndarray):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.target_const = torch.tensor(target_const, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt.shape[0]

    def __getitem__(self, i: int):
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "target_const": self.target_const[i],
        }



class DistillPredictorDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        hlt_const: np.ndarray,
        target_const: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.hlt_const = torch.tensor(hlt_const, dtype=torch.float32)
        self.target_const = torch.tensor(target_const, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt.shape[0]

    def __getitem__(self, i: int):
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "hlt_const": self.hlt_const[i],
            "target_const": self.target_const[i],
        }


class FixedKHungarianPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        k_fixed: int = 8,
        embed_dim: int = 192,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k_fixed = int(k_fixed)
        self.in_proj = nn.Linear(input_dim, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.query = nn.Parameter(torch.randn(self.k_fixed, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.q_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 4),  # [log_pt, eta, phi, log_E]
        )

    def forward(self, feat_hlt: torch.Tensor, mask_hlt: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(feat_hlt)
        x = x.masked_fill(~mask_hlt.unsqueeze(-1), 0.0)
        x = self.encoder(x, src_key_padding_mask=~mask_hlt)

        mask_f = mask_hlt.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1, keepdim=True) / (mask_f.sum(dim=1, keepdim=True).clamp(min=1.0))

        q = self.query.unsqueeze(0).expand(x.shape[0], -1, -1) + pooled
        q2, _ = self.cross_attn(q, x, x, key_padding_mask=~mask_hlt, need_weights=False)
        q = self.q_norm(q + q2)
        out = self.head(q)
        # keep phi wrapped
        out_phi = wrap_phi_t(out[..., 2])
        out = torch.cat([out[..., :2], out_phi.unsqueeze(-1), out[..., 3:4]], dim=-1)
        return out


def const_to_target_feat(const: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pt = const[..., 0].clamp(min=eps)
    eta = const[..., 1]
    phi = wrap_phi_t(const[..., 2])
    E = const[..., 3].clamp(min=eps)
    log_pt = torch.log(pt)
    log_E = torch.log(E)
    return torch.stack([log_pt, eta, phi, log_E], dim=-1)


def pred_to_const(pred_feat: torch.Tensor, pt_max: float = 1e8, e_max: float = 1e8) -> torch.Tensor:
    log_pt = pred_feat[..., 0]
    eta = pred_feat[..., 1].clamp(min=-5.0, max=5.0)
    phi = wrap_phi_t(pred_feat[..., 2])
    log_E = pred_feat[..., 3]

    pt = torch.exp(log_pt).clamp(min=1e-6, max=pt_max)
    E_raw = torch.exp(log_E).clamp(min=1e-6, max=e_max)
    # enforce rough physical floor E >= pt*cosh(eta)
    E_floor = pt * torch.cosh(eta)
    E = torch.maximum(E_raw, E_floor)

    return torch.stack([pt, eta, phi, E], dim=-1)


def hungarian_loss(
    pred_feat: torch.Tensor,
    target_const: torch.Tensor,
    w_logpt: float,
    w_eta: float,
    w_phi: float,
    w_loge: float,
    w_sep: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    pred_feat: [B,K,4] in target feature space [log_pt, eta, phi, log_E]
    target_const: [B,K,4] raw [pt,eta,phi,E]
    """
    device = pred_feat.device
    bsz, k, _ = pred_feat.shape
    tgt_feat = const_to_target_feat(target_const)

    # Build cost matrix for Hungarian (detached for assignment indices).
    dp = pred_feat[:, :, None, 0] - tgt_feat[:, None, :, 0]
    de = pred_feat[:, :, None, 1] - tgt_feat[:, None, :, 1]
    dphi = wrap_phi_t(pred_feat[:, :, None, 2] - tgt_feat[:, None, :, 2])
    dE = pred_feat[:, :, None, 3] - tgt_feat[:, None, :, 3]

    cost = (
        float(w_logpt) * dp.abs()
        + float(w_eta) * de.abs()
        + float(w_phi) * dphi.abs()
        + float(w_loge) * dE.abs()
    )

    perms = []
    cost_np = cost.detach().cpu().numpy()
    for i in range(bsz):
        r, c = linear_sum_assignment(cost_np[i])
        perm = np.zeros((k,), dtype=np.int64)
        perm[r] = c
        perms.append(perm)
    perm_t = torch.tensor(np.stack(perms, axis=0), dtype=torch.long, device=device)

    # Reorder targets to matched order and compute regression losses.
    tgt_matched = torch.gather(tgt_feat, dim=1, index=perm_t.unsqueeze(-1).expand(-1, -1, 4))

    l_logpt = F.smooth_l1_loss(pred_feat[..., 0], tgt_matched[..., 0])
    l_eta = F.smooth_l1_loss(pred_feat[..., 1], tgt_matched[..., 1])
    l_phi = F.smooth_l1_loss(wrap_phi_t(pred_feat[..., 2] - tgt_matched[..., 2]), torch.zeros_like(pred_feat[..., 2]))
    l_loge = F.smooth_l1_loss(pred_feat[..., 3], tgt_matched[..., 3])

    # Small repulsion in eta-phi to discourage duplicate outputs.
    pe = pred_feat[..., 1]
    pp = pred_feat[..., 2]
    d_eta = pe[:, :, None] - pe[:, None, :]
    d_phi = wrap_phi_t(pp[:, :, None] - pp[:, None, :])
    dR2 = d_eta.pow(2) + d_phi.pow(2)
    eye = torch.eye(k, device=device).unsqueeze(0)
    rep = torch.exp(-dR2 / 0.01) * (1.0 - eye)
    l_sep = rep.mean()

    total = (
        float(w_logpt) * l_logpt
        + float(w_eta) * l_eta
        + float(w_phi) * l_phi
        + float(w_loge) * l_loge
        + float(w_sep) * l_sep
    )
    comps = {
        "total": total.detach(),
        "logpt": l_logpt.detach(),
        "eta": l_eta.detach(),
        "phi": l_phi.detach(),
        "loge": l_loge.detach(),
        "sep": l_sep.detach(),
    }
    return total, comps


def train_predictor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    loss_w: Dict[str, float],
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_val = float("inf")
    best_state = None
    no_improve = 0
    history: List[Dict[str, float]] = []

    for ep in range(1, int(epochs) + 1):
        model.train()
        tr_losses = []
        tr_comp = {k: [] for k in ["logpt", "eta", "phi", "loge", "sep"]}

        for batch in train_loader:
            feat = batch["feat_hlt"].to(device)
            mask = batch["mask_hlt"].to(device)
            tgt = batch["target_const"].to(device)

            pred = model(feat, mask)
            loss, comps = hungarian_loss(
                pred_feat=pred,
                target_const=tgt,
                w_logpt=float(loss_w["logpt"]),
                w_eta=float(loss_w["eta"]),
                w_phi=float(loss_w["phi"]),
                w_loge=float(loss_w["loge"]),
                w_sep=float(loss_w["sep"]),
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            tr_losses.append(float(loss.item()))
            for k in tr_comp:
                tr_comp[k].append(float(comps[k].item()))

        # validation
        model.eval()
        va_losses = []
        va_comp = {k: [] for k in ["logpt", "eta", "phi", "loge", "sep"]}
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["feat_hlt"].to(device)
                mask = batch["mask_hlt"].to(device)
                tgt = batch["target_const"].to(device)
                pred = model(feat, mask)
                loss, comps = hungarian_loss(
                    pred_feat=pred,
                    target_const=tgt,
                    w_logpt=float(loss_w["logpt"]),
                    w_eta=float(loss_w["eta"]),
                    w_phi=float(loss_w["phi"]),
                    w_loge=float(loss_w["loge"]),
                    w_sep=float(loss_w["sep"]),
                )
                va_losses.append(float(loss.item()))
                for k in va_comp:
                    va_comp[k].append(float(comps[k].item()))

        sch.step()

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        row = {
            "epoch": float(ep),
            "train_total": tr,
            "val_total": va,
            "train_logpt": float(np.mean(tr_comp["logpt"])) if tr_comp["logpt"] else float("nan"),
            "train_eta": float(np.mean(tr_comp["eta"])) if tr_comp["eta"] else float("nan"),
            "train_phi": float(np.mean(tr_comp["phi"])) if tr_comp["phi"] else float("nan"),
            "train_loge": float(np.mean(tr_comp["loge"])) if tr_comp["loge"] else float("nan"),
            "train_sep": float(np.mean(tr_comp["sep"])) if tr_comp["sep"] else float("nan"),
            "val_logpt": float(np.mean(va_comp["logpt"])) if va_comp["logpt"] else float("nan"),
            "val_eta": float(np.mean(va_comp["eta"])) if va_comp["eta"] else float("nan"),
            "val_phi": float(np.mean(va_comp["phi"])) if va_comp["phi"] else float("nan"),
            "val_loge": float(np.mean(va_comp["loge"])) if va_comp["loge"] else float("nan"),
            "val_sep": float(np.mean(va_comp["sep"])) if va_comp["sep"] else float("nan"),
        }
        history.append(row)

        if ep % 5 == 0:
            print(
                f"Predictor ep {ep}: train_total={tr:.4f}, val_total={va:.4f}, "
                f"val(logpt/eta/phi/loge/sep)=({row['val_logpt']:.4f}/{row['val_eta']:.4f}/{row['val_phi']:.4f}/{row['val_loge']:.4f}/{row['val_sep']:.4f})"
            )

        if np.isfinite(va) and va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(patience):
            print(f"Early stopping predictor at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def compute_features_torch(const: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pt = torch.clamp(const[..., 0], min=1e-8)
    eta = torch.clamp(const[..., 1], min=-5.0, max=5.0)
    phi = const[..., 2]
    ene = torch.clamp(const[..., 3], min=1e-8)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    mask_f = mask.float()
    jet_px = (px * mask_f).sum(dim=1, keepdim=True)
    jet_py = (py * mask_f).sum(dim=1, keepdim=True)
    jet_pz = (pz * mask_f).sum(dim=1, keepdim=True)
    jet_e = (ene * mask_f).sum(dim=1, keepdim=True)

    jet_pt = torch.sqrt(jet_px * jet_px + jet_py * jet_py) + 1e-8
    jet_p = torch.sqrt(jet_px * jet_px + jet_py * jet_py + jet_pz * jet_pz) + 1e-8
    jet_eta = 0.5 * torch.log(torch.clamp((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), min=1e-8, max=1e8))
    jet_phi = torch.atan2(jet_py, jet_px)

    d_eta = eta - jet_eta
    d_phi = torch.atan2(torch.sin(phi - jet_phi), torch.cos(phi - jet_phi))

    log_pt = torch.log(pt + 1e-8)
    log_e = torch.log(ene + 1e-8)
    log_pt_rel = torch.log(pt / jet_pt + 1e-8)
    log_e_rel = torch.log(ene / (jet_e + 1e-8) + 1e-8)
    d_r = torch.sqrt(d_eta * d_eta + d_phi * d_phi)

    feat = torch.stack([d_eta, d_phi, log_pt, log_e, log_pt_rel, log_e_rel, d_r], dim=-1)
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = torch.clamp(feat, min=-20.0, max=20.0)
    feat = feat * mask.unsqueeze(-1).float()
    return feat


def standardize_torch(feat: torch.Tensor, mask: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    out = (feat - means) / stds
    out = torch.clamp(out, min=-10.0, max=10.0)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = out * mask.unsqueeze(-1).float()
    return out


def train_predictor_with_frozen_oracle_logits(
    predictor: nn.Module,
    frozen_oracle_tagger: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    w_logit: float,
    w_hungarian: float,
    loss_w: Dict[str, float],
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    for p in frozen_oracle_tagger.parameters():
        p.requires_grad = False
    frozen_oracle_tagger.eval()

    means_t = torch.tensor(means.reshape(1, 1, 7), dtype=torch.float32, device=device)
    stds_t = torch.tensor(stds.reshape(1, 1, 7), dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(predictor.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_val = float('inf')
    best_state = None
    no_imp = 0
    hist: List[Dict[str, float]] = []

    def run_epoch(loader: DataLoader, train: bool):
        if train:
            predictor.train()
        else:
            predictor.eval()

        ls_tot, ls_logit, ls_hung = [], [], []
        for batch in loader:
            feat_hlt = batch['feat_hlt'].to(device)
            mask_hlt = batch['mask_hlt'].to(device)
            hlt_const = batch['hlt_const'].to(device)
            tgt_const = batch['target_const'].to(device)

            if train:
                opt.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                pred_feat = predictor(feat_hlt, mask_hlt)
                pred_const = pred_to_const(pred_feat)

                bsz = pred_const.size(0)
                k = pred_const.size(1)
                k_mask = torch.ones((bsz, k), dtype=torch.bool, device=device)

                aug_pred_const = torch.cat([hlt_const, pred_const], dim=1)
                aug_pred_mask = torch.cat([mask_hlt, k_mask], dim=1)

                aug_orc_const = torch.cat([hlt_const, tgt_const], dim=1)
                aug_orc_mask = torch.cat([mask_hlt, k_mask], dim=1)

                feat_pred = compute_features_torch(aug_pred_const, aug_pred_mask)
                feat_pred_std = standardize_torch(feat_pred, aug_pred_mask, means_t, stds_t)

                feat_orc = compute_features_torch(aug_orc_const, aug_orc_mask)
                feat_orc_std = standardize_torch(feat_orc, aug_orc_mask, means_t, stds_t)

                logits_pred = frozen_oracle_tagger(feat_pred_std, aug_pred_mask).squeeze(1)
                with torch.no_grad():
                    logits_orc = frozen_oracle_tagger(feat_orc_std, aug_orc_mask).squeeze(1)

                loss_logit = F.mse_loss(logits_pred, logits_orc)
                loss_hung, _ = hungarian_loss(
                    pred_feat=pred_feat,
                    target_const=tgt_const,
                    w_logpt=float(loss_w['logpt']),
                    w_eta=float(loss_w['eta']),
                    w_phi=float(loss_w['phi']),
                    w_loge=float(loss_w['loge']),
                    w_sep=float(loss_w['sep']),
                )
                loss = float(w_logit) * loss_logit + float(w_hungarian) * loss_hung

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                    opt.step()

            ls_tot.append(float(loss.item()))
            ls_logit.append(float(loss_logit.item()))
            ls_hung.append(float(loss_hung.item()))

        return (
            float(np.mean(ls_tot)) if ls_tot else float('nan'),
            float(np.mean(ls_logit)) if ls_logit else float('nan'),
            float(np.mean(ls_hung)) if ls_hung else float('nan'),
        )

    for ep in range(1, int(epochs) + 1):
        tr_tot, tr_logit, tr_hung = run_epoch(train_loader, True)
        va_tot, va_logit, va_hung = run_epoch(val_loader, False)
        sch.step()

        row = {
            'epoch': float(ep),
            'train_total': tr_tot,
            'train_logit': tr_logit,
            'train_hung': tr_hung,
            'val_total': va_tot,
            'val_logit': va_logit,
            'val_hung': va_hung,
        }
        hist.append(row)

        if ep % 2 == 0 or ep == 1:
            print(
                f"Distill ep {ep}: train(total/logit/hung)=({tr_tot:.4f}/{tr_logit:.4f}/{tr_hung:.4f}) "
                f"val(total/logit/hung)=({va_tot:.4f}/{va_logit:.4f}/{va_hung:.4f})"
            )

        if np.isfinite(va_tot) and va_tot < best_val:
            best_val = float(va_tot)
            best_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= int(patience):
            print(f"Early stopping distill at epoch {ep}")
            break

    if best_state is not None:
        predictor.load_state_dict(best_state)
    return predictor, hist


# ------------------------------ Main flow ------------------------------ #
def build_loaders(
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pin = torch.cuda.is_available()
    ds_tr = JetDataset(feat[train_idx], mask[train_idx], labels[train_idx])
    ds_va = JetDataset(feat[val_idx], mask[val_idx], labels[val_idx])
    ds_te = JetDataset(feat[test_idx], mask[test_idx], labels[test_idx])

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return dl_tr, dl_va, dl_te


def train_eval_tagger(
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg_training: Dict,
    cfg_model: Dict,
    device: torch.device,
    num_workers: int,
    name: str,
    return_model: bool = False,
):
    bs = int(cfg_training["batch_size"])
    dl_tr, dl_va, dl_te = build_loaders(feat, mask, labels, train_idx, val_idx, test_idx, bs, num_workers)
    model = ParticleTransformer(input_dim=7, **cfg_model).to(device)
    model = train_single_view_classifier_auc(model, dl_tr, dl_va, device, cfg_training, name=name)
    auc, preds, labs = eval_classifier(model, dl_te, device)
    fpr, tpr, _ = roc_curve(labs, preds)
    metrics = {
        "auc": float(auc),
        "fpr30": float(fpr_at_target_tpr(fpr, tpr, 0.30)),
        "fpr50": float(fpr_at_target_tpr(fpr, tpr, 0.50)),
    }
    if return_model:
        return metrics, model
    return metrics


def build_augmented_constituents(
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    add_const: np.ndarray,
    aug_max_constits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = const_hlt.shape[0]
    out_const = np.zeros((n, aug_max_constits, 4), dtype=np.float32)
    out_mask = np.zeros((n, aug_max_constits), dtype=bool)

    for i in range(n):
        h = const_hlt[i, mask_hlt[i]]
        a = add_const[i]
        merged = np.concatenate([h, a], axis=0)
        if merged.shape[0] > 1:
            ord_idx = np.argsort(-merged[:, 0])
            merged = merged[ord_idx]
        keep = min(aug_max_constits, merged.shape[0])
        out_const[i, :keep] = merged[:keep]
        out_mask[i, :keep] = True
    return out_const, out_mask


def predict_added_tokens(
    model: nn.Module,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    n = feat_hlt.shape[0]
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            xb = torch.tensor(feat_hlt[s:e], dtype=torch.float32, device=device)
            mb = torch.tensor(mask_hlt[s:e], dtype=torch.bool, device=device)
            pred_feat = model(xb, mb)
            pred_const = pred_to_const(pred_feat)
            outs.append(pred_const.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)



def build_augmented_view_from_predictor(
    predictor: nn.Module,
    feat_hlt_std: np.ndarray,
    hlt_const: np.ndarray,
    hlt_mask: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
    pred_batch_size: int,
    aug_max_constits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_add = predict_added_tokens(
        model=predictor,
        feat_hlt=feat_hlt_std[idx],
        mask_hlt=hlt_mask[idx],
        device=device,
        batch_size=int(pred_batch_size),
    )
    aug_const, aug_mask = build_augmented_constituents(
        const_hlt=hlt_const[idx],
        mask_hlt=hlt_mask[idx],
        add_const=pred_add,
        aug_max_constits=int(aug_max_constits),
    )
    feat_aug = compute_features(aug_const, aug_mask)
    feat_aug_std = standardize(feat_aug, aug_mask, means, stds)
    return feat_aug_std, aug_mask


def eval_tagger_model_on_arrays(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, float]:
    pin = torch.cuda.is_available()
    ds = JetDataset(feat, mask, labels)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=pin)
    auc, preds, labs = eval_classifier(model, dl, device)
    fpr, tpr, _ = roc_curve(labs, preds)
    return {
        "auc": float(auc),
        "fpr30": float(fpr_at_target_tpr(fpr, tpr, 0.30)),
        "fpr50": float(fpr_at_target_tpr(fpr, tpr, 0.50)),
    }


def joint_finetune_predictor_and_tagger_alternating(
    predictor: nn.Module,
    tagger: nn.Module,
    feat_hlt_std: np.ndarray,
    hlt_const: np.ndarray,
    hlt_mask: np.ndarray,
    labels: np.ndarray,
    target_const: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg_training: Dict,
    device: torch.device,
    num_workers: int,
    pred_batch_size: int,
    pred_weight_decay: float,
    aug_max_constits: int,
    loss_w: Dict[str, float],
    joint_epochs: int,
    joint_patience: int,
    joint_lr_pred: float,
    joint_lr_tagger: float,
) -> Tuple[nn.Module, nn.Module, List[Dict[str, float]]]:
    pin = torch.cuda.is_available()

    ds_pred_tr = PredictorDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], target_const[train_idx])
    ds_pred_va = PredictorDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], target_const[val_idx])
    dl_pred_tr = DataLoader(
        ds_pred_tr,
        batch_size=int(pred_batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(num_workers),
        pin_memory=pin,
    )
    dl_pred_va = DataLoader(
        ds_pred_va,
        batch_size=int(pred_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin,
    )

    bs_cls = int(cfg_training["batch_size"])
    opt_pred = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(joint_lr_pred),
        weight_decay=float(pred_weight_decay),
    )
    opt_tagger = torch.optim.AdamW(
        tagger.parameters(),
        lr=float(joint_lr_tagger),
        weight_decay=float(cfg_training["weight_decay"]),
    )

    best_auc = float("-inf")
    no_imp = 0
    best_pred_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
    best_tag_state = {k: v.detach().cpu().clone() for k, v in tagger.state_dict().items()}
    history: List[Dict[str, float]] = []

    for ep in range(1, int(joint_epochs) + 1):
        predictor.train()
        pred_tr_losses: List[float] = []
        for batch in dl_pred_tr:
            feat = batch["feat_hlt"].to(device)
            mask = batch["mask_hlt"].to(device)
            tgt = batch["target_const"].to(device)
            opt_pred.zero_grad(set_to_none=True)
            pred_feat = predictor(feat, mask)
            loss_pred, _ = hungarian_loss(
                pred_feat=pred_feat,
                target_const=tgt,
                w_logpt=float(loss_w["logpt"]),
                w_eta=float(loss_w["eta"]),
                w_phi=float(loss_w["phi"]),
                w_loge=float(loss_w["loge"]),
                w_sep=float(loss_w["sep"]),
            )
            loss_pred.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt_pred.step()
            pred_tr_losses.append(float(loss_pred.item()))

        predictor.eval()
        pred_va_losses: List[float] = []
        with torch.no_grad():
            for batch in dl_pred_va:
                feat = batch["feat_hlt"].to(device)
                mask = batch["mask_hlt"].to(device)
                tgt = batch["target_const"].to(device)
                pred_feat = predictor(feat, mask)
                loss_pred, _ = hungarian_loss(
                    pred_feat=pred_feat,
                    target_const=tgt,
                    w_logpt=float(loss_w["logpt"]),
                    w_eta=float(loss_w["eta"]),
                    w_phi=float(loss_w["phi"]),
                    w_loge=float(loss_w["loge"]),
                    w_sep=float(loss_w["sep"]),
                )
                pred_va_losses.append(float(loss_pred.item()))

        feat_aug_tr, mask_aug_tr = build_augmented_view_from_predictor(
            predictor=predictor,
            feat_hlt_std=feat_hlt_std,
            hlt_const=hlt_const,
            hlt_mask=hlt_mask,
            means=means,
            stds=stds,
            idx=train_idx,
            device=device,
            pred_batch_size=int(pred_batch_size),
            aug_max_constits=int(aug_max_constits),
        )
        feat_aug_va, mask_aug_va = build_augmented_view_from_predictor(
            predictor=predictor,
            feat_hlt_std=feat_hlt_std,
            hlt_const=hlt_const,
            hlt_mask=hlt_mask,
            means=means,
            stds=stds,
            idx=val_idx,
            device=device,
            pred_batch_size=int(pred_batch_size),
            aug_max_constits=int(aug_max_constits),
        )

        ds_cls_tr = JetDataset(feat_aug_tr, mask_aug_tr, labels[train_idx])
        ds_cls_va = JetDataset(feat_aug_va, mask_aug_va, labels[val_idx])
        dl_cls_tr = DataLoader(
            ds_cls_tr,
            batch_size=bs_cls,
            shuffle=True,
            drop_last=True,
            num_workers=int(num_workers),
            pin_memory=pin,
        )
        dl_cls_va = DataLoader(
            ds_cls_va,
            batch_size=bs_cls,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=pin,
        )

        cls_train_loss, cls_train_auc = train_classifier(tagger, dl_cls_tr, opt_tagger, device)
        val_auc, val_preds, val_labs = eval_classifier(tagger, dl_cls_va, device)
        val_fpr, val_tpr, _ = roc_curve(val_labs, val_preds)
        val_fpr50 = fpr_at_target_tpr(val_fpr, val_tpr, 0.50)

        row = {
            "epoch": float(ep),
            "pred_train_loss": float(np.mean(pred_tr_losses)) if pred_tr_losses else float("nan"),
            "pred_val_loss": float(np.mean(pred_va_losses)) if pred_va_losses else float("nan"),
            "cls_train_loss": float(cls_train_loss),
            "cls_train_auc": float(cls_train_auc),
            "cls_val_auc": float(val_auc),
            "cls_val_fpr50": float(val_fpr50),
        }
        history.append(row)

        print(
            f"Joint ep {ep}: pred(train/val)=({row['pred_train_loss']:.4f}/{row['pred_val_loss']:.4f}) "
            f"cls(train_auc/val_auc/val_fpr50)=({row['cls_train_auc']:.4f}/{row['cls_val_auc']:.4f}/{row['cls_val_fpr50']:.6f})"
        )

        if np.isfinite(val_auc) and float(val_auc) > best_auc:
            best_auc = float(val_auc)
            no_imp = 0
            best_pred_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
            best_tag_state = {k: v.detach().cpu().clone() for k, v in tagger.state_dict().items()}
        else:
            no_imp += 1

        if no_imp >= int(joint_patience):
            print(f"Early stopping joint fine-tune at epoch {ep}")
            break

    predictor.load_state_dict(best_pred_state)
    tagger.load_state_dict(best_tag_state)
    return predictor, tagger, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-K Hungarian predictor for additive constituent recovery")
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="checkpoints/fixedk_hungarian_predictor")
    parser.add_argument("--run_name", type=str, default="fixedk8_hungarian_run")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--n_train_jets", type=int, default=95000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--n_train_split", type=int, default=35000)
    parser.add_argument("--n_val_split", type=int, default=10000)
    parser.add_argument("--n_test_split", type=int, default=50000)

    parser.add_argument("--merge_radius", type=float, default=0.01)
    parser.add_argument("--eff_plateau_barrel", type=float, default=0.98)
    parser.add_argument("--eff_plateau_endcap", type=float, default=0.94)
    parser.add_argument("--smear_a", type=float, default=0.35)
    parser.add_argument("--smear_b", type=float, default=0.012)
    parser.add_argument("--smear_c", type=float, default=0.08)

    parser.add_argument("--k_fixed", type=int, default=8)
    parser.add_argument("--novel_dr_match", type=float, default=0.02)
    parser.add_argument("--ig_steps", type=int, default=8)
    parser.add_argument("--greedy_pool", type=int, default=12)
    parser.add_argument("--greedy_gain_min", type=float, default=0.0)

    parser.add_argument("--predictor_embed_dim", type=int, default=192)
    parser.add_argument("--predictor_heads", type=int, default=8)
    parser.add_argument("--predictor_layers", type=int, default=4)
    parser.add_argument("--predictor_ff_dim", type=int, default=512)
    parser.add_argument("--predictor_dropout", type=float, default=0.1)

    parser.add_argument("--pred_epochs", type=int, default=60)
    parser.add_argument("--pred_patience", type=int, default=12)
    parser.add_argument("--pred_batch_size", type=int, default=256)
    parser.add_argument("--pred_lr", type=float, default=3e-4)
    parser.add_argument("--pred_weight_decay", type=float, default=1e-5)
    parser.add_argument("--pred_warmup_epochs", type=int, default=3)

    parser.add_argument("--loss_w_logpt", type=float, default=1.0)
    parser.add_argument("--loss_w_eta", type=float, default=0.6)
    parser.add_argument("--loss_w_phi", type=float, default=0.6)
    parser.add_argument("--loss_w_loge", type=float, default=0.7)
    parser.add_argument("--loss_w_sep", type=float, default=0.02)

    parser.add_argument("--aug_max_constits", type=int, default=-1)

    parser.add_argument("--distill_epochs", type=int, default=20)
    parser.add_argument("--distill_patience", type=int, default=6)
    parser.add_argument("--distill_lr", type=float, default=1e-4)
    parser.add_argument("--distill_weight_decay", type=float, default=1e-5)
    parser.add_argument("--distill_warmup_epochs", type=int, default=2)
    parser.add_argument("--distill_w_logit", type=float, default=1.0)
    parser.add_argument("--distill_w_hungarian", type=float, default=0.15)

    parser.add_argument("--enable_joint_finetune", action="store_true")
    parser.add_argument("--joint_epochs", type=int, default=12)
    parser.add_argument("--joint_patience", type=int, default=4)
    parser.add_argument("--joint_lr_pred", type=float, default=1e-4)
    parser.add_argument("--joint_lr_tagger", type=float, default=2e-5)

    # Optional classifier override for teacher/baseline/added
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=-1.0)
    parser.add_argument("--weight_decay", type=float, default=-1.0)
    parser.add_argument("--warmup_epochs", type=int, default=-1)

    args = parser.parse_args()
    set_seed(int(args.seed))

    cfg = deepcopy(BASE_CONFIG)
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    if int(args.batch_size) > 0:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if int(args.epochs) > 0:
        cfg["training"]["epochs"] = int(args.epochs)
    if int(args.patience) > 0:
        cfg["training"]["patience"] = int(args.patience)
    if float(args.lr) > 0:
        cfg["training"]["lr"] = float(args.lr)
    if float(args.weight_decay) > 0:
        cfg["training"]["weight_decay"] = float(args.weight_decay)
    if int(args.warmup_epochs) >= 0:
        cfg["training"]["warmup_epochs"] = int(args.warmup_epochs)

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

    max_jets_needed = int(args.offset_jets + args.n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=int(args.max_constits),
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}"
        )

    const_raw = all_const_full[args.offset_jets : args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets : args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, _, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )

    print("Computing base features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    n = len(labels)
    idx = np.arange(n)
    n_train_split = int(args.n_train_split)
    n_val_split = int(args.n_val_split)
    n_test_split = int(args.n_test_split)
    total_need = n_train_split + n_val_split + n_test_split
    if total_need > n:
        raise ValueError(f"Split counts exceed available jets: {total_need} > {n}")
    if total_need < n:
        idx_use, _ = train_test_split(
            idx,
            train_size=total_need,
            random_state=int(args.seed),
            stratify=labels,
        )
    else:
        idx_use = idx

    train_idx, rem_idx = train_test_split(
        idx_use,
        train_size=n_train_split,
        random_state=int(args.seed),
        stratify=labels[idx_use],
    )
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=n_val_split,
        test_size=n_test_split,
        random_state=int(args.seed),
        stratify=labels[rem_idx],
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    bs_cls = int(cfg["training"]["batch_size"])
    nw = int(args.num_workers)

    # ----------------- Teacher / baseline ----------------- #
    print("\n" + "=" * 70)
    print("Training Teacher (Offline)")
    print("=" * 70)
    dl_tr_off, dl_va_off, dl_te_off = build_loaders(
        feat_off_std, masks_off, labels, train_idx, val_idx, test_idx, bs_cls, nw
    )
    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = train_single_view_classifier_auc(teacher, dl_tr_off, dl_va_off, device, cfg["training"], name="Teacher")
    auc_teacher, preds_teacher, labs_t = eval_classifier(teacher, dl_te_off, device)
    fpr_t, tpr_t, _ = roc_curve(labs_t, preds_teacher)
    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.30)
    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)

    print("\n" + "=" * 70)
    print("Training Baseline (HLT)")
    print("=" * 70)
    dl_tr_hlt, dl_va_hlt, dl_te_hlt = build_loaders(
        feat_hlt_std, hlt_mask, labels, train_idx, val_idx, test_idx, bs_cls, nw
    )
    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = train_single_view_classifier_auc(baseline, dl_tr_hlt, dl_va_hlt, device, cfg["training"], name="Baseline-HLT")
    auc_baseline, preds_baseline, labs_b = eval_classifier(baseline, dl_te_hlt, device)
    assert np.array_equal(labs_t.astype(np.float32), labs_b.astype(np.float32))
    fpr_b, tpr_b, _ = roc_curve(labs_b, preds_baseline)
    fpr30_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.30)
    fpr50_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.50)

    # ----------------- Build fixed-K oracle targets ----------------- #
    print("\nComputing novelty mask...")
    novel_mask = compute_novel_mask(
        const_off=const_off,
        mask_off=masks_off,
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
        dr_match=float(args.novel_dr_match),
        chunk_size=2048,
    )

    print("Computing IG scores (for greedy pool + fallback)...")
    ig_scores = compute_ig_scores(
        model=teacher,
        feat=feat_off_std,
        mask=masks_off,
        labels=labels,
        device=device,
        batch_size=bs_cls,
        steps=int(args.ig_steps),
    )

    k_fixed = int(args.k_fixed)
    greedy_pool = int(args.greedy_pool)
    greedy_cand_idx = np.full((n, greedy_pool), -1, dtype=np.int64)
    for i in range(n):
        idxv = np.where(novel_mask[i])[0]
        if idxv.size == 0:
            continue
        sc = ig_scores[i, idxv]
        ordv = idxv[np.argsort(-sc)]
        take = min(greedy_pool, ordv.size)
        greedy_cand_idx[i, :take] = ordv[:take]

    print("Computing greedy oracle order...")
    greedy_order = compute_greedy_insertion_order(
        model=teacher,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        feat_off=feat_off_std,
        labels=labels,
        candidate_pool_idx=greedy_cand_idx,
        k_max=k_fixed,
        gain_min=float(args.greedy_gain_min),
        device=device,
        batch_size=bs_cls,
    )

    target_idx = build_fixed_k_target_indices(
        const_off=const_off,
        mask_off=masks_off,
        novel_mask=novel_mask,
        greedy_order=greedy_order,
        ig_scores=ig_scores,
        k_fixed=k_fixed,
    )
    target_const = gather_target_tokens(const_off, target_idx)

    # ----------------- Train predictor ----------------- #
    print("\n" + "=" * 70)
    print(f"Training Fixed-K Predictor (K={k_fixed}) with Hungarian loss")
    print("=" * 70)

    ds_pred_train = PredictorDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], target_const[train_idx])
    ds_pred_val = PredictorDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], target_const[val_idx])

    pin = torch.cuda.is_available()
    dl_pred_train = DataLoader(
        ds_pred_train,
        batch_size=int(args.pred_batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=pin,
    )
    dl_pred_val = DataLoader(
        ds_pred_val,
        batch_size=int(args.pred_batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )

    predictor = FixedKHungarianPredictor(
        input_dim=7,
        k_fixed=k_fixed,
        embed_dim=int(args.predictor_embed_dim),
        num_heads=int(args.predictor_heads),
        num_layers=int(args.predictor_layers),
        ff_dim=int(args.predictor_ff_dim),
        dropout=float(args.predictor_dropout),
    ).to(device)

    predictor, pred_hist = train_predictor(
        model=predictor,
        train_loader=dl_pred_train,
        val_loader=dl_pred_val,
        device=device,
        epochs=int(args.pred_epochs),
        patience=int(args.pred_patience),
        lr=float(args.pred_lr),
        weight_decay=float(args.pred_weight_decay),
        warmup_epochs=int(args.pred_warmup_epochs),
        loss_w={
            "logpt": float(args.loss_w_logpt),
            "eta": float(args.loss_w_eta),
            "phi": float(args.loss_w_phi),
            "loge": float(args.loss_w_loge),
            "sep": float(args.loss_w_sep),
        },
    )

    # ----------------- Build predicted-added view ----------------- #
    print("\nGenerating predicted added tokens...")
    pred_add_const = predict_added_tokens(
        model=predictor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        device=device,
        batch_size=int(args.pred_batch_size),
    )

    if int(args.aug_max_constits) > 0:
        aug_max_constits = int(args.aug_max_constits)
    else:
        aug_max_constits = int(args.max_constits) + int(k_fixed)

    aug_pred_const, aug_pred_mask = build_augmented_constituents(
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
        add_const=pred_add_const,
        aug_max_constits=aug_max_constits,
    )

    # Oracle reference (HLT + true fixed-K targets)
    oracle_add_const = target_const.copy()
    aug_oracle_const, aug_oracle_mask = build_augmented_constituents(
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
        add_const=oracle_add_const,
        aug_max_constits=aug_max_constits,
    )

    feat_aug_pred = compute_features(aug_pred_const, aug_pred_mask)
    feat_aug_oracle = compute_features(aug_oracle_const, aug_oracle_mask)
    feat_aug_pred_std = standardize(feat_aug_pred, aug_pred_mask, means, stds)
    feat_aug_oracle_std = standardize(feat_aug_oracle, aug_oracle_mask, means, stds)

    # ----------------- Oracle-tagger distillation path ----------------- #
    print("\n" + "=" * 70)
    print("Training oracle added-view tagger (HLT + Oracle-K)")
    print("=" * 70)
    oracle_metrics_train, oracle_tagger = train_eval_tagger(
        feat=feat_aug_oracle_std,
        mask=aug_oracle_mask,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cfg_training=cfg["training"],
        cfg_model=cfg["model"],
        device=device,
        num_workers=nw,
        name=f"OracleTagger-K{k_fixed}",
        return_model=True,
    )
    torch.save(oracle_tagger.state_dict(), save_root / "oracle_tagger_frozen.pt")

    print("\nEvaluating frozen oracle tagger on predicted-added view (pre-distill)")
    pred_metrics_pre = eval_tagger_model_on_arrays(
        model=oracle_tagger,
        feat=feat_aug_pred_std[test_idx],
        mask=aug_pred_mask[test_idx],
        labels=labels[test_idx],
        device=device,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=nw,
    )

    print("\n" + "=" * 70)
    print("Distilling predictor with frozen oracle-tagger logits")
    print("=" * 70)
    ds_dist_tr = DistillPredictorDataset(
        feat_hlt=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        hlt_const=hlt_const[train_idx],
        target_const=target_const[train_idx],
    )
    ds_dist_va = DistillPredictorDataset(
        feat_hlt=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        hlt_const=hlt_const[val_idx],
        target_const=target_const[val_idx],
    )
    pin = torch.cuda.is_available()
    dl_dist_tr = DataLoader(
        ds_dist_tr,
        batch_size=int(args.pred_batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=pin,
    )
    dl_dist_va = DataLoader(
        ds_dist_va,
        batch_size=int(args.pred_batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )

    predictor_pre_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
    torch.save(predictor_pre_state, save_root / "predictor_predistill.pt")

    predictor, distill_hist = train_predictor_with_frozen_oracle_logits(
        predictor=predictor,
        frozen_oracle_tagger=oracle_tagger,
        train_loader=dl_dist_tr,
        val_loader=dl_dist_va,
        means=means,
        stds=stds,
        device=device,
        epochs=int(args.distill_epochs),
        patience=int(args.distill_patience),
        lr=float(args.distill_lr),
        weight_decay=float(args.distill_weight_decay),
        warmup_epochs=int(args.distill_warmup_epochs),
        w_logit=float(args.distill_w_logit),
        w_hungarian=float(args.distill_w_hungarian),
        loss_w={
            "logpt": float(args.loss_w_logpt),
            "eta": float(args.loss_w_eta),
            "phi": float(args.loss_w_phi),
            "loge": float(args.loss_w_loge),
            "sep": float(args.loss_w_sep),
        },
    )

    torch.save(predictor.state_dict(), save_root / "predictor_postdistill.pt")

    pred_add_const_post = predict_added_tokens(
        model=predictor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        device=device,
        batch_size=int(args.pred_batch_size),
    )
    aug_pred_post_const, aug_pred_post_mask = build_augmented_constituents(
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
        add_const=pred_add_const_post,
        aug_max_constits=aug_max_constits,
    )
    feat_aug_pred_post = compute_features(aug_pred_post_const, aug_pred_post_mask)
    feat_aug_pred_post_std = standardize(feat_aug_pred_post, aug_pred_post_mask, means, stds)

    print("\nEvaluating frozen oracle tagger on predicted-added view (post-distill)")
    pred_metrics_post = eval_tagger_model_on_arrays(
        model=oracle_tagger,
        feat=feat_aug_pred_post_std[test_idx],
        mask=aug_pred_post_mask[test_idx],
        labels=labels[test_idx],
        device=device,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=nw,
    )

    oracle_metrics = eval_tagger_model_on_arrays(
        model=oracle_tagger,
        feat=feat_aug_oracle_std[test_idx],
        mask=aug_oracle_mask[test_idx],
        labels=labels[test_idx],
        device=device,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=nw,
    )

    # Recovery metrics
    def rec_auc(x):
        den = float(auc_teacher - auc_baseline)
        return float((x - auc_baseline) / den) if abs(den) > 1e-12 else float("nan")

    def rec_fpr50(x):
        den = float(fpr50_baseline - fpr50_teacher)
        return float((fpr50_baseline - x) / den) if abs(den) > 1e-12 else float("nan")

    summary = {
        "teacher": {
            "auc": float(auc_teacher),
            "fpr30": float(fpr30_teacher),
            "fpr50": float(fpr50_teacher),
        },
        "baseline_hlt": {
            "auc": float(auc_baseline),
            "fpr30": float(fpr30_baseline),
            "fpr50": float(fpr50_baseline),
        },
        "oracle_tagger_on_oracle": {
            **oracle_metrics,
            "recovery_auc": rec_auc(oracle_metrics["auc"]),
            "recovery_fpr50": rec_fpr50(oracle_metrics["fpr50"]),
        },
        "oracle_tagger_on_pred_pre": {
            **pred_metrics_pre,
            "recovery_auc": rec_auc(pred_metrics_pre["auc"]),
            "recovery_fpr50": rec_fpr50(pred_metrics_pre["fpr50"]),
        },
        "oracle_tagger_on_pred_post": {
            **pred_metrics_post,
            "recovery_auc": rec_auc(pred_metrics_post["auc"]),
            "recovery_fpr50": rec_fpr50(pred_metrics_post["fpr50"]),
        },
        "k_fixed": int(k_fixed),
        "mean_added_pred_tokens_pre": float(np.mean(np.sum(pred_add_const[:, :, 0] > 0, axis=1))),
        "mean_added_pred_tokens_post": float(np.mean(np.sum(pred_add_const_post[:, :, 0] > 0, axis=1))),
        "mean_added_oracle_tokens": float(k_fixed),
    }

    with open(save_root / "oraclelogit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(save_root / "predictor_history.json", "w", encoding="utf-8") as f:
        json.dump(pred_hist, f, indent=2)
    with open(save_root / "distill_history.json", "w", encoding="utf-8") as f:
        json.dump(distill_hist, f, indent=2)

    run_cfg = {
        "args": vars(args),
        "training": cfg["training"],
        "model": cfg["model"],
        "hlt_effects": cfg["hlt_effects"],
        "split": {
            "n_train_split": int(len(train_idx)),
            "n_val_split": int(len(val_idx)),
            "n_test_split": int(len(test_idx)),
        },
    }
    with open(save_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Teacher  : AUC={auc_teacher:.4f} FPR50={fpr50_teacher:.6f}")
    print(f"Baseline : AUC={auc_baseline:.4f} FPR50={fpr50_baseline:.6f}")
    print(
        f"OracleTagger on Oracle-K: AUC={oracle_metrics['auc']:.4f} FPR50={oracle_metrics['fpr50']:.6f} "
        f"RecAUC={summary['oracle_tagger_on_oracle']['recovery_auc']:.3f} "
        f"RecFPR50={summary['oracle_tagger_on_oracle']['recovery_fpr50']:.3f}"
    )
    print(
        f"OracleTagger on Pred-K PreDistill: AUC={pred_metrics_pre['auc']:.4f} FPR50={pred_metrics_pre['fpr50']:.6f} "
        f"RecAUC={summary['oracle_tagger_on_pred_pre']['recovery_auc']:.3f} "
        f"RecFPR50={summary['oracle_tagger_on_pred_pre']['recovery_fpr50']:.3f}"
    )
    print(
        f"OracleTagger on Pred-K PostDistill: AUC={pred_metrics_post['auc']:.4f} FPR50={pred_metrics_post['fpr50']:.6f} "
        f"RecAUC={summary['oracle_tagger_on_pred_post']['recovery_auc']:.3f} "
        f"RecFPR50={summary['oracle_tagger_on_pred_post']['recovery_fpr50']:.3f}"
    )
    print(f"Saved outputs to: {save_root}")


if __name__ == "__main__":
    main()

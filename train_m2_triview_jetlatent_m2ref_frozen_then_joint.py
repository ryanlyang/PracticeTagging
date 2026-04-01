#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tri-view top-tagger:
  View-1: HLT
  View-2: jet-latent set2set reconstructor corrected view
  View-3: reference m2 reconstructor corrected view

Workflow:
1) Load dataset/splits from reference m2 run.
2) Load reconstructors:
   - Jet-latent set2set reconstructor (from dependent run output)
   - Reference m2 reconstructor (from provided run output)
3) Train tri-view tagger with reconstructors frozen.
4) Joint-finetune tri-view tagger + both reconstructors.
5) Save metrics and val/test score arrays.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as m2mod
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_jetlatent_set2set as jlmod


FEAT_CLIP_ABS = 50.0


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def safe_load_state(model: nn.Module, state: Dict[str, torch.Tensor], name: str) -> None:
    miss, unexp = model.load_state_dict(state, strict=False)
    if len(miss) > 0 or len(unexp) > 0:
        print(f"[{name}] strict=False load: missing={len(miss)}, unexpected={len(unexp)}")


def sanitize_numpy_features(x: np.ndarray, clip_abs: float = FEAT_CLIP_ABS) -> np.ndarray:
    y = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(clip_abs) > 0:
        np.clip(y, -float(clip_abs), float(clip_abs), out=y)
    return y


def sanitize_torch_features(x: torch.Tensor, clip_abs: float = FEAT_CLIP_ABS) -> torch.Tensor:
    y = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if float(clip_abs) > 0:
        y = torch.clamp(y, min=-float(clip_abs), max=float(clip_abs))
    return y


def sanitize_torch_logits(z: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0)


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> float:
    labels = labels.astype(np.float32)
    scores = np.nan_to_num(scores.astype(np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    finite = np.isfinite(labels) & np.isfinite(scores)
    if not np.any(finite):
        return float("nan")
    labels = labels[finite]
    scores = scores[finite]
    if np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(m2mod.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))


def auc_and_fpr50(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> Tuple[float, float]:
    labels = labels.astype(np.float32)
    scores = np.nan_to_num(scores.astype(np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    finite = np.isfinite(labels) & np.isfinite(scores)
    if not np.any(finite):
        return float("nan"), float("nan")
    labels = labels[finite]
    scores = scores[finite]
    auc = float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else float("nan")
    return auc, fpr_at_tpr(labels, scores, target_tpr)


def low_fpr_surrogate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_tpr: float = 0.50,
    tau: float = 0.05,
) -> torch.Tensor:
    logits = sanitize_torch_logits(logits)
    probs = torch.sigmoid(logits)
    pos = probs[labels > 0.5]
    neg = probs[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.zeros((), device=logits.device)
    q = float(max(0.0, min(1.0, 1.0 - target_tpr)))
    thr = torch.quantile(pos.detach(), q=q)
    neg_term = torch.sigmoid((neg - thr) / max(float(tau), 1e-4)).mean()
    pos_term = torch.sigmoid((thr - pos) / max(float(tau), 1e-4)).mean()
    return neg_term + 0.5 * pos_term


class FrozenTriViewDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        feat_jl: np.ndarray,
        mask_jl: np.ndarray,
        feat_m2: np.ndarray,
        mask_m2: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt = feat_hlt.astype(np.float32)
        self.mask_hlt = mask_hlt.astype(bool)
        self.feat_jl = feat_jl.astype(np.float32)
        self.mask_jl = mask_jl.astype(bool)
        self.feat_m2 = feat_m2.astype(np.float32)
        self.mask_m2 = mask_m2.astype(bool)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "feat_jl": self.feat_jl[i],
            "mask_jl": self.mask_jl[i],
            "feat_m2": self.feat_m2[i],
            "mask_m2": self.mask_m2[i],
            "label": self.labels[i],
        }


class JointRecoDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt = feat_hlt.astype(np.float32)
        self.mask_hlt = mask_hlt.astype(bool)
        self.const_hlt = const_hlt.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "label": self.labels[i],
        }


class LiteViewEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(d_model))
        self.head = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), 1),
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(feat)
        x = self.encoder(x, src_key_padding_mask=~mask)
        w = mask.float().unsqueeze(-1)
        pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        pooled = self.norm(pooled)
        return self.head(pooled).squeeze(1)


class TriViewTopTagger(nn.Module):
    def __init__(
        self,
        d_model: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.v_hlt = LiteViewEncoder(7, d_model=d_model, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout)
        self.v_jl = LiteViewEncoder(10, d_model=d_model, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout)
        self.v_m2 = LiteViewEncoder(10, d_model=d_model, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        feat_jl: torch.Tensor,
        mask_jl: torch.Tensor,
        feat_m2: torch.Tensor,
        mask_m2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        l_hlt = self.v_hlt(feat_hlt, mask_hlt)
        l_jl = self.v_jl(feat_jl, mask_jl)
        l_m2 = self.v_m2(feat_m2, mask_m2)
        per_view = torch.stack([l_hlt, l_jl, l_m2], dim=1)
        out = self.fuse(per_view).squeeze(1)
        return out, per_view


@torch.no_grad()
def predict_single_view_logits(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out = np.zeros(idx.shape[0], dtype=np.float64)
    p = 0
    for start in range(0, len(idx), int(batch_size)):
        end = min(start + int(batch_size), len(idx))
        sl = idx[start:end]
        x = torch.tensor(feat[sl], dtype=torch.float32, device=device)
        m = torch.tensor(mask[sl], dtype=torch.bool, device=device)
        z = model(x, m).squeeze(1)
        k = end - start
        out[p:p + k] = z.detach().cpu().numpy().astype(np.float64)
        p += k
    return out


@torch.no_grad()
def eval_frozen_loader(
    tagger: TriViewTopTagger,
    loader: DataLoader,
    device: torch.device,
    target_tpr: float,
) -> Dict[str, np.ndarray | float]:
    tagger.eval()
    preds = []
    labs = []
    for batch in loader:
        feat_hlt = sanitize_torch_features(batch["feat_hlt"].to(device))
        mask_hlt = batch["mask_hlt"].to(device)
        feat_jl = sanitize_torch_features(batch["feat_jl"].to(device))
        mask_jl = batch["mask_jl"].to(device)
        feat_m2 = sanitize_torch_features(batch["feat_m2"].to(device))
        mask_m2 = batch["mask_m2"].to(device)
        y = batch["label"].to(device)

        z, _ = tagger(feat_hlt, mask_hlt, feat_jl, mask_jl, feat_m2, mask_m2)
        z = sanitize_torch_logits(z)
        p = torch.sigmoid(z)
        preds.append(p.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds).astype(np.float64) if preds else np.zeros(0, dtype=np.float64)
    labs = np.concatenate(labs).astype(np.float32) if labs else np.zeros(0, dtype=np.float32)
    auc, fpr50 = auc_and_fpr50(labs, preds, target_tpr)
    return {"preds": preds, "labels": labs, "auc": float(auc), "fpr50": float(fpr50)}


@torch.no_grad()
def build_corrected_view_numpy_generic(
    reconstructor: nn.Module,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    device: torch.device,
    batch_size: int,
    corrected_weight_floor: float,
    builder_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    n, seq_len, _ = feat_hlt.shape
    feat_b = np.zeros((n, seq_len, 10), dtype=np.float32)
    mask_b = np.zeros((n, seq_len), dtype=bool)

    reconstructor.eval()
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        x = torch.tensor(feat_hlt[start:end], dtype=torch.float32, device=device)
        m = torch.tensor(mask_hlt[start:end], dtype=torch.bool, device=device)
        c = torch.tensor(const_hlt[start:end], dtype=torch.float32, device=device)
        out = reconstructor(x, m, c, stage_scale=1.0)
        fb, mb = builder_fn(
            out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=False,
        )
        feat_b[start:end] = fb.detach().cpu().numpy().astype(np.float32)
        mask_b[start:end] = mb.detach().cpu().numpy().astype(bool)
    return feat_b, mask_b


@torch.no_grad()
def build_triview_numpy(
    split_name: str,
    split_idx: np.ndarray,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    jl_reco: nn.Module,
    m2_reco: nn.Module,
    device: torch.device,
    batch_size: int,
    corrected_weight_floor: float,
) -> Dict[str, np.ndarray]:
    print(f"Building frozen tri-view tensors for {split_name}...")
    feat_hlt_s = feat_hlt_std[split_idx]
    mask_hlt_s = hlt_mask[split_idx]
    const_hlt_s = hlt_const[split_idx]

    feat_jl, mask_jl = build_corrected_view_numpy_generic(
        reconstructor=jl_reco,
        feat_hlt=feat_hlt_s,
        mask_hlt=mask_hlt_s,
        const_hlt=const_hlt_s,
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        builder_fn=jlmod.build_soft_corrected_view_set2set,
    )
    feat_m2, mask_m2 = build_corrected_view_numpy_generic(
        reconstructor=m2_reco,
        feat_hlt=feat_hlt_s,
        mask_hlt=mask_hlt_s,
        const_hlt=const_hlt_s,
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        builder_fn=m2mod.build_soft_corrected_view,
    )

    feat_hlt_s = sanitize_numpy_features(feat_hlt_s)
    feat_jl = sanitize_numpy_features(feat_jl)
    feat_m2 = sanitize_numpy_features(feat_m2)

    return {
        "feat_hlt": feat_hlt_s.astype(np.float32),
        "mask_hlt": mask_hlt_s.astype(bool),
        "feat_jl": feat_jl.astype(np.float32),
        "mask_jl": mask_jl.astype(bool),
        "feat_m2": feat_m2.astype(np.float32),
        "mask_m2": mask_m2.astype(bool),
    }


def train_frozen_phase(
    tagger: TriViewTopTagger,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_rank: float,
    rank_tau: float,
    target_tpr: float,
    select_metric: str,
) -> Dict[str, float]:
    opt = torch.optim.AdamW(tagger.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = m2mod.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for ep in range(int(epochs)):
        tagger.train()
        tr_loss = tr_cls = tr_rank = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt = sanitize_torch_features(batch["feat_hlt"].to(device))
            mask_hlt = batch["mask_hlt"].to(device)
            feat_jl = sanitize_torch_features(batch["feat_jl"].to(device))
            mask_jl = batch["mask_jl"].to(device)
            feat_m2 = sanitize_torch_features(batch["feat_m2"].to(device))
            mask_m2 = batch["mask_m2"].to(device)
            y = batch["label"].to(device)

            opt.zero_grad()
            logits, _ = tagger(feat_hlt, mask_hlt, feat_jl, mask_jl, feat_m2, mask_m2)
            logits = sanitize_torch_logits(logits)
            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = low_fpr_surrogate_loss(logits, y, target_tpr=float(target_tpr), tau=float(rank_tau))
            loss = loss_cls + float(lambda_rank) * loss_rank
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            tr_loss += float(loss.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            tr_rank += float(loss_rank.item()) * bs
            n_tr += bs

        sch.step()
        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)

        val_pack = eval_frozen_loader(tagger, val_loader, device, float(target_tpr))
        va_auc = float(val_pack["auc"])
        va_fpr50 = float(val_pack["fpr50"])

        if str(select_metric).lower() == "fpr50":
            sel = va_fpr50
            improved = np.isfinite(sel) and (sel < best_sel)
        else:
            sel = va_auc
            improved = np.isfinite(sel) and (sel > best_sel)

        if improved:
            best_sel = float(sel)
            best_state = {k: v.detach().cpu().clone() for k, v in tagger.state_dict().items()}
            best_metrics = {
                "best_epoch": int(ep + 1),
                "best_select_metric": str(select_metric).lower(),
                "best_sel": float(best_sel),
                "best_val_auc": float(va_auc),
                "best_val_fpr50": float(va_fpr50),
                "best_train_loss": float(tr_loss),
                "best_train_loss_cls": float(tr_cls),
                "best_train_loss_rank": float(tr_rank),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"FrozenTri ep {ep+1}: train_loss={tr_loss:.5f} (cls={tr_cls:.5f}, rank={tr_rank:.5f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_sel={best_sel:.6f}"
            )
        if no_improve >= int(patience):
            print(f"Early stopping frozen tri-view at epoch {ep+1}")
            break

    if best_state is not None:
        tagger.load_state_dict(best_state)
    return best_metrics


def build_views_torch_from_batch(
    feat_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_hlt: torch.Tensor,
    jl_reco: nn.Module,
    m2_reco: nn.Module,
    corrected_weight_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_jl = jl_reco(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
    feat_jl, mask_jl = jlmod.build_soft_corrected_view_set2set(
        out_jl,
        weight_floor=float(corrected_weight_floor),
        scale_features_by_weight=True,
        include_flags=False,
    )
    out_m2 = m2_reco(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
    feat_m2, mask_m2 = m2mod.build_soft_corrected_view(
        out_m2,
        weight_floor=float(corrected_weight_floor),
        scale_features_by_weight=True,
        include_flags=False,
    )
    feat_jl = sanitize_torch_features(feat_jl)
    feat_m2 = sanitize_torch_features(feat_m2)
    return feat_jl, mask_jl, feat_m2, mask_m2


@torch.no_grad()
def eval_joint_dynamic(
    tagger: TriViewTopTagger,
    loader: DataLoader,
    device: torch.device,
    target_tpr: float,
    jl_reco: nn.Module,
    m2_reco: nn.Module,
    corrected_weight_floor: float,
) -> Dict[str, np.ndarray | float]:
    tagger.eval()
    jl_reco.eval()
    m2_reco.eval()
    preds = []
    labs = []
    for batch in loader:
        feat_hlt = sanitize_torch_features(batch["feat_hlt"].to(device))
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = torch.nan_to_num(batch["const_hlt"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        y = batch["label"].to(device)

        feat_jl, mask_jl, feat_m2, mask_m2 = build_views_torch_from_batch(
            feat_hlt=feat_hlt,
            mask_hlt=mask_hlt,
            const_hlt=const_hlt,
            jl_reco=jl_reco,
            m2_reco=m2_reco,
            corrected_weight_floor=float(corrected_weight_floor),
        )
        z, _ = tagger(feat_hlt, mask_hlt, feat_jl, mask_jl, feat_m2, mask_m2)
        z = sanitize_torch_logits(z)
        p = torch.sigmoid(z)
        preds.append(p.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds).astype(np.float64) if preds else np.zeros(0, dtype=np.float64)
    labs = np.concatenate(labs).astype(np.float32) if labs else np.zeros(0, dtype=np.float32)
    auc, fpr50 = auc_and_fpr50(labs, preds, target_tpr)
    return {"preds": preds, "labels": labs, "auc": float(auc), "fpr50": float(fpr50)}


def train_joint_phase(
    tagger: TriViewTopTagger,
    jl_reco: nn.Module,
    m2_reco: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr_tagger: float,
    lr_reco: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_rank: float,
    rank_tau: float,
    corrected_weight_floor: float,
    target_tpr: float,
    select_metric: str,
) -> Dict[str, float]:
    for p in jl_reco.parameters():
        p.requires_grad_(True)
    for p in m2_reco.parameters():
        p.requires_grad_(True)

    params = [
        {"params": tagger.parameters(), "lr": float(lr_tagger)},
        {"params": jl_reco.parameters(), "lr": float(lr_reco)},
        {"params": m2_reco.parameters(), "lr": float(lr_reco)},
    ]
    opt = torch.optim.AdamW(params, lr=float(lr_tagger), weight_decay=float(weight_decay))
    sch = m2mod.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    no_improve = 0
    best_metrics: Dict[str, float] = {}

    for ep in range(int(epochs)):
        tagger.train()
        jl_reco.train()
        m2_reco.train()

        tr_loss = tr_cls = tr_rank = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt = sanitize_torch_features(batch["feat_hlt"].to(device))
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = torch.nan_to_num(batch["const_hlt"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
            y = batch["label"].to(device)

            opt.zero_grad()
            feat_jl, mask_jl, feat_m2, mask_m2 = build_views_torch_from_batch(
                feat_hlt=feat_hlt,
                mask_hlt=mask_hlt,
                const_hlt=const_hlt,
                jl_reco=jl_reco,
                m2_reco=m2_reco,
                corrected_weight_floor=float(corrected_weight_floor),
            )
            logits, _ = tagger(feat_hlt, mask_hlt, feat_jl, mask_jl, feat_m2, mask_m2)
            logits = sanitize_torch_logits(logits)
            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = low_fpr_surrogate_loss(logits, y, target_tpr=float(target_tpr), tau=float(rank_tau))
            loss = loss_cls + float(lambda_rank) * loss_rank
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(jl_reco.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(m2_reco.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            tr_loss += float(loss.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            tr_rank += float(loss_rank.item()) * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)

        val_pack = eval_joint_dynamic(
            tagger=tagger,
            loader=val_loader,
            device=device,
            target_tpr=float(target_tpr),
            jl_reco=jl_reco,
            m2_reco=m2_reco,
            corrected_weight_floor=float(corrected_weight_floor),
        )
        va_auc = float(val_pack["auc"])
        va_fpr50 = float(val_pack["fpr50"])

        if str(select_metric).lower() == "fpr50":
            sel = va_fpr50
            improved = np.isfinite(sel) and (sel < best_sel)
        else:
            sel = va_auc
            improved = np.isfinite(sel) and (sel > best_sel)

        if improved:
            best_sel = float(sel)
            best_state = {
                "tagger": {k: v.detach().cpu().clone() for k, v in tagger.state_dict().items()},
                "jl": {k: v.detach().cpu().clone() for k, v in jl_reco.state_dict().items()},
                "m2": {k: v.detach().cpu().clone() for k, v in m2_reco.state_dict().items()},
            }
            best_metrics = {
                "best_epoch": int(ep + 1),
                "best_select_metric": str(select_metric).lower(),
                "best_sel": float(best_sel),
                "best_val_auc": float(va_auc),
                "best_val_fpr50": float(va_fpr50),
                "best_train_loss": float(tr_loss),
                "best_train_loss_cls": float(tr_cls),
                "best_train_loss_rank": float(tr_rank),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 2 == 0:
            print(
                f"JointTri ep {ep+1}: train_loss={tr_loss:.5f} (cls={tr_cls:.5f}, rank={tr_rank:.5f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_sel={best_sel:.6f}"
            )
        if no_improve >= int(patience):
            print(f"Early stopping joint tri-view at epoch {ep+1}")
            break

    if best_state is not None:
        tagger.load_state_dict(best_state["tagger"])
        jl_reco.load_state_dict(best_state["jl"])
        m2_reco.load_state_dict(best_state["m2"])
    return best_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetlatent_run_dir", type=str, required=True)
    ap.add_argument("--m2_ref_run_dir", type=str, required=True)
    ap.add_argument("--jetlatent_reco_ckpt", type=str, default="offline_reconstructor.pt")
    ap.add_argument("--m2_reco_ckpt", type=str, default="offline_reconstructor.pt")
    ap.add_argument("--m2_baseline_ckpt", type=str, default="baseline.pt")

    ap.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_triview_jetlatent_m2ref",
    )
    ap.add_argument("--run_name", type=str, default="model2_triview_jetlatent_m2ref_150k75k150k_seed0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--corrected_weight_floor", type=float, default=0.03)
    ap.add_argument("--reco_eval_batch_size", type=int, default=256)
    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--select_metric", type=str, choices=["fpr50", "auc"], default="auc")

    ap.add_argument("--frozen_epochs", type=int, default=40)
    ap.add_argument("--frozen_patience", type=int, default=10)
    ap.add_argument("--frozen_batch_size", type=int, default=256)
    ap.add_argument("--frozen_lr", type=float, default=3e-4)
    ap.add_argument("--frozen_weight_decay", type=float, default=1e-4)
    ap.add_argument("--frozen_warmup_epochs", type=int, default=5)
    ap.add_argument("--frozen_lambda_rank", type=float, default=0.2)
    ap.add_argument("--frozen_rank_tau", type=float, default=0.05)

    ap.add_argument("--joint_epochs", type=int, default=12)
    ap.add_argument("--joint_patience", type=int, default=6)
    ap.add_argument("--joint_batch_size", type=int, default=128)
    ap.add_argument("--joint_lr_tagger", type=float, default=1e-4)
    ap.add_argument("--joint_lr_reco", type=float, default=2e-6)
    ap.add_argument("--joint_weight_decay", type=float, default=1e-4)
    ap.add_argument("--joint_warmup_epochs", type=int, default=3)
    ap.add_argument("--joint_lambda_rank", type=float, default=0.2)
    ap.add_argument("--joint_rank_tau", type=float, default=0.05)

    args = ap.parse_args()
    set_seed(int(args.seed))

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    jetlatent_run = Path(args.jetlatent_run_dir)
    m2_ref_run = Path(args.m2_ref_run_dir)

    # Keep exact data/splits from reference m2 run.
    setup_path = m2_ref_run / "data_setup.json"
    split_path = m2_ref_run / "data_splits.npz"
    if not setup_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Missing m2 reference setup/splits in {m2_ref_run}")

    with open(setup_path, "r", encoding="utf-8") as f:
        data_setup = json.load(f)
    split_npz = np.load(split_path)
    train_idx = split_npz["train_idx"].astype(np.int64)
    val_idx = split_npz["val_idx"].astype(np.int64)
    test_idx = split_npz["test_idx"].astype(np.int64)
    means = split_npz["means"].astype(np.float32)
    stds = split_npz["stds"].astype(np.float32)

    train_files = [Path(p) for p in data_setup.get("train_files", [])]
    if len(train_files) == 0:
        raise RuntimeError("data_setup.json has no train_files")

    n_train_jets = int(data_setup.get("n_train_jets"))
    offset_jets = int(data_setup.get("offset_jets", 0))
    max_constits = int(data_setup.get("max_constits", 100))
    hlt_cfg = dict(data_setup.get("hlt_effects", {}))

    cfg = m2mod._deepcopy_config()
    cfg["hlt_effects"].update(hlt_cfg)

    max_jets_needed = int(offset_jets + n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = m2mod.load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}")

    const_raw = all_const_full[offset_jets: offset_jets + n_train_jets]
    labels = all_labels_full[offset_jets: offset_jets + n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, _ = m2mod.apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(data_setup.get("seed", args.seed)),
    )

    print("Computing standardized HLT features...")
    feat_hlt = m2mod.compute_features(hlt_const, hlt_mask)
    feat_hlt_std = m2mod.standardize(feat_hlt, hlt_mask, means, stds)

    # Baseline HLT reference scores for saved arrays.
    baseline_ref = m2mod.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline_path = m2_ref_run / args.m2_baseline_ckpt
    if baseline_path.exists():
        safe_load_state(baseline_ref, load_model_state(baseline_path, device), "m2_baseline")
        preds_hlt_val = 1.0 / (1.0 + np.exp(-predict_single_view_logits(
            baseline_ref, feat_hlt_std, hlt_mask, val_idx, device, int(args.reco_eval_batch_size)
        )))
        preds_hlt_test = 1.0 / (1.0 + np.exp(-predict_single_view_logits(
            baseline_ref, feat_hlt_std, hlt_mask, test_idx, device, int(args.reco_eval_batch_size)
        )))
    else:
        print(f"Warning: baseline checkpoint not found: {baseline_path}")
        preds_hlt_val = np.zeros(val_idx.shape[0], dtype=np.float64)
        preds_hlt_test = np.zeros(test_idx.shape[0], dtype=np.float64)

    # Load reconstructors.
    jl_reco = jlmod.OfflineReconstructorJetLatentSet2Set(
        input_dim=7,
        **m2mod.BASE_CONFIG["reconstructor_model"],
    ).to(device)
    m2_reco = m2mod.OfflineReconstructor(
        input_dim=7,
        **m2mod.BASE_CONFIG["reconstructor_model"],
    ).to(device)

    jl_ckpt = jetlatent_run / args.jetlatent_reco_ckpt
    m2_ckpt = m2_ref_run / args.m2_reco_ckpt
    if not jl_ckpt.exists():
        raise FileNotFoundError(f"Jet-latent reconstructor checkpoint not found: {jl_ckpt}")
    if not m2_ckpt.exists():
        raise FileNotFoundError(f"Reference m2 reconstructor checkpoint not found: {m2_ckpt}")

    safe_load_state(jl_reco, load_model_state(jl_ckpt, device), "jetlatent_reco")
    safe_load_state(m2_reco, load_model_state(m2_ckpt, device), "m2_reco")
    m2_reco = m2mod.wrap_reconstructor_unmerge_only(m2_reco)

    for p in jl_reco.parameters():
        p.requires_grad_(False)
    for p in m2_reco.parameters():
        p.requires_grad_(False)

    print("\n" + "=" * 70)
    print("STEP 1: BUILD FROZEN TRI-VIEW TENSORS")
    print("=" * 70)
    train_views = build_triview_numpy(
        split_name="train",
        split_idx=train_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        jl_reco=jl_reco,
        m2_reco=m2_reco,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    val_views = build_triview_numpy(
        split_name="val",
        split_idx=val_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        jl_reco=jl_reco,
        m2_reco=m2_reco,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    test_views = build_triview_numpy(
        split_name="test",
        split_idx=test_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        jl_reco=jl_reco,
        m2_reco=m2_reco,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )

    ds_train_f = FrozenTriViewDataset(
        train_views["feat_hlt"], train_views["mask_hlt"],
        train_views["feat_jl"], train_views["mask_jl"],
        train_views["feat_m2"], train_views["mask_m2"],
        labels[train_idx],
    )
    ds_val_f = FrozenTriViewDataset(
        val_views["feat_hlt"], val_views["mask_hlt"],
        val_views["feat_jl"], val_views["mask_jl"],
        val_views["feat_m2"], val_views["mask_m2"],
        labels[val_idx],
    )
    ds_test_f = FrozenTriViewDataset(
        test_views["feat_hlt"], test_views["mask_hlt"],
        test_views["feat_jl"], test_views["mask_jl"],
        test_views["feat_m2"], test_views["mask_m2"],
        labels[test_idx],
    )

    dl_train_f = DataLoader(ds_train_f, batch_size=int(args.frozen_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_f = DataLoader(ds_val_f, batch_size=int(args.frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_f = DataLoader(ds_test_f, batch_size=int(args.frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))

    print("\n" + "=" * 70)
    print("STEP 2: TRAIN TRI-VIEW TAGGER (FROZEN RECONSTRUCTORS)")
    print("=" * 70)
    tagger = TriViewTopTagger().to(device)
    frozen_train_metrics = train_frozen_phase(
        tagger=tagger,
        train_loader=dl_train_f,
        val_loader=dl_val_f,
        device=device,
        epochs=int(args.frozen_epochs),
        patience=int(args.frozen_patience),
        lr=float(args.frozen_lr),
        weight_decay=float(args.frozen_weight_decay),
        warmup_epochs=int(args.frozen_warmup_epochs),
        lambda_rank=float(args.frozen_lambda_rank),
        rank_tau=float(args.frozen_rank_tau),
        target_tpr=float(args.target_tpr),
        select_metric=str(args.select_metric),
    )

    frozen_val = eval_frozen_loader(tagger, dl_val_f, device, float(args.target_tpr))
    frozen_test = eval_frozen_loader(tagger, dl_test_f, device, float(args.target_tpr))
    print(
        f"FrozenTri: val_auc={float(frozen_val['auc']):.4f}, val_fpr50={float(frozen_val['fpr50']):.6f} | "
        f"test_auc={float(frozen_test['auc']):.4f}, test_fpr50={float(frozen_test['fpr50']):.6f}"
    )

    print("\n" + "=" * 70)
    print("STEP 3: JOINT FINETUNE (UNFREEZE JETLATENT + M2 RECONSTRUCTORS)")
    print("=" * 70)

    ds_train_j = JointRecoDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], hlt_const[train_idx], labels[train_idx])
    ds_val_j = JointRecoDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], hlt_const[val_idx], labels[val_idx])
    ds_test_j = JointRecoDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], hlt_const[test_idx], labels[test_idx])

    dl_train_j = DataLoader(ds_train_j, batch_size=int(args.joint_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_j = DataLoader(ds_val_j, batch_size=int(args.joint_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_j = DataLoader(ds_test_j, batch_size=int(args.joint_batch_size), shuffle=False, num_workers=int(args.num_workers))

    joint_train_metrics: Dict[str, float] = {}
    joint_val = {"preds": np.zeros(0, dtype=np.float64), "labels": np.zeros(0, dtype=np.float32), "auc": float("nan"), "fpr50": float("nan")}
    joint_test = {"preds": np.zeros(0, dtype=np.float64), "labels": np.zeros(0, dtype=np.float32), "auc": float("nan"), "fpr50": float("nan")}

    if int(args.joint_epochs) > 0:
        joint_train_metrics = train_joint_phase(
            tagger=tagger,
            jl_reco=jl_reco,
            m2_reco=m2_reco,
            train_loader=dl_train_j,
            val_loader=dl_val_j,
            device=device,
            epochs=int(args.joint_epochs),
            patience=int(args.joint_patience),
            lr_tagger=float(args.joint_lr_tagger),
            lr_reco=float(args.joint_lr_reco),
            weight_decay=float(args.joint_weight_decay),
            warmup_epochs=int(args.joint_warmup_epochs),
            lambda_rank=float(args.joint_lambda_rank),
            rank_tau=float(args.joint_rank_tau),
            corrected_weight_floor=float(args.corrected_weight_floor),
            target_tpr=float(args.target_tpr),
            select_metric=str(args.select_metric),
        )
        joint_val = eval_joint_dynamic(
            tagger=tagger,
            loader=dl_val_j,
            device=device,
            target_tpr=float(args.target_tpr),
            jl_reco=jl_reco,
            m2_reco=m2_reco,
            corrected_weight_floor=float(args.corrected_weight_floor),
        )
        joint_test = eval_joint_dynamic(
            tagger=tagger,
            loader=dl_test_j,
            device=device,
            target_tpr=float(args.target_tpr),
            jl_reco=jl_reco,
            m2_reco=m2_reco,
            corrected_weight_floor=float(args.corrected_weight_floor),
        )
        print(
            f"JointTri : val_auc={float(joint_val['auc']):.4f}, val_fpr50={float(joint_val['fpr50']):.6f} | "
            f"test_auc={float(joint_test['auc']):.4f}, test_fpr50={float(joint_test['fpr50']):.6f}"
        )

    np.savez_compressed(
        save_root / "triview_jetlatent_m2ref_scores.npz",
        labels_val=labels[val_idx].astype(np.float32),
        labels_test=labels[test_idx].astype(np.float32),
        preds_hlt_val=preds_hlt_val.astype(np.float64),
        preds_hlt_test=preds_hlt_test.astype(np.float64),
        preds_triview_frozen_val=np.asarray(frozen_val["preds"], dtype=np.float64),
        preds_triview_frozen_test=np.asarray(frozen_test["preds"], dtype=np.float64),
        preds_triview_joint_val=np.asarray(joint_val["preds"], dtype=np.float64),
        preds_triview_joint_test=np.asarray(joint_test["preds"], dtype=np.float64),
        target_tpr=float(args.target_tpr),
    )

    metrics = {
        "variant": "triview_hlt_plus_jetlatent_plus_m2ref",
        "target_tpr": float(args.target_tpr),
        "jetlatent": {
            "run_dir": str(jetlatent_run),
            "reco_ckpt": str(args.jetlatent_reco_ckpt),
        },
        "m2_ref": {
            "run_dir": str(m2_ref_run),
            "reco_ckpt": str(args.m2_reco_ckpt),
            "baseline_ckpt": str(args.m2_baseline_ckpt),
        },
        "hlt_stats": hlt_stats,
        "frozen_train": frozen_train_metrics,
        "frozen_eval": {
            "val_auc": float(frozen_val["auc"]),
            "val_fpr50": float(frozen_val["fpr50"]),
            "test_auc": float(frozen_test["auc"]),
            "test_fpr50": float(frozen_test["fpr50"]),
        },
        "joint_train": joint_train_metrics,
        "joint_eval": {
            "val_auc": float(joint_val["auc"]),
            "val_fpr50": float(joint_val["fpr50"]),
            "test_auc": float(joint_test["auc"]),
            "test_fpr50": float(joint_test["fpr50"]),
        },
    }
    with open(save_root / "triview_jetlatent_m2ref_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    torch.save({"model": tagger.state_dict(), "metrics": frozen_train_metrics}, save_root / "triview_frozen_tagger.pt")
    if int(args.joint_epochs) > 0:
        torch.save({"model": tagger.state_dict(), "metrics": joint_train_metrics}, save_root / "triview_joint_tagger.pt")
        torch.save({"model": jl_reco.state_dict()}, save_root / "jetlatent_reco_after_joint.pt")
        torch.save({"model": m2_reco.state_dict()}, save_root / "m2ref_reco_after_joint.pt")

    print(f"Saved tri-view outputs to: {save_root}")


if __name__ == "__main__":
    main()

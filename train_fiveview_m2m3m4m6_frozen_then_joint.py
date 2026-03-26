#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Five-view top-tagging pipeline using HLT + 4 reconstructor views (m2/m3/m4/m6):

1) Load the same dataset/splits used by the 6-model fusion runs (from m2 run dir).
2) Load reconstructors:
   - m2 pre-joint checkpoint (offline_reconstructor_stage2.pt)
   - m3 stage-A checkpoint
   - m4 stage-A checkpoint
   - m6 stage-A checkpoint
3) Build five-view training data (HLT + 4 corrected soft views) and train a
   frozen-reconstructor five-view top tagger.
4) Joint-finetune all reconstructors + five-view tagger end-to-end.
5) Save checkpoints, metrics, and val/test score arrays for both phases.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as b
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as m2mod


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
        print(
            f"[{name}] load_state_dict strict=False: missing={len(miss)}, unexpected={len(unexp)}"
        )


FEAT_CLIP_ABS = 50.0


def sanitize_numpy_features(x: np.ndarray, clip_abs: float = FEAT_CLIP_ABS) -> np.ndarray:
    y = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if clip_abs > 0:
        np.clip(y, -float(clip_abs), float(clip_abs), out=y)
    return y


def sanitize_numpy_scores(y_score: np.ndarray) -> np.ndarray:
    return np.nan_to_num(y_score.astype(np.float64), nan=0.5, posinf=1.0, neginf=0.0)


def sanitize_torch_features(x: torch.Tensor, clip_abs: float = FEAT_CLIP_ABS) -> torch.Tensor:
    y = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_abs > 0:
        y = torch.clamp(y, min=-float(clip_abs), max=float(clip_abs))
    return y


def sanitize_torch_logits(z: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0)


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> float:
    labels = labels.astype(np.float32)
    scores = sanitize_numpy_scores(scores)
    finite = np.isfinite(labels) & np.isfinite(scores)
    if not np.any(finite):
        return float("nan")
    labels = labels[finite]
    scores = scores[finite]
    if np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(b.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))


def auc_and_fpr50(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> Tuple[float, float]:
    labels = labels.astype(np.float32)
    scores = sanitize_numpy_scores(scores)
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


class FrozenFiveViewDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        feat_m2: np.ndarray,
        mask_m2: np.ndarray,
        feat_m3: np.ndarray,
        mask_m3: np.ndarray,
        feat_m4: np.ndarray,
        mask_m4: np.ndarray,
        feat_m6: np.ndarray,
        mask_m6: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt = feat_hlt.astype(np.float32)
        self.mask_hlt = mask_hlt.astype(bool)
        self.feat_m2 = feat_m2.astype(np.float32)
        self.mask_m2 = mask_m2.astype(bool)
        self.feat_m3 = feat_m3.astype(np.float32)
        self.mask_m3 = mask_m3.astype(bool)
        self.feat_m4 = feat_m4.astype(np.float32)
        self.mask_m4 = mask_m4.astype(bool)
        self.feat_m6 = feat_m6.astype(np.float32)
        self.mask_m6 = mask_m6.astype(bool)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "feat_m2": self.feat_m2[i],
            "mask_m2": self.mask_m2[i],
            "feat_m3": self.feat_m3[i],
            "mask_m3": self.mask_m3[i],
            "feat_m4": self.feat_m4[i],
            "mask_m4": self.mask_m4[i],
            "feat_m6": self.feat_m6[i],
            "mask_m6": self.mask_m6[i],
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


class FiveViewTopTagger(nn.Module):
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
        self.v_m2 = LiteViewEncoder(10, d_model=d_model, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout)
        self.v_m3 = LiteViewEncoder(10, d_model=d_model, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout)
        self.v_m4 = LiteViewEncoder(10, d_model=d_model, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout)
        self.v_m6 = LiteViewEncoder(10, d_model=d_model, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(5, 24),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(24, 1),
        )

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        feat_m2: torch.Tensor,
        mask_m2: torch.Tensor,
        feat_m3: torch.Tensor,
        mask_m3: torch.Tensor,
        feat_m4: torch.Tensor,
        mask_m4: torch.Tensor,
        feat_m6: torch.Tensor,
        mask_m6: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        l_hlt = self.v_hlt(feat_hlt, mask_hlt)
        l_m2 = self.v_m2(feat_m2, mask_m2)
        l_m3 = self.v_m3(feat_m3, mask_m3)
        l_m4 = self.v_m4(feat_m4, mask_m4)
        l_m6 = self.v_m6(feat_m6, mask_m6)
        per_view = torch.stack([l_hlt, l_m2, l_m3, l_m4, l_m6], dim=1)
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
    tagger: FiveViewTopTagger,
    loader: DataLoader,
    device: torch.device,
    target_tpr: float,
) -> Dict[str, np.ndarray | float]:
    tagger.eval()
    preds = []
    labs = []
    for batch in loader:
        feat_hlt = batch["feat_hlt"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        feat_m2 = batch["feat_m2"].to(device)
        mask_m2 = batch["mask_m2"].to(device)
        feat_m3 = batch["feat_m3"].to(device)
        mask_m3 = batch["mask_m3"].to(device)
        feat_m4 = batch["feat_m4"].to(device)
        mask_m4 = batch["mask_m4"].to(device)
        feat_m6 = batch["feat_m6"].to(device)
        mask_m6 = batch["mask_m6"].to(device)
        y = batch["label"].to(device)

        feat_hlt = sanitize_torch_features(feat_hlt)
        feat_m2 = sanitize_torch_features(feat_m2)
        feat_m3 = sanitize_torch_features(feat_m3)
        feat_m4 = sanitize_torch_features(feat_m4)
        feat_m6 = sanitize_torch_features(feat_m6)

        z, _ = tagger(
            feat_hlt, mask_hlt,
            feat_m2, mask_m2,
            feat_m3, mask_m3,
            feat_m4, mask_m4,
            feat_m6, mask_m6,
        )
        z = sanitize_torch_logits(z)
        p = torch.sigmoid(z)
        preds.append(p.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds).astype(np.float64) if preds else np.zeros(0, dtype=np.float64)
    labs = np.concatenate(labs).astype(np.float32) if labs else np.zeros(0, dtype=np.float32)
    auc, fpr50 = auc_and_fpr50(labs, preds, target_tpr)
    return {
        "preds": preds,
        "labels": labs,
        "auc": float(auc),
        "fpr50": float(fpr50),
    }


@torch.no_grad()
def build_five_views_numpy(
    split_name: str,
    split_idx: np.ndarray,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    m2_reco: nn.Module,
    m3_reco: nn.Module,
    m4_reco: nn.Module,
    m6_reco: nn.Module,
    device: torch.device,
    batch_size: int,
    corrected_weight_floor: float,
) -> Dict[str, np.ndarray]:
    print(f"Building frozen five-view tensors for {split_name}...")
    feat_hlt_s = feat_hlt_std[split_idx]
    mask_hlt_s = hlt_mask[split_idx]
    const_hlt_s = hlt_const[split_idx]

    feat_m2, mask_m2 = m2mod.build_corrected_view_numpy(
        reconstructor=m2_reco,
        feat_hlt=feat_hlt_s,
        mask_hlt=mask_hlt_s,
        const_hlt=const_hlt_s,
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_m3, mask_m3 = b.build_corrected_view_numpy(
        reconstructor=m3_reco,
        feat_hlt=feat_hlt_s,
        mask_hlt=mask_hlt_s,
        const_hlt=const_hlt_s,
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_m4, mask_m4 = b.build_corrected_view_numpy(
        reconstructor=m4_reco,
        feat_hlt=feat_hlt_s,
        mask_hlt=mask_hlt_s,
        const_hlt=const_hlt_s,
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_m6, mask_m6 = b.build_corrected_view_numpy(
        reconstructor=m6_reco,
        feat_hlt=feat_hlt_s,
        mask_hlt=mask_hlt_s,
        const_hlt=const_hlt_s,
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=False,
    )

    feat_hlt_s = sanitize_numpy_features(feat_hlt_s)
    feat_m2 = sanitize_numpy_features(feat_m2)
    feat_m3 = sanitize_numpy_features(feat_m3)
    feat_m4 = sanitize_numpy_features(feat_m4)
    feat_m6 = sanitize_numpy_features(feat_m6)

    return {
        "feat_hlt": feat_hlt_s.astype(np.float32),
        "mask_hlt": mask_hlt_s.astype(bool),
        "feat_m2": feat_m2.astype(np.float32),
        "mask_m2": mask_m2.astype(bool),
        "feat_m3": feat_m3.astype(np.float32),
        "mask_m3": mask_m3.astype(bool),
        "feat_m4": feat_m4.astype(np.float32),
        "mask_m4": mask_m4.astype(bool),
        "feat_m6": feat_m6.astype(np.float32),
        "mask_m6": mask_m6.astype(bool),
    }


def train_frozen_phase(
    tagger: FiveViewTopTagger,
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
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for ep in range(int(epochs)):
        tagger.train()
        tr_loss = tr_cls = tr_rank = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            feat_m2 = batch["feat_m2"].to(device)
            mask_m2 = batch["mask_m2"].to(device)
            feat_m3 = batch["feat_m3"].to(device)
            mask_m3 = batch["mask_m3"].to(device)
            feat_m4 = batch["feat_m4"].to(device)
            mask_m4 = batch["mask_m4"].to(device)
            feat_m6 = batch["feat_m6"].to(device)
            mask_m6 = batch["mask_m6"].to(device)
            y = batch["label"].to(device)

            feat_hlt = sanitize_torch_features(feat_hlt)
            feat_m2 = sanitize_torch_features(feat_m2)
            feat_m3 = sanitize_torch_features(feat_m3)
            feat_m4 = sanitize_torch_features(feat_m4)
            feat_m6 = sanitize_torch_features(feat_m6)

            opt.zero_grad()
            logits, _ = tagger(
                feat_hlt, mask_hlt,
                feat_m2, mask_m2,
                feat_m3, mask_m3,
                feat_m4, mask_m4,
                feat_m6, mask_m6,
            )
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
                f"Frozen5V ep {ep+1}: train_loss={tr_loss:.5f} (cls={tr_cls:.5f}, rank={tr_rank:.5f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping frozen five-view at epoch {ep+1}")
            break

    if best_state is not None:
        tagger.load_state_dict(best_state)
    return best_metrics


def build_views_torch_from_batch(
    feat_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_hlt: torch.Tensor,
    m2_reco: nn.Module,
    m3_reco: nn.Module,
    m4_reco: nn.Module,
    m6_reco: nn.Module,
    corrected_weight_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out2 = m2_reco(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
    feat_m2, mask_m2 = m2mod.build_soft_corrected_view(
        out2,
        weight_floor=float(corrected_weight_floor),
        scale_features_by_weight=True,
        include_flags=False,
    )

    out3 = m3_reco(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
    feat_m3, mask_m3 = b.build_soft_corrected_view(
        out3,
        weight_floor=float(corrected_weight_floor),
        scale_features_by_weight=True,
        include_flags=False,
    )

    out4 = m4_reco(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
    feat_m4, mask_m4 = b.build_soft_corrected_view(
        out4,
        weight_floor=float(corrected_weight_floor),
        scale_features_by_weight=True,
        include_flags=False,
    )

    out6 = m6_reco(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
    feat_m6, mask_m6 = b.build_soft_corrected_view(
        out6,
        weight_floor=float(corrected_weight_floor),
        scale_features_by_weight=True,
        include_flags=False,
    )

    feat_m2 = sanitize_torch_features(feat_m2)
    feat_m3 = sanitize_torch_features(feat_m3)
    feat_m4 = sanitize_torch_features(feat_m4)
    feat_m6 = sanitize_torch_features(feat_m6)

    return feat_m2, mask_m2, feat_m3, mask_m3, feat_m4, mask_m4, feat_m6, mask_m6


@torch.no_grad()
def eval_joint_dynamic(
    tagger: FiveViewTopTagger,
    loader: DataLoader,
    device: torch.device,
    target_tpr: float,
    m2_reco: nn.Module,
    m3_reco: nn.Module,
    m4_reco: nn.Module,
    m6_reco: nn.Module,
    corrected_weight_floor: float,
) -> Dict[str, np.ndarray | float]:
    tagger.eval()
    m2_reco.eval()
    m3_reco.eval()
    m4_reco.eval()
    m6_reco.eval()

    preds = []
    labs = []
    for batch in loader:
        feat_hlt = batch["feat_hlt"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        feat_m2, mask_m2, feat_m3, mask_m3, feat_m4, mask_m4, feat_m6, mask_m6 = build_views_torch_from_batch(
            feat_hlt=feat_hlt,
            mask_hlt=mask_hlt,
            const_hlt=const_hlt,
            m2_reco=m2_reco,
            m3_reco=m3_reco,
            m4_reco=m4_reco,
            m6_reco=m6_reco,
            corrected_weight_floor=float(corrected_weight_floor),
        )

        z, _ = tagger(
            feat_hlt, mask_hlt,
            feat_m2, mask_m2,
            feat_m3, mask_m3,
            feat_m4, mask_m4,
            feat_m6, mask_m6,
        )
        z = sanitize_torch_logits(z)
        p = torch.sigmoid(z)
        preds.append(p.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds).astype(np.float64) if preds else np.zeros(0, dtype=np.float64)
    labs = np.concatenate(labs).astype(np.float32) if labs else np.zeros(0, dtype=np.float32)
    auc, fpr50 = auc_and_fpr50(labs, preds, target_tpr)
    return {
        "preds": preds,
        "labels": labs,
        "auc": float(auc),
        "fpr50": float(fpr50),
    }


def train_joint_phase(
    tagger: FiveViewTopTagger,
    m2_reco: nn.Module,
    m3_reco: nn.Module,
    m4_reco: nn.Module,
    m6_reco: nn.Module,
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
    for p in m2_reco.parameters():
        p.requires_grad_(True)
    for p in m3_reco.parameters():
        p.requires_grad_(True)
    for p in m4_reco.parameters():
        p.requires_grad_(True)
    for p in m6_reco.parameters():
        p.requires_grad_(True)

    params = [
        {"params": tagger.parameters(), "lr": float(lr_tagger)},
        {"params": m2_reco.parameters(), "lr": float(lr_reco)},
        {"params": m3_reco.parameters(), "lr": float(lr_reco)},
        {"params": m4_reco.parameters(), "lr": float(lr_reco)},
        {"params": m6_reco.parameters(), "lr": float(lr_reco)},
    ]
    opt = torch.optim.AdamW(params, lr=float(lr_tagger), weight_decay=float(weight_decay))
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    no_improve = 0
    best_metrics: Dict[str, float] = {}

    for ep in range(int(epochs)):
        tagger.train()
        m2_reco.train()
        m3_reco.train()
        m4_reco.train()
        m6_reco.train()

        tr_loss = tr_cls = tr_rank = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            y = batch["label"].to(device)

            feat_hlt = sanitize_torch_features(feat_hlt)
            const_hlt = torch.nan_to_num(const_hlt, nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad()

            feat_m2, mask_m2, feat_m3, mask_m3, feat_m4, mask_m4, feat_m6, mask_m6 = build_views_torch_from_batch(
                feat_hlt=feat_hlt,
                mask_hlt=mask_hlt,
                const_hlt=const_hlt,
                m2_reco=m2_reco,
                m3_reco=m3_reco,
                m4_reco=m4_reco,
                m6_reco=m6_reco,
                corrected_weight_floor=float(corrected_weight_floor),
            )

            logits, _ = tagger(
                feat_hlt, mask_hlt,
                feat_m2, mask_m2,
                feat_m3, mask_m3,
                feat_m4, mask_m4,
                feat_m6, mask_m6,
            )

            logits = sanitize_torch_logits(logits)
            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = low_fpr_surrogate_loss(logits, y, target_tpr=float(target_tpr), tau=float(rank_tau))
            loss = loss_cls + float(lambda_rank) * loss_rank
            if not torch.isfinite(loss):
                continue
            loss.backward()

            torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(m2_reco.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(m3_reco.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(m4_reco.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(m6_reco.parameters(), 1.0)
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
            m2_reco=m2_reco,
            m3_reco=m3_reco,
            m4_reco=m4_reco,
            m6_reco=m6_reco,
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
                "m2": {k: v.detach().cpu().clone() for k, v in m2_reco.state_dict().items()},
                "m3": {k: v.detach().cpu().clone() for k, v in m3_reco.state_dict().items()},
                "m4": {k: v.detach().cpu().clone() for k, v in m4_reco.state_dict().items()},
                "m6": {k: v.detach().cpu().clone() for k, v in m6_reco.state_dict().items()},
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
                f"Joint5V ep {ep+1}: train_loss={tr_loss:.5f} (cls={tr_cls:.5f}, rank={tr_rank:.5f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping joint five-view at epoch {ep+1}")
            break

    if best_state is not None:
        tagger.load_state_dict(best_state["tagger"])
        m2_reco.load_state_dict(best_state["m2"])
        m3_reco.load_state_dict(best_state["m3"])
        m4_reco.load_state_dict(best_state["m4"])
        m6_reco.load_state_dict(best_state["m6"])

    return best_metrics


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--m2_run_dir", type=str, required=True)
    ap.add_argument("--m3_run_dir", type=str, required=True)
    ap.add_argument("--m4_run_dir", type=str, required=True)
    ap.add_argument("--m6_run_dir", type=str, required=True)

    ap.add_argument("--m2_reco_ckpt", type=str, default="offline_reconstructor_stage2.pt")
    ap.add_argument("--m3_reco_ckpt", type=str, default="offline_reconstructor_stageA.pt")
    ap.add_argument("--m4_reco_ckpt", type=str, default="offline_reconstructor_stageA.pt")
    ap.add_argument("--m6_reco_ckpt", type=str, default="offline_reconstructor_stageA.pt")
    ap.add_argument("--m2_baseline_ckpt", type=str, default="baseline.pt")

    ap.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model14_fiveview_m2m3m4m6")
    ap.add_argument("--run_name", type=str, default="model14_fiveview_m2m3m4m6_150k75k150k_seed0")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--corrected_weight_floor", type=float, default=0.03)
    ap.add_argument("--reco_eval_batch_size", type=int, default=256)
    ap.add_argument("--target_tpr", type=float, default=0.50)

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

    ap.add_argument("--select_metric", type=str, choices=["fpr50", "auc"], default="fpr50")

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

    m2_run = Path(args.m2_run_dir)
    m3_run = Path(args.m3_run_dir)
    m4_run = Path(args.m4_run_dir)
    m6_run = Path(args.m6_run_dir)

    setup_path = m2_run / "data_setup.json"
    split_path = m2_run / "data_splits.npz"
    if not setup_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Missing m2 setup/splits in {m2_run}")

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

    cfg = b._deepcopy_config()
    cfg["hlt_effects"].update(hlt_cfg)

    max_jets_needed = int(offset_jets + n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = b.load_raw_constituents_from_h5(
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
    hlt_const, hlt_mask, hlt_stats, _ = b.apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(data_setup.get("seed", args.seed)),
    )

    print("Computing standardized HLT/offline features...")
    feat_hlt = b.compute_features(hlt_const, hlt_mask)
    feat_hlt_std = b.standardize(feat_hlt, hlt_mask, means, stds)

    # Load m2 baseline for HLT reference scores.
    baseline_ref = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline_path = m2_run / args.m2_baseline_ckpt
    if baseline_path.exists():
        safe_load_state(baseline_ref, load_model_state(baseline_path, device), "m2_baseline")
        preds_hlt_val = 1.0 / (1.0 + np.exp(-predict_single_view_logits(baseline_ref, feat_hlt_std, hlt_mask, val_idx, device, int(args.reco_eval_batch_size))))
        preds_hlt_test = 1.0 / (1.0 + np.exp(-predict_single_view_logits(baseline_ref, feat_hlt_std, hlt_mask, test_idx, device, int(args.reco_eval_batch_size))))
    else:
        print(f"Warning: baseline checkpoint not found: {baseline_path}")
        preds_hlt_val = np.zeros(val_idx.shape[0], dtype=np.float64)
        preds_hlt_test = np.zeros(test_idx.shape[0], dtype=np.float64)

    # Load reconstructors.
    m2_reco = m2mod.OfflineReconstructor(input_dim=7, **m2mod.BASE_CONFIG["reconstructor_model"]).to(device)
    m3_reco = b.OfflineReconstructor(input_dim=7, **b.BASE_CONFIG["reconstructor_model"]).to(device)
    m4_reco = b.OfflineReconstructor(input_dim=7, **b.BASE_CONFIG["reconstructor_model"]).to(device)
    m6_reco = b.OfflineReconstructor(input_dim=7, **b.BASE_CONFIG["reconstructor_model"]).to(device)

    safe_load_state(m2_reco, load_model_state(m2_run / args.m2_reco_ckpt, device), "m2_reco")
    safe_load_state(m3_reco, load_model_state(m3_run / args.m3_reco_ckpt, device), "m3_reco")
    safe_load_state(m4_reco, load_model_state(m4_run / args.m4_reco_ckpt, device), "m4_reco")
    safe_load_state(m6_reco, load_model_state(m6_run / args.m6_reco_ckpt, device), "m6_reco")

    for p in m2_reco.parameters():
        p.requires_grad_(False)
    for p in m3_reco.parameters():
        p.requires_grad_(False)
    for p in m4_reco.parameters():
        p.requires_grad_(False)
    for p in m6_reco.parameters():
        p.requires_grad_(False)

    print("\n" + "=" * 70)
    print("STEP 1: BUILD FROZEN FIVE-VIEW TENSORS")
    print("=" * 70)
    train_views = build_five_views_numpy(
        split_name="train",
        split_idx=train_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        m2_reco=m2_reco,
        m3_reco=m3_reco,
        m4_reco=m4_reco,
        m6_reco=m6_reco,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    val_views = build_five_views_numpy(
        split_name="val",
        split_idx=val_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        m2_reco=m2_reco,
        m3_reco=m3_reco,
        m4_reco=m4_reco,
        m6_reco=m6_reco,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    test_views = build_five_views_numpy(
        split_name="test",
        split_idx=test_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        m2_reco=m2_reco,
        m3_reco=m3_reco,
        m4_reco=m4_reco,
        m6_reco=m6_reco,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )

    ds_train_f = FrozenFiveViewDataset(
        train_views["feat_hlt"], train_views["mask_hlt"],
        train_views["feat_m2"], train_views["mask_m2"],
        train_views["feat_m3"], train_views["mask_m3"],
        train_views["feat_m4"], train_views["mask_m4"],
        train_views["feat_m6"], train_views["mask_m6"],
        labels[train_idx],
    )
    ds_val_f = FrozenFiveViewDataset(
        val_views["feat_hlt"], val_views["mask_hlt"],
        val_views["feat_m2"], val_views["mask_m2"],
        val_views["feat_m3"], val_views["mask_m3"],
        val_views["feat_m4"], val_views["mask_m4"],
        val_views["feat_m6"], val_views["mask_m6"],
        labels[val_idx],
    )
    ds_test_f = FrozenFiveViewDataset(
        test_views["feat_hlt"], test_views["mask_hlt"],
        test_views["feat_m2"], test_views["mask_m2"],
        test_views["feat_m3"], test_views["mask_m3"],
        test_views["feat_m4"], test_views["mask_m4"],
        test_views["feat_m6"], test_views["mask_m6"],
        labels[test_idx],
    )

    dl_train_f = DataLoader(ds_train_f, batch_size=int(args.frozen_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_f = DataLoader(ds_val_f, batch_size=int(args.frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_f = DataLoader(ds_test_f, batch_size=int(args.frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))

    print("\n" + "=" * 70)
    print("STEP 2: TRAIN FIVE-VIEW TAGGER (FROZEN RECONSTRUCTORS)")
    print("=" * 70)
    tagger = FiveViewTopTagger().to(device)
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
        f"Frozen5V: val_auc={float(frozen_val['auc']):.4f}, val_fpr50={float(frozen_val['fpr50']):.6f} | "
        f"test_auc={float(frozen_test['auc']):.4f}, test_fpr50={float(frozen_test['fpr50']):.6f}"
    )

    print("\n" + "=" * 70)
    print("STEP 3: JOINT FINETUNE (UNFREEZE M2/M3/M4/M6 RECONSTRUCTORS)")
    print("=" * 70)

    ds_train_j = JointRecoDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], hlt_const[train_idx], labels[train_idx])
    ds_val_j = JointRecoDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], hlt_const[val_idx], labels[val_idx])
    ds_test_j = JointRecoDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], hlt_const[test_idx], labels[test_idx])

    dl_train_j = DataLoader(ds_train_j, batch_size=int(args.joint_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_j = DataLoader(ds_val_j, batch_size=int(args.joint_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_j = DataLoader(ds_test_j, batch_size=int(args.joint_batch_size), shuffle=False, num_workers=int(args.num_workers))

    joint_train_metrics = {}
    joint_val = {"preds": np.zeros(0, dtype=np.float64), "labels": np.zeros(0, dtype=np.float32), "auc": float("nan"), "fpr50": float("nan")}
    joint_test = {"preds": np.zeros(0, dtype=np.float64), "labels": np.zeros(0, dtype=np.float32), "auc": float("nan"), "fpr50": float("nan")}

    if int(args.joint_epochs) > 0:
        joint_train_metrics = train_joint_phase(
            tagger=tagger,
            m2_reco=m2_reco,
            m3_reco=m3_reco,
            m4_reco=m4_reco,
            m6_reco=m6_reco,
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
            m2_reco=m2_reco,
            m3_reco=m3_reco,
            m4_reco=m4_reco,
            m6_reco=m6_reco,
            corrected_weight_floor=float(args.corrected_weight_floor),
        )
        joint_test = eval_joint_dynamic(
            tagger=tagger,
            loader=dl_test_j,
            device=device,
            target_tpr=float(args.target_tpr),
            m2_reco=m2_reco,
            m3_reco=m3_reco,
            m4_reco=m4_reco,
            m6_reco=m6_reco,
            corrected_weight_floor=float(args.corrected_weight_floor),
        )

        print(
            f"Joint5V : val_auc={float(joint_val['auc']):.4f}, val_fpr50={float(joint_val['fpr50']):.6f} | "
            f"test_auc={float(joint_test['auc']):.4f}, test_fpr50={float(joint_test['fpr50']):.6f}"
        )

    np.savez_compressed(
        save_root / "fiveview_m2m3m4m6_scores.npz",
        labels_val=labels[val_idx].astype(np.float32),
        labels_test=labels[test_idx].astype(np.float32),
        preds_hlt_val=preds_hlt_val.astype(np.float64),
        preds_hlt_test=preds_hlt_test.astype(np.float64),
        preds_fiveview_frozen_val=np.asarray(frozen_val["preds"], dtype=np.float64),
        preds_fiveview_frozen_test=np.asarray(frozen_test["preds"], dtype=np.float64),
        preds_fiveview_joint_val=np.asarray(joint_val["preds"], dtype=np.float64),
        preds_fiveview_joint_test=np.asarray(joint_test["preds"], dtype=np.float64),
        target_tpr=float(args.target_tpr),
    )

    metrics = {
        "variant": "fiveview_hlt_plus_m2m3m4m6",
        "target_tpr": float(args.target_tpr),
        "m2": {
            "run_dir": str(m2_run),
            "reco_ckpt": str(args.m2_reco_ckpt),
            "note": "pre-joint checkpoint requested",
        },
        "m3": {"run_dir": str(m3_run), "reco_ckpt": str(args.m3_reco_ckpt)},
        "m4": {"run_dir": str(m4_run), "reco_ckpt": str(args.m4_reco_ckpt)},
        "m6": {"run_dir": str(m6_run), "reco_ckpt": str(args.m6_reco_ckpt)},
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
    with open(save_root / "fiveview_m2m3m4m6_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    torch.save({"model": tagger.state_dict(), "metrics": frozen_train_metrics}, save_root / "fiveview_frozen_tagger.pt")
    if int(args.joint_epochs) > 0:
        torch.save({"model": tagger.state_dict(), "metrics": joint_train_metrics}, save_root / "fiveview_joint_tagger.pt")
        torch.save({"model": m2_reco.state_dict()}, save_root / "m2_reco_after_joint.pt")
        torch.save({"model": m3_reco.state_dict()}, save_root / "m3_reco_after_joint.pt")
        torch.save({"model": m4_reco.state_dict()}, save_root / "m4_reco_after_joint.pt")
        torch.save({"model": m6_reco.state_dict()}, save_root / "m6_reco_after_joint.pt")

    print(f"Saved five-view outputs to: {save_root}")


if __name__ == "__main__":
    main()

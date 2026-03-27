#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model-20: Dual-reconstructor dualview residual correction on top of HLT.

Setup:
1) Load fixed train/val/test split + preprocessing config from m2 run dir.
2) Load pretrained reconstructors:
   - Reco-A: m6 stage-A checkpoint
   - Reco-B: m2 pre-joint checkpoint
3) Build corrected views A/B from HLT input and train a gated residual model:
      final_logit = hlt_logit + gate * delta
   with label-first objective (BCE), weak teacher KD, and gate sparsity.
4) Optional light joint finetune: unfreeze both reconstructors + residual model
   with tiny reco LRs and parameter anchors.
5) Save pre-joint and post-joint val/test predictions + checkpoints.
"""

from __future__ import annotations

import argparse
import copy
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

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as b
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as m2mod


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
    if clip_abs > 0:
        np.clip(y, -float(clip_abs), float(clip_abs), out=y)
    return y


def sanitize_numpy_scores(s: np.ndarray) -> np.ndarray:
    return np.nan_to_num(s.astype(np.float64), nan=0.5, posinf=1.0, neginf=0.0)


def sanitize_torch_features(x: torch.Tensor, clip_abs: float = FEAT_CLIP_ABS) -> torch.Tensor:
    y = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_abs > 0:
        y = torch.clamp(y, min=-float(clip_abs), max=float(clip_abs))
    return y


def sanitize_torch_logits(z: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0)


def auc_and_fpr(labels: np.ndarray, probs: np.ndarray, target_tpr: float) -> Tuple[float, float]:
    labels = labels.astype(np.float32)
    probs = sanitize_numpy_scores(probs)
    finite = np.isfinite(labels) & np.isfinite(probs)
    if not np.any(finite):
        return float("nan"), float("nan")
    labels = labels[finite]
    probs = probs[finite]
    auc = float(roc_auc_score(labels, probs)) if np.unique(labels).size > 1 else float("nan")
    if np.unique(labels).size < 2:
        return auc, float("nan")
    fpr, tpr, _ = roc_curve(labels, probs)
    return auc, float(b.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))


def threshold_at_target_tpr(labels: np.ndarray, probs: np.ndarray, target_tpr: float) -> float:
    fpr, tpr, thr = roc_curve(labels.astype(np.float32), probs.astype(np.float64))
    if len(thr) == 0:
        return 0.5
    valid = np.isfinite(thr)
    if not np.any(valid):
        return float(np.median(probs))
    idx_valid = np.where(valid)[0]
    idx = int(idx_valid[np.argmin(np.abs(tpr[idx_valid] - float(target_tpr)))])
    return float(thr[idx])


def fpr_from_val_threshold(labels_test: np.ndarray, probs_test: np.ndarray, thr_val: float) -> float:
    y = labels_test.astype(np.float32) > 0.5
    neg = ~y
    pred = probs_test >= float(thr_val)
    fp = int((pred & neg).sum())
    nneg = int(neg.sum())
    return float(fp / max(nneg, 1))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def predict_single_view_logits(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    split_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    idx = split_idx.astype(np.int64)
    out = np.zeros(idx.shape[0], dtype=np.float64)
    ptr = 0
    with torch.no_grad():
        for start in range(0, len(idx), int(batch_size)):
            end = min(start + int(batch_size), len(idx))
            sl = idx[start:end]
            x = torch.tensor(feat[sl], dtype=torch.float32, device=device)
            m = torch.tensor(mask[sl], dtype=torch.bool, device=device)
            z = model(x, m).squeeze(1)
            k = end - start
            out[ptr: ptr + k] = z.detach().cpu().numpy().astype(np.float64)
            ptr += k
    return out


def build_aux_features_np(mask_hlt: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    c_h = mask_hlt.astype(np.float32).sum(axis=1)
    c_a = mask_a.astype(np.float32).sum(axis=1)
    c_b = mask_b.astype(np.float32).sum(axis=1)
    denom_h = np.maximum(c_h, 1.0)
    denom_a = np.maximum(c_a, 1.0)
    denom_b = np.maximum(c_b, 1.0)
    feats = np.stack(
        [
            c_h,
            c_a,
            c_b,
            c_a - c_h,
            c_b - c_h,
            c_a - c_b,
            c_a / denom_h,
            c_b / denom_h,
            (c_a - c_h) / denom_a,
            (c_b - c_h) / denom_b,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def standardize_aux(train_aux: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train_aux.mean(axis=0).astype(np.float32)
    sd = train_aux.std(axis=0).astype(np.float32)
    sd = np.clip(sd, 1e-6, None)
    return ((x - mu) / sd).astype(np.float32), mu, sd


class FrozenDualResidualDataset(Dataset):
    def __init__(
        self,
        feat_a: np.ndarray,
        mask_a: np.ndarray,
        feat_b: np.ndarray,
        mask_b: np.ndarray,
        aux: np.ndarray,
        hlt_logit: np.ndarray,
        teacher_logit: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_a = feat_a.astype(np.float32)
        self.mask_a = mask_a.astype(bool)
        self.feat_b = feat_b.astype(np.float32)
        self.mask_b = mask_b.astype(bool)
        self.aux = aux.astype(np.float32)
        self.hlt_logit = hlt_logit.astype(np.float32)
        self.teacher_logit = teacher_logit.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return {
            "feat_a": self.feat_a[i],
            "mask_a": self.mask_a[i],
            "feat_b": self.feat_b[i],
            "mask_b": self.mask_b[i],
            "aux": self.aux[i],
            "hlt_logit": self.hlt_logit[i],
            "teacher_logit": self.teacher_logit[i],
            "label": self.labels[i],
        }


class JointDualResidualDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        hlt_logit: np.ndarray,
        teacher_logit: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt = feat_hlt.astype(np.float32)
        self.mask_hlt = mask_hlt.astype(bool)
        self.const_hlt = const_hlt.astype(np.float32)
        self.hlt_logit = hlt_logit.astype(np.float32)
        self.teacher_logit = teacher_logit.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "hlt_logit": self.hlt_logit[i],
            "teacher_logit": self.teacher_logit[i],
            "label": self.labels[i],
        }


class GatedDualResidual(nn.Module):
    def __init__(self, aux_dim: int, dual_cfg: Dict[str, int | float], hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.dual = b.DualViewCrossAttnClassifier(input_dim_a=10, input_dim_b=10, **dual_cfg)
        in_dim = int(aux_dim) + 2  # dual_logit + hlt_logit + aux
        self.delta_head = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden // 2), 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden // 2), 1),
        )

    def forward(
        self,
        feat_a: torch.Tensor,
        mask_a: torch.Tensor,
        feat_b: torch.Tensor,
        mask_b: torch.Tensor,
        hlt_logit: torch.Tensor,
        aux: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feat_a = sanitize_torch_features(feat_a)
        feat_b = sanitize_torch_features(feat_b)
        z_dual = sanitize_torch_logits(self.dual(feat_a, mask_a, feat_b, mask_b).squeeze(1))
        x = torch.cat([z_dual.unsqueeze(1), hlt_logit.unsqueeze(1), aux], dim=1)
        delta = sanitize_torch_logits(self.delta_head(x).squeeze(1))
        gate = torch.sigmoid(self.gate_head(x).squeeze(1))
        final_logit = sanitize_torch_logits(hlt_logit + gate * delta)
        return {
            "dual_logit": z_dual,
            "delta": delta,
            "gate": gate,
            "final_logit": final_logit,
        }


def select_is_better(select_metric: str, curr_auc: float, curr_fpr50: float, best: float) -> bool:
    if str(select_metric).lower() == "fpr50":
        return np.isfinite(curr_fpr50) and curr_fpr50 < best
    return np.isfinite(curr_auc) and curr_auc > best


@torch.no_grad()
def eval_frozen_loader(
    model: GatedDualResidual,
    loader: DataLoader,
    device: torch.device,
    target_tpr: float,
) -> Dict[str, np.ndarray | float]:
    model.eval()
    all_probs = []
    all_labels = []
    all_gates = []
    all_delta = []
    for batch in loader:
        feat_a = batch["feat_a"].to(device=device, dtype=torch.float32)
        mask_a = batch["mask_a"].to(device=device, dtype=torch.bool)
        feat_b = batch["feat_b"].to(device=device, dtype=torch.float32)
        mask_b = batch["mask_b"].to(device=device, dtype=torch.bool)
        aux = batch["aux"].to(device=device, dtype=torch.float32)
        hlt_logit = batch["hlt_logit"].to(device=device, dtype=torch.float32)
        y = batch["label"].to(device=device, dtype=torch.float32)

        out = model(feat_a, mask_a, feat_b, mask_b, hlt_logit, aux)
        probs = torch.sigmoid(out["final_logit"])
        all_probs.append(probs.detach().cpu().numpy().astype(np.float64))
        all_labels.append(y.detach().cpu().numpy().astype(np.float32))
        all_gates.append(out["gate"].detach().cpu().numpy().astype(np.float64))
        all_delta.append(out["delta"].detach().cpu().numpy().astype(np.float64))

    probs = np.concatenate(all_probs) if all_probs else np.zeros((0,), dtype=np.float64)
    labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.float32)
    gates = np.concatenate(all_gates) if all_gates else np.zeros((0,), dtype=np.float64)
    delta = np.concatenate(all_delta) if all_delta else np.zeros((0,), dtype=np.float64)
    auc, fpr50 = auc_and_fpr(labels, probs, float(target_tpr))
    return {"probs": probs, "labels": labels, "gates": gates, "delta": delta, "auc": auc, "fpr50": fpr50}


def train_frozen(
    model: GatedDualResidual,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    kd_temp: float,
    target_tpr: float,
    select_metric: str,
    lambda_cls: float,
    lambda_kd: float,
    lambda_residual: float,
    lambda_gate: float,
) -> Tuple[GatedDualResidual, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0
    t = float(max(kd_temp, 1e-3))

    for ep in range(int(epochs)):
        model.train()
        run_tot = run_cls = run_kd = run_res = run_gate = 0.0
        n_seen = 0

        for batch in train_loader:
            feat_a = batch["feat_a"].to(device=device, dtype=torch.float32)
            mask_a = batch["mask_a"].to(device=device, dtype=torch.bool)
            feat_b = batch["feat_b"].to(device=device, dtype=torch.float32)
            mask_b = batch["mask_b"].to(device=device, dtype=torch.bool)
            aux = batch["aux"].to(device=device, dtype=torch.float32)
            hlt_logit = batch["hlt_logit"].to(device=device, dtype=torch.float32)
            teacher_logit = batch["teacher_logit"].to(device=device, dtype=torch.float32)
            y = batch["label"].to(device=device, dtype=torch.float32)

            opt.zero_grad()
            out = model(feat_a, mask_a, feat_b, mask_b, hlt_logit, aux)
            final_logit = out["final_logit"]
            gate = out["gate"]
            delta = out["delta"]

            l_cls = F.binary_cross_entropy_with_logits(final_logit, y)
            soft_t = torch.sigmoid(teacher_logit / t)
            l_kd = F.binary_cross_entropy_with_logits(final_logit / t, soft_t) * (t * t)
            l_res = F.smooth_l1_loss(gate * delta, teacher_logit - hlt_logit)
            l_gate = gate.mean()

            loss = (
                float(lambda_cls) * l_cls
                + float(lambda_kd) * l_kd
                + float(lambda_residual) * l_res
                + float(lambda_gate) * l_gate
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            run_tot += float(loss.item()) * bs
            run_cls += float(l_cls.item()) * bs
            run_kd += float(l_kd.item()) * bs
            run_res += float(l_res.item()) * bs
            run_gate += float(l_gate.item()) * bs
            n_seen += bs

        sch.step()

        val_pack = eval_frozen_loader(model, val_loader, device, float(target_tpr))
        auc_v = float(val_pack["auc"])
        fpr50_v = float(val_pack["fpr50"])

        if select_is_better(str(select_metric), auc_v, fpr50_v, best_sel):
            best_sel = float(fpr50_v) if str(select_metric).lower() == "fpr50" else float(auc_v)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "best_epoch": int(ep + 1),
                "best_select_metric": str(select_metric).lower(),
                "best_sel": float(best_sel),
                "best_val_auc": float(auc_v),
                "best_val_fpr50": float(fpr50_v),
                "best_train_total": float(run_tot / max(n_seen, 1)),
                "best_train_cls": float(run_cls / max(n_seen, 1)),
                "best_train_kd": float(run_kd / max(n_seen, 1)),
                "best_train_res": float(run_res / max(n_seen, 1)),
                "best_train_gate": float(run_gate / max(n_seen, 1)),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Frozen ep {ep+1}: train_total={run_tot/max(n_seen,1):.5f} "
                f"(cls={run_cls/max(n_seen,1):.5f}, kd={run_kd/max(n_seen,1):.5f}, "
                f"res={run_res/max(n_seen,1):.5f}, gate={run_gate/max(n_seen,1):.5f}) | "
                f"val_auc={auc_v:.4f}, val_fpr50={fpr50_v:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping Frozen at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def compute_aux_from_masks_torch(mask_hlt: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    c_h = mask_hlt.float().sum(dim=1)
    c_a = mask_a.float().sum(dim=1)
    c_b = mask_b.float().sum(dim=1)
    denom_h = torch.clamp(c_h, min=1.0)
    denom_a = torch.clamp(c_a, min=1.0)
    denom_b = torch.clamp(c_b, min=1.0)
    return torch.stack(
        [
            c_h,
            c_a,
            c_b,
            c_a - c_h,
            c_b - c_h,
            c_a - c_b,
            c_a / denom_h,
            c_b / denom_h,
            (c_a - c_h) / denom_a,
            (c_b - c_h) / denom_b,
        ],
        dim=1,
    )


def l2_anchor_to_state(model: nn.Module, ref_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    vals = []
    for n, p in model.named_parameters():
        if n in ref_state:
            ref = ref_state[n].to(device=p.device, dtype=p.dtype)
            vals.append((p - ref).pow(2).mean())
    if len(vals) == 0:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(vals).mean()


@torch.no_grad()
def eval_joint_dynamic(
    model: GatedDualResidual,
    reco_a: nn.Module,
    reco_b: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    aux_mean_t: torch.Tensor,
    aux_std_t: torch.Tensor,
    target_tpr: float,
) -> Dict[str, np.ndarray | float]:
    model.eval()
    reco_a.eval()
    reco_b.eval()

    all_probs = []
    all_labels = []
    all_gates = []
    all_delta = []

    for batch in loader:
        feat_hlt = batch["feat_hlt"].to(device=device, dtype=torch.float32)
        mask_hlt = batch["mask_hlt"].to(device=device, dtype=torch.bool)
        const_hlt = batch["const_hlt"].to(device=device, dtype=torch.float32)
        hlt_logit = batch["hlt_logit"].to(device=device, dtype=torch.float32)
        y = batch["label"].to(device=device, dtype=torch.float32)

        out_a = reco_a(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
        feat_a, mask_a = b.build_soft_corrected_view(
            out_a,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=False,
        )
        out_b = reco_b(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = m2mod.build_soft_corrected_view(
            out_b,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=False,
        )

        aux = compute_aux_from_masks_torch(mask_hlt, mask_a, mask_b)
        aux = (aux - aux_mean_t) / aux_std_t

        out = model(feat_a, mask_a, feat_b, mask_b, hlt_logit, aux)
        probs = torch.sigmoid(out["final_logit"])

        all_probs.append(probs.detach().cpu().numpy().astype(np.float64))
        all_labels.append(y.detach().cpu().numpy().astype(np.float32))
        all_gates.append(out["gate"].detach().cpu().numpy().astype(np.float64))
        all_delta.append(out["delta"].detach().cpu().numpy().astype(np.float64))

    probs = np.concatenate(all_probs) if all_probs else np.zeros((0,), dtype=np.float64)
    labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.float32)
    gates = np.concatenate(all_gates) if all_gates else np.zeros((0,), dtype=np.float64)
    delta = np.concatenate(all_delta) if all_delta else np.zeros((0,), dtype=np.float64)
    auc, fpr50 = auc_and_fpr(labels, probs, float(target_tpr))
    return {"probs": probs, "labels": labels, "gates": gates, "delta": delta, "auc": auc, "fpr50": fpr50}


def train_joint(
    model: GatedDualResidual,
    reco_a: nn.Module,
    reco_b: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr_model: float,
    lr_reco_a: float,
    lr_reco_b: float,
    weight_decay: float,
    warmup_epochs: int,
    kd_temp: float,
    target_tpr: float,
    select_metric: str,
    lambda_cls: float,
    lambda_kd: float,
    lambda_residual: float,
    lambda_gate: float,
    lambda_anchor_a: float,
    lambda_anchor_b: float,
    corrected_weight_floor: float,
    aux_mean_t: torch.Tensor,
    aux_std_t: torch.Tensor,
) -> Tuple[GatedDualResidual, nn.Module, nn.Module, Dict[str, float]]:
    init_a = {k: v.detach().cpu().clone() for k, v in reco_a.state_dict().items()}
    init_b = {k: v.detach().cpu().clone() for k, v in reco_b.state_dict().items()}

    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": float(lr_model)},
            {"params": reco_a.parameters(), "lr": float(lr_reco_a)},
            {"params": reco_b.parameters(), "lr": float(lr_reco_b)},
        ],
        lr=float(lr_model),
        weight_decay=float(weight_decay),
    )
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0
    t = float(max(kd_temp, 1e-3))

    for ep in range(int(epochs)):
        model.train()
        reco_a.train()
        reco_b.train()

        run_tot = run_cls = run_kd = run_res = run_gate = run_anc_a = run_anc_b = 0.0
        n_seen = 0

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device=device, dtype=torch.float32)
            mask_hlt = batch["mask_hlt"].to(device=device, dtype=torch.bool)
            const_hlt = batch["const_hlt"].to(device=device, dtype=torch.float32)
            hlt_logit = batch["hlt_logit"].to(device=device, dtype=torch.float32)
            teacher_logit = batch["teacher_logit"].to(device=device, dtype=torch.float32)
            y = batch["label"].to(device=device, dtype=torch.float32)

            opt.zero_grad()

            out_a = reco_a(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
            feat_a, mask_a = b.build_soft_corrected_view(
                out_a,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            out_b = reco_b(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
            feat_b, mask_b = m2mod.build_soft_corrected_view(
                out_b,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )

            aux = compute_aux_from_masks_torch(mask_hlt, mask_a, mask_b)
            aux = (aux - aux_mean_t) / aux_std_t

            out = model(feat_a, mask_a, feat_b, mask_b, hlt_logit, aux)
            final_logit = out["final_logit"]
            gate = out["gate"]
            delta = out["delta"]

            l_cls = F.binary_cross_entropy_with_logits(final_logit, y)
            soft_t = torch.sigmoid(teacher_logit / t)
            l_kd = F.binary_cross_entropy_with_logits(final_logit / t, soft_t) * (t * t)
            l_res = F.smooth_l1_loss(gate * delta, teacher_logit - hlt_logit)
            l_gate = gate.mean()

            l_anc_a = l2_anchor_to_state(reco_a, init_a)
            l_anc_b = l2_anchor_to_state(reco_b, init_b)

            loss = (
                float(lambda_cls) * l_cls
                + float(lambda_kd) * l_kd
                + float(lambda_residual) * l_res
                + float(lambda_gate) * l_gate
                + float(lambda_anchor_a) * l_anc_a
                + float(lambda_anchor_b) * l_anc_b
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reco_a.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reco_b.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            run_tot += float(loss.item()) * bs
            run_cls += float(l_cls.item()) * bs
            run_kd += float(l_kd.item()) * bs
            run_res += float(l_res.item()) * bs
            run_gate += float(l_gate.item()) * bs
            run_anc_a += float(l_anc_a.item()) * bs
            run_anc_b += float(l_anc_b.item()) * bs
            n_seen += bs

        sch.step()

        val_pack = eval_joint_dynamic(
            model=model,
            reco_a=reco_a,
            reco_b=reco_b,
            loader=val_loader,
            device=device,
            corrected_weight_floor=float(corrected_weight_floor),
            aux_mean_t=aux_mean_t,
            aux_std_t=aux_std_t,
            target_tpr=float(target_tpr),
        )
        auc_v = float(val_pack["auc"])
        fpr50_v = float(val_pack["fpr50"])

        if select_is_better(str(select_metric), auc_v, fpr50_v, best_sel):
            best_sel = float(fpr50_v) if str(select_metric).lower() == "fpr50" else float(auc_v)
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "reco_a": {k: v.detach().cpu().clone() for k, v in reco_a.state_dict().items()},
                "reco_b": {k: v.detach().cpu().clone() for k, v in reco_b.state_dict().items()},
            }
            best_metrics = {
                "best_epoch": int(ep + 1),
                "best_select_metric": str(select_metric).lower(),
                "best_sel": float(best_sel),
                "best_val_auc": float(auc_v),
                "best_val_fpr50": float(fpr50_v),
                "best_train_total": float(run_tot / max(n_seen, 1)),
                "best_train_cls": float(run_cls / max(n_seen, 1)),
                "best_train_kd": float(run_kd / max(n_seen, 1)),
                "best_train_res": float(run_res / max(n_seen, 1)),
                "best_train_gate": float(run_gate / max(n_seen, 1)),
                "best_train_anchor_a": float(run_anc_a / max(n_seen, 1)),
                "best_train_anchor_b": float(run_anc_b / max(n_seen, 1)),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Joint ep {ep+1}: train_total={run_tot/max(n_seen,1):.5f} "
                f"(cls={run_cls/max(n_seen,1):.5f}, kd={run_kd/max(n_seen,1):.5f}, "
                f"res={run_res/max(n_seen,1):.5f}, gate={run_gate/max(n_seen,1):.5f}, "
                f"ancA={run_anc_a/max(n_seen,1):.5f}, ancB={run_anc_b/max(n_seen,1):.5f}) | "
                f"val_auc={auc_v:.4f}, val_fpr50={fpr50_v:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping Joint at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        reco_a.load_state_dict(best_state["reco_a"])
        reco_b.load_state_dict(best_state["reco_b"])

    return model, reco_a, reco_b, best_metrics


def build_two_views_numpy(
    split_name: str,
    split_idx: np.ndarray,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    reco_a: nn.Module,
    reco_b: nn.Module,
    device: torch.device,
    batch_size: int,
    corrected_weight_floor: float,
) -> Dict[str, np.ndarray]:
    print(f"Building corrected views ({split_name})...")
    feat_a, mask_a = b.build_corrected_view_numpy(
        reconstructor=reco_a,
        feat_hlt=feat_hlt_std[split_idx],
        mask_hlt=hlt_mask[split_idx],
        const_hlt=hlt_const[split_idx],
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_b, mask_b = m2mod.build_corrected_view_numpy(
        reconstructor=reco_b,
        feat_hlt=feat_hlt_std[split_idx],
        mask_hlt=hlt_mask[split_idx],
        const_hlt=hlt_const[split_idx],
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_a = sanitize_numpy_features(feat_a)
    feat_b = sanitize_numpy_features(feat_b)
    return {
        "feat_a": feat_a.astype(np.float32),
        "mask_a": mask_a.astype(bool),
        "feat_b": feat_b.astype(np.float32),
        "mask_b": mask_b.astype(bool),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2_run_dir", type=str, required=True)
    ap.add_argument("--m6_run_dir", type=str, required=True)
    ap.add_argument("--m2_reco_ckpt", type=str, default="offline_reconstructor_stage2.pt")
    ap.add_argument("--m6_reco_ckpt", type=str, default="offline_reconstructor_stageA.pt")
    ap.add_argument("--m2_baseline_ckpt", type=str, default="baseline.pt")
    ap.add_argument("--teacher_ckpt", type=str, default="teacher.pt")

    ap.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "reco_teacher_joint_fusion_6model_150k75k150k" / "model20_dualview_residual_m2m6"))
    ap.add_argument("--run_name", type=str, default="model20_dualview_residual_m2m6_150k75k150k_seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--skip_save_models", action="store_true")

    ap.add_argument("--reco_eval_batch_size", type=int, default=256)
    ap.add_argument("--corrected_weight_floor", type=float, default=0.03)
    ap.add_argument("--target_tpr", type=float, default=0.50)

    ap.add_argument("--frozen_epochs", type=int, default=45)
    ap.add_argument("--frozen_patience", type=int, default=12)
    ap.add_argument("--frozen_batch_size", type=int, default=256)
    ap.add_argument("--frozen_lr", type=float, default=3e-4)
    ap.add_argument("--frozen_weight_decay", type=float, default=1e-4)
    ap.add_argument("--frozen_warmup_epochs", type=int, default=5)

    ap.add_argument("--joint_epochs", type=int, default=12)
    ap.add_argument("--joint_patience", type=int, default=6)
    ap.add_argument("--joint_batch_size", type=int, default=128)
    ap.add_argument("--joint_lr_model", type=float, default=1e-4)
    ap.add_argument("--joint_lr_reco_a", type=float, default=2e-6)
    ap.add_argument("--joint_lr_reco_b", type=float, default=2e-6)
    ap.add_argument("--joint_weight_decay", type=float, default=1e-4)
    ap.add_argument("--joint_warmup_epochs", type=int, default=3)

    ap.add_argument("--lambda_cls", type=float, default=1.0)
    ap.add_argument("--lambda_kd", type=float, default=0.10)
    ap.add_argument("--lambda_residual", type=float, default=0.05)
    ap.add_argument("--lambda_gate", type=float, default=0.01)
    ap.add_argument("--lambda_anchor_a", type=float, default=0.02)
    ap.add_argument("--lambda_anchor_b", type=float, default=0.02)
    ap.add_argument("--kd_temp", type=float, default=2.5)

    ap.add_argument("--select_metric", type=str, choices=["auc", "fpr50"], default="auc")

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

    print("Computing standardized HLT features...")
    feat_hlt = b.compute_features(hlt_const, hlt_mask)
    feat_hlt_std = b.standardize(feat_hlt, hlt_mask, means, stds)
    feat_hlt_std = sanitize_numpy_features(feat_hlt_std)

    # Load baseline + teacher logits on HLT view.
    baseline = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)

    baseline_path = m2_run / args.m2_baseline_ckpt
    teacher_path = m2_run / args.teacher_ckpt
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_path}")
    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_path}")

    safe_load_state(baseline, load_model_state(baseline_path, device), "baseline")
    safe_load_state(teacher, load_model_state(teacher_path, device), "teacher")

    hlt_logits_train = predict_single_view_logits(baseline, feat_hlt_std, hlt_mask, train_idx, device, int(args.reco_eval_batch_size))
    hlt_logits_val = predict_single_view_logits(baseline, feat_hlt_std, hlt_mask, val_idx, device, int(args.reco_eval_batch_size))
    hlt_logits_test = predict_single_view_logits(baseline, feat_hlt_std, hlt_mask, test_idx, device, int(args.reco_eval_batch_size))

    teacher_logits_train = predict_single_view_logits(teacher, feat_hlt_std, hlt_mask, train_idx, device, int(args.reco_eval_batch_size))
    teacher_logits_val = predict_single_view_logits(teacher, feat_hlt_std, hlt_mask, val_idx, device, int(args.reco_eval_batch_size))
    teacher_logits_test = predict_single_view_logits(teacher, feat_hlt_std, hlt_mask, test_idx, device, int(args.reco_eval_batch_size))

    # Load pretrained reconstructors.
    reco_a = b.OfflineReconstructor(input_dim=7, **b.BASE_CONFIG["reconstructor_model"]).to(device)
    reco_b = m2mod.OfflineReconstructor(input_dim=7, **m2mod.BASE_CONFIG["reconstructor_model"]).to(device)

    reco_a_path = m6_run / args.m6_reco_ckpt
    reco_b_path = m2_run / args.m2_reco_ckpt
    if not reco_a_path.exists():
        raise FileNotFoundError(f"Missing m6 reco checkpoint: {reco_a_path}")
    if not reco_b_path.exists():
        raise FileNotFoundError(f"Missing m2 reco checkpoint: {reco_b_path}")

    safe_load_state(reco_a, load_model_state(reco_a_path, device), "reco_a_m6")
    safe_load_state(reco_b, load_model_state(reco_b_path, device), "reco_b_m2")

    for p in reco_a.parameters():
        p.requires_grad_(False)
    for p in reco_b.parameters():
        p.requires_grad_(False)

    print("\n" + "=" * 70)
    print("STEP 1: BUILD FROZEN TWO-VIEW TENSORS")
    print("=" * 70)
    train_views = build_two_views_numpy(
        split_name="train",
        split_idx=train_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        reco_a=reco_a,
        reco_b=reco_b,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    val_views = build_two_views_numpy(
        split_name="val",
        split_idx=val_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        reco_a=reco_a,
        reco_b=reco_b,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    test_views = build_two_views_numpy(
        split_name="test",
        split_idx=test_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        reco_a=reco_a,
        reco_b=reco_b,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )

    aux_train_raw = build_aux_features_np(hlt_mask[train_idx], train_views["mask_a"], train_views["mask_b"])
    aux_val_raw = build_aux_features_np(hlt_mask[val_idx], val_views["mask_a"], val_views["mask_b"])
    aux_test_raw = build_aux_features_np(hlt_mask[test_idx], test_views["mask_a"], test_views["mask_b"])

    aux_train, aux_mu, aux_sd = standardize_aux(aux_train_raw, aux_train_raw)
    aux_val, _, _ = standardize_aux(aux_train_raw, aux_val_raw)
    aux_test, _, _ = standardize_aux(aux_train_raw, aux_test_raw)

    ds_train_f = FrozenDualResidualDataset(
        feat_a=train_views["feat_a"],
        mask_a=train_views["mask_a"],
        feat_b=train_views["feat_b"],
        mask_b=train_views["mask_b"],
        aux=aux_train,
        hlt_logit=hlt_logits_train,
        teacher_logit=teacher_logits_train,
        labels=labels[train_idx],
    )
    ds_val_f = FrozenDualResidualDataset(
        feat_a=val_views["feat_a"],
        mask_a=val_views["mask_a"],
        feat_b=val_views["feat_b"],
        mask_b=val_views["mask_b"],
        aux=aux_val,
        hlt_logit=hlt_logits_val,
        teacher_logit=teacher_logits_val,
        labels=labels[val_idx],
    )
    ds_test_f = FrozenDualResidualDataset(
        feat_a=test_views["feat_a"],
        mask_a=test_views["mask_a"],
        feat_b=test_views["feat_b"],
        mask_b=test_views["mask_b"],
        aux=aux_test,
        hlt_logit=hlt_logits_test,
        teacher_logit=teacher_logits_test,
        labels=labels[test_idx],
    )

    dl_train_f = DataLoader(ds_train_f, batch_size=int(args.frozen_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_f = DataLoader(ds_val_f, batch_size=int(args.frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_f = DataLoader(ds_test_f, batch_size=int(args.frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))

    model = GatedDualResidual(aux_dim=int(aux_train.shape[1]), dual_cfg=cfg["model"]).to(device)

    print("\n" + "=" * 70)
    print("STEP 2: TRAIN FROZEN DUALVIEW RESIDUAL (LABEL-FIRST)")
    print("=" * 70)
    model, frozen_metrics = train_frozen(
        model=model,
        train_loader=dl_train_f,
        val_loader=dl_val_f,
        device=device,
        epochs=int(args.frozen_epochs),
        patience=int(args.frozen_patience),
        lr=float(args.frozen_lr),
        weight_decay=float(args.frozen_weight_decay),
        warmup_epochs=int(args.frozen_warmup_epochs),
        kd_temp=float(args.kd_temp),
        target_tpr=float(args.target_tpr),
        select_metric=str(args.select_metric),
        lambda_cls=float(args.lambda_cls),
        lambda_kd=float(args.lambda_kd),
        lambda_residual=float(args.lambda_residual),
        lambda_gate=float(args.lambda_gate),
    )

    frozen_val = eval_frozen_loader(model, dl_val_f, device, float(args.target_tpr))
    frozen_test = eval_frozen_loader(model, dl_test_f, device, float(args.target_tpr))

    print(
        f"FrozenResidual: val_auc={float(frozen_val['auc']):.4f}, val_fpr50={float(frozen_val['fpr50']):.6f} | "
        f"test_auc={float(frozen_test['auc']):.4f}, test_fpr50={float(frozen_test['fpr50']):.6f}"
    )

    for p in reco_a.parameters():
        p.requires_grad_(True)
    for p in reco_b.parameters():
        p.requires_grad_(True)

    ds_train_j = JointDualResidualDataset(
        feat_hlt=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        hlt_logit=hlt_logits_train,
        teacher_logit=teacher_logits_train,
        labels=labels[train_idx],
    )
    ds_val_j = JointDualResidualDataset(
        feat_hlt=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        hlt_logit=hlt_logits_val,
        teacher_logit=teacher_logits_val,
        labels=labels[val_idx],
    )
    ds_test_j = JointDualResidualDataset(
        feat_hlt=feat_hlt_std[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_hlt=hlt_const[test_idx],
        hlt_logit=hlt_logits_test,
        teacher_logit=teacher_logits_test,
        labels=labels[test_idx],
    )

    dl_train_j = DataLoader(ds_train_j, batch_size=int(args.joint_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_j = DataLoader(ds_val_j, batch_size=int(args.joint_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_j = DataLoader(ds_test_j, batch_size=int(args.joint_batch_size), shuffle=False, num_workers=int(args.num_workers))

    aux_mean_t = torch.tensor(aux_mu, dtype=torch.float32, device=device)
    aux_std_t = torch.tensor(aux_sd, dtype=torch.float32, device=device)

    joint_metrics: Dict[str, float] = {}
    joint_val = {
        "probs": np.zeros((0,), dtype=np.float64),
        "labels": np.zeros((0,), dtype=np.float32),
        "gates": np.zeros((0,), dtype=np.float64),
        "delta": np.zeros((0,), dtype=np.float64),
        "auc": float("nan"),
        "fpr50": float("nan"),
    }
    joint_test = copy.deepcopy(joint_val)

    if int(args.joint_epochs) > 0:
        print("\n" + "=" * 70)
        print("STEP 3: LIGHT JOINT FINETUNE (RECO-A + RECO-B + RESIDUAL)")
        print("=" * 70)
        model, reco_a, reco_b, joint_metrics = train_joint(
            model=model,
            reco_a=reco_a,
            reco_b=reco_b,
            train_loader=dl_train_j,
            val_loader=dl_val_j,
            device=device,
            epochs=int(args.joint_epochs),
            patience=int(args.joint_patience),
            lr_model=float(args.joint_lr_model),
            lr_reco_a=float(args.joint_lr_reco_a),
            lr_reco_b=float(args.joint_lr_reco_b),
            weight_decay=float(args.joint_weight_decay),
            warmup_epochs=int(args.joint_warmup_epochs),
            kd_temp=float(args.kd_temp),
            target_tpr=float(args.target_tpr),
            select_metric=str(args.select_metric),
            lambda_cls=float(args.lambda_cls),
            lambda_kd=float(args.lambda_kd),
            lambda_residual=float(args.lambda_residual),
            lambda_gate=float(args.lambda_gate),
            lambda_anchor_a=float(args.lambda_anchor_a),
            lambda_anchor_b=float(args.lambda_anchor_b),
            corrected_weight_floor=float(args.corrected_weight_floor),
            aux_mean_t=aux_mean_t,
            aux_std_t=aux_std_t,
        )

        joint_val = eval_joint_dynamic(
            model=model,
            reco_a=reco_a,
            reco_b=reco_b,
            loader=dl_val_j,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            aux_mean_t=aux_mean_t,
            aux_std_t=aux_std_t,
            target_tpr=float(args.target_tpr),
        )
        joint_test = eval_joint_dynamic(
            model=model,
            reco_a=reco_a,
            reco_b=reco_b,
            loader=dl_test_j,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            aux_mean_t=aux_mean_t,
            aux_std_t=aux_std_t,
            target_tpr=float(args.target_tpr),
        )

        print(
            f"JointResidual: val_auc={float(joint_val['auc']):.4f}, val_fpr50={float(joint_val['fpr50']):.6f} | "
            f"test_auc={float(joint_test['auc']):.4f}, test_fpr50={float(joint_test['fpr50']):.6f}"
        )

    # Threshold-based test FPR at val-selected threshold (frozen and joint).
    thr_frozen = threshold_at_target_tpr(frozen_val["labels"], frozen_val["probs"], float(args.target_tpr))
    fpr_test_thr_frozen = fpr_from_val_threshold(frozen_test["labels"], frozen_test["probs"], thr_frozen)

    if joint_val["probs"].size > 0:
        thr_joint = threshold_at_target_tpr(joint_val["labels"], joint_val["probs"], float(args.target_tpr))
        fpr_test_thr_joint = fpr_from_val_threshold(joint_test["labels"], joint_test["probs"], thr_joint)
    else:
        thr_joint = float("nan")
        fpr_test_thr_joint = float("nan")

    # Reference HLT / Teacher metrics on same splits.
    probs_hlt_val = sigmoid_np(hlt_logits_val)
    probs_hlt_test = sigmoid_np(hlt_logits_test)
    probs_teacher_val = sigmoid_np(teacher_logits_val)
    probs_teacher_test = sigmoid_np(teacher_logits_test)
    hlt_auc_val, hlt_fpr50_val = auc_and_fpr(labels[val_idx], probs_hlt_val, float(args.target_tpr))
    hlt_auc_test, hlt_fpr50_test = auc_and_fpr(labels[test_idx], probs_hlt_test, float(args.target_tpr))
    teacher_auc_val, teacher_fpr50_val = auc_and_fpr(labels[val_idx], probs_teacher_val, float(args.target_tpr))
    teacher_auc_test, teacher_fpr50_test = auc_and_fpr(labels[test_idx], probs_teacher_test, float(args.target_tpr))

    np.savez_compressed(
        save_root / "dualview_residual_scores.npz",
        labels_val=labels[val_idx].astype(np.float32),
        labels_test=labels[test_idx].astype(np.float32),
        preds_hlt_val=probs_hlt_val.astype(np.float64),
        preds_hlt_test=probs_hlt_test.astype(np.float64),
        preds_teacher_val=probs_teacher_val.astype(np.float64),
        preds_teacher_test=probs_teacher_test.astype(np.float64),
        preds_residual_frozen_val=frozen_val["probs"].astype(np.float64),
        preds_residual_frozen_test=frozen_test["probs"].astype(np.float64),
        preds_residual_joint_val=joint_val["probs"].astype(np.float64),
        preds_residual_joint_test=joint_test["probs"].astype(np.float64),
        gates_frozen_val=frozen_val["gates"].astype(np.float64),
        gates_frozen_test=frozen_test["gates"].astype(np.float64),
        gates_joint_val=joint_val["gates"].astype(np.float64),
        gates_joint_test=joint_test["gates"].astype(np.float64),
    )

    out_json = {
        "variant": "model20_dualview_residual_m2m6",
        "seed": int(args.seed),
        "select_metric": str(args.select_metric).lower(),
        "target_tpr": float(args.target_tpr),
        "m2": {
            "run_dir": str(m2_run),
            "reco_ckpt": str(args.m2_reco_ckpt),
            "baseline_ckpt": str(args.m2_baseline_ckpt),
            "teacher_ckpt": str(args.teacher_ckpt),
        },
        "m6": {
            "run_dir": str(m6_run),
            "reco_ckpt": str(args.m6_reco_ckpt),
        },
        "weights": {
            "lambda_cls": float(args.lambda_cls),
            "lambda_kd": float(args.lambda_kd),
            "lambda_residual": float(args.lambda_residual),
            "lambda_gate": float(args.lambda_gate),
            "lambda_anchor_a": float(args.lambda_anchor_a),
            "lambda_anchor_b": float(args.lambda_anchor_b),
            "kd_temp": float(args.kd_temp),
        },
        "hlt": {
            "auc_val": float(hlt_auc_val),
            "auc_test": float(hlt_auc_test),
            "fpr50_val": float(hlt_fpr50_val),
            "fpr50_test": float(hlt_fpr50_test),
        },
        "teacher": {
            "auc_val": float(teacher_auc_val),
            "auc_test": float(teacher_auc_test),
            "fpr50_val": float(teacher_fpr50_val),
            "fpr50_test": float(teacher_fpr50_test),
        },
        "residual_frozen_train": frozen_metrics,
        "residual_frozen_eval": {
            "auc_val": float(frozen_val["auc"]),
            "fpr50_val": float(frozen_val["fpr50"]),
            "auc_test": float(frozen_test["auc"]),
            "fpr50_test": float(frozen_test["fpr50"]),
            "fpr_test_at_val_thr": float(fpr_test_thr_frozen),
            "val_threshold": float(thr_frozen),
            "gate_mean_val": float(np.mean(frozen_val["gates"])) if frozen_val["gates"].size else float("nan"),
            "gate_mean_test": float(np.mean(frozen_test["gates"])) if frozen_test["gates"].size else float("nan"),
        },
        "residual_joint_train": joint_metrics,
        "residual_joint_eval": {
            "auc_val": float(joint_val["auc"]),
            "fpr50_val": float(joint_val["fpr50"]),
            "auc_test": float(joint_test["auc"]),
            "fpr50_test": float(joint_test["fpr50"]),
            "fpr_test_at_val_thr": float(fpr_test_thr_joint),
            "val_threshold": float(thr_joint),
            "gate_mean_val": float(np.mean(joint_val["gates"])) if joint_val["gates"].size else float("nan"),
            "gate_mean_test": float(np.mean(joint_test["gates"])) if joint_test["gates"].size else float("nan"),
        },
    }

    with open(save_root / "dualview_residual_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    if not args.skip_save_models:
        torch.save(
            {
                "model": model.state_dict(),
                "stage": "frozen_or_latest",
                "metrics": frozen_metrics,
                "aux_mean": aux_mu.astype(np.float32),
                "aux_std": aux_sd.astype(np.float32),
            },
            save_root / "dualview_residual_prejoint.pt",
        )
        torch.save(
            {
                "model": model.state_dict(),
                "stage": "post_joint",
                "metrics": joint_metrics,
                "aux_mean": aux_mu.astype(np.float32),
                "aux_std": aux_sd.astype(np.float32),
            },
            save_root / "dualview_residual_postjoint.pt",
        )
        torch.save({"model": reco_a.state_dict()}, save_root / "offline_reconstructor_A_postjoint.pt")
        torch.save({"model": reco_b.state_dict()}, save_root / "offline_reconstructor_B_postjoint.pt")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"HLT AUC (val/test): {hlt_auc_val:.4f} / {hlt_auc_test:.4f}\n"
        f"Teacher AUC (val/test): {teacher_auc_val:.4f} / {teacher_auc_test:.4f}\n"
        f"ResidualFrozen AUC (val/test): {float(frozen_val['auc']):.4f} / {float(frozen_test['auc']):.4f}\n"
        f"ResidualJoint AUC (val/test): {float(joint_val['auc']):.4f} / {float(joint_test['auc']):.4f}\n"
        f"FPR@50 HLT / Teacher / ResidualFrozen / ResidualJoint (test): "
        f"{hlt_fpr50_test:.6f} / {teacher_fpr50_test:.6f} / {float(frozen_test['fpr50']):.6f} / {float(joint_test['fpr50']):.6f}"
    )
    print(f"\nSaved model-20 dualview residual results to: {save_root}")


if __name__ == "__main__":
    main()

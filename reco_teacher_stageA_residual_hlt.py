#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-A RecoTeacher + residual correction pipeline.

Pipeline:
1) Train offline teacher and HLT baseline (single-view).
2) Train Stage-A reconstructor with teacher-guided losses (s09-style supported,
   stagewise best-reload between scale phases).
3) Freeze reconstructor, build corrected view, and train residual head to predict
   r* = teacher_logit(offline) - hlt_logit.
4) Compose final score as hlt_logit + alpha * r_hat, where alpha is selected on val.
5) Optional light joint finetune of (reconstructor + residual head) with anchor.

Outputs:
- stageA_residual_scores.npz
- stageA_residual_metrics.json
- model checkpoints for teacher/baseline/reconstructor/residual head
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as b
import reco_teacher_stageA_only_delta_curriculum as sA

_ORIG_TEACHER_GUIDED_RECO_LOSSES = b._compute_teacher_guided_reco_losses
_RATIO_AWARE_COUNT_CFG: Dict[str, float | bool] = {
    "enabled": False,
    "eps": 0.015,
    "under_lambda": 1.0,
    "over_lambda": 0.25,
    "over_margin_base": 2.0,
    "over_margin_scale": 6.0,
    "over_ratio_gamma": 0.7,
    "over_lambda_floor": 0.05,
}


def _compute_ratio_aware_count_budget_hinge(
    reco_out: Dict[str, torch.Tensor],
    mask_hlt: torch.Tensor,
    mask_off: torch.Tensor,
    cfg: Dict[str, float | bool],
) -> torch.Tensor:
    true_count = mask_off.float().sum(dim=1)
    hlt_count = mask_hlt.float().sum(dim=1)
    true_added = (true_count - hlt_count).clamp(min=0.0)

    pred_added = reco_out["child_weight"].sum(dim=1) + reco_out["gen_weight"].sum(dim=1)

    ratio = hlt_count / torch.clamp(true_count, min=1.0)
    deficit = torch.clamp(1.0 - ratio, min=0.0, max=1.0)

    over_margin = float(cfg["over_margin_base"]) + float(cfg["over_margin_scale"]) * deficit
    over_lambda = float(cfg["over_lambda"]) * (1.0 - float(cfg["over_ratio_gamma"]) * deficit)
    over_lambda = torch.clamp(over_lambda, min=float(cfg["over_lambda_floor"]))
    under_lambda = float(cfg["under_lambda"])
    eps = max(float(cfg["eps"]), 0.0)

    under = F.relu(true_added - pred_added - eps)
    over = F.relu(pred_added - true_added - eps - over_margin)
    loss_vec = under_lambda * under.square() + over_lambda * over.square()
    return loss_vec.mean()


def _compute_teacher_guided_reco_losses_ratio_aware(
    reco_out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    teacher_model: nn.Module,
    means_t: torch.Tensor,
    stds_t: torch.Tensor,
    loss_cfg: Dict,
    kd_temperature: float,
    budget_eps: float,
    budget_weight_floor: float,
) -> Dict[str, torch.Tensor]:
    losses = _ORIG_TEACHER_GUIDED_RECO_LOSSES(
        reco_out=reco_out,
        const_hlt=const_hlt,
        mask_hlt=mask_hlt,
        const_off=const_off,
        mask_off=mask_off,
        budget_merge_true=budget_merge_true,
        budget_eff_true=budget_eff_true,
        teacher_model=teacher_model,
        means_t=means_t,
        stds_t=stds_t,
        loss_cfg=loss_cfg,
        kd_temperature=kd_temperature,
        budget_eps=budget_eps,
        budget_weight_floor=budget_weight_floor,
    )
    if bool(_RATIO_AWARE_COUNT_CFG.get("enabled", False)):
        losses["budget_hinge"] = _compute_ratio_aware_count_budget_hinge(
            reco_out=reco_out,
            mask_hlt=mask_hlt,
            mask_off=mask_off,
            cfg=_RATIO_AWARE_COUNT_CFG,
        )
    return losses



def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def threshold_at_target_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> Tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(labels.astype(np.float32), scores.astype(np.float64))
    if len(thr) == 0:
        return 0.5, float("nan"), float("nan")
    valid = np.isfinite(thr)
    if not np.any(valid):
        return float(np.median(scores)), float("nan"), float("nan")
    idx_valid = np.where(valid)[0]
    idx = int(idx_valid[np.argmin(np.abs(tpr[idx_valid] - float(target_tpr)))])
    return float(thr[idx]), float(tpr[idx]), float(fpr[idx])


def rates_from_threshold(labels: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    labels_b = labels.astype(np.float32) > 0.5
    neg_b = ~labels_b
    pred = scores >= float(thr)
    tp = int((pred & labels_b).sum())
    fp = int((pred & neg_b).sum())
    n_pos = int(labels_b.sum())
    n_neg = int(neg_b.sum())
    return {
        "tp": tp,
        "fp": fp,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tpr": float(tp / max(n_pos, 1)),
        "fpr": float(fp / max(n_neg, 1)),
    }


def auc_and_fpr_at_target(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> Dict[str, float]:
    labels = labels.astype(np.float32)
    scores = scores.astype(np.float64)
    auc = float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr_at = float(b.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))
    return {"auc": auc, "fpr_at_target_tpr": fpr_at}


@torch.no_grad()
def predict_logits_single_view(
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
    for start in range(0, len(idx), int(batch_size)):
        end = min(start + int(batch_size), len(idx))
        sl = idx[start:end]
        x = torch.tensor(feat[sl], dtype=torch.float32, device=device)
        m = torch.tensor(mask[sl], dtype=torch.bool, device=device)
        logits = model(x, m).squeeze(1)
        k = end - start
        out[ptr: ptr + k] = logits.detach().cpu().numpy().astype(np.float64)
        ptr += k
    return out


class ResidualDataset(Dataset):
    def __init__(
        self,
        feat_corr: np.ndarray,
        mask_corr: np.ndarray,
        hlt_logit: np.ndarray,
        teacher_logit: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat = torch.tensor(feat_corr, dtype=torch.float32)
        self.mask = torch.tensor(mask_corr, dtype=torch.bool)
        self.hlt_logit = torch.tensor(hlt_logit.astype(np.float32), dtype=torch.float32)
        self.teacher_logit = torch.tensor(teacher_logit.astype(np.float32), dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat": self.feat[i],
            "mask": self.mask[i],
            "hlt_logit": self.hlt_logit[i],
            "teacher_logit": self.teacher_logit[i],
            "label": self.labels[i],
        }


class ResidualJointDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        labels: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
        hlt_logit: np.ndarray,
        teacher_logit: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        self.budget_merge_true = torch.tensor(budget_merge_true.astype(np.float32), dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true.astype(np.float32), dtype=torch.float32)
        self.hlt_logit = torch.tensor(hlt_logit.astype(np.float32), dtype=torch.float32)
        self.teacher_logit = torch.tensor(teacher_logit.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "label": self.labels[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
            "hlt_logit": self.hlt_logit[i],
            "teacher_logit": self.teacher_logit[i],
        }


class OfflineDropoutTeacherDataset(Dataset):
    def __init__(
        self,
        feat_off: np.ndarray,
        mask_off: np.ndarray,
        mask_hlt: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        drop_prob: float = 0.0,
        seed: int = 0,
        drop_mode: str = "random",
        num_banks: int = 1,
    ):
        self.feat_off = feat_off
        self.mask_off = mask_off
        self.mask_hlt = mask_hlt
        self.labels = labels.astype(np.float32)
        self.indices = indices.astype(np.int64)
        self.drop_prob = float(drop_prob)
        self.drop_mode = str(drop_mode)
        self.num_banks = int(max(1, num_banks))
        self.current_bank = 0
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(int(seed))

    def set_drop_prob(self, p: float) -> None:
        self.drop_prob = float(np.clip(p, 0.0, 1.0))

    def set_current_bank(self, bank: int) -> None:
        self.current_bank = int(bank) % int(self.num_banks)

    def _deterministic_keep_extra(self, n_extra: int, idx: int) -> np.ndarray:
        # Stable per-jet/per-bank Bernoulli mask for reproducible dropout.
        key = (
            (self.base_seed * 1315423911)
            ^ (int(self.current_bank) * 2654435761)
            ^ (int(idx) * 2246822519)
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(np.uint64(key))
        return rng.random(int(n_extra)) >= float(self.drop_prob)

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        idx = int(self.indices[i])
        feat_full = self.feat_off[idx].astype(np.float32, copy=True)
        mask_full = self.mask_off[idx].astype(bool, copy=True)
        mask_hlt = self.mask_hlt[idx].astype(bool, copy=False)

        extra_mask = mask_full & (~mask_hlt)
        if self.drop_prob > 0.0 and np.any(extra_mask):
            if self.drop_mode == "deterministic_bank":
                keep_extra = self._deterministic_keep_extra(int(extra_mask.sum()), idx)
            else:
                keep_extra = self.rng.random(int(extra_mask.sum())) >= self.drop_prob
            drop_mask = extra_mask.copy()
            drop_mask[extra_mask] = ~keep_extra
            mask_drop = mask_full & (~drop_mask)
        else:
            mask_drop = mask_full.copy()

        # Safety: never return an empty constituent set.
        if not np.any(mask_drop):
            if np.any(mask_hlt):
                mask_drop = mask_hlt.copy()
            elif np.any(mask_full):
                first = int(np.flatnonzero(mask_full)[0])
                mask_drop[first] = True

        feat_drop = feat_full.copy()
        feat_drop[~mask_drop] = 0.0

        return {
            "feat_full": feat_full,
            "mask_full": mask_full.astype(bool),
            "feat_drop": feat_drop,
            "mask_drop": mask_drop.astype(bool),
            "label": np.float32(self.labels[idx]),
        }


def train_single_view_teacher_with_offline_dropout(
    model: nn.Module,
    feat_off: np.ndarray,
    mask_off: np.ndarray,
    mask_hlt: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_loader_full: DataLoader,
    device: torch.device,
    train_cfg: Dict[str, float | int],
    target_tpr: float,
    drop_prob_max: float,
    drop_warmup_epochs: int,
    lambda_drop_cls: float,
    use_consistency: bool,
    consistency_temp: float,
    lambda_consistency: float,
    drop_mode: str,
    drop_num_banks: int,
    drop_bank_cycle_epochs: int,
    seed: int,
    name: str = "TeacherDrop",
) -> nn.Module:
    epochs = int(train_cfg.get("epochs", 60))
    patience = int(train_cfg.get("patience", 10))
    batch_size = int(train_cfg.get("batch_size", 512))
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))

    train_ds = OfflineDropoutTeacherDataset(
        feat_off=feat_off,
        mask_off=mask_off,
        mask_hlt=mask_hlt,
        labels=labels,
        indices=train_idx,
        drop_prob=0.0,
        seed=int(seed),
        drop_mode=str(drop_mode),
        num_banks=int(max(1, drop_num_banks)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_auc = float("-inf")
    best_state = None
    no_improve = 0

    t = float(max(consistency_temp, 1e-3))
    use_consistency = bool(use_consistency)
    bank_cycle_epochs = int(max(1, drop_bank_cycle_epochs))

    for ep in range(epochs):
        p = float(drop_prob_max) * min(1.0, float(ep + 1) / float(max(int(drop_warmup_epochs), 1)))
        train_ds.set_drop_prob(p)
        if str(drop_mode) == "deterministic_bank":
            bank = (int(ep) // int(bank_cycle_epochs)) % int(max(1, drop_num_banks))
            train_ds.set_current_bank(bank)
        else:
            bank = -1

        model.train()
        running_loss = 0.0
        running_l_full = 0.0
        running_l_drop = 0.0
        running_l_cons = 0.0
        n_seen = 0

        for batch in train_loader:
            x_full = batch["feat_full"].to(device=device, dtype=torch.float32)
            m_full = batch["mask_full"].to(device=device, dtype=torch.bool)
            x_drop = batch["feat_drop"].to(device=device, dtype=torch.float32)
            m_drop = batch["mask_drop"].to(device=device, dtype=torch.bool)
            y = batch["label"].to(device=device, dtype=torch.float32)

            opt.zero_grad()
            z_full = model(x_full, m_full).squeeze(1)
            z_drop = model(x_drop, m_drop).squeeze(1)

            l_full = F.binary_cross_entropy_with_logits(z_full, y)
            l_drop = F.binary_cross_entropy_with_logits(z_drop, y)

            if use_consistency:
                target_soft = torch.sigmoid(z_full.detach() / t)
                l_cons = F.binary_cross_entropy_with_logits(z_drop / t, target_soft) * (t * t)
            else:
                l_cons = torch.zeros((), device=device)

            loss = l_full + float(lambda_drop_cls) * l_drop + float(lambda_consistency) * l_cons
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x_full.size(0)
            running_loss += float(loss.item()) * bs
            running_l_full += float(l_full.item()) * bs
            running_l_drop += float(l_drop.item()) * bs
            running_l_cons += float(l_cons.item()) * bs
            n_seen += bs

        sch.step()

        tr_loss = running_loss / max(n_seen, 1)
        tr_l_full = running_l_full / max(n_seen, 1)
        tr_l_drop = running_l_drop / max(n_seen, 1)
        tr_l_cons = running_l_cons / max(n_seen, 1)

        val_auc, val_scores, val_labels = b.eval_classifier(model, val_loader_full, device)
        if np.unique(val_labels).size > 1 and val_scores.size > 0:
            fpr_v, tpr_v, _ = roc_curve(val_labels.astype(np.float32), val_scores.astype(np.float64))
            val_fpr50 = float(b.fpr_at_target_tpr(fpr_v, tpr_v, float(target_tpr)))
        else:
            val_fpr50 = float("nan")

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            bank_str = f", bank={bank}" if str(drop_mode) == "deterministic_bank" else ""
            print(
                f"{name} ep {ep+1}: train_loss={tr_loss:.5f} (full={tr_l_full:.5f}, drop={tr_l_drop:.5f}, cons={tr_l_cons:.5f}, p={p:.3f}{bank_str}) | "
                f"val_auc={float(val_auc):.4f}, val_fpr50={val_fpr50:.6f}, best_auc={best_auc:.4f}"
            )

        if no_improve >= patience:
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class ResidualHead(nn.Module):
    def __init__(self, input_dim: int, model_cfg: Dict[str, int | float]):
        super().__init__()
        self.model = b.ParticleTransformer(input_dim=int(input_dim), **model_cfg)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(feat, mask).squeeze(1)


@torch.no_grad()
def predict_residual_head(
    model: nn.Module,
    feat_corr: np.ndarray,
    mask_corr: np.ndarray,
    split_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    idx = split_idx.astype(np.int64)
    out = np.zeros(idx.shape[0], dtype=np.float64)
    ptr = 0
    for start in range(0, len(idx), int(batch_size)):
        end = min(start + int(batch_size), len(idx))
        sl = idx[start:end]
        x = torch.tensor(feat_corr[sl], dtype=torch.float32, device=device)
        m = torch.tensor(mask_corr[sl], dtype=torch.bool, device=device)
        r_hat = model(x, m)
        k = end - start
        out[ptr: ptr + k] = r_hat.detach().cpu().numpy().astype(np.float64)
        ptr += k
    return out


@torch.no_grad()
def predict_residual_head_with_reco(
    reconstructor: nn.Module,
    residual_head: nn.Module,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    split_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
    corrected_weight_floor: float,
) -> np.ndarray:
    reconstructor.eval()
    residual_head.eval()
    idx = split_idx.astype(np.int64)
    out = np.zeros(idx.shape[0], dtype=np.float64)
    ptr = 0
    for start in range(0, len(idx), int(batch_size)):
        end = min(start + int(batch_size), len(idx))
        sl = idx[start:end]
        x = torch.tensor(feat_hlt[sl], dtype=torch.float32, device=device)
        m = torch.tensor(mask_hlt[sl], dtype=torch.bool, device=device)
        c = torch.tensor(const_hlt[sl], dtype=torch.float32, device=device)

        reco_out = reconstructor(x, m, c, stage_scale=1.0)
        feat_corr_t, mask_corr_t = b.build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=False,
        )
        r_hat = residual_head(feat_corr_t, mask_corr_t)
        k = end - start
        out[ptr: ptr + k] = r_hat.detach().cpu().numpy().astype(np.float64)
        ptr += k
    return out


def select_alpha_on_val_and_eval_test(
    labels_val: np.ndarray,
    hlt_logits_val: np.ndarray,
    rhat_val: np.ndarray,
    labels_test: np.ndarray,
    hlt_logits_test: np.ndarray,
    rhat_test: np.ndarray,
    alpha_grid: List[float],
    target_tpr: float,
) -> Dict[str, object]:
    labels_val = labels_val.astype(np.float32)
    labels_test = labels_test.astype(np.float32)

    best = {
        "alpha": float("nan"),
        "fpr_val": float("inf"),
        "tpr_val": float("nan"),
        "threshold_val": float("nan"),
        "auc_val": float("nan"),
    }

    for alpha in alpha_grid:
        s_val = sigmoid_np(hlt_logits_val + float(alpha) * rhat_val)
        thr, tpr_v, fpr_v = threshold_at_target_tpr(labels_val, s_val, float(target_tpr))
        auc_v = float(roc_auc_score(labels_val, s_val)) if np.unique(labels_val).size > 1 else float("nan")

        replace = False
        if fpr_v < best["fpr_val"]:
            replace = True
        elif np.isclose(fpr_v, best["fpr_val"]):
            if abs(tpr_v - float(target_tpr)) < abs(best["tpr_val"] - float(target_tpr)):
                replace = True
        if replace:
            best = {
                "alpha": float(alpha),
                "fpr_val": float(fpr_v),
                "tpr_val": float(tpr_v),
                "threshold_val": float(thr),
                "auc_val": float(auc_v),
            }

    a = float(best["alpha"])
    s_test = sigmoid_np(hlt_logits_test + a * rhat_test)
    rates_test = rates_from_threshold(labels_test, s_test, float(best["threshold_val"]))
    auc_test = float(roc_auc_score(labels_test, s_test)) if np.unique(labels_test).size > 1 else float("nan")
    fpr_test_exact = auc_and_fpr_at_target(labels_test, s_test, float(target_tpr))["fpr_at_target_tpr"]

    # Oracle alpha reference on test (post-hoc only)
    best_oracle = {
        "alpha": float("nan"),
        "fpr_test": float("inf"),
        "tpr_test": float("nan"),
        "threshold_test": float("nan"),
        "auc_test": float("nan"),
    }
    for alpha in alpha_grid:
        s_t = sigmoid_np(hlt_logits_test + float(alpha) * rhat_test)
        thr_t, tpr_t, fpr_t = threshold_at_target_tpr(labels_test, s_t, float(target_tpr))
        auc_t = float(roc_auc_score(labels_test, s_t)) if np.unique(labels_test).size > 1 else float("nan")

        replace = False
        if fpr_t < best_oracle["fpr_test"]:
            replace = True
        elif np.isclose(fpr_t, best_oracle["fpr_test"]):
            if abs(tpr_t - float(target_tpr)) < abs(best_oracle["tpr_test"] - float(target_tpr)):
                replace = True
        if replace:
            best_oracle = {
                "alpha": float(alpha),
                "fpr_test": float(fpr_t),
                "tpr_test": float(tpr_t),
                "threshold_test": float(thr_t),
                "auc_test": float(auc_t),
            }

    return {
        "selection": {
            "source": "val",
            "target_tpr": float(target_tpr),
            **best,
        },
        "test_eval": {
            "target_tpr": float(target_tpr),
            "alpha": float(best["alpha"]),
            "threshold_from_val": float(best["threshold_val"]),
            "tpr": float(rates_test["tpr"]),
            "fpr": float(rates_test["fpr"]),
            "tp": int(rates_test["tp"]),
            "fp": int(rates_test["fp"]),
            "auc": float(auc_test),
            "fpr_at_target_tpr_exact": float(fpr_test_exact),
        },
        "test_oracle": {
            "target_tpr": float(target_tpr),
            **best_oracle,
        },
    }


def train_residual_head(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_res: float,
    lambda_kd: float,
    lambda_cls: float,
    kd_temp: float,
    target_tpr: float,
    select_metric: str,
) -> Tuple[nn.Module, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    kd_temp = float(max(kd_temp, 1e-3))

    for ep in range(int(epochs)):
        model.train()
        tr_loss = tr_res = tr_kd = tr_cls = 0.0
        n_tr = 0

        for batch in train_loader:
            feat = batch["feat"].to(device)
            mask = batch["mask"].to(device)
            hlt_logit = batch["hlt_logit"].to(device)
            teacher_logit = batch["teacher_logit"].to(device)
            y = batch["label"].to(device)

            opt.zero_grad()
            r_hat = model(feat, mask)
            r_target = teacher_logit - hlt_logit
            final_logit = hlt_logit + r_hat

            loss_res = F.smooth_l1_loss(r_hat, r_target)
            target_soft = torch.sigmoid(teacher_logit / kd_temp)
            loss_kd = (
                F.binary_cross_entropy_with_logits(final_logit / kd_temp, target_soft, reduction="mean")
                * (kd_temp * kd_temp)
            )
            loss_cls = F.binary_cross_entropy_with_logits(final_logit, y)

            loss = float(lambda_res) * loss_res + float(lambda_kd) * loss_kd + float(lambda_cls) * loss_cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = feat.size(0)
            tr_loss += float(loss.item()) * bs
            tr_res += float(loss_res.item()) * bs
            tr_kd += float(loss_kd.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_res /= max(n_tr, 1)
        tr_kd /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)

        model.eval()
        va_loss = va_res = va_kd = va_cls = 0.0
        n_va = 0
        va_scores = []
        va_labels = []

        with torch.no_grad():
            for batch in val_loader:
                feat = batch["feat"].to(device)
                mask = batch["mask"].to(device)
                hlt_logit = batch["hlt_logit"].to(device)
                teacher_logit = batch["teacher_logit"].to(device)
                y = batch["label"].to(device)

                r_hat = model(feat, mask)
                r_target = teacher_logit - hlt_logit
                final_logit = hlt_logit + r_hat

                loss_res = F.smooth_l1_loss(r_hat, r_target)
                target_soft = torch.sigmoid(teacher_logit / kd_temp)
                loss_kd = (
                    F.binary_cross_entropy_with_logits(final_logit / kd_temp, target_soft, reduction="mean")
                    * (kd_temp * kd_temp)
                )
                loss_cls = F.binary_cross_entropy_with_logits(final_logit, y)
                loss = float(lambda_res) * loss_res + float(lambda_kd) * loss_kd + float(lambda_cls) * loss_cls

                bs = feat.size(0)
                va_loss += float(loss.item()) * bs
                va_res += float(loss_res.item()) * bs
                va_kd += float(loss_kd.item()) * bs
                va_cls += float(loss_cls.item()) * bs
                n_va += bs

                va_scores.append(sigmoid_np(final_logit.detach().cpu().numpy()))
                va_labels.append(y.detach().cpu().numpy().astype(np.float32))

        va_loss /= max(n_va, 1)
        va_res /= max(n_va, 1)
        va_kd /= max(n_va, 1)
        va_cls /= max(n_va, 1)

        s_val = np.concatenate(va_scores) if va_scores else np.zeros((0,), dtype=np.float64)
        y_val = np.concatenate(va_labels) if va_labels else np.zeros((0,), dtype=np.float32)

        if np.unique(y_val).size > 1 and s_val.size > 0:
            auc_v = float(roc_auc_score(y_val, s_val))
            fpr_v, tpr_v, _ = roc_curve(y_val, s_val)
            fpr50_v = float(b.fpr_at_target_tpr(fpr_v, tpr_v, float(target_tpr)))
        else:
            auc_v, fpr50_v = float("nan"), float("nan")

        if str(select_metric).lower() == "fpr50":
            sel_val = fpr50_v
            is_better = np.isfinite(sel_val) and (sel_val < best_sel)
        else:
            sel_val = auc_v
            is_better = np.isfinite(sel_val) and (sel_val > best_sel)

        if is_better:
            best_sel = float(sel_val)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "best_epoch": int(ep + 1),
                "best_select_metric": str(select_metric).lower(),
                "best_sel": float(best_sel),
                "best_val_auc": float(auc_v),
                "best_val_fpr50": float(fpr50_v),
                "best_val_loss": float(va_loss),
                "best_val_loss_res": float(va_res),
                "best_val_loss_kd": float(va_kd),
                "best_val_loss_cls": float(va_cls),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Residual ep {ep+1}: train_loss={tr_loss:.5f} (res={tr_res:.5f}, kd={tr_kd:.5f}, cls={tr_cls:.5f}) | "
                f"val_loss={va_loss:.5f}, val_auc={auc_v:.4f}, val_fpr50={fpr50_v:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping Residual head at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def train_residual_joint(
    reconstructor: nn.Module,
    residual_head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    teacher_model: nn.Module,
    device: torch.device,
    means: np.ndarray,
    stds: np.ndarray,
    epochs: int,
    patience: int,
    lr_reco: float,
    lr_head: float,
    weight_decay: float,
    warmup_epochs: int,
    corrected_weight_floor: float,
    lambda_res: float,
    lambda_kd: float,
    lambda_cls: float,
    kd_temp: float,
    lambda_reco_anchor: float,
    stageA_lambda_kd: float,
    stageA_lambda_emb: float,
    stageA_lambda_tok: float,
    stageA_lambda_phys: float,
    stageA_lambda_budget_hinge: float,
    stageA_budget_eps: float,
    stageA_budget_weight_floor: float,
    target_tpr: float,
    select_metric: str,
) -> Tuple[nn.Module, nn.Module, Dict[str, float]]:
    opt = torch.optim.AdamW(
        [
            {"params": residual_head.parameters(), "lr": float(lr_head)},
            {"params": reconstructor.parameters(), "lr": float(lr_reco)},
        ],
        lr=float(lr_head),
        weight_decay=float(weight_decay),
    )
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(np.clip(stds, 1e-6, None), dtype=torch.float32, device=device)
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    teacher_model.eval()

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state_reco = None
    best_state_head = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0
    kd_temp = float(max(kd_temp, 1e-3))

    for ep in range(int(epochs)):
        reconstructor.train()
        residual_head.train()

        tr_loss = tr_res = tr_kd = tr_cls = tr_anchor = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            y = batch["label"].to(device)
            b_merge = batch["budget_merge_true"].to(device)
            b_eff = batch["budget_eff_true"].to(device)
            hlt_logit = batch["hlt_logit"].to(device)
            teacher_logit = batch["teacher_logit"].to(device)

            opt.zero_grad()
            reco_out = reconstructor(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
            feat_corr_t, mask_corr_t = b.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            r_hat = residual_head(feat_corr_t, mask_corr_t)
            r_target = teacher_logit - hlt_logit
            final_logit = hlt_logit + r_hat

            loss_res = F.smooth_l1_loss(r_hat, r_target)
            target_soft = torch.sigmoid(teacher_logit / kd_temp)
            loss_kd = (
                F.binary_cross_entropy_with_logits(final_logit / kd_temp, target_soft, reduction="mean")
                * (kd_temp * kd_temp)
            )
            loss_cls = F.binary_cross_entropy_with_logits(final_logit, y)

            loss = float(lambda_res) * loss_res + float(lambda_kd) * loss_kd + float(lambda_cls) * loss_cls

            anchor_val = torch.zeros((), device=device)
            if float(lambda_reco_anchor) > 0.0:
                reco_losses = b._compute_teacher_guided_reco_losses(
                    reco_out=reco_out,
                    const_hlt=const_hlt,
                    mask_hlt=mask_hlt,
                    const_off=const_off,
                    mask_off=mask_off,
                    budget_merge_true=b_merge,
                    budget_eff_true=b_eff,
                    teacher_model=teacher_model,
                    means_t=means_t,
                    stds_t=stds_t,
                    loss_cfg=b.BASE_CONFIG["loss"],
                    kd_temperature=float(max(kd_temp, 1e-3)),
                    budget_eps=float(max(stageA_budget_eps, 0.0)),
                    budget_weight_floor=float(max(stageA_budget_weight_floor, 0.0)),
                )
                anchor_val = (
                    float(max(stageA_lambda_kd, 0.0)) * reco_losses["kd"]
                    + float(max(stageA_lambda_emb, 0.0)) * reco_losses["emb"]
                    + float(max(stageA_lambda_tok, 0.0)) * reco_losses["tok"]
                    + float(max(stageA_lambda_phys, 0.0)) * reco_losses["phys"]
                    + float(max(stageA_lambda_budget_hinge, 0.0)) * reco_losses["budget_hinge"]
                )
                loss = loss + float(lambda_reco_anchor) * anchor_val

            loss.backward()
            torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(residual_head.parameters(), 1.0)
            opt.step()

            bs = feat_hlt.size(0)
            tr_loss += float(loss.item()) * bs
            tr_res += float(loss_res.item()) * bs
            tr_kd += float(loss_kd.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            tr_anchor += float(anchor_val.item()) * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_res /= max(n_tr, 1)
        tr_kd /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_anchor /= max(n_tr, 1)

        reconstructor.eval()
        residual_head.eval()

        va_scores = []
        va_labels = []
        va_loss = 0.0
        n_va = 0

        with torch.no_grad():
            for batch in val_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                y = batch["label"].to(device)
                hlt_logit = batch["hlt_logit"].to(device)
                teacher_logit = batch["teacher_logit"].to(device)

                reco_out = reconstructor(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
                feat_corr_t, mask_corr_t = b.build_soft_corrected_view(
                    reco_out,
                    weight_floor=float(corrected_weight_floor),
                    scale_features_by_weight=True,
                    include_flags=False,
                )
                r_hat = residual_head(feat_corr_t, mask_corr_t)
                final_logit = hlt_logit + r_hat
                target_soft = torch.sigmoid(teacher_logit / kd_temp)
                loss_res = F.smooth_l1_loss(r_hat, teacher_logit - hlt_logit)
                loss_kd = (
                    F.binary_cross_entropy_with_logits(final_logit / kd_temp, target_soft, reduction="mean")
                    * (kd_temp * kd_temp)
                )
                loss_cls = F.binary_cross_entropy_with_logits(final_logit, y)
                loss = float(lambda_res) * loss_res + float(lambda_kd) * loss_kd + float(lambda_cls) * loss_cls

                bs = feat_hlt.size(0)
                va_loss += float(loss.item()) * bs
                n_va += bs
                va_scores.append(sigmoid_np(final_logit.detach().cpu().numpy()))
                va_labels.append(y.detach().cpu().numpy().astype(np.float32))

        va_loss /= max(n_va, 1)
        s_val = np.concatenate(va_scores) if va_scores else np.zeros((0,), dtype=np.float64)
        y_val = np.concatenate(va_labels) if va_labels else np.zeros((0,), dtype=np.float32)

        if np.unique(y_val).size > 1 and s_val.size > 0:
            auc_v = float(roc_auc_score(y_val, s_val))
            fpr_v, tpr_v, _ = roc_curve(y_val, s_val)
            fpr50_v = float(b.fpr_at_target_tpr(fpr_v, tpr_v, float(target_tpr)))
        else:
            auc_v, fpr50_v = float("nan"), float("nan")

        if str(select_metric).lower() == "fpr50":
            sel_val = fpr50_v
            is_better = np.isfinite(sel_val) and (sel_val < best_sel)
        else:
            sel_val = auc_v
            is_better = np.isfinite(sel_val) and (sel_val > best_sel)

        if is_better:
            best_sel = float(sel_val)
            best_state_reco = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            best_state_head = {k: v.detach().cpu().clone() for k, v in residual_head.state_dict().items()}
            best_metrics = {
                "best_epoch": int(ep + 1),
                "best_select_metric": str(select_metric).lower(),
                "best_sel": float(best_sel),
                "best_val_auc": float(auc_v),
                "best_val_fpr50": float(fpr50_v),
                "best_val_loss": float(va_loss),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"ResidualJoint ep {ep+1}: train_loss={tr_loss:.5f} (res={tr_res:.5f}, kd={tr_kd:.5f}, cls={tr_cls:.5f}, anchor={tr_anchor:.5f}) | "
                f"val_loss={va_loss:.5f}, val_auc={auc_v:.4f}, val_fpr50={fpr50_v:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping Residual joint at epoch {ep+1}")
            break

    if best_state_reco is not None:
        reconstructor.load_state_dict(best_state_reco)
    if best_state_head is not None:
        residual_head.load_state_dict(best_state_head)

    return reconstructor, residual_head, best_metrics


def parse_alpha_grid(s: str) -> List[float]:
    vals: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if len(vals) == 0:
        vals = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    vals = sorted(set(float(v) for v in vals))
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--n_train_jets", type=int, default=250000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--n_train_split", type=int, default=75000)
    ap.add_argument("--n_val_split", type=int, default=25000)
    ap.add_argument("--n_test_split", type=int, default=150000)
    ap.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_joint_stageA_residual"))
    ap.add_argument("--run_name", type=str, default="reco_teacher_stageA_residual_75k25k150k_seed0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--skip_save_models", action="store_true")
    ap.add_argument("--seed", type=int, default=b.RANDOM_SEED)

    ap.add_argument("--teacher_use_offline_dropout", action="store_true")
    ap.add_argument("--teacher_drop_prob_max", type=float, default=0.5)
    ap.add_argument("--teacher_drop_warmup_epochs", type=int, default=20)
    ap.add_argument("--teacher_drop_mode", type=str, choices=["random", "deterministic_bank"], default="random")
    ap.add_argument("--teacher_drop_num_banks", type=int, default=3)
    ap.add_argument("--teacher_drop_bank_cycle_epochs", type=int, default=1)
    ap.add_argument("--teacher_lambda_drop_cls", type=float, default=1.0)
    ap.add_argument("--teacher_use_consistency", action="store_true")
    ap.add_argument("--teacher_consistency_temp", type=float, default=2.0)
    ap.add_argument("--teacher_lambda_consistency", type=float, default=0.2)

    ap.add_argument("--merge_radius", type=float, default=b.BASE_CONFIG["hlt_effects"]["merge_radius"])
    ap.add_argument("--eff_plateau_barrel", type=float, default=b.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    ap.add_argument("--eff_plateau_endcap", type=float, default=b.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    ap.add_argument("--smear_a", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_a"])
    ap.add_argument("--smear_b", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_b"])
    ap.add_argument("--smear_c", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_c"])

    ap.add_argument("--stageA_epochs", type=int, default=90)
    ap.add_argument("--stageA_patience", type=int, default=18)
    ap.add_argument("--stageA_kd_temp", type=float, default=2.5)
    ap.add_argument("--stageA_lambda_kd", type=float, default=1.0)
    ap.add_argument("--stageA_lambda_emb", type=float, default=1.2)
    ap.add_argument("--stageA_lambda_tok", type=float, default=0.6)
    ap.add_argument("--stageA_lambda_phys", type=float, default=0.2)
    ap.add_argument("--stageA_lambda_budget_hinge", type=float, default=0.03)
    ap.add_argument("--stageA_budget_eps", type=float, default=0.015)
    ap.add_argument("--stageA_budget_weight_floor", type=float, default=1e-4)
    ap.add_argument("--stageA_ratio_count_tolerant", action="store_true")
    ap.add_argument("--stageA_ratio_count_under_lambda", type=float, default=1.0)
    ap.add_argument("--stageA_ratio_count_over_lambda", type=float, default=0.25)
    ap.add_argument("--stageA_ratio_count_over_margin_base", type=float, default=2.0)
    ap.add_argument("--stageA_ratio_count_over_margin_scale", type=float, default=6.0)
    ap.add_argument("--stageA_ratio_count_over_ratio_gamma", type=float, default=0.7)
    ap.add_argument("--stageA_ratio_count_over_lambda_floor", type=float, default=0.05)
    ap.add_argument("--stageA_ratio_count_eps", type=float, default=-1.0)
    ap.add_argument("--stageA_target_tpr", type=float, default=0.50)
    ap.add_argument("--disable_stageA_loss_normalization", action="store_true")
    ap.add_argument("--stageA_loss_norm_ema_decay", type=float, default=0.98)
    ap.add_argument("--stageA_loss_norm_eps", type=float, default=1e-6)
    ap.add_argument("--disable_stageA_stagewise_best_reload", action="store_true")

    ap.add_argument("--stageA_lambda_delta", type=float, default=0.15)
    ap.add_argument("--stageA_delta_tau", type=float, default=0.05)
    ap.add_argument("--stageA_delta_lambda_fp", type=float, default=3.0)

    ap.add_argument("--added_target_scale", type=float, default=0.90)
    ap.add_argument("--reco_weight_threshold", type=float, default=0.03)
    ap.add_argument("--reco_eval_batch_size", type=int, default=256)

    ap.add_argument("--residual_epochs", type=int, default=45)
    ap.add_argument("--residual_patience", type=int, default=12)
    ap.add_argument("--residual_lr", type=float, default=3e-4)
    ap.add_argument("--residual_weight_decay", type=float, default=1e-4)
    ap.add_argument("--residual_warmup_epochs", type=int, default=5)
    ap.add_argument("--residual_lambda_res", type=float, default=1.0)
    ap.add_argument("--residual_lambda_kd", type=float, default=0.2)
    ap.add_argument("--residual_lambda_cls", type=float, default=0.1)
    ap.add_argument("--residual_kd_temp", type=float, default=2.5)
    ap.add_argument("--residual_select_metric", type=str, choices=["fpr50", "auc"], default="fpr50")
    ap.add_argument("--residual_alpha_grid", type=str, default="0.0,0.25,0.5,0.75,1.0,1.25,1.5")

    ap.add_argument("--residual_joint_epochs", type=int, default=0)
    ap.add_argument("--residual_joint_patience", type=int, default=10)
    ap.add_argument("--residual_joint_lr_reco", type=float, default=2e-6)
    ap.add_argument("--residual_joint_lr_head", type=float, default=1e-4)
    ap.add_argument("--residual_joint_weight_decay", type=float, default=1e-4)
    ap.add_argument("--residual_joint_warmup_epochs", type=int, default=4)
    ap.add_argument("--residual_joint_lambda_reco_anchor", type=float, default=0.02)

    ap.add_argument("--report_target_tpr", type=float, default=0.50)
    args = ap.parse_args()

    alpha_grid = parse_alpha_grid(args.residual_alpha_grid)

    b.set_seed(int(args.seed))

    if bool(args.stageA_ratio_count_tolerant):
        ratio_eps = float(args.stageA_ratio_count_eps)
        if ratio_eps < 0.0:
            ratio_eps = float(args.stageA_budget_eps)
        _RATIO_AWARE_COUNT_CFG.update(
            {
                "enabled": True,
                "eps": float(max(ratio_eps, 0.0)),
                "under_lambda": float(max(args.stageA_ratio_count_under_lambda, 0.0)),
                "over_lambda": float(max(args.stageA_ratio_count_over_lambda, 0.0)),
                "over_margin_base": float(max(args.stageA_ratio_count_over_margin_base, 0.0)),
                "over_margin_scale": float(max(args.stageA_ratio_count_over_margin_scale, 0.0)),
                "over_ratio_gamma": float(max(args.stageA_ratio_count_over_ratio_gamma, 0.0)),
                "over_lambda_floor": float(max(args.stageA_ratio_count_over_lambda_floor, 0.0)),
            }
        )
        b._compute_teacher_guided_reco_losses = _compute_teacher_guided_reco_losses_ratio_aware
        sA.b._compute_teacher_guided_reco_losses = _compute_teacher_guided_reco_losses_ratio_aware
        print(
            "[RatioBudget] Enabled ratio-aware asymmetric count tolerance: "
            f"eps={_RATIO_AWARE_COUNT_CFG['eps']:.4f}, "
            f"under={_RATIO_AWARE_COUNT_CFG['under_lambda']:.3f}, "
            f"over={_RATIO_AWARE_COUNT_CFG['over_lambda']:.3f}, "
            f"margin_base={_RATIO_AWARE_COUNT_CFG['over_margin_base']:.3f}, "
            f"margin_scale={_RATIO_AWARE_COUNT_CFG['over_margin_scale']:.3f}, "
            f"gamma={_RATIO_AWARE_COUNT_CFG['over_ratio_gamma']:.3f}, "
            f"over_floor={_RATIO_AWARE_COUNT_CFG['over_lambda_floor']:.3f}"
        )

    cfg = b._deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    cfg["reconstructor_training"]["epochs"] = int(args.stageA_epochs)
    cfg["reconstructor_training"]["patience"] = int(args.stageA_patience)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
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

    max_jets_needed = int(args.offset_jets) + int(args.n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = b.load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}")

    const_raw = all_const_full[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, _budget_truth = b.apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )

    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    rho = b._clamp_target_scale(float(args.added_target_scale))
    budget_merge_true = (rho * true_added_raw).astype(np.float32)
    budget_eff_true = ((1.0 - rho) * true_added_raw).astype(np.float32)

    print(
        f"Non-priv rho split setup: rho={rho:.3f}, "
        f"mean_true_added_raw={float(true_added_raw.mean()):.3f}, "
        f"mean_target_merge={float(budget_merge_true.mean()):.3f}, "
        f"mean_target_eff={float(budget_eff_true.mean()):.3f}"
    )

    print("Computing features...")
    feat_off = b.compute_features(const_off, masks_off)
    feat_hlt = b.compute_features(hlt_const, hlt_mask)

    idx = np.arange(len(labels))
    total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > len(idx):
        raise ValueError(f"Requested split counts exceed available jets: {total_need} > {len(idx)}")

    if total_need < len(idx):
        idx_use, _ = train_test_split(
            idx,
            train_size=total_need,
            random_state=int(args.seed),
            stratify=labels[idx],
        )
    else:
        idx_use = idx

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
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} (custom_counts=True)")

    means, stds = b.get_stats(feat_off, masks_off, train_idx)
    feat_off_std = b.standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = b.standardize(feat_hlt, hlt_mask, means, stds)

    data_setup = {
        "train_path_arg": str(args.train_path),
        "train_files": [str(p.resolve()) for p in train_files],
        "n_train_jets": int(args.n_train_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "seed": int(args.seed),
        "split": {
            "mode": "custom_counts",
            "n_train_split": int(len(train_idx)),
            "n_val_split": int(len(val_idx)),
            "n_test_split": int(len(test_idx)),
        },
        "hlt_effects": cfg["hlt_effects"],
        "variant": "stageA_residual_hlt",
        "rho": float(rho),
        "mean_true_added_raw": float(true_added_raw.mean()),
        "mean_target_merge": float(budget_merge_true.mean()),
        "mean_target_eff": float(budget_eff_true.mean()),
        "alpha_grid": [float(x) for x in alpha_grid],
    }
    with open(save_root / "data_setup.json", "w", encoding="utf-8") as f:
        json.dump(data_setup, f, indent=2)
    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
    )

    print("\n" + "=" * 70)
    print("STEP 1: TEACHER + BASELINE")
    print("=" * 70)
    BS = int(cfg["training"]["batch_size"])

    ds_train_off = b.JetDataset(feat_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    ds_val_off = b.JetDataset(feat_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    ds_test_off = b.JetDataset(feat_off_std[test_idx], masks_off[test_idx], labels[test_idx])
    dl_train_off = DataLoader(ds_train_off, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_off = DataLoader(ds_val_off, batch_size=BS, shuffle=False)
    dl_test_off = DataLoader(ds_test_off, batch_size=BS, shuffle=False)

    teacher = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    if bool(args.teacher_use_offline_dropout):
        print("Teacher mode: offline-dropout")
        teacher = train_single_view_teacher_with_offline_dropout(
            model=teacher,
            feat_off=feat_off_std,
            mask_off=masks_off,
            mask_hlt=hlt_mask,
            labels=labels.astype(np.float32),
            train_idx=train_idx,
            val_loader_full=dl_val_off,
            device=device,
            train_cfg=cfg["training"],
            target_tpr=float(args.stageA_target_tpr),
            drop_prob_max=float(args.teacher_drop_prob_max),
            drop_warmup_epochs=int(args.teacher_drop_warmup_epochs),
            lambda_drop_cls=float(args.teacher_lambda_drop_cls),
            use_consistency=bool(args.teacher_use_consistency),
            consistency_temp=float(args.teacher_consistency_temp),
            lambda_consistency=float(args.teacher_lambda_consistency),
            drop_mode=str(args.teacher_drop_mode),
            drop_num_banks=int(args.teacher_drop_num_banks),
            drop_bank_cycle_epochs=int(args.teacher_drop_bank_cycle_epochs),
            seed=int(args.seed),
            name="TeacherDrop",
        )
    else:
        teacher = b.train_single_view_classifier_auc(
            teacher, dl_train_off, dl_val_off, device, cfg["training"], name="Teacher"
        )
    auc_teacher, preds_teacher_test_prob, labs_test_teacher = b.eval_classifier(teacher, dl_test_off, device)
    auc_teacher_val, preds_teacher_val_prob, labs_val_teacher = b.eval_classifier(teacher, dl_val_off, device)

    ds_train_hlt = b.JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx])
    ds_val_hlt = b.JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    ds_test_hlt = b.JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_train_hlt = DataLoader(ds_train_hlt, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=BS, shuffle=False)
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=BS, shuffle=False)

    baseline = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = b.train_single_view_classifier_auc(
        baseline, dl_train_hlt, dl_val_hlt, device, cfg["training"], name="Baseline"
    )
    auc_baseline, preds_baseline_test_prob, labs_test_hlt = b.eval_classifier(baseline, dl_test_hlt, device)
    auc_baseline_val, preds_baseline_val_prob, labs_val_baseline = b.eval_classifier(baseline, dl_val_hlt, device)

    assert np.array_equal(labs_val_teacher.astype(np.float32), labs_val_baseline.astype(np.float32))
    assert np.array_equal(labs_test_teacher.astype(np.float32), labs_test_hlt.astype(np.float32))

    hlt_thr_prob, hlt_thr_tpr, hlt_thr_fpr = threshold_at_target_tpr(
        labs_val_baseline.astype(np.float32),
        preds_baseline_val_prob.astype(np.float64),
        float(args.stageA_target_tpr),
    )
    print(
        f"StageA delta HLT reference @TPR={float(args.stageA_target_tpr):.2f}: "
        f"threshold_prob={hlt_thr_prob:.6f}, val_tpr={hlt_thr_tpr:.6f}, val_fpr={hlt_thr_fpr:.6f}"
    )

    print("\n" + "=" * 70)
    print("STEP 2: STAGE A (TEACHER-GUIDED RECONSTRUCTOR)")
    print("=" * 70)
    ds_train_reco = b.StageAReconstructionDataset(
        feat_hlt_std[train_idx], hlt_mask[train_idx], hlt_const[train_idx],
        const_off[train_idx], masks_off[train_idx], labels[train_idx],
        budget_merge_true[train_idx], budget_eff_true[train_idx],
    )
    ds_val_reco = b.StageAReconstructionDataset(
        feat_hlt_std[val_idx], hlt_mask[val_idx], hlt_const[val_idx],
        const_off[val_idx], masks_off[val_idx], labels[val_idx],
        budget_merge_true[val_idx], budget_eff_true[val_idx],
    )
    dl_train_reco = DataLoader(
        ds_train_reco,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_reco = DataLoader(
        ds_val_reco,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    reconstructor = b.OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    b.BASE_CONFIG["loss"] = cfg["loss"]
    reconstructor, reco_val_metrics = sA.train_reconstructor_teacher_guided_stagewise_delta(
        model=reconstructor,
        train_loader=dl_train_reco,
        val_loader=dl_val_reco,
        device=device,
        train_cfg=cfg["reconstructor_training"],
        loss_cfg=cfg["loss"],
        teacher_model=teacher,
        hlt_model=baseline,
        hlt_threshold_prob=float(hlt_thr_prob),
        feat_means=means.astype(np.float32),
        feat_stds=stds.astype(np.float32),
        kd_temperature=float(args.stageA_kd_temp),
        lambda_kd=float(args.stageA_lambda_kd),
        lambda_emb=float(args.stageA_lambda_emb),
        lambda_tok=float(args.stageA_lambda_tok),
        lambda_phys=float(args.stageA_lambda_phys),
        lambda_budget_hinge=float(args.stageA_lambda_budget_hinge),
        lambda_delta=float(args.stageA_lambda_delta),
        delta_tau=float(args.stageA_delta_tau),
        delta_lambda_fp=float(args.stageA_delta_lambda_fp),
        budget_eps=float(args.stageA_budget_eps),
        budget_weight_floor=float(args.stageA_budget_weight_floor),
        target_tpr_for_fpr=float(args.stageA_target_tpr),
        normalize_loss_terms=not bool(args.disable_stageA_loss_normalization),
        loss_norm_ema_decay=float(args.stageA_loss_norm_ema_decay),
        loss_norm_eps=float(args.stageA_loss_norm_eps),
        reload_best_at_stage_transition=not bool(args.disable_stageA_stagewise_best_reload),
    )

    print("\n" + "=" * 70)
    print("STEP 3: STAGE A SOFT-RECO EVAL")
    print("=" * 70)
    auc_reco_teacher_val, preds_reco_teacher_val_prob, labs_reco_val, fpr50_reco_teacher_val = sA.eval_teacher_on_soft_reco_split(
        reconstructor=reconstructor,
        teacher=teacher,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        labels=labels,
        split_idx=val_idx,
        means=means,
        stds=stds,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        weight_floor=float(args.reco_weight_threshold),
        target_tpr=float(args.report_target_tpr),
    )
    auc_reco_teacher_test, preds_reco_teacher_test_prob, labs_reco_test, fpr50_reco_teacher_test = sA.eval_teacher_on_soft_reco_split(
        reconstructor=reconstructor,
        teacher=teacher,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        labels=labels,
        split_idx=test_idx,
        means=means,
        stds=stds,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        weight_floor=float(args.reco_weight_threshold),
        target_tpr=float(args.report_target_tpr),
    )

    assert np.array_equal(labs_reco_val.astype(np.float32), labs_val_teacher.astype(np.float32))
    assert np.array_equal(labs_reco_test.astype(np.float32), labs_test_teacher.astype(np.float32))

    print("\n" + "=" * 70)
    print("STEP 4: RESIDUAL HEAD (FROZEN RECONSTRUCTOR)")
    print("=" * 70)

    # Precompute logits for residual target r* = teacher_logit - hlt_logit.
    all_idx = np.arange(len(labels), dtype=np.int64)
    hlt_logits_all = predict_logits_single_view(
        model=baseline,
        feat=feat_hlt_std,
        mask=hlt_mask,
        split_idx=all_idx,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
    )
    teacher_logits_all = predict_logits_single_view(
        model=teacher,
        feat=feat_off_std,
        mask=masks_off,
        split_idx=all_idx,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
    )

    # Fixed corrected view from Stage-A reconstructor.
    feat_corr_all, mask_corr_all = b.build_corrected_view_numpy(
        reconstructor=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=False,
    )

    ds_train_res = ResidualDataset(
        feat_corr_all[train_idx],
        mask_corr_all[train_idx],
        hlt_logits_all[train_idx],
        teacher_logits_all[train_idx],
        labels[train_idx],
    )
    ds_val_res = ResidualDataset(
        feat_corr_all[val_idx],
        mask_corr_all[val_idx],
        hlt_logits_all[val_idx],
        teacher_logits_all[val_idx],
        labels[val_idx],
    )

    dl_train_res = DataLoader(ds_train_res, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_res = DataLoader(ds_val_res, batch_size=BS, shuffle=False)

    residual_head = ResidualHead(input_dim=int(feat_corr_all.shape[-1]), model_cfg=cfg["model"]).to(device)
    residual_head, residual_frozen_train_metrics = train_residual_head(
        model=residual_head,
        train_loader=dl_train_res,
        val_loader=dl_val_res,
        device=device,
        epochs=int(args.residual_epochs),
        patience=int(args.residual_patience),
        lr=float(args.residual_lr),
        weight_decay=float(args.residual_weight_decay),
        warmup_epochs=int(args.residual_warmup_epochs),
        lambda_res=float(args.residual_lambda_res),
        lambda_kd=float(args.residual_lambda_kd),
        lambda_cls=float(args.residual_lambda_cls),
        kd_temp=float(args.residual_kd_temp),
        target_tpr=float(args.report_target_tpr),
        select_metric=str(args.residual_select_metric),
    )

    rhat_val_frozen = predict_residual_head(
        model=residual_head,
        feat_corr=feat_corr_all,
        mask_corr=mask_corr_all,
        split_idx=val_idx,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
    )
    rhat_test_frozen = predict_residual_head(
        model=residual_head,
        feat_corr=feat_corr_all,
        mask_corr=mask_corr_all,
        split_idx=test_idx,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
    )

    alpha_eval_frozen = select_alpha_on_val_and_eval_test(
        labels_val=labels[val_idx].astype(np.float32),
        hlt_logits_val=hlt_logits_all[val_idx],
        rhat_val=rhat_val_frozen,
        labels_test=labels[test_idx].astype(np.float32),
        hlt_logits_test=hlt_logits_all[test_idx],
        rhat_test=rhat_test_frozen,
        alpha_grid=alpha_grid,
        target_tpr=float(args.report_target_tpr),
    )

    preds_residual_frozen_val_prob = sigmoid_np(
        hlt_logits_all[val_idx] + float(alpha_eval_frozen["selection"]["alpha"]) * rhat_val_frozen
    )
    preds_residual_frozen_test_prob = sigmoid_np(
        hlt_logits_all[test_idx] + float(alpha_eval_frozen["selection"]["alpha"]) * rhat_test_frozen
    )

    # Optional light joint stage.
    residual_joint_train_metrics: Dict[str, float] | None = None
    alpha_eval_joint: Dict[str, object] | None = None
    preds_residual_joint_val_prob = np.zeros(0, dtype=np.float64)
    preds_residual_joint_test_prob = np.zeros(0, dtype=np.float64)

    if int(args.residual_joint_epochs) > 0:
        print("\n" + "=" * 70)
        print("STEP 5: OPTIONAL LIGHT JOINT FINETUNE (RECO + RESIDUAL)")
        print("=" * 70)

        ds_train_joint = ResidualJointDataset(
            feat_hlt=feat_hlt_std[train_idx],
            mask_hlt=hlt_mask[train_idx],
            const_hlt=hlt_const[train_idx],
            const_off=const_off[train_idx],
            mask_off=masks_off[train_idx],
            labels=labels[train_idx],
            budget_merge_true=budget_merge_true[train_idx],
            budget_eff_true=budget_eff_true[train_idx],
            hlt_logit=hlt_logits_all[train_idx],
            teacher_logit=teacher_logits_all[train_idx],
        )
        ds_val_joint = ResidualJointDataset(
            feat_hlt=feat_hlt_std[val_idx],
            mask_hlt=hlt_mask[val_idx],
            const_hlt=hlt_const[val_idx],
            const_off=const_off[val_idx],
            mask_off=masks_off[val_idx],
            labels=labels[val_idx],
            budget_merge_true=budget_merge_true[val_idx],
            budget_eff_true=budget_eff_true[val_idx],
            hlt_logit=hlt_logits_all[val_idx],
            teacher_logit=teacher_logits_all[val_idx],
        )

        dl_train_joint = DataLoader(
            ds_train_joint,
            batch_size=int(cfg["reconstructor_training"]["batch_size"]),
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        dl_val_joint = DataLoader(
            ds_val_joint,
            batch_size=int(cfg["reconstructor_training"]["batch_size"]),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        reconstructor, residual_head, residual_joint_train_metrics = train_residual_joint(
            reconstructor=reconstructor,
            residual_head=residual_head,
            train_loader=dl_train_joint,
            val_loader=dl_val_joint,
            teacher_model=teacher,
            device=device,
            means=means,
            stds=stds,
            epochs=int(args.residual_joint_epochs),
            patience=int(args.residual_joint_patience),
            lr_reco=float(args.residual_joint_lr_reco),
            lr_head=float(args.residual_joint_lr_head),
            weight_decay=float(args.residual_joint_weight_decay),
            warmup_epochs=int(args.residual_joint_warmup_epochs),
            corrected_weight_floor=float(args.reco_weight_threshold),
            lambda_res=float(args.residual_lambda_res),
            lambda_kd=float(args.residual_lambda_kd),
            lambda_cls=float(args.residual_lambda_cls),
            kd_temp=float(args.residual_kd_temp),
            lambda_reco_anchor=float(args.residual_joint_lambda_reco_anchor),
            stageA_lambda_kd=float(args.stageA_lambda_kd),
            stageA_lambda_emb=float(args.stageA_lambda_emb),
            stageA_lambda_tok=float(args.stageA_lambda_tok),
            stageA_lambda_phys=float(args.stageA_lambda_phys),
            stageA_lambda_budget_hinge=float(args.stageA_lambda_budget_hinge),
            stageA_budget_eps=float(args.stageA_budget_eps),
            stageA_budget_weight_floor=float(args.stageA_budget_weight_floor),
            target_tpr=float(args.report_target_tpr),
            select_metric=str(args.residual_select_metric),
        )

        rhat_val_joint = predict_residual_head_with_reco(
            reconstructor=reconstructor,
            residual_head=residual_head,
            feat_hlt=feat_hlt_std,
            mask_hlt=hlt_mask,
            const_hlt=hlt_const,
            split_idx=val_idx,
            device=device,
            batch_size=int(args.reco_eval_batch_size),
            corrected_weight_floor=float(args.reco_weight_threshold),
        )
        rhat_test_joint = predict_residual_head_with_reco(
            reconstructor=reconstructor,
            residual_head=residual_head,
            feat_hlt=feat_hlt_std,
            mask_hlt=hlt_mask,
            const_hlt=hlt_const,
            split_idx=test_idx,
            device=device,
            batch_size=int(args.reco_eval_batch_size),
            corrected_weight_floor=float(args.reco_weight_threshold),
        )

        alpha_eval_joint = select_alpha_on_val_and_eval_test(
            labels_val=labels[val_idx].astype(np.float32),
            hlt_logits_val=hlt_logits_all[val_idx],
            rhat_val=rhat_val_joint,
            labels_test=labels[test_idx].astype(np.float32),
            hlt_logits_test=hlt_logits_all[test_idx],
            rhat_test=rhat_test_joint,
            alpha_grid=alpha_grid,
            target_tpr=float(args.report_target_tpr),
        )

        preds_residual_joint_val_prob = sigmoid_np(
            hlt_logits_all[val_idx] + float(alpha_eval_joint["selection"]["alpha"]) * rhat_val_joint
        )
        preds_residual_joint_test_prob = sigmoid_np(
            hlt_logits_all[test_idx] + float(alpha_eval_joint["selection"]["alpha"]) * rhat_test_joint
        )

    print("\n" + "=" * 70)
    print("FINAL STAGEA+RESIDUAL EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"RecoTeacher Soft AUC (val/test): {auc_reco_teacher_val:.4f} / {auc_reco_teacher_test:.4f}")
    print(
        f"Residual frozen (VAL-selected alpha={alpha_eval_frozen['selection']['alpha']:.3f}) -> "
        f"AUC_test={alpha_eval_frozen['test_eval']['auc']:.4f}, "
        f"FPR_test@thr_from_val={alpha_eval_frozen['test_eval']['fpr']:.6f}, "
        f"TPR_test@thr_from_val={alpha_eval_frozen['test_eval']['tpr']:.6f}, "
        f"FPR_test@TPR={float(args.report_target_tpr):.2f} exact={alpha_eval_frozen['test_eval']['fpr_at_target_tpr_exact']:.6f}"
    )
    if alpha_eval_joint is not None:
        print(
            f"Residual JOINT (VAL-selected alpha={alpha_eval_joint['selection']['alpha']:.3f}) -> "
            f"AUC_test={alpha_eval_joint['test_eval']['auc']:.4f}, "
            f"FPR_test@thr_from_val={alpha_eval_joint['test_eval']['fpr']:.6f}, "
            f"TPR_test@thr_from_val={alpha_eval_joint['test_eval']['tpr']:.6f}, "
            f"FPR_test@TPR={float(args.report_target_tpr):.2f} exact={alpha_eval_joint['test_eval']['fpr_at_target_tpr_exact']:.6f}"
        )

    np.savez_compressed(
        save_root / "stageA_residual_scores.npz",
        labels_val=labels[val_idx].astype(np.float32),
        labels_test=labels[test_idx].astype(np.float32),
        preds_teacher_val=preds_teacher_val_prob.astype(np.float64),
        preds_teacher_test=preds_teacher_test_prob.astype(np.float64),
        preds_hlt_val=preds_baseline_val_prob.astype(np.float64),
        preds_hlt_test=preds_baseline_test_prob.astype(np.float64),
        preds_reco_teacher_val=preds_reco_teacher_val_prob.astype(np.float64),
        preds_reco_teacher_test=preds_reco_teacher_test_prob.astype(np.float64),
        preds_residual_frozen_val=preds_residual_frozen_val_prob.astype(np.float64),
        preds_residual_frozen_test=preds_residual_frozen_test_prob.astype(np.float64),
        preds_residual_joint_val=preds_residual_joint_val_prob.astype(np.float64),
        preds_residual_joint_test=preds_residual_joint_test_prob.astype(np.float64),
        alpha_residual_frozen=float(alpha_eval_frozen["selection"]["alpha"]),
        alpha_residual_joint=float(alpha_eval_joint["selection"]["alpha"]) if alpha_eval_joint is not None else float("nan"),
        target_tpr=float(args.report_target_tpr),
    )

    with open(save_root / "stageA_residual_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": "stageA_reco_plus_residual_hlt",
                "rho": float(rho),
                "alpha_grid": [float(x) for x in alpha_grid],
                "stageA_reconstructor": reco_val_metrics,
                "stageA_ratio_count_budget": {
                    "enabled": bool(args.stageA_ratio_count_tolerant),
                    "eps": float(_RATIO_AWARE_COUNT_CFG["eps"]),
                    "under_lambda": float(_RATIO_AWARE_COUNT_CFG["under_lambda"]),
                    "over_lambda": float(_RATIO_AWARE_COUNT_CFG["over_lambda"]),
                    "over_margin_base": float(_RATIO_AWARE_COUNT_CFG["over_margin_base"]),
                    "over_margin_scale": float(_RATIO_AWARE_COUNT_CFG["over_margin_scale"]),
                    "over_ratio_gamma": float(_RATIO_AWARE_COUNT_CFG["over_ratio_gamma"]),
                    "over_lambda_floor": float(_RATIO_AWARE_COUNT_CFG["over_lambda_floor"]),
                },
                "teacher": {
                    "auc_val": float(auc_teacher_val),
                    "auc_test": float(auc_teacher),
                    "offline_dropout": {
                        "enabled": bool(args.teacher_use_offline_dropout),
                        "drop_prob_max": float(args.teacher_drop_prob_max),
                        "drop_warmup_epochs": int(args.teacher_drop_warmup_epochs),
                        "drop_mode": str(args.teacher_drop_mode),
                        "drop_num_banks": int(args.teacher_drop_num_banks),
                        "drop_bank_cycle_epochs": int(args.teacher_drop_bank_cycle_epochs),
                        "lambda_drop_cls": float(args.teacher_lambda_drop_cls),
                        "use_consistency": bool(args.teacher_use_consistency),
                        "consistency_temp": float(args.teacher_consistency_temp),
                        "lambda_consistency": float(args.teacher_lambda_consistency),
                    },
                },
                "hlt": {
                    "auc_val": float(auc_baseline_val),
                    "auc_test": float(auc_baseline),
                    "delta_ref_threshold_prob": float(hlt_thr_prob),
                    "delta_ref_val_tpr": float(hlt_thr_tpr),
                    "delta_ref_val_fpr": float(hlt_thr_fpr),
                },
                "reco_teacher_soft": {
                    "auc_val": float(auc_reco_teacher_val),
                    "auc_test": float(auc_reco_teacher_test),
                    "fpr50_val": float(fpr50_reco_teacher_val),
                    "fpr50_test": float(fpr50_reco_teacher_test),
                },
                "residual_frozen_train": residual_frozen_train_metrics,
                "residual_frozen_eval": alpha_eval_frozen,
                "residual_joint_train": residual_joint_train_metrics,
                "residual_joint_eval": alpha_eval_joint,
            },
            f,
            indent=2,
        )

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg["hlt_effects"], "stats": hlt_stats}, f, indent=2)

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": float(auc_teacher)}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": float(auc_baseline)}, save_root / "baseline.pt")
        torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor_stageA.pt")
        torch.save({"model": residual_head.state_dict(), "metrics": residual_frozen_train_metrics}, save_root / "residual_head.pt")

    print(f"\nSaved StageA+Residual results to: {save_root}")


if __name__ == "__main__":
    main()

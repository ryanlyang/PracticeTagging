#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual-reconstructor + dualview pipeline (m9 offdrop variant, no residual head).

Flow:
1) Train Teacher + Baseline on offline/HLT (teacher may use deterministic offline-dropout).
2) Train Reco-A with teacher-guided Stage-A curriculum (s01/s09-style Stage-A stack).
3) Train Reco-B with m2-style reconstruction losses on deterministic masked-offline targets,
   using ratio-aware asymmetric count budget tolerance.
4) Train dualview tagger on [Reco-A corrected view, Reco-B corrected view] with frozen reconstructors.
5) Optional light joint finetune (unfreeze Reco-A, Reco-B, dualview).

Outputs:
- dualreco_dualview_scores.npz
- dualreco_dualview_metrics.json
- checkpoints for teacher/baseline/recoA/recoB/dual_frozen/dual_joint
"""

from __future__ import annotations

import argparse
import copy
import json
import random
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
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as m2mod
import reco_teacher_stageA_only_delta_curriculum as sA
from unmerge_correct_hlt import DualViewJetDataset


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auc_and_fpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> Tuple[float, float]:
    labels = labels.astype(np.float32)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr_t = float(b.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))
    return auc, fpr_t


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


def build_concat_constituents(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    max_concat_constits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    const_cat = np.concatenate([const_off, const_hlt], axis=1)
    mask_cat = np.concatenate([mask_off, mask_hlt], axis=1)

    full_l = int(const_cat.shape[1])
    out_l = int(max_concat_constits)
    if out_l <= 0:
        out_l = full_l

    if out_l < full_l:
        const_cat = const_cat[:, :out_l, :]
        mask_cat = mask_cat[:, :out_l]
    elif out_l > full_l:
        n = int(const_cat.shape[0])
        pad_const = np.zeros((n, out_l - full_l, const_cat.shape[2]), dtype=const_cat.dtype)
        pad_mask = np.zeros((n, out_l - full_l), dtype=bool)
        const_cat = np.concatenate([const_cat, pad_const], axis=1)
        mask_cat = np.concatenate([mask_cat, pad_mask], axis=1)

    const_cat = const_cat.copy()
    const_cat[~mask_cat] = 0.0
    return const_cat.astype(np.float32), mask_cat.astype(bool)


def _norm_feature_ablation_mode(mode: str) -> str:
    m = str(mode).lower()
    if m in {"none", "no_angle", "no_scale", "core_shape"}:
        return m
    raise ValueError(f"Unknown feature ablation mode: {mode}")


def apply_feature_ablation_to_corrected_np(feat: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    m = _norm_feature_ablation_mode(mode)
    if m == "none":
        return feat
    out = feat.copy()
    out[:, :, :7] = sA.apply_teacher_feature_ablation_np(out[:, :, :7], mask, m)
    out[~mask] = 0.0
    return out.astype(np.float32)


def apply_feature_ablation_to_corrected_torch(feat: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    m = _norm_feature_ablation_mode(mode)
    if m == "none":
        return feat
    out = feat.clone()
    out[:, :, :7] = sA.apply_teacher_feature_ablation_torch(out[:, :, :7], mask, m)
    out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
    return out


def _reco_b_ablation_profile(mode: str) -> Dict[str, float]:
    m = _norm_feature_ablation_mode(mode)
    if m == "none":
        return {
            "c_logpt": 1.00,
            "c_eta": 0.60,
            "c_phi": 0.60,
            "c_logE": 0.25,
            "w_phys_vec": 1.0,
            "w_phys_E": 1.0,
            "w_pt_ratio": 1.0,
            "w_e_ratio": 1.0,
            "w_local": 1.0,
        }
    if m == "no_angle":
        return {
            "c_logpt": 1.00,
            "c_eta": 0.00,
            "c_phi": 0.00,
            "c_logE": 0.25,
            "w_phys_vec": 0.0,
            "w_phys_E": 1.0,
            "w_pt_ratio": 1.0,
            "w_e_ratio": 1.0,
            "w_local": 0.0,
        }
    if m == "no_scale":
        return {
            "c_logpt": 0.00,
            "c_eta": 0.60,
            "c_phi": 0.60,
            "c_logE": 0.00,
            "w_phys_vec": 0.0,
            "w_phys_E": 0.0,
            "w_pt_ratio": 0.0,
            "w_e_ratio": 0.0,
            "w_local": 1.0,
        }
    # core_shape: keep geometry + relative-shape tendency; de-emphasize absolute energy-scale terms.
    return {
        "c_logpt": 0.25,
        "c_eta": 0.60,
        "c_phi": 0.60,
        "c_logE": 0.00,
        "w_phys_vec": 0.0,
        "w_phys_E": 0.0,
        "w_pt_ratio": 0.25,
        "w_e_ratio": 0.0,
        "w_local": 1.0,
    }


def compute_reco_b_losses(
    out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    loss_cfg: Dict,
    feature_ablation_mode: str = "none",
    target_token_weights: torch.Tensor | None = None,
    added_min_frac: float = 0.0,
    added_min_hinge_lambda: float = 0.0,
) -> Dict[str, torch.Tensor]:
    m = _norm_feature_ablation_mode(feature_ablation_mode)
    if (
        m == "none"
        and target_token_weights is None
        and float(added_min_frac) <= 0.0
        and float(added_min_hinge_lambda) <= 0.0
    ):
        return m2mod.compute_reconstruction_losses(
            out,
            const_hlt,
            mask_hlt,
            const_off,
            mask_off,
            budget_merge_true,
            budget_eff_true,
            loss_cfg,
        )

    prof = _reco_b_ablation_profile(m)
    eps = 1e-8

    pred = out["cand_tokens"]
    w = out["cand_weights"].clamp(0.0, 1.0)

    p_pt = pred[:, :, 0].clamp(min=eps).unsqueeze(2)
    t_pt = const_off[:, :, 0].clamp(min=eps).unsqueeze(1)
    p_eta = pred[:, :, 1].unsqueeze(2)
    t_eta = const_off[:, :, 1].unsqueeze(1)
    p_phi = pred[:, :, 2].unsqueeze(2)
    t_phi = const_off[:, :, 2].unsqueeze(1)
    p_E = pred[:, :, 3].clamp(min=eps).unsqueeze(2)
    t_E = const_off[:, :, 3].clamp(min=eps).unsqueeze(1)

    d_logpt = torch.abs(torch.log(p_pt) - torch.log(t_pt))
    d_eta = torch.abs(p_eta - t_eta)
    d_phi = torch.abs(torch.atan2(torch.sin(p_phi - t_phi), torch.cos(p_phi - t_phi)))
    d_logE = torch.abs(torch.log(p_E) - torch.log(t_E))

    cost = (
        float(prof["c_logpt"]) * d_logpt
        + float(prof["c_eta"]) * d_eta
        + float(prof["c_phi"]) * d_phi
        + float(prof["c_logE"]) * d_logE
    )

    valid_tgt = mask_off.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))
    pred_to_tgt = cost.min(dim=2).values
    loss_pred_to_tgt = (w * pred_to_tgt).sum(dim=1) / (w.sum(dim=1) + eps)

    penalty = float(loss_cfg["unselected_penalty"]) * (1.0 - w).unsqueeze(2)
    tgt_to_pred = (cost + penalty).min(dim=1).values
    if target_token_weights is None:
        tgt_w = mask_off.float()
    else:
        tgt_w = target_token_weights.float() * mask_off.float()
    loss_tgt_to_pred = (tgt_to_pred * tgt_w).sum(dim=1) / (tgt_w.sum(dim=1) + eps)
    loss_set = (loss_pred_to_tgt + loss_tgt_to_pred).mean()

    pt_pred = pred[:, :, 0]
    eta_pred = pred[:, :, 1]
    phi_pred = pred[:, :, 2]
    E_pred = pred[:, :, 3]
    px_pred = (w * pt_pred * torch.cos(phi_pred)).sum(dim=1)
    py_pred = (w * pt_pred * torch.sin(phi_pred)).sum(dim=1)
    pz_pred = (w * pt_pred * torch.sinh(eta_pred)).sum(dim=1)
    Es_pred = (w * E_pred).sum(dim=1)

    wt = mask_off.float()
    pt_true = const_off[:, :, 0]
    eta_true = const_off[:, :, 1]
    phi_true = const_off[:, :, 2]
    E_true = const_off[:, :, 3]
    px_true = (wt * pt_true * torch.cos(phi_true)).sum(dim=1)
    py_true = (wt * pt_true * torch.sin(phi_true)).sum(dim=1)
    pz_true = (wt * pt_true * torch.sinh(eta_true)).sum(dim=1)
    Es_true = (wt * E_true).sum(dim=1)

    norm_vec = (px_true.abs() + py_true.abs() + pz_true.abs() + 1.0)
    loss_phys_vec = ((px_pred - px_true).abs() + (py_pred - py_true).abs() + (pz_pred - pz_true).abs()) / norm_vec
    loss_phys_e = (Es_pred - Es_true).abs() / (Es_true.abs() + 1.0)
    loss_phys = (float(prof["w_phys_vec"]) * loss_phys_vec + float(prof["w_phys_E"]) * loss_phys_e).mean()

    pred_pt = torch.sqrt(px_pred.pow(2) + py_pred.pow(2) + eps)
    true_pt = torch.sqrt(px_true.pow(2) + py_true.pow(2) + eps)
    loss_pt_ratio = F.smooth_l1_loss(pred_pt / (true_pt + eps), torch.ones_like(true_pt))
    loss_e_ratio = F.smooth_l1_loss(Es_pred / (Es_true + eps), torch.ones_like(Es_true))

    true_count = mask_off.float().sum(dim=1)
    hlt_count = mask_hlt.float().sum(dim=1)
    true_added = (true_count - hlt_count).clamp(min=0.0)
    pred_count = w.sum(dim=1)
    pred_added = out["child_weight"].sum(dim=1) + out["gen_weight"].sum(dim=1)
    pred_added_merge = out["child_weight"].sum(dim=1)
    pred_added_eff = out["gen_weight"].sum(dim=1)
    budget_total = out["budget_total"]
    budget_merge = out["budget_merge"]
    budget_eff = out["budget_eff"]
    budget_true_total = budget_merge_true + budget_eff_true

    loss_budget = (
        F.smooth_l1_loss(pred_count, true_count)
        + F.smooth_l1_loss(budget_total, true_count)
        + F.smooth_l1_loss(pred_added, true_added)
        + F.smooth_l1_loss(budget_merge + budget_eff, true_added)
        + F.smooth_l1_loss(budget_merge, budget_merge_true)
        + F.smooth_l1_loss(budget_eff, budget_eff_true)
        + F.smooth_l1_loss(pred_added_merge, budget_merge_true)
        + F.smooth_l1_loss(pred_added_eff, budget_eff_true)
        + F.smooth_l1_loss(budget_merge + budget_eff, budget_true_total)
    )

    loss_sparse = out["child_weight"].mean() + out["gen_weight"].mean()

    split_delta = out["split_delta"]
    split_step = torch.sqrt(split_delta[..., 1].pow(2) + split_delta[..., 2].pow(2) + 1e-8)
    split_w = out["child_weight"].reshape(split_step.shape)
    loss_local_split = (split_w * split_step).sum() / (split_w.sum() + eps)

    gen_tokens = out["gen_tokens"]
    gen_w = out["gen_weight"]
    h_eta = const_hlt[:, :, 1]
    h_phi = const_hlt[:, :, 2]
    g_eta = gen_tokens[:, :, 1].unsqueeze(2)
    g_phi = gen_tokens[:, :, 2].unsqueeze(2)
    d_eta_l = g_eta - h_eta.unsqueeze(1)
    d_phi_l = torch.atan2(torch.sin(g_phi - h_phi.unsqueeze(1)), torch.cos(g_phi - h_phi.unsqueeze(1)))
    dR = torch.sqrt(d_eta_l.pow(2) + d_phi_l.pow(2) + 1e-8)
    dR = torch.where(mask_hlt.unsqueeze(1), dR, torch.full_like(dR, 1e4))
    nearest = dR.min(dim=2).values
    excess = F.relu(nearest - float(loss_cfg["gen_local_radius"]))
    loss_local_gen = (gen_w * excess).sum() / (gen_w.sum() + eps)
    loss_local = (loss_local_split + loss_local_gen) * float(prof["w_local"])

    if float(added_min_hinge_lambda) > 0.0 and float(added_min_frac) > 0.0:
        min_added = float(added_min_frac) * true_added
        loss_added_min = F.relu(min_added - pred_added).mean()
    else:
        loss_added_min = torch.zeros((), device=pred.device)

    total = (
        float(loss_cfg["w_set"]) * loss_set
        + float(loss_cfg["w_phys"]) * loss_phys
        + float(loss_cfg["w_pt_ratio"]) * float(prof["w_pt_ratio"]) * loss_pt_ratio
        + float(loss_cfg["w_e_ratio"]) * float(prof["w_e_ratio"]) * loss_e_ratio
        + float(loss_cfg["w_budget"]) * loss_budget
        + float(loss_cfg["w_sparse"]) * loss_sparse
        + float(loss_cfg["w_local"]) * loss_local
        + float(added_min_hinge_lambda) * loss_added_min
    )

    return {
        "total": total,
        "set": loss_set,
        "phys": loss_phys,
        "pt_ratio": loss_pt_ratio,
        "e_ratio": loss_e_ratio,
        "budget": loss_budget,
        "sparse": loss_sparse,
        "local": loss_local,
        "added_min_hinge": loss_added_min,
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
            "mask_full": mask_full,
            "feat_drop": feat_drop,
            "mask_drop": mask_drop,
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
    train_cfg: Dict,
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
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    patience = int(train_cfg["patience"])
    lr = float(train_cfg["lr"])
    wd = float(train_cfg["weight_decay"])
    warmup = int(train_cfg["warmup_epochs"])

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

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = b.get_scheduler(opt, warmup, epochs)

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
        n_tr = 0

        for batch in train_loader:
            feat_full = batch["feat_full"].to(device)
            mask_full = batch["mask_full"].to(device)
            feat_drop = batch["feat_drop"].to(device)
            mask_drop = batch["mask_drop"].to(device)
            y = batch["label"].to(device)

            opt.zero_grad()
            logits_full = model(feat_full, mask_full).squeeze(1)
            logits_drop = model(feat_drop, mask_drop).squeeze(1)

            loss_full = F.binary_cross_entropy_with_logits(logits_full, y)
            loss_drop = F.binary_cross_entropy_with_logits(logits_drop, y)

            if use_consistency:
                p_full = torch.sigmoid(logits_full / t)
                p_drop = torch.sigmoid(logits_drop / t)
                loss_cons = F.smooth_l1_loss(p_drop, p_full.detach())
            else:
                loss_cons = torch.zeros((), device=device)

            loss = loss_full + float(lambda_drop_cls) * loss_drop + float(lambda_consistency) * loss_cons
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            running_loss += float(loss.item()) * bs
            running_l_full += float(loss_full.item()) * bs
            running_l_drop += float(loss_drop.item()) * bs
            running_l_cons += float(loss_cons.item()) * bs
            n_tr += bs

        sch.step()

        tr_loss = running_loss / max(n_tr, 1)
        tr_l_full = running_l_full / max(n_tr, 1)
        tr_l_drop = running_l_drop / max(n_tr, 1)
        tr_l_cons = running_l_cons / max(n_tr, 1)

        va_auc, va_preds, va_labs = b.eval_classifier(model, val_loader_full, device)
        va_fpr, va_tpr, _ = roc_curve(va_labs, va_preds)
        va_fpr50 = b.fpr_at_target_tpr(va_fpr, va_tpr, float(target_tpr))

        if np.isfinite(va_auc) and float(va_auc) > best_auc:
            best_auc = float(va_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            bank_str = f", bank={bank}" if str(drop_mode) == "deterministic_bank" else ""
            print(
                f"{name} ep {ep+1}: train_loss={tr_loss:.5f} "
                f"(full={tr_l_full:.5f}, drop={tr_l_drop:.5f}, cons={tr_l_cons:.5f}, p={p:.3f}{bank_str}) | "
                f"val_auc={float(va_auc):.4f}, val_fpr50={va_fpr50:.6f}, best_auc={best_auc:.4f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _deterministic_drop_mask(
    mask_off_row: np.ndarray,
    mask_hlt_row: np.ndarray,
    drop_prob: float,
    seed: int,
    bank: int,
    jet_idx: int,
) -> np.ndarray:
    mask_full = mask_off_row.astype(bool, copy=True)
    mask_hlt_src = mask_hlt_row.astype(bool, copy=False)
    l_full = int(mask_full.shape[0])
    l_hlt = int(mask_hlt_src.shape[0])

    # Align HLT mask length to target mask length.
    # For concat targets we use [offline || hlt], so HLT occupancy is aligned to
    # the trailing slice of the target sequence.
    if l_hlt == l_full:
        mask_hlt = mask_hlt_src
    elif l_hlt < l_full:
        mask_hlt = np.zeros((l_full,), dtype=bool)
        mask_hlt[(l_full - l_hlt):] = mask_hlt_src
    else:
        mask_hlt = mask_hlt_src[:l_full]

    extra_mask = mask_full & (~mask_hlt)

    if drop_prob > 0.0 and np.any(extra_mask):
        key = ((int(seed) * 1315423911) ^ (int(bank) * 2654435761) ^ (int(jet_idx) * 2246822519)) & 0xFFFFFFFF
        rng = np.random.default_rng(np.uint64(key))
        keep_extra = rng.random(int(extra_mask.sum())) >= float(drop_prob)
        drop_mask = extra_mask.copy()
        drop_mask[extra_mask] = ~keep_extra
        mask_out = mask_full & (~drop_mask)
    else:
        mask_out = mask_full

    if not np.any(mask_out):
        if np.any(mask_hlt):
            mask_out = mask_hlt.copy()
        elif np.any(mask_full):
            mask_out = np.zeros_like(mask_full, dtype=bool)
            mask_out[int(np.flatnonzero(mask_full)[0])] = True
    return mask_out


def build_masked_targets_for_indices(
    const_off: np.ndarray,
    masks_off: np.ndarray,
    hlt_mask: np.ndarray,
    indices: np.ndarray,
    drop_prob: float,
    seed: int,
    bank: int,
    rho: float,
    strict_m2_budget: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = indices.astype(np.int64)
    c_full = const_off[idx]
    m_full = masks_off[idx]
    m_hlt = hlt_mask[idx]

    c_tgt = c_full.copy()
    m_tgt = np.zeros_like(m_full, dtype=bool)

    for i in range(idx.shape[0]):
        m = _deterministic_drop_mask(
            mask_off_row=m_full[i],
            mask_hlt_row=m_hlt[i],
            drop_prob=float(drop_prob),
            seed=int(seed),
            bank=int(bank),
            jet_idx=int(idx[i]),
        )
        m_tgt[i] = m
        c_tgt[i][~m] = 0.0

    true_count = m_tgt.sum(axis=1).astype(np.float32)
    hlt_count = m_hlt.sum(axis=1).astype(np.float32)
    true_added = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    b_merge = (float(rho) * true_added).astype(np.float32)
    if bool(strict_m2_budget):
        # Strict m2-style unmerge-only budgeting: no efficiency split target.
        b_eff = np.zeros_like(true_added, dtype=np.float32)
    else:
        b_eff = ((1.0 - float(rho)) * true_added).astype(np.float32)
    return c_tgt.astype(np.float32), m_tgt.astype(bool), b_merge, b_eff


def _build_concat_target_weights(
    mask_off_target: torch.Tensor,
    offline_prefix_len: int,
    offline_token_weight: float,
    hlt_token_weight: float,
) -> torch.Tensor:
    l = int(mask_off_target.shape[1])
    pref = min(max(int(offline_prefix_len), 0), l)
    w = mask_off_target.float() * float(hlt_token_weight)
    if pref > 0:
        w[:, :pref] = mask_off_target[:, :pref].float() * float(offline_token_weight)
    return w


def ratio_aware_budget_loss_from_out(
    out: Dict[str, torch.Tensor],
    mask_hlt: torch.Tensor,
    mask_off_target: torch.Tensor,
    eps: float,
    under_lambda: float,
    over_lambda: float,
    over_margin_base: float,
    over_margin_scale: float,
    over_ratio_gamma: float,
    over_lambda_floor: float,
) -> torch.Tensor:
    true_count = mask_off_target.float().sum(dim=1)
    hlt_count = mask_hlt.float().sum(dim=1)
    true_added = (true_count - hlt_count).clamp(min=0.0)
    pred_added = out["child_weight"].sum(dim=1) + out["gen_weight"].sum(dim=1)

    ratio = hlt_count / torch.clamp(true_count, min=1.0)
    deficit = torch.clamp(1.0 - ratio, min=0.0, max=1.0)

    over_margin = float(over_margin_base) + float(over_margin_scale) * deficit
    over_w = float(over_lambda) * (1.0 - float(over_ratio_gamma) * deficit)
    over_w = torch.clamp(over_w, min=float(over_lambda_floor))

    under = F.relu(true_added - pred_added - float(max(eps, 0.0)))
    over = F.relu(pred_added - true_added - float(max(eps, 0.0)) - over_margin)
    return (float(under_lambda) * under.square() + over_w * over.square()).mean()


def train_reco_b_masked_m2(
    model: nn.Module,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    const_off: np.ndarray,
    masks_off: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    num_workers: int,
    cfg: Dict,
    rho: float,
    drop_prob: float,
    drop_num_banks: int,
    drop_bank_cycle_epochs: int,
    seed: int,
    ratio_eps: float,
    ratio_under_lambda: float,
    ratio_over_lambda: float,
    ratio_margin_base: float,
    ratio_margin_scale: float,
    ratio_gamma: float,
    ratio_over_floor: float,
    target_mode: str,
    max_constits: int,
    concat_offline_token_weight: float,
    concat_hlt_token_weight: float,
    concat_added_min_frac: float,
    concat_added_min_hinge_lambda: float,
    recoB_reload_best_at_stage_transition: bool,
    use_ratio_budget: bool,
    strict_m2_budget: bool = False,
    feature_ablation_mode: str = "none",
) -> Tuple[nn.Module, Dict[str, float]]:
    train_cfg = cfg["recoB_training"]
    loss_cfg = cfg["loss"]

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = b.get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    # fixed val target (bank 0)
    c_val, m_val, bm_val, be_val = build_masked_targets_for_indices(
        const_off=const_off,
        masks_off=masks_off,
        hlt_mask=hlt_mask,
        indices=val_idx,
        drop_prob=float(drop_prob),
        seed=int(seed),
        bank=0,
        rho=float(rho),
        strict_m2_budget=bool(strict_m2_budget),
    )
    ds_val = m2mod.WeightedReconstructionDataset(
        feat_hlt=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        const_off=c_val,
        mask_off=m_val,
        budget_merge_true=bm_val,
        budget_eff_true=be_val,
        sample_weight_reco=np.ones((len(val_idx),), dtype=np.float32),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    best_val = float("inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    total_epochs = int(train_cfg["epochs"])
    stage1_end = int(max(int(train_cfg.get("stage1_epochs", 0)), 0))
    stage2_end = int(max(int(train_cfg.get("stage2_epochs", stage1_end)), stage1_end))
    stage1_end = min(stage1_end, total_epochs)
    stage2_end = min(max(stage2_end, stage1_end), total_epochs)

    min_stop_epoch = int(stage2_end) + int(train_cfg.get("min_full_scale_epochs", 5))
    is_concat_target = str(target_mode).lower() == "concat"

    stage_best_val = float("inf")
    stage_best_state = None
    stage_best_epoch = -1

    for ep in range(total_epochs):
        bank = (int(ep) // int(max(1, drop_bank_cycle_epochs))) % int(max(1, drop_num_banks))

        c_tr, m_tr, bm_tr, be_tr = build_masked_targets_for_indices(
            const_off=const_off,
            masks_off=masks_off,
            hlt_mask=hlt_mask,
            indices=train_idx,
            drop_prob=float(drop_prob),
            seed=int(seed),
            bank=int(bank),
            rho=float(rho),
            strict_m2_budget=bool(strict_m2_budget),
        )
        ds_train = m2mod.WeightedReconstructionDataset(
            feat_hlt=feat_hlt_std[train_idx],
            mask_hlt=hlt_mask[train_idx],
            const_hlt=hlt_const[train_idx],
            const_off=c_tr,
            mask_off=m_tr,
            budget_merge_true=bm_tr,
            budget_eff_true=be_tr,
            sample_weight_reco=np.ones((len(train_idx),), dtype=np.float32),
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            drop_last=True,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
        )

        if ep < stage1_end:
            sc = 0.35
        elif ep < stage2_end:
            sc = 0.70
        else:
            sc = 1.0

        model.train()
        tr_total = tr_budget_asym = 0.0
        n_tr = 0
        w_budget = float(loss_cfg["w_budget"])

        for batch in dl_train:
            feat_hlt_b = batch["feat_hlt"].to(device)
            mask_hlt_b = batch["mask_hlt"].to(device)
            const_hlt_b = batch["const_hlt"].to(device)
            const_off_b = batch["const_off"].to(device)
            mask_off_b = batch["mask_off"].to(device)
            bm = batch["budget_merge_true"].to(device)
            be = batch["budget_eff_true"].to(device)

            opt.zero_grad()
            out = model(feat_hlt_b, mask_hlt_b, const_hlt_b, stage_scale=float(sc))
            target_token_w = None
            if is_concat_target:
                target_token_w = _build_concat_target_weights(
                    mask_off_target=mask_off_b,
                    offline_prefix_len=int(max_constits),
                    offline_token_weight=float(concat_offline_token_weight),
                    hlt_token_weight=float(concat_hlt_token_weight),
                )
            losses = compute_reco_b_losses(
                out,
                const_hlt_b,
                mask_hlt_b,
                const_off_b,
                mask_off_b,
                bm,
                be,
                loss_cfg,
                feature_ablation_mode=str(feature_ablation_mode),
                target_token_weights=target_token_w,
                added_min_frac=float(concat_added_min_frac) if is_concat_target else 0.0,
                added_min_hinge_lambda=float(concat_added_min_hinge_lambda) if is_concat_target else 0.0,
            )
            if bool(use_ratio_budget):
                if bool(use_ratio_budget):
                    loss_budget_asym = ratio_aware_budget_loss_from_out(
                        out=out,
                        mask_hlt=mask_hlt_b,
                        mask_off_target=mask_off_b,
                        eps=float(ratio_eps),
                        under_lambda=float(ratio_under_lambda),
                        over_lambda=float(ratio_over_lambda),
                        over_margin_base=float(ratio_margin_base),
                        over_margin_scale=float(ratio_margin_scale),
                        over_ratio_gamma=float(ratio_gamma),
                        over_lambda_floor=float(ratio_over_floor),
                    )
                    loss = losses["total"] - w_budget * losses["budget"] + w_budget * loss_budget_asym
                else:
                    loss_budget_asym = torch.zeros((), device=feat_hlt_b.device)
                    loss = losses["total"]
            else:
                loss_budget_asym = torch.zeros((), device=feat_hlt_b.device)
                loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = feat_hlt_b.size(0)
            tr_total += float(loss.item()) * bs
            tr_budget_asym += float(loss_budget_asym.item()) * bs
            n_tr += bs

        sch.step()

        model.eval()
        va_total = va_budget_asym = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in dl_val:
                feat_hlt_b = batch["feat_hlt"].to(device)
                mask_hlt_b = batch["mask_hlt"].to(device)
                const_hlt_b = batch["const_hlt"].to(device)
                const_off_b = batch["const_off"].to(device)
                mask_off_b = batch["mask_off"].to(device)
                bm = batch["budget_merge_true"].to(device)
                be = batch["budget_eff_true"].to(device)

                out = model(feat_hlt_b, mask_hlt_b, const_hlt_b, stage_scale=1.0)
                target_token_w = None
                if is_concat_target:
                    target_token_w = _build_concat_target_weights(
                        mask_off_target=mask_off_b,
                        offline_prefix_len=int(max_constits),
                        offline_token_weight=float(concat_offline_token_weight),
                        hlt_token_weight=float(concat_hlt_token_weight),
                    )
                losses = compute_reco_b_losses(
                    out,
                    const_hlt_b,
                    mask_hlt_b,
                    const_off_b,
                    mask_off_b,
                    bm,
                    be,
                    loss_cfg,
                    feature_ablation_mode=str(feature_ablation_mode),
                    target_token_weights=target_token_w,
                    added_min_frac=float(concat_added_min_frac) if is_concat_target else 0.0,
                    added_min_hinge_lambda=float(concat_added_min_hinge_lambda) if is_concat_target else 0.0,
                )
                if bool(use_ratio_budget):
                    loss_budget_asym = ratio_aware_budget_loss_from_out(
                        out=out,
                        mask_hlt=mask_hlt_b,
                        mask_off_target=mask_off_b,
                        eps=float(ratio_eps),
                        under_lambda=float(ratio_under_lambda),
                        over_lambda=float(ratio_over_lambda),
                        over_margin_base=float(ratio_margin_base),
                        over_margin_scale=float(ratio_margin_scale),
                        over_ratio_gamma=float(ratio_gamma),
                        over_lambda_floor=float(ratio_over_floor),
                    )
                    loss = losses["total"] - w_budget * losses["budget"] + w_budget * loss_budget_asym
                else:
                    loss_budget_asym = torch.zeros((), device=feat_hlt_b.device)
                    loss = losses["total"]

                bs = feat_hlt_b.size(0)
                va_total += float(loss.item()) * bs
                va_budget_asym += float(loss_budget_asym.item()) * bs
                n_va += bs

        tr_total /= max(n_tr, 1)
        tr_budget_asym /= max(n_tr, 1)
        va_total /= max(n_va, 1)
        va_budget_asym /= max(n_va, 1)

        if va_total < best_val:
            best_val = float(va_total)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "best_epoch": int(ep + 1),
                "best_val_total": float(va_total),
                "best_val_budget_asym": float(va_budget_asym),
                "best_train_total": float(tr_total),
                "best_train_budget_asym": float(tr_budget_asym),
                "best_bank": int(bank),
            }
            no_improve = 0
        else:
            no_improve += 1

        if va_total < stage_best_val:
            stage_best_val = float(va_total)
            stage_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stage_best_epoch = int(ep + 1)

        if (ep + 1) % 5 == 0:
            print(
                f"RecoB ep {ep+1}: train_total={tr_total:.5f}, val_total={va_total:.5f}, "
                f"train_budget_asym={tr_budget_asym:.5f}, val_budget_asym={va_budget_asym:.5f}, "
                f"bank={bank}, best_val={best_val:.5f}, stage_best={stage_best_val:.5f}"
            )

        if bool(recoB_reload_best_at_stage_transition) and (ep + 1 in {int(stage1_end), int(stage2_end)}) and stage_best_state is not None:
            model.load_state_dict(stage_best_state)
            print(
                f"Reloaded best RecoB checkpoint at stage boundary epoch={ep+1} "
                f"(stage_best_epoch={stage_best_epoch}, stage_best_val={stage_best_val:.5f})"
            )
            stage_best_val = float("inf")
            stage_best_state = None
            stage_best_epoch = -1
            no_improve = 0

        if (ep + 1) >= int(min_stop_epoch) and no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping RecoB at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


@torch.no_grad()
def eval_dual_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    auc, preds, labs = b.eval_classifier_dual(model, loader, device)
    if preds.size == 0:
        return float("nan"), preds, labs, float("nan")
    fpr, tpr, _ = roc_curve(labs, preds)
    fpr50 = b.fpr_at_target_tpr(fpr, tpr, 0.50)
    return float(auc), preds.astype(np.float64), labs.astype(np.float32), float(fpr50)


def train_dual_frozen(
    model: nn.Module,
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
) -> Tuple[nn.Module, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for ep in range(int(epochs)):
        model.train()
        tr_loss = tr_cls = tr_rank = 0.0
        n_tr = 0

        for batch in train_loader:
            xa = batch["feat_a"].to(device)
            ma = batch["mask_a"].to(device)
            xb = batch["feat_b"].to(device)
            mb = batch["mask_b"].to(device)
            y = batch["label"].to(device)

            opt.zero_grad()
            logits = model(xa, ma, xb, mb).squeeze(1)
            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = b.low_fpr_surrogate_loss(logits, y, target_tpr=float(target_tpr), tau=float(rank_tau))
            loss = loss_cls + float(lambda_rank) * loss_rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        va_auc, _, _, va_fpr50 = eval_dual_model(model, val_loader, device)

        if str(select_metric).lower() == "fpr50":
            sel = va_fpr50
            improved = np.isfinite(sel) and (sel < best_sel)
        else:
            sel = va_auc
            improved = np.isfinite(sel) and (sel > best_sel)

        if improved:
            best_sel = float(sel)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
                f"DualFrozen ep {ep+1}: train_loss={tr_loss:.5f} (cls={tr_cls:.5f}, rank={tr_rank:.5f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping DualFrozen at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


class JointTwoRecoDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        labels: np.ndarray,
        const_off_full: np.ndarray,
        mask_off_full: np.ndarray,
        budget_merge_full: np.ndarray,
        budget_eff_full: np.ndarray,
        const_off_b: np.ndarray,
        mask_off_b: np.ndarray,
        budget_merge_b: np.ndarray,
        budget_eff_b: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        self.const_off_full = torch.tensor(const_off_full, dtype=torch.float32)
        self.mask_off_full = torch.tensor(mask_off_full, dtype=torch.bool)
        self.budget_merge_full = torch.tensor(budget_merge_full, dtype=torch.float32)
        self.budget_eff_full = torch.tensor(budget_eff_full, dtype=torch.float32)
        self.const_off_b = torch.tensor(const_off_b, dtype=torch.float32)
        self.mask_off_b = torch.tensor(mask_off_b, dtype=torch.bool)
        self.budget_merge_b = torch.tensor(budget_merge_b, dtype=torch.float32)
        self.budget_eff_b = torch.tensor(budget_eff_b, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "label": self.labels[i],
            "const_off_full": self.const_off_full[i],
            "mask_off_full": self.mask_off_full[i],
            "budget_merge_full": self.budget_merge_full[i],
            "budget_eff_full": self.budget_eff_full[i],
            "const_off_b": self.const_off_b[i],
            "mask_off_b": self.mask_off_b[i],
            "budget_merge_b": self.budget_merge_b[i],
            "budget_eff_b": self.budget_eff_b[i],
        }


@torch.no_grad()
def eval_dual_joint_dynamic(
    reco_a: nn.Module,
    reco_b: nn.Module,
    dual: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    feature_ablation_mode: str = "none",
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    reco_a.eval()
    reco_b.eval()
    dual.eval()

    preds = []
    labs = []
    for batch in loader:
        feat_hlt = batch["feat_hlt"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

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
        feat_b = apply_feature_ablation_to_corrected_torch(feat_b, mask_b, str(feature_ablation_mode))
        logits = dual(feat_a, mask_a, feat_b, mask_b).squeeze(1)
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds).astype(np.float64) if preds else np.zeros(0, dtype=np.float64)
    labs = np.concatenate(labs).astype(np.float32) if labs else np.zeros(0, dtype=np.float32)
    auc, fpr50 = auc_and_fpr(labs, preds, target_tpr=0.50)
    return auc, preds, labs, fpr50


def train_dual_joint_two_reco(
    reco_a: nn.Module,
    reco_b: nn.Module,
    dual: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    teacher_model: nn.Module,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    lr_dual: float,
    lr_reco_a: float,
    lr_reco_b: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_rank: float,
    rank_tau: float,
    corrected_weight_floor: float,
    select_metric: str,
    lambda_anchor_a: float,
    lambda_anchor_b: float,
    stageA_lambda_kd: float,
    stageA_lambda_emb: float,
    stageA_lambda_tok: float,
    stageA_lambda_phys: float,
    stageA_lambda_budget_hinge: float,
    stageA_budget_eps: float,
    stageA_budget_weight_floor: float,
    recoB_ratio_eps: float,
    recoB_ratio_under_lambda: float,
    recoB_ratio_over_lambda: float,
    recoB_ratio_margin_base: float,
    recoB_ratio_margin_scale: float,
    recoB_ratio_gamma: float,
    recoB_ratio_over_floor: float,
    recoB_loss_cfg: Dict,
    target_mode: str,
    max_constits: int,
    concat_offline_token_weight: float,
    concat_hlt_token_weight: float,
    concat_added_min_frac: float,
    concat_added_min_hinge_lambda: float,
    recoB_use_ratio_budget: bool,
    feature_ablation_mode: str = "none",
) -> Dict[str, float]:
    for p in reco_a.parameters():
        p.requires_grad_(True)
    for p in reco_b.parameters():
        p.requires_grad_(True)

    for p in teacher_model.parameters():
        p.requires_grad_(False)
    teacher_model.eval()

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(np.clip(stds, 1e-6, None), dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(
        [
            {"params": dual.parameters(), "lr": float(lr_dual)},
            {"params": reco_a.parameters(), "lr": float(lr_reco_a)},
            {"params": reco_b.parameters(), "lr": float(lr_reco_b)},
        ],
        lr=float(lr_dual),
        weight_decay=float(weight_decay),
    )
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    w_budget_b = float(recoB_loss_cfg["w_budget"])

    for ep in range(int(epochs)):
        reco_a.train()
        reco_b.train()
        dual.train()

        tr_loss = tr_cls = tr_rank = tr_anchor_a = tr_anchor_b = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            y = batch["label"].to(device)

            const_off_full = batch["const_off_full"].to(device)
            mask_off_full = batch["mask_off_full"].to(device)
            bmf = batch["budget_merge_full"].to(device)
            bef = batch["budget_eff_full"].to(device)

            const_off_b = batch["const_off_b"].to(device)
            mask_off_b = batch["mask_off_b"].to(device)
            bmb = batch["budget_merge_b"].to(device)
            beb = batch["budget_eff_b"].to(device)

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
            feat_b = apply_feature_ablation_to_corrected_torch(feat_b, mask_b, str(feature_ablation_mode))

            logits = dual(feat_a, mask_a, feat_b, mask_b).squeeze(1)
            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = b.low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=float(rank_tau))

            if float(lambda_anchor_a) > 0.0:
                losses_a = b._compute_teacher_guided_reco_losses(
                    reco_out=out_a,
                    const_hlt=const_hlt,
                    mask_hlt=mask_hlt,
                    const_off=const_off_full,
                    mask_off=mask_off_full,
                    budget_merge_true=bmf,
                    budget_eff_true=bef,
                    teacher_model=teacher_model,
                    means_t=means_t,
                    stds_t=stds_t,
                    loss_cfg=b.BASE_CONFIG["loss"],
                    kd_temperature=2.5,
                    budget_eps=float(max(stageA_budget_eps, 0.0)),
                    budget_weight_floor=float(max(stageA_budget_weight_floor, 0.0)),
                )
                loss_anchor_a = (
                    float(max(stageA_lambda_kd, 0.0)) * losses_a["kd"]
                    + float(max(stageA_lambda_emb, 0.0)) * losses_a["emb"]
                    + float(max(stageA_lambda_tok, 0.0)) * losses_a["tok"]
                    + float(max(stageA_lambda_phys, 0.0)) * losses_a["phys"]
                    + float(max(stageA_lambda_budget_hinge, 0.0)) * losses_a["budget_hinge"]
                )
            else:
                loss_anchor_a = torch.zeros((), device=device)

            if float(lambda_anchor_b) > 0.0:
                target_token_w_b = None
                if str(target_mode).lower() == "concat":
                    target_token_w_b = _build_concat_target_weights(
                        mask_off_target=mask_off_b,
                        offline_prefix_len=int(max_constits),
                        offline_token_weight=float(concat_offline_token_weight),
                        hlt_token_weight=float(concat_hlt_token_weight),
                    )
                losses_b = compute_reco_b_losses(
                    out_b,
                    const_hlt,
                    mask_hlt,
                    const_off_b,
                    mask_off_b,
                    bmb,
                    beb,
                    recoB_loss_cfg,
                    feature_ablation_mode=str(feature_ablation_mode),
                    target_token_weights=target_token_w_b,
                    added_min_frac=float(concat_added_min_frac) if str(target_mode).lower() == "concat" else 0.0,
                    added_min_hinge_lambda=float(concat_added_min_hinge_lambda) if str(target_mode).lower() == "concat" else 0.0,
                )
                if bool(recoB_use_ratio_budget):
                    loss_budget_asym_b = ratio_aware_budget_loss_from_out(
                        out=out_b,
                        mask_hlt=mask_hlt,
                        mask_off_target=mask_off_b,
                        eps=float(recoB_ratio_eps),
                        under_lambda=float(recoB_ratio_under_lambda),
                        over_lambda=float(recoB_ratio_over_lambda),
                        over_margin_base=float(recoB_ratio_margin_base),
                        over_margin_scale=float(recoB_ratio_margin_scale),
                        over_ratio_gamma=float(recoB_ratio_gamma),
                        over_lambda_floor=float(recoB_ratio_over_floor),
                    )
                    loss_anchor_b = losses_b["total"] - w_budget_b * losses_b["budget"] + w_budget_b * loss_budget_asym_b
                else:
                    loss_anchor_b = losses_b["total"]
            else:
                loss_anchor_b = torch.zeros((), device=device)

            loss = (
                loss_cls
                + float(lambda_rank) * loss_rank
                + float(lambda_anchor_a) * loss_anchor_a
                + float(lambda_anchor_b) * loss_anchor_b
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reco_a.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reco_b.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            tr_loss += float(loss.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            tr_rank += float(loss_rank.item()) * bs
            tr_anchor_a += float(loss_anchor_a.item()) * bs
            tr_anchor_b += float(loss_anchor_b.item()) * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_anchor_a /= max(n_tr, 1)
        tr_anchor_b /= max(n_tr, 1)

        va_auc, _, _, va_fpr50 = eval_dual_joint_dynamic(
            reco_a=reco_a,
            reco_b=reco_b,
            dual=dual,
            loader=val_loader,
            device=device,
            corrected_weight_floor=float(corrected_weight_floor),
            feature_ablation_mode=str(feature_ablation_mode),
        )

        if str(select_metric).lower() == "fpr50":
            sel = va_fpr50
            improved = np.isfinite(sel) and (sel < best_sel)
        else:
            sel = va_auc
            improved = np.isfinite(sel) and (sel > best_sel)

        if improved:
            best_sel = float(sel)
            best_state = {
                "dual": {k: v.detach().cpu().clone() for k, v in dual.state_dict().items()},
                "reco_a": {k: v.detach().cpu().clone() for k, v in reco_a.state_dict().items()},
                "reco_b": {k: v.detach().cpu().clone() for k, v in reco_b.state_dict().items()},
            }
            best_metrics = {
                "best_epoch": int(ep + 1),
                "best_select_metric": str(select_metric).lower(),
                "best_sel": float(best_sel),
                "best_val_auc": float(va_auc),
                "best_val_fpr50": float(va_fpr50),
                "best_train_loss": float(tr_loss),
                "best_train_cls": float(tr_cls),
                "best_train_rank": float(tr_rank),
                "best_train_anchor_a": float(tr_anchor_a),
                "best_train_anchor_b": float(tr_anchor_b),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 2 == 0:
            print(
                f"DualJoint ep {ep+1}: train_loss={tr_loss:.5f} (cls={tr_cls:.5f}, rank={tr_rank:.5f}, "
                f"anchorA={tr_anchor_a:.5f}, anchorB={tr_anchor_b:.5f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping DualJoint at epoch {ep+1}")
            break

    if best_state is not None:
        dual.load_state_dict(best_state["dual"])
        reco_a.load_state_dict(best_state["reco_a"])
        reco_b.load_state_dict(best_state["reco_b"])

    return best_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--n_train_jets", type=int, default=375000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--n_train_split", type=int, default=150000)
    ap.add_argument("--n_val_split", type=int, default=75000)
    ap.add_argument("--n_test_split", type=int, default=150000)
    ap.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "reco_teacher_joint_fusion_6model_150k75k150k" / "model9_dualreco_dualview"))
    ap.add_argument("--run_name", type=str, default="model9_dualreco_dualview_150k75k150k_seed0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_save_models", action="store_true")

    ap.add_argument("--teacher_use_offline_dropout", action="store_true")
    ap.add_argument("--teacher_drop_prob_max", type=float, default=0.50)
    ap.add_argument("--teacher_drop_warmup_epochs", type=int, default=20)
    ap.add_argument("--teacher_drop_mode", type=str, choices=["random", "deterministic_bank"], default="deterministic_bank")
    ap.add_argument("--teacher_drop_num_banks", type=int, default=3)
    ap.add_argument("--teacher_drop_bank_cycle_epochs", type=int, default=1)
    ap.add_argument("--teacher_lambda_drop_cls", type=float, default=1.0)
    ap.add_argument("--teacher_use_consistency", action="store_true")
    ap.add_argument("--teacher_consistency_temp", type=float, default=2.0)
    ap.add_argument("--teacher_lambda_consistency", type=float, default=0.2)

    ap.add_argument("--teacher_use_anti_overlap", action="store_true")
    ap.add_argument("--teacher_anti_lambda", type=float, default=0.02)
    ap.add_argument("--teacher_anti_tau", type=float, default=0.05)
    ap.add_argument("--teacher_anti_beta", type=float, default=0.10)
    ap.add_argument("--teacher_anti_warmup_epochs", type=int, default=12)

    ap.add_argument("--merge_radius", type=float, default=b.BASE_CONFIG["hlt_effects"]["merge_radius"])
    ap.add_argument("--eff_plateau_barrel", type=float, default=b.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    ap.add_argument("--eff_plateau_endcap", type=float, default=b.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    ap.add_argument("--smear_a", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_a"])
    ap.add_argument("--smear_b", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_b"])
    ap.add_argument("--smear_c", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_c"])

    # Reco-A teacher-guided Stage-A
    ap.add_argument("--stageA_epochs", type=int, default=90)
    ap.add_argument("--stageA_patience", type=int, default=18)
    ap.add_argument("--stageA_kd_temp", type=float, default=2.5)
    ap.add_argument("--stageA_lambda_kd", type=float, default=5.0)
    ap.add_argument("--stageA_lambda_emb", type=float, default=0.0)
    ap.add_argument("--stageA_lambda_tok", type=float, default=0.0)
    ap.add_argument("--stageA_lambda_phys", type=float, default=0.05)
    ap.add_argument("--stageA_lambda_budget_hinge", type=float, default=1.0)
    ap.add_argument("--stageA_lambda_delta", type=float, default=0.15)
    ap.add_argument("--stageA_delta_tau", type=float, default=0.05)
    ap.add_argument("--stageA_delta_lambda_fp", type=float, default=3.0)
    ap.add_argument("--stageA_budget_eps", type=float, default=0.015)
    ap.add_argument("--stageA_budget_weight_floor", type=float, default=1e-4)
    ap.add_argument("--stageA_target_tpr", type=float, default=0.50)
    ap.add_argument("--disable_stageA_loss_normalization", action="store_true")
    ap.add_argument("--stageA_loss_norm_ema_decay", type=float, default=0.98)
    ap.add_argument("--stageA_loss_norm_eps", type=float, default=1e-6)
    ap.add_argument("--disable_stageA_stagewise_best_reload", action="store_true")

    # Reco-B masked m2-style
    ap.add_argument("--recoB_epochs", type=int, default=90)
    ap.add_argument("--recoB_patience", type=int, default=18)
    ap.add_argument("--recoB_lr", type=float, default=3e-4)
    ap.add_argument("--recoB_weight_decay", type=float, default=1e-4)
    ap.add_argument("--recoB_warmup_epochs", type=int, default=5)
    ap.add_argument("--recoB_stage1_epochs", type=int, default=20)
    ap.add_argument("--recoB_stage2_epochs", type=int, default=55)
    ap.add_argument("--recoB_min_full_scale_epochs", type=int, default=5)
    ap.add_argument("--target_drop_prob_max", type=float, default=0.50)
    ap.add_argument("--target_drop_num_banks", type=int, default=3)
    ap.add_argument("--target_drop_bank_cycle_epochs", type=int, default=1)
    ap.add_argument("--recoB_ratio_count_eps", type=float, default=0.015)
    ap.add_argument("--recoB_ratio_count_under_lambda", type=float, default=1.0)
    ap.add_argument("--recoB_ratio_count_over_lambda", type=float, default=0.25)
    ap.add_argument("--recoB_ratio_count_over_margin_base", type=float, default=2.0)
    ap.add_argument("--recoB_ratio_count_over_margin_scale", type=float, default=6.0)
    ap.add_argument("--recoB_ratio_count_over_ratio_gamma", type=float, default=0.70)
    ap.add_argument("--recoB_ratio_count_over_lambda_floor", type=float, default=0.05)
    ap.add_argument("--max_concat_constits", type=int, default=-1)
    ap.add_argument("--recoB_concat_offline_token_weight", type=float, default=3.0)
    ap.add_argument("--recoB_concat_hlt_token_weight", type=float, default=0.25)
    ap.add_argument("--recoB_concat_added_min_frac", type=float, default=0.75)
    ap.add_argument("--recoB_concat_added_min_hinge_lambda", type=float, default=0.20)
    ap.add_argument("--disable_recoB_stagewise_best_reload", action="store_true")
    ap.add_argument("--disable_recoB_ratio_budget", action="store_true")
    ap.add_argument("--recoB_strict_m2_mode", action="store_true")

    # Dual frozen
    ap.add_argument("--dual_frozen_epochs", type=int, default=45)
    ap.add_argument("--dual_frozen_patience", type=int, default=12)
    ap.add_argument("--dual_frozen_batch_size", type=int, default=256)
    ap.add_argument("--dual_frozen_lr", type=float, default=3e-4)
    ap.add_argument("--dual_frozen_weight_decay", type=float, default=1e-4)
    ap.add_argument("--dual_frozen_warmup_epochs", type=int, default=5)
    ap.add_argument("--dual_frozen_lambda_rank", type=float, default=0.2)
    ap.add_argument("--dual_frozen_rank_tau", type=float, default=0.05)

    # Dual joint
    ap.add_argument("--dual_joint_epochs", type=int, default=12)
    ap.add_argument("--dual_joint_patience", type=int, default=6)
    ap.add_argument("--dual_joint_batch_size", type=int, default=128)
    ap.add_argument("--dual_joint_lr_dual", type=float, default=1e-4)
    ap.add_argument("--dual_joint_lr_reco_a", type=float, default=2e-6)
    ap.add_argument("--dual_joint_lr_reco_b", type=float, default=2e-6)
    ap.add_argument("--dual_joint_weight_decay", type=float, default=1e-4)
    ap.add_argument("--dual_joint_warmup_epochs", type=int, default=3)
    ap.add_argument("--dual_joint_lambda_rank", type=float, default=0.2)
    ap.add_argument("--dual_joint_rank_tau", type=float, default=0.05)
    ap.add_argument("--dual_joint_lambda_anchor_a", type=float, default=0.02)
    ap.add_argument("--dual_joint_lambda_anchor_b", type=float, default=0.02)

    ap.add_argument("--added_target_scale", type=float, default=0.90)
    ap.add_argument("--corrected_weight_floor", type=float, default=0.03)
    ap.add_argument("--reco_eval_batch_size", type=int, default=256)
    ap.add_argument("--select_metric", type=str, choices=["auc", "fpr50"], default="auc")
    ap.add_argument("--report_target_tpr", type=float, default=0.50)
    ap.add_argument("--target_mode", type=str, choices=["offdrop", "topk", "feature_ablation", "concat"], default="offdrop")
    ap.add_argument("--offline_target_topk_pt", type=int, default=0)
    ap.add_argument("--target_feature_ablation", type=str, choices=["none", "no_angle", "no_scale", "core_shape"], default="none")

    args = ap.parse_args()

    if bool(args.teacher_use_anti_overlap) and bool(args.teacher_use_offline_dropout):
        raise ValueError("Use either --teacher_use_anti_overlap or --teacher_use_offline_dropout, not both.")

    set_seed(int(args.seed))
    b.set_seed(int(args.seed))

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
    masks_off_full = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off_full = const_raw.copy()
    const_off_full[~masks_off_full] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, _ = b.apply_hlt_effects_realistic_nomap(
        const_off_full,
        masks_off_full,
        cfg,
        seed=int(args.seed),
    )

    target_mode = str(args.target_mode).lower()
    topk_target = int(args.offline_target_topk_pt)
    feature_ablation_mode = "none"
    max_concat_constits = int(args.max_concat_constits)
    if max_concat_constits <= 0:
        max_concat_constits = int(args.max_constits) * 2

    if target_mode == "topk":
        if 0 < topk_target < int(args.max_constits):
            const_off_target, masks_off_target = sA.apply_offline_topk_target_mask_by_pt(
                const_off_full,
                masks_off_full,
                topk_target,
            )
            print(
                f"Target mode=topk, applying offline top-k target mask: k={topk_target} "
                f"(HLT input max_constits stays {int(args.max_constits)})."
            )
        else:
            print(
                f"Target mode=topk but offline_target_topk_pt={topk_target} is invalid for "
                f"max_constits={int(args.max_constits)}; falling back to full offline target."
            )
            target_mode = "offdrop"
            const_off_target = const_off_full
            masks_off_target = masks_off_full
    elif target_mode == "feature_ablation":
        feature_ablation_mode = _norm_feature_ablation_mode(str(args.target_feature_ablation))
        if feature_ablation_mode == "none":
            raise ValueError("target_mode=feature_ablation requires --target_feature_ablation != none")
        const_off_target = const_off_full
        masks_off_target = masks_off_full
        print(f"Target mode=feature_ablation, mode={feature_ablation_mode} (HLT input remains full).")
    elif target_mode == "concat":
        const_off_target, masks_off_target = build_concat_constituents(
            const_off=const_off_full,
            mask_off=masks_off_full,
            const_hlt=hlt_const,
            mask_hlt=hlt_mask,
            max_concat_constits=int(max_concat_constits),
        )
        print(
            f"Target mode=concat, teacher/recoB target uses offline||HLT with max_concat_constits={int(max_concat_constits)}; "
            f"Reco-A physical target remains full offline."
        )
    else:
        const_off_target = const_off_full
        masks_off_target = masks_off_full

    true_count = masks_off_target.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    rho = b._clamp_target_scale(float(args.added_target_scale))
    budget_merge_true = (rho * true_added_raw).astype(np.float32)
    budget_eff_true = ((1.0 - rho) * true_added_raw).astype(np.float32)

    true_count_full = masks_off_full.sum(axis=1).astype(np.float32)
    true_added_raw_full = np.maximum(true_count_full - hlt_count, 0.0).astype(np.float32)
    budget_merge_true_full = (rho * true_added_raw_full).astype(np.float32)
    budget_eff_true_full = ((1.0 - rho) * true_added_raw_full).astype(np.float32)

    if target_mode == "concat":
        const_off_stageA = const_off_full
        masks_off_stageA = masks_off_full
        budget_merge_true_stageA = budget_merge_true_full
        budget_eff_true_stageA = budget_eff_true_full
    else:
        const_off_stageA = const_off_target
        masks_off_stageA = masks_off_target
        budget_merge_true_stageA = budget_merge_true
        budget_eff_true_stageA = budget_eff_true

    print(
        f"Non-priv rho split setup: rho={rho:.3f}, "
        f"mean_true_added_raw(target)={float(true_added_raw.mean()):.3f}, "
        f"mean_target_merge={float(budget_merge_true.mean()):.3f}, "
        f"mean_target_eff={float(budget_eff_true.mean()):.3f}"
    )

    print("Computing features...")
    feat_off = b.compute_features(const_off_target, masks_off_target)
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

    means, stds = b.get_stats(feat_off, masks_off_target, train_idx)
    feat_off_std = b.standardize(feat_off, masks_off_target, means, stds)
    feat_hlt_std = b.standardize(feat_hlt, hlt_mask, means, stds)
    feat_off_teacher_std = sA.apply_teacher_feature_ablation_np(
        feat_off_std,
        masks_off_target,
        feature_ablation_mode if target_mode == "feature_ablation" else "none",
    )

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
        "variant": "m9_dualreco_dualview_offdrop",
        "rho": float(rho),
        "target_mode": str(target_mode),
        "offline_target_topk_pt": int(topk_target),
        "target_feature_ablation": str(feature_ablation_mode),
        "max_concat_constits": int(max_concat_constits),
        "stageA_target": "full_offline" if target_mode == "concat" else "same_as_teacher_target",
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

    ds_train_off = b.JetDataset(feat_off_teacher_std[train_idx], masks_off_target[train_idx], labels[train_idx])
    ds_val_off = b.JetDataset(feat_off_teacher_std[val_idx], masks_off_target[val_idx], labels[val_idx])
    ds_test_off = b.JetDataset(feat_off_teacher_std[test_idx], masks_off_target[test_idx], labels[test_idx])
    dl_train_off = DataLoader(ds_train_off, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_off = DataLoader(ds_val_off, batch_size=BS, shuffle=False)
    dl_test_off = DataLoader(ds_test_off, batch_size=BS, shuffle=False)

    ds_train_hlt = b.JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx])
    ds_val_hlt = b.JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    ds_test_hlt = b.JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_train_hlt = DataLoader(ds_train_hlt, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=BS, shuffle=False)
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=BS, shuffle=False)

    baseline = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = b.train_single_view_classifier_auc(baseline, dl_train_hlt, dl_val_hlt, device, cfg["training"], name="Baseline")
    auc_hlt_test, preds_hlt_test, _ = b.eval_classifier(baseline, dl_test_hlt, device)
    auc_hlt_val, preds_hlt_val, _ = b.eval_classifier(baseline, dl_val_hlt, device)

    hlt_thr_prob, hlt_thr_tpr, hlt_thr_fpr = threshold_at_target_tpr(labels[val_idx], preds_hlt_val, float(args.stageA_target_tpr))
    print(
        f"StageA delta HLT reference @TPR={float(args.stageA_target_tpr):.2f}: "
        f"threshold_prob={hlt_thr_prob:.6f}, val_tpr={hlt_thr_tpr:.6f}, val_fpr={hlt_thr_fpr:.6f}"
    )

    teacher = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    if bool(args.teacher_use_anti_overlap):
        print(
            "Teacher mode: anti-overlap "
            f"(lambda={float(args.teacher_anti_lambda):.4f}, tau={float(args.teacher_anti_tau):.4f}, "
            f"beta={float(args.teacher_anti_beta):.4f}, warmup={int(args.teacher_anti_warmup_epochs)})"
        )
        ds_train_teacher_anti = sA.TeacherAntiOverlapDataset(
            feat_off=feat_off_teacher_std[train_idx],
            mask_off=masks_off_target[train_idx],
            feat_hlt=feat_hlt_std[train_idx],
            mask_hlt=hlt_mask[train_idx],
            labels=labels[train_idx],
        )
        dl_train_teacher_anti = DataLoader(ds_train_teacher_anti, batch_size=BS, shuffle=True, drop_last=True)
        teacher = sA.train_single_view_teacher_anti_overlap(
            model=teacher,
            train_loader=dl_train_teacher_anti,
            val_loader_off=dl_val_off,
            hlt_model=baseline,
            hlt_threshold_prob=float(hlt_thr_prob),
            device=device,
            train_cfg=cfg["training"],
            target_tpr=float(args.stageA_target_tpr),
            anti_lambda=float(args.teacher_anti_lambda),
            anti_tau=float(args.teacher_anti_tau),
            anti_beta=float(args.teacher_anti_beta),
            anti_warmup_epochs=int(args.teacher_anti_warmup_epochs),
            name="TeacherAntiOverlap",
        )
    elif bool(args.teacher_use_offline_dropout):
        print("Teacher mode: offline-dropout")
        teacher = train_single_view_teacher_with_offline_dropout(
            model=teacher,
            feat_off=feat_off_teacher_std,
            mask_off=masks_off_target,
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
        teacher = b.train_single_view_classifier_auc(teacher, dl_train_off, dl_val_off, device, cfg["training"], name="Teacher")

    auc_teacher_test, preds_teacher_test, _ = b.eval_classifier(teacher, dl_test_off, device)
    auc_teacher_val, preds_teacher_val, _ = b.eval_classifier(teacher, dl_val_off, device)

    print("\n" + "=" * 70)
    print("STEP 2: TRAIN RECO-A (TEACHER-GUIDED STAGE-A)")
    print("=" * 70)
    ds_train_reco_a = b.StageAReconstructionDataset(
        feat_hlt_std[train_idx], hlt_mask[train_idx], hlt_const[train_idx],
        const_off_stageA[train_idx], masks_off_stageA[train_idx], labels[train_idx],
        budget_merge_true_stageA[train_idx], budget_eff_true_stageA[train_idx],
    )
    ds_val_reco_a = b.StageAReconstructionDataset(
        feat_hlt_std[val_idx], hlt_mask[val_idx], hlt_const[val_idx],
        const_off_stageA[val_idx], masks_off_stageA[val_idx], labels[val_idx],
        budget_merge_true_stageA[val_idx], budget_eff_true_stageA[val_idx],
    )
    dl_train_reco_a = DataLoader(ds_train_reco_a, batch_size=int(cfg["reconstructor_training"]["batch_size"]), shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    dl_val_reco_a = DataLoader(ds_val_reco_a, batch_size=int(cfg["reconstructor_training"]["batch_size"]), shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    reco_a = b.OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    b.BASE_CONFIG["loss"] = cfg["loss"]
    reco_a, reco_a_val_metrics = sA.train_reconstructor_teacher_guided_stagewise_delta(
        model=reco_a,
        train_loader=dl_train_reco_a,
        val_loader=dl_val_reco_a,
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
    if bool(args.recoB_strict_m2_mode):
        print("STEP 3: TRAIN RECO-B (STRICT M2 MODE: UNMERGE-ONLY + PLAIN BUDGET)")
    else:
        print("STEP 3: TRAIN RECO-B (M2-STYLE MASKED TARGET + RATIO BUDGET)")
    print("=" * 70)

    target_drop_prob_for_reco_b = float(args.target_drop_prob_max) if target_mode == "offdrop" else 0.0
    strict_m2_budget = bool(args.recoB_strict_m2_mode)

    cfg_reco_b = {
        "recoB_training": {
            "epochs": int(args.recoB_epochs),
            "patience": int(args.recoB_patience),
            "batch_size": int(cfg["reconstructor_training"]["batch_size"]),
            "lr": float(args.recoB_lr),
            "weight_decay": float(args.recoB_weight_decay),
            "warmup_epochs": int(args.recoB_warmup_epochs),
            "stage1_epochs": int(args.recoB_stage1_epochs),
            "stage2_epochs": int(args.recoB_stage2_epochs),
            "min_full_scale_epochs": int(args.recoB_min_full_scale_epochs),
        },
        "loss": copy.deepcopy(m2mod.BASE_CONFIG["loss"]),
    }

    reco_b = m2mod.OfflineReconstructor(input_dim=7, **m2mod.BASE_CONFIG["reconstructor_model"]).to(device)
    if bool(strict_m2_budget):
        reco_b = m2mod.wrap_reconstructor_unmerge_only(reco_b)
    reco_b, reco_b_metrics = train_reco_b_masked_m2(
        model=reco_b,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        const_off=const_off_target,
        masks_off=masks_off_target,
        train_idx=train_idx,
        val_idx=val_idx,
        device=device,
        num_workers=int(args.num_workers),
        cfg=cfg_reco_b,
        rho=float(rho),
        drop_prob=float(target_drop_prob_for_reco_b),
        drop_num_banks=int(args.target_drop_num_banks),
        drop_bank_cycle_epochs=int(args.target_drop_bank_cycle_epochs),
        seed=int(args.seed),
        ratio_eps=float(args.recoB_ratio_count_eps),
        ratio_under_lambda=float(args.recoB_ratio_count_under_lambda),
        ratio_over_lambda=float(args.recoB_ratio_count_over_lambda),
        ratio_margin_base=float(args.recoB_ratio_count_over_margin_base),
        ratio_margin_scale=float(args.recoB_ratio_count_over_margin_scale),
        ratio_gamma=float(args.recoB_ratio_count_over_ratio_gamma),
        ratio_over_floor=float(args.recoB_ratio_count_over_lambda_floor),
        target_mode=str(target_mode),
        max_constits=int(args.max_constits),
        concat_offline_token_weight=float(args.recoB_concat_offline_token_weight),
        concat_hlt_token_weight=float(args.recoB_concat_hlt_token_weight),
        concat_added_min_frac=float(args.recoB_concat_added_min_frac),
        concat_added_min_hinge_lambda=float(args.recoB_concat_added_min_hinge_lambda),
        recoB_reload_best_at_stage_transition=not bool(args.disable_recoB_stagewise_best_reload),
        use_ratio_budget=not bool(args.disable_recoB_ratio_budget),
        strict_m2_budget=bool(strict_m2_budget),
        feature_ablation_mode=str(feature_ablation_mode),
    )

    print("\n" + "=" * 70)
    print("STEP 4: DUALVIEW TAGGER (FROZEN RECO-A + RECO-B)")
    print("=" * 70)

    feat_a_train, mask_a_train = b.build_corrected_view_numpy(
        reconstructor=reco_a,
        feat_hlt=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_b_train, mask_b_train = m2mod.build_corrected_view_numpy(
        reconstructor=reco_b,
        feat_hlt=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=False,
    )

    feat_b_train = apply_feature_ablation_to_corrected_np(feat_b_train, mask_b_train, str(feature_ablation_mode))

    feat_a_val, mask_a_val = b.build_corrected_view_numpy(
        reconstructor=reco_a,
        feat_hlt=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_b_val, mask_b_val = m2mod.build_corrected_view_numpy(
        reconstructor=reco_b,
        feat_hlt=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=False,
    )

    feat_b_val = apply_feature_ablation_to_corrected_np(feat_b_val, mask_b_val, str(feature_ablation_mode))

    feat_a_test, mask_a_test = b.build_corrected_view_numpy(
        reconstructor=reco_a,
        feat_hlt=feat_hlt_std[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_hlt=hlt_const[test_idx],
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_b_test, mask_b_test = m2mod.build_corrected_view_numpy(
        reconstructor=reco_b,
        feat_hlt=feat_hlt_std[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_hlt=hlt_const[test_idx],
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=False,
    )

    feat_b_test = apply_feature_ablation_to_corrected_np(feat_b_test, mask_b_test, str(feature_ablation_mode))

    ds_train_dual = DualViewJetDataset(feat_a_train, mask_a_train, feat_b_train, mask_b_train, labels[train_idx])
    ds_val_dual = DualViewJetDataset(feat_a_val, mask_a_val, feat_b_val, mask_b_val, labels[val_idx])
    ds_test_dual = DualViewJetDataset(feat_a_test, mask_a_test, feat_b_test, mask_b_test, labels[test_idx])

    dl_train_dual = DataLoader(ds_train_dual, batch_size=int(args.dual_frozen_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_dual = DataLoader(ds_val_dual, batch_size=int(args.dual_frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_dual = DataLoader(ds_test_dual, batch_size=int(args.dual_frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))

    dual = b.DualViewCrossAttnClassifier(input_dim_a=10, input_dim_b=10, **cfg["model"]).to(device)
    dual, dual_frozen_train_metrics = train_dual_frozen(
        model=dual,
        train_loader=dl_train_dual,
        val_loader=dl_val_dual,
        device=device,
        epochs=int(args.dual_frozen_epochs),
        patience=int(args.dual_frozen_patience),
        lr=float(args.dual_frozen_lr),
        weight_decay=float(args.dual_frozen_weight_decay),
        warmup_epochs=int(args.dual_frozen_warmup_epochs),
        lambda_rank=float(args.dual_frozen_lambda_rank),
        rank_tau=float(args.dual_frozen_rank_tau),
        target_tpr=float(args.report_target_tpr),
        select_metric=str(args.select_metric),
    )

    auc_dual_frozen_val, preds_dual_frozen_val, labs_dual_val, fpr50_dual_frozen_val = eval_dual_model(dual, dl_val_dual, device)
    auc_dual_frozen_test, preds_dual_frozen_test, labs_dual_test, fpr50_dual_frozen_test = eval_dual_model(dual, dl_test_dual, device)

    print(
        f"DualFrozen: val_auc={auc_dual_frozen_val:.4f}, val_fpr50={fpr50_dual_frozen_val:.6f} | "
        f"test_auc={auc_dual_frozen_test:.4f}, test_fpr50={fpr50_dual_frozen_test:.6f}"
    )

    print("\n" + "=" * 70)
    print("STEP 5: DUALVIEW JOINT FINETUNE (UNFREEZE RECO-A + RECO-B)")
    print("=" * 70)

    # fixed bank-0 targets for branch-B anchor during joint
    c_tr_b0, m_tr_b0, bm_tr_b0, be_tr_b0 = build_masked_targets_for_indices(
        const_off=const_off_target,
        masks_off=masks_off_target,
        hlt_mask=hlt_mask,
        indices=train_idx,
        drop_prob=float(target_drop_prob_for_reco_b),
        seed=int(args.seed),
        bank=0,
        rho=float(rho),
        strict_m2_budget=bool(strict_m2_budget),
    )
    c_val_b0, m_val_b0, bm_val_b0, be_val_b0 = build_masked_targets_for_indices(
        const_off=const_off_target,
        masks_off=masks_off_target,
        hlt_mask=hlt_mask,
        indices=val_idx,
        drop_prob=float(target_drop_prob_for_reco_b),
        seed=int(args.seed),
        bank=0,
        rho=float(rho),
        strict_m2_budget=bool(strict_m2_budget),
    )
    c_test_b0, m_test_b0, bm_test_b0, be_test_b0 = build_masked_targets_for_indices(
        const_off=const_off_target,
        masks_off=masks_off_target,
        hlt_mask=hlt_mask,
        indices=test_idx,
        drop_prob=float(target_drop_prob_for_reco_b),
        seed=int(args.seed),
        bank=0,
        rho=float(rho),
        strict_m2_budget=bool(strict_m2_budget),
    )

    ds_train_joint = JointTwoRecoDataset(
        feat_hlt=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        labels=labels[train_idx],
        const_off_full=const_off_stageA[train_idx],
        mask_off_full=masks_off_stageA[train_idx],
        budget_merge_full=budget_merge_true_stageA[train_idx],
        budget_eff_full=budget_eff_true_stageA[train_idx],
        const_off_b=c_tr_b0,
        mask_off_b=m_tr_b0,
        budget_merge_b=bm_tr_b0,
        budget_eff_b=be_tr_b0,
    )
    ds_val_joint = JointTwoRecoDataset(
        feat_hlt=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        labels=labels[val_idx],
        const_off_full=const_off_stageA[val_idx],
        mask_off_full=masks_off_stageA[val_idx],
        budget_merge_full=budget_merge_true_stageA[val_idx],
        budget_eff_full=budget_eff_true_stageA[val_idx],
        const_off_b=c_val_b0,
        mask_off_b=m_val_b0,
        budget_merge_b=bm_val_b0,
        budget_eff_b=be_val_b0,
    )
    ds_test_joint = JointTwoRecoDataset(
        feat_hlt=feat_hlt_std[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_hlt=hlt_const[test_idx],
        labels=labels[test_idx],
        const_off_full=const_off_stageA[test_idx],
        mask_off_full=masks_off_stageA[test_idx],
        budget_merge_full=budget_merge_true_stageA[test_idx],
        budget_eff_full=budget_eff_true_stageA[test_idx],
        const_off_b=c_test_b0,
        mask_off_b=m_test_b0,
        budget_merge_b=bm_test_b0,
        budget_eff_b=be_test_b0,
    )

    dl_train_joint = DataLoader(ds_train_joint, batch_size=int(args.dual_joint_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_joint = DataLoader(ds_val_joint, batch_size=int(args.dual_joint_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_joint = DataLoader(ds_test_joint, batch_size=int(args.dual_joint_batch_size), shuffle=False, num_workers=int(args.num_workers))

    dual_joint_train_metrics: Dict[str, float] = {}
    auc_dual_joint_val = float("nan")
    fpr50_dual_joint_val = float("nan")
    auc_dual_joint_test = float("nan")
    fpr50_dual_joint_test = float("nan")
    preds_dual_joint_val = np.zeros(0, dtype=np.float64)
    preds_dual_joint_test = np.zeros(0, dtype=np.float64)

    if int(args.dual_joint_epochs) > 0:
        dual_joint_train_metrics = train_dual_joint_two_reco(
            reco_a=reco_a,
            reco_b=reco_b,
            dual=dual,
            train_loader=dl_train_joint,
            val_loader=dl_val_joint,
            teacher_model=teacher,
            means=means,
            stds=stds,
            device=device,
            epochs=int(args.dual_joint_epochs),
            patience=int(args.dual_joint_patience),
            lr_dual=float(args.dual_joint_lr_dual),
            lr_reco_a=float(args.dual_joint_lr_reco_a),
            lr_reco_b=float(args.dual_joint_lr_reco_b),
            weight_decay=float(args.dual_joint_weight_decay),
            warmup_epochs=int(args.dual_joint_warmup_epochs),
            lambda_rank=float(args.dual_joint_lambda_rank),
            rank_tau=float(args.dual_joint_rank_tau),
            corrected_weight_floor=float(args.corrected_weight_floor),
            select_metric=str(args.select_metric),
            lambda_anchor_a=float(args.dual_joint_lambda_anchor_a),
            lambda_anchor_b=float(args.dual_joint_lambda_anchor_b),
            stageA_lambda_kd=float(args.stageA_lambda_kd),
            stageA_lambda_emb=float(args.stageA_lambda_emb),
            stageA_lambda_tok=float(args.stageA_lambda_tok),
            stageA_lambda_phys=float(args.stageA_lambda_phys),
            stageA_lambda_budget_hinge=float(args.stageA_lambda_budget_hinge),
            stageA_budget_eps=float(args.stageA_budget_eps),
            stageA_budget_weight_floor=float(args.stageA_budget_weight_floor),
            recoB_ratio_eps=float(args.recoB_ratio_count_eps),
            recoB_ratio_under_lambda=float(args.recoB_ratio_count_under_lambda),
            recoB_ratio_over_lambda=float(args.recoB_ratio_count_over_lambda),
            recoB_ratio_margin_base=float(args.recoB_ratio_count_over_margin_base),
            recoB_ratio_margin_scale=float(args.recoB_ratio_count_over_margin_scale),
            recoB_ratio_gamma=float(args.recoB_ratio_count_over_ratio_gamma),
            recoB_ratio_over_floor=float(args.recoB_ratio_count_over_lambda_floor),
            recoB_loss_cfg=cfg_reco_b["loss"],
            target_mode=str(target_mode),
            max_constits=int(args.max_constits),
            concat_offline_token_weight=float(args.recoB_concat_offline_token_weight),
            concat_hlt_token_weight=float(args.recoB_concat_hlt_token_weight),
            concat_added_min_frac=float(args.recoB_concat_added_min_frac),
            concat_added_min_hinge_lambda=float(args.recoB_concat_added_min_hinge_lambda),
            recoB_use_ratio_budget=not bool(args.disable_recoB_ratio_budget),
            feature_ablation_mode=str(feature_ablation_mode),
        )

        auc_dual_joint_val, preds_dual_joint_val, _, fpr50_dual_joint_val = eval_dual_joint_dynamic(
            reco_a=reco_a,
            reco_b=reco_b,
            dual=dual,
            loader=dl_val_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            feature_ablation_mode=str(feature_ablation_mode),
        )
        auc_dual_joint_test, preds_dual_joint_test, _, fpr50_dual_joint_test = eval_dual_joint_dynamic(
            reco_a=reco_a,
            reco_b=reco_b,
            dual=dual,
            loader=dl_test_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            feature_ablation_mode=str(feature_ablation_mode),
        )

        print(
            f"DualJoint: val_auc={auc_dual_joint_val:.4f}, val_fpr50={fpr50_dual_joint_val:.6f} | "
            f"test_auc={auc_dual_joint_test:.4f}, test_fpr50={fpr50_dual_joint_test:.6f}"
        )

    np.savez_compressed(
        save_root / "dualreco_dualview_scores.npz",
        labels_val=labels[val_idx].astype(np.float32),
        labels_test=labels[test_idx].astype(np.float32),
        preds_hlt_val=preds_hlt_val.astype(np.float64),
        preds_hlt_test=preds_hlt_test.astype(np.float64),
        preds_teacher_val=preds_teacher_val.astype(np.float64),
        preds_teacher_test=preds_teacher_test.astype(np.float64),
        preds_dual_frozen_val=preds_dual_frozen_val.astype(np.float64),
        preds_dual_frozen_test=preds_dual_frozen_test.astype(np.float64),
        preds_dual_joint_val=preds_dual_joint_val.astype(np.float64),
        preds_dual_joint_test=preds_dual_joint_test.astype(np.float64),
    )

    out_json = {
        "variant": "m9_dualreco_dualview_offdrop",
        "rho": float(rho),
        "teacher": {
            "auc_val": float(auc_teacher_val),
            "auc_test": float(auc_teacher_test),
        },
        "hlt": {
            "auc_val": float(auc_hlt_val),
            "auc_test": float(auc_hlt_test),
            "delta_ref_threshold_prob": float(hlt_thr_prob),
            "delta_ref_val_tpr": float(hlt_thr_tpr),
            "delta_ref_val_fpr": float(hlt_thr_fpr),
        },
        "recoA_stageA": reco_a_val_metrics,
        "recoB_stageA": reco_b_metrics,
        "dual_frozen_train": dual_frozen_train_metrics,
        "dual_frozen_eval": {
            "auc_val": float(auc_dual_frozen_val),
            "fpr50_val": float(fpr50_dual_frozen_val),
            "auc_test": float(auc_dual_frozen_test),
            "fpr50_test": float(fpr50_dual_frozen_test),
        },
        "dual_joint_train": dual_joint_train_metrics,
        "dual_joint_eval": {
            "auc_val": float(auc_dual_joint_val),
            "fpr50_val": float(fpr50_dual_joint_val),
            "auc_test": float(auc_dual_joint_test),
            "fpr50_test": float(fpr50_dual_joint_test),
        },
        "teacher_mode": (
            "anti_overlap" if bool(args.teacher_use_anti_overlap) else (
                "offline_dropout" if bool(args.teacher_use_offline_dropout) else "standard"
            )
        ),
        "teacher_dropout": {
            "enabled": bool(args.teacher_use_offline_dropout),
            "drop_prob_max": float(args.teacher_drop_prob_max),
            "drop_mode": str(args.teacher_drop_mode),
            "drop_num_banks": int(args.teacher_drop_num_banks),
            "drop_bank_cycle_epochs": int(args.teacher_drop_bank_cycle_epochs),
        },
        "teacher_anti_overlap": {
            "enabled": bool(args.teacher_use_anti_overlap),
            "lambda": float(args.teacher_anti_lambda),
            "tau": float(args.teacher_anti_tau),
            "beta": float(args.teacher_anti_beta),
            "warmup_epochs": int(args.teacher_anti_warmup_epochs),
        },
        "target_mask": {
            "target_mode": str(target_mode),
            "drop_prob_max": float(target_drop_prob_for_reco_b),
            "drop_num_banks": int(args.target_drop_num_banks),
            "drop_bank_cycle_epochs": int(args.target_drop_bank_cycle_epochs),
            "offline_target_topk_pt": int(topk_target),
            "feature_ablation": str(feature_ablation_mode),
            "max_concat_constits": int(max_concat_constits),
            "concat_offline_token_weight": float(args.recoB_concat_offline_token_weight),
            "concat_hlt_token_weight": float(args.recoB_concat_hlt_token_weight),
            "concat_added_min_frac": float(args.recoB_concat_added_min_frac),
            "concat_added_min_hinge_lambda": float(args.recoB_concat_added_min_hinge_lambda),
        },
        "recoB_use_ratio_budget": bool(not args.disable_recoB_ratio_budget),
        "recoB_strict_m2_mode": bool(args.recoB_strict_m2_mode),
        "recoB_ratio_count_budget": {
            "eps": float(args.recoB_ratio_count_eps),
            "under_lambda": float(args.recoB_ratio_count_under_lambda),
            "over_lambda": float(args.recoB_ratio_count_over_lambda),
            "over_margin_base": float(args.recoB_ratio_count_over_margin_base),
            "over_margin_scale": float(args.recoB_ratio_count_over_margin_scale),
            "over_ratio_gamma": float(args.recoB_ratio_count_over_ratio_gamma),
            "over_lambda_floor": float(args.recoB_ratio_count_over_lambda_floor),
        },
    }
    with open(save_root / "dualreco_dualview_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": float(auc_teacher_test)}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": float(auc_hlt_test)}, save_root / "baseline.pt")
        torch.save({"model": reco_a.state_dict(), "val": reco_a_val_metrics}, save_root / "offline_reconstructor_A_stageA.pt")
        torch.save({"model": reco_b.state_dict(), "val": reco_b_metrics}, save_root / "offline_reconstructor_B_stageA.pt")
        torch.save({"model": dual.state_dict(), "val": dual_frozen_train_metrics}, save_root / "dualview_frozen.pt")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"HLT AUC (val/test): {auc_hlt_val:.4f} / {auc_hlt_test:.4f}\n"
        f"Teacher AUC (val/test): {auc_teacher_val:.4f} / {auc_teacher_test:.4f}\n"
        f"DualFrozen AUC (val/test): {auc_dual_frozen_val:.4f} / {auc_dual_frozen_test:.4f}\n"
        f"DualJoint AUC (val/test): {auc_dual_joint_val:.4f} / {auc_dual_joint_test:.4f}\n"
        f"FPR@50 HLT / Teacher / DualFrozen / DualJoint (test): "
        f"{auc_and_fpr(labels[test_idx], preds_hlt_test, 0.50)[1]:.6f} / "
        f"{auc_and_fpr(labels[test_idx], preds_teacher_test, 0.50)[1]:.6f} / "
        f"{fpr50_dual_frozen_test:.6f} / {fpr50_dual_joint_test:.6f}"
    )

    print(f"\nSaved dual-reco dualview results to: {save_root}")


if __name__ == "__main__":
    main()

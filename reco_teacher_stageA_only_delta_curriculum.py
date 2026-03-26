#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StageA-only RecoTeacher training with:
- Teacher-guided reconstruction losses (s09-style weights supported)
- Stage-scale curriculum (0.35 -> 0.70 -> 1.00)
- Phase-boundary reload of best val checkpoint including model+optimizer+scheduler+EMA
- Optional complement loss L_delta against frozen HLT baseline

Outputs include val/test score arrays for downstream fusion analysis.
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


def _teacher_ablation_zero_indices(mode: str) -> List[int]:
    m = str(mode).lower()
    if m == "none":
        return []
    if m == "no_angle":
        return [0, 1, 6]
    if m == "no_scale":
        return [2, 3, 4, 5]
    if m == "core_shape":
        return [2, 3, 5]
    raise ValueError(f"Unknown teacher_feature_ablation: {mode}")


def apply_teacher_feature_ablation_np(feat: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    idxs = _teacher_ablation_zero_indices(mode)
    if len(idxs) == 0:
        return feat
    out = feat.copy()
    out[:, :, idxs] = 0.0
    out[~mask] = 0.0
    return out.astype(np.float32)


def apply_teacher_feature_ablation_torch(feat: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    idxs = _teacher_ablation_zero_indices(mode)
    if len(idxs) == 0:
        return feat
    out = feat.clone()
    out[:, :, idxs] = 0.0
    out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
    return out


def apply_offline_topk_target_mask_by_pt(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    k = int(max(k, 0))
    n, l, _ = const_off.shape
    if k <= 0 or k >= l:
        return const_off.copy().astype(np.float32), mask_off.copy().astype(bool)

    pt = const_off[:, :, 0]
    masked_pt = np.where(mask_off, pt, -np.inf)
    topk_idx = np.argpartition(-masked_pt, kth=k - 1, axis=1)[:, :k]

    new_mask = np.zeros_like(mask_off, dtype=bool)
    row_idx = np.broadcast_to(np.arange(n)[:, None], topk_idx.shape)
    valid_sel = mask_off[row_idx, topk_idx]
    new_mask[row_idx[valid_sel], topk_idx[valid_sel]] = True

    new_const = const_off.copy()
    new_const[~new_mask] = 0.0
    return new_const.astype(np.float32), new_mask


class TeacherAntiOverlapDataset(Dataset):
    def __init__(
        self,
        feat_off: np.ndarray,
        mask_off: np.ndarray,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_off = torch.tensor(feat_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_off": self.feat_off[i],
            "mask_off": self.mask_off[i],
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "label": self.labels[i],
        }


def train_single_view_teacher_anti_overlap(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader_off: DataLoader,
    hlt_model: nn.Module,
    hlt_threshold_prob: float,
    device: torch.device,
    train_cfg: Dict,
    target_tpr: float,
    anti_lambda: float,
    anti_tau: float,
    anti_beta: float,
    anti_warmup_epochs: int,
    name: str = "TeacherAntiOverlap",
) -> nn.Module:
    anti_lambda = float(max(anti_lambda, 0.0))
    anti_tau = float(max(anti_tau, 1e-6))
    anti_beta = float(max(anti_beta, 1e-6))
    anti_warmup_epochs = int(max(1, anti_warmup_epochs))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = b.get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    hlt_model.eval()
    for p in hlt_model.parameters():
        p.requires_grad_(False)

    thr_prob = float(np.clip(hlt_threshold_prob, 1e-6, 1.0 - 1e-6))
    thr_logit = float(np.log(thr_prob / (1.0 - thr_prob)))

    best_val_auc = float("-inf")
    best_state = None
    no_improve = 0

    for ep in range(int(train_cfg["epochs"])):
        model.train()
        tr_loss = tr_bce = tr_anti = 0.0
        n_tr = 0

        lam_ep = anti_lambda * min(1.0, float(ep + 1) / float(anti_warmup_epochs))

        for batch in train_loader:
            feat_off = batch["feat_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            labels_batch = batch["label"].to(device).float()

            opt.zero_grad()

            logits_t = model(feat_off, mask_off).squeeze(1)
            with torch.no_grad():
                logits_h = hlt_model(feat_hlt, mask_hlt).squeeze(1)

            loss_bce = F.binary_cross_entropy_with_logits(logits_t, labels_batch)

            p_hlt = torch.sigmoid((logits_h - thr_logit) / anti_tau)
            p_t = torch.sigmoid((logits_t - thr_logit) / anti_tau)
            w_band = torch.exp(-torch.abs(logits_h - thr_logit) / anti_beta)
            neg = (1.0 - labels_batch)

            overlap_pen = (neg * w_band * p_hlt * p_t).mean()
            loss = loss_bce + lam_ep * overlap_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = feat_off.size(0)
            tr_loss += float(loss.item()) * bs
            tr_bce += float(loss_bce.item()) * bs
            tr_anti += float(overlap_pen.item()) * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_bce /= max(n_tr, 1)
        tr_anti /= max(n_tr, 1)

        va_auc, va_preds, va_labs = b.eval_classifier(model, val_loader_off, device)
        va_fpr, va_tpr, _ = roc_curve(va_labs, va_preds)
        va_fpr50 = b.fpr_at_target_tpr(va_fpr, va_tpr, float(target_tpr))

        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_loss={tr_loss:.5f} (bce={tr_bce:.5f}, anti={tr_anti:.5f}, lam={lam_ep:.5f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_auc={best_val_auc:.4f}"
            )

        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_teacher_on_soft_reco_split(
    reconstructor: nn.Module,
    teacher: nn.Module,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    labels: np.ndarray,
    split_idx: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    batch_size: int,
    weight_floor: float,
    target_tpr: float,
    teacher_feature_ablation: str = "none",
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    reconstructor.eval()
    teacher.eval()

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(np.clip(stds, 1e-6, None), dtype=torch.float32, device=device)

    preds_list: List[np.ndarray] = []
    labs_list: List[np.ndarray] = []

    idx = split_idx.astype(np.int64)
    for start in range(0, len(idx), int(batch_size)):
        end = min(start + int(batch_size), len(idx))
        sl = idx[start:end]
        x = torch.tensor(feat_hlt_std[sl], dtype=torch.float32, device=device)
        m = torch.tensor(hlt_mask[sl], dtype=torch.bool, device=device)
        c = torch.tensor(hlt_const[sl], dtype=torch.float32, device=device)

        reco_out = reconstructor(x, m, c, stage_scale=1.0)
        feat_reco_t, mask_reco_t = b._build_teacher_reco_features_from_output(
            reco_out,
            c,
            m,
            weight_floor=float(weight_floor),
        )
        feat_reco_std_t = b._standardize_features_torch(feat_reco_t, mask_reco_t, means_t, stds_t)
        feat_reco_std_t = apply_teacher_feature_ablation_torch(feat_reco_std_t, mask_reco_t, teacher_feature_ablation)
        logits = teacher(feat_reco_std_t, mask_reco_t).squeeze(1)
        p = torch.sigmoid(logits)

        preds_list.append(p.detach().cpu().numpy().astype(np.float64))
        labs_list.append(labels[sl].astype(np.float32))

    preds = np.concatenate(preds_list) if preds_list else np.zeros(0, dtype=np.float64)
    labs = np.concatenate(labs_list) if labs_list else np.zeros(0, dtype=np.float32)

    auc = float(roc_auc_score(labs, preds)) if len(np.unique(labs)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(labs, preds)
    fpr_at = float(b.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))
    return auc, preds, labs, fpr_at


def train_reconstructor_teacher_guided_stagewise_delta(
    model: b.OfflineReconstructor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
    teacher_model: nn.Module,
    hlt_model: nn.Module,
    hlt_threshold_prob: float,
    feat_means: np.ndarray,
    feat_stds: np.ndarray,
    kd_temperature: float,
    lambda_kd: float,
    lambda_emb: float,
    lambda_tok: float,
    lambda_phys: float,
    lambda_budget_hinge: float,
    lambda_delta: float,
    delta_tau: float,
    delta_lambda_fp: float,
    budget_eps: float,
    budget_weight_floor: float,
    target_tpr_for_fpr: float,
    normalize_loss_terms: bool,
    loss_norm_ema_decay: float,
    loss_norm_eps: float,
    reload_best_at_stage_transition: bool,
) -> Tuple[b.OfflineReconstructor, Dict[str, object]]:
    epochs_total = int(train_cfg["epochs"])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = b.get_scheduler(opt, int(train_cfg["warmup_epochs"]), epochs_total)

    kd_temperature = max(float(kd_temperature), 1e-3)
    lambda_kd = float(max(lambda_kd, 0.0))
    lambda_emb = float(max(lambda_emb, 0.0))
    lambda_tok = float(max(lambda_tok, 0.0))
    lambda_phys = float(max(lambda_phys, 0.0))
    lambda_budget_hinge = float(max(lambda_budget_hinge, 0.0))
    lambda_delta = float(max(lambda_delta, 0.0))
    delta_tau = float(max(delta_tau, 1e-6))
    delta_lambda_fp = float(max(delta_lambda_fp, 0.0))
    budget_eps = float(max(budget_eps, 0.0))
    budget_weight_floor = float(max(budget_weight_floor, 0.0))
    loss_norm_ema_decay = float(np.clip(loss_norm_ema_decay, 0.0, 0.9999))
    loss_norm_eps = float(max(loss_norm_eps, 1e-12))

    means_t = torch.tensor(feat_means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(np.clip(feat_stds, 1e-6, None), dtype=torch.float32, device=device)

    teacher_model.eval()
    hlt_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    for p in hlt_model.parameters():
        p.requires_grad_(False)

    reco_loss_ema_state = {
        "kd": 1.0,
        "emb": 1.0,
        "tok": 1.0,
        "phys": 1.0,
        "budget": 1.0,
    }

    s1 = int(train_cfg.get("stage1_epochs", max(1, epochs_total // 4)))
    s2 = int(train_cfg.get("stage2_epochs", max(s1 + 1, (2 * epochs_total) // 3)))
    s1 = max(0, min(s1, epochs_total))
    s2 = max(s1, min(s2, epochs_total))
    phase_defs = [
        (0, s1, 0.35, "phase_035"),
        (s1, s2, 0.70, "phase_070"),
        (s2, epochs_total, 1.00, "phase_100"),
    ]
    phase_defs = [p for p in phase_defs if p[0] < p[1]]

    best_global_auc = float("-inf")
    best_global_state = None
    best_global_metrics: Dict[str, object] = {}
    phase_summaries: List[Dict[str, object]] = []

    for p_start, p_end, sc, p_name in phase_defs:
        print("-" * 70)
        print(f"Stage-A curriculum: {p_name} | epochs [{p_start+1},{p_end}] | stage_scale={sc:.2f}")
        print("-" * 70)
        phase_best_auc = float("-inf")
        phase_best_pack = None
        phase_no_improve = 0
        min_phase_epochs = min(5, max(1, p_end - p_start))

        for ep in range(p_start, p_end):
            model.train()

            tr_total = tr_kd = tr_emb = tr_tok = tr_phys = tr_budget_hinge = 0.0
            tr_delta = tr_delta_gain = tr_delta_cost = 0.0
            n_tr = 0
            tr_probs_all = []
            tr_labels_all = []

            for batch in train_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                const_off = batch["const_off"].to(device)
                mask_off = batch["mask_off"].to(device)
                labels_batch = batch["label"].to(device)
                budget_merge_true = batch["budget_merge_true"].to(device)
                budget_eff_true = batch["budget_eff_true"].to(device)

                opt.zero_grad()
                out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=float(sc))

                losses = b._compute_teacher_guided_reco_losses(
                    reco_out=out,
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
                loss_total, _, reco_loss_ema_state = b._compose_teacher_guided_reco_total(
                    losses_raw=losses,
                    ema_state=reco_loss_ema_state,
                    normalize_terms=bool(normalize_loss_terms),
                    ema_decay=loss_norm_ema_decay,
                    norm_eps=loss_norm_eps,
                    w_logit=lambda_kd,
                    w_emb=lambda_emb,
                    w_tok=lambda_tok,
                    w_phys=lambda_phys,
                    w_budget=lambda_budget_hinge,
                    update_ema=True,
                )

                delta_term = torch.zeros((), device=device)
                delta_gain = torch.zeros((), device=device)
                delta_cost = torch.zeros((), device=device)
                if lambda_delta > 0.0:
                    with torch.no_grad():
                        logits_hlt = hlt_model(feat_hlt, mask_hlt).squeeze(1)
                        p_hlt = torch.sigmoid(logits_hlt)
                        h_soft = torch.sigmoid((p_hlt - float(hlt_threshold_prob)) / float(delta_tau))
                    y = labels_batch.float().view(-1)
                    p_reco = torch.sigmoid(losses["logits_teacher_reco"]).view(-1)
                    miss_hlt = (1.0 - h_soft).view(-1)
                    delta_gain = (y * miss_hlt * p_reco).mean()
                    delta_cost = ((1.0 - y) * miss_hlt * p_reco).mean()
                    delta_term = -delta_gain + float(delta_lambda_fp) * delta_cost
                    loss_total = loss_total + float(lambda_delta) * delta_term

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                probs_reco = torch.sigmoid(losses["logits_teacher_reco"]).detach().cpu().numpy()
                tr_probs_all.append(probs_reco)
                tr_labels_all.append(labels_batch.detach().cpu().numpy().astype(np.int64))

                bs = feat_hlt.size(0)
                tr_total += loss_total.item() * bs
                tr_kd += losses["kd"].item() * bs
                tr_emb += losses["emb"].item() * bs
                tr_tok += losses["tok"].item() * bs
                tr_phys += losses["phys"].item() * bs
                tr_budget_hinge += losses["budget_hinge"].item() * bs
                tr_delta += delta_term.item() * bs
                tr_delta_gain += delta_gain.item() * bs
                tr_delta_cost += delta_cost.item() * bs
                n_tr += bs

            model.eval()
            va_total = va_kd = va_emb = va_tok = va_phys = va_budget_hinge = 0.0
            va_delta = va_delta_gain = va_delta_cost = 0.0
            n_va = 0
            va_probs_all = []
            va_labels_all = []

            with torch.no_grad():
                for batch in val_loader:
                    feat_hlt = batch["feat_hlt"].to(device)
                    mask_hlt = batch["mask_hlt"].to(device)
                    const_hlt = batch["const_hlt"].to(device)
                    const_off = batch["const_off"].to(device)
                    mask_off = batch["mask_off"].to(device)
                    labels_batch = batch["label"].to(device)
                    budget_merge_true = batch["budget_merge_true"].to(device)
                    budget_eff_true = batch["budget_eff_true"].to(device)

                    out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
                    losses = b._compute_teacher_guided_reco_losses(
                        reco_out=out,
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
                    loss_total, _, _ = b._compose_teacher_guided_reco_total(
                        losses_raw=losses,
                        ema_state=reco_loss_ema_state,
                        normalize_terms=bool(normalize_loss_terms),
                        ema_decay=loss_norm_ema_decay,
                        norm_eps=loss_norm_eps,
                        w_logit=lambda_kd,
                        w_emb=lambda_emb,
                        w_tok=lambda_tok,
                        w_phys=lambda_phys,
                        w_budget=lambda_budget_hinge,
                        update_ema=False,
                    )

                    delta_term = torch.zeros((), device=device)
                    delta_gain = torch.zeros((), device=device)
                    delta_cost = torch.zeros((), device=device)
                    if lambda_delta > 0.0:
                        logits_hlt = hlt_model(feat_hlt, mask_hlt).squeeze(1)
                        p_hlt = torch.sigmoid(logits_hlt)
                        h_soft = torch.sigmoid((p_hlt - float(hlt_threshold_prob)) / float(delta_tau))
                        y = labels_batch.float().view(-1)
                        p_reco = torch.sigmoid(losses["logits_teacher_reco"]).view(-1)
                        miss_hlt = (1.0 - h_soft).view(-1)
                        delta_gain = (y * miss_hlt * p_reco).mean()
                        delta_cost = ((1.0 - y) * miss_hlt * p_reco).mean()
                        delta_term = -delta_gain + float(delta_lambda_fp) * delta_cost
                        loss_total = loss_total + float(lambda_delta) * delta_term

                    probs_reco = torch.sigmoid(losses["logits_teacher_reco"]).detach().cpu().numpy()
                    va_probs_all.append(probs_reco)
                    va_labels_all.append(labels_batch.detach().cpu().numpy().astype(np.int64))

                    bs = feat_hlt.size(0)
                    va_total += loss_total.item() * bs
                    va_kd += losses["kd"].item() * bs
                    va_emb += losses["emb"].item() * bs
                    va_tok += losses["tok"].item() * bs
                    va_phys += losses["phys"].item() * bs
                    va_budget_hinge += losses["budget_hinge"].item() * bs
                    va_delta += delta_term.item() * bs
                    va_delta_gain += delta_gain.item() * bs
                    va_delta_cost += delta_cost.item() * bs
                    n_va += bs

            sch.step()

            tr_total /= max(n_tr, 1)
            tr_kd /= max(n_tr, 1)
            tr_emb /= max(n_tr, 1)
            tr_tok /= max(n_tr, 1)
            tr_phys /= max(n_tr, 1)
            tr_budget_hinge /= max(n_tr, 1)
            tr_delta /= max(n_tr, 1)
            tr_delta_gain /= max(n_tr, 1)
            tr_delta_cost /= max(n_tr, 1)

            va_total /= max(n_va, 1)
            va_kd /= max(n_va, 1)
            va_emb /= max(n_va, 1)
            va_tok /= max(n_va, 1)
            va_phys /= max(n_va, 1)
            va_budget_hinge /= max(n_va, 1)
            va_delta /= max(n_va, 1)
            va_delta_gain /= max(n_va, 1)
            va_delta_cost /= max(n_va, 1)

            tr_probs = np.concatenate(tr_probs_all, axis=0) if tr_probs_all else np.zeros((0,), dtype=np.float32)
            tr_labels = np.concatenate(tr_labels_all, axis=0) if tr_labels_all else np.zeros((0,), dtype=np.int64)
            va_probs = np.concatenate(va_probs_all, axis=0) if va_probs_all else np.zeros((0,), dtype=np.float32)
            va_labels = np.concatenate(va_labels_all, axis=0) if va_labels_all else np.zeros((0,), dtype=np.int64)

            if np.unique(tr_labels).size > 1 and tr_probs.size > 0:
                tr_auc = float(roc_auc_score(tr_labels, tr_probs))
                tr_fpr, tr_tpr, _ = roc_curve(tr_labels, tr_probs)
                tr_fpr50 = float(b.fpr_at_target_tpr(tr_fpr, tr_tpr, float(target_tpr_for_fpr)))
            else:
                tr_auc, tr_fpr50 = float("nan"), float("nan")

            if np.unique(va_labels).size > 1 and va_probs.size > 0:
                va_auc = float(roc_auc_score(va_labels, va_probs))
                va_fpr, va_tpr, _ = roc_curve(va_labels, va_probs)
                va_fpr50 = float(b.fpr_at_target_tpr(va_fpr, va_tpr, float(target_tpr_for_fpr)))
            else:
                va_auc, va_fpr50 = float("nan"), float("nan")

            metrics_ep = {
                "epoch": int(ep + 1),
                "phase": p_name,
                "stage_scale": float(sc),
                "train_total": float(tr_total),
                "train_kd": float(tr_kd),
                "train_emb": float(tr_emb),
                "train_tok": float(tr_tok),
                "train_phys": float(tr_phys),
                "train_budget_hinge": float(tr_budget_hinge),
                "train_delta": float(tr_delta),
                "train_delta_gain": float(tr_delta_gain),
                "train_delta_cost": float(tr_delta_cost),
                "train_teacher_auc": float(tr_auc),
                "train_teacher_fpr50": float(tr_fpr50),
                "val_total": float(va_total),
                "val_kd": float(va_kd),
                "val_emb": float(va_emb),
                "val_tok": float(va_tok),
                "val_phys": float(va_phys),
                "val_budget_hinge": float(va_budget_hinge),
                "val_delta": float(va_delta),
                "val_delta_gain": float(va_delta_gain),
                "val_delta_cost": float(va_delta_cost),
                "val_teacher_auc": float(va_auc),
                "val_teacher_fpr50": float(va_fpr50),
                "loss_normalized": bool(normalize_loss_terms),
                "loss_norm_ema_decay": float(loss_norm_ema_decay),
            }

            if np.isfinite(va_auc) and (va_auc > phase_best_auc):
                phase_best_auc = float(va_auc)
                phase_no_improve = 0
                phase_best_pack = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "opt": copy.deepcopy(opt.state_dict()),
                    "sch": copy.deepcopy(sch.state_dict()),
                    "ema": {k: float(v) for k, v in reco_loss_ema_state.items()},
                    "metrics": copy.deepcopy(metrics_ep),
                }
            else:
                phase_no_improve += 1

            if np.isfinite(va_auc) and (va_auc > best_global_auc):
                best_global_auc = float(va_auc)
                best_global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_global_metrics = copy.deepcopy(metrics_ep)
                best_global_metrics["selected_metric"] = "teacher_on_reco_val_auc"

            if (ep + 1) % 5 == 0:
                print(
                    f"Ep {ep+1} [{p_name}]: train_total={tr_total:.4f}, val_total={va_total:.4f}, "
                    f"train_teacher_auc={tr_auc:.4f}, val_teacher_auc={va_auc:.4f}, "
                    f"val_teacher_fpr50={va_fpr50:.6f}, best_global_auc={best_global_auc:.4f} | "
                    f"kd={va_kd:.4f}, emb={va_emb:.4f}, tok={va_tok:.4f}, phys={va_phys:.4f}, "
                    f"budget_hinge={va_budget_hinge:.4f}, delta={va_delta:.4f}, "
                    f"delta_gain={va_delta_gain:.4f}, delta_cost={va_delta_cost:.4f}, stage_scale={sc:.2f}"
                )

            if (ep - p_start + 1) >= min_phase_epochs and phase_no_improve >= int(train_cfg["patience"]):
                print(f"Early stop within {p_name} at epoch {ep+1} (no_improve={phase_no_improve})")
                break

        if phase_best_pack is not None:
            phase_summary = {
                "phase": p_name,
                "scale": float(sc),
                "epoch_start": int(p_start + 1),
                "epoch_end": int(p_end),
                "best_val_auc": float(phase_best_auc),
                "best_epoch": int(phase_best_pack["metrics"]["epoch"]),
                "best_val_fpr50": float(phase_best_pack["metrics"]["val_teacher_fpr50"]),
            }
            phase_summaries.append(phase_summary)
            if bool(reload_best_at_stage_transition):
                model.load_state_dict(phase_best_pack["model"])
                opt.load_state_dict(phase_best_pack["opt"])
                sch.load_state_dict(phase_best_pack["sch"])
                reco_loss_ema_state = {k: float(v) for k, v in phase_best_pack["ema"].items()}
                print(
                    f"Reloaded best checkpoint for {p_name} "
                    f"(epoch={phase_best_pack['metrics']['epoch']}, val_auc={phase_best_auc:.4f})"
                )

    if best_global_state is not None:
        model.load_state_dict(best_global_state)

    best_global_metrics = dict(best_global_metrics)
    best_global_metrics["phase_summaries"] = phase_summaries
    best_global_metrics["stagewise_best_reload"] = bool(reload_best_at_stage_transition)
    best_global_metrics["delta_enabled"] = bool(lambda_delta > 0.0)
    best_global_metrics["delta_lambda"] = float(lambda_delta)
    best_global_metrics["delta_tau"] = float(delta_tau)
    best_global_metrics["delta_lambda_fp"] = float(delta_lambda_fp)
    best_global_metrics["hlt_threshold_prob"] = float(hlt_threshold_prob)
    return model, best_global_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--n_train_jets", type=int, default=250000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--n_train_split", type=int, default=75000)
    ap.add_argument("--n_val_split", type=int, default=25000)
    ap.add_argument("--n_test_split", type=int, default=150000)
    ap.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_joint_stageAonly"))
    ap.add_argument("--run_name", type=str, default="reco_teacher_stageAonly_delta_75k25k150k_seed0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--skip_save_models", action="store_true")
    ap.add_argument("--seed", type=int, default=b.RANDOM_SEED)

    ap.add_argument("--teacher_use_anti_overlap", action="store_true")
    ap.add_argument("--teacher_anti_lambda", type=float, default=0.01)
    ap.add_argument("--teacher_anti_tau", type=float, default=0.05)
    ap.add_argument("--teacher_anti_beta", type=float, default=0.10)
    ap.add_argument("--teacher_anti_warmup_epochs", type=int, default=12)
    ap.add_argument("--teacher_feature_ablation", type=str, choices=["none", "no_angle", "no_scale", "core_shape"], default="none")
    ap.add_argument(
        "--offline_target_topk_pt",
        type=int,
        default=0,
        help="If >0, keeps only top-k offline constituents by pT for teacher/StageA targets while keeping HLT inputs at full max_constits.",
    )

    ap.add_argument("--merge_radius", type=float, default=b.BASE_CONFIG["hlt_effects"]["merge_radius"])
    ap.add_argument("--eff_plateau_barrel", type=float, default=b.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    ap.add_argument("--eff_plateau_endcap", type=float, default=b.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    ap.add_argument("--smear_a", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_a"])
    ap.add_argument("--smear_b", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_b"])
    ap.add_argument("--smear_c", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_c"])

    ap.add_argument("--stageA_epochs", type=int, default=90)
    ap.add_argument("--stageA_patience", type=int, default=18)
    ap.add_argument("--stageA_kd_temp", type=float, default=2.5)
    ap.add_argument("--stageA_lambda_kd", type=float, default=5.0)
    ap.add_argument("--stageA_lambda_emb", type=float, default=0.0)
    ap.add_argument("--stageA_lambda_tok", type=float, default=0.0)
    ap.add_argument("--stageA_lambda_phys", type=float, default=0.05)
    ap.add_argument("--stageA_lambda_budget_hinge", type=float, default=1.0)
    ap.add_argument("--stageA_budget_eps", type=float, default=0.015)
    ap.add_argument("--stageA_budget_weight_floor", type=float, default=1e-4)
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
    ap.add_argument("--train_corrected_only_post_stageA", action="store_true")

    ap.add_argument("--report_target_tpr", type=float, default=0.50)
    ap.add_argument("--combo_weight_step", type=float, default=0.01)
    args = ap.parse_args()

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
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off_full = const_raw.copy()
    const_off_full[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, _budget_truth = b.apply_hlt_effects_realistic_nomap(
        const_off_full,
        masks_off,
        cfg,
        seed=int(args.seed),
    )

    const_off = const_off_full
    topk_target = int(args.offline_target_topk_pt)
    if 0 < topk_target < int(args.max_constits):
        const_off, masks_off = apply_offline_topk_target_mask_by_pt(
            const_off_full,
            masks_off,
            topk_target,
        )
        print(
            f"Applying offline top-k target mask: k={topk_target} "
            f"(teacher/StageA targets only); HLT input max_constits stays {int(args.max_constits)}."
        )
    elif topk_target >= int(args.max_constits):
        print(
            f"offline_target_topk_pt={topk_target} >= max_constits={int(args.max_constits)}; "
            "disabling top-k target mask."
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

    feat_off_teacher_std = apply_teacher_feature_ablation_np(feat_off_std, masks_off, args.teacher_feature_ablation)

    data_setup = {
        "train_path_arg": str(args.train_path),
        "train_files": [str(p.resolve()) for p in train_files],
        "n_train_jets": int(args.n_train_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "offline_target_topk_pt": int(topk_target),
        "seed": int(args.seed),
        "split": {
            "mode": "custom_counts",
            "n_train_split": int(len(train_idx)),
            "n_val_split": int(len(val_idx)),
            "n_test_split": int(len(test_idx)),
        },
        "hlt_effects": cfg["hlt_effects"],
        "variant": "stageA_only_recoteacher_delta",
        "rho": float(rho),
        "mean_true_added_raw": float(true_added_raw.mean()),
        "mean_target_merge": float(budget_merge_true.mean()),
        "mean_target_eff": float(budget_eff_true.mean()),
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

    ds_train_off = b.JetDataset(feat_off_teacher_std[train_idx], masks_off[train_idx], labels[train_idx])
    ds_val_off = b.JetDataset(feat_off_teacher_std[val_idx], masks_off[val_idx], labels[val_idx])
    ds_test_off = b.JetDataset(feat_off_teacher_std[test_idx], masks_off[test_idx], labels[test_idx])
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
    baseline = b.train_single_view_classifier_auc(
        baseline, dl_train_hlt, dl_val_hlt, device, cfg["training"], name="Baseline"
    )
    auc_baseline, preds_baseline, labs_test_hlt = b.eval_classifier(baseline, dl_test_hlt, device)
    auc_baseline_val, preds_baseline_val, labs_val_baseline = b.eval_classifier(baseline, dl_val_hlt, device)

    hlt_thr_prob, hlt_thr_tpr, hlt_thr_fpr = threshold_at_target_tpr(
        labs_val_baseline.astype(np.float32),
        preds_baseline_val.astype(np.float64),
        float(args.stageA_target_tpr),
    )
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
        ds_train_teacher_anti = TeacherAntiOverlapDataset(
            feat_off=feat_off_std[train_idx],
            mask_off=masks_off[train_idx],
            feat_hlt=feat_hlt_std[train_idx],
            mask_hlt=hlt_mask[train_idx],
            labels=labels[train_idx],
        )
        dl_train_teacher_anti = DataLoader(ds_train_teacher_anti, batch_size=BS, shuffle=True, drop_last=True)
        teacher = train_single_view_teacher_anti_overlap(
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
    else:
        teacher = b.train_single_view_classifier_auc(
            teacher, dl_train_off, dl_val_off, device, cfg["training"], name="Teacher"
        )

    auc_teacher, preds_teacher, labs_test_teacher = b.eval_classifier(teacher, dl_test_off, device)
    auc_teacher_val, preds_teacher_val, labs_val_teacher = b.eval_classifier(teacher, dl_val_off, device)

    assert np.array_equal(labs_val_teacher.astype(np.float32), labs_val_baseline.astype(np.float32))
    assert np.array_equal(labs_test_teacher.astype(np.float32), labs_test_hlt.astype(np.float32))

    print("\n" + "=" * 70)
    print("STEP 2: STAGE A (RECONSTRUCTOR PRETRAIN, STAGEWISE BEST RELOAD)")
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
    reconstructor, reco_val_metrics = train_reconstructor_teacher_guided_stagewise_delta(
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
    print("STEP 3: STAGE A SOFT-RECO EVALUATION")
    print("=" * 70)
    auc_reco_teacher_val, preds_reco_teacher_val, labs_reco_val, fpr50_reco_teacher_val = eval_teacher_on_soft_reco_split(
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
        teacher_feature_ablation=str(args.teacher_feature_ablation),
    )
    auc_reco_teacher_test, preds_reco_teacher_test, labs_reco_test, fpr50_reco_teacher_test = eval_teacher_on_soft_reco_split(
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
        teacher_feature_ablation=str(args.teacher_feature_ablation),
    )

    assert np.array_equal(labs_reco_val.astype(np.float32), labs_val_teacher.astype(np.float32))
    assert np.array_equal(labs_reco_test.astype(np.float32), labs_test_teacher.astype(np.float32))

    corrected_only = None
    auc_corr_val = float("nan")
    auc_corr_test = float("nan")
    preds_corr_val = None
    preds_corr_test = None
    fpr50_corr_val = float("nan")
    fpr50_corr_test = float("nan")
    combo_hlt_corr_valsel = None
    combo_hlt_corr_oracle = None

    if bool(args.train_corrected_only_post_stageA):
        print("\n" + "=" * 70)
        print("STEP 4: CORRECTED-ONLY TAGGER (FROZEN STAGE-A RECONSTRUCTOR)")
        print("=" * 70)
        feat_corr_all, mask_corr_all = b.build_corrected_view_numpy(
            reconstructor=reconstructor,
            feat_hlt=feat_hlt_std,
            mask_hlt=hlt_mask,
            const_hlt=hlt_const,
            device=device,
            batch_size=int(BS),
            corrected_weight_floor=float(args.reco_weight_threshold),
            corrected_use_flags=False,
        )

        ds_train_corr = b.JetDataset(feat_corr_all[train_idx], mask_corr_all[train_idx], labels[train_idx])
        ds_val_corr = b.JetDataset(feat_corr_all[val_idx], mask_corr_all[val_idx], labels[val_idx])
        ds_test_corr = b.JetDataset(feat_corr_all[test_idx], mask_corr_all[test_idx], labels[test_idx])
        dl_train_corr = DataLoader(ds_train_corr, batch_size=BS, shuffle=True, drop_last=True)
        dl_val_corr = DataLoader(ds_val_corr, batch_size=BS, shuffle=False)
        dl_test_corr = DataLoader(ds_test_corr, batch_size=BS, shuffle=False)

        corrected_only = b.ParticleTransformer(input_dim=int(feat_corr_all.shape[-1]), **cfg["model"]).to(device)
        corrected_only = b.train_single_view_classifier_auc(
            corrected_only,
            dl_train_corr,
            dl_val_corr,
            device,
            cfg["training"],
            name="CorrectedOnly-PostStageA",
        )
        auc_corr_val, preds_corr_val, labs_corr_val = b.eval_classifier(corrected_only, dl_val_corr, device)
        auc_corr_test, preds_corr_test, labs_corr_test = b.eval_classifier(corrected_only, dl_test_corr, device)

        assert np.array_equal(labs_corr_val.astype(np.float32), labs_val_teacher.astype(np.float32))
        assert np.array_equal(labs_corr_test.astype(np.float32), labs_test_teacher.astype(np.float32))

        fpr_corr_v, tpr_corr_v, _ = roc_curve(labs_corr_val, preds_corr_val)
        fpr_corr_t, tpr_corr_t, _ = roc_curve(labs_corr_test, preds_corr_test)
        fpr50_corr_val = float(b.fpr_at_target_tpr(fpr_corr_v, tpr_corr_v, 0.50))
        fpr50_corr_test = float(b.fpr_at_target_tpr(fpr_corr_t, tpr_corr_t, 0.50))

        combo_hlt_corr_valsel = b.select_weighted_combo_on_val_and_eval_test(
            labels_val=labs_val_teacher.astype(np.float32),
            preds_a_val=preds_baseline_val,
            preds_b_val=preds_corr_val,
            labels_test=labs_test_teacher.astype(np.float32),
            preds_a_test=preds_baseline,
            preds_b_test=preds_corr_test,
            name_a="hlt",
            name_b="corrected_only",
            target_tpr=float(args.report_target_tpr),
            weight_step=float(args.combo_weight_step),
        )
        combo_hlt_corr_oracle = b.search_best_weighted_combo_at_tpr(
            labels=labs_test_teacher.astype(np.float32),
            preds_a=preds_baseline,
            preds_b=preds_corr_test,
            name_a="hlt",
            name_b="corrected_only",
            target_tpr=float(args.report_target_tpr),
            weight_step=float(args.combo_weight_step),
        )

    overlap_model_preds = {
        "hlt": preds_baseline,
        "reco_teacher": preds_reco_teacher_test,
    }
    if preds_corr_test is not None:
        overlap_model_preds["corrected_only"] = preds_corr_test

    overlap_report = b.build_overlap_report_at_tpr(
        labels=labs_reco_test.astype(np.float32),
        model_preds=overlap_model_preds,
        target_tpr=float(args.report_target_tpr),
    )
    best_combo_valsel = b.select_weighted_combo_on_val_and_eval_test(
        labels_val=labs_val_teacher.astype(np.float32),
        preds_a_val=preds_baseline_val,
        preds_b_val=preds_reco_teacher_val,
        labels_test=labs_reco_test.astype(np.float32),
        preds_a_test=preds_baseline,
        preds_b_test=preds_reco_teacher_test,
        name_a="hlt",
        name_b="reco_teacher",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )
    best_combo_test_oracle = b.search_best_weighted_combo_at_tpr(
        labels=labs_reco_test.astype(np.float32),
        preds_a=preds_baseline,
        preds_b=preds_reco_teacher_test,
        name_a="hlt",
        name_b="reco_teacher",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )

    fpr_t, tpr_t, _ = roc_curve(labs_test_teacher, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs_test_teacher, preds_baseline)
    fpr_r, tpr_r, _ = roc_curve(labs_reco_test, preds_reco_teacher_test)
    fpr50_teacher = float(b.fpr_at_target_tpr(fpr_t, tpr_t, 0.50))
    fpr50_hlt = float(b.fpr_at_target_tpr(fpr_b, tpr_b, 0.50))
    fpr50_reco = float(b.fpr_at_target_tpr(fpr_r, tpr_r, 0.50))

    print("\n" + "=" * 70)
    print("FINAL STAGE-A ONLY EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"RecoTeacher Soft AUC (val/test): {auc_reco_teacher_val:.4f} / {auc_reco_teacher_test:.4f}")
    print(f"FPR@50 Teacher/HLT/RecoTeacherSoft: {fpr50_teacher:.6f} / {fpr50_hlt:.6f} / {fpr50_reco:.6f}")
    if preds_corr_test is not None:
        print(f"CorrectedOnly AUC (val/test): {auc_corr_val:.4f} / {auc_corr_test:.4f}")
        print(f"FPR@50 CorrectedOnly (val/test): {fpr50_corr_val:.6f} / {fpr50_corr_test:.6f}")
    pair = overlap_report["pairs"].get("hlt__reco_teacher", {})
    if not pair:
        pair = overlap_report["pairs"].get("reco_teacher__hlt", {})
    print(
        f"TP overlap @TPR={float(args.report_target_tpr):.2f} (HLT vs RecoTeacher): "
        f"{int(pair.get('overlap_tp_count', 0))} shared TP | "
        f"overlap frac of HLT TP={float(pair.get('overlap_tp_frac_of_a_tp', float('nan'))):.3f}, "
        f"of RecoTeacher TP={float(pair.get('overlap_tp_frac_of_b_tp', float('nan'))):.3f}"
    )
    print(
        f"Best weighted combo @TPR={float(args.report_target_tpr):.2f} (VAL-selected -> TEST): "
        f"w_hlt={best_combo_valsel['test_eval']['w_a']:.3f}, "
        f"w_reco={best_combo_valsel['test_eval']['w_b']:.3f}, "
        f"FPR_test={best_combo_valsel['test_eval']['fpr']:.6f}"
    )
    print(
        f"Best weighted combo @TPR={float(args.report_target_tpr):.2f} (TEST post-hoc): "
        f"w_hlt={best_combo_test_oracle['w_a']:.3f}, "
        f"w_reco={best_combo_test_oracle['w_b']:.3f}, "
        f"FPR={best_combo_test_oracle['fpr']:.6f}"
    )
    if combo_hlt_corr_valsel is not None and combo_hlt_corr_oracle is not None:
        print(
            f"Best weighted combo @TPR={float(args.report_target_tpr):.2f} (HLT+CorrectedOnly, VAL-selected -> TEST): "
            f"w_hlt={combo_hlt_corr_valsel['test_eval']['w_a']:.3f}, "
            f"w_corr={combo_hlt_corr_valsel['test_eval']['w_b']:.3f}, "
            f"FPR_test={combo_hlt_corr_valsel['test_eval']['fpr']:.6f}"
        )
        print(
            f"Best weighted combo @TPR={float(args.report_target_tpr):.2f} (HLT+CorrectedOnly, TEST post-hoc): "
            f"w_hlt={combo_hlt_corr_oracle['w_a']:.3f}, "
            f"w_corr={combo_hlt_corr_oracle['w_b']:.3f}, "
            f"FPR={combo_hlt_corr_oracle['fpr']:.6f}"
        )

    preds_corr_val_out = (
        np.asarray(preds_corr_val, dtype=np.float64)
        if preds_corr_val is not None
        else np.zeros(0, dtype=np.float64)
    )
    preds_corr_test_out = (
        np.asarray(preds_corr_test, dtype=np.float64)
        if preds_corr_test is not None
        else np.zeros(0, dtype=np.float64)
    )

    np.savez_compressed(
        save_root / "stageA_only_scores.npz",
        labels_val=labs_val_teacher.astype(np.float32),
        labels_test=labs_test_teacher.astype(np.float32),
        preds_teacher_val=preds_teacher_val.astype(np.float64),
        preds_teacher_test=preds_teacher.astype(np.float64),
        preds_hlt_val=preds_baseline_val.astype(np.float64),
        preds_hlt_test=preds_baseline.astype(np.float64),
        preds_reco_teacher_val=preds_reco_teacher_val.astype(np.float64),
        preds_reco_teacher_test=preds_reco_teacher_test.astype(np.float64),
        preds_corrected_only_val=preds_corr_val_out,
        preds_corrected_only_test=preds_corr_test_out,
        auc_teacher_val=float(auc_teacher_val),
        auc_teacher_test=float(auc_teacher),
        auc_hlt_val=float(auc_baseline_val),
        auc_hlt_test=float(auc_baseline),
        auc_reco_teacher_val=float(auc_reco_teacher_val),
        auc_reco_teacher_test=float(auc_reco_teacher_test),
        auc_corrected_only_val=float(auc_corr_val),
        auc_corrected_only_test=float(auc_corr_test),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_hlt=float(fpr50_hlt),
        fpr50_reco_teacher=float(fpr50_reco),
        fpr50_corrected_only_val=float(fpr50_corr_val),
        fpr50_corrected_only_test=float(fpr50_corr_test),
        has_corrected_only=bool(preds_corr_test is not None),
        target_tpr=float(args.report_target_tpr),
    )

    with open(save_root / "stageA_only_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": "stageA_only_recoteacher_delta",
                "rho": float(rho),
                "stageA_reconstructor": reco_val_metrics,
                "teacher": {
                    "auc_val": float(auc_teacher_val),
                    "auc_test": float(auc_teacher),
                    "feature_ablation": str(args.teacher_feature_ablation),
                    "anti_overlap": {
                        "enabled": bool(args.teacher_use_anti_overlap),
                        "lambda": float(args.teacher_anti_lambda),
                        "tau": float(args.teacher_anti_tau),
                        "beta": float(args.teacher_anti_beta),
                        "warmup_epochs": int(args.teacher_anti_warmup_epochs),
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
                "corrected_only": {
                    "enabled": bool(preds_corr_test is not None),
                    "auc_val": float(auc_corr_val),
                    "auc_test": float(auc_corr_test),
                    "fpr50_val": float(fpr50_corr_val),
                    "fpr50_test": float(fpr50_corr_test),
                },
                "overlap_report_tpr": overlap_report,
                "best_combo_hlt_reco_val_selected_eval_test": best_combo_valsel,
                "best_combo_hlt_reco_test_posthoc": best_combo_test_oracle,
                "best_combo_hlt_corrected_val_selected_eval_test": combo_hlt_corr_valsel,
                "best_combo_hlt_corrected_test_posthoc": combo_hlt_corr_oracle,
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
        torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor.pt")
        if corrected_only is not None:
            torch.save({"model": corrected_only.state_dict(), "auc": float(auc_corr_test)}, save_root / "corrected_only_tagger.pt")

    print(f"\nSaved Stage-A only RecoTeacher results to: {save_root}")


if __name__ == "__main__":
    main()

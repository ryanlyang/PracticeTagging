#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ConcatTeacher-guided Stage-A reconstructor pipeline:
1) Train HLT baseline and ConcatTeacher (single-view on offline||HLT constituents).
2) Freeze ConcatTeacher and train Stage-A reconstructor with teacher-guided losses.
   - Physics/budget supervision stays anchored to offline targets.
   - Teacher KD/embedding/token losses use ConcatTeacher as target signal.
   - Stage-scale curriculum with phase-best reload is supported.
3) Freeze reconstructor and train corrected-only top tagger (single-view, no dual-view).
4) Joint-finetune corrected-only path (reconstructor + corrected-only head), and report pre/post.
5) Train dual-view tagger on frozen Stage-A reconstructor outputs, then joint-finetune, and report pre/post.

Outputs include val/test score arrays, overlap reports, and combo metrics.
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


class StageAConcatTeacherDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        const_teacher: np.ndarray,
        mask_teacher: np.ndarray,
        labels: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.const_teacher = torch.tensor(const_teacher, dtype=torch.float32)
        self.mask_teacher = torch.tensor(mask_teacher, dtype=torch.bool)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "const_teacher": self.const_teacher[i],
            "mask_teacher": self.mask_teacher[i],
            "label": self.labels[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
        }


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


def _compute_concat_teacher_guided_reco_losses(
    reco_out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    const_teacher: torch.Tensor,
    mask_teacher: torch.Tensor,
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
    # Keep physical supervision anchored to offline target.
    aux_losses = b.compute_reconstruction_losses(
        reco_out,
        const_hlt,
        mask_hlt,
        const_off,
        mask_off,
        budget_merge_true,
        budget_eff_true,
        loss_cfg,
    )
    loss_phys = aux_losses["phys"]

    # Teacher target path uses concatenated (offline||HLT) view.
    with torch.no_grad():
        feat_teacher_raw = b.compute_features_torch(const_teacher, mask_teacher)
        feat_teacher_std = b._standardize_features_torch(feat_teacher_raw, mask_teacher, means_t, stds_t)
        teacher_pack = teacher_model(feat_teacher_std, mask_teacher, return_attention=True, return_embedding=True)
        logits_teacher_target = teacher_pack[0].view(-1)
        attn_teacher_target = teacher_pack[1]
        emb_teacher_target = teacher_pack[2]

    feat_reco_raw, mask_reco = b._build_teacher_reco_features_from_output(
        reco_out,
        const_hlt,
        mask_hlt,
        weight_floor=budget_weight_floor,
    )
    feat_reco_std = b._standardize_features_torch(feat_reco_raw, mask_reco, means_t, stds_t)
    reco_pack = teacher_model(feat_reco_std, mask_reco, return_attention=True, return_embedding=True)
    logits_teacher_reco = reco_pack[0].view(-1)
    attn_teacher_reco = reco_pack[1]
    emb_teacher_reco = reco_pack[2]

    target_soft = torch.sigmoid(logits_teacher_target / kd_temperature)
    kd_vec = (
        F.binary_cross_entropy_with_logits(
            logits_teacher_reco / kd_temperature,
            target_soft,
            reduction="none",
        )
        * (kd_temperature * kd_temperature)
    )
    loss_kd = b._weighted_batch_mean(kd_vec, None)

    emb_target_n = F.normalize(emb_teacher_target, dim=1)
    emb_reco_n = F.normalize(emb_teacher_reco, dim=1)
    loss_emb = (1.0 - (emb_target_n * emb_reco_n).sum(dim=1)).mean()

    # Reco path and concat-teacher path may have different token lengths (e.g. 100 vs 200).
    # Align to shared prefix length for stable attention-token KD.
    # NOTE: b._attention_kl_loss_masked expects [B, L] attention vectors.
    def _attn_to_token_vec(attn: torch.Tensor, l_take: int) -> torch.Tensor:
        if attn.dim() == 2:
            # [B, L]
            return attn[:, :l_take]
        if attn.dim() == 3:
            # [B, L, L] -> pooled [B, L]
            return attn[:, :l_take, :l_take].mean(dim=1)
        if attn.dim() == 4:
            # [B, H, L, L] -> pooled [B, L]
            return attn[:, :, :l_take, :l_take].mean(dim=(1, 2))
        raise RuntimeError(f"Unexpected attention tensor rank={attn.dim()} (shape={tuple(attn.shape)})")

    l_reco = int(mask_reco.shape[1])
    l_teacher = int(mask_teacher.shape[1])
    l_common = int(min(l_reco, l_teacher))
    if l_common > 0:
        attn_pred_tok = _attn_to_token_vec(attn_teacher_reco, l_common)
        attn_tgt_tok = _attn_to_token_vec(attn_teacher_target, l_common)
        mask_pred_tok = mask_reco[:, :l_common]
        mask_tgt_tok = mask_teacher[:, :l_common]
        loss_tok = b._attention_kl_loss_masked(
            attn_pred=attn_pred_tok,
            attn_target=attn_tgt_tok,
            mask_pred=mask_pred_tok,
            mask_target=mask_tgt_tok,
        )
    else:
        loss_tok = torch.zeros((), device=mask_reco.device)

    reco_tokens = reco_out["cand_tokens"][:, : const_hlt.shape[1], :]
    mean_edit_vec = b._sorted_edit_budget_vec(reco_tokens, const_hlt, mask_hlt)
    budget_hinge_vec = F.relu(mean_edit_vec - budget_eps)
    loss_budget_hinge = b._weighted_batch_mean(budget_hinge_vec, None)

    return {
        "kd": loss_kd,
        "emb": loss_emb,
        "tok": loss_tok,
        "phys": loss_phys,
        "budget_hinge": loss_budget_hinge,
        "logits_teacher_reco": logits_teacher_reco,
    }


def train_reconstructor_concat_teacher_stagewise(
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
                const_teacher = batch["const_teacher"].to(device)
                mask_teacher = batch["mask_teacher"].to(device)
                labels_batch = batch["label"].to(device)
                budget_merge_true = batch["budget_merge_true"].to(device)
                budget_eff_true = batch["budget_eff_true"].to(device)

                opt.zero_grad()
                out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=float(sc))

                losses = _compute_concat_teacher_guided_reco_losses(
                    reco_out=out,
                    const_hlt=const_hlt,
                    mask_hlt=mask_hlt,
                    const_off=const_off,
                    mask_off=mask_off,
                    const_teacher=const_teacher,
                    mask_teacher=mask_teacher,
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
                    const_teacher = batch["const_teacher"].to(device)
                    mask_teacher = batch["mask_teacher"].to(device)
                    labels_batch = batch["label"].to(device)
                    budget_merge_true = batch["budget_merge_true"].to(device)
                    budget_eff_true = batch["budget_eff_true"].to(device)

                    out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
                    losses = _compute_concat_teacher_guided_reco_losses(
                        reco_out=out,
                        const_hlt=const_hlt,
                        mask_hlt=mask_hlt,
                        const_off=const_off,
                        mask_off=mask_off,
                        const_teacher=const_teacher,
                        mask_teacher=mask_teacher,
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



def _build_concat_teacher_batch_torch(
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    max_concat_constits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    const_cat = torch.cat([const_off, const_hlt], dim=1)
    mask_cat = torch.cat([mask_off, mask_hlt], dim=1)

    full_l = int(const_cat.shape[1])
    out_l = int(max_concat_constits)
    if out_l <= 0:
        out_l = full_l

    if out_l < full_l:
        const_cat = const_cat[:, :out_l, :]
        mask_cat = mask_cat[:, :out_l]
    elif out_l > full_l:
        n = int(const_cat.shape[0])
        pad_const = torch.zeros((n, out_l - full_l, const_cat.shape[2]), dtype=const_cat.dtype, device=const_cat.device)
        pad_mask = torch.zeros((n, out_l - full_l), dtype=torch.bool, device=const_cat.device)
        const_cat = torch.cat([const_cat, pad_const], dim=1)
        mask_cat = torch.cat([mask_cat, pad_mask], dim=1)

    const_cat = const_cat * mask_cat.unsqueeze(-1).float()
    return const_cat, mask_cat


@torch.no_grad()
def eval_corrected_joint_model(
    reconstructor: b.OfflineReconstructor,
    corrected_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    corrected_model.eval()
    reconstructor.eval()

    preds: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_corr, mask_corr = b.build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=bool(corrected_use_flags),
        )
        logits = corrected_model(feat_corr, mask_corr).squeeze(1)
        p = torch.sigmoid(logits)
        preds.append(p.detach().cpu().numpy().astype(np.float64))
        labs.append(y.detach().cpu().numpy().astype(np.float32))

    preds_np = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float64)
    labs_np = np.concatenate(labs) if labs else np.zeros(0, dtype=np.float32)
    if preds_np.size == 0:
        return float("nan"), preds_np, labs_np, float("nan")
    auc = float(roc_auc_score(labs_np, preds_np)) if len(np.unique(labs_np)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(labs_np, preds_np)
    fpr50 = float(b.fpr_at_target_tpr(fpr, tpr, 0.50))
    return auc, preds_np, labs_np, fpr50


def _combo_reports(
    labels_val: np.ndarray,
    labels_test: np.ndarray,
    preds_hlt_val: np.ndarray,
    preds_hlt_test: np.ndarray,
    preds_other_val: np.ndarray,
    preds_other_test: np.ndarray,
    other_name: str,
    target_tpr: float,
    weight_step: float,
) -> Tuple[Dict, Dict]:
    valsel = b.select_weighted_combo_on_val_and_eval_test(
        labels_val=labels_val,
        preds_a_val=preds_hlt_val,
        preds_b_val=preds_other_val,
        labels_test=labels_test,
        preds_a_test=preds_hlt_test,
        preds_b_test=preds_other_test,
        name_a="hlt",
        name_b=other_name,
        target_tpr=float(target_tpr),
        weight_step=float(weight_step),
    )
    oracle = b.search_best_weighted_combo_at_tpr(
        labels=labels_test,
        preds_a=preds_hlt_test,
        preds_b=preds_other_test,
        name_a="hlt",
        name_b=other_name,
        target_tpr=float(target_tpr),
        weight_step=float(weight_step),
    )
    return valsel, oracle


def train_corrected_only_joint_concat(
    reconstructor: b.OfflineReconstructor,
    corrected_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    freeze_reconstructor: bool,
    epochs: int,
    patience: int,
    lr_model: float,
    lr_reco: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_reco: float,
    lambda_rank: float,
    lambda_cons: float,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    min_epochs: int,
    select_metric: str,
    max_concat_constits: int,
    teacher_model: nn.Module | None = None,
    feat_means: np.ndarray | None = None,
    feat_stds: np.ndarray | None = None,
    reco_kd_temperature: float = 2.5,
    reco_lambda_kd: float = 1.0,
    reco_lambda_emb: float = 1.2,
    reco_lambda_tok: float = 0.6,
    reco_lambda_phys: float = 0.2,
    reco_lambda_budget_hinge: float = 0.03,
    reco_budget_eps: float = 0.015,
    reco_budget_weight_floor: float = 1e-4,
    reco_normalize_loss_terms: bool = True,
    reco_loss_norm_ema_decay: float = 0.98,
    reco_loss_norm_eps: float = 1e-6,
) -> Tuple[b.OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    for p in reconstructor.parameters():
        p.requires_grad = not freeze_reconstructor

    params = [{"params": corrected_model.parameters(), "lr": float(lr_model)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_model), weight_decay=float(weight_decay))
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    means_t = None
    stds_t = None
    reco_loss_ema_state = None
    if teacher_model is not None and feat_means is not None and feat_stds is not None:
        teacher_model.eval()
        for p_t in teacher_model.parameters():
            p_t.requires_grad_(False)
        means_t = torch.tensor(feat_means, dtype=torch.float32, device=device)
        stds_t = torch.tensor(np.clip(feat_stds, 1e-6, None), dtype=torch.float32, device=device)
        reco_loss_ema_state = {
            "kd": 1.0,
            "emb": 1.0,
            "tok": 1.0,
            "phys": 1.0,
            "budget": 1.0,
        }

    best_state_model_sel = None
    best_state_reco_sel = None
    best_state_model_auc = None
    best_state_reco_auc = None
    best_state_model_fpr = None
    best_state_reco_fpr = None

    best_val_fpr50 = float("inf")
    best_val_auc = float("-inf")
    best_sel_score = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    sel_val_fpr50 = float("nan")
    sel_val_auc = float("nan")
    no_improve = 0

    for ep in range(int(epochs)):
        corrected_model.train()
        if freeze_reconstructor:
            reconstructor.eval()
        else:
            reconstructor.train()

        tr_loss = tr_cls = tr_rank = tr_reco = tr_cons = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt_reco = batch["feat_hlt_reco"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            b_merge = batch["budget_merge_true"].to(device)
            b_eff = batch["budget_eff_true"].to(device)
            y = batch["label"].to(device)

            opt.zero_grad()

            if freeze_reconstructor:
                with torch.no_grad():
                    reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            else:
                reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)

            feat_corr, mask_corr = b.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=bool(corrected_use_flags),
            )
            logits = corrected_model(feat_corr, mask_corr).squeeze(1)

            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = b.low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()

            if float(lambda_reco) > 0.0 and teacher_model is not None and means_t is not None and stds_t is not None:
                const_teacher, mask_teacher = _build_concat_teacher_batch_torch(
                    const_off=const_off,
                    mask_off=mask_off,
                    const_hlt=const_hlt,
                    mask_hlt=mask_hlt,
                    max_concat_constits=int(max_concat_constits),
                )
                reco_losses = _compute_concat_teacher_guided_reco_losses(
                    reco_out=reco_out,
                    const_hlt=const_hlt,
                    mask_hlt=mask_hlt,
                    const_off=const_off,
                    mask_off=mask_off,
                    const_teacher=const_teacher,
                    mask_teacher=mask_teacher,
                    budget_merge_true=b_merge,
                    budget_eff_true=b_eff,
                    teacher_model=teacher_model,
                    means_t=means_t,
                    stds_t=stds_t,
                    loss_cfg=b.BASE_CONFIG["loss"],
                    kd_temperature=float(max(reco_kd_temperature, 1e-3)),
                    budget_eps=float(max(reco_budget_eps, 0.0)),
                    budget_weight_floor=float(max(reco_budget_weight_floor, 0.0)),
                )
                loss_reco, _, reco_loss_ema_state = b._compose_teacher_guided_reco_total(
                    losses_raw=reco_losses,
                    ema_state=reco_loss_ema_state,
                    normalize_terms=bool(reco_normalize_loss_terms),
                    ema_decay=float(np.clip(reco_loss_norm_ema_decay, 0.0, 0.9999)),
                    norm_eps=float(max(reco_loss_norm_eps, 1e-12)),
                    w_logit=float(max(reco_lambda_kd, 0.0)),
                    w_emb=float(max(reco_lambda_emb, 0.0)),
                    w_tok=float(max(reco_lambda_tok, 0.0)),
                    w_phys=float(max(reco_lambda_phys, 0.0)),
                    w_budget=float(max(reco_lambda_budget_hinge, 0.0)),
                    update_ema=True,
                )
            elif float(lambda_reco) > 0.0:
                reco_losses = b.compute_reconstruction_losses(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_merge,
                    b_eff,
                    b.BASE_CONFIG["loss"],
                )
                loss_reco = reco_losses["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            loss = (
                loss_cls
                + float(lambda_rank) * loss_rank
                + float(lambda_reco) * loss_reco
                + float(lambda_cons) * loss_cons
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corrected_model.parameters(), 1.0)
            if not freeze_reconstructor:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = feat_hlt_reco.size(0)
            tr_loss += float(loss.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            tr_rank += float(loss_rank.item()) * bs
            tr_reco += float(loss_reco.item()) * bs
            tr_cons += float(loss_cons.item()) * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)

        va_auc, _, _, va_fpr50 = eval_corrected_joint_model(
            reconstructor=reconstructor,
            corrected_model=corrected_model,
            loader=val_loader,
            device=device,
            corrected_weight_floor=float(corrected_weight_floor),
            corrected_use_flags=bool(corrected_use_flags),
        )

        if np.isfinite(va_fpr50) and float(va_fpr50) < best_val_fpr50:
            best_val_fpr50 = float(va_fpr50)
            best_state_model_fpr = {k: v.detach().cpu().clone() for k, v in corrected_model.state_dict().items()}
            best_state_reco_fpr = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state_model_auc = {k: v.detach().cpu().clone() for k, v in corrected_model.state_dict().items()}
            best_state_reco_auc = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}

        if str(select_metric).lower() == "auc":
            improved = np.isfinite(va_auc) and (float(va_auc) > best_sel_score)
            current_score = float(va_auc) if np.isfinite(va_auc) else float("-inf")
        else:
            improved = np.isfinite(va_fpr50) and (float(va_fpr50) < best_sel_score)
            current_score = float(va_fpr50) if np.isfinite(va_fpr50) else float("inf")

        if improved:
            best_sel_score = current_score
            sel_val_fpr50 = float(va_fpr50)
            sel_val_auc = float(va_auc)
            best_state_model_sel = {k: v.detach().cpu().clone() for k, v in corrected_model.state_dict().items()}
            best_state_reco_sel = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print_every = 1 if str(stage_name).startswith("StageC") else 5
        if (ep + 1) % print_every == 0:
            print(
                f"{stage_name} ep {ep+1}: train_loss={tr_loss:.4f} "
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, "
                f"select={str(select_metric).lower()}, best_sel={best_sel_score:.6f}"
            )

        if (ep + 1) >= int(min_epochs) and no_improve >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_model_sel is not None:
        corrected_model.load_state_dict(best_state_model_sel)
    if best_state_reco_sel is not None:
        reconstructor.load_state_dict(best_state_reco_sel)

    metrics = {
        "selection_metric": str(select_metric).lower(),
        "selected_val_fpr50": float(sel_val_fpr50),
        "selected_val_auc": float(sel_val_auc),
        "best_val_fpr50_seen": float(best_val_fpr50),
        "best_val_auc_seen": float(best_val_auc),
    }
    state_pack = {
        "selected": {"model": best_state_model_sel, "reco": best_state_reco_sel},
        "auc": {"model": best_state_model_auc, "reco": best_state_reco_auc},
        "fpr50": {"model": best_state_model_fpr, "reco": best_state_reco_fpr},
    }
    return reconstructor, corrected_model, metrics, state_pack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--n_train_jets", type=int, default=250000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--max_concat_constits", type=int, default=-1)
    ap.add_argument("--n_train_split", type=int, default=75000)
    ap.add_argument("--n_val_split", type=int, default=25000)
    ap.add_argument("--n_test_split", type=int, default=150000)
    ap.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_joint_concat_teacher"))
    ap.add_argument("--run_name", type=str, default="concat_teacher_stageA_then_corrected_75k25k150k_seed0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--skip_save_models", action="store_true")
    ap.add_argument("--seed", type=int, default=b.RANDOM_SEED)

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
    ap.add_argument("--stageA_target_tpr", type=float, default=0.50)
    ap.add_argument("--disable_stageA_loss_normalization", action="store_true")
    ap.add_argument("--stageA_loss_norm_ema_decay", type=float, default=0.98)
    ap.add_argument("--stageA_loss_norm_eps", type=float, default=1e-6)
    ap.add_argument("--disable_stageA_stagewise_best_reload", action="store_true")

    ap.add_argument("--stageA_lambda_delta", type=float, default=0.0)
    ap.add_argument("--stageA_delta_tau", type=float, default=0.05)
    ap.add_argument("--stageA_delta_lambda_fp", type=float, default=3.0)

    ap.add_argument("--added_target_scale", type=float, default=0.90)
    ap.add_argument("--reco_weight_threshold", type=float, default=0.03)
    ap.add_argument("--reco_eval_batch_size", type=int, default=256)

    ap.add_argument("--report_target_tpr", type=float, default=0.50)
    ap.add_argument("--combo_weight_step", type=float, default=0.01)
    ap.add_argument("--joint_select_metric", type=str, default="auc", choices=["auc", "fpr50"])
    ap.add_argument("--stageB_epochs", type=int, default=45)
    ap.add_argument("--stageB_patience", type=int, default=12)
    ap.add_argument("--stageB_min_epochs", type=int, default=12)
    ap.add_argument("--stageB_lr_dual", type=float, default=4e-4)
    ap.add_argument("--stageB_lambda_rank", type=float, default=0.0)
    ap.add_argument("--stageB_lambda_cons", type=float, default=0.0)

    ap.add_argument("--stageC_epochs", type=int, default=65)
    ap.add_argument("--stageC_patience", type=int, default=14)
    ap.add_argument("--stageC_min_epochs", type=int, default=25)
    ap.add_argument("--stageC_lr_dual", type=float, default=2e-4)
    ap.add_argument("--stageC_lr_reco", type=float, default=1e-4)
    ap.add_argument("--stageC_lambda_reco", type=float, default=0.4)
    ap.add_argument("--stageC_lambda_rank", type=float, default=0.0)
    ap.add_argument("--stageC_lambda_cons", type=float, default=0.06)

    ap.add_argument("--use_corrected_flags", action="store_true")
    ap.add_argument(
        "--stop_after_corrected_only",
        action="store_true",
        help="Stop after Stage-4 corrected-only evaluation and skip corrected-joint/dual-joint finetuning.",
    )
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

    max_concat_constits = int(args.max_concat_constits)
    if max_concat_constits <= 0:
        max_concat_constits = int(args.max_constits) * 2
    const_concat, mask_concat = build_concat_constituents(
        const_off=const_off,
        mask_off=masks_off,
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
        max_concat_constits=max_concat_constits,
    )
    feat_concat = b.compute_features(const_concat, mask_concat)

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

    means_off, stds_off = b.get_stats(feat_off, masks_off, train_idx)
    feat_off_std = b.standardize(feat_off, masks_off, means_off, stds_off)
    feat_hlt_std = b.standardize(feat_hlt, hlt_mask, means_off, stds_off)

    means_concat, stds_concat = b.get_stats(feat_concat, mask_concat, train_idx)
    feat_concat_std = b.standardize(feat_concat, mask_concat, means_concat, stds_concat)

    data_setup = {
        "train_path_arg": str(args.train_path),
        "train_files": [str(p.resolve()) for p in train_files],
        "n_train_jets": int(args.n_train_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "max_concat_constits": int(max_concat_constits),
        "seed": int(args.seed),
        "split": {
            "mode": "custom_counts",
            "n_train_split": int(len(train_idx)),
            "n_val_split": int(len(val_idx)),
            "n_test_split": int(len(test_idx)),
        },
        "hlt_effects": cfg["hlt_effects"],
        "variant": "concat_teacher_stageA_then_corrected",
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
        means_off=means_off.astype(np.float32),
        stds_off=stds_off.astype(np.float32),
        means_concat=means_concat.astype(np.float32),
        stds_concat=stds_concat.astype(np.float32),
    )

    print("\n" + "=" * 70)
    print("STEP 1: HLT BASELINE + CONCAT TEACHER")
    print("=" * 70)
    bs_cls = int(cfg["training"]["batch_size"])

    ds_train_hlt = b.JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx])
    ds_val_hlt = b.JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    ds_test_hlt = b.JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_train_hlt = DataLoader(ds_train_hlt, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=bs_cls, shuffle=False)
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=bs_cls, shuffle=False)

    baseline = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = b.train_single_view_classifier_auc(
        baseline,
        dl_train_hlt,
        dl_val_hlt,
        device,
        cfg["training"],
        name="Baseline-HLT",
    )
    auc_hlt_test, preds_hlt_test, labs_hlt_test = b.eval_classifier(baseline, dl_test_hlt, device)
    auc_hlt_val, preds_hlt_val, labs_hlt_val = b.eval_classifier(baseline, dl_val_hlt, device)

    ds_train_concat = b.JetDataset(feat_concat_std[train_idx], mask_concat[train_idx], labels[train_idx])
    ds_val_concat = b.JetDataset(feat_concat_std[val_idx], mask_concat[val_idx], labels[val_idx])
    ds_test_concat = b.JetDataset(feat_concat_std[test_idx], mask_concat[test_idx], labels[test_idx])
    dl_train_concat = DataLoader(ds_train_concat, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_val_concat = DataLoader(ds_val_concat, batch_size=bs_cls, shuffle=False)
    dl_test_concat = DataLoader(ds_test_concat, batch_size=bs_cls, shuffle=False)

    concat_teacher = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    concat_teacher = b.train_single_view_classifier_auc(
        concat_teacher,
        dl_train_concat,
        dl_val_concat,
        device,
        cfg["training"],
        name="ConcatTeacher",
    )
    auc_concat_test, preds_concat_test, labs_concat_test = b.eval_classifier(concat_teacher, dl_test_concat, device)
    auc_concat_val, preds_concat_val, labs_concat_val = b.eval_classifier(concat_teacher, dl_val_concat, device)

    assert np.array_equal(labs_hlt_val.astype(np.float32), labs_concat_val.astype(np.float32))
    assert np.array_equal(labs_hlt_test.astype(np.float32), labs_concat_test.astype(np.float32))

    hlt_thr_prob, hlt_thr_tpr, hlt_thr_fpr = threshold_at_target_tpr(
        labs_hlt_val.astype(np.float32),
        preds_hlt_val.astype(np.float64),
        float(args.stageA_target_tpr),
    )
    print(
        f"StageA delta HLT reference @TPR={float(args.stageA_target_tpr):.2f}: "
        f"threshold_prob={hlt_thr_prob:.6f}, val_tpr={hlt_thr_tpr:.6f}, val_fpr={hlt_thr_fpr:.6f}"
    )

    print("\n" + "=" * 70)
    print("STEP 2: STAGE A (CONCAT-TEACHER-GUIDED RECONSTRUCTOR PRETRAIN)")
    print("=" * 70)
    ds_train_reco = StageAConcatTeacherDataset(
        feat_hlt=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        const_teacher=const_concat[train_idx],
        mask_teacher=mask_concat[train_idx],
        labels=labels[train_idx],
        budget_merge_true=budget_merge_true[train_idx],
        budget_eff_true=budget_eff_true[train_idx],
    )
    ds_val_reco = StageAConcatTeacherDataset(
        feat_hlt=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        const_teacher=const_concat[val_idx],
        mask_teacher=mask_concat[val_idx],
        labels=labels[val_idx],
        budget_merge_true=budget_merge_true[val_idx],
        budget_eff_true=budget_eff_true[val_idx],
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
    reconstructor, reco_val_metrics = train_reconstructor_concat_teacher_stagewise(
        model=reconstructor,
        train_loader=dl_train_reco,
        val_loader=dl_val_reco,
        device=device,
        train_cfg=cfg["reconstructor_training"],
        loss_cfg=cfg["loss"],
        teacher_model=concat_teacher,
        hlt_model=baseline,
        hlt_threshold_prob=float(hlt_thr_prob),
        feat_means=means_concat.astype(np.float32),
        feat_stds=stds_concat.astype(np.float32),
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
    print("STEP 3: CONCAT-TEACHER ON SOFT RECONSTRUCTED VIEW")
    print("=" * 70)
    auc_reco_teacher_val, preds_reco_teacher_val, labs_reco_val, fpr50_reco_teacher_val = eval_teacher_on_soft_reco_split(
        reconstructor=reconstructor,
        teacher=concat_teacher,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        labels=labels,
        split_idx=val_idx,
        means=means_concat,
        stds=stds_concat,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        weight_floor=float(args.reco_weight_threshold),
        target_tpr=float(args.report_target_tpr),
    )
    auc_reco_teacher_test, preds_reco_teacher_test, labs_reco_test, fpr50_reco_teacher_test = eval_teacher_on_soft_reco_split(
        reconstructor=reconstructor,
        teacher=concat_teacher,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        labels=labels,
        split_idx=test_idx,
        means=means_concat,
        stds=stds_concat,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        weight_floor=float(args.reco_weight_threshold),
        target_tpr=float(args.report_target_tpr),
    )

    print("\n" + "=" * 70)
    print("STEP 4: CORRECTED-ONLY TAGGER (FROZEN STAGE-A RECONSTRUCTOR)")
    print("=" * 70)
    corrected_use_flags = bool(args.use_corrected_flags)
    feat_corr_all, mask_corr_all = b.build_corrected_view_numpy(
        reconstructor=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        device=device,
        batch_size=int(bs_cls),
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )

    ds_train_corr = b.JetDataset(feat_corr_all[train_idx], mask_corr_all[train_idx], labels[train_idx])
    ds_val_corr = b.JetDataset(feat_corr_all[val_idx], mask_corr_all[val_idx], labels[val_idx])
    ds_test_corr = b.JetDataset(feat_corr_all[test_idx], mask_corr_all[test_idx], labels[test_idx])
    dl_train_corr = DataLoader(ds_train_corr, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_val_corr = DataLoader(ds_val_corr, batch_size=bs_cls, shuffle=False)
    dl_test_corr = DataLoader(ds_test_corr, batch_size=bs_cls, shuffle=False)

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

    if bool(args.stop_after_corrected_only):
        fpr_hlt, tpr_hlt, _ = roc_curve(labs_hlt_test, preds_hlt_test)
        fpr_concat, tpr_concat, _ = roc_curve(labs_concat_test, preds_concat_test)
        fpr_reco, tpr_reco, _ = roc_curve(labs_reco_test, preds_reco_teacher_test)
        fpr_corr, tpr_corr, _ = roc_curve(labs_corr_test, preds_corr_test)

        fpr50_hlt = float(b.fpr_at_target_tpr(fpr_hlt, tpr_hlt, 0.50))
        fpr50_concat = float(b.fpr_at_target_tpr(fpr_concat, tpr_concat, 0.50))
        fpr50_reco = float(b.fpr_at_target_tpr(fpr_reco, tpr_reco, 0.50))
        fpr50_corr = float(b.fpr_at_target_tpr(fpr_corr, tpr_corr, 0.50))

        overlap_report = b.build_overlap_report_at_tpr(
            labels=labs_hlt_test.astype(np.float32),
            model_preds={
                "hlt": preds_hlt_test,
                "concat_teacher": preds_concat_test,
                "reco_teacher_soft": preds_reco_teacher_test,
                "corrected_only": preds_corr_test,
            },
            target_tpr=float(args.report_target_tpr),
        )

        combo_hlt_reco_valsel, combo_hlt_reco_oracle = _combo_reports(
            labels_val=labs_hlt_val.astype(np.float32),
            labels_test=labs_hlt_test.astype(np.float32),
            preds_hlt_val=preds_hlt_val,
            preds_hlt_test=preds_hlt_test,
            preds_other_val=preds_reco_teacher_val,
            preds_other_test=preds_reco_teacher_test,
            other_name="reco_teacher_soft",
            target_tpr=float(args.report_target_tpr),
            weight_step=float(args.combo_weight_step),
        )
        combo_hlt_corr_valsel, combo_hlt_corr_oracle = _combo_reports(
            labels_val=labs_hlt_val.astype(np.float32),
            labels_test=labs_hlt_test.astype(np.float32),
            preds_hlt_val=preds_hlt_val,
            preds_hlt_test=preds_hlt_test,
            preds_other_val=preds_corr_val,
            preds_other_test=preds_corr_test,
            other_name="corrected_only",
            target_tpr=float(args.report_target_tpr),
            weight_step=float(args.combo_weight_step),
        )

        print("\n" + "=" * 70)
        print("FINAL EVALUATION (STOP AFTER CORRECTED-ONLY)")
        print("=" * 70)
        print(f"HLT baseline AUC (val/test): {auc_hlt_val:.4f} / {auc_hlt_test:.4f}")
        print(f"ConcatTeacher AUC (val/test): {auc_concat_val:.4f} / {auc_concat_test:.4f}")
        print(f"RecoTeacherSoft AUC (val/test): {auc_reco_teacher_val:.4f} / {auc_reco_teacher_test:.4f}")
        print(f"CorrectedOnly AUC (val/test): {auc_corr_val:.4f} / {auc_corr_test:.4f}")
        print(
            "FPR@50 HLT / ConcatTeacher / RecoTeacherSoft / CorrectedOnly: "
            f"{fpr50_hlt:.6f} / {fpr50_concat:.6f} / {fpr50_reco:.6f} / {fpr50_corr:.6f}"
        )

        np.savez_compressed(
            save_root / "concat_teacher_stageA_scores.npz",
            labels_val=labs_hlt_val.astype(np.float32),
            labels_test=labs_hlt_test.astype(np.float32),
            preds_hlt_val=preds_hlt_val.astype(np.float64),
            preds_hlt_test=preds_hlt_test.astype(np.float64),
            preds_concat_teacher_val=preds_concat_val.astype(np.float64),
            preds_concat_teacher_test=preds_concat_test.astype(np.float64),
            preds_reco_teacher_val=preds_reco_teacher_val.astype(np.float64),
            preds_reco_teacher_test=preds_reco_teacher_test.astype(np.float64),
            preds_corrected_only_val=preds_corr_val.astype(np.float64),
            preds_corrected_only_test=preds_corr_test.astype(np.float64),
            auc_hlt_val=float(auc_hlt_val),
            auc_hlt_test=float(auc_hlt_test),
            auc_concat_teacher_val=float(auc_concat_val),
            auc_concat_teacher_test=float(auc_concat_test),
            auc_reco_teacher_val=float(auc_reco_teacher_val),
            auc_reco_teacher_test=float(auc_reco_teacher_test),
            auc_corrected_only_val=float(auc_corr_val),
            auc_corrected_only_test=float(auc_corr_test),
            fpr50_hlt=float(fpr50_hlt),
            fpr50_concat_teacher=float(fpr50_concat),
            fpr50_reco_teacher=float(fpr50_reco),
            fpr50_corrected_only=float(fpr50_corr),
            target_tpr=float(args.report_target_tpr),
            stop_after_corrected_only=np.array([1], dtype=np.int32),
        )

        with open(save_root / "concat_teacher_stageA_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "variant": "concat_teacher_stageA_then_corrected",
                    "rho": float(rho),
                    "max_concat_constits": int(max_concat_constits),
                    "stop_after_corrected_only": True,
                    "stageA_reconstructor": reco_val_metrics,
                    "stageB_dual_pre": None,
                    "stageC_dual_joint": None,
                    "stageC_corrected_only_joint": None,
                    "hlt": {
                        "auc_val": float(auc_hlt_val),
                        "auc_test": float(auc_hlt_test),
                        "delta_ref_threshold_prob": float(hlt_thr_prob),
                        "delta_ref_val_tpr": float(hlt_thr_tpr),
                        "delta_ref_val_fpr": float(hlt_thr_fpr),
                    },
                    "concat_teacher": {
                        "auc_val": float(auc_concat_val),
                        "auc_test": float(auc_concat_test),
                        "fpr50_test": float(fpr50_concat),
                    },
                    "reco_teacher_soft": {
                        "auc_val": float(auc_reco_teacher_val),
                        "auc_test": float(auc_reco_teacher_test),
                        "fpr50_val": float(fpr50_reco_teacher_val),
                        "fpr50_test": float(fpr50_reco_teacher_test),
                    },
                    "corrected_only": {
                        "auc_val": float(auc_corr_val),
                        "auc_test": float(auc_corr_test),
                        "fpr50_test": float(fpr50_corr),
                    },
                    "overlap_report_tpr": overlap_report,
                    "best_combo_hlt_reco_val_selected_eval_test": combo_hlt_reco_valsel,
                    "best_combo_hlt_reco_test_posthoc": combo_hlt_reco_oracle,
                    "best_combo_hlt_corrected_val_selected_eval_test": combo_hlt_corr_valsel,
                    "best_combo_hlt_corrected_test_posthoc": combo_hlt_corr_oracle,
                },
                f,
                indent=2,
            )

        with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
            json.dump({"config": cfg["hlt_effects"], "stats": hlt_stats}, f, indent=2)

        if not args.skip_save_models:
            torch.save({"model": baseline.state_dict(), "auc": float(auc_hlt_test)}, save_root / "baseline.pt")
            torch.save({"model": concat_teacher.state_dict(), "auc": float(auc_concat_test)}, save_root / "concat_teacher.pt")
            torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor_stageA.pt")
            torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor.pt")
            torch.save({"model": corrected_only.state_dict(), "auc": float(auc_corr_test)}, save_root / "corrected_only_tagger.pt")

        print(f"\nSaved concat-teacher Stage-A + corrected-only results to: {save_root}")
        return

    # Shared joint datasets (for corrected-only joint and dual-view branches).
    ds_train_joint = b.JointDualDataset(
        feat_hlt_std[train_idx], feat_hlt_std[train_idx], hlt_mask[train_idx], hlt_const[train_idx],
        const_off[train_idx], masks_off[train_idx],
        budget_merge_true[train_idx], budget_eff_true[train_idx],
        labels[train_idx],
    )
    ds_val_joint = b.JointDualDataset(
        feat_hlt_std[val_idx], feat_hlt_std[val_idx], hlt_mask[val_idx], hlt_const[val_idx],
        const_off[val_idx], masks_off[val_idx],
        budget_merge_true[val_idx], budget_eff_true[val_idx],
        labels[val_idx],
    )
    ds_test_joint = b.JointDualDataset(
        feat_hlt_std[test_idx], feat_hlt_std[test_idx], hlt_mask[test_idx], hlt_const[test_idx],
        const_off[test_idx], masks_off[test_idx],
        budget_merge_true[test_idx], budget_eff_true[test_idx],
        labels[test_idx],
    )

    dl_train_joint = DataLoader(
        ds_train_joint,
        batch_size=bs_cls,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint = DataLoader(
        ds_val_joint,
        batch_size=bs_cls,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_test_joint = DataLoader(
        ds_test_joint,
        batch_size=bs_cls,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    stageA_reco_state = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
    corrected_only_pre_state = {k: v.detach().cpu().clone() for k, v in corrected_only.state_dict().items()}

    print("\n" + "=" * 70)
    print("STEP 5: CORRECTED-ONLY JOINT FINETUNE (PRE->POST)")
    print("=" * 70)
    reconstructor_corr_joint = b.OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor_corr_joint.load_state_dict(stageA_reco_state)
    corrected_only_joint = b.ParticleTransformer(input_dim=int(feat_corr_all.shape[-1]), **cfg["model"]).to(device)
    corrected_only_joint.load_state_dict(corrected_only_pre_state)

    reconstructor_corr_joint, corrected_only_joint, corrected_joint_metrics, corrected_joint_states = train_corrected_only_joint_concat(
        reconstructor=reconstructor_corr_joint,
        corrected_model=corrected_only_joint,
        train_loader=dl_train_joint,
        val_loader=dl_val_joint,
        device=device,
        stage_name="StageC-CorrectedOnlyJoint",
        freeze_reconstructor=False,
        epochs=int(args.stageC_epochs),
        patience=int(args.stageC_patience),
        lr_model=float(args.stageC_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        lambda_reco=float(args.stageC_lambda_reco),
        lambda_rank=float(args.stageC_lambda_rank),
        lambda_cons=float(args.stageC_lambda_cons),
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
        min_epochs=int(args.stageC_min_epochs),
        select_metric=str(args.joint_select_metric),
        max_concat_constits=int(max_concat_constits),
        teacher_model=concat_teacher,
        feat_means=means_concat.astype(np.float32),
        feat_stds=stds_concat.astype(np.float32),
        reco_kd_temperature=float(args.stageA_kd_temp),
        reco_lambda_kd=float(args.stageA_lambda_kd),
        reco_lambda_emb=float(args.stageA_lambda_emb),
        reco_lambda_tok=float(args.stageA_lambda_tok),
        reco_lambda_phys=float(args.stageA_lambda_phys),
        reco_lambda_budget_hinge=float(args.stageA_lambda_budget_hinge),
        reco_budget_eps=float(args.stageA_budget_eps),
        reco_budget_weight_floor=float(args.stageA_budget_weight_floor),
        reco_normalize_loss_terms=not bool(args.disable_stageA_loss_normalization),
        reco_loss_norm_ema_decay=float(args.stageA_loss_norm_ema_decay),
        reco_loss_norm_eps=float(args.stageA_loss_norm_eps),
    )

    auc_corr_joint_val, preds_corr_joint_val, labs_corr_joint_val, fpr50_corr_joint_val = eval_corrected_joint_model(
        reconstructor=reconstructor_corr_joint,
        corrected_model=corrected_only_joint,
        loader=dl_val_joint,
        device=device,
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )
    auc_corr_joint_test, preds_corr_joint_test, labs_corr_joint_test, fpr50_corr_joint_test = eval_corrected_joint_model(
        reconstructor=reconstructor_corr_joint,
        corrected_model=corrected_only_joint,
        loader=dl_test_joint,
        device=device,
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )

    print("\n" + "=" * 70)
    print("STEP 6: DUAL-VIEW PRETRAIN (FROZEN STAGE-A RECO) -> JOINT FINETUNE")
    print("=" * 70)
    reconstructor_dual = b.OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor_dual.load_state_dict(stageA_reco_state)

    dual_input_dim_a = int(feat_hlt_std.shape[-1])
    dual_input_dim_b = 12 if corrected_use_flags else 10
    dual_model = b.DualViewCrossAttnClassifier(input_dim_a=dual_input_dim_a, input_dim_b=dual_input_dim_b, **cfg["model"]).to(device)

    reconstructor_dual, dual_model, stageB_dual_metrics, stageB_dual_states = b.train_joint_dual(
        reconstructor=reconstructor_dual,
        dual_model=dual_model,
        train_loader=dl_train_joint,
        val_loader=dl_val_joint,
        device=device,
        stage_name="StageB-DualPretrain",
        freeze_reconstructor=True,
        epochs=int(args.stageB_epochs),
        patience=int(args.stageB_patience),
        lr_dual=float(args.stageB_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        lambda_reco=0.0,
        lambda_rank=float(args.stageB_lambda_rank),
        lambda_cons=float(args.stageB_lambda_cons),
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
        min_epochs=int(args.stageB_min_epochs),
        select_metric=str(args.joint_select_metric),
    )

    auc_dual_pre_val, preds_dual_pre_val, labs_dual_pre_val, fpr50_dual_pre_val = b.eval_joint_model(
        reconstructor_dual,
        dual_model,
        dl_val_joint,
        device,
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )
    auc_dual_pre_test, preds_dual_pre_test, labs_dual_pre_test, fpr50_dual_pre_test = b.eval_joint_model(
        reconstructor_dual,
        dual_model,
        dl_test_joint,
        device,
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )

    reconstructor_dual, dual_model, stageC_dual_metrics, stageC_dual_states = b.train_joint_dual(
        reconstructor=reconstructor_dual,
        dual_model=dual_model,
        train_loader=dl_train_joint,
        val_loader=dl_val_joint,
        device=device,
        stage_name="StageC-DualJoint",
        freeze_reconstructor=False,
        epochs=int(args.stageC_epochs),
        patience=int(args.stageC_patience),
        lr_dual=float(args.stageC_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        lambda_reco=float(args.stageC_lambda_reco),
        lambda_rank=float(args.stageC_lambda_rank),
        lambda_cons=float(args.stageC_lambda_cons),
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
        min_epochs=int(args.stageC_min_epochs),
        select_metric=str(args.joint_select_metric),
        teacher_model=concat_teacher,
        feat_means=means_concat.astype(np.float32),
        feat_stds=stds_concat.astype(np.float32),
        reco_kd_temperature=float(args.stageA_kd_temp),
        reco_lambda_kd=float(args.stageA_lambda_kd),
        reco_lambda_emb=float(args.stageA_lambda_emb),
        reco_lambda_tok=float(args.stageA_lambda_tok),
        reco_lambda_phys=float(args.stageA_lambda_phys),
        reco_lambda_budget_hinge=float(args.stageA_lambda_budget_hinge),
        reco_budget_eps=float(args.stageA_budget_eps),
        reco_budget_weight_floor=float(args.stageA_budget_weight_floor),
        reco_normalize_loss_terms=not bool(args.disable_stageA_loss_normalization),
        reco_loss_norm_ema_decay=float(args.stageA_loss_norm_ema_decay),
        reco_loss_norm_eps=float(args.stageA_loss_norm_eps),
    )

    auc_dual_joint_val, preds_dual_joint_val, labs_dual_joint_val, fpr50_dual_joint_val = b.eval_joint_model(
        reconstructor_dual,
        dual_model,
        dl_val_joint,
        device,
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )
    auc_dual_joint_test, preds_dual_joint_test, labs_dual_joint_test, fpr50_dual_joint_test = b.eval_joint_model(
        reconstructor_dual,
        dual_model,
        dl_test_joint,
        device,
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )

    assert np.array_equal(labs_reco_val.astype(np.float32), labs_hlt_val.astype(np.float32))
    assert np.array_equal(labs_reco_test.astype(np.float32), labs_hlt_test.astype(np.float32))
    assert np.array_equal(labs_corr_val.astype(np.float32), labs_hlt_val.astype(np.float32))
    assert np.array_equal(labs_corr_test.astype(np.float32), labs_hlt_test.astype(np.float32))
    assert np.array_equal(labs_corr_joint_val.astype(np.float32), labs_hlt_val.astype(np.float32))
    assert np.array_equal(labs_corr_joint_test.astype(np.float32), labs_hlt_test.astype(np.float32))
    assert np.array_equal(labs_dual_pre_val.astype(np.float32), labs_hlt_val.astype(np.float32))
    assert np.array_equal(labs_dual_pre_test.astype(np.float32), labs_hlt_test.astype(np.float32))
    assert np.array_equal(labs_dual_joint_val.astype(np.float32), labs_hlt_val.astype(np.float32))
    assert np.array_equal(labs_dual_joint_test.astype(np.float32), labs_hlt_test.astype(np.float32))

    fpr_concat, tpr_concat, _ = roc_curve(labs_concat_test, preds_concat_test)
    fpr_hlt, tpr_hlt, _ = roc_curve(labs_hlt_test, preds_hlt_test)
    fpr_reco, tpr_reco, _ = roc_curve(labs_reco_test, preds_reco_teacher_test)
    fpr_corr, tpr_corr, _ = roc_curve(labs_corr_test, preds_corr_test)
    fpr_corr_joint, tpr_corr_joint, _ = roc_curve(labs_corr_joint_test, preds_corr_joint_test)
    fpr_dual_pre, tpr_dual_pre, _ = roc_curve(labs_dual_pre_test, preds_dual_pre_test)
    fpr_dual_joint, tpr_dual_joint, _ = roc_curve(labs_dual_joint_test, preds_dual_joint_test)

    fpr50_concat = float(b.fpr_at_target_tpr(fpr_concat, tpr_concat, 0.50))
    fpr50_hlt = float(b.fpr_at_target_tpr(fpr_hlt, tpr_hlt, 0.50))
    fpr50_reco = float(b.fpr_at_target_tpr(fpr_reco, tpr_reco, 0.50))
    fpr50_corr = float(b.fpr_at_target_tpr(fpr_corr, tpr_corr, 0.50))
    fpr50_corr_joint = float(b.fpr_at_target_tpr(fpr_corr_joint, tpr_corr_joint, 0.50))
    fpr50_dual_pre = float(b.fpr_at_target_tpr(fpr_dual_pre, tpr_dual_pre, 0.50))
    fpr50_dual_joint = float(b.fpr_at_target_tpr(fpr_dual_joint, tpr_dual_joint, 0.50))

    overlap_report = b.build_overlap_report_at_tpr(
        labels=labs_hlt_test.astype(np.float32),
        model_preds={
            "hlt": preds_hlt_test,
            "concat_teacher": preds_concat_test,
            "reco_teacher_soft": preds_reco_teacher_test,
            "corrected_only": preds_corr_test,
            "corrected_only_joint": preds_corr_joint_test,
            "dual_pre": preds_dual_pre_test,
            "dual_joint": preds_dual_joint_test,
        },
        target_tpr=float(args.report_target_tpr),
    )

    combo_hlt_reco_valsel, combo_hlt_reco_oracle = _combo_reports(
        labels_val=labs_hlt_val.astype(np.float32),
        labels_test=labs_hlt_test.astype(np.float32),
        preds_hlt_val=preds_hlt_val,
        preds_hlt_test=preds_hlt_test,
        preds_other_val=preds_reco_teacher_val,
        preds_other_test=preds_reco_teacher_test,
        other_name="reco_teacher_soft",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )
    combo_hlt_corr_valsel, combo_hlt_corr_oracle = _combo_reports(
        labels_val=labs_hlt_val.astype(np.float32),
        labels_test=labs_hlt_test.astype(np.float32),
        preds_hlt_val=preds_hlt_val,
        preds_hlt_test=preds_hlt_test,
        preds_other_val=preds_corr_val,
        preds_other_test=preds_corr_test,
        other_name="corrected_only",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )
    combo_hlt_corr_joint_valsel, combo_hlt_corr_joint_oracle = _combo_reports(
        labels_val=labs_hlt_val.astype(np.float32),
        labels_test=labs_hlt_test.astype(np.float32),
        preds_hlt_val=preds_hlt_val,
        preds_hlt_test=preds_hlt_test,
        preds_other_val=preds_corr_joint_val,
        preds_other_test=preds_corr_joint_test,
        other_name="corrected_only_joint",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )
    combo_hlt_dual_pre_valsel, combo_hlt_dual_pre_oracle = _combo_reports(
        labels_val=labs_hlt_val.astype(np.float32),
        labels_test=labs_hlt_test.astype(np.float32),
        preds_hlt_val=preds_hlt_val,
        preds_hlt_test=preds_hlt_test,
        preds_other_val=preds_dual_pre_val,
        preds_other_test=preds_dual_pre_test,
        other_name="dual_pre",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )
    combo_hlt_dual_joint_valsel, combo_hlt_dual_joint_oracle = _combo_reports(
        labels_val=labs_hlt_val.astype(np.float32),
        labels_test=labs_hlt_test.astype(np.float32),
        preds_hlt_val=preds_hlt_val,
        preds_hlt_test=preds_hlt_test,
        preds_other_val=preds_dual_joint_val,
        preds_other_test=preds_dual_joint_test,
        other_name="dual_joint",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    print(f"HLT baseline AUC (val/test): {auc_hlt_val:.4f} / {auc_hlt_test:.4f}")
    print(f"ConcatTeacher AUC (val/test): {auc_concat_val:.4f} / {auc_concat_test:.4f}")
    print(f"RecoTeacherSoft AUC (val/test): {auc_reco_teacher_val:.4f} / {auc_reco_teacher_test:.4f}")
    print(f"CorrectedOnly pre-joint AUC (val/test): {auc_corr_val:.4f} / {auc_corr_test:.4f}")
    print(f"CorrectedOnly post-joint AUC (val/test): {auc_corr_joint_val:.4f} / {auc_corr_joint_test:.4f}")
    print(f"Dual pre-joint AUC (val/test): {auc_dual_pre_val:.4f} / {auc_dual_pre_test:.4f}")
    print(f"Dual post-joint AUC (val/test): {auc_dual_joint_val:.4f} / {auc_dual_joint_test:.4f}")
    print(
        "FPR@50 HLT / ConcatTeacher / RecoTeacherSoft / CorrectedOnly(pre) / CorrectedOnly(post) / "
        f"Dual(pre) / Dual(post): {fpr50_hlt:.6f} / {fpr50_concat:.6f} / {fpr50_reco:.6f} / "
        f"{fpr50_corr:.6f} / {fpr50_corr_joint:.6f} / {fpr50_dual_pre:.6f} / {fpr50_dual_joint:.6f}"
    )
    pair = overlap_report["pairs"].get("hlt__reco_teacher_soft", {})
    if not pair:
        pair = overlap_report["pairs"].get("reco_teacher_soft__hlt", {})
    print(
        f"TP overlap @TPR={float(args.report_target_tpr):.2f} (HLT vs RecoTeacherSoft): "
        f"{int(pair.get('overlap_tp_count', 0))} shared TP | "
        f"overlap frac HLT={float(pair.get('overlap_tp_frac_of_a_tp', float('nan'))):.3f}, "
        f"RecoTeacherSoft={float(pair.get('overlap_tp_frac_of_b_tp', float('nan'))):.3f}"
    )

    np.savez_compressed(
        save_root / "concat_teacher_stageA_scores.npz",
        labels_val=labs_hlt_val.astype(np.float32),
        labels_test=labs_hlt_test.astype(np.float32),
        preds_hlt_val=preds_hlt_val.astype(np.float64),
        preds_hlt_test=preds_hlt_test.astype(np.float64),
        preds_concat_teacher_val=preds_concat_val.astype(np.float64),
        preds_concat_teacher_test=preds_concat_test.astype(np.float64),
        preds_reco_teacher_val=preds_reco_teacher_val.astype(np.float64),
        preds_reco_teacher_test=preds_reco_teacher_test.astype(np.float64),
        preds_corrected_only_val=preds_corr_val.astype(np.float64),
        preds_corrected_only_test=preds_corr_test.astype(np.float64),
        preds_corrected_only_joint_val=preds_corr_joint_val.astype(np.float64),
        preds_corrected_only_joint_test=preds_corr_joint_test.astype(np.float64),
        preds_dual_pre_val=preds_dual_pre_val.astype(np.float64),
        preds_dual_pre_test=preds_dual_pre_test.astype(np.float64),
        preds_dual_joint_val=preds_dual_joint_val.astype(np.float64),
        preds_dual_joint_test=preds_dual_joint_test.astype(np.float64),
        auc_hlt_val=float(auc_hlt_val),
        auc_hlt_test=float(auc_hlt_test),
        auc_concat_teacher_val=float(auc_concat_val),
        auc_concat_teacher_test=float(auc_concat_test),
        auc_reco_teacher_val=float(auc_reco_teacher_val),
        auc_reco_teacher_test=float(auc_reco_teacher_test),
        auc_corrected_only_val=float(auc_corr_val),
        auc_corrected_only_test=float(auc_corr_test),
        auc_corrected_only_joint_val=float(auc_corr_joint_val),
        auc_corrected_only_joint_test=float(auc_corr_joint_test),
        auc_dual_pre_val=float(auc_dual_pre_val),
        auc_dual_pre_test=float(auc_dual_pre_test),
        auc_dual_joint_val=float(auc_dual_joint_val),
        auc_dual_joint_test=float(auc_dual_joint_test),
        fpr50_hlt=float(fpr50_hlt),
        fpr50_concat_teacher=float(fpr50_concat),
        fpr50_reco_teacher=float(fpr50_reco),
        fpr50_corrected_only=float(fpr50_corr),
        fpr50_corrected_only_joint=float(fpr50_corr_joint),
        fpr50_dual_pre=float(fpr50_dual_pre),
        fpr50_dual_joint=float(fpr50_dual_joint),
        target_tpr=float(args.report_target_tpr),
    )

    with open(save_root / "concat_teacher_stageA_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": "concat_teacher_stageA_then_corrected",
                "rho": float(rho),
                "max_concat_constits": int(max_concat_constits),
                "stageA_reconstructor": reco_val_metrics,
                "stageB_dual_pre": stageB_dual_metrics,
                "stageC_dual_joint": stageC_dual_metrics,
                "stageC_corrected_only_joint": corrected_joint_metrics,
                "hlt": {
                    "auc_val": float(auc_hlt_val),
                    "auc_test": float(auc_hlt_test),
                    "delta_ref_threshold_prob": float(hlt_thr_prob),
                    "delta_ref_val_tpr": float(hlt_thr_tpr),
                    "delta_ref_val_fpr": float(hlt_thr_fpr),
                },
                "concat_teacher": {
                    "auc_val": float(auc_concat_val),
                    "auc_test": float(auc_concat_test),
                    "fpr50_test": float(fpr50_concat),
                },
                "reco_teacher_soft": {
                    "auc_val": float(auc_reco_teacher_val),
                    "auc_test": float(auc_reco_teacher_test),
                    "fpr50_val": float(fpr50_reco_teacher_val),
                    "fpr50_test": float(fpr50_reco_teacher_test),
                },
                "corrected_only": {
                    "auc_val": float(auc_corr_val),
                    "auc_test": float(auc_corr_test),
                    "fpr50_test": float(fpr50_corr),
                },
                "corrected_only_joint": {
                    "auc_val": float(auc_corr_joint_val),
                    "auc_test": float(auc_corr_joint_test),
                    "fpr50_val": float(fpr50_corr_joint_val),
                    "fpr50_test": float(fpr50_corr_joint),
                },
                "dual_pre": {
                    "auc_val": float(auc_dual_pre_val),
                    "auc_test": float(auc_dual_pre_test),
                    "fpr50_val": float(fpr50_dual_pre_val),
                    "fpr50_test": float(fpr50_dual_pre),
                },
                "dual_joint": {
                    "auc_val": float(auc_dual_joint_val),
                    "auc_test": float(auc_dual_joint_test),
                    "fpr50_val": float(fpr50_dual_joint_val),
                    "fpr50_test": float(fpr50_dual_joint),
                },
                "overlap_report_tpr": overlap_report,
                "best_combo_hlt_reco_val_selected_eval_test": combo_hlt_reco_valsel,
                "best_combo_hlt_reco_test_posthoc": combo_hlt_reco_oracle,
                "best_combo_hlt_corrected_val_selected_eval_test": combo_hlt_corr_valsel,
                "best_combo_hlt_corrected_test_posthoc": combo_hlt_corr_oracle,
                "best_combo_hlt_corrected_joint_val_selected_eval_test": combo_hlt_corr_joint_valsel,
                "best_combo_hlt_corrected_joint_test_posthoc": combo_hlt_corr_joint_oracle,
                "best_combo_hlt_dual_pre_val_selected_eval_test": combo_hlt_dual_pre_valsel,
                "best_combo_hlt_dual_pre_test_posthoc": combo_hlt_dual_pre_oracle,
                "best_combo_hlt_dual_joint_val_selected_eval_test": combo_hlt_dual_joint_valsel,
                "best_combo_hlt_dual_joint_test_posthoc": combo_hlt_dual_joint_oracle,
            },
            f,
            indent=2,
        )

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg["hlt_effects"], "stats": hlt_stats}, f, indent=2)

    if not args.skip_save_models:
        torch.save({"model": baseline.state_dict(), "auc": float(auc_hlt_test)}, save_root / "baseline.pt")
        torch.save({"model": concat_teacher.state_dict(), "auc": float(auc_concat_test)}, save_root / "concat_teacher.pt")
        torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor_stageA.pt")
        torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor.pt")
        torch.save({"model": corrected_only.state_dict(), "auc": float(auc_corr_test)}, save_root / "corrected_only_tagger.pt")
        torch.save({"model": corrected_only_joint.state_dict(), "auc": float(auc_corr_joint_test)}, save_root / "corrected_only_joint_tagger.pt")
        torch.save({"model": dual_model.state_dict(), "auc": float(auc_dual_joint_test)}, save_root / "dual_joint_tagger.pt")
        if stageB_dual_states.get("selected", {}).get("dual") is not None:
            torch.save({"model": stageB_dual_states["selected"]["dual"], "auc": float(auc_dual_pre_test)}, save_root / "dual_pre_tagger.pt")
        torch.save({"model": reconstructor_corr_joint.state_dict(), "val": corrected_joint_metrics}, save_root / "offline_reconstructor_corrected_joint.pt")
        torch.save({"model": reconstructor_dual.state_dict(), "val": stageC_dual_metrics}, save_root / "offline_reconstructor_dual_joint.pt")

    print(f"\nSaved concat-teacher Stage-A + corrected-only/joint/dual results to: {save_root}")


if __name__ == "__main__":
    main()

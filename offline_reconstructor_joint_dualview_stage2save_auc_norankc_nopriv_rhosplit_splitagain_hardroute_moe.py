#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hard-routed MoE variant of nopriv rho-split + split-again pipeline.

Design:
1) Build pseudo-HLT from offline jets.
2) Train teacher (offline) + baseline (HLT).
3) Stage A: train two reconstructors with hard routing by HLT constituent count.
4) Stage B: train two dual-view taggers with frozen branch reconstructors.
   - Both taggers train on full train split.
   - Classification BCE is reweighted: in-branch route gets higher weight.
5) Stage C: hard-routed joint finetune per branch.
6) Test-time hard routing and merged prediction.

Routing rule:
- low branch:  hlt_count <= route_hlt_count_thr
- high branch: hlt_count >  route_hlt_count_thr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from unmerge_correct_hlt import (
    RANDOM_SEED,
    DualViewCrossAttnClassifier,
    JetDataset,
    ParticleTransformer,
    compute_features,
    get_scheduler,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
    eval_classifier,
)
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    OfflineReconstructor,
    ReconstructionDataset,
    apply_hlt_effects_realistic_nomap,
    compute_reconstruction_losses,
    fpr_at_target_tpr,
    plot_roc,
    train_reconstructor,
)

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain as base


def _deepcopy_config() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _clamp_target_scale(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


class WeightedJointDualDataset(base.JointDualDataset):
    def __init__(
        self,
        feat_hlt_reco: np.ndarray,
        feat_hlt_dual: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        super().__init__(
            feat_hlt_reco=feat_hlt_reco,
            feat_hlt_dual=feat_hlt_dual,
            mask_hlt=mask_hlt,
            const_hlt=const_hlt,
            const_off=const_off,
            mask_off=mask_off,
            budget_merge_true=budget_merge_true,
            budget_eff_true=budget_eff_true,
            labels=labels,
        )
        if sample_weight is None:
            sample_weight = np.ones((labels.shape[0],), dtype=np.float32)
        self.sample_weight = torch.tensor(sample_weight.astype(np.float32), dtype=torch.float32)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        out = super().__getitem__(i)
        out["sample_weight"] = self.sample_weight[i]
        return out


def _safe_auc_fpr50(labels: np.ndarray, preds: np.ndarray) -> Tuple[float, float]:
    if preds.size == 0:
        return float("nan"), float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels, preds))
    fpr, tpr, _ = roc_curve(labels, preds)
    fpr50 = float(fpr_at_target_tpr(fpr, tpr, 0.50))
    return auc, fpr50


def _safe_auc_fpr50_weighted(
    labels: np.ndarray, preds: np.ndarray, sample_weight: Optional[np.ndarray]
) -> Tuple[float, float]:
    if sample_weight is None:
        return _safe_auc_fpr50(labels, preds)
    if preds.size == 0 or sample_weight.size != preds.size:
        return float("nan"), float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels, preds, sample_weight=sample_weight))
    fpr, tpr, _ = roc_curve(labels, preds, sample_weight=sample_weight)
    fpr50 = float(fpr_at_target_tpr(fpr, tpr, 0.50))
    return auc, fpr50


@torch.no_grad()
def eval_joint_model_safe(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    dual_model.eval()
    reconstructor.eval()

    preds = []
    labs = []
    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        feat_hlt_dual = batch["feat_hlt_dual"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = base.build_soft_corrected_view(
            reco_out,
            weight_floor=corrected_weight_floor,
            scale_features_by_weight=True,
            include_flags=corrected_use_flags,
        )
        logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)
        p = torch.sigmoid(logits)
        preds.append(p.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    labs = np.concatenate(labs) if labs else np.zeros(0, dtype=np.float32)

    auc, fpr50 = _safe_auc_fpr50(labs, preds)
    return auc, preds, labs, fpr50


@torch.no_grad()
def eval_joint_model_safe_with_weights(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, float, float, float]:
    dual_model.eval()
    reconstructor.eval()

    preds = []
    labs = []
    weights = []
    has_weight = False
    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        feat_hlt_dual = batch["feat_hlt_dual"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = base.build_soft_corrected_view(
            reco_out,
            weight_floor=corrected_weight_floor,
            scale_features_by_weight=True,
            include_flags=corrected_use_flags,
        )
        logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)
        p = torch.sigmoid(logits)
        preds.append(p.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())
        if "sample_weight" in batch:
            has_weight = True
            weights.append(batch["sample_weight"].detach().cpu().numpy())

    preds = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    labs = np.concatenate(labs) if labs else np.zeros(0, dtype=np.float32)
    sw = np.concatenate(weights).astype(np.float32) if has_weight and weights else None

    auc_unw, fpr50_unw = _safe_auc_fpr50(labs, preds)
    auc_w, fpr50_w = _safe_auc_fpr50_weighted(labs, preds, sw)
    return auc_unw, preds, labs, fpr50_unw, auc_w, fpr50_w


def train_joint_dual_weighted(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    freeze_reconstructor: bool,
    epochs: int,
    patience: int,
    lr_dual: float,
    lr_reco: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_reco: float,
    lambda_rank: float,
    lambda_cons: float,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    min_epochs: int,
    select_metric: str = "auc",
    use_sample_weight: bool = False,
) -> Tuple[OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    for p in reconstructor.parameters():
        p.requires_grad = not freeze_reconstructor

    params = [{"params": dual_model.parameters(), "lr": float(lr_dual)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_dual), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_state_dual_sel = None
    best_state_reco_sel = None
    best_state_dual_auc = None
    best_state_reco_auc = None
    best_state_dual_fpr = None
    best_state_reco_fpr = None

    best_val_fpr50 = float("inf")
    best_val_auc = float("-inf")
    best_val_fpr50_unw = float("inf")
    best_val_auc_unw = float("-inf")
    best_val_fpr50_w = float("inf")
    best_val_auc_w = float("-inf")
    best_sel_score = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    sel_val_fpr50 = float("nan")
    sel_val_auc = float("nan")
    sel_val_fpr50_unw = float("nan")
    sel_val_auc_unw = float("nan")
    sel_val_fpr50_w = float("nan")
    sel_val_auc_w = float("nan")
    no_improve = 0

    for ep in tqdm(range(int(epochs)), desc=stage_name):
        dual_model.train()
        if freeze_reconstructor:
            reconstructor.eval()
        else:
            reconstructor.train()

        tr_loss = 0.0
        tr_cls = 0.0
        tr_rank = 0.0
        tr_reco = 0.0
        tr_cons = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt_reco = batch["feat_hlt_reco"].to(device)
            feat_hlt_dual = batch["feat_hlt_dual"].to(device)
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

            feat_b, mask_b = base.build_soft_corrected_view(
                reco_out,
                weight_floor=corrected_weight_floor,
                scale_features_by_weight=True,
                include_flags=corrected_use_flags,
            )
            logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)

            if use_sample_weight and "sample_weight" in batch:
                w = batch["sample_weight"].to(device)
                w = w / torch.clamp(w.mean(), min=1e-6)
                loss_cls_all = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                loss_cls = (loss_cls_all * w).mean()
            else:
                loss_cls = F.binary_cross_entropy_with_logits(logits, y)

            loss_rank = base.low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()

            if float(lambda_reco) > 0.0:
                reco_losses = compute_reconstruction_losses(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_merge,
                    b_eff,
                    BASE_CONFIG["loss"],
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
            torch.nn.utils.clip_grad_norm_(dual_model.parameters(), 1.0)
            if not freeze_reconstructor:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = feat_hlt_reco.size(0)
            tr_loss += loss.item() * bs
            tr_cls += loss_cls.item() * bs
            tr_rank += loss_rank.item() * bs
            tr_reco += loss_reco.item() * bs
            tr_cons += loss_cons.item() * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)

        va_auc_unw, _, _, va_fpr50_unw, va_auc_w, va_fpr50_w = eval_joint_model_safe_with_weights(
            reconstructor=reconstructor,
            dual_model=dual_model,
            loader=val_loader,
            device=device,
            corrected_weight_floor=corrected_weight_floor,
            corrected_use_flags=corrected_use_flags,
        )
        if not np.isfinite(va_auc_w):
            va_auc_w = va_auc_unw
        if not np.isfinite(va_fpr50_w):
            va_fpr50_w = va_fpr50_unw

        va_auc = va_auc_w if use_sample_weight else va_auc_unw
        va_fpr50 = va_fpr50_w if use_sample_weight else va_fpr50_unw

        if np.isfinite(va_fpr50_unw) and float(va_fpr50_unw) < best_val_fpr50_unw:
            best_val_fpr50_unw = float(va_fpr50_unw)
        if np.isfinite(va_auc_unw) and float(va_auc_unw) > best_val_auc_unw:
            best_val_auc_unw = float(va_auc_unw)
        if np.isfinite(va_fpr50_w) and float(va_fpr50_w) < best_val_fpr50_w:
            best_val_fpr50_w = float(va_fpr50_w)
        if np.isfinite(va_auc_w) and float(va_auc_w) > best_val_auc_w:
            best_val_auc_w = float(va_auc_w)

        if np.isfinite(va_fpr50) and float(va_fpr50) < best_val_fpr50:
            best_val_fpr50 = float(va_fpr50)
            best_state_dual_fpr = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco_fpr = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state_dual_auc = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
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
            sel_val_fpr50_unw = float(va_fpr50_unw)
            sel_val_auc_unw = float(va_auc_unw)
            sel_val_fpr50_w = float(va_fpr50_w)
            sel_val_auc_w = float(va_auc_w)
            best_state_dual_sel = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco_sel = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print_every = 1 if str(stage_name).startswith("StageC") else 5
        if (ep + 1) % print_every == 0:
            val_msg = (
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}"
                if not use_sample_weight
                else (
                    f"val_auc_sel={va_auc:.4f}, val_fpr50_sel={va_fpr50:.6f} "
                    f"(unw_auc={va_auc_unw:.4f}, unw_fpr50={va_fpr50_unw:.6f}, "
                    f"w_auc={va_auc_w:.4f}, w_fpr50={va_fpr50_w:.6f})"
                )
            )
            print(
                f"{stage_name} ep {ep+1}: train_loss={tr_loss:.4f} "
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}) | "
                f"{val_msg}, "
                f"select={str(select_metric).lower()}, best_sel={best_sel_score:.6f}"
            )

        if (ep + 1) >= int(min_epochs) and no_improve >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_dual_sel is not None:
        dual_model.load_state_dict(best_state_dual_sel)
    if best_state_reco_sel is not None:
        reconstructor.load_state_dict(best_state_reco_sel)

    metrics = {
        "selection_metric": str(select_metric).lower(),
        "selection_source": "weighted" if use_sample_weight else "unweighted",
        "selected_val_fpr50": float(sel_val_fpr50),
        "selected_val_auc": float(sel_val_auc),
        "selected_val_fpr50_unweighted": float(sel_val_fpr50_unw),
        "selected_val_auc_unweighted": float(sel_val_auc_unw),
        "selected_val_fpr50_weighted": float(sel_val_fpr50_w),
        "selected_val_auc_weighted": float(sel_val_auc_w),
        "best_val_fpr50_seen": float(best_val_fpr50),
        "best_val_auc_seen": float(best_val_auc),
        "best_val_fpr50_seen_unweighted": float(best_val_fpr50_unw),
        "best_val_auc_seen_unweighted": float(best_val_auc_unw),
        "best_val_fpr50_seen_weighted": float(best_val_fpr50_w),
        "best_val_auc_seen_weighted": float(best_val_auc_w),
    }
    state_pack = {
        "selected": {"dual": best_state_dual_sel, "reco": best_state_reco_sel},
        "auc": {"dual": best_state_dual_auc, "reco": best_state_reco_auc},
        "fpr50": {"dual": best_state_dual_fpr, "reco": best_state_reco_fpr},
    }
    return reconstructor, dual_model, metrics, state_pack


def _subset_joint_dataset(
    feat_hlt_std: np.ndarray,
    feat_hlt_dual: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    const_off: np.ndarray,
    masks_off: np.ndarray,
    budget_merge_true: np.ndarray,
    budget_eff_true: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> WeightedJointDualDataset:
    return WeightedJointDualDataset(
        feat_hlt_reco=feat_hlt_std[idx],
        feat_hlt_dual=feat_hlt_dual[idx],
        mask_hlt=hlt_mask[idx],
        const_hlt=hlt_const[idx],
        const_off=const_off[idx],
        mask_off=masks_off[idx],
        budget_merge_true=budget_merge_true[idx],
        budget_eff_true=budget_eff_true[idx],
        labels=labels[idx],
        sample_weight=sample_weight,
    )


def _subset_reco_dataset(
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    const_off: np.ndarray,
    masks_off: np.ndarray,
    budget_merge_true: np.ndarray,
    budget_eff_true: np.ndarray,
    idx: np.ndarray,
) -> ReconstructionDataset:
    return ReconstructionDataset(
        feat_hlt_std[idx], hlt_mask[idx], hlt_const[idx],
        const_off[idx], masks_off[idx],
        budget_merge_true[idx], budget_eff_true[idx],
    )


def _route_eval(
    reconstructor_low: OfflineReconstructor,
    dual_low: nn.Module,
    reconstructor_high: OfflineReconstructor,
    dual_high: nn.Module,
    feat_hlt_std: np.ndarray,
    feat_hlt_dual: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    const_off: np.ndarray,
    masks_off: np.ndarray,
    budget_merge_true: np.ndarray,
    budget_eff_true: np.ndarray,
    labels: np.ndarray,
    eval_idx: np.ndarray,
    route_low_mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
) -> Tuple[float, np.ndarray, np.ndarray, float, Dict[str, Dict[str, float]]]:
    local_route_low = route_low_mask[eval_idx]
    idx_low = eval_idx[local_route_low]
    idx_high = eval_idx[~local_route_low]

    preds = np.zeros((eval_idx.shape[0],), dtype=np.float32)
    labs = labels[eval_idx].astype(np.float32)

    branch_metrics: Dict[str, Dict[str, float]] = {}

    if idx_low.size > 0:
        ds_low = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const, const_off, masks_off,
            budget_merge_true, budget_eff_true, labels, idx_low,
        )
        dl_low = DataLoader(
            ds_low, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
        auc_low, p_low, l_low, fpr50_low = eval_joint_model_safe(
            reconstructor_low,
            dual_low,
            dl_low,
            device,
            corrected_weight_floor,
            corrected_use_flags,
        )
        preds[local_route_low] = p_low.astype(np.float32)
        if l_low.size == labs[local_route_low].size:
            assert np.array_equal(l_low.astype(np.float32), labs[local_route_low].astype(np.float32))
        fpr30_low = float("nan")
        if l_low.size > 0 and len(np.unique(l_low)) > 1:
            fpr_l, tpr_l, _ = roc_curve(l_low, p_low)
            fpr30_low = float(fpr_at_target_tpr(fpr_l, tpr_l, 0.30))
        branch_metrics["low"] = {
            "n": int(idx_low.size),
            "auc": float(auc_low),
            "fpr30": float(fpr30_low),
            "fpr50": float(fpr50_low),
        }
    else:
        branch_metrics["low"] = {"n": 0, "auc": float("nan"), "fpr30": float("nan"), "fpr50": float("nan")}

    if idx_high.size > 0:
        ds_high = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const, const_off, masks_off,
            budget_merge_true, budget_eff_true, labels, idx_high,
        )
        dl_high = DataLoader(
            ds_high, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
        auc_high, p_high, l_high, fpr50_high = eval_joint_model_safe(
            reconstructor_high,
            dual_high,
            dl_high,
            device,
            corrected_weight_floor,
            corrected_use_flags,
        )
        preds[~local_route_low] = p_high.astype(np.float32)
        if l_high.size == labs[~local_route_low].size:
            assert np.array_equal(l_high.astype(np.float32), labs[~local_route_low].astype(np.float32))
        fpr30_high = float("nan")
        if l_high.size > 0 and len(np.unique(l_high)) > 1:
            fpr_h, tpr_h, _ = roc_curve(l_high, p_high)
            fpr30_high = float(fpr_at_target_tpr(fpr_h, tpr_h, 0.30))
        branch_metrics["high"] = {
            "n": int(idx_high.size),
            "auc": float(auc_high),
            "fpr30": float(fpr30_high),
            "fpr50": float(fpr50_high),
        }
    else:
        branch_metrics["high"] = {"n": 0, "auc": float("nan"), "fpr30": float("nan"), "fpr50": float("nan")}

    auc_all, fpr50_all = _safe_auc_fpr50(labs, preds)
    return auc_all, preds, labs, fpr50_all, branch_metrics


def _subset_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    if labels.size == 0:
        return {"n": 0, "auc": float("nan"), "fpr30": float("nan"), "fpr50": float("nan")}
    auc, fpr50 = _safe_auc_fpr50(labels, preds)
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, preds)
        fpr30 = float(fpr_at_target_tpr(fpr, tpr, 0.30))
    else:
        fpr30 = float("nan")
    return {"n": int(labels.size), "auc": float(auc), "fpr30": float(fpr30), "fpr50": float(fpr50)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=50000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument(
        "--n_train_split",
        type=int,
        default=-1,
        help="If >0 and paired with --n_val_split/--n_test_split, use exact split counts instead of 70/15/15.",
    )
    parser.add_argument("--n_val_split", type=int, default=-1)
    parser.add_argument("--n_test_split", type=int, default=-1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_joint"),
    )
    parser.add_argument("--run_name", type=str, default="joint_hardroute_default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--skip_save_models", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    # HLT controls
    parser.add_argument("--merge_radius", type=float, default=BASE_CONFIG["hlt_effects"]["merge_radius"])
    parser.add_argument("--eff_plateau_barrel", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    parser.add_argument("--eff_plateau_endcap", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    parser.add_argument("--smear_a", type=float, default=BASE_CONFIG["hlt_effects"]["smear_a"])
    parser.add_argument("--smear_b", type=float, default=BASE_CONFIG["hlt_effects"]["smear_b"])
    parser.add_argument("--smear_c", type=float, default=BASE_CONFIG["hlt_effects"]["smear_c"])

    # Stage A
    parser.add_argument("--stageA_epochs", type=int, default=90)
    parser.add_argument("--stageA_patience", type=int, default=18)

    # Stage B
    parser.add_argument("--stageB_epochs", type=int, default=45)
    parser.add_argument("--stageB_patience", type=int, default=12)
    parser.add_argument("--stageB_min_epochs", type=int, default=12)
    parser.add_argument("--stageB_lr_dual", type=float, default=4e-4)
    parser.add_argument("--stageB_lambda_rank", type=float, default=0.0)
    parser.add_argument("--stageB_lambda_cons", type=float, default=0.0)

    parser.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])

    # Stage C
    parser.add_argument("--stageC_epochs", type=int, default=65)
    parser.add_argument("--stageC_patience", type=int, default=14)
    parser.add_argument("--stageC_min_epochs", type=int, default=25)
    parser.add_argument("--stageC_lr_dual", type=float, default=2e-4)
    parser.add_argument("--stageC_lr_reco", type=float, default=1e-4)
    parser.add_argument("--lambda_reco", type=float, default=0.35)
    parser.add_argument("--lambda_rank", type=float, default=0.0)
    parser.add_argument("--lambda_cons", type=float, default=0.06)
    parser.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    parser.add_argument("--use_corrected_flags", action="store_true")

    # split-again controls
    parser.add_argument("--disable_split_again", action="store_true")
    parser.add_argument("--split_again_exist_thr", type=float, default=0.30)
    parser.add_argument("--split_again_psplit_thr", type=float, default=0.20)
    parser.add_argument("--split_again_dr_thr", type=float, default=0.10)
    parser.add_argument("--split_again_alpha", type=float, default=8.0)
    parser.add_argument("--split_again_beta", type=float, default=4.0)
    parser.add_argument("--split_again_gamma", type=float, default=3.0)
    parser.add_argument("--split_again_score_power", type=float, default=1.25)
    parser.add_argument("--split_again_budget_frac", type=float, default=1.0)
    parser.add_argument("--split_again_max_parent_added", type=float, default=2.0)

    parser.add_argument("--loss_w_pt_ratio", type=float, default=BASE_CONFIG["loss"]["w_pt_ratio"])
    parser.add_argument("--loss_w_e_ratio", type=float, default=BASE_CONFIG["loss"]["w_e_ratio"])
    parser.add_argument("--loss_w_budget", type=float, default=BASE_CONFIG["loss"]["w_budget"])
    parser.add_argument("--loss_w_sparse", type=float, default=BASE_CONFIG["loss"]["w_sparse"])
    parser.add_argument("--loss_w_local", type=float, default=BASE_CONFIG["loss"]["w_local"])

    parser.add_argument(
        "--added_target_scale",
        type=float,
        default=0.90,
        help="rho in [0,1] for non-privileged split targets: merge=rho*true_added, eff=(1-rho)*true_added.",
    )

    # Hard-route MoE controls
    parser.add_argument("--route_hlt_count_thr", type=int, default=26)
    parser.add_argument("--stageB_route_weight", type=float, default=5.0)
    parser.add_argument(
        "--stageB_hard_route",
        action="store_true",
        help="If set, Stage B also trains/validates each branch on its own hard-routed subset only.",
    )
    parser.add_argument("--route_boundary_band", type=int, default=2)

    # API compatibility
    parser.add_argument("--disable_final_kd", action="store_true")

    args = parser.parse_args()
    base.set_seed(int(args.seed))

    selection_metric = "auc"
    if str(args.selection_metric).lower() != "auc":
        print(f"Note: overriding --selection_metric={args.selection_metric} to 'auc' in this script.")

    cfg = _deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)
    cfg["loss"]["w_pt_ratio"] = float(args.loss_w_pt_ratio)
    cfg["loss"]["w_e_ratio"] = float(args.loss_w_e_ratio)
    cfg["loss"]["w_budget"] = float(args.loss_w_budget)
    cfg["loss"]["w_sparse"] = float(args.loss_w_sparse)
    cfg["loss"]["w_local"] = float(args.loss_w_local)

    base.SPLIT_AGAIN_CFG["enabled"] = not bool(args.disable_split_again)
    base.SPLIT_AGAIN_CFG["exist_thr"] = float(args.split_again_exist_thr)
    base.SPLIT_AGAIN_CFG["psplit_thr"] = float(args.split_again_psplit_thr)
    base.SPLIT_AGAIN_CFG["dr_thr"] = float(args.split_again_dr_thr)
    base.SPLIT_AGAIN_CFG["alpha"] = float(args.split_again_alpha)
    base.SPLIT_AGAIN_CFG["beta"] = float(args.split_again_beta)
    base.SPLIT_AGAIN_CFG["gamma"] = float(args.split_again_gamma)
    base.SPLIT_AGAIN_CFG["score_power"] = float(args.split_again_score_power)
    base.SPLIT_AGAIN_CFG["budget_frac"] = float(args.split_again_budget_frac)
    base.SPLIT_AGAIN_CFG["max_parent_added"] = float(args.split_again_max_parent_added)

    cfg["reconstructor_training"]["epochs"] = int(args.stageA_epochs)
    cfg["reconstructor_training"]["patience"] = int(args.stageA_patience)

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

    max_jets_needed = args.offset_jets + args.n_train_jets
    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}"
        )

    const_raw = all_const_full[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )

    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    rho = _clamp_target_scale(float(args.added_target_scale))
    budget_merge_true_raw = (rho * true_added_raw).astype(np.float32)
    budget_eff_true_raw = ((1.0 - rho) * true_added_raw).astype(np.float32)
    budget_merge_true = budget_merge_true_raw.copy()
    budget_eff_true = budget_eff_true_raw.copy()
    print(
        f"Non-priv rho split setup: rho={rho:.3f}, "
        f"mean_true_added_raw={float(true_added_raw.mean()):.3f}, "
        f"mean_target_merge={float(budget_merge_true.mean()):.3f}, "
        f"mean_target_eff={float(budget_eff_true.mean()):.3f}"
    )

    print("Computing features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    n_train_split = int(args.n_train_split)
    n_val_split = int(args.n_val_split)
    n_test_split = int(args.n_test_split)
    custom_split = (n_train_split > 0 and n_val_split > 0 and n_test_split > 0)

    idx = np.arange(len(labels))
    if custom_split:
        total_need = int(n_train_split + n_val_split + n_test_split)
        if total_need > len(idx):
            raise ValueError(
                f"Requested split counts exceed available jets: "
                f"{n_train_split}+{n_val_split}+{n_test_split} > {len(idx)}"
            )
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
            train_size=int(n_train_split),
            random_state=int(args.seed),
            stratify=labels[idx_use],
        )
        val_idx, test_idx = train_test_split(
            rem_idx,
            train_size=int(n_val_split),
            test_size=int(n_test_split),
            random_state=int(args.seed),
            stratify=labels[rem_idx],
        )
    else:
        train_idx, temp_idx = train_test_split(
            idx, test_size=0.30, random_state=int(args.seed), stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=int(args.seed), stratify=labels[temp_idx]
        )
    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} "
        f"(custom_counts={custom_split})"
    )

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)
    feat_hlt_dual = feat_hlt_std.astype(np.float32, copy=True)

    route_thr = int(args.route_hlt_count_thr)
    route_weight = float(args.stageB_route_weight)
    stageB_hard_route = bool(args.stageB_hard_route)
    route_low_mask = (hlt_mask.sum(axis=1).astype(np.int32) <= route_thr)

    train_low = int(route_low_mask[train_idx].sum())
    val_low = int(route_low_mask[val_idx].sum())
    test_low = int(route_low_mask[test_idx].sum())
    print(
        f"Hard routing: low if hlt_count <= {route_thr}. "
        f"Train low/high={train_low}/{len(train_idx)-train_low}, "
        f"Val low/high={val_low}/{len(val_idx)-val_low}, "
        f"Test low/high={test_low}/{len(test_idx)-test_low}"
    )

    data_setup = {
        "variant": "nopriv_rhosplit_splitagain_hardroute_moe",
        "n_train_jets": int(args.n_train_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "seed": int(args.seed),
        "split": (
            {
                "mode": "custom_counts",
                "n_train_split": int(len(train_idx)),
                "n_val_split": int(len(val_idx)),
                "n_test_split": int(len(test_idx)),
            }
            if custom_split
            else {"mode": "fractions", "train_frac": 0.70, "val_frac": 0.15, "test_frac": 0.15}
        ),
        "rho": float(rho),
        "route_hlt_count_thr": int(route_thr),
        "stageB_route_weight": float(route_weight),
        "stageB_hard_route": bool(stageB_hard_route),
        "split_again": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in base.SPLIT_AGAIN_CFG.items()},
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
        route_low_mask=route_low_mask.astype(np.uint8),
    )

    print("\n" + "=" * 70)
    print("STEP 1: TEACHER + BASELINE")
    print("=" * 70)
    BS = int(cfg["training"]["batch_size"])

    ds_train_off = JetDataset(feat_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    ds_val_off = JetDataset(feat_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    ds_test_off = JetDataset(feat_off_std[test_idx], masks_off[test_idx], labels[test_idx])
    dl_train_off = DataLoader(ds_train_off, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_off = DataLoader(ds_val_off, batch_size=BS, shuffle=False)
    dl_test_off = DataLoader(ds_test_off, batch_size=BS, shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = base.train_single_view_classifier_auc(
        teacher, dl_train_off, dl_val_off, device, cfg["training"], name="Teacher"
    )
    auc_teacher, preds_teacher, labs = eval_classifier(teacher, dl_test_off, device)

    ds_train_hlt = JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx])
    ds_val_hlt = JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    ds_test_hlt = JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_train_hlt = DataLoader(ds_train_hlt, batch_size=BS, shuffle=True, drop_last=True)
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=BS, shuffle=False)
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=BS, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = base.train_single_view_classifier_auc(
        baseline, dl_train_hlt, dl_val_hlt, device, cfg["training"], name="Baseline"
    )
    auc_baseline, preds_baseline, _ = eval_classifier(baseline, dl_test_hlt, device)

    train_idx_low = train_idx[route_low_mask[train_idx]]
    train_idx_high = train_idx[~route_low_mask[train_idx]]
    val_idx_low = val_idx[route_low_mask[val_idx]]
    val_idx_high = val_idx[~route_low_mask[val_idx]]

    if train_idx_low.size == 0 or train_idx_high.size == 0:
        raise RuntimeError("Routing split is degenerate; one branch has zero training jets.")

    print("\n" + "=" * 70)
    print("STEP 2A: STAGE A LOW-BRANCH (RECONSTRUCTOR PRETRAIN)")
    print("=" * 70)
    ds_train_reco_low = _subset_reco_dataset(
        feat_hlt_std, hlt_mask, hlt_const, const_off, masks_off,
        budget_merge_true, budget_eff_true, train_idx_low,
    )
    ds_val_reco_low = _subset_reco_dataset(
        feat_hlt_std, hlt_mask, hlt_const, const_off, masks_off,
        budget_merge_true, budget_eff_true, val_idx_low,
    )
    dl_train_reco_low = DataLoader(
        ds_train_reco_low,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_reco_low = DataLoader(
        ds_val_reco_low,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    reconstructor_low = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    BASE_CONFIG["loss"] = cfg["loss"]
    reconstructor_low, reco_low_metrics = train_reconstructor(
        reconstructor_low,
        dl_train_reco_low,
        dl_val_reco_low,
        device,
        cfg["reconstructor_training"],
        cfg["loss"],
    )

    print("\n" + "=" * 70)
    print("STEP 2B: STAGE A HIGH-BRANCH (RECONSTRUCTOR PRETRAIN)")
    print("=" * 70)
    ds_train_reco_high = _subset_reco_dataset(
        feat_hlt_std, hlt_mask, hlt_const, const_off, masks_off,
        budget_merge_true, budget_eff_true, train_idx_high,
    )
    ds_val_reco_high = _subset_reco_dataset(
        feat_hlt_std, hlt_mask, hlt_const, const_off, masks_off,
        budget_merge_true, budget_eff_true, val_idx_high,
    )
    dl_train_reco_high = DataLoader(
        ds_train_reco_high,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_reco_high = DataLoader(
        ds_val_reco_high,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    reconstructor_high = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor_high, reco_high_metrics = train_reconstructor(
        reconstructor_high,
        dl_train_reco_high,
        dl_val_reco_high,
        device,
        cfg["reconstructor_training"],
        cfg["loss"],
    )

    print("\n" + "=" * 70)
    if stageB_hard_route:
        print("STEP 3: STAGE B (DUAL PRETRAIN, FROZEN RECO, HARD ROUTED)")
    else:
        print("STEP 3: STAGE B (DUAL PRETRAIN, FROZEN RECO, FULL TRAIN + WEIGHTED BCE)")
    print("=" * 70)

    if stageB_hard_route:
        stageb_use_weight = False
        print(
            f"Stage-B hard route enabled: "
            f"low train/val={len(train_idx_low)}/{len(val_idx_low)}, "
            f"high train/val={len(train_idx_high)}/{len(val_idx_high)}"
        )
        ds_train_joint_low = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
            const_off, masks_off, budget_merge_true, budget_eff_true,
            labels, train_idx_low,
        )
        ds_train_joint_high = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
            const_off, masks_off, budget_merge_true, budget_eff_true,
            labels, train_idx_high,
        )
        ds_val_joint_low_stageb = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
            const_off, masks_off, budget_merge_true, budget_eff_true,
            labels, val_idx_low,
        )
        ds_val_joint_high_stageb = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
            const_off, masks_off, budget_merge_true, budget_eff_true,
            labels, val_idx_high,
        )
    else:
        stageb_use_weight = True
        w_low = np.where(route_low_mask[train_idx], route_weight, 1.0).astype(np.float32)
        w_high = np.where(route_low_mask[train_idx], 1.0, route_weight).astype(np.float32)
        w_low_val = np.where(route_low_mask[val_idx], route_weight, 1.0).astype(np.float32)
        w_high_val = np.where(route_low_mask[val_idx], 1.0, route_weight).astype(np.float32)
        print(
            f"Stage-B route weighting: in-route weight={route_weight:.3f}, out-route weight=1.000 | "
            f"mean(w_low)={float(w_low.mean()):.3f}, mean(w_high)={float(w_high.mean()):.3f}"
        )
        ds_train_joint_low = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
            const_off, masks_off, budget_merge_true, budget_eff_true,
            labels, train_idx, sample_weight=w_low,
        )
        ds_train_joint_high = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
            const_off, masks_off, budget_merge_true, budget_eff_true,
            labels, train_idx, sample_weight=w_high,
        )
        ds_val_joint_low_stageb = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
            const_off, masks_off, budget_merge_true, budget_eff_true,
            labels, val_idx, sample_weight=w_low_val,
        )
        ds_val_joint_high_stageb = _subset_joint_dataset(
            feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
            const_off, masks_off, budget_merge_true, budget_eff_true,
            labels, val_idx, sample_weight=w_high_val,
        )
    ds_val_joint_low_stagec = _subset_joint_dataset(
        feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
        const_off, masks_off, budget_merge_true, budget_eff_true,
        labels, val_idx_low,
    )
    ds_val_joint_high_stagec = _subset_joint_dataset(
        feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
        const_off, masks_off, budget_merge_true, budget_eff_true,
        labels, val_idx_high,
    )

    dl_train_joint_low = DataLoader(
        ds_train_joint_low,
        batch_size=BS,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_train_joint_high = DataLoader(
        ds_train_joint_high,
        batch_size=BS,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint_low_stageb = DataLoader(
        ds_val_joint_low_stageb,
        batch_size=BS,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint_high_stageb = DataLoader(
        ds_val_joint_high_stageb,
        batch_size=BS,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint_low_stagec = DataLoader(
        ds_val_joint_low_stagec,
        batch_size=BS,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint_high_stagec = DataLoader(
        ds_val_joint_high_stagec,
        batch_size=BS,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dual_input_dim_a = int(feat_hlt_dual.shape[-1])
    dual_input_dim_b = 12 if bool(args.use_corrected_flags) else 10

    dual_low = DualViewCrossAttnClassifier(
        input_dim_a=dual_input_dim_a,
        input_dim_b=dual_input_dim_b,
        **cfg["model"],
    ).to(device)
    dual_high = DualViewCrossAttnClassifier(
        input_dim_a=dual_input_dim_a,
        input_dim_b=dual_input_dim_b,
        **cfg["model"],
    ).to(device)

    reconstructor_low, dual_low, stageB_low_metrics, stageB_low_states = train_joint_dual_weighted(
        reconstructor=reconstructor_low,
        dual_model=dual_low,
        train_loader=dl_train_joint_low,
        val_loader=dl_val_joint_low_stageb,
        device=device,
        stage_name="StageB-DualPretrain-Low",
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
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
        min_epochs=int(args.stageB_min_epochs),
        select_metric=selection_metric,
        use_sample_weight=stageb_use_weight,
    )

    reconstructor_high, dual_high, stageB_high_metrics, stageB_high_states = train_joint_dual_weighted(
        reconstructor=reconstructor_high,
        dual_model=dual_high,
        train_loader=dl_train_joint_high,
        val_loader=dl_val_joint_high_stageb,
        device=device,
        stage_name="StageB-DualPretrain-High",
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
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
        min_epochs=int(args.stageB_min_epochs),
        select_metric=selection_metric,
        use_sample_weight=stageb_use_weight,
    )

    stage2_reco_low_state = {k: v.detach().cpu().clone() for k, v in reconstructor_low.state_dict().items()}
    stage2_dual_low_state = {k: v.detach().cpu().clone() for k, v in dual_low.state_dict().items()}
    stage2_reco_high_state = {k: v.detach().cpu().clone() for k, v in reconstructor_high.state_dict().items()}
    stage2_dual_high_state = {k: v.detach().cpu().clone() for k, v in dual_high.state_dict().items()}

    auc_stage2, preds_stage2, labs_stage2, _, stage2_branch_metrics = _route_eval(
        reconstructor_low,
        dual_low,
        reconstructor_high,
        dual_high,
        feat_hlt_std,
        feat_hlt_dual,
        hlt_mask,
        hlt_const,
        const_off,
        masks_off,
        budget_merge_true,
        budget_eff_true,
        labels,
        test_idx,
        route_low_mask,
        device,
        BS,
        int(args.num_workers),
        float(args.corrected_weight_floor),
        bool(args.use_corrected_flags),
    )
    assert np.array_equal(labs.astype(np.float32), labs_stage2.astype(np.float32))

    print("\n" + "=" * 70)
    print("STEP 4: STAGE C (JOINT FINETUNE, HARD ROUTED)")
    print("=" * 70)

    ds_train_joint_low_c = _subset_joint_dataset(
        feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
        const_off, masks_off, budget_merge_true, budget_eff_true,
        labels, train_idx_low,
    )
    ds_train_joint_high_c = _subset_joint_dataset(
        feat_hlt_std, feat_hlt_dual, hlt_mask, hlt_const,
        const_off, masks_off, budget_merge_true, budget_eff_true,
        labels, train_idx_high,
    )

    dl_train_joint_low_c = DataLoader(
        ds_train_joint_low_c,
        batch_size=BS,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_train_joint_high_c = DataLoader(
        ds_train_joint_high_c,
        batch_size=BS,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    reconstructor_low, dual_low, stageC_low_metrics, stageC_low_states = train_joint_dual_weighted(
        reconstructor=reconstructor_low,
        dual_model=dual_low,
        train_loader=dl_train_joint_low_c,
        val_loader=dl_val_joint_low_stagec,
        device=device,
        stage_name="StageC-Joint-Low",
        freeze_reconstructor=False,
        epochs=int(args.stageC_epochs),
        patience=int(args.stageC_patience),
        lr_dual=float(args.stageC_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        lambda_reco=float(args.lambda_reco),
        lambda_rank=0.0,
        lambda_cons=float(args.lambda_cons),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
        min_epochs=int(args.stageC_min_epochs),
        select_metric=selection_metric,
        use_sample_weight=False,
    )

    reconstructor_high, dual_high, stageC_high_metrics, stageC_high_states = train_joint_dual_weighted(
        reconstructor=reconstructor_high,
        dual_model=dual_high,
        train_loader=dl_train_joint_high_c,
        val_loader=dl_val_joint_high_stagec,
        device=device,
        stage_name="StageC-Joint-High",
        freeze_reconstructor=False,
        epochs=int(args.stageC_epochs),
        patience=int(args.stageC_patience),
        lr_dual=float(args.stageC_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        lambda_reco=float(args.lambda_reco),
        lambda_rank=0.0,
        lambda_cons=float(args.lambda_cons),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
        min_epochs=int(args.stageC_min_epochs),
        select_metric=selection_metric,
        use_sample_weight=False,
    )

    auc_joint, preds_joint, labs_joint, _, joint_branch_metrics = _route_eval(
        reconstructor_low,
        dual_low,
        reconstructor_high,
        dual_high,
        feat_hlt_std,
        feat_hlt_dual,
        hlt_mask,
        hlt_const,
        const_off,
        masks_off,
        budget_merge_true,
        budget_eff_true,
        labels,
        test_idx,
        route_low_mask,
        device,
        BS,
        int(args.num_workers),
        float(args.corrected_weight_floor),
        bool(args.use_corrected_flags),
    )
    assert np.array_equal(labs.astype(np.float32), labs_joint.astype(np.float32))

    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_s2, tpr_s2, _ = roc_curve(labs, preds_stage2)
    fpr_j, tpr_j, _ = roc_curve(labs, preds_joint)

    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.30)
    fpr30_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.30)
    fpr30_stage2 = fpr_at_target_tpr(fpr_s2, tpr_s2, 0.30)
    fpr30_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.30)

    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.50)
    fpr50_stage2 = fpr_at_target_tpr(fpr_s2, tpr_s2, 0.50)
    fpr50_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.50)

    hlt_count_test = hlt_mask[test_idx].sum(axis=1).astype(np.int32)
    b = int(args.route_boundary_band)
    boundary_lo = int(route_thr - b)
    boundary_hi = int(route_thr + b)
    boundary_mask = (hlt_count_test >= boundary_lo) & (hlt_count_test <= boundary_hi)
    boundary_metrics = {
        "range": [int(boundary_lo), int(boundary_hi)],
        "teacher": _subset_metrics(labs[boundary_mask], preds_teacher[boundary_mask]),
        "baseline": _subset_metrics(labs[boundary_mask], preds_baseline[boundary_mask]),
        "stage2": _subset_metrics(labs[boundary_mask], preds_stage2[boundary_mask]),
        "joint": _subset_metrics(labs[boundary_mask], preds_joint[boundary_mask]),
    }

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"Stage2 (PreJoint Routed) AUC: {auc_stage2:.4f}")
    print(f"Joint Dual-View (HardRoute) AUC: {auc_joint:.4f}")
    print()
    print(
        f"FPR@30 Teacher/Baseline/Stage2/Joint: "
        f"{fpr30_teacher:.6f} / {fpr30_baseline:.6f} / {fpr30_stage2:.6f} / {fpr30_joint:.6f}"
    )
    print(
        f"FPR@50 Teacher/Baseline/Stage2/Joint: "
        f"{fpr50_teacher:.6f} / {fpr50_baseline:.6f} / {fpr50_stage2:.6f} / {fpr50_joint:.6f}"
    )
    print(
        f"Boundary band HLT count [{boundary_lo},{boundary_hi}] n={int(boundary_mask.sum())}: "
        f"Baseline FPR50={boundary_metrics['baseline']['fpr50']:.6f}, "
        f"Joint FPR50={boundary_metrics['joint']['fpr50']:.6f}"
    )

    plot_lines = [
        (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
        (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
        (tpr_s2, fpr_s2, "-.", f"Stage2 Routed (AUC={auc_stage2:.3f})", "darkorange"),
        (tpr_j, fpr_j, "-.", f"Joint HardRoute (AUC={auc_joint:.3f})", "darkslateblue"),
    ]
    plot_roc(
        plot_lines,
        save_root / "results_teacher_baseline_joint_hardroute.png",
        min_fpr=1e-4,
    )

    np.savez(
        save_root / "results.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_stage2=auc_stage2,
        auc_joint=auc_joint,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_stage2=fpr_s2,
        tpr_stage2=tpr_s2,
        fpr_joint=fpr_j,
        tpr_joint=tpr_j,
        fpr30_teacher=fpr30_teacher,
        fpr30_baseline=fpr30_baseline,
        fpr30_stage2=fpr30_stage2,
        fpr30_joint=fpr30_joint,
        fpr50_teacher=fpr50_teacher,
        fpr50_baseline=fpr50_baseline,
        fpr50_stage2=fpr50_stage2,
        fpr50_joint=fpr50_joint,
        hlt_count_test=hlt_count_test,
        route_low_test=(hlt_count_test <= route_thr).astype(np.uint8),
    )

    with open(save_root / "joint_stage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": {
                    "mode": "nopriv_rhosplit_splitagain_hardroute_moe",
                    "rho": float(rho),
                    "route_hlt_count_thr": int(route_thr),
                    "stageB_route_weight": float(route_weight),
                    "stageB_hard_route": bool(stageB_hard_route),
                    "split_again": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in base.SPLIT_AGAIN_CFG.items()},
                    "boundary_band": int(args.route_boundary_band),
                },
                "routing_counts": {
                    "train_low": int(train_idx_low.size),
                    "train_high": int(train_idx_high.size),
                    "val_low": int(val_idx_low.size),
                    "val_high": int(val_idx_high.size),
                    "test_low": int(test_low),
                    "test_high": int(len(test_idx) - test_low),
                },
                "stageA_reconstructor_low": reco_low_metrics,
                "stageA_reconstructor_high": reco_high_metrics,
                "stageB_low": stageB_low_metrics,
                "stageB_high": stageB_high_metrics,
                "stageC_low": stageC_low_metrics,
                "stageC_high": stageC_high_metrics,
                "test_branch_metrics": {
                    "stage2": stage2_branch_metrics,
                    "joint": joint_branch_metrics,
                },
                "boundary_metrics": boundary_metrics,
                "test": {
                    "auc_teacher": float(auc_teacher),
                    "auc_baseline": float(auc_baseline),
                    "auc_stage2": float(auc_stage2),
                    "auc_joint": float(auc_joint),
                    "fpr30_teacher": float(fpr30_teacher),
                    "fpr30_baseline": float(fpr30_baseline),
                    "fpr30_stage2": float(fpr30_stage2),
                    "fpr30_joint": float(fpr30_joint),
                    "fpr50_teacher": float(fpr50_teacher),
                    "fpr50_baseline": float(fpr50_baseline),
                    "fpr50_stage2": float(fpr50_stage2),
                    "fpr50_joint": float(fpr50_joint),
                },
            },
            f,
            indent=2,
        )

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg["hlt_effects"], "stats": hlt_stats}, f, indent=2)

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": auc_teacher}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": auc_baseline}, save_root / "baseline.pt")

        torch.save({"model": stage2_reco_low_state, "val": reco_low_metrics}, save_root / "offline_reconstructor_low_stage2.pt")
        torch.save({"model": stage2_reco_high_state, "val": reco_high_metrics}, save_root / "offline_reconstructor_high_stage2.pt")
        torch.save({"model": reconstructor_low.state_dict(), "val": reco_low_metrics}, save_root / "offline_reconstructor_low.pt")
        torch.save({"model": reconstructor_high.state_dict(), "val": reco_high_metrics}, save_root / "offline_reconstructor_high.pt")

        torch.save(
            {
                "model": stage2_dual_low_state,
                "branch": "low",
                "auc_stage2": float(auc_stage2),
            },
            save_root / "dual_joint_low_stage2.pt",
        )
        torch.save(
            {
                "model": stage2_dual_high_state,
                "branch": "high",
                "auc_stage2": float(auc_stage2),
            },
            save_root / "dual_joint_high_stage2.pt",
        )
        torch.save({"model": dual_low.state_dict(), "branch": "low", "auc": float(auc_joint)}, save_root / "dual_joint_low.pt")
        torch.save({"model": dual_high.state_dict(), "branch": "high", "auc": float(auc_joint)}, save_root / "dual_joint_high.pt")

    print(f"\nSaved hard-route joint results to: {save_root}")


if __name__ == "__main__":
    main()

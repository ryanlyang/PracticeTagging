#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Adds Stage-A-only task-aware auxiliary KD objective:
    reconstructed-view -> frozen offline teacher logits

Design:
- Auxiliary only active during Stage-A training updates.
- Reco-only view for student branch (no HLT shortcut).
- Teacher target from offline view (same jet), detached.
- Small weighted loss with warmup ramp.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


_AUX_LAMBDA = float(os.environ.get("STAGEA_TASKAUX_LAMBDA", "0.02"))
_AUX_TEMP = float(os.environ.get("STAGEA_TASKAUX_TEMP", "2.0"))
_AUX_RAMP_STEPS = int(max(int(os.environ.get("STAGEA_TASKAUX_RAMP_STEPS", "1200")), 1))
_AUX_WEIGHT_FLOOR = float(os.environ.get("STAGEA_TASKAUX_WEIGHT_FLOOR", "1e-4"))
_AUX_INCLUDE_FLAGS = bool(int(os.environ.get("STAGEA_TASKAUX_INCLUDE_FLAGS", "0")))

_IN_STAGEA_TRAIN = False
_STAGEA_STEP = 0
_STAGEA_TEACHER: Optional[torch.nn.Module] = None

_ORIG_COMPUTE_RECO_LOSSES = base.compute_reconstruction_losses_weighted
_ORIG_TRAIN_RECO = base.train_reconstructor_weighted
_ORIG_TRAIN_SINGLE = base.train_single_view_classifier_auc


def _train_single_view_classifier_auc_capture_teacher(*args, **kwargs):
    trained = _ORIG_TRAIN_SINGLE(*args, **kwargs)
    name = kwargs.get("name", None)
    if name is None and len(args) >= 6:
        name = args[5]
    if str(name).strip().lower() == "teacher":
        global _STAGEA_TEACHER
        _STAGEA_TEACHER = trained
        _STAGEA_TEACHER.eval()
        for p in _STAGEA_TEACHER.parameters():
            p.requires_grad_(False)
        print(
            "[StageA-TaskAux] Captured frozen teacher for auxiliary KD "
            f"(lambda={_AUX_LAMBDA:.6f}, temp={_AUX_TEMP:.3f}, ramp_steps={_AUX_RAMP_STEPS})"
        )
    return trained


def _compute_reco_losses_with_taskaux(
    out,
    const_hlt,
    mask_hlt,
    const_off,
    mask_off,
    budget_merge_true,
    budget_eff_true,
    loss_cfg,
    sample_weight=None,
):
    global _STAGEA_STEP
    losses = _ORIG_COMPUTE_RECO_LOSSES(
        out,
        const_hlt,
        mask_hlt,
        const_off,
        mask_off,
        budget_merge_true,
        budget_eff_true,
        loss_cfg,
        sample_weight=sample_weight,
    )

    use_aux = (
        _IN_STAGEA_TRAIN
        and torch.is_grad_enabled()
        and (_STAGEA_TEACHER is not None)
        and (_AUX_LAMBDA > 0.0)
    )
    if not use_aux:
        losses["task_aux"] = torch.zeros_like(losses["total"])
        losses["task_aux_scale"] = torch.zeros_like(losses["total"])
        return losses

    # Student input: reconstructed-only view (first 7 channels).
    feat_b, mask_b = base.build_soft_corrected_view(
        out,
        weight_floor=float(_AUX_WEIGHT_FLOOR),
        scale_features_by_weight=True,
        include_flags=bool(_AUX_INCLUDE_FLAGS),
    )
    feat_corr = feat_b[..., :7]

    # Teacher target: offline view, detached.
    feat_off = base.compute_features_torch(const_off, mask_off)
    with torch.no_grad():
        logits_t = _STAGEA_TEACHER(feat_off, mask_off).squeeze(1)

    logits_s = _STAGEA_TEACHER(feat_corr, mask_b).squeeze(1)
    temp = max(float(_AUX_TEMP), 1e-4)
    target_prob = torch.sigmoid(logits_t / temp)
    loss_aux = F.binary_cross_entropy_with_logits(logits_s / temp, target_prob) * (temp * temp)

    _STAGEA_STEP += 1
    scale = min(1.0, float(_STAGEA_STEP) / float(_AUX_RAMP_STEPS))
    losses["task_aux"] = loss_aux
    losses["task_aux_scale"] = torch.full_like(loss_aux, float(scale))
    losses["total"] = losses["total"] + float(_AUX_LAMBDA) * float(scale) * loss_aux
    return losses


def _train_reconstructor_weighted_stagea_taskaux(*args, **kwargs):
    global _IN_STAGEA_TRAIN, _STAGEA_STEP
    prev = _IN_STAGEA_TRAIN
    _IN_STAGEA_TRAIN = True
    _STAGEA_STEP = 0
    print(
        "[StageA-TaskAux] enabled "
        f"(lambda={_AUX_LAMBDA:.6f}, temp={_AUX_TEMP:.3f}, ramp_steps={_AUX_RAMP_STEPS})"
    )
    try:
        return _ORIG_TRAIN_RECO(*args, **kwargs)
    finally:
        _IN_STAGEA_TRAIN = prev


base.train_single_view_classifier_auc = _train_single_view_classifier_auc_capture_teacher
base.compute_reconstruction_losses_weighted = _compute_reco_losses_with_taskaux
base.train_reconstructor_weighted = _train_reconstructor_weighted_stagea_taskaux


if __name__ == "__main__":
    base.main()


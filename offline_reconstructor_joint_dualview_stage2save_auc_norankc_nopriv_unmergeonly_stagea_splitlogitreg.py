#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Adds Stage-A-only split-exist logit regularization:
    L_reg = lambda * mean(|split_exist_raw|)
  (masked over valid HLT tokens)
- Stage B / Stage C behavior is unchanged.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


_ORIG_TRAIN_RECO_WEIGHTED = base.train_reconstructor_weighted
_ORIG_COMPUTE_RECO_LOSSES_WEIGHTED = base.compute_reconstruction_losses_weighted

_STAGEA_SPLIT_LOGIT_L1 = float(max(float(os.environ.get("STAGEA_SPLIT_LOGIT_L1", "1e-4")), 0.0))
_STAGEA_SPLIT_LOGIT_REG_ENABLE = int(os.environ.get("STAGEA_SPLIT_LOGIT_REG_ENABLE", "1")) != 0

_REG_ACTIVE = False
_LAST_SPLIT_EXIST_RAW: Optional[torch.Tensor] = None


def _compute_split_logit_reg(
    split_exist_raw: torch.Tensor,
    mask_hlt: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    eps = 1e-8
    m = mask_hlt.float().unsqueeze(-1)
    abs_raw = split_exist_raw.abs() * m
    per_jet = abs_raw.sum(dim=(1, 2)) / (m.sum(dim=(1, 2)) + eps)
    return base._weighted_batch_mean(per_jet, sample_weight)


def compute_reconstruction_losses_weighted_with_stagea_splitlogitreg(
    out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    loss_cfg: Dict,
    sample_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    losses = _ORIG_COMPUTE_RECO_LOSSES_WEIGHTED(
        out=out,
        const_hlt=const_hlt,
        mask_hlt=mask_hlt,
        const_off=const_off,
        mask_off=mask_off,
        budget_merge_true=budget_merge_true,
        budget_eff_true=budget_eff_true,
        loss_cfg=loss_cfg,
        sample_weight=sample_weight,
    )

    add_reg = bool(_REG_ACTIVE and _STAGEA_SPLIT_LOGIT_REG_ENABLE and _STAGEA_SPLIT_LOGIT_L1 > 0.0)
    if add_reg and _LAST_SPLIT_EXIST_RAW is not None:
        reg = _compute_split_logit_reg(_LAST_SPLIT_EXIST_RAW, mask_hlt, sample_weight)
    else:
        reg = losses["total"].new_zeros(())

    losses["split_logit_l1"] = reg
    losses["total"] = losses["total"] + (float(_STAGEA_SPLIT_LOGIT_L1) * reg)
    return losses


def train_reconstructor_weighted_with_stagea_splitlogitreg(
    model: base.OfflineReconstructor,
    train_loader,
    val_loader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
    apply_reco_weight: bool,
    reload_best_at_stage_transition: bool,
):
    global _REG_ACTIVE, _LAST_SPLIT_EXIST_RAW

    _REG_ACTIVE = bool(_STAGEA_SPLIT_LOGIT_REG_ENABLE and _STAGEA_SPLIT_LOGIT_L1 > 0.0)
    _LAST_SPLIT_EXIST_RAW = None

    handles = []
    split_head = getattr(model, "split_exist_head", None)
    if isinstance(split_head, nn.Module):
        def _split_exist_hook(_m, _inp, out):
            global _LAST_SPLIT_EXIST_RAW
            if torch.is_tensor(out):
                _LAST_SPLIT_EXIST_RAW = out

        handles.append(split_head.register_forward_hook(_split_exist_hook))

    try:
        return _ORIG_TRAIN_RECO_WEIGHTED(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            train_cfg=train_cfg,
            loss_cfg=loss_cfg,
            apply_reco_weight=apply_reco_weight,
            reload_best_at_stage_transition=reload_best_at_stage_transition,
        )
    finally:
        _REG_ACTIVE = False
        _LAST_SPLIT_EXIST_RAW = None
        for h in handles:
            h.remove()


base.compute_reconstruction_losses_weighted = compute_reconstruction_losses_weighted_with_stagea_splitlogitreg
base.train_reconstructor_weighted = train_reconstructor_weighted_with_stagea_splitlogitreg


if __name__ == "__main__":
    print(
        "[StageA-SplitLogitReg] "
        f"enabled={int(_STAGEA_SPLIT_LOGIT_REG_ENABLE)} "
        f"lambda={_STAGEA_SPLIT_LOGIT_L1:.6g}"
    )
    base.main()


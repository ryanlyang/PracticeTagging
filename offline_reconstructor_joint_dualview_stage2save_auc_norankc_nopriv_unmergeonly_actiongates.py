#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses the full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps in reconstructor with independent sigmoid action gates.
- Adds optional gate sparsity regularization in reconstruction loss.
"""

from __future__ import annotations

import os
from typing import Dict

import torch

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2_actiongates import OfflineReconstructorActionGates

base.OfflineReconstructor = OfflineReconstructorActionGates

_orig_compute_reco_losses = base.compute_reconstruction_losses_weighted


def _compute_reco_losses_with_action_gate_reg(
    out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    loss_cfg: Dict,
    sample_weight: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    losses = _orig_compute_reco_losses(
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
    if "action_gates" not in out:
        return losses

    w_gate_l1 = float(loss_cfg.get("w_action_gate_l1", 0.0))
    if w_gate_l1 <= 0.0:
        return losses

    eps = 1e-8
    gates = out["action_gates"].clamp(0.0, 1.0)  # [B, L, 4]
    non_keep = gates[..., 1:]  # unsmear/split/reassign
    m = mask_hlt.float()  # [B, L]
    denom = (m.sum(dim=1).clamp(min=1.0) * float(non_keep.shape[-1]))
    gate_l1_vec = (non_keep * m.unsqueeze(-1)).sum(dim=(1, 2)) / (denom + eps)
    gate_l1 = base._weighted_batch_mean(gate_l1_vec, sample_weight)

    out_losses = dict(losses)
    out_losses["gate_l1"] = gate_l1
    out_losses["total"] = out_losses["total"] + w_gate_l1 * gate_l1
    return out_losses


base.compute_reconstruction_losses_weighted = _compute_reco_losses_with_action_gate_reg


def _configure_gate_regularization_from_env() -> float:
    raw = os.getenv("ACTION_GATE_L1", "0.01").strip()
    try:
        val = float(raw)
    except Exception:
        val = 0.01
    val = max(val, 0.0)
    base.BASE_CONFIG["loss"]["w_action_gate_l1"] = val
    return val


if __name__ == "__main__":
    gate_l1 = _configure_gate_regularization_from_env()
    print(f"[ActionGates] Using w_action_gate_l1={gate_l1:.6f} (from ACTION_GATE_L1 env).")
    base.main()


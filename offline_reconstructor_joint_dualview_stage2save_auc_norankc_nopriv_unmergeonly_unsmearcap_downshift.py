#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper runner: identical to
offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py,
with:
- unsmear hard-cap (via OfflineReconstructorUnsmearCap), and
- asymmetric log-pt shift penalty to suppress pT-collapse behavior.

Penalty:
  L_down = mean(ReLU((-d_logpt) - tau_down)^2)
  L_up   = mean(ReLU(( d_logpt) - tau_up)^2)
  total += lambda_down * L_down + lambda_up * L_up

Choose lambda_down > lambda_up to penalize negative shifts more strongly.

Controls (environment variables):
- UNSMEAR_LOGPT_CAP (default: 0.25)
- UNSMEAR_LOGE_CAP  (default: same as logpt cap)
- UNSMEAR_DOWNSHIFT_LAMBDA (default: 0.08)
- UNSMEAR_DOWNSHIFT_TAU    (default: 0.15)
- UNSMEAR_UPSHIFT_LAMBDA   (default: 0.02)
- UNSMEAR_UPSHIFT_TAU      (default: 0.15)
"""

from __future__ import annotations

import os
from typing import Dict

import torch

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base_run
from offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_unsmearcap import (
    OfflineReconstructorUnsmearCap,
)


_ORIG_COMPUTE = base_run.compute_reconstruction_losses_weighted


def _compute_reconstruction_losses_weighted_with_downshift_pen(
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
    losses = _ORIG_COMPUTE(
        out,
        const_hlt,
        mask_hlt,
        const_off,
        mask_off,
        budget_merge_true,
        budget_eff_true,
        loss_cfg,
        sample_weight,
    )

    lam_down = float(os.environ.get("UNSMEAR_DOWNSHIFT_LAMBDA", "0.08"))
    tau_down = float(os.environ.get("UNSMEAR_DOWNSHIFT_TAU", "0.15"))
    lam_up = float(os.environ.get("UNSMEAR_UPSHIFT_LAMBDA", "0.02"))
    tau_up = float(os.environ.get("UNSMEAR_UPSHIFT_TAU", "0.15"))

    tau_down = max(tau_down, 1e-6)
    tau_up = max(tau_up, 1e-6)

    d_logpt = out.get("d_logpt", None)
    if d_logpt is None:
        zero = losses["total"] * 0.0
        losses["unsmear_downshift"] = zero
        losses["unsmear_upshift"] = zero
        return losses

    down_excess = torch.relu((-d_logpt) - tau_down)
    up_excess = torch.relu(d_logpt - tau_up)

    down_vec = (down_excess * down_excess).mean(dim=1)
    up_vec = (up_excess * up_excess).mean(dim=1)

    down_pen = base_run._weighted_batch_mean(down_vec, sample_weight)
    up_pen = base_run._weighted_batch_mean(up_vec, sample_weight)

    total_add = 0.0
    if lam_down > 0.0:
        total_add = total_add + lam_down * down_pen
    if lam_up > 0.0:
        total_add = total_add + lam_up * up_pen

    losses["total"] = losses["total"] + total_add
    losses["unsmear_downshift"] = down_pen
    losses["unsmear_upshift"] = up_pen
    return losses


def main() -> None:
    base_run.OfflineReconstructor = OfflineReconstructorUnsmearCap
    base_run.compute_reconstruction_losses_weighted = _compute_reconstruction_losses_weighted_with_downshift_pen
    base_run.main()


if __name__ == "__main__":
    main()

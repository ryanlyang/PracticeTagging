#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper runner: identical to
offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py,
with:
- unsmear hard-cap (via OfflineReconstructorUnsmearCap), and
- trust-region penalty on unsmear log-pt shift:
    L_unsmear_clip = mean(ReLU(|d_logpt|-tau)^2)

Controls (environment variables):
- UNSMEAR_LOGPT_CAP (default: 0.25)
- UNSMEAR_LOGE_CAP  (default: same as logpt cap)
- UNSMEAR_TRUST_LAMBDA (default: 0.05)
- UNSMEAR_TRUST_TAU    (default: 0.18)
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


def _compute_reconstruction_losses_weighted_with_unsmear_trust(
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

    lam = float(os.environ.get("UNSMEAR_TRUST_LAMBDA", "0.05"))
    tau = float(os.environ.get("UNSMEAR_TRUST_TAU", "0.18"))
    tau = max(tau, 1e-6)

    if lam <= 0.0:
        losses["unsmear_clip"] = losses["total"] * 0.0
        return losses

    d_logpt = out.get("d_logpt", None)
    if d_logpt is None:
        trust_pen = losses["total"] * 0.0
    else:
        excess = torch.relu(torch.abs(d_logpt) - tau)
        trust_vec = (excess * excess).mean(dim=1)
        trust_pen = base_run._weighted_batch_mean(trust_vec, sample_weight)

    losses["total"] = losses["total"] + lam * trust_pen
    losses["unsmear_clip"] = trust_pen
    return losses


def main() -> None:
    base_run.OfflineReconstructor = OfflineReconstructorUnsmearCap
    base_run.compute_reconstruction_losses_weighted = _compute_reconstruction_losses_weighted_with_unsmear_trust
    base_run.main()


if __name__ == "__main__":
    main()

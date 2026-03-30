#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses the full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Adds a Stage-A-only action entropy regularizer to sharpen action mixing.

Design:
- Keep base action softmax (no hard routing).
- Add tiny entropy penalty only during Stage-A *training* steps.
- Do not alter Stage-A validation objective.
- Do not alter Stage B/C behavior.
"""

from __future__ import annotations

import math
import os

import torch

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


_STAGEA_ACTION_ENTROPY_LAMBDA = float(os.environ.get("STAGEA_ACTION_ENTROPY_LAMBDA", "0.01"))
_STAGEA_ACTION_SOFTMAX_TEMP = float(os.environ.get("STAGEA_ACTION_SOFTMAX_TEMP", "1.0"))
_IN_STAGEA_TRAIN = False

_ORIG_COMPUTE_RECO_LOSSES = base.compute_reconstruction_losses_weighted
_ORIG_TRAIN_RECO = base.train_reconstructor_weighted


def _compute_reco_losses_with_stagea_action_entropy(
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

    # Apply only in Stage-A training path:
    # - toggled by _IN_STAGEA_TRAIN from wrapped train_reconstructor_weighted
    # - gated by torch.is_grad_enabled() so val/no_grad is unaffected
    if _IN_STAGEA_TRAIN and torch.is_grad_enabled() and (_STAGEA_ACTION_ENTROPY_LAMBDA > 0.0):
        action_prob = out.get("action_prob", None)
        if action_prob is not None:
            eps = 1e-8
            p = action_prob.clamp(min=eps, max=1.0)

            # Optional mild temperature sharpening in the entropy computation space.
            temp = max(float(_STAGEA_ACTION_SOFTMAX_TEMP), 1e-4)
            if abs(temp - 1.0) > 1e-8:
                p = torch.softmax(torch.log(p) / temp, dim=-1)

            entropy = -(p * torch.log(p.clamp(min=eps))).sum(dim=-1)  # [B, L]
            m = mask_hlt.float()
            entropy_mean = (entropy * m).sum() / (m.sum() + eps)
            entropy_norm = entropy_mean / max(math.log(float(p.shape[-1])), eps)  # normalize to ~[0,1]

            losses["action_entropy"] = entropy_norm
            losses["total"] = losses["total"] + float(_STAGEA_ACTION_ENTROPY_LAMBDA) * entropy_norm
            return losses

    losses["action_entropy"] = torch.zeros_like(losses["total"])
    return losses


def _train_reconstructor_weighted_stagea_entropy(*args, **kwargs):
    global _IN_STAGEA_TRAIN
    prev = _IN_STAGEA_TRAIN
    _IN_STAGEA_TRAIN = True
    print(
        "[StageA-ActionEntropy] enabled "
        f"(lambda={_STAGEA_ACTION_ENTROPY_LAMBDA:.6f}, temp={_STAGEA_ACTION_SOFTMAX_TEMP:.4f})"
    )
    try:
        return _ORIG_TRAIN_RECO(*args, **kwargs)
    finally:
        _IN_STAGEA_TRAIN = prev


base.compute_reconstruction_losses_weighted = _compute_reco_losses_with_stagea_action_entropy
base.train_reconstructor_weighted = _train_reconstructor_weighted_stagea_entropy


if __name__ == "__main__":
    base.main()


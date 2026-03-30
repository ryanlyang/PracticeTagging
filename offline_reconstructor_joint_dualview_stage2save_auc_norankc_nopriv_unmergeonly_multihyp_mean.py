#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Adds probabilistic multi-hypothesis corrected-view scoring:
  * reconstructor still outputs one soft candidate set
  * derive H hypothesis corrected views via candidate-weight temperature transforms
  * score each hypothesis independently with a shared dual model
  * aggregate logits by mean (default) or log-sum-exp mean

This is an ablation for ambiguity handling without changing reconstructor losses.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


def _parse_temps(s: str) -> Tuple[float, ...]:
    vals = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(max(float(tok), 1e-3))
    if not vals:
        vals = [1.0]
    return tuple(vals)


_HYP_TEMPS = _parse_temps(os.environ.get("MULTIHYP_TEMPS", "0.85,1.00"))
_HYP_AGG = str(os.environ.get("MULTIHYP_AGG", "mean")).strip().lower()
if _HYP_AGG not in {"mean", "lse"}:
    _HYP_AGG = "mean"

# Cache last reco_out from build_soft_corrected_view call so dual forward can build hypotheses.
_LAST_RECO_OUT: Optional[dict] = None
_LAST_BUILD_ARGS = {
    "weight_floor": 1e-4,
    "scale_features_by_weight": True,
    "include_flags": False,
}

_ORIG_BUILD_SOFT_VIEW = base.build_soft_corrected_view
_ORIG_DUAL_CLASS = base.DualViewCrossAttnClassifier


def _weights_with_temperature(w: torch.Tensor, temp: float) -> torch.Tensor:
    eps = 1e-6
    temp = max(float(temp), 1e-4)
    w0 = w.clamp(min=eps, max=1.0 - eps)
    logit = torch.log(w0) - torch.log1p(-w0)
    wt = torch.sigmoid(logit / temp)

    # Keep global mass comparable per jet to avoid trivial scale effects.
    base_sum = w.sum(dim=1, keepdim=True)
    hyp_sum = wt.sum(dim=1, keepdim=True).clamp(min=eps)
    scale = (base_sum / hyp_sum).clamp(min=0.5, max=2.0)
    wt = (wt * scale).clamp(min=0.0, max=1.0)
    return wt


def _build_soft_corrected_view_from_reco_override(
    reco_out: dict,
    weight_floor: float,
    scale_features_by_weight: bool,
    include_flags: bool,
    cand_weights_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reco_out = base.enforce_unmerge_only_output(reco_out)
    eps = 1e-8

    L = int(reco_out["action_prob"].shape[1])
    cand_weights = reco_out["cand_weights"].clamp(0.0, 1.0)
    if cand_weights_override is not None:
        cand_weights = cand_weights_override.clamp(0.0, 1.0)

    tok_tokens = reco_out["cand_tokens"][:, :L, :]
    tok_w = cand_weights[:, :L]
    mask_b = tok_w > float(weight_floor)
    none_valid = ~mask_b.any(dim=1)
    if none_valid.any():
        mask_b = mask_b.clone()
        mask_b[none_valid, 0] = True

    feat7 = base.compute_features_torch(tok_tokens, mask_b)
    if bool(scale_features_by_weight):
        feat7 = feat7 * tok_w.unsqueeze(-1)

    # Parent-level split mass per token, from child segment of candidate weights.
    child_len = int(reco_out["child_weight"].shape[1])
    K = max(int(child_len // max(L, 1)), 1)
    child_start = L
    child_end = child_start + child_len
    child_w = cand_weights[:, child_start:child_end]
    if child_w.numel() == 0:
        parent_added = torch.zeros_like(tok_w)
    else:
        parent_added = child_w.reshape(child_w.shape[0], L, K).sum(dim=2).clamp(0.0, 1.0)

    # Keep the original budget-head share signal for stability.
    valid_count = mask_b.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    eff_share = (reco_out["budget_eff"].unsqueeze(1) / valid_count).clamp(0.0, 1.0)
    eff_share = eff_share * mask_b.float()

    extra = torch.stack([tok_w, parent_added, eff_share], dim=-1)
    if include_flags:
        tok_merge_flag = reco_out["cand_merge_flags"][:, :L].clamp(0.0, 1.0)
        tok_eff_flag = reco_out["cand_eff_flags"][:, :L].clamp(0.0, 1.0)
        extra = torch.cat([extra, tok_merge_flag.unsqueeze(-1), tok_eff_flag.unsqueeze(-1)], dim=-1)

    feat_b = torch.cat([feat7, extra], dim=-1)
    feat_b = torch.nan_to_num(feat_b, nan=0.0, posinf=0.0, neginf=0.0)
    feat_b = feat_b * mask_b.unsqueeze(-1).float()
    return feat_b, mask_b


def _build_soft_corrected_view_cache(
    reco_out,
    weight_floor: float = 1e-4,
    scale_features_by_weight: bool = True,
    include_flags: bool = False,
):
    global _LAST_RECO_OUT, _LAST_BUILD_ARGS
    _LAST_RECO_OUT = reco_out
    _LAST_BUILD_ARGS = {
        "weight_floor": float(weight_floor),
        "scale_features_by_weight": bool(scale_features_by_weight),
        "include_flags": bool(include_flags),
    }
    return _ORIG_BUILD_SOFT_VIEW(
        reco_out,
        weight_floor=weight_floor,
        scale_features_by_weight=scale_features_by_weight,
        include_flags=include_flags,
    )


class DualViewCrossAttnClassifierMultiHypMean(nn.Module):
    """
    Shared dual-view scorer across multiple corrected-view hypotheses.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base_model = _ORIG_DUAL_CLASS(*args, **kwargs)
        self.hyp_temps = tuple(_HYP_TEMPS)
        self.agg_mode = str(_HYP_AGG)

    def forward(self, feat_a, mask_a, feat_b, mask_b):
        # Fallback path if no cached reco output is available.
        if _LAST_RECO_OUT is None:
            return self.base_model(feat_a, mask_a, feat_b, mask_b)

        # Guard against stale cache from a different batch size.
        if int(_LAST_RECO_OUT["cand_weights"].shape[0]) != int(feat_a.shape[0]):
            return self.base_model(feat_a, mask_a, feat_b, mask_b)

        reco_out = _LAST_RECO_OUT
        build_args = dict(_LAST_BUILD_ARGS)

        logits_all = []
        for t in self.hyp_temps:
            if abs(float(t) - 1.0) < 1e-8:
                feat_h, mask_h = feat_b, mask_b
            else:
                w_h = _weights_with_temperature(reco_out["cand_weights"], float(t))
                feat_h, mask_h = _build_soft_corrected_view_from_reco_override(
                    reco_out,
                    weight_floor=float(build_args["weight_floor"]),
                    scale_features_by_weight=bool(build_args["scale_features_by_weight"]),
                    include_flags=bool(build_args["include_flags"]),
                    cand_weights_override=w_h,
                )
            logits_h = self.base_model(feat_a, mask_a, feat_h, mask_h)  # [B,1]
            logits_all.append(logits_h)

        stack = torch.stack(logits_all, dim=0)  # [H,B,1]
        if self.agg_mode == "lse":
            out = torch.logsumexp(stack, dim=0) - math.log(float(stack.shape[0]))
        else:
            out = stack.mean(dim=0)
        return out


base.build_soft_corrected_view = _build_soft_corrected_view_cache
base.DualViewCrossAttnClassifier = DualViewCrossAttnClassifierMultiHypMean


if __name__ == "__main__":
    print(
        "[MultiHyp] enabled "
        f"(temps={','.join(f'{t:.3f}' for t in _HYP_TEMPS)}, agg={_HYP_AGG})"
    )
    base.main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses the full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Adds multi-hypothesis corrected-view handling with learned aggregation:
  * K=3 corrected-view hypotheses (default via cand-weight temperatures)
  * shared dual scorer per hypothesis
  * learned gate/attention over hypothesis logits for final logit
  * reconstruction loss aggregated via soft-min across hypotheses
    (instead of implicit single-view averaging)
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

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
    if len(vals) == 0:
        vals = [0.75, 1.00, 1.25]
    return tuple(vals)


_HYP_TEMPS = _parse_temps(os.environ.get("MULTIHYP_TEMPS", "0.75,1.00,1.25"))
_RECO_SOFTMIN_TAU = max(float(os.environ.get("MULTIHYP_RECO_SOFTMIN_TAU", "0.05")), 1e-4)
_GATE_HIDDEN = max(int(os.environ.get("MULTIHYP_GATE_HIDDEN", "32")), 4)
_GATE_DROPOUT = float(max(min(float(os.environ.get("MULTIHYP_GATE_DROPOUT", "0.05")), 0.5), 0.0))

_LAST_RECO_OUT: Optional[dict] = None
_LAST_BUILD_ARGS = {
    "weight_floor": 1e-4,
    "scale_features_by_weight": True,
    "include_flags": False,
}

_ORIG_BUILD_SOFT_VIEW = base.build_soft_corrected_view
_ORIG_DUAL_CLASS = base.DualViewCrossAttnClassifier
_ORIG_RECO_LOSS_FN = base.compute_reconstruction_losses_weighted


def _weights_with_temperature(w: torch.Tensor, temp: float) -> torch.Tensor:
    eps = 1e-6
    t = max(float(temp), 1e-4)
    w0 = w.clamp(min=eps, max=1.0 - eps)
    logit = torch.log(w0) - torch.log1p(-w0)
    wt = torch.sigmoid(logit / t)

    # Keep per-jet candidate mass near baseline to avoid trivial scale changes.
    base_sum = w.sum(dim=1, keepdim=True)
    hyp_sum = wt.sum(dim=1, keepdim=True).clamp(min=eps)
    scale = (base_sum / hyp_sum).clamp(min=0.5, max=2.0)
    wt = (wt * scale).clamp(min=0.0, max=1.0)
    return wt


def _reco_out_with_cand_weights(reco_out: Dict[str, torch.Tensor], cand_w_override: torch.Tensor) -> Dict[str, torch.Tensor]:
    out_h = dict(reco_out)
    cand_w = cand_w_override.clamp(0.0, 1.0)
    out_h["cand_weights"] = cand_w

    L = int(reco_out["action_prob"].shape[1])
    n_child = int(reco_out["child_weight"].shape[1])
    n_gen = int(reco_out["gen_weight"].shape[1])

    if n_child > 0:
        out_h["child_weight"] = cand_w[:, L : L + n_child]
    if n_gen > 0:
        out_h["gen_weight"] = cand_w[:, L + n_child : L + n_child + n_gen]
    return out_h


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


class DualViewCrossAttnClassifierMultiHypLearned(nn.Module):
    """
    Shared dual scorer on K hypotheses + learned per-sample hypothesis gate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base_model = _ORIG_DUAL_CLASS(*args, **kwargs)
        self.hyp_temps = tuple(_HYP_TEMPS)
        h = len(self.hyp_temps)
        gate_in_dim = 2 * h  # [hyp_logits, hyp_mass]
        self.hyp_gate = nn.Sequential(
            nn.Linear(gate_in_dim, _GATE_HIDDEN),
            nn.GELU(),
            nn.Dropout(_GATE_DROPOUT),
            nn.Linear(_GATE_HIDDEN, h),
        )

    def forward(self, feat_a, mask_a, feat_b, mask_b):
        # Fallback path if no cached reco output is available.
        if _LAST_RECO_OUT is None:
            return self.base_model(feat_a, mask_a, feat_b, mask_b)

        reco_out = _LAST_RECO_OUT
        if int(reco_out["cand_weights"].shape[0]) != int(feat_a.shape[0]):
            return self.base_model(feat_a, mask_a, feat_b, mask_b)

        build_args = dict(_LAST_BUILD_ARGS)

        hyp_logits = []
        hyp_mass = []
        for t in self.hyp_temps:
            if abs(float(t) - 1.0) < 1e-8:
                feat_h, mask_h = feat_b, mask_b
                w_h = reco_out["cand_weights"]
            else:
                w_h = _weights_with_temperature(reco_out["cand_weights"], float(t))
                out_h = _reco_out_with_cand_weights(reco_out, w_h)
                feat_h, mask_h = _ORIG_BUILD_SOFT_VIEW(
                    out_h,
                    weight_floor=float(build_args["weight_floor"]),
                    scale_features_by_weight=bool(build_args["scale_features_by_weight"]),
                    include_flags=bool(build_args["include_flags"]),
                )

            logits_h = self.base_model(feat_a, mask_a, feat_h, mask_h).squeeze(1)  # [B]
            mass_h = w_h.sum(dim=1) / max(float(w_h.shape[1]), 1.0)  # [B]

            hyp_logits.append(logits_h)
            hyp_mass.append(mass_h)

        logits_stack = torch.stack(hyp_logits, dim=1)  # [B,H]
        mass_stack = torch.stack(hyp_mass, dim=1)  # [B,H]
        gate_in = torch.cat([logits_stack, mass_stack], dim=1)  # [B,2H]
        gate_logits = self.hyp_gate(gate_in)  # [B,H]
        gate_alpha = torch.softmax(gate_logits, dim=1)

        out = (gate_alpha * logits_stack).sum(dim=1, keepdim=True)  # [B,1]
        return out


def _compute_reco_losses_multihyp(
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
    # Build per-hypothesis reco losses using same criterion on different
    # candidate-weight temperatures, then aggregate with soft-min.
    loss_dicts = []
    for t in _HYP_TEMPS:
        if abs(float(t) - 1.0) < 1e-8:
            out_h = out
        else:
            w_h = _weights_with_temperature(out["cand_weights"], float(t))
            out_h = _reco_out_with_cand_weights(out, w_h)

        l_h = _ORIG_RECO_LOSS_FN(
            out_h,
            const_hlt,
            mask_hlt,
            const_off,
            mask_off,
            budget_merge_true,
            budget_eff_true,
            loss_cfg,
            sample_weight=sample_weight,
        )
        loss_dicts.append(l_h)

    keys = list(loss_dicts[0].keys())
    totals = torch.stack([d["total"] for d in loss_dicts], dim=0)  # [H]
    tau = float(_RECO_SOFTMIN_TAU)
    z = -totals / tau
    alpha = torch.softmax(z, dim=0)

    out_loss: Dict[str, torch.Tensor] = {}
    out_loss["total"] = -tau * torch.logsumexp(z, dim=0)
    for k in keys:
        if k == "total":
            continue
        vals = torch.stack([d[k] for d in loss_dicts], dim=0)
        out_loss[k] = (alpha * vals).sum(dim=0)
    return out_loss


base.build_soft_corrected_view = _build_soft_corrected_view_cache
base.DualViewCrossAttnClassifier = DualViewCrossAttnClassifierMultiHypLearned
base.compute_reconstruction_losses_weighted = _compute_reco_losses_multihyp


if __name__ == "__main__":
    print(
        "[MultiHypLearned] enabled "
        f"(temps={','.join(f'{t:.3f}' for t in _HYP_TEMPS)}, "
        f"reco_softmin_tau={_RECO_SOFTMIN_TAU:.4f}, "
        f"gate_hidden={_GATE_HIDDEN}, gate_dropout={_GATE_DROPOUT:.3f})"
    )
    base.main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Goal-conditioned wrapper for the m2 hybrid-ops JetClass reconstructor."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_hybrid_ops as hybrid_ops

IDX_PT = 0
IDX_ETA = 1
IDX_PHI = 2
IDX_E = 3


def _wrap_phi_t(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _weighted_p4_t(tokens4: torch.Tensor, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    eps = 1e-8
    pt = tokens4[:, :, IDX_PT].clamp(min=0.0)
    eta = tokens4[:, :, IDX_ETA].clamp(min=-5.0, max=5.0)
    phi = tokens4[:, :, IDX_PHI]
    ene = tokens4[:, :, IDX_E].clamp(min=0.0)
    w = weights.float().clamp(min=0.0)
    px = (pt * torch.cos(phi) * w).sum(dim=1)
    py = (pt * torch.sin(phi) * w).sum(dim=1)
    pz = (pt * torch.sinh(eta) * w).sum(dim=1)
    e = (ene * w).sum(dim=1)
    jet_pt = torch.sqrt(px.pow(2) + py.pow(2) + eps)
    jet_eta = torch.asinh(pz / jet_pt.clamp(min=eps))
    jet_phi = torch.atan2(py, px)
    return {"pt": jet_pt, "eta": jet_eta, "phi": jet_phi, "px": px, "py": py, "pz": pz, "e": e}


class OfflineReconstructorGoalConditionedHybridOps(hybrid_ops.OfflineReconstructorHybridOps):
    """HLT-only global pT/axis correction head on top of hybrid operations."""

    def __init__(
        self,
        *args,
        goal_apply_scale: float = 0.45,
        goal_max_dlogpt: float = 1.25,
        goal_max_deta: float = 0.55,
        goal_max_dphi: float = 0.55,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        hidden = int(getattr(self, "embed_dim", 128))
        self.goal_apply_scale = float(goal_apply_scale)
        self.goal_max_dlogpt = float(goal_max_dlogpt)
        self.goal_max_deta = float(goal_max_deta)
        self.goal_max_dphi = float(goal_max_dphi)
        self.goal_head = nn.Sequential(
            nn.Linear(12, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def _goal_features(self, const_hlt: torch.Tensor, mask_hlt: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        p4 = _weighted_p4_t(const_hlt, mask_hlt.float())
        pt = const_hlt[:, :, IDX_PT].clamp(min=0.0)
        eta = const_hlt[:, :, IDX_ETA]
        phi = const_hlt[:, :, IDX_PHI]
        w = mask_hlt.float()
        n = w.sum(dim=1).clamp(min=1.0)
        scalar_pt = (pt * w).sum(dim=1)
        leading_pt = (pt * w).max(dim=1).values
        mean_eta = (eta * w).sum(dim=1) / n
        mean_sin_phi = (torch.sin(phi) * w).sum(dim=1) / n
        mean_cos_phi = (torch.cos(phi) * w).sum(dim=1) / n
        return torch.stack(
            [
                torch.log(p4["pt"].clamp(min=eps)),
                p4["eta"],
                torch.sin(p4["phi"]),
                torch.cos(p4["phi"]),
                torch.log(p4["e"].clamp(min=eps) + 1.0),
                torch.log(n),
                torch.log(scalar_pt.clamp(min=eps) + 1.0),
                leading_pt / scalar_pt.clamp(min=eps),
                scalar_pt / p4["pt"].clamp(min=eps),
                mean_eta,
                mean_sin_phi,
                mean_cos_phi,
            ],
            dim=1,
        )

    def _predict_goal(self, const_hlt: torch.Tensor, mask_hlt: torch.Tensor) -> torch.Tensor:
        raw = self.goal_head(self._goal_features(const_hlt, mask_hlt))
        return torch.stack(
            [
                self.goal_max_dlogpt * torch.tanh(raw[:, 0]),
                self.goal_max_deta * torch.tanh(raw[:, 1]),
                self.goal_max_dphi * torch.tanh(raw[:, 2]),
            ],
            dim=1,
        )

    def _apply_goal_to_tokens(self, tokens: torch.Tensor, goal_pred: torch.Tensor) -> torch.Tensor:
        if tokens.numel() == 0 or self.goal_apply_scale <= 0.0:
            return tokens
        dlogpt = (self.goal_apply_scale * goal_pred[:, 0]).view(-1, 1)
        deta = (self.goal_apply_scale * goal_pred[:, 1]).view(-1, 1)
        dphi = (self.goal_apply_scale * goal_pred[:, 2]).view(-1, 1)
        pt = (tokens[:, :, 0].clamp(min=1e-8) * torch.exp(dlogpt)).clamp(min=1e-8)
        eta = (tokens[:, :, 1] + deta).clamp(min=-5.0, max=5.0)
        phi = _wrap_phi_t(tokens[:, :, 2] + dphi)
        ene = (tokens[:, :, 3].clamp(min=1e-8) * torch.exp(dlogpt)).clamp(min=1e-8)
        ene = torch.maximum(ene, pt * torch.cosh(eta))
        return torch.stack([pt, eta, phi, ene], dim=-1)

    def forward(self, feat_hlt, mask_hlt, const_hlt, stage_scale: float = 1.0):
        goal_pred = self._predict_goal(const_hlt, mask_hlt)
        out = super().forward(feat_hlt, mask_hlt, const_hlt, stage_scale=stage_scale)
        out["goal_pred"] = goal_pred
        for key in ("cand_tokens", "tok_tokens", "gen_tokens"):
            if key in out:
                out[key] = self._apply_goal_to_tokens(out[key], goal_pred)
        return out


# Reuse the corrected-view and loss helpers from the base hybrid-ops module.
build_soft_corrected_view_hybrid_ops = hybrid_ops.build_soft_corrected_view_hybrid_ops
compute_reconstruction_losses_weighted_hybrid_ops = hybrid_ops.compute_reconstruction_losses_weighted_hybrid_ops

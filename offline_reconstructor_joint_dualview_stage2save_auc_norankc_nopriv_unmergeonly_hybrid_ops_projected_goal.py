#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projected-goal wrapper for the m2 hybrid-ops JetClass reconstructor."""

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
    eta = tokens4[:, :, IDX_ETA].clamp(min=-8.0, max=8.0)
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


class OfflineReconstructorProjectedGoalHybridOps(hybrid_ops.OfflineReconstructorHybridOps):
    """HLT-only global pT/axis target head plus exact candidate-cloud projection on top of hybrid operations."""

    def __init__(
        self,
        *args,
        goal_apply_scale: float = 1.0,
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

    def _target_from_goal(
        self,
        const_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        goal_pred: torch.Tensor,
        stage_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        hlt_p4 = _weighted_p4_t(const_hlt, mask_hlt.float())
        scale = float(self.goal_apply_scale) * max(0.0, min(float(stage_scale), 1.0))
        target_pt = hlt_p4["pt"].clamp(min=1e-8) * torch.exp(scale * goal_pred[:, 0])
        target_eta = hlt_p4["eta"] + scale * goal_pred[:, 1]
        target_phi = _wrap_phi_t(hlt_p4["phi"] + scale * goal_pred[:, 2])
        target_pz = target_pt * torch.sinh(target_eta)
        return {
            "pt": target_pt.clamp(min=1e-8),
            "eta": target_eta,
            "phi": target_phi,
            "px": target_pt * torch.cos(target_phi),
            "py": target_pt * torch.sin(target_phi),
            "pz": target_pz,
        }

    def _projection_params(
        self,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        raw = _weighted_p4_t(tokens, weights)
        pt = tokens[:, :, IDX_PT].clamp(min=0.0)
        w = weights.float().clamp(min=0.0)
        scalar_pt = (pt * w).sum(dim=1).clamp(min=eps)
        return {
            "pt_scale": (target["pt"] / raw["pt"].clamp(min=eps)).clamp(min=1.0e-4, max=1.0e4),
            "dphi": _wrap_phi_t(target["phi"] - raw["phi"]),
            "delta_pz": target["pz"] - raw["pz"],
            "scalar_pt": scalar_pt,
            "raw_pt": raw["pt"],
            "raw_eta": raw["eta"],
            "raw_phi": raw["phi"],
        }

    def _project_tokens(self, tokens: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if tokens.numel() == 0:
            return tokens

        eps = 1e-8
        pt = tokens[:, :, IDX_PT].clamp(min=eps)
        eta = tokens[:, :, IDX_ETA]
        phi = tokens[:, :, IDX_PHI]
        ene = tokens[:, :, IDX_E].clamp(min=eps)

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        mass2 = (ene.pow(2) - px.pow(2) - py.pow(2) - pz.pow(2)).clamp(min=0.0)

        s = params["pt_scale"].view(-1, 1)
        c = torch.cos(params["dphi"]).view(-1, 1)
        q = torch.sin(params["dphi"]).view(-1, 1)
        px_rot = c * px - q * py
        py_rot = q * px + c * py
        px_new = s * px_rot
        py_new = s * py_rot

        # Allocate the required longitudinal momentum residual by token pT.
        # For the full candidate cloud this makes sum(w*pz_new) exactly match
        # the predicted target pz, avoiding unstable raw_pz division near zero.
        pz_share = pt / params["scalar_pt"].view(-1, 1)
        pz_new = pz + pz_share * params["delta_pz"].view(-1, 1)

        pt_new = torch.sqrt(px_new.pow(2) + py_new.pow(2) + eps)
        eta_new = torch.asinh(pz_new / pt_new.clamp(min=eps))
        phi_new = torch.atan2(py_new, px_new)
        mass2_new = mass2 * s.pow(2)
        ene_new = torch.sqrt(px_new.pow(2) + py_new.pow(2) + pz_new.pow(2) + mass2_new + eps)
        return torch.stack(
            [
                torch.nan_to_num(pt_new, nan=eps, posinf=1e4, neginf=eps).clamp(min=eps),
                torch.nan_to_num(eta_new, nan=0.0, posinf=8.0, neginf=-8.0).clamp(min=-8.0, max=8.0),
                _wrap_phi_t(torch.nan_to_num(phi_new, nan=0.0, posinf=0.0, neginf=0.0)),
                torch.nan_to_num(ene_new, nan=eps, posinf=1e6, neginf=eps).clamp(min=eps),
            ],
            dim=-1,
        )

    def forward(self, feat_hlt, mask_hlt, const_hlt, stage_scale: float = 1.0):
        goal_pred = self._predict_goal(const_hlt, mask_hlt)
        out = super().forward(feat_hlt, mask_hlt, const_hlt, stage_scale=stage_scale)
        out["goal_pred"] = goal_pred
        target = self._target_from_goal(const_hlt, mask_hlt, goal_pred, stage_scale=stage_scale)
        params = self._projection_params(out["cand_tokens"], out["cand_weights"], target)
        for key in ("cand_tokens", "tok_tokens", "gen_tokens"):
            if key in out:
                out[key] = self._project_tokens(out[key], params)
        projected = _weighted_p4_t(out["cand_tokens"], out["cand_weights"])
        out["goal_target_pt"] = target["pt"]
        out["goal_target_eta"] = target["eta"]
        out["goal_target_phi"] = target["phi"]
        out["projection_raw_pt"] = params["raw_pt"]
        out["projection_raw_eta"] = params["raw_eta"]
        out["projection_raw_phi"] = params["raw_phi"]
        out["projection_pt_relerr"] = (projected["pt"] / target["pt"].clamp(min=1e-8) - 1.0).abs()
        out["projection_axis_dr"] = torch.sqrt(
            (projected["eta"] - target["eta"]).pow(2)
            + _wrap_phi_t(projected["phi"] - target["phi"]).pow(2)
            + 1e-12
        )
        return out


# Reuse the corrected-view and loss helpers from the base hybrid-ops module.
build_soft_corrected_view_hybrid_ops = hybrid_ops.build_soft_corrected_view_hybrid_ops
compute_reconstruction_losses_weighted_hybrid_ops = hybrid_ops.compute_reconstruction_losses_weighted_hybrid_ops

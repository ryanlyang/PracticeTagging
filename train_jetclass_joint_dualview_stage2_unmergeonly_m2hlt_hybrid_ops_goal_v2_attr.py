#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JetClass V2-Attr + m2-style HLT + goal-conditioned hybrid-ops reconstructor.

Purpose:
- Keep the V2 constrained attribute-head training/eval pipeline.
- Swap the base reconstructor to hybrid ops.
- Keep m2-style HLT corruption used by the current jetlatent runs.
- Predict a jet-level correction target from HLT only:
    log(pT_offline / pT_HLT), eta_offline - eta_HLT, phi_offline - phi_HLT.
- Use that prediction to globally condition/correct the hybrid operation output
  while keeping edit/split/generate branches available.
- Add final reconstructed jet response and axis losses in Stage-A.

This wrapper avoids modifying existing PracticeTagging scripts.
"""

from __future__ import annotations

import importlib
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


RAW_DIM = 14
IDX_PT = 0
IDX_ETA = 1
IDX_PHI = 2
IDX_E = 3
IDX_CHARGE = 4
IDX_PID0 = 5
IDX_PID4 = 9
IDX_D0 = 10
IDX_D0ERR = 11
IDX_DZ = 12
IDX_DZERR = 13


def _find_practice_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "PracticeTagging",
        Path("/home/ryreu/atlas/PracticeTagging"),
        Path("/home/ryan/ComputerScience/ATLAS/HLT_Reco/PracticeTagging"),
    ]
    required = [
        "train_jetclass_joint_dualview_stage2_unmergeonly_v2_attr.py",
        "train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt.py",
        "offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_hybrid_ops.py",
        "offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py",
    ]
    for c in candidates:
        if c.is_dir() and all((c / r).exists() for r in required):
            return c
    raise FileNotFoundError(
        "Could not locate PracticeTagging with required JetClass V2/jetlatent modules."
    )


def _identity_wrap(model):
    return model


def _identity_enforce(out: Dict):
    return out


def main() -> None:
    practice_root = _find_practice_root()
    sys.path.insert(0, str(practice_root))

    v2 = importlib.import_module("train_jetclass_joint_dualview_stage2_unmergeonly_v2_attr")
    m2hlt = importlib.import_module("train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt")
    jetlatent = importlib.import_module(
        "offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_hybrid_ops"
    )
    reco_joint = importlib.import_module(
        "offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly"
    )

    TYPE_CH = int(getattr(v2, "TYPE_CH", 0))
    TYPE_NH = int(getattr(v2, "TYPE_NH", 1))
    TYPE_GAM = int(getattr(v2, "TYPE_GAM", 2))
    TYPE_ELE = int(getattr(v2, "TYPE_ELE", 3))
    TYPE_MU = int(getattr(v2, "TYPE_MU", 4))
    TYPE_UNK = int(getattr(v2, "TYPE_UNK", 5))

    MERGE_MODE_NONE = int(getattr(v2, "MERGE_MODE_NONE", 0))
    MERGE_MODE_SAME_TYPE = int(getattr(v2, "MERGE_MODE_SAME_TYPE", 1))
    MERGE_MODE_ELE_GAM = int(getattr(v2, "MERGE_MODE_ELE_GAM", 2))
    MERGE_MODE_CH_NH = int(getattr(v2, "MERGE_MODE_CH_NH", 3))
    _stagea_aux_queue: List[Dict[str, np.ndarray]] = []

    # Parse wrapper-only knobs first; the imported V2 parser rejects unknown args.
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--target_drop_prob_max", type=float, default=0.0)
    wrapper_parser.add_argument("--target_drop_warmup_epochs", type=int, default=20)
    wrapper_parser.add_argument("--target_drop_mode", type=str, default="deterministic_bank")
    wrapper_parser.add_argument("--target_drop_num_banks", type=int, default=3)
    wrapper_parser.add_argument("--target_drop_bank_cycle_epochs", type=int, default=1)
    wrapper_parser.add_argument("--goal_apply_scale", type=float, default=0.45)
    wrapper_parser.add_argument("--goal_lambda_head", type=float, default=0.15)
    wrapper_parser.add_argument("--goal_lambda_response", type=float, default=0.30)
    wrapper_parser.add_argument("--goal_lambda_axis", type=float, default=0.12)
    wrapper_parser.add_argument("--goal_max_dlogpt", type=float, default=1.25)
    wrapper_parser.add_argument("--goal_max_deta", type=float, default=0.55)
    wrapper_parser.add_argument("--goal_max_dphi", type=float, default=0.55)
    wrapper_args, remaining_argv = wrapper_parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_argv]

    # Parse once so patched Stage-A trainer can use the exact run-time knobs.
    args = v2.parse_args()
    args.target_drop_prob_max = float(wrapper_args.target_drop_prob_max)
    args.target_drop_warmup_epochs = int(wrapper_args.target_drop_warmup_epochs)
    args.target_drop_mode = str(wrapper_args.target_drop_mode)
    args.target_drop_num_banks = int(wrapper_args.target_drop_num_banks)
    args.target_drop_bank_cycle_epochs = int(wrapper_args.target_drop_bank_cycle_epochs)
    args.goal_apply_scale = float(wrapper_args.goal_apply_scale)
    args.goal_lambda_head = float(wrapper_args.goal_lambda_head)
    args.goal_lambda_response = float(wrapper_args.goal_lambda_response)
    args.goal_lambda_axis = float(wrapper_args.goal_lambda_axis)
    args.goal_max_dlogpt = float(wrapper_args.goal_max_dlogpt)
    args.goal_max_deta = float(wrapper_args.goal_max_deta)
    args.goal_max_dphi = float(wrapper_args.goal_max_dphi)

    stagea_attr_lam_mode = float(args.lambda_attr_mode)
    stagea_attr_lam_type = float(args.lambda_attr_type)
    stagea_attr_lam_charge = float(args.lambda_attr_charge)
    stagea_attr_lam_track = float(args.lambda_attr_track)
    stagea_mode_none_weight = float(args.v2_mode_none_weight)
    stagea_mode_label_smoothing = float(args.v2_mode_label_smoothing)
    stagea_track_weight = float(args.v2_track_weight)

    # Non-split anchor (unsmear/reassign branch): keep non-kin attrs tied to HLT parent.
    stagea_anchor_type = float(args.lambda_attr_type)
    stagea_anchor_charge = float(args.lambda_attr_charge)
    stagea_anchor_track = float(args.lambda_attr_track)
    target_drop_prob_max = float(max(0.0, min(1.0, args.target_drop_prob_max)))
    target_drop_warmup_epochs = int(max(1, args.target_drop_warmup_epochs))
    target_drop_mode = str(args.target_drop_mode)
    target_drop_num_banks = int(max(1, args.target_drop_num_banks))
    target_drop_bank_cycle_epochs = int(max(1, args.target_drop_bank_cycle_epochs))
    added_target_scale = float(max(0.0, min(1.0, args.added_target_scale)))
    goal_apply_scale = float(max(0.0, args.goal_apply_scale))
    goal_lambda_head = float(max(0.0, args.goal_lambda_head))
    goal_lambda_response = float(max(0.0, args.goal_lambda_response))
    goal_lambda_axis = float(max(0.0, args.goal_lambda_axis))
    goal_max_dlogpt = float(max(0.05, args.goal_max_dlogpt))
    goal_max_deta = float(max(0.01, args.goal_max_deta))
    goal_max_dphi = float(max(0.01, args.goal_max_dphi))

    def _infer_type_id(token: np.ndarray) -> int:
        pid = token[IDX_PID0:IDX_PID4 + 1]
        if np.max(pid) <= 0:
            return TYPE_UNK
        return int(np.argmax(pid))

    def _infer_merge_mode(ti: int, tj: int) -> int:
        if ti == tj and ti != TYPE_UNK:
            return MERGE_MODE_SAME_TYPE
        pair = {int(ti), int(tj)}
        if pair == {TYPE_ELE, TYPE_GAM}:
            return MERGE_MODE_ELE_GAM
        if pair == {TYPE_CH, TYPE_NH}:
            return MERGE_MODE_CH_NH
        return MERGE_MODE_NONE

    def _wrap_phi_np(x: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(x), np.cos(x))

    def _wrap_phi_t(x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

    def _jet_p4_np(tokens4: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        pt = np.maximum(tokens4[:, :, IDX_PT].astype(np.float64), 0.0)
        eta = tokens4[:, :, IDX_ETA].astype(np.float64)
        phi = tokens4[:, :, IDX_PHI].astype(np.float64)
        ene = np.maximum(tokens4[:, :, IDX_E].astype(np.float64), 0.0)
        w = mask.astype(np.float64)
        px = (pt * np.cos(phi) * w).sum(axis=1)
        py = (pt * np.sin(phi) * w).sum(axis=1)
        pz = (pt * np.sinh(eta) * w).sum(axis=1)
        e = (ene * w).sum(axis=1)
        jet_pt = np.sqrt(px * px + py * py)
        jet_eta = np.arcsinh(pz / np.clip(jet_pt, 1e-8, np.inf))
        jet_phi = np.arctan2(py, px)
        return {
            "pt": jet_pt.astype(np.float32),
            "eta": jet_eta.astype(np.float32),
            "phi": jet_phi.astype(np.float32),
            "px": px.astype(np.float32),
            "py": py.astype(np.float32),
            "pz": pz.astype(np.float32),
            "e": e.astype(np.float32),
        }

    def _goal_target_np(off_tokens4: np.ndarray, off_mask: np.ndarray, hlt_tokens4: np.ndarray, hlt_mask: np.ndarray) -> np.ndarray:
        off = _jet_p4_np(off_tokens4, off_mask)
        hlt = _jet_p4_np(hlt_tokens4, hlt_mask)
        dlogpt = np.log(np.clip(off["pt"], 1e-8, np.inf)) - np.log(np.clip(hlt["pt"], 1e-8, np.inf))
        deta = off["eta"] - hlt["eta"]
        dphi = _wrap_phi_np(off["phi"] - hlt["phi"])
        target = np.stack([dlogpt, deta, dphi], axis=1).astype(np.float32)
        target[:, 0] = np.clip(target[:, 0], -goal_max_dlogpt, goal_max_dlogpt)
        target[:, 1] = np.clip(target[:, 1], -goal_max_deta, goal_max_deta)
        target[:, 2] = np.clip(target[:, 2], -goal_max_dphi, goal_max_dphi)
        return target

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

    def _batch_weighted_mean(vec: torch.Tensor, sw: torch.Tensor | None) -> torch.Tensor:
        if sw is None:
            return vec.mean()
        w = sw.float().clamp(min=0.0).to(vec.device)
        return (vec * w).sum() / w.sum().clamp(min=1e-8)

    class OfflineReconstructorGoalConditionedHybridOps(jetlatent.OfflineReconstructorHybridOps):
        """
        HLT-only global correction head plus unchanged hybrid operation branches.

        The head predicts a jet-level correction and applies a bounded global
        correction to operation candidates. Split/generate/reassign/unsmear all
        remain active; the global target acts as a shared pT/axis goal.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            hidden = int(getattr(self, "embed_dim", 128))
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
            abs_eta = p4["eta"].abs()
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
                    mean_cos_phi + 0.0 * abs_eta,
                ],
                dim=1,
            )

        def _predict_goal(self, const_hlt: torch.Tensor, mask_hlt: torch.Tensor) -> torch.Tensor:
            raw = self.goal_head(self._goal_features(const_hlt, mask_hlt))
            return torch.stack(
                [
                    goal_max_dlogpt * torch.tanh(raw[:, 0]),
                    goal_max_deta * torch.tanh(raw[:, 1]),
                    goal_max_dphi * torch.tanh(raw[:, 2]),
                ],
                dim=1,
            )

        def _apply_goal_to_tokens(self, tokens: torch.Tensor, goal_pred: torch.Tensor) -> torch.Tensor:
            if tokens.numel() == 0 or goal_apply_scale <= 0.0:
                return tokens
            dlogpt = (goal_apply_scale * goal_pred[:, 0]).view(-1, 1)
            deta = (goal_apply_scale * goal_pred[:, 1]).view(-1, 1)
            dphi = (goal_apply_scale * goal_pred[:, 2]).view(-1, 1)
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

    def _empty_unmerge_provenance(max_constits: int) -> Dict[str, np.ndarray]:
        return {
            "split_target_mask": np.zeros((max_constits,), dtype=bool),
            "split_mode_target": np.full((max_constits,), MERGE_MODE_NONE, dtype=np.int64),
            "child_type_a_target": np.full((max_constits,), TYPE_UNK, dtype=np.int64),
            "child_type_b_target": np.full((max_constits,), TYPE_UNK, dtype=np.int64),
            "child_attr_a_target": np.zeros((max_constits, 5), dtype=np.float32),
            "child_attr_b_target": np.zeros((max_constits, 5), dtype=np.float32),
        }

    def _default_meta() -> Dict[str, object]:
        return {
            "is_merged_token": False,
            "split_mode_target": MERGE_MODE_NONE,
            "child_type_a_target": TYPE_UNK,
            "child_type_b_target": TYPE_UNK,
            "child_attr_a_target": np.zeros((5,), dtype=np.float32),
            "child_attr_b_target": np.zeros((5,), dtype=np.float32),
        }

    def _apply_hlt_single_jet_m2style_with_provenance(
        tok: np.ndarray,
        msk: np.ndarray,
        params,
        rng: np.random.RandomState,
        max_constits: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
        diag = {
            "n_offline": 0.0,
            "n_after_eff": 0.0,
            "n_after_threshold": 0.0,
            "n_after_merge": 0.0,
            "drop_eff": 0.0,
            "drop_threshold": 0.0,
            "drop_merge": 0.0,
            "drop_total": 0.0,
            "merge_count": 0.0,
        }

        valid = tok[msk].copy()
        meta: List[Dict[str, object]] = [_default_meta() for _ in range(len(valid))]

        n0 = int(len(valid))
        diag["n_offline"] = float(n0)
        empty_prov = _empty_unmerge_provenance(max_constits)
        if n0 == 0:
            return (
                np.zeros((max_constits, RAW_DIM), dtype=np.float32),
                np.zeros((max_constits,), dtype=bool),
                diag,
                empty_prov,
            )

        # 1) Pre-threshold (m2-like)
        pt_thr = float(params.hlt_pt_threshold)
        keep_thr = valid[:, IDX_PT] >= pt_thr
        valid = valid[keep_thr]
        meta = [meta[k] for k in np.where(keep_thr)[0]]
        n_thr = int(len(valid))
        diag["n_after_threshold"] = float(n_thr)
        if n_thr == 0:
            diag["drop_threshold"] = float(n0)
            diag["drop_total"] = float(n0)
            return (
                np.zeros((max_constits, RAW_DIM), dtype=np.float32),
                np.zeros((max_constits,), dtype=bool),
                diag,
                empty_prov,
            )

        # 2) Type-agnostic local merging (deterministic pair scan like m2)
        merge_radius = 0.01 * float(max(0.05, params.merge_prob_scale))
        n_merged = 0
        if merge_radius > 0:
            to_remove: set[int] = set()
            for i in range(len(valid)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(valid)):
                    if j in to_remove:
                        continue
                    deta = float(valid[i, IDX_ETA] - valid[j, IDX_ETA])
                    dphi = float(
                        math.atan2(
                            math.sin(float(valid[i, IDX_PHI] - valid[j, IDX_PHI])),
                            math.cos(float(valid[i, IDX_PHI] - valid[j, IDX_PHI])),
                        )
                    )
                    dR = math.sqrt(deta * deta + dphi * dphi)
                    if dR >= merge_radius:
                        continue

                    t_i = valid[i].copy()
                    t_j = valid[j].copy()
                    m_i = meta[i]
                    m_j = meta[j]

                    merged = m2hlt._merge_tokens_copy_dominant(t_i, t_j)

                    split_mode_target = MERGE_MODE_NONE
                    child_type_a_target = TYPE_UNK
                    child_type_b_target = TYPE_UNK
                    child_attr_a_target = np.zeros((5,), dtype=np.float32)
                    child_attr_b_target = np.zeros((5,), dtype=np.float32)

                    # Only supervise simple first-level merges.
                    if (not bool(m_i["is_merged_token"])) and (not bool(m_j["is_merged_token"])):
                        ti = _infer_type_id(t_i)
                        tj = _infer_type_id(t_j)
                        mode = _infer_merge_mode(ti, tj)
                        if mode != MERGE_MODE_NONE:
                            split_mode_target = int(mode)
                            a, b = (t_i, t_j) if float(t_i[IDX_E]) >= float(t_j[IDX_E]) else (t_j, t_i)
                            child_type_a_target = int(_infer_type_id(a))
                            child_type_b_target = int(_infer_type_id(b))
                            child_attr_a_target = a[[IDX_CHARGE, IDX_D0, IDX_D0ERR, IDX_DZ, IDX_DZERR]].astype(np.float32)
                            child_attr_b_target = b[[IDX_CHARGE, IDX_D0, IDX_D0ERR, IDX_DZ, IDX_DZERR]].astype(np.float32)

                    merged_meta = {
                        "is_merged_token": True,
                        "split_mode_target": int(split_mode_target),
                        "child_type_a_target": int(child_type_a_target),
                        "child_type_b_target": int(child_type_b_target),
                        "child_attr_a_target": child_attr_a_target,
                        "child_attr_b_target": child_attr_b_target,
                    }

                    valid[i] = merged
                    meta[i] = merged_meta
                    to_remove.add(j)
                    n_merged += 1

            if to_remove:
                keep_idx = [k for k in range(len(valid)) if k not in to_remove]
                valid = valid[keep_idx]
                meta = [meta[k] for k in keep_idx]

        n_after_merge_raw = int(len(valid))

        # 3) Efficiency model (m2-style compact adaptation)
        if n_after_merge_raw > 0:
            eta = valid[:, IDX_ETA]
            phi = valid[:, IDX_PHI]
            pt = np.maximum(valid[:, IDX_PT], 1e-8)
            abs_eta = np.abs(eta)

            dens = m2hlt._compute_local_density_np(eta=eta, phi=phi, valid_idx=np.arange(len(valid)), radius=0.04)
            jq = np.clip(rng.lognormal(mean=0.0, sigma=0.08), 0.75, 1.35)

            plateau = np.where(abs_eta < 1.5, float(params.eff_plateau_barrel), float(params.eff_plateau_endcap))
            pt50 = np.where(abs_eta < 1.5, float(params.eff_turnon_pt), float(params.eff_turnon_pt) + 0.30)
            width = np.where(abs_eta < 1.5, float(params.eff_width_pt), 1.25 * float(params.eff_width_pt))
            turn_on = 1.0 / (1.0 + np.exp(-(pt - pt50) / np.maximum(width, 1e-6)))
            density_term = np.exp(-0.055 * dens)
            q_eff = np.clip(jq, 0.90, 1.06)

            eps = plateau * turn_on * density_term * q_eff
            eps = np.clip(eps, 0.02, 0.995)
            keep_eff = rng.random_sample(len(valid)) < eps
            valid = valid[keep_eff]
            meta = [meta[k] for k in np.where(keep_eff)[0]]

        n_eff = int(len(valid))
        diag["n_after_eff"] = float(n_eff)

        # 4) Smearing + tails + local reassignment
        if n_eff > 0:
            pt = np.maximum(valid[:, IDX_PT], 1e-8)
            eta = valid[:, IDX_ETA]
            phi = valid[:, IDX_PHI]
            abs_eta = np.abs(eta)
            dens = m2hlt._compute_local_density_np(eta=eta, phi=phi, valid_idx=np.arange(len(valid)), radius=0.04)
            q = float(np.clip(rng.lognormal(mean=0.0, sigma=0.08), 0.75, 1.35))

            smear_scale = float(max(0.0, params.smear_scale))
            reassign_scale = float(max(0.0, params.reassign_scale))

            sigma_rel = np.sqrt(
                ((0.35 * smear_scale) / np.sqrt(pt)) ** 2
                + (0.012 * smear_scale) ** 2
                + ((0.08 * smear_scale) / pt) ** 2
            )
            sigma_rel = sigma_rel * (1.0 + 0.08 * abs_eta) * q
            sigma_rel = np.clip(sigma_rel, 0.004, 0.40)

            tail_prob = 0.015 + 0.010 * abs_eta + 0.010 * dens
            tail_prob = np.clip(tail_prob, 0.0, 0.25)
            is_tail = rng.random_sample(len(valid)) < tail_prob

            ratio = rng.normal(loc=1.0, scale=sigma_rel)
            tail_sigma = 2.5 * sigma_rel + 0.015
            ratio_tail = rng.normal(loc=0.98, scale=tail_sigma)
            ratio[is_tail] = ratio_tail[is_tail]
            ratio = np.clip(ratio, 0.40, 1.60)
            pt_new = np.clip(pt * ratio, 1e-8, None)

            sigma_eta = (0.0008 * smear_scale + (0.010 * smear_scale) / np.sqrt(pt)) * (1.0 + 0.08 * abs_eta) * q
            sigma_phi = (0.0008 * smear_scale + (0.010 * smear_scale) / np.sqrt(pt)) * (1.0 + 0.08 * abs_eta) * q
            eta_new = eta + rng.normal(loc=0.0, scale=sigma_eta)
            phi_new = m2hlt.wrap_phi_np(phi + rng.normal(loc=0.0, scale=sigma_phi))

            if len(valid) > 1 and reassign_scale > 0.0:
                p_reassign = (0.01 + 0.006 * dens) * reassign_scale
                p_reassign = np.clip(p_reassign, 0.0, 0.08)
                do_reassign = rng.random_sample(len(valid)) < p_reassign
                for ii in np.where(do_reassign)[0]:
                    deta = eta_new[ii] - eta_new
                    dphi = m2hlt.wrap_phi_np(phi_new[ii] - phi_new)
                    dR = np.sqrt(deta * deta + dphi * dphi)
                    dR[ii] = 1e9
                    nn = int(np.argmin(dR))
                    if dR[nn] > 0.08:
                        continue
                    lam = rng.uniform(0.20, 0.65)
                    eta_new[ii] = (1.0 - lam) * eta_new[ii] + lam * eta_new[nn]
                    phi_new[ii] = math.atan2(
                        (1.0 - lam) * math.sin(phi_new[ii]) + lam * math.sin(phi_new[nn]),
                        (1.0 - lam) * math.cos(phi_new[ii]) + lam * math.cos(phi_new[nn]),
                    )

            eta_new = np.clip(eta_new, -5.0, 5.0)
            phi_new = m2hlt.wrap_phi_np(phi_new)
            e_new = pt_new * np.cosh(eta_new)

            valid[:, IDX_PT] = pt_new
            valid[:, IDX_ETA] = eta_new
            valid[:, IDX_PHI] = phi_new
            valid[:, IDX_E] = np.maximum(e_new, 1e-8)

        final = valid
        order = np.argsort(-final[:, IDX_PT]) if len(final) > 0 else np.array([], dtype=np.int64)
        final = final[order] if len(order) > 0 else final
        meta = [meta[int(k)] for k in order] if len(order) > 0 else meta

        take = min(len(final), max_constits)
        out_tok = np.zeros((max_constits, RAW_DIM), dtype=np.float32)
        out_mask = np.zeros((max_constits,), dtype=bool)
        if take > 0:
            out_tok[:take] = final[:take]
            out_mask[:take] = True

        n_final_raw = int(len(final))
        diag["n_after_merge"] = float(n_final_raw)
        diag["drop_threshold"] = float(max(n0 - n_thr, 0))
        diag["drop_merge"] = float(max(n_thr - n_after_merge_raw, 0))
        diag["drop_eff"] = float(max(n_after_merge_raw - n_eff, 0))
        diag["drop_total"] = float(max(n0 - n_final_raw, 0))
        diag["merge_count"] = float(n_merged)

        prov = _empty_unmerge_provenance(max_constits)
        for i in range(take):
            m = meta[i]
            mode = int(m["split_mode_target"])
            prov["split_mode_target"][i] = mode
            prov["split_target_mask"][i] = bool(mode != MERGE_MODE_NONE)
            prov["child_type_a_target"][i] = int(m["child_type_a_target"])
            prov["child_type_b_target"][i] = int(m["child_type_b_target"])
            prov["child_attr_a_target"][i] = np.asarray(m["child_attr_a_target"], dtype=np.float32)
            prov["child_attr_b_target"][i] = np.asarray(m["child_attr_b_target"], dtype=np.float32)

        return out_tok, out_mask, diag, prov

    def _build_hlt_view_m2style_with_provenance(
        tok: np.ndarray,
        msk: np.ndarray,
        params,
        seed: int,
        return_provenance: bool = False,
    ):
        n = len(tok)
        out_tok = np.zeros_like(tok, dtype=np.float32)
        out_msk = np.zeros_like(msk, dtype=bool)
        diag_rows: List[Dict[str, float]] = []
        prov_rows: List[Dict[str, np.ndarray]] = []

        for i in range(n):
            rng = np.random.RandomState(int(seed) + i * 37 + 11)
            ti, mi, di, pi = _apply_hlt_single_jet_m2style_with_provenance(tok[i], msk[i], params, rng, tok.shape[1])
            out_tok[i] = ti
            out_msk[i] = mi
            diag_rows.append(di)
            if return_provenance:
                prov_rows.append(pi)

        keys = [
            "n_offline",
            "n_after_eff",
            "n_after_threshold",
            "n_after_merge",
            "drop_eff",
            "drop_threshold",
            "drop_merge",
            "drop_total",
            "merge_count",
        ]
        per_jet = {k: np.array([row[k] for row in diag_rows], dtype=np.float32) for k in keys}

        if not return_provenance:
            return out_tok, out_msk, per_jet

        pkeys = [
            "split_target_mask",
            "split_mode_target",
            "child_type_a_target",
            "child_type_b_target",
            "child_attr_a_target",
            "child_attr_b_target",
        ]
        prov = {k: np.stack([row[k] for row in prov_rows], axis=0) for k in pkeys}
        _stagea_aux_queue.append(
            {
                "hlt_tok_raw": out_tok.copy(),
                "split_target_mask": prov["split_target_mask"].copy(),
                "split_mode_target": prov["split_mode_target"].copy(),
                "child_type_a_target": prov["child_type_a_target"].copy(),
                "child_type_b_target": prov["child_type_b_target"].copy(),
                "child_attr_a_target": prov["child_attr_a_target"].copy(),
                "child_attr_b_target": prov["child_attr_b_target"].copy(),
            }
        )
        return out_tok, out_msk, per_jet, prov

    class _WeightedReconstructionDatasetFullInfo(Dataset):
        """
        Stage-A reconstruction dataset with full-info supervision fields:
        - split provenance targets for V2 attr losses,
        - parent HLT type/charge/track targets for non-split anchor.
        """

        def __init__(
            self,
            feat_hlt: np.ndarray,
            mask_hlt: np.ndarray,
            const_hlt: np.ndarray,
            const_off: np.ndarray,
            mask_off: np.ndarray,
            budget_merge_true: np.ndarray,
            budget_eff_true: np.ndarray,
            sample_weight_reco: np.ndarray | None = None,
        ):
            self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
            self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
            self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
            self.const_off = torch.tensor(const_off, dtype=torch.float32)
            self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
            self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
            self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)
            self.target_drop_prob = 0.0
            self.target_drop_bank = 0
            goal_target = _goal_target_np(
                const_off[:, :, :4].astype(np.float32),
                mask_off.astype(bool),
                const_hlt[:, :, :4].astype(np.float32),
                mask_hlt.astype(bool),
            )
            self.goal_target = torch.tensor(goal_target, dtype=torch.float32)

            n = int(feat_hlt.shape[0])
            if sample_weight_reco is None:
                sw = np.ones((n,), dtype=np.float32)
            else:
                sw = np.asarray(sample_weight_reco, dtype=np.float32)
                if sw.shape[0] != n:
                    raise ValueError(f"sample_weight_reco length mismatch: {sw.shape[0]} vs {n}")
            self.sample_weight_reco = torch.tensor(sw, dtype=torch.float32)

            aux = _stagea_aux_queue.pop(0) if _stagea_aux_queue else None
            ok = False
            if aux is not None:
                hlt_tok_raw = np.asarray(aux["hlt_tok_raw"], dtype=np.float32)
                ok = (
                    hlt_tok_raw.ndim == 3
                    and hlt_tok_raw.shape[0] == feat_hlt.shape[0]
                    and hlt_tok_raw.shape[1] == feat_hlt.shape[1]
                )
            if not ok:
                hlt_tok_raw = np.zeros((feat_hlt.shape[0], feat_hlt.shape[1], RAW_DIM), dtype=np.float32)
                split_target_mask = np.zeros((feat_hlt.shape[0], feat_hlt.shape[1]), dtype=bool)
                split_mode_target = np.full((feat_hlt.shape[0], feat_hlt.shape[1]), MERGE_MODE_NONE, dtype=np.int64)
                child_type_a_target = np.full((feat_hlt.shape[0], feat_hlt.shape[1]), TYPE_UNK, dtype=np.int64)
                child_type_b_target = np.full((feat_hlt.shape[0], feat_hlt.shape[1]), TYPE_UNK, dtype=np.int64)
                child_attr_a_target = np.zeros((feat_hlt.shape[0], feat_hlt.shape[1], 5), dtype=np.float32)
                child_attr_b_target = np.zeros((feat_hlt.shape[0], feat_hlt.shape[1], 5), dtype=np.float32)
            else:
                split_target_mask = np.asarray(aux["split_target_mask"], dtype=bool)
                split_mode_target = np.asarray(aux["split_mode_target"], dtype=np.int64)
                child_type_a_target = np.asarray(aux["child_type_a_target"], dtype=np.int64)
                child_type_b_target = np.asarray(aux["child_type_b_target"], dtype=np.int64)
                child_attr_a_target = np.asarray(aux["child_attr_a_target"], dtype=np.float32)
                child_attr_b_target = np.asarray(aux["child_attr_b_target"], dtype=np.float32)

            # Parent (HLT) non-kin targets used to anchor non-split branch.
            pid_block = hlt_tok_raw[:, :, IDX_PID0:IDX_PID4 + 1]
            parent_type = np.argmax(pid_block, axis=-1).astype(np.int64)
            parent_type[np.max(pid_block, axis=-1) <= 0.0] = TYPE_UNK
            parent_type[~mask_hlt] = TYPE_UNK

            parent_charge = hlt_tok_raw[:, :, IDX_CHARGE].astype(np.float32)
            parent_charge[~mask_hlt] = 0.0

            parent_track = hlt_tok_raw[:, :, IDX_D0:IDX_DZERR + 1].astype(np.float32)
            parent_track = np.where(mask_hlt[:, :, None], parent_track, 0.0)

            self.split_target_mask = torch.tensor(split_target_mask, dtype=torch.bool)
            self.split_mode_target = torch.tensor(split_mode_target, dtype=torch.long)
            self.child_type_a_target = torch.tensor(child_type_a_target, dtype=torch.long)
            self.child_type_b_target = torch.tensor(child_type_b_target, dtype=torch.long)
            self.child_attr_a_target = torch.tensor(child_attr_a_target, dtype=torch.float32)
            self.child_attr_b_target = torch.tensor(child_attr_b_target, dtype=torch.float32)
            self.parent_type_target = torch.tensor(parent_type, dtype=torch.long)
            self.parent_charge_target = torch.tensor(parent_charge, dtype=torch.float32)
            self.parent_track_target = torch.tensor(parent_track, dtype=torch.float32)

        def __len__(self) -> int:
            return int(self.feat_hlt.shape[0])

        def set_target_drop_state(self, prob: float, bank: int = 0) -> None:
            self.target_drop_prob = float(np.clip(prob, 0.0, 1.0))
            self.target_drop_bank = int(bank) % int(max(1, target_drop_num_banks))

        def _deterministic_keep_extra(self, n_extra: int, idx: int) -> np.ndarray:
            key = (
                (int(args.seed) * 1315423911)
                ^ (int(self.target_drop_bank) * 2654435761)
                ^ (int(idx) * 2246822519)
            ) & 0xFFFFFFFF
            rng = np.random.default_rng(np.uint64(key))
            return rng.random(int(n_extra)) >= float(self.target_drop_prob)

        def _target_dropped_mask_and_budget(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
            mask = self.mask_off[i]
            if target_drop_prob_max <= 0.0 or self.target_drop_prob <= 0.0:
                return mask, self.budget_merge_true[i]

            valid = torch.nonzero(mask, as_tuple=False).flatten()
            n_off = int(valid.numel())
            if n_off <= 1:
                return mask, self.budget_merge_true[i]

            # JetClass HLT tokens are sorted/rebuilt, not index-aligned to offline tokens.
            # Preserve the leading pT-ranked offline core up to the HLT count and apply
            # offdrop only to the extra reconstruction target budget.
            n_hlt = int(self.mask_hlt[i].sum().item())
            n_preserve = int(min(max(n_hlt, 1), n_off))
            extra = valid[n_preserve:]
            if int(extra.numel()) <= 0:
                return mask, self.budget_merge_true[i]

            if target_drop_mode == "deterministic_bank":
                keep_extra = self._deterministic_keep_extra(int(extra.numel()), int(i))
            else:
                rng = np.random.default_rng(np.uint64((int(args.seed) + int(i) * 1000003) & 0xFFFFFFFF))
                keep_extra = rng.random(int(extra.numel())) >= float(self.target_drop_prob)

            out = mask.clone()
            drop_extra = extra[torch.from_numpy(~keep_extra).to(extra.device)]
            if int(drop_extra.numel()) > 0:
                out[drop_extra] = False
            if not bool(out.any()):
                out[valid[0]] = True

            n_target = int(out.sum().item())
            budget = torch.tensor(
                added_target_scale * max(float(n_target - n_hlt), 0.0),
                dtype=self.budget_merge_true.dtype,
            )
            return out, budget

        def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
            mask_off_i, budget_merge_i = self._target_dropped_mask_and_budget(i)
            return {
                "feat_hlt": self.feat_hlt[i],
                "mask_hlt": self.mask_hlt[i],
                "const_hlt": self.const_hlt[i],
                "const_off": self.const_off[i],
                "mask_off": mask_off_i,
                "budget_merge_true": budget_merge_i,
                "budget_eff_true": self.budget_eff_true[i],
                "sample_weight_reco": self.sample_weight_reco[i],
                "goal_target": self.goal_target[i],
                "split_target_mask": self.split_target_mask[i],
                "split_mode_target": self.split_mode_target[i],
                "child_type_a_target": self.child_type_a_target[i],
                "child_type_b_target": self.child_type_b_target[i],
                "child_charge_a_target": self.child_attr_a_target[i, :, 0],
                "child_charge_b_target": self.child_attr_b_target[i, :, 0],
                "child_track_a_target": self.child_attr_a_target[i, :, 1:5],
                "child_track_b_target": self.child_attr_b_target[i, :, 1:5],
                "parent_type_target": self.parent_type_target[i],
                "parent_charge_target": self.parent_charge_target[i],
                "parent_track_target": self.parent_track_target[i],
            }

    def _set_target_drop_state(dataset: Dataset, prob: float, bank: int) -> None:
        if hasattr(dataset, "set_target_drop_state"):
            dataset.set_target_drop_state(prob, bank)  # type: ignore[attr-defined]
        elif isinstance(dataset, Subset):
            _set_target_drop_state(dataset.dataset, prob, bank)

    def _compose_stagea_fullinfo_losses(
        reco_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        *,
        loss_cfg: Dict,
        sample_weight: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        device = reco_out["cand_tokens"].device
        losses_reco = jetlatent.compute_reconstruction_losses_weighted_hybrid_ops(
            reco_out,
            batch["const_hlt"].to(device),
            batch["mask_hlt"].to(device),
            batch["const_off"].to(device),
            batch["mask_off"].to(device),
            batch["budget_merge_true"].to(device),
            batch["budget_eff_true"].to(device),
            loss_cfg,
            sample_weight=sample_weight,
        )

        losses_attr = v2.compute_v2_attr_losses(
            reco_out,
            batch,
            mode_none_weight=stagea_mode_none_weight,
            mode_label_smoothing=stagea_mode_label_smoothing,
            track_weight=stagea_track_weight,
        )
        loss_attr_main = (
            stagea_attr_lam_mode * losses_attr["mode"]
            + stagea_attr_lam_type * losses_attr["type"]
            + stagea_attr_lam_charge * losses_attr["charge"]
            + stagea_attr_lam_track * losses_attr["track"]
        )

        # Non-split anchor: for unsmear/reassign path, keep parent attrs close to HLT input attrs.
        zero = torch.zeros((), device=device)
        mask_hlt = batch["mask_hlt"].to(device)
        split_target_mask = batch.get("split_target_mask", torch.zeros_like(mask_hlt)).to(device)
        nonsplit = mask_hlt & (~split_target_mask)
        if nonsplit.any() and ("child_type_logits" in reco_out):
            type_logits = reco_out["child_type_logits"][:, :, 0, :]
            type_tgt = batch["parent_type_target"].to(device)
            loss_anchor_type = F.cross_entropy(type_logits[nonsplit], type_tgt[nonsplit])
        else:
            loss_anchor_type = zero

        if nonsplit.any() and ("child_charge_pred" in reco_out):
            charge_pred = reco_out["child_charge_pred"][:, :, 0]
            charge_tgt = batch["parent_charge_target"].to(device)
            type_tgt = batch["parent_type_target"].to(device)
            track_like = (type_tgt == TYPE_CH) | (type_tgt == TYPE_ELE) | (type_tgt == TYPE_MU)
            charge_mask = nonsplit & track_like
            if charge_mask.any():
                loss_anchor_charge = F.smooth_l1_loss(charge_pred[charge_mask], charge_tgt[charge_mask])
            else:
                loss_anchor_charge = zero
        else:
            loss_anchor_charge = zero

        if nonsplit.any() and ("child_track_pred" in reco_out):
            track_pred = reco_out["child_track_pred"][:, :, 0, :]
            track_tgt = batch["parent_track_target"].to(device)
            type_tgt = batch["parent_type_target"].to(device)
            track_like = (type_tgt == TYPE_CH) | (type_tgt == TYPE_ELE) | (type_tgt == TYPE_MU)
            track_mask = (nonsplit & track_like).unsqueeze(-1).expand(-1, -1, 4)
            if track_mask.any():
                loss_anchor_track = F.smooth_l1_loss(track_pred[track_mask], track_tgt[track_mask])
            else:
                loss_anchor_track = zero
        else:
            loss_anchor_track = zero

        loss_anchor = (
            stagea_anchor_type * loss_anchor_type
            + stagea_anchor_charge * loss_anchor_charge
            + stagea_anchor_track * loss_anchor_track
        )

        goal_pred = reco_out.get("goal_pred", None)
        if goal_pred is not None and "goal_target" in batch:
            goal_target = batch["goal_target"].to(device)
            goal_head_vec = F.smooth_l1_loss(goal_pred, goal_target, reduction="none").mean(dim=1)
            loss_goal_head = _batch_weighted_mean(goal_head_vec, sample_weight)

            pred_p4 = _weighted_p4_t(reco_out["cand_tokens"], reco_out["cand_weights"])
            true_p4 = _weighted_p4_t(batch["const_off"].to(device), batch["mask_off"].to(device).float())
            response_vec = F.smooth_l1_loss(
                pred_p4["pt"] / true_p4["pt"].clamp(min=1e-8),
                torch.ones_like(pred_p4["pt"]),
                reduction="none",
            )
            loss_goal_response = _batch_weighted_mean(response_vec, sample_weight)

            deta = pred_p4["eta"] - true_p4["eta"]
            dphi = _wrap_phi_t(pred_p4["phi"] - true_p4["phi"])
            axis_vec = F.smooth_l1_loss(deta, torch.zeros_like(deta), reduction="none") + F.smooth_l1_loss(
                dphi,
                torch.zeros_like(dphi),
                reduction="none",
            )
            loss_goal_axis = _batch_weighted_mean(axis_vec, sample_weight)
        else:
            loss_goal_head = zero
            loss_goal_response = zero
            loss_goal_axis = zero

        loss_goal = (
            goal_lambda_head * loss_goal_head
            + goal_lambda_response * loss_goal_response
            + goal_lambda_axis * loss_goal_axis
        )

        total = losses_reco["total"] + loss_attr_main + loss_anchor + loss_goal
        return {
            "total": total,
            "set": losses_reco["set"],
            "budget": losses_reco["budget"],
            "pt_ratio": losses_reco["pt_ratio"],
            "local": losses_reco["local"],
            "attr_main": loss_attr_main,
            "anchor": loss_anchor,
            "goal": loss_goal,
            "goal_head": loss_goal_head,
            "goal_response": loss_goal_response,
            "goal_axis": loss_goal_axis,
        }

    def _train_reconstructor_weighted_fullinfo(
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        train_cfg: Dict,
        loss_cfg: Dict,
        apply_reco_weight: bool,
        reload_best_at_stage_transition: bool,
    ):
        # `reload_best_at_stage_transition` kept for API compatibility.
        _ = reload_best_at_stage_transition
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        sch = reco_joint.get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

        best_state = None
        best_val = 1e9
        no_improve = 0
        min_stop_epoch = int(train_cfg.get("stage2_epochs", 0)) + int(train_cfg.get("min_full_scale_epochs", 5))

        for ep in tqdm(range(int(train_cfg["epochs"])), desc="Reconstructor"):
            drop_prob = target_drop_prob_max * min(1.0, float(ep + 1) / float(target_drop_warmup_epochs))
            if target_drop_mode == "deterministic_bank":
                drop_bank = (int(ep) // int(target_drop_bank_cycle_epochs)) % int(max(1, target_drop_num_banks))
            else:
                drop_bank = 0
            _set_target_drop_state(train_loader.dataset, drop_prob, drop_bank)
            _set_target_drop_state(val_loader.dataset, drop_prob, drop_bank)

            model.train()
            sc = reco_joint.stage_scale_local(ep, train_cfg)
            tr_total = tr_set = tr_budget = tr_pt = tr_local = tr_attr = tr_anchor = tr_goal = 0.0
            n_tr = 0
            for batch in train_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                sw_reco = batch.get("sample_weight_reco", None)
                if sw_reco is not None:
                    sw_reco = sw_reco.to(device)

                opt.zero_grad()
                out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=sc)
                losses = _compose_stagea_fullinfo_losses(
                    out,
                    batch,
                    loss_cfg=loss_cfg,
                    sample_weight=(sw_reco if (bool(apply_reco_weight) and sw_reco is not None) else None),
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                bs = int(feat_hlt.size(0))
                tr_total += float(losses["total"].item()) * bs
                tr_set += float(losses["set"].item()) * bs
                tr_budget += float(losses["budget"].item()) * bs
                tr_pt += float(losses["pt_ratio"].item()) * bs
                tr_local += float(losses["local"].item()) * bs
                tr_attr += float(losses["attr_main"].item()) * bs
                tr_anchor += float(losses["anchor"].item()) * bs
                tr_goal += float(losses["goal"].item()) * bs
                n_tr += bs

            model.eval()
            va_total_u = va_set_u = va_budget_u = va_pt_u = va_local_u = va_attr_u = va_anchor_u = va_goal_u = 0.0
            va_total_w = 0.0
            n_va = 0
            with torch.no_grad():
                for batch in val_loader:
                    feat_hlt = batch["feat_hlt"].to(device)
                    mask_hlt = batch["mask_hlt"].to(device)
                    const_hlt = batch["const_hlt"].to(device)
                    sw_reco = batch.get("sample_weight_reco", None)
                    if sw_reco is not None:
                        sw_reco = sw_reco.to(device)

                    out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
                    losses_u = _compose_stagea_fullinfo_losses(
                        out,
                        batch,
                        loss_cfg=loss_cfg,
                        sample_weight=None,
                    )
                    if bool(apply_reco_weight) and sw_reco is not None:
                        losses_w = _compose_stagea_fullinfo_losses(
                            out,
                            batch,
                            loss_cfg=loss_cfg,
                            sample_weight=sw_reco,
                        )
                    else:
                        losses_w = losses_u

                    bs = int(feat_hlt.size(0))
                    va_total_u += float(losses_u["total"].item()) * bs
                    va_set_u += float(losses_u["set"].item()) * bs
                    va_budget_u += float(losses_u["budget"].item()) * bs
                    va_pt_u += float(losses_u["pt_ratio"].item()) * bs
                    va_local_u += float(losses_u["local"].item()) * bs
                    va_attr_u += float(losses_u["attr_main"].item()) * bs
                    va_anchor_u += float(losses_u["anchor"].item()) * bs
                    va_goal_u += float(losses_u["goal"].item()) * bs
                    va_total_w += float(losses_w["total"].item()) * bs
                    n_va += bs

            sch.step()
            tr_total /= max(n_tr, 1)
            tr_set /= max(n_tr, 1)
            tr_budget /= max(n_tr, 1)
            tr_pt /= max(n_tr, 1)
            tr_local /= max(n_tr, 1)
            tr_attr /= max(n_tr, 1)
            tr_anchor /= max(n_tr, 1)
            tr_goal /= max(n_tr, 1)

            va_total_u /= max(n_va, 1)
            va_set_u /= max(n_va, 1)
            va_budget_u /= max(n_va, 1)
            va_pt_u /= max(n_va, 1)
            va_local_u /= max(n_va, 1)
            va_attr_u /= max(n_va, 1)
            va_anchor_u /= max(n_va, 1)
            va_goal_u /= max(n_va, 1)
            va_total_w /= max(n_va, 1)

            select_metric = va_total_w if bool(apply_reco_weight) else va_total_u
            if select_metric < best_val:
                best_val = float(select_metric)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(
                    f"Ep {ep+1}: train_total={tr_total:.4f}, val_total_unw={va_total_u:.4f}, "
                    f"val_total_w={va_total_w:.4f}, select={'weighted' if bool(apply_reco_weight) else 'unweighted'}, "
                    f"best_sel={best_val:.4f} | set_unw={va_set_u:.4f}, attr_unw={va_attr_u:.4f}, "
                    f"anchor_unw={va_anchor_u:.4f}, goal_unw={va_goal_u:.4f}, budget_unw={va_budget_u:.4f}, "
                    f"w_set={float(loss_cfg.get('w_set', 0.0)):.3f}, w_budget={float(loss_cfg.get('w_budget', 0.0)):.3f}, "
                    f"w_pt={float(loss_cfg.get('w_pt_ratio', 0.0)):.3f}, w_local={float(loss_cfg.get('w_local', 0.0)):.3f}, "
                    f"goal_w(head/resp/axis)={goal_lambda_head:.2f}/{goal_lambda_response:.2f}/{goal_lambda_axis:.2f}, "
                    f"stage_scale={sc:.2f}, target_drop={drop_prob:.3f}, bank={drop_bank}"
                )

            if no_improve >= int(train_cfg["patience"]) and (ep + 1) >= int(max(min_stop_epoch, 1)):
                print(f"Early stopping reconstructor at epoch {ep+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, {"val_total": float(best_val)}

    # HLT profile: match current m2-style setup with V2-compatible API and real provenance.
    v2.build_hlt_view = _build_hlt_view_m2style_with_provenance

    # Reconstructor/loss/corrected-view: hybrid ops + full-info Stage-A dataset/trainer.
    v2.OfflineReconstructor = OfflineReconstructorGoalConditionedHybridOps
    v2.compute_reconstruction_losses_weighted = jetlatent.compute_reconstruction_losses_weighted_hybrid_ops
    v2.build_soft_corrected_view = jetlatent.build_soft_corrected_view_hybrid_ops
    v2.wrap_reconstructor_unmerge_only = _identity_wrap
    v2.WeightedReconstructionDataset = _WeightedReconstructionDatasetFullInfo
    v2.train_reconstructor_weighted = _train_reconstructor_weighted_fullinfo

    # Stage-A trainer calls globals from reco_joint; patch there too.
    reco_joint.OfflineReconstructor = OfflineReconstructorGoalConditionedHybridOps
    reco_joint.compute_reconstruction_losses_weighted = jetlatent.compute_reconstruction_losses_weighted_hybrid_ops
    reco_joint.enforce_unmerge_only_output = _identity_enforce
    reco_joint.wrap_reconstructor_unmerge_only = _identity_wrap
    reco_joint.WeightedReconstructionDataset = _WeightedReconstructionDatasetFullInfo

    v2.run(args)


if __name__ == "__main__":
    main()

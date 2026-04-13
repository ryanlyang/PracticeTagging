#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JetClass V2-Attr + m2-style HLT + jetlatent set2set base reconstructor.

Purpose:
- Keep the V2 constrained attribute-head training/eval pipeline.
- Swap the base reconstructor to jetlatent set2set.
- Keep m2-style HLT corruption used by the current jetlatent runs.

This wrapper avoids modifying existing PracticeTagging scripts.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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
        "offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_jetlatent_set2set.py",
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
        "offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_jetlatent_set2set"
    )
    reco_joint = importlib.import_module(
        "offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly"
    )

    TYPE_CH = int(getattr(v2, "TYPE_CH", 0))
    TYPE_NH = int(getattr(v2, "TYPE_NH", 1))
    TYPE_GAM = int(getattr(v2, "TYPE_GAM", 2))
    TYPE_ELE = int(getattr(v2, "TYPE_ELE", 3))
    TYPE_UNK = int(getattr(v2, "TYPE_UNK", 5))

    MERGE_MODE_NONE = int(getattr(v2, "MERGE_MODE_NONE", 0))
    MERGE_MODE_SAME_TYPE = int(getattr(v2, "MERGE_MODE_SAME_TYPE", 1))
    MERGE_MODE_ELE_GAM = int(getattr(v2, "MERGE_MODE_ELE_GAM", 2))
    MERGE_MODE_CH_NH = int(getattr(v2, "MERGE_MODE_CH_NH", 3))

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
        return out_tok, out_msk, per_jet, prov

    # HLT profile: match current m2-style setup with V2-compatible API and real provenance.
    v2.build_hlt_view = _build_hlt_view_m2style_with_provenance

    # Reconstructor/loss/corrected-view: swap to jetlatent set2set.
    v2.OfflineReconstructor = jetlatent.OfflineReconstructorJetLatentSet2Set
    v2.compute_reconstruction_losses_weighted = jetlatent.compute_reconstruction_losses_weighted_set2set
    v2.build_soft_corrected_view = jetlatent.build_soft_corrected_view_set2set
    v2.wrap_reconstructor_unmerge_only = _identity_wrap

    # Stage-A trainer calls globals from reco_joint; patch there too.
    reco_joint.compute_reconstruction_losses_weighted = jetlatent.compute_reconstruction_losses_weighted_set2set
    reco_joint.enforce_unmerge_only_output = _identity_enforce
    reco_joint.wrap_reconstructor_unmerge_only = _identity_wrap

    args = v2.parse_args()
    v2.run(args)


if __name__ == "__main__":
    main()

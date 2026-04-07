#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JetClass joint dual-view (V1 model) with m2-style HLT generation.

Differences from the default JetClass HLT builder:
1) Type-agnostic local merging (no type veto).
2) Merged token identity copies the dominant-energy source token.
3) Efficiency/smear/reassign behavior follows the m2-like style used in
   offline_reconstructor_no_gt_local30kv2.py (with a compact adaptation for JetClass).

This file is a wrapper that monkey-patches only this process, so existing runs are unaffected.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import train_jetclass_joint_dualview_stage2_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2 import _compute_local_density_np, wrap_phi_np


RAW_DIM = 14
IDX_PT = 0
IDX_ETA = 1
IDX_PHI = 2
IDX_E = 3


def _merge_tokens_copy_dominant(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Merge kinematics while copying non-kin fields from dominant-energy source."""
    e1 = float(t1[IDX_E])
    e2 = float(t2[IDX_E])
    if e2 > e1 or (abs(e2 - e1) < 1e-8 and float(t2[IDX_PT]) > float(t1[IDX_PT])):
        dom = t2
    else:
        dom = t1

    out = dom.copy()
    pt1 = max(float(t1[IDX_PT]), 1e-8)
    pt2 = max(float(t2[IDX_PT]), 1e-8)
    pt_sum = pt1 + pt2
    w1 = pt1 / max(pt_sum, 1e-8)
    w2 = pt2 / max(pt_sum, 1e-8)

    eta = w1 * float(t1[IDX_ETA]) + w2 * float(t2[IDX_ETA])
    phi = math.atan2(
        w1 * math.sin(float(t1[IDX_PHI])) + w2 * math.sin(float(t2[IDX_PHI])),
        w1 * math.cos(float(t1[IDX_PHI])) + w2 * math.cos(float(t2[IDX_PHI])),
    )
    e = max(float(t1[IDX_E] + t2[IDX_E]), 1e-8)

    out[IDX_PT] = pt_sum
    out[IDX_ETA] = np.clip(eta, -5.0, 5.0)
    out[IDX_PHI] = wrap_phi_np(np.array([phi], dtype=np.float64))[0]
    out[IDX_E] = e
    return out.astype(np.float32)


def _apply_hlt_single_jet_m2style(
    tok: np.ndarray,
    msk: np.ndarray,
    params,
    rng: np.random.RandomState,
    max_constits: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    diag = {
        "n_offline": 0.0,
        "n_after_eff": 0.0,
        "n_after_threshold": 0.0,
        "n_after_merge": 0.0,  # final count after all effects (for compatibility)
        "drop_eff": 0.0,
        "drop_threshold": 0.0,
        "drop_merge": 0.0,
        "drop_total": 0.0,
        "merge_count": 0.0,
    }

    valid = tok[msk].copy()
    n0 = int(len(valid))
    diag["n_offline"] = float(n0)
    if n0 == 0:
        return (
            np.zeros((max_constits, RAW_DIM), dtype=np.float32),
            np.zeros((max_constits,), dtype=bool),
            diag,
        )

    # 1) Pre-threshold (m2-like)
    pt_thr = float(params.hlt_pt_threshold)
    keep_thr = valid[:, IDX_PT] >= pt_thr
    valid = valid[keep_thr]
    n_thr = int(len(valid))
    diag["n_after_threshold"] = float(n_thr)
    if n_thr == 0:
        diag["drop_threshold"] = float(n0)
        diag["drop_total"] = float(n0)
        return (
            np.zeros((max_constits, RAW_DIM), dtype=np.float32),
            np.zeros((max_constits,), dtype=bool),
            diag,
        )

    # 2) Type-agnostic local merging (deterministic pair scan like m2)
    # Use merge_prob_scale as radius multiplier so legacy runner knobs stay meaningful.
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
                valid[i] = _merge_tokens_copy_dominant(valid[i], valid[j])
                to_remove.add(j)
                n_merged += 1
        if to_remove:
            keep_idx = [k for k in range(len(valid)) if k not in to_remove]
            valid = valid[keep_idx]
    n_after_merge_raw = int(len(valid))

    # 3) Efficiency model (m2-style compact adaptation)
    if n_after_merge_raw > 0:
        eta = valid[:, IDX_ETA]
        phi = valid[:, IDX_PHI]
        pt = np.maximum(valid[:, IDX_PT], 1e-8)
        abs_eta = np.abs(eta)

        dens = _compute_local_density_np(eta=eta, phi=phi, valid_idx=np.arange(len(valid)), radius=0.04)
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
    n_eff = int(len(valid))
    diag["n_after_eff"] = float(n_eff)

    # 4) Smearing + tails + local reassignment (m2-like style, scaled by runner knobs)
    if n_eff > 0:
        pt = np.maximum(valid[:, IDX_PT], 1e-8)
        eta = valid[:, IDX_ETA]
        phi = valid[:, IDX_PHI]
        abs_eta = np.abs(eta)
        dens = _compute_local_density_np(eta=eta, phi=phi, valid_idx=np.arange(len(valid)), radius=0.04)
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
        phi_new = wrap_phi_np(phi + rng.normal(loc=0.0, scale=sigma_phi))

        # Local reassignment
        if len(valid) > 1 and reassign_scale > 0.0:
            p_reassign = (0.01 + 0.006 * dens) * reassign_scale
            p_reassign = np.clip(p_reassign, 0.0, 0.08)
            do_reassign = rng.random_sample(len(valid)) < p_reassign
            for ii in np.where(do_reassign)[0]:
                deta = eta_new[ii] - eta_new
                dphi = wrap_phi_np(phi_new[ii] - phi_new)
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
        phi_new = wrap_phi_np(phi_new)
        e_new = pt_new * np.cosh(eta_new)

        valid[:, IDX_PT] = pt_new
        valid[:, IDX_ETA] = eta_new
        valid[:, IDX_PHI] = phi_new
        valid[:, IDX_E] = np.maximum(e_new, 1e-8)

    # Optional post-smear threshold kept at 0.0 (m2 default), so no extra drop.
    final = valid
    order = np.argsort(-final[:, IDX_PT]) if len(final) > 0 else np.array([], dtype=np.int64)
    final = final[order] if len(order) > 0 else final
    take = min(len(final), max_constits)
    out_tok = np.zeros((max_constits, RAW_DIM), dtype=np.float32)
    out_mask = np.zeros((max_constits,), dtype=bool)
    if take > 0:
        out_tok[:take] = final[:take]
        out_mask[:take] = True

    n_final = int(take if len(final) <= max_constits else max_constits)
    # For diagnostics, treat merge/eff shares by true step deltas before truncation.
    n_final_raw = int(len(final))
    diag["n_after_merge"] = float(n_final_raw)
    diag["drop_threshold"] = float(max(n0 - n_thr, 0))
    diag["drop_merge"] = float(max(n_thr - n_after_merge_raw, 0))
    diag["drop_eff"] = float(max(n_after_merge_raw - n_eff, 0))
    diag["drop_total"] = float(max(n0 - n_final_raw, 0))
    diag["merge_count"] = float(n_merged)
    return out_tok, out_mask, diag


def _build_hlt_view_m2style(
    tok: np.ndarray,
    msk: np.ndarray,
    params,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    n = len(tok)
    out_tok = np.zeros_like(tok, dtype=np.float32)
    out_msk = np.zeros_like(msk, dtype=bool)
    diag_rows: List[Dict[str, float]] = []
    for i in tqdm(range(n), desc="Applying m2-style HLT corruption"):
        rng = np.random.RandomState(int(seed) + i * 37 + 11)
        ti, mi, di = _apply_hlt_single_jet_m2style(tok[i], msk[i], params, rng, tok.shape[1])
        out_tok[i] = ti
        out_msk[i] = mi
        diag_rows.append(di)

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
    return out_tok, out_msk, per_jet


def main() -> None:
    # Patch only this entrypoint.
    base.build_hlt_view = _build_hlt_view_m2style
    args = base.parse_args()
    base.run(args)


if __name__ == "__main__":
    main()


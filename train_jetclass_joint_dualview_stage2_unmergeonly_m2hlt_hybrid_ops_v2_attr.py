#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JetClass V2-Attr + m2-style HLT + hybrid-ops base reconstructor.

Purpose:
- Keep the V2 constrained attribute-head training/eval pipeline.
- Swap the base reconstructor to hybrid ops.
- Keep m2-style HLT corruption used by the current jetlatent runs.
- Enable full-info Stage-A from epoch 1 by adding attr losses directly into
  the main reconstructor pretrain objective (not only StageA-Attr calibration).

This wrapper avoids modifying existing PracticeTagging scripts.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

    # Parse once so patched Stage-A trainer can use the exact run-time knobs.
    args = v2.parse_args()
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

        def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
            return {
                "feat_hlt": self.feat_hlt[i],
                "mask_hlt": self.mask_hlt[i],
                "const_hlt": self.const_hlt[i],
                "const_off": self.const_off[i],
                "mask_off": self.mask_off[i],
                "budget_merge_true": self.budget_merge_true[i],
                "budget_eff_true": self.budget_eff_true[i],
                "sample_weight_reco": self.sample_weight_reco[i],
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

        total = losses_reco["total"] + loss_attr_main + loss_anchor
        return {
            "total": total,
            "set": losses_reco["set"],
            "budget": losses_reco["budget"],
            "pt_ratio": losses_reco["pt_ratio"],
            "local": losses_reco["local"],
            "attr_main": loss_attr_main,
            "anchor": loss_anchor,
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
            model.train()
            sc = reco_joint.stage_scale_local(ep, train_cfg)
            tr_total = tr_set = tr_budget = tr_pt = tr_local = tr_attr = tr_anchor = 0.0
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
                n_tr += bs

            model.eval()
            va_total_u = va_set_u = va_budget_u = va_pt_u = va_local_u = va_attr_u = va_anchor_u = 0.0
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

            va_total_u /= max(n_va, 1)
            va_set_u /= max(n_va, 1)
            va_budget_u /= max(n_va, 1)
            va_pt_u /= max(n_va, 1)
            va_local_u /= max(n_va, 1)
            va_attr_u /= max(n_va, 1)
            va_anchor_u /= max(n_va, 1)
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
                    f"anchor_unw={va_anchor_u:.4f}, budget_unw={va_budget_u:.4f}, "
                    f"w_set={float(loss_cfg.get('w_set', 0.0)):.3f}, w_budget={float(loss_cfg.get('w_budget', 0.0)):.3f}, "
                    f"w_pt={float(loss_cfg.get('w_pt_ratio', 0.0)):.3f}, w_local={float(loss_cfg.get('w_local', 0.0)):.3f}, "
                    f"stage_scale={sc:.2f}"
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
    v2.OfflineReconstructor = jetlatent.OfflineReconstructorHybridOps
    v2.compute_reconstruction_losses_weighted = jetlatent.compute_reconstruction_losses_weighted_hybrid_ops
    v2.build_soft_corrected_view = jetlatent.build_soft_corrected_view_hybrid_ops
    v2.wrap_reconstructor_unmerge_only = _identity_wrap
    v2.WeightedReconstructionDataset = _WeightedReconstructionDatasetFullInfo
    v2.train_reconstructor_weighted = _train_reconstructor_weighted_fullinfo

    # Stage-A trainer calls globals from reco_joint; patch there too.
    reco_joint.compute_reconstruction_losses_weighted = jetlatent.compute_reconstruction_losses_weighted_hybrid_ops
    reco_joint.enforce_unmerge_only_output = _identity_enforce
    reco_joint.wrap_reconstructor_unmerge_only = _identity_wrap
    reco_joint.WeightedReconstructionDataset = _WeightedReconstructionDatasetFullInfo

    v2.run(args)


if __name__ == "__main__":
    main()

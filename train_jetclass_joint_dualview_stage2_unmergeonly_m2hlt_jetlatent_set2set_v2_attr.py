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
import sys
from pathlib import Path
from typing import Dict

import numpy as np


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

    def _build_hlt_view_m2style_with_provenance(
        tok: np.ndarray,
        msk: np.ndarray,
        params,
        seed: int,
        return_provenance: bool = False,
    ):
        # m2-style builder returns only (tok, mask, diag).
        out_tok, out_msk, per_jet = m2hlt._build_hlt_view_m2style(tok, msk, params=params, seed=seed)
        if not return_provenance:
            return out_tok, out_msk, per_jet

        # V2 expects provenance tensors; m2-style HLT does not currently expose
        # per-token merge ancestry, so provide a shape-compatible "no split"
        # placeholder to keep Stage A/B/C training runnable.
        n_jets, max_constits = out_msk.shape
        mode_none = int(getattr(v2, "MERGE_MODE_NONE", 0))
        type_unk = int(getattr(v2, "TYPE_UNK", 5))
        prov = {
            "split_target_mask": np.zeros((n_jets, max_constits), dtype=bool),
            "split_mode_target": np.full((n_jets, max_constits), mode_none, dtype=np.int64),
            "child_type_a_target": np.full((n_jets, max_constits), type_unk, dtype=np.int64),
            "child_type_b_target": np.full((n_jets, max_constits), type_unk, dtype=np.int64),
            "child_attr_a_target": np.zeros((n_jets, max_constits, 5), dtype=np.float32),
            "child_attr_b_target": np.zeros((n_jets, max_constits, 5), dtype=np.float32),
        }
        return out_tok, out_msk, per_jet, prov

    # HLT profile: match current m2-style setup with V2-compatible API.
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

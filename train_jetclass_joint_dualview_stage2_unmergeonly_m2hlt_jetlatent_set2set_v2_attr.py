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

    # HLT profile: match current m2-style setup.
    v2.build_hlt_view = m2hlt._build_hlt_view_m2style

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

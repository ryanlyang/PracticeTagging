#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full joint-training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps in reconstructor variant:
  K=3 split children + soft-gating split allocation.
"""

from __future__ import annotations

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2_splitk3_softgate import OfflineReconstructorSplitK3SoftGate

base.OfflineReconstructor = OfflineReconstructorSplitK3SoftGate
base.reco_base.OfflineReconstructor = OfflineReconstructorSplitK3SoftGate


if __name__ == "__main__":
    print("[SplitK3-SoftGate] Using K=3 split children with soft-gated allocation")
    base.main()


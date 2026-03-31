#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full joint-training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps in reconstructor variant with token log-pT upper clamp 10.0 (from 9.0).
"""

from __future__ import annotations

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2_ptclamp10 import OfflineReconstructorPtClamp10

# Swap reconstructor class used by the runner and any reco_base references.
base.OfflineReconstructor = OfflineReconstructorPtClamp10
base.reco_base.OfflineReconstructor = OfflineReconstructorPtClamp10


if __name__ == "__main__":
    print("[PtClamp10] Using reconstructor token log-pT upper clamp = 10.0")
    base.main()


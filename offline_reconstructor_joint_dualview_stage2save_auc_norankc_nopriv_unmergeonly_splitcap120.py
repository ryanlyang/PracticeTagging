#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full joint-training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps in reconstructor variant with higher split child pT cap
  (total child pT up to 1.2 * parent).
"""

from __future__ import annotations

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2_splitcap120 import OfflineReconstructorSplitCap120

base.OfflineReconstructor = OfflineReconstructorSplitCap120
base.reco_base.OfflineReconstructor = OfflineReconstructorSplitCap120


if __name__ == "__main__":
    print("[SplitCap120] Using split total child pT cap = 1.2 * parent")
    base.main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full joint-training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps in reconstructor variant with split-angle cap 0.35.
"""

from __future__ import annotations

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2_anglecap035 import OfflineReconstructorAngleCap035

base.OfflineReconstructor = OfflineReconstructorAngleCap035
base.reco_base.OfflineReconstructor = OfflineReconstructorAngleCap035


if __name__ == "__main__":
    print("[AngleCap035] Using split child angle cap 0.35")
    base.main()


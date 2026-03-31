#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full joint-training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps in reconstructor variant with budget high clamp 6.0.
"""

from __future__ import annotations

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2_budgetclamp6 import OfflineReconstructorBudgetClamp6

base.OfflineReconstructor = OfflineReconstructorBudgetClamp6
base.reco_base.OfflineReconstructor = OfflineReconstructorBudgetClamp6


if __name__ == "__main__":
    print("[BudgetClamp6] Using budget calibration high clamp 6.0")
    base.main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses the full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps in reconstructor with factorized edit decision:
    edit-vs-keep gate + edit-type softmax.
"""

from __future__ import annotations

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2_factorized_edit import OfflineReconstructorFactorizedEdit

base.OfflineReconstructor = OfflineReconstructorFactorizedEdit


if __name__ == "__main__":
    base.main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses the full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps in expressive split heads only (no context conditioning).
"""

from __future__ import annotations

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2_splitheads_only import OfflineReconstructorSplitHeadsOnly

base.OfflineReconstructor = OfflineReconstructorSplitHeadsOnly


if __name__ == "__main__":
    base.main()


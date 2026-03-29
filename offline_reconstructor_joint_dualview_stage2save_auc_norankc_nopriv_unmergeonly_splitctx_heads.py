#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses the full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps in a context-conditioned, more expressive split-head reconstructor.
"""

from __future__ import annotations

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
from offline_reconstructor_no_gt_local30kv2_splitctx_heads import OfflineReconstructorSplitCtx

# Override only the reconstructor class used by the base pipeline.
base.OfflineReconstructor = OfflineReconstructorSplitCtx


if __name__ == "__main__":
    base.main()


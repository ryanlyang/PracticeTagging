#!/usr/bin/env python3
"""Run JetClass stacked-fusion analyzer with m2 HLT and simple3 reconstructor."""

from __future__ import annotations

import numpy as np

import analyze_jetclass_four_model_stacked_fusion as fusion
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_simple3_ops as simple3_ops
from train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt import _build_hlt_view_m2style


def _empty_provenance(tok_shape):
    n, max_constits = int(tok_shape[0]), int(tok_shape[1])
    return {
        "split_target_mask": np.zeros((n, max_constits), dtype=bool),
        "split_mode_target": np.zeros((n, max_constits), dtype=np.int64),
        "child_type_a_target": np.zeros((n, max_constits), dtype=np.int64),
        "child_type_b_target": np.zeros((n, max_constits), dtype=np.int64),
        "child_attr_a_target": np.zeros((n, max_constits, 5), dtype=np.float32),
        "child_attr_b_target": np.zeros((n, max_constits, 5), dtype=np.float32),
    }


def _build_hlt_view_m2style_compat(tok, msk, params, seed, return_provenance=False):
    out_tok, out_msk, diag = _build_hlt_view_m2style(tok, msk, params=params, seed=seed)
    if return_provenance:
        return out_tok, out_msk, diag, _empty_provenance(tok.shape)
    return out_tok, out_msk, diag


def main() -> None:
    if fusion.v2 is None or fusion.hybrid_ops is None:
        raise RuntimeError("JetClass fusion dependencies failed to import; activate atlas_kd.")
    fusion.v2.build_hlt_view = _build_hlt_view_m2style_compat
    fusion.hybrid_ops.OfflineReconstructorHybridOps = simple3_ops.OfflineReconstructorSimple3Ops
    fusion.hybrid_ops.build_soft_corrected_view_hybrid_ops = simple3_ops.build_soft_corrected_view_simple3_ops
    fusion.main()


if __name__ == "__main__":
    main()

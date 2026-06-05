#!/usr/bin/env python3
"""
Run the standard JetClass stacked-fusion analyzer with the m2-style HLT builder.

The baseline-HLT seed-ensemble control is trained on m2-style fixed HLT. The
standard analyzer imports the default V2 HLT builder, so this wrapper patches
the analyzer process before calling its main().
"""

from __future__ import annotations

import numpy as np

import analyze_jetclass_four_model_stacked_fusion as fusion
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
    if fusion.v2 is None:
        raise RuntimeError("JetClass fusion dependencies failed to import; activate atlas_kd.")
    fusion.v2.build_hlt_view = _build_hlt_view_m2style_compat
    fusion.main()


if __name__ == "__main__":
    main()

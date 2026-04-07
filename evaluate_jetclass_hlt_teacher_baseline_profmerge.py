#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JetClass teacher-vs-baseline experiment variant:
- Type-agnostic HLT merging (no pair-type veto),
- Merged output particle type follows dominant-energy constituent
  (implemented by leaving merge type override unset).

This file is intentionally separate so ongoing runs using
`evaluate_jetclass_hlt_teacher_baseline.py` are unaffected.
"""

from __future__ import annotations

import numpy as np

import evaluate_jetclass_hlt_teacher_baseline as base


def _allowed_merge_and_output_type_type_agnostic(ti: int, tj: int):
    # Allow any pair except both-unknown. Output type is chosen by dominant energy
    # in merge_two_tokens when override is None.
    if ti == base.TYPE_UNK and tj == base.TYPE_UNK:
        return False, None
    return True, None


def _get_type_config_type_agnostic():
    cfg = base.get_type_config()

    # Flatten pair-dependent merge behavior so merging is type-agnostic.
    # Strength still controlled by --merge_prob_scale at runtime.
    r = np.full_like(cfg["merge_radius"], 0.018, dtype=np.float64)
    p = np.full_like(cfg["merge_prob"], 0.28, dtype=np.float64)

    # Keep unknown interactions conservative.
    unk = int(base.TYPE_UNK)
    r[unk, :] = np.minimum(r[unk, :], 0.010)
    r[:, unk] = np.minimum(r[:, unk], 0.010)
    p[unk, :] = np.minimum(p[unk, :], 0.08)
    p[:, unk] = np.minimum(p[:, unk], 0.08)

    cfg["merge_radius"] = r
    cfg["merge_prob"] = p
    return cfg


def main() -> None:
    # Monkey-patch only for this process.
    original_get_type_config = base.get_type_config

    def _get_type_config_type_agnostic_no_recursion():
        cfg = original_get_type_config()

        r = np.full_like(cfg["merge_radius"], 0.018, dtype=np.float64)
        p = np.full_like(cfg["merge_prob"], 0.28, dtype=np.float64)

        unk = int(base.TYPE_UNK)
        r[unk, :] = np.minimum(r[unk, :], 0.010)
        r[:, unk] = np.minimum(r[:, unk], 0.010)
        p[unk, :] = np.minimum(p[unk, :], 0.08)
        p[:, unk] = np.minimum(p[:, unk], 0.08)

        cfg["merge_radius"] = r
        cfg["merge_prob"] = p
        return cfg

    base.allowed_merge_and_output_type = _allowed_merge_and_output_type_type_agnostic
    base.get_type_config = _get_type_config_type_agnostic_no_recursion
    args = base.parse_args()
    base.run_experiment(args)


if __name__ == "__main__":
    main()

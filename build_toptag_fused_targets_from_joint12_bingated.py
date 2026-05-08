#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build TPR-agnostic fused soft targets for top-tagging from joint12 bin-gated output.

Input:
  - bin_gated_scores.npz produced by analyze_hlt_joint31_bin_gated_fusion.py

Output:
  - fused_targets_train_val_test.npz
  - fused_targets_metadata.json

The output includes:
  - per-family fused targets (bin/global)
  - overall fused targets (default sourced from selected family, reduced over TPR keys)
  - aliases for both {fit,ref,test} and {train,val,test}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _reduce_stack(arrs: List[np.ndarray], mode: str) -> np.ndarray:
    x = np.stack(arrs, axis=0).astype(np.float64)
    mode_l = str(mode).strip().lower()
    if mode_l == "median":
        out = np.median(x, axis=0)
    else:
        out = np.mean(x, axis=0)
    return np.asarray(out, dtype=np.float32)


def _suffix_sort_key(s: str) -> Tuple[int, float, str]:
    """
    Prefer numeric ordering for suffixes like 'tpr0p500', then fallback to lexical.
    """
    m = re.match(r"^tpr(\d+)p(\d+)$", str(s))
    if m is not None:
        whole = int(m.group(1))
        frac = int(m.group(2))
        scale = 10 ** len(m.group(2))
        return (0, float(whole) + float(frac) / float(scale), str(s))
    try:
        return (1, float(s), str(s))
    except Exception:
        return (2, 0.0, str(s))


def _collect_family_split(
    arr: np.lib.npyio.NpzFile,
    family: str,
    split: str,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    # Accept flexible suffixes, e.g. tpr0p500, 0, 1, etc.
    pat = re.compile(rf"^fused_{re.escape(family)}_{re.escape(split)}_(.+)$")
    keys = []
    out: Dict[str, np.ndarray] = {}
    for k in arr.files:
        m = pat.match(k)
        if m is None:
            continue
        suffix = m.group(1)
        keys.append(k)
        out[suffix] = np.asarray(arr[k], dtype=np.float32).reshape(-1)
    keys.sort()
    return keys, out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build top-tagging fused targets from joint12 bin-gated scores")
    ap.add_argument("--scores_npz", type=Path, required=True, help="Path to bin_gated_scores.npz")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    ap.add_argument(
        "--overall_family",
        type=str,
        default="bin",
        choices=["bin", "global"],
        help="Which fused family to use for probs_fused_overall_*.",
    )
    ap.add_argument(
        "--reduction",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Reduction over available TPR-specific fused scores.",
    )
    args = ap.parse_args()

    scores_npz = args.scores_npz.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.load(scores_npz)
    required = ["labels_fit", "labels_ref", "labels_test"]
    for k in required:
        if k not in arr:
            raise KeyError(f"Missing required key `{k}` in {scores_npz}")

    labels_fit = np.asarray(arr["labels_fit"], dtype=np.float32).reshape(-1)
    labels_ref = np.asarray(arr["labels_ref"], dtype=np.float32).reshape(-1)
    labels_test = np.asarray(arr["labels_test"], dtype=np.float32).reshape(-1)

    built: Dict[str, np.ndarray] = {}
    meta: Dict[str, object] = {
        "scores_npz": str(scores_npz),
        "overall_family": str(args.overall_family),
        "reduction": str(args.reduction),
        "available_tpr_suffixes": {},
        "source_keys": {},
    }

    # Build per-family reductions.
    for family in ("bin", "global"):
        fam_suffixes = None
        fam_keys: Dict[str, List[str]] = {}
        for split_src, split_alias in (("fit", "fit"), ("cal", "ref"), ("test", "test")):
            keys, by_suffix = _collect_family_split(arr, family=family, split=split_src)
            fam_keys[split_src] = keys
            if len(by_suffix) == 0:
                continue
            suffixes = sorted(by_suffix.keys(), key=_suffix_sort_key)
            if fam_suffixes is None:
                fam_suffixes = suffixes
            else:
                fam_suffixes = sorted(
                    set(fam_suffixes).intersection(set(suffixes)),
                    key=_suffix_sort_key,
                )
            if len(fam_suffixes) == 0:
                raise RuntimeError(f"No common TPR suffixes found for family={family} split={split_src}.")
            reduced = _reduce_stack([by_suffix[s] for s in fam_suffixes], mode=args.reduction)
            built[f"probs_fused_{family}_{split_alias}"] = reduced
            for s in fam_suffixes:
                built[f"probs_fused_{family}_{split_alias}_tpr{s}"] = by_suffix[s].astype(np.float32)

        meta["available_tpr_suffixes"][family] = fam_suffixes or []
        meta["source_keys"][family] = fam_keys

    need_keys = [
        "probs_fused_bin_fit",
        "probs_fused_bin_ref",
        "probs_fused_bin_test",
        "probs_fused_global_fit",
        "probs_fused_global_ref",
        "probs_fused_global_test",
    ]
    for k in need_keys:
        if k not in built:
            raise KeyError(
                f"Could not build `{k}` from {scores_npz}. "
                f"Check that fused_bin_* and fused_global_* keys are present."
            )

    fam = str(args.overall_family).strip().lower()
    built["probs_fused_overall_fit"] = built[f"probs_fused_{fam}_fit"]
    built["probs_fused_overall_ref"] = built[f"probs_fused_{fam}_ref"]
    built["probs_fused_overall_test"] = built[f"probs_fused_{fam}_test"]

    # train/val aliases
    built["probs_fused_bin_train"] = built["probs_fused_bin_fit"]
    built["probs_fused_bin_val"] = built["probs_fused_bin_ref"]
    built["probs_fused_global_train"] = built["probs_fused_global_fit"]
    built["probs_fused_global_val"] = built["probs_fused_global_ref"]
    built["probs_fused_overall_train"] = built["probs_fused_overall_fit"]
    built["probs_fused_overall_val"] = built["probs_fused_overall_ref"]

    out_npz = out_dir / "fused_targets_train_val_test.npz"
    np.savez_compressed(
        out_npz,
        labels_fit=labels_fit,
        labels_ref=labels_ref,
        labels_test=labels_test,
        labels_train=labels_fit,
        labels_val=labels_ref,
        labels_eval=labels_test,
        **built,
        **{
            k: np.asarray(arr[k])
            for k in ("idx_fit", "idx_ref")
            if k in arr
        },
    )

    meta.update(
        {
            "n_fit": int(labels_fit.shape[0]),
            "n_ref": int(labels_ref.shape[0]),
            "n_test": int(labels_test.shape[0]),
            "output_npz": str(out_npz),
        }
    )
    out_meta = out_dir / "fused_targets_metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2))

    print("============================================================")
    print("Built top-tagging fused targets from joint12 bin-gated scores")
    print("============================================================")
    print(f"Scores NPZ: {scores_npz}")
    print(f"Out dir:    {out_dir}")
    print(f"Overall:    family={fam}, reduction={args.reduction}")
    print(f"Saved NPZ:  {out_npz}")
    print(f"Saved meta: {out_meta}")


if __name__ == "__main__":
    main()

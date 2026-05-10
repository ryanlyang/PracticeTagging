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


def _parse_csv_list(s: str) -> List[str]:
    out: List[str] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_float_list(s: str, default: List[float]) -> List[float]:
    out: List[float] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except Exception:
            continue
    if not out:
        out = [float(x) for x in default]
    return out


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(s)).strip("_")


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
    ap.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Optional bin_gated_report.json. Required when building fixed-model mapping targets.",
    )
    ap.add_argument(
        "--fixed_models",
        type=str,
        default="",
        help="Comma-separated model names for fixed mapping (e.g. joint_delta,dual_m17_antioverlap,...).",
    )
    ap.add_argument(
        "--fixed_prefix",
        type=str,
        default="probs_fixedmap",
        help="Key prefix for fixed-map targets.",
    )
    ap.add_argument(
        "--fixed_reduction",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Reduction for fixed-map aggregate target over selected models.",
    )
    ap.add_argument(
        "--fixed_strategy",
        type=str,
        default="mean",
        choices=["mean", "greedy_global"],
        help="How to build fixed-map aggregate: plain reduction or greedy global blend over fixed models.",
    )
    ap.add_argument(
        "--fixed_anchor_model",
        type=str,
        default="",
        help="Anchor model for greedy_global fixed-map strategy. Default: first model in --fixed_models.",
    )
    ap.add_argument(
        "--fixed_target_tprs",
        type=str,
        default="0.50,0.30",
        help="Comma-separated target TPRs for greedy_global fixed-map strategy.",
    )
    ap.add_argument(
        "--fixed_tpr_reduction",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Reduction over per-TPR greedy blends for fixed-map strategy.",
    )
    ap.add_argument(
        "--fixed_calibration",
        type=str,
        default="iso",
        choices=["raw", "iso", "platt"],
        help="Calibration mode before greedy blending in fixed-map strategy.",
    )
    ap.add_argument(
        "--fixed_w_step",
        type=float,
        default=0.005,
        help="Blend weight scan step for greedy_global fixed-map strategy.",
    )
    ap.add_argument(
        "--fixed_max_add",
        type=int,
        default=6,
        help="Max greedy additions for fixed-map strategy.",
    )
    ap.add_argument(
        "--fixed_min_improve",
        type=float,
        default=2e-7,
        help="Minimum fit-FPR improvement for accepting a greedy blend step.",
    )
    ap.add_argument(
        "--fixed_head_select_mode",
        type=str,
        default="best_val_fpr",
        choices=["first", "best_val_auc", "best_val_fpr"],
        help="How to choose head keys for models with multiple output heads.",
    )
    ap.add_argument(
        "--fixed_head_select_tpr",
        type=float,
        default=0.50,
        help="TPR used when fixed_head_select_mode=best_val_fpr.",
    )
    ap.add_argument(
        "--fixed_include_per_model",
        action="store_true",
        help="Also emit per-model targets for selected fixed models.",
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
        "fixed_map": {},
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

    fixed_models = _parse_csv_list(args.fixed_models)
    if len(fixed_models) > 0:
        if args.report_json is None:
            raise ValueError("--report_json is required when --fixed_models is provided.")
        report_json = args.report_json.expanduser().resolve()
        if not report_json.is_file():
            raise FileNotFoundError(f"Missing report json: {report_json}")

        report = json.loads(report_json.read_text())
        fusion_json = report.get("fusion_json", "")
        if not isinstance(fusion_json, str) or len(fusion_json.strip()) == 0:
            raise KeyError(f"Missing fusion_json in report: {report_json}")

        # Reuse score loading logic from analyzer to avoid key drift.
        import analyze_hlt_joint31_bin_gated_fusion as ana

        y_val, y_test, scores_val, scores_test, used_paths, skipped_models = ana._load_required_scores(
            Path(fusion_json).expanduser().resolve(),
            required_models=fixed_models,
            head_select_mode=str(args.fixed_head_select_mode),
            head_select_tpr=float(args.fixed_head_select_tpr),
        )

        idx_fit = np.asarray(arr["idx_fit"], dtype=np.int64).reshape(-1) if "idx_fit" in arr else None
        idx_ref = np.asarray(arr["idx_ref"], dtype=np.int64).reshape(-1) if "idx_ref" in arr else None
        if idx_fit is None or idx_ref is None:
            raise KeyError(
                f"Missing idx_fit/idx_ref in {scores_npz}; cannot build fixed-map train/val targets."
            )

        if not np.array_equal(np.asarray(y_val, dtype=np.float32).reshape(-1), labels_ref):
            raise RuntimeError(
                "Validation labels mismatch between report/fusion scores and bin_gated_scores.npz. "
                f"report={report_json}"
            )
        if not np.array_equal(np.asarray(y_test, dtype=np.float32).reshape(-1), labels_test):
            raise RuntimeError(
                "Test labels mismatch between report/fusion scores and bin_gated_scores.npz. "
                f"report={report_json}"
            )

        y_fit = np.asarray(y_val, dtype=np.float32).reshape(-1)[idx_fit]
        y_ref = np.asarray(y_val, dtype=np.float32).reshape(-1)[idx_ref]
        y_tst = np.asarray(y_test, dtype=np.float32).reshape(-1)

        fit_map_raw: Dict[str, np.ndarray] = {}
        ref_map_raw: Dict[str, np.ndarray] = {}
        test_map_raw: Dict[str, np.ndarray] = {}
        per_model_meta: Dict[str, str] = {}

        for m in fixed_models:
            if m not in scores_val or m not in scores_test:
                why = skipped_models.get(m, "missing_from_score_maps")
                raise KeyError(f"Fixed model `{m}` unavailable: {why}")
            s_fit = np.asarray(scores_val[m], dtype=np.float32).reshape(-1)[idx_fit]
            s_ref = np.asarray(scores_val[m], dtype=np.float32).reshape(-1)[idx_ref]
            s_test = np.asarray(scores_test[m], dtype=np.float32).reshape(-1)
            fit_map_raw[m] = s_fit
            ref_map_raw[m] = s_ref
            test_map_raw[m] = s_test

            if args.fixed_include_per_model:
                km = _sanitize_name(m)
                built[f"{args.fixed_prefix}_{km}_fit"] = s_fit
                built[f"{args.fixed_prefix}_{km}_ref"] = s_ref
                built[f"{args.fixed_prefix}_{km}_test"] = s_test
                built[f"{args.fixed_prefix}_{km}_train"] = s_fit
                built[f"{args.fixed_prefix}_{km}_val"] = s_ref

            per_model_meta[m] = used_paths.get(m, "")

        strategy = str(args.fixed_strategy).strip().lower()
        if strategy == "mean":
            fit_list = [fit_map_raw[m] for m in fixed_models]
            ref_list = [ref_map_raw[m] for m in fixed_models]
            test_list = [test_map_raw[m] for m in fixed_models]
            agg_fit = _reduce_stack(fit_list, mode=args.fixed_reduction)
            agg_ref = _reduce_stack(ref_list, mode=args.fixed_reduction)
            agg_test = _reduce_stack(test_list, mode=args.fixed_reduction)
            fixed_details: Dict[str, object] = {
                "strategy": strategy,
                "reduction": str(args.fixed_reduction),
            }
        elif strategy == "greedy_global":
            target_tprs = _parse_float_list(str(args.fixed_target_tprs), [0.50, 0.30])
            anchor_model = str(args.fixed_anchor_model).strip() or str(fixed_models[0])
            if anchor_model not in fixed_models:
                raise ValueError(
                    f"fixed_anchor_model={anchor_model} must be in fixed_models={fixed_models}"
                )

            blend_fit_list: List[np.ndarray] = []
            blend_ref_list: List[np.ndarray] = []
            blend_test_list: List[np.ndarray] = []
            per_tpr_logs: Dict[str, List[Dict[str, object]]] = {}

            for tpr in target_tprs:
                fit_map = {}
                cal_map = {}
                test_map = {}
                for m in fixed_models:
                    sf, sr, st = ana.calibrate_scores(
                        y_fit=y_fit,
                        s_fit=np.asarray(fit_map_raw[m], dtype=np.float64),
                        s_cal=np.asarray(ref_map_raw[m], dtype=np.float64),
                        s_test=np.asarray(test_map_raw[m], dtype=np.float64),
                        mode=str(args.fixed_calibration),
                    )
                    fit_map[m] = sf
                    cal_map[m] = sr
                    test_map[m] = st

                s_fit_b, s_ref_b, s_test_b, rows_b = ana._greedy_global_blend(
                    y_fit=np.asarray(y_fit, dtype=np.float32),
                    s_fit_map=fit_map,
                    s_cal_map=cal_map,
                    s_test_map=test_map,
                    anchor_model=anchor_model,
                    candidates=list(fixed_models),
                    target_tpr=float(tpr),
                    max_add=int(min(max(1, int(args.fixed_max_add)), len(fixed_models) - 1)),
                    w_step=float(args.fixed_w_step),
                    min_improve=float(args.fixed_min_improve),
                )
                k = f"tpr{tpr:.3f}".replace(".", "p")
                built[f"{args.fixed_prefix}_fit_{k}"] = np.asarray(s_fit_b, dtype=np.float32)
                built[f"{args.fixed_prefix}_ref_{k}"] = np.asarray(s_ref_b, dtype=np.float32)
                built[f"{args.fixed_prefix}_test_{k}"] = np.asarray(s_test_b, dtype=np.float32)
                per_tpr_logs[k] = rows_b
                blend_fit_list.append(np.asarray(s_fit_b, dtype=np.float32))
                blend_ref_list.append(np.asarray(s_ref_b, dtype=np.float32))
                blend_test_list.append(np.asarray(s_test_b, dtype=np.float32))

            agg_fit = _reduce_stack(blend_fit_list, mode=args.fixed_tpr_reduction)
            agg_ref = _reduce_stack(blend_ref_list, mode=args.fixed_tpr_reduction)
            agg_test = _reduce_stack(blend_test_list, mode=args.fixed_tpr_reduction)

            if not np.array_equal(np.asarray(y_ref, dtype=np.float32).reshape(-1), labels_ref):
                raise RuntimeError("Internal fixed-map ref labels drifted unexpectedly.")
            if not np.array_equal(np.asarray(y_tst, dtype=np.float32).reshape(-1), labels_test):
                raise RuntimeError("Internal fixed-map test labels drifted unexpectedly.")

            fixed_details = {
                "strategy": strategy,
                "anchor_model": anchor_model,
                "target_tprs": [float(x) for x in target_tprs],
                "tpr_reduction": str(args.fixed_tpr_reduction),
                "calibration": str(args.fixed_calibration),
                "w_step": float(args.fixed_w_step),
                "max_add": int(args.fixed_max_add),
                "min_improve": float(args.fixed_min_improve),
                "head_select_mode": str(args.fixed_head_select_mode),
                "head_select_tpr": float(args.fixed_head_select_tpr),
                "per_tpr_logs": per_tpr_logs,
            }
        else:
            raise ValueError(f"Unknown fixed_strategy: {args.fixed_strategy}")

        pref = str(args.fixed_prefix).strip()
        built[f"{pref}_fit"] = agg_fit
        built[f"{pref}_ref"] = agg_ref
        built[f"{pref}_test"] = agg_test
        built[f"{pref}_train"] = agg_fit
        built[f"{pref}_val"] = agg_ref

        meta["fixed_map"] = {
            "enabled": True,
            "report_json": str(report_json),
            "fusion_json": str(Path(fusion_json).expanduser().resolve()),
            "models": fixed_models,
            "prefix": pref,
            "reduction": str(args.fixed_reduction),
            "include_per_model": bool(args.fixed_include_per_model),
            "used_paths": per_model_meta,
            "strategy_details": fixed_details,
        }

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
    if len(fixed_models) > 0:
        print(
            "Fixed-map targets: "
            f"prefix={str(args.fixed_prefix).strip()}, models={','.join(fixed_models)}, "
            f"strategy={str(args.fixed_strategy).strip().lower()}, reduction={args.fixed_reduction}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze9 bin-gated-only fusion for the finished weighted 5m1m1m RecoTeacher set.

This adds the completed m5 joint run to the Analyze8 bin-gated candidate set:
  - m4_corrected
  - m5_joint
  - m9_mid
  - m9_high
  - m12_dual
  - m15_mid_dual
  - m15_high_dual
  - m16_dual
  - m17_dual
plus HLT.

The underlying bin-gated method still has to build a global blend internally,
because bin-local gates are updates on top of that seed. This wrapper filters
the saved user-facing report/summary to only the final bin_gated_blend rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


BASE_DIR = "checkpoints/reco_teacher_joint_fusion_6model_150k75k150k"


DEFAULT_CANDIDATES = (
    "corrected_s01,"
    "joint_s01,"
    "offdrop_mid,"
    "offdrop_high,"
    "dual_m12_noscale,"
    "dual_m15_offdrop_mid,"
    "dual_m15_offdrop_high,"
    "dual_m16_topk60,"
    "dual_m17_antioverlap,"
    "hlt"
)


def _resolve(raw: str) -> Path:
    return Path(str(raw)).expanduser().resolve()


def _first_key(z: np.lib.npyio.NpzFile, keys: List[str]) -> str:
    for key in keys:
        if key in z:
            return key
    raise KeyError(f"None of keys found: {keys}; available={list(z.keys())}")


def _assert_same_labels(z: np.lib.npyio.NpzFile, y_val: np.ndarray, y_test: np.ndarray, label: str) -> None:
    yv = np.asarray(z["labels_val"], dtype=np.float32)
    yt = np.asarray(z["labels_test"], dtype=np.float32)
    if not np.array_equal(y_val, yv):
        raise RuntimeError(f"Validation labels mismatch for {label}")
    if not np.array_equal(y_test, yt):
        raise RuntimeError(f"Test labels mismatch for {label}")


def _load_hlt_override(
    step1_ref_npz: str,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None, str]:
    if not str(step1_ref_npz).strip():
        return None, None, ""
    path = _resolve(step1_ref_npz)
    if not path.exists():
        return None, None, ""
    z = np.load(path)
    if "labels_val" not in z or "labels_test" not in z:
        return None, None, ""
    if "preds_hlt_val" not in z or "preds_hlt_test" not in z:
        return None, None, ""
    yv = np.asarray(z["labels_val"], dtype=np.float32)
    yt = np.asarray(z["labels_test"], dtype=np.float32)
    if not np.array_equal(y_val, yv) or not np.array_equal(y_test, yt):
        print(f"[prep] STEP1 labels mismatch; keeping HLT from reference model: {path}", flush=True)
        return None, None, ""
    print(f"[prep] using STEP1 HLT override from: {path}", flush=True)
    return (
        np.asarray(z["preds_hlt_val"], dtype=np.float64),
        np.asarray(z["preds_hlt_test"], dtype=np.float64),
        str(path),
    )


def _write_fusion_json(args: argparse.Namespace, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    m9mid_npz = _resolve(args.m9mid_npz)
    if not m9mid_npz.exists():
        raise FileNotFoundError(f"Missing m9mid npz: {m9mid_npz}")
    z_mid = np.load(m9mid_npz)
    y_val = np.asarray(z_mid["labels_val"], dtype=np.float32)
    y_test = np.asarray(z_mid["labels_test"], dtype=np.float32)

    required = {
        "m4": _resolve(args.m4_npz),
        "m5": _resolve(args.m5_npz),
        "m9mid": m9mid_npz,
        "m9high": _resolve(args.m9high_npz),
        "m12": _resolve(args.m12_npz),
        "m15mid": _resolve(args.m15mid_npz),
        "m15high": _resolve(args.m15high_npz),
        "m16": _resolve(args.m16_npz),
        "m17": _resolve(args.m17_npz),
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required score file for {label}: {path}")
        if label != "m9mid":
            _assert_same_labels(np.load(path), y_val, y_test, label)

    k_anchor_val = _first_key(
        z_mid,
        ["preds_residual_joint_val", "preds_residual_frozen_val", "preds_reco_teacher_val", "preds_hlt_val"],
    )
    k_anchor_test = _first_key(
        z_mid,
        ["preds_residual_joint_test", "preds_residual_frozen_test", "preds_reco_teacher_test", "preds_hlt_test"],
    )
    k_hlt_val = _first_key(z_mid, ["preds_hlt_val"])
    k_hlt_test = _first_key(z_mid, ["preds_hlt_test"])

    hlt_val = np.asarray(z_mid[k_hlt_val], dtype=np.float64)
    hlt_test = np.asarray(z_mid[k_hlt_test], dtype=np.float64)
    hlt_source = str(m9mid_npz)
    hlt_override_val, hlt_override_test, hlt_override_source = _load_hlt_override(
        args.step1_ref_npz,
        y_val=y_val,
        y_test=y_test,
    )
    if hlt_override_val is not None and hlt_override_test is not None:
        hlt_val = hlt_override_val
        hlt_test = hlt_override_test
        hlt_source = hlt_override_source

    compat_npz = out_dir / "joint_delta_compat_from_analyze9_anchor.npz"
    np.savez_compressed(
        compat_npz,
        labels_val=y_val.astype(np.float32),
        labels_test=y_test.astype(np.float32),
        preds_hlt_val=hlt_val.astype(np.float64),
        preds_hlt_test=hlt_test.astype(np.float64),
        preds_joint_val=np.asarray(z_mid[k_anchor_val], dtype=np.float64),
        preds_joint_test=np.asarray(z_mid[k_anchor_test], dtype=np.float64),
    )
    print(
        "[prep] wrote compatibility NPZ: "
        f"{compat_npz} (anchor from m9mid val={k_anchor_val}, test={k_anchor_test})",
        flush=True,
    )

    score_files = {
        "joint_delta": str(compat_npz.resolve()),
        "hlt": str(compat_npz.resolve()),
        "corrected_s01": str(required["m4"].resolve()),
        "joint_s01": str(required["m5"].resolve()),
        "offdrop_mid": str(required["m9mid"].resolve()),
        "offdrop_high": str(required["m9high"].resolve()),
        "dual_m12_noscale": str(required["m12"].resolve()),
        "dual_m15_offdrop_mid": str(required["m15mid"].resolve()),
        "dual_m15_offdrop_high": str(required["m15high"].resolve()),
        "dual_m16_topk60": str(required["m16"].resolve()),
        "dual_m17_antioverlap": str(required["m17"].resolve()),
    }
    fusion_json = out_dir / "fusion_hlt_joint9_finished_weighted_5m1m1m.json"
    fusion = {
        "run_dirs": {"score_files": score_files},
        "analyze9_meta": {
            "hlt_source": hlt_source,
            "anchor_source": str(m9mid_npz),
            "anchor_val_key": k_anchor_val,
            "anchor_test_key": k_anchor_test,
            "candidate_models": args.candidate_models,
        },
    }
    fusion_json.write_text(json.dumps(fusion, indent=2))
    print(f"[prep] wrote fusion JSON: {fusion_json}", flush=True)
    return compat_npz, fusion_json


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        if not keys:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def _filter_bin_gated_outputs(out_dir: Path, full_report_json: Path, report_json: Path) -> None:
    summary_csv = out_dir / "bin_gated_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Expected bin-gated summary not found: {summary_csv}")

    rows = _read_csv(summary_csv)
    keep = [r for r in rows if str(r.get("method", "")) == "bin_gated_blend"]
    if not keep:
        raise RuntimeError(f"No bin_gated_blend rows found in {summary_csv}")

    full_summary_csv = out_dir / "bin_gated_full_summary.csv"
    blend_summary_csv = out_dir / "bin_gated_blend_only_summary.csv"
    _write_csv(full_summary_csv, rows)
    _write_csv(blend_summary_csv, keep)
    _write_csv(summary_csv, keep)

    full_report = json.loads(full_report_json.read_text()) if full_report_json.exists() else {}
    by_tpr = {}
    for key, obj in dict(full_report.get("by_tpr", {})).items():
        if not isinstance(obj, dict):
            continue
        metrics = obj.get("metrics", {})
        by_tpr[key] = {
            "target_tpr": obj.get("target_tpr"),
            "candidates": obj.get("candidates", []),
            "metrics": {
                "bin_gated_blend": metrics.get("bin_gated_blend", {}),
            },
            "bins": obj.get("bins", []),
        }

    report = {
        "source_full_report": str(full_report_json.resolve()),
        "out_dir": str(out_dir.resolve()),
        "settings": full_report.get("settings", {}),
        "score_files_used": full_report.get("score_files_used", {}),
        "target_tprs": full_report.get("target_tprs", []),
        "by_tpr": by_tpr,
        "files": {
            "blend_only_summary_csv": str(blend_summary_csv.resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "full_summary_csv": str(full_summary_csv.resolve()),
            "update_log_csv": str((out_dir / "bin_gated_update_log.csv").resolve()),
            "scores_npz": str((out_dir / "bin_gated_scores.npz").resolve()),
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2))

    print("=" * 72)
    print("Analyze9 bin_gated_blend only")
    print("=" * 72)
    for row in keep:
        print(
            f"TPR={float(row['target_tpr']):.3f} "
            f"AUC_test={float(row['auc_test']):.6f} "
            f"FPR_test={float(row['fpr_test']):.6f} "
            f"(cal={float(row['fpr_cal']):.6f})"
        )
    print(f"Saved blend-only report:  {report_json}")
    print(f"Saved blend-only summary: {blend_summary_csv}")
    print(f"Saved full engine summary: {full_summary_csv}")
    print("=" * 72)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Analyze9 bin-gated-only fusion for finished weighted 5m1m1m runs.")
    ap.add_argument(
        "--m4_npz",
        default=(
            f"{BASE_DIR}/model4_recoteacher_s01_corrected_weighted_5m1m1m/"
            "model4_recoteacher_s01_corrected_weighted_5m1m1m_seed0/stageA_only_scores.npz"
        ),
    )
    ap.add_argument(
        "--m5_npz",
        default=(
            f"{BASE_DIR}/model5_joint_s01_full_weighted_5m1m1m/"
            "model5_joint_s01_full_weighted_5m1m1m_seed0/fusion_scores_val_test.npz"
        ),
    )
    ap.add_argument(
        "--m9mid_npz",
        default=(
            f"{BASE_DIR}/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m/"
            "model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m_seed0/stageA_residual_scores.npz"
        ),
    )
    ap.add_argument(
        "--m9high_npz",
        default=(
            f"{BASE_DIR}/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m/"
            "model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m_seed0/stageA_residual_scores.npz"
        ),
    )
    ap.add_argument(
        "--m12_npz",
        default=(
            f"{BASE_DIR}/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_from_recoonly/"
            "model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
    )
    ap.add_argument(
        "--m15mid_npz",
        default=(
            f"{BASE_DIR}/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_from_recoonly/"
            "model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
    )
    ap.add_argument(
        "--m15high_npz",
        default=(
            f"{BASE_DIR}/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_from_recoonly/"
            "model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
    )
    ap.add_argument(
        "--m16_npz",
        default=(
            f"{BASE_DIR}/model16_dualreco_dualview_topk60_weighted_5m1m1m_from_recoonly/"
            "model16_dualreco_dualview_topk60_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
    )
    ap.add_argument(
        "--m17_npz",
        default=(
            f"{BASE_DIR}/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_from_recoonly/"
            "model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
    )
    ap.add_argument(
        "--step1_ref_npz",
        default=(
            f"{BASE_DIR}/teacher_hlt_only_weighted_5m1m1m/"
            "teacher_hlt_only_weighted_5m1m1m_seed0/results_step1_teacher_baseline.npz"
        ),
    )
    ap.add_argument("--target_tprs", default="0.50,0.30")
    ap.add_argument("--anchor_model", default="offdrop_mid")
    ap.add_argument("--candidate_models", default=DEFAULT_CANDIDATES)
    ap.add_argument("--selection_mode", default="valsel", choices=["split", "valsel"])
    ap.add_argument("--calibration", default="iso", choices=["raw", "iso", "platt"])
    ap.add_argument("--head_select_mode", default="best_val_fpr", choices=["first", "best_val_auc", "best_val_fpr"])
    ap.add_argument("--head_select_tpr", type=float, default=0.50)
    ap.add_argument("--router_cal_frac", type=float, default=0.40)
    ap.add_argument("--score_band_edges", default="0.0,0.8,0.9,1.0")
    ap.add_argument("--dist_near_cut", type=float, default=0.0384)
    ap.add_argument("--dist_mid_low", type=float, default=0.06285)
    ap.add_argument("--dist_mid_high", type=float, default=0.07386)
    ap.add_argument("--global_max_add", type=int, default=8)
    ap.add_argument("--bin_max_add", type=int, default=6)
    ap.add_argument("--w_step", type=float, default=0.0025)
    ap.add_argument("--min_bin_fit", type=int, default=2000)
    ap.add_argument("--min_global_improve", type=float, default=2e-7)
    ap.add_argument("--min_bin_improve", type=float, default=1e-6)
    ap.add_argument("--expand_prepost_variants", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin_gated_script", default="analyze_hlt_joint31_bin_gated_fusion.py")
    ap.add_argument(
        "--out_dir",
        default=f"{BASE_DIR}/analyze9_finished_weighted_5m1m1m/bin_gated",
    )
    ap.add_argument("--report_json", default="")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = _resolve(args.out_dir)
    report_json = _resolve(args.report_json) if str(args.report_json).strip() else out_dir / "bin_gated_report.json"
    full_report_json = out_dir / "bin_gated_full_report.json"

    _compat_npz, fusion_json = _write_fusion_json(args, out_dir=out_dir)

    script = Path(args.bin_gated_script).expanduser()
    if not script.is_absolute():
        script = Path(__file__).resolve().parent / script
    script = script.resolve()
    if not script.exists():
        raise FileNotFoundError(f"Missing bin-gated script: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--fusion_json",
        str(fusion_json),
        "--target_tprs",
        str(args.target_tprs),
        "--anchor_model",
        str(args.anchor_model),
        "--selection_mode",
        str(args.selection_mode),
        "--candidate_models_all",
        str(args.candidate_models),
        "--expand_prepost_variants",
        str(int(args.expand_prepost_variants)),
        "--router_cal_frac",
        str(float(args.router_cal_frac)),
        "--seed",
        str(int(args.seed)),
        "--calibration",
        str(args.calibration),
        "--head_select_mode",
        str(args.head_select_mode),
        "--head_select_tpr",
        str(float(args.head_select_tpr)),
        "--score_band_edges",
        str(args.score_band_edges),
        "--dist_near_cut",
        str(float(args.dist_near_cut)),
        "--dist_mid_low",
        str(float(args.dist_mid_low)),
        "--dist_mid_high",
        str(float(args.dist_mid_high)),
        "--global_max_add",
        str(int(args.global_max_add)),
        "--bin_max_add",
        str(int(args.bin_max_add)),
        "--w_step",
        str(float(args.w_step)),
        "--min_bin_fit",
        str(int(args.min_bin_fit)),
        "--min_global_improve",
        str(float(args.min_global_improve)),
        "--min_bin_improve",
        str(float(args.min_bin_improve)),
        "--out_dir",
        str(out_dir),
        "--report_json",
        str(full_report_json),
    ]

    print("=" * 72)
    print("Analyze9 Bin-Gated Fusion")
    print("=" * 72)
    print(f"Fusion JSON: {fusion_json}")
    print(f"Out dir:     {out_dir}")
    print(f"Candidates:  {args.candidate_models}")
    print(f"Report:      {report_json} (bin_gated_blend only)")
    print(f"Full report: {full_report_json}")
    print("=" * 72)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    engine_stdout = out_dir / "bin_gated_engine_stdout.log"
    engine_stderr = out_dir / "bin_gated_engine_stderr.log"
    engine_stdout.write_text(result.stdout or "")
    engine_stderr.write_text(result.stderr or "")
    if result.returncode != 0:
        print(f"ERROR: bin-gated engine failed with exit code {result.returncode}", file=sys.stderr)
        if result.stdout:
            print("--- engine stdout tail ---", file=sys.stderr)
            print(result.stdout[-4000:], file=sys.stderr)
        if result.stderr:
            print("--- engine stderr tail ---", file=sys.stderr)
            print(result.stderr[-4000:], file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    _filter_bin_gated_outputs(out_dir=out_dir, full_report_json=full_report_json, report_json=report_json)


if __name__ == "__main__":
    main()

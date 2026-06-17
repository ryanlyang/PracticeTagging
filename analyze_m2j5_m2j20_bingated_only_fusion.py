#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bin-gated-only fusion for the two M2 5m1m1m weighted runs.

Inputs:
  - HLT baseline
  - m2 delta005 Stage2
  - m2 delta005 Joint
  - m2 delta020 Stage2
  - m2 delta020 Joint

The underlying bin-gated engine reports anchor/global/HLT diagnostics as well,
but this wrapper filters the user-facing report and summary to bin_gated_blend
only, matching the Analyze9 bin-gated-only behavior.
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
DEFAULT_CANDIDATES = "joint_delta_stage2,joint_delta_joint,hlt,joint_delta020_stage2,joint_delta020_joint"


def _resolve(raw: str) -> Path:
    return Path(str(raw)).expanduser().resolve()


def _assert_score_file(path: Path, label: str) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"Missing required score file for {label}: {path}")
    z = np.load(path)
    required = [
        "labels_val",
        "labels_test",
        "preds_hlt_val",
        "preds_hlt_test",
        "preds_stage2_val",
        "preds_stage2_test",
        "preds_joint_val",
        "preds_joint_test",
    ]
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"{label} score file is missing keys {missing}: {path}")
    return z


def _assert_same_labels(z_ref: np.lib.npyio.NpzFile, z_other: np.lib.npyio.NpzFile, label: str) -> None:
    yv_ref = np.asarray(z_ref["labels_val"], dtype=np.float32)
    yt_ref = np.asarray(z_ref["labels_test"], dtype=np.float32)
    yv = np.asarray(z_other["labels_val"], dtype=np.float32)
    yt = np.asarray(z_other["labels_test"], dtype=np.float32)
    if not np.array_equal(yv_ref, yv):
        raise RuntimeError(f"Validation labels mismatch for {label}")
    if not np.array_equal(yt_ref, yt):
        raise RuntimeError(f"Test labels mismatch for {label}")


def _step1_ref_is_compatible(step1_ref_npz: str, z_ref: np.lib.npyio.NpzFile) -> bool:
    if not str(step1_ref_npz).strip():
        return False
    path = _resolve(step1_ref_npz)
    if not path.exists():
        return False
    z = np.load(path)
    needed = ["labels_val", "labels_test", "preds_hlt_val", "preds_hlt_test"]
    if any(k not in z for k in needed):
        return False
    return (
        np.array_equal(np.asarray(z_ref["labels_val"], dtype=np.float32), np.asarray(z["labels_val"], dtype=np.float32))
        and np.array_equal(np.asarray(z_ref["labels_test"], dtype=np.float32), np.asarray(z["labels_test"], dtype=np.float32))
    )


def _write_fusion_json(args: argparse.Namespace, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    m2j5_npz = _resolve(args.m2j5_npz)
    m2j20_npz = _resolve(args.m2j20_npz)
    z5 = _assert_score_file(m2j5_npz, "m2j5")
    z20 = _assert_score_file(m2j20_npz, "m2j20")
    _assert_same_labels(z5, z20, "m2j20")

    score_files = {
        # The bin-gated engine names this family "joint_delta"; here it points
        # at the actual delta005 run requested by this wrapper.
        "joint_delta": str(m2j5_npz.resolve()),
        "joint_delta020": str(m2j20_npz.resolve()),
    }
    hlt_source = str(m2j5_npz.resolve())
    if _step1_ref_is_compatible(args.step1_ref_npz, z5):
        score_files["hlt"] = str(_resolve(args.step1_ref_npz))
        hlt_source = str(_resolve(args.step1_ref_npz))
        print(f"[prep] using STEP1 HLT override from: {hlt_source}", flush=True)
    else:
        print(f"[prep] using HLT scores from m2j5 npz: {hlt_source}", flush=True)

    fusion_json = out_dir / "fusion_m2j5_m2j20_bingated_weighted_5m1m1m.json"
    fusion = {
        "run_dirs": {"score_files": score_files},
        "m2_bin_gated_meta": {
            "m2j5_npz": str(m2j5_npz.resolve()),
            "m2j20_npz": str(m2j20_npz.resolve()),
            "hlt_source": hlt_source,
            "candidate_models": str(args.candidate_models),
            "display_models": [
                "m2j5_stage2",
                "m2j5_joint",
                "hlt",
                "m2j20_stage2",
                "m2j20_joint",
            ],
            "engine_model_mapping": {
                "m2j5_stage2": "joint_delta_stage2",
                "m2j5_joint": "joint_delta_joint",
                "m2j20_stage2": "joint_delta020_stage2",
                "m2j20_joint": "joint_delta020_joint",
            },
        },
    }
    fusion_json.write_text(json.dumps(fusion, indent=2))
    print(f"[prep] wrote fusion JSON: {fusion_json}", flush=True)
    return fusion_json


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        if not keys:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


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
            "metrics": {"bin_gated_blend": metrics.get("bin_gated_blend", {})},
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
    print("M2 bin_gated_blend only")
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
    ap = argparse.ArgumentParser(description="M2 delta005/delta020 bin-gated-only fusion.")
    ap.add_argument(
        "--m2j5_npz",
        default=(
            f"{BASE_DIR}/model2_joint_delta005_weighted_5m1m1m/"
            "model2_joint_delta005_weighted_5m1m1m_seed0/fusion_scores_val_test.npz"
        ),
    )
    ap.add_argument(
        "--m2j20_npz",
        default=(
            f"{BASE_DIR}/model2_joint_delta020_weighted_5m1m1m/"
            "model2_joint_delta020_weighted_5m1m1m_seed0/fusion_scores_val_test.npz"
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
    ap.add_argument("--anchor_model", default="joint_delta_joint")
    ap.add_argument("--candidate_models", default=DEFAULT_CANDIDATES)
    ap.add_argument("--selection_mode", default="valsel", choices=["split", "valsel"])
    ap.add_argument("--calibration", default="iso", choices=["raw", "iso", "platt"])
    ap.add_argument("--head_select_mode", default="first", choices=["first", "best_val_auc", "best_val_fpr"])
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
    ap.add_argument("--expand_prepost_variants", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin_gated_script", default="analyze_hlt_joint31_bin_gated_fusion.py")
    ap.add_argument(
        "--out_dir",
        default=f"{BASE_DIR}/analyze_m2j5_m2j20_weighted_5m1m1m/bin_gated",
    )
    ap.add_argument("--report_json", default="")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = _resolve(args.out_dir)
    report_json = _resolve(args.report_json) if str(args.report_json).strip() else out_dir / "bin_gated_report.json"
    full_report_json = out_dir / "bin_gated_full_report.json"

    fusion_json = _write_fusion_json(args, out_dir=out_dir)

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
    print("M2J5/M2J20 Bin-Gated Fusion")
    print("=" * 72)
    print(f"Fusion JSON: {fusion_json}")
    print(f"Out dir:     {out_dir}")
    print(f"Candidates:  {args.candidate_models}")
    print(f"Anchor:      {args.anchor_model}")
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

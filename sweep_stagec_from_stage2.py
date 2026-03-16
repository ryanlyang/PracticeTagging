#!/usr/bin/env python3
"""
Run a Stage-C finetune hyperparameter sweep from saved Stage2 checkpoints.

This launcher runs multiple finetune jobs sequentially by invoking:
  finetune_stagec_from_stage2.py
for each config in a predefined suite, then writes an aggregated summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SWEEP_SUITES: Dict[str, List[Dict[str, float]]] = {
    # Mostly reduce lambda_cons at fixed lambda_reco.
    "debug_1": [
        {"name": "r035_c006", "lambda_reco": 0.35, "lambda_cons": 0.06},
        {"name": "r035_c004", "lambda_reco": 0.35, "lambda_cons": 0.04},
        {"name": "r035_c002", "lambda_reco": 0.35, "lambda_cons": 0.02},
        {"name": "r035_c000", "lambda_reco": 0.35, "lambda_cons": 0.00},
    ],
    # Mostly reduce lambda_reco at low lambda_cons.
    "debug_2": [
        {"name": "r030_c002", "lambda_reco": 0.30, "lambda_cons": 0.02},
        {"name": "r025_c002", "lambda_reco": 0.25, "lambda_cons": 0.02},
        {"name": "r020_c002", "lambda_reco": 0.20, "lambda_cons": 0.02},
        {"name": "r015_c002", "lambda_reco": 0.15, "lambda_cons": 0.02},
    ],
    # Increase lambda_reco at fixed low lambda_cons.
    "tier3_1": [
        {"name": "r040_c002", "lambda_reco": 0.40, "lambda_cons": 0.02},
        {"name": "r045_c002", "lambda_reco": 0.45, "lambda_cons": 0.02},
        {"name": "r055_c002", "lambda_reco": 0.55, "lambda_cons": 0.02},
        {"name": "r070_c002", "lambda_reco": 0.70, "lambda_cons": 0.02},
    ],
    # High lambda_reco with varied lambda_cons.
    "tier3_2": [
        {"name": "r050_c000", "lambda_reco": 0.50, "lambda_cons": 0.00},
        {"name": "r050_c004", "lambda_reco": 0.50, "lambda_cons": 0.04},
        {"name": "r065_c000", "lambda_reco": 0.65, "lambda_cons": 0.00},
        {"name": "r070_c000", "lambda_reco": 0.70, "lambda_cons": 0.00},
    ],
}


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _nested_float(d: Dict, *keys: str) -> float:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return float("nan")
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return float("nan")


def _fmt(x: float) -> str:
    if x is None or not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.6f}"


def _run_one(args: argparse.Namespace, cfg: Dict[str, float]) -> Dict:
    run_name = f"{args.run_prefix}_{args.suite}_{cfg['name']}"
    run_out_dir = Path(args.save_dir) / run_name
    metrics_path = run_out_dir / "stagec_refine_metrics.json"

    if args.skip_existing and metrics_path.exists():
        print(f"[skip] {run_name} (metrics already exist)")
        metrics = _read_json(metrics_path)
        return _extract_row(cfg=cfg, run_name=run_name, rc=0, metrics=metrics, skipped=True)

    cmd = [
        sys.executable,
        "finetune_stagec_from_stage2.py",
        "--run_dir", args.run_dir,
        "--save_dir", args.save_dir,
        "--run_name", run_name,
        "--n_train_jets", str(args.n_train_jets),
        "--offset_jets", str(args.offset_jets),
        "--max_constits", str(args.max_constits),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--stageC_epochs", str(args.stageC_epochs),
        "--stageC_patience", str(args.stageC_patience),
        "--stageC_min_epochs", str(args.stageC_min_epochs),
        "--stageC_lr_dual", str(args.stageC_lr_dual),
        "--stageC_lr_reco", str(args.stageC_lr_reco),
        "--lambda_reco", str(cfg["lambda_reco"]),
        "--lambda_cons", str(cfg["lambda_cons"]),
        "--selection_metric", str(args.selection_metric),
        "--device", str(args.device),
    ]
    if args.use_corrected_flags:
        cmd.append("--use_corrected_flags")

    print("\n" + "=" * 96)
    print(f"[run] {run_name}")
    print(" ".join(cmd))
    print("=" * 96)
    proc = subprocess.run(cmd, check=False)
    rc = int(proc.returncode)

    metrics = _read_json(metrics_path)
    row = _extract_row(cfg=cfg, run_name=run_name, rc=rc, metrics=metrics, skipped=False)
    return row


def _extract_row(cfg: Dict[str, float], run_name: str, rc: int, metrics: Dict, skipped: bool) -> Dict:
    row = {
        "run_name": run_name,
        "cfg_name": cfg["name"],
        "lambda_reco": float(cfg["lambda_reco"]),
        "lambda_cons": float(cfg["lambda_cons"]),
        "return_code": int(rc),
        "skipped": bool(skipped),
        "status": "ok" if (rc == 0 and len(metrics) > 0) else "failed",
    }

    row["teacher_auc"] = _nested_float(metrics, "test_teacher_loaded", "auc")
    row["teacher_fpr30"] = _nested_float(metrics, "test_teacher_loaded", "fpr30")
    row["teacher_fpr50"] = _nested_float(metrics, "test_teacher_loaded", "fpr50")
    row["baseline_auc"] = _nested_float(metrics, "test_baseline_loaded", "auc")
    row["baseline_fpr30"] = _nested_float(metrics, "test_baseline_loaded", "fpr30")
    row["baseline_fpr50"] = _nested_float(metrics, "test_baseline_loaded", "fpr50")

    row["stage2_auc"] = _nested_float(metrics, "test_stage2_loaded", "auc")
    row["stage2_fpr30"] = _nested_float(metrics, "test_stage2_loaded", "fpr30")
    row["stage2_fpr50"] = _nested_float(metrics, "test_stage2_loaded", "fpr50")
    row["stagec_auc"] = _nested_float(metrics, "test_stageC_selected", "auc")
    row["stagec_fpr30"] = _nested_float(metrics, "test_stageC_selected", "fpr30")
    row["stagec_fpr50"] = _nested_float(metrics, "test_stageC_selected", "fpr50")

    # Positive is better for all "improve_vs_baseline" numbers.
    if math.isfinite(row["stagec_auc"]) and math.isfinite(row["baseline_auc"]):
        row["auc_improve_vs_baseline"] = row["stagec_auc"] - row["baseline_auc"]
    else:
        row["auc_improve_vs_baseline"] = float("nan")

    if math.isfinite(row["baseline_fpr30"]) and math.isfinite(row["stagec_fpr30"]):
        row["fpr30_improve_vs_baseline"] = row["baseline_fpr30"] - row["stagec_fpr30"]
    else:
        row["fpr30_improve_vs_baseline"] = float("nan")

    if math.isfinite(row["baseline_fpr50"]) and math.isfinite(row["stagec_fpr50"]):
        row["fpr50_improve_vs_baseline"] = row["baseline_fpr50"] - row["stagec_fpr50"]
    else:
        row["fpr50_improve_vs_baseline"] = float("nan")

    return row


def _print_table(rows: List[Dict]) -> None:
    print("\n" + "#" * 112)
    print("STAGE-C SWEEP SUMMARY")
    print("#" * 112)
    header = (
        "run_name".ljust(42)
        + "  "
        + "r".rjust(6)
        + "  "
        + "c".rjust(6)
        + "  "
        + "AUC".rjust(8)
        + "  "
        + "FPR30".rjust(10)
        + "  "
        + "FPR50".rjust(10)
        + "  "
        + "dAUC_vsB".rjust(10)
        + "  "
        + "d30_vsB".rjust(10)
        + "  "
        + "d50_vsB".rjust(10)
        + "  "
        + "status".rjust(7)
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            row["run_name"][:42].ljust(42)
            + "  "
            + f"{row['lambda_reco']:.2f}".rjust(6)
            + "  "
            + f"{row['lambda_cons']:.2f}".rjust(6)
            + "  "
            + _fmt(row["stagec_auc"]).rjust(8)
            + "  "
            + _fmt(row["stagec_fpr30"]).rjust(10)
            + "  "
            + _fmt(row["stagec_fpr50"]).rjust(10)
            + "  "
            + _fmt(row["auc_improve_vs_baseline"]).rjust(10)
            + "  "
            + _fmt(row["fpr30_improve_vs_baseline"]).rjust(10)
            + "  "
            + _fmt(row["fpr50_improve_vs_baseline"]).rjust(10)
            + "  "
            + str(row["status"]).rjust(7)
        )
        print(line)


def _write_outputs(rows: List[Dict], out_dir: Path, suite: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"stagec_sweep_{suite}_summary.json"
    csv_path = out_dir / f"stagec_sweep_{suite}_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "run_name",
        "cfg_name",
        "lambda_reco",
        "lambda_cons",
        "status",
        "return_code",
        "skipped",
        "teacher_auc",
        "teacher_fpr30",
        "teacher_fpr50",
        "baseline_auc",
        "baseline_fpr30",
        "baseline_fpr50",
        "stage2_auc",
        "stage2_fpr30",
        "stage2_fpr50",
        "stagec_auc",
        "stagec_fpr30",
        "stagec_fpr50",
        "auc_improve_vs_baseline",
        "fpr30_improve_vs_baseline",
        "fpr50_improve_vs_baseline",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote summary JSON: {json_path}")
    print(f"Wrote summary CSV:  {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="checkpoints/offline_reconstructor_joint_stagec_sweeps")
    p.add_argument("--suite", type=str, required=True, choices=sorted(SWEEP_SUITES.keys()))
    p.add_argument("--run_prefix", type=str, default="stagec_sweep")

    p.add_argument("--n_train_jets", type=int, default=100000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=80)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--stageC_epochs", type=int, default=100)
    p.add_argument("--stageC_patience", type=int, default=14)
    p.add_argument("--stageC_min_epochs", type=int, default=25)
    p.add_argument("--stageC_lr_dual", type=float, default=2e-5)
    p.add_argument("--stageC_lr_reco", type=float, default=1e-5)
    p.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_corrected_flags", action="store_true")

    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--fail_fast", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    suite_cfgs = SWEEP_SUITES[args.suite]
    print(f"Running suite: {args.suite} ({len(suite_cfgs)} configs)")
    print(f"run_dir: {run_dir}")
    print(f"save_dir: {args.save_dir}")
    print(f"base LRs: dual={args.stageC_lr_dual}, reco={args.stageC_lr_reco}")
    print(f"selection_metric: {args.selection_metric}")

    rows: List[Dict] = []
    for cfg in suite_cfgs:
        row = _run_one(args, cfg)
        rows.append(row)
        if row["status"] != "ok" and args.fail_fast:
            print(f"Fail-fast enabled; stopping after failure in {row['run_name']}")
            break

    _print_table(rows)
    _write_outputs(rows, Path(args.save_dir), args.suite)


if __name__ == "__main__":
    main()


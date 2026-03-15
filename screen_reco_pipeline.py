#!/usr/bin/env python3
"""
Multi-fidelity screening pipeline for offline reconstructor experiments.

Purpose:
- Run many reconstruction candidates.
- Rank by a downstream metric (default: fpr50_dual_flag_kd).
- Apply hard diagnostic gates (resolution/count/budget sanity).
- Promote top candidates through larger stages.

This script is an orchestrator: it launches an existing reconstructor script
(e.g. offline_reconstructor_no_gt_nopriv_staged.py), then reads that run's
artifacts (results.npz + summaries) to decide promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


@dataclass
class StageSpec:
    name: str
    n_train_jets: int
    reco_epochs: int
    reco_patience: int
    promote_top_k: int | None = None
    promote_top_frac: float | None = None


def _default_stages() -> List[StageSpec]:
    return [
        StageSpec(
            name="stage_a_30k",
            n_train_jets=30000,
            reco_epochs=50,
            reco_patience=12,
            promote_top_frac=0.50,
        ),
        StageSpec(
            name="stage_b_50k",
            n_train_jets=50000,
            reco_epochs=80,
            reco_patience=16,
            promote_top_frac=0.40,
        ),
        StageSpec(
            name="stage_c_120k",
            n_train_jets=120000,
            reco_epochs=110,
            reco_patience=20,
            promote_top_k=3,
        ),
    ]


def _load_stages(path: Path | None) -> List[StageSpec]:
    if path is None:
        return _default_stages()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[StageSpec] = []
    for row in data:
        out.append(
            StageSpec(
                name=str(row["name"]),
                n_train_jets=int(row["n_train_jets"]),
                reco_epochs=int(row["reco_epochs"]),
                reco_patience=int(row.get("reco_patience", 20)),
                promote_top_k=None if row.get("promote_top_k") is None else int(row["promote_top_k"]),
                promote_top_frac=None if row.get("promote_top_frac") is None else float(row["promote_top_frac"]),
            )
        )
    if len(out) == 0:
        raise ValueError("Stages JSON is empty.")
    return out


def _default_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "name": "baseline",
            "args": {},
        }
    ]


def _load_candidates(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return _default_candidates()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Candidates JSON must be a non-empty list.")
    out: List[Dict[str, Any]] = []
    for row in data:
        if "name" not in row:
            raise ValueError("Each candidate must include 'name'.")
        out.append(
            {
                "name": str(row["name"]),
                "args": dict(row.get("args", {})),
            }
        )
    return out


def _cli_from_args(arg_dict: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    for k, v in arg_dict.items():
        if v is None:
            continue
        key = f"--{k}"
        if isinstance(v, bool):
            if v:
                cli.append(key)
            continue
        if isinstance(v, (list, tuple)):
            cli.extend([key, ",".join(str(x) for x in v)])
            continue
        cli.extend([key, str(v)])
    return cli


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metric(npz_path: Path, metric_key: str) -> float:
    if not npz_path.exists():
        return float("nan")
    d = np.load(npz_path)
    if metric_key not in d:
        return float("nan")
    return float(np.array(d[metric_key]).reshape(()))


def _compute_response_health(npz_path: Path) -> Tuple[float, float]:
    """
    Returns:
    - mean_abs_response_bias: mean(|resp_corr - 1|)
    - mean_resolution_ratio: mean(reso_corr / reso_hlt)
    """
    if not npz_path.exists():
        return float("nan"), float("nan")
    d = np.load(npz_path)
    need = ["jet_response_hlt_std", "jet_response_corrected_std", "jet_response_corrected_mean"]
    if any(k not in d for k in need):
        return float("nan"), float("nan")
    hlt_std = np.asarray(d["jet_response_hlt_std"], dtype=np.float64)
    corr_std = np.asarray(d["jet_response_corrected_std"], dtype=np.float64)
    corr_mean = np.asarray(d["jet_response_corrected_mean"], dtype=np.float64)
    if hlt_std.size == 0 or corr_std.size == 0 or corr_mean.size == 0:
        return float("nan"), float("nan")
    reso_ratio = float(np.mean(corr_std / np.maximum(hlt_std, 1e-8)))
    resp_bias = float(np.mean(np.abs(corr_mean - 1.0)))
    return resp_bias, reso_ratio


def _apply_gates(
    metric_value: float,
    metric_lower_is_better: bool,
    count_summary: Dict[str, Any],
    budget_summary: Dict[str, Any],
    npz_path: Path,
    gate_max_count_mae: float,
    gate_max_budget_total_mae: float,
    gate_max_response_bias: float,
    gate_max_resolution_ratio: float,
) -> Tuple[bool, List[str], Dict[str, float]]:
    notes: List[str] = []
    diag: Dict[str, float] = {}

    if not np.isfinite(metric_value):
        notes.append("missing_or_nan_target_metric")
        return False, notes, diag

    reco_count_mae = float(count_summary.get("reco_count_mae_vs_offline", np.nan))
    budget_total_mae = float(budget_summary.get("total_mae", np.nan))
    response_bias, resolution_ratio = _compute_response_health(npz_path)

    diag["reco_count_mae"] = reco_count_mae
    diag["budget_total_mae"] = budget_total_mae
    diag["response_bias"] = response_bias
    diag["resolution_ratio"] = resolution_ratio

    ok = True
    if np.isfinite(reco_count_mae) and reco_count_mae > gate_max_count_mae:
        ok = False
        notes.append(f"count_mae>{gate_max_count_mae}")
    if np.isfinite(budget_total_mae) and budget_total_mae > gate_max_budget_total_mae:
        ok = False
        notes.append(f"budget_total_mae>{gate_max_budget_total_mae}")
    if np.isfinite(response_bias) and response_bias > gate_max_response_bias:
        ok = False
        notes.append(f"response_bias>{gate_max_response_bias}")
    if np.isfinite(resolution_ratio) and resolution_ratio > gate_max_resolution_ratio:
        ok = False
        notes.append(f"resolution_ratio>{gate_max_resolution_ratio}")

    # metric direction is used for ranking, but sanity-check finiteness here only.
    _ = metric_lower_is_better
    return ok, notes, diag


def _rank(
    rows: List[Dict[str, Any]],
    metric_key: str,
    lower_is_better: bool,
) -> List[Dict[str, Any]]:
    def sort_key(x: Dict[str, Any]) -> float:
        v = float(x.get(metric_key, np.nan))
        if not np.isfinite(v):
            return float("inf")
        return v if lower_is_better else -v

    return sorted(rows, key=sort_key)


def _pick_promoted(stage: StageSpec, ranked_ok: List[Dict[str, Any]]) -> List[str]:
    if len(ranked_ok) == 0:
        return []
    if stage.promote_top_k is not None:
        k = max(1, min(int(stage.promote_top_k), len(ranked_ok)))
        return [r["candidate"] for r in ranked_ok[:k]]
    frac = 0.5 if stage.promote_top_frac is None else float(stage.promote_top_frac)
    k = max(1, min(len(ranked_ok), int(math.ceil(frac * len(ranked_ok)))))
    return [r["candidate"] for r in ranked_ok[:k]]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reco_script", type=str, default="offline_reconstructor_no_gt_nopriv_staged.py")
    parser.add_argument("--candidates_json", type=str, default="")
    parser.add_argument("--stages_json", type=str, default="")
    parser.add_argument("--save_root", type=str, default="checkpoints/reco_screening")
    parser.add_argument("--screen_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--target_metric", type=str, default="fpr50_dual_flag_kd")
    parser.add_argument("--metric_lower_is_better", action="store_true", default=True)
    parser.add_argument("--metric_higher_is_better", action="store_true")
    parser.add_argument("--skip_save_models", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--extra_cli", type=str, default="")

    # Hard gates for quick rejection.
    parser.add_argument("--gate_max_count_mae", type=float, default=8.0)
    parser.add_argument("--gate_max_budget_total_mae", type=float, default=8.0)
    parser.add_argument("--gate_max_response_bias", type=float, default=0.25)
    parser.add_argument("--gate_max_resolution_ratio", type=float, default=1.60)

    args = parser.parse_args()
    if args.metric_higher_is_better:
        metric_lower_is_better = False
    else:
        metric_lower_is_better = True

    reco_script = Path(args.reco_script)
    if not reco_script.exists():
        raise FileNotFoundError(f"Reconstructor script not found: {reco_script}")

    candidates = _load_candidates(Path(args.candidates_json) if args.candidates_json else None)
    stages = _load_stages(Path(args.stages_json) if args.stages_json else None)

    run_root = Path(args.save_root) / args.screen_name
    run_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    active_candidates = [c["name"] for c in candidates]
    cand_map = {c["name"]: c for c in candidates}
    extra_cli = shlex.split(args.extra_cli) if args.extra_cli else []

    print(f"Screening root: {run_root}")
    print(f"Candidates: {len(candidates)}")
    print(f"Stages: {[s.name for s in stages]}")
    print(f"Target metric: {args.target_metric} ({'min' if metric_lower_is_better else 'max'})")

    for s in stages:
        if len(active_candidates) == 0:
            print(f"[{s.name}] no candidates left; stopping.")
            break
        print(f"\n=== {s.name} | candidates={len(active_candidates)} | jets={s.n_train_jets} ===")

        stage_rows: List[Dict[str, Any]] = []
        stage_save_dir = run_root / s.name
        stage_save_dir.mkdir(parents=True, exist_ok=True)

        for cname in active_candidates:
            cand = cand_map[cname]
            run_name = cname
            cmd = [
                sys.executable,
                str(reco_script),
                "--save_dir",
                str(stage_save_dir),
                "--run_name",
                run_name,
                "--n_train_jets",
                str(s.n_train_jets),
                "--offset_jets",
                str(args.offset_jets),
                "--max_constits",
                str(args.max_constits),
                "--num_workers",
                str(args.num_workers),
                "--device",
                str(args.device),
                "--reco_epochs",
                str(s.reco_epochs),
                "--reco_patience",
                str(s.reco_patience),
            ]
            if args.skip_save_models:
                cmd.append("--skip_save_models")
            cmd.extend(_cli_from_args(cand.get("args", {})))
            cmd.extend(extra_cli)

            print(f"[{s.name}] run {cname}")
            print("  " + " ".join(shlex.quote(x) for x in cmd))

            rc = 0
            if not args.dry_run:
                proc = subprocess.run(cmd, check=False)
                rc = int(proc.returncode)

            run_dir = stage_save_dir / run_name
            npz_path = run_dir / "results.npz"
            count_json = run_dir / "constituent_count_summary.json"
            budget_json = run_dir / "budget_summary_test.json"

            count_summary = _load_json_if_exists(count_json)
            budget_summary = _load_json_if_exists(budget_json)
            metric_value = _extract_metric(npz_path, args.target_metric)

            ok, gate_notes, gate_diag = _apply_gates(
                metric_value=metric_value,
                metric_lower_is_better=metric_lower_is_better,
                count_summary=count_summary,
                budget_summary=budget_summary,
                npz_path=npz_path,
                gate_max_count_mae=float(args.gate_max_count_mae),
                gate_max_budget_total_mae=float(args.gate_max_budget_total_mae),
                gate_max_response_bias=float(args.gate_max_response_bias),
                gate_max_resolution_ratio=float(args.gate_max_resolution_ratio),
            )

            row = {
                "stage": s.name,
                "candidate": cname,
                "return_code": rc,
                "status": "ok" if (rc == 0 and ok) else "rejected",
                "gate_notes": ";".join(gate_notes),
                args.target_metric: metric_value,
                "run_dir": str(run_dir),
            }
            row.update(gate_diag)
            stage_rows.append(row)
            all_rows.append(row)

        ranked = _rank(
            [r for r in stage_rows if r["status"] == "ok"],
            metric_key=args.target_metric,
            lower_is_better=metric_lower_is_better,
        )
        promoted = _pick_promoted(s, ranked)
        active_candidates = promoted

        print(f"[{s.name}] promoted: {promoted if promoted else 'none'}")

        _write_csv(stage_save_dir / "stage_summary.csv", stage_rows)
        with open(stage_save_dir / "stage_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage": s.name,
                    "target_metric": args.target_metric,
                    "metric_lower_is_better": metric_lower_is_better,
                    "rows": stage_rows,
                    "promoted": promoted,
                },
                f,
                indent=2,
            )

    _write_csv(run_root / "screen_summary.csv", all_rows)
    with open(run_root / "screen_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_metric": args.target_metric,
                "metric_lower_is_better": metric_lower_is_better,
                "rows": all_rows,
                "final_promoted": active_candidates,
            },
            f,
            indent=2,
        )

    print("\nScreening complete.")
    print(f"Summary CSV: {run_root / 'screen_summary.csv'}")
    print(f"Summary JSON: {run_root / 'screen_summary.json'}")
    if len(active_candidates) > 0:
        print(f"Final promoted: {active_candidates}")
    else:
        print("No candidates survived gates.")


if __name__ == "__main__":
    main()

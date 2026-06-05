#!/usr/bin/env python3
"""
Analyze model contributions to JetClass stacked-logistic fusion.

This script consumes a saved `fusion_scores.npz` from
analyze_jetclass_four_model_stacked_fusion.py and retrains lightweight stacked
logistic regressions on different subsets of the base-model logits/probs.

It does not rerun neural inference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _infer_model_names(files: Sequence[str]) -> List[str]:
    names = []
    for k in files:
        if k.startswith("logits_val_"):
            name = k[len("logits_val_") :]
            if f"logits_test_{name}" in files and f"probs_val_{name}" in files and f"probs_test_{name}" in files:
                names.append(name)
    names = sorted(set(names))
    if not names:
        raise ValueError("No base model keys found. Expected logits_val_<name>/probs_val_<name> keys.")
    return names


def _features(d: np.lib.npyio.NpzFile, names: Sequence[str], split: str, mode: str) -> np.ndarray:
    parts: List[np.ndarray] = []
    if mode in {"logits", "logits_probs"}:
        parts.extend(np.asarray(d[f"logits_{split}_{name}"], dtype=np.float32) for name in names)
    if mode in {"probs", "logits_probs"}:
        parts.extend(np.asarray(d[f"probs_{split}_{name}"], dtype=np.float32) for name in names)
    if not parts:
        raise ValueError(f"Unknown feature mode: {mode}")
    return np.concatenate(parts, axis=1).astype(np.float32)


def _auc_macro(y: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _fit_eval_subset(
    d: np.lib.npyio.NpzFile,
    names: Sequence[str],
    feature_mode: str,
    Cs: Sequence[float],
    cv: int,
    max_iter: int,
    n_jobs: int,
    compute_auc: bool,
) -> Dict[str, object]:
    y_val = np.asarray(d["y_val"], dtype=np.int64)
    y_test = np.asarray(d["y_test"], dtype=np.int64)
    x_val = _features(d, names, "val", feature_mode)
    x_test = _features(d, names, "test", feature_mode)

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegressionCV(
            Cs=[float(c) for c in Cs],
            cv=int(cv),
            solver="lbfgs",
            scoring="accuracy",
            max_iter=int(max_iter),
            n_jobs=int(n_jobs),
            refit=True,
        ),
    )
    clf.fit(x_val, y_val)
    p_val = clf.predict_proba(x_val).astype(np.float32)
    p_test = clf.predict_proba(x_test).astype(np.float32)
    pred_val = np.argmax(p_val, axis=1)
    pred_test = np.argmax(p_test, axis=1)
    lr = clf.named_steps["logisticregressioncv"]
    best_c = float(np.mean(lr.C_)) if getattr(lr, "C_", np.array([])).size else float("nan")
    return {
        "models": list(names),
        "n_models": int(len(names)),
        "val_acc": float(accuracy_score(y_val, pred_val)),
        "test_acc": float(accuracy_score(y_test, pred_test)),
        "val_auc_macro_ovr": _auc_macro(y_val, p_val) if bool(compute_auc) else float("nan"),
        "test_auc_macro_ovr": _auc_macro(y_test, p_test) if bool(compute_auc) else float("nan"),
        "best_C_mean": best_c,
    }


def _all_combos(names: Sequence[str], max_size: int) -> Iterable[Tuple[str, ...]]:
    for k in range(1, int(max_size) + 1):
        yield from itertools.combinations(names, k)


def _top_by_val(rows: Sequence[Dict[str, object]], n: int) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda r: (_safe_float(r["val_acc"]), _safe_float(r["test_acc"])), reverse=True)[: int(n)]


def _greedy_forward(
    d: np.lib.npyio.NpzFile,
    names: Sequence[str],
    feature_mode: str,
    Cs: Sequence[float],
    cv: int,
    max_iter: int,
    n_jobs: int,
    max_steps: int,
    compute_auc: bool,
) -> List[Dict[str, object]]:
    selected: List[str] = []
    remaining = list(names)
    history: List[Dict[str, object]] = []
    for step in range(1, min(int(max_steps), len(names)) + 1):
        candidates: List[Dict[str, object]] = []
        for name in remaining:
            subset = tuple(selected + [name])
            row = _fit_eval_subset(d, subset, feature_mode, Cs, cv, max_iter, n_jobs, compute_auc)
            row["added"] = name
            row["step"] = step
            candidates.append(row)
        best = _top_by_val(candidates, 1)[0]
        selected = list(best["models"])
        remaining = [n for n in remaining if n not in selected]
        history.append(best)
    return history


def _load_report(report_path: Path | None) -> Dict[str, object]:
    if report_path is None:
        return {}
    if not report_path.exists():
        print(f"[warn] report_json not found: {report_path}")
        return {}
    return json.loads(report_path.read_text())


def _individual_from_report(report: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    metrics = report.get("individual_metrics", {})
    if isinstance(metrics, dict):
        for name, obj in metrics.items():
            if not isinstance(obj, dict) or "test" not in obj:
                continue
            test = obj.get("test", {})
            val = obj.get("val", {})
            out[str(name)] = {
                "val_acc": _safe_float(val.get("acc")),
                "test_acc": _safe_float(test.get("acc")),
                "val_auc_macro_ovr": _safe_float(val.get("auc_macro_ovr")),
                "test_auc_macro_ovr": _safe_float(test.get("auc_macro_ovr")),
            }
    return out


def _print_row(row: Dict[str, object], prefix: str = "") -> None:
    models = ",".join(row["models"])
    print(
        f"{prefix}{models:80s} "
        f"val_acc={float(row['val_acc']):.6f} test_acc={float(row['test_acc']):.6f} "
        f"val_auc={float(row['val_auc_macro_ovr']):.6f} test_auc={float(row['test_auc_macro_ovr']):.6f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--feature_mode", type=str, default="logits_probs", choices=["logits", "probs", "logits_probs"])
    ap.add_argument("--max_combo_size", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--greedy_steps", type=int, default=8)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--skip_auc", action="store_true", help="Skip macro-AUC computations for faster subset scans.")
    ap.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated model-name subset to consider. Defaults to all models in NPZ.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.scores_npz)
    names = _infer_model_names(d.files)
    if args.models.strip():
        wanted = [x.strip() for x in args.models.split(",") if x.strip()]
        missing = [x for x in wanted if x not in names]
        if missing:
            raise ValueError(f"Requested models not found in NPZ: {missing}; available={names}")
        names = wanted

    report = _load_report(args.report_json)
    individual = _individual_from_report(report)

    print("============================================================")
    print("JetClass Stacked Subset Contribution Analysis")
    print("============================================================")
    print(f"Scores: {args.scores_npz.resolve()}")
    print(f"Models ({len(names)}): {', '.join(names)}")
    print(f"Feature mode: {args.feature_mode}")

    print("\nFitting all-model stack...")
    compute_auc = not bool(args.skip_auc)

    all_row = _fit_eval_subset(d, tuple(names), args.feature_mode, args.Cs, args.cv, args.max_iter, args.n_jobs, compute_auc)
    _print_row(all_row, prefix="ALL ")

    print("\nFitting leave-one-out stacks...")
    loo_rows: List[Dict[str, object]] = []
    for name in names:
        subset = tuple(n for n in names if n != name)
        row = _fit_eval_subset(d, subset, args.feature_mode, args.Cs, args.cv, args.max_iter, args.n_jobs, compute_auc)
        row["removed"] = name
        row["val_acc_drop"] = float(all_row["val_acc"] - row["val_acc"])
        row["test_acc_drop"] = float(all_row["test_acc"] - row["test_acc"])
        loo_rows.append(row)
        print(
            f"remove {name:20s} val_drop={row['val_acc_drop']:+.6f} "
            f"test_drop={row['test_acc_drop']:+.6f} "
            f"without_val={row['val_acc']:.6f} without_test={row['test_acc']:.6f}"
        )

    print(f"\nFitting exhaustive combos up to size {args.max_combo_size}...")
    combo_rows: List[Dict[str, object]] = []
    total = sum(math_comb(len(names), k) for k in range(1, int(args.max_combo_size) + 1))
    for i, combo in enumerate(_all_combos(names, int(args.max_combo_size)), start=1):
        row = _fit_eval_subset(d, combo, args.feature_mode, args.Cs, args.cv, args.max_iter, args.n_jobs, compute_auc)
        combo_rows.append(row)
        if i % 50 == 0 or i == total:
            print(f"  combos {i}/{total}")

    print("\nFitting greedy forward selection...")
    greedy_rows = _greedy_forward(
        d,
        names,
        args.feature_mode,
        args.Cs,
        args.cv,
        args.max_iter,
        args.n_jobs,
        int(args.greedy_steps),
        compute_auc,
    )
    for row in greedy_rows:
        _print_row(row, prefix=f"step {row['step']:02d} add {row['added']:20s} ")

    top_combos = _top_by_val(combo_rows, int(args.top_k))
    top_by_size: Dict[str, List[Dict[str, object]]] = {}
    for k in range(1, int(args.max_combo_size) + 1):
        top_by_size[str(k)] = _top_by_val([r for r in combo_rows if int(r["n_models"]) == k], int(args.top_k))

    loo_sorted = sorted(loo_rows, key=lambda r: float(r["val_acc_drop"]), reverse=True)
    singleton_rows = [r for r in combo_rows if int(r["n_models"]) == 1]
    singleton_sorted = sorted(singleton_rows, key=lambda r: float(r["val_acc"]), reverse=True)

    summary = {
        "scores_npz": str(args.scores_npz.resolve()),
        "report_json": str(args.report_json.resolve()) if args.report_json else "",
        "feature_mode": str(args.feature_mode),
        "compute_auc": bool(compute_auc),
        "models": list(names),
        "all_models": all_row,
        "individual_from_report": individual,
        "leave_one_out": loo_sorted,
        "singletons": singleton_sorted,
        "top_combos_overall": top_combos,
        "top_combos_by_size": top_by_size,
        "greedy_forward": greedy_rows,
    }
    out_json = out_dir / "stacked_subset_contribution_report.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("\nTop leave-one-out contributors by val_acc_drop:")
    for row in loo_sorted:
        print(
            f"  {row['removed']:20s} val_drop={row['val_acc_drop']:+.6f} "
            f"test_drop={row['test_acc_drop']:+.6f}"
        )

    print(f"\nTop {min(args.top_k, len(top_combos))} combos by val accuracy:")
    for row in top_combos[: int(args.top_k)]:
        _print_row(row, prefix="  ")

    print(f"\nSaved: {out_json}")


def math_comb(n: int, k: int) -> int:
    try:
        return int(math.comb(n, k))  # type: ignore[name-defined]
    except Exception:
        return len(list(itertools.combinations(range(n), k)))


if __name__ == "__main__":
    import math

    main()

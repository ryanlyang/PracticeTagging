#!/usr/bin/env python3
"""
Backward-greedy pruning for JetClass stacked-logistic fusion.

Loads a saved fusion_scores.npz and repeatedly removes one base model from the
current stack. At each step it tries every single-model removal, refits the
stacked logistic regression on the remaining models, and keeps the removal that
maximizes the chosen metric. No neural inference is rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MetricRow = Dict[str, object]


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _infer_model_names(files: Sequence[str]) -> List[str]:
    names: List[str] = []
    files_set = set(files)
    for k in files:
        if not k.startswith("logits_val_"):
            continue
        name = k[len("logits_val_") :]
        required = {
            f"logits_test_{name}",
            f"probs_val_{name}",
            f"probs_test_{name}",
        }
        if required.issubset(files_set):
            names.append(name)
    names = sorted(set(names))
    if not names:
        raise ValueError("No base model score keys found in NPZ.")
    return names


def _features(d: np.lib.npyio.NpzFile, names: Sequence[str], split: str, mode: str) -> np.ndarray:
    parts: List[np.ndarray] = []
    if mode in {"logits", "logits_probs"}:
        parts.extend(np.asarray(d[f"logits_{split}_{name}"], dtype=np.float32) for name in names)
    if mode in {"probs", "logits_probs"}:
        parts.extend(np.asarray(d[f"probs_{split}_{name}"], dtype=np.float32) for name in names)
    if not parts:
        raise ValueError(f"Unknown feature_mode: {mode}")
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


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
) -> MetricRow:
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
    if not isinstance(metrics, dict):
        return out
    for name, obj in metrics.items():
        if not isinstance(obj, dict):
            continue
        val = obj.get("val", {}) if isinstance(obj.get("val", {}), dict) else {}
        test = obj.get("test", {}) if isinstance(obj.get("test", {}), dict) else {}
        out[str(name)] = {
            "val_acc": _safe_float(val.get("acc")),
            "test_acc": _safe_float(test.get("acc")),
            "val_auc_macro_ovr": _safe_float(val.get("auc_macro_ovr")),
            "test_auc_macro_ovr": _safe_float(test.get("auc_macro_ovr")),
        }
    return out


def _metric_key(selection_metric: str) -> str:
    if selection_metric not in {"val_acc", "test_acc"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    return selection_metric


def _candidate_sort_key(row: MetricRow, selection_metric: str) -> Tuple[float, float, float]:
    primary = _safe_float(row[selection_metric])
    secondary_name = "test_acc" if selection_metric == "val_acc" else "val_acc"
    secondary = _safe_float(row[secondary_name])
    return primary, secondary, _safe_float(row.get("val_auc_macro_ovr"))


def _backward_prune(
    d: np.lib.npyio.NpzFile,
    all_names: Sequence[str],
    selection_metric: str,
    stop_n: int,
    feature_mode: str,
    Cs: Sequence[float],
    cv: int,
    max_iter: int,
    n_jobs: int,
    compute_auc: bool,
    cache: Dict[Tuple[str, ...], MetricRow],
) -> Dict[str, object]:
    selection_metric = _metric_key(selection_metric)
    current = tuple(all_names)

    def fit(names: Sequence[str]) -> MetricRow:
        key = tuple(names)
        if key not in cache:
            cache[key] = _fit_eval_subset(d, key, feature_mode, Cs, cv, max_iter, n_jobs, compute_auc)
        return cache[key]

    full = fit(current)
    path: List[Dict[str, object]] = []
    print(f"\nBackward pruning path selected by {selection_metric}")
    print(
        f"start n={len(current):02d} val_acc={full['val_acc']:.6f} "
        f"test_acc={full['test_acc']:.6f} models={','.join(current)}"
    )

    step = 0
    while len(current) > int(stop_n):
        before = fit(current)
        candidates: List[MetricRow] = []
        for removed in current:
            remaining = tuple(n for n in current if n != removed)
            row = dict(fit(remaining))
            row["removed"] = removed
            row["remaining_models"] = list(remaining)
            row["delta_val_acc_vs_before"] = _safe_float(row["val_acc"]) - _safe_float(before["val_acc"])
            row["delta_test_acc_vs_before"] = _safe_float(row["test_acc"]) - _safe_float(before["test_acc"])
            row["drop_val_acc_vs_before"] = _safe_float(before["val_acc"]) - _safe_float(row["val_acc"])
            row["drop_test_acc_vs_before"] = _safe_float(before["test_acc"]) - _safe_float(row["test_acc"])
            candidates.append(row)

        best = sorted(candidates, key=lambda r: _candidate_sort_key(r, selection_metric), reverse=True)[0]
        current = tuple(best["remaining_models"])  # type: ignore[arg-type]
        step += 1
        record = {
            "step": step,
            "n_before": int(len(current) + 1),
            "n_after": int(len(current)),
            "selection_metric": selection_metric,
            "removed": best["removed"],
            "before": before,
            "after": {k: v for k, v in best.items() if k not in {"remaining_models"}},
            "remaining_models": list(current),
            "candidate_removals": sorted(
                candidates,
                key=lambda r: _candidate_sort_key(r, selection_metric),
                reverse=True,
            ),
        }
        path.append(record)
        print(
            f"step {step:02d}: remove {best['removed']:20s} "
            f"n={len(current):02d} val_acc={best['val_acc']:.6f} "
            f"test_acc={best['test_acc']:.6f} "
            f"dval={best['delta_val_acc_vs_before']:+.6f} "
            f"dtest={best['delta_test_acc_vs_before']:+.6f}"
        )

    final = fit(current)
    return {
        "selection_metric": selection_metric,
        "stop_n": int(stop_n),
        "start_models": list(all_names),
        "full_stack": full,
        "path": path,
        "final_subset": list(current),
        "final_metrics": final,
    }


def _write_trace_tsv(out_path: Path, result: Dict[str, object]) -> None:
    path = result.get("path", [])
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "selection_metric",
                "step",
                "n_before",
                "n_after",
                "removed",
                "val_acc",
                "test_acc",
                "delta_val_acc_vs_before",
                "delta_test_acc_vs_before",
                "remaining_models",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for item in path:  # type: ignore[assignment]
            after = item["after"]
            writer.writerow(
                {
                    "selection_metric": result["selection_metric"],
                    "step": item["step"],
                    "n_before": item["n_before"],
                    "n_after": item["n_after"],
                    "removed": item["removed"],
                    "val_acc": after["val_acc"],
                    "test_acc": after["test_acc"],
                    "delta_val_acc_vs_before": after["delta_val_acc_vs_before"],
                    "delta_test_acc_vs_before": after["delta_test_acc_vs_before"],
                    "remaining_models": ",".join(item["remaining_models"]),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--feature_mode", type=str, default="logits_probs", choices=["logits", "probs", "logits_probs"])
    ap.add_argument("--stop_n", type=int, default=5)
    ap.add_argument("--selection_metrics", type=str, nargs="+", default=["val_acc", "test_acc"], choices=["val_acc", "test_acc"])
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--skip_auc", action="store_true")
    ap.add_argument("--models", type=str, default="", help="Optional comma-separated model list. Defaults to all NPZ models.")
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
    compute_auc = not bool(args.skip_auc)
    cache: Dict[Tuple[str, ...], MetricRow] = {}

    print("============================================================")
    print("JetClass Stacked Backward-Greedy Pruning")
    print("============================================================")
    print(f"Scores: {args.scores_npz.resolve()}")
    print(f"Report: {args.report_json.resolve() if args.report_json else ''}")
    print(f"Out:    {out_dir}")
    print(f"Models ({len(names)}): {', '.join(names)}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Stop n: {args.stop_n}")
    print(f"Selection metrics: {', '.join(args.selection_metrics)}")
    print(f"Skip AUC: {args.skip_auc}")

    results = []
    for metric in args.selection_metrics:
        results.append(
            _backward_prune(
                d=d,
                all_names=names,
                selection_metric=metric,
                stop_n=int(args.stop_n),
                feature_mode=args.feature_mode,
                Cs=args.Cs,
                cv=int(args.cv),
                max_iter=int(args.max_iter),
                n_jobs=int(args.n_jobs),
                compute_auc=compute_auc,
                cache=cache,
            )
        )

    summary = {
        "scores_npz": str(args.scores_npz.resolve()),
        "report_json": str(args.report_json.resolve()) if args.report_json else "",
        "feature_mode": str(args.feature_mode),
        "stop_n": int(args.stop_n),
        "compute_auc": bool(compute_auc),
        "models": list(names),
        "individual_from_report": individual,
        "n_unique_subset_fits": int(len(cache)),
        "results": results,
    }
    out_json = out_dir / "backward_prune_report.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    for result in results:
        trace_path = out_dir / f"backward_prune_trace_select_{result['selection_metric']}.tsv"
        _write_trace_tsv(trace_path, result)
        final = result["final_metrics"]
        print(
            f"\nFinal select={result['selection_metric']}: "
            f"models={','.join(result['final_subset'])} "
            f"val_acc={final['val_acc']:.6f} test_acc={final['test_acc']:.6f}"
        )
        print(f"Trace: {trace_path}")

    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


FEATURE_NAMES = [
    "teacher_logit",
    "teacher_prob",
    "teacher_entropy",
    "nconst_norm",
    "count_diff_hlt_norm",
    "pt_ratio_hlt",
    "pt_ratio_hlt_abs1",
]


def _reshape_bkf_to_flat(x_bkf: np.ndarray) -> np.ndarray:
    # [N,K,F] -> [N, K*F]
    n, k, f = x_bkf.shape
    return x_bkf.reshape(n, k * f)


def _subset_flat(x_bkf: np.ndarray, feat_ids: List[int]) -> np.ndarray:
    # keep selected per-hyp features, then flatten
    xs = x_bkf[:, :, feat_ids]
    return _reshape_bkf_to_flat(xs)


def _fit_eval(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    xte: np.ndarray,
    yte: np.ndarray,
) -> Tuple[float, float]:
    clf = LogisticRegression(
        max_iter=1200,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=1,
    )
    clf.fit(xtr, ytr)
    pva = clf.predict(xva)
    pte = clf.predict(xte)
    return float(accuracy_score(yva, pva)), float(accuracy_score(yte, pte))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Run directory containing selector_outputs.npz and data_splits.npz")
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    sel_npz = np.load(run_dir / "selector_outputs.npz")
    split_npz = np.load(run_dir / "data_splits.npz")

    x_bkf = sel_npz["selector_feat_all"].astype(np.float32)
    y = sel_npz["selector_target_all"].astype(np.int64)
    tr = split_npz["train_idx"].astype(np.int64)
    va = split_npz["val_idx"].astype(np.int64)
    te = split_npz["test_idx"].astype(np.int64)

    groups: Dict[str, List[int]] = {
        "all": list(range(len(FEATURE_NAMES))),
        "teacher_only": [0, 1, 2],
        "count_only": [3, 4],
        "pt_only": [5, 6],
        "teacher_plus_count": [0, 1, 2, 3, 4],
        "teacher_plus_pt": [0, 1, 2, 5, 6],
        "count_plus_pt": [3, 4, 5, 6],
    }

    rows = []
    best_name = None
    best_val = -1.0
    best_test = float("nan")
    for name, ids in groups.items():
        xf = _subset_flat(x_bkf, ids)
        va_acc, te_acc = _fit_eval(xf[tr], y[tr], xf[va], y[va], xf[te], y[te])
        row = {
            "group": name,
            "feature_ids": ids,
            "features": [FEATURE_NAMES[i] for i in ids],
            "val_acc": va_acc,
            "test_acc": te_acc,
        }
        rows.append(row)
        if va_acc > best_val:
            best_val = va_acc
            best_test = te_acc
            best_name = name

    rows = sorted(rows, key=lambda r: r["val_acc"], reverse=True)

    print("Selector feature-group sweep:")
    for r in rows:
        print(f"  {r['group']:>18s} | val_acc={r['val_acc']:.4f} | test_acc={r['test_acc']:.4f}")
    print(f"\nBest by val_acc: {best_name} | val_acc={best_val:.4f} | test_acc={best_test:.4f}")

    out_csv = Path(args.out_csv) if args.out_csv else (run_dir / "selector_signal_sweep.csv")
    out_json = Path(args.out_json) if args.out_json else (run_dir / "selector_signal_sweep.json")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "val_acc", "test_acc", "feature_ids", "features"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "group": r["group"],
                    "val_acc": f"{r['val_acc']:.8f}",
                    "test_acc": f"{r['test_acc']:.8f}",
                    "feature_ids": json.dumps(r["feature_ids"]),
                    "features": json.dumps(r["features"]),
                }
            )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_group_by_val_acc": best_name,
                "best_val_acc": best_val,
                "best_test_acc_for_best_val_group": best_test,
                "groups": rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

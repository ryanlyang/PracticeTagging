#!/usr/bin/env python3
"""Audit why JetClass stacked-logistic fusion gives large singleton gains.

This script consumes saved fusion_scores.npz files and never reruns neural
inference. It treats each base model identically and asks:

- Does raw argmax in the NPZ match the reported individual accuracy?
- Is the correct class usually in top-2/top-3?
- Can a hard class remap explain the gain?
- How much gain comes from a restricted per-class calibrator vs full logreg?
- Do permuted-label / row-shuffled controls collapse?
- Are label arrays identical across compared fusion reports?

The point is to distinguish true richer logits from stacker misuse/bookkeeping.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy should be available in atlas_kd
    linear_sum_assignment = None


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)


def _topk_acc(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argpartition(probs, kth=probs.shape[1] - k, axis=1)[:, -k:]
    return float(np.mean([int(y[i]) in set(order[i].tolist()) for i in range(len(y))]))


def _infer_names(files: Iterable[str]) -> List[str]:
    fs = set(files)
    out = []
    for k in fs:
        if not k.startswith("logits_val_"):
            continue
        name = k[len("logits_val_") :]
        required = [
            f"logits_test_{name}",
            f"probs_val_{name}",
            f"probs_test_{name}",
        ]
        if all(x in fs for x in required):
            out.append(name)
    return sorted(set(out))


def _features(d: np.lib.npyio.NpzFile, name: str, split: str, mode: str) -> np.ndarray:
    parts: List[np.ndarray] = []
    if mode in {"logits", "logits_probs"}:
        parts.append(np.asarray(d[f"logits_{split}_{name}"], dtype=np.float32))
    if mode in {"probs", "logits_probs"}:
        parts.append(np.asarray(d[f"probs_{split}_{name}"], dtype=np.float32))
    if not parts:
        raise ValueError(f"Unknown feature mode: {mode}")
    return np.concatenate(parts, axis=1).astype(np.float32)


def _subset_indices(n: int, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or max_rows >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(max_rows), replace=False))


def _fit_full_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    cv: int,
    cs: Sequence[float],
    max_iter: int,
    n_jobs: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegressionCV(
            Cs=[float(c) for c in cs],
            cv=int(cv),
            solver="lbfgs",
            scoring="accuracy",
            max_iter=int(max_iter),
            n_jobs=int(n_jobs),
            refit=True,
        ),
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_eval).astype(np.float32)
    lr = clf.named_steps["logisticregressioncv"]
    info = {
        "C_mean": float(np.mean(lr.C_)) if getattr(lr, "C_", np.array([])).size else float("nan"),
        "n_features": int(x_train.shape[1]),
    }
    return probs, info


def _fit_diag_ovr(
    logits_train: np.ndarray,
    probs_train: np.ndarray,
    y_train: np.ndarray,
    logits_eval: np.ndarray,
    probs_eval: np.ndarray,
    *,
    max_iter: int,
) -> np.ndarray:
    """Restricted calibrator: class c can only see logit_c and prob_c.

    This is still supervised, but it cannot learn cross-class rules like
    "if H4q first and Hbb second, predict Hbb". It separates calibration from
    full reclassification.
    """
    n_classes = int(probs_train.shape[1])
    scores = np.zeros((logits_eval.shape[0], n_classes), dtype=np.float32)
    for c in range(n_classes):
        x_tr = np.stack([logits_train[:, c], probs_train[:, c]], axis=1).astype(np.float32)
        x_ev = np.stack([logits_eval[:, c], probs_eval[:, c]], axis=1).astype(np.float32)
        y_bin = (y_train == c).astype(np.int64)
        if len(np.unique(y_bin)) < 2:
            scores[:, c] = -1e6
            continue
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=int(max_iter)),
        )
        clf.fit(x_tr, y_bin)
        scores[:, c] = clf.decision_function(x_ev).astype(np.float32)
    return _softmax(scores)


def _hard_remap_acc(y_val: np.ndarray, pred_val: np.ndarray, y_test: np.ndarray, pred_test: np.ndarray) -> Tuple[float, Dict[int, int]]:
    if linear_sum_assignment is None:
        return float("nan"), {}
    n_classes = int(max(y_val.max(), pred_val.max(), y_test.max(), pred_test.max()) + 1)
    cm = confusion_matrix(y_val, pred_val, labels=np.arange(n_classes))
    row, col = linear_sum_assignment(-cm)
    mapping = {int(c): int(r) for r, c in zip(row, col)}
    remapped = np.array([mapping.get(int(p), int(p)) for p in pred_test], dtype=np.int64)
    return float(accuracy_score(y_test, remapped)), mapping


@dataclass
class DatasetSpec:
    label: str
    scores: Path
    report: Optional[Path]


def _load_report(path: Optional[Path]) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _report_acc(report: Dict[str, object], name: str) -> Optional[float]:
    try:
        return float(report["individual_metrics"][name]["test"]["acc"])
    except Exception:
        return None


def _audit_dataset(
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, np.ndarray]]:
    d = np.load(spec.scores)
    report = _load_report(spec.report)
    names = _infer_names(d.files)
    if args.model_subset:
        wanted = [x.strip() for x in args.model_subset.split(",") if x.strip()]
        names = [n for n in names if n in wanted]
    if not names:
        raise ValueError(f"No models found in {spec.scores}")

    y_val_full = np.asarray(d["y_val"], dtype=np.int64)
    y_test_full = np.asarray(d["y_test"], dtype=np.int64)
    tr_idx = _subset_indices(len(y_val_full), int(args.max_train_rows), int(args.seed))
    te_idx = _subset_indices(len(y_test_full), int(args.max_test_rows), int(args.seed) + 1)
    y_val = y_val_full[tr_idx]
    y_test = y_test_full[te_idx]

    rows: List[Dict[str, object]] = []
    control_rows: List[Dict[str, object]] = []

    print(f"\nDataset: {spec.label}")
    print(f"  scores: {spec.scores}")
    print(f"  models ({len(names)}): {', '.join(names)}")
    print(f"  train rows: {len(y_val)} / {len(y_val_full)} | test rows: {len(y_test)} / {len(y_test_full)}")

    for name in names:
        pv_full = np.asarray(d[f"probs_val_{name}"], dtype=np.float32)
        pt_full = np.asarray(d[f"probs_test_{name}"], dtype=np.float32)
        lv_full = np.asarray(d[f"logits_val_{name}"], dtype=np.float32)
        lt_full = np.asarray(d[f"logits_test_{name}"], dtype=np.float32)
        pv = pv_full[tr_idx]
        pt = pt_full[te_idx]
        lv = lv_full[tr_idx]
        lt = lt_full[te_idx]

        pred_val = pv.argmax(axis=1)
        pred_test = pt.argmax(axis=1)
        raw_val = float(accuracy_score(y_val, pred_val))
        raw_test = float(accuracy_score(y_test, pred_test))
        hard_test, mapping = _hard_remap_acc(y_val, pred_val, y_test, pred_test)
        diag_probs = _fit_diag_ovr(lv, pv, y_val, lt, pt, max_iter=int(args.diag_max_iter))
        diag_test = float(accuracy_score(y_test, diag_probs.argmax(axis=1)))

        row: Dict[str, object] = {
            "dataset": spec.label,
            "model": name,
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "report_test_acc": _report_acc(report, name),
            "raw_val_acc": raw_val,
            "raw_test_acc": raw_test,
            "raw_test_nll": float(log_loss(y_test, pt, labels=np.arange(pt.shape[1]))),
            "top2_test_acc": _topk_acc(pt, y_test, 2),
            "top3_test_acc": _topk_acc(pt, y_test, 3),
            "hard_remap_test_acc": hard_test,
            "hard_remap_is_identity": bool(all(mapping.get(i, i) == i for i in range(pt.shape[1]))),
            "diag_ovr_test_acc": diag_test,
            "diag_ovr_gain": diag_test - raw_test,
        }

        for mode in args.feature_modes:
            xv = _features(d, name, "val", mode)[tr_idx]
            xt = _features(d, name, "test", mode)[te_idx]
            p_stack, info = _fit_full_logreg(
                xv,
                y_val,
                xt,
                cv=int(args.cv),
                cs=args.Cs,
                max_iter=int(args.max_iter),
                n_jobs=int(args.n_jobs),
            )
            acc = float(accuracy_score(y_test, p_stack.argmax(axis=1)))
            row[f"full_{mode}_test_acc"] = acc
            row[f"full_{mode}_gain"] = acc - raw_test
            row[f"full_{mode}_C_mean"] = info["C_mean"]

        rows.append(row)
        print(
            f"  {name:20s} raw={raw_test:.6f} top2={row['top2_test_acc']:.6f} "
            f"diag={diag_test:.6f} full_lp={row.get('full_logits_probs_test_acc', float('nan')):.6f} "
            f"gain={row.get('full_logits_probs_gain', float('nan')):+.6f}"
        )

    # Negative controls for the highest-gain model and any HLT-looking model.
    def gain_key(r: Dict[str, object]) -> float:
        return float(r.get("full_logits_probs_gain", float("nan")))

    candidates = []
    hlt_like = [r for r in rows if "hlt" in str(r["model"]).lower()]
    if hlt_like:
        candidates.append(hlt_like[0])
    candidates.append(max(rows, key=gain_key))
    seen = set()
    rng = np.random.default_rng(int(args.seed) + 77)
    for base in candidates:
        name = str(base["model"])
        key = (spec.label, name)
        if key in seen:
            continue
        seen.add(key)
        xv = _features(d, name, "val", "logits_probs")[tr_idx]
        xt = _features(d, name, "test", "logits_probs")[te_idx]
        ctrl_rows = min(int(args.control_rows), len(y_val)) if int(args.control_rows) > 0 else len(y_val)
        cidx = _subset_indices(len(y_val), ctrl_rows, int(args.seed) + 99)
        xctrl = xv[cidx]
        yctrl = y_val[cidx].copy()

        y_perm = yctrl.copy()
        rng.shuffle(y_perm)
        p_perm, _ = _fit_full_logreg(xctrl, y_perm, xt, cv=max(2, min(int(args.cv), 3)), cs=args.Cs, max_iter=int(args.max_iter), n_jobs=int(args.n_jobs))
        control_rows.append(
            {
                "dataset": spec.label,
                "model": name,
                "control": "permuted_labels",
                "train_rows": int(len(y_perm)),
                "test_acc": float(accuracy_score(y_test, p_perm.argmax(axis=1))),
            }
        )

        x_shuf = xctrl.copy()
        rng.shuffle(x_shuf, axis=0)
        p_shuf, _ = _fit_full_logreg(x_shuf, yctrl, xt, cv=max(2, min(int(args.cv), 3)), cs=args.Cs, max_iter=int(args.max_iter), n_jobs=int(args.n_jobs))
        control_rows.append(
            {
                "dataset": spec.label,
                "model": name,
                "control": "row_shuffled_features",
                "train_rows": int(len(yctrl)),
                "test_acc": float(accuracy_score(y_test, p_shuf.argmax(axis=1))),
            }
        )
        print(
            f"    controls {name}: perm={control_rows[-2]['test_acc']:.6f} "
            f"row_shuffle={control_rows[-1]['test_acc']:.6f}"
        )

    labels = {"y_val": y_val_full, "y_test": y_test_full}
    d.close()
    return rows, control_rows, labels


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _existing_spec(label: str, scores: str, report: str) -> Optional[DatasetSpec]:
    p = Path(scores)
    if not p.exists():
        print(f"[skip] {label}: missing {p}")
        return None
    rp = Path(report) if report else None
    return DatasetSpec(label=label, scores=p, report=rp)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary_scores", default="checkpoints/jetclass_joint_dualview/fusion_reports/samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/fusion_scores.npz")
    p.add_argument("--primary_report", default="checkpoints/jetclass_joint_dualview/fusion_reports/samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/report.json")
    p.add_argument("--hlt5_scores", default="checkpoints/jetclass_hlt_seed_ensemble/fusion_reports/hlt5_1m250k1m_fixedhlt_stacked_acc/fusion_scores.npz")
    p.add_argument("--hlt5_report", default="checkpoints/jetclass_hlt_seed_ensemble/fusion_reports/hlt5_1m250k1m_fixedhlt_stacked_acc/report.json")
    p.add_argument("--legacy12_scores", default="checkpoints/jetclass_joint_dualview/fusion_reports/twelve_model_1m250k1m_m2hybrid_stacked_acc/fusion_scores.npz")
    p.add_argument("--legacy12_report", default="checkpoints/jetclass_joint_dualview/fusion_reports/twelve_model_1m250k1m_m2hybrid_stacked_acc/report.json")
    p.add_argument("--out_dir", default="checkpoints/jetclass_joint_dualview/fusion_reports/stacker_singleton_behavior_audit")
    p.add_argument("--feature_modes", nargs="+", default=["logits_probs", "logits", "probs"], choices=["logits_probs", "logits", "probs"])
    p.add_argument("--model_subset", default="", help="Optional comma-separated model names to audit within each NPZ.")
    p.add_argument("--max_train_rows", type=int, default=0, help="0 means use all stack-train/val rows saved in NPZ.")
    p.add_argument("--max_test_rows", type=int, default=0, help="0 means use all test rows saved in NPZ.")
    p.add_argument("--control_rows", type=int, default=120000)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--Cs", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--max_iter", type=int, default=1200)
    p.add_argument("--diag_max_iter", type=int, default=500)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=52)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        _existing_spec("primary_samehlt7_plus_hlt", args.primary_scores, args.primary_report),
        _existing_spec("hlt5_seed_control", args.hlt5_scores, args.hlt5_report),
        _existing_spec("legacy12_m2hybrid", args.legacy12_scores, args.legacy12_report),
    ]
    specs = [s for s in specs if s is not None]
    if not specs:
        raise SystemExit("No input score files found.")

    all_rows: List[Dict[str, object]] = []
    all_controls: List[Dict[str, object]] = []
    labels_by_dataset: Dict[str, Dict[str, np.ndarray]] = {}
    for spec in specs:
        rows, controls, labels = _audit_dataset(spec, args)
        all_rows.extend(rows)
        all_controls.extend(controls)
        labels_by_dataset[spec.label] = labels

    label_checks: List[Dict[str, object]] = []
    labels_items = list(labels_by_dataset.items())
    for i in range(len(labels_items)):
        for j in range(i + 1, len(labels_items)):
            a_name, a_lab = labels_items[i]
            b_name, b_lab = labels_items[j]
            for split in ("y_val", "y_test"):
                a = a_lab[split]
                b = b_lab[split]
                same_len = len(a) == len(b)
                label_checks.append(
                    {
                        "dataset_a": a_name,
                        "dataset_b": b_name,
                        "split": split,
                        "same_length": bool(same_len),
                        "len_a": int(len(a)),
                        "len_b": int(len(b)),
                        "arrays_equal": bool(same_len and np.array_equal(a, b)),
                        "prefix_equal_len": int(min(len(a), len(b))),
                        "prefix_equal_fraction": float(np.mean(a[: min(len(a), len(b))] == b[: min(len(a), len(b))])) if min(len(a), len(b)) else float("nan"),
                    }
                )

    _write_csv(out_dir / "singleton_audit_rows.csv", all_rows)
    _write_csv(out_dir / "control_rows.csv", all_controls)
    _write_csv(out_dir / "label_array_checks.csv", label_checks)

    report = {
        "args": vars(args),
        "rows": all_rows,
        "controls": all_controls,
        "label_array_checks": label_checks,
    }
    (out_dir / "stacker_singleton_audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    print("\nTop singleton full-logits_probs gains:")
    for r in sorted(all_rows, key=lambda x: float(x.get("full_logits_probs_gain", float("nan"))), reverse=True)[:20]:
        print(
            f"  {r['dataset']:28s} {r['model']:20s} raw={float(r['raw_test_acc']):.6f} "
            f"top2={float(r['top2_test_acc']):.6f} diag={float(r['diag_ovr_test_acc']):.6f} "
            f"full={float(r.get('full_logits_probs_test_acc', float('nan'))):.6f} "
            f"gain={float(r.get('full_logits_probs_gain', float('nan'))):+.6f}"
        )

    print("\nControls:")
    for c in all_controls:
        print(f"  {c['dataset']:28s} {c['model']:20s} {c['control']:22s} acc={float(c['test_acc']):.6f}")

    print("\nLabel array checks:")
    for c in label_checks:
        print(
            f"  {c['dataset_a']} vs {c['dataset_b']} {c['split']}: "
            f"same_len={c['same_length']} arrays_equal={c['arrays_equal']} "
            f"prefix_equal_fraction={float(c['prefix_equal_fraction']):.6f}"
        )

    print(f"\nSaved audit: {out_dir / 'stacker_singleton_audit_report.json'}")
    print(f"Saved rows:  {out_dir / 'singleton_audit_rows.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def _fpr_at_target_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    y = np.asarray(y).astype(np.int64)
    p = _clip_probs(p)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    idx = int(np.argmin(np.abs(tpr - float(target_tpr))))
    return float(fpr[idx])


def _score_metrics(y: np.ndarray, s: np.ndarray, hlt_ref: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(np.int64)
    s = _clip_probs(s)
    h = _clip_probs(hlt_ref)
    out = {
        "auc": float(roc_auc_score(y, s)) if np.unique(y).size > 1 else float("nan"),
        "fpr30": _fpr_at_target_tpr(y, s, 0.30),
        "fpr50": _fpr_at_target_tpr(y, s, 0.50),
    }
    bce_s = -(y * np.log(s) + (1.0 - y) * np.log(1.0 - s))
    bce_h = -(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))
    db = bce_s - bce_h
    out["harm_bce_frac_all_vs_hlt"] = float(np.mean(db > 0.0))
    neg = (y == 0)
    out["harm_bce_frac_neg_vs_hlt"] = float(np.mean(db[neg] > 0.0)) if np.any(neg) else float("nan")
    return out


def _save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        path.write_text("")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _hard_route(g: np.ndarray, p_joint: np.ndarray, p_hlt: np.ndarray, threshold: float, direction: str) -> Tuple[np.ndarray, np.ndarray]:
    g = np.asarray(g, dtype=np.float64)
    if direction == ">=":
        m = g >= float(threshold)
    elif direction == "<=":
        m = g <= float(threshold)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    s = np.where(m, p_joint, p_hlt)
    return s, m


def main() -> None:
    ap = argparse.ArgumentParser(description="Posthoc hard-route analysis for oracle-route gate MoE outputs")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--scores_npz", type=str, default="")
    ap.add_argument("--objective", type=str, default="fpr50", choices=["fpr50", "auc"])
    ap.add_argument("--directions", type=str, default="both", choices=["both", ">=", "<="])
    ap.add_argument("--num_thresholds", type=int, default=2001)
    ap.add_argument("--topk_sweep_rows", type=int, default=200)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    scores_npz = (
        Path(args.scores_npz).expanduser().resolve()
        if str(args.scores_npz).strip()
        else (run_dir / "router_gate_scores_val_test.npz")
    )
    if not scores_npz.exists():
        raise FileNotFoundError(f"scores npz not found: {scores_npz}")

    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (run_dir / "router_gate_posthoc_hardroute")
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(scores_npz)
    required = [
        "val_labels", "val_hlt", "val_joint", "val_fused", "val_gate",
        "test_labels", "test_hlt", "test_joint", "test_fused", "test_gate",
    ]
    miss = [k for k in required if k not in z]
    if miss:
        raise RuntimeError(f"Missing keys in {scores_npz}: {miss}")

    y_val = z["val_labels"].astype(np.int64)
    ph_val = _clip_probs(z["val_hlt"])
    pj_val = _clip_probs(z["val_joint"])
    pf_val = _clip_probs(z["val_fused"])
    g_val = np.asarray(z["val_gate"], dtype=np.float64)

    y_test = z["test_labels"].astype(np.int64)
    ph_test = _clip_probs(z["test_hlt"])
    pj_test = _clip_probs(z["test_joint"])
    pf_test = _clip_probs(z["test_fused"])
    g_test = np.asarray(z["test_gate"], dtype=np.float64)

    # Threshold grid includes edges to allow all-HLT/all-Joint routing.
    nqt = max(17, int(args.num_thresholds))
    qs = np.linspace(0.0, 1.0, nqt)
    thr = np.quantile(g_val, qs)
    thr = np.unique(np.concatenate([
        np.array([float(np.min(g_val)) - 1e-8], dtype=np.float64),
        thr.astype(np.float64),
        np.array([float(np.max(g_val)) + 1e-8], dtype=np.float64),
    ]))

    dirs = [args.directions] if args.directions in (">=", "<=") else [">=", "<="]

    sweep_rows: List[Dict[str, object]] = []
    best = None

    for d in dirs:
        for t in thr:
            s_val, m_val = _hard_route(g_val, pj_val, ph_val, float(t), d)
            mv = _score_metrics(y_val, s_val, ph_val)
            row = {
                "direction": d,
                "threshold": float(t),
                "joint_route_frac_val": float(np.mean(m_val)),
                "val_auc": float(mv["auc"]),
                "val_fpr30": float(mv["fpr30"]),
                "val_fpr50": float(mv["fpr50"]),
            }
            sweep_rows.append(row)

            if best is None:
                best = row
            else:
                if str(args.objective) == "auc":
                    key = (row["val_auc"], -row["val_fpr50"], -row["val_fpr30"])
                    key_b = (best["val_auc"], -best["val_fpr50"], -best["val_fpr30"])
                    if key > key_b:
                        best = row
                else:
                    key = (-row["val_fpr50"], row["val_auc"], -row["val_fpr30"])
                    key_b = (-best["val_fpr50"], best["val_auc"], -best["val_fpr30"])
                    if key > key_b:
                        best = row

    if best is None:
        raise RuntimeError("No threshold candidates evaluated")

    # Evaluate selected hard route on test.
    s_val_best, route_mask_val = _hard_route(g_val, pj_val, ph_val, float(best["threshold"]), str(best["direction"]))
    s_test_best, route_mask_test = _hard_route(g_test, pj_test, ph_test, float(best["threshold"]), str(best["direction"]))

    m_val_best = _score_metrics(y_val, s_val_best, ph_val)
    m_test_best = _score_metrics(y_test, s_test_best, ph_test)

    # Baselines.
    base_val = {
        "hlt": _score_metrics(y_val, ph_val, ph_val),
        "joint": _score_metrics(y_val, pj_val, ph_val),
        "soft_fused": _score_metrics(y_val, pf_val, ph_val),
        "hard_05": _score_metrics(y_val, np.where(g_val >= 0.5, pj_val, ph_val), ph_val),
    }
    base_test = {
        "hlt": _score_metrics(y_test, ph_test, ph_test),
        "joint": _score_metrics(y_test, pj_test, ph_test),
        "soft_fused": _score_metrics(y_test, pf_test, ph_test),
        "hard_05": _score_metrics(y_test, np.where(g_test >= 0.5, pj_test, ph_test), ph_test),
    }

    # Sort sweep rows by chosen objective.
    if str(args.objective) == "auc":
        sweep_rows_sorted = sorted(sweep_rows, key=lambda r: (r["val_auc"], -r["val_fpr50"], -r["val_fpr30"]), reverse=True)
    else:
        sweep_rows_sorted = sorted(sweep_rows, key=lambda r: (r["val_fpr50"], -r["val_auc"], r["val_fpr30"]))

    topk = max(1, int(args.topk_sweep_rows))
    _save_csv(out_dir / "hardroute_threshold_sweep_val.csv", sweep_rows_sorted[:topk])

    report = {
        "run_dir": str(run_dir),
        "scores_npz": str(scores_npz),
        "objective": str(args.objective),
        "directions": dirs,
        "num_thresholds": int(nqt),
        "selected": {
            "direction": str(best["direction"]),
            "threshold": float(best["threshold"]),
            "joint_route_frac_val": float(np.mean(route_mask_val)),
            "joint_route_frac_test": float(np.mean(route_mask_test)),
        },
        "val": {
            "selected_hard_route": m_val_best,
            "baselines": base_val,
        },
        "test": {
            "selected_hard_route": m_test_best,
            "baselines": base_test,
        },
        "files": {
            "sweep_csv": str((out_dir / "hardroute_threshold_sweep_val.csv").resolve()),
        },
    }

    rep_path = Path(args.report_json).expanduser().resolve() if str(args.report_json).strip() else (out_dir / "hardroute_posthoc_report.json")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with rep_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("Posthoc Hard-Route Analysis")
    print("=" * 72)
    print(f"Run dir: {run_dir}")
    print(f"Scores: {scores_npz}")
    print(f"Selected: direction={best['direction']} threshold={best['threshold']:.8f}")
    print(f"Val  route_frac={np.mean(route_mask_val):.4f} | AUC={m_val_best['auc']:.6f} FPR30={m_val_best['fpr30']:.6f} FPR50={m_val_best['fpr50']:.6f}")
    print(f"Test route_frac={np.mean(route_mask_test):.4f} | AUC={m_test_best['auc']:.6f} FPR30={m_test_best['fpr30']:.6f} FPR50={m_test_best['fpr50']:.6f}")
    print()
    print(
        "Test baselines: "
        f"HLT AUC={base_test['hlt']['auc']:.6f} FPR50={base_test['hlt']['fpr50']:.6f} | "
        f"Joint AUC={base_test['joint']['auc']:.6f} FPR50={base_test['joint']['fpr50']:.6f} | "
        f"SoftFused AUC={base_test['soft_fused']['auc']:.6f} FPR50={base_test['soft_fused']['fpr50']:.6f}"
    )
    print()
    print(f"Saved report: {rep_path}")
    print(f"Saved sweep CSV: {out_dir / 'hardroute_threshold_sweep_val.csv'}")


if __name__ == "__main__":
    main()

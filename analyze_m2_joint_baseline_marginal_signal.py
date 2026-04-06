#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int64)
    p = _clip_probs(p)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _fpr_at_target_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    y = np.asarray(y).astype(np.int64)
    p = _clip_probs(p)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    idx = int(np.argmin(np.abs(tpr - float(target_tpr))))
    return float(fpr[idx])


def _threshold_for_target_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    y = np.asarray(y).astype(np.int64)
    p = _clip_probs(p)
    if y.size == 0 or np.unique(y).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    idx = np.where(tpr >= float(target_tpr))[0]
    if idx.size == 0:
        return float(thr[-1])
    # Highest threshold that still reaches target TPR.
    return float(thr[idx[0]])


def _bce_per_jet(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    p = _clip_probs(p)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _metrics(y: np.ndarray, p: np.ndarray, target_tprs: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {"auc": _safe_auc(y, p)}
    for t in target_tprs:
        out[f"fpr_at_tpr_{int(round(t * 100))}"] = _fpr_at_target_tpr(y, p, t)
    return out


def _build_gate_features(p_hlt: np.ndarray, p_joint: np.ndarray) -> np.ndarray:
    p_hlt = _clip_probs(p_hlt)
    p_joint = _clip_probs(p_joint)
    diff = p_joint - p_hlt
    abs_diff = np.abs(diff)
    conf_hlt = np.abs(p_hlt - 0.5) * 2.0
    conf_joint = np.abs(p_joint - 0.5) * 2.0
    return np.stack(
        [
            p_hlt,
            p_joint,
            diff,
            abs_diff,
            conf_hlt,
            conf_joint,
            conf_joint - conf_hlt,
            p_hlt * p_joint,
        ],
        axis=1,
    )


def _transition_counts(
    y_test: np.ndarray,
    p_hlt_test: np.ndarray,
    p_joint_test: np.ndarray,
    y_val: np.ndarray,
    p_hlt_val: np.ndarray,
    p_joint_val: np.ndarray,
    target_tpr: float,
) -> Dict[str, float]:
    thr_hlt = _threshold_for_target_tpr(y_val, p_hlt_val, target_tpr)
    thr_joint = _threshold_for_target_tpr(y_val, p_joint_val, target_tpr)

    pred_hlt = p_hlt_test >= thr_hlt
    pred_joint = p_joint_test >= thr_joint

    neg = y_test == 0
    pos = y_test == 1

    fp_hlt = pred_hlt & neg
    fp_joint = pred_joint & neg
    tp_hlt = pred_hlt & pos
    tp_joint = pred_joint & pos

    out = {
        "target_tpr": float(target_tpr),
        "thr_hlt_val": float(thr_hlt),
        "thr_joint_val": float(thr_joint),
        "n_neg_test": int(np.sum(neg)),
        "n_pos_test": int(np.sum(pos)),
        "fp_hlt_only": int(np.sum(fp_hlt & (~fp_joint))),
        "fp_joint_only": int(np.sum(fp_joint & (~fp_hlt))),
        "fp_both": int(np.sum(fp_hlt & fp_joint)),
        "fp_neither": int(np.sum((~fp_hlt) & (~fp_joint) & neg)),
        "tp_hlt_only": int(np.sum(tp_hlt & (~tp_joint))),
        "tp_joint_only": int(np.sum(tp_joint & (~tp_hlt))),
        "tp_both": int(np.sum(tp_hlt & tp_joint)),
        "tp_neither": int(np.sum((~tp_hlt) & (~tp_joint) & pos)),
    }
    return out


def _bin_report(
    y: np.ndarray,
    p_hlt: np.ndarray,
    p_joint: np.ndarray,
    delta_bce: np.ndarray,
    bin_values: np.ndarray,
    edges: np.ndarray,
    bin_name: str,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    idx = np.digitize(bin_values, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges) - 2)

    for b in range(len(edges) - 1):
        m = idx == b
        n = int(np.sum(m))
        if n == 0:
            row = {
                "bin": b,
                f"{bin_name}_lo": float(edges[b]),
                f"{bin_name}_hi": float(edges[b + 1]),
                "n": 0,
                "frac": 0.0,
                "mean_delta_bce_joint_minus_hlt": float("nan"),
                "help_frac": float("nan"),
                "hurt_frac": float("nan"),
                "auc_hlt": float("nan"),
                "auc_joint": float("nan"),
            }
            rows.append(row)
            continue

        d = delta_bce[m]
        row = {
            "bin": b,
            f"{bin_name}_lo": float(edges[b]),
            f"{bin_name}_hi": float(edges[b + 1]),
            "n": n,
            "frac": float(n / max(len(y), 1)),
            "mean_delta_bce_joint_minus_hlt": float(np.mean(d)),
            "help_frac": float(np.mean(d < 0.0)),
            "hurt_frac": float(np.mean(d > 0.0)),
            "auc_hlt": _safe_auc(y[m], p_hlt[m]),
            "auc_joint": _safe_auc(y[m], p_joint[m]),
        }
        rows.append(row)
    return rows


def _save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Marginal signal diagnostics for HLT baseline vs joint dual-view.")
    ap.add_argument("--run_dir", type=str, default="", help="Run dir containing fusion_scores_val_test.npz")
    ap.add_argument("--scores_npz", type=str, default="", help="Path to fusion_scores_val_test.npz")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--target_tprs", type=str, default="0.30,0.50")
    ap.add_argument("--n_conf_bins", type=int, default=10)
    ap.add_argument("--n_disagree_bins", type=int, default=10)
    ap.add_argument("--gate_c", type=float, default=0.2)
    ap.add_argument("--disable_learned_gate", action="store_true")
    ap.add_argument("--save_per_jet_npz", action="store_true")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve() if str(args.run_dir).strip() else None
    if str(args.scores_npz).strip():
        npz_path = Path(args.scores_npz).expanduser().resolve()
    else:
        if run_dir is None:
            raise ValueError("Provide --run_dir or --scores_npz")
        npz_path = run_dir / "fusion_scores_val_test.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Score file not found: {npz_path}")

    if str(args.out_dir).strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        if run_dir is None:
            out_dir = npz_path.parent / "marginal_signal"
        else:
            out_dir = run_dir / "marginal_signal"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_tprs = [float(x) for x in str(args.target_tprs).split(",") if str(x).strip()]
    if not target_tprs:
        target_tprs = [0.30, 0.50]

    z = np.load(npz_path)
    split = str(args.split).strip().lower()

    y = z[f"labels_{split}"].astype(np.int64)
    p_hlt = _clip_probs(z[f"preds_hlt_{split}"])
    p_joint = _clip_probs(z[f"preds_joint_{split}"])

    y_val = z["labels_val"].astype(np.int64)
    p_hlt_val = _clip_probs(z["preds_hlt_val"])
    p_joint_val = _clip_probs(z["preds_joint_val"])

    bce_hlt = _bce_per_jet(y, p_hlt)
    bce_joint = _bce_per_jet(y, p_joint)
    delta_bce = bce_joint - bce_hlt

    help_mask = delta_bce < 0.0
    hurt_mask = delta_bce > 0.0
    tie_mask = ~(help_mask | hurt_mask)

    p_oracle = np.where(help_mask, p_joint, p_hlt)

    metrics = {
        "hlt": _metrics(y, p_hlt, target_tprs),
        "joint": _metrics(y, p_joint, target_tprs),
        "oracle_best_of_two": _metrics(y, p_oracle, target_tprs),
    }

    summary = {
        "n_jets": int(len(y)),
        "help_frac_joint_better": float(np.mean(help_mask)),
        "hurt_frac_joint_worse": float(np.mean(hurt_mask)),
        "tie_frac": float(np.mean(tie_mask)),
        "mean_delta_bce_joint_minus_hlt": float(np.mean(delta_bce)),
        "mean_help_bce_reduction": float(-np.mean(delta_bce[help_mask])) if np.any(help_mask) else float("nan"),
        "mean_hurt_bce_increase": float(np.mean(delta_bce[hurt_mask])) if np.any(hurt_mask) else float("nan"),
    }

    gate_report: Dict[str, Dict[str, float]] = {}
    gate_outputs: Dict[str, np.ndarray] = {}
    if not bool(args.disable_learned_gate):
        y_gate_val = (_bce_per_jet(y_val, p_joint_val) < _bce_per_jet(y_val, p_hlt_val)).astype(np.int64)
        if np.unique(y_gate_val).size >= 2:
            X_val = _build_gate_features(p_hlt_val, p_joint_val)
            X = _build_gate_features(p_hlt, p_joint)
            clf = LogisticRegression(C=float(args.gate_c), max_iter=2000, class_weight="balanced", solver="lbfgs")
            clf.fit(X_val, y_gate_val)
            pj = clf.predict_proba(X)[:, 1]
            sel_joint = pj >= 0.5
            p_gate_hard = np.where(sel_joint, p_joint, p_hlt)
            p_gate_soft = pj * p_joint + (1.0 - pj) * p_hlt

            gate_report["learned_gate_hard"] = _metrics(y, p_gate_hard, target_tprs)
            gate_report["learned_gate_softmix"] = _metrics(y, p_gate_soft, target_tprs)
            gate_outputs["gate_prob_joint"] = pj.astype(np.float64)
            gate_outputs["gate_select_joint_hard"] = sel_joint.astype(np.int8)
        else:
            gate_report["learned_gate"] = {"auc": float("nan")}

    transitions = {
        f"tpr{int(round(t * 100))}": _transition_counts(y, p_hlt, p_joint, y_val, p_hlt_val, p_joint_val, t)
        for t in target_tprs
    }

    conf_hlt = np.abs(p_hlt - 0.5) * 2.0
    disagree = np.abs(p_joint - p_hlt)
    conf_edges = np.linspace(0.0, 1.0, int(max(args.n_conf_bins, 2)) + 1)
    dis_edges = np.linspace(0.0, 1.0, int(max(args.n_disagree_bins, 2)) + 1)

    conf_rows = _bin_report(y, p_hlt, p_joint, delta_bce, conf_hlt, conf_edges, "conf_hlt")
    dis_rows = _bin_report(y, p_hlt, p_joint, delta_bce, disagree, dis_edges, "abs_disagree")

    _save_csv(out_dir / f"{split}_by_hlt_confidence.csv", conf_rows)
    _save_csv(out_dir / f"{split}_by_hlt_joint_disagreement.csv", dis_rows)

    report = {
        "scores_npz": str(npz_path),
        "split": split,
        "target_tprs": target_tprs,
        "metrics": metrics,
        "summary": summary,
        "gate_report": gate_report,
        "transitions": transitions,
        "artifacts": {
            "confidence_csv": str((out_dir / f"{split}_by_hlt_confidence.csv").resolve()),
            "disagreement_csv": str((out_dir / f"{split}_by_hlt_joint_disagreement.csv").resolve()),
        },
    }

    report_path = Path(args.report_json).expanduser().resolve() if str(args.report_json).strip() else (out_dir / f"{split}_marginal_signal_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    if bool(args.save_per_jet_npz):
        np.savez_compressed(
            out_dir / f"{split}_marginal_signal_perjet.npz",
            labels=y.astype(np.int8),
            preds_hlt=p_hlt.astype(np.float64),
            preds_joint=p_joint.astype(np.float64),
            bce_hlt=bce_hlt.astype(np.float64),
            bce_joint=bce_joint.astype(np.float64),
            delta_bce_joint_minus_hlt=delta_bce.astype(np.float64),
            help_mask=help_mask.astype(np.int8),
            hurt_mask=hurt_mask.astype(np.int8),
            oracle_choose_joint=help_mask.astype(np.int8),
            oracle_best_of_two_preds=p_oracle.astype(np.float64),
            **gate_outputs,
        )

    print("=" * 60)
    print("Marginal Signal Report")
    print("=" * 60)
    print(f"scores_npz: {npz_path}")
    print(f"split: {split}")
    print(f"n_jets: {summary['n_jets']}")
    print(
        "help/hurt/tie frac: "
        f"{summary['help_frac_joint_better']:.4f} / "
        f"{summary['hurt_frac_joint_worse']:.4f} / "
        f"{summary['tie_frac']:.4f}"
    )
    print(
        "mean delta BCE (joint-hlt): "
        f"{summary['mean_delta_bce_joint_minus_hlt']:.6f} "
        f"(help mean={summary['mean_help_bce_reduction']:.6f}, "
        f"hurt mean={summary['mean_hurt_bce_increase']:.6f})"
    )

    for name, m in metrics.items():
        print(f"[{name}] AUC={m['auc']:.6f}")
        for t in target_tprs:
            k = f"fpr_at_tpr_{int(round(t * 100))}"
            print(f"  FPR@TPR{int(round(t*100))}: {m.get(k, float('nan')):.6f}")

    if gate_report:
        for name, m in gate_report.items():
            print(f"[{name}] AUC={m.get('auc', float('nan')):.6f}")
            for t in target_tprs:
                k = f"fpr_at_tpr_{int(round(t * 100))}"
                if k in m:
                    print(f"  FPR@TPR{int(round(t*100))}: {m[k]:.6f}")

    print("Saved:")
    print(f"  {report_path}")
    print(f"  {out_dir / f'{split}_by_hlt_confidence.csv'}")
    print(f"  {out_dir / f'{split}_by_hlt_joint_disagreement.csv'}")


if __name__ == "__main__":
    main()

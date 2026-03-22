#!/usr/bin/env python3
"""
Posthoc fusion analysis between normal-joint and specialist-joint predictions.

Given saved results.npz files (with labels + prediction vectors), this script:
- computes standalone metrics at target TPR
- computes TP overlap at target TPR (default 50%)
- evaluates multiple fusion families and reports best FPR@target
- exports detailed tables for further analysis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-8


def prob_threshold_at_target_tpr(pos_probs: np.ndarray, target_tpr: float) -> float:
    if pos_probs.size == 0:
        return 0.5
    q = float(np.clip(1.0 - float(target_tpr), 0.0, 1.0))
    return float(np.quantile(pos_probs, q))


def fpr_at_target_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    tgt = float(np.clip(target_tpr, 0.0, 1.0))
    return float(np.interp(tgt, tpr, fpr))


@dataclass
class ScoreSet:
    y: np.ndarray
    p_normal: np.ndarray
    p_special: np.ndarray


def _load_npz_vector(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"Key '{key}' not in {path}. Available: {z.files}")
        arr = np.asarray(z[key]).reshape(-1)
    return arr.astype(np.float32)


def _load_scores(
    normal_npz: Path,
    special_npz: Path,
    normal_key: str,
    special_key: str,
    label_key: str,
) -> ScoreSet:
    y_n = _load_npz_vector(normal_npz, label_key).astype(np.int64)
    y_s = _load_npz_vector(special_npz, label_key).astype(np.int64)
    if y_n.shape != y_s.shape:
        raise ValueError(f"Label size mismatch: normal={y_n.shape}, special={y_s.shape}")
    if not np.array_equal(y_n, y_s):
        raise ValueError("Label arrays differ between normal and specialist results.")

    p_n = _load_npz_vector(normal_npz, normal_key)
    p_s = _load_npz_vector(special_npz, special_key)
    if p_n.shape != y_n.shape or p_s.shape != y_n.shape:
        raise ValueError(
            f"Prediction size mismatch: labels={y_n.shape}, normal={p_n.shape}, special={p_s.shape}"
        )

    return ScoreSet(y=y_n, p_normal=np.clip(p_n, 0.0, 1.0), p_special=np.clip(p_s, 0.0, 1.0))


def _metrics_at_target(y: np.ndarray, p: np.ndarray, target_tpr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    if np.unique(y).size <= 1:
        return {
            "auc": float("nan"),
            "fpr_at_target_tpr": float("nan"),
            "threshold": float("nan"),
            "achieved_tpr": float("nan"),
            "achieved_fpr": float("nan"),
        }

    auc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    fpr_t = float(fpr_at_target_tpr(fpr, tpr, float(target_tpr)))
    thr = float(prob_threshold_at_target_tpr(p[y == 1], float(target_tpr)))
    pred = (p >= thr)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    tpr_emp = float(tp / n_pos) if n_pos > 0 else float("nan")
    fpr_emp = float(fp / n_neg) if n_neg > 0 else float("nan")

    return {
        "auc": auc,
        "fpr_at_target_tpr": fpr_t,
        "threshold": thr,
        "achieved_tpr": tpr_emp,
        "achieved_fpr": fpr_emp,
    }


def _tp_overlap_at_target(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, target_tpr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    pos = (y == 1)
    n_pos = int(np.sum(pos))

    t_a = float(prob_threshold_at_target_tpr(p_a[pos], float(target_tpr)))
    t_b = float(prob_threshold_at_target_tpr(p_b[pos], float(target_tpr)))

    sel_a = (p_a >= t_a) & pos
    sel_b = (p_b >= t_b) & pos

    a = int(np.sum(sel_a))
    b = int(np.sum(sel_b))
    inter = int(np.sum(sel_a & sel_b))
    union = int(np.sum(sel_a | sel_b))

    return {
        "target_tpr": float(target_tpr),
        "n_pos": n_pos,
        "tp_a": a,
        "tp_b": b,
        "tp_intersection": inter,
        "tp_union": union,
        "overlap_frac_of_a": float(inter / a) if a > 0 else float("nan"),
        "overlap_frac_of_b": float(inter / b) if b > 0 else float("nan"),
        "jaccard": float(inter / union) if union > 0 else float("nan"),
        "a_only": int(a - inter),
        "b_only": int(b - inter),
    }


def _eval_scores(y: np.ndarray, p: np.ndarray, target_tpr: float) -> Tuple[float, float]:
    if np.unique(y).size <= 1:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    fpr_t = float(fpr_at_target_tpr(fpr, tpr, float(target_tpr)))
    return auc, fpr_t


def _row(method: str, params: Dict[str, float], auc: float, fpr_t: float) -> Dict[str, float]:
    out: Dict[str, float] = {"method": method, "auc": float(auc), "fpr_at_target_tpr": float(fpr_t)}
    out.update({k: float(v) for k, v in params.items()})
    return out


def _sweep_weight_grid(step: float) -> np.ndarray:
    inv = max(1, int(round(1.0 / max(step, 1e-4))))
    return np.linspace(0.0, 1.0, inv + 1, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    pp = np.clip(p, EPS, 1.0 - EPS)
    return np.log(pp / (1.0 - pp))


def _analyze_fusions(scores: ScoreSet, target_tpr: float, weight_step: float) -> Dict[str, List[Dict[str, float]]]:
    y = scores.y
    pn = scores.p_normal.astype(np.float32)
    ps = scores.p_special.astype(np.float32)

    ws = _sweep_weight_grid(float(weight_step))

    tables: Dict[str, List[Dict[str, float]]] = {
        "linear": [],
        "logit": [],
        "geometric": [],
        "harmonic": [],
        "minmax": [],
    }

    # Linear, logit, geometric, harmonic sweeps.
    ln = _logit(pn)
    ls = _logit(ps)

    for w in ws:
        wn = float(w)
        ws_ = float(1.0 - w)

        p_lin = wn * pn + ws_ * ps
        auc, fpr_t = _eval_scores(y, p_lin, target_tpr)
        tables["linear"].append(_row("linear", {"w_normal": wn, "w_special": ws_}, auc, fpr_t))

        z = wn * ln + ws_ * ls
        p_logit = _sigmoid(z)
        auc, fpr_t = _eval_scores(y, p_logit, target_tpr)
        tables["logit"].append(_row("logit", {"w_normal": wn, "w_special": ws_}, auc, fpr_t))

        p_geo = np.clip((pn ** wn) * (ps ** ws_), 0.0, 1.0)
        auc, fpr_t = _eval_scores(y, p_geo, target_tpr)
        tables["geometric"].append(_row("geometric", {"w_normal": wn, "w_special": ws_}, auc, fpr_t))

        denom = (wn / np.clip(pn, EPS, 1.0)) + (ws_ / np.clip(ps, EPS, 1.0))
        p_hm = np.clip(1.0 / np.clip(denom, EPS, np.inf), 0.0, 1.0)
        auc, fpr_t = _eval_scores(y, p_hm, target_tpr)
        tables["harmonic"].append(_row("harmonic", {"w_normal": wn, "w_special": ws_}, auc, fpr_t))

    # Min/max soft fusions.
    p_min = np.minimum(pn, ps)
    auc, fpr_t = _eval_scores(y, p_min, target_tpr)
    tables["minmax"].append(_row("min", {}, auc, fpr_t))

    p_max = np.maximum(pn, ps)
    auc, fpr_t = _eval_scores(y, p_max, target_tpr)
    tables["minmax"].append(_row("max", {}, auc, fpr_t))

    return tables


def _analyze_hard_gates(
    scores: ScoreSet,
    target_tpr: float,
    tpr_min: float,
    tpr_max: float,
    tpr_step: float,
) -> List[Dict[str, float]]:
    y = scores.y.astype(np.int64)
    pos = (y == 1)
    neg = (y == 0)
    pn = scores.p_normal.astype(np.float32)
    ps = scores.p_special.astype(np.float32)

    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))

    grid = np.arange(float(tpr_min), float(tpr_max) + 1e-9, float(tpr_step), dtype=np.float32)
    rows: List[Dict[str, float]] = []

    for tn in grid:
        thr_n = float(prob_threshold_at_target_tpr(pn[pos], float(tn)))
        pred_n = pn >= thr_n
        for ts in grid:
            thr_s = float(prob_threshold_at_target_tpr(ps[pos], float(ts)))
            pred_s = ps >= thr_s

            pred_and = pred_n & pred_s
            tp = int(np.sum(pred_and & pos))
            fp = int(np.sum(pred_and & neg))
            tpr_u = float(tp / n_pos) if n_pos > 0 else float("nan")
            fpr_u = float(fp / n_neg) if n_neg > 0 else float("nan")
            rows.append(
                {
                    "mode": "hard_and",
                    "tpr_normal_target": float(tn),
                    "tpr_special_target": float(ts),
                    "thr_normal": float(thr_n),
                    "thr_special": float(thr_s),
                    "tpr_union": tpr_u,
                    "fpr_union": fpr_u,
                    "target_tpr": float(target_tpr),
                }
            )

            pred_or = pred_n | pred_s
            tp = int(np.sum(pred_or & pos))
            fp = int(np.sum(pred_or & neg))
            tpr_u = float(tp / n_pos) if n_pos > 0 else float("nan")
            fpr_u = float(fp / n_neg) if n_neg > 0 else float("nan")
            rows.append(
                {
                    "mode": "hard_or",
                    "tpr_normal_target": float(tn),
                    "tpr_special_target": float(ts),
                    "thr_normal": float(thr_n),
                    "thr_special": float(thr_s),
                    "tpr_union": tpr_u,
                    "fpr_union": fpr_u,
                    "target_tpr": float(target_tpr),
                }
            )

    return rows


def _best_rows(rows: List[Dict[str, float]], metric_key: str, top_k: int) -> List[Dict[str, float]]:
    def _key(r: Dict[str, float]) -> Tuple[float, float]:
        m = float(r.get(metric_key, float("inf")))
        a = float(r.get("auc", float("-inf")))
        if not np.isfinite(m):
            m = float("inf")
        if not np.isfinite(a):
            a = float("-inf")
        return (m, -a)

    s = sorted(rows, key=_key)
    return s[: max(1, int(top_k))]


def _best_hard_gate_near_target(
    rows: List[Dict[str, float]],
    mode: str,
    target_tpr: float,
    tol: float,
    top_k: int,
) -> List[Dict[str, float]]:
    cand = [r for r in rows if str(r.get("mode", "")) == mode]
    cand = [r for r in cand if abs(float(r.get("tpr_union", np.nan)) - float(target_tpr)) <= float(tol)]
    s = sorted(cand, key=lambda r: float(r.get("fpr_union", float("inf"))))
    return s[: max(1, int(top_k))]


def _write_tsv(path: Path, rows: List[Dict[str, float]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.10g}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze normal-vs-specialist fusion strategies.")
    ap.add_argument("--normal_npz", type=str, default="", help="Path to normal run results.npz")
    ap.add_argument("--special_npz", type=str, default="", help="Path to specialist run results.npz")
    ap.add_argument("--normal_run_dir", type=str, default="download_checkpoints/stagec_normal_extval100k_exttest300k_combo_seed0")
    ap.add_argument("--special_run_dir", type=str, default="")
    ap.add_argument("--normal_key", type=str, default="preds_stagec")
    ap.add_argument("--special_key", type=str, default="preds_stagec")
    ap.add_argument("--label_key", type=str, default="labels")
    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--weight_step", type=float, default=0.01)
    ap.add_argument("--hard_tpr_min", type=float, default=0.30)
    ap.add_argument("--hard_tpr_max", type=float, default=0.60)
    ap.add_argument("--hard_tpr_step", type=float, default=0.01)
    ap.add_argument("--hard_target_tol", type=float, default=0.003)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--save_dir", type=str, default="download_checkpoints/fusion_analysis_normal_special")
    args = ap.parse_args()

    normal_npz = Path(args.normal_npz) if str(args.normal_npz).strip() else (Path(args.normal_run_dir) / "results.npz")
    special_npz = Path(args.special_npz) if str(args.special_npz).strip() else (Path(args.special_run_dir) / "results.npz")

    if not normal_npz.exists():
        raise FileNotFoundError(f"Normal results not found: {normal_npz}")
    if not special_npz.exists():
        raise FileNotFoundError(
            f"Specialist results not found: {special_npz}. Provide --special_npz or --special_run_dir."
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    scores = _load_scores(normal_npz, special_npz, args.normal_key, args.special_key, args.label_key)
    y = scores.y

    m_norm = _metrics_at_target(y, scores.p_normal, args.target_tpr)
    m_spec = _metrics_at_target(y, scores.p_special, args.target_tpr)
    overlap = _tp_overlap_at_target(y, scores.p_normal, scores.p_special, args.target_tpr)

    fusion_tables = _analyze_fusions(scores, args.target_tpr, args.weight_step)
    hard_rows = _analyze_hard_gates(
        scores,
        args.target_tpr,
        args.hard_tpr_min,
        args.hard_tpr_max,
        args.hard_tpr_step,
    )

    # Build compact best summary table.
    best_summary: List[Dict[str, float]] = []
    for name, rows in fusion_tables.items():
        best = _best_rows(rows, "fpr_at_target_tpr", 1)[0]
        rec = dict(best)
        rec["family"] = name
        best_summary.append(rec)

    best_and = _best_hard_gate_near_target(
        hard_rows, "hard_and", args.target_tpr, args.hard_target_tol, 1
    )
    best_or = _best_hard_gate_near_target(
        hard_rows, "hard_or", args.target_tpr, args.hard_target_tol, 1
    )
    if best_and:
        rec = dict(best_and[0])
        rec["family"] = "hard_and"
        rec["method"] = "hard_and"
        rec["fpr_at_target_tpr"] = float(rec.get("fpr_union", float("nan")))
        best_summary.append(rec)
    if best_or:
        rec = dict(best_or[0])
        rec["family"] = "hard_or"
        rec["method"] = "hard_or"
        rec["fpr_at_target_tpr"] = float(rec.get("fpr_union", float("nan")))
        best_summary.append(rec)

    best_summary = sorted(
        best_summary,
        key=lambda r: float(r.get("fpr_at_target_tpr", float("inf"))),
    )

    # Write detailed tables.
    _write_tsv(
        save_dir / "fusion_best_summary.tsv",
        best_summary,
        [
            "family",
            "method",
            "fpr_at_target_tpr",
            "auc",
            "w_normal",
            "w_special",
            "tpr_normal_target",
            "tpr_special_target",
            "tpr_union",
            "fpr_union",
            "thr_normal",
            "thr_special",
        ],
    )

    for name, rows in fusion_tables.items():
        _write_tsv(
            save_dir / f"fusion_{name}_top.tsv",
            _best_rows(rows, "fpr_at_target_tpr", args.top_k),
            ["method", "fpr_at_target_tpr", "auc", "w_normal", "w_special"],
        )

    _write_tsv(
        save_dir / "fusion_hard_gates_top.tsv",
        sorted(hard_rows, key=lambda r: (float(r.get("fpr_union", float("inf"))), abs(float(r.get("tpr_union", np.nan)) - float(args.target_tpr))))[: max(1, int(args.top_k))],
        [
            "mode",
            "tpr_normal_target",
            "tpr_special_target",
            "thr_normal",
            "thr_special",
            "tpr_union",
            "fpr_union",
            "target_tpr",
        ],
    )

    out_json = {
        "inputs": {
            "normal_npz": str(normal_npz),
            "special_npz": str(special_npz),
            "normal_key": str(args.normal_key),
            "special_key": str(args.special_key),
            "label_key": str(args.label_key),
            "target_tpr": float(args.target_tpr),
            "weight_step": float(args.weight_step),
            "hard_tpr_min": float(args.hard_tpr_min),
            "hard_tpr_max": float(args.hard_tpr_max),
            "hard_tpr_step": float(args.hard_tpr_step),
            "hard_target_tol": float(args.hard_target_tol),
            "n_samples": int(y.size),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
        },
        "standalone": {
            "normal": m_norm,
            "special": m_spec,
        },
        "tp_overlap_at_target": overlap,
        "best_fusions": best_summary,
    }

    with (save_dir / "fusion_analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print("Fusion analysis complete.")
    print(f"Saved: {save_dir / 'fusion_analysis_summary.json'}")
    print(f"Normal@TPR{args.target_tpr:.2f}: FPR={m_norm['fpr_at_target_tpr']:.6f}, AUC={m_norm['auc']:.6f}")
    print(f"Special@TPR{args.target_tpr:.2f}: FPR={m_spec['fpr_at_target_tpr']:.6f}, AUC={m_spec['auc']:.6f}")
    print(
        "TP overlap @ target: "
        f"intersection={overlap['tp_intersection']}, union={overlap['tp_union']}, "
        f"overlap_of_normal={overlap['overlap_frac_of_a']:.4f}, "
        f"overlap_of_special={overlap['overlap_frac_of_b']:.4f}"
    )
    if len(best_summary) > 0:
        b = best_summary[0]
        print(
            "Best fusion family: "
            f"{b.get('family', '')} | FPR@target={float(b.get('fpr_at_target_tpr', float('nan'))):.6f}"
        )


if __name__ == "__main__":
    main()

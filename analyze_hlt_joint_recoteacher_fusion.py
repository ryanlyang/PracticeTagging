#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze fusion strategies across three models:
  - HLT baseline (from joint run)
  - Joint dual-view tagger (from joint run)
  - RecoTeacher StageA-only model (from stageA-only run)

Reports:
  - Individual model AUC/FPR@targetTPR
  - Overlap at target TPR
  - Raw weighted fusion (2-model + 3-model)
  - Calibrated weighted fusion (Platt / Isotonic)
  - Meta-fuser on [HLT, Joint, RecoTeacher, |HLT-Joint|, |HLT-RecoTeacher|]
  - Val-selected -> test evaluation and oracle test post-hoc references
  - Stepwise gain table vs HLT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as m


def threshold_for_target_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> float:
    labels = labels.astype(np.float32)
    scores = scores.astype(np.float64)
    target_tpr = float(np.clip(target_tpr, 0.0, 1.0))
    pos = scores[labels > 0.5]
    if pos.size == 0:
        return float("inf")
    q = float(np.clip(1.0 - target_tpr, 0.0, 1.0))
    return float(np.quantile(pos, q=q))


def rates_from_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    labels_b = labels.astype(np.float32) > 0.5
    neg_b = ~labels_b
    pred = scores >= float(threshold)
    tp = int((pred & labels_b).sum())
    fp = int((pred & neg_b).sum())
    n_pos = int(labels_b.sum())
    n_neg = int(neg_b.sum())
    tpr = float(tp / max(n_pos, 1))
    fpr = float(fp / max(n_neg, 1))
    return {
        "tp": tp,
        "fp": fp,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tpr": tpr,
        "fpr": fpr,
    }


def auc_and_fpr_at_target(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> Dict[str, float]:
    labels = labels.astype(np.float32)
    scores = scores.astype(np.float64)
    auc = float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr_at = float(m.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))
    return {"auc": auc, "fpr_at_target_tpr": fpr_at}


def calibrate_platt(y_val: np.ndarray, s_val: np.ndarray, s_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    y_val = y_val.astype(np.int64)
    s_val = s_val.astype(np.float64)
    s_test = s_test.astype(np.float64)
    if np.unique(y_val).size < 2:
        return s_val.copy(), s_test.copy(), {"ok": False, "reason": "single_class"}

    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
    )
    lr.fit(s_val.reshape(-1, 1), y_val)
    pv = lr.predict_proba(s_val.reshape(-1, 1))[:, 1].astype(np.float64)
    pt = lr.predict_proba(s_test.reshape(-1, 1))[:, 1].astype(np.float64)
    return pv, pt, {
        "ok": True,
        "coef": float(lr.coef_.ravel()[0]),
        "intercept": float(lr.intercept_.ravel()[0]),
    }


def calibrate_isotonic(y_val: np.ndarray, s_val: np.ndarray, s_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    y_val = y_val.astype(np.int64)
    s_val = s_val.astype(np.float64)
    s_test = s_test.astype(np.float64)
    if np.unique(y_val).size < 2:
        return s_val.copy(), s_test.copy(), {"ok": False, "reason": "single_class"}

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(s_val, y_val.astype(np.float64))
    pv = np.asarray(iso.transform(s_val), dtype=np.float64)
    pt = np.asarray(iso.transform(s_test), dtype=np.float64)
    return pv, pt, {"ok": True}


def search_best_weighted_combo_3_at_tpr(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    preds_c: np.ndarray,
    name_a: str,
    name_b: str,
    name_c: str,
    target_tpr: float,
    weight_step: float,
) -> Dict[str, float]:
    labels = labels.astype(np.float32)
    pa = np.asarray(preds_a, dtype=np.float64)
    pb = np.asarray(preds_b, dtype=np.float64)
    pc = np.asarray(preds_c, dtype=np.float64)
    step = float(max(weight_step, 1e-4))

    best = {
        "name_a": name_a,
        "name_b": name_b,
        "name_c": name_c,
        "target_tpr": float(target_tpr),
        "weight_step": float(step),
        "w_a": float("nan"),
        "w_b": float("nan"),
        "w_c": float("nan"),
        "threshold": float("nan"),
        "tpr": float("nan"),
        "fpr": float("inf"),
        "tp": 0,
        "fp": 0,
    }

    w_vals = np.arange(0.0, 1.0 + 0.5 * step, step, dtype=np.float64)
    for w_a in w_vals:
        max_b = 1.0 - w_a
        if max_b < -1e-12:
            continue
        for w_b in np.arange(0.0, max_b + 0.5 * step, step, dtype=np.float64):
            w_c = 1.0 - w_a - w_b
            if w_c < -1e-12:
                continue
            w_c = float(max(0.0, w_c))
            score = w_a * pa + w_b * pb + w_c * pc
            thr = threshold_for_target_tpr(labels, score, target_tpr)
            rates = rates_from_threshold(labels, score, thr)
            fpr = float(rates["fpr"])
            tpr = float(rates["tpr"])

            replace = False
            if fpr < float(best["fpr"]):
                replace = True
            elif np.isfinite(fpr) and np.isclose(fpr, float(best["fpr"])):
                if abs(tpr - float(target_tpr)) < abs(float(best["tpr"]) - float(target_tpr)):
                    replace = True

            if replace:
                best = {
                    "name_a": name_a,
                    "name_b": name_b,
                    "name_c": name_c,
                    "target_tpr": float(target_tpr),
                    "weight_step": float(step),
                    "w_a": float(w_a),
                    "w_b": float(w_b),
                    "w_c": float(w_c),
                    "threshold": float(thr),
                    "tpr": float(tpr),
                    "fpr": float(fpr),
                    "tp": int(rates["tp"]),
                    "fp": int(rates["fp"]),
                }

    return best


def select_weighted_combo_3_on_val_eval_test(
    labels_val: np.ndarray,
    preds_a_val: np.ndarray,
    preds_b_val: np.ndarray,
    preds_c_val: np.ndarray,
    labels_test: np.ndarray,
    preds_a_test: np.ndarray,
    preds_b_test: np.ndarray,
    preds_c_test: np.ndarray,
    name_a: str,
    name_b: str,
    name_c: str,
    target_tpr: float,
    weight_step: float,
) -> Dict[str, object]:
    best_val = search_best_weighted_combo_3_at_tpr(
        labels=labels_val,
        preds_a=preds_a_val,
        preds_b=preds_b_val,
        preds_c=preds_c_val,
        name_a=name_a,
        name_b=name_b,
        name_c=name_c,
        target_tpr=target_tpr,
        weight_step=weight_step,
    )

    w_a = float(best_val.get("w_a", float("nan")))
    w_b = float(best_val.get("w_b", float("nan")))
    w_c = float(best_val.get("w_c", float("nan")))
    thr = float(best_val.get("threshold", float("nan")))

    if not (np.isfinite(w_a) and np.isfinite(w_b) and np.isfinite(w_c) and np.isfinite(thr)):
        return {
            "selection": {"source": "val", "best": best_val},
            "test_eval": {
                "name_a": name_a,
                "name_b": name_b,
                "name_c": name_c,
                "target_tpr": float(target_tpr),
                "w_a": float("nan"),
                "w_b": float("nan"),
                "w_c": float("nan"),
                "threshold_from_val": float("nan"),
                "tpr": float("nan"),
                "fpr": float("nan"),
                "tp": 0,
                "fp": 0,
            },
        }

    score_test = (
        w_a * np.asarray(preds_a_test, dtype=np.float64)
        + w_b * np.asarray(preds_b_test, dtype=np.float64)
        + w_c * np.asarray(preds_c_test, dtype=np.float64)
    )
    rates = rates_from_threshold(labels_test.astype(np.float32), score_test, thr)

    return {
        "selection": {"source": "val", "best": best_val},
        "test_eval": {
            "name_a": name_a,
            "name_b": name_b,
            "name_c": name_c,
            "target_tpr": float(target_tpr),
            "w_a": float(w_a),
            "w_b": float(w_b),
            "w_c": float(w_c),
            "threshold_from_val": float(thr),
            "tpr": float(rates["tpr"]),
            "fpr": float(rates["fpr"]),
            "tp": int(rates["tp"]),
            "fp": int(rates["fp"]),
        },
    }


def build_meta_features(hlt: np.ndarray, joint: np.ndarray, reco: np.ndarray) -> np.ndarray:
    h = np.asarray(hlt, dtype=np.float64)
    j = np.asarray(joint, dtype=np.float64)
    r = np.asarray(reco, dtype=np.float64)
    return np.column_stack([
        h,
        j,
        r,
        np.abs(h - j),
        np.abs(h - r),
    ]).astype(np.float64)


def train_select_meta_fuser(
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_tpr: float,
    seed: int,
    sel_frac: float,
    c_grid: np.ndarray,
) -> Dict[str, object]:
    y_val_i = y_val.astype(np.int64)
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    if np.unique(y_val_i).size < 2:
        return {
            "selection": {"source": "val_split", "ok": False, "reason": "single_class_val"},
            "test_eval": {"tpr": float("nan"), "fpr": float("nan"), "tp": 0, "fp": 0},
            "test_oracle": {"tpr": float("nan"), "fpr": float("nan"), "tp": 0, "fp": 0},
        }

    sel_frac = float(np.clip(sel_frac, 0.1, 0.9))
    fit_size = max(1, int(round((1.0 - sel_frac) * len(y_val_i))))
    if fit_size >= len(y_val_i):
        fit_size = max(1, len(y_val_i) - 1)

    idx = np.arange(len(y_val_i), dtype=np.int64)
    idx_fit, idx_sel = train_test_split(
        idx,
        train_size=fit_size,
        random_state=int(seed),
        stratify=y_val_i,
    )

    X_fit = X_val[idx_fit]
    y_fit = y_val_i[idx_fit]
    X_sel = X_val[idx_sel]
    y_sel = y_val_i[idx_sel].astype(np.float32)

    best = {
        "ok": False,
        "C": float("nan"),
        "fpr_sel": float("inf"),
        "tpr_sel": float("nan"),
        "threshold_sel": float("nan"),
    }

    for c_val in c_grid:
        clf = LogisticRegression(
            C=float(c_val),
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
        )
        clf.fit(X_fit, y_fit)
        pred_sel = clf.predict_proba(X_sel)[:, 1].astype(np.float64)
        thr_sel = threshold_for_target_tpr(y_sel, pred_sel, target_tpr)
        rates_sel = rates_from_threshold(y_sel, pred_sel, thr_sel)

        replace = False
        if rates_sel["fpr"] < best["fpr_sel"]:
            replace = True
        elif np.isclose(rates_sel["fpr"], best["fpr_sel"]):
            if abs(rates_sel["tpr"] - float(target_tpr)) < abs(best["tpr_sel"] - float(target_tpr)):
                replace = True

        if replace:
            best = {
                "ok": True,
                "C": float(c_val),
                "fpr_sel": float(rates_sel["fpr"]),
                "tpr_sel": float(rates_sel["tpr"]),
                "threshold_sel": float(thr_sel),
            }

    if not bool(best["ok"]):
        return {
            "selection": {"source": "val_split", **best},
            "test_eval": {"tpr": float("nan"), "fpr": float("nan"), "tp": 0, "fp": 0},
            "test_oracle": {"tpr": float("nan"), "fpr": float("nan"), "tp": 0, "fp": 0},
        }

    clf_full = LogisticRegression(
        C=float(best["C"]),
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
    )
    clf_full.fit(X_val, y_val_i)
    pred_val_full = clf_full.predict_proba(X_val)[:, 1].astype(np.float64)
    pred_test = clf_full.predict_proba(X_test)[:, 1].astype(np.float64)

    thr_from_val = threshold_for_target_tpr(y_val.astype(np.float32), pred_val_full, target_tpr)
    test_eval_rates = rates_from_threshold(y_test.astype(np.float32), pred_test, thr_from_val)

    thr_test_oracle = threshold_for_target_tpr(y_test.astype(np.float32), pred_test, target_tpr)
    test_oracle_rates = rates_from_threshold(y_test.astype(np.float32), pred_test, thr_test_oracle)

    return {
        "selection": {
            "source": "val_split",
            **best,
            "meta_fit_count": int(len(idx_fit)),
            "meta_sel_count": int(len(idx_sel)),
            "c_grid": [float(x) for x in c_grid.tolist()],
        },
        "threshold_from_val_full": float(thr_from_val),
        "test_eval": {
            "tpr": float(test_eval_rates["tpr"]),
            "fpr": float(test_eval_rates["fpr"]),
            "tp": int(test_eval_rates["tp"]),
            "fp": int(test_eval_rates["fp"]),
        },
        "test_oracle": {
            "threshold": float(thr_test_oracle),
            "tpr": float(test_oracle_rates["tpr"]),
            "fpr": float(test_oracle_rates["fpr"]),
            "tp": int(test_oracle_rates["tp"]),
            "fp": int(test_oracle_rates["fp"]),
        },
    }




def _quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n_bins = int(max(1, n_bins))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, qs)
    edges = np.asarray(edges, dtype=np.float64)
    edges[0] = float(np.min(x))
    edges[-1] = float(np.max(x))
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([float(np.min(x)), float(np.max(x)) + 1e-9], dtype=np.float64)
    return edges


def _rates_on_subset(labels: np.ndarray, scores: np.ndarray, threshold: float, mask: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.float32)
    scores = np.asarray(scores, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    y = labels[m]
    s = scores[m]
    if y.size == 0:
        return {
            "count": 0,
            "n_pos": 0,
            "n_neg": 0,
            "tp": 0,
            "fp": 0,
            "tpr": float("nan"),
            "fpr": float("nan"),
        }
    r = rates_from_threshold(y, s, float(threshold))
    return {
        "count": int(y.size),
        "n_pos": int(r["n_pos"]),
        "n_neg": int(r["n_neg"]),
        "tp": int(r["tp"]),
        "fp": int(r["fp"]),
        "tpr": float(r["tpr"]),
        "fpr": float(r["fpr"]),
    }


def build_bucket_delta_report(
    labels: np.ndarray,
    hlt_scores: np.ndarray,
    hlt_threshold: float,
    methods: Dict[str, Dict[str, object]],
    bucket_values: np.ndarray,
    edges: np.ndarray,
    bucket_name: str,
) -> Dict[str, object]:
    labels = labels.astype(np.float32)
    x = np.asarray(bucket_values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 2:
        return {"bucket_name": str(bucket_name), "edges": [], "bins": []}

    bins = []
    for i in range(edges.size - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < edges.size - 2:
            m = (x >= lo) & (x < hi)
        else:
            m = (x >= lo) & (x <= hi)

        hlt_r = _rates_on_subset(labels, hlt_scores, hlt_threshold, m)
        row = {
            "bin_index": int(i),
            "low": lo,
            "high": hi,
            "count": int(hlt_r["count"]),
            "hlt": hlt_r,
            "methods": {},
        }
        for name, pack in methods.items():
            ms = np.asarray(pack["scores"], dtype=np.float64)
            mt = float(pack["threshold"])
            mr = _rates_on_subset(labels, ms, mt, m)
            dtpr = float(mr["tpr"] - hlt_r["tpr"]) if np.isfinite(mr["tpr"]) and np.isfinite(hlt_r["tpr"]) else float("nan")
            dfpr = float(mr["fpr"] - hlt_r["fpr"]) if np.isfinite(mr["fpr"]) and np.isfinite(hlt_r["fpr"]) else float("nan")
            d_tp = int(mr["tp"] - hlt_r["tp"])
            d_fp = int(mr["fp"] - hlt_r["fp"])
            tp_per_fp = float(d_tp / d_fp) if d_fp != 0 else (float("inf") if d_tp > 0 else float("nan"))
            row["methods"][str(name)] = {
                **mr,
                "delta_tpr_vs_hlt": dtpr,
                "delta_fpr_vs_hlt": dfpr,
                "delta_tp_vs_hlt": d_tp,
                "delta_fp_vs_hlt": d_fp,
                "delta_tp_per_delta_fp_vs_hlt": tp_per_fp,
            }
        bins.append(row)

    return {
        "bucket_name": str(bucket_name),
        "edges": [float(e) for e in edges.tolist()],
        "bins": bins,
    }

def make_gain_row(name: str, fpr_hlt: float, fpr_method: float) -> Dict[str, float]:
    gain_abs = float(fpr_hlt - fpr_method)
    gain_rel = float(100.0 * gain_abs / fpr_hlt) if np.isfinite(fpr_hlt) and fpr_hlt > 0 else float("nan")
    return {
        "method": str(name),
        "fpr_test": float(fpr_method),
        "gain_vs_hlt_abs": float(gain_abs),
        "gain_vs_hlt_rel_percent": float(gain_rel),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fusion analysis for HLT + Joint + RecoTeacher")
    ap.add_argument("--stagea_run_dir", type=str, required=True)
    ap.add_argument("--joint_run_dir", type=str, required=True)
    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--weight_step_2", type=float, default=0.01)
    ap.add_argument("--weight_step_3", type=float, default=0.02)
    ap.add_argument("--meta_sel_frac", type=float, default=0.30)
    ap.add_argument("--meta_c_grid", type=str, default="0.05,0.1,0.3,1,3,10,30")
    ap.add_argument("--bucket_deciles", type=int, default=10)
    ap.add_argument("--bucket_pt_bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_name", type=str, default="fusion_hlt_joint_recoteacher_analysis.json")
    args = ap.parse_args()

    stagea_dir = Path(args.stagea_run_dir)
    joint_dir = Path(args.joint_run_dir)
    if not stagea_dir.exists():
        raise FileNotFoundError(f"stagea_run_dir not found: {stagea_dir}")
    if not joint_dir.exists():
        raise FileNotFoundError(f"joint_run_dir not found: {joint_dir}")

    stagea_npz = stagea_dir / "stageA_only_scores.npz"
    joint_npz = joint_dir / "fusion_scores_val_test.npz"
    if not stagea_npz.exists():
        raise FileNotFoundError(f"StageA score file not found: {stagea_npz}")
    if not joint_npz.exists():
        raise FileNotFoundError(
            f"Joint fusion score file not found: {joint_npz} (run joint with --save_fusion_scores)."
        )

    sA = np.load(stagea_npz)
    jn = np.load(joint_npz)

    y_val_a = np.asarray(sA["labels_val"], dtype=np.float32)
    y_test_a = np.asarray(sA["labels_test"], dtype=np.float32)
    y_val_j = np.asarray(jn["labels_val"], dtype=np.float32)
    y_test_j = np.asarray(jn["labels_test"], dtype=np.float32)

    if not np.array_equal(y_val_a, y_val_j):
        raise RuntimeError("Validation label mismatch between StageA-only and Joint runs.")
    if not np.array_equal(y_test_a, y_test_j):
        raise RuntimeError("Test label mismatch between StageA-only and Joint runs.")

    y_val = y_val_j
    y_test = y_test_j

    hlt_val = np.asarray(jn["preds_hlt_val"], dtype=np.float64)
    hlt_test = np.asarray(jn["preds_hlt_test"], dtype=np.float64)
    joint_val = np.asarray(jn["preds_joint_val"], dtype=np.float64)
    joint_test = np.asarray(jn["preds_joint_test"], dtype=np.float64)
    reco_val = np.asarray(sA["preds_reco_teacher_val"], dtype=np.float64)
    reco_test = np.asarray(sA["preds_reco_teacher_test"], dtype=np.float64)

    target_tpr = float(args.target_tpr)

    indiv = {
        "hlt": {
            **auc_and_fpr_at_target(y_val, hlt_val, target_tpr),
            "auc_test": auc_and_fpr_at_target(y_test, hlt_test, target_tpr)["auc"],
            "fpr_test": auc_and_fpr_at_target(y_test, hlt_test, target_tpr)["fpr_at_target_tpr"],
        },
        "joint": {
            **auc_and_fpr_at_target(y_val, joint_val, target_tpr),
            "auc_test": auc_and_fpr_at_target(y_test, joint_test, target_tpr)["auc"],
            "fpr_test": auc_and_fpr_at_target(y_test, joint_test, target_tpr)["fpr_at_target_tpr"],
        },
        "reco_teacher": {
            **auc_and_fpr_at_target(y_val, reco_val, target_tpr),
            "auc_test": auc_and_fpr_at_target(y_test, reco_test, target_tpr)["auc"],
            "fpr_test": auc_and_fpr_at_target(y_test, reco_test, target_tpr)["fpr_at_target_tpr"],
        },
    }

    overlap_test = m.build_overlap_report_at_tpr(
        labels=y_test,
        model_preds={
            "hlt": hlt_test,
            "joint": joint_test,
            "reco_teacher": reco_test,
        },
        target_tpr=target_tpr,
    )

    # Raw 2-model combos.
    raw_hlt_joint_valsel = m.select_weighted_combo_on_val_and_eval_test(
        labels_val=y_val,
        preds_a_val=hlt_val,
        preds_b_val=joint_val,
        labels_test=y_test,
        preds_a_test=hlt_test,
        preds_b_test=joint_test,
        name_a="hlt",
        name_b="joint",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_2),
    )
    raw_hlt_joint_oracle = m.search_best_weighted_combo_at_tpr(
        labels=y_test,
        preds_a=hlt_test,
        preds_b=joint_test,
        name_a="hlt",
        name_b="joint",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_2),
    )

    raw_hlt_reco_valsel = m.select_weighted_combo_on_val_and_eval_test(
        labels_val=y_val,
        preds_a_val=hlt_val,
        preds_b_val=reco_val,
        labels_test=y_test,
        preds_a_test=hlt_test,
        preds_b_test=reco_test,
        name_a="hlt",
        name_b="reco_teacher",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_2),
    )
    raw_hlt_reco_oracle = m.search_best_weighted_combo_at_tpr(
        labels=y_test,
        preds_a=hlt_test,
        preds_b=reco_test,
        name_a="hlt",
        name_b="reco_teacher",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_2),
    )

    # Raw 3-model combos.
    raw_3_valsel = select_weighted_combo_3_on_val_eval_test(
        labels_val=y_val,
        preds_a_val=hlt_val,
        preds_b_val=joint_val,
        preds_c_val=reco_val,
        labels_test=y_test,
        preds_a_test=hlt_test,
        preds_b_test=joint_test,
        preds_c_test=reco_test,
        name_a="hlt",
        name_b="joint",
        name_c="reco_teacher",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_3),
    )
    raw_3_oracle = search_best_weighted_combo_3_at_tpr(
        labels=y_test,
        preds_a=hlt_test,
        preds_b=joint_test,
        preds_c=reco_test,
        name_a="hlt",
        name_b="joint",
        name_c="reco_teacher",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_3),
    )

    # Calibrations.
    hlt_platt_v, hlt_platt_t, hlt_platt_meta = calibrate_platt(y_val, hlt_val, hlt_test)
    joint_platt_v, joint_platt_t, joint_platt_meta = calibrate_platt(y_val, joint_val, joint_test)
    reco_platt_v, reco_platt_t, reco_platt_meta = calibrate_platt(y_val, reco_val, reco_test)

    hlt_iso_v, hlt_iso_t, hlt_iso_meta = calibrate_isotonic(y_val, hlt_val, hlt_test)
    joint_iso_v, joint_iso_t, joint_iso_meta = calibrate_isotonic(y_val, joint_val, joint_test)
    reco_iso_v, reco_iso_t, reco_iso_meta = calibrate_isotonic(y_val, reco_val, reco_test)

    platt_3_valsel = select_weighted_combo_3_on_val_eval_test(
        labels_val=y_val,
        preds_a_val=hlt_platt_v,
        preds_b_val=joint_platt_v,
        preds_c_val=reco_platt_v,
        labels_test=y_test,
        preds_a_test=hlt_platt_t,
        preds_b_test=joint_platt_t,
        preds_c_test=reco_platt_t,
        name_a="hlt_platt",
        name_b="joint_platt",
        name_c="reco_platt",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_3),
    )
    platt_3_oracle = search_best_weighted_combo_3_at_tpr(
        labels=y_test,
        preds_a=hlt_platt_t,
        preds_b=joint_platt_t,
        preds_c=reco_platt_t,
        name_a="hlt_platt",
        name_b="joint_platt",
        name_c="reco_platt",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_3),
    )

    iso_3_valsel = select_weighted_combo_3_on_val_eval_test(
        labels_val=y_val,
        preds_a_val=hlt_iso_v,
        preds_b_val=joint_iso_v,
        preds_c_val=reco_iso_v,
        labels_test=y_test,
        preds_a_test=hlt_iso_t,
        preds_b_test=joint_iso_t,
        preds_c_test=reco_iso_t,
        name_a="hlt_iso",
        name_b="joint_iso",
        name_c="reco_iso",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_3),
    )
    iso_3_oracle = search_best_weighted_combo_3_at_tpr(
        labels=y_test,
        preds_a=hlt_iso_t,
        preds_b=joint_iso_t,
        preds_c=reco_iso_t,
        name_a="hlt_iso",
        name_b="joint_iso",
        name_c="reco_iso",
        target_tpr=target_tpr,
        weight_step=float(args.weight_step_3),
    )

    c_grid = np.array([float(x) for x in str(args.meta_c_grid).split(",") if x.strip()], dtype=np.float64)
    if c_grid.size == 0:
        c_grid = np.array([0.1, 0.3, 1.0, 3.0, 10.0], dtype=np.float64)

    meta_raw = train_select_meta_fuser(
        X_val=build_meta_features(hlt_val, joint_val, reco_val),
        y_val=y_val,
        X_test=build_meta_features(hlt_test, joint_test, reco_test),
        y_test=y_test,
        target_tpr=target_tpr,
        seed=int(args.seed),
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
    )
    meta_platt = train_select_meta_fuser(
        X_val=build_meta_features(hlt_platt_v, joint_platt_v, reco_platt_v),
        y_val=y_val,
        X_test=build_meta_features(hlt_platt_t, joint_platt_t, reco_platt_t),
        y_test=y_test,
        target_tpr=target_tpr,
        seed=int(args.seed),
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
    )
    meta_iso = train_select_meta_fuser(
        X_val=build_meta_features(hlt_iso_v, joint_iso_v, reco_iso_v),
        y_val=y_val,
        X_test=build_meta_features(hlt_iso_t, joint_iso_t, reco_iso_t),
        y_test=y_test,
        target_tpr=target_tpr,
        seed=int(args.seed),
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
    )

    fpr_hlt_test = float(indiv["hlt"]["fpr_test"])
    # Bucket diagnostics (where gains come from)
    thr_hlt_test = threshold_for_target_tpr(y_test, hlt_test, target_tpr)
    thr_joint_test = threshold_for_target_tpr(y_test, joint_test, target_tpr)
    thr_reco_test = threshold_for_target_tpr(y_test, reco_test, target_tpr)

    raw3_te = raw_3_valsel["test_eval"]
    score_raw3_test = (
        float(raw3_te["w_a"]) * hlt_test
        + float(raw3_te["w_b"]) * joint_test
        + float(raw3_te["w_c"]) * reco_test
    )
    thr_raw3 = float(raw3_te["threshold_from_val"])

    iso3_te = iso_3_valsel["test_eval"]
    score_iso3_test = (
        float(iso3_te["w_a"]) * hlt_iso_t
        + float(iso3_te["w_b"]) * joint_iso_t
        + float(iso3_te["w_c"]) * reco_iso_t
    )
    thr_iso3 = float(iso3_te["threshold_from_val"])

    bucket_methods = {
        "joint_single": {"scores": joint_test, "threshold": thr_joint_test},
        "reco_teacher_single": {"scores": reco_test, "threshold": thr_reco_test},
        "raw3_val_selected": {"scores": score_raw3_test, "threshold": thr_raw3},
        "iso3_val_selected": {"scores": score_iso3_test, "threshold": thr_iso3},
    }

    bucket_reports = {
        "hlt_score_deciles": build_bucket_delta_report(
            labels=y_test,
            hlt_scores=hlt_test,
            hlt_threshold=thr_hlt_test,
            methods=bucket_methods,
            bucket_values=hlt_test,
            edges=_quantile_edges(hlt_test, int(args.bucket_deciles)),
            bucket_name="hlt_score_deciles",
        ),
    }

    if "hlt_nconst_test" in jn:
        nconst = np.asarray(jn["hlt_nconst_test"], dtype=np.float64)
        n_edges = np.array([0, 10, 20, 30, 40, 50, 60, 80, 120], dtype=np.float64)
        bucket_reports["hlt_nconst_bins"] = build_bucket_delta_report(
            labels=y_test,
            hlt_scores=hlt_test,
            hlt_threshold=thr_hlt_test,
            methods=bucket_methods,
            bucket_values=nconst,
            edges=n_edges,
            bucket_name="hlt_nconst_bins",
        )

    if "hlt_jet_pt_test" in jn:
        pt = np.asarray(jn["hlt_jet_pt_test"], dtype=np.float64)
        bucket_reports["hlt_pt_quantiles"] = build_bucket_delta_report(
            labels=y_test,
            hlt_scores=hlt_test,
            hlt_threshold=thr_hlt_test,
            methods=bucket_methods,
            bucket_values=pt,
            edges=_quantile_edges(pt, int(args.bucket_pt_bins)),
            bucket_name="hlt_pt_quantiles",
        )

    stepwise = [
        make_gain_row("HLT single", fpr_hlt_test, fpr_hlt_test),
        make_gain_row("Joint single", fpr_hlt_test, float(indiv["joint"]["fpr_test"])),
        make_gain_row("RecoTeacher single", fpr_hlt_test, float(indiv["reco_teacher"]["fpr_test"])),
        make_gain_row("Raw 2-model HLT+Joint (val->test)", fpr_hlt_test, float(raw_hlt_joint_valsel["test_eval"]["fpr"])),
        make_gain_row("Raw 2-model HLT+RecoTeacher (val->test)", fpr_hlt_test, float(raw_hlt_reco_valsel["test_eval"]["fpr"])),
        make_gain_row("Raw 3-model weighted (val->test)", fpr_hlt_test, float(raw_3_valsel["test_eval"]["fpr"])),
        make_gain_row("Platt-calibrated 3-model weighted (val->test)", fpr_hlt_test, float(platt_3_valsel["test_eval"]["fpr"])),
        make_gain_row("Isotonic-calibrated 3-model weighted (val->test)", fpr_hlt_test, float(iso_3_valsel["test_eval"]["fpr"])),
        make_gain_row("Meta-fuser raw (val-split selected -> test)", fpr_hlt_test, float(meta_raw["test_eval"]["fpr"])),
        make_gain_row("Meta-fuser Platt inputs (val-split selected -> test)", fpr_hlt_test, float(meta_platt["test_eval"]["fpr"])),
        make_gain_row("Meta-fuser Isotonic inputs (val-split selected -> test)", fpr_hlt_test, float(meta_iso["test_eval"]["fpr"])),
    ]

    out = {
        "target_tpr": target_tpr,
        "inputs": {
            "stagea_run_dir": str(stagea_dir),
            "joint_run_dir": str(joint_dir),
            "stagea_scores": str(stagea_npz),
            "joint_scores": str(joint_npz),
        },
        "individual_models": indiv,
        "test_overlap": overlap_test,
        "raw_combos": {
            "hlt_joint_val_selected_eval_test": raw_hlt_joint_valsel,
            "hlt_joint_test_oracle": raw_hlt_joint_oracle,
            "hlt_reco_val_selected_eval_test": raw_hlt_reco_valsel,
            "hlt_reco_test_oracle": raw_hlt_reco_oracle,
            "hlt_joint_reco_val_selected_eval_test": raw_3_valsel,
            "hlt_joint_reco_test_oracle": raw_3_oracle,
        },
        "calibration": {
            "platt": {
                "hlt": hlt_platt_meta,
                "joint": joint_platt_meta,
                "reco_teacher": reco_platt_meta,
                "combo3_val_selected_eval_test": platt_3_valsel,
                "combo3_test_oracle": platt_3_oracle,
            },
            "isotonic": {
                "hlt": hlt_iso_meta,
                "joint": joint_iso_meta,
                "reco_teacher": reco_iso_meta,
                "combo3_val_selected_eval_test": iso_3_valsel,
                "combo3_test_oracle": iso_3_oracle,
            },
        },
        "meta_fuser": {
            "raw": meta_raw,
            "platt_inputs": meta_platt,
            "isotonic_inputs": meta_iso,
        },
        "stepwise_gain_vs_hlt": stepwise,
        "bucket_analysis": bucket_reports,
    }

    out_path = joint_dir / str(args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=" * 70)
    print("HLT + Joint + RecoTeacher Fusion Analysis")
    print("=" * 70)
    print(f"Saved: {out_path}")
    print(f"HLT test FPR@TPR={target_tpr:.2f}: {fpr_hlt_test:.6f}")
    best_row = min(stepwise, key=lambda r: float(r["fpr_test"]))
    print(
        f"Best val-selected strategy: {best_row['method']} | "
        f"FPR_test={best_row['fpr_test']:.6f} | "
        f"gain_vs_hlt_abs={best_row['gain_vs_hlt_abs']:.6f}"
    )


if __name__ == "__main__":
    main()

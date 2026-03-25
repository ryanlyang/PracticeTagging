#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusion analysis for five independently trained models:
  1) HLT baseline (from model-2 joint run)
  2) Joint with weak L_delta (from model-2 joint run)
  3) RecoTeacher-soft (s09 StageA-only)
  4) Corrected-only tagger (s01 StageA-only)
  5) Joint dual-view (s01 full StageA/B/C run)

Reports val-selected->test and oracle test references for:
  - individual models
  - HLT+X weighted pairs
  - all-5 weighted fusion (raw / Platt / Isotonic)
  - meta-fuser (raw / Platt / Isotonic inputs)
  - overlap and bucket diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as m


MODEL_ORDER = [
    "hlt",
    "joint_delta",
    "reco_teacher_s09",
    "corrected_s01",
    "joint_s01",
]


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


def generate_weight_candidates(
    n_models: int,
    n_random: int,
    seed: int,
    include_pair_grid: bool,
    pair_step: float,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    eye = np.eye(n_models, dtype=np.float64)
    rows.extend([eye[i] for i in range(n_models)])
    rows.append(np.ones(n_models, dtype=np.float64) / float(n_models))

    if include_pair_grid and n_models >= 2:
        ws = np.arange(0.0, 1.0 + 0.5 * pair_step, pair_step, dtype=np.float64)
        for j in range(1, n_models):
            for w0 in ws:
                v = np.zeros(n_models, dtype=np.float64)
                v[0] = float(w0)
                v[j] = float(1.0 - w0)
                rows.append(v)

    if int(n_random) > 0:
        rng = np.random.default_rng(int(seed))
        rand_w = rng.dirichlet(alpha=np.ones(n_models, dtype=np.float64), size=int(n_random)).astype(np.float64)
        rows.extend([rand_w[i] for i in range(rand_w.shape[0])])

    W = np.vstack(rows).astype(np.float64)
    W = np.clip(W, 0.0, None)
    W /= np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)

    # Deduplicate rounded copies for stable search cost.
    key = np.round(W, 6)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return W[uniq_idx]


def search_best_weighted_combo_multi_at_tpr(
    labels: np.ndarray,
    score_mat: np.ndarray,
    model_names: List[str],
    target_tpr: float,
    weight_candidates: np.ndarray,
) -> Dict[str, object]:
    labels = labels.astype(np.float32)
    score_mat = np.asarray(score_mat, dtype=np.float64)
    W = np.asarray(weight_candidates, dtype=np.float64)

    best = {
        "model_names": list(model_names),
        "target_tpr": float(target_tpr),
        "weights": [float("nan")] * len(model_names),
        "threshold": float("nan"),
        "tpr": float("nan"),
        "fpr": float("inf"),
        "tp": 0,
        "fp": 0,
    }

    for i in range(W.shape[0]):
        w = W[i]
        score = np.sum(score_mat * w[:, None], axis=0)
        thr = threshold_for_target_tpr(labels, score, target_tpr)
        rates = rates_from_threshold(labels, score, thr)
        fpr = float(rates["fpr"])
        tpr = float(rates["tpr"])

        replace = False
        if fpr < float(best["fpr"]):
            replace = True
        elif np.isclose(fpr, float(best["fpr"])):
            if abs(tpr - float(target_tpr)) < abs(float(best["tpr"]) - float(target_tpr)):
                replace = True

        if replace:
            best = {
                "model_names": list(model_names),
                "target_tpr": float(target_tpr),
                "weights": [float(x) for x in w.tolist()],
                "threshold": float(thr),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "tp": int(rates["tp"]),
                "fp": int(rates["fp"]),
            }

    return best


def select_weighted_combo_multi_on_val_eval_test(
    y_val: np.ndarray,
    score_mat_val: np.ndarray,
    y_test: np.ndarray,
    score_mat_test: np.ndarray,
    model_names: List[str],
    target_tpr: float,
    weight_candidates: np.ndarray,
) -> Dict[str, object]:
    best_val = search_best_weighted_combo_multi_at_tpr(
        labels=y_val,
        score_mat=score_mat_val,
        model_names=model_names,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    weights = np.asarray(best_val["weights"], dtype=np.float64)
    thr = float(best_val["threshold"])
    if not np.all(np.isfinite(weights)) or not np.isfinite(thr):
        return {
            "selection": {"source": "val", "best": best_val},
            "test_eval": {
                "model_names": list(model_names),
                "target_tpr": float(target_tpr),
                "weights": [float("nan")] * len(model_names),
                "threshold_from_val": float("nan"),
                "tpr": float("nan"),
                "fpr": float("nan"),
                "tp": 0,
                "fp": 0,
            },
        }

    score_test = np.sum(np.asarray(score_mat_test, dtype=np.float64) * weights[:, None], axis=0)
    rates = rates_from_threshold(y_test.astype(np.float32), score_test, thr)
    return {
        "selection": {"source": "val", "best": best_val},
        "test_eval": {
            "model_names": list(model_names),
            "target_tpr": float(target_tpr),
            "weights": [float(x) for x in weights.tolist()],
            "threshold_from_val": float(thr),
            "tpr": float(rates["tpr"]),
            "fpr": float(rates["fpr"]),
            "tp": int(rates["tp"]),
            "fp": int(rates["fp"]),
        },
    }


def build_meta_features(score_dict: Dict[str, np.ndarray]) -> np.ndarray:
    h = np.asarray(score_dict["hlt"], dtype=np.float64)
    jd = np.asarray(score_dict["joint_delta"], dtype=np.float64)
    rs = np.asarray(score_dict["reco_teacher_s09"], dtype=np.float64)
    cs = np.asarray(score_dict["corrected_s01"], dtype=np.float64)
    j5 = np.asarray(score_dict["joint_s01"], dtype=np.float64)

    return np.column_stack([
        h, jd, rs, cs, j5,
        np.abs(h - jd), np.abs(h - rs), np.abs(h - cs), np.abs(h - j5),
        np.abs(jd - rs), np.abs(jd - cs), np.abs(jd - j5),
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
) -> Tuple[Dict[str, object], np.ndarray, float]:
    y_val_i = y_val.astype(np.int64)
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    if np.unique(y_val_i).size < 2:
        return ({
            "selection": {"source": "val_split", "ok": False, "reason": "single_class_val"},
            "test_eval": {"tpr": float("nan"), "fpr": float("nan"), "tp": 0, "fp": 0},
            "test_oracle": {"tpr": float("nan"), "fpr": float("nan"), "tp": 0, "fp": 0},
        }, np.zeros(len(y_test), dtype=np.float64), float("nan"))

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
        return ({
            "selection": {"source": "val_split", **best},
            "test_eval": {"tpr": float("nan"), "fpr": float("nan"), "tp": 0, "fp": 0},
            "test_oracle": {"tpr": float("nan"), "fpr": float("nan"), "tp": 0, "fp": 0},
        }, np.zeros(len(y_test), dtype=np.float64), float("nan"))

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

    out = {
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
    return out, pred_test, float(thr_from_val)


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
    msk = np.asarray(mask, dtype=bool)
    y = labels[msk]
    s = scores[msk]
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
            msk = (x >= lo) & (x < hi)
        else:
            msk = (x >= lo) & (x <= hi)

        hlt_r = _rates_on_subset(labels, hlt_scores, hlt_threshold, msk)
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
            mr = _rates_on_subset(labels, ms, mt, msk)
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


def _load_required(npz_path: Path, key: str) -> np.ndarray:
    z = np.load(npz_path)
    if key not in z:
        raise KeyError(f"Missing key '{key}' in {npz_path}")
    return np.asarray(z[key])


def main() -> None:
    ap = argparse.ArgumentParser(description="Fusion analysis for HLT + four additional models")
    ap.add_argument("--joint_delta_run_dir", type=str, required=True)
    ap.add_argument("--reco_teacher_s09_run_dir", type=str, required=True)
    ap.add_argument("--corrected_s01_run_dir", type=str, required=True)
    ap.add_argument("--joint_s01_run_dir", type=str, required=True)
    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--weight_step_2", type=float, default=0.01)
    ap.add_argument("--weight_samples_multi", type=int, default=4000)
    ap.add_argument("--pair_grid_step_multi", type=float, default=0.05)
    ap.add_argument("--meta_sel_frac", type=float, default=0.30)
    ap.add_argument("--meta_c_grid", type=str, default="0.05,0.1,0.3,1,3,10,30")
    ap.add_argument("--bucket_deciles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_name", type=str, default="fusion_hlt_joint5_analysis.json")
    args = ap.parse_args()

    dir_joint_delta = Path(args.joint_delta_run_dir)
    dir_reco_s09 = Path(args.reco_teacher_s09_run_dir)
    dir_corr_s01 = Path(args.corrected_s01_run_dir)
    dir_joint_s01 = Path(args.joint_s01_run_dir)

    joint_delta_npz = dir_joint_delta / "fusion_scores_val_test.npz"
    reco_s09_npz = dir_reco_s09 / "stageA_only_scores.npz"
    corr_s01_npz = dir_corr_s01 / "stageA_only_scores.npz"
    joint_s01_npz = dir_joint_s01 / "fusion_scores_val_test.npz"

    for pth in [joint_delta_npz, reco_s09_npz, corr_s01_npz, joint_s01_npz]:
        if not pth.exists():
            raise FileNotFoundError(f"Required score file not found: {pth}")

    z2 = np.load(joint_delta_npz)
    z3 = np.load(reco_s09_npz)
    z4 = np.load(corr_s01_npz)
    z5 = np.load(joint_s01_npz)

    y_val = np.asarray(z2["labels_val"], dtype=np.float32)
    y_test = np.asarray(z2["labels_test"], dtype=np.float32)

    for nm, z in [("model3", z3), ("model4", z4), ("model5", z5)]:
        yv = np.asarray(z["labels_val"], dtype=np.float32)
        yt = np.asarray(z["labels_test"], dtype=np.float32)
        if not np.array_equal(y_val, yv):
            raise RuntimeError(f"Validation labels mismatch: {nm}")
        if not np.array_equal(y_test, yt):
            raise RuntimeError(f"Test labels mismatch: {nm}")

    if "preds_corrected_only_val" not in z4 or "preds_corrected_only_test" not in z4:
        raise KeyError(f"Model4 scores missing corrected-only keys in: {corr_s01_npz}")
    if np.asarray(z4["preds_corrected_only_test"]).size != y_test.size:
        raise RuntimeError(
            "Model4 corrected-only predictions missing or wrong size. "
            "Rerun model4 with --train_corrected_only_post_stageA."
        )

    scores_val = {
        "hlt": np.asarray(z2["preds_hlt_val"], dtype=np.float64),
        "joint_delta": np.asarray(z2["preds_joint_val"], dtype=np.float64),
        "reco_teacher_s09": np.asarray(z3["preds_reco_teacher_val"], dtype=np.float64),
        "corrected_s01": np.asarray(z4["preds_corrected_only_val"], dtype=np.float64),
        "joint_s01": np.asarray(z5["preds_joint_val"], dtype=np.float64),
    }
    scores_test = {
        "hlt": np.asarray(z2["preds_hlt_test"], dtype=np.float64),
        "joint_delta": np.asarray(z2["preds_joint_test"], dtype=np.float64),
        "reco_teacher_s09": np.asarray(z3["preds_reco_teacher_test"], dtype=np.float64),
        "corrected_s01": np.asarray(z4["preds_corrected_only_test"], dtype=np.float64),
        "joint_s01": np.asarray(z5["preds_joint_test"], dtype=np.float64),
    }

    target_tpr = float(args.target_tpr)

    indiv = {}
    for name in MODEL_ORDER:
        v = auc_and_fpr_at_target(y_val, scores_val[name], target_tpr)
        t = auc_and_fpr_at_target(y_test, scores_test[name], target_tpr)
        indiv[name] = {
            "auc_val": float(v["auc"]),
            "fpr_val": float(v["fpr_at_target_tpr"]),
            "auc_test": float(t["auc"]),
            "fpr_test": float(t["fpr_at_target_tpr"]),
        }

    overlap_test = m.build_overlap_report_at_tpr(
        labels=y_test,
        model_preds={k: scores_test[k] for k in MODEL_ORDER},
        target_tpr=target_tpr,
    )

    # Pair combos: HLT + each other model.
    pair_results_valsel: Dict[str, Dict[str, object]] = {}
    pair_results_oracle: Dict[str, Dict[str, object]] = {}
    for other in MODEL_ORDER[1:]:
        key = f"hlt_plus_{other}"
        pair_results_valsel[key] = m.select_weighted_combo_on_val_and_eval_test(
            labels_val=y_val,
            preds_a_val=scores_val["hlt"],
            preds_b_val=scores_val[other],
            labels_test=y_test,
            preds_a_test=scores_test["hlt"],
            preds_b_test=scores_test[other],
            name_a="hlt",
            name_b=other,
            target_tpr=target_tpr,
            weight_step=float(args.weight_step_2),
        )
        pair_results_oracle[key] = m.search_best_weighted_combo_at_tpr(
            labels=y_test,
            preds_a=scores_test["hlt"],
            preds_b=scores_test[other],
            name_a="hlt",
            name_b=other,
            target_tpr=target_tpr,
            weight_step=float(args.weight_step_2),
        )

    # All-6 weighted fusion: raw scores.
    W = generate_weight_candidates(
        n_models=len(MODEL_ORDER),
        n_random=int(args.weight_samples_multi),
        seed=int(args.seed),
        include_pair_grid=True,
        pair_step=float(args.pair_grid_step_multi),
    )
    mat_val = np.vstack([scores_val[n] for n in MODEL_ORDER])
    mat_test = np.vstack([scores_test[n] for n in MODEL_ORDER])

    raw_all5_valsel = select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_val,
        y_test=y_test,
        score_mat_test=mat_test,
        model_names=MODEL_ORDER,
        target_tpr=target_tpr,
        weight_candidates=W,
    )
    raw_all5_oracle = search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_test,
        model_names=MODEL_ORDER,
        target_tpr=target_tpr,
        weight_candidates=W,
    )

    # Calibrate every model, then all-6 weighted fusion on calibrated scores.
    cal_platt = {}
    cal_iso = {}
    scores_platt_val: Dict[str, np.ndarray] = {}
    scores_platt_test: Dict[str, np.ndarray] = {}
    scores_iso_val: Dict[str, np.ndarray] = {}
    scores_iso_test: Dict[str, np.ndarray] = {}

    for name in MODEL_ORDER:
        pv, pt, pm = calibrate_platt(y_val, scores_val[name], scores_test[name])
        iv, it, im = calibrate_isotonic(y_val, scores_val[name], scores_test[name])
        scores_platt_val[name] = pv
        scores_platt_test[name] = pt
        scores_iso_val[name] = iv
        scores_iso_test[name] = it
        cal_platt[name] = pm
        cal_iso[name] = im

    mat_platt_val = np.vstack([scores_platt_val[n] for n in MODEL_ORDER])
    mat_platt_test = np.vstack([scores_platt_test[n] for n in MODEL_ORDER])
    mat_iso_val = np.vstack([scores_iso_val[n] for n in MODEL_ORDER])
    mat_iso_test = np.vstack([scores_iso_test[n] for n in MODEL_ORDER])

    platt_all5_valsel = select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_platt_val,
        y_test=y_test,
        score_mat_test=mat_platt_test,
        model_names=MODEL_ORDER,
        target_tpr=target_tpr,
        weight_candidates=W,
    )
    platt_all5_oracle = search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_platt_test,
        model_names=MODEL_ORDER,
        target_tpr=target_tpr,
        weight_candidates=W,
    )

    iso_all5_valsel = select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_iso_val,
        y_test=y_test,
        score_mat_test=mat_iso_test,
        model_names=MODEL_ORDER,
        target_tpr=target_tpr,
        weight_candidates=W,
    )
    iso_all5_oracle = search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_iso_test,
        model_names=MODEL_ORDER,
        target_tpr=target_tpr,
        weight_candidates=W,
    )

    c_grid = np.array([float(x) for x in str(args.meta_c_grid).split(",") if x.strip()], dtype=np.float64)
    if c_grid.size == 0:
        c_grid = np.array([0.1, 0.3, 1.0, 3.0, 10.0], dtype=np.float64)

    meta_raw, meta_raw_pred_test, meta_raw_thr = train_select_meta_fuser(
        X_val=build_meta_features(scores_val),
        y_val=y_val,
        X_test=build_meta_features(scores_test),
        y_test=y_test,
        target_tpr=target_tpr,
        seed=int(args.seed),
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
    )
    meta_platt, meta_platt_pred_test, meta_platt_thr = train_select_meta_fuser(
        X_val=build_meta_features(scores_platt_val),
        y_val=y_val,
        X_test=build_meta_features(scores_platt_test),
        y_test=y_test,
        target_tpr=target_tpr,
        seed=int(args.seed),
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
    )
    meta_iso, meta_iso_pred_test, meta_iso_thr = train_select_meta_fuser(
        X_val=build_meta_features(scores_iso_val),
        y_val=y_val,
        X_test=build_meta_features(scores_iso_test),
        y_test=y_test,
        target_tpr=target_tpr,
        seed=int(args.seed),
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
    )

    # Build stepwise gain summary.
    fpr_hlt_test = float(indiv["hlt"]["fpr_test"])
    stepwise = [make_gain_row("HLT single", fpr_hlt_test, fpr_hlt_test)]

    for name in MODEL_ORDER[1:]:
        stepwise.append(make_gain_row(f"{name} single", fpr_hlt_test, float(indiv[name]["fpr_test"])))

    for key, pack in pair_results_valsel.items():
        stepwise.append(make_gain_row(f"pair {key} (val->test)", fpr_hlt_test, float(pack["test_eval"]["fpr"])))

    stepwise.extend([
        make_gain_row("all5 raw weighted (val->test)", fpr_hlt_test, float(raw_all5_valsel["test_eval"]["fpr"])),
        make_gain_row("all5 Platt weighted (val->test)", fpr_hlt_test, float(platt_all5_valsel["test_eval"]["fpr"])),
        make_gain_row("all5 Isotonic weighted (val->test)", fpr_hlt_test, float(iso_all5_valsel["test_eval"]["fpr"])),
        make_gain_row("meta raw (val-split selected -> test)", fpr_hlt_test, float(meta_raw["test_eval"]["fpr"])),
        make_gain_row("meta Platt inputs (val-split selected -> test)", fpr_hlt_test, float(meta_platt["test_eval"]["fpr"])),
        make_gain_row("meta Isotonic inputs (val-split selected -> test)", fpr_hlt_test, float(meta_iso["test_eval"]["fpr"])),
    ])

    # Bucket diagnostics on key methods.
    thr_hlt_test = threshold_for_target_tpr(y_test, scores_test["hlt"], target_tpr)

    pair_best_key = min(pair_results_valsel.keys(), key=lambda k: float(pair_results_valsel[k]["test_eval"]["fpr"]))
    pair_best = pair_results_valsel[pair_best_key]["test_eval"]
    pair_best_other = str(pair_best_key).replace("hlt_plus_", "")
    pair_best_score = (
        float(pair_best["w_a"]) * scores_test["hlt"]
        + float(pair_best["w_b"]) * scores_test[pair_best_other]
    )

    raw_w = np.asarray(raw_all5_valsel["test_eval"]["weights"], dtype=np.float64)
    raw_score = np.sum(mat_test * raw_w[:, None], axis=0)
    raw_thr = float(raw_all5_valsel["test_eval"]["threshold_from_val"])

    iso_w = np.asarray(iso_all5_valsel["test_eval"]["weights"], dtype=np.float64)
    iso_score = np.sum(mat_iso_test * iso_w[:, None], axis=0)
    iso_thr = float(iso_all5_valsel["test_eval"]["threshold_from_val"])

    methods_bucket = {
        f"pair_best_{pair_best_key}": {
            "scores": pair_best_score,
            "threshold": float(pair_best["threshold_from_val"]),
        },
        "all5_raw_valsel": {
            "scores": raw_score,
            "threshold": raw_thr,
        },
        "all5_iso_valsel": {
            "scores": iso_score,
            "threshold": iso_thr,
        },
        "meta_iso": {
            "scores": meta_iso_pred_test,
            "threshold": float(meta_iso_thr),
        },
    }

    bucket_reports = {
        "hlt_score_deciles": build_bucket_delta_report(
            labels=y_test,
            hlt_scores=scores_test["hlt"],
            hlt_threshold=thr_hlt_test,
            methods=methods_bucket,
            bucket_values=scores_test["hlt"],
            edges=_quantile_edges(scores_test["hlt"], int(args.bucket_deciles)),
            bucket_name="hlt_score_deciles",
        )
    }

    if "hlt_nconst_test" in z2:
        nconst = np.asarray(z2["hlt_nconst_test"], dtype=np.float64)
        n_edges = np.array([0, 10, 20, 30, 40, 50, 60, 80, 120], dtype=np.float64)
        bucket_reports["hlt_nconst_bins"] = build_bucket_delta_report(
            labels=y_test,
            hlt_scores=scores_test["hlt"],
            hlt_threshold=thr_hlt_test,
            methods=methods_bucket,
            bucket_values=nconst,
            edges=n_edges,
            bucket_name="hlt_nconst_bins",
        )

    out = {
        "target_tpr": float(target_tpr),
        "inputs": {
            "joint_delta_run_dir": str(dir_joint_delta),
            "reco_teacher_s09_run_dir": str(dir_reco_s09),
            "corrected_s01_run_dir": str(dir_corr_s01),
            "joint_s01_run_dir": str(dir_joint_s01),
            "score_files": {
                "joint_delta": str(joint_delta_npz),
                "reco_teacher_s09": str(reco_s09_npz),
                "corrected_s01": str(corr_s01_npz),
                "joint_s01": str(joint_s01_npz),
            },
        },
        "models_order": MODEL_ORDER,
        "individual_models": indiv,
        "test_overlap": overlap_test,
        "pair_combos_hlt_plus_x": {
            "val_selected_eval_test": pair_results_valsel,
            "test_oracle": pair_results_oracle,
        },
        "all5_weighted": {
            "candidate_count": int(W.shape[0]),
            "raw_val_selected_eval_test": raw_all5_valsel,
            "raw_test_oracle": raw_all5_oracle,
            "platt_val_selected_eval_test": platt_all5_valsel,
            "platt_test_oracle": platt_all5_oracle,
            "isotonic_val_selected_eval_test": iso_all5_valsel,
            "isotonic_test_oracle": iso_all5_oracle,
        },
        "calibration": {
            "platt": cal_platt,
            "isotonic": cal_iso,
        },
        "meta_fuser": {
            "raw": meta_raw,
            "platt_inputs": meta_platt,
            "isotonic_inputs": meta_iso,
        },
        "stepwise_gain_vs_hlt": stepwise,
        "bucket_analysis": bucket_reports,
    }

    out_path = dir_joint_delta / str(args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=" * 70)
    print("HLT + Joint(2x) + RecoTeacher + Corrected Fusion Analysis")
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

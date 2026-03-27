#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusion analysis for HLT + 17 trained models (18 total including HLT).

Outputs:
- Individual model metrics (val/test AUC + FPR@target TPR)
- HLT+X weighted pair combos (val-selected->test and test oracle)
- All-model weighted fusion (raw/platt/isotonic)
- Meta-fuser (raw/platt/isotonic features)
- Overlap report on test
- Ranked summary of best non-oracle and oracle test FPR@target TPR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

    lr = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced")
    lr.fit(s_val.reshape(-1, 1), y_val)
    pv = lr.predict_proba(s_val.reshape(-1, 1))[:, 1].astype(np.float64)
    pt = lr.predict_proba(s_test.reshape(-1, 1))[:, 1].astype(np.float64)
    return pv, pt, {"ok": True, "coef": float(lr.coef_.ravel()[0]), "intercept": float(lr.intercept_.ravel()[0])}


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
    n_sparse_random: int = 0,
    sparse_k_grid: Sequence[int] | None = None,
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

    rng = np.random.default_rng(int(seed))

    if int(n_random) > 0:
        rand_w = rng.dirichlet(alpha=np.ones(n_models, dtype=np.float64), size=int(n_random)).astype(np.float64)
        rows.extend([rand_w[i] for i in range(rand_w.shape[0])])

    n_sparse = int(max(0, n_sparse_random))
    if n_sparse > 0:
        if sparse_k_grid is None:
            sparse_k_grid = [2, 3, 4, 5, 6, 8, 10, 12]
        ks = sorted(set(max(1, min(int(k), n_models)) for k in sparse_k_grid))
        if not ks:
            ks = [min(4, n_models)]

        k_choices = np.asarray(ks, dtype=np.int64)
        k_idx = rng.integers(0, len(k_choices), size=n_sparse)

        for i in range(n_sparse):
            k = int(k_choices[int(k_idx[i])])
            idx = rng.choice(n_models, size=k, replace=False)
            w_sub = rng.dirichlet(alpha=np.ones(k, dtype=np.float64)).astype(np.float64)
            v = np.zeros(n_models, dtype=np.float64)
            v[idx] = w_sub
            rows.append(v)

    W = np.vstack(rows).astype(np.float64)
    W = np.clip(W, 0.0, None)
    W /= np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)

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
                "auc": float("nan"),
                "fpr_at_target_tpr_exact": float("nan"),
            },
        }

    score_test = np.sum(np.asarray(score_mat_test, dtype=np.float64) * weights[:, None], axis=0)
    rates = rates_from_threshold(y_test.astype(np.float32), score_test, thr)
    a = auc_and_fpr_at_target(y_test, score_test, target_tpr)
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
            "auc": float(a["auc"]),
            "fpr_at_target_tpr_exact": float(a["fpr_at_target_tpr"]),
        },
    }


def build_meta_features(model_order: List[str], score_dict: Dict[str, np.ndarray]) -> np.ndarray:
    mat = np.column_stack([np.asarray(score_dict[n], dtype=np.float64) for n in model_order])
    hlt_idx = model_order.index("hlt")
    hlt = mat[:, [hlt_idx]]
    abs_hlt = np.abs(mat - hlt)

    non_hlt_mask = np.ones(mat.shape[1], dtype=bool)
    non_hlt_mask[hlt_idx] = False
    others = mat[:, non_hlt_mask]
    mean_other = others.mean(axis=1, keepdims=True)
    max_other = others.max(axis=1, keepdims=True)
    min_other = others.min(axis=1, keepdims=True)
    std_other = others.std(axis=1, keepdims=True)

    return np.concatenate([mat, abs_hlt, mean_other, max_other, min_other, std_other], axis=1).astype(np.float64)


def train_select_meta_fuser(
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_tpr: float,
    sel_frac: float,
    c_grid: List[float],
    seed: int,
) -> Dict[str, object]:
    yv = y_val.astype(np.int64)
    if np.unique(yv).size < 2:
        return {
            "selection": {"ok": False, "reason": "single_class_val"},
            "test_eval": {"auc": float("nan"), "fpr": float("nan"), "tpr": float("nan"), "fpr_at_target_tpr_exact": float("nan")},
            "oracle_test": {"auc": float("nan"), "fpr_at_target_tpr_exact": float("nan")},
        }

    sel_frac = float(np.clip(sel_frac, 0.1, 0.8))
    idx = np.arange(yv.shape[0])
    idx_fit, idx_sel = train_test_split(
        idx,
        test_size=sel_frac,
        random_state=int(seed),
        stratify=yv,
    )

    best = None
    for c in c_grid:
        lr = LogisticRegression(
            C=float(c),
            solver="lbfgs",
            max_iter=4000,
            class_weight="balanced",
        )
        lr.fit(X_val[idx_fit], yv[idx_fit])
        s_sel = lr.predict_proba(X_val[idx_sel])[:, 1].astype(np.float64)
        thr_sel = threshold_for_target_tpr(y_val[idx_sel], s_sel, target_tpr)
        rates_sel = rates_from_threshold(y_val[idx_sel], s_sel, thr_sel)
        cand = {
            "C": float(c),
            "model": lr,
            "thr_sel": float(thr_sel),
            "fpr_sel": float(rates_sel["fpr"]),
            "tpr_sel": float(rates_sel["tpr"]),
        }
        if best is None or cand["fpr_sel"] < best["fpr_sel"]:
            best = cand

    assert best is not None
    lr_best = best["model"]

    s_val = lr_best.predict_proba(X_val)[:, 1].astype(np.float64)
    s_test = lr_best.predict_proba(X_test)[:, 1].astype(np.float64)

    thr_val = threshold_for_target_tpr(y_val, s_val, target_tpr)
    rates_test = rates_from_threshold(y_test, s_test, thr_val)
    a_test = auc_and_fpr_at_target(y_test, s_test, target_tpr)

    thr_oracle = threshold_for_target_tpr(y_test, s_test, target_tpr)
    rates_oracle = rates_from_threshold(y_test, s_test, thr_oracle)

    return {
        "selection": {
            "ok": True,
            "C": float(best["C"]),
            "thr_sel": float(best["thr_sel"]),
            "fpr_sel": float(best["fpr_sel"]),
            "tpr_sel": float(best["tpr_sel"]),
            "thr_val": float(thr_val),
        },
        "test_eval": {
            "auc": float(a_test["auc"]),
            "fpr": float(rates_test["fpr"]),
            "tpr": float(rates_test["tpr"]),
            "tp": int(rates_test["tp"]),
            "fp": int(rates_test["fp"]),
            "fpr_at_target_tpr_exact": float(a_test["fpr_at_target_tpr"]),
        },
        "oracle_test": {
            "auc": float(a_test["auc"]),
            "fpr": float(rates_oracle["fpr"]),
            "tpr": float(rates_oracle["tpr"]),
            "tp": int(rates_oracle["tp"]),
            "fp": int(rates_oracle["fp"]),
            "fpr_at_target_tpr_exact": float(a_test["fpr_at_target_tpr"]),
        },
        "scores_val": s_val,
        "scores_test": s_test,
    }


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"Missing score file: {path}")
    return np.load(path)


def _pick_score(z: np.lib.npyio.NpzFile, key_candidates: List[str], split: str, ref_len: int) -> np.ndarray:
    for k in key_candidates:
        if k in z:
            arr = np.asarray(z[k], dtype=np.float64)
            if arr.size == ref_len:
                return arr
    raise KeyError(
        f"Could not find usable {split} score among keys={key_candidates}; "
        f"available keys={list(z.keys())}"
    )


def _collect_candidates(results: Dict[str, object]) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []

    def _flt(v: object, default: float) -> float:
        try:
            f = float(v)
        except Exception:
            return float(default)
        if not np.isfinite(f):
            return float(default)
        return float(f)

    def _get(d: object, key: str, default: float) -> float:
        if isinstance(d, dict):
            return _flt(d.get(key, default), default)
        return float(default)

    for name, met in results["individual"].items():
        out.append(
            {
                "name": f"indiv::{name}",
                "fpr": _get(met, "fpr_test", float("inf")),
                "auc": _get(met, "auc_test", float("nan")),
                "oracle": False,
            }
        )

    for name, pack in results["pair_results_valsel"].items():
        te = pack.get("test_eval", {}) if isinstance(pack, dict) else {}
        out.append(
            {
                "name": f"pair_valsel::{name}",
                "fpr": _get(te, "fpr", float("inf")),
                "auc": _get(te, "auc", float("nan")),
                "oracle": False,
            }
        )

    for name, pack in results["pair_results_oracle"].items():
        out.append(
            {
                "name": f"pair_oracle::{name}",
                "fpr": _get(pack, "fpr", float("inf")),
                "auc": _get(pack, "auc", float("nan")),
                "oracle": True,
            }
        )

    for k in [
        "all18_weighted_raw_valsel",
        "all18_weighted_platt_valsel",
        "all18_weighted_iso_valsel",
    ]:
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        te = pack.get("test_eval", {}) if isinstance(pack, dict) else {}
        out.append(
            {
                "name": k,
                "fpr": _get(te, "fpr", float("inf")),
                "auc": _get(te, "auc", float("nan")),
                "oracle": False,
            }
        )

    for k in [
        "all18_weighted_raw_oracle",
        "all18_weighted_platt_oracle",
        "all18_weighted_iso_oracle",
    ]:
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        out.append(
            {
                "name": k,
                "fpr": _get(pack, "fpr", float("inf")),
                "auc": _get(pack, "auc", float("nan")),
                "oracle": True,
            }
        )

    for k in ["meta_raw", "meta_platt", "meta_iso"]:
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        te = pack.get("test_eval", {}) if isinstance(pack, dict) else {}
        orc = pack.get("oracle_test", {}) if isinstance(pack, dict) else {}
        out.append(
            {
                "name": f"{k}::valsel",
                "fpr": _get(te, "fpr", float("inf")),
                "auc": _get(te, "auc", float("nan")),
                "oracle": False,
            }
        )
        out.append(
            {
                "name": f"{k}::oracle",
                "fpr": _get(orc, "fpr", float("inf")),
                "auc": _get(orc, "auc", float("nan")),
                "oracle": True,
            }
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fusion analysis for HLT + 17 additional models")
    ap.add_argument("--joint_delta_run_dir", type=str, required=True)
    ap.add_argument("--reco_teacher_s09_run_dir", type=str, required=True)
    ap.add_argument("--corrected_s01_run_dir", type=str, required=True)
    ap.add_argument("--joint_s01_run_dir", type=str, required=True)
    ap.add_argument("--concat_run_dir", type=str, required=True)

    ap.add_argument("--m7_residual_run_dir", type=str, required=True)
    ap.add_argument("--m8_direct_residual_run_dir", type=str, required=True)
    ap.add_argument("--m9_low_run_dir", type=str, required=True)
    ap.add_argument("--m9_mid_run_dir", type=str, required=True)
    ap.add_argument("--m9_high_run_dir", type=str, required=True)

    ap.add_argument("--m4_k40_run_dir", type=str, required=True)
    ap.add_argument("--m4_k60_run_dir", type=str, required=True)
    ap.add_argument("--m4_k80_run_dir", type=str, required=True)

    ap.add_argument("--m10_run_dir", type=str, required=True)
    ap.add_argument("--m11_run_dir", type=str, required=True)
    ap.add_argument("--m12_run_dir", type=str, required=True)
    ap.add_argument("--m13_run_dir", type=str, required=True)

    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--weight_step_2", type=float, default=0.01)
    ap.add_argument("--weight_samples_multi", type=int, default=10000)
    ap.add_argument("--pair_grid_step_multi", type=float, default=0.10)
    ap.add_argument("--meta_sel_frac", type=float, default=0.30)
    ap.add_argument("--meta_c_grid", type=str, default="0.05,0.1,0.3,1,3,10,30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_name", type=str, default="fusion_hlt_joint18_analysis.json")
    args = ap.parse_args()

    target_tpr = float(args.target_tpr)

    dir_joint_delta = Path(args.joint_delta_run_dir)
    z2 = _load_npz(dir_joint_delta / "fusion_scores_val_test.npz")
    y_val = np.asarray(z2["labels_val"], dtype=np.float32)
    y_test = np.asarray(z2["labels_test"], dtype=np.float32)

    scores_val: Dict[str, np.ndarray] = {
        "hlt": np.asarray(z2["preds_hlt_val"], dtype=np.float64),
        "joint_delta": np.asarray(z2["preds_joint_val"], dtype=np.float64),
    }
    scores_test: Dict[str, np.ndarray] = {
        "hlt": np.asarray(z2["preds_hlt_test"], dtype=np.float64),
        "joint_delta": np.asarray(z2["preds_joint_test"], dtype=np.float64),
    }

    model_specs = [
        ("reco_teacher_s09", Path(args.reco_teacher_s09_run_dir) / "stageA_only_scores.npz", ["preds_reco_teacher_val"], ["preds_reco_teacher_test"]),
        ("corrected_s01", Path(args.corrected_s01_run_dir) / "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("joint_s01", Path(args.joint_s01_run_dir) / "fusion_scores_val_test.npz", ["preds_joint_val"], ["preds_joint_test"]),
        ("concat_corrected", Path(args.concat_run_dir) / "concat_teacher_stageA_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),

        ("residual_m7", Path(args.m7_residual_run_dir) / "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("direct_residual_m8", Path(args.m8_direct_residual_run_dir) / "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("offdrop_low", Path(args.m9_low_run_dir) / "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("offdrop_mid", Path(args.m9_mid_run_dir) / "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("offdrop_high", Path(args.m9_high_run_dir) / "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),

        ("corrected_k40", Path(args.m4_k40_run_dir) / "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("corrected_k60", Path(args.m4_k60_run_dir) / "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("corrected_k80", Path(args.m4_k80_run_dir) / "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),

        ("antioverlap_m10", Path(args.m10_run_dir) / "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("feat_noangle_m11", Path(args.m11_run_dir) / "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("feat_noscale_m12", Path(args.m12_run_dir) / "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("feat_coreshape_m13", Path(args.m13_run_dir) / "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
    ]

    score_files = {"joint_delta": str((dir_joint_delta / "fusion_scores_val_test.npz").resolve())}
    for name, npz_path, k_val, k_test in model_specs:
        z = _load_npz(npz_path)
        yv = np.asarray(z["labels_val"], dtype=np.float32)
        yt = np.asarray(z["labels_test"], dtype=np.float32)
        if not np.array_equal(y_val, yv):
            raise RuntimeError(f"Validation labels mismatch: {name} ({npz_path})")
        if not np.array_equal(y_test, yt):
            raise RuntimeError(f"Test labels mismatch: {name} ({npz_path})")

        scores_val[name] = _pick_score(z, k_val, "val", ref_len=y_val.size)
        scores_test[name] = _pick_score(z, k_test, "test", ref_len=y_test.size)
        score_files[name] = str(npz_path.resolve())

    model_order = [
        "hlt",
        "joint_delta",
        "reco_teacher_s09",
        "corrected_s01",
        "joint_s01",
        "concat_corrected",
        "residual_m7",
        "direct_residual_m8",
        "offdrop_low",
        "offdrop_mid",
        "offdrop_high",
        "corrected_k40",
        "corrected_k60",
        "corrected_k80",
        "antioverlap_m10",
        "feat_noangle_m11",
        "feat_noscale_m12",
        "feat_coreshape_m13",
    ]

    for n in model_order:
        if n not in scores_val:
            raise KeyError(f"Missing model score: {n}")

    indiv: Dict[str, Dict[str, float]] = {}
    for name in model_order:
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
        model_preds={k: scores_test[k] for k in model_order},
        target_tpr=target_tpr,
    )

    pair_results_valsel: Dict[str, Dict[str, object]] = {}
    pair_results_oracle: Dict[str, Dict[str, object]] = {}
    for other in model_order[1:]:
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
        po = m.search_best_weighted_combo_at_tpr(
            labels=y_test,
            preds_a=scores_test["hlt"],
            preds_b=scores_test[other],
            name_a="hlt",
            name_b=other,
            target_tpr=target_tpr,
            weight_step=float(args.weight_step_2),
        )
        # add auc/fpr_at_target exact for consistent summary
        ps = po["w_a"] * scores_test["hlt"] + po["w_b"] * scores_test[other]
        pa = auc_and_fpr_at_target(y_test, ps, target_tpr)
        po["auc"] = float(pa["auc"])
        po["fpr_at_target_tpr_exact"] = float(pa["fpr_at_target_tpr"])
        pair_results_oracle[key] = po

    weight_candidates = generate_weight_candidates(
        n_models=len(model_order),
        n_random=int(args.weight_samples_multi),
        seed=int(args.seed),
        include_pair_grid=True,
        pair_step=float(args.pair_grid_step_multi),
    )

    mat_val = np.vstack([scores_val[n] for n in model_order])
    mat_test = np.vstack([scores_test[n] for n in model_order])

    all18_weighted_raw_valsel = select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_val,
        y_test=y_test,
        score_mat_test=mat_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )
    all18_weighted_raw_oracle = search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    cal_platt_meta = {}
    cal_iso_meta = {}
    scores_platt_val: Dict[str, np.ndarray] = {}
    scores_platt_test: Dict[str, np.ndarray] = {}
    scores_iso_val: Dict[str, np.ndarray] = {}
    scores_iso_test: Dict[str, np.ndarray] = {}

    for name in model_order:
        pv, pt, pm = calibrate_platt(y_val, scores_val[name], scores_test[name])
        iv, it, im = calibrate_isotonic(y_val, scores_val[name], scores_test[name])
        scores_platt_val[name] = pv
        scores_platt_test[name] = pt
        scores_iso_val[name] = iv
        scores_iso_test[name] = it
        cal_platt_meta[name] = pm
        cal_iso_meta[name] = im

    mat_platt_val = np.vstack([scores_platt_val[n] for n in model_order])
    mat_platt_test = np.vstack([scores_platt_test[n] for n in model_order])
    mat_iso_val = np.vstack([scores_iso_val[n] for n in model_order])
    mat_iso_test = np.vstack([scores_iso_test[n] for n in model_order])

    all18_weighted_platt_valsel = select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_platt_val,
        y_test=y_test,
        score_mat_test=mat_platt_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )
    all18_weighted_platt_oracle = search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_platt_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    all18_weighted_iso_valsel = select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_iso_val,
        y_test=y_test,
        score_mat_test=mat_iso_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )
    all18_weighted_iso_oracle = search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_iso_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    c_grid = [float(x.strip()) for x in str(args.meta_c_grid).split(",") if x.strip()]
    meta_raw = train_select_meta_fuser(
        X_val=build_meta_features(model_order, scores_val),
        y_val=y_val,
        X_test=build_meta_features(model_order, scores_test),
        y_test=y_test,
        target_tpr=target_tpr,
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
        seed=int(args.seed),
    )
    meta_platt = train_select_meta_fuser(
        X_val=build_meta_features(model_order, scores_platt_val),
        y_val=y_val,
        X_test=build_meta_features(model_order, scores_platt_test),
        y_test=y_test,
        target_tpr=target_tpr,
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
        seed=int(args.seed),
    )
    meta_iso = train_select_meta_fuser(
        X_val=build_meta_features(model_order, scores_iso_val),
        y_val=y_val,
        X_test=build_meta_features(model_order, scores_iso_test),
        y_test=y_test,
        target_tpr=target_tpr,
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
        seed=int(args.seed),
    )

    results = {
        "config": {
            "target_tpr": target_tpr,
            "weight_step_2": float(args.weight_step_2),
            "weight_samples_multi": int(args.weight_samples_multi),
            "pair_grid_step_multi": float(args.pair_grid_step_multi),
            "meta_sel_frac": float(args.meta_sel_frac),
            "meta_c_grid": c_grid,
            "seed": int(args.seed),
        },
        "run_dirs": {
            "joint_delta_run_dir": str(Path(args.joint_delta_run_dir)),
            "reco_teacher_s09_run_dir": str(Path(args.reco_teacher_s09_run_dir)),
            "corrected_s01_run_dir": str(Path(args.corrected_s01_run_dir)),
            "joint_s01_run_dir": str(Path(args.joint_s01_run_dir)),
            "concat_run_dir": str(Path(args.concat_run_dir)),
            "m7_residual_run_dir": str(Path(args.m7_residual_run_dir)),
            "m8_direct_residual_run_dir": str(Path(args.m8_direct_residual_run_dir)),
            "m9_low_run_dir": str(Path(args.m9_low_run_dir)),
            "m9_mid_run_dir": str(Path(args.m9_mid_run_dir)),
            "m9_high_run_dir": str(Path(args.m9_high_run_dir)),
            "m4_k40_run_dir": str(Path(args.m4_k40_run_dir)),
            "m4_k60_run_dir": str(Path(args.m4_k60_run_dir)),
            "m4_k80_run_dir": str(Path(args.m4_k80_run_dir)),
            "m10_run_dir": str(Path(args.m10_run_dir)),
            "m11_run_dir": str(Path(args.m11_run_dir)),
            "m12_run_dir": str(Path(args.m12_run_dir)),
            "m13_run_dir": str(Path(args.m13_run_dir)),
            "score_files": score_files,
        },
        "models_order": model_order,
        "individual": indiv,
        "overlap_test": overlap_test,
        "pair_results_valsel": pair_results_valsel,
        "pair_results_oracle": pair_results_oracle,
        "all18_weighted_raw_valsel": all18_weighted_raw_valsel,
        "all18_weighted_raw_oracle": all18_weighted_raw_oracle,
        "all18_weighted_platt_valsel": all18_weighted_platt_valsel,
        "all18_weighted_platt_oracle": all18_weighted_platt_oracle,
        "all18_weighted_iso_valsel": all18_weighted_iso_valsel,
        "all18_weighted_iso_oracle": all18_weighted_iso_oracle,
        "calibration": {
            "platt": cal_platt_meta,
            "isotonic": cal_iso_meta,
        },
        "meta_raw": {
            "selection": meta_raw["selection"],
            "test_eval": meta_raw["test_eval"],
            "oracle_test": meta_raw["oracle_test"],
        },
        "meta_platt": {
            "selection": meta_platt["selection"],
            "test_eval": meta_platt["test_eval"],
            "oracle_test": meta_platt["oracle_test"],
        },
        "meta_iso": {
            "selection": meta_iso["selection"],
            "test_eval": meta_iso["test_eval"],
            "oracle_test": meta_iso["oracle_test"],
        },
    }

    all_candidates = _collect_candidates(results)
    non_oracle = [x for x in all_candidates if not bool(x["oracle"])]
    oracle = [x for x in all_candidates if bool(x["oracle"])]
    non_oracle_sorted = sorted(non_oracle, key=lambda d: float(d["fpr"]))
    oracle_sorted = sorted(oracle, key=lambda d: float(d["fpr"]))
    results["best_summary"] = {
        "best_non_oracle": non_oracle_sorted[0] if non_oracle_sorted else None,
        "best_oracle": oracle_sorted[0] if oracle_sorted else None,
        "top10_non_oracle": non_oracle_sorted[:10],
        "top10_oracle": oracle_sorted[:10],
    }

    out_path = dir_joint_delta / str(args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    b = results["best_summary"]["best_non_oracle"]
    if b is not None:
        print(
            f"Best non-oracle @TPR={target_tpr:.2f}: {b['name']} | "
            f"FPR={float(b['fpr']):.6f} | AUC={float(b['auc']):.6f}"
        )
    bo = results["best_summary"]["best_oracle"]
    if bo is not None:
        print(
            f"Best oracle @TPR={target_tpr:.2f}: {bo['name']} | "
            f"FPR={float(bo['fpr']):.6f}"
        )

    print(f"Saved fusion analysis to: {out_path}")


if __name__ == "__main__":
    main()

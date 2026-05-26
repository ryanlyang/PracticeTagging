#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stacked-logreg fusion for the 3 finished weighted 5m1m1m runs (+ optional HLT).

This script is score-level only (no model retraining). It loads saved NPZ score
artifacts, optionally calibrates each source score on validation data, then
compares:
  - uniform_prob_avg
  - weighted_prob_avg
  - weighted_logit_avg
  - stacked_logreg

It also supports pre/post-joint style source selection per run:
  - m4 aliases: prejoint->reco_teacher, postjoint->corrected_only
  - m9 aliases: prejoint->residual_frozen, postjoint->residual_joint
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


M4_KEYS: Dict[str, Tuple[str, str]] = {
    "hlt": ("preds_hlt_val", "preds_hlt_test"),
    "reco_teacher": ("preds_reco_teacher_val", "preds_reco_teacher_test"),
    "corrected_only": ("preds_corrected_only_val", "preds_corrected_only_test"),
}

M9_KEYS: Dict[str, Tuple[str, str]] = {
    "hlt": ("preds_hlt_val", "preds_hlt_test"),
    "reco_teacher": ("preds_reco_teacher_val", "preds_reco_teacher_test"),
    "residual_frozen": ("preds_residual_frozen_val", "preds_residual_frozen_test"),
    "residual_joint": ("preds_residual_joint_val", "preds_residual_joint_test"),
}

M4_ALIASES: Dict[str, str] = {
    "pre": "reco_teacher",
    "post": "corrected_only",
    "prejoint": "reco_teacher",
    "postjoint": "corrected_only",
}

M9_ALIASES: Dict[str, str] = {
    "pre": "residual_frozen",
    "post": "residual_joint",
    "prejoint": "residual_frozen",
    "postjoint": "residual_joint",
}


def _parse_csv_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _safe_logit(p: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=np.float64)
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _threshold_for_target_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> float:
    y = np.asarray(labels, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float64)
    target_tpr = float(np.clip(target_tpr, 0.0, 1.0))
    pos = s[y > 0.5]
    if pos.size == 0:
        return float("inf")
    q = float(np.clip(1.0 - target_tpr, 0.0, 1.0))
    return float(np.quantile(pos, q=q))


def _rates_from_threshold(labels: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    y_pos = np.asarray(labels, dtype=np.float32) > 0.5
    y_neg = ~y_pos
    pred = np.asarray(scores, dtype=np.float64) >= float(thr)
    tp = int((pred & y_pos).sum())
    fp = int((pred & y_neg).sum())
    n_pos = int(y_pos.sum())
    n_neg = int(y_neg.sum())
    tpr = float(tp / max(n_pos, 1))
    fpr = float(fp / max(n_neg, 1))
    return {"tp": tp, "fp": fp, "n_pos": n_pos, "n_neg": n_neg, "tpr": tpr, "fpr": fpr}


def _eval_metrics(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float64)
    thr = _threshold_for_target_tpr(y, s, float(target_tpr))
    rr = _rates_from_threshold(y, s, thr)
    auc = float(roc_auc_score(y, s)) if np.unique(y.astype(np.int64)).size > 1 else float("nan")
    return {
        "auc": float(auc),
        "threshold_at_target_tpr": float(thr),
        "fpr_at_target_tpr": float(rr["fpr"]),
        "tpr_at_target_tpr": float(rr["tpr"]),
        "tp_at_target_tpr": float(rr["tp"]),
        "fp_at_target_tpr": float(rr["fp"]),
    }


def _objective(metrics: Dict[str, float], optimize_for: str) -> float:
    if optimize_for == "fpr_at_tpr":
        fpr = float(metrics.get("fpr_at_target_tpr", float("nan")))
        return -fpr if np.isfinite(fpr) else float("-inf")
    if optimize_for == "auc":
        auc = float(metrics.get("auc", float("nan")))
        return auc if np.isfinite(auc) else float("-inf")
    raise ValueError(f"Unknown optimize_for={optimize_for}")


def _n_simplex_grid_candidates(n_models: int, step: float) -> int:
    k = int(round(1.0 / float(step)))
    if n_models < 2 or k <= 0:
        raise ValueError(f"Invalid simplex config: n_models={n_models}, step={step}")
    return int(math.comb(k + n_models - 1, n_models - 1))


def _simplex_grid(n_models: int, step: float) -> np.ndarray:
    k = int(round(1.0 / float(step)))
    out: List[List[int]] = []

    def rec(i: int, rem: int, cur: List[int]) -> None:
        if i == n_models - 1:
            out.append(cur + [rem])
            return
        for v in range(rem + 1):
            rec(i + 1, rem - v, cur + [v])

    rec(0, k, [])
    return np.asarray(out, dtype=np.float64) / float(k)


def _sample_simplex_dirichlet(n_models: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    sampled = rng.dirichlet(alpha=np.ones((n_models,), dtype=np.float64), size=int(max(1, n_samples)))
    uniform = np.full((1, n_models), 1.0 / float(n_models), dtype=np.float64)
    onehots = np.eye(n_models, dtype=np.float64)
    return np.concatenate([uniform, onehots, sampled], axis=0)


def _build_weight_candidates(
    n_models: int,
    step: float,
    mode: str,
    max_candidates: int,
    random_samples: int,
    random_seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    m = str(mode).strip().lower()
    if m not in {"auto", "grid", "dirichlet"}:
        raise ValueError(f"Unsupported weight_search_mode={mode}")

    grid_count = _n_simplex_grid_candidates(n_models=n_models, step=step)
    max_c = int(max(1, max_candidates))
    info: Dict[str, object] = {
        "requested_mode": m,
        "grid_candidate_count": int(grid_count),
        "max_weight_candidates": int(max_c),
    }

    if m == "grid":
        if grid_count > max_c:
            raise ValueError(
                f"Grid has {grid_count} candidates (> {max_c}). "
                "Use --weight_search_mode dirichlet or larger --weight_step."
            )
        w = _simplex_grid(n_models=n_models, step=step)
        info.update({"strategy": "grid", "actual_candidate_count": int(w.shape[0])})
        return w, info

    if m == "dirichlet":
        w = _sample_simplex_dirichlet(n_models=n_models, n_samples=random_samples, seed=random_seed)
        info.update(
            {
                "strategy": "dirichlet",
                "actual_candidate_count": int(w.shape[0]),
                "dirichlet_samples": int(max(1, random_samples)),
                "dirichlet_seed": int(random_seed),
            }
        )
        return w, info

    if grid_count <= max_c:
        w = _simplex_grid(n_models=n_models, step=step)
        info.update({"strategy": "grid_auto", "actual_candidate_count": int(w.shape[0])})
        return w, info

    w = _sample_simplex_dirichlet(n_models=n_models, n_samples=random_samples, seed=random_seed)
    info.update(
        {
            "strategy": "dirichlet_auto",
            "actual_candidate_count": int(w.shape[0]),
            "dirichlet_samples": int(max(1, random_samples)),
            "dirichlet_seed": int(random_seed),
        }
    )
    return w, info


def _fuse_prob(weights: np.ndarray, score_mat: np.ndarray) -> np.ndarray:
    return np.asarray(score_mat, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)


def _fuse_logit(weights: np.ndarray, score_mat: np.ndarray) -> np.ndarray:
    lg = _safe_logit(score_mat)
    return _sigmoid(np.asarray(lg, dtype=np.float64) @ np.asarray(weights, dtype=np.float64))


def _search_best_weights(
    weight_candidates: np.ndarray,
    y_val: np.ndarray,
    score_mat_val: np.ndarray,
    target_tpr: float,
    optimize_for: str,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    best_w = np.asarray(weight_candidates[0], dtype=np.float64).copy()
    best_score = float("-inf")
    best_tie = float("-inf")
    best_metrics: Dict[str, float] = {}

    for w in weight_candidates:
        if mode == "prob":
            sv = _fuse_prob(w, score_mat_val)
        elif mode == "logit":
            sv = _fuse_logit(w, score_mat_val)
        else:
            raise ValueError(f"Unknown search mode={mode}")

        mv = _eval_metrics(y_val, sv, target_tpr)
        score = _objective(mv, optimize_for)
        tie = float(mv.get("auc", float("nan")))
        tie = tie if np.isfinite(tie) else float("-inf")

        if (score > best_score) or (np.isclose(score, best_score) and tie > best_tie):
            best_score = float(score)
            best_tie = float(tie)
            best_w = np.asarray(w, dtype=np.float64).copy()
            best_metrics = dict(mv)

    return best_w.astype(np.float64), best_metrics


def _calibrate_binary_scores(
    y_val: np.ndarray,
    s_val: np.ndarray,
    s_test: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    mode = str(mode).strip().lower()
    y = np.asarray(y_val, dtype=np.int64)
    sv = np.asarray(s_val, dtype=np.float64)
    st = np.asarray(s_test, dtype=np.float64)

    if mode == "raw" or np.unique(y).size < 2:
        return sv.copy(), st.copy(), {"mode": "raw", "ok": True}

    if mode == "platt":
        lr = LogisticRegression(solver="lbfgs", max_iter=4000, class_weight="balanced")
        lr.fit(sv.reshape(-1, 1), y)
        return (
            lr.predict_proba(sv.reshape(-1, 1))[:, 1].astype(np.float64),
            lr.predict_proba(st.reshape(-1, 1))[:, 1].astype(np.float64),
            {
                "mode": "platt",
                "ok": True,
                "coef": float(lr.coef_.ravel()[0]),
                "intercept": float(lr.intercept_.ravel()[0]),
            },
        )

    if mode == "iso":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(sv.astype(np.float64), y.astype(np.float64))
        return (
            np.asarray(iso.transform(sv.astype(np.float64)), dtype=np.float64),
            np.asarray(iso.transform(st.astype(np.float64)), dtype=np.float64),
            {"mode": "iso", "ok": True},
        )

    raise ValueError(f"Unknown base_calibration={mode}")


def _assert_same_labels(npz: np.lib.npyio.NpzFile, yv_ref: np.ndarray, yt_ref: np.ndarray, tag: str) -> None:
    yv = np.asarray(npz["labels_val"], dtype=np.float32)
    yt = np.asarray(npz["labels_test"], dtype=np.float32)
    if not np.array_equal(yv_ref, yv):
        raise RuntimeError(f"Validation labels mismatch for {tag}")
    if not np.array_equal(yt_ref, yt):
        raise RuntimeError(f"Test labels mismatch for {tag}")


def _resolve_sources(
    npz: np.lib.npyio.NpzFile,
    requested_sources: Iterable[str],
    key_map: Dict[str, Tuple[str, str]],
    alias_map: Dict[str, str],
    prefix: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, str, str]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray, str, str]] = {}
    for raw_tok in requested_sources:
        tok = str(raw_tok).strip().lower()
        if not tok:
            continue
        canonical = alias_map.get(tok, tok)
        if canonical not in key_map:
            raise KeyError(f"Unsupported source `{raw_tok}` for {prefix}. Allowed: {sorted(key_map.keys())}")
        kv, kt = key_map[canonical]
        if kv not in npz or kt not in npz:
            raise KeyError(f"Missing keys for {prefix}:{canonical} -> ({kv}, {kt})")
        name = f"{prefix}_{canonical}"
        out[name] = (
            np.asarray(npz[kv], dtype=np.float64),
            np.asarray(npz[kt], dtype=np.float64),
            kv,
            kt,
        )
    if not out:
        raise ValueError(f"No valid sources resolved for {prefix}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="3-run stacked-logreg fusion (weighted 5m1m1m)")

    ap.add_argument(
        "--m4_npz",
        type=str,
        default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model4_recoteacher_s01_corrected_weighted_5m1m1m/model4_recoteacher_s01_corrected_weighted_5m1m1m_seed0/stageA_only_scores.npz",
    )
    ap.add_argument(
        "--m9mid_npz",
        type=str,
        default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m_seed0/stageA_residual_scores.npz",
    )
    ap.add_argument(
        "--m9high_npz",
        type=str,
        default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m_seed0/stageA_residual_scores.npz",
    )
    ap.add_argument(
        "--hlt_npz",
        type=str,
        default="",
        help="Optional external HLT source NPZ; if empty, HLT is taken from m9mid_npz.",
    )

    ap.add_argument(
        "--m4_sources",
        type=str,
        default="corrected_only",
        help="Comma list. Supports corrected_only,reco_teacher,hlt,prejoint,postjoint.",
    )
    ap.add_argument(
        "--m9mid_sources",
        type=str,
        default="residual_frozen,residual_joint",
        help="Comma list. Supports residual_frozen,residual_joint,reco_teacher,hlt,prejoint,postjoint.",
    )
    ap.add_argument(
        "--m9high_sources",
        type=str,
        default="residual_frozen,residual_joint",
        help="Comma list. Supports residual_frozen,residual_joint,reco_teacher,hlt,prejoint,postjoint.",
    )
    ap.add_argument("--include_hlt", type=int, default=1)

    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--optimize_for", type=str, default="fpr_at_tpr", choices=["fpr_at_tpr", "auc"])
    ap.add_argument("--base_calibration", type=str, default="raw", choices=["raw", "platt", "iso"])

    ap.add_argument("--weight_step", type=float, default=0.05)
    ap.add_argument("--weight_search_mode", type=str, default="auto", choices=["auto", "grid", "dirichlet"])
    ap.add_argument("--max_weight_candidates", type=int, default=250000)
    ap.add_argument("--weight_random_samples", type=int, default=5000)
    ap.add_argument("--weight_random_seed", type=int, default=52)

    ap.add_argument("--stack_features", type=str, default="logits_probs", choices=["logits", "probs", "logits_probs"])
    ap.add_argument("--stack_Cs", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    ap.add_argument("--stack_cv", type=int, default=5)
    ap.add_argument("--stack_max_iter", type=int, default=5000)
    ap.add_argument("--stack_n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/stacked_fusion_3_weighted_5m1m1m",
    )
    args = ap.parse_args()

    m4_npz_path = Path(args.m4_npz).expanduser().resolve()
    m9mid_npz_path = Path(args.m9mid_npz).expanduser().resolve()
    m9high_npz_path = Path(args.m9high_npz).expanduser().resolve()
    if not m4_npz_path.exists():
        raise FileNotFoundError(f"Missing m4 npz: {m4_npz_path}")
    if not m9mid_npz_path.exists():
        raise FileNotFoundError(f"Missing m9mid npz: {m9mid_npz_path}")
    if not m9high_npz_path.exists():
        raise FileNotFoundError(f"Missing m9high npz: {m9high_npz_path}")

    z_m9mid = np.load(m9mid_npz_path)
    y_val = np.asarray(z_m9mid["labels_val"], dtype=np.float32)
    y_test = np.asarray(z_m9mid["labels_test"], dtype=np.float32)

    z_m4 = np.load(m4_npz_path)
    z_m9high = np.load(m9high_npz_path)
    _assert_same_labels(z_m4, y_val, y_test, "m4")
    _assert_same_labels(z_m9high, y_val, y_test, "m9high")

    scores_val_raw: Dict[str, np.ndarray] = {}
    scores_test_raw: Dict[str, np.ndarray] = {}
    source_meta: Dict[str, Dict[str, str]] = {}

    m4_sources = _resolve_sources(z_m4, _parse_csv_list(args.m4_sources), M4_KEYS, M4_ALIASES, "m4")
    m9mid_sources = _resolve_sources(z_m9mid, _parse_csv_list(args.m9mid_sources), M9_KEYS, M9_ALIASES, "m9mid")
    m9high_sources = _resolve_sources(z_m9high, _parse_csv_list(args.m9high_sources), M9_KEYS, M9_ALIASES, "m9high")

    for name, (sv, st, kv, kt) in {**m4_sources, **m9mid_sources, **m9high_sources}.items():
        scores_val_raw[name] = sv
        scores_test_raw[name] = st
        source_meta[name] = {"val_key": kv, "test_key": kt}

    if int(args.include_hlt) != 0:
        if str(args.hlt_npz).strip():
            hlt_path = Path(args.hlt_npz).expanduser().resolve()
            if not hlt_path.exists():
                raise FileNotFoundError(f"Missing hlt npz: {hlt_path}")
            z_hlt = np.load(hlt_path)
            _assert_same_labels(z_hlt, y_val, y_test, "hlt_override")
            kv, kt = "preds_hlt_val", "preds_hlt_test"
            if kv not in z_hlt or kt not in z_hlt:
                raise KeyError(f"HLT override missing keys ({kv}, {kt}): {hlt_path}")
            sv = np.asarray(z_hlt[kv], dtype=np.float64)
            st = np.asarray(z_hlt[kt], dtype=np.float64)
            src_path = str(hlt_path)
        else:
            kv, kt = "preds_hlt_val", "preds_hlt_test"
            if kv not in z_m9mid or kt not in z_m9mid:
                raise KeyError(f"m9mid NPZ missing HLT keys ({kv}, {kt})")
            sv = np.asarray(z_m9mid[kv], dtype=np.float64)
            st = np.asarray(z_m9mid[kt], dtype=np.float64)
            src_path = str(m9mid_npz_path)
        scores_val_raw["hlt"] = sv
        scores_test_raw["hlt"] = st
        source_meta["hlt"] = {"val_key": kv, "test_key": kt, "path": src_path}

    model_names = sorted(scores_val_raw.keys())
    if len(model_names) < 2:
        raise RuntimeError(f"Need at least 2 model score sources; got: {model_names}")

    score_mat_val_raw = np.column_stack([scores_val_raw[n] for n in model_names]).astype(np.float64)
    score_mat_test_raw = np.column_stack([scores_test_raw[n] for n in model_names]).astype(np.float64)

    score_mat_val = np.zeros_like(score_mat_val_raw, dtype=np.float64)
    score_mat_test = np.zeros_like(score_mat_test_raw, dtype=np.float64)
    calibration_report: Dict[str, Dict[str, float]] = {}
    for i, n in enumerate(model_names):
        sv, st, meta = _calibrate_binary_scores(
            y_val=y_val,
            s_val=score_mat_val_raw[:, i],
            s_test=score_mat_test_raw[:, i],
            mode=str(args.base_calibration),
        )
        score_mat_val[:, i] = sv
        score_mat_test[:, i] = st
        calibration_report[n] = meta

    weights_uniform = np.full((len(model_names),), 1.0 / float(len(model_names)), dtype=np.float64)
    s_uni_val = _fuse_prob(weights_uniform, score_mat_val)
    s_uni_test = _fuse_prob(weights_uniform, score_mat_test)

    w_cands, w_info = _build_weight_candidates(
        n_models=len(model_names),
        step=float(args.weight_step),
        mode=str(args.weight_search_mode),
        max_candidates=int(args.max_weight_candidates),
        random_samples=int(args.weight_random_samples),
        random_seed=int(args.weight_random_seed),
    )

    w_prob, w_prob_search_metrics = _search_best_weights(
        weight_candidates=w_cands,
        y_val=y_val,
        score_mat_val=score_mat_val,
        target_tpr=float(args.target_tpr),
        optimize_for=str(args.optimize_for),
        mode="prob",
    )
    s_wprob_val = _fuse_prob(w_prob, score_mat_val)
    s_wprob_test = _fuse_prob(w_prob, score_mat_test)

    w_logit, w_logit_search_metrics = _search_best_weights(
        weight_candidates=w_cands,
        y_val=y_val,
        score_mat_val=score_mat_val,
        target_tpr=float(args.target_tpr),
        optimize_for=str(args.optimize_for),
        mode="logit",
    )
    s_wlog_val = _fuse_logit(w_logit, score_mat_val)
    s_wlog_test = _fuse_logit(w_logit, score_mat_test)

    if str(args.stack_features) == "logits":
        x_val = _safe_logit(score_mat_val).astype(np.float32)
        x_test = _safe_logit(score_mat_test).astype(np.float32)
    elif str(args.stack_features) == "probs":
        x_val = score_mat_val.astype(np.float32)
        x_test = score_mat_test.astype(np.float32)
    else:
        x_val = np.concatenate([_safe_logit(score_mat_val), score_mat_val], axis=1).astype(np.float32)
        x_test = np.concatenate([_safe_logit(score_mat_test), score_mat_test], axis=1).astype(np.float32)

    stack_pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegressionCV(
            Cs=[float(c) for c in args.stack_Cs],
            cv=int(args.stack_cv),
            scoring="roc_auc",
            solver="lbfgs",
            max_iter=int(args.stack_max_iter),
            n_jobs=int(args.stack_n_jobs),
            random_state=int(args.seed),
            refit=True,
        ),
    )
    stack_pipe.fit(x_val, y_val.astype(np.int64))
    s_stack_val = stack_pipe.predict_proba(x_val)[:, 1].astype(np.float64)
    s_stack_test = stack_pipe.predict_proba(x_test)[:, 1].astype(np.float64)

    lr_cv = stack_pipe.named_steps["logisticregressioncv"]
    c_arr = np.asarray(getattr(lr_cv, "C_", np.array([])), dtype=np.float64)
    stack_best_c_mean = float(np.mean(c_arr)) if c_arr.size > 0 else float("nan")

    method_scores_val = {
        "uniform_prob_avg": s_uni_val,
        "weighted_prob_avg": s_wprob_val,
        "weighted_logit_avg": s_wlog_val,
        "stacked_logreg": s_stack_val,
    }
    method_scores_test = {
        "uniform_prob_avg": s_uni_test,
        "weighted_prob_avg": s_wprob_test,
        "weighted_logit_avg": s_wlog_test,
        "stacked_logreg": s_stack_test,
    }

    method_metrics: Dict[str, Dict[str, object]] = {}
    best_method = ""
    best_obj = float("-inf")
    for name in method_scores_val:
        mv = _eval_metrics(y_val, method_scores_val[name], float(args.target_tpr))
        mt = _eval_metrics(y_test, method_scores_test[name], float(args.target_tpr))
        obj = _objective(mv, str(args.optimize_for))
        method_metrics[name] = {"val": mv, "test": mt, "val_objective": float(obj)}
        if obj > best_obj:
            best_obj = float(obj)
            best_method = str(name)

    individual_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i, n in enumerate(model_names):
        individual_metrics[n] = {
            "val_raw": _eval_metrics(y_val, score_mat_val_raw[:, i], float(args.target_tpr)),
            "test_raw": _eval_metrics(y_test, score_mat_test_raw[:, i], float(args.target_tpr)),
            "val_calibrated": _eval_metrics(y_val, score_mat_val[:, i], float(args.target_tpr)),
            "test_calibrated": _eval_metrics(y_test, score_mat_test[:, i], float(args.target_tpr)),
        }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "setup": {
            "target_tpr": float(args.target_tpr),
            "optimize_for": str(args.optimize_for),
            "base_calibration": str(args.base_calibration),
            "stack_features": str(args.stack_features),
            "stack_Cs": [float(c) for c in args.stack_Cs],
            "stack_cv": int(args.stack_cv),
            "stack_best_C_mean": float(stack_best_c_mean),
            "weight_step": float(args.weight_step),
            "weight_search_mode": str(args.weight_search_mode),
            "n_weight_candidates": int(w_cands.shape[0]),
            "weight_search_info": w_info,
            "val_size": int(y_val.size),
            "test_size": int(y_test.size),
        },
        "inputs": {
            "m4_npz": str(m4_npz_path),
            "m9mid_npz": str(m9mid_npz_path),
            "m9high_npz": str(m9high_npz_path),
            "hlt_npz": str(args.hlt_npz),
            "m4_sources": _parse_csv_list(args.m4_sources),
            "m9mid_sources": _parse_csv_list(args.m9mid_sources),
            "m9high_sources": _parse_csv_list(args.m9high_sources),
            "include_hlt": bool(int(args.include_hlt)),
            "source_meta": source_meta,
        },
        "model_names": model_names,
        "calibration_report": calibration_report,
        "individual_metrics": individual_metrics,
        "method_metrics": method_metrics,
        "weighted_prob_weights": {n: float(w_prob[i]) for i, n in enumerate(model_names)},
        "weighted_logit_weights": {n: float(w_logit[i]) for i, n in enumerate(model_names)},
        "weighted_prob_search_metrics": w_prob_search_metrics,
        "weighted_logit_search_metrics": w_logit_search_metrics,
        "best_method_by_val_objective": best_method,
    }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    npz_payload: Dict[str, np.ndarray] = {
        "y_val": y_val.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }
    for i, n in enumerate(model_names):
        safe = n.replace(" ", "_")
        npz_payload[f"scores_val_raw_{safe}"] = score_mat_val_raw[:, i].astype(np.float32)
        npz_payload[f"scores_test_raw_{safe}"] = score_mat_test_raw[:, i].astype(np.float32)
        npz_payload[f"scores_val_cal_{safe}"] = score_mat_val[:, i].astype(np.float32)
        npz_payload[f"scores_test_cal_{safe}"] = score_mat_test[:, i].astype(np.float32)
    for k in method_scores_val:
        npz_payload[f"scores_val_{k}"] = method_scores_val[k].astype(np.float32)
        npz_payload[f"scores_test_{k}"] = method_scores_test[k].astype(np.float32)
    np.savez_compressed(out_dir / "fusion_scores.npz", **npz_payload)

    print("============================================================")
    print("3-Run Stacked Fusion (Stacked Logistic Regression)")
    print("============================================================")
    print(f"Out dir: {out_dir}")
    print(f"Models ({len(model_names)}): {', '.join(model_names)}")
    print(
        f"Objective={args.optimize_for} | target_tpr={float(args.target_tpr):.3f} | "
        f"base_cal={args.base_calibration} | stack_features={args.stack_features}"
    )
    print(
        f"Weight search={w_info.get('strategy', 'unknown')} | "
        f"candidates={int(w_cands.shape[0])}"
    )
    print("------------------------------------------------------------")
    for name, info in method_metrics.items():
        mv = info["val"]
        mt = info["test"]
        print(
            f"{name:20s} "
            f"val(auc/fpr50)={float(mv.get('auc', float('nan'))):.4f}/"
            f"{float(mv.get('fpr_at_target_tpr', float('nan'))):.6f} | "
            f"test(auc/fpr50)={float(mt.get('auc', float('nan'))):.4f}/"
            f"{float(mt.get('fpr_at_target_tpr', float('nan'))):.6f}"
        )
    print("------------------------------------------------------------")
    print(f"Best method by val objective: {best_method}")
    print(f"Weighted prob weights:  {report['weighted_prob_weights']}")
    print(f"Weighted logit weights: {report['weighted_logit_weights']}")
    print(f"Saved report: {report_path}")
    print(f"Saved scores: {out_dir / 'fusion_scores.npz'}")


if __name__ == "__main__":
    main()

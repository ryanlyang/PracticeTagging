#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze the finished weighted 5m1m1m RecoTeacher score artifacts.

This is a temporary "Analyze8" path for the artifacts that are already done:
  - m4_corrected
  - m9_mid
  - m9_high
  - m12_dual
  - m15_mid_dual
  - m15_high_dual
  - m16_dual
  - m17_dual
plus HLT as an extra score source.

It runs two score-level fusion families:
  1. JetClass-style stacked logistic regression, plus uniform/weighted blends.
  2. Analyze12-style bin-gated fusion by generating a small compatibility
     fusion JSON and calling analyze_hlt_joint31_bin_gated_fusion.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = "checkpoints/reco_teacher_joint_fusion_6model_150k75k150k"


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    arg_name: str
    default_npz: str
    key_pairs: Tuple[Tuple[str, str, str], ...]


SCORE_SPECS: Tuple[ScoreSpec, ...] = (
    ScoreSpec(
        name="m4_corrected",
        arg_name="m4_npz",
        default_npz=(
            f"{BASE_DIR}/model4_recoteacher_s01_corrected_weighted_5m1m1m/"
            "model4_recoteacher_s01_corrected_weighted_5m1m1m_seed0/"
            "stageA_only_scores.npz"
        ),
        key_pairs=(
            ("corrected_only", "preds_corrected_only_val", "preds_corrected_only_test"),
        ),
    ),
    ScoreSpec(
        name="m9_mid",
        arg_name="m9mid_npz",
        default_npz=(
            f"{BASE_DIR}/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m/"
            "model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m_seed0/"
            "stageA_residual_scores.npz"
        ),
        key_pairs=(
            ("residual_frozen", "preds_residual_frozen_val", "preds_residual_frozen_test"),
            ("residual_joint", "preds_residual_joint_val", "preds_residual_joint_test"),
        ),
    ),
    ScoreSpec(
        name="m9_high",
        arg_name="m9high_npz",
        default_npz=(
            f"{BASE_DIR}/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m/"
            "model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m_seed0/"
            "stageA_residual_scores.npz"
        ),
        key_pairs=(
            ("residual_frozen", "preds_residual_frozen_val", "preds_residual_frozen_test"),
            ("residual_joint", "preds_residual_joint_val", "preds_residual_joint_test"),
        ),
    ),
    ScoreSpec(
        name="m12_dual",
        arg_name="m12_npz",
        default_npz=(
            f"{BASE_DIR}/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_from_recoonly/"
            "model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
        key_pairs=(
            ("dual_frozen", "preds_dual_frozen_val", "preds_dual_frozen_test"),
            ("dualview_frozen", "preds_dualview_frozen_val", "preds_dualview_frozen_test"),
            ("dual_joint", "preds_dual_joint_val", "preds_dual_joint_test"),
        ),
    ),
    ScoreSpec(
        name="m15_mid_dual",
        arg_name="m15mid_npz",
        default_npz=(
            f"{BASE_DIR}/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_from_recoonly/"
            "model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
        key_pairs=(
            ("dual_frozen", "preds_dual_frozen_val", "preds_dual_frozen_test"),
            ("dualview_frozen", "preds_dualview_frozen_val", "preds_dualview_frozen_test"),
            ("dual_joint", "preds_dual_joint_val", "preds_dual_joint_test"),
        ),
    ),
    ScoreSpec(
        name="m15_high_dual",
        arg_name="m15high_npz",
        default_npz=(
            f"{BASE_DIR}/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_from_recoonly/"
            "model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
        key_pairs=(
            ("dual_frozen", "preds_dual_frozen_val", "preds_dual_frozen_test"),
            ("dualview_frozen", "preds_dualview_frozen_val", "preds_dualview_frozen_test"),
            ("dual_joint", "preds_dual_joint_val", "preds_dual_joint_test"),
        ),
    ),
    ScoreSpec(
        name="m16_dual",
        arg_name="m16_npz",
        default_npz=(
            f"{BASE_DIR}/model16_dualreco_dualview_topk60_weighted_5m1m1m_from_recoonly/"
            "model16_dualreco_dualview_topk60_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
        key_pairs=(
            ("dual_frozen", "preds_dual_frozen_val", "preds_dual_frozen_test"),
            ("dualview_frozen", "preds_dualview_frozen_val", "preds_dualview_frozen_test"),
            ("dual_joint", "preds_dual_joint_val", "preds_dual_joint_test"),
        ),
    ),
    ScoreSpec(
        name="m17_dual",
        arg_name="m17_npz",
        default_npz=(
            f"{BASE_DIR}/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_from_recoonly/"
            "model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_from_recoonly/"
            "dualreco_dualview_scores.npz"
        ),
        key_pairs=(
            ("dual_frozen", "preds_dual_frozen_val", "preds_dual_frozen_test"),
            ("dualview_frozen", "preds_dualview_frozen_val", "preds_dualview_frozen_test"),
            ("dual_joint", "preds_dual_joint_val", "preds_dual_joint_test"),
        ),
    ),
)


BIN_GATED_SCORE_FILE_KEYS: Dict[str, str] = {
    "corrected_s01": "m4_npz",
    "offdrop_mid": "m9mid_npz",
    "offdrop_high": "m9high_npz",
    "dual_m12_noscale": "m12_npz",
    "dual_m15_offdrop_mid": "m15mid_npz",
    "dual_m15_offdrop_high": "m15high_npz",
    "dual_m16_topk60": "m16_npz",
    "dual_m17_antioverlap": "m17_npz",
}


DEFAULT_BIN_GATED_CANDIDATES = (
    "corrected_s01,"
    "offdrop_mid,"
    "offdrop_high,"
    "dual_m12_noscale,"
    "dual_m15_offdrop_mid,"
    "dual_m15_offdrop_high,"
    "dual_m16_topk60,"
    "dual_m17_antioverlap,"
    "hlt"
)


def _parse_csv_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_float_list(raw: str, default: Sequence[float]) -> List[float]:
    out: List[float] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    return out if out else [float(x) for x in default]


def _safe_logit(p: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=np.float64)
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _sanitize_scores(scores: np.ndarray, label: str) -> Tuple[np.ndarray, Dict[str, float]]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(s)
    n_bad = int((~finite).sum())
    if n_bad == 0:
        return s, {"nonfinite_replaced": 0, "replacement": float("nan")}
    if finite.any():
        repl = float(np.median(s[finite]))
    else:
        repl = 0.5
    fixed = s.copy()
    fixed[~finite] = repl
    print(f"[warn] {label}: replaced {n_bad} non-finite scores with {repl:.6g}", flush=True)
    return fixed, {"nonfinite_replaced": n_bad, "replacement": repl}


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
    return {
        "tp": tp,
        "fp": fp,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tpr": float(tp / max(n_pos, 1)),
        "fpr": float(fp / max(n_neg, 1)),
    }


def _eval_metrics(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float64)
    thr = _threshold_for_target_tpr(y, s, float(target_tpr))
    rr = _rates_from_threshold(y, s, thr)
    auc = float(roc_auc_score(y, s)) if np.unique(y.astype(np.int64)).size > 1 else float("nan")
    return {
        "auc": auc,
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
    return _sigmoid(_safe_logit(score_mat) @ np.asarray(weights, dtype=np.float64))


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

        mv = _eval_metrics(y_val, sv, float(target_tpr))
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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mode = str(mode).strip().lower()
    y = np.asarray(y_val, dtype=np.int64)
    sv = np.asarray(s_val, dtype=np.float64)
    st = np.asarray(s_test, dtype=np.float64)
    if mode == "raw" or np.unique(y).size < 2:
        return sv.copy(), st.copy(), {"mode": "raw", "ok": True}
    if mode == "platt":
        lr = LogisticRegression(solver="lbfgs", max_iter=4000)
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


def _assert_same_labels(
    npz: np.lib.npyio.NpzFile,
    yv_ref: np.ndarray,
    yt_ref: np.ndarray,
    tag: str,
) -> None:
    yv = np.asarray(npz["labels_val"], dtype=np.float32)
    yt = np.asarray(npz["labels_test"], dtype=np.float32)
    if not np.array_equal(yv_ref, yv):
        raise RuntimeError(f"Validation labels mismatch for {tag}")
    if not np.array_equal(yt_ref, yt):
        raise RuntimeError(f"Test labels mismatch for {tag}")


def _select_key_pair(
    npz: np.lib.npyio.NpzFile,
    key_pairs: Sequence[Tuple[str, str, str]],
    y_val: np.ndarray,
    mode: str,
    target_tpr: float,
    source_name: str,
) -> Tuple[str, str, str]:
    present: List[Tuple[str, str, str]] = []
    for label, kv, kt in key_pairs:
        if kv in npz and kt in npz:
            present.append((label, kv, kt))
    if not present:
        raise KeyError(
            f"No supported score keys found for {source_name}. "
            f"Tried: {[(kv, kt) for _, kv, kt in key_pairs]}"
        )
    m = str(mode).strip().lower()
    if m == "first" or len(present) == 1:
        return present[0]

    best = present[0]
    best_primary = None
    best_auc = float("-inf")
    yv = np.asarray(y_val, dtype=np.float32)
    for item in present:
        label, kv, _ = item
        sv = np.asarray(npz[kv], dtype=np.float64)
        finite = np.isfinite(sv)
        if not finite.all():
            repl = float(np.median(sv[finite])) if finite.any() else 0.5
            sv = sv.copy()
            sv[~finite] = repl
        auc = float(roc_auc_score(yv, sv)) if np.unique(yv.astype(np.int64)).size > 1 else float("nan")
        auc_cmp = auc if np.isfinite(auc) else float("-inf")
        if m == "best_val_auc":
            primary = auc_cmp
            better = (best_primary is None) or (primary > float(best_primary))
        elif m == "best_val_fpr":
            thr = _threshold_for_target_tpr(yv, sv, float(target_tpr))
            rr = _rates_from_threshold(yv, sv, thr)
            primary = -float(rr["fpr"])
            better = (best_primary is None) or (primary > float(best_primary)) or (
                np.isclose(primary, float(best_primary)) and auc_cmp > best_auc
            )
        else:
            raise ValueError(f"Unknown head_select_mode={mode}")
        if better:
            best = item
            best_primary = float(primary)
            best_auc = auc_cmp
        print(
            f"[head] {source_name}:{label} val_auc={auc_cmp:.6f}",
            flush=True,
        )
    return best


def _save_csv_dynamic(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        if not keys:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _resolve_path(raw: str) -> Path:
    return Path(str(raw)).expanduser().resolve()


def _load_finished_scores(args: argparse.Namespace) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, object]],
    Dict[str, Dict[str, object]],
]:
    spec_by_arg = {s.arg_name: s for s in SCORE_SPECS}
    ref_spec = spec_by_arg["m9mid_npz"]
    ref_path = _resolve_path(getattr(args, ref_spec.arg_name))
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference score file {ref_spec.name}: {ref_path}")
    z_ref = np.load(ref_path)
    y_val = np.asarray(z_ref["labels_val"], dtype=np.float32)
    y_test = np.asarray(z_ref["labels_test"], dtype=np.float32)

    scores_val: Dict[str, np.ndarray] = {}
    scores_test: Dict[str, np.ndarray] = {}
    source_meta: Dict[str, Dict[str, object]] = {}
    sanitize_report: Dict[str, Dict[str, object]] = {}

    for spec in SCORE_SPECS:
        path = _resolve_path(getattr(args, spec.arg_name))
        if not path.exists():
            if int(args.allow_missing) != 0:
                print(f"[warn] skipping missing score file for {spec.name}: {path}", flush=True)
                continue
            raise FileNotFoundError(f"Missing score file for {spec.name}: {path}")
        z = z_ref if path == ref_path else np.load(path)
        _assert_same_labels(z, y_val, y_test, spec.name)
        head_label, kv, kt = _select_key_pair(
            z,
            spec.key_pairs,
            y_val=y_val,
            mode=str(args.head_select_mode),
            target_tpr=float(args.head_select_tpr),
            source_name=spec.name,
        )
        sv, rep_v = _sanitize_scores(np.asarray(z[kv], dtype=np.float64), f"{spec.name}:{kv}")
        st, rep_t = _sanitize_scores(np.asarray(z[kt], dtype=np.float64), f"{spec.name}:{kt}")
        scores_val[spec.name] = sv
        scores_test[spec.name] = st
        source_meta[spec.name] = {
            "path": str(path),
            "selected_head": head_label,
            "val_key": kv,
            "test_key": kt,
        }
        sanitize_report[spec.name] = {"val": rep_v, "test": rep_t}

    if len(scores_val) != len(SCORE_SPECS) and int(args.allow_missing) == 0:
        raise RuntimeError(f"Loaded {len(scores_val)} model score sets, expected {len(SCORE_SPECS)}")

    hlt_sv, hlt_st, hlt_meta = _load_hlt_scores(args, z_ref, y_val, y_test, ref_path)
    hlt_sv, rep_v = _sanitize_scores(hlt_sv, "hlt:val")
    hlt_st, rep_t = _sanitize_scores(hlt_st, "hlt:test")
    scores_val["hlt"] = hlt_sv
    scores_test["hlt"] = hlt_st
    source_meta["hlt"] = hlt_meta
    sanitize_report["hlt"] = {"val": rep_v, "test": rep_t}

    return y_val, y_test, scores_val, scores_test, source_meta, sanitize_report


def _load_hlt_scores(
    args: argparse.Namespace,
    z_ref: np.lib.npyio.NpzFile,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ref_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    candidates: List[Tuple[str, Path]] = []
    if str(args.hlt_npz).strip():
        candidates.append(("hlt_npz", _resolve_path(str(args.hlt_npz))))
    if str(args.step1_ref_npz).strip():
        p = _resolve_path(str(args.step1_ref_npz))
        if p.exists():
            candidates.append(("step1_ref_npz", p))

    for label, path in candidates:
        if not path.exists():
            if label == "hlt_npz":
                raise FileNotFoundError(f"Missing --hlt_npz: {path}")
            continue
        z = np.load(path)
        if "preds_hlt_val" not in z or "preds_hlt_test" not in z:
            print(f"[warn] {label} lacks preds_hlt_val/test, ignoring: {path}", flush=True)
            continue
        if "labels_val" in z and "labels_test" in z:
            yv = np.asarray(z["labels_val"], dtype=np.float32)
            yt = np.asarray(z["labels_test"], dtype=np.float32)
            if not np.array_equal(y_val, yv) or not np.array_equal(y_test, yt):
                print(f"[warn] {label} labels mismatch, ignoring: {path}", flush=True)
                continue
        else:
            print(f"[warn] {label} has no labels for compatibility check, ignoring: {path}", flush=True)
            continue
        return (
            np.asarray(z["preds_hlt_val"], dtype=np.float64),
            np.asarray(z["preds_hlt_test"], dtype=np.float64),
            {
                "path": str(path),
                "source": label,
                "val_key": "preds_hlt_val",
                "test_key": "preds_hlt_test",
            },
        )

    if "preds_hlt_val" not in z_ref or "preds_hlt_test" not in z_ref:
        raise KeyError(f"Reference npz missing HLT keys: {ref_path}")
    return (
        np.asarray(z_ref["preds_hlt_val"], dtype=np.float64),
        np.asarray(z_ref["preds_hlt_test"], dtype=np.float64),
        {
            "path": str(ref_path),
            "source": "reference_npz",
            "val_key": "preds_hlt_val",
            "test_key": "preds_hlt_test",
        },
    )


def _build_stack_features(score_mat_val: np.ndarray, score_mat_test: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "logits":
        return _safe_logit(score_mat_val).astype(np.float32), _safe_logit(score_mat_test).astype(np.float32)
    if mode == "probs":
        return score_mat_val.astype(np.float32), score_mat_test.astype(np.float32)
    if mode == "logits_probs":
        return (
            np.concatenate([_safe_logit(score_mat_val), score_mat_val], axis=1).astype(np.float32),
            np.concatenate([_safe_logit(score_mat_test), score_mat_test], axis=1).astype(np.float32),
        )
    raise ValueError(f"Unknown stack_features={mode}")


def _run_stacked_fusion(
    args: argparse.Namespace,
    target_tprs: Sequence[float],
    y_val: np.ndarray,
    y_test: np.ndarray,
    scores_val_raw: Dict[str, np.ndarray],
    scores_test_raw: Dict[str, np.ndarray],
    source_meta: Dict[str, Dict[str, object]],
    sanitize_report: Dict[str, Dict[str, object]],
    out_dir: Path,
) -> Dict[str, object]:
    model_names = ["hlt"] + [s.name for s in SCORE_SPECS if s.name in scores_val_raw]
    if int(args.allow_missing) != 0:
        model_names = [n for n in model_names if n in scores_val_raw]
    if len(model_names) < 2:
        raise RuntimeError(f"Need at least two score sources; got {model_names}")

    score_mat_val_raw = np.column_stack([scores_val_raw[n] for n in model_names]).astype(np.float64)
    score_mat_test_raw = np.column_stack([scores_test_raw[n] for n in model_names]).astype(np.float64)

    score_mat_val = np.zeros_like(score_mat_val_raw, dtype=np.float64)
    score_mat_test = np.zeros_like(score_mat_test_raw, dtype=np.float64)
    calibration_report: Dict[str, Dict[str, object]] = {}
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

    x_val, x_test = _build_stack_features(score_mat_val, score_mat_test, str(args.stack_features))
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

    method_scores_val_base = {
        "uniform_prob_avg": s_uni_val,
        "stacked_logreg": s_stack_val,
    }
    method_scores_test_base = {
        "uniform_prob_avg": s_uni_test,
        "stacked_logreg": s_stack_test,
    }
    target_reports: Dict[str, object] = {}
    summary_rows: List[Dict[str, object]] = []
    individual_rows: List[Dict[str, object]] = []
    weights_by_tpr: Dict[str, object] = {}
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

    for name, sv in method_scores_val_base.items():
        npz_payload[f"scores_val_{name}"] = sv.astype(np.float32)
        npz_payload[f"scores_test_{name}"] = method_scores_test_base[name].astype(np.float32)

    for tpr in target_tprs:
        tpr_key = f"tpr{float(tpr):.3f}".replace(".", "p")
        w_prob, w_prob_search_metrics = _search_best_weights(
            weight_candidates=w_cands,
            y_val=y_val,
            score_mat_val=score_mat_val,
            target_tpr=float(tpr),
            optimize_for=str(args.optimize_for),
            mode="prob",
        )
        s_wprob_val = _fuse_prob(w_prob, score_mat_val)
        s_wprob_test = _fuse_prob(w_prob, score_mat_test)

        w_logit, w_logit_search_metrics = _search_best_weights(
            weight_candidates=w_cands,
            y_val=y_val,
            score_mat_val=score_mat_val,
            target_tpr=float(tpr),
            optimize_for=str(args.optimize_for),
            mode="logit",
        )
        s_wlog_val = _fuse_logit(w_logit, score_mat_val)
        s_wlog_test = _fuse_logit(w_logit, score_mat_test)

        method_scores_val = dict(method_scores_val_base)
        method_scores_test = dict(method_scores_test_base)
        method_scores_val[f"weighted_prob_avg_{tpr_key}"] = s_wprob_val
        method_scores_test[f"weighted_prob_avg_{tpr_key}"] = s_wprob_test
        method_scores_val[f"weighted_logit_avg_{tpr_key}"] = s_wlog_val
        method_scores_test[f"weighted_logit_avg_{tpr_key}"] = s_wlog_test

        npz_payload[f"scores_val_weighted_prob_avg_{tpr_key}"] = s_wprob_val.astype(np.float32)
        npz_payload[f"scores_test_weighted_prob_avg_{tpr_key}"] = s_wprob_test.astype(np.float32)
        npz_payload[f"scores_val_weighted_logit_avg_{tpr_key}"] = s_wlog_val.astype(np.float32)
        npz_payload[f"scores_test_weighted_logit_avg_{tpr_key}"] = s_wlog_test.astype(np.float32)

        method_metrics: Dict[str, Dict[str, object]] = {}
        best_method = ""
        best_obj = float("-inf")
        best_auc_tie = float("-inf")
        for name in method_scores_val:
            mv = _eval_metrics(y_val, method_scores_val[name], float(tpr))
            mt = _eval_metrics(y_test, method_scores_test[name], float(tpr))
            obj = _objective(mv, str(args.optimize_for))
            auc_tie = float(mv.get("auc", float("nan")))
            auc_tie = auc_tie if np.isfinite(auc_tie) else float("-inf")
            method_metrics[name] = {"val": mv, "test": mt, "val_objective": float(obj)}
            if (obj > best_obj) or (np.isclose(obj, best_obj) and auc_tie > best_auc_tie):
                best_obj = float(obj)
                best_auc_tie = float(auc_tie)
                best_method = str(name)
            summary_rows.append(
                {
                    "target_tpr": float(tpr),
                    "method": name,
                    "auc_val": float(mv["auc"]),
                    "fpr_val": float(mv["fpr_at_target_tpr"]),
                    "tpr_val": float(mv["tpr_at_target_tpr"]),
                    "auc_test": float(mt["auc"]),
                    "fpr_test": float(mt["fpr_at_target_tpr"]),
                    "tpr_test": float(mt["tpr_at_target_tpr"]),
                    "val_objective": float(obj),
                }
            )

        individual_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for i, n in enumerate(model_names):
            m_raw_v = _eval_metrics(y_val, score_mat_val_raw[:, i], float(tpr))
            m_raw_t = _eval_metrics(y_test, score_mat_test_raw[:, i], float(tpr))
            m_cal_v = _eval_metrics(y_val, score_mat_val[:, i], float(tpr))
            m_cal_t = _eval_metrics(y_test, score_mat_test[:, i], float(tpr))
            individual_metrics[n] = {
                "val_raw": m_raw_v,
                "test_raw": m_raw_t,
                "val_calibrated": m_cal_v,
                "test_calibrated": m_cal_t,
            }
            individual_rows.append(
                {
                    "target_tpr": float(tpr),
                    "source": n,
                    "selected_head": source_meta.get(n, {}).get("selected_head", ""),
                    "auc_val_raw": float(m_raw_v["auc"]),
                    "fpr_val_raw": float(m_raw_v["fpr_at_target_tpr"]),
                    "auc_test_raw": float(m_raw_t["auc"]),
                    "fpr_test_raw": float(m_raw_t["fpr_at_target_tpr"]),
                    "auc_val_calibrated": float(m_cal_v["auc"]),
                    "fpr_val_calibrated": float(m_cal_v["fpr_at_target_tpr"]),
                    "auc_test_calibrated": float(m_cal_t["auc"]),
                    "fpr_test_calibrated": float(m_cal_t["fpr_at_target_tpr"]),
                }
            )

        weights_by_tpr[f"{float(tpr):.4f}"] = {
            "weighted_prob_weights": {n: float(w_prob[i]) for i, n in enumerate(model_names)},
            "weighted_logit_weights": {n: float(w_logit[i]) for i, n in enumerate(model_names)},
            "weighted_prob_search_metrics": w_prob_search_metrics,
            "weighted_logit_search_metrics": w_logit_search_metrics,
        }
        target_reports[f"{float(tpr):.4f}"] = {
            "target_tpr": float(tpr),
            "method_metrics": method_metrics,
            "individual_metrics": individual_metrics,
            "best_method_by_val_objective": best_method,
        }

    report = {
        "setup": {
            "target_tprs": [float(x) for x in target_tprs],
            "optimize_for": str(args.optimize_for),
            "base_calibration": str(args.base_calibration),
            "head_select_mode": str(args.head_select_mode),
            "head_select_tpr": float(args.head_select_tpr),
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
        "model_names": model_names,
        "source_meta": source_meta,
        "sanitize_report": sanitize_report,
        "calibration_report": calibration_report,
        "weights_by_tpr": weights_by_tpr,
        "by_tpr": target_reports,
        "files": {
            "summary_csv": str((out_dir / "stacked_summary.csv").resolve()),
            "individual_csv": str((out_dir / "stacked_individual_summary.csv").resolve()),
            "scores_npz": str((out_dir / "stacked_scores.npz").resolve()),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stacked_report.json").write_text(json.dumps(report, indent=2))
    _save_csv_dynamic(out_dir / "stacked_summary.csv", summary_rows)
    _save_csv_dynamic(out_dir / "stacked_individual_summary.csv", individual_rows)
    np.savez_compressed(out_dir / "stacked_scores.npz", **npz_payload)

    print("=" * 72)
    print("Analyze8 Stacked Logistic Fusion")
    print("=" * 72)
    print(f"Out dir: {out_dir}")
    print(f"Sources ({len(model_names)}): {', '.join(model_names)}")
    print(
        f"Objective={args.optimize_for} | TPRs={','.join(f'{x:.3f}' for x in target_tprs)} | "
        f"calibration={args.base_calibration} | stack_features={args.stack_features}"
    )
    print(
        f"Weight search={w_info.get('strategy', 'unknown')} | "
        f"candidates={int(w_cands.shape[0])} | stack_C={stack_best_c_mean:.6g}"
    )
    print("-" * 72)
    for tpr in target_tprs:
        key = f"{float(tpr):.4f}"
        methods = target_reports[key]["method_metrics"]
        ranked = sorted(
            methods.items(),
            key=lambda kv: (
                float(kv[1]["test"]["fpr_at_target_tpr"]),
                -float(kv[1]["test"]["auc"]),
            ),
        )
        print(f"TPR={float(tpr):.3f} best methods by test FPR then AUC:")
        for name, info in ranked:
            mt = info["test"]
            mv = info["val"]
            print(
                f"  {name:28s} "
                f"val_auc/fpr={float(mv['auc']):.6f}/{float(mv['fpr_at_target_tpr']):.6f} | "
                f"test_auc/fpr={float(mt['auc']):.6f}/{float(mt['fpr_at_target_tpr']):.6f}"
            )
        print()
    print(f"Saved report: {out_dir / 'stacked_report.json'}")
    print(f"Saved summary: {out_dir / 'stacked_summary.csv'}")
    print(f"Saved scores: {out_dir / 'stacked_scores.npz'}")
    return report


def _write_bin_gated_inputs(
    args: argparse.Namespace,
    y_val: np.ndarray,
    y_test: np.ndarray,
    scores_val: Dict[str, np.ndarray],
    scores_test: Dict[str, np.ndarray],
    source_meta: Dict[str, Dict[str, object]],
    out_dir: Path,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = str(args.bin_anchor_model)
    anchor_score_name = {
        "offdrop_mid": "m9_mid",
        "offdrop_high": "m9_high",
        "corrected_s01": "m4_corrected",
        "dual_m12_noscale": "m12_dual",
        "dual_m15_offdrop_mid": "m15_mid_dual",
        "dual_m15_offdrop_high": "m15_high_dual",
        "dual_m16_topk60": "m16_dual",
        "dual_m17_antioverlap": "m17_dual",
        "hlt": "hlt",
    }.get(anchor, "m9_mid")
    if anchor_score_name not in scores_val:
        raise KeyError(f"Cannot build bin-gated compat anchor from {anchor_score_name}; missing score")

    compat_npz = out_dir / "joint_delta_compat_from_analyze8_anchor.npz"
    np.savez_compressed(
        compat_npz,
        labels_val=y_val.astype(np.float32),
        labels_test=y_test.astype(np.float32),
        preds_hlt_val=np.asarray(scores_val["hlt"], dtype=np.float64),
        preds_hlt_test=np.asarray(scores_test["hlt"], dtype=np.float64),
        preds_joint_val=np.asarray(scores_val[anchor_score_name], dtype=np.float64),
        preds_joint_test=np.asarray(scores_test[anchor_score_name], dtype=np.float64),
    )

    score_files = {
        "joint_delta": str(compat_npz.resolve()),
        "hlt": str(compat_npz.resolve()),
    }
    for model_name, arg_name in BIN_GATED_SCORE_FILE_KEYS.items():
        score_files[model_name] = str(_resolve_path(getattr(args, arg_name)))

    fusion_json = out_dir / "fusion_hlt_joint8_finished_weighted_5m1m1m.json"
    fusion_payload = {
        "run_dirs": {"score_files": score_files},
        "analyze8_meta": {
            "compat_anchor_model": anchor,
            "compat_anchor_score_name": anchor_score_name,
            "source_meta": source_meta,
        },
    }
    fusion_json.write_text(json.dumps(fusion_payload, indent=2))
    return compat_npz, fusion_json


def _run_bin_gated_fusion(
    args: argparse.Namespace,
    target_tprs: Sequence[float],
    y_val: np.ndarray,
    y_test: np.ndarray,
    scores_val: Dict[str, np.ndarray],
    scores_test: Dict[str, np.ndarray],
    source_meta: Dict[str, Dict[str, object]],
    out_dir: Path,
) -> None:
    compat_npz, fusion_json = _write_bin_gated_inputs(
        args=args,
        y_val=y_val,
        y_test=y_test,
        scores_val=scores_val,
        scores_test=scores_test,
        source_meta=source_meta,
        out_dir=out_dir,
    )

    bin_script = Path(args.bin_gated_script).expanduser()
    if not bin_script.is_absolute():
        bin_script = Path(__file__).resolve().parent / bin_script
    bin_script = bin_script.resolve()
    if not bin_script.exists():
        raise FileNotFoundError(f"Missing bin-gated script: {bin_script}")

    cmd = [
        sys.executable,
        str(bin_script),
        "--fusion_json",
        str(fusion_json),
        "--target_tprs",
        ",".join(f"{float(x):.2f}" for x in target_tprs),
        "--anchor_model",
        str(args.bin_anchor_model),
        "--selection_mode",
        str(args.bin_selection_mode),
        "--candidate_models_all",
        str(args.bin_candidate_models),
        "--expand_prepost_variants",
        str(int(args.bin_expand_prepost_variants)),
        "--router_cal_frac",
        str(float(args.bin_router_cal_frac)),
        "--seed",
        str(int(args.seed)),
        "--calibration",
        str(args.bin_calibration),
        "--head_select_mode",
        str(args.bin_head_select_mode),
        "--head_select_tpr",
        str(float(args.bin_head_select_tpr)),
        "--score_band_edges",
        str(args.bin_score_band_edges),
        "--dist_near_cut",
        str(float(args.bin_dist_near_cut)),
        "--dist_mid_low",
        str(float(args.bin_dist_mid_low)),
        "--dist_mid_high",
        str(float(args.bin_dist_mid_high)),
        "--global_max_add",
        str(int(args.bin_global_max_add)),
        "--bin_max_add",
        str(int(args.bin_bin_max_add)),
        "--w_step",
        str(float(args.bin_w_step)),
        "--min_bin_fit",
        str(int(args.bin_min_bin_fit)),
        "--min_global_improve",
        str(float(args.bin_min_global_improve)),
        "--min_bin_improve",
        str(float(args.bin_min_bin_improve)),
        "--out_dir",
        str(out_dir),
        "--report_json",
        str(out_dir / "bin_gated_report.json"),
    ]

    print("=" * 72)
    print("Analyze8 Bin-Gated Fusion")
    print("=" * 72)
    print(f"Compat NPZ:  {compat_npz}")
    print(f"Fusion JSON: {fusion_json}")
    print(f"Out dir:     {out_dir}")
    print(f"Candidates:  {args.bin_candidate_models}")
    print(f"Command:     {' '.join(cmd)}")
    print("=" * 72, flush=True)
    subprocess.run(cmd, check=True)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Analyze8 stacked-logreg plus Analyze12-style bin-gated fusion for finished 5m1m1m runs."
    )
    for spec in SCORE_SPECS:
        ap.add_argument(f"--{spec.arg_name}", type=str, default=spec.default_npz)
    ap.add_argument(
        "--hlt_npz",
        type=str,
        default="",
        help="Optional HLT override NPZ with matching labels and preds_hlt_val/test.",
    )
    ap.add_argument(
        "--step1_ref_npz",
        type=str,
        default=(
            f"{BASE_DIR}/teacher_hlt_only_weighted_5m1m1m/"
            "teacher_hlt_only_weighted_5m1m1m_seed0/"
            "results_step1_teacher_baseline.npz"
        ),
        help="Optional Step-1 HLT reference; used only if present and labels match.",
    )
    ap.add_argument("--allow_missing", type=int, default=0)

    ap.add_argument("--target_tprs", type=str, default="0.50,0.30")
    ap.add_argument("--optimize_for", type=str, default="fpr_at_tpr", choices=["fpr_at_tpr", "auc"])
    ap.add_argument("--base_calibration", type=str, default="iso", choices=["raw", "platt", "iso"])
    ap.add_argument(
        "--head_select_mode",
        type=str,
        default="best_val_fpr",
        choices=["first", "best_val_auc", "best_val_fpr"],
        help="How stacked-logreg picks one head per finished model family.",
    )
    ap.add_argument("--head_select_tpr", type=float, default=0.50)

    ap.add_argument("--weight_step", type=float, default=0.05)
    ap.add_argument("--weight_search_mode", type=str, default="auto", choices=["auto", "grid", "dirichlet"])
    ap.add_argument("--max_weight_candidates", type=int, default=250000)
    ap.add_argument("--weight_random_samples", type=int, default=20000)
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
        default=f"{BASE_DIR}/analyze8_finished_weighted_5m1m1m",
    )
    ap.add_argument("--skip_stacked", type=int, default=0)
    ap.add_argument("--skip_bin_gated", type=int, default=0)

    ap.add_argument("--bin_gated_script", type=str, default="analyze_hlt_joint31_bin_gated_fusion.py")
    ap.add_argument("--bin_anchor_model", type=str, default="offdrop_mid")
    ap.add_argument("--bin_candidate_models", type=str, default=DEFAULT_BIN_GATED_CANDIDATES)
    ap.add_argument("--bin_expand_prepost_variants", type=int, default=1)
    ap.add_argument("--bin_selection_mode", type=str, default="valsel", choices=["split", "valsel"])
    ap.add_argument("--bin_calibration", type=str, default="iso", choices=["raw", "iso", "platt"])
    ap.add_argument(
        "--bin_head_select_mode",
        type=str,
        default="best_val_fpr",
        choices=["first", "best_val_auc", "best_val_fpr"],
    )
    ap.add_argument("--bin_head_select_tpr", type=float, default=0.50)
    ap.add_argument("--bin_router_cal_frac", type=float, default=0.40)
    ap.add_argument("--bin_score_band_edges", type=str, default="0.0,0.8,0.9,1.0")
    ap.add_argument("--bin_dist_near_cut", type=float, default=0.0384)
    ap.add_argument("--bin_dist_mid_low", type=float, default=0.06285)
    ap.add_argument("--bin_dist_mid_high", type=float, default=0.07386)
    ap.add_argument("--bin_global_max_add", type=int, default=8)
    ap.add_argument("--bin_bin_max_add", type=int, default=6)
    ap.add_argument("--bin_w_step", type=float, default=0.0025)
    ap.add_argument("--bin_min_bin_fit", type=int, default=2000)
    ap.add_argument("--bin_min_global_improve", type=float, default=2e-7)
    ap.add_argument("--bin_min_bin_improve", type=float, default=1e-6)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    target_tprs = _parse_float_list(args.target_tprs, [0.50, 0.30])
    out_root = _resolve_path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    y_val, y_test, scores_val, scores_test, source_meta, sanitize_report = _load_finished_scores(args)

    if int(args.skip_stacked) == 0:
        _run_stacked_fusion(
            args=args,
            target_tprs=target_tprs,
            y_val=y_val,
            y_test=y_test,
            scores_val_raw=scores_val,
            scores_test_raw=scores_test,
            source_meta=source_meta,
            sanitize_report=sanitize_report,
            out_dir=out_root / "stacked_logreg",
        )

    if int(args.skip_bin_gated) == 0:
        _run_bin_gated_fusion(
            args=args,
            target_tprs=target_tprs,
            y_val=y_val,
            y_test=y_test,
            scores_val=scores_val,
            scores_test=scores_test,
            source_meta=source_meta,
            out_dir=out_root / "bin_gated",
        )

    print("=" * 72)
    print("Analyze8 complete")
    print("=" * 72)
    print(f"Out root: {out_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rolling-chunk greedy fusion over 31-model score artifacts.

Goal:
- Reduce selection overfitting by rotating fit/cal chunks across a large val pool.
- At each step:
  1) pick fit chunk i, cal chunk i+1 (cyclic),
  2) choose best candidate merge on fit chunk,
  3) accept only if it improves current cal chunk and cumulative cal union.

No model retraining; score-level fusion only.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import analyze_hlt_joint18_recoteacher_fusion as base
import analyze_hlt_joint31_specialization_atlas as atlas


def _parse_str_list(spec: str) -> List[str]:
    out: List[str] = []
    for tok in str(spec).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _save_csv_dynamic(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _eval_on_indices(
    y: np.ndarray,
    s: np.ndarray,
    idx_rel: np.ndarray,
    target_tpr: float,
) -> Dict[str, float]:
    y_i = np.asarray(y[idx_rel], dtype=np.float32)
    s_i = np.asarray(s[idx_rel], dtype=np.float64)
    thr = float(base.threshold_for_target_tpr(y_i, s_i, float(target_tpr)))
    rr = base.rates_from_threshold(y_i, s_i, thr)
    auc = float(roc_auc_score(y_i, s_i)) if np.unique(y_i).size > 1 else float("nan")
    return {
        "threshold": thr,
        "fpr": float(rr["fpr"]),
        "tpr": float(rr["tpr"]),
        "auc": auc,
        "tp": float(rr["tp"]),
        "fp": float(rr["fp"]),
        "n_pos": float(rr["n_pos"]),
        "n_neg": float(rr["n_neg"]),
    }


def _eval_test_from_ref(
    y_ref: np.ndarray,
    s_ref: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    target_tpr: float,
) -> Dict[str, float]:
    thr = float(base.threshold_for_target_tpr(y_ref, s_ref, float(target_tpr)))
    rr_ref = base.rates_from_threshold(y_ref, s_ref, thr)
    rr_test = base.rates_from_threshold(y_test, s_test, thr)
    auc_ref = float(roc_auc_score(y_ref, s_ref)) if np.unique(y_ref).size > 1 else float("nan")
    auc_test = float(roc_auc_score(y_test, s_test)) if np.unique(y_test).size > 1 else float("nan")
    return {
        "threshold_from_ref": thr,
        "fpr_ref": float(rr_ref["fpr"]),
        "tpr_ref": float(rr_ref["tpr"]),
        "auc_ref": auc_ref,
        "fpr_test": float(rr_test["fpr"]),
        "tpr_test": float(rr_test["tpr"]),
        "auc_test": auc_test,
        "tp_test": float(rr_test["tp"]),
        "fp_test": float(rr_test["fp"]),
    }


def _make_stratified_chunks(y_pool: np.ndarray, n_chunks: int, seed: int) -> List[np.ndarray]:
    yb = (np.asarray(y_pool, dtype=np.float32) > 0.5)
    pos = np.where(yb)[0]
    neg = np.where(~yb)[0]
    if pos.size == 0 or neg.size == 0:
        raise RuntimeError("Val pool has only one class; cannot build stratified chunks.")
    rng = np.random.default_rng(int(seed))
    pos = np.asarray(pos[rng.permutation(pos.size)], dtype=np.int64)
    neg = np.asarray(neg[rng.permutation(neg.size)], dtype=np.int64)
    pos_parts = np.array_split(pos, int(n_chunks))
    neg_parts = np.array_split(neg, int(n_chunks))
    out: List[np.ndarray] = []
    for i in range(int(n_chunks)):
        c = np.concatenate([pos_parts[i], neg_parts[i]], axis=0)
        c = np.asarray(c[rng.permutation(c.size)], dtype=np.int64)
        out.append(c)
    return out


def _calibrate_from_fit(
    y_fit: np.ndarray,
    s_fit: np.ndarray,
    s_pool: np.ndarray,
    s_test: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mode = str(mode).lower()
    if mode == "raw":
        return s_pool.astype(np.float64), s_test.astype(np.float64), {"ok": True, "mode": "raw"}

    y_fit_i = np.asarray(y_fit, dtype=np.int64)
    sf = np.asarray(s_fit, dtype=np.float64)
    sp = np.asarray(s_pool, dtype=np.float64)
    st = np.asarray(s_test, dtype=np.float64)
    if np.unique(y_fit_i).size < 2:
        return sp, st, {"ok": False, "mode": mode, "reason": "single_class_fit"}

    if mode == "iso":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(sf, y_fit_i.astype(np.float64))
        return (
            np.asarray(iso.transform(sp), dtype=np.float64),
            np.asarray(iso.transform(st), dtype=np.float64),
            {"ok": True, "mode": "iso"},
        )
    if mode == "platt":
        lr = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced")
        lr.fit(sf.reshape(-1, 1), y_fit_i)
        return (
            lr.predict_proba(sp.reshape(-1, 1))[:, 1].astype(np.float64),
            lr.predict_proba(st.reshape(-1, 1))[:, 1].astype(np.float64),
            {
                "ok": True,
                "mode": "platt",
                "coef": float(lr.coef_.ravel()[0]),
                "intercept": float(lr.intercept_.ravel()[0]),
            },
        )
    raise ValueError(f"Unknown calibration mode: {mode}")


def _fpr_only_fit(y_fit: np.ndarray, s_fit: np.ndarray, target_tpr: float) -> float:
    thr = float(base.threshold_for_target_tpr(y_fit, s_fit, float(target_tpr)))
    rr = base.rates_from_threshold(y_fit, s_fit, thr)
    return float(rr["fpr"])


def _weights_to_str(w: np.ndarray, names: List[str], eps: float = 1e-12) -> str:
    parts = []
    for i, v in enumerate(w):
        vv = float(v)
        if vv > float(eps):
            parts.append(f"{names[i]}:{vv:.6f}")
    return ",".join(parts)


def _load_precomputed_scores(
    npz_path: Path,
    manifest_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str], List[str]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"precomputed_scores_npz not found: {npz_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"precomputed_manifest_json not found: {manifest_path}")

    z = np.load(npz_path, allow_pickle=False)
    need = ["labels_dev", "labels_test", "scores_dev", "scores_test"]
    missing = [k for k in need if k not in z]
    if missing:
        raise KeyError(f"precomputed npz missing keys: {missing}")

    labels_dev = np.asarray(z["labels_dev"], dtype=np.float32)
    labels_test = np.asarray(z["labels_test"], dtype=np.float32)
    scores_dev_mat = np.asarray(z["scores_dev"], dtype=np.float64)
    scores_test_mat = np.asarray(z["scores_test"], dtype=np.float64)

    if scores_dev_mat.ndim != 2 or scores_test_mat.ndim != 2:
        raise RuntimeError(
            f"precomputed score matrices must be 2D, got dev={scores_dev_mat.shape}, test={scores_test_mat.shape}"
        )
    if scores_dev_mat.shape[1] != labels_dev.shape[0]:
        raise RuntimeError(
            f"scores_dev shape mismatch: {scores_dev_mat.shape} vs labels_dev={labels_dev.shape}"
        )
    if scores_test_mat.shape[1] != labels_test.shape[0]:
        raise RuntimeError(
            f"scores_test shape mismatch: {scores_test_mat.shape} vs labels_test={labels_test.shape}"
        )
    if scores_dev_mat.shape[0] != scores_test_mat.shape[0]:
        raise RuntimeError(
            f"precomputed model-axis mismatch: dev={scores_dev_mat.shape[0]} vs test={scores_test_mat.shape[0]}"
        )

    man = json.loads(manifest_path.read_text())
    model_order = [str(x) for x in man.get("model_order", [])]
    if not model_order:
        raise KeyError(f"manifest missing non-empty model_order: {manifest_path}")
    if len(model_order) != int(scores_dev_mat.shape[0]):
        raise RuntimeError(
            f"manifest model_order length={len(model_order)} != scores rows={scores_dev_mat.shape[0]}"
        )

    score_files_raw = man.get("score_files_used", {})
    score_files: Dict[str, str] = {}
    if isinstance(score_files_raw, dict):
        for k, v in score_files_raw.items():
            if isinstance(v, str):
                score_files[str(k)] = str(v)

    scores_dev: Dict[str, np.ndarray] = {}
    scores_test: Dict[str, np.ndarray] = {}
    for i, m in enumerate(model_order):
        scores_dev[m] = np.asarray(scores_dev_mat[i], dtype=np.float64)
        scores_test[m] = np.asarray(scores_test_mat[i], dtype=np.float64)

    return labels_dev, labels_test, scores_dev, scores_test, score_files, model_order


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling-chunk greedy fusion on 31-model artifacts")
    ap.add_argument("--fusion_json", type=str, required=True)
    ap.add_argument("--precomputed_scores_npz", type=str, default="")
    ap.add_argument("--precomputed_manifest_json", type=str, default="")
    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--anchor_model", type=str, default="joint_delta")
    ap.add_argument("--candidate_models", type=str, default="")
    ap.add_argument("--candidate_topk_fit", type=int, default=16)
    ap.add_argument("--val_pool_size", type=int, default=1000000)
    ap.add_argument("--val_pool_offset", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--w_step", type=float, default=0.02)
    ap.add_argument("--calibration", type=str, default="raw", choices=["raw", "iso", "platt"])
    ap.add_argument("--min_fit_gain", type=float, default=1e-6)
    ap.add_argument("--min_cal_gain", type=float, default=1e-6)
    ap.add_argument("--min_cum_cal_gain", type=float, default=1e-6)
    ap.add_argument("--topk_diagnostic", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    fusion_json = Path(args.fusion_json).expanduser().resolve()
    if not fusion_json.exists():
        raise FileNotFoundError(f"fusion_json not found: {fusion_json}")

    fusion = json.loads(fusion_json.read_text())
    pre_npz = str(args.precomputed_scores_npz).strip()
    pre_manifest = str(args.precomputed_manifest_json).strip()
    using_precomputed = len(pre_npz) > 0

    if using_precomputed:
        if not pre_manifest:
            raise ValueError("--precomputed_manifest_json is required when --precomputed_scores_npz is provided")
        y_val_full, y_test, scores_val_full, scores_test, score_files, pre_model_order = _load_precomputed_scores(
            npz_path=Path(pre_npz).expanduser().resolve(),
            manifest_path=Path(pre_manifest).expanduser().resolve(),
        )
        model_order = list(fusion.get("models_order", []))
        if not model_order:
            model_order = list(pre_model_order)
        missing_models = [m for m in model_order if m not in scores_val_full]
        if missing_models:
            raise KeyError(
                "Requested models from fusion_json are missing in precomputed artifact: "
                + ",".join(missing_models)
            )
    else:
        y_val_full, y_test, scores_val_full, scores_test, score_files, _run_dirs, model_order = atlas._load_scores_from_fusion_json(
            fusion_obj=fusion,
            fusion_json_path=fusion_json,
        )
    if str(args.anchor_model) not in model_order:
        raise KeyError(f"anchor_model={args.anchor_model} is not in model_order")

    n_val_avail = int(y_val_full.shape[0])
    val_pool_size = int(args.val_pool_size)
    val_pool_offset = int(args.val_pool_offset)
    if val_pool_size <= 0:
        raise ValueError("--val_pool_size must be > 0")
    if val_pool_offset < 0:
        raise ValueError("--val_pool_offset must be >= 0")
    if (val_pool_offset + val_pool_size) > n_val_avail:
        raise RuntimeError(
            f"Requested val pool offset+size = {val_pool_offset}+{val_pool_size} exceeds "
            f"available validation jets = {n_val_avail}. "
            "Use a larger score artifact or reduce --val_pool_size."
        )

    pool_abs = np.arange(val_pool_offset, val_pool_offset + val_pool_size, dtype=np.int64)
    y_pool = np.asarray(y_val_full[pool_abs], dtype=np.float32)
    scores_pool_raw: Dict[str, np.ndarray] = {m: np.asarray(scores_val_full[m][pool_abs], dtype=np.float64) for m in model_order}
    scores_test_raw: Dict[str, np.ndarray] = {m: np.asarray(scores_test[m], dtype=np.float64) for m in model_order}

    # Candidate set.
    cand_user = _parse_str_list(args.candidate_models)
    if cand_user:
        candidates = [m for m in cand_user if m in model_order and m != str(args.anchor_model)]
    else:
        candidates = [m for m in model_order if m != str(args.anchor_model)]
    if not candidates:
        raise RuntimeError("No candidate models available after filtering.")

    # Optional top-k prefilter by single-model fit score on pool.
    prefilter_rows: List[Dict[str, object]] = []
    for m in candidates:
        em = _eval_on_indices(y_pool, scores_pool_raw[m], np.arange(val_pool_size, dtype=np.int64), float(args.target_tpr))
        prefilter_rows.append(
            {
                "model": m,
                "fpr_pool": float(em["fpr"]),
                "auc_pool": float(em["auc"]),
                "threshold_pool": float(em["threshold"]),
            }
        )
    prefilter_rows = sorted(prefilter_rows, key=lambda r: (float(r["fpr_pool"]), -float(r["auc_pool"])))
    topk = int(args.candidate_topk_fit)
    if topk > 0:
        candidates = [r["model"] for r in prefilter_rows[:topk]]

    # Stratified chunks over pool (relative indices).
    n_chunks = int(max(2, args.n_chunks))
    chunks = _make_stratified_chunks(y_pool, n_chunks=n_chunks, seed=int(args.seed))
    chunk_rows: List[Dict[str, object]] = []
    for i, c in enumerate(chunks):
        y_c = y_pool[c]
        chunk_rows.append(
            {
                "chunk_id": int(i),
                "n": int(c.size),
                "n_pos": int((y_c > 0.5).sum()),
                "n_neg": int((y_c < 0.5).sum()),
                "pos_frac": float((y_c > 0.5).mean()),
            }
        )

    # Score arrays optionally calibrated ONCE using first fit chunk.
    fit0 = chunks[0]
    scores_pool: Dict[str, np.ndarray] = {}
    scores_t: Dict[str, np.ndarray] = {}
    calibration_rows: List[Dict[str, object]] = []
    for m in model_order:
        sp, st, diag = _calibrate_from_fit(
            y_fit=y_pool[fit0],
            s_fit=scores_pool_raw[m][fit0],
            s_pool=scores_pool_raw[m],
            s_test=scores_test_raw[m],
            mode=str(args.calibration),
        )
        scores_pool[m] = sp
        scores_t[m] = st
        row = {"model": m}
        row.update({k: diag.get(k, "") for k in ["ok", "mode", "reason", "coef", "intercept"]})
        calibration_rows.append(row)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (fusion_json.parent / "rolling_chunk_greedy_31")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Greedy rolling.
    name_to_idx = {m: i for i, m in enumerate(model_order)}
    w_full = np.zeros((len(model_order),), dtype=np.float64)
    w_full[name_to_idx[str(args.anchor_model)]] = 1.0
    selected = [str(args.anchor_model)]
    remaining = [m for m in candidates if m != str(args.anchor_model)]

    s_cur_pool = scores_pool[str(args.anchor_model)].copy()
    s_cur_test = scores_t[str(args.anchor_model)].copy()
    s_best_cum_pool = s_cur_pool.copy()
    s_best_cum_test = s_cur_test.copy()
    best_cum_fpr = float("inf")
    best_cum_step = 0
    best_cum_weights = w_full.copy()
    best_cum_selected = list(selected)

    cal_seen = np.zeros((val_pool_size,), dtype=bool)
    w_grid = np.arange(float(args.w_step), 1.0 + 1e-12, float(args.w_step), dtype=np.float64)
    w_grid = w_grid[(w_grid > 0.0) & (w_grid <= 1.0)]
    if w_grid.size == 0:
        w_grid = np.asarray([0.5], dtype=np.float64)

    step_rows: List[Dict[str, object]] = []
    topk_rows: List[Dict[str, object]] = []
    no_improve = 0
    reject_counts = {"fit": 0, "cal": 0, "cum": 0, "none": 0}

    max_steps = int(max(1, args.max_steps))
    for step in range(1, max_steps + 1):
        if not remaining:
            break
        fit_id = int((step - 1) % n_chunks)
        cal_id = int(step % n_chunks)
        idx_fit = chunks[fit_id]
        idx_cal = chunks[cal_id]
        cal_seen[idx_cal] = True
        idx_cum = np.where(cal_seen)[0]
        if idx_cum.size == 0:
            idx_cum = idx_cal

        cur_fit = _eval_on_indices(y_pool, s_cur_pool, idx_fit, float(args.target_tpr))
        cur_cal = _eval_on_indices(y_pool, s_cur_pool, idx_cal, float(args.target_tpr))
        cur_cum = _eval_on_indices(y_pool, s_cur_pool, idx_cum, float(args.target_tpr))

        best = None
        cand_fit_best_rows: List[Dict[str, object]] = []
        y_fit = y_pool[idx_fit]
        s_cur_fit = s_cur_pool[idx_fit]
        for cand in remaining:
            s_c_fit = scores_pool[cand][idx_fit]
            best_c = None
            for w in w_grid:
                s_try_fit = (1.0 - w) * s_cur_fit + w * s_c_fit
                fpr_fit = _fpr_only_fit(y_fit, s_try_fit, float(args.target_tpr))
                if best_c is None or float(fpr_fit) < float(best_c["fpr_fit"]) - 1e-15:
                    best_c = {"cand": cand, "w": float(w), "fpr_fit": float(fpr_fit)}
            if best_c is None:
                continue
            cand_fit_best_rows.append(best_c)
            if best is None:
                best = dict(best_c)
            else:
                if float(best_c["fpr_fit"]) < float(best["fpr_fit"]) - 1e-15:
                    best = dict(best_c)

        cand_fit_best_rows = sorted(cand_fit_best_rows, key=lambda r: float(r["fpr_fit"]))
        for rank, r in enumerate(cand_fit_best_rows[: int(max(1, args.topk_diagnostic))], start=1):
            topk_rows.append(
                {
                    "step": int(step),
                    "rank": int(rank),
                    "fit_chunk": int(fit_id),
                    "cal_chunk": int(cal_id),
                    "candidate": str(r["cand"]),
                    "best_w_fit": float(r["w"]),
                    "best_fpr_fit": float(r["fpr_fit"]),
                    "current_fpr_fit": float(cur_fit["fpr"]),
                    "gain_fit": float(cur_fit["fpr"] - float(r["fpr_fit"])),
                }
            )

        if best is None:
            break

        cand = str(best["cand"])
        w = float(best["w"])
        s_new_pool = (1.0 - w) * s_cur_pool + w * scores_pool[cand]
        s_new_test = (1.0 - w) * s_cur_test + w * scores_t[cand]

        new_fit = _eval_on_indices(y_pool, s_new_pool, idx_fit, float(args.target_tpr))
        new_cal = _eval_on_indices(y_pool, s_new_pool, idx_cal, float(args.target_tpr))
        new_cum = _eval_on_indices(y_pool, s_new_pool, idx_cum, float(args.target_tpr))

        gain_fit = float(cur_fit["fpr"] - new_fit["fpr"])
        gain_cal = float(cur_cal["fpr"] - new_cal["fpr"])
        gain_cum = float(cur_cum["fpr"] - new_cum["fpr"])

        accept = True
        reason = "accept"
        if gain_fit < float(args.min_fit_gain):
            accept = False
            reason = "reject_fit"
            reject_counts["fit"] += 1
        elif gain_cal < float(args.min_cal_gain):
            accept = False
            reason = "reject_cal"
            reject_counts["cal"] += 1
        elif gain_cum < float(args.min_cum_cal_gain):
            accept = False
            reason = "reject_cum"
            reject_counts["cum"] += 1

        if accept:
            s_cur_pool = s_new_pool
            s_cur_test = s_new_test
            w_full *= (1.0 - w)
            w_full[name_to_idx[cand]] += w
            selected.append(cand)
            remaining = [m for m in remaining if m != cand]
            no_improve = 0
            if float(new_cum["fpr"]) < float(best_cum_fpr):
                best_cum_fpr = float(new_cum["fpr"])
                best_cum_step = int(step)
                s_best_cum_pool = s_cur_pool.copy()
                s_best_cum_test = s_cur_test.copy()
                best_cum_weights = w_full.copy()
                best_cum_selected = list(selected)
        else:
            no_improve += 1

        step_rows.append(
            {
                "step": int(step),
                "fit_chunk": int(fit_id),
                "cal_chunk": int(cal_id),
                "n_fit": int(idx_fit.size),
                "n_cal": int(idx_cal.size),
                "n_cum_cal": int(idx_cum.size),
                "best_candidate_fit": cand,
                "best_w_fit": float(w),
                "action": reason,
                "accepted": int(1 if accept else 0),
                "fpr_fit_before": float(cur_fit["fpr"]),
                "fpr_fit_after": float(new_fit["fpr"]),
                "gain_fit": float(gain_fit),
                "fpr_cal_before": float(cur_cal["fpr"]),
                "fpr_cal_after": float(new_cal["fpr"]),
                "gain_cal": float(gain_cal),
                "fpr_cum_before": float(cur_cum["fpr"]),
                "fpr_cum_after": float(new_cum["fpr"]),
                "gain_cum": float(gain_cum),
                "auc_fit_before": float(cur_fit["auc"]),
                "auc_fit_after": float(new_fit["auc"]),
                "auc_cal_before": float(cur_cal["auc"]),
                "auc_cal_after": float(new_cal["auc"]),
                "auc_cum_before": float(cur_cum["auc"]),
                "auc_cum_after": float(new_cum["auc"]),
                "n_selected": int(len(selected)),
                "selected_models": ",".join(selected),
                "weight_vector": _weights_to_str(w_full, model_order),
                "remaining_candidates": int(len(remaining)),
            }
        )

        if no_improve >= int(max(1, args.patience)):
            break

    # Final reference set: union of seen cal chunks (fallback full pool).
    idx_final_ref = np.where(cal_seen)[0]
    if idx_final_ref.size == 0:
        idx_final_ref = np.arange(val_pool_size, dtype=np.int64)
        reject_counts["none"] += 1
    y_ref = y_pool[idx_final_ref]

    # Snapshot metrics.
    anchor_pool = scores_pool[str(args.anchor_model)]
    anchor_test = scores_t[str(args.anchor_model)]

    final_eval = _eval_test_from_ref(
        y_ref=y_ref,
        s_ref=s_cur_pool[idx_final_ref],
        y_test=y_test,
        s_test=s_cur_test,
        target_tpr=float(args.target_tpr),
    )
    anchor_eval = _eval_test_from_ref(
        y_ref=y_ref,
        s_ref=anchor_pool[idx_final_ref],
        y_test=y_test,
        s_test=anchor_test,
        target_tpr=float(args.target_tpr),
    )
    best_cum_eval = _eval_test_from_ref(
        y_ref=y_ref,
        s_ref=s_best_cum_pool[idx_final_ref],
        y_test=y_test,
        s_test=s_best_cum_test,
        target_tpr=float(args.target_tpr),
    )

    # Oracle threshold on test for diagnostic only (not selection).
    thr_oracle_final = float(base.threshold_for_target_tpr(y_test, s_cur_test, float(args.target_tpr)))
    rr_oracle_final = base.rates_from_threshold(y_test, s_cur_test, thr_oracle_final)
    oracle_diag = {
        "threshold_from_test": thr_oracle_final,
        "fpr_test_oracle_threshold": float(rr_oracle_final["fpr"]),
        "tpr_test_oracle_threshold": float(rr_oracle_final["tpr"]),
    }

    summary_rows = [
        {
            "method": "anchor",
            "target_tpr": float(args.target_tpr),
            **anchor_eval,
            "selected_models": str(args.anchor_model),
            "weight_vector": _weights_to_str(np.eye(len(model_order))[name_to_idx[str(args.anchor_model)]], model_order),
        },
        {
            "method": "rolling_final",
            "target_tpr": float(args.target_tpr),
            **final_eval,
            "selected_models": ",".join(selected),
            "weight_vector": _weights_to_str(w_full, model_order),
        },
        {
            "method": "rolling_best_cum",
            "target_tpr": float(args.target_tpr),
            **best_cum_eval,
            "selected_models": ",".join(best_cum_selected),
            "weight_vector": _weights_to_str(best_cum_weights, model_order),
            "best_cum_step": int(best_cum_step),
        },
    ]

    _save_csv_dynamic(out_dir / "rolling_step_log.csv", step_rows)
    _save_csv_dynamic(out_dir / "rolling_fit_topk_by_step.csv", topk_rows)
    _save_csv_dynamic(out_dir / "rolling_chunk_stats.csv", chunk_rows)
    _save_csv_dynamic(out_dir / "rolling_calibration_diagnostics.csv", calibration_rows)
    _save_csv_dynamic(out_dir / "rolling_prefilter_single_model.csv", prefilter_rows)
    _save_csv_dynamic(out_dir / "rolling_summary.csv", summary_rows)

    np.savez_compressed(
        out_dir / "rolling_scores.npz",
        pool_abs=pool_abs.astype(np.int64),
        idx_final_ref=idx_final_ref.astype(np.int64),
        labels_pool=y_pool.astype(np.float32),
        labels_test=np.asarray(y_test, dtype=np.float32),
        score_anchor_pool=anchor_pool.astype(np.float32),
        score_anchor_test=anchor_test.astype(np.float32),
        score_final_pool=s_cur_pool.astype(np.float32),
        score_final_test=s_cur_test.astype(np.float32),
        score_best_cum_pool=s_best_cum_pool.astype(np.float32),
        score_best_cum_test=s_best_cum_test.astype(np.float32),
    )

    report = {
        "fusion_json": str(fusion_json),
        "out_dir": str(out_dir),
        "settings": vars(args),
        "using_precomputed_scores": bool(using_precomputed),
        "precomputed_scores_npz": str(Path(pre_npz).expanduser().resolve()) if using_precomputed else "",
        "precomputed_manifest_json": str(Path(pre_manifest).expanduser().resolve()) if using_precomputed else "",
        "n_models_total": int(len(model_order)),
        "n_val_available": int(n_val_avail),
        "val_pool_size": int(val_pool_size),
        "val_pool_offset": int(val_pool_offset),
        "target_tpr": float(args.target_tpr),
        "anchor_model": str(args.anchor_model),
        "candidate_models_initial": [str(r["model"]) for r in prefilter_rows],
        "candidate_models_used": [m for m in candidates],
        "chunk_stats": chunk_rows,
        "selected_models_final": selected,
        "selected_models_best_cum": best_cum_selected,
        "best_cum_step": int(best_cum_step),
        "weights_final": {model_order[i]: float(w_full[i]) for i in range(len(model_order)) if float(w_full[i]) > 1e-12},
        "weights_best_cum": {model_order[i]: float(best_cum_weights[i]) for i in range(len(model_order)) if float(best_cum_weights[i]) > 1e-12},
        "anchor_eval": anchor_eval,
        "final_eval": final_eval,
        "best_cum_eval": best_cum_eval,
        "oracle_diag_final": oracle_diag,
        "reject_counts": reject_counts,
        "timing_sec": float(time.time() - t0),
        "score_files_used": score_files,
        "files": {
            "summary_csv": str((out_dir / "rolling_summary.csv").resolve()),
            "step_log_csv": str((out_dir / "rolling_step_log.csv").resolve()),
            "fit_topk_csv": str((out_dir / "rolling_fit_topk_by_step.csv").resolve()),
            "chunk_stats_csv": str((out_dir / "rolling_chunk_stats.csv").resolve()),
            "calibration_diag_csv": str((out_dir / "rolling_calibration_diagnostics.csv").resolve()),
            "prefilter_csv": str((out_dir / "rolling_prefilter_single_model.csv").resolve()),
            "scores_npz": str((out_dir / "rolling_scores.npz").resolve()),
        },
    }

    report_json = (
        Path(args.report_json).expanduser().resolve()
        if str(args.report_json).strip()
        else (out_dir / "rolling_chunk_greedy_report.json")
    )
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("31-Model Rolling-Chunk Greedy Fusion")
    print("=" * 72)
    print(f"Fusion json: {fusion_json}")
    print(f"Out dir:     {out_dir}")
    print(f"Val avail:   {n_val_avail}")
    print(f"Val pool:    offset={val_pool_offset}, size={val_pool_size}")
    print(f"Chunks:      {n_chunks} (cyclic fit/cal)")
    print(f"Target TPR:  {float(args.target_tpr):.3f}")
    print(f"Anchor:      {args.anchor_model}")
    print(f"Candidates:  {len(candidates)}")
    print(f"Calibration: {args.calibration} (fit-once on chunk0)")
    print()
    print(
        f"Anchor      AUC_test={float(anchor_eval['auc_test']):.6f} "
        f"FPR_test={float(anchor_eval['fpr_test']):.6f} "
        f"(ref_fpr={float(anchor_eval['fpr_ref']):.6f})"
    )
    print(
        f"Rolling final AUC_test={float(final_eval['auc_test']):.6f} "
        f"FPR_test={float(final_eval['fpr_test']):.6f} "
        f"(ref_fpr={float(final_eval['fpr_ref']):.6f})"
    )
    print(
        f"Best cum    AUC_test={float(best_cum_eval['auc_test']):.6f} "
        f"FPR_test={float(best_cum_eval['fpr_test']):.6f} "
        f"(step={int(best_cum_step)})"
    )
    print(f"Selected final: {','.join(selected)}")
    print(f"Saved report: {report_json}")
    print(f"Saved summary: {out_dir / 'rolling_summary.csv'}")
    print(f"Saved step log: {out_dir / 'rolling_step_log.csv'}")


if __name__ == "__main__":
    main()

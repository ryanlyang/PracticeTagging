#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusion analysis for:
- Previous 18-model fusion setup
- Additional dualreco-dualview model family (using frozen/pre-joint scores)
- Additional m2 delta ablations (delta000, delta020) using joint scores

Outputs mirror the joint18 analyzer, with weighted keys named by model count:
- all{N}_weighted_raw/platt/iso_{valsel|oracle}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

import analyze_hlt_joint18_recoteacher_fusion as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as m




def _parse_float_grid(spec: str, default: Sequence[float]) -> List[float]:
    vals: List[float] = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except Exception:
            continue
    if not vals:
        vals = [float(x) for x in default]
    out: List[float] = []
    for v in vals:
        if np.isfinite(v):
            out.append(float(v))
    return out if out else [float(x) for x in default]


def _parse_int_grid(spec: str, default: Sequence[int]) -> List[int]:
    vals: List[int] = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except Exception:
            continue
    if not vals:
        vals = [int(x) for x in default]
    out = sorted(set(max(1, int(v)) for v in vals))
    return out if out else [int(x) for x in default]


def _parse_hidden_grid(spec: str, default: Sequence[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for blk in str(spec).split('|'):
        blk = blk.strip()
        if not blk:
            continue
        vals: List[int] = []
        for tok in blk.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = int(tok)
            except Exception:
                continue
            if v > 0:
                vals.append(v)
        if vals:
            out.append(tuple(vals))
    if not out:
        out = [tuple(x) for x in default]
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError('simplex projection expects 1D vector')
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u - (cssv - 1.0) / (np.arange(v.size) + 1.0) > 0.0)[0]
    if rho.size == 0:
        out = np.ones_like(v) / float(v.size)
        return out
    rho_idx = int(rho[-1])
    theta = (cssv[rho_idx] - 1.0) / float(rho_idx + 1)
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        return np.ones_like(v) / float(v.size)
    return w / s


def _fit_sparse_simplex_logit(
    X: np.ndarray,
    y: np.ndarray,
    l1: float,
    lr: float,
    steps: int,
) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape

    w = np.ones(d, dtype=np.float64) / float(max(d, 1))
    b = 0.0

    best_loss = float('inf')
    best_w = w.copy()
    best_b = float(b)

    l1 = float(max(0.0, l1))
    lr = float(max(1e-5, lr))
    steps = int(max(10, steps))

    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        err = (p - y)

        grad_w = (X.T @ err) / float(n)
        grad_b = float(err.mean())

        w = w - lr * grad_w
        b = b - lr * grad_b

        if l1 > 0.0:
            w = np.maximum(0.0, w - lr * l1)
        w = _project_to_simplex(w)

        z2 = X @ w + b
        p2 = _sigmoid(z2)
        p2 = np.clip(p2, 1e-6, 1.0 - 1e-6)
        bce = -np.mean(y * np.log(p2) + (1.0 - y) * np.log(1.0 - p2))
        loss = float(bce)
        if loss < best_loss:
            best_loss = loss
            best_w = w.copy()
            best_b = float(b)

    return best_w, best_b


def _apply_sparse_simplex_logit(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _sigmoid(np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64) + float(b)).astype(np.float64)


def _choose_score_calibration(
    y_cal: np.ndarray,
    s_cal_raw: np.ndarray,
    s_test_raw: np.ndarray,
    target_tpr: float,
) -> Dict[str, object]:
    y_cal_i = np.asarray(y_cal, dtype=np.int64)
    s_cal_raw = np.asarray(s_cal_raw, dtype=np.float64)
    s_test_raw = np.asarray(s_test_raw, dtype=np.float64)

    candidates: Dict[str, Dict[str, object]] = {}

    def _pack(name: str, s_cal: np.ndarray, s_test: np.ndarray, meta: Dict[str, object]) -> None:
        thr = base.threshold_for_target_tpr(y_cal_i, s_cal, target_tpr)
        rates = base.rates_from_threshold(y_cal_i, s_cal, thr)
        auc_cal = base.auc_and_fpr_at_target(y_cal_i, s_cal, target_tpr)
        candidates[name] = {
            'scores_cal': np.asarray(s_cal, dtype=np.float64),
            'scores_test': np.asarray(s_test, dtype=np.float64),
            'threshold_cal': float(thr),
            'fpr_cal': float(rates['fpr']),
            'tpr_cal': float(rates['tpr']),
            'auc_cal': float(auc_cal['auc']),
            'meta': dict(meta),
        }

    _pack('raw', s_cal_raw, s_test_raw, {'ok': True})

    if np.unique(y_cal_i).size >= 2:
        try:
            lr = LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced')
            lr.fit(s_cal_raw.reshape(-1, 1), y_cal_i)
            s_cal_p = lr.predict_proba(s_cal_raw.reshape(-1, 1))[:, 1].astype(np.float64)
            s_test_p = lr.predict_proba(s_test_raw.reshape(-1, 1))[:, 1].astype(np.float64)
            _pack(
                'platt',
                s_cal_p,
                s_test_p,
                {
                    'ok': True,
                    'coef': float(lr.coef_.ravel()[0]),
                    'intercept': float(lr.intercept_.ravel()[0]),
                },
            )
        except Exception as e:
            candidates['platt_error'] = {'error': repr(e)}

        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(s_cal_raw, y_cal_i.astype(np.float64))
            s_cal_i = np.asarray(iso.transform(s_cal_raw), dtype=np.float64)
            s_test_i = np.asarray(iso.transform(s_test_raw), dtype=np.float64)
            _pack('isotonic', s_cal_i, s_test_i, {'ok': True})
        except Exception as e:
            candidates['isotonic_error'] = {'error': repr(e)}

    valid = [(k, v) for k, v in candidates.items() if isinstance(v, dict) and 'fpr_cal' in v]
    valid_sorted = sorted(valid, key=lambda kv: (float(kv[1]['fpr_cal']), -float(kv[1]['auc_cal'])))
    if not valid_sorted:
        return {
            'chosen': 'raw',
            'scores_test': s_test_raw,
            'threshold_cal': float('nan'),
            'diagnostics': candidates,
        }

    best_name, best_pack = valid_sorted[0]
    return {
        'chosen': str(best_name),
        'scores_test': np.asarray(best_pack['scores_test'], dtype=np.float64),
        'threshold_cal': float(best_pack['threshold_cal']),
        'diagnostics': {
            k: {
                kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                for kk, vv in v.items()
                if kk not in {'scores_cal', 'scores_test'}
            }
            for k, v in candidates.items()
            if isinstance(v, dict) and 'scores_cal' in v
        },
    }


def _build_eval_pack(y_test: np.ndarray, s_test: np.ndarray, threshold: float, target_tpr: float) -> Dict[str, object]:
    rates_test = base.rates_from_threshold(y_test, s_test, threshold)
    a_test = base.auc_and_fpr_at_target(y_test, s_test, target_tpr)
    return {
        'threshold_from_cal': float(threshold),
        'auc': float(a_test['auc']),
        'fpr': float(rates_test['fpr']),
        'tpr': float(rates_test['tpr']),
        'tp': int(rates_test['tp']),
        'fp': int(rates_test['fp']),
        'fpr_at_target_tpr_exact': float(a_test['fpr_at_target_tpr']),
    }


def _train_sparse_stable_stacker(
    score_mat_val: np.ndarray,
    y_val: np.ndarray,
    score_mat_test: np.ndarray,
    y_test: np.ndarray,
    model_order: List[str],
    target_tpr: float,
    sel_frac: float,
    seed: int,
    l1_grid: Sequence[float],
    folds: int,
    stability_seeds: Sequence[int],
    min_freq: float,
    select_threshold: float,
    train_steps: int,
    train_lr: float,
) -> Dict[str, object]:
    X_val = np.asarray(score_mat_val, dtype=np.float64).T
    X_test = np.asarray(score_mat_test, dtype=np.float64).T
    yv = np.asarray(y_val, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64)

    idx = np.arange(yv.size)
    idx_fit, idx_cal = train_test_split(
        idx,
        test_size=float(np.clip(sel_frac, 0.1, 0.8)),
        random_state=int(seed),
        stratify=yv.astype(np.int64),
    )

    X_fit = X_val[idx_fit]
    y_fit = yv[idx_fit]
    X_cal = X_val[idx_cal]
    y_cal = yv[idx_cal]

    n_models = X_val.shape[1]
    counts = np.zeros(n_models, dtype=np.float64)
    total = 0.0

    for s in stability_seeds:
        skf = StratifiedKFold(n_splits=max(2, int(folds)), shuffle=True, random_state=int(s))
        for tr_loc, va_loc in skf.split(X_fit, y_fit.astype(np.int64)):
            X_tr = X_fit[tr_loc]
            y_tr = y_fit[tr_loc]
            X_va = X_fit[va_loc]
            y_va = y_fit[va_loc]

            best = None
            for l1 in l1_grid:
                w, b = _fit_sparse_simplex_logit(
                    X=X_tr,
                    y=y_tr,
                    l1=float(l1),
                    lr=float(train_lr),
                    steps=int(train_steps),
                )
                s_va = _apply_sparse_simplex_logit(X_va, w, b)
                thr = base.threshold_for_target_tpr(y_va, s_va, target_tpr)
                rates = base.rates_from_threshold(y_va, s_va, thr)
                cand = {
                    'l1': float(l1),
                    'w': w,
                    'b': float(b),
                    'fpr': float(rates['fpr']),
                }
                if best is None or cand['fpr'] < best['fpr']:
                    best = cand

            if best is None:
                continue
            counts += (best['w'] > float(select_threshold)).astype(np.float64)
            total += 1.0

    if total <= 0.0:
        freq = np.ones(n_models, dtype=np.float64)
    else:
        freq = counts / total

    selected = freq >= float(min_freq)
    if selected.size > 0:
        selected[0] = True  # keep HLT anchor
    if not bool(selected.any()):
        top_k = min(6, n_models)
        idx_top = np.argsort(-freq)[:top_k]
        selected[idx_top] = True

    sel_idx = np.where(selected)[0]
    sel_names = [model_order[i] for i in sel_idx.tolist()]

    X_fit_s = X_fit[:, sel_idx]
    X_cal_s = X_cal[:, sel_idx]
    X_test_s = X_test[:, sel_idx]

    best_final = None
    for l1 in l1_grid:
        w, b = _fit_sparse_simplex_logit(
            X=X_fit_s,
            y=y_fit,
            l1=float(l1),
            lr=float(train_lr),
            steps=int(train_steps),
        )
        s_cal = _apply_sparse_simplex_logit(X_cal_s, w, b)
        thr = base.threshold_for_target_tpr(y_cal, s_cal, target_tpr)
        rates = base.rates_from_threshold(y_cal, s_cal, thr)
        cand = {
            'l1': float(l1),
            'w': w,
            'b': float(b),
            'fpr_cal': float(rates['fpr']),
            'thr_cal': float(thr),
        }
        if best_final is None or cand['fpr_cal'] < best_final['fpr_cal']:
            best_final = cand

    if best_final is None:
        return {
            'selection': {'ok': False, 'reason': 'fit_failed'},
            'test_eval': {'auc': float('nan'), 'fpr': float('nan'), 'tpr': float('nan'), 'fpr_at_target_tpr_exact': float('nan')},
            'oracle_test': {'auc': float('nan'), 'fpr': float('nan'), 'tpr': float('nan'), 'fpr_at_target_tpr_exact': float('nan')},
        }

    s_cal_raw = _apply_sparse_simplex_logit(X_cal_s, best_final['w'], best_final['b'])
    s_test_raw = _apply_sparse_simplex_logit(X_test_s, best_final['w'], best_final['b'])

    cal_pick = _choose_score_calibration(
        y_cal=y_cal,
        s_cal_raw=s_cal_raw,
        s_test_raw=s_test_raw,
        target_tpr=target_tpr,
    )

    s_test = np.asarray(cal_pick['scores_test'], dtype=np.float64)
    thr_cal = float(cal_pick['threshold_cal'])

    test_eval = _build_eval_pack(yt, s_test, thr_cal, target_tpr)
    thr_oracle = base.threshold_for_target_tpr(yt, s_test, target_tpr)
    oracle_test = _build_eval_pack(yt, s_test, thr_oracle, target_tpr)

    full_w = np.zeros(n_models, dtype=np.float64)
    full_w[sel_idx] = np.asarray(best_final['w'], dtype=np.float64)

    return {
        'selection': {
            'ok': True,
            'selected_models': sel_names,
            'selected_count': int(len(sel_names)),
            'stability_frequencies': {model_order[i]: float(freq[i]) for i in range(n_models)},
            'l1': float(best_final['l1']),
            'calibration': str(cal_pick['chosen']),
            'calibration_diagnostics': cal_pick['diagnostics'],
            'weights': {model_order[i]: float(full_w[i]) for i in range(n_models)},
            'bias': float(best_final['b']),
        },
        'test_eval': test_eval,
        'oracle_test': oracle_test,
    }


def _build_nonlinear_features(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    hlt = P[:, [0]]
    d_hlt = np.abs(P - hlt)
    lg = _safe_logit(P)
    m = P.mean(axis=1, keepdims=True)
    s = P.std(axis=1, keepdims=True)
    pmax = P.max(axis=1, keepdims=True)
    pmin = P.min(axis=1, keepdims=True)
    part = np.partition(P, kth=max(0, P.shape[1] - 2), axis=1)
    top2 = part[:, -2:]
    top1 = np.max(top2, axis=1, keepdims=True)
    top2v = np.min(top2, axis=1, keepdims=True)
    gap = top1 - top2v
    return np.concatenate([P, lg, d_hlt, m, s, pmax, pmin, gap], axis=1).astype(np.float64)


def _train_tiny_mlp_stacker(
    score_mat_val: np.ndarray,
    y_val: np.ndarray,
    score_mat_test: np.ndarray,
    y_test: np.ndarray,
    target_tpr: float,
    sel_frac: float,
    seed: int,
    hidden_grid: Sequence[Tuple[int, ...]],
    alpha_grid: Sequence[float],
    max_iter: int,
) -> Dict[str, object]:
    P_val = np.asarray(score_mat_val, dtype=np.float64).T
    P_test = np.asarray(score_mat_test, dtype=np.float64).T
    yv = np.asarray(y_val, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64)

    idx = np.arange(yv.size)
    idx_fit, idx_cal = train_test_split(
        idx,
        test_size=float(np.clip(sel_frac, 0.1, 0.8)),
        random_state=int(seed),
        stratify=yv.astype(np.int64),
    )

    X_val = _build_nonlinear_features(P_val)
    X_test = _build_nonlinear_features(P_test)

    X_fit = X_val[idx_fit]
    y_fit = yv[idx_fit].astype(np.int64)
    X_cal = X_val[idx_cal]
    y_cal = yv[idx_cal]

    mu = X_fit.mean(axis=0, keepdims=True)
    sd = X_fit.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)

    X_fit_s = (X_fit - mu) / sd
    X_cal_s = (X_cal - mu) / sd
    X_test_s = (X_test - mu) / sd

    best = None
    for hidden in hidden_grid:
        for alpha in alpha_grid:
            clf = MLPClassifier(
                hidden_layer_sizes=tuple(int(x) for x in hidden),
                activation='relu',
                solver='adam',
                alpha=float(max(0.0, alpha)),
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=int(max(50, max_iter)),
                early_stopping=True,
                n_iter_no_change=12,
                random_state=int(seed),
            )
            try:
                clf.fit(X_fit_s, y_fit)
                s_cal = clf.predict_proba(X_cal_s)[:, 1].astype(np.float64)
            except Exception:
                continue

            thr = base.threshold_for_target_tpr(y_cal, s_cal, target_tpr)
            rates = base.rates_from_threshold(y_cal, s_cal, thr)
            cand = {
                'hidden': tuple(int(x) for x in hidden),
                'alpha': float(alpha),
                'model': clf,
                'fpr_cal': float(rates['fpr']),
                'thr_cal': float(thr),
            }
            if best is None or cand['fpr_cal'] < best['fpr_cal']:
                best = cand

    if best is None:
        return {
            'selection': {'ok': False, 'reason': 'fit_failed'},
            'test_eval': {'auc': float('nan'), 'fpr': float('nan'), 'tpr': float('nan'), 'fpr_at_target_tpr_exact': float('nan')},
            'oracle_test': {'auc': float('nan'), 'fpr': float('nan'), 'tpr': float('nan'), 'fpr_at_target_tpr_exact': float('nan')},
        }

    clf_best = best['model']
    s_cal_raw = clf_best.predict_proba(X_cal_s)[:, 1].astype(np.float64)
    s_test_raw = clf_best.predict_proba(X_test_s)[:, 1].astype(np.float64)

    cal_pick = _choose_score_calibration(
        y_cal=y_cal,
        s_cal_raw=s_cal_raw,
        s_test_raw=s_test_raw,
        target_tpr=target_tpr,
    )

    s_test = np.asarray(cal_pick['scores_test'], dtype=np.float64)
    thr_cal = float(cal_pick['threshold_cal'])

    test_eval = _build_eval_pack(yt, s_test, thr_cal, target_tpr)
    thr_oracle = base.threshold_for_target_tpr(yt, s_test, target_tpr)
    oracle_test = _build_eval_pack(yt, s_test, thr_oracle, target_tpr)

    return {
        'selection': {
            'ok': True,
            'hidden': [int(x) for x in best['hidden']],
            'alpha': float(best['alpha']),
            'calibration': str(cal_pick['chosen']),
            'calibration_diagnostics': cal_pick['diagnostics'],
            'n_iter': int(getattr(clf_best, 'n_iter_', -1)),
        },
        'test_eval': test_eval,
        'oracle_test': oracle_test,
    }


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    den = np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)
    return ez / den


def _build_gate_features(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    hlt = P[:, [0]]
    d_hlt = np.abs(P - hlt)
    m = P.mean(axis=1, keepdims=True)
    s = P.std(axis=1, keepdims=True)
    pmax = P.max(axis=1, keepdims=True)
    pmin = P.min(axis=1, keepdims=True)
    return np.concatenate([P, d_hlt, m, s, pmax, pmin], axis=1).astype(np.float64)


def _moe_predict(
    P: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    temp: float,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    temp = float(max(1e-3, temp))
    z = (np.asarray(X, dtype=np.float64) @ np.asarray(W, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / temp
    g = _softmax_rows(z)

    if int(topk) > 0 and int(topk) < g.shape[1]:
        k = int(topk)
        idx = np.argpartition(-g, kth=k - 1, axis=1)[:, :k]
        mask = np.zeros_like(g, dtype=bool)
        rows = np.arange(g.shape[0])[:, None]
        mask[rows, idx] = True
        g = np.where(mask, g, 0.0)
        g = g / np.clip(g.sum(axis=1, keepdims=True), 1e-12, None)

    s = np.sum(g * np.asarray(P, dtype=np.float64), axis=1)
    s = np.clip(s, 1e-6, 1.0 - 1e-6)
    return s.astype(np.float64), g.astype(np.float64)


def _fit_moe_gate_adam(
    X: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    temp: float,
    entropy_lambda: float,
    l2: float,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n, d = X.shape
    m_models = P.shape[1]

    rng = np.random.default_rng(int(seed))
    W = rng.normal(0.0, 0.02, size=(d, m_models)).astype(np.float64)
    b = np.zeros(m_models, dtype=np.float64)

    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    bs = int(max(128, min(int(batch_size), n)))
    t = 0

    for _ in range(int(max(20, steps))):
        t += 1
        idx = rng.integers(0, n, size=bs)
        Xb = X[idx]
        Pb = P[idx]
        yb = y[idx]

        s, g = _moe_predict(P=Pb, X=Xb, W=W, b=b, temp=temp, topk=0)
        s = np.clip(s, 1e-6, 1.0 - 1e-6)

        dlds = (s - yb) / (s * (1.0 - s)) / float(bs)
        dldg = dlds[:, None] * Pb

        if float(entropy_lambda) > 0.0:
            dHdg = -(np.log(np.clip(g, 1e-9, 1.0)) + 1.0)
            dldg = dldg + (float(entropy_lambda) / float(bs)) * dHdg

        tmp = np.sum(dldg * g, axis=1, keepdims=True)
        dlda = g * (dldg - tmp)
        dldz = dlda / float(max(1e-3, temp))

        gradW = Xb.T @ dldz + float(max(0.0, l2)) * W
        gradb = np.sum(dldz, axis=0)

        mW = beta1 * mW + (1.0 - beta1) * gradW
        vW = beta2 * vW + (1.0 - beta2) * (gradW * gradW)
        mb = beta1 * mb + (1.0 - beta1) * gradb
        vb = beta2 * vb + (1.0 - beta2) * (gradb * gradb)

        mW_hat = mW / (1.0 - beta1 ** t)
        vW_hat = vW / (1.0 - beta2 ** t)
        mb_hat = mb / (1.0 - beta1 ** t)
        vb_hat = vb / (1.0 - beta2 ** t)

        W = W - float(lr) * mW_hat / (np.sqrt(vW_hat) + eps)
        b = b - float(lr) * mb_hat / (np.sqrt(vb_hat) + eps)

    return W, b


def _train_moe_gated_stacker(
    score_mat_val: np.ndarray,
    y_val: np.ndarray,
    score_mat_test: np.ndarray,
    y_test: np.ndarray,
    target_tpr: float,
    sel_frac: float,
    seed: int,
    entropy_grid: Sequence[float],
    l2_grid: Sequence[float],
    temp_grid: Sequence[float],
    topk_grid: Sequence[int],
    steps: int,
    batch_size: int,
    lr: float,
) -> Dict[str, object]:
    P_val = np.asarray(score_mat_val, dtype=np.float64).T
    P_test = np.asarray(score_mat_test, dtype=np.float64).T
    yv = np.asarray(y_val, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64)

    idx = np.arange(yv.size)
    idx_fit, idx_cal = train_test_split(
        idx,
        test_size=float(np.clip(sel_frac, 0.1, 0.8)),
        random_state=int(seed),
        stratify=yv.astype(np.int64),
    )

    X_val = _build_gate_features(P_val)
    X_test = _build_gate_features(P_test)

    X_fit = X_val[idx_fit]
    y_fit = yv[idx_fit]
    P_fit = P_val[idx_fit]

    X_cal = X_val[idx_cal]
    y_cal = yv[idx_cal]
    P_cal = P_val[idx_cal]

    mu = X_fit.mean(axis=0, keepdims=True)
    sd = X_fit.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)

    X_fit_s = (X_fit - mu) / sd
    X_cal_s = (X_cal - mu) / sd
    X_test_s = (X_test - mu) / sd

    best = None
    cfg_idx = 0
    for ent in entropy_grid:
        for l2 in l2_grid:
            for temp in temp_grid:
                for topk in topk_grid:
                    cfg_idx += 1
                    W, b = _fit_moe_gate_adam(
                        X=X_fit_s,
                        P=P_fit,
                        y=y_fit,
                        temp=float(temp),
                        entropy_lambda=float(ent),
                        l2=float(l2),
                        steps=int(steps),
                        batch_size=int(batch_size),
                        lr=float(lr),
                        seed=int(seed + 101 * cfg_idx),
                    )
                    s_cal, g_cal = _moe_predict(P=P_cal, X=X_cal_s, W=W, b=b, temp=float(temp), topk=int(topk))
                    thr = base.threshold_for_target_tpr(y_cal, s_cal, target_tpr)
                    rates = base.rates_from_threshold(y_cal, s_cal, thr)
                    cand = {
                        'entropy_lambda': float(ent),
                        'l2': float(l2),
                        'temp': float(temp),
                        'topk': int(topk),
                        'W': W,
                        'b': b,
                        'fpr_cal': float(rates['fpr']),
                        'avg_entropy_cal': float((-g_cal * np.log(np.clip(g_cal, 1e-9, 1.0))).sum(axis=1).mean()),
                        'avg_active_cal': float((g_cal > 1e-4).sum(axis=1).mean()),
                    }
                    if best is None or cand['fpr_cal'] < best['fpr_cal']:
                        best = cand

    if best is None:
        return {
            'selection': {'ok': False, 'reason': 'fit_failed'},
            'test_eval': {'auc': float('nan'), 'fpr': float('nan'), 'tpr': float('nan'), 'fpr_at_target_tpr_exact': float('nan')},
            'oracle_test': {'auc': float('nan'), 'fpr': float('nan'), 'tpr': float('nan'), 'fpr_at_target_tpr_exact': float('nan')},
        }

    s_cal_raw, g_cal = _moe_predict(
        P=P_cal,
        X=X_cal_s,
        W=best['W'],
        b=best['b'],
        temp=float(best['temp']),
        topk=int(best['topk']),
    )
    s_test_raw, g_test = _moe_predict(
        P=P_test,
        X=X_test_s,
        W=best['W'],
        b=best['b'],
        temp=float(best['temp']),
        topk=int(best['topk']),
    )

    cal_pick = _choose_score_calibration(
        y_cal=y_cal,
        s_cal_raw=s_cal_raw,
        s_test_raw=s_test_raw,
        target_tpr=target_tpr,
    )

    s_test = np.asarray(cal_pick['scores_test'], dtype=np.float64)
    thr_cal = float(cal_pick['threshold_cal'])

    test_eval = _build_eval_pack(yt, s_test, thr_cal, target_tpr)
    thr_oracle = base.threshold_for_target_tpr(yt, s_test, target_tpr)
    oracle_test = _build_eval_pack(yt, s_test, thr_oracle, target_tpr)

    return {
        'selection': {
            'ok': True,
            'entropy_lambda': float(best['entropy_lambda']),
            'l2': float(best['l2']),
            'temp': float(best['temp']),
            'topk': int(best['topk']),
            'calibration': str(cal_pick['chosen']),
            'calibration_diagnostics': cal_pick['diagnostics'],
            'avg_entropy_cal': float(best['avg_entropy_cal']),
            'avg_active_cal': float(best['avg_active_cal']),
            'avg_entropy_test': float((-g_test * np.log(np.clip(g_test, 1e-9, 1.0))).sum(axis=1).mean()),
            'avg_active_test': float((g_test > 1e-4).sum(axis=1).mean()),
        },
        'test_eval': test_eval,
        'oracle_test': oracle_test,
    }


def _collect_candidates(results: Dict[str, object], n_models: int) -> List[Dict[str, float | str]]:
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

    for name, met in results['individual'].items():
        out.append(
            {
                'name': f'indiv::{name}',
                'fpr': _get(met, 'fpr_test', float('inf')),
                'auc': _get(met, 'auc_test', float('nan')),
                'oracle': False,
            }
        )

    for name, pack in results['pair_results_valsel'].items():
        te = pack.get('test_eval', {}) if isinstance(pack, dict) else {}
        out.append(
            {
                'name': f'pair_valsel::{name}',
                'fpr': _get(te, 'fpr', float('inf')),
                'auc': _get(te, 'auc', float('nan')),
                'oracle': False,
            }
        )

    for name, pack in results['pair_results_oracle'].items():
        out.append(
            {
                'name': f'pair_oracle::{name}',
                'fpr': _get(pack, 'fpr', float('inf')),
                'auc': _get(pack, 'auc', float('nan')),
                'oracle': True,
            }
        )

    for k in [
        f'all{int(n_models)}_weighted_raw_valsel',
        f'all{int(n_models)}_weighted_platt_valsel',
        f'all{int(n_models)}_weighted_iso_valsel',
    ]:
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        te = pack.get('test_eval', {}) if isinstance(pack, dict) else {}
        out.append(
            {
                'name': k,
                'fpr': _get(te, 'fpr', float('inf')),
                'auc': _get(te, 'auc', float('nan')),
                'oracle': False,
            }
        )

    for k in [
        f'all{int(n_models)}_weighted_raw_oracle',
        f'all{int(n_models)}_weighted_platt_oracle',
        f'all{int(n_models)}_weighted_iso_oracle',
    ]:
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        out.append(
            {
                'name': k,
                'fpr': _get(pack, 'fpr', float('inf')),
                'auc': _get(pack, 'auc', float('nan')),
                'oracle': True,
            }
        )

    for k in ['meta_raw', 'meta_platt', 'meta_iso']:
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        te = pack.get('test_eval', {}) if isinstance(pack, dict) else {}
        orc = pack.get('oracle_test', {}) if isinstance(pack, dict) else {}
        out.append(
            {
                'name': f'{k}::valsel',
                'fpr': _get(te, 'fpr', float('inf')),
                'auc': _get(te, 'auc', float('nan')),
                'oracle': False,
            }
        )
        out.append(
            {
                'name': f'{k}::oracle',
                'fpr': _get(orc, 'fpr', float('inf')),
                'auc': _get(orc, 'auc', float('nan')),
                'oracle': True,
            }
        )

    for k in ['sparse_linear_stable', 'tiny_mlp_stacker', 'moe_gated_stacker']:
        if k not in results:
            continue
        pack = results.get(k, {}) if isinstance(results, dict) else {}
        te = pack.get('test_eval', {}) if isinstance(pack, dict) else {}
        orc = pack.get('oracle_test', {}) if isinstance(pack, dict) else {}
        out.append(
            {
                'name': f'{k}::valsel',
                'fpr': _get(te, 'fpr', float('inf')),
                'auc': _get(te, 'auc', float('nan')),
                'oracle': False,
            }
        )
        out.append(
            {
                'name': f'{k}::oracle',
                'fpr': _get(orc, 'fpr', float('inf')),
                'auc': _get(orc, 'auc', float('nan')),
                'oracle': True,
            }
        )

    return out

def _load_model_scores(
    model_name: str,
    run_dir: Path,
    file_name: str,
    val_keys: List[str],
    test_keys: List[str],
    y_val_ref: np.ndarray,
    y_test_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    npz_path = run_dir / file_name
    z = base._load_npz(npz_path)

    yv = np.asarray(z["labels_val"], dtype=np.float32)
    yt = np.asarray(z["labels_test"], dtype=np.float32)
    if not np.array_equal(y_val_ref, yv):
        raise RuntimeError(f"Validation labels mismatch: {model_name} ({npz_path})")
    if not np.array_equal(y_test_ref, yt):
        raise RuntimeError(f"Test labels mismatch: {model_name} ({npz_path})")

    s_val = base._pick_score(z, val_keys, "val", ref_len=y_val_ref.size)
    s_test = base._pick_score(z, test_keys, "test", ref_len=y_test_ref.size)
    return s_val, s_test, str(npz_path.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fusion analysis for previous 18 + dualreco frozen family + m2 delta000/delta020"
    )

    # Previous 18-model set
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

    # New m2 delta ablations
    ap.add_argument("--m2_delta000_run_dir", type=str, required=True)
    ap.add_argument("--m2_delta020_run_dir", type=str, required=True)

    # New dualreco-dualview family (use frozen scores)

    ap.add_argument("--m11_dual_run_dir", type=str, required=True)
    ap.add_argument("--m12_dual_run_dir", type=str, required=True)
    ap.add_argument("--m13_dual_run_dir", type=str, required=True)
    ap.add_argument("--m15_dual_low_run_dir", type=str, required=True)
    ap.add_argument("--m15_dual_mid_run_dir", type=str, required=True)
    ap.add_argument("--m15_dual_high_run_dir", type=str, required=True)
    ap.add_argument("--m16_dual_k40_run_dir", type=str, required=True)
    ap.add_argument("--m16_dual_k60_run_dir", type=str, required=True)
    ap.add_argument("--m16_dual_k80_run_dir", type=str, required=True)
    ap.add_argument("--m17_dual_run_dir", type=str, required=True)
    ap.add_argument("--m19_dual_run_dir", type=str, required=True)

    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--weight_step_2", type=float, default=0.01)
    ap.add_argument("--weight_samples_multi", type=int, default=12000)
    ap.add_argument("--weight_samples_multi_sparse", type=int, default=20000)
    ap.add_argument("--weight_sparse_k_grid", type=str, default="2,3,4,5,6,8,10,12")
    ap.add_argument("--pair_grid_step_multi", type=float, default=0.05)
    ap.add_argument("--meta_sel_frac", type=float, default=0.30)
    ap.add_argument("--meta_c_grid", type=str, default="0.05,0.1,0.3,1,3,10,30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_name", type=str, default="fusion_hlt_joint31_analysis.json")

    ap.add_argument("--disable_sparse_stacker", action="store_true")
    ap.add_argument("--disable_tiny_mlp_stacker", action="store_true")
    ap.add_argument("--disable_moe_gated_stacker", action="store_true")

    ap.add_argument("--sparse_l1_grid", type=str, default="0.0,1e-4,3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--sparse_folds", type=int, default=5)
    ap.add_argument("--sparse_stability_seeds", type=str, default="0,1,2")
    ap.add_argument("--sparse_min_freq", type=float, default=0.60)
    ap.add_argument("--sparse_select_threshold", type=float, default=0.01)
    ap.add_argument("--sparse_train_steps", type=int, default=800)
    ap.add_argument("--sparse_lr", type=float, default=0.05)

    ap.add_argument("--mlp_hidden_grid", type=str, default="64,32|32,16|64")
    ap.add_argument("--mlp_alpha_grid", type=str, default="1e-4,3e-4,1e-3")
    ap.add_argument("--mlp_max_iter", type=int, default=260)

    ap.add_argument("--moe_entropy_grid", type=str, default="0.01,0.03")
    ap.add_argument("--moe_l2_grid", type=str, default="1e-4,3e-4")
    ap.add_argument("--moe_temp_grid", type=str, default="1.0,1.3")
    ap.add_argument("--moe_topk_grid", type=str, default="4,8,12")
    ap.add_argument("--moe_steps", type=int, default=220)
    ap.add_argument("--moe_batch_size", type=int, default=2048)
    ap.add_argument("--moe_lr", type=float, default=0.005)
    args = ap.parse_args()

    target_tpr = float(args.target_tpr)

    dir_joint_delta = Path(args.joint_delta_run_dir)
    z2 = base._load_npz(dir_joint_delta / "fusion_scores_val_test.npz")
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

    score_files = {"joint_delta": str((dir_joint_delta / "fusion_scores_val_test.npz").resolve())}

    model_specs = [
        # Previous 18 additions beyond hlt/joint_delta
        ("reco_teacher_s09", Path(args.reco_teacher_s09_run_dir), "stageA_only_scores.npz", ["preds_reco_teacher_val"], ["preds_reco_teacher_test"]),
        ("corrected_s01", Path(args.corrected_s01_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("joint_s01", Path(args.joint_s01_run_dir), "fusion_scores_val_test.npz", ["preds_joint_val"], ["preds_joint_test"]),
        ("concat_corrected", Path(args.concat_run_dir), "concat_teacher_stageA_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),

        ("residual_m7", Path(args.m7_residual_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("direct_residual_m8", Path(args.m8_direct_residual_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("offdrop_low", Path(args.m9_low_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("offdrop_mid", Path(args.m9_mid_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),
        ("offdrop_high", Path(args.m9_high_run_dir), "stageA_residual_scores.npz", ["preds_residual_joint_val", "preds_residual_frozen_val"], ["preds_residual_joint_test", "preds_residual_frozen_test"]),

        ("corrected_k40", Path(args.m4_k40_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("corrected_k60", Path(args.m4_k60_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("corrected_k80", Path(args.m4_k80_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),

        ("antioverlap_m10", Path(args.m10_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("feat_noangle_m11", Path(args.m11_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("feat_noscale_m12", Path(args.m12_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),
        ("feat_coreshape_m13", Path(args.m13_run_dir), "stageA_only_scores.npz", ["preds_corrected_only_val"], ["preds_corrected_only_test"]),

        # New m2 delta ablations
        ("joint_delta000", Path(args.m2_delta000_run_dir), "fusion_scores_val_test.npz", ["preds_joint_val"], ["preds_joint_test"]),
        ("joint_delta020", Path(args.m2_delta020_run_dir), "fusion_scores_val_test.npz", ["preds_joint_val"], ["preds_joint_test"]),

        # New dualreco frozen additions

        ("dual_m11_noangle", Path(args.m11_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m12_noscale", Path(args.m12_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m13_coreshape", Path(args.m13_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m15_offdrop_low", Path(args.m15_dual_low_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m15_offdrop_mid", Path(args.m15_dual_mid_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m15_offdrop_high", Path(args.m15_dual_high_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m16_topk40", Path(args.m16_dual_k40_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m16_topk60", Path(args.m16_dual_k60_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m16_topk80", Path(args.m16_dual_k80_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m17_antioverlap", Path(args.m17_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
        ("dual_m19_basic", Path(args.m19_dual_run_dir), "dualreco_dualview_scores.npz", ["preds_dual_frozen_val", "preds_dualview_frozen_val"], ["preds_dual_frozen_test", "preds_dualview_frozen_test"]),
    ]

    for name, run_dir, file_name, val_keys, test_keys in model_specs:
        s_val, s_test, src = _load_model_scores(
            model_name=name,
            run_dir=run_dir,
            file_name=file_name,
            val_keys=val_keys,
            test_keys=test_keys,
            y_val_ref=y_val,
            y_test_ref=y_test,
        )
        scores_val[name] = s_val
        scores_test[name] = s_test
        score_files[name] = src

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
        "joint_delta000",
        "joint_delta020",
        "dual_m11_noangle",
        "dual_m12_noscale",
        "dual_m13_coreshape",
        "dual_m15_offdrop_low",
        "dual_m15_offdrop_mid",
        "dual_m15_offdrop_high",
        "dual_m16_topk40",
        "dual_m16_topk60",
        "dual_m16_topk80",
        "dual_m17_antioverlap",
        "dual_m19_basic",
    ]

    for n in model_order:
        if n not in scores_val:
            raise KeyError(f"Missing model score: {n}")

    indiv: Dict[str, Dict[str, float]] = {}
    for name in model_order:
        v = base.auc_and_fpr_at_target(y_val, scores_val[name], target_tpr)
        t = base.auc_and_fpr_at_target(y_test, scores_test[name], target_tpr)
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
        ps = po["w_a"] * scores_test["hlt"] + po["w_b"] * scores_test[other]
        pa = base.auc_and_fpr_at_target(y_test, ps, target_tpr)
        po["auc"] = float(pa["auc"])
        po["fpr_at_target_tpr_exact"] = float(pa["fpr_at_target_tpr"])
        pair_results_oracle[key] = po

    sparse_k_grid = _parse_int_grid(args.weight_sparse_k_grid, [2, 3, 4, 5, 6, 8, 10, 12])
    weight_candidates = base.generate_weight_candidates(
        n_models=len(model_order),
        n_random=int(args.weight_samples_multi),
        seed=int(args.seed),
        include_pair_grid=True,
        pair_step=float(args.pair_grid_step_multi),
        n_sparse_random=int(args.weight_samples_multi_sparse),
        sparse_k_grid=sparse_k_grid,
    )
    print(f"Weight candidates total: {int(weight_candidates.shape[0])} (dense={int(args.weight_samples_multi)}, sparse={int(args.weight_samples_multi_sparse)})")

    mat_val = np.vstack([scores_val[n] for n in model_order])
    mat_test = np.vstack([scores_test[n] for n in model_order])

    all_weighted_raw_valsel = base.select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_val,
        y_test=y_test,
        score_mat_test=mat_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )
    all_weighted_raw_oracle = base.search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    cal_platt_meta: Dict[str, Dict[str, float]] = {}
    cal_iso_meta: Dict[str, Dict[str, float]] = {}
    scores_platt_val: Dict[str, np.ndarray] = {}
    scores_platt_test: Dict[str, np.ndarray] = {}
    scores_iso_val: Dict[str, np.ndarray] = {}
    scores_iso_test: Dict[str, np.ndarray] = {}

    for name in model_order:
        pv, pt, pm = base.calibrate_platt(y_val, scores_val[name], scores_test[name])
        iv, it, im = base.calibrate_isotonic(y_val, scores_val[name], scores_test[name])
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

    all_weighted_platt_valsel = base.select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_platt_val,
        y_test=y_test,
        score_mat_test=mat_platt_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )
    all_weighted_platt_oracle = base.search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_platt_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    all_weighted_iso_valsel = base.select_weighted_combo_multi_on_val_eval_test(
        y_val=y_val,
        score_mat_val=mat_iso_val,
        y_test=y_test,
        score_mat_test=mat_iso_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )
    all_weighted_iso_oracle = base.search_best_weighted_combo_multi_at_tpr(
        labels=y_test,
        score_mat=mat_iso_test,
        model_names=model_order,
        target_tpr=target_tpr,
        weight_candidates=weight_candidates,
    )

    c_grid = [float(x.strip()) for x in str(args.meta_c_grid).split(",") if x.strip()]
    meta_raw = base.train_select_meta_fuser(
        X_val=base.build_meta_features(model_order, scores_val),
        y_val=y_val,
        X_test=base.build_meta_features(model_order, scores_test),
        y_test=y_test,
        target_tpr=target_tpr,
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
        seed=int(args.seed),
    )
    meta_platt = base.train_select_meta_fuser(
        X_val=base.build_meta_features(model_order, scores_platt_val),
        y_val=y_val,
        X_test=base.build_meta_features(model_order, scores_platt_test),
        y_test=y_test,
        target_tpr=target_tpr,
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
        seed=int(args.seed),
    )

    meta_iso = base.train_select_meta_fuser(
        X_val=base.build_meta_features(model_order, scores_iso_val),
        y_val=y_val,
        X_test=base.build_meta_features(model_order, scores_iso_test),
        y_test=y_test,
        target_tpr=target_tpr,
        sel_frac=float(args.meta_sel_frac),
        c_grid=c_grid,
        seed=int(args.seed),
    )

    sparse_result = None
    tiny_mlp_result = None
    moe_gated_result = None

    if not bool(args.disable_sparse_stacker):
        sparse_result = _train_sparse_stable_stacker(
            score_mat_val=mat_val,
            y_val=y_val,
            score_mat_test=mat_test,
            y_test=y_test,
            model_order=model_order,
            target_tpr=target_tpr,
            sel_frac=float(args.meta_sel_frac),
            seed=int(args.seed),
            l1_grid=_parse_float_grid(args.sparse_l1_grid, [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
            folds=int(max(2, args.sparse_folds)),
            stability_seeds=_parse_int_grid(args.sparse_stability_seeds, [0, 1, 2]),
            min_freq=float(args.sparse_min_freq),
            select_threshold=float(args.sparse_select_threshold),
            train_steps=int(args.sparse_train_steps),
            train_lr=float(args.sparse_lr),
        )

    if not bool(args.disable_tiny_mlp_stacker):
        tiny_mlp_result = _train_tiny_mlp_stacker(
            score_mat_val=mat_val,
            y_val=y_val,
            score_mat_test=mat_test,
            y_test=y_test,
            target_tpr=target_tpr,
            sel_frac=float(args.meta_sel_frac),
            seed=int(args.seed),
            hidden_grid=_parse_hidden_grid(args.mlp_hidden_grid, [(64, 32), (32, 16), (64,)]),
            alpha_grid=_parse_float_grid(args.mlp_alpha_grid, [1e-4, 3e-4, 1e-3]),
            max_iter=int(args.mlp_max_iter),
        )

    if not bool(args.disable_moe_gated_stacker):
        moe_gated_result = _train_moe_gated_stacker(
            score_mat_val=mat_val,
            y_val=y_val,
            score_mat_test=mat_test,
            y_test=y_test,
            target_tpr=target_tpr,
            sel_frac=float(args.meta_sel_frac),
            seed=int(args.seed),
            entropy_grid=_parse_float_grid(args.moe_entropy_grid, [0.01, 0.03]),
            l2_grid=_parse_float_grid(args.moe_l2_grid, [1e-4, 3e-4]),
            temp_grid=_parse_float_grid(args.moe_temp_grid, [1.0, 1.3]),
            topk_grid=_parse_int_grid(args.moe_topk_grid, [4, 8, 12]),
            steps=int(args.moe_steps),
            batch_size=int(args.moe_batch_size),
            lr=float(args.moe_lr),
        )

    n_models = len(model_order)

    run_dirs = {
        k: str(v)
        for k, v in vars(args).items()
        if k.endswith("_run_dir")
    }
    run_dirs["score_files"] = score_files

    results = {
        "config": {
            "target_tpr": target_tpr,
            "weight_step_2": float(args.weight_step_2),
            "weight_samples_multi": int(args.weight_samples_multi),
            "weight_samples_multi_sparse": int(args.weight_samples_multi_sparse),
            "weight_sparse_k_grid": sparse_k_grid,
            "pair_grid_step_multi": float(args.pair_grid_step_multi),
            "meta_sel_frac": float(args.meta_sel_frac),
            "meta_c_grid": c_grid,
            "seed": int(args.seed),
            "n_models": int(n_models),
        },
        "run_dirs": run_dirs,
        "models_order": model_order,
        "individual": indiv,
        "overlap_test": overlap_test,
        "pair_results_valsel": pair_results_valsel,
        "pair_results_oracle": pair_results_oracle,
        f"all{int(n_models)}_weighted_raw_valsel": all_weighted_raw_valsel,
        f"all{int(n_models)}_weighted_raw_oracle": all_weighted_raw_oracle,
        f"all{int(n_models)}_weighted_platt_valsel": all_weighted_platt_valsel,
        f"all{int(n_models)}_weighted_platt_oracle": all_weighted_platt_oracle,
        f"all{int(n_models)}_weighted_iso_valsel": all_weighted_iso_valsel,
        f"all{int(n_models)}_weighted_iso_oracle": all_weighted_iso_oracle,
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


    if sparse_result is not None:
        results['sparse_linear_stable'] = sparse_result
    if tiny_mlp_result is not None:
        results['tiny_mlp_stacker'] = tiny_mlp_result
    if moe_gated_result is not None:
        results['moe_gated_stacker'] = moe_gated_result

    results['config']['advanced_stackers'] = {
        'sparse_enabled': not bool(args.disable_sparse_stacker),
        'tiny_mlp_enabled': not bool(args.disable_tiny_mlp_stacker),
        'moe_enabled': not bool(args.disable_moe_gated_stacker),
        'sparse_l1_grid': _parse_float_grid(args.sparse_l1_grid, [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
        'mlp_hidden_grid': [list(h) for h in _parse_hidden_grid(args.mlp_hidden_grid, [(64, 32), (32, 16), (64,)])],
        'mlp_alpha_grid': _parse_float_grid(args.mlp_alpha_grid, [1e-4, 3e-4, 1e-3]),
        'moe_entropy_grid': _parse_float_grid(args.moe_entropy_grid, [0.01, 0.03]),
        'moe_l2_grid': _parse_float_grid(args.moe_l2_grid, [1e-4, 3e-4]),
        'moe_temp_grid': _parse_float_grid(args.moe_temp_grid, [1.0, 1.3]),
        'moe_topk_grid': _parse_int_grid(args.moe_topk_grid, [4, 8, 12]),
    }

    all_candidates = _collect_candidates(results, n_models=n_models)
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

    print(f"Models fused: {n_models}")
    print(f"Saved fusion analysis to: {out_path}")


if __name__ == "__main__":
    main()

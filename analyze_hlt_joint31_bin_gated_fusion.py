#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bin-gated fusion over existing 31-model score artifacts.

Core idea:
1) Build a strong global blend from a small candidate set (greedy, fit split).
2) Add piecewise/bin-local blend updates on top of that global blend.
3) Select operating threshold on held-out calibration split.
4) Report held-out test metrics.

No model retraining; this is score-level fusion only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# model_name -> (run_dir_key, file_name, val_keys, test_keys)
MODEL_SCORE_SPECS: Dict[str, Tuple[str, str, List[str], List[str]]] = {
    "reco_teacher_s09": (
        "reco_teacher_s09_run_dir",
        "stageA_only_scores.npz",
        ["preds_reco_teacher_val"],
        ["preds_reco_teacher_test"],
    ),
    "corrected_s01": (
        "corrected_s01_run_dir",
        "stageA_only_scores.npz",
        ["preds_corrected_only_val"],
        ["preds_corrected_only_test"],
    ),
    "joint_s01": (
        "joint_s01_run_dir",
        "fusion_scores_val_test.npz",
        ["preds_joint_val"],
        ["preds_joint_test"],
    ),
    "concat_corrected": (
        "concat_run_dir",
        "concat_teacher_stageA_scores.npz",
        ["preds_corrected_only_val"],
        ["preds_corrected_only_test"],
    ),
    "residual_m7": (
        "m7_residual_run_dir",
        "stageA_residual_scores.npz",
        ["preds_residual_joint_val", "preds_residual_frozen_val"],
        ["preds_residual_joint_test", "preds_residual_frozen_test"],
    ),
    "direct_residual_m8": (
        "m8_direct_residual_run_dir",
        "stageA_residual_scores.npz",
        ["preds_residual_joint_val", "preds_residual_frozen_val"],
        ["preds_residual_joint_test", "preds_residual_frozen_test"],
    ),
    "offdrop_low": (
        "m9_low_run_dir",
        "stageA_residual_scores.npz",
        ["preds_residual_joint_val", "preds_residual_frozen_val"],
        ["preds_residual_joint_test", "preds_residual_frozen_test"],
    ),
    "offdrop_mid": (
        "m9_mid_run_dir",
        "stageA_residual_scores.npz",
        ["preds_residual_joint_val", "preds_residual_frozen_val"],
        ["preds_residual_joint_test", "preds_residual_frozen_test"],
    ),
    "offdrop_high": (
        "m9_high_run_dir",
        "stageA_residual_scores.npz",
        ["preds_residual_joint_val", "preds_residual_frozen_val"],
        ["preds_residual_joint_test", "preds_residual_frozen_test"],
    ),
    "corrected_k40": (
        "m4_k40_run_dir",
        "stageA_only_scores.npz",
        ["preds_corrected_only_val"],
        ["preds_corrected_only_test"],
    ),
    "corrected_k60": (
        "m4_k60_run_dir",
        "stageA_only_scores.npz",
        ["preds_corrected_only_val"],
        ["preds_corrected_only_test"],
    ),
    "corrected_k80": (
        "m4_k80_run_dir",
        "stageA_only_scores.npz",
        ["preds_corrected_only_val"],
        ["preds_corrected_only_test"],
    ),
    "antioverlap_m10": (
        "m10_run_dir",
        "stageA_only_scores.npz",
        ["preds_corrected_only_val"],
        ["preds_corrected_only_test"],
    ),
    "feat_noangle_m11": (
        "m11_run_dir",
        "stageA_only_scores.npz",
        ["preds_corrected_only_val"],
        ["preds_corrected_only_test"],
    ),
    "feat_noscale_m12": (
        "m12_run_dir",
        "stageA_only_scores.npz",
        ["preds_corrected_only_val"],
        ["preds_corrected_only_test"],
    ),
    "feat_coreshape_m13": (
        "m13_run_dir",
        "stageA_only_scores.npz",
        ["preds_corrected_only_val"],
        ["preds_corrected_only_test"],
    ),
    "joint_delta000": (
        "m2_delta000_run_dir",
        "fusion_scores_val_test.npz",
        ["preds_joint_val"],
        ["preds_joint_test"],
    ),
    "joint_delta020": (
        "m2_delta020_run_dir",
        "fusion_scores_val_test.npz",
        ["preds_joint_val"],
        ["preds_joint_test"],
    ),
    "dual_m11_noangle": (
        "m11_dual_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m12_noscale": (
        "m12_dual_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m13_coreshape": (
        "m13_dual_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m15_offdrop_low": (
        "m15_dual_low_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m15_offdrop_mid": (
        "m15_dual_mid_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m15_offdrop_high": (
        "m15_dual_high_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m16_topk40": (
        "m16_dual_k40_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m16_topk60": (
        "m16_dual_k60_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m16_topk80": (
        "m16_dual_k80_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m17_antioverlap": (
        "m17_dual_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
    "dual_m19_basic": (
        "m19_dual_run_dir",
        "dualreco_dualview_scores.npz",
        ["preds_dual_frozen_val", "preds_dualview_frozen_val"],
        ["preds_dual_frozen_test", "preds_dualview_frozen_test"],
    ),
}


def _parse_float_list(spec: str, default: List[float]) -> List[float]:
    out: List[float] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            continue
    if not out:
        out = [float(x) for x in default]
    return out


def _parse_str_list(spec: str) -> List[str]:
    out: List[str] = []
    for tok in str(spec).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _safe_path(raw: str, repo_root: Path) -> Path:
    p = Path(str(raw)).expanduser()
    cands: List[Path] = []
    if p.exists():
        return p.resolve()

    s = str(p)
    if "/checkpoints/" in s:
        tail = s.split("/checkpoints/", 1)[1]
        cands.append((repo_root / "checkpoints" / tail).resolve())
        cands.append((repo_root / "download_checkpoints" / tail).resolve())
    if "/download_checkpoints/" in s:
        tail = s.split("/download_checkpoints/", 1)[1]
        cands.append((repo_root / "download_checkpoints" / tail).resolve())
        cands.append((repo_root / "checkpoints" / tail).resolve())

    if not p.is_absolute():
        cands.append((repo_root / p).resolve())
        cands.append((repo_root / "download_checkpoints" / p).resolve())
        if s.startswith("checkpoints/"):
            tail = s[len("checkpoints/") :]
            cands.append((repo_root / "download_checkpoints" / tail).resolve())

    seen = set()
    for c in cands:
        cs = str(c)
        if cs in seen:
            continue
        seen.add(cs)
        if c.exists():
            return c

    if cands:
        return cands[0]
    return p.resolve()


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


def _first_existing_key(npz: np.lib.npyio.NpzFile, names: Iterable[str]) -> str:
    for n in names:
        if n in npz:
            return n
    raise KeyError(f"None of keys found in npz: {list(names)}")


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
    return {"tp": tp, "fp": fp, "n_pos": n_pos, "n_neg": n_neg, "tpr": tpr, "fpr": fpr}


def eval_from_ref(
    y_ref: np.ndarray,
    s_ref: np.ndarray,
    y_eval: np.ndarray,
    s_eval: np.ndarray,
    target_tpr: float,
) -> Dict[str, float]:
    thr = threshold_for_target_tpr(y_ref, s_ref, target_tpr)
    rr = rates_from_threshold(y_ref, s_ref, thr)
    re = rates_from_threshold(y_eval, s_eval, thr)
    return {
        "threshold_from_ref": float(thr),
        "auc_ref": float(roc_auc_score(y_ref, s_ref)) if np.unique(y_ref).size > 1 else float("nan"),
        "auc_eval": float(roc_auc_score(y_eval, s_eval)) if np.unique(y_eval).size > 1 else float("nan"),
        "fpr_ref": float(rr["fpr"]),
        "tpr_ref": float(rr["tpr"]),
        "fpr_eval": float(re["fpr"]),
        "tpr_eval": float(re["tpr"]),
        "tp_eval": float(re["tp"]),
        "fp_eval": float(re["fp"]),
    }


def calibrate_scores(
    y_fit: np.ndarray,
    s_fit: np.ndarray,
    s_cal: np.ndarray,
    s_test: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = str(mode).lower()
    if mode == "raw":
        return s_fit.copy(), s_cal.copy(), s_test.copy()
    if np.unique(y_fit.astype(np.int64)).size < 2:
        return s_fit.copy(), s_cal.copy(), s_test.copy()
    if mode == "iso":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s_fit.astype(np.float64), y_fit.astype(np.float64))
        return (
            np.asarray(iso.transform(s_fit.astype(np.float64)), dtype=np.float64),
            np.asarray(iso.transform(s_cal.astype(np.float64)), dtype=np.float64),
            np.asarray(iso.transform(s_test.astype(np.float64)), dtype=np.float64),
        )
    if mode == "platt":
        lr = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced")
        lr.fit(s_fit.reshape(-1, 1).astype(np.float64), y_fit.astype(np.int64))
        return (
            lr.predict_proba(s_fit.reshape(-1, 1).astype(np.float64))[:, 1].astype(np.float64),
            lr.predict_proba(s_cal.reshape(-1, 1).astype(np.float64))[:, 1].astype(np.float64),
            lr.predict_proba(s_test.reshape(-1, 1).astype(np.float64))[:, 1].astype(np.float64),
        )
    raise ValueError(f"Unknown calibration mode: {mode}")


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"Missing score file: {path}")
    return np.load(path)


def _load_required_scores(
    fusion_json_path: Path,
    required_models: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]:
    repo_root = Path(__file__).resolve().parent
    fusion = json.loads(fusion_json_path.read_text())
    run_dirs_raw = dict(fusion.get("run_dirs", {}))
    score_files_raw = dict(run_dirs_raw.get("score_files", {})) if isinstance(run_dirs_raw.get("score_files", {}), dict) else {}

    resolved_run_dirs: Dict[str, Path] = {}
    for k, v in run_dirs_raw.items():
        if k == "score_files":
            continue
        if isinstance(v, str):
            resolved_run_dirs[k] = _safe_path(v, repo_root)

    joint_score_path = score_files_raw.get("joint_delta")
    if not isinstance(joint_score_path, str):
        joint_dir = resolved_run_dirs.get("joint_delta_run_dir")
        if joint_dir is None:
            raise KeyError("Could not resolve joint_delta score path from fusion json")
        joint_score_path = str(joint_dir / "fusion_scores_val_test.npz")
    joint_npz_path = _safe_path(joint_score_path, repo_root)
    z2 = _load_npz(joint_npz_path)
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
    used_paths: Dict[str, str] = {"joint_delta": str(joint_npz_path)}

    if "preds_teacher_val" in z2 and "preds_teacher_test" in z2:
        scores_val["teacher"] = np.asarray(z2["preds_teacher_val"], dtype=np.float64)
        scores_test["teacher"] = np.asarray(z2["preds_teacher_test"], dtype=np.float64)
        used_paths["teacher"] = str(joint_npz_path)

    for name in sorted(set(required_models)):
        if name in scores_val:
            continue
        if name not in MODEL_SCORE_SPECS:
            raise KeyError(f"Unsupported model requested: {name}")
        run_key, file_name, val_keys, test_keys = MODEL_SCORE_SPECS[name]
        sf = score_files_raw.get(name)
        if isinstance(sf, str):
            npz_path = _safe_path(sf, repo_root)
        else:
            rd = resolved_run_dirs.get(run_key)
            if rd is None:
                raise KeyError(f"Missing run dir for {name}: key={run_key}")
            npz_path = _safe_path(str(rd / file_name), repo_root)

        z = _load_npz(npz_path)
        yv = np.asarray(z["labels_val"], dtype=np.float32)
        yt = np.asarray(z["labels_test"], dtype=np.float32)
        if not np.array_equal(y_val, yv):
            raise RuntimeError(f"Validation labels mismatch for model={name} npz={npz_path}")
        if not np.array_equal(y_test, yt):
            raise RuntimeError(f"Test labels mismatch for model={name} npz={npz_path}")
        k_val = _first_existing_key(z, val_keys)
        k_test = _first_existing_key(z, test_keys)
        scores_val[name] = np.asarray(z[k_val], dtype=np.float64)
        scores_test[name] = np.asarray(z[k_test], dtype=np.float64)
        used_paths[name] = str(npz_path)

    return y_val, y_test, scores_val, scores_test, used_paths


def _score_band(scores: np.ndarray, edges: List[float]) -> np.ndarray:
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("score band edges must have at least two values")
    if not np.all(np.diff(e) > 0.0):
        raise ValueError("score band edges must be strictly increasing")
    return np.digitize(scores, e[1:-1], right=False).astype(np.int64)


def _dist_band(dist: np.ndarray, near_cut: float, mid_lo: float, mid_hi: float) -> np.ndarray:
    b = np.full(dist.shape, 2, dtype=np.int64)  # other
    b[dist < float(near_cut)] = 0  # near
    mid = (dist >= float(mid_lo)) & (dist < float(mid_hi))
    b[mid] = 1  # mid band
    return b


def _make_bin_ids(joint_score: np.ndarray, dist_to_joint_thr: np.ndarray, score_edges: List[float], near_cut: float, mid_lo: float, mid_hi: float) -> np.ndarray:
    jb = _score_band(joint_score, score_edges)  # 0..(len(edges)-2)
    db = _dist_band(dist_to_joint_thr, near_cut, mid_lo, mid_hi)  # 0..2
    return (jb * 3 + db).astype(np.int64)


def _bin_label(bin_id: int, score_edges: List[float]) -> str:
    n_j = len(score_edges) - 1
    j = int(bin_id // 3)
    d = int(bin_id % 3)
    if j < 0 or j >= n_j:
        return f"bin_{bin_id}"
    jlab = f"[{score_edges[j]:.3g},{score_edges[j+1]:.3g})"
    dlab = ["near", "mid", "other"][d] if 0 <= d <= 2 else f"d{d}"
    return f"joint{jlab}|dist={dlab}"


def _objective(y: np.ndarray, s: np.ndarray, target_tpr: float) -> Tuple[float, float]:
    thr = threshold_for_target_tpr(y, s, target_tpr)
    rr = rates_from_threshold(y, s, thr)
    auc = float(roc_auc_score(y, s)) if np.unique(y).size > 1 else float("nan")
    return float(rr["fpr"]), auc


def _greedy_global_blend(
    y_fit: np.ndarray,
    s_fit_map: Dict[str, np.ndarray],
    s_cal_map: Dict[str, np.ndarray],
    s_test_map: Dict[str, np.ndarray],
    anchor_model: str,
    candidates: List[str],
    target_tpr: float,
    max_add: int,
    w_step: float,
    min_improve: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
    s_fit = np.asarray(s_fit_map[anchor_model], dtype=np.float64).copy()
    s_cal = np.asarray(s_cal_map[anchor_model], dtype=np.float64).copy()
    s_test = np.asarray(s_test_map[anchor_model], dtype=np.float64).copy()
    rows: List[Dict[str, object]] = []
    selected = [anchor_model]
    fpr0, auc0 = _objective(y_fit, s_fit, target_tpr)
    rows.append(
        {
            "stage": "global",
            "step": 0,
            "action": "init_anchor",
            "added_model": anchor_model,
            "blend_w": 0.0,
            "fpr_fit": fpr0,
            "auc_fit": auc0,
            "selected_models": ",".join(selected),
        }
    )
    remaining = [m for m in candidates if m != anchor_model]
    for step in range(1, int(max_add) + 1):
        cur_fpr, cur_auc = _objective(y_fit, s_fit, target_tpr)
        best = None
        for cand in remaining:
            sv_c = s_fit_map[cand]
            sc_c = s_cal_map[cand]
            st_c = s_test_map[cand]
            for w in np.arange(float(w_step), 1.0, float(w_step), dtype=np.float64):
                sf = (1.0 - w) * s_fit + w * sv_c
                fpr, auc = _objective(y_fit, sf, target_tpr)
                key = (fpr, -auc)
                if best is None or key < best["key"]:
                    best = {
                        "cand": cand,
                        "w": float(w),
                        "fpr": float(fpr),
                        "auc": float(auc),
                        "sf": sf,
                        "sc": (1.0 - w) * s_cal + w * sc_c,
                        "st": (1.0 - w) * s_test + w * st_c,
                        "key": key,
                    }
        if best is None:
            break
        improve = cur_fpr - float(best["fpr"])
        if improve <= float(min_improve):
            rows.append(
                {
                    "stage": "global",
                    "step": step,
                    "action": "stop_no_improve",
                    "added_model": "",
                    "blend_w": 0.0,
                    "fpr_fit": cur_fpr,
                    "auc_fit": cur_auc,
                    "selected_models": ",".join(selected),
                }
            )
            break

        s_fit = np.asarray(best["sf"], dtype=np.float64)
        s_cal = np.asarray(best["sc"], dtype=np.float64)
        s_test = np.asarray(best["st"], dtype=np.float64)
        selected.append(str(best["cand"]))
        remaining = [m for m in remaining if m != best["cand"]]
        rows.append(
            {
                "stage": "global",
                "step": step,
                "action": "add_model",
                "added_model": str(best["cand"]),
                "blend_w": float(best["w"]),
                "fpr_fit": float(best["fpr"]),
                "auc_fit": float(best["auc"]),
                "fpr_fit_prev": float(cur_fpr),
                "fpr_fit_gain": float(improve),
                "selected_models": ",".join(selected),
            }
        )
        if not remaining:
            break
    return s_fit, s_cal, s_test, rows


def _binwise_updates(
    y_fit: np.ndarray,
    s_fit_init: np.ndarray,
    s_cal_init: np.ndarray,
    s_test_init: np.ndarray,
    s_fit_map: Dict[str, np.ndarray],
    s_cal_map: Dict[str, np.ndarray],
    s_test_map: Dict[str, np.ndarray],
    candidates: List[str],
    target_tpr: float,
    bin_fit: np.ndarray,
    bin_cal: np.ndarray,
    bin_test: np.ndarray,
    score_edges: List[float],
    min_bin_fit: int,
    bin_max_add: int,
    w_step: float,
    min_bin_improve: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
    s_fit = np.asarray(s_fit_init, dtype=np.float64).copy()
    s_cal = np.asarray(s_cal_init, dtype=np.float64).copy()
    s_test = np.asarray(s_test_init, dtype=np.float64).copy()
    logs: List[Dict[str, object]] = []

    unique_bins = sorted(int(x) for x in np.unique(bin_fit))
    for b in unique_bins:
        idx_f = np.where(bin_fit == b)[0]
        idx_c = np.where(bin_cal == b)[0]
        idx_t = np.where(bin_test == b)[0]
        n_f = int(idx_f.size)
        n_c = int(idx_c.size)
        n_t = int(idx_t.size)
        if n_f < int(min_bin_fit):
            logs.append(
                {
                    "stage": "bin",
                    "bin_id": b,
                    "bin_label": _bin_label(b, score_edges),
                    "action": "skip_small_bin",
                    "n_fit": n_f,
                    "n_cal": n_c,
                    "n_test": n_t,
                }
            )
            continue

        for step in range(1, int(bin_max_add) + 1):
            cur_fpr, cur_auc = _objective(y_fit, s_fit, target_tpr)
            required = float(min_bin_improve) * float(np.sqrt(max(1.0, float(min_bin_fit) / float(max(n_f, 1)))))
            best = None
            for cand in candidates:
                sv_c = s_fit_map[cand]
                sc_c = s_cal_map[cand]
                st_c = s_test_map[cand]
                for w in np.arange(float(w_step), 1.0, float(w_step), dtype=np.float64):
                    sf = s_fit.copy()
                    sc = s_cal.copy()
                    st = s_test.copy()
                    sf[idx_f] = (1.0 - w) * sf[idx_f] + w * sv_c[idx_f]
                    if idx_c.size > 0:
                        sc[idx_c] = (1.0 - w) * sc[idx_c] + w * sc_c[idx_c]
                    if idx_t.size > 0:
                        st[idx_t] = (1.0 - w) * st[idx_t] + w * st_c[idx_t]
                    fpr, auc = _objective(y_fit, sf, target_tpr)
                    improve = cur_fpr - float(fpr)
                    key = (fpr, -auc)
                    if best is None or key < best["key"]:
                        best = {
                            "cand": cand,
                            "w": float(w),
                            "fpr": float(fpr),
                            "auc": float(auc),
                            "improve": float(improve),
                            "required": float(required),
                            "sf": sf,
                            "sc": sc,
                            "st": st,
                            "key": key,
                        }
            if best is None:
                break
            if float(best["improve"]) <= float(best["required"]):
                logs.append(
                    {
                        "stage": "bin",
                        "bin_id": b,
                        "bin_label": _bin_label(b, score_edges),
                        "step": step,
                        "action": "stop_no_improve",
                        "n_fit": n_f,
                        "n_cal": n_c,
                        "n_test": n_t,
                        "fpr_fit": cur_fpr,
                        "auc_fit": cur_auc,
                        "required_gain": float(best["required"]),
                    }
                )
                break
            s_fit = np.asarray(best["sf"], dtype=np.float64)
            s_cal = np.asarray(best["sc"], dtype=np.float64)
            s_test = np.asarray(best["st"], dtype=np.float64)
            logs.append(
                {
                    "stage": "bin",
                    "bin_id": b,
                    "bin_label": _bin_label(b, score_edges),
                    "step": step,
                    "action": "add_model",
                    "added_model": str(best["cand"]),
                    "blend_w": float(best["w"]),
                    "n_fit": n_f,
                    "n_cal": n_c,
                    "n_test": n_t,
                    "fpr_fit_prev": cur_fpr,
                    "fpr_fit": float(best["fpr"]),
                    "fpr_fit_gain": float(best["improve"]),
                    "required_gain": float(best["required"]),
                    "auc_fit": float(best["auc"]),
                }
            )
    return s_fit, s_cal, s_test, logs


def _default_candidates_for_tpr(tpr: float) -> List[str]:
    if tpr >= 0.45:
        return [
            "joint_delta",
            "dual_m15_offdrop_mid",
            "concat_corrected",
            "hlt",
            "dual_m17_antioverlap",
            "dual_m16_topk60",
            "offdrop_high",
        ]
    return [
        "joint_delta",
        "joint_s01",
        "corrected_s01",
        "offdrop_mid",
        "dual_m12_noscale",
        "dual_m16_topk60",
        "dual_m15_offdrop_high",
        "joint_delta020",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Bin-gated fusion on 31-model score artifacts")
    ap.add_argument("--fusion_json", type=str, required=True)
    ap.add_argument("--target_tprs", type=str, default="0.50,0.30")
    ap.add_argument("--anchor_model", type=str, default="joint_delta")
    ap.add_argument("--candidate_models_all", type=str, default="")
    ap.add_argument("--candidate_models_tpr50", type=str, default="")
    ap.add_argument("--candidate_models_tpr30", type=str, default="")
    ap.add_argument("--selection_mode", type=str, default="split", choices=["split", "valsel"])
    ap.add_argument("--router_cal_frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", type=str, default="iso", choices=["raw", "iso", "platt"])
    ap.add_argument("--score_band_edges", type=str, default="0.0,0.8,0.9,1.0")
    ap.add_argument("--dist_near_cut", type=float, default=0.0384)
    ap.add_argument("--dist_mid_low", type=float, default=0.06285)
    ap.add_argument("--dist_mid_high", type=float, default=0.07386)
    ap.add_argument("--global_max_add", type=int, default=6)
    ap.add_argument("--bin_max_add", type=int, default=3)
    ap.add_argument("--w_step", type=float, default=0.01)
    ap.add_argument("--min_bin_fit", type=int, default=1200)
    ap.add_argument("--min_global_improve", type=float, default=1e-6)
    ap.add_argument("--min_bin_improve", type=float, default=5e-6)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    args = ap.parse_args()

    fusion_json = Path(args.fusion_json).expanduser().resolve()
    if not fusion_json.exists():
        raise FileNotFoundError(f"fusion_json not found: {fusion_json}")

    target_tprs = _parse_float_list(args.target_tprs, [0.50, 0.30])
    target_tprs = [float(x) for x in target_tprs]
    score_edges = _parse_float_list(args.score_band_edges, [0.0, 0.8, 0.9, 1.0])
    if len(score_edges) < 2:
        raise ValueError("score_band_edges must have at least two values")
    if not np.all(np.diff(np.asarray(score_edges, dtype=np.float64)) > 0.0):
        raise ValueError("score_band_edges must be strictly increasing")

    # Resolve candidate sets per TPR first so we only load what we need.
    c_all = _parse_str_list(args.candidate_models_all)
    c_50 = _parse_str_list(args.candidate_models_tpr50)
    c_30 = _parse_str_list(args.candidate_models_tpr30)
    candidates_by_tpr: Dict[float, List[str]] = {}
    for tpr in target_tprs:
        if c_all:
            cands = c_all
        elif tpr >= 0.45 and c_50:
            cands = c_50
        elif tpr < 0.45 and c_30:
            cands = c_30
        else:
            cands = _default_candidates_for_tpr(tpr)
        if args.anchor_model not in cands:
            cands = [args.anchor_model] + list(cands)
        # Keep stable unique order.
        seen = set()
        dedup: List[str] = []
        for m in cands:
            if m in seen:
                continue
            seen.add(m)
            dedup.append(m)
        candidates_by_tpr[float(tpr)] = dedup

    required_models = sorted(set(["hlt", args.anchor_model] + [m for v in candidates_by_tpr.values() for m in v]))
    y_val, y_test, scores_val_raw, scores_test_raw, score_paths = _load_required_scores(fusion_json, required_models)
    if args.anchor_model not in scores_val_raw:
        raise KeyError(f"anchor_model not available: {args.anchor_model}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (fusion_json.parent / "bin_gated_fusion_31")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validation partition for fitting vs threshold selection.
    idx = np.arange(y_val.shape[0], dtype=np.int64)
    yv_int = (y_val > 0.5).astype(np.int64)
    if str(args.selection_mode).lower() == "valsel":
        idx_fit = idx.copy()
        idx_ref = idx.copy()
    else:
        idx_fit, idx_ref = train_test_split(
            idx,
            test_size=float(np.clip(args.router_cal_frac, 0.1, 0.8)),
            random_state=int(args.seed),
            stratify=yv_int,
        )
    y_fit = y_val[idx_fit].astype(np.float32)
    y_ref = y_val[idx_ref].astype(np.float32)
    y_te = y_test.astype(np.float32)

    update_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    score_dump: Dict[str, np.ndarray] = {}
    report_tpr: Dict[str, object] = {}

    for tpr in target_tprs:
        cands = [m for m in candidates_by_tpr[float(tpr)] if m in scores_val_raw]
        if args.anchor_model not in cands:
            cands = [args.anchor_model] + cands
        cands = [m for i, m in enumerate(cands) if m not in cands[:i]]
        if len(cands) < 2:
            raise RuntimeError(f"Need at least anchor + one candidate for tpr={tpr}")

        # Build fit/ref/test maps and calibrated variants for candidate optimization.
        fit_map_raw = {m: np.asarray(scores_val_raw[m][idx_fit], dtype=np.float64) for m in cands}
        ref_map_raw = {m: np.asarray(scores_val_raw[m][idx_ref], dtype=np.float64) for m in cands}
        test_map_raw = {m: np.asarray(scores_test_raw[m], dtype=np.float64) for m in cands}

        fit_map = {}
        cal_map = {}
        test_map = {}
        for m in cands:
            sf, sr, st = calibrate_scores(
                y_fit=y_fit,
                s_fit=fit_map_raw[m],
                s_cal=ref_map_raw[m],
                s_test=test_map_raw[m],
                mode=str(args.calibration),
            )
            fit_map[m] = sf
            cal_map[m] = sr
            test_map[m] = st

        # 1) Global greedy blend.
        s_fit_g, s_cal_g, s_test_g, rows_g = _greedy_global_blend(
            y_fit=y_fit,
            s_fit_map=fit_map,
            s_cal_map=cal_map,
            s_test_map=test_map,
            anchor_model=str(args.anchor_model),
            candidates=cands,
            target_tpr=float(tpr),
            max_add=int(args.global_max_add),
            w_step=float(args.w_step),
            min_improve=float(args.min_global_improve),
        )
        for r in rows_g:
            rr = dict(r)
            rr["target_tpr"] = float(tpr)
            update_rows.append(rr)

        # 2) Build bins from anchor raw score + distance-to-anchor-threshold.
        thr_anchor_fit = threshold_for_target_tpr(y_fit, fit_map_raw[str(args.anchor_model)], float(tpr))
        dist_fit = np.abs(fit_map_raw[str(args.anchor_model)] - float(thr_anchor_fit))
        dist_cal = np.abs(ref_map_raw[str(args.anchor_model)] - float(thr_anchor_fit))
        dist_test = np.abs(test_map_raw[str(args.anchor_model)] - float(thr_anchor_fit))
        bin_fit = _make_bin_ids(
            joint_score=fit_map_raw[str(args.anchor_model)],
            dist_to_joint_thr=dist_fit,
            score_edges=score_edges,
            near_cut=float(args.dist_near_cut),
            mid_lo=float(args.dist_mid_low),
            mid_hi=float(args.dist_mid_high),
        )
        bin_cal = _make_bin_ids(
            joint_score=ref_map_raw[str(args.anchor_model)],
            dist_to_joint_thr=dist_cal,
            score_edges=score_edges,
            near_cut=float(args.dist_near_cut),
            mid_lo=float(args.dist_mid_low),
            mid_hi=float(args.dist_mid_high),
        )
        bin_test = _make_bin_ids(
            joint_score=test_map_raw[str(args.anchor_model)],
            dist_to_joint_thr=dist_test,
            score_edges=score_edges,
            near_cut=float(args.dist_near_cut),
            mid_lo=float(args.dist_mid_low),
            mid_hi=float(args.dist_mid_high),
        )

        # 3) Bin-local updates on top of global blend.
        s_fit_b, s_cal_b, s_test_b, rows_b = _binwise_updates(
            y_fit=y_fit,
            s_fit_init=s_fit_g,
            s_cal_init=s_cal_g,
            s_test_init=s_test_g,
            s_fit_map=fit_map,
            s_cal_map=cal_map,
            s_test_map=test_map,
            candidates=cands,
            target_tpr=float(tpr),
            bin_fit=bin_fit,
            bin_cal=bin_cal,
            bin_test=bin_test,
            score_edges=score_edges,
            min_bin_fit=int(args.min_bin_fit),
            bin_max_add=int(args.bin_max_add),
            w_step=float(args.w_step),
            min_bin_improve=float(args.min_bin_improve),
        )
        for r in rows_b:
            rr = dict(r)
            rr["target_tpr"] = float(tpr)
            update_rows.append(rr)

        # Save fused score arrays.
        k = f"tpr{tpr:.3f}".replace(".", "p")
        score_dump[f"fused_global_fit_{k}"] = s_fit_g.astype(np.float32)
        score_dump[f"fused_global_cal_{k}"] = s_cal_g.astype(np.float32)
        score_dump[f"fused_global_test_{k}"] = s_test_g.astype(np.float32)
        score_dump[f"fused_bin_fit_{k}"] = s_fit_b.astype(np.float32)
        score_dump[f"fused_bin_cal_{k}"] = s_cal_b.astype(np.float32)
        score_dump[f"fused_bin_test_{k}"] = s_test_b.astype(np.float32)

        # 4) Compare methods with threshold selected on reference split.
        methods = {
            "anchor": (ref_map_raw[str(args.anchor_model)], test_map_raw[str(args.anchor_model)]),
            "global_blend": (s_cal_g, s_test_g),
            "bin_gated_blend": (s_cal_b, s_test_b),
            "hlt": (scores_val_raw["hlt"][idx_ref], scores_test_raw["hlt"]),
        }
        if "teacher" in scores_val_raw and "teacher" in scores_test_raw:
            methods["teacher"] = (scores_val_raw["teacher"][idx_ref], scores_test_raw["teacher"])

        tpr_report = {"target_tpr": float(tpr), "candidates": cands, "metrics": {}}
        for mname, (sc, st) in methods.items():
            ev = eval_from_ref(y_ref=y_ref, s_ref=np.asarray(sc, dtype=np.float64), y_eval=y_te, s_eval=np.asarray(st, dtype=np.float64), target_tpr=float(tpr))
            row = {
                "target_tpr": float(tpr),
                "method": mname,
                "auc_cal": float(ev["auc_ref"]),
                "auc_test": float(ev["auc_eval"]),
                "fpr_cal": float(ev["fpr_ref"]),
                "fpr_test": float(ev["fpr_eval"]),
                "tpr_cal": float(ev["tpr_ref"]),
                "tpr_test": float(ev["tpr_eval"]),
                "threshold_from_ref": float(ev["threshold_from_ref"]),
            }
            summary_rows.append(row)
            tpr_report["metrics"][mname] = row

        # Bin counts for interpretability.
        bin_rows = []
        for b in sorted(int(x) for x in np.unique(bin_fit)):
            bin_rows.append(
                {
                    "bin_id": int(b),
                    "label": _bin_label(int(b), score_edges),
                    "n_fit": int((bin_fit == b).sum()),
                    "n_cal": int((bin_cal == b).sum()),
                    "n_test": int((bin_test == b).sum()),
                }
            )
        tpr_report["bins"] = bin_rows
        report_tpr[f"{tpr:.4f}"] = tpr_report

    # Write outputs.
    _save_csv_dynamic(out_dir / "bin_gated_update_log.csv", update_rows)
    _save_csv_dynamic(out_dir / "bin_gated_summary.csv", summary_rows)
    np.savez_compressed(
        out_dir / "bin_gated_scores.npz",
        idx_fit=idx_fit.astype(np.int64),
        idx_ref=idx_ref.astype(np.int64),
        labels_fit=y_fit.astype(np.float32),
        labels_ref=y_ref.astype(np.float32),
        labels_test=y_te.astype(np.float32),
        **score_dump,
    )

    report = {
        "fusion_json": str(fusion_json),
        "out_dir": str(out_dir),
        "settings": vars(args),
        "score_files_used": score_paths,
        "target_tprs": target_tprs,
        "by_tpr": report_tpr,
        "files": {
            "summary_csv": str((out_dir / "bin_gated_summary.csv").resolve()),
            "update_log_csv": str((out_dir / "bin_gated_update_log.csv").resolve()),
            "scores_npz": str((out_dir / "bin_gated_scores.npz").resolve()),
        },
    }
    report_json = (
        Path(args.report_json).expanduser().resolve()
        if str(args.report_json).strip()
        else (out_dir / "bin_gated_report.json")
    )
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("31-Model Bin-Gated Fusion")
    print("=" * 72)
    print(f"Fusion json: {fusion_json}")
    print(f"Out dir:     {out_dir}")
    print(f"Anchor:      {args.anchor_model}")
    print(f"TPRs:        {','.join(f'{x:.3f}' for x in target_tprs)}")
    print(f"Selection:   {args.selection_mode}")
    print(f"Calibration: {args.calibration}")
    print(f"Split fit/ref: {len(idx_fit)} / {len(idx_ref)}")
    print()
    for tpr in target_tprs:
        key = f"{tpr:.4f}"
        m = report_tpr[key]["metrics"]
        ranked = sorted(m.items(), key=lambda kv: (float(kv[1]["fpr_test"]), -float(kv[1]["auc_test"])))
        print(f"TPR={tpr:.3f} best methods (by test FPR then AUC):")
        for name, rr in ranked:
            print(
                f"  {name:16s} AUC_test={float(rr['auc_test']):.6f} "
                f"FPR_test={float(rr['fpr_test']):.6f} "
                f"(cal={float(rr['fpr_cal']):.6f})"
            )
        print()
    print(f"Saved report: {report_json}")
    print(f"Saved summary: {out_dir / 'bin_gated_summary.csv'}")
    print(f"Saved updates: {out_dir / 'bin_gated_update_log.csv'}")
    print(f"Saved scores: {out_dir / 'bin_gated_scores.npz'}")


if __name__ == "__main__":
    main()

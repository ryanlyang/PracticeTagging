#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Specialization atlas for 31-model fusion artifacts.

What this does:
1) Re-load the 31 model val/test score arrays used by fusion_hlt_joint31_* analyses.
2) Build per-model tail metrics at one or more target TPR operating points.
3) Build model-pair overlap/rescue diagnostics.
4) Build bin-wise specialization diagnostics (counts, pT, score regimes, etc.).
5) Run a greedy marginal subset-addition analysis (val-selected, test-evaluated).

No model retraining; inference-only diagnostics using existing score files and dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import analyze_hlt_joint18_recoteacher_fusion as base
from analyze_m2_router_signal_sweep import _build_train_file_list, _offline_mask
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
)
from unmerge_correct_hlt import load_raw_constituents_from_h5


# 31-model order used by analyze_hlt_joint35_recoteacher_fusion.py
MODEL_ORDER_31: List[str] = [
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


def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


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


def _safe_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


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
            out = {k: r.get(k, "") for k in keys}
            w.writerow(out)


def _first_existing_key(npz: np.lib.npyio.NpzFile, names: Iterable[str]) -> str:
    for n in names:
        if n in npz:
            return n
    raise KeyError(f"None of keys found in npz: {list(names)}")


def _load_scores_from_fusion_json(
    fusion_obj: Dict[str, object],
    fusion_json_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str], Dict[str, Path], List[str]]:
    _ = fusion_json_path  # kept for signature clarity
    repo_root = Path(__file__).resolve().parent
    run_dirs_raw = dict(fusion_obj.get("run_dirs", {}))
    score_files_raw = dict(run_dirs_raw.get("score_files", {})) if isinstance(run_dirs_raw.get("score_files", {}), dict) else {}

    resolved_run_dirs: Dict[str, Path] = {}
    for k, v in run_dirs_raw.items():
        if k == "score_files":
            continue
        if isinstance(v, str):
            resolved_run_dirs[k] = _safe_path(v, repo_root)

    model_order = list(fusion_obj.get("models_order", MODEL_ORDER_31))
    if not model_order:
        model_order = list(MODEL_ORDER_31)

    # joint_delta anchor scores / labels
    joint_score_path = score_files_raw.get("joint_delta")
    if not isinstance(joint_score_path, str):
        joint_dir = resolved_run_dirs.get("joint_delta_run_dir")
        if joint_dir is None:
            raise KeyError("Could not resolve joint_delta score path from fusion json")
        joint_score_path = str(joint_dir / "fusion_scores_val_test.npz")
    joint_npz_path = _safe_path(joint_score_path, repo_root)
    z2 = base._load_npz(joint_npz_path)
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
    score_file_used: Dict[str, str] = {"joint_delta": str(joint_npz_path)}

    # Generic fallback for additional models whose score npz follows
    # standard fusion naming.
    generic_val_keys = [
        "preds_joint_val",
        "preds_joint_post_val",
        "preds_joint_bestfpr_val",
        "preds_fused_val",
        "preds_val",
    ]
    generic_test_keys = [
        "preds_joint_test",
        "preds_joint_post_test",
        "preds_joint_bestfpr_test",
        "preds_fused_test",
        "preds_test",
    ]

    for name in model_order:
        if name in ("hlt", "joint_delta"):
            continue

        if name in MODEL_SCORE_SPECS:
            run_key, file_name, val_keys, test_keys = MODEL_SCORE_SPECS[name]
            sf = score_files_raw.get(name)
            if isinstance(sf, str):
                npz_path = _safe_path(sf, repo_root)
            else:
                rd = resolved_run_dirs.get(run_key)
                if rd is None:
                    raise KeyError(f"Missing run dir for {name}: key={run_key}")
                npz_path = _safe_path(str(rd / file_name), repo_root)
        else:
            sf = score_files_raw.get(name)
            if isinstance(sf, str):
                npz_path = _safe_path(sf, repo_root)
            else:
                rd = resolved_run_dirs.get(f"{name}_run_dir")
                if rd is None:
                    raise KeyError(
                        f"Missing score path for model={name}. "
                        f"Add run_dirs.score_files['{name}'] or run_dirs['{name}_run_dir']."
                    )
                npz_path = _safe_path(str(rd / "fusion_scores_val_test.npz"), repo_root)
            val_keys = generic_val_keys
            test_keys = generic_test_keys

        z = base._load_npz(npz_path)
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
        score_file_used[name] = str(npz_path)

    for n in model_order:
        if n not in scores_val:
            raise KeyError(f"Model in models_order missing score arrays: {n}")
    return y_val, y_test, scores_val, scores_test, score_file_used, resolved_run_dirs, model_order


def _extract_split_indices(split_npz: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_train = "train_idx" if "train_idx" in split_npz else "idx_train"
    k_val = "val_idx" if "val_idx" in split_npz else "idx_val"
    k_test = "test_idx" if "test_idx" in split_npz else "idx_test"
    if k_train not in split_npz or k_val not in split_npz or k_test not in split_npz:
        raise KeyError(f"Could not find train/val/test indices in split npz keys={list(split_npz.files)}")
    train_idx = np.asarray(split_npz[k_train], dtype=np.int64)
    val_idx = np.asarray(split_npz[k_val], dtype=np.int64)
    test_idx = np.asarray(split_npz[k_test], dtype=np.int64)
    return train_idx, val_idx, test_idx


def _basic_jet_features(
    const: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> Dict[str, np.ndarray]:
    eps = 1e-8
    m = mask.astype(np.float64)
    pt = np.where(mask, const[:, :, 0], 0.0).astype(np.float64)
    eta = np.where(mask, const[:, :, 1], 0.0).astype(np.float64)
    phi = np.where(mask, const[:, :, 2], 0.0).astype(np.float64)
    ene = np.where(mask, const[:, :, 3], 0.0).astype(np.float64)

    n_const = m.sum(axis=1)
    jet_pt = pt.sum(axis=1)
    jet_E = ene.sum(axis=1)
    px = (pt * np.cos(phi)).sum(axis=1)
    py = (pt * np.sin(phi)).sum(axis=1)
    pz = (pt * np.sinh(eta)).sum(axis=1)
    p2 = px * px + py * py + pz * pz
    jet_mass = np.sqrt(np.maximum(jet_E * jet_E - p2, 0.0))

    eta_axis = np.where(jet_pt > eps, (pt * eta).sum(axis=1) / np.maximum(jet_pt, eps), 0.0)
    sin_phi_axis = np.where(jet_pt > eps, (pt * np.sin(phi)).sum(axis=1) / np.maximum(jet_pt, eps), 0.0)
    cos_phi_axis = np.where(jet_pt > eps, (pt * np.cos(phi)).sum(axis=1) / np.maximum(jet_pt, eps), 1.0)
    phi_axis = np.arctan2(sin_phi_axis, cos_phi_axis)

    deta = eta - eta_axis[:, None]
    dphi = np.arctan2(np.sin(phi - phi_axis[:, None]), np.cos(phi - phi_axis[:, None]))
    dR = np.sqrt(deta * deta + dphi * dphi)
    jet_width = np.where(jet_pt > eps, (pt * dR).sum(axis=1) / np.maximum(jet_pt, eps), 0.0)
    ptD = np.where(jet_pt > eps, np.sqrt((pt * pt).sum(axis=1)) / np.maximum(jet_pt, eps), 0.0)
    topk = min(const.shape[1], 5)
    pt_top5 = np.partition(pt, -topk, axis=1)[:, -topk:].sum(axis=1)
    pt_top5_frac = np.where(jet_pt > eps, pt_top5 / np.maximum(jet_pt, eps), 0.0)
    density_proxy = n_const / np.maximum(jet_width + 1e-3, 1e-3)

    return {
        f"{prefix}_nconst": n_const.astype(np.float32),
        f"{prefix}_jet_pt": jet_pt.astype(np.float32),
        f"{prefix}_jet_mass": jet_mass.astype(np.float32),
        f"{prefix}_jet_width": jet_width.astype(np.float32),
        f"{prefix}_ptD": ptD.astype(np.float32),
        f"{prefix}_pt_top5_frac": pt_top5_frac.astype(np.float32),
        f"{prefix}_density_proxy": density_proxy.astype(np.float32),
    }


def _calc_thresholds_for_tprs(
    y_val: np.ndarray,
    scores_val: Dict[str, np.ndarray],
    model_order: List[str],
    tprs: List[float],
) -> Dict[float, Dict[str, float]]:
    out: Dict[float, Dict[str, float]] = {}
    for tpr in tprs:
        th: Dict[str, float] = {}
        for m in model_order:
            th[m] = float(base.threshold_for_target_tpr(y_val, scores_val[m], tpr))
        out[float(tpr)] = th
    return out


def _global_rows(
    y_val: np.ndarray,
    y_test: np.ndarray,
    scores_val: Dict[str, np.ndarray],
    scores_test: Dict[str, np.ndarray],
    model_order: List[str],
    thresholds: Dict[float, Dict[str, float]],
    anchor_model: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    neg_test = y_test < 0.5
    pos_test = y_test > 0.5
    n_neg = max(int(neg_test.sum()), 1)
    n_pos = max(int(pos_test.sum()), 1)
    for tpr, thr_map in thresholds.items():
        anchor_pred = scores_test[anchor_model] >= float(thr_map[anchor_model])
        for m in model_order:
            thr = float(thr_map[m])
            pred_t = scores_test[m] >= thr
            pred_v = scores_val[m] >= thr

            fp_t = neg_test & pred_t
            tp_t = pos_test & pred_t
            fp_v = (y_val < 0.5) & pred_v
            tp_v = (y_val > 0.5) & pred_v

            rescue = neg_test & anchor_pred & ~pred_t
            harm = neg_test & ~anchor_pred & pred_t

            auc_v = float(base.auc_and_fpr_at_target(y_val, scores_val[m], tpr)["auc"])
            auc_t = float(base.auc_and_fpr_at_target(y_test, scores_test[m], tpr)["auc"])

            rows.append(
                {
                    "target_tpr": float(tpr),
                    "model": m,
                    "threshold_from_val": thr,
                    "auc_val": auc_v,
                    "auc_test": auc_t,
                    "fpr_test": float(fp_t.sum() / n_neg),
                    "tpr_test": float(tp_t.sum() / n_pos),
                    "fpr_val_at_val_thr": float(fp_v.sum() / max(int((y_val < 0.5).sum()), 1)),
                    "tpr_val_at_val_thr": float(tp_v.sum() / max(int((y_val > 0.5).sum()), 1)),
                    "rescue_rate_vs_anchor": float(rescue.sum() / n_neg),
                    "harm_rate_vs_anchor": float(harm.sum() / n_neg),
                    "net_rate_vs_anchor": float((rescue.sum() - harm.sum()) / n_neg),
                    "is_anchor": bool(m == anchor_model),
                }
            )
    return rows


def _pairwise_rows(
    y_test: np.ndarray,
    scores_test: Dict[str, np.ndarray],
    model_order: List[str],
    thresholds: Dict[float, Dict[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    neg = y_test < 0.5
    n_neg = max(int(neg.sum()), 1)
    for tpr, thr_map in thresholds.items():
        preds = {m: (scores_test[m] >= float(thr_map[m])) for m in model_order}
        fps = {m: (neg & preds[m]) for m in model_order}
        for i, a in enumerate(model_order):
            for b in model_order[i + 1 :]:
                fp_a = fps[a]
                fp_b = fps[b]
                inter = int(np.logical_and(fp_a, fp_b).sum())
                union = int(np.logical_or(fp_a, fp_b).sum())
                jacc = float(inter / max(union, 1))

                rescue_a_to_b = int(np.logical_and(fp_a, ~fp_b).sum())
                rescue_b_to_a = int(np.logical_and(fp_b, ~fp_a).sum())
                harm_a_to_b = rescue_b_to_a
                harm_b_to_a = rescue_a_to_b

                rows.append(
                    {
                        "target_tpr": float(tpr),
                        "model_a": a,
                        "model_b": b,
                        "fp_intersection": inter,
                        "fp_union": union,
                        "fp_jaccard": jacc,
                        "rescue_rate_a_to_b": float(rescue_a_to_b / n_neg),
                        "rescue_rate_b_to_a": float(rescue_b_to_a / n_neg),
                        "harm_rate_a_to_b": float(harm_a_to_b / n_neg),
                        "harm_rate_b_to_a": float(harm_b_to_a / n_neg),
                        "net_rate_a_to_b": float((rescue_a_to_b - harm_a_to_b) / n_neg),
                        "net_rate_b_to_a": float((rescue_b_to_a - harm_b_to_a) / n_neg),
                    }
                )
    return rows


def _make_fixed_bins(values: np.ndarray, edges: List[float]) -> Tuple[np.ndarray, List[str]]:
    v = np.asarray(values, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("fixed bin edges must have >=2 entries")
    idx = np.digitize(v, e[1:-1], right=False).astype(np.int64)
    labels = [f"[{e[i]:g},{e[i+1]:g})" for i in range(e.size - 1)]
    return idx, labels


def _make_quantile_bins(values: np.ndarray, q: int) -> Tuple[np.ndarray, List[str]]:
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if finite.sum() < 20:
        return np.zeros(v.shape[0], dtype=np.int64), ["all"]
    qs = np.linspace(0.0, 1.0, max(2, int(q) + 1))
    edges = np.quantile(v[finite], qs)
    edges = np.unique(edges)
    if edges.size < 3:
        return np.zeros(v.shape[0], dtype=np.int64), ["all"]
    idx = np.digitize(v, edges[1:-1], right=False).astype(np.int64)
    labels = [f"[{edges[i]:.4g},{edges[i+1]:.4g})" for i in range(edges.size - 1)]
    return idx, labels


def _bin_rows(
    y_test: np.ndarray,
    feature_test: Dict[str, np.ndarray],
    feature_val: Dict[str, np.ndarray],
    scores_test: Dict[str, np.ndarray],
    thresholds: Dict[float, Dict[str, float]],
    model_order: List[str],
    anchor_model: str,
    min_bin_neg: int,
    count_edges: List[float],
    quantile_bins: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    neg = y_test < 0.5
    pos = y_test > 0.5

    for feat_name, v_test in feature_test.items():
        v_val = feature_val.get(feat_name)
        if v_val is None:
            continue

        if "nconst" in feat_name:
            bidx_test, blabs = _make_fixed_bins(v_test, count_edges)
        elif feat_name in {"hlt_score", "joint_score"}:
            bidx_test, blabs = _make_fixed_bins(v_test, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        else:
            bidx_test, blabs = _make_quantile_bins(v_val, quantile_bins)
            # bidx_test built from val bins only if quantiles are non-degenerate.
            if len(blabs) > 1:
                edges = np.quantile(np.asarray(v_val, dtype=np.float64), np.linspace(0.0, 1.0, len(blabs) + 1))
                edges = np.unique(edges)
                if edges.size >= 3:
                    bidx_test = np.digitize(np.asarray(v_test, dtype=np.float64), edges[1:-1], right=False).astype(np.int64)
                    blabs = [f"[{edges[i]:.4g},{edges[i+1]:.4g})" for i in range(edges.size - 1)]

        if len(blabs) <= 1:
            continue

        for b in range(len(blabs)):
            m_bin = bidx_test == b
            n_bin = int(m_bin.sum())
            if n_bin <= 0:
                continue
            n_neg_bin = int((neg & m_bin).sum())
            n_pos_bin = int((pos & m_bin).sum())
            if n_neg_bin < int(min_bin_neg):
                continue

            for tpr, thr_map in thresholds.items():
                anchor_pred = scores_test[anchor_model] >= float(thr_map[anchor_model])
                for m in model_order:
                    pred = scores_test[m] >= float(thr_map[m])
                    fp = neg & pred & m_bin
                    tp = pos & pred & m_bin
                    rescue = neg & m_bin & anchor_pred & ~pred
                    harm = neg & m_bin & ~anchor_pred & pred
                    rows.append(
                        {
                            "feature": feat_name,
                            "bin_id": int(b),
                            "bin_label": blabs[b],
                            "target_tpr": float(tpr),
                            "model": m,
                            "n_bin": n_bin,
                            "n_neg_bin": n_neg_bin,
                            "n_pos_bin": n_pos_bin,
                            "fpr_bin": float(fp.sum() / max(n_neg_bin, 1)),
                            "tpr_bin": float(tp.sum() / max(n_pos_bin, 1)),
                            "rescue_rate_vs_anchor_bin": float(rescue.sum() / max(n_neg_bin, 1)),
                            "harm_rate_vs_anchor_bin": float(harm.sum() / max(n_neg_bin, 1)),
                            "net_rate_vs_anchor_bin": float((rescue.sum() - harm.sum()) / max(n_neg_bin, 1)),
                        }
                    )
    return rows


def _greedy_subset_rows(
    y_val: np.ndarray,
    y_test: np.ndarray,
    scores_val: Dict[str, np.ndarray],
    scores_test: Dict[str, np.ndarray],
    model_order: List[str],
    anchor_model: str,
    tpr: float,
    max_add: int,
    w_step: float,
    calibration: str,
) -> List[Dict[str, object]]:
    # Optional per-model calibration for fair score-space mixing.
    cal_val: Dict[str, np.ndarray] = {}
    cal_test: Dict[str, np.ndarray] = {}
    for m in model_order:
        sv = scores_val[m]
        st = scores_test[m]
        if calibration == "iso":
            sv2, st2, _ = base.calibrate_isotonic(y_val, sv, st)
            cal_val[m] = sv2
            cal_test[m] = st2
        elif calibration == "platt":
            sv2, st2, _ = base.calibrate_platt(y_val, sv, st)
            cal_val[m] = sv2
            cal_test[m] = st2
        else:
            cal_val[m] = np.asarray(sv, dtype=np.float64)
            cal_test[m] = np.asarray(st, dtype=np.float64)

    rows: List[Dict[str, object]] = []
    selected: List[str] = [anchor_model]
    n_models = len(model_order)
    w_full = np.zeros((n_models,), dtype=np.float64)
    idx_model = {m: i for i, m in enumerate(model_order)}
    w_full[idx_model[anchor_model]] = 1.0

    s_val_cur = cal_val[anchor_model].copy()
    s_test_cur = cal_test[anchor_model].copy()

    def _eval_pack(sv: np.ndarray, st: np.ndarray) -> Dict[str, float]:
        thr = float(base.threshold_for_target_tpr(y_val, sv, tpr))
        rv = base.rates_from_threshold(y_val, sv, thr)
        rt = base.rates_from_threshold(y_test, st, thr)
        av = base.auc_and_fpr_at_target(y_val, sv, tpr)
        at = base.auc_and_fpr_at_target(y_test, st, tpr)
        return {
            "threshold": thr,
            "fpr_val": float(rv["fpr"]),
            "tpr_val": float(rv["tpr"]),
            "fpr_test": float(rt["fpr"]),
            "tpr_test": float(rt["tpr"]),
            "auc_val": float(av["auc"]),
            "auc_test": float(at["auc"]),
        }

    base_pack = _eval_pack(s_val_cur, s_test_cur)
    rows.append(
        {
            "target_tpr": float(tpr),
            "step": 0,
            "action": "init_anchor",
            "added_model": anchor_model,
            "blend_w_to_new": 0.0,
            "selected_models": ",".join(selected),
            **base_pack,
        }
    )

    w_grid = np.linspace(0.0, 1.0, max(2, int(round(1.0 / max(1e-6, w_step))) + 1))
    for step in range(1, int(max_add) + 1):
        cur_best = None
        for cand in model_order:
            if cand in selected:
                continue
            sv_c = cal_val[cand]
            st_c = cal_test[cand]
            best_c = None
            for w in w_grid:
                sv = (1.0 - w) * s_val_cur + w * sv_c
                st = (1.0 - w) * s_test_cur + w * st_c
                pack = _eval_pack(sv, st)
                cand_pack = {
                    "model": cand,
                    "w": float(w),
                    "sv": sv,
                    "st": st,
                    **pack,
                }
                if best_c is None:
                    best_c = cand_pack
                else:
                    key = (cand_pack["fpr_val"], -cand_pack["auc_val"])
                    key_b = (best_c["fpr_val"], -best_c["auc_val"])
                    if key < key_b:
                        best_c = cand_pack
            if best_c is None:
                continue
            if cur_best is None:
                cur_best = best_c
            else:
                key = (best_c["fpr_val"], -best_c["auc_val"])
                key_b = (cur_best["fpr_val"], -cur_best["auc_val"])
                if key < key_b:
                    cur_best = best_c

        if cur_best is None:
            break

        prev = rows[-1]
        if float(cur_best["fpr_val"]) >= float(prev["fpr_val"]) - 1e-12:
            rows.append(
                {
                    "target_tpr": float(tpr),
                    "step": step,
                    "action": "stop_no_val_improve",
                    "added_model": "",
                    "blend_w_to_new": 0.0,
                    "selected_models": ",".join(selected),
                    **_eval_pack(s_val_cur, s_test_cur),
                }
            )
            break

        cand = str(cur_best["model"])
        w = float(cur_best["w"])
        selected.append(cand)
        s_val_cur = np.asarray(cur_best["sv"], dtype=np.float64)
        s_test_cur = np.asarray(cur_best["st"], dtype=np.float64)
        w_full *= (1.0 - w)
        w_full[idx_model[cand]] += w

        rows.append(
            {
                "target_tpr": float(tpr),
                "step": step,
                "action": "add_model",
                "added_model": cand,
                "blend_w_to_new": w,
                "selected_models": ",".join(selected),
                "weight_vector": ",".join(f"{model_order[i]}:{w_full[i]:.6f}" for i in range(n_models) if w_full[i] > 1e-12),
                **_eval_pack(s_val_cur, s_test_cur),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="31-model specialization atlas and greedy subset diagnostics")
    ap.add_argument("--fusion_json", type=str, required=True)
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--target_tprs", type=str, default="0.50,0.30")
    ap.add_argument("--anchor_model", type=str, default="joint_delta")
    ap.add_argument("--quantile_bins", type=int, default=10)
    ap.add_argument("--count_edges", type=str, default="0,10,20,30,40,60,80,100,200")
    ap.add_argument("--min_bin_neg", type=int, default=200)
    ap.add_argument("--greedy_max_add", type=int, default=12)
    ap.add_argument("--greedy_w_step", type=float, default=0.01)
    ap.add_argument("--greedy_calibration", type=str, default="iso", choices=["raw", "iso", "platt"])
    ap.add_argument("--hlt_seed", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    args = ap.parse_args()

    fusion_json = Path(args.fusion_json).expanduser().resolve()
    if not fusion_json.exists():
        raise FileNotFoundError(f"fusion_json not found: {fusion_json}")

    fusion = json.loads(fusion_json.read_text())
    y_val, y_test, scores_val, scores_test, score_files, run_dirs, model_order = _load_scores_from_fusion_json(fusion, fusion_json)

    if str(args.anchor_model) not in model_order:
        raise KeyError(f"anchor_model={args.anchor_model} is not in model_order")
    tprs = _parse_float_list(args.target_tprs, [0.50, 0.30])

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (fusion_json.parent / "specialization_atlas_31")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    count_edges = _parse_float_list(args.count_edges, [0, 10, 20, 30, 40, 60, 80, 100, 200])
    if len(count_edges) < 2:
        raise ValueError("--count_edges must provide at least two edges")

    # Global model thresholds and metrics.
    thresholds = _calc_thresholds_for_tprs(y_val, scores_val, model_order, tprs)
    global_rows = _global_rows(
        y_val=y_val,
        y_test=y_test,
        scores_val=scores_val,
        scores_test=scores_test,
        model_order=model_order,
        thresholds=thresholds,
        anchor_model=str(args.anchor_model),
    )
    pair_rows = _pairwise_rows(
        y_test=y_test,
        scores_test=scores_test,
        model_order=model_order,
        thresholds=thresholds,
    )

    # Per-jet descriptors (val/test aligned with score arrays).
    feature_val: Dict[str, np.ndarray] = {}
    feature_test: Dict[str, np.ndarray] = {}
    jet_feature_status = {"ok": False, "message": ""}

    try:
        joint_dir = run_dirs.get("joint_delta_run_dir")
        if joint_dir is None:
            raise KeyError("joint_delta_run_dir missing in fusion run_dirs")
        data_setup_path = joint_dir / "data_setup.json"
        split_path = joint_dir / "data_splits.npz"
        hlt_stats_path = joint_dir / "hlt_stats.json"
        if not data_setup_path.exists():
            raise FileNotFoundError(f"Missing {data_setup_path}")
        if not split_path.exists():
            raise FileNotFoundError(f"Missing {split_path}")
        if not hlt_stats_path.exists():
            raise FileNotFoundError(f"Missing {hlt_stats_path}")

        data_setup = json.loads(data_setup_path.read_text())
        split_npz = np.load(split_path)
        train_idx, val_idx, test_idx = _extract_split_indices(split_npz)
        if val_idx.shape[0] != y_val.shape[0] or test_idx.shape[0] != y_test.shape[0]:
            raise RuntimeError(
                "data_splits val/test lengths do not match fusion score labels "
                f"({val_idx.shape[0]}/{test_idx.shape[0]} vs {y_val.shape[0]}/{y_test.shape[0]})"
            )

        cfg = _deepcopy_cfg()
        hlt_obj = json.loads(hlt_stats_path.read_text())
        hcfg = hlt_obj.get("config", {})
        for k in list(cfg.get("hlt_effects", {}).keys()):
            if k in hcfg:
                cfg["hlt_effects"][k] = hcfg[k]
        pt_thr_off = float(cfg["hlt_effects"]["pt_threshold_offline"])

        train_files = _build_train_file_list(data_setup, args.train_path)
        n_need = int(max(val_idx.max(), test_idx.max()) + 1)
        const_raw_all, _labels_all = load_raw_constituents_from_h5(
            files=train_files,
            max_jets=n_need,
            max_constits=int(args.max_constits),
        )
        const_off_all, mask_off_all = _offline_mask(const_raw_all, pt_thr_off)
        hlt_const_all, hlt_mask_all, _hlt_stats_gen, _ = apply_hlt_effects_realistic_nomap(
            const_off_all,
            mask_off_all,
            cfg,
            seed=int(args.hlt_seed),
        )

        off_val = _basic_jet_features(const_off_all[val_idx], mask_off_all[val_idx], prefix="off")
        off_test = _basic_jet_features(const_off_all[test_idx], mask_off_all[test_idx], prefix="off")
        hlt_val = _basic_jet_features(hlt_const_all[val_idx], hlt_mask_all[val_idx], prefix="hlt")
        hlt_test = _basic_jet_features(hlt_const_all[test_idx], hlt_mask_all[test_idx], prefix="hlt")

        feature_val.update(off_val)
        feature_val.update(hlt_val)
        feature_test.update(off_test)
        feature_test.update(hlt_test)

        # Delta descriptors.
        feature_val["delta_nconst_off_minus_hlt"] = feature_val["off_nconst"] - feature_val["hlt_nconst"]
        feature_test["delta_nconst_off_minus_hlt"] = feature_test["off_nconst"] - feature_test["hlt_nconst"]
        feature_val["delta_pt_off_minus_hlt"] = feature_val["off_jet_pt"] - feature_val["hlt_jet_pt"]
        feature_test["delta_pt_off_minus_hlt"] = feature_test["off_jet_pt"] - feature_test["hlt_jet_pt"]
        feature_val["delta_mass_off_minus_hlt"] = feature_val["off_jet_mass"] - feature_val["hlt_jet_mass"]
        feature_test["delta_mass_off_minus_hlt"] = feature_test["off_jet_mass"] - feature_test["hlt_jet_mass"]

        jet_feature_status["ok"] = True
        jet_feature_status["message"] = "Loaded raw-jet descriptors from data_splits + pseudo-HLT generation."
    except Exception as e:
        jet_feature_status["ok"] = False
        jet_feature_status["message"] = f"Jet feature extraction unavailable: {repr(e)}"

    # Score-derived descriptors always available.
    feature_val["hlt_score"] = np.asarray(scores_val["hlt"], dtype=np.float64)
    feature_test["hlt_score"] = np.asarray(scores_test["hlt"], dtype=np.float64)
    feature_val["joint_score"] = np.asarray(scores_val["joint_delta"], dtype=np.float64)
    feature_test["joint_score"] = np.asarray(scores_test["joint_delta"], dtype=np.float64)

    feature_val["score_gap_joint_minus_hlt"] = feature_val["joint_score"] - feature_val["hlt_score"]
    feature_test["score_gap_joint_minus_hlt"] = feature_test["joint_score"] - feature_test["hlt_score"]
    feature_val["abs_score_gap_joint_hlt"] = np.abs(feature_val["score_gap_joint_minus_hlt"])
    feature_test["abs_score_gap_joint_hlt"] = np.abs(feature_test["score_gap_joint_minus_hlt"])
    feature_val["hlt_entropy"] = _safe_entropy(feature_val["hlt_score"])
    feature_test["hlt_entropy"] = _safe_entropy(feature_test["hlt_score"])
    feature_val["joint_entropy"] = _safe_entropy(feature_val["joint_score"])
    feature_test["joint_entropy"] = _safe_entropy(feature_test["joint_score"])
    feature_val["joint_conf_minus_hlt_conf"] = np.abs(feature_val["joint_score"] - 0.5) - np.abs(feature_val["hlt_score"] - 0.5)
    feature_test["joint_conf_minus_hlt_conf"] = np.abs(feature_test["joint_score"] - 0.5) - np.abs(feature_test["hlt_score"] - 0.5)

    # Threshold-distance features at first target TPR.
    tpr_ref = float(tprs[0])
    thr_h = float(thresholds[tpr_ref]["hlt"])
    thr_j = float(thresholds[tpr_ref]["joint_delta"])
    feature_val["dist_to_hlt_thr"] = np.abs(feature_val["hlt_score"] - thr_h)
    feature_test["dist_to_hlt_thr"] = np.abs(feature_test["hlt_score"] - thr_h)
    feature_val["dist_to_joint_thr"] = np.abs(feature_val["joint_score"] - thr_j)
    feature_test["dist_to_joint_thr"] = np.abs(feature_test["joint_score"] - thr_j)
    feature_val["dist_gap_joint_minus_hlt"] = feature_val["dist_to_joint_thr"] - feature_val["dist_to_hlt_thr"]
    feature_test["dist_gap_joint_minus_hlt"] = feature_test["dist_to_joint_thr"] - feature_test["dist_to_hlt_thr"]

    bin_rows = _bin_rows(
        y_test=y_test,
        feature_test=feature_test,
        feature_val=feature_val,
        scores_test=scores_test,
        thresholds=thresholds,
        model_order=model_order,
        anchor_model=str(args.anchor_model),
        min_bin_neg=int(args.min_bin_neg),
        count_edges=count_edges,
        quantile_bins=int(args.quantile_bins),
    )

    # Greedy marginal subset selection.
    greedy_rows: List[Dict[str, object]] = []
    for tpr in tprs:
        greedy_rows.extend(
            _greedy_subset_rows(
                y_val=y_val,
                y_test=y_test,
                scores_val=scores_val,
                scores_test=scores_test,
                model_order=model_order,
                anchor_model=str(args.anchor_model),
                tpr=float(tpr),
                max_add=int(args.greedy_max_add),
                w_step=float(args.greedy_w_step),
                calibration=str(args.greedy_calibration),
            )
        )

    # Per-model top specialization bins (largest positive net rescue-vs-harm).
    top_bin_rows: List[Dict[str, object]] = []
    for tpr in tprs:
        for m in model_order:
            if m == str(args.anchor_model):
                continue
            cand = [
                r
                for r in bin_rows
                if float(r["target_tpr"]) == float(tpr)
                and str(r["model"]) == m
                and int(r["n_neg_bin"]) >= int(args.min_bin_neg)
            ]
            cand = sorted(cand, key=lambda r: float(r["net_rate_vs_anchor_bin"]), reverse=True)
            for rank, r in enumerate(cand[:3], start=1):
                rr = dict(r)
                rr["rank"] = rank
                top_bin_rows.append(rr)

    # Write outputs.
    _save_csv_dynamic(out_dir / "atlas_model_global_metrics.csv", global_rows)
    _save_csv_dynamic(out_dir / "atlas_pairwise_overlap_rescue.csv", pair_rows)
    _save_csv_dynamic(out_dir / "atlas_bin_specialization_metrics.csv", bin_rows)
    _save_csv_dynamic(out_dir / "atlas_top_specialization_bins.csv", top_bin_rows)
    _save_csv_dynamic(out_dir / "atlas_greedy_marginal_curve.csv", greedy_rows)

    # Compact summaries for easy scanning.
    best_by_tpr: Dict[str, Dict[str, object]] = {}
    for tpr in tprs:
        rr = [r for r in global_rows if float(r["target_tpr"]) == float(tpr)]
        rr = sorted(rr, key=lambda r: (float(r["fpr_test"]), -float(r["auc_test"])))
        best_by_tpr[f"{tpr:.4f}"] = rr[0] if rr else {}

    report = {
        "fusion_json": str(fusion_json),
        "out_dir": str(out_dir),
        "settings": vars(args),
        "target_tprs": tprs,
        "model_order": model_order,
        "anchor_model": str(args.anchor_model),
        "n_models": int(len(model_order)),
        "n_val": int(y_val.shape[0]),
        "n_test": int(y_test.shape[0]),
        "jet_feature_status": jet_feature_status,
        "score_files_used": score_files,
        "best_model_by_tpr_test": best_by_tpr,
        "files": {
            "global_csv": str((out_dir / "atlas_model_global_metrics.csv").resolve()),
            "pairwise_csv": str((out_dir / "atlas_pairwise_overlap_rescue.csv").resolve()),
            "bins_csv": str((out_dir / "atlas_bin_specialization_metrics.csv").resolve()),
            "top_bins_csv": str((out_dir / "atlas_top_specialization_bins.csv").resolve()),
            "greedy_csv": str((out_dir / "atlas_greedy_marginal_curve.csv").resolve()),
        },
    }

    report_json = (
        Path(args.report_json).expanduser().resolve()
        if str(args.report_json).strip()
        else (out_dir / "specialization_atlas_report.json")
    )
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("31-Model Specialization Atlas")
    print("=" * 72)
    print(f"Fusion json: {fusion_json}")
    print(f"Out dir:     {out_dir}")
    print(f"Models:      {len(model_order)}")
    print(f"Val/Test:    {y_val.shape[0]} / {y_test.shape[0]}")
    print(f"Anchor:      {args.anchor_model}")
    print(f"Target TPRs: {','.join(f'{x:.3f}' for x in tprs)}")
    print(f"Jet feats:   {jet_feature_status['message']}")
    print()
    print("Best single model by test FPR (per target TPR):")
    for k, v in best_by_tpr.items():
        if not v:
            continue
        print(
            f"  TPR={k}: {v.get('model', '?')} "
            f"FPR={float(v.get('fpr_test', float('nan'))):.6f} "
            f"AUC={float(v.get('auc_test', float('nan'))):.6f}"
        )
    print()
    print(f"Saved report: {report_json}")
    print(f"Saved global metrics: {out_dir / 'atlas_model_global_metrics.csv'}")
    print(f"Saved pairwise metrics: {out_dir / 'atlas_pairwise_overlap_rescue.csv'}")
    print(f"Saved bin metrics: {out_dir / 'atlas_bin_specialization_metrics.csv'}")
    print(f"Saved top-bin summary: {out_dir / 'atlas_top_specialization_bins.csv'}")
    print(f"Saved greedy curve: {out_dir / 'atlas_greedy_marginal_curve.csv'}")


if __name__ == "__main__":
    main()

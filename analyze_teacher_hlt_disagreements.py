#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze teacher-vs-HLT disagreement jets for a saved run directory.

This script reloads:
  - data_setup.json / data_splits.npz
  - teacher.pt / baseline.pt

Then it rebuilds the exact pseudo-HLT dataset, runs teacher + baseline on val/test,
defines operating thresholds at target TPR (default 0.50), and reports diagnostics for:
  - teacher-right / hlt-wrong
  - hlt-right / teacher-wrong
  - critical negatives (background where HLT false-positive, teacher correct)
  - critical positives (signal where HLT false-negative, teacher correct)

It also exports subset jets for downstream manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from unmerge_correct_hlt import (
    load_raw_constituents_from_h5,
    compute_features,
    get_stats,
    standardize,
    ParticleTransformer,
    JetDataset,
    eval_classifier,
    compute_jet_pt,
)
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
    fpr_at_target_tpr,
)


def threshold_at_target_tpr(probs: np.ndarray, labels: np.ndarray, target_tpr: float) -> float:
    pos = probs[labels == 1]
    if pos.size == 0:
        return 0.5
    q = float(np.clip(1.0 - float(target_tpr), 0.0, 1.0))
    return float(np.quantile(pos, q))


def safe_stat(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10.0)),
        "p90": float(np.percentile(x, 90.0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def jet_mass(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pt = np.maximum(const[:, :, 0], 0.0)
    eta = const[:, :, 1]
    phi = const[:, :, 2]
    e = np.maximum(const[:, :, 3], 0.0)
    w = mask.astype(np.float32)
    px = (pt * np.cos(phi) * w).sum(axis=1)
    py = (pt * np.sin(phi) * w).sum(axis=1)
    pz = (pt * np.sinh(eta) * w).sum(axis=1)
    ee = (e * w).sum(axis=1)
    m2 = ee * ee - (px * px + py * py + pz * pz)
    return np.sqrt(np.maximum(m2, 0.0)).astype(np.float32)


def summarize_subset(name: str, idx: np.ndarray, fields: Dict[str, np.ndarray], y: np.ndarray) -> Dict:
    out: Dict[str, object] = {
        "name": str(name),
        "n": int(idx.size),
        "fraction_of_test": float(idx.size / max(1, y.size)),
        "label_positive_rate": float(np.mean(y[idx])) if idx.size > 0 else float("nan"),
        "label_counts": {
            "n_background": int(np.sum(y[idx] == 0)) if idx.size > 0 else 0,
            "n_signal": int(np.sum(y[idx] == 1)) if idx.size > 0 else 0,
        },
        "metrics": {},
    }
    for k, arr in fields.items():
        out["metrics"][k] = safe_stat(arr[idx])
    return out


def write_per_jet_tsv(path: Path, columns: Dict[str, np.ndarray]) -> None:
    names = list(columns.keys())
    n = len(columns[names[0]]) if names else 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(names)
        for i in range(n):
            row = []
            for k in names:
                v = columns[k][i]
                if isinstance(v, (np.floating, float)):
                    row.append(f"{float(v):.8g}")
                elif isinstance(v, (np.integer, int, np.bool_, bool)):
                    row.append(str(int(v)))
                else:
                    row.append(str(v))
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Saved run dir containing teacher.pt/baseline.pt/data_setup.json.")
    p.add_argument("--save_subdir", type=str, default="teacher_hlt_disagreement_analysis")
    p.add_argument("--target_tpr", type=float, default=0.50)
    p.add_argument("--threshold_source", type=str, default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=-1)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_export_per_subset", type=int, default=50000)
    p.add_argument("--export_all_subset_jets", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    setup_path = run_dir / "data_setup.json"
    splits_path = run_dir / "data_splits.npz"
    teacher_path = run_dir / "teacher.pt"
    baseline_path = run_dir / "baseline.pt"

    for fp in [setup_path, splits_path, teacher_path, baseline_path]:
        if not fp.exists():
            raise FileNotFoundError(f"Required file not found: {fp}")

    out_dir = run_dir / args.save_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(setup_path, "r", encoding="utf-8") as f:
        setup = json.load(f)
    splits = np.load(splits_path, allow_pickle=False)
    train_idx = splits["train_idx"].astype(np.int64)
    val_idx = splits["val_idx"].astype(np.int64)
    test_idx = splits["test_idx"].astype(np.int64)
    means = splits["means"].astype(np.float32)
    stds = splits["stds"].astype(np.float32)

    cfg = json.loads(json.dumps(BASE_CONFIG))
    if "hlt_effects" in setup:
        cfg["hlt_effects"].update(setup["hlt_effects"])

    n_train_jets = int(setup["n_train_jets"])
    offset_jets = int(setup["offset_jets"])
    max_constits = int(setup["max_constits"])
    seed = int(setup["seed"])
    device = torch.device(args.device)

    train_files = [Path(x) for x in setup.get("train_files", [])]
    if len(train_files) == 0:
        raise RuntimeError("No train_files found in data_setup.json.")

    max_jets_needed = offset_jets + n_train_jets
    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=max_constits,
    )
    const_raw = all_const_full[offset_jets : offset_jets + n_train_jets]
    labels = all_labels_full[offset_jets : offset_jets + n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT deterministically...")
    hlt_const, hlt_mask, hlt_stats, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=seed,
    )

    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    batch_size = int(cfg["training"]["batch_size"]) if int(args.batch_size) <= 0 else int(args.batch_size)

    ds_val_off = JetDataset(feat_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    ds_val_hlt = JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    ds_test_off = JetDataset(feat_off_std[test_idx], masks_off[test_idx], labels[test_idx])
    ds_test_hlt = JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_val_off = DataLoader(ds_val_off, batch_size=batch_size, shuffle=False, num_workers=int(args.num_workers))
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=batch_size, shuffle=False, num_workers=int(args.num_workers))
    dl_test_off = DataLoader(ds_test_off, batch_size=batch_size, shuffle=False, num_workers=int(args.num_workers))
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=batch_size, shuffle=False, num_workers=int(args.num_workers))

    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher_ckpt = torch.load(teacher_path, map_location=device)
    baseline_ckpt = torch.load(baseline_path, map_location=device)
    teacher.load_state_dict(teacher_ckpt["model"])
    baseline.load_state_dict(baseline_ckpt["model"])

    print("Evaluating teacher/baseline on val/test...")
    auc_val_teacher, p_val_teacher, y_val_teacher = eval_classifier(teacher, dl_val_off, device)
    auc_val_hlt, p_val_hlt, y_val_hlt = eval_classifier(baseline, dl_val_hlt, device)
    auc_test_teacher, p_test_teacher, y_test_teacher = eval_classifier(teacher, dl_test_off, device)
    auc_test_hlt, p_test_hlt, y_test_hlt = eval_classifier(baseline, dl_test_hlt, device)

    if not np.array_equal(y_val_teacher.astype(np.int64), y_val_hlt.astype(np.int64)):
        raise RuntimeError("Val label mismatch between teacher and HLT.")
    if not np.array_equal(y_test_teacher.astype(np.int64), y_test_hlt.astype(np.int64)):
        raise RuntimeError("Test label mismatch between teacher and HLT.")

    y_val = y_val_teacher.astype(np.int64)
    y_test = y_test_teacher.astype(np.int64)

    if args.threshold_source == "test":
        thr_teacher = threshold_at_target_tpr(p_test_teacher, y_test, float(args.target_tpr))
        thr_hlt = threshold_at_target_tpr(p_test_hlt, y_test, float(args.target_tpr))
    else:
        thr_teacher = threshold_at_target_tpr(p_val_teacher, y_val, float(args.target_tpr))
        thr_hlt = threshold_at_target_tpr(p_val_hlt, y_val, float(args.target_tpr))

    pred_teacher = (p_test_teacher >= thr_teacher).astype(np.int64)
    pred_hlt = (p_test_hlt >= thr_hlt).astype(np.int64)
    teacher_right = pred_teacher == y_test
    hlt_right = pred_hlt == y_test

    # Core disagreement subsets.
    idx_teacher_right_hlt_wrong = np.where(teacher_right & (~hlt_right))[0]
    idx_hlt_right_teacher_wrong = np.where(hlt_right & (~teacher_right))[0]
    idx_both_right = np.where(teacher_right & hlt_right)[0]
    idx_both_wrong = np.where((~teacher_right) & (~hlt_right))[0]

    # Directional subsets.
    idx_critical_neg = np.where((y_test == 0) & teacher_right & (~hlt_right))[0]
    idx_critical_pos = np.where((y_test == 1) & teacher_right & (~hlt_right))[0]
    idx_hlt_rescue_neg = np.where((y_test == 0) & hlt_right & (~teacher_right))[0]
    idx_hlt_rescue_pos = np.where((y_test == 1) & hlt_right & (~teacher_right))[0]

    # Jet/constituent-level fields on test split.
    c_off_t = const_off[test_idx]
    m_off_t = masks_off[test_idx]
    c_hlt_t = hlt_const[test_idx]
    m_hlt_t = hlt_mask[test_idx]

    n_off = m_off_t.sum(axis=1).astype(np.float32)
    n_hlt = m_hlt_t.sum(axis=1).astype(np.float32)
    n_missing = np.maximum(n_off - n_hlt, 0.0).astype(np.float32)
    merge_lost = budget_truth["merge_lost_per_jet"][test_idx].astype(np.float32)
    eff_lost = budget_truth["eff_lost_per_jet"][test_idx].astype(np.float32)

    pt_off = compute_jet_pt(c_off_t, m_off_t).astype(np.float32)
    pt_hlt = compute_jet_pt(c_hlt_t, m_hlt_t).astype(np.float32)
    pt_resp = (pt_hlt / np.maximum(pt_off, 1e-8)).astype(np.float32)

    m_off = jet_mass(c_off_t, m_off_t)
    m_hlt = jet_mass(c_hlt_t, m_hlt_t)
    m_resp = (m_hlt / np.maximum(m_off, 1e-8)).astype(np.float32)

    p_gap = (p_test_teacher - p_test_hlt).astype(np.float32)
    p_gap_abs = np.abs(p_gap).astype(np.float32)

    fields = {
        "p_teacher": p_test_teacher.astype(np.float32),
        "p_hlt": p_test_hlt.astype(np.float32),
        "p_gap_teacher_minus_hlt": p_gap,
        "p_gap_abs": p_gap_abs,
        "n_const_offline": n_off,
        "n_const_hlt": n_hlt,
        "n_const_missing": n_missing,
        "merge_lost_true": merge_lost,
        "eff_lost_true": eff_lost,
        "jet_pt_offline": pt_off,
        "jet_pt_hlt": pt_hlt,
        "jet_pt_response_hlt_over_offline": pt_resp,
        "jet_mass_offline": m_off,
        "jet_mass_hlt": m_hlt,
        "jet_mass_response_hlt_over_offline": m_resp,
    }

    subset_summaries = {
        "teacher_right_hlt_wrong": summarize_subset("teacher_right_hlt_wrong", idx_teacher_right_hlt_wrong, fields, y_test),
        "hlt_right_teacher_wrong": summarize_subset("hlt_right_teacher_wrong", idx_hlt_right_teacher_wrong, fields, y_test),
        "critical_neg_bg_teacher_right_hlt_fp": summarize_subset("critical_neg_bg_teacher_right_hlt_fp", idx_critical_neg, fields, y_test),
        "critical_pos_sig_teacher_right_hlt_fn": summarize_subset("critical_pos_sig_teacher_right_hlt_fn", idx_critical_pos, fields, y_test),
        "hlt_rescue_neg_bg_hlt_right_teacher_fp": summarize_subset("hlt_rescue_neg_bg_hlt_right_teacher_fp", idx_hlt_rescue_neg, fields, y_test),
        "hlt_rescue_pos_sig_hlt_right_teacher_fn": summarize_subset("hlt_rescue_pos_sig_hlt_right_teacher_fn", idx_hlt_rescue_pos, fields, y_test),
        "both_right": summarize_subset("both_right", idx_both_right, fields, y_test),
        "both_wrong": summarize_subset("both_wrong", idx_both_wrong, fields, y_test),
    }

    # Core model metrics on test.
    fpr_t, tpr_t, _ = np.array([]), np.array([]), None
    fpr_h, tpr_h, _ = np.array([]), np.array([]), None
    try:
        from sklearn.metrics import roc_curve
        fpr_t, tpr_t, _ = roc_curve(y_test, p_test_teacher)
        fpr_h, tpr_h, _ = roc_curve(y_test, p_test_hlt)
    except Exception:
        pass
    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.30) if fpr_t.size > 0 else float("nan")
    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50) if fpr_t.size > 0 else float("nan")
    fpr30_hlt = fpr_at_target_tpr(fpr_h, tpr_h, 0.30) if fpr_h.size > 0 else float("nan")
    fpr50_hlt = fpr_at_target_tpr(fpr_h, tpr_h, 0.50) if fpr_h.size > 0 else float("nan")

    summary = {
        "run_dir": str(run_dir),
        "analysis_dir": str(out_dir),
        "target_tpr": float(args.target_tpr),
        "threshold_source": str(args.threshold_source),
        "thresholds": {
            "teacher_threshold": float(thr_teacher),
            "hlt_threshold": float(thr_hlt),
        },
        "metrics": {
            "val_auc_teacher": float(auc_val_teacher),
            "val_auc_hlt": float(auc_val_hlt),
            "test_auc_teacher": float(auc_test_teacher),
            "test_auc_hlt": float(auc_test_hlt),
            "test_fpr30_teacher": float(fpr30_teacher),
            "test_fpr50_teacher": float(fpr50_teacher),
            "test_fpr30_hlt": float(fpr30_hlt),
            "test_fpr50_hlt": float(fpr50_hlt),
        },
        "counts": {
            "n_test": int(y_test.size),
            "teacher_right_hlt_wrong": int(idx_teacher_right_hlt_wrong.size),
            "hlt_right_teacher_wrong": int(idx_hlt_right_teacher_wrong.size),
            "both_right": int(idx_both_right.size),
            "both_wrong": int(idx_both_wrong.size),
            "critical_neg": int(idx_critical_neg.size),
            "critical_pos": int(idx_critical_pos.size),
            "hlt_rescue_neg": int(idx_hlt_rescue_neg.size),
            "hlt_rescue_pos": int(idx_hlt_rescue_pos.size),
        },
        "hlt_generation_stats": hlt_stats,
        "subsets": subset_summaries,
    }

    # Per-jet table for all test jets.
    per_jet_cols = {
        "test_local_idx": np.arange(y_test.size, dtype=np.int64),
        "global_idx": test_idx.astype(np.int64),
        "label": y_test.astype(np.int64),
        "p_teacher": p_test_teacher.astype(np.float32),
        "p_hlt": p_test_hlt.astype(np.float32),
        "pred_teacher": pred_teacher.astype(np.int64),
        "pred_hlt": pred_hlt.astype(np.int64),
        "teacher_right": teacher_right.astype(np.int64),
        "hlt_right": hlt_right.astype(np.int64),
        "teacher_right_hlt_wrong": (teacher_right & (~hlt_right)).astype(np.int64),
        "hlt_right_teacher_wrong": (hlt_right & (~teacher_right)).astype(np.int64),
        "critical_neg": ((y_test == 0) & teacher_right & (~hlt_right)).astype(np.int64),
        "critical_pos": ((y_test == 1) & teacher_right & (~hlt_right)).astype(np.int64),
        "n_const_offline": n_off.astype(np.float32),
        "n_const_hlt": n_hlt.astype(np.float32),
        "n_const_missing": n_missing.astype(np.float32),
        "merge_lost_true": merge_lost.astype(np.float32),
        "eff_lost_true": eff_lost.astype(np.float32),
        "jet_pt_offline": pt_off.astype(np.float32),
        "jet_pt_hlt": pt_hlt.astype(np.float32),
        "jet_pt_response_hlt_over_offline": pt_resp.astype(np.float32),
        "jet_mass_offline": m_off.astype(np.float32),
        "jet_mass_hlt": m_hlt.astype(np.float32),
        "jet_mass_response_hlt_over_offline": m_resp.astype(np.float32),
        "p_gap_teacher_minus_hlt": p_gap.astype(np.float32),
    }
    write_per_jet_tsv(out_dir / "test_per_jet_disagreement_metrics.tsv", per_jet_cols)
    np.savez_compressed(out_dir / "test_per_jet_disagreement_metrics.npz", **per_jet_cols)

    def export_subset(name: str, idx_local: np.ndarray) -> Dict[str, object]:
        if idx_local.size == 0:
            return {"name": name, "n_selected": 0, "path": None}
        score = np.abs(p_gap[idx_local])
        order = np.argsort(-score)
        idx_sorted = idx_local[order]
        if args.export_all_subset_jets:
            keep = idx_sorted
        else:
            keep = idx_sorted[: int(max(1, args.max_export_per_subset))]
        gidx = test_idx[keep].astype(np.int64)
        path = out_dir / f"{name}_jets.npz"
        np.savez_compressed(
            path,
            test_local_idx=keep.astype(np.int64),
            global_idx=gidx,
            label=y_test[keep].astype(np.int64),
            p_teacher=p_test_teacher[keep].astype(np.float32),
            p_hlt=p_test_hlt[keep].astype(np.float32),
            pred_teacher=pred_teacher[keep].astype(np.int64),
            pred_hlt=pred_hlt[keep].astype(np.int64),
            const_offline=c_off_t[keep].astype(np.float32),
            mask_offline=m_off_t[keep].astype(bool),
            const_hlt=c_hlt_t[keep].astype(np.float32),
            mask_hlt=m_hlt_t[keep].astype(bool),
            n_const_offline=n_off[keep].astype(np.float32),
            n_const_hlt=n_hlt[keep].astype(np.float32),
            n_const_missing=n_missing[keep].astype(np.float32),
            merge_lost_true=merge_lost[keep].astype(np.float32),
            eff_lost_true=eff_lost[keep].astype(np.float32),
            jet_pt_offline=pt_off[keep].astype(np.float32),
            jet_pt_hlt=pt_hlt[keep].astype(np.float32),
            jet_pt_response_hlt_over_offline=pt_resp[keep].astype(np.float32),
        )
        return {"name": name, "n_selected": int(keep.size), "path": str(path)}

    exports = [
        export_subset("teacher_right_hlt_wrong", idx_teacher_right_hlt_wrong),
        export_subset("hlt_right_teacher_wrong", idx_hlt_right_teacher_wrong),
        export_subset("critical_neg_bg_teacher_right_hlt_fp", idx_critical_neg),
        export_subset("critical_pos_sig_teacher_right_hlt_fn", idx_critical_pos),
    ]
    summary["exports"] = exports

    with open(out_dir / "disagreement_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Disagreement analysis complete")
    print("=" * 70)
    print(f"Run dir: {run_dir}")
    print(f"Output dir: {out_dir}")
    print(
        "Counts | "
        f"T-right/H-wrong={summary['counts']['teacher_right_hlt_wrong']}, "
        f"H-right/T-wrong={summary['counts']['hlt_right_teacher_wrong']}, "
        f"critical_neg={summary['counts']['critical_neg']}, "
        f"critical_pos={summary['counts']['critical_pos']}"
    )
    print(f"Teacher/HLT thresholds @TPR={args.target_tpr:.2f}: {thr_teacher:.6f} / {thr_hlt:.6f}")
    print(f"Saved summary: {out_dir / 'disagreement_summary.json'}")


if __name__ == "__main__":
    main()


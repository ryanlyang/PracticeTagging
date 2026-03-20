#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended disagreement analysis for Teacher / HLT / Joint (Unmerge) models.

What this script does:
1. Loads a saved run directory (teacher.pt, baseline.pt, dual_joint.pt, offline_reconstructor.pt).
2. Rebuilds pseudo-HLT deterministically from raw offline jets.
3. Evaluates teacher, HLT baseline, and joint dual-view model on a fresh evaluation slice
   (default: 500k jets starting after original training range).
4. Builds rich per-jet exports and extensive bucket tables across:
   - HLT top-tagger score buckets (p_hlt)
   - Joint top-tagger score buckets (p_joint)
   - HLT data buckets (n_const_hlt, jet_pt_hlt, jet_mass_hlt)
   - combined buckets (count x score, count x pt, score x score, etc.)

Outputs are written under:
  <run_dir>/<save_subdir>/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly import (
    JointDualDataset,
    build_soft_corrected_view,
    eval_joint_model,
)
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
    fpr_at_target_tpr,
)
from unmerge_correct_hlt import (
    DualViewCrossAttnClassifier,
    JetDataset,
    ParticleTransformer,
    compute_features,
    compute_jet_pt,
    eval_classifier,
    load_raw_constituents_from_h5,
    standardize,
)


def _deepcopy_config() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _load_checkpoint(path: Path, map_location: torch.device) -> Dict:
    """Load checkpoint dict with compatibility across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    if len(np.unique(y.astype(np.int64))) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _threshold_at_target_tpr(probs: np.ndarray, labels: np.ndarray, target_tpr: float) -> float:
    pos = probs[labels == 1]
    if pos.size == 0:
        return 0.5
    q = float(np.clip(1.0 - float(target_tpr), 0.0, 1.0))
    return float(np.quantile(pos, q))


def _safe_stat(x: np.ndarray) -> Dict[str, float]:
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


def _jet_mass(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
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


def _write_tsv(path: Path, columns: Dict[str, np.ndarray]) -> None:
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


def _resolve_run_dir(run_dir_arg: str) -> Path:
    p = Path(run_dir_arg).expanduser()
    if p.exists():
        return p.resolve()
    # Convenience fallback: if user passes checkpoints/... name but only download_checkpoints exists locally.
    cand = Path.cwd() / "download_checkpoints" / p.name
    if cand.exists():
        return cand.resolve()
    raise FileNotFoundError(f"Run directory not found: {p}")


def _resolve_data_files(setup: Dict, explicit_data_file: Optional[str]) -> List[Path]:
    if explicit_data_file:
        p = Path(explicit_data_file).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--data_file not found: {p}")
        return [p]

    files = [Path(x).expanduser() for x in setup.get("train_files", [])]
    existing = [x.resolve() for x in files if x.exists()]
    if existing:
        return existing

    local_fallback = Path.cwd() / "data" / "test.h5"
    if local_fallback.exists():
        return [local_fallback.resolve()]

    raise FileNotFoundError(
        "No usable data file found. train_files from data_setup.json are missing and fallback data/test.h5 is absent."
    )


def _build_quantile_band_masks(
    arr: np.ndarray,
    quantiles: Sequence[float],
    value_name: str,
) -> List[Tuple[str, np.ndarray]]:
    q = np.asarray(quantiles, dtype=np.float64)
    q = q[(q >= 0.0) & (q <= 1.0)]
    q = np.unique(q)
    if q.size < 2:
        return []
    edges = np.unique(np.quantile(arr, q))
    if edges.size < 2:
        return []
    out: List[Tuple[str, np.ndarray]] = []
    for i in range(edges.size - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if hi <= lo:
            continue
        if i < edges.size - 2:
            m = (arr >= lo) & (arr < hi)
        else:
            m = (arr >= lo) & (arr <= hi)
        out.append((f"{lo:.6g}<={value_name}<{'=' if i == edges.size - 2 else ''}{hi:.6g}", m))
    return out


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


@torch.no_grad()
def _eval_recoonly_model(
    reconstructor: nn.Module,
    recoonly_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    reconstructor.eval()
    recoonly_model.eval()
    preds: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].detach().cpu().numpy().astype(np.float32)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=bool(corrected_use_flags),
        )
        logits = recoonly_model(feat_b, mask_b).squeeze(1)
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        preds.append(p)
        labs.append(y)

    if len(preds) == 0:
        return float("nan"), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64), float("nan")
    pred = np.concatenate(preds).astype(np.float32)
    lab = np.concatenate(labs).astype(np.int64)
    auc = _safe_auc(lab, pred)
    if np.isfinite(auc):
        fpr, tpr, _ = roc_curve(lab, pred)
        fpr50 = float(fpr_at_target_tpr(fpr, tpr, 0.50))
    else:
        fpr50 = float("nan")
    return float(auc), pred, lab, float(fpr50)


def _build_bucket_row(
    name: str,
    family: str,
    mask: np.ndarray,
    y: np.ndarray,
    p_teacher: np.ndarray,
    p_hlt: np.ndarray,
    p_joint: np.ndarray,
    pred_teacher: np.ndarray,
    pred_hlt: np.ndarray,
    pred_joint: np.ndarray,
    teacher_right: np.ndarray,
    hlt_right: np.ndarray,
    joint_right: np.ndarray,
    global_counts: Dict[str, int],
) -> Dict[str, object]:
    idx = mask
    n = int(np.sum(idx))
    yb = y[idx]
    n_pos = int(np.sum(yb == 1))
    n_neg = int(np.sum(yb == 0))

    p_t = p_teacher[idx]
    p_h = p_hlt[idx]
    p_j = p_joint[idx]
    pr_t = pred_teacher[idx]
    pr_h = pred_hlt[idx]
    pr_j = pred_joint[idx]
    r_t = teacher_right[idx]
    r_h = hlt_right[idx]
    r_j = joint_right[idx]

    auc_t = _safe_auc(yb, p_t)
    auc_h = _safe_auc(yb, p_h)
    auc_j = _safe_auc(yb, p_j)

    neg_mask = (yb == 0)
    pos_mask = (yb == 1)
    fp_t = int(np.sum(neg_mask & (pr_t == 1)))
    fp_h = int(np.sum(neg_mask & (pr_h == 1)))
    fp_j = int(np.sum(neg_mask & (pr_j == 1)))
    tp_t = int(np.sum(pos_mask & (pr_t == 1)))
    tp_h = int(np.sum(pos_mask & (pr_h == 1)))
    tp_j = int(np.sum(pos_mask & (pr_j == 1)))

    # Pairwise disagreement directions in-bucket.
    t_right_h_wrong = int(np.sum(r_t & (~r_h)))
    h_right_t_wrong = int(np.sum(r_h & (~r_t)))
    t_right_j_wrong = int(np.sum(r_t & (~r_j)))
    j_right_t_wrong = int(np.sum(r_j & (~r_t)))
    j_right_h_wrong = int(np.sum(r_j & (~r_h)))
    h_right_j_wrong = int(np.sum(r_h & (~r_j)))

    # Critical negatives (background where A right and B wrong).
    cn_t_h = int(np.sum((yb == 0) & r_t & (~r_h)))
    cn_t_j = int(np.sum((yb == 0) & r_t & (~r_j)))
    cn_j_h = int(np.sum((yb == 0) & r_j & (~r_h)))

    return {
        "bucket": name,
        "family": family,
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "traffic": float(n / max(1, y.size)),
        "auc_teacher": auc_t,
        "auc_hlt": auc_h,
        "auc_joint": auc_j,
        "auc_gap_teacher_minus_hlt": float(auc_t - auc_h) if np.isfinite(auc_t) and np.isfinite(auc_h) else float("nan"),
        "auc_gap_teacher_minus_joint": float(auc_t - auc_j) if np.isfinite(auc_t) and np.isfinite(auc_j) else float("nan"),
        "auc_gap_joint_minus_hlt": float(auc_j - auc_h) if np.isfinite(auc_j) and np.isfinite(auc_h) else float("nan"),
        "fpr_teacher": _safe_rate(fp_t, n_neg),
        "fpr_hlt": _safe_rate(fp_h, n_neg),
        "fpr_joint": _safe_rate(fp_j, n_neg),
        "tpr_teacher": _safe_rate(tp_t, n_pos),
        "tpr_hlt": _safe_rate(tp_h, n_pos),
        "tpr_joint": _safe_rate(tp_j, n_pos),
        "fpr_gap_hlt_minus_teacher": _safe_rate(fp_h, n_neg) - _safe_rate(fp_t, n_neg) if n_neg > 0 else float("nan"),
        "fpr_gap_hlt_minus_joint": _safe_rate(fp_h, n_neg) - _safe_rate(fp_j, n_neg) if n_neg > 0 else float("nan"),
        "delta_teacher_hlt": _safe_rate(t_right_h_wrong - h_right_t_wrong, n),
        "delta_teacher_joint": _safe_rate(t_right_j_wrong - j_right_t_wrong, n),
        "delta_joint_hlt": _safe_rate(j_right_h_wrong - h_right_j_wrong, n),
        "critical_neg_teacher_over_hlt": _safe_rate(cn_t_h, n_neg),
        "critical_neg_teacher_over_joint": _safe_rate(cn_t_j, n_neg),
        "critical_neg_joint_over_hlt": _safe_rate(cn_j_h, n_neg),
        "capture_critical_neg_teacher_over_hlt": _safe_rate(cn_t_h, global_counts["cn_t_h"]),
        "capture_critical_neg_teacher_over_joint": _safe_rate(cn_t_j, global_counts["cn_t_j"]),
        "capture_critical_neg_joint_over_hlt": _safe_rate(cn_j_h, global_counts["cn_j_h"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default="checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_1MJ100C",
        help="Run directory containing teacher.pt, baseline.pt, dual_joint.pt, offline_reconstructor.pt.",
    )
    parser.add_argument("--save_subdir", type=str, default="teacher_hlt_joint_disagreement_analysis_500k")
    parser.add_argument("--data_file", type=str, default=None, help="Optional override data h5 path.")
    parser.add_argument("--n_eval_jets", type=int, default=500000)
    parser.add_argument(
        "--offset_eval_jets",
        type=int,
        default=-1,
        help="Start offset for fresh evaluation slice; negative -> setup.offset_jets + setup.n_train_jets.",
    )
    parser.add_argument("--max_constits", type=int, default=-1)
    parser.add_argument("--target_tpr", type=float, default=0.50)
    parser.add_argument("--threshold_source", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--threshold_val_frac", type=float, default=0.20)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bucket_min_count", type=int, default=1000)
    parser.add_argument("--bucket_min_pos", type=int, default=300)
    parser.add_argument("--bucket_min_neg", type=int, default=300)
    parser.add_argument("--max_export_per_subset", type=int, default=100000)
    parser.add_argument("--export_all_subset_jets", action="store_true")
    parser.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    parser.add_argument("--joint_mode", type=str, default="dual", choices=["dual", "recoonly"])
    parser.add_argument("--dual_ckpt_name", type=str, default="dual_joint.pt")
    parser.add_argument("--joint_recoonly_ckpt_name", type=str, default="recoonly_classifier.pt")
    parser.add_argument("--reco_ckpt_name", type=str, default="offline_reconstructor.pt")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    out_dir = (run_dir / args.save_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_path = run_dir / "data_setup.json"
    splits_path = run_dir / "data_splits.npz"
    teacher_path = run_dir / "teacher.pt"
    baseline_path = run_dir / "baseline.pt"
    dual_path = run_dir / args.dual_ckpt_name
    joint_recoonly_path = run_dir / args.joint_recoonly_ckpt_name
    reco_path = run_dir / args.reco_ckpt_name

    required = [setup_path, splits_path, teacher_path, baseline_path, reco_path]
    if str(args.joint_mode).lower() == "dual":
        required.append(dual_path)
    else:
        required.append(joint_recoonly_path)

    for fp in required:
        if not fp.exists():
            raise FileNotFoundError(f"Required file not found: {fp}")

    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    splits = np.load(splits_path, allow_pickle=False)
    means = splits["means"].astype(np.float32)
    stds = splits["stds"].astype(np.float32)

    cfg = _deepcopy_config()
    if "hlt_effects" in setup:
        cfg["hlt_effects"].update(setup["hlt_effects"])

    seed = int(setup.get("seed", 52))
    n_train_jets_setup = int(setup.get("n_train_jets", 0))
    offset_setup = int(setup.get("offset_jets", 0))
    if int(args.offset_eval_jets) < 0:
        offset_eval = int(offset_setup + n_train_jets_setup)
    else:
        offset_eval = int(args.offset_eval_jets)
    n_eval_jets = int(args.n_eval_jets)
    max_constits = int(setup.get("max_constits", 100) if int(args.max_constits) <= 0 else int(args.max_constits))
    data_files = _resolve_data_files(setup, args.data_file)

    max_jets_needed = int(offset_eval + n_eval_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        data_files,
        max_jets=max_jets_needed,
        max_constits=max_constits,
    )
    n_available = int(all_const_full.shape[0])
    if offset_eval >= n_available:
        raise RuntimeError(
            f"offset_eval_jets={offset_eval} exceeds available jets={n_available} in data source."
        )
    end_eval = min(offset_eval + n_eval_jets, n_available)
    if end_eval - offset_eval < n_eval_jets:
        print(
            f"Warning: requested n_eval_jets={n_eval_jets}, available={end_eval - offset_eval}. Using available slice."
        )
    const_raw = all_const_full[offset_eval:end_eval]
    labels = all_labels_full[offset_eval:end_eval].astype(np.int64)

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

    print("Computing standardized features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)
    feat_off_std = standardize(feat_off, masks_off, means, stds).astype(np.float32)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds).astype(np.float32)

    # Non-priv unmerge-only targets needed for JointDualDataset (not used directly in eval path, but required by dataset).
    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    added_target_scale = 1.0
    metrics_path = run_dir / "joint_stage_metrics.json"
    if metrics_path.exists():
        try:
            jm = json.loads(metrics_path.read_text(encoding="utf-8"))
            added_target_scale = float(jm.get("variant", {}).get("added_target_scale", 1.0))
        except Exception:
            added_target_scale = 1.0
    budget_merge_true = (added_target_scale * true_added_raw).astype(np.float32)
    budget_eff_true = np.zeros_like(true_added_raw, dtype=np.float32)

    device = torch.device(args.device)
    bs = int(cfg["training"]["batch_size"]) if int(args.batch_size) <= 0 else int(args.batch_size)

    # Build threshold source split.
    n_eval = int(labels.shape[0])
    all_idx = np.arange(n_eval, dtype=np.int64)
    if args.threshold_source == "val":
        frac = float(np.clip(args.threshold_val_frac, 0.01, 0.90))
        n_val = max(1, int(round(frac * n_eval)))
        rng = np.random.default_rng(seed + 31415)
        perm = rng.permutation(n_eval)
        idx_val = perm[:n_val]
        idx_test = perm[n_val:]
        if idx_test.size == 0:
            idx_test = idx_val.copy()
    else:
        idx_val = all_idx
        idx_test = all_idx

    # -------------------- Load models -------------------- #
    print("Loading teacher/HLT/joint checkpoints...")
    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher_ckpt = _load_checkpoint(teacher_path, device)
    baseline_ckpt = _load_checkpoint(baseline_path, device)
    reco_ckpt = _load_checkpoint(reco_path, device)

    teacher.load_state_dict(teacher_ckpt["model"])
    baseline.load_state_dict(baseline_ckpt["model"])
    reco_state = reco_ckpt["model"]

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(reco_state)
    joint_mode = str(args.joint_mode).lower()
    corrected_use_flags = False
    feat_hlt_dual = feat_hlt_std.astype(np.float32, copy=True)
    dual_joint = None
    recoonly_model = None

    if joint_mode == "dual":
        dual_ckpt = _load_checkpoint(dual_path, device)
        dual_state = dual_ckpt["model"]

        # Infer dual input dimensions from checkpoint.
        key_a = "input_proj_a.0.weight"
        key_b = "input_proj_b.0.weight"
        if key_a not in dual_state or key_b not in dual_state:
            raise RuntimeError("Could not infer dual input dims from dual checkpoint.")
        dual_input_dim_a = int(dual_state[key_a].shape[1])
        dual_input_dim_b = int(dual_state[key_b].shape[1])
        corrected_use_flags = bool(dual_input_dim_b == 12)

        if feat_hlt_dual.shape[-1] != dual_input_dim_a:
            raise RuntimeError(
                "Dual input_dim_a mismatch. "
                f"Checkpoint expects {dual_input_dim_a}, but analysis currently builds {feat_hlt_dual.shape[-1]}. "
                "This usually means the original run used extra dual features (e.g., jet regressor)."
            )

        dual_joint = DualViewCrossAttnClassifier(
            input_dim_a=dual_input_dim_a,
            input_dim_b=dual_input_dim_b,
            **cfg["model"],
        ).to(device)
        dual_joint.load_state_dict(dual_state)
    else:
        recoonly_ckpt = _load_checkpoint(joint_recoonly_path, device)
        recoonly_state = recoonly_ckpt["model"]
        key_x = "input_proj.0.weight"
        if key_x not in recoonly_state:
            raise RuntimeError(
                f"Could not infer reco-only input dim from checkpoint {joint_recoonly_path}; missing {key_x}."
            )
        recoonly_input_dim = int(recoonly_state[key_x].shape[1])
        corrected_use_flags = bool(recoonly_input_dim == 12)
        recoonly_model = ParticleTransformer(input_dim=recoonly_input_dim, **cfg["model"]).to(device)
        recoonly_model.load_state_dict(recoonly_state)

    # -------------------- Eval probs -------------------- #
    def eval_single(feat: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ds = JetDataset(feat, mask, labels)
        dl = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
        )
        _, p, y_ = eval_classifier(teacher if feat is feat_off_std else baseline, dl, device)
        return p.astype(np.float32), y_.astype(np.int64)

    print("Evaluating teacher on offline view...")
    ds_eval_off = JetDataset(feat_off_std, masks_off, labels)
    dl_eval_off = DataLoader(
        ds_eval_off,
        batch_size=bs,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    auc_teacher, p_teacher, y_teacher = eval_classifier(teacher, dl_eval_off, device)
    p_teacher = p_teacher.astype(np.float32)
    y_teacher = y_teacher.astype(np.int64)

    print("Evaluating HLT baseline...")
    ds_eval_hlt = JetDataset(feat_hlt_std, hlt_mask, labels)
    dl_eval_hlt = DataLoader(
        ds_eval_hlt,
        batch_size=bs,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    auc_hlt, p_hlt, y_hlt = eval_classifier(baseline, dl_eval_hlt, device)
    p_hlt = p_hlt.astype(np.float32)
    y_hlt = y_hlt.astype(np.int64)

    if not np.array_equal(y_teacher, y_hlt):
        raise RuntimeError("Label mismatch between teacher and HLT eval outputs.")
    y_eval = y_teacher.astype(np.int64)

    if joint_mode == "dual":
        print("Evaluating joint dual-view model...")
    else:
        print("Evaluating joint reco-only corrected-view model...")
    ds_eval_joint = JointDualDataset(
        feat_hlt_reco=feat_hlt_std,
        feat_hlt_dual=feat_hlt_dual,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        const_off=const_off,
        mask_off=masks_off,
        budget_merge_true=budget_merge_true,
        budget_eff_true=budget_eff_true,
        labels=labels,
    )
    dl_eval_joint = DataLoader(
        ds_eval_joint,
        batch_size=bs,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    if joint_mode == "dual":
        auc_joint, p_joint, y_joint, fpr50_joint_direct = eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_joint,
            loader=dl_eval_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=corrected_use_flags,
        )
    else:
        auc_joint, p_joint, y_joint, fpr50_joint_direct = _eval_recoonly_model(
            reconstructor=reconstructor,
            recoonly_model=recoonly_model,
            loader=dl_eval_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=corrected_use_flags,
        )
    p_joint = p_joint.astype(np.float32)
    y_joint = y_joint.astype(np.int64)
    if not np.array_equal(y_eval, y_joint):
        raise RuntimeError("Label mismatch between joint and teacher/HLT eval outputs.")

    # -------------------- Thresholds / decisions -------------------- #
    y_src = y_eval[idx_val]
    thr_teacher = _threshold_at_target_tpr(p_teacher[idx_val], y_src, float(args.target_tpr))
    thr_hlt = _threshold_at_target_tpr(p_hlt[idx_val], y_src, float(args.target_tpr))
    thr_joint = _threshold_at_target_tpr(p_joint[idx_val], y_src, float(args.target_tpr))

    pred_teacher = (p_teacher >= thr_teacher).astype(np.int64)
    pred_hlt = (p_hlt >= thr_hlt).astype(np.int64)
    pred_joint = (p_joint >= thr_joint).astype(np.int64)

    teacher_right = (pred_teacher == y_eval)
    hlt_right = (pred_hlt == y_eval)
    joint_right = (pred_joint == y_eval)

    # Pairwise key subsets.
    idx_t_right_h_wrong = np.where(teacher_right & (~hlt_right))[0]
    idx_h_right_t_wrong = np.where(hlt_right & (~teacher_right))[0]
    idx_t_right_j_wrong = np.where(teacher_right & (~joint_right))[0]
    idx_j_right_t_wrong = np.where(joint_right & (~teacher_right))[0]
    idx_j_right_h_wrong = np.where(joint_right & (~hlt_right))[0]
    idx_h_right_j_wrong = np.where(hlt_right & (~joint_right))[0]

    idx_cn_t_h = np.where((y_eval == 0) & teacher_right & (~hlt_right))[0]
    idx_cn_t_j = np.where((y_eval == 0) & teacher_right & (~joint_right))[0]
    idx_cn_j_h = np.where((y_eval == 0) & joint_right & (~hlt_right))[0]

    # -------------------- Jet fields -------------------- #
    n_off = masks_off.sum(axis=1).astype(np.float32)
    n_hlt = hlt_mask.sum(axis=1).astype(np.float32)
    n_missing = np.maximum(n_off - n_hlt, 0.0).astype(np.float32)
    merge_lost = budget_truth["merge_lost_per_jet"].astype(np.float32)
    eff_lost = budget_truth["eff_lost_per_jet"].astype(np.float32)

    pt_off = compute_jet_pt(const_off, masks_off).astype(np.float32)
    pt_hlt = compute_jet_pt(hlt_const, hlt_mask).astype(np.float32)
    pt_resp = (pt_hlt / np.maximum(pt_off, 1e-8)).astype(np.float32)

    mass_off = _jet_mass(const_off, masks_off)
    mass_hlt = _jet_mass(hlt_const, hlt_mask)
    mass_resp = (mass_hlt / np.maximum(mass_off, 1e-8)).astype(np.float32)

    p_gap_t_h = (p_teacher - p_hlt).astype(np.float32)
    p_gap_t_j = (p_teacher - p_joint).astype(np.float32)
    p_gap_j_h = (p_joint - p_hlt).astype(np.float32)

    # -------------------- Core metrics -------------------- #
    def model_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
        auc = _safe_auc(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        return {
            "auc": float(auc),
            "fpr30": float(fpr_at_target_tpr(fpr, tpr, 0.30)),
            "fpr50": float(fpr_at_target_tpr(fpr, tpr, 0.50)),
        }

    m_teacher = model_metrics(y_eval[idx_test], p_teacher[idx_test])
    m_hlt_metrics = model_metrics(y_eval[idx_test], p_hlt[idx_test])
    m_joint = model_metrics(y_eval[idx_test], p_joint[idx_test])

    # -------------------- Per-jet export -------------------- #
    per_jet_cols = {
        "eval_local_idx": np.arange(y_eval.size, dtype=np.int64),
        "global_idx": np.arange(offset_eval, offset_eval + y_eval.size, dtype=np.int64),
        "label": y_eval.astype(np.int64),
        "p_teacher": p_teacher.astype(np.float32),
        "p_hlt": p_hlt.astype(np.float32),
        "p_joint": p_joint.astype(np.float32),
        "pred_teacher": pred_teacher.astype(np.int64),
        "pred_hlt": pred_hlt.astype(np.int64),
        "pred_joint": pred_joint.astype(np.int64),
        "teacher_right": teacher_right.astype(np.int64),
        "hlt_right": hlt_right.astype(np.int64),
        "joint_right": joint_right.astype(np.int64),
        "teacher_right_hlt_wrong": (teacher_right & (~hlt_right)).astype(np.int64),
        "hlt_right_teacher_wrong": (hlt_right & (~teacher_right)).astype(np.int64),
        "teacher_right_joint_wrong": (teacher_right & (~joint_right)).astype(np.int64),
        "joint_right_teacher_wrong": (joint_right & (~teacher_right)).astype(np.int64),
        "joint_right_hlt_wrong": (joint_right & (~hlt_right)).astype(np.int64),
        "hlt_right_joint_wrong": (hlt_right & (~joint_right)).astype(np.int64),
        "critical_neg_teacher_over_hlt": ((y_eval == 0) & teacher_right & (~hlt_right)).astype(np.int64),
        "critical_neg_teacher_over_joint": ((y_eval == 0) & teacher_right & (~joint_right)).astype(np.int64),
        "critical_neg_joint_over_hlt": ((y_eval == 0) & joint_right & (~hlt_right)).astype(np.int64),
        "n_const_offline": n_off.astype(np.float32),
        "n_const_hlt": n_hlt.astype(np.float32),
        "n_const_missing": n_missing.astype(np.float32),
        "merge_lost_true": merge_lost.astype(np.float32),
        "eff_lost_true": eff_lost.astype(np.float32),
        "jet_pt_offline": pt_off.astype(np.float32),
        "jet_pt_hlt": pt_hlt.astype(np.float32),
        "jet_pt_response_hlt_over_offline": pt_resp.astype(np.float32),
        "jet_mass_offline": mass_off.astype(np.float32),
        "jet_mass_hlt": mass_hlt.astype(np.float32),
        "jet_mass_response_hlt_over_offline": mass_resp.astype(np.float32),
        "p_gap_teacher_minus_hlt": p_gap_t_h.astype(np.float32),
        "p_gap_teacher_minus_joint": p_gap_t_j.astype(np.float32),
        "p_gap_joint_minus_hlt": p_gap_j_h.astype(np.float32),
    }
    _write_tsv(out_dir / "eval_per_jet_teacher_hlt_joint.tsv", per_jet_cols)
    np.savez_compressed(out_dir / "eval_per_jet_teacher_hlt_joint.npz", **per_jet_cols)

    # -------------------- Subset exports -------------------- #
    def export_subset(name: str, idx_local: np.ndarray) -> Dict[str, object]:
        if idx_local.size == 0:
            return {"name": name, "n_selected": 0, "path": None}
        score = np.abs(p_gap_t_h[idx_local]) + np.abs(p_gap_t_j[idx_local]) + np.abs(p_gap_j_h[idx_local])
        order = np.argsort(-score)
        idx_sorted = idx_local[order]
        keep = idx_sorted if args.export_all_subset_jets else idx_sorted[: int(max(1, args.max_export_per_subset))]
        gidx = np.arange(offset_eval, offset_eval + y_eval.size, dtype=np.int64)[keep]
        path = out_dir / f"{name}_jets.npz"
        np.savez_compressed(
            path,
            eval_local_idx=keep.astype(np.int64),
            global_idx=gidx.astype(np.int64),
            label=y_eval[keep].astype(np.int64),
            p_teacher=p_teacher[keep].astype(np.float32),
            p_hlt=p_hlt[keep].astype(np.float32),
            p_joint=p_joint[keep].astype(np.float32),
            pred_teacher=pred_teacher[keep].astype(np.int64),
            pred_hlt=pred_hlt[keep].astype(np.int64),
            pred_joint=pred_joint[keep].astype(np.int64),
            const_offline=const_off[keep].astype(np.float32),
            mask_offline=masks_off[keep].astype(bool),
            const_hlt=hlt_const[keep].astype(np.float32),
            mask_hlt=hlt_mask[keep].astype(bool),
            n_const_offline=n_off[keep].astype(np.float32),
            n_const_hlt=n_hlt[keep].astype(np.float32),
            jet_pt_offline=pt_off[keep].astype(np.float32),
            jet_pt_hlt=pt_hlt[keep].astype(np.float32),
            jet_mass_offline=mass_off[keep].astype(np.float32),
            jet_mass_hlt=mass_hlt[keep].astype(np.float32),
        )
        return {"name": name, "n_selected": int(keep.size), "path": str(path)}

    exports = [
        export_subset("teacher_right_hlt_wrong", idx_t_right_h_wrong),
        export_subset("hlt_right_teacher_wrong", idx_h_right_t_wrong),
        export_subset("teacher_right_joint_wrong", idx_t_right_j_wrong),
        export_subset("joint_right_teacher_wrong", idx_j_right_t_wrong),
        export_subset("joint_right_hlt_wrong", idx_j_right_h_wrong),
        export_subset("hlt_right_joint_wrong", idx_h_right_j_wrong),
        export_subset("critical_neg_teacher_over_hlt", idx_cn_t_h),
        export_subset("critical_neg_teacher_over_joint", idx_cn_t_j),
        export_subset("critical_neg_joint_over_hlt", idx_cn_j_h),
    ]

    # -------------------- Bucket tables -------------------- #
    print("Computing extensive bucket metrics...")
    yb = y_eval
    p_t = p_teacher
    p_h = p_hlt
    p_j = p_joint
    pr_t = pred_teacher
    pr_h = pred_hlt
    pr_j = pred_joint
    r_t = teacher_right
    r_h = hlt_right
    r_j = joint_right

    global_counts = {
        "cn_t_h": int(len(idx_cn_t_h)),
        "cn_t_j": int(len(idx_cn_t_j)),
        "cn_j_h": int(len(idx_cn_j_h)),
    }

    bucket_rows: List[Dict[str, object]] = []

    def maybe_add(name: str, family: str, mask: np.ndarray) -> None:
        n = int(mask.sum())
        if n < int(args.bucket_min_count):
            return
        n_pos = int(np.sum(yb[mask] == 1))
        n_neg = int(np.sum(yb[mask] == 0))
        if n_pos < int(args.bucket_min_pos) or n_neg < int(args.bucket_min_neg):
            return
        bucket_rows.append(
            _build_bucket_row(
                name=name,
                family=family,
                mask=mask,
                y=yb,
                p_teacher=p_t,
                p_hlt=p_h,
                p_joint=p_j,
                pred_teacher=pr_t,
                pred_hlt=pr_h,
                pred_joint=pr_j,
                teacher_right=r_t,
                hlt_right=r_h,
                joint_right=r_j,
                global_counts=global_counts,
            )
        )

    # HLT-data families.
    count_edges = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 150, 200], dtype=np.float32)
    for i in range(len(count_edges) - 1):
        lo = float(count_edges[i])
        hi = float(count_edges[i + 1])
        m = (n_hlt > lo) & (n_hlt <= hi)
        maybe_add(f"{lo:.0f}<n_const_hlt<={hi:.0f}", "hlt_count_band", m)

    for k in count_edges[1:]:
        m = (n_hlt > 0.0) & (n_hlt <= float(k))
        maybe_add(f"0<n_const_hlt<={float(k):.0f}", "hlt_count_cum", m)

    for name, m in _build_quantile_band_masks(pt_hlt, np.linspace(0.0, 1.0, 21), "jet_pt_hlt"):
        maybe_add(name, "hlt_pt_quantile_band", m)
    for name, m in _build_quantile_band_masks(mass_hlt, np.linspace(0.0, 1.0, 21), "jet_mass_hlt"):
        maybe_add(name, "hlt_mass_quantile_band", m)

    # HLT-score families.
    for name, m in _build_quantile_band_masks(p_h, np.linspace(0.0, 1.0, 21), "p_hlt"):
        maybe_add(name, "hlt_score_quantile_band", m)
    hlt_thr_quantiles = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]
    for q in hlt_thr_quantiles:
        thr = float(np.quantile(p_h, q))
        maybe_add(f"p_hlt>={thr:.6g} (q{q:.2f})", "hlt_score_threshold", p_h >= thr)

    # Joint-score families.
    for name, m in _build_quantile_band_masks(p_j, np.linspace(0.0, 1.0, 21), "p_joint"):
        maybe_add(name, "joint_score_quantile_band", m)
    for q in hlt_thr_quantiles:
        thr = float(np.quantile(p_j, q))
        maybe_add(f"p_joint>={thr:.6g} (q{q:.2f})", "joint_score_threshold", p_j >= thr)

    # Combined families.
    coarse_count = np.array([0, 10, 20, 30, 40, 50, 80, 200], dtype=np.float32)
    p_h_edges = np.unique(np.quantile(p_h, np.linspace(0.0, 1.0, 11)))
    p_j_edges = np.unique(np.quantile(p_j, np.linspace(0.0, 1.0, 11)))
    pt_edges = np.unique(np.quantile(pt_hlt, np.linspace(0.0, 1.0, 9)))

    for i in range(len(coarse_count) - 1):
        clo = float(coarse_count[i])
        chi = float(coarse_count[i + 1])
        count_mask = (n_hlt > clo) & (n_hlt <= chi)
        for j in range(len(p_h_edges) - 1):
            slo = float(p_h_edges[j])
            shi = float(p_h_edges[j + 1])
            if j < len(p_h_edges) - 2:
                s_mask = (p_h >= slo) & (p_h < shi)
            else:
                s_mask = (p_h >= slo) & (p_h <= shi)
            maybe_add(
                f"{clo:.0f}<n_const_hlt<={chi:.0f} & {slo:.6g}<=p_hlt<{'=' if j == len(p_h_edges)-2 else ''}{shi:.6g}",
                "count_x_hlt_score_band",
                count_mask & s_mask,
            )
        for j in range(len(p_j_edges) - 1):
            slo = float(p_j_edges[j])
            shi = float(p_j_edges[j + 1])
            if j < len(p_j_edges) - 2:
                s_mask = (p_j >= slo) & (p_j < shi)
            else:
                s_mask = (p_j >= slo) & (p_j <= shi)
            maybe_add(
                f"{clo:.0f}<n_const_hlt<={chi:.0f} & {slo:.6g}<=p_joint<{'=' if j == len(p_j_edges)-2 else ''}{shi:.6g}",
                "count_x_joint_score_band",
                count_mask & s_mask,
            )
        for j in range(len(pt_edges) - 1):
            plo = float(pt_edges[j])
            phi = float(pt_edges[j + 1])
            if j < len(pt_edges) - 2:
                p_mask = (pt_hlt >= plo) & (pt_hlt < phi)
            else:
                p_mask = (pt_hlt >= plo) & (pt_hlt <= phi)
            maybe_add(
                f"{clo:.0f}<n_const_hlt<={chi:.0f} & {plo:.6g}<=jet_pt_hlt<{'=' if j == len(pt_edges)-2 else ''}{phi:.6g}",
                "count_x_hlt_pt_band",
                count_mask & p_mask,
            )

    # Cross-score band family (HLT score x Joint score).
    coarse_q = np.linspace(0.0, 1.0, 6)
    p_h_c = np.unique(np.quantile(p_h, coarse_q))
    p_j_c = np.unique(np.quantile(p_j, coarse_q))
    for i in range(len(p_h_c) - 1):
        h_lo = float(p_h_c[i])
        h_hi = float(p_h_c[i + 1])
        if i < len(p_h_c) - 2:
            m_h_band = (p_h >= h_lo) & (p_h < h_hi)
        else:
            m_h_band = (p_h >= h_lo) & (p_h <= h_hi)
        for j in range(len(p_j_c) - 1):
            j_lo = float(p_j_c[j])
            j_hi = float(p_j_c[j + 1])
            if j < len(p_j_c) - 2:
                mjm = (p_j >= j_lo) & (p_j < j_hi)
            else:
                mjm = (p_j >= j_lo) & (p_j <= j_hi)
            maybe_add(
                f"{h_lo:.6g}<=p_hlt<{'=' if i == len(p_h_c)-2 else ''}{h_hi:.6g} & "
                f"{j_lo:.6g}<=p_joint<{'=' if j == len(p_j_c)-2 else ''}{j_hi:.6g}",
                "hlt_score_x_joint_score_band",
                m_h_band & mjm,
            )

    if len(bucket_rows) == 0:
        bucket_df_out = None
    else:
        import pandas as pd

        bucket_df_out = pd.DataFrame(bucket_rows)
        bucket_df_out.sort_values(["family", "n"], ascending=[True, False], inplace=True)
        bucket_df_out.to_csv(out_dir / "bucket_metrics_all.tsv", sep="\t", index=False)
        for fam, fam_df in bucket_df_out.groupby("family", sort=True):
            fam_path = out_dir / f"bucket_metrics_{fam}.tsv"
            fam_df.sort_values("n", ascending=False).to_csv(fam_path, sep="\t", index=False)

    # -------------------- Summary json -------------------- #
    summary = {
        "run_dir": str(run_dir),
        "analysis_dir": str(out_dir),
        "joint_mode": str(joint_mode),
        "data_file_used": [str(x) for x in data_files],
        "n_eval": int(n_eval),
        "offset_eval_jets": int(offset_eval),
        "target_tpr": float(args.target_tpr),
        "threshold_source": str(args.threshold_source),
        "thresholds": {
            "teacher": float(thr_teacher),
            "hlt": float(thr_hlt),
            "joint": float(thr_joint),
        },
        "metrics": {
            "teacher": {
                "auc": float(m_teacher["auc"]),
                "fpr30": float(m_teacher["fpr30"]),
                "fpr50": float(m_teacher["fpr50"]),
            },
            "hlt": {
                "auc": float(m_hlt_metrics["auc"]),
                "fpr30": float(m_hlt_metrics["fpr30"]),
                "fpr50": float(m_hlt_metrics["fpr50"]),
            },
            "joint": {
                "auc": float(m_joint["auc"]),
                "fpr30": float(m_joint["fpr30"]),
                "fpr50": float(m_joint["fpr50"]),
                "fpr50_direct_eval_joint_model": float(fpr50_joint_direct),
            },
        },
        "counts": {
            "teacher_right_hlt_wrong": int(idx_t_right_h_wrong.size),
            "hlt_right_teacher_wrong": int(idx_h_right_t_wrong.size),
            "teacher_right_joint_wrong": int(idx_t_right_j_wrong.size),
            "joint_right_teacher_wrong": int(idx_j_right_t_wrong.size),
            "joint_right_hlt_wrong": int(idx_j_right_h_wrong.size),
            "hlt_right_joint_wrong": int(idx_h_right_j_wrong.size),
            "critical_neg_teacher_over_hlt": int(idx_cn_t_h.size),
            "critical_neg_teacher_over_joint": int(idx_cn_t_j.size),
            "critical_neg_joint_over_hlt": int(idx_cn_j_h.size),
        },
        "hlt_generation_stats": hlt_stats,
        "subset_exports": exports,
        "bucket_metrics_files": (
            sorted([str(x.name) for x in out_dir.glob("bucket_metrics_*.tsv")])
            if len(bucket_rows) > 0
            else []
        ),
    }
    with open(out_dir / "disagreement_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Teacher/HLT/Joint disagreement analysis complete")
    print("=" * 72)
    print(f"Run dir: {run_dir}")
    print(f"Output dir: {out_dir}")
    print(
        "Test metrics | "
        f"Teacher AUC={summary['metrics']['teacher']['auc']:.4f} FPR50={summary['metrics']['teacher']['fpr50']:.6f} | "
        f"HLT AUC={summary['metrics']['hlt']['auc']:.4f} FPR50={summary['metrics']['hlt']['fpr50']:.6f} | "
        f"Joint AUC={summary['metrics']['joint']['auc']:.4f} FPR50={summary['metrics']['joint']['fpr50']:.6f}"
    )
    print(
        "Disagreements | "
        f"T>H={summary['counts']['teacher_right_hlt_wrong']} "
        f"H>T={summary['counts']['hlt_right_teacher_wrong']} "
        f"T>J={summary['counts']['teacher_right_joint_wrong']} "
        f"J>H={summary['counts']['joint_right_hlt_wrong']}"
    )
    print(f"Thresholds @TPR={args.target_tpr:.2f}: T={thr_teacher:.6f} H={thr_hlt:.6f} J={thr_joint:.6f}")
    if len(bucket_rows) > 0:
        print(f"Bucket rows exported: {len(bucket_rows)}")
        print(f"All buckets table: {out_dir / 'bucket_metrics_all.tsv'}")
    print(f"Summary: {out_dir / 'disagreement_summary.json'}")


if __name__ == "__main__":
    main()

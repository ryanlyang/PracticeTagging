#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finalize m5 fusion-score artifacts after an OOM in diagnostics/final export.

This script does not train. It reloads the same data split, loads the saved
teacher, HLT baseline, Stage-B, and Stage-C checkpoints, evaluates val/test
scores, and writes the missing fusion_scores_val_test.npz, results.npz, and
joint_stage_metrics.json artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as m5
from offline_reconstructor_no_gt_local30kv2 import (
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
    get_stats,
    standardize,
)


BASE = "checkpoints/reco_teacher_joint_fusion_6model_150k75k150k"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Finalize m5 fusion scores from saved checkpoints.")
    ap.add_argument("--train_path", default="./data/train_quarter.h5")
    ap.add_argument("--use_train_weights", action="store_true")
    ap.add_argument("--save_dir", default=f"{BASE}/model5_joint_s01_full_weighted_5m1m1m")
    ap.add_argument("--run_name", default="model5_joint_s01_full_weighted_5m1m1m_seed0")
    ap.add_argument("--n_train_jets", type=int, default=7000000)
    ap.add_argument("--n_train_split", type=int, default=5000000)
    ap.add_argument("--n_val_split", type=int, default=1000000)
    ap.add_argument("--n_test_split", type=int, default=1000000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=-1)
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    ap.add_argument("--use_corrected_flags", action="store_true")
    ap.add_argument("--added_target_scale", type=float, default=0.90)
    ap.add_argument("--report_target_tpr", type=float, default=0.50)
    ap.add_argument("--teacher_ckpt", default="")
    ap.add_argument("--baseline_ckpt", default="")
    ap.add_argument("--stage2_reco_ckpt", default="")
    ap.add_argument("--stage2_dual_ckpt", default="")
    ap.add_argument("--stage2_reco_fpr_ckpt", default="")
    ap.add_argument("--stage2_dual_fpr_ckpt", default="")
    ap.add_argument("--stageC_reco_ckpt", default="")
    ap.add_argument("--stageC_dual_ckpt", default="")
    ap.add_argument("--stageC_reco_fpr_ckpt", default="")
    ap.add_argument("--stageC_dual_fpr_ckpt", default="")
    return ap


def _parse_h5_path_arg(raw: str) -> list[Path]:
    p = Path(str(raw)).expanduser()
    if p.is_dir():
        files = sorted(p.glob("*.h5"))
    else:
        files = [Path(x.strip()).expanduser() for x in str(raw).split(",") if x.strip()]
    if not files:
        raise FileNotFoundError(f"No HDF5 input files found for --train_path={raw!r}")
    return files


def _resolve_path(raw: str, default: Path) -> Path:
    return Path(raw).expanduser().resolve() if str(raw).strip() else default.resolve()


def _resolve_ckpts(args: argparse.Namespace, save_root: Path) -> Dict[str, Path]:
    defaults = {
        "teacher_ckpt": save_root / "teacher.pt",
        "baseline_ckpt": save_root / "baseline.pt",
        "stage2_reco_ckpt": save_root / "offline_reconstructor_stage2.pt",
        "stage2_dual_ckpt": save_root / "dual_joint_stage2.pt",
        "stage2_reco_fpr_ckpt": save_root / "offline_reconstructor_stage2_bestfpr50.pt",
        "stage2_dual_fpr_ckpt": save_root / "dual_joint_stage2_bestfpr50.pt",
        "stageC_reco_ckpt": save_root / "offline_reconstructor_stageC_selected_pre_eval.pt",
        "stageC_dual_ckpt": save_root / "dual_joint_stageC_selected_pre_eval.pt",
        "stageC_reco_fpr_ckpt": save_root / "offline_reconstructor_stageC_bestfpr50_pre_eval.pt",
        "stageC_dual_fpr_ckpt": save_root / "dual_joint_stageC_bestfpr50_pre_eval.pt",
    }
    return {key: _resolve_path(str(getattr(args, key)), default) for key, default in defaults.items()}


def _require_inputs(paths: Dict[str, Path]) -> None:
    required = [
        "teacher_ckpt",
        "baseline_ckpt",
        "stage2_reco_ckpt",
        "stage2_dual_ckpt",
        "stageC_reco_ckpt",
        "stageC_dual_ckpt",
    ]
    missing = [f"{key}: {paths[key]}" for key in required if not paths[key].is_file()]
    if missing:
        raise FileNotFoundError("Missing required checkpoints:\n" + "\n".join(missing))


def _load_raw(args: argparse.Namespace, cfg: Dict) -> Dict[str, np.ndarray]:
    train_files = _parse_h5_path_arg(str(args.train_path))
    max_jets_needed = int(args.offset_jets) + int(args.n_train_jets)
    print("Loading offline constituents...", flush=True)
    all_const_full, all_labels_full, all_train_w_full = m5.load_raw_constituents_labels_weights_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=int(args.max_constits),
        use_train_weights=bool(args.use_train_weights),
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}")

    start = int(args.offset_jets)
    stop = start + int(args.n_train_jets)
    const_raw = all_const_full[start:stop]
    labels = all_labels_full[start:stop].astype(np.int64, copy=True)
    del all_labels_full, all_train_w_full
    gc.collect()

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0
    del all_const_full, const_raw, raw_mask
    gc.collect()

    print("Generating pseudo-HLT...", flush=True)
    hlt_const, hlt_mask, _hlt_stats, _budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )
    true_added_raw = np.maximum(
        masks_off.sum(axis=1).astype(np.float32) - hlt_mask.sum(axis=1).astype(np.float32),
        0.0,
    ).astype(np.float32)
    rho = m5._clamp_target_scale(float(args.added_target_scale))
    print(
        f"Non-priv rho split setup: rho={rho:.3f}, "
        f"mean_true_added_raw={float(true_added_raw.mean()):.3f}",
        flush=True,
    )
    return {
        "labels": labels,
        "const_off": const_off,
        "masks_off": masks_off,
        "hlt_const": hlt_const,
        "hlt_mask": hlt_mask,
        "true_added_raw": true_added_raw,
        "rho": np.array(float(rho), dtype=np.float32),
    }


def _load_or_rebuild_split_stats(
    args: argparse.Namespace,
    save_root: Path,
    labels: np.ndarray,
    const_off: np.ndarray,
    masks_off: np.ndarray,
) -> Dict[str, np.ndarray]:
    split_path = save_root / "data_splits.npz"
    if split_path.is_file():
        z = np.load(split_path)
        required = ["train_idx", "val_idx", "test_idx", "means", "stds"]
        missing = [k for k in required if k not in z]
        if missing:
            raise KeyError(f"{split_path} is missing required keys: {missing}")
        print(f"Loaded exact split/statistics from {split_path}", flush=True)
        return {k: np.asarray(z[k]) for k in required}

    print("WARNING: data_splits.npz missing; rebuilding split/statistics from seed.", flush=True)
    idx = np.arange(len(labels))
    total_need = int(args.n_train_split) + int(args.n_val_split) + int(args.n_test_split)
    if total_need > len(idx):
        raise ValueError(f"Requested split counts exceed available jets: {total_need} > {len(idx)}")
    if total_need < len(idx):
        idx_use, _ = train_test_split(
            idx,
            train_size=total_need,
            random_state=int(args.seed),
            stratify=labels[idx],
        )
    else:
        idx_use = idx
    train_idx, rem_idx = train_test_split(
        idx_use,
        train_size=int(args.n_train_split),
        random_state=int(args.seed),
        stratify=labels[idx_use],
    )
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=int(args.n_val_split),
        test_size=int(args.n_test_split),
        random_state=int(args.seed),
        stratify=labels[rem_idx],
    )
    print("Computing train features for fallback statistics...", flush=True)
    feat_off_train = compute_features(const_off[train_idx], masks_off[train_idx])
    local_idx = np.arange(len(train_idx), dtype=np.int64)
    means, stds = get_stats(feat_off_train, masks_off[train_idx], local_idx)
    del feat_off_train
    gc.collect()
    return {
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "means": means.astype(np.float32),
        "stds": stds.astype(np.float32),
    }


def _standardized_features(const: np.ndarray, mask: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    feat = compute_features(const, mask)
    out = standardize(feat, mask, means, stds)
    del feat
    gc.collect()
    return out.astype(np.float32, copy=False)


def _eval_single(
    model: torch.nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    ds = JetDataset(feat, mask, labels)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=torch.cuda.is_available())
    return eval_classifier(model, dl, device)


def _make_joint_loader(
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    const_off: np.ndarray,
    mask_off: np.ndarray,
    budget_merge_true: np.ndarray,
    budget_eff_true: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    ds = m5.JointDualDataset(
        feat_hlt,
        feat_hlt,
        mask_hlt,
        const_hlt,
        const_off,
        mask_off,
        budget_merge_true,
        budget_eff_true,
        labels,
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=torch.cuda.is_available())


def _load_model(model: torch.nn.Module, ckpt: Path, device: torch.device) -> None:
    state, _meta = m5._load_model_state_from_checkpoint(ckpt, device)
    model.load_state_dict(state, strict=True)


def _eval_pair(
    reco: torch.nn.Module,
    dual: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    return m5.eval_joint_model(
        reco,
        dual,
        loader,
        device,
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=bool(corrected_use_flags),
    )


def _fpr_curves(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    fpr, tpr, _ = roc_curve(labels, np.asarray(scores, dtype=np.float64))
    return fpr, tpr, float(fpr_at_target_tpr(fpr, tpr, 0.30)), float(fpr_at_target_tpr(fpr, tpr, 0.50))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64))) if len(np.unique(labels)) > 1 else float("nan")


def main() -> None:
    args = _build_parser().parse_args()
    m5.set_seed(int(args.seed))
    cfg = m5._deepcopy_config()
    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    ckpts = _resolve_ckpts(args, save_root)
    _require_inputs(ckpts)

    device = torch.device(str(args.device))
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg["training"]["batch_size"])
    print(f"Device: {device}", flush=True)
    print(f"Save dir: {save_root}", flush=True)

    raw = _load_raw(args, cfg)
    split = _load_or_rebuild_split_stats(
        args,
        save_root,
        raw["labels"],
        raw["const_off"],
        raw["masks_off"],
    )
    val_idx = np.asarray(split["val_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    means = np.asarray(split["means"], dtype=np.float32)
    stds = np.asarray(split["stds"], dtype=np.float32)
    print(f"Split sizes: Val={len(val_idx)}, Test={len(test_idx)}", flush=True)

    y_val = raw["labels"][val_idx].astype(np.float32)
    y_test = raw["labels"][test_idx].astype(np.float32)
    rho = float(raw["rho"])
    budget_merge_val = (rho * raw["true_added_raw"][val_idx]).astype(np.float32)
    budget_merge_test = (rho * raw["true_added_raw"][test_idx]).astype(np.float32)
    budget_eff_val = ((1.0 - rho) * raw["true_added_raw"][val_idx]).astype(np.float32)
    budget_eff_test = ((1.0 - rho) * raw["true_added_raw"][test_idx]).astype(np.float32)

    print("Computing val/test standardized features...", flush=True)
    feat_off_val = _standardized_features(raw["const_off"][val_idx], raw["masks_off"][val_idx], means, stds)
    feat_hlt_val = _standardized_features(raw["hlt_const"][val_idx], raw["hlt_mask"][val_idx], means, stds)
    feat_off_test = _standardized_features(raw["const_off"][test_idx], raw["masks_off"][test_idx], means, stds)
    feat_hlt_test = _standardized_features(raw["hlt_const"][test_idx], raw["hlt_mask"][test_idx], means, stds)

    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    _load_model(teacher, ckpts["teacher_ckpt"], device)
    _load_model(baseline, ckpts["baseline_ckpt"], device)

    print("Evaluating teacher/HLT validation...", flush=True)
    auc_teacher_val, preds_teacher_val, labs_teacher_val = _eval_single(
        teacher, feat_off_val, raw["masks_off"][val_idx], raw["labels"][val_idx], batch_size, device, int(args.num_workers)
    )
    auc_hlt_val, preds_hlt_val, labs_hlt_val = _eval_single(
        baseline, feat_hlt_val, raw["hlt_mask"][val_idx], raw["labels"][val_idx], batch_size, device, int(args.num_workers)
    )
    print("Evaluating teacher/HLT test...", flush=True)
    auc_teacher_test, preds_teacher_test, labs_teacher_test = _eval_single(
        teacher, feat_off_test, raw["masks_off"][test_idx], raw["labels"][test_idx], batch_size, device, int(args.num_workers)
    )
    auc_hlt_test, preds_hlt_test, labs_hlt_test = _eval_single(
        baseline, feat_hlt_test, raw["hlt_mask"][test_idx], raw["labels"][test_idx], batch_size, device, int(args.num_workers)
    )
    for name, labs, target in [
        ("teacher_val", labs_teacher_val, y_val),
        ("hlt_val", labs_hlt_val, y_val),
        ("teacher_test", labs_teacher_test, y_test),
        ("hlt_test", labs_hlt_test, y_test),
    ]:
        if not np.array_equal(np.asarray(labs, dtype=np.float32), target):
            raise RuntimeError(f"{name} labels mismatch.")

    del teacher, baseline, feat_off_val, feat_off_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    dual_input_dim_a = int(feat_hlt_val.shape[-1])
    dual_input_dim_b = 12 if bool(args.use_corrected_flags) else 10
    dual_joint = DualViewCrossAttnClassifier(input_dim_a=dual_input_dim_a, input_dim_b=dual_input_dim_b, **cfg["model"]).to(device)

    print("Building validation joint loader...", flush=True)
    dl_val = _make_joint_loader(
        feat_hlt_val,
        raw["hlt_mask"][val_idx],
        raw["hlt_const"][val_idx],
        raw["const_off"][val_idx],
        raw["masks_off"][val_idx],
        budget_merge_val,
        budget_eff_val,
        raw["labels"][val_idx],
        batch_size,
        int(args.num_workers),
    )
    print("Evaluating Stage2 validation...", flush=True)
    _load_model(reconstructor, ckpts["stage2_reco_ckpt"], device)
    _load_model(dual_joint, ckpts["stage2_dual_ckpt"], device)
    auc_stage2_val, preds_stage2_val, labs_stage2_val, fpr50_stage2_val = _eval_pair(
        reconstructor,
        dual_joint,
        dl_val,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    print("Evaluating StageC validation...", flush=True)
    _load_model(reconstructor, ckpts["stageC_reco_ckpt"], device)
    _load_model(dual_joint, ckpts["stageC_dual_ckpt"], device)
    auc_joint_val, preds_joint_val, labs_joint_val, fpr50_joint_val = _eval_pair(
        reconstructor,
        dual_joint,
        dl_val,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    del dl_val, feat_hlt_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Building test joint loader...", flush=True)
    dl_test = _make_joint_loader(
        feat_hlt_test,
        raw["hlt_mask"][test_idx],
        raw["hlt_const"][test_idx],
        raw["const_off"][test_idx],
        raw["masks_off"][test_idx],
        budget_merge_test,
        budget_eff_test,
        raw["labels"][test_idx],
        batch_size,
        int(args.num_workers),
    )
    print("Evaluating Stage2 test...", flush=True)
    _load_model(reconstructor, ckpts["stage2_reco_ckpt"], device)
    _load_model(dual_joint, ckpts["stage2_dual_ckpt"], device)
    auc_stage2_test, preds_stage2_test, labs_stage2_test, _ = _eval_pair(
        reconstructor,
        dual_joint,
        dl_test,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )

    auc_stage2_fprsel = float("nan")
    preds_stage2_fprsel = None
    labs_stage2_fprsel = None
    if ckpts["stage2_reco_fpr_ckpt"].is_file() and ckpts["stage2_dual_fpr_ckpt"].is_file():
        print("Evaluating Stage2 best-FPR test...", flush=True)
        _load_model(reconstructor, ckpts["stage2_reco_fpr_ckpt"], device)
        _load_model(dual_joint, ckpts["stage2_dual_fpr_ckpt"], device)
        auc_stage2_fprsel, preds_stage2_fprsel, labs_stage2_fprsel, _ = _eval_pair(
            reconstructor,
            dual_joint,
            dl_test,
            device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )

    print("Evaluating StageC test...", flush=True)
    _load_model(reconstructor, ckpts["stageC_reco_ckpt"], device)
    _load_model(dual_joint, ckpts["stageC_dual_ckpt"], device)
    auc_joint_test, preds_joint_test, labs_joint_test, _ = _eval_pair(
        reconstructor,
        dual_joint,
        dl_test,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )

    auc_joint_fprsel = float("nan")
    preds_joint_fprsel = None
    labs_joint_fprsel = None
    if ckpts["stageC_reco_fpr_ckpt"].is_file() and ckpts["stageC_dual_fpr_ckpt"].is_file():
        print("Evaluating StageC best-FPR test...", flush=True)
        _load_model(reconstructor, ckpts["stageC_reco_fpr_ckpt"], device)
        _load_model(dual_joint, ckpts["stageC_dual_fpr_ckpt"], device)
        auc_joint_fprsel, preds_joint_fprsel, labs_joint_fprsel, _ = _eval_pair(
            reconstructor,
            dual_joint,
            dl_test,
            device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
    del dl_test, feat_hlt_test, reconstructor, dual_joint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for name, labs, target in [
        ("stage2_val", labs_stage2_val, y_val),
        ("joint_val", labs_joint_val, y_val),
        ("stage2_test", labs_stage2_test, y_test),
        ("joint_test", labs_joint_test, y_test),
        ("stage2_bestfpr_test", labs_stage2_fprsel, y_test),
        ("joint_bestfpr_test", labs_joint_fprsel, y_test),
    ]:
        if labs is not None and not np.array_equal(np.asarray(labs, dtype=np.float32), target):
            raise RuntimeError(f"{name} labels mismatch.")

    fpr_t, tpr_t, fpr30_teacher, fpr50_teacher = _fpr_curves(y_test, preds_teacher_test)
    fpr_b, tpr_b, fpr30_hlt, fpr50_hlt = _fpr_curves(y_test, preds_hlt_test)
    fpr_s2, tpr_s2, fpr30_stage2, fpr50_stage2 = _fpr_curves(y_test, preds_stage2_test)
    fpr_j, tpr_j, fpr30_joint, fpr50_joint = _fpr_curves(y_test, preds_joint_test)
    if preds_stage2_fprsel is not None:
        fpr_s2_fprsel, tpr_s2_fprsel, fpr30_stage2_fprsel, fpr50_stage2_fprsel = _fpr_curves(y_test, preds_stage2_fprsel)
    else:
        fpr_s2_fprsel, tpr_s2_fprsel = np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        fpr30_stage2_fprsel, fpr50_stage2_fprsel = float("nan"), float("nan")
    if preds_joint_fprsel is not None:
        fpr_j_fprsel, tpr_j_fprsel, fpr30_joint_fprsel, fpr50_joint_fprsel = _fpr_curves(y_test, preds_joint_fprsel)
    else:
        fpr_j_fprsel, tpr_j_fprsel = np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        fpr30_joint_fprsel, fpr50_joint_fprsel = float("nan"), float("nan")

    fusion_path = save_root / "fusion_scores_val_test.npz"
    np.savez_compressed(
        fusion_path,
        labels_val=y_val.astype(np.float32),
        labels_test=y_test.astype(np.float32),
        preds_teacher_val=np.asarray(preds_teacher_val, dtype=np.float64),
        preds_teacher_test=np.asarray(preds_teacher_test, dtype=np.float64),
        preds_hlt_val=np.asarray(preds_hlt_val, dtype=np.float64),
        preds_hlt_test=np.asarray(preds_hlt_test, dtype=np.float64),
        preds_stage2_val=np.asarray(preds_stage2_val, dtype=np.float64),
        preds_stage2_test=np.asarray(preds_stage2_test, dtype=np.float64),
        preds_joint_val=np.asarray(preds_joint_val, dtype=np.float64),
        preds_joint_test=np.asarray(preds_joint_test, dtype=np.float64),
        auc_teacher_val=float(auc_teacher_val),
        auc_teacher_test=float(auc_teacher_test),
        auc_hlt_val=float(auc_hlt_val),
        auc_hlt_test=float(auc_hlt_test),
        auc_stage2_val=float(auc_stage2_val),
        auc_stage2_test=float(auc_stage2_test),
        auc_joint_val=float(auc_joint_val),
        auc_joint_test=float(auc_joint_test),
        fpr50_stage2_val=float(fpr50_stage2_val),
        fpr50_stage2_test=float(fpr50_stage2),
        fpr50_joint_val=float(fpr50_joint_val),
        fpr50_joint_test=float(fpr50_joint),
        hlt_nconst_val=np.asarray(raw["hlt_mask"][val_idx].sum(axis=1), dtype=np.int32),
        hlt_nconst_test=np.asarray(raw["hlt_mask"][test_idx].sum(axis=1), dtype=np.float32),
        hlt_jet_pt_val=np.asarray(compute_jet_pt(raw["hlt_const"][val_idx], raw["hlt_mask"][val_idx]), dtype=np.float64),
        hlt_jet_pt_test=np.asarray(compute_jet_pt(raw["hlt_const"][test_idx], raw["hlt_mask"][test_idx]), dtype=np.float64),
        off_jet_pt_val=np.asarray(compute_jet_pt(raw["const_off"][val_idx], raw["masks_off"][val_idx]), dtype=np.float64),
        off_jet_pt_test=np.asarray(compute_jet_pt(raw["const_off"][test_idx], raw["masks_off"][test_idx]), dtype=np.float64),
        target_tpr=np.array(float(args.report_target_tpr), dtype=np.float64),
    )

    np.savez(
        save_root / "results.npz",
        auc_teacher=float(auc_teacher_test),
        auc_baseline=float(auc_hlt_test),
        auc_reco_only=float("nan"),
        auc_stage2=float(auc_stage2_test),
        auc_stage2_fprsel=float(auc_stage2_fprsel),
        auc_joint=float(auc_joint_test),
        auc_joint_fprsel=float(auc_joint_fprsel),
        auc_joint_kd=float("nan"),
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_stage2=fpr_s2,
        tpr_stage2=tpr_s2,
        fpr_reco_only=np.array([], dtype=np.float64),
        tpr_reco_only=np.array([], dtype=np.float64),
        fpr_stage2_fprsel=fpr_s2_fprsel,
        tpr_stage2_fprsel=tpr_s2_fprsel,
        fpr_joint=fpr_j,
        tpr_joint=tpr_j,
        fpr_joint_fprsel=fpr_j_fprsel,
        tpr_joint_fprsel=tpr_j_fprsel,
        fpr_joint_kd=np.array([], dtype=np.float64),
        tpr_joint_kd=np.array([], dtype=np.float64),
        fpr30_teacher=float(fpr30_teacher),
        fpr30_baseline=float(fpr30_hlt),
        fpr30_reco_only=float("nan"),
        fpr30_stage2=float(fpr30_stage2),
        fpr30_stage2_fprsel=float(fpr30_stage2_fprsel),
        fpr30_joint=float(fpr30_joint),
        fpr30_joint_fprsel=float(fpr30_joint_fprsel),
        fpr30_joint_kd=float("nan"),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_baseline=float(fpr50_hlt),
        fpr50_reco_only=float("nan"),
        fpr50_stage2=float(fpr50_stage2),
        fpr50_stage2_fprsel=float(fpr50_stage2_fprsel),
        fpr50_joint=float(fpr50_joint),
        fpr50_joint_fprsel=float(fpr50_joint_fprsel),
        fpr50_joint_kd=float("nan"),
        rho=float(rho),
    )

    metrics = {
        "finalized_from_checkpoints": True,
        "checkpoints": {k: str(v) for k, v in ckpts.items()},
        "variant": {
            "mode": "nopriv_rhosplit_splitagain",
            "rho": float(rho),
            "split_again": {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in m5.SPLIT_AGAIN_CFG.items()
            },
            "mean_true_added_raw": float(np.mean(raw["true_added_raw"])),
            "mean_target_merge": float(np.mean(rho * raw["true_added_raw"])),
            "mean_target_eff": float(np.mean((1.0 - rho) * raw["true_added_raw"])),
        },
        "test_stage2": {
            "auc_stage2": float(auc_stage2_test),
            "auc_stage2_fprsel": float(auc_stage2_fprsel) if preds_stage2_fprsel is not None else None,
            "fpr30_stage2": float(fpr30_stage2),
            "fpr30_stage2_fprsel": float(fpr30_stage2_fprsel) if preds_stage2_fprsel is not None else None,
            "fpr50_stage2": float(fpr50_stage2),
            "fpr50_stage2_fprsel": float(fpr50_stage2_fprsel) if preds_stage2_fprsel is not None else None,
        },
        "test": {
            "auc_teacher": float(auc_teacher_test),
            "auc_baseline": float(auc_hlt_test),
            "auc_reco_only": None,
            "auc_stage2": float(auc_stage2_test),
            "auc_stage2_fprsel": float(auc_stage2_fprsel) if preds_stage2_fprsel is not None else None,
            "auc_joint": float(auc_joint_test),
            "auc_joint_fprsel": float(auc_joint_fprsel) if preds_joint_fprsel is not None else None,
            "auc_joint_kd": None,
            "fpr30_teacher": float(fpr30_teacher),
            "fpr30_baseline": float(fpr30_hlt),
            "fpr30_reco_only": None,
            "fpr30_stage2": float(fpr30_stage2),
            "fpr30_stage2_fprsel": float(fpr30_stage2_fprsel) if preds_stage2_fprsel is not None else None,
            "fpr30_joint": float(fpr30_joint),
            "fpr30_joint_fprsel": float(fpr30_joint_fprsel) if preds_joint_fprsel is not None else None,
            "fpr30_joint_kd": None,
            "fpr50_teacher": float(fpr50_teacher),
            "fpr50_baseline": float(fpr50_hlt),
            "fpr50_reco_only": None,
            "fpr50_stage2": float(fpr50_stage2),
            "fpr50_stage2_fprsel": float(fpr50_stage2_fprsel) if preds_stage2_fprsel is not None else None,
            "fpr50_joint": float(fpr50_joint),
            "fpr50_joint_fprsel": float(fpr50_joint_fprsel) if preds_joint_fprsel is not None else None,
            "fpr50_joint_kd": None,
        },
        "validation": {
            "auc_teacher": float(auc_teacher_val),
            "auc_baseline": float(auc_hlt_val),
            "auc_stage2": float(auc_stage2_val),
            "auc_joint": float(auc_joint_val),
            "fpr50_stage2": float(fpr50_stage2_val),
            "fpr50_joint": float(fpr50_joint_val),
        },
    }
    with open(save_root / "joint_stage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("FINALIZED TEST EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher_test:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_hlt_test:.4f}")
    print(f"Stage2 (PreJoint) AUC: {auc_stage2_test:.4f}")
    print(f"Stage2 (BestValFPR50) AUC: {auc_stage2_fprsel:.4f}")
    print(f"Joint Dual-View  AUC: {auc_joint_test:.4f}")
    print(f"Joint Dual-View (BestValFPR50) AUC: {auc_joint_fprsel:.4f}")
    print(
        f"FPR@30 Teacher/Baseline/Stage2/Joint: "
        f"{fpr30_teacher:.6f} / {fpr30_hlt:.6f} / {fpr30_stage2:.6f} / {fpr30_joint:.6f}"
    )
    print(
        f"FPR@30 Stage2BestFPR / JointBestFPR: "
        f"{fpr30_stage2_fprsel:.6f} / {fpr30_joint_fprsel:.6f}"
    )
    print(
        f"FPR@50 Teacher/Baseline/Stage2/Joint: "
        f"{fpr50_teacher:.6f} / {fpr50_hlt:.6f} / {fpr50_stage2:.6f} / {fpr50_joint:.6f}"
    )
    print(
        f"FPR@50 Stage2BestFPR / JointBestFPR: "
        f"{fpr50_stage2_fprsel:.6f} / {fpr50_joint_fprsel:.6f}"
    )
    print(f"Saved fusion scores: {fusion_path}")
    print(f"Saved results:       {save_root / 'results.npz'}")
    print(f"Saved metrics:       {save_root / 'joint_stage_metrics.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finalize m2 delta020 fusion-score artifacts after an OOM in diagnostics.

This script does not train. It reloads the dataset/split, loads the saved
Stage-B and Stage-C checkpoints, evaluates val/test scores, and writes the
missing fusion_scores_val_test.npz plus lightweight results summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as m2
from unmerge_correct_hlt import (
    DualViewCrossAttnClassifier,
    compute_features,
    compute_jet_pt,
    get_stats,
    standardize,
)
from offline_reconstructor_no_gt_local30kv2 import (
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
    fpr_at_target_tpr,
)


BASE = "checkpoints/reco_teacher_joint_fusion_6model_150k75k150k"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Finalize m2j20 fusion scores from saved checkpoints.")
    ap.add_argument("--train_path", default="./data/train_quarter.h5")
    ap.add_argument("--use_train_weights", action="store_true")
    ap.add_argument("--force_m5_step1", action="store_true")
    ap.add_argument("--save_dir", default=f"{BASE}/model2_joint_delta020_weighted_5m1m1m")
    ap.add_argument("--run_name", default="model2_joint_delta020_weighted_5m1m1m_seed0")
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
    ap.add_argument(
        "--reference_npz",
        default=(
            f"{BASE}/model2_joint_delta005_weighted_5m1m1m/"
            "model2_joint_delta005_weighted_5m1m1m_seed0/fusion_scores_val_test.npz"
        ),
        help="Reference score file supplying labels, teacher, and HLT scores.",
    )
    ap.add_argument("--stage2_reco_ckpt", default="")
    ap.add_argument("--stage2_dual_ckpt", default="")
    ap.add_argument("--stage2_reco_fpr_ckpt", default="")
    ap.add_argument("--stage2_dual_fpr_ckpt", default="")
    ap.add_argument("--stageC_reco_ckpt", default="")
    ap.add_argument("--stageC_dual_ckpt", default="")
    ap.add_argument("--stageC_reco_fpr_ckpt", default="")
    ap.add_argument("--stageC_dual_fpr_ckpt", default="")
    return ap


def _resolve_ckpts(args: argparse.Namespace, save_root: Path) -> Dict[str, Path]:
    defaults = {
        "stage2_reco_ckpt": save_root / "offline_reconstructor_stage2.pt",
        "stage2_dual_ckpt": save_root / "dual_joint_stage2.pt",
        "stage2_reco_fpr_ckpt": save_root / "offline_reconstructor_stage2_bestfpr50.pt",
        "stage2_dual_fpr_ckpt": save_root / "dual_joint_stage2_bestfpr50.pt",
        "stageC_reco_ckpt": save_root / "offline_reconstructor_stageC_selected_pre_diagnostics.pt",
        "stageC_dual_ckpt": save_root / "dual_joint_stageC_selected_pre_diagnostics.pt",
        "stageC_reco_fpr_ckpt": save_root / "offline_reconstructor_stageC_bestfpr50_pre_diagnostics.pt",
        "stageC_dual_fpr_ckpt": save_root / "dual_joint_stageC_bestfpr50_pre_diagnostics.pt",
    }
    out: Dict[str, Path] = {}
    for key, default in defaults.items():
        raw = str(getattr(args, key)).strip()
        out[key] = Path(raw).expanduser().resolve() if raw else default.resolve()
    return out


def _load_data(args: argparse.Namespace, cfg: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train_files = m2._parse_h5_path_arg(str(args.train_path))
    max_jets_needed = int(args.offset_jets) + int(args.n_train_jets)
    print("Loading offline constituents...", flush=True)
    all_const_full, all_labels_full, all_train_w_full = m2.load_raw_constituents_labels_weights_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=int(args.max_constits),
        use_train_weights=bool(args.use_train_weights),
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}")

    const_raw = all_const_full[int(args.offset_jets) : int(args.offset_jets) + int(args.n_train_jets)]
    labels = all_labels_full[int(args.offset_jets) : int(args.offset_jets) + int(args.n_train_jets)].astype(np.int64)
    train_weight = all_train_w_full[int(args.offset_jets) : int(args.offset_jets) + int(args.n_train_jets)].astype(np.float32)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...", flush=True)
    hlt_const, hlt_mask, _hlt_stats, _budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )

    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    added_target_scale = float(np.clip(float(args.added_target_scale), 0.0, 1.0))
    budget_merge_true = (added_target_scale * true_added_raw).astype(np.float32)
    budget_eff_true = np.zeros_like(true_added_raw, dtype=np.float32)

    print("Computing features and split...", flush=True)
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

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
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}", flush=True)

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)
    feat_hlt_dual = feat_hlt_std.astype(np.float32, copy=True)

    arrays = {
        "labels": labels,
        "const_off": const_off,
        "masks_off": masks_off,
        "hlt_const": hlt_const,
        "hlt_mask": hlt_mask,
        "feat_hlt_std": feat_hlt_std,
        "feat_hlt_dual": feat_hlt_dual,
        "budget_merge_true": budget_merge_true,
        "budget_eff_true": budget_eff_true,
        "train_weight": train_weight,
    }
    splits = {
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "means": means.astype(np.float32),
        "stds": stds.astype(np.float32),
    }
    return arrays, splits


def _make_loader(arrays: Dict[str, np.ndarray], idx: np.ndarray, batch_size: int, num_workers: int) -> DataLoader:
    ds = m2.JointDualDataset(
        arrays["feat_hlt_std"][idx],
        arrays["feat_hlt_dual"][idx],
        arrays["hlt_mask"][idx],
        arrays["hlt_const"][idx],
        arrays["const_off"][idx],
        arrays["masks_off"][idx],
        arrays["budget_merge_true"][idx],
        arrays["budget_eff_true"][idx],
        arrays["labels"][idx],
        sample_weight_cls=np.ones((len(idx),), dtype=np.float32),
        sample_weight_reco=np.ones((len(idx),), dtype=np.float32),
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=torch.cuda.is_available())


def _load_pair(
    reco: torch.nn.Module,
    dual: torch.nn.Module,
    reco_ckpt: Path,
    dual_ckpt: Path,
    device: torch.device,
) -> None:
    if not reco_ckpt.exists():
        raise FileNotFoundError(f"Missing reconstructor checkpoint: {reco_ckpt}")
    if not dual_ckpt.exists():
        raise FileNotFoundError(f"Missing dual checkpoint: {dual_ckpt}")
    reco.load_state_dict(m2._load_checkpoint_model_state(str(reco_ckpt), device), strict=True)
    dual.load_state_dict(m2._load_checkpoint_model_state(str(dual_ckpt), device), strict=True)


def _eval_pair(
    reco: torch.nn.Module,
    dual: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    return m2.eval_joint_model(
        reco,
        dual,
        loader,
        device,
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=bool(corrected_use_flags),
    )


def _auc_from_ref(z: np.lib.npyio.NpzFile, key_auc: str, key_label: str, key_pred: str) -> float:
    if key_auc in z:
        return float(z[key_auc])
    return float(roc_auc_score(np.asarray(z[key_label], dtype=np.float32), np.asarray(z[key_pred], dtype=np.float64)))


def main() -> None:
    args = _build_parser().parse_args()
    m2.set_seed(int(args.seed))
    cfg = m2._deepcopy_config()
    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    ckpts = _resolve_ckpts(args, save_root)
    device = torch.device(str(args.device))
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg["training"]["batch_size"])

    print(f"Device: {device}")
    print(f"Save dir: {save_root}")
    print(f"Reference NPZ: {args.reference_npz}")
    arrays, splits = _load_data(args, cfg)

    ref_path = Path(args.reference_npz).expanduser().resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference NPZ with teacher/HLT scores: {ref_path}")
    zref = np.load(ref_path)
    required_ref = [
        "labels_val",
        "labels_test",
        "preds_teacher_val",
        "preds_teacher_test",
        "preds_hlt_val",
        "preds_hlt_test",
    ]
    missing = [k for k in required_ref if k not in zref]
    if missing:
        raise KeyError(f"Reference NPZ missing required keys {missing}: {ref_path}")

    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]
    y_val = arrays["labels"][val_idx].astype(np.float32)
    y_test = arrays["labels"][test_idx].astype(np.float32)
    if not np.array_equal(y_val, np.asarray(zref["labels_val"], dtype=np.float32)):
        raise RuntimeError("Reference validation labels do not match regenerated m2j20 split.")
    if not np.array_equal(y_test, np.asarray(zref["labels_test"], dtype=np.float32)):
        raise RuntimeError("Reference test labels do not match regenerated m2j20 split.")

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor = m2.wrap_reconstructor_unmerge_only(reconstructor)
    dual_input_dim_a = int(arrays["feat_hlt_dual"].shape[-1])
    dual_input_dim_b = 12 if bool(args.use_corrected_flags) else 10
    dual_joint = DualViewCrossAttnClassifier(input_dim_a=dual_input_dim_a, input_dim_b=dual_input_dim_b, **cfg["model"]).to(device)

    print("Building validation loader...", flush=True)
    dl_val = _make_loader(arrays, val_idx, batch_size=batch_size, num_workers=int(args.num_workers))
    print("Evaluating Stage2 validation...", flush=True)
    _load_pair(reconstructor, dual_joint, ckpts["stage2_reco_ckpt"], ckpts["stage2_dual_ckpt"], device)
    auc_stage2_val, preds_stage2_val, labs_stage2_val, fpr50_stage2_val = _eval_pair(
        reconstructor,
        dual_joint,
        dl_val,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    print("Evaluating StageC validation...", flush=True)
    _load_pair(reconstructor, dual_joint, ckpts["stageC_reco_ckpt"], ckpts["stageC_dual_ckpt"], device)
    auc_joint_val, preds_joint_val, labs_joint_val, fpr50_joint_val = _eval_pair(
        reconstructor,
        dual_joint,
        dl_val,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    del dl_val

    print("Building test loader...", flush=True)
    dl_test = _make_loader(arrays, test_idx, batch_size=batch_size, num_workers=int(args.num_workers))
    print("Evaluating Stage2 test...", flush=True)
    _load_pair(reconstructor, dual_joint, ckpts["stage2_reco_ckpt"], ckpts["stage2_dual_ckpt"], device)
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
    if ckpts["stage2_reco_fpr_ckpt"].exists() and ckpts["stage2_dual_fpr_ckpt"].exists():
        print("Evaluating Stage2 best-FPR test...", flush=True)
        _load_pair(reconstructor, dual_joint, ckpts["stage2_reco_fpr_ckpt"], ckpts["stage2_dual_fpr_ckpt"], device)
        auc_stage2_fprsel, preds_stage2_fprsel, labs_stage2_fprsel, _ = _eval_pair(
            reconstructor,
            dual_joint,
            dl_test,
            device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        if not np.array_equal(labs_stage2_fprsel.astype(np.float32), y_test):
            raise RuntimeError("Stage2 best-FPR test labels mismatch.")

    print("Evaluating StageC test...", flush=True)
    _load_pair(reconstructor, dual_joint, ckpts["stageC_reco_ckpt"], ckpts["stageC_dual_ckpt"], device)
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
    if ckpts["stageC_reco_fpr_ckpt"].exists() and ckpts["stageC_dual_fpr_ckpt"].exists():
        print("Evaluating StageC best-FPR test...", flush=True)
        _load_pair(reconstructor, dual_joint, ckpts["stageC_reco_fpr_ckpt"], ckpts["stageC_dual_fpr_ckpt"], device)
        auc_joint_fprsel, preds_joint_fprsel, labs_joint_fprsel, _ = _eval_pair(
            reconstructor,
            dual_joint,
            dl_test,
            device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        if not np.array_equal(labs_joint_fprsel.astype(np.float32), y_test):
            raise RuntimeError("StageC best-FPR test labels mismatch.")
    del dl_test

    for name, labs in [
        ("stage2_val", labs_stage2_val),
        ("joint_val", labs_joint_val),
        ("stage2_test", labs_stage2_test),
        ("joint_test", labs_joint_test),
    ]:
        target = y_val if name.endswith("_val") else y_test
        if not np.array_equal(np.asarray(labs, dtype=np.float32), target):
            raise RuntimeError(f"{name} labels mismatch.")

    preds_teacher_val = np.asarray(zref["preds_teacher_val"], dtype=np.float64)
    preds_teacher_test = np.asarray(zref["preds_teacher_test"], dtype=np.float64)
    preds_hlt_val = np.asarray(zref["preds_hlt_val"], dtype=np.float64)
    preds_hlt_test = np.asarray(zref["preds_hlt_test"], dtype=np.float64)

    auc_teacher_val = _auc_from_ref(zref, "auc_teacher_val", "labels_val", "preds_teacher_val")
    auc_teacher_test = _auc_from_ref(zref, "auc_teacher_test", "labels_test", "preds_teacher_test")
    auc_hlt_val = _auc_from_ref(zref, "auc_hlt_val", "labels_val", "preds_hlt_val")
    auc_hlt_test = _auc_from_ref(zref, "auc_hlt_test", "labels_test", "preds_hlt_test")

    def _fprs(scores: np.ndarray) -> Tuple[float, float]:
        fpr, tpr, _ = roc_curve(y_test, np.asarray(scores, dtype=np.float64))
        return float(fpr_at_target_tpr(fpr, tpr, 0.30)), float(fpr_at_target_tpr(fpr, tpr, 0.50))

    fpr30_teacher, fpr50_teacher = _fprs(preds_teacher_test)
    fpr30_hlt, fpr50_hlt = _fprs(preds_hlt_test)
    fpr30_stage2, fpr50_stage2 = _fprs(preds_stage2_test)
    fpr30_joint, fpr50_joint = _fprs(preds_joint_test)
    fpr30_stage2_fprsel, fpr50_stage2_fprsel = (
        _fprs(preds_stage2_fprsel) if preds_stage2_fprsel is not None else (float("nan"), float("nan"))
    )
    fpr30_joint_fprsel, fpr50_joint_fprsel = (
        _fprs(preds_joint_fprsel) if preds_joint_fprsel is not None else (float("nan"), float("nan"))
    )

    fusion_path = save_root / "fusion_scores_val_test.npz"
    np.savez_compressed(
        fusion_path,
        labels_val=y_val.astype(np.float32),
        labels_test=y_test.astype(np.float32),
        preds_teacher_val=preds_teacher_val.astype(np.float64),
        preds_teacher_test=preds_teacher_test.astype(np.float64),
        preds_hlt_val=preds_hlt_val.astype(np.float64),
        preds_hlt_test=preds_hlt_test.astype(np.float64),
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
        hlt_nconst_val=np.asarray(arrays["hlt_mask"][val_idx].sum(axis=1), dtype=np.int32),
        hlt_nconst_test=np.asarray(arrays["hlt_mask"][test_idx].sum(axis=1), dtype=np.int32),
        hlt_jet_pt_val=np.asarray(compute_jet_pt(arrays["hlt_const"][val_idx], arrays["hlt_mask"][val_idx]), dtype=np.float64),
        hlt_jet_pt_test=np.asarray(compute_jet_pt(arrays["hlt_const"][test_idx], arrays["hlt_mask"][test_idx]), dtype=np.float64),
        off_jet_pt_val=np.asarray(compute_jet_pt(arrays["const_off"][val_idx], arrays["masks_off"][val_idx]), dtype=np.float64),
        off_jet_pt_test=np.asarray(compute_jet_pt(arrays["const_off"][test_idx], arrays["masks_off"][test_idx]), dtype=np.float64),
    )

    np.savez(
        save_root / "results.npz",
        labels=y_test.astype(np.float32),
        preds_teacher=preds_teacher_test.astype(np.float64),
        preds_baseline=preds_hlt_test.astype(np.float64),
        preds_stage2=np.asarray(preds_stage2_test, dtype=np.float64),
        preds_joint=np.asarray(preds_joint_test, dtype=np.float64),
        auc_teacher=float(auc_teacher_test),
        auc_baseline=float(auc_hlt_test),
        auc_stage2=float(auc_stage2_test),
        auc_joint=float(auc_joint_test),
    )

    metrics = {
        "finalized_from_checkpoints": True,
        "reference_npz": str(ref_path),
        "checkpoints": {k: str(v) for k, v in ckpts.items()},
        "auc_teacher_test": float(auc_teacher_test),
        "auc_hlt_test": float(auc_hlt_test),
        "auc_stage2_test": float(auc_stage2_test),
        "auc_stage2_bestfpr50_test": float(auc_stage2_fprsel),
        "auc_joint_test": float(auc_joint_test),
        "auc_joint_bestfpr50_test": float(auc_joint_fprsel),
        "fpr30_teacher": float(fpr30_teacher),
        "fpr30_hlt": float(fpr30_hlt),
        "fpr30_stage2": float(fpr30_stage2),
        "fpr30_stage2_bestfpr50": float(fpr30_stage2_fprsel),
        "fpr30_joint": float(fpr30_joint),
        "fpr30_joint_bestfpr50": float(fpr30_joint_fprsel),
        "fpr50_teacher": float(fpr50_teacher),
        "fpr50_hlt": float(fpr50_hlt),
        "fpr50_stage2": float(fpr50_stage2),
        "fpr50_stage2_bestfpr50": float(fpr50_stage2_fprsel),
        "fpr50_joint": float(fpr50_joint),
        "fpr50_joint_bestfpr50": float(fpr50_joint_fprsel),
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
    print(f"Saved metrics:       {save_root / 'joint_stage_metrics.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze overlap/correlation at a fixed operating point between:
  - HLT baseline model score
  - Teacher score on Stage-2 (pre-StageC) reconstructor outputs

This script is designed to run from a saved teacherkd run directory produced by:
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as m


def load_model_state(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def resolve_train_files(data_setup: dict, data_file_override: str) -> List[Path]:
    if data_file_override:
        p = Path(data_file_override)
        if not p.exists():
            raise FileNotFoundError(f"--data_file not found: {p}")
        return [p]

    saved_files = [Path(p) for p in data_setup.get("train_files", [])]
    existing_saved = [p for p in saved_files if p.exists()]
    if existing_saved:
        return existing_saved

    train_path_arg = data_setup.get("train_path_arg", "./data")
    tp = Path(train_path_arg)
    if tp.is_dir():
        files = sorted(tp.glob("*.h5"))
    else:
        files = [tp]
    files = [p for p in files if p.exists()]
    if files:
        return files

    raise FileNotFoundError(
        "Could not resolve local HDF5 input files from data_setup.json; "
        "provide --data_file explicitly."
    )


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    ox = np.argsort(x, kind="mergesort")
    oy = np.argsort(y, kind="mergesort")
    rx = np.empty_like(ox, dtype=np.float64)
    ry = np.empty_like(oy, dtype=np.float64)
    rx[ox] = np.arange(len(x), dtype=np.float64)
    ry[oy] = np.arange(len(y), dtype=np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="HLT vs teacher-on-Stage2-reco overlap analysis")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--reco_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--weight_threshold", type=float, default=0.03)
    ap.add_argument("--disable_budget_topk", action="store_true")
    ap.add_argument("--data_file", type=str, default="")
    ap.add_argument("--teacher_ckpt", type=str, default="teacher.pt")
    ap.add_argument("--hlt_ckpt", type=str, default="baseline.pt")
    ap.add_argument("--reco_ckpt", type=str, default="offline_reconstructor_stage2.pt")
    ap.add_argument("--output_name", type=str, default="hlt_vs_teacher_on_stage2_reco_overlap_tpr50.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    data_setup_path = run_dir / "data_setup.json"
    split_path = run_dir / "data_splits.npz"
    if not data_setup_path.exists() or not split_path.exists():
        raise FileNotFoundError("Missing data_setup.json or data_splits.npz in run_dir")

    with open(data_setup_path, "r", encoding="utf-8") as f:
        data_setup = json.load(f)
    split = np.load(split_path)

    test_idx = split["test_idx"]
    means = split["means"]
    stds = split["stds"]

    seed = int(data_setup["seed"])
    offset_jets = int(data_setup["offset_jets"])
    n_train_jets = int(data_setup["n_train_jets"])
    max_constits = int(data_setup["max_constits"])

    train_files = resolve_train_files(data_setup, args.data_file)
    print("Resolved input HDF5 files:")
    for p in train_files:
        print(f"  - {p}")

    max_jets_needed = offset_jets + n_train_jets
    print("Loading raw constituents...")
    all_const_full, all_labels_full = m.load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=max_constits,
    )

    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets loaded: need {max_jets_needed}, got {all_const_full.shape[0]}"
        )

    const_raw = all_const_full[offset_jets: offset_jets + n_train_jets]
    labels = all_labels_full[offset_jets: offset_jets + n_train_jets].astype(np.int64)

    print("Regenerating pseudo-HLT deterministically...")
    cfg = m._deepcopy_config()
    for k, v in data_setup.get("hlt_effects", {}).items():
        if k in cfg["hlt_effects"]:
            cfg["hlt_effects"][k] = v

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    hlt_const, hlt_mask, _, _ = m.apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=seed,
    )

    print("Computing standardized features...")
    feat_hlt = m.compute_features(hlt_const, hlt_mask)
    feat_hlt_std = m.standardize(feat_hlt, hlt_mask, means, stds)

    # Device fallback
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but torch.cuda.is_available() is False; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Models
    teacher = m.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher.load_state_dict(load_model_state(run_dir / args.teacher_ckpt, device))
    teacher.eval()

    hlt_model = m.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    hlt_model.load_state_dict(load_model_state(run_dir / args.hlt_ckpt, device))
    hlt_model.eval()

    reconstructor = m.OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(load_model_state(run_dir / args.reco_ckpt, device))
    reconstructor.eval()

    # HLT baseline on test
    print("Evaluating HLT baseline on test split...")
    ds_test_hlt = m.JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_test_hlt = DataLoader(
        ds_test_hlt,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    auc_hlt, preds_hlt, labs_hlt = m.eval_classifier(hlt_model, dl_test_hlt, device)

    # Reconstruct test with Stage2 reconstructor
    print("Reconstructing test split with Stage2 reconstructor...")
    (
        reco_const_test,
        reco_mask_test,
        _reco_merge_flag_test,
        _reco_eff_flag_test,
        _created_merge_count_test,
        _created_eff_count_test,
        _pred_budget_total_test,
        _pred_budget_merge_test,
        _pred_budget_eff_test,
    ) = m.reconstruct_dataset(
        model=reconstructor,
        feat_hlt=feat_hlt_std[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_hlt=hlt_const[test_idx],
        max_constits=max_constits,
        device=device,
        batch_size=int(args.reco_batch_size),
        weight_threshold=float(args.weight_threshold),
        use_budget_topk=not bool(args.disable_budget_topk),
    )

    feat_reco_test = m.compute_features(reco_const_test, reco_mask_test)
    feat_reco_test_std = m.standardize(feat_reco_test, reco_mask_test, means, stds)

    print("Evaluating teacher on reconstructed test split...")
    ds_test_reco = m.JetDataset(feat_reco_test_std, reco_mask_test, labels[test_idx])
    dl_test_reco = DataLoader(
        ds_test_reco,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    auc_teacher_reco, preds_teacher_reco, labs_teacher_reco = m.eval_classifier(teacher, dl_test_reco, device)

    labs_ref = labels[test_idx].astype(np.float32)
    if not np.array_equal(labs_hlt.astype(np.float32), labs_ref):
        raise RuntimeError("Label mismatch: HLT eval labels do not align with test_idx labels")
    if not np.array_equal(labs_teacher_reco.astype(np.float32), labs_ref):
        raise RuntimeError("Label mismatch: teacher-on-reco labels do not align with test_idx labels")

    print(f"Building overlap report at TPR={float(args.target_tpr):.3f}...")
    overlap_report = m.build_overlap_report_at_tpr(
        labels=labs_ref,
        model_preds={
            "hlt": preds_hlt,
            "teacher_reco_stage2": preds_teacher_reco,
        },
        target_tpr=float(args.target_tpr),
    )

    pair = overlap_report["pairs"].get("hlt__teacher_reco_stage2", {})
    if not pair:
        pair = overlap_report["pairs"].get("teacher_reco_stage2__hlt", {})

    preds_hlt = np.asarray(preds_hlt, dtype=np.float64)
    preds_teacher_reco = np.asarray(preds_teacher_reco, dtype=np.float64)
    pearson_all = float(np.corrcoef(preds_hlt, preds_teacher_reco)[0, 1])
    spearman_all = rank_corr(preds_hlt, preds_teacher_reco)

    pos = labs_ref > 0.5
    neg = ~pos
    pearson_signal = float(np.corrcoef(preds_hlt[pos], preds_teacher_reco[pos])[0, 1]) if pos.sum() > 1 else float("nan")
    pearson_background = float(np.corrcoef(preds_hlt[neg], preds_teacher_reco[neg])[0, 1]) if neg.sum() > 1 else float("nan")

    out = {
        "run_dir": str(run_dir),
        "device": str(device),
        "n_test": int(len(test_idx)),
        "target_tpr": float(args.target_tpr),
        "ckpts": {
            "teacher": args.teacher_ckpt,
            "hlt": args.hlt_ckpt,
            "reconstructor": args.reco_ckpt,
        },
        "auc": {
            "hlt": float(auc_hlt),
            "teacher_on_stage2_reco": float(auc_teacher_reco),
        },
        "overlap_report": overlap_report,
        "pair_hlt_vs_teacher_reco_stage2": pair,
        "score_correlation": {
            "pearson_all": pearson_all,
            "spearman_all": spearman_all,
            "pearson_signal_only": pearson_signal,
            "pearson_background_only": pearson_background,
        },
    }

    out_path = run_dir / args.output_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved overlap analysis to:", out_path)
    print("AUC(HLT):", out["auc"]["hlt"])
    print("AUC(Teacher on Stage2 Reco):", out["auc"]["teacher_on_stage2_reco"])
    print("Pair overlap (HLT vs TeacherRecoStage2):")
    print(json.dumps(pair, indent=2))
    print("Score correlation:")
    print(json.dumps(out["score_correlation"], indent=2))


if __name__ == "__main__":
    main()

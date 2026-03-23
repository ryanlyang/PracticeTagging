#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze overlap/correlation at a fixed operating point between:
  - HLT baseline model score
  - Teacher score on Stage-2 (pre-StageC) reconstructor outputs

Also reports weighted-combo metrics:
  - best combo selected on val and evaluated on test
  - oracle best combo directly on test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


@torch.no_grad()
def build_teacher_soft_reco_features_numpy(
    reconstructor: torch.nn.Module,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    device: torch.device,
    batch_size: int,
    weight_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(m, "_build_teacher_reco_features_from_output"):
        raise RuntimeError(
            "soft reco_eval_mode requires _build_teacher_reco_features_from_output in module"
        )

    n_jets, seq_len, _ = feat_hlt_std.shape
    feat_reco = np.zeros((n_jets, seq_len, 7), dtype=np.float32)
    reco_mask = np.zeros((n_jets, seq_len), dtype=bool)

    reconstructor.eval()
    for start in range(0, n_jets, int(batch_size)):
        end = min(start + int(batch_size), n_jets)
        x = torch.tensor(feat_hlt_std[start:end], dtype=torch.float32, device=device)
        mh = torch.tensor(hlt_mask[start:end], dtype=torch.bool, device=device)
        ch = torch.tensor(hlt_const[start:end], dtype=torch.float32, device=device)
        reco_out = reconstructor(x, mh, ch, stage_scale=1.0)
        feat_reco_t, reco_mask_t = m._build_teacher_reco_features_from_output(
            reco_out,
            ch,
            mh,
            weight_floor=float(weight_threshold),
        )
        feat_reco[start:end] = feat_reco_t.detach().cpu().numpy().astype(np.float32)
        reco_mask[start:end] = reco_mask_t.detach().cpu().numpy().astype(bool)

    return feat_reco, reco_mask


def eval_hlt_and_teacher_reco_on_split(
    split_name: str,
    split_idx: np.ndarray,
    labels: np.ndarray,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    teacher: torch.nn.Module,
    hlt_model: torch.nn.Module,
    reconstructor: torch.nn.Module,
    max_constits: int,
    device: torch.device,
    eval_batch_size: int,
    reco_batch_size: int,
    num_workers: int,
    weight_threshold: float,
    use_budget_topk: bool,
    reco_eval_mode: str,
) -> Dict[str, object]:
    print(f"Evaluating HLT baseline on {split_name} split...")
    ds_hlt = m.JetDataset(feat_hlt_std[split_idx], hlt_mask[split_idx], labels[split_idx])
    dl_hlt = DataLoader(
        ds_hlt,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    auc_hlt, preds_hlt, labs_hlt = m.eval_classifier(hlt_model, dl_hlt, device)

    feat_hlt_split = feat_hlt_std[split_idx]
    hlt_mask_split = hlt_mask[split_idx]
    hlt_const_split = hlt_const[split_idx]

    print(
        f"Building {split_name} reconstructed teacher-view features "
        f"(mode={reco_eval_mode})..."
    )
    if reco_eval_mode == "soft":
        feat_reco, reco_mask = build_teacher_soft_reco_features_numpy(
            reconstructor=reconstructor,
            feat_hlt_std=feat_hlt_split,
            hlt_mask=hlt_mask_split,
            hlt_const=hlt_const_split,
            device=device,
            batch_size=int(reco_batch_size),
            weight_threshold=float(weight_threshold),
        )
    elif reco_eval_mode == "hard":
        (
            reco_const,
            reco_mask,
            _reco_merge_flag,
            _reco_eff_flag,
            _created_merge_count,
            _created_eff_count,
            _pred_budget_total,
            _pred_budget_merge,
            _pred_budget_eff,
        ) = m.reconstruct_dataset(
            model=reconstructor,
            feat_hlt=feat_hlt_split,
            mask_hlt=hlt_mask_split,
            const_hlt=hlt_const_split,
            max_constits=max_constits,
            device=device,
            batch_size=int(reco_batch_size),
            weight_threshold=float(weight_threshold),
            use_budget_topk=bool(use_budget_topk),
        )
        feat_reco = m.compute_features(reco_const, reco_mask)
    else:
        raise ValueError(f"Unsupported reco_eval_mode: {reco_eval_mode}")

    feat_reco_std = m.standardize(feat_reco, reco_mask, means, stds)

    print(f"Evaluating teacher on reconstructed {split_name} split...")
    ds_reco = m.JetDataset(feat_reco_std, reco_mask, labels[split_idx])
    dl_reco = DataLoader(
        ds_reco,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    auc_teacher_reco, preds_teacher_reco, labs_teacher_reco = m.eval_classifier(teacher, dl_reco, device)

    labs_ref = labels[split_idx].astype(np.float32)
    if not np.array_equal(labs_hlt.astype(np.float32), labs_ref):
        raise RuntimeError(f"Label mismatch on {split_name}: HLT labels do not align with split labels")
    if not np.array_equal(labs_teacher_reco.astype(np.float32), labs_ref):
        raise RuntimeError(f"Label mismatch on {split_name}: teacher-on-reco labels do not align with split labels")

    return {
        "auc_hlt": float(auc_hlt),
        "auc_teacher_reco": float(auc_teacher_reco),
        "preds_hlt": np.asarray(preds_hlt, dtype=np.float64),
        "preds_teacher_reco": np.asarray(preds_teacher_reco, dtype=np.float64),
        "labels": labs_ref,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="HLT vs teacher-on-Stage2-reco overlap analysis")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--target_tpr", type=float, default=0.50)
    ap.add_argument("--combo_weight_step", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--reco_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--weight_threshold", type=float, default=0.03)
    ap.add_argument("--disable_budget_topk", action="store_true")
    ap.add_argument("--reco_eval_mode", type=str, default="soft", choices=["soft", "hard"])
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

    val_idx = split["val_idx"]
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

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but torch.cuda.is_available() is False; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    teacher = m.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher.load_state_dict(load_model_state(run_dir / args.teacher_ckpt, device))
    teacher.eval()

    hlt_model = m.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    hlt_model.load_state_dict(load_model_state(run_dir / args.hlt_ckpt, device))
    hlt_model.eval()

    reconstructor = m.OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(load_model_state(run_dir / args.reco_ckpt, device))
    reconstructor.eval()

    val_eval = eval_hlt_and_teacher_reco_on_split(
        split_name="val",
        split_idx=val_idx,
        labels=labels,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        means=means,
        stds=stds,
        teacher=teacher,
        hlt_model=hlt_model,
        reconstructor=reconstructor,
        max_constits=max_constits,
        device=device,
        eval_batch_size=int(args.eval_batch_size),
        reco_batch_size=int(args.reco_batch_size),
        num_workers=int(args.num_workers),
        weight_threshold=float(args.weight_threshold),
        use_budget_topk=not bool(args.disable_budget_topk),
        reco_eval_mode=str(args.reco_eval_mode),
    )

    test_eval = eval_hlt_and_teacher_reco_on_split(
        split_name="test",
        split_idx=test_idx,
        labels=labels,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        means=means,
        stds=stds,
        teacher=teacher,
        hlt_model=hlt_model,
        reconstructor=reconstructor,
        max_constits=max_constits,
        device=device,
        eval_batch_size=int(args.eval_batch_size),
        reco_batch_size=int(args.reco_batch_size),
        num_workers=int(args.num_workers),
        weight_threshold=float(args.weight_threshold),
        use_budget_topk=not bool(args.disable_budget_topk),
        reco_eval_mode=str(args.reco_eval_mode),
    )

    labels_test = test_eval["labels"]
    preds_hlt_test = test_eval["preds_hlt"]
    preds_reco_teacher_test = test_eval["preds_teacher_reco"]

    print(f"Building overlap report at TPR={float(args.target_tpr):.3f}...")
    overlap_report = m.build_overlap_report_at_tpr(
        labels=labels_test,
        model_preds={
            "hlt": preds_hlt_test,
            "teacher_reco_stage2": preds_reco_teacher_test,
        },
        target_tpr=float(args.target_tpr),
    )

    pair = overlap_report["pairs"].get("hlt__teacher_reco_stage2", {})
    if not pair:
        pair = overlap_report["pairs"].get("teacher_reco_stage2__hlt", {})

    combo_valsel = m.select_weighted_combo_on_val_and_eval_test(
        labels_val=val_eval["labels"],
        preds_a_val=val_eval["preds_hlt"],
        preds_b_val=val_eval["preds_teacher_reco"],
        labels_test=test_eval["labels"],
        preds_a_test=test_eval["preds_hlt"],
        preds_b_test=test_eval["preds_teacher_reco"],
        name_a="hlt",
        name_b="teacher_reco_stage2",
        target_tpr=float(args.target_tpr),
        weight_step=float(max(args.combo_weight_step, 1e-4)),
    )

    combo_test_oracle = m.search_best_weighted_combo_at_tpr(
        labels=test_eval["labels"],
        preds_a=test_eval["preds_hlt"],
        preds_b=test_eval["preds_teacher_reco"],
        name_a="hlt",
        name_b="teacher_reco_stage2",
        target_tpr=float(args.target_tpr),
        weight_step=float(max(args.combo_weight_step, 1e-4)),
    )

    pearson_all = float(np.corrcoef(preds_hlt_test, preds_reco_teacher_test)[0, 1])
    spearman_all = rank_corr(preds_hlt_test, preds_reco_teacher_test)

    pos = labels_test > 0.5
    neg = ~pos
    pearson_signal = float(np.corrcoef(preds_hlt_test[pos], preds_reco_teacher_test[pos])[0, 1]) if pos.sum() > 1 else float("nan")
    pearson_background = float(np.corrcoef(preds_hlt_test[neg], preds_reco_teacher_test[neg])[0, 1]) if neg.sum() > 1 else float("nan")

    out = {
        "run_dir": str(run_dir),
        "device": str(device),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "target_tpr": float(args.target_tpr),
        "combo_weight_step": float(max(args.combo_weight_step, 1e-4)),
        "reco_eval_mode": str(args.reco_eval_mode),
        "ckpts": {
            "teacher": args.teacher_ckpt,
            "hlt": args.hlt_ckpt,
            "reconstructor": args.reco_ckpt,
        },
        "auc": {
            "val_hlt": float(val_eval["auc_hlt"]),
            "val_teacher_on_stage2_reco": float(val_eval["auc_teacher_reco"]),
            "test_hlt": float(test_eval["auc_hlt"]),
            "test_teacher_on_stage2_reco": float(test_eval["auc_teacher_reco"]),
        },
        "overlap_report": overlap_report,
        "pair_hlt_vs_teacher_reco_stage2": pair,
        "best_combo_hlt_teacher_reco_val_selected_eval_test": combo_valsel,
        "best_combo_hlt_teacher_reco_test_posthoc_oracle": combo_test_oracle,
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
    print("Reco eval mode:", out["reco_eval_mode"])
    print("AUC(HLT test):", out["auc"]["test_hlt"])
    print("AUC(Teacher on Stage2 Reco test):", out["auc"]["test_teacher_on_stage2_reco"])
    print("Pair overlap (HLT vs TeacherRecoStage2):")
    print(json.dumps(pair, indent=2))
    print("Best weighted combo (val-selected, evaluated on test):")
    print(json.dumps(combo_valsel, indent=2))
    print("Best weighted combo (oracle on test):")
    print(json.dumps(combo_test_oracle, indent=2))
    print("Score correlation:")
    print(json.dumps(out["score_correlation"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fast Stage-C finetuning starting from a saved Stage2 checkpoint.

This script re-creates data/splits deterministically, loads:
  - offline_reconstructor_stage2.pt
  - dual_joint_stage2.pt
from a previous run folder, then runs Stage C only.
If available, it consumes run_dir/data_setup.json and run_dir/data_splits.npz
to reproduce the exact original data setup and split indices.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

import offline_reconstructor_joint_dualview_stage2save_auc_norankc as joint
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as LOCAL30K_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
    fpr_at_target_tpr,
)
from unmerge_correct_hlt import (
    RANDOM_SEED,
    DualViewCrossAttnClassifier,
    JetDataset,
    ParticleTransformer,
    compute_features,
    eval_classifier,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_checkpoint_state(path: Path, device: torch.device, tag: str) -> Dict[str, torch.Tensor]:
    """
    Load checkpoint and return a plain state_dict.

    Supports:
    - plain state_dict files
    - wrapped dicts like {"model": state_dict, ...}
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    keys = list(ckpt.keys())[:8] if isinstance(ckpt, dict) else [type(ckpt).__name__]
    raise RuntimeError(
        f"Unsupported checkpoint format for {tag}: {path}. "
        f"Top-level keys/type preview: {keys}"
    )


def load_cfg_from_run(run_dir: Path) -> Dict:
    cfg = joint._deepcopy_config()
    hlt_stats_path = run_dir / "hlt_stats.json"
    if hlt_stats_path.exists():
        h = json.load(open(hlt_stats_path, "r", encoding="utf-8"))
        hcfg = h.get("config", {})
        for k, v in hcfg.items():
            if k in cfg["hlt_effects"]:
                cfg["hlt_effects"][k] = v
    return cfg


def load_saved_data_setup(run_dir: Path) -> Dict:
    path = run_dir / "data_setup.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            out = json.load(f)
        return out if isinstance(out, dict) else {}
    except Exception as e:
        print(f"Warning: failed to read saved data setup {path}: {e}")
        return {}


def load_saved_splits(run_dir: Path) -> Dict[str, np.ndarray]:
    path = run_dir / "data_splits.npz"
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:
        print(f"Warning: failed to read saved splits {path}: {e}")
        return {}


def maybe_build_jetreg_features(
    run_dir: Path,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    const_off: np.ndarray,
    masks_off: np.ndarray,
    hlt_const: np.ndarray,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> np.ndarray:
    jet_reg_ckpt = run_dir / "jet_regressor.pt"
    jet_meta_path = run_dir / "jet_regression_metrics.json"

    enabled_in_run = False
    if jet_meta_path.exists():
        try:
            enabled_in_run = bool(json.load(open(jet_meta_path, "r", encoding="utf-8")).get("enabled", False))
        except Exception:
            enabled_in_run = False

    if (not jet_reg_ckpt.exists()) or (not enabled_in_run):
        return feat_hlt_std.astype(np.float32, copy=True)

    target_off, target_hlt_ref, _ = joint.compute_jet_regression_targets(
        const_off=const_off,
        mask_off=masks_off,
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
    )
    target_dim = int(target_off.shape[1])

    model = joint.JetLevelRegressor(
        input_dim=7,
        output_dim=target_dim,
        embed_dim=int(args.jet_reg_embed_dim),
        num_heads=int(args.jet_reg_num_heads),
        num_layers=int(args.jet_reg_num_layers),
        ff_dim=int(args.jet_reg_ff_dim),
        dropout=float(args.jet_reg_dropout),
    ).to(device)
    model.load_state_dict(torch.load(jet_reg_ckpt, map_location=device))

    pred_log_all = joint.predict_jet_regressor(
        model=model,
        feat=feat_hlt_std,
        mask=hlt_mask,
        device=device,
        batch_size=int(batch_size),
    )
    delta_vs_hlt = pred_log_all - target_hlt_ref
    extra_global = np.concatenate([pred_log_all, delta_vs_hlt], axis=-1).astype(np.float32)
    extra_global = np.repeat(extra_global[:, None, :], feat_hlt_std.shape[1], axis=1)
    feat_hlt_dual = np.concatenate([feat_hlt_std, extra_global], axis=-1).astype(np.float32)
    feat_hlt_dual[~hlt_mask] = 0.0
    return feat_hlt_dual


def maybe_eval_single_view_checkpoint(
    ckpt_path: Path,
    tag: str,
    feat_test: np.ndarray,
    mask_test: np.ndarray,
    labels_test: np.ndarray,
    model_cfg: Dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Load a single-view classifier checkpoint (teacher/baseline), evaluate on test split,
    and return AUC/FPR@30/FPR@50 metrics. Returns {} if ckpt is missing.
    """
    if not ckpt_path.exists():
        print(f"Warning: {tag} checkpoint not found: {ckpt_path}")
        return {}

    model = ParticleTransformer(input_dim=7, **model_cfg).to(device)
    state = _load_checkpoint_state(ckpt_path, device, tag)
    model.load_state_dict(state)

    ds = JetDataset(feat_test, mask_test, labels_test)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    auc, preds, labs = eval_classifier(model, dl, device)
    fpr, tpr, _ = roc_curve(labs, preds)
    return {
        "auc": float(auc),
        "fpr30": float(fpr_at_target_tpr(fpr, tpr, 0.30)),
        "fpr50": float(fpr_at_target_tpr(fpr, tpr, 0.50)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Previous run folder with Stage2 checkpoints")
    p.add_argument("--save_dir", type=str, default="", help="If empty, defaults to <run_dir>/stagec_refine")
    p.add_argument("--run_name", type=str, default="stagec_refine")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--n_train_jets", type=int, default=100000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=80)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=-1)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--ignore_saved_data_setup",
        action="store_true",
        help="Ignore run_dir/data_setup.json and run_dir/data_splits.npz; rebuild from CLI args.",
    )

    p.add_argument("--reco_ckpt", type=str, default="")
    p.add_argument("--dual_ckpt", type=str, default="")
    p.add_argument(
        "--fresh_dual_init",
        action="store_true",
        help="Initialize dual-view classifier from scratch instead of loading dual Stage2 checkpoint.",
    )

    # Stage C knobs for fast iteration.
    p.add_argument("--stageC_epochs", type=int, default=35)
    p.add_argument("--stageC_patience", type=int, default=8)
    p.add_argument("--stageC_min_epochs", type=int, default=8)
    p.add_argument(
        "--stageC_freeze_reco_epochs",
        type=int,
        default=0,
        help=(
            "Freeze reconstructor for the first N Stage-C epochs, then unfreeze for the remaining epochs. "
            "0 means never frozen."
        ),
    )
    p.add_argument("--stageC_lr_dual", type=float, default=2e-5)
    p.add_argument("--stageC_lr_reco", type=float, default=1e-5)
    p.add_argument("--stageC_lambda_rank", type=float, default=0.0)
    p.add_argument("--lambda_reco", type=float, default=0.35)
    p.add_argument("--lambda_cons", type=float, default=0.0)
    p.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])
    p.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    p.add_argument("--use_corrected_flags", action="store_true")

    # Optional overrides for reconstruction loss weights during Stage C.
    p.add_argument("--loss_w_pt_ratio", type=float, default=-1.0)
    p.add_argument("--loss_w_e_ratio", type=float, default=-1.0)
    p.add_argument("--loss_w_budget", type=float, default=-1.0)
    p.add_argument("--loss_w_sparse", type=float, default=-1.0)
    p.add_argument("--loss_w_local", type=float, default=-1.0)

    # Jet reg model architecture defaults (used only if a jet reg ckpt exists in run_dir).
    p.add_argument("--jet_reg_embed_dim", type=int, default=128)
    p.add_argument("--jet_reg_num_heads", type=int, default=8)
    p.add_argument("--jet_reg_num_layers", type=int, default=4)
    p.add_argument("--jet_reg_ff_dim", type=int, default=512)
    p.add_argument("--jet_reg_dropout", type=float, default=0.1)

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    saved_setup = {}
    saved_splits = {}
    use_saved_data_setup = False
    if not bool(args.ignore_saved_data_setup):
        saved_setup = load_saved_data_setup(run_dir)
        saved_splits = load_saved_splits(run_dir)
        use_saved_data_setup = len(saved_setup) > 0

    eff_seed = int(saved_setup.get("seed", args.seed)) if use_saved_data_setup else int(args.seed)
    eff_n_train_jets = int(saved_setup.get("n_train_jets", args.n_train_jets)) if use_saved_data_setup else int(args.n_train_jets)
    eff_offset_jets = int(saved_setup.get("offset_jets", args.offset_jets)) if use_saved_data_setup else int(args.offset_jets)
    eff_max_constits = int(saved_setup.get("max_constits", args.max_constits)) if use_saved_data_setup else int(args.max_constits)
    set_seed(eff_seed)

    out_root = Path(args.save_dir) if str(args.save_dir).strip() else (run_dir / "stagec_refine")
    save_root = out_root / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg = load_cfg_from_run(run_dir)

    if args.loss_w_pt_ratio >= 0:
        cfg["loss"]["w_pt_ratio"] = float(args.loss_w_pt_ratio)
    if args.loss_w_e_ratio >= 0:
        cfg["loss"]["w_e_ratio"] = float(args.loss_w_e_ratio)
    if args.loss_w_budget >= 0:
        cfg["loss"]["w_budget"] = float(args.loss_w_budget)
    if args.loss_w_sparse >= 0:
        cfg["loss"]["w_sparse"] = float(args.loss_w_sparse)
    if args.loss_w_local >= 0:
        cfg["loss"]["w_local"] = float(args.loss_w_local)

    print(f"Device: {device}")
    print(f"Load run dir: {run_dir}")
    print(f"Save dir: {save_root}")

    if use_saved_data_setup:
        train_files = [Path(p) for p in saved_setup.get("train_files", [])]
        train_files = [p for p in train_files if p.exists()]
        if len(train_files) == 0:
            print("Warning: saved train_files unavailable; falling back to --train_path")
    else:
        train_files = []

    if len(train_files) == 0:
        train_path = Path(args.train_path)
        if train_path.is_dir():
            train_files = sorted(list(train_path.glob("*.h5")))
        else:
            train_files = [Path(x) for x in str(args.train_path).split(",") if x.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    print(
        f"Data setup source: {'saved data_setup.json' if use_saved_data_setup else 'CLI args'} | "
        f"seed={eff_seed}, n_train_jets={eff_n_train_jets}, offset_jets={eff_offset_jets}, max_constits={eff_max_constits}"
    )

    max_jets_needed = int(eff_offset_jets) + int(eff_n_train_jets)
    print("Loading offline constituents...")
    all_const, all_labels = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=int(eff_max_constits),
    )
    if all_const.shape[0] < max_jets_needed:
        raise RuntimeError(f"Requested {max_jets_needed} jets but got {all_const.shape[0]}")

    const_raw = all_const[int(eff_offset_jets): int(eff_offset_jets) + int(eff_n_train_jets)]
    labels = all_labels[int(eff_offset_jets): int(eff_offset_jets) + int(eff_n_train_jets)].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT deterministically...")
    hlt_const, hlt_mask, _, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(eff_seed),
    )
    budget_merge_true = budget_truth["merge_lost_per_jet"].astype(np.float32)
    budget_eff_true = budget_truth["eff_lost_per_jet"].astype(np.float32)

    print("Computing features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    splits_source = "recomputed"
    has_saved_split_idx = (
        isinstance(saved_splits, dict)
        and "train_idx" in saved_splits
        and "val_idx" in saved_splits
        and "test_idx" in saved_splits
    )
    if use_saved_data_setup and has_saved_split_idx:
        train_idx = np.asarray(saved_splits["train_idx"], dtype=np.int64)
        val_idx = np.asarray(saved_splits["val_idx"], dtype=np.int64)
        test_idx = np.asarray(saved_splits["test_idx"], dtype=np.int64)
        all_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)
        max_idx = int(np.max(all_idx)) if all_idx.size > 0 else -1
        if max_idx >= len(labels):
            raise RuntimeError(
                f"Saved split indices exceed available labels: max_idx={max_idx}, n_labels={len(labels)}"
            )
        splits_source = "saved data_splits.npz"
    else:
        idx = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(
            idx, test_size=0.30, random_state=int(eff_seed), stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=int(eff_seed), stratify=labels[temp_idx]
        )
    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} "
        f"(source: {splits_source})"
    )

    if use_saved_data_setup and isinstance(saved_splits, dict) and "means" in saved_splits and "stds" in saved_splits:
        means = np.asarray(saved_splits["means"], dtype=np.float32)
        stds = np.asarray(saved_splits["stds"], dtype=np.float32)
        if means.shape[-1] != feat_off.shape[-1] or stds.shape[-1] != feat_off.shape[-1]:
            print("Warning: saved means/stds shape mismatch; recomputing from train split.")
            means, stds = get_stats(feat_off, masks_off, train_idx)
    else:
        means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    bs = int(cfg["training"]["batch_size"]) if int(args.batch_size) <= 0 else int(args.batch_size)
    feat_hlt_dual = maybe_build_jetreg_features(
        run_dir=run_dir,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        const_off=const_off,
        masks_off=masks_off,
        hlt_const=hlt_const,
        device=device,
        batch_size=bs,
        args=args,
    )

    ds_train_joint = joint.JointDualDataset(
        feat_hlt_std[train_idx], feat_hlt_dual[train_idx], hlt_mask[train_idx], hlt_const[train_idx],
        const_off[train_idx], masks_off[train_idx], budget_merge_true[train_idx], budget_eff_true[train_idx],
        labels[train_idx],
    )
    ds_val_joint = joint.JointDualDataset(
        feat_hlt_std[val_idx], feat_hlt_dual[val_idx], hlt_mask[val_idx], hlt_const[val_idx],
        const_off[val_idx], masks_off[val_idx], budget_merge_true[val_idx], budget_eff_true[val_idx],
        labels[val_idx],
    )
    ds_test_joint = joint.JointDualDataset(
        feat_hlt_std[test_idx], feat_hlt_dual[test_idx], hlt_mask[test_idx], hlt_const[test_idx],
        const_off[test_idx], masks_off[test_idx], budget_merge_true[test_idx], budget_eff_true[test_idx],
        labels[test_idx],
    )

    dl_train_joint = DataLoader(
        ds_train_joint, batch_size=bs, shuffle=True, drop_last=True,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint = DataLoader(
        ds_val_joint, batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_test_joint = DataLoader(
        ds_test_joint, batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    dual_input_dim_a = int(feat_hlt_dual.shape[-1])
    dual_input_dim_b = 12 if bool(args.use_corrected_flags) else 10
    dual_joint = DualViewCrossAttnClassifier(
        input_dim_a=dual_input_dim_a,
        input_dim_b=dual_input_dim_b,
        **cfg["model"],
    ).to(device)

    reco_ckpt = Path(args.reco_ckpt) if str(args.reco_ckpt).strip() else (run_dir / "offline_reconstructor_stage2.pt")
    dual_ckpt = Path(args.dual_ckpt) if str(args.dual_ckpt).strip() else (run_dir / "dual_joint_stage2.pt")
    if not reco_ckpt.exists():
        raise FileNotFoundError(f"Missing reconstructor checkpoint: {reco_ckpt}")
    if (not bool(args.fresh_dual_init)) and (not dual_ckpt.exists()):
        raise FileNotFoundError(f"Missing dual checkpoint: {dual_ckpt}")
    if bool(args.fresh_dual_init) and (not dual_ckpt.exists()):
        raise FileNotFoundError(
            f"Missing dual checkpoint for Stage2 reference metrics (fresh-dual mode): {dual_ckpt}"
        )

    reco_state = _load_checkpoint_state(reco_ckpt, device, "reconstructor")
    reconstructor.load_state_dict(reco_state)
    if bool(args.fresh_dual_init):
        # Still evaluate loaded Stage2 model for reference metrics.
        dual_ref = DualViewCrossAttnClassifier(
            input_dim_a=dual_input_dim_a,
            input_dim_b=dual_input_dim_b,
            **cfg["model"],
        ).to(device)
        dual_ref.load_state_dict(_load_checkpoint_state(dual_ckpt, device, "dual"))
        print(f"Dual init mode: FRESH (training model randomly initialized), reference Stage2 dual loaded from {dual_ckpt}")
        auc_stage2, preds_stage2, labs_stage2, _ = joint.eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_ref,
            loader=dl_test_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        del dual_ref
    else:
        dual_state = _load_checkpoint_state(dual_ckpt, device, "dual")
        dual_joint.load_state_dict(dual_state)
        print(f"Dual init mode: LOADED from {dual_ckpt}")
        auc_stage2, preds_stage2, labs_stage2, _ = joint.eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_joint,
            loader=dl_test_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
    fpr_s2, tpr_s2, _ = roc_curve(labs_stage2, preds_stage2)
    fpr30_stage2 = float(fpr_at_target_tpr(fpr_s2, tpr_s2, 0.30))
    fpr50_stage2 = float(fpr_at_target_tpr(fpr_s2, tpr_s2, 0.50))

    # Evaluate teacher/baseline from source run on the exact same rebuilt test split.
    teacher_metrics = maybe_eval_single_view_checkpoint(
        ckpt_path=run_dir / "teacher.pt",
        tag="teacher",
        feat_test=feat_off_std[test_idx],
        mask_test=masks_off[test_idx],
        labels_test=labels[test_idx],
        model_cfg=cfg["model"],
        batch_size=bs,
        num_workers=int(args.num_workers),
        device=device,
    )
    baseline_metrics = maybe_eval_single_view_checkpoint(
        ckpt_path=run_dir / "baseline.pt",
        tag="baseline",
        feat_test=feat_hlt_std[test_idx],
        mask_test=hlt_mask[test_idx],
        labels_test=labels[test_idx],
        model_cfg=cfg["model"],
        batch_size=bs,
        num_workers=int(args.num_workers),
        device=device,
    )

    print("\n" + "=" * 70)
    print("FAST STAGE C: JOINT FINETUNE FROM SAVED STAGE2")
    print("=" * 70)

    LOCAL30K_CONFIG["loss"] = cfg["loss"]
    total_stagec_epochs = int(args.stageC_epochs)
    freeze_epochs = max(0, min(int(args.stageC_freeze_reco_epochs), total_stagec_epochs))
    unfreeze_epochs = max(0, total_stagec_epochs - freeze_epochs)
    if freeze_epochs > 0:
        print(
            f"Stage-C schedule: freeze reconstructor for {freeze_epochs} epoch(s), "
            f"then unfreeze for {unfreeze_epochs} epoch(s)."
        )
    else:
        print("Stage-C schedule: reconstructor unfrozen from epoch 1.")

    def _is_auc_mode() -> bool:
        return str(args.selection_metric).lower() == "auc"

    def _better_selected(new_m: Dict[str, float], cur_m: Dict[str, float] | None) -> bool:
        if cur_m is None:
            return True
        if _is_auc_mode():
            return float(new_m.get("selected_val_auc", float("-inf"))) > float(cur_m.get("selected_val_auc", float("-inf")))
        return float(new_m.get("selected_val_fpr50", float("inf"))) < float(cur_m.get("selected_val_fpr50", float("inf")))

    def _better_auc(new_m: Dict[str, float], cur_m: Dict[str, float] | None) -> bool:
        if cur_m is None:
            return True
        return float(new_m.get("best_val_auc_seen", float("-inf"))) > float(cur_m.get("best_val_auc_seen", float("-inf")))

    def _better_fpr(new_m: Dict[str, float], cur_m: Dict[str, float] | None) -> bool:
        if cur_m is None:
            return True
        return float(new_m.get("best_val_fpr50_seen", float("inf"))) < float(cur_m.get("best_val_fpr50_seen", float("inf")))

    selected_metrics = None
    auc_metrics = None
    fpr_metrics = None
    selected_states = None
    auc_states = None
    fpr_states = None
    frozen_selected_metrics = None
    frozen_selected_states = None
    phase_reports = []

    def _run_phase(phase_name: str, freeze_reco: bool, epochs: int, patience: int, min_epochs: int) -> None:
        nonlocal reconstructor, dual_joint
        nonlocal selected_metrics, auc_metrics, fpr_metrics
        nonlocal selected_states, auc_states, fpr_states
        nonlocal frozen_selected_metrics, frozen_selected_states
        if int(epochs) <= 0:
            return
        reconstructor, dual_joint, ph_metrics, ph_states = joint.train_joint_dual(
            reconstructor=reconstructor,
            dual_model=dual_joint,
            train_loader=dl_train_joint,
            val_loader=dl_val_joint,
            device=device,
            stage_name=phase_name,
            freeze_reconstructor=bool(freeze_reco),
            epochs=int(epochs),
            patience=int(patience),
            lr_dual=float(args.stageC_lr_dual),
            lr_reco=float(args.stageC_lr_reco),
            weight_decay=float(cfg["training"]["weight_decay"]),
            warmup_epochs=int(cfg["training"]["warmup_epochs"]),
            lambda_reco=float(args.lambda_reco),
            lambda_rank=float(args.stageC_lambda_rank),
            lambda_cons=float(args.lambda_cons),
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
            min_epochs=int(min_epochs),
            select_metric=str(args.selection_metric),
        )
        phase_reports.append(
            {
                "phase_name": phase_name,
                "freeze_reconstructor": bool(freeze_reco),
                "epochs": int(epochs),
                "metrics": ph_metrics,
            }
        )
        if _better_selected(ph_metrics, selected_metrics):
            selected_metrics = ph_metrics
            selected_states = ph_states.get("selected", {})
        if _better_auc(ph_metrics, auc_metrics):
            auc_metrics = ph_metrics
            auc_states = ph_states.get("auc", {})
        if _better_fpr(ph_metrics, fpr_metrics):
            fpr_metrics = ph_metrics
            fpr_states = ph_states.get("fpr50", {})
        if bool(freeze_reco):
            frozen_selected_metrics = ph_metrics
            frozen_selected_states = ph_states.get("selected", {})

    if freeze_epochs > 0:
        _run_phase(
            phase_name="StageC-FromSavedStage2-FrozenReco",
            freeze_reco=True,
            epochs=int(freeze_epochs),
            patience=max(int(freeze_epochs) + 1, int(args.stageC_patience)),
            min_epochs=int(freeze_epochs),
        )
    _run_phase(
        phase_name="StageC-FromSavedStage2",
        freeze_reco=False,
        epochs=int(unfreeze_epochs if freeze_epochs > 0 else total_stagec_epochs),
        patience=int(args.stageC_patience),
        min_epochs=min(int(args.stageC_min_epochs), int(unfreeze_epochs if freeze_epochs > 0 else total_stagec_epochs)),
    )

    stageC_metrics = {
        "selection_metric": str(args.selection_metric).lower(),
        "selected_val_fpr50": float(selected_metrics.get("selected_val_fpr50", float("nan"))) if selected_metrics else float("nan"),
        "selected_val_auc": float(selected_metrics.get("selected_val_auc", float("nan"))) if selected_metrics else float("nan"),
        "best_val_fpr50_seen": float(fpr_metrics.get("best_val_fpr50_seen", float("nan"))) if fpr_metrics else float("nan"),
        "best_val_auc_seen": float(auc_metrics.get("best_val_auc_seen", float("nan"))) if auc_metrics else float("nan"),
    }
    stageC_states = {
        "selected": {"dual": (selected_states or {}).get("dual"), "reco": (selected_states or {}).get("reco")},
        "auc": {"dual": (auc_states or {}).get("dual"), "reco": (auc_states or {}).get("reco")},
        "fpr50": {"dual": (fpr_states or {}).get("dual"), "reco": (fpr_states or {}).get("reco")},
        "frozen_selected": {"dual": (frozen_selected_states or {}).get("dual"), "reco": (frozen_selected_states or {}).get("reco")},
        "phase_reports": phase_reports,
    }

    auc_joint, preds_joint, labs_joint, _ = joint.eval_joint_model(
        reconstructor=reconstructor,
        dual_model=dual_joint,
        loader=dl_test_joint,
        device=device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    fpr_j, tpr_j, _ = roc_curve(labs_joint, preds_joint)
    fpr30_joint = float(fpr_at_target_tpr(fpr_j, tpr_j, 0.30))
    fpr50_joint = float(fpr_at_target_tpr(fpr_j, tpr_j, 0.50))

    # Optional frozen-phase selected checkpoint eval (if freeze phase was used).
    auc_joint_frozen = float("nan")
    fpr30_joint_frozen = float("nan")
    fpr50_joint_frozen = float("nan")
    if stageC_states.get("frozen_selected", {}).get("dual") is not None and stageC_states.get("frozen_selected", {}).get("reco") is not None:
        torch.save({"model": stageC_states["frozen_selected"]["reco"]}, save_root / "offline_reconstructor_stagec_frozen_ckpt.pt")
        torch.save({"model": stageC_states["frozen_selected"]["dual"]}, save_root / "dual_joint_stagec_frozen_ckpt.pt")
        reconstructor.load_state_dict(stageC_states["frozen_selected"]["reco"])
        dual_joint.load_state_dict(stageC_states["frozen_selected"]["dual"])
        auc_joint_frozen, preds_joint_frozen, labs_joint_frozen, _ = joint.eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_joint,
            loader=dl_test_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        fpr_fr, tpr_fr, _ = roc_curve(labs_joint_frozen, preds_joint_frozen)
        fpr30_joint_frozen = float(fpr_at_target_tpr(fpr_fr, tpr_fr, 0.30))
        fpr50_joint_frozen = float(fpr_at_target_tpr(fpr_fr, tpr_fr, 0.50))

    # Also evaluate Stage-C best-val_fpr50 checkpoint for comparison.
    auc_joint_fprsel = float("nan")
    fpr30_joint_fprsel = float("nan")
    fpr50_joint_fprsel = float("nan")
    if stageC_states.get("fpr50", {}).get("dual") is not None and stageC_states.get("fpr50", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["fpr50"]["reco"])
        dual_joint.load_state_dict(stageC_states["fpr50"]["dual"])
        auc_joint_fprsel, preds_joint_fprsel, labs_joint_fprsel, _ = joint.eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_joint,
            loader=dl_test_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        fpr_f, tpr_f, _ = roc_curve(labs_joint_fprsel, preds_joint_fprsel)
        fpr30_joint_fprsel = float(fpr_at_target_tpr(fpr_f, tpr_f, 0.30))
        fpr50_joint_fprsel = float(fpr_at_target_tpr(fpr_f, tpr_f, 0.50))

    # Restore selected state before saving.
    if stageC_states.get("selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["selected"]["reco"])
    if stageC_states.get("selected", {}).get("dual") is not None:
        dual_joint.load_state_dict(stageC_states["selected"]["dual"])

    # Save selected checkpoint in both legacy (plain state_dict) and analyzer-compatible formats.
    torch.save({"model": reconstructor.state_dict()}, save_root / "offline_reconstructor_stagec_selected_ckpt.pt")
    torch.save({"model": dual_joint.state_dict()}, save_root / "dual_joint_stagec_selected_ckpt.pt")
    torch.save(reconstructor.state_dict(), save_root / "offline_reconstructor.pt")
    torch.save(dual_joint.state_dict(), save_root / "dual_joint.pt")

    # Copy source-run assets required by the disagreement analyzer into this output folder.
    for fname in ["data_setup.json", "data_splits.npz", "teacher.pt", "baseline.pt", "hlt_stats.json"]:
        src = run_dir / fname
        if src.exists():
            try:
                shutil.copy2(src, save_root / fname)
            except Exception as e:
                print(f"Warning: failed to copy {src} -> {save_root / fname}: {e}")

    base_test = {}
    base_path = run_dir / "joint_stage_metrics.json"
    if base_path.exists():
        try:
            base_test = json.load(open(base_path, "r", encoding="utf-8")).get("test", {})
        except Exception:
            base_test = {}

    out_metrics = {
        "source_run_dir": str(run_dir),
        "source_reco_ckpt": str(reco_ckpt),
        "source_dual_ckpt": str(dual_ckpt),
        "stageC_args": {
            "stageC_epochs": int(args.stageC_epochs),
            "stageC_patience": int(args.stageC_patience),
            "stageC_min_epochs": int(args.stageC_min_epochs),
            "stageC_freeze_reco_epochs": int(args.stageC_freeze_reco_epochs),
            "stageC_lr_dual": float(args.stageC_lr_dual),
            "stageC_lr_reco": float(args.stageC_lr_reco),
            "stageC_lambda_rank": float(args.stageC_lambda_rank),
            "lambda_reco": float(args.lambda_reco),
            "lambda_cons": float(args.lambda_cons),
            "selection_metric": str(args.selection_metric),
            "corrected_weight_floor": float(args.corrected_weight_floor),
            "use_corrected_flags": bool(args.use_corrected_flags),
            "fresh_dual_init": bool(args.fresh_dual_init),
        },
        "data_reload": {
            "setup_source": "saved data_setup.json" if use_saved_data_setup else "cli args",
            "splits_source": splits_source,
            "seed_effective": int(eff_seed),
            "n_train_jets_effective": int(eff_n_train_jets),
            "offset_jets_effective": int(eff_offset_jets),
            "max_constits_effective": int(eff_max_constits),
            "train_files_used": [str(p) for p in train_files],
            "ignore_saved_data_setup": bool(args.ignore_saved_data_setup),
        },
        "stageC_metrics": stageC_metrics,
        "stageC_phase_reports": phase_reports,
        "test_stage2_loaded": {
            "auc": float(auc_stage2),
            "fpr30": float(fpr30_stage2),
            "fpr50": float(fpr50_stage2),
        },
        "test_stageC_selected": {
            "auc": float(auc_joint),
            "fpr30": float(fpr30_joint),
            "fpr50": float(fpr50_joint),
        },
        "test_stageC_frozen_selected": {
            "auc": float(auc_joint_frozen),
            "fpr30": float(fpr30_joint_frozen),
            "fpr50": float(fpr50_joint_frozen),
        },
        "test_stageC_bestfpr50": {
            "auc": float(auc_joint_fprsel),
            "fpr30": float(fpr30_joint_fprsel),
            "fpr50": float(fpr50_joint_fprsel),
        },
        "test_teacher_loaded": teacher_metrics,
        "test_baseline_loaded": baseline_metrics,
        "baseline_teacher_from_source": base_test,
    }

    with open(save_root / "stagec_refine_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    np.savez_compressed(
        save_root / "results.npz",
        labels=labs_joint.astype(np.float32),
        preds_stage2=preds_stage2.astype(np.float32),
        preds_stagec_frozen=(preds_joint_frozen.astype(np.float32) if np.isfinite(auc_joint_frozen) else np.array([], dtype=np.float32)),
        preds_stagec=preds_joint.astype(np.float32),
    )

    print("\n" + "=" * 70)
    print("FAST STAGE C RESULTS")
    print("=" * 70)
    if len(teacher_metrics) > 0:
        print(
            f"Teacher (loaded): AUC={teacher_metrics['auc']:.4f}, "
            f"FPR30={teacher_metrics['fpr30']:.6f}, FPR50={teacher_metrics['fpr50']:.6f}"
        )
    if len(baseline_metrics) > 0:
        print(
            f"Baseline (loaded): AUC={baseline_metrics['auc']:.4f}, "
            f"FPR30={baseline_metrics['fpr30']:.6f}, FPR50={baseline_metrics['fpr50']:.6f}"
        )
    print(f"Loaded Stage2: AUC={auc_stage2:.4f}, FPR30={fpr30_stage2:.6f}, FPR50={fpr50_stage2:.6f}")
    if np.isfinite(auc_joint_frozen):
        print(f"StageC FrozenSelected: AUC={auc_joint_frozen:.4f}, FPR30={fpr30_joint_frozen:.6f}, FPR50={fpr50_joint_frozen:.6f}")
    print(f"StageC Selected: AUC={auc_joint:.4f}, FPR30={fpr30_joint:.6f}, FPR50={fpr50_joint:.6f}")
    if np.isfinite(auc_joint_fprsel):
        print(
            f"StageC BestValFPR50: AUC={auc_joint_fprsel:.4f}, "
            f"FPR30={fpr30_joint_fprsel:.6f}, FPR50={fpr50_joint_fprsel:.6f}"
        )
    print(f"Saved to: {save_root}")


if __name__ == "__main__":
    main()

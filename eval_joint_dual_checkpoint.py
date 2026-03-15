#!/usr/bin/env python3
"""
Evaluate a saved joint dual-view checkpoint on the test split used by
offline_reconstructor_joint_dualview.py.

Typical Stage-B-only usage:
  python eval_joint_dual_checkpoint.py \
    --run_dir checkpoints/offline_reconstructor_joint/my_stageb_run \
    --dual_ckpt checkpoints/offline_reconstructor_joint/my_stageb_run/dual_joint.pt \
    --reco_ckpt checkpoints/offline_reconstructor_joint/my_stageb_run/offline_reconstructor.pt \
    --n_train_jets 50000 --max_constits 80 --seed 52 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from unmerge_correct_hlt import (
    RANDOM_SEED,
    DualViewCrossAttnClassifier,
    compute_features,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
)
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
    fpr_at_target_tpr,
)
from offline_reconstructor_joint_dualview import (
    JointDualDataset,
    JetLevelRegressor,
    compute_jet_regression_targets,
    eval_joint_model,
    predict_jet_regressor,
)


def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _extract_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _infer_dual_input_dims(dual_sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    wa = dual_sd["input_proj_a.weight"]
    wb = dual_sd["input_proj_b.weight"]
    return int(wa.shape[1]), int(wb.shape[1])


def _load_hlt_cfg_from_run_dir(run_dir: Path, cfg: Dict) -> Dict:
    hlt_path = run_dir / "hlt_stats.json"
    if not hlt_path.exists():
        return cfg
    hlt = json.loads(hlt_path.read_text())
    src = hlt.get("config", {})
    for k in list(cfg.get("hlt_effects", {}).keys()):
        if k in src:
            cfg["hlt_effects"][k] = src[k]
    return cfg


def _build_feat_hlt_dual(
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    const_off: np.ndarray,
    masks_off: np.ndarray,
    hlt_const: np.ndarray,
    run_dir: Path,
    dual_input_dim_a: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if dual_input_dim_a == 7:
        return feat_hlt_std.astype(np.float32, copy=True)

    if dual_input_dim_a != 23:
        raise RuntimeError(
            f"Unsupported dual input_dim_a={dual_input_dim_a}. "
            "Expected 7 (no jet reg) or 23 (jet reg extras)."
        )

    jet_ckpt = run_dir / "jet_regressor.pt"
    if not jet_ckpt.exists():
        raise FileNotFoundError(
            f"dual_input_dim_a={dual_input_dim_a} implies jet-reg extras, "
            f"but no jet regressor checkpoint found: {jet_ckpt}"
        )

    # Recreate target transforms exactly as in training to form delta-vs-HLT features.
    target_off, target_hlt_ref, _ = compute_jet_regression_targets(
        const_off=const_off,
        mask_off=masks_off,
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
    )
    target_dim = int(target_off.shape[1])

    jet_sd = _extract_state_dict(jet_ckpt)
    jet_model = JetLevelRegressor(
        input_dim=7,
        output_dim=target_dim,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
    ).to(device)
    jet_model.load_state_dict(jet_sd, strict=True)

    pred_log_all = predict_jet_regressor(
        model=jet_model,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--dual_ckpt", type=str, required=True)
    parser.add_argument("--reco_ckpt", type=str, required=True)
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=50000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    dual_ckpt = Path(args.dual_ckpt)
    reco_ckpt = Path(args.reco_ckpt)
    device = torch.device(args.device)

    cfg = _deepcopy_cfg()
    cfg = _load_hlt_cfg_from_run_dir(run_dir, cfg)

    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = int(args.offset_jets + args.n_train_jets)
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=int(args.max_constits),
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}"
        )

    const_raw = all_const_full[args.offset_jets : args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets : args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    hlt_const, hlt_mask, _, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )
    budget_merge_true = budget_truth["merge_lost_per_jet"].astype(np.float32)
    budget_eff_true = budget_truth["eff_lost_per_jet"].astype(np.float32)

    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=int(args.seed),
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=int(args.seed),
        stratify=labels[temp_idx],
    )

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    dual_sd = _extract_state_dict(dual_ckpt)
    dual_input_dim_a, dual_input_dim_b = _infer_dual_input_dims(dual_sd)
    use_corrected_flags = bool(dual_input_dim_b == 12)

    feat_hlt_dual = _build_feat_hlt_dual(
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        const_off=const_off,
        masks_off=masks_off,
        hlt_const=hlt_const,
        run_dir=run_dir,
        dual_input_dim_a=dual_input_dim_a,
        device=device,
        batch_size=int(args.batch_size),
    )

    reco_model = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reco_sd = _extract_state_dict(reco_ckpt)
    reco_model.load_state_dict(reco_sd, strict=True)

    dual_model = DualViewCrossAttnClassifier(
        input_dim_a=dual_input_dim_a,
        input_dim_b=dual_input_dim_b,
        **cfg["model"],
    ).to(device)
    dual_model.load_state_dict(dual_sd, strict=True)

    ds_test = JointDualDataset(
        feat_hlt_std[test_idx],
        feat_hlt_dual[test_idx],
        hlt_mask[test_idx],
        hlt_const[test_idx],
        const_off[test_idx],
        masks_off[test_idx],
        budget_merge_true[test_idx],
        budget_eff_true[test_idx],
        labels[test_idx],
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    auc, preds, labs, fpr50 = eval_joint_model(
        reco_model,
        dual_model,
        dl_test,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=use_corrected_flags,
    )
    fpr, tpr, _ = roc_curve(labs, preds)
    fpr30 = fpr_at_target_tpr(fpr, tpr, 0.30)
    fpr50_chk = fpr_at_target_tpr(fpr, tpr, 0.50)
    auc_chk = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else float("nan")

    out = {
        "auc": float(auc),
        "auc_check": float(auc_chk),
        "fpr30": float(fpr30),
        "fpr50": float(fpr50),
        "fpr50_check": float(fpr50_chk),
        "n_test": int(len(test_idx)),
        "seed": int(args.seed),
        "n_train_jets": int(args.n_train_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "dual_input_dim_a": int(dual_input_dim_a),
        "dual_input_dim_b": int(dual_input_dim_b),
        "use_corrected_flags": bool(use_corrected_flags),
        "dual_ckpt": str(dual_ckpt),
        "reco_ckpt": str(reco_ckpt),
    }

    print("\nStage-B/Test Evaluation")
    print(f"AUC:    {out['auc']:.6f}")
    print(f"FPR@30: {out['fpr30']:.6f}")
    print(f"FPR@50: {out['fpr50']:.6f}")
    print(f"n_test: {out['n_test']}")

    out_path = run_dir / "stageb_test_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

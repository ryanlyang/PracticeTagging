#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oracle two-view ceiling check for m2 family:
  View-A = HLT features
  View-B = TRUE OFFLINE features

Purpose:
- Isolate fusion capacity from reconstructor quality.
- If this run is strong, the bottleneck is reconstruction (not fusion).
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
from tqdm import tqdm

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as m2
from unmerge_correct_hlt import train_classifier_dual


def train_dual_view_classifier_auc(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> torch.nn.Module:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = m2.get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    best_val_auc = float("-inf")
    best_fpr50 = float("nan")
    best_state = None
    no_improve = 0

    for ep in tqdm(range(int(train_cfg["epochs"])), desc=name):
        _, tr_auc = train_classifier_dual(model, train_loader, opt, device)
        va_auc, va_preds, va_labs = m2.eval_classifier_dual(model, val_loader, device)
        va_fpr, va_tpr, _ = roc_curve(va_labs, va_preds)
        va_fpr50 = m2.fpr_at_target_tpr(va_fpr, va_tpr, 0.50)
        sch.step()

        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_fpr50 = float(va_fpr50)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, "
                f"val_fpr50={va_fpr50:.6f}, best_auc={best_val_auc:.4f}, fpr50@best={best_fpr50:.6f}"
            )

        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def auc_and_fpr_at(labels: np.ndarray, preds: np.ndarray, target_tpr: float) -> Tuple[float, float]:
    if preds.size == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels, preds))
    fpr, tpr, _ = roc_curve(labels, preds)
    fpr_t = float(m2.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))
    return auc, fpr_t


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle two-view ceiling: HLT + TRUE offline")
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--n_train_jets", type=int, default=375000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--n_train_split", type=int, default=150000)
    ap.add_argument("--n_val_split", type=int, default=75000)
    ap.add_argument("--n_test_split", type=int, default=150000)
    ap.add_argument(
        "--save_dir",
        type=str,
        default=str(Path().cwd() / "checkpoints" / "reco_teacher_joint_fusion_6model_150k75k150k" / "model2_oracle_hlt_trueoffline_dualview"),
    )
    ap.add_argument("--run_name", type=str, default="model2_oracle_hlt_trueoffline_dualview_150k75k150k_seed0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_fusion_scores", action="store_true")

    # Optional training overrides.
    ap.add_argument("--train_epochs", type=int, default=-1)
    ap.add_argument("--train_patience", type=int, default=-1)
    ap.add_argument("--train_batch_size", type=int, default=-1)
    ap.add_argument("--train_lr", type=float, default=-1.0)
    ap.add_argument("--train_weight_decay", type=float, default=-1.0)
    ap.add_argument("--train_warmup_epochs", type=int, default=-1)

    args = ap.parse_args()

    m2.set_seed(int(args.seed))
    cfg = m2._deepcopy_config()
    if int(args.train_epochs) > 0:
        cfg["training"]["epochs"] = int(args.train_epochs)
    if int(args.train_patience) > 0:
        cfg["training"]["patience"] = int(args.train_patience)
    if int(args.train_batch_size) > 0:
        cfg["training"]["batch_size"] = int(args.train_batch_size)
    if float(args.train_lr) > 0.0:
        cfg["training"]["lr"] = float(args.train_lr)
    if float(args.train_weight_decay) >= 0.0:
        cfg["training"]["weight_decay"] = float(args.train_weight_decay)
    if int(args.train_warmup_epochs) >= 0:
        cfg["training"]["warmup_epochs"] = int(args.train_warmup_epochs)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = int(args.offset_jets) + int(args.n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = m2.load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}")

    const_raw = all_const_full[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, _, _ = m2.apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )

    print("Computing features...")
    features_off = m2.compute_features(const_off, masks_off)
    features_hlt = m2.compute_features(hlt_const, hlt_mask)

    idx = np.arange(len(labels))
    total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
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
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} (custom_counts=True)")

    means, stds = m2.get_stats(features_off, masks_off, train_idx)
    features_off_std = m2.standardize(features_off, masks_off, means, stds)
    features_hlt_std = m2.standardize(features_hlt, hlt_mask, means, stds)

    bs = int(cfg["training"]["batch_size"])
    pin = torch.cuda.is_available()

    # Teacher (offline only)
    train_ds_off = m2.JetDataset(features_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    val_ds_off = m2.JetDataset(features_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    test_ds_off = m2.JetDataset(features_off_std[test_idx], masks_off[test_idx], labels[test_idx])
    train_loader_off = DataLoader(train_ds_off, batch_size=bs, shuffle=True, drop_last=True, num_workers=int(args.num_workers), pin_memory=pin)
    val_loader_off = DataLoader(val_ds_off, batch_size=bs, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    test_loader_off = DataLoader(test_ds_off, batch_size=bs, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)

    # Baseline (hlt only)
    train_ds_hlt = m2.JetDataset(features_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx])
    val_ds_hlt = m2.JetDataset(features_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    test_ds_hlt = m2.JetDataset(features_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    train_loader_hlt = DataLoader(train_ds_hlt, batch_size=bs, shuffle=True, drop_last=True, num_workers=int(args.num_workers), pin_memory=pin)
    val_loader_hlt = DataLoader(val_ds_hlt, batch_size=bs, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    test_loader_hlt = DataLoader(test_ds_hlt, batch_size=bs, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)

    # Oracle dual-view (hlt + true offline)
    kd_train_ds = m2.DualViewKDDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        labels[train_idx],
    )
    kd_val_ds = m2.DualViewKDDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        labels[val_idx],
    )
    kd_test_ds = m2.DualViewKDDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        labels[test_idx],
    )
    kd_train_loader = DataLoader(kd_train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=int(args.num_workers), pin_memory=pin)
    kd_val_loader = DataLoader(kd_val_ds, batch_size=bs, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    kd_test_loader = DataLoader(kd_test_ds, batch_size=bs, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)

    print("\n" + "=" * 70)
    print("STEP 1: TEACHER + BASELINE")
    print("=" * 70)
    teacher = m2.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = m2.train_single_view_classifier_auc(teacher, train_loader_off, val_loader_off, device, cfg["training"], name="Teacher")
    auc_teacher_val, preds_teacher_val, y_val = m2.eval_classifier(teacher, val_loader_off, device)
    auc_teacher_test, preds_teacher_test, y_test = m2.eval_classifier(teacher, test_loader_off, device)

    baseline = m2.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = m2.train_single_view_classifier_auc(baseline, train_loader_hlt, val_loader_hlt, device, cfg["training"], name="Baseline")
    auc_baseline_val, preds_baseline_val, _ = m2.eval_classifier(baseline, val_loader_hlt, device)
    auc_baseline_test, preds_baseline_test, _ = m2.eval_classifier(baseline, test_loader_hlt, device)

    print("\n" + "=" * 70)
    print("STEP 2: ORACLE DUAL-VIEW (HLT + TRUE OFFLINE)")
    print("=" * 70)
    dual_oracle = m2.DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **cfg["model"]).to(device)
    dual_oracle = train_dual_view_classifier_auc(
        dual_oracle,
        kd_train_loader,
        kd_val_loader,
        device,
        cfg["training"],
        name="OracleDualView",
    )
    auc_oracle_val, preds_oracle_val, _ = m2.eval_classifier_dual(dual_oracle, kd_val_loader, device)
    auc_oracle_test, preds_oracle_test, _ = m2.eval_classifier_dual(dual_oracle, kd_test_loader, device)

    _, teacher_fpr30 = auc_and_fpr_at(y_test, preds_teacher_test, 0.30)
    _, baseline_fpr30 = auc_and_fpr_at(y_test, preds_baseline_test, 0.30)
    _, oracle_fpr30 = auc_and_fpr_at(y_test, preds_oracle_test, 0.30)

    _, teacher_fpr50 = auc_and_fpr_at(y_test, preds_teacher_test, 0.50)
    _, baseline_fpr50 = auc_and_fpr_at(y_test, preds_baseline_test, 0.50)
    _, oracle_fpr50 = auc_and_fpr_at(y_test, preds_oracle_test, 0.50)

    metrics = {
        "run_dir": str(save_root),
        "model": "m2_oracle_hlt_trueoffline_dualview",
        "split_counts": {
            "n_train_split": int(len(train_idx)),
            "n_val_split": int(len(val_idx)),
            "n_test_split": int(len(test_idx)),
        },
        "val_auc": {
            "teacher_offline": float(auc_teacher_val),
            "baseline_hlt": float(auc_baseline_val),
            "oracle_dual_hlt_offline": float(auc_oracle_val),
        },
        "test_auc": {
            "teacher_offline": float(auc_teacher_test),
            "baseline_hlt": float(auc_baseline_test),
            "oracle_dual_hlt_offline": float(auc_oracle_test),
        },
        "test_fpr30": {
            "teacher_offline": float(teacher_fpr30),
            "baseline_hlt": float(baseline_fpr30),
            "oracle_dual_hlt_offline": float(oracle_fpr30),
        },
        "test_fpr50": {
            "teacher_offline": float(teacher_fpr50),
            "baseline_hlt": float(baseline_fpr50),
            "oracle_dual_hlt_offline": float(oracle_fpr50),
        },
    }

    with open(save_root / "oracle_hlt_trueoffline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.save_fusion_scores:
        np.savez_compressed(
            save_root / "oracle_hlt_trueoffline_scores_val_test.npz",
            y_val=y_val.astype(np.float32),
            y_test=y_test.astype(np.float32),
            preds_teacher_val=preds_teacher_val.astype(np.float32),
            preds_teacher_test=preds_teacher_test.astype(np.float32),
            preds_baseline_val=preds_baseline_val.astype(np.float32),
            preds_baseline_test=preds_baseline_test.astype(np.float32),
            preds_oracle_dual_val=preds_oracle_val.astype(np.float32),
            preds_oracle_dual_test=preds_oracle_test.astype(np.float32),
        )
        print(f"Saved score arrays to: {save_root / 'oracle_hlt_trueoffline_scores_val_test.npz'}")

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION (Oracle HLT+Offline Fusion)")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher_test:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline_test:.4f}")
    print(f"Oracle Dual-View AUC: {auc_oracle_test:.4f}")
    print("")
    print(
        "FPR@30 Teacher/Baseline/OracleDual: "
        f"{teacher_fpr30:.6f} / {baseline_fpr30:.6f} / {oracle_fpr30:.6f}"
    )
    print(
        "FPR@50 Teacher/Baseline/OracleDual: "
        f"{teacher_fpr50:.6f} / {baseline_fpr50:.6f} / {oracle_fpr50:.6f}"
    )
    print(f"\nSaved metrics to: {save_root / 'oracle_hlt_trueoffline_metrics.json'}")
    print(f"Done: {save_root}")


if __name__ == "__main__":
    main()

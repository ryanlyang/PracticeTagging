#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ablation: does over-adding constituents hurt top-tagging?

This script:
1) Loads offline jets.
2) Generates pseudo-HLT with the same realistic HLT effects used in the unmerge-only pipeline.
3) Builds an "added" view by concatenating each jet's Offline + HLT constituents.
4) Trains/evaluates three single-view taggers on identical splits:
   - Teacher (Offline view)
   - Baseline (HLT view)
   - Added view (Offline+HLT constituents)

Outputs:
- metrics_summary.json
- metrics_summary.npz
- roc_compare_logfpr.png
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
)
from unmerge_correct_hlt import (
    JetDataset,
    ParticleTransformer,
    compute_features,
    eval_classifier,
    get_scheduler,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
    train_classifier,
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


def fpr_at_target_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    target = float(np.clip(target_tpr, 0.0, 1.0))
    idx = int(np.argmin(np.abs(tpr - target)))
    return float(fpr[idx])


def _train_single_view_classifier_auc(
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
    sch = get_scheduler(
        opt,
        int(train_cfg["warmup_epochs"]),
        int(train_cfg["epochs"]),
    )

    best_val_auc = float("-inf")
    best_state = None
    no_improve = 0

    for ep in tqdm(range(int(train_cfg["epochs"])), desc=name):
        _, tr_auc = train_classifier(model, train_loader, opt, device)
        va_auc, va_preds, va_labs = eval_classifier(model, val_loader, device)
        va_fpr, va_tpr, _ = roc_curve(va_labs, va_preds)
        va_fpr50 = fpr_at_target_tpr(va_fpr, va_tpr, 0.50)
        sch.step()

        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_auc={best_val_auc:.4f}"
            )
        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _build_added_view(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    added_max_constits: int,
    sort_by_pt: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    add_const = np.concatenate([const_off, const_hlt], axis=1)
    add_mask = np.concatenate([mask_off, mask_hlt], axis=1)

    if sort_by_pt:
        pt = add_const[:, :, 0].copy()
        pt[~add_mask] = -1.0
        order = np.argsort(-pt, axis=1)
        add_const = np.take_along_axis(add_const, order[:, :, None], axis=1)
        add_mask = np.take_along_axis(add_mask, order, axis=1)

    if added_max_constits > 0 and add_const.shape[1] > added_max_constits:
        add_const = add_const[:, :added_max_constits, :]
        add_mask = add_mask[:, :added_max_constits]

    add_const = add_const.astype(np.float32, copy=False)
    add_mask = add_mask.astype(bool, copy=False)
    add_const[~add_mask] = 0.0
    return add_const, add_mask


def _plot_roc(
    lines: list,
    save_path: Path,
    min_fpr: float = 1e-4,
) -> None:
    plt.figure(figsize=(8, 6))
    for tpr, fpr, style, label, color in lines:
        plt.plot(tpr, fpr, style, lw=2.0, color=color, label=label)
    plt.yscale("log")
    plt.xlim([0.0, 1.0])
    plt.ylim([min_fpr, 1.0])
    plt.xlabel("TPR")
    plt.ylabel("FPR")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation for over-adding constituents: Offline vs HLT vs Offline+HLT.",
    )
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="checkpoints/added_constituents_ablation")
    parser.add_argument("--run_name", type=str, default="added_constituents_eval")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--n_train_jets", type=int, default=100000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--added_max_constits", type=int, default=-1)
    parser.add_argument("--n_train_split", type=int, default=-1)
    parser.add_argument("--n_val_split", type=int, default=-1)
    parser.add_argument("--n_test_split", type=int, default=-1)

    parser.add_argument("--merge_radius", type=float, default=0.01)
    parser.add_argument("--eff_plateau_barrel", type=float, default=0.98)
    parser.add_argument("--eff_plateau_endcap", type=float, default=0.94)
    parser.add_argument("--smear_a", type=float, default=0.35)
    parser.add_argument("--smear_b", type=float, default=0.012)
    parser.add_argument("--smear_c", type=float, default=0.08)

    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=-1.0)
    parser.add_argument("--weight_decay", type=float, default=-1.0)
    parser.add_argument("--warmup_epochs", type=int, default=-1)

    args = parser.parse_args()
    set_seed(int(args.seed))

    cfg = deepcopy(BASE_CONFIG)
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    if int(args.batch_size) > 0:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if int(args.epochs) > 0:
        cfg["training"]["epochs"] = int(args.epochs)
    if int(args.patience) > 0:
        cfg["training"]["patience"] = int(args.patience)
    if float(args.lr) > 0.0:
        cfg["training"]["lr"] = float(args.lr)
    if float(args.weight_decay) > 0.0:
        cfg["training"]["weight_decay"] = float(args.weight_decay)
    if int(args.warmup_epochs) >= 0:
        cfg["training"]["warmup_epochs"] = int(args.warmup_epochs)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
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

    max_jets_needed = int(args.offset_jets + args.n_train_jets)
    print("Loading offline constituents...")
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
    masks_off = raw_mask & (
        const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"])
    )
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, _, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )

    if int(args.added_max_constits) > 0:
        added_max_constits = int(args.added_max_constits)
    else:
        added_max_constits = int(args.max_constits) * 2

    print("Building added view (Offline + HLT)...")
    add_const, add_mask = _build_added_view(
        const_off=const_off,
        mask_off=masks_off,
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
        added_max_constits=added_max_constits,
        sort_by_pt=True,
    )

    print("Computing features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)
    feat_add = compute_features(add_const, add_mask)

    idx = np.arange(len(labels))
    custom_split = (
        int(args.n_train_split) > 0
        and int(args.n_val_split) > 0
        and int(args.n_test_split) > 0
    )
    if custom_split:
        n_train_split = int(args.n_train_split)
        n_val_split = int(args.n_val_split)
        n_test_split = int(args.n_test_split)
        total_need = n_train_split + n_val_split + n_test_split
        if total_need > len(idx):
            raise ValueError(
                f"Requested split counts exceed available jets: "
                f"{n_train_split}+{n_val_split}+{n_test_split} > {len(idx)}"
            )
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
            train_size=n_train_split,
            random_state=int(args.seed),
            stratify=labels[idx_use],
        )
        val_idx, test_idx = train_test_split(
            rem_idx,
            train_size=n_val_split,
            test_size=n_test_split,
            random_state=int(args.seed),
            stratify=labels[rem_idx],
        )
    else:
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
    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} "
        f"(custom_counts={custom_split})"
    )

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)
    feat_add_std = standardize(feat_add, add_mask, means, stds)

    print(
        f"Mean constituents/jet: offline={float(masks_off.sum(axis=1).mean()):.2f}, "
        f"hlt={float(hlt_mask.sum(axis=1).mean()):.2f}, "
        f"added={float(add_mask.sum(axis=1).mean()):.2f}"
    )

    bs = int(cfg["training"]["batch_size"])
    nw = int(args.num_workers)
    pin = torch.cuda.is_available()

    def build_loaders(feat: np.ndarray, mask: np.ndarray):
        ds_tr = JetDataset(feat[train_idx], mask[train_idx], labels[train_idx])
        ds_va = JetDataset(feat[val_idx], mask[val_idx], labels[val_idx])
        ds_te = JetDataset(feat[test_idx], mask[test_idx], labels[test_idx])
        dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw, pin_memory=pin)
        dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
        dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
        return dl_tr, dl_va, dl_te

    dl_train_off, dl_val_off, dl_test_off = build_loaders(feat_off_std, masks_off)
    dl_train_hlt, dl_val_hlt, dl_test_hlt = build_loaders(feat_hlt_std, hlt_mask)
    dl_train_add, dl_val_add, dl_test_add = build_loaders(feat_add_std, add_mask)

    model_kwargs = dict(cfg["model"])
    train_cfg = dict(cfg["training"])

    print("\n" + "=" * 70)
    print("Training Teacher (Offline)")
    print("=" * 70)
    teacher = ParticleTransformer(input_dim=7, **model_kwargs).to(device)
    teacher = _train_single_view_classifier_auc(
        teacher, dl_train_off, dl_val_off, device, train_cfg, name="Teacher"
    )
    auc_teacher, preds_teacher, labs_test = eval_classifier(teacher, dl_test_off, device)

    print("\n" + "=" * 70)
    print("Training Baseline (HLT)")
    print("=" * 70)
    baseline = ParticleTransformer(input_dim=7, **model_kwargs).to(device)
    baseline = _train_single_view_classifier_auc(
        baseline, dl_train_hlt, dl_val_hlt, device, train_cfg, name="Baseline-HLT"
    )
    auc_baseline, preds_baseline, labs_hlt = eval_classifier(baseline, dl_test_hlt, device)
    assert np.array_equal(labs_test.astype(np.float32), labs_hlt.astype(np.float32))

    print("\n" + "=" * 70)
    print("Training Added View (Offline + HLT)")
    print("=" * 70)
    added_model = ParticleTransformer(input_dim=7, **model_kwargs).to(device)
    added_model = _train_single_view_classifier_auc(
        added_model, dl_train_add, dl_val_add, device, train_cfg, name="Added-OfflinePlusHLT"
    )
    auc_added, preds_added, labs_added = eval_classifier(added_model, dl_test_add, device)
    assert np.array_equal(labs_test.astype(np.float32), labs_added.astype(np.float32))

    fpr_t, tpr_t, _ = roc_curve(labs_test, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs_test, preds_baseline)
    fpr_a, tpr_a, _ = roc_curve(labs_test, preds_added)

    metrics = {
        "auc_teacher_offline": float(auc_teacher),
        "auc_baseline_hlt": float(auc_baseline),
        "auc_added_offline_plus_hlt": float(auc_added),
        "fpr30_teacher_offline": float(fpr_at_target_tpr(fpr_t, tpr_t, 0.30)),
        "fpr30_baseline_hlt": float(fpr_at_target_tpr(fpr_b, tpr_b, 0.30)),
        "fpr30_added_offline_plus_hlt": float(fpr_at_target_tpr(fpr_a, tpr_a, 0.30)),
        "fpr50_teacher_offline": float(fpr_at_target_tpr(fpr_t, tpr_t, 0.50)),
        "fpr50_baseline_hlt": float(fpr_at_target_tpr(fpr_b, tpr_b, 0.50)),
        "fpr50_added_offline_plus_hlt": float(fpr_at_target_tpr(fpr_a, tpr_a, 0.50)),
        "mean_constituents_offline": float(masks_off.sum(axis=1).mean()),
        "mean_constituents_hlt": float(hlt_mask.sum(axis=1).mean()),
        "mean_constituents_added": float(add_mask.sum(axis=1).mean()),
        "added_max_constits": int(added_max_constits),
        "n_jets": int(len(labels)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }

    print("\n" + "=" * 70)
    print("FINAL TEST METRICS")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {metrics['auc_teacher_offline']:.4f}")
    print(f"Baseline (HLT)   AUC: {metrics['auc_baseline_hlt']:.4f}")
    print(f"Added (Off+HLT)  AUC: {metrics['auc_added_offline_plus_hlt']:.4f}")
    print(
        "FPR@30 Teacher/Baseline/Added: "
        f"{metrics['fpr30_teacher_offline']:.6f} / "
        f"{metrics['fpr30_baseline_hlt']:.6f} / "
        f"{metrics['fpr30_added_offline_plus_hlt']:.6f}"
    )
    print(
        "FPR@50 Teacher/Baseline/Added: "
        f"{metrics['fpr50_teacher_offline']:.6f} / "
        f"{metrics['fpr50_baseline_hlt']:.6f} / "
        f"{metrics['fpr50_added_offline_plus_hlt']:.6f}"
    )

    _plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher Offline (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"Baseline HLT (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_a, fpr_a, "-.", f"Added Off+HLT (AUC={auc_added:.3f})", "darkgreen"),
        ],
        save_root / "roc_compare_logfpr.png",
        min_fpr=1e-4,
    )

    np.savez_compressed(
        save_root / "metrics_summary.npz",
        labs=labs_test.astype(np.float32),
        preds_teacher=preds_teacher.astype(np.float32),
        preds_baseline=preds_baseline.astype(np.float32),
        preds_added=preds_added.astype(np.float32),
        fpr_teacher=fpr_t.astype(np.float32),
        tpr_teacher=tpr_t.astype(np.float32),
        fpr_baseline=fpr_b.astype(np.float32),
        tpr_baseline=tpr_b.astype(np.float32),
        fpr_added=fpr_a.astype(np.float32),
        tpr_added=tpr_a.astype(np.float32),
    )
    with open(save_root / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(save_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "hlt_effects": cfg["hlt_effects"],
                "training": cfg["training"],
                "model": cfg["model"],
            },
            f,
            indent=2,
        )

    print(f"\nSaved outputs to: {save_root}")


if __name__ == "__main__":
    main()

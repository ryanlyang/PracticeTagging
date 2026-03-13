#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate an existing unmerge_model run on a NEW, non-overlapping jet subset.

Workflow:
  1) Load enough jets to cover a base subset (for stats) and a new eval subset.
  2) Recompute feature standardization stats on the BASE subset (first N jets).
  3) Recompute unmerge target normalization (tgt_mean/std) on the BASE subset
     using the saved merge-count predictor (to match training-time scaling).
  4) Use saved merge-count + unmerge predictor to build unmerged dataset
     on the NEW eval subset.
  5) Run teacher, baseline, unmerge classifier, and unmerge+KD (if present)
     on the NEW eval subset.
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import unmerge_model as um


def _wrap_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unmerge_model" / "default"))
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unmerge_model_eval"))
    parser.add_argument("--run_name", type=str, default="eval_newdata")

    parser.add_argument("--n_eval_jets", type=int, default=200000)
    parser.add_argument("--offset_jets", type=int, default=200000, help="Start index for eval jets (skip first N jets).")
    parser.add_argument("--stats_jets", type=int, default=200000, help="Number of base jets used to compute stats.")
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--max_merge_count", type=int, default=10)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=um.RANDOM_SEED)

    args = parser.parse_args()

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    device = torch.device(args.device)

    if args.stats_jets > args.offset_jets:
        raise ValueError("stats_jets must be <= offset_jets to ensure a non-overlapping eval set.")

    # Load data
    train_path = Path(args.train_path)
    train_files = sorted(list(train_path.glob("*.h5")))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {train_path}")

    max_needed = max(args.offset_jets + args.n_eval_jets, args.stats_jets)
    print("Loading data via utils.load_from_files...")
    all_data, all_labels, _, _, _ = um.utils.load_from_files(
        train_files,
        max_jets=max_needed,
        max_constits=args.max_constits,
        use_train_weights=False,
    )
    all_labels = all_labels.astype(np.int64)
    print(f"Loaded: data={all_data.shape}, labels={all_labels.shape}")

    if all_data.shape[0] < args.offset_jets + args.n_eval_jets:
        raise RuntimeError(
            f"Not enough jets available. Need {args.offset_jets + args.n_eval_jets}, got {all_data.shape[0]}."
        )

    base_data = all_data[:args.stats_jets]
    base_labels = all_labels[:args.stats_jets]
    eval_data = all_data[args.offset_jets:args.offset_jets + args.n_eval_jets]
    eval_labels = all_labels[args.offset_jets:args.offset_jets + args.n_eval_jets]

    # Build constituents for base + eval
    def build_const(data):
        eta = data[:, :, um.ETA_IDX].astype(np.float32)
        phi = data[:, :, um.PHI_IDX].astype(np.float32)
        pt = data[:, :, um.PT_IDX].astype(np.float32)
        mask_raw = pt > 0
        E = pt * np.cosh(np.clip(eta, -5, 5))
        const = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)
        return const, mask_raw

    base_const, base_mask_raw = build_const(base_data)
    eval_const, eval_mask_raw = build_const(eval_data)

    # Apply HLT effects (no smearing, merging + efficiency only per CONFIG)
    print("Applying HLT effects (base subset)...")
    hlt_base, hlt_mask_base, origin_counts_base, origin_lists_base, stats_base = um.apply_hlt_effects_with_tracking(
        base_const, base_mask_raw, um.CONFIG, seed=args.seed
    )
    print("Applying HLT effects (eval subset)...")
    hlt_eval, hlt_mask_eval, _, _, stats_eval = um.apply_hlt_effects_with_tracking(
        eval_const, eval_mask_raw, um.CONFIG, seed=args.seed
    )

    # Offline masks
    pt_threshold_off = um.CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off_base = base_mask_raw & (base_const[:, :, 0] >= pt_threshold_off)
    const_off_base = base_const.copy()
    const_off_base[~masks_off_base] = 0

    masks_off_eval = eval_mask_raw & (eval_const[:, :, 0] >= pt_threshold_off)
    const_off_eval = eval_const.copy()
    const_off_eval[~masks_off_eval] = 0

    print("Base HLT stats:")
    print(f"  Offline particles: {stats_base['n_initial']:,}")
    print(f"  Lost to pT threshold ({um.CONFIG['hlt_effects']['pt_threshold_hlt']}): {stats_base['n_lost_threshold']:,}")
    print(f"  Lost to merging (dR<{um.CONFIG['hlt_effects']['merge_radius']}): {stats_base['n_merged']:,}")
    print(f"  Lost to efficiency: {stats_base['n_lost_eff']:,}")
    print(f"  HLT particles: {stats_base['n_final']:,}")
    print(f"  Avg per jet: Offline={masks_off_base.sum(axis=1).mean():.1f}, HLT={hlt_mask_base.sum(axis=1).mean():.1f}")

    # Compute features
    print("Computing features...")
    features_off_base = um.compute_features(const_off_base, masks_off_base)
    features_hlt_base = um.compute_features(hlt_base, hlt_mask_base)
    features_off_eval = um.compute_features(const_off_eval, masks_off_eval)
    features_hlt_eval = um.compute_features(hlt_eval, hlt_mask_eval)

    # Train/val/test split on base subset (to replicate training stats)
    idx = np.arange(len(base_labels))
    train_idx, temp_idx = um.train_test_split(idx, test_size=0.30, random_state=um.RANDOM_SEED, stratify=base_labels)
    val_idx, test_idx = um.train_test_split(temp_idx, test_size=0.50, random_state=um.RANDOM_SEED, stratify=base_labels[temp_idx])
    print(f"Base split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    feat_means, feat_stds = um.get_stats(features_off_base, masks_off_base, train_idx)
    features_off_base_std = um.standardize(features_off_base, masks_off_base, feat_means, feat_stds)
    features_hlt_base_std = um.standardize(features_hlt_base, hlt_mask_base, feat_means, feat_stds)

    features_off_eval_std = um.standardize(features_off_eval, masks_off_eval, feat_means, feat_stds)
    features_hlt_eval_std = um.standardize(features_hlt_eval, hlt_mask_eval, feat_means, feat_stds)

    # Load merge-count predictor
    max_count = max(int(args.max_merge_count), 2)
    count_ckpt = torch.load(ckpt_dir / "merge_count.pt", map_location=device)
    count_model = um.MergeCountPredictor(input_dim=7, num_classes=max_count, **um.CONFIG["merge_count_model"]).to(device)
    count_model.load_state_dict(count_ckpt["model"])

    bs_cnt = args.batch_size or um.CONFIG["merge_count_training"]["batch_size"]
    pred_counts_base = um.predict_counts(count_model, features_hlt_base_std, hlt_mask_base, bs_cnt, device, max_count)
    pred_counts_eval = um.predict_counts(count_model, features_hlt_eval_std, hlt_mask_eval, bs_cnt, device, max_count)

    # Build tgt_mean/std on base subset (training split)
    print("Computing unmerge target stats from base subset...")
    samples = []
    for j in range(len(base_labels)):
        for idx in range(args.max_constits):
            origin = origin_lists_base[j][idx]
            if hlt_mask_base[j, idx] and len(origin) > 1:
                if len(origin) > max_count:
                    continue
                pc = int(pred_counts_base[j, idx])
                if pc < 2:
                    pc = 2
                if pc > max_count:
                    pc = max_count
                samples.append((j, idx, origin, pc))

    train_idx_set = set(train_idx)
    train_samples = [s for s in samples if s[0] in train_idx_set]
    if len(train_samples) == 0:
        raise RuntimeError("No merged samples in base training split; cannot compute tgt_mean/std.")

    train_targets = [const_off_base[s[0], s[2], :4] for s in train_samples]
    flat_train = np.concatenate(train_targets, axis=0)
    tgt_mean = flat_train.mean(axis=0)
    tgt_std = flat_train.std(axis=0) + 1e-8

    # Load unmerge predictor
    unmerge_ckpt = torch.load(ckpt_dir / "unmerge_predictor.pt", map_location=device)
    unmerge_model = um.UnmergePredictor(input_dim=7, max_count=max_count, **um.CONFIG["unmerge_model"]).to(device)
    unmerge_model.load_state_dict(unmerge_ckpt["model"])

    # Build unmerged dataset for eval subset
    bs_un = args.batch_size or um.CONFIG["unmerge_training"]["batch_size"]
    print("Building unmerged dataset for eval subset...")
    unmerged_const_eval, unmerged_mask_eval = um.build_unmerged_dataset(
        features_hlt_eval_std,
        hlt_mask_eval,
        hlt_eval,
        pred_counts_eval,
        unmerge_model,
        tgt_mean,
        tgt_std,
        max_count,
        args.max_constits,
        device,
        bs_un,
    )

    features_unmerged_eval = um.compute_features(unmerged_const_eval, unmerged_mask_eval)
    features_unmerged_eval_std = um.standardize(features_unmerged_eval, unmerged_mask_eval, feat_means, feat_stds)

    # Load classifiers
    teacher_ckpt = torch.load(ckpt_dir / "teacher.pt", map_location=device)
    baseline_ckpt = torch.load(ckpt_dir / "baseline.pt", map_location=device)
    unmerge_cls_ckpt = torch.load(ckpt_dir / "unmerge_classifier.pt", map_location=device)

    teacher = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    baseline = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    unmerge_cls = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)

    teacher.load_state_dict(teacher_ckpt["model"])
    baseline.load_state_dict(baseline_ckpt["model"])
    unmerge_cls.load_state_dict(unmerge_cls_ckpt["model"])

    unmerge_kd = None
    if (ckpt_dir / "unmerge_kd.pt").exists():
        kd_ckpt = torch.load(ckpt_dir / "unmerge_kd.pt", map_location=device)
        unmerge_kd = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
        unmerge_kd.load_state_dict(kd_ckpt["model"])

    # Build loaders for eval subset (treated as test)
    bs_eval = args.batch_size or um.CONFIG["training"]["batch_size"]
    test_ds_off = um.JetDataset(features_off_eval_std, masks_off_eval, eval_labels)
    test_ds_hlt = um.JetDataset(features_hlt_eval_std, hlt_mask_eval, eval_labels)
    test_ds_unm = um.JetDataset(features_unmerged_eval_std, unmerged_mask_eval, eval_labels)

    test_loader_off = torch.utils.data.DataLoader(test_ds_off, batch_size=bs_eval, shuffle=False, num_workers=args.num_workers)
    test_loader_hlt = torch.utils.data.DataLoader(test_ds_hlt, batch_size=bs_eval, shuffle=False, num_workers=args.num_workers)
    test_loader_unm = torch.utils.data.DataLoader(test_ds_unm, batch_size=bs_eval, shuffle=False, num_workers=args.num_workers)

    print("\nEvaluating classifiers on NEW eval subset...")
    auc_teacher, preds_teacher, labs = um.eval_classifier(teacher, test_loader_off, device)
    auc_baseline, preds_baseline, _ = um.eval_classifier(baseline, test_loader_hlt, device)
    auc_unmerge, preds_unmerge, _ = um.eval_classifier(unmerge_cls, test_loader_unm, device)

    auc_unmerge_kd, preds_unmerge_kd = None, None
    if unmerge_kd is not None:
        auc_unmerge_kd, preds_unmerge_kd, _ = um.eval_classifier(unmerge_kd, test_loader_unm, device)

    print("\nFINAL EVAL (new 200k jets)")
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"Unmerge Model    AUC: {auc_unmerge:.4f}")
    if auc_unmerge_kd is not None:
        print(f"Unmerge + KD     AUC: {auc_unmerge_kd:.4f}")

    # Save results
    np.savez(
        save_root / "results_newdata.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_unmerge=auc_unmerge,
        auc_unmerge_kd=(auc_unmerge_kd if auc_unmerge_kd is not None else np.nan),
        preds_teacher=preds_teacher,
        preds_baseline=preds_baseline,
        preds_unmerge=preds_unmerge,
        preds_unmerge_kd=(preds_unmerge_kd if preds_unmerge_kd is not None else np.array([])),
        labels=labs,
        offset_jets=args.offset_jets,
        n_eval_jets=args.n_eval_jets,
        stats_jets=args.stats_jets,
    )

    # ROC plot
    fpr_t, tpr_t, _ = um.roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = um.roc_curve(labs, preds_baseline)
    fpr_u, tpr_u, _ = um.roc_curve(labs, preds_unmerge)
    lines = [
        (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
        (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
        (tpr_u, fpr_u, ":", f"Unmerge (AUC={auc_unmerge:.3f})", "forestgreen"),
    ]
    if auc_unmerge_kd is not None:
        fpr_k, tpr_k, _ = um.roc_curve(labs, preds_unmerge_kd)
        lines.append((tpr_k, fpr_k, "-.", f"Unmerge+KD (AUC={auc_unmerge_kd:.3f})", "darkorange"))

    plt.figure(figsize=(8, 6))
    for tpr, fpr, style, label, color in lines:
        plt.plot(tpr, fpr, style, label=label, color=color, linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_root / "results_newdata.png", dpi=300)
    plt.close()

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()

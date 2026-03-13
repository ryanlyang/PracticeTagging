#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train unmerge-model classifiers on a noise-injected unmerged dataset
without retraining the unmerger. Uses saved merge-count + unmerge predictor
from an existing unmerge_model run.

Outputs:
  - checkpoints/<save_dir>/<run_name>/{unmerge_noise.pt, unmerge_kd_noise.pt}
  - results.npz + ROC plots
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import unmerge_model as um


def _wrap_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def apply_unmerge_noise_inplace(const, mask, jet_indices, cfg, rng):
    if len(jet_indices) == 0:
        return
    sel = const[jet_indices]
    m = mask[jet_indices]
    valid = m.copy()

    # Drop tokens
    if cfg["drop_prob"] > 0:
        drop = (rng.random(size=m.shape) < cfg["drop_prob"]) & m
        m[drop] = False
        sel[drop] = 0.0
        valid = m

    # Jitter / scale
    if cfg["pt_scale"] > 0:
        scale = rng.normal(1.0, cfg["pt_scale"], size=m.shape)
        scale = np.clip(scale, 0.2, 3.0)
        sel[..., 0] = np.where(valid, sel[..., 0] * scale, 0.0)

    if cfg["pt_jitter"] > 0:
        pt_noise = rng.normal(0.0, cfg["pt_jitter"], size=m.shape)
        sel[..., 0] = np.where(valid, np.clip(sel[..., 0] + pt_noise, 0.0, None), 0.0)

    if cfg["eta_jitter"] > 0:
        eta_noise = rng.normal(0.0, cfg["eta_jitter"], size=m.shape)
        sel[..., 1] = np.where(valid, np.clip(sel[..., 1] + eta_noise, -5.0, 5.0), 0.0)

    if cfg["phi_jitter"] > 0:
        phi_noise = rng.normal(0.0, cfg["phi_jitter"], size=m.shape)
        new_phi = sel[..., 2] + phi_noise
        sel[..., 2] = np.where(valid, _wrap_phi(new_phi), 0.0)

    if cfg["energy_jitter"] > 0 and not cfg["recompute_E"]:
        e_noise = rng.normal(0.0, cfg["energy_jitter"], size=m.shape)
        sel[..., 3] = np.where(valid, np.clip(sel[..., 3] + e_noise, 0.0, None), 0.0)

    if cfg["recompute_E"]:
        sel[..., 3] = np.where(valid, sel[..., 0] * np.cosh(np.clip(sel[..., 1], -5.0, 5.0)), 0.0)

    const[jet_indices] = sel
    mask[jet_indices] = m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unmerge_model_noise"))
    parser.add_argument("--run_name", type=str, default="noise_injection")
    parser.add_argument("--ckpt_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unmerge_model" / "default"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_save_models", action="store_true")

    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--max_merge_count", type=int, default=10)

    # Noise injection knobs
    parser.add_argument("--drop_prob", type=float, default=0.05)
    parser.add_argument("--pt_scale", type=float, default=0.10)
    parser.add_argument("--pt_jitter", type=float, default=0.0)
    parser.add_argument("--eta_jitter", type=float, default=0.01)
    parser.add_argument("--phi_jitter", type=float, default=0.01)
    parser.add_argument("--energy_jitter", type=float, default=0.0)
    parser.add_argument("--no_recompute_E", action="store_true")
    parser.add_argument("--noise_seed", type=int, default=1337)

    args = parser.parse_args()

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Save dir: {save_root}")
    print(f"Using checkpoints from: {args.ckpt_dir}")

    # Load data
    train_path = Path(args.train_path)
    train_files = sorted(list(train_path.glob("*.h5")))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {train_path}")

    print("Loading data via utils.load_from_files...")
    all_data, all_labels, _, _, _ = um.utils.load_from_files(
        train_files,
        max_jets=args.n_train_jets,
        max_constits=args.max_constits,
        use_train_weights=False,
    )
    all_labels = all_labels.astype(np.int64)
    print(f"Loaded: data={all_data.shape}, labels={all_labels.shape}")

    eta = all_data[:, :, um.ETA_IDX].astype(np.float32)
    phi = all_data[:, :, um.PHI_IDX].astype(np.float32)
    pt = all_data[:, :, um.PT_IDX].astype(np.float32)
    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    constituents_raw = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    print("Applying HLT effects...")
    hlt_const, hlt_mask, origin_counts, origin_lists, stats = um.apply_hlt_effects_with_tracking(
        constituents_raw, mask_raw, um.CONFIG, seed=um.RANDOM_SEED
    )
    pt_threshold_off = um.CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    const_off = constituents_raw.copy()
    const_off[~masks_off] = 0

    print("Computing features...")
    features_off = um.compute_features(const_off, masks_off)
    features_hlt = um.compute_features(hlt_const, hlt_mask)

    idx = np.arange(len(all_labels))
    train_idx, temp_idx = um.train_test_split(idx, test_size=0.30, random_state=um.RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = um.train_test_split(
        temp_idx, test_size=0.50, random_state=um.RANDOM_SEED, stratify=all_labels[temp_idx]
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    feat_means, feat_stds = um.get_stats(features_off, masks_off, train_idx)
    features_off_std = um.standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = um.standardize(features_hlt, hlt_mask, feat_means, feat_stds)

    max_count = max(int(args.max_merge_count), 2)

    # Load checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    teacher_ckpt = ckpt_dir / "teacher.pt"
    baseline_ckpt = ckpt_dir / "baseline.pt"
    count_ckpt = ckpt_dir / "merge_count.pt"
    unmerge_ckpt = ckpt_dir / "unmerge_predictor.pt"

    teacher = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    baseline = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    count_model = um.MergeCountPredictor(input_dim=7, num_classes=max_count, **um.CONFIG["merge_count_model"]).to(device)
    unmerge_model = um.UnmergePredictor(input_dim=7, max_count=max_count, **um.CONFIG["unmerge_model"]).to(device)

    if teacher_ckpt.exists():
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device)["model"])
        print(f"Loaded teacher: {teacher_ckpt}")
    if baseline_ckpt.exists():
        baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device)["model"])
        print(f"Loaded baseline: {baseline_ckpt}")
    if count_ckpt.exists():
        count_model.load_state_dict(torch.load(count_ckpt, map_location=device)["model"])
        print(f"Loaded merge-count: {count_ckpt}")
    if unmerge_ckpt.exists():
        unmerge_model.load_state_dict(torch.load(unmerge_ckpt, map_location=device)["model"])
        print(f"Loaded unmerge predictor: {unmerge_ckpt}")

    # Predict counts and build unmerged dataset
    BS_cnt = um.CONFIG["merge_count_training"]["batch_size"]
    pred_counts = um.predict_counts(count_model, features_hlt_std, hlt_mask, BS_cnt, device, max_count)

    # Build target stats from train samples (same as unmerge_model)
    samples = []
    for j in range(len(all_labels)):
        for idx_t in range(args.max_constits):
            origin = origin_lists[j][idx_t]
            if hlt_mask[j, idx_t] and len(origin) > 1:
                if len(origin) > max_count:
                    continue
                pc = int(pred_counts[j, idx_t])
                if pc < 2:
                    pc = 2
                if pc > max_count:
                    pc = max_count
                samples.append((j, idx_t, origin, pc))
    train_idx_set = set(train_idx)
    train_samples = [s for s in samples if s[0] in train_idx_set]
    train_targets = [const_off[s[0], s[2], :4] for s in train_samples]
    flat_train = np.concatenate(train_targets, axis=0)
    tgt_mean = flat_train.mean(axis=0)
    tgt_std = flat_train.std(axis=0) + 1e-8

    BS_un = um.CONFIG["unmerge_training"]["batch_size"]
    unmerged_const, unmerged_mask = um.build_unmerged_dataset(
        features_hlt_std,
        hlt_mask,
        hlt_const,
        pred_counts,
        unmerge_model,
        tgt_mean,
        tgt_std,
        max_count,
        args.max_constits,
        device,
        BS_un,
    )

    # Apply noise injection on train indices only
    noise_cfg = {
        "drop_prob": float(args.drop_prob),
        "pt_scale": float(args.pt_scale),
        "pt_jitter": float(args.pt_jitter),
        "eta_jitter": float(args.eta_jitter),
        "phi_jitter": float(args.phi_jitter),
        "energy_jitter": float(args.energy_jitter),
        "recompute_E": (not args.no_recompute_E),
    }
    rng = np.random.default_rng(args.noise_seed)
    unmerged_const_noisy = unmerged_const.copy()
    unmerged_mask_noisy = unmerged_mask.copy()
    apply_unmerge_noise_inplace(unmerged_const_noisy, unmerged_mask_noisy, train_idx, noise_cfg, rng)

    # Features for noisy unmerged
    features_unmerged = um.compute_features(unmerged_const_noisy, unmerged_mask_noisy)
    features_unmerged_std = um.standardize(features_unmerged, unmerged_mask_noisy, feat_means, feat_stds)

    # Datasets/loaders
    BS = um.CONFIG["training"]["batch_size"]
    train_ds_um = um.JetDataset(features_unmerged_std[train_idx], unmerged_mask_noisy[train_idx], all_labels[train_idx])
    val_ds_um = um.JetDataset(features_unmerged_std[val_idx], unmerged_mask_noisy[val_idx], all_labels[val_idx])
    test_ds_um = um.JetDataset(features_unmerged_std[test_idx], unmerged_mask_noisy[test_idx], all_labels[test_idx])
    train_loader_um = um.DataLoader(train_ds_um, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_um = um.DataLoader(val_ds_um, batch_size=BS, shuffle=False)
    test_loader_um = um.DataLoader(test_ds_um, batch_size=BS, shuffle=False)

    # Baseline/teacher eval
    test_ds_off = um.JetDataset(features_off_std[test_idx], masks_off[test_idx], all_labels[test_idx])
    test_ds_hlt = um.JetDataset(features_hlt_std[test_idx], hlt_mask[test_idx], all_labels[test_idx])
    test_loader_off = um.DataLoader(test_ds_off, batch_size=BS, shuffle=False)
    test_loader_hlt = um.DataLoader(test_ds_hlt, batch_size=BS, shuffle=False)
    auc_teacher, preds_teacher, labs = um.eval_classifier(teacher, test_loader_off, device)
    auc_baseline, preds_baseline, _ = um.eval_classifier(baseline, test_loader_hlt, device)

    # Train classifier on noisy unmerged
    print("\n" + "=" * 70)
    print("TRAIN: UNMERGE CLASSIFIER (noise-injected train)")
    print("=" * 70)
    unmerge_cls = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    opt_u = torch.optim.AdamW(unmerge_cls.parameters(), lr=um.CONFIG["training"]["lr"], weight_decay=um.CONFIG["training"]["weight_decay"])
    sch_u = um.get_scheduler(opt_u, um.CONFIG["training"]["warmup_epochs"], um.CONFIG["training"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(um.CONFIG["training"]["epochs"]), desc="UnmergeNoise"):
        _, train_auc = um.train_classifier(unmerge_cls, train_loader_um, opt_u, device)
        val_auc, _, _ = um.eval_classifier(unmerge_cls, val_loader_um, device)
        sch_u.step()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in unmerge_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")
        if no_improve >= um.CONFIG["training"]["patience"]:
            print(f"Early stopping at epoch {ep+1}")
            break
    if best_state is not None:
        unmerge_cls.load_state_dict(best_state)
    auc_unmerge, preds_unmerge, _ = um.eval_classifier(unmerge_cls, test_loader_um, device)

    # Train KD student on noisy unmerged
    print("\n" + "=" * 70)
    print("TRAIN: UNMERGE + KD (noise-injected train)")
    print("=" * 70)
    kd_train_ds = um.UnmergeKDDataset(
        features_unmerged_std[train_idx],
        unmerged_mask_noisy[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        all_labels[train_idx],
    )
    kd_val_ds = um.UnmergeKDDataset(
        features_unmerged_std[val_idx],
        unmerged_mask_noisy[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        all_labels[val_idx],
    )
    kd_test_ds = um.UnmergeKDDataset(
        features_unmerged_std[test_idx],
        unmerged_mask_noisy[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        all_labels[test_idx],
    )
    kd_train_loader = um.DataLoader(kd_train_ds, batch_size=BS, shuffle=True, drop_last=True)
    kd_val_loader = um.DataLoader(kd_val_ds, batch_size=BS, shuffle=False)
    kd_test_loader = um.DataLoader(kd_test_ds, batch_size=BS, shuffle=False)

    kd_student = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    opt_kd = torch.optim.AdamW(kd_student.parameters(), lr=um.CONFIG["training"]["lr"], weight_decay=um.CONFIG["training"]["weight_decay"])
    sch_kd = um.get_scheduler(opt_kd, um.CONFIG["training"]["warmup_epochs"], um.CONFIG["training"]["epochs"])

    kd_cfg = um.CONFIG["kd"].copy()
    best_auc_kd, best_state_kd, no_improve = 0.0, None, 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    last_val_loss = None

    for ep in tqdm(range(um.CONFIG["training"]["epochs"]), desc="Unmerge+KDNoise"):
        if not kd_active:
            kd_cfg_ep = kd_cfg.copy()
            kd_cfg_ep["alpha_kd"] = 0.0
        else:
            kd_cfg_ep = kd_cfg
        _, train_auc = um.train_kd_epoch(kd_student, teacher, kd_train_loader, opt_kd, device, kd_cfg_ep)
        val_loss = um.evaluate_bce_loss_unmerged(kd_student, kd_val_loader, device)
        val_auc, _, _ = um.evaluate_kd(kd_student, kd_val_loader, device)
        sch_kd.step()

        if not kd_active and kd_cfg["adaptive_alpha"]:
            if last_val_loss is not None:
                if abs(last_val_loss - val_loss) <= kd_cfg["alpha_stable_delta"]:
                    stable_count += 1
                else:
                    stable_count = 0
            last_val_loss = val_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
                print(f"Activating KD ramp at epoch {ep+1} (val_loss={val_loss:.4f})")

        if val_auc > best_auc_kd:
            best_auc_kd = val_auc
            best_state_kd = {k: v.detach().cpu().clone() for k, v in kd_student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: val_auc={val_auc:.4f}, best={best_auc_kd:.4f} | alpha_kd={kd_cfg_ep['alpha_kd']:.2f}")
        if no_improve >= um.CONFIG["training"]["patience"]:
            print(f"Early stopping KD student at epoch {ep+1}")
            break
    if best_state_kd is not None:
        kd_student.load_state_dict(best_state_kd)
    auc_unmerge_kd, preds_unmerge_kd, _ = um.evaluate_kd(kd_student, kd_test_loader, device)

    # Save models
    if not args.skip_save_models:
        torch.save({"model": unmerge_cls.state_dict(), "auc": auc_unmerge}, save_root / "unmerge_noise.pt")
        torch.save({"model": kd_student.state_dict(), "auc": auc_unmerge_kd}, save_root / "unmerge_kd_noise.pt")

    # ROC curves
    fpr_t, tpr_t, _ = um.roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = um.roc_curve(labs, preds_baseline)
    fpr_u, tpr_u, _ = um.roc_curve(labs, preds_unmerge)
    fpr_k, tpr_k, _ = um.roc_curve(labs, preds_unmerge_kd)

    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={auc_teacher:.3f})", color="crimson", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline (AUC={auc_baseline:.3f})", color="steelblue", linewidth=2)
    plt.plot(tpr_u, fpr_u, ":", label=f"Unmerge+Noise (AUC={auc_unmerge:.3f})", color="forestgreen", linewidth=2)
    plt.plot(tpr_k, fpr_k, "-.", label=f"Unmerge+KD+Noise (AUC={auc_unmerge_kd:.3f})", color="darkorange", linewidth=2)
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_root / "results_all.png", dpi=300)
    plt.close()

    # Save results
    np.savez(
        save_root / "results.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_unmerge_noise=auc_unmerge,
        auc_unmerge_kd_noise=auc_unmerge_kd,
        preds_teacher=preds_teacher,
        preds_baseline=preds_baseline,
        preds_unmerge=preds_unmerge,
        preds_unmerge_kd=preds_unmerge_kd,
        labs=labs,
        noise_cfg=noise_cfg,
    )

    print("\nResults:")
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"Unmerge+Noise    AUC: {auc_unmerge:.4f}")
    print(f"Unmerge+KD+Noise AUC: {auc_unmerge_kd:.4f}")
    print(f"Saved results to: {save_root}")


if __name__ == "__main__":
    main()

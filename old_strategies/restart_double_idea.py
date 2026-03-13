#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Restart script for double_idea.py when it gets killed during consistency training.

This script:
1. Skips teacher, baseline, and mixed training (loads from checkpoints)
2. Trains ONLY the consistency model with reduced batch size to avoid OOM
3. Saves results when done

Usage:
  python restart_double_idea.py --run_name double_test --batch_size 256 --device cuda
"""

import sys
import argparse
from pathlib import Path

# Import everything from double_idea
sys.path.insert(0, str(Path(__file__).parent))
from old_ideas.double_idea import *

def main_restart():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=10000)
    parser.add_argument("--max_constits", type=int, default=80)

    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "transformer_twohlt"))
    parser.add_argument("--run_name", type=str, required=True, help="Must match original run name")

    parser.add_argument("--device", type=str, default="cpu")

    # Batch size (keep same as original)
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")

    # two HLT seeds (must match original)
    parser.add_argument("--hlt_seed_a", type=int, default=123)
    parser.add_argument("--hlt_seed_b", type=int, default=456)

    # consistency hyperparams
    parser.add_argument("--lam_cons", type=float, default=1.0)
    parser.add_argument("--cons_ramp_frac", type=float, default=0.2)
    parser.add_argument("--gate_thr", type=float, default=None)

    args = parser.parse_args()

    save_dir = Path(args.save_dir) / args.run_name
    if not save_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {save_dir}\nMake sure --run_name matches the original run.")

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")
    print(f"Batch size: {args.batch_size}")

    # Check that teacher, baseline, mixed exist
    teacher_path = save_dir / "teacher_offline.pt"
    baseline_path = save_dir / "baseline_hlt_a.pt"
    mixed_path = save_dir / "mixed_hlt_ab.pt"
    consistency_path = save_dir / "consistency_hlt_ab.pt"

    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
    if not mixed_path.exists():
        raise FileNotFoundError(f"Mixed checkpoint not found: {mixed_path}")

    print("\n✓ Found existing checkpoints for teacher, baseline, and mixed")
    print("  Will load these and train ONLY the consistency model\n")

    # ------------------- Load dataset (same as original) ------------------- #
    train_path = Path(args.train_path)
    train_files = sorted(list(train_path.glob("*.h5")))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {train_path}")

    print("Loading data via utils.load_from_files...")
    all_data, all_labels, _, _, all_pt = utils.load_from_files(
        train_files,
        max_jets=args.n_train_jets,
        max_constits=args.max_constits,
        use_train_weights=False,
    )
    all_labels = all_labels.astype(np.int64)
    print(f"Loaded: data={all_data.shape}, labels={all_labels.shape}")

    # ------------------- Convert to [pt, eta, phi, E] ------------------- #
    eta = all_data[:, :, ETA_IDX].astype(np.float32)
    phi = all_data[:, :, PHI_IDX].astype(np.float32)
    pt  = all_data[:, :, PT_IDX].astype(np.float32)

    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    constituents_raw = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    # ------------------- Offline threshold ------------------- #
    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0

    # ------------------- Two HLT views (same seeds as original) ------------------- #
    print(f"\nGenerating HLT-A with seed={args.hlt_seed_a}")
    constituents_hlt_a, masks_hlt_a = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=args.hlt_seed_a)

    print(f"Generating HLT-B with seed={args.hlt_seed_b}")
    constituents_hlt_b, masks_hlt_b = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=args.hlt_seed_b)

    # ------------------- Compute features ------------------- #
    print("\nComputing features...")
    features_off = compute_features(constituents_off, masks_off)
    features_a   = compute_features(constituents_hlt_a, masks_hlt_a)
    features_b   = compute_features(constituents_hlt_b, masks_hlt_b)

    # ------------------- Split (same as original) ------------------- #
    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # ------------------- Standardize ------------------- #
    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_a_std   = standardize(features_a,   masks_hlt_a, feat_means, feat_stds)
    features_b_std   = standardize(features_b,   masks_hlt_b, feat_means, feat_stds)

    # ------------------- Build datasets/loaders with REDUCED batch size ------------------- #
    train_ds = JetDatasetTwoHLT(
        features_off_std[train_idx],
        features_a_std[train_idx],
        features_b_std[train_idx],
        all_labels[train_idx],
        masks_off[train_idx],
        masks_hlt_a[train_idx],
        masks_hlt_b[train_idx],
    )
    val_ds = JetDatasetTwoHLT(
        features_off_std[val_idx],
        features_a_std[val_idx],
        features_b_std[val_idx],
        all_labels[val_idx],
        masks_off[val_idx],
        masks_hlt_a[val_idx],
        masks_hlt_b[val_idx],
    )
    test_ds = JetDatasetTwoHLT(
        features_off_std[test_idx],
        features_a_std[test_idx],
        features_b_std[test_idx],
        all_labels[test_idx],
        masks_off[test_idx],
        masks_hlt_a[test_idx],
        masks_hlt_b[test_idx],
    )

    # REDUCED BATCH SIZE
    BS = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BS, shuffle=False)

    # ------------------- Load teacher, baseline, mixed (for final eval) ------------------- #
    def make_model():
        return ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)

    print("\nLoading existing models...")
    teacher = make_model()
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=False)["model"])
    print(f"✓ Loaded teacher from {teacher_path}")

    baseline = make_model()
    baseline.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=False)["model"])
    print(f"✓ Loaded baseline from {baseline_path}")

    mixed = make_model()
    mixed.load_state_dict(torch.load(mixed_path, map_location=device, weights_only=False)["model"])
    print(f"✓ Loaded mixed from {mixed_path}")

    # ------------------- Train consistency model (fresh or resume) ------------------- #
    print("\n" + "=" * 70)
    print("TRAINING CONSISTENCY MODEL")
    print("=" * 70)

    cons = make_model()

    # Check if consistency checkpoint exists (partial training)
    if consistency_path.exists():
        print(f"\n⚠ Found existing consistency checkpoint: {consistency_path}")
        print("Loading it and continuing training from where it left off...")
        ckpt = torch.load(consistency_path, map_location=device, weights_only=False)
        cons.load_state_dict(ckpt["model"])
        best_auc_cons = ckpt.get("auc", 0.0)
        history_cons = ckpt.get("history", [])
        start_epoch = len(history_cons)
        print(f"Resuming from epoch {start_epoch}, best val AUC so far: {best_auc_cons:.4f}")
    else:
        print("No existing consistency checkpoint found. Training from scratch.")
        best_auc_cons = 0.0
        history_cons = []
        start_epoch = 0

    opt = torch.optim.AdamW(cons.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

    # Fast-forward scheduler to current epoch
    for _ in range(start_epoch):
        sch.step()

    best_state = {k: v.detach().cpu().clone() for k, v in cons.state_dict().items()}
    no_improve = 0

    total_epochs = CONFIG["training"]["epochs"]

    for ep in tqdm(range(start_epoch, total_epochs), desc="Consistency", initial=start_epoch, total=total_epochs):
        lam = ramp_value(ep, total_epochs, args.lam_cons, args.cons_ramp_frac)
        train_loss, train_auc = train_epoch_consistency(cons, train_loader, opt, device, lam_cons=lam, gate_thr=args.gate_thr)
        val_auc, _, _ = evaluate(cons, val_loader, device, "hlt_a", "mask_a")
        sch.step()

        history_cons.append((ep + 1, train_loss, train_auc, val_auc, lam))

        if val_auc > best_auc_cons:
            best_auc_cons = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in cons.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: lam={lam:.3f}, train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_cons:.4f}")

        # Save checkpoint every 5 epochs (in case of another crash)
        if (ep + 1) % 5 == 0:
            torch.save({"model": cons.state_dict(), "auc": best_auc_cons, "history": history_cons}, consistency_path)

        if no_improve >= CONFIG["training"]["patience"] + 5:
            print(f"Early stopping consistency at epoch {ep+1}")
            break

    cons.load_state_dict(best_state)
    torch.save({"model": cons.state_dict(), "auc": best_auc_cons, "history": history_cons}, consistency_path)
    print(f"\n✓ Saved consistency: {consistency_path} (best val AUC={best_auc_cons:.4f})")

    # ------------------- Final evaluation on TEST ------------------- #
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    auc_teacher, preds_teacher, labs = evaluate(teacher, test_loader, device, "off", "mask_off")
    auc_baseline, preds_baseline, _ = evaluate(baseline, test_loader, device, "hlt_a", "mask_a")
    auc_mixed, preds_mixed, _ = evaluate(mixed, test_loader, device, "hlt_a", "mask_a")
    auc_cons, preds_cons, _ = evaluate(cons, test_loader, device, "hlt_a", "mask_a")

    print(f"\n{'Model':<40} {'AUC':>10}")
    print("-" * 52)
    print(f"{'Teacher (Offline test)':<40} {auc_teacher:>10.4f}")
    print(f"{'Baseline (HLT-A test)':<40} {auc_baseline:>10.4f}")
    print(f"{'Mixed (HLT-A test)':<40} {auc_mixed:>10.4f}")
    print(f"{'Consistency (HLT-A test)':<40} {auc_cons:>10.4f}")
    print("-" * 52)

    # Background rejection
    def br_at_wp(labs_np, preds_np, wp=0.5):
        fpr, tpr, _ = roc_curve(labs_np, preds_np)
        idx_wp = np.argmax(tpr >= wp)
        return (1.0 / fpr[idx_wp]) if (fpr[idx_wp] > 0) else 0.0

    br_baseline = br_at_wp(labs, preds_baseline, wp=0.5)
    br_mixed    = br_at_wp(labs, preds_mixed, wp=0.5)
    br_cons     = br_at_wp(labs, preds_cons, wp=0.5)

    print("\nBackground Rejection @ 50% signal efficiency (HLT-A tested):")
    print(f"  Baseline:    {br_baseline:.2f}")
    print(f"  Mixed:       {br_mixed:.2f}")
    print(f"  Consistency: {br_cons:.2f}")

    # ROC curves
    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_m, tpr_m, _ = roc_curve(labs, preds_mixed)
    fpr_c, tpr_c, _ = roc_curve(labs, preds_cons)

    # Save results
    np.savez(
        save_dir / "results_twohlt.npz",
        labs=labs,
        preds_teacher=preds_teacher,
        preds_baseline=preds_baseline,
        preds_mixed=preds_mixed,
        preds_consistency=preds_cons,
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_mixed=auc_mixed,
        auc_consistency=auc_cons,
        br_baseline=br_baseline,
        br_mixed=br_mixed,
        br_consistency=br_cons,
        fpr_teacher=fpr_t, tpr_teacher=tpr_t,
        fpr_baseline=fpr_b, tpr_baseline=tpr_b,
        fpr_mixed=fpr_m, tpr_mixed=tpr_m,
        fpr_consistency=fpr_c, tpr_consistency=tpr_c,
        hlt_seed_a=args.hlt_seed_a,
        hlt_seed_b=args.hlt_seed_b,
        lam_cons=args.lam_cons,
        cons_ramp_frac=args.cons_ramp_frac,
        gate_thr=args.gate_thr,
    )

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-",  label=f"Teacher OFF (AUC={auc_teacher:.3f})", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline HLT-A (AUC={auc_baseline:.3f})", linewidth=2)
    plt.plot(tpr_m, fpr_m, "-.", label=f"Mixed HLT-A/B (AUC={auc_mixed:.3f})", linewidth=2)
    plt.plot(tpr_c, fpr_c, ":",  label=f"Consistency (AUC={auc_cons:.3f})", linewidth=2)

    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=11, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "results_twohlt.png", dpi=300)
    plt.close()

    # Log summary
    summary_file = Path(args.save_dir) / "run_summaries_twohlt.txt"
    with open(summary_file, "a") as f:
        f.write(f"\nRun: {args.run_name} (RESTARTED)\n")
        f.write(f"  hlt_seed_a={args.hlt_seed_a}, hlt_seed_b={args.hlt_seed_b}\n")
        f.write(f"  lam_cons={args.lam_cons}, cons_ramp_frac={args.cons_ramp_frac}, gate_thr={args.gate_thr}\n")
        f.write(f"  batch_size={args.batch_size}\n")
        f.write(f"  AUC teacher(off)={auc_teacher:.4f}\n")
        f.write(f"  AUC baseline(hlt_a)={auc_baseline:.4f}\n")
        f.write(f"  AUC mixed(hlt_a)={auc_mixed:.4f}\n")
        f.write(f"  AUC consistency(hlt_a)={auc_cons:.4f}\n")
        f.write(f"  BR@50 baseline={br_baseline:.2f}, mixed={br_mixed:.2f}, consistency={br_cons:.2f}\n")
        f.write(f"  Saved to: {save_dir}\n")
        f.write("=" * 70 + "\n")

    print(f"\n✓ Saved results to: {save_dir / 'results_twohlt.npz'} and {save_dir / 'results_twohlt.png'}")
    print(f"✓ Logged to: {summary_file}")
    print("\n✓ RESTART COMPLETE!")


if __name__ == "__main__":
    main_restart()

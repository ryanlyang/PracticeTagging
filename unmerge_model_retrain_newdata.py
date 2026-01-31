#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrain classifiers on a NEW jet subset using a pretrained unmerger + merge-count predictor.

Workflow:
  1) Load enough jets to cover base stats subset + new eval subset.
  2) Compute feature normalization stats on BASE subset (to feed predictors as trained).
  3) Compute unmerge target mean/std on BASE subset using predicted counts (to match unmerger scale).
  4) Apply merge+eff (no smearing), build unmerged dataset for NEW subset.
  5) Train fresh teacher, baseline, unmerge classifier, KD, merge-flag classifier, merge-flag KD
     on the NEW subset splits.
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import unmerge_model as um


def build_unmerged_dataset_with_flag(
    feat_hlt_std,
    mask_hlt,
    hlt_const,
    pred_counts,
    unmerge_model,
    tgt_mean,
    tgt_std,
    max_count,
    max_constits,
    device,
    batch_size,
):
    n_jets, max_part, _ = hlt_const.shape
    pred_map = {}
    samples = []
    for j in range(n_jets):
        for idx in range(max_part):
            if mask_hlt[j, idx] and pred_counts[j, idx] > 1:
                samples.append((j, idx, int(pred_counts[j, idx])))

    if len(samples) > 0:
        unmerge_model.eval()
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                chunk = samples[i:i + batch_size]
                jet_idx = [c[0] for c in chunk]
                tok_idx = [c[1] for c in chunk]
                counts = [c[2] for c in chunk]
                x = torch.tensor(feat_hlt_std[jet_idx], dtype=torch.float32, device=device)
                m = torch.tensor(mask_hlt[jet_idx], dtype=torch.bool, device=device)
                token_idx = torch.tensor(tok_idx, dtype=torch.long, device=device)
                count = torch.tensor(counts, dtype=torch.long, device=device)
                preds = unmerge_model(x, m, token_idx, count).cpu().numpy()
                for k in range(len(chunk)):
                    c = counts[k]
                    pred = preds[k, :c]
                    pred = pred * tgt_std + tgt_mean
                    pred[:, 0] = np.clip(pred[:, 0], 0.0, None)
                    pred[:, 1] = np.clip(pred[:, 1], -5.0, 5.0)
                    pred[:, 2] = np.arctan2(np.sin(pred[:, 2]), np.cos(pred[:, 2]))
                    pred[:, 3] = np.clip(pred[:, 3], 0.0, None)
                    pred_map[(chunk[k][0], chunk[k][1])] = pred

    new_const = np.zeros((n_jets, max_constits, 4), dtype=np.float32)
    new_mask = np.zeros((n_jets, max_constits), dtype=bool)
    new_flag = np.zeros((n_jets, max_constits), dtype=np.int32)

    for j in range(n_jets):
        parts = []
        flags = []
        for idx in range(max_part):
            if not mask_hlt[j, idx]:
                continue
            if pred_counts[j, idx] <= 1:
                parts.append(hlt_const[j, idx])
                flags.append(0)
            else:
                pred = pred_map.get((j, idx))
                if pred is not None:
                    parts.extend(list(pred))
                    flags.extend([1] * len(pred))
        if len(parts) == 0:
            continue
        n_keep = min(len(parts), max_constits)
        new_const[j, :n_keep] = parts[:n_keep]
        new_mask[j, :n_keep] = True
        new_flag[j, :n_keep] = flags[:n_keep]

    return new_const, new_mask, new_flag


def fit_classifier(model, train_loader, val_loader, device, epochs, patience):
    opt = torch.optim.AdamW(model.parameters(), lr=um.CONFIG["training"]["lr"], weight_decay=um.CONFIG["training"]["weight_decay"])
    sch = um.get_scheduler(opt, um.CONFIG["training"]["warmup_epochs"], um.CONFIG["training"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(epochs), desc="Classifier"):
        _, train_auc = um.train_classifier(model, train_loader, opt, device)
        val_auc, _, _ = um.eval_classifier(model, val_loader, device)
        sch.step()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")
        if no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def fit_kd(student, teacher, train_loader, val_loader, device, cfg):
    opt = torch.optim.AdamW(student.parameters(), lr=um.CONFIG["training"]["lr"], weight_decay=um.CONFIG["training"]["weight_decay"])
    sch = um.get_scheduler(opt, um.CONFIG["training"]["warmup_epochs"], um.CONFIG["training"]["epochs"])

    best_auc, best_state, no_improve = 0.0, None, 0
    kd_active = not cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None

    for ep in tqdm(range(um.CONFIG["training"]["epochs"]), desc="Unmerge+KD"):
        current_alpha = cfg["alpha_kd"] if kd_active else 0.0
        cfg_ep = dict(cfg)
        cfg_ep["alpha_kd"] = current_alpha
        train_loss, train_auc = um.train_kd_epoch(student, teacher, train_loader, opt, device, cfg_ep)
        val_auc, _, _ = um.evaluate_kd(student, val_loader, device)
        sch.step()

        if not kd_active and cfg["adaptive_alpha"]:
            val_loss = um.evaluate_bce_loss_unmerged(student, val_loader, device)
            if prev_val_loss is not None and abs(prev_val_loss - val_loss) < cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = val_loss
            if ep + 1 >= cfg["alpha_warmup_min_epochs"] and stable_count >= cfg["alpha_stable_patience"]:
                kd_active = True
                print(f"Activating KD ramp at epoch {ep+1} (val_loss={val_loss:.4f})")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f} | alpha_kd={current_alpha:.2f}")
        if no_improve >= um.CONFIG["training"]["patience"]:
            print(f"Early stopping KD student at epoch {ep+1}")
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    if cfg["self_train"]:
        print("\nSTEP: SELF-TRAIN")
        opt_st = torch.optim.AdamW(student.parameters(), lr=cfg["self_train_lr"])
        best_auc_st = best_auc
        no_improve = 0
        for ep in range(cfg["self_train_epochs"]):
            st_loss = um.self_train_student(student, teacher, train_loader, opt_st, device, cfg)
            val_auc, _, _ = um.evaluate_kd(student, val_loader, device)
            if val_auc > best_auc_st:
                best_auc_st = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if (ep + 1) % 2 == 0:
                print(f"Self ep {ep+1}: loss={st_loss:.4f}, val_auc={val_auc:.4f}, best={best_auc_st:.4f}")
            if no_improve >= cfg["self_train_patience"]:
                break
        if best_state is not None:
            student.load_state_dict(best_state)
    return student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unmerge_model" / "default"))
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unmerge_model_retrain"))
    parser.add_argument("--run_name", type=str, default="retrain_newdata")

    parser.add_argument("--n_eval_jets", type=int, default=200000)
    parser.add_argument("--offset_jets", type=int, default=200000)
    parser.add_argument("--stats_jets", type=int, default=200000)
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

    print("Applying HLT effects (base subset)...")
    hlt_base, hlt_mask_base, origin_counts_base, origin_lists_base, stats_base = um.apply_hlt_effects_with_tracking(
        base_const, base_mask_raw, um.CONFIG, seed=args.seed
    )
    print("Applying HLT effects (eval subset)...")
    hlt_eval, hlt_mask_eval, _, _, stats_eval = um.apply_hlt_effects_with_tracking(
        eval_const, eval_mask_raw, um.CONFIG, seed=args.seed
    )

    pt_threshold_off = um.CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off_base = base_mask_raw & (base_const[:, :, 0] >= pt_threshold_off)
    const_off_base = base_const.copy()
    const_off_base[~masks_off_base] = 0

    masks_off_eval = eval_mask_raw & (eval_const[:, :, 0] >= pt_threshold_off)
    const_off_eval = eval_const.copy()
    const_off_eval[~masks_off_eval] = 0

    print("Computing features...")
    features_off_base = um.compute_features(const_off_base, masks_off_base)
    features_hlt_base = um.compute_features(hlt_base, hlt_mask_base)
    features_off_eval = um.compute_features(const_off_eval, masks_off_eval)
    features_hlt_eval = um.compute_features(hlt_eval, hlt_mask_eval)

    # Base split for stats
    idx_base = np.arange(len(base_labels))
    train_idx_base, temp_idx_base = train_test_split(idx_base, test_size=0.30, random_state=um.RANDOM_SEED, stratify=base_labels)
    val_idx_base, test_idx_base = train_test_split(temp_idx_base, test_size=0.50, random_state=um.RANDOM_SEED, stratify=base_labels[temp_idx_base])

    feat_means_base, feat_stds_base = um.get_stats(features_off_base, masks_off_base, train_idx_base)
    features_hlt_base_std = um.standardize(features_hlt_base, hlt_mask_base, feat_means_base, feat_stds_base)
    features_hlt_eval_std = um.standardize(features_hlt_eval, hlt_mask_eval, feat_means_base, feat_stds_base)

    # Load merge-count predictor + unmerge predictor
    max_count = max(int(args.max_merge_count), 2)
    count_ckpt = torch.load(ckpt_dir / "merge_count.pt", map_location=device)
    count_model = um.MergeCountPredictor(input_dim=7, num_classes=max_count, **um.CONFIG["merge_count_model"]).to(device)
    count_model.load_state_dict(count_ckpt["model"])

    unmerge_ckpt = torch.load(ckpt_dir / "unmerge_predictor.pt", map_location=device)
    unmerge_model = um.UnmergePredictor(input_dim=7, max_count=max_count, **um.CONFIG["unmerge_model"]).to(device)
    unmerge_model.load_state_dict(unmerge_ckpt["model"])

    bs_cnt = args.batch_size or um.CONFIG["merge_count_training"]["batch_size"]
    pred_counts_base = um.predict_counts(count_model, features_hlt_base_std, hlt_mask_base, bs_cnt, device, max_count)
    pred_counts_eval = um.predict_counts(count_model, features_hlt_eval_std, hlt_mask_eval, bs_cnt, device, max_count)

    # Unmerge target stats from base subset
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
    train_idx_set = set(train_idx_base)
    train_samples = [s for s in samples if s[0] in train_idx_set]
    if len(train_samples) == 0:
        raise RuntimeError("No merged samples in base training split; cannot compute tgt_mean/std.")
    train_targets = [const_off_base[s[0], s[2], :4] for s in train_samples]
    flat_train = np.concatenate(train_targets, axis=0)
    tgt_mean = flat_train.mean(axis=0)
    tgt_std = flat_train.std(axis=0) + 1e-8

    # Build unmerged dataset for eval subset (with merge flag)
    bs_un = args.batch_size or um.CONFIG["unmerge_training"]["batch_size"]
    print("Building unmerged dataset for eval subset...")
    unmerged_const_eval, unmerged_mask_eval, unmerged_flag_eval = build_unmerged_dataset_with_flag(
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

    # Train/val/test split on eval subset
    idx_eval = np.arange(len(eval_labels))
    train_idx, temp_idx = train_test_split(idx_eval, test_size=0.30, random_state=um.RANDOM_SEED, stratify=eval_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=um.RANDOM_SEED, stratify=eval_labels[temp_idx])
    print(f"Eval split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # New stats for classifier training
    feat_means_new, feat_stds_new = um.get_stats(features_off_eval, masks_off_eval, train_idx)
    features_off_eval_std = um.standardize(features_off_eval, masks_off_eval, feat_means_new, feat_stds_new)
    features_hlt_eval_std_cls = um.standardize(features_hlt_eval, hlt_mask_eval, feat_means_new, feat_stds_new)
    features_unmerged_eval_std = um.standardize(features_unmerged_eval, unmerged_mask_eval, feat_means_new, feat_stds_new)

    # Merge-flag features (append as last channel)
    flag_channel = unmerged_flag_eval[..., None].astype(np.float32)
    features_unmerged_flag = np.concatenate([features_unmerged_eval_std, flag_channel], axis=2)

    # Dataloaders
    BS = args.batch_size or um.CONFIG["training"]["batch_size"]
    train_loader_off = torch.utils.data.DataLoader(um.JetDataset(features_off_eval_std[train_idx], masks_off_eval[train_idx], eval_labels[train_idx]),
                                                   batch_size=BS, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader_off = torch.utils.data.DataLoader(um.JetDataset(features_off_eval_std[val_idx], masks_off_eval[val_idx], eval_labels[val_idx]),
                                                 batch_size=BS, shuffle=False, num_workers=args.num_workers)
    test_loader_off = torch.utils.data.DataLoader(um.JetDataset(features_off_eval_std[test_idx], masks_off_eval[test_idx], eval_labels[test_idx]),
                                                  batch_size=BS, shuffle=False, num_workers=args.num_workers)

    train_loader_hlt = torch.utils.data.DataLoader(um.JetDataset(features_hlt_eval_std_cls[train_idx], hlt_mask_eval[train_idx], eval_labels[train_idx]),
                                                   batch_size=BS, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader_hlt = torch.utils.data.DataLoader(um.JetDataset(features_hlt_eval_std_cls[val_idx], hlt_mask_eval[val_idx], eval_labels[val_idx]),
                                                 batch_size=BS, shuffle=False, num_workers=args.num_workers)
    test_loader_hlt = torch.utils.data.DataLoader(um.JetDataset(features_hlt_eval_std_cls[test_idx], hlt_mask_eval[test_idx], eval_labels[test_idx]),
                                                  batch_size=BS, shuffle=False, num_workers=args.num_workers)

    train_loader_un = torch.utils.data.DataLoader(um.JetDataset(features_unmerged_eval_std[train_idx], unmerged_mask_eval[train_idx], eval_labels[train_idx]),
                                                  batch_size=BS, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader_un = torch.utils.data.DataLoader(um.JetDataset(features_unmerged_eval_std[val_idx], unmerged_mask_eval[val_idx], eval_labels[val_idx]),
                                                batch_size=BS, shuffle=False, num_workers=args.num_workers)
    test_loader_un = torch.utils.data.DataLoader(um.JetDataset(features_unmerged_eval_std[test_idx], unmerged_mask_eval[test_idx], eval_labels[test_idx]),
                                                 batch_size=BS, shuffle=False, num_workers=args.num_workers)

    train_loader_flag = torch.utils.data.DataLoader(um.JetDataset(features_unmerged_flag[train_idx], unmerged_mask_eval[train_idx], eval_labels[train_idx]),
                                                    batch_size=BS, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader_flag = torch.utils.data.DataLoader(um.JetDataset(features_unmerged_flag[val_idx], unmerged_mask_eval[val_idx], eval_labels[val_idx]),
                                                  batch_size=BS, shuffle=False, num_workers=args.num_workers)
    test_loader_flag = torch.utils.data.DataLoader(um.JetDataset(features_unmerged_flag[test_idx], unmerged_mask_eval[test_idx], eval_labels[test_idx]),
                                                   batch_size=BS, shuffle=False, num_workers=args.num_workers)

    # KD datasets
    kd_train_ds = um.UnmergeKDDataset(
        features_unmerged_eval_std[train_idx], unmerged_mask_eval[train_idx],
        features_off_eval_std[train_idx], masks_off_eval[train_idx], eval_labels[train_idx]
    )
    kd_val_ds = um.UnmergeKDDataset(
        features_unmerged_eval_std[val_idx], unmerged_mask_eval[val_idx],
        features_off_eval_std[val_idx], masks_off_eval[val_idx], eval_labels[val_idx]
    )
    kd_test_ds = um.UnmergeKDDataset(
        features_unmerged_eval_std[test_idx], unmerged_mask_eval[test_idx],
        features_off_eval_std[test_idx], masks_off_eval[test_idx], eval_labels[test_idx]
    )
    kd_train_loader = torch.utils.data.DataLoader(kd_train_ds, batch_size=BS, shuffle=True, drop_last=True, num_workers=args.num_workers)
    kd_val_loader = torch.utils.data.DataLoader(kd_val_ds, batch_size=BS, shuffle=False, num_workers=args.num_workers)
    kd_test_loader = torch.utils.data.DataLoader(kd_test_ds, batch_size=BS, shuffle=False, num_workers=args.num_workers)

    kd_train_ds_f = um.UnmergeKDDataset(
        features_unmerged_flag[train_idx], unmerged_mask_eval[train_idx],
        features_off_eval_std[train_idx], masks_off_eval[train_idx], eval_labels[train_idx]
    )
    kd_val_ds_f = um.UnmergeKDDataset(
        features_unmerged_flag[val_idx], unmerged_mask_eval[val_idx],
        features_off_eval_std[val_idx], masks_off_eval[val_idx], eval_labels[val_idx]
    )
    kd_test_ds_f = um.UnmergeKDDataset(
        features_unmerged_flag[test_idx], unmerged_mask_eval[test_idx],
        features_off_eval_std[test_idx], masks_off_eval[test_idx], eval_labels[test_idx]
    )
    kd_train_loader_f = torch.utils.data.DataLoader(kd_train_ds_f, batch_size=BS, shuffle=True, drop_last=True, num_workers=args.num_workers)
    kd_val_loader_f = torch.utils.data.DataLoader(kd_val_ds_f, batch_size=BS, shuffle=False, num_workers=args.num_workers)
    kd_test_loader_f = torch.utils.data.DataLoader(kd_test_ds_f, batch_size=BS, shuffle=False, num_workers=args.num_workers)

    # Train teacher
    print("\nSTEP 1: TEACHER (Offline)")
    teacher = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    teacher, _ = fit_classifier(teacher, train_loader_off, val_loader_off, device, um.CONFIG["training"]["epochs"], um.CONFIG["training"]["patience"])
    auc_teacher, preds_teacher, labs = um.eval_classifier(teacher, test_loader_off, device)

    # Train baseline
    print("\nSTEP 2: BASELINE (HLT)")
    baseline = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    baseline, _ = fit_classifier(baseline, train_loader_hlt, val_loader_hlt, device, um.CONFIG["training"]["epochs"], um.CONFIG["training"]["patience"])
    auc_baseline, preds_baseline, _ = um.eval_classifier(baseline, test_loader_hlt, device)

    # Unmerge classifier
    print("\nSTEP 3: UNMERGE CLASSIFIER")
    unmerge_cls = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    unmerge_cls, _ = fit_classifier(unmerge_cls, train_loader_un, val_loader_un, device, um.CONFIG["training"]["epochs"], um.CONFIG["training"]["patience"])
    auc_unmerge, preds_unmerge, _ = um.eval_classifier(unmerge_cls, test_loader_un, device)

    # Unmerge + KD
    print("\nSTEP 4: UNMERGE + KD")
    kd_cfg = um.CONFIG["kd"]
    kd_student = um.ParticleTransformer(input_dim=7, **um.CONFIG["model"]).to(device)
    kd_student = fit_kd(kd_student, teacher, kd_train_loader, kd_val_loader, device, kd_cfg)
    auc_unmerge_kd, preds_unmerge_kd, _ = um.evaluate_kd(kd_student, kd_test_loader, device)

    # Merge-flag classifier
    print("\nSTEP 5: UNMERGE + MERGE-FLAG CLASSIFIER")
    unmerge_flag_cls = um.ParticleTransformer(input_dim=8, **um.CONFIG["model"]).to(device)
    unmerge_flag_cls, _ = fit_classifier(unmerge_flag_cls, train_loader_flag, val_loader_flag, device, um.CONFIG["training"]["epochs"], um.CONFIG["training"]["patience"])
    auc_unmerge_flag, preds_unmerge_flag, _ = um.eval_classifier(unmerge_flag_cls, test_loader_flag, device)

    # Merge-flag + KD
    print("\nSTEP 6: UNMERGE + MERGE-FLAG + KD")
    kd_student_f = um.ParticleTransformer(input_dim=8, **um.CONFIG["model"]).to(device)
    kd_student_f = fit_kd(kd_student_f, teacher, kd_train_loader_f, kd_val_loader_f, device, kd_cfg)
    auc_unmerge_flag_kd, preds_unmerge_flag_kd, _ = um.evaluate_kd(kd_student_f, kd_test_loader_f, device)

    print("\nFINAL EVAL (new 200k jets, retrained classifiers)")
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"Unmerge Model    AUC: {auc_unmerge:.4f}")
    print(f"Unmerge + KD     AUC: {auc_unmerge_kd:.4f}")
    print(f"Unmerge + Flag   AUC: {auc_unmerge_flag:.4f}")
    print(f"Unmerge + Flag + KD AUC: {auc_unmerge_flag_kd:.4f}")

    np.savez(
        save_root / "results_newdata_retrain.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_unmerge=auc_unmerge,
        auc_unmerge_kd=auc_unmerge_kd,
        auc_unmerge_mergeflag=auc_unmerge_flag,
        auc_unmerge_mergeflag_kd=auc_unmerge_flag_kd,
        preds_teacher=preds_teacher,
        preds_baseline=preds_baseline,
        preds_unmerge=preds_unmerge,
        preds_unmerge_kd=preds_unmerge_kd,
        preds_unmerge_mergeflag=preds_unmerge_flag,
        preds_unmerge_mergeflag_kd=preds_unmerge_flag_kd,
        labels=labs,
        offset_jets=args.offset_jets,
        n_eval_jets=args.n_eval_jets,
        stats_jets=args.stats_jets,
    )

    fpr_t, tpr_t, _ = um.roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = um.roc_curve(labs, preds_baseline)
    fpr_u, tpr_u, _ = um.roc_curve(labs, preds_unmerge)
    fpr_k, tpr_k, _ = um.roc_curve(labs, preds_unmerge_kd)
    fpr_f, tpr_f, _ = um.roc_curve(labs, preds_unmerge_flag)
    fpr_fk, tpr_fk, _ = um.roc_curve(labs, preds_unmerge_flag_kd)

    def plot_roc(lines, out_name):
        plt.figure(figsize=(8, 6))
        for tpr, fpr, style, label, color in lines:
            plt.plot(tpr, fpr, style, label=label, color=color, linewidth=2)
        plt.ylabel("False Positive Rate", fontsize=12)
        plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
        plt.legend(fontsize=12, frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_root / out_name, dpi=300)
        plt.close()

    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_u, fpr_u, ":", f"Unmerge (AUC={auc_unmerge:.3f})", "forestgreen"),
            (tpr_k, fpr_k, "-.", f"Unmerge+KD (AUC={auc_unmerge_kd:.3f})", "darkorange"),
            (tpr_f, fpr_f, "-", f"Unmerge+Flag (AUC={auc_unmerge_flag:.3f})", "purple"),
            (tpr_fk, fpr_fk, "--", f"Unmerge+Flag+KD (AUC={auc_unmerge_flag_kd:.3f})", "brown"),
        ],
        "results_all.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
        ],
        "results_teacher_baseline.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_u, fpr_u, ":", f"Unmerge (AUC={auc_unmerge:.3f})", "forestgreen"),
        ],
        "results_teacher_unmerge.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_k, fpr_k, "-.", f"Unmerge+KD (AUC={auc_unmerge_kd:.3f})", "darkorange"),
        ],
        "results_teacher_unmerge_kd.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_f, fpr_f, "-", f"Unmerge+Flag (AUC={auc_unmerge_flag:.3f})", "purple"),
        ],
        "results_teacher_unmerge_mergeflag.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_fk, fpr_fk, "--", f"Unmerge+Flag+KD (AUC={auc_unmerge_flag_kd:.3f})", "brown"),
        ],
        "results_teacher_unmerge_mergeflag_kd.png",
    )

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()

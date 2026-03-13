#!/usr/bin/env python3
"""
Rerun classifier training/eval using pretrained k-fold merge-count + unmerge predictors.

Intended use:
- Point --kfold_model_dir at a directory containing fold_{0..K-1}/merge_count.pt and unmerge_predictor.pt
- Rebuild HLT for the current dataset slice
- Generate OOF unmerged jets for train split
- Generate val/test unmerged jets via random-per-jet selection across folds (optional)
- Train and evaluate:
  Teacher, Baseline, Baseline+KD, Unmerged, Unmerged+KD,
  DualView, DualView+MF, DualView+KD, DualView+MF+KD
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve

import utils

from unmerge_distr_model_unsmear import (
    CONFIG,
    RANDOM_SEED,
    ETA_IDX,
    PHI_IDX,
    PT_IDX,
    apply_hlt_effects_with_tracking,
    compute_features,
    get_stats,
    standardize,
    ParticleTransformer,
    MergeCountPredictor,
    UnmergePredictor,
    DualViewCrossAttnClassifier,
    DualViewKDDataset,
    UnmergeKDDataset,
    JetDataset,
    predict_counts,
    predict_counts_ensemble,
    build_unmerged_dataset_subset,
    build_unmerged_dataset_random_ensemble_subset,
    get_scheduler,
    train_classifier,
    eval_classifier,
    train_kd_epoch,
    train_kd_epoch_dual,
    evaluate_kd,
    evaluate_kd_dual,
    evaluate_bce_loss_dual,
    self_train_student,
    self_train_student_dual,
)


def _load_fold_models(kfold_model_dir: Path, fold_id: int, device: torch.device, max_count: int):
    fold_dir = kfold_model_dir / f"fold_{fold_id}"
    ckpt_c = torch.load(fold_dir / "merge_count.pt", map_location=device)
    count_model = MergeCountPredictor(input_dim=7, num_classes=max_count, **CONFIG["merge_count_model"]).to(device)
    count_model.load_state_dict(ckpt_c["model"])

    ckpt_u = torch.load(fold_dir / "unmerge_predictor.pt", map_location=device)
    state = ckpt_u["model"]

    # Architecture depends on global CONFIG["unmerge_training"]["distributional"].
    out_w = state.get("out.4.weight", None)
    if out_w is None:
        raise KeyError("Expected key 'out.4.weight' in unmerge_predictor checkpoint.")
    out_dim = int(out_w.shape[0])
    if out_dim not in (4, 8):
        raise ValueError(f"Unexpected unmerge out_dim={out_dim} from checkpoint.")
    dist = out_dim == 8

    relpos_mode = ckpt_u.get("config", {}).get("relpos_mode", None)
    if relpos_mode is None:
        # Infer from keys
        relpos_mode = "attn" if any(k.startswith("relpos_mlp.") for k in state.keys()) else "none"

    # Temporarily set global for correct module construction.
    prev_dist = bool(CONFIG["unmerge_training"]["distributional"])
    CONFIG["unmerge_training"]["distributional"] = dist
    try:
        unmerge_model = UnmergePredictor(
            input_dim=7,
            max_count=max_count,
            relpos_mode=relpos_mode,
            **CONFIG["unmerge_model"],
        ).to(device)
        unmerge_model.load_state_dict(state)
    finally:
        CONFIG["unmerge_training"]["distributional"] = prev_dist

    tgt_mean = ckpt_u["tgt_mean"]
    tgt_std = ckpt_u["tgt_std"]
    return count_model, unmerge_model, tgt_mean, tgt_std, dist, relpos_mode


@torch.no_grad()
def _evaluate_bce_loss_single_compat(model, loader, device):
    """
    Compatibility BCE evaluator for single-view KD loaders.
    Supports either:
      - JetDataset style: {"feat", "mask", "label"}
      - UnmergeKDDataset style: {"unmerged", "mask_unmerged", "label"}
    """
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        if "feat" in batch and "mask" in batch:
            x = batch["feat"].to(device)
            mask = batch["mask"].to(device)
        elif "unmerged" in batch and "mask_unmerged" in batch:
            x = batch["unmerged"].to(device)
            mask = batch["mask_unmerged"].to(device)
        else:
            raise KeyError(
                f"Unsupported batch keys for BCE eval: {sorted(list(batch.keys()))}"
            )
        y = batch["label"].to(device)
        logits = model(x, mask).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        total += float(loss.item()) * len(y)
        count += len(y)
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--n_train_jets", type=int, default=200000)
    ap.add_argument("--offset_jets", type=int, default=0, help="Skip this many jets before taking n_train_jets.")
    ap.add_argument("--max_constits", type=int, default=80)
    ap.add_argument("--max_merge_count", type=int, default=10)
    ap.add_argument("--kfold_model_dir", type=str, required=True)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--save_dir", type=str, default="checkpoints/rerun_kfold_unmerge")
    ap.add_argument("--run_name", type=str, default="rerun")
    ap.add_argument("--efficiency_loss", type=float, default=0.05, help="Override HLT efficiency loss to match the fold models.")
    ap.add_argument("--random_valtest", action="store_true", help="Randomly select unmerge outputs for val/test across folds (per-jet).")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    # Override HLT efficiency for this rerun to match the trained fold models.
    CONFIG["hlt_effects"]["efficiency_loss"] = float(args.efficiency_loss)

    # Load dataset slice
    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(train_path.glob("*.h5"))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if not train_files:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = args.offset_jets + args.n_train_jets
    data_full, labels_full, _, _, _ = utils.load_from_files(
        [str(p) for p in train_files],
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
        use_train_weights=False,
    )
    if data_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Not enough jets for offset {args.offset_jets} + n_train_jets {args.n_train_jets}. Got {data_full.shape[0]}.")
    data = data_full[args.offset_jets:args.offset_jets + args.n_train_jets]
    labels = labels_full[args.offset_jets:args.offset_jets + args.n_train_jets].astype(np.int64)

    eta = data[:, :, ETA_IDX].astype(np.float32)
    phi = data[:, :, PHI_IDX].astype(np.float32)
    pt = data[:, :, PT_IDX].astype(np.float32)
    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    const_off = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)
    masks_off = mask_raw

    # Build HLT (must match fold training as closely as possible)
    hlt_const, hlt_mask, origin_counts, origin_lists, _ = apply_hlt_effects_with_tracking(
        const_off, masks_off, CONFIG, seed=RANDOM_SEED
    )

    features_off = compute_features(const_off, masks_off)
    features_hlt = compute_features(hlt_const, hlt_mask)

    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=labels[temp_idx])

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = standardize(features_hlt, hlt_mask, feat_means, feat_stds)

    # Load fold models
    kfold_model_dir = Path(args.kfold_model_dir)
    max_count = args.max_merge_count
    count_models = []
    unmerge_models = []
    tgt_stats = []
    dist_flags = []
    relpos_modes = []
    for fid in range(args.k_folds):
        cm, um, tm, ts, dist, relpos = _load_fold_models(kfold_model_dir, fid, device, max_count)
        count_models.append(cm)
        unmerge_models.append(um)
        tgt_stats.append((tm, ts))
        dist_flags.append(dist)
        relpos_modes.append(relpos)
    if len(set(dist_flags)) != 1:
        raise RuntimeError(f"Folds disagree on distributional mode: {dist_flags}")
    if len(set(relpos_modes)) != 1:
        print(f"Warning: folds disagree on relpos modes: {relpos_modes}")

    # Predict counts OOF on train split, ensemble on val/test
    BS_cnt = CONFIG["merge_count_training"]["batch_size"]
    pred_counts = np.zeros_like(hlt_mask, dtype=np.int64)
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=RANDOM_SEED)
    train_idx_array = np.array(train_idx)
    holdouts = []
    for fold_id, (_, hold_rel) in enumerate(kf.split(train_idx_array)):
        hold_sub = train_idx_array[hold_rel]
        holdouts.append(hold_sub)
        pred_counts[hold_sub] = predict_counts(count_models[fold_id], features_hlt_std[hold_sub], hlt_mask[hold_sub], BS_cnt, device, max_count)
    # For val/test, use averaged ensemble counts (stable), regardless of random unmerge.
    pred_counts[val_idx] = predict_counts_ensemble(count_models, features_hlt_std[val_idx], hlt_mask[val_idx], BS_cnt, device, max_count)
    pred_counts[test_idx] = predict_counts_ensemble(count_models, features_hlt_std[test_idx], hlt_mask[test_idx], BS_cnt, device, max_count)

    # Build unmerged jets: OOF on train, random-per-jet or averaged on val/test
    n_jets = len(labels)
    unmerged_const = np.zeros((n_jets, args.max_constits, 4), dtype=np.float32)
    unmerged_mask = np.zeros((n_jets, args.max_constits), dtype=bool)
    unmerged_flag = np.zeros((n_jets, args.max_constits), dtype=np.float32)

    BS_un = CONFIG["unmerge_training"]["batch_size"]
    for fold_id, hold_sub in enumerate(holdouts):
        sub_const, sub_mask, sub_flag = build_unmerged_dataset_subset(
            hold_sub,
            features_hlt_std,
            hlt_mask,
            hlt_const,
            pred_counts,
            unmerge_models[fold_id],
            tgt_stats[fold_id][0],
            tgt_stats[fold_id][1],
            max_count,
            args.max_constits,
            device,
            BS_un,
        )
        unmerged_const[hold_sub] = sub_const
        unmerged_mask[hold_sub] = sub_mask
        unmerged_flag[hold_sub] = sub_flag

    rng = np.random.default_rng(RANDOM_SEED)
    pred_counts_list_val = [pred_counts[val_idx]] * len(unmerge_models)
    pred_counts_list_test = [pred_counts[test_idx]] * len(unmerge_models)

    if args.random_valtest:
        val_const, val_mask, val_flag = build_unmerged_dataset_random_ensemble_subset(
            val_idx,
            features_hlt_std,
            hlt_mask,
            hlt_const,
            pred_counts_list_val,
            unmerge_models,
            tgt_stats,
            max_count,
            args.max_constits,
            device,
            BS_un,
            rng,
            mode="jet",
        )
        test_const, test_mask, test_flag = build_unmerged_dataset_random_ensemble_subset(
            test_idx,
            features_hlt_std,
            hlt_mask,
            hlt_const,
            pred_counts_list_test,
            unmerge_models,
            tgt_stats,
            max_count,
            args.max_constits,
            device,
            BS_un,
            rng,
            mode="jet",
        )
        unmerged_const[val_idx] = val_const
        unmerged_mask[val_idx] = val_mask
        unmerged_flag[val_idx] = val_flag
        unmerged_const[test_idx] = test_const
        unmerged_mask[test_idx] = test_mask
        unmerged_flag[test_idx] = test_flag
    else:
        # Default to using the last fold model for val/test deterministically.
        for indices, name in [(val_idx, "val"), (test_idx, "test")]:
            sub_const, sub_mask, sub_flag = build_unmerged_dataset_subset(
                indices,
                features_hlt_std,
                hlt_mask,
                hlt_const,
                pred_counts,
                unmerge_models[-1],
                tgt_stats[-1][0],
                tgt_stats[-1][1],
                max_count,
                args.max_constits,
                device,
                BS_un,
            )
            unmerged_const[indices] = sub_const
            unmerged_mask[indices] = sub_mask
            unmerged_flag[indices] = sub_flag

    features_unmerged = compute_features(unmerged_const, unmerged_mask)
    features_unmerged_std = standardize(features_unmerged, unmerged_mask, feat_means, feat_stds)
    features_unmerged_flag = np.concatenate([features_unmerged_std, unmerged_flag[..., None]], axis=-1).astype(np.float32)

    # DataLoaders
    DL = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"), persistent_workers=args.num_workers > 0)
    BS = CONFIG["training"]["batch_size"]
    train_loader_off = torch.utils.data.DataLoader(JetDataset(features_off_std[train_idx], masks_off[train_idx], labels[train_idx]), batch_size=BS, shuffle=True, drop_last=True, **DL)
    val_loader_off = torch.utils.data.DataLoader(JetDataset(features_off_std[val_idx], masks_off[val_idx], labels[val_idx]), batch_size=BS, shuffle=False, **DL)
    test_loader_off = torch.utils.data.DataLoader(JetDataset(features_off_std[test_idx], masks_off[test_idx], labels[test_idx]), batch_size=BS, shuffle=False, **DL)

    train_loader_hlt = torch.utils.data.DataLoader(JetDataset(features_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx]), batch_size=BS, shuffle=True, drop_last=True, **DL)
    val_loader_hlt = torch.utils.data.DataLoader(JetDataset(features_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx]), batch_size=BS, shuffle=False, **DL)
    test_loader_hlt = torch.utils.data.DataLoader(JetDataset(features_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx]), batch_size=BS, shuffle=False, **DL)

    train_loader_um = torch.utils.data.DataLoader(JetDataset(features_unmerged_std[train_idx], unmerged_mask[train_idx], labels[train_idx]), batch_size=BS, shuffle=True, drop_last=True, **DL)
    val_loader_um = torch.utils.data.DataLoader(JetDataset(features_unmerged_std[val_idx], unmerged_mask[val_idx], labels[val_idx]), batch_size=BS, shuffle=False, **DL)
    test_loader_um = torch.utils.data.DataLoader(JetDataset(features_unmerged_std[test_idx], unmerged_mask[test_idx], labels[test_idx]), batch_size=BS, shuffle=False, **DL)

    # Teacher
    teacher = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_t = get_scheduler(opt_t, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_t, best_state_t, no_improve = 0.0, None, 0
    kd_cfg = CONFIG["kd"]
    ema = None
    if kd_cfg.get("ema_teacher", False):
        from unmerge_distr_model_unsmear import EMA  # local import

        ema = EMA(teacher, decay=kd_cfg["ema_decay"])
    for ep in range(CONFIG["training"]["epochs"]):
        _, train_auc = train_classifier(teacher, train_loader_off, opt_t, device, ema=ema)
        val_auc, _, _ = eval_classifier(teacher, val_loader_off, device)
        sch_t.step()
        if val_auc > best_auc_t:
            best_auc_t = val_auc
            best_state_t = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Teacher ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_t:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            break
    if best_state_t is not None:
        teacher.load_state_dict(best_state_t)
    if ema is not None:
        ema.apply_to(teacher)

    # Baseline
    baseline = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_b = get_scheduler(opt_b, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_b, best_state_b, no_improve = 0.0, None, 0
    for ep in range(CONFIG["training"]["epochs"]):
        _, train_auc = train_classifier(baseline, train_loader_hlt, opt_b, device)
        val_auc, _, _ = eval_classifier(baseline, val_loader_hlt, device)
        sch_b.step()
        if val_auc > best_auc_b:
            best_auc_b = val_auc
            best_state_b = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Baseline ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_b:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            break
    if best_state_b is not None:
        baseline.load_state_dict(best_state_b)

    auc_teacher, preds_teacher, labs = eval_classifier(teacher, test_loader_off, device)
    auc_baseline, preds_baseline, _ = eval_classifier(baseline, test_loader_hlt, device)

    # Baseline + KD (HLT student distilled from offline teacher)
    kd_train_hlt = UnmergeKDDataset(features_hlt_std[train_idx], hlt_mask[train_idx], features_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    kd_val_hlt = UnmergeKDDataset(features_hlt_std[val_idx], hlt_mask[val_idx], features_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    kd_test_hlt = UnmergeKDDataset(features_hlt_std[test_idx], hlt_mask[test_idx], features_off_std[test_idx], masks_off[test_idx], labels[test_idx])
    kd_train_loader_hlt = torch.utils.data.DataLoader(kd_train_hlt, batch_size=BS, shuffle=True, drop_last=True, **DL)
    kd_val_loader_hlt = torch.utils.data.DataLoader(kd_val_hlt, batch_size=BS, shuffle=False, **DL)
    kd_test_loader_hlt = torch.utils.data.DataLoader(kd_test_hlt, batch_size=BS, shuffle=False, **DL)

    hlt_kd = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_hlt_kd = torch.optim.AdamW(hlt_kd.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_hlt_kd = get_scheduler(opt_hlt_kd, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_kd, best_state_kd, no_improve = 0.0, None, 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None
    for ep in range(CONFIG["training"]["epochs"]):
        current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
        kd_cfg_ep = dict(kd_cfg)
        kd_cfg_ep["alpha_kd"] = current_alpha
        _, train_auc = train_kd_epoch(hlt_kd, teacher, kd_train_loader_hlt, opt_hlt_kd, device, kd_cfg_ep)
        val_auc, _, _ = evaluate_kd(hlt_kd, kd_val_loader_hlt, device)
        sch_hlt_kd.step()
        if not kd_active and kd_cfg["adaptive_alpha"]:
            # KD loaders use UnmergeKDDataset keys ("unmerged", "mask_unmerged", ...),
            # so we must use the matching loss helper.
            val_loss = _evaluate_bce_loss_single_compat(hlt_kd, kd_val_loader_hlt, device)
            if prev_val_loss is not None and abs(prev_val_loss - val_loss) < kd_cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = val_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
        if val_auc > best_auc_kd:
            best_auc_kd = val_auc
            best_state_kd = {k: v.detach().cpu().clone() for k, v in hlt_kd.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Baseline+KD ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_kd:.4f} | alpha_kd={current_alpha:.2f}")
        if no_improve >= CONFIG["training"]["patience"]:
            break
    if best_state_kd is not None:
        hlt_kd.load_state_dict(best_state_kd)
    if kd_cfg["self_train"]:
        opt_st = torch.optim.AdamW(hlt_kd.parameters(), lr=kd_cfg["self_train_lr"])
        best_auc_st = best_auc_kd
        no_improve = 0
        for ep in range(kd_cfg["self_train_epochs"]):
            _ = self_train_student(hlt_kd, teacher, kd_train_loader_hlt, opt_st, device, kd_cfg)
            val_auc, _, _ = evaluate_kd(hlt_kd, kd_val_loader_hlt, device)
            if val_auc > best_auc_st:
                best_auc_st = val_auc
                best_state_kd = {k: v.detach().cpu().clone() for k, v in hlt_kd.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= kd_cfg["self_train_patience"]:
                break
        if best_state_kd is not None:
            hlt_kd.load_state_dict(best_state_kd)
    auc_baseline_kd, preds_baseline_kd, _ = evaluate_kd(hlt_kd, kd_test_loader_hlt, device)

    # Unmerged
    unmerge_cls = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_u = torch.optim.AdamW(unmerge_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_u = get_scheduler(opt_u, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_u, best_state_u, no_improve = 0.0, None, 0
    for ep in range(CONFIG["training"]["epochs"]):
        _, train_auc = train_classifier(unmerge_cls, train_loader_um, opt_u, device)
        val_auc, _, _ = eval_classifier(unmerge_cls, val_loader_um, device)
        sch_u.step()
        if val_auc > best_auc_u:
            best_auc_u = val_auc
            best_state_u = {k: v.detach().cpu().clone() for k, v in unmerge_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Unmerged ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_u:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            break
    if best_state_u is not None:
        unmerge_cls.load_state_dict(best_state_u)
    auc_unmerged, preds_unmerged, _ = eval_classifier(unmerge_cls, test_loader_um, device)

    # Unmerged + KD
    kd_train_um = UnmergeKDDataset(features_unmerged_std[train_idx], unmerged_mask[train_idx], features_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    kd_val_um = UnmergeKDDataset(features_unmerged_std[val_idx], unmerged_mask[val_idx], features_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    kd_test_um = UnmergeKDDataset(features_unmerged_std[test_idx], unmerged_mask[test_idx], features_off_std[test_idx], masks_off[test_idx], labels[test_idx])
    kd_train_loader_um = torch.utils.data.DataLoader(kd_train_um, batch_size=BS, shuffle=True, drop_last=True, **DL)
    kd_val_loader_um = torch.utils.data.DataLoader(kd_val_um, batch_size=BS, shuffle=False, **DL)
    kd_test_loader_um = torch.utils.data.DataLoader(kd_test_um, batch_size=BS, shuffle=False, **DL)

    unmerged_kd = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_ukd = torch.optim.AdamW(unmerged_kd.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_ukd = get_scheduler(opt_ukd, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_ukd, best_state_ukd, no_improve = 0.0, None, 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None
    for ep in range(CONFIG["training"]["epochs"]):
        current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
        kd_cfg_ep = dict(kd_cfg)
        kd_cfg_ep["alpha_kd"] = current_alpha
        _, train_auc = train_kd_epoch(unmerged_kd, teacher, kd_train_loader_um, opt_ukd, device, kd_cfg_ep)
        val_auc, _, _ = evaluate_kd(unmerged_kd, kd_val_loader_um, device)
        sch_ukd.step()
        if not kd_active and kd_cfg["adaptive_alpha"]:
            val_loss = _evaluate_bce_loss_single_compat(unmerged_kd, kd_val_loader_um, device)
            if prev_val_loss is not None and abs(prev_val_loss - val_loss) < kd_cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = val_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
        if val_auc > best_auc_ukd:
            best_auc_ukd = val_auc
            best_state_ukd = {k: v.detach().cpu().clone() for k, v in unmerged_kd.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Unmerged+KD ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_ukd:.4f} | alpha_kd={current_alpha:.2f}")
        if no_improve >= CONFIG["training"]["patience"]:
            break
    if best_state_ukd is not None:
        unmerged_kd.load_state_dict(best_state_ukd)
    if kd_cfg["self_train"]:
        opt_st = torch.optim.AdamW(unmerged_kd.parameters(), lr=kd_cfg["self_train_lr"])
        best_auc_st = best_auc_ukd
        no_improve = 0
        for ep in range(kd_cfg["self_train_epochs"]):
            _ = self_train_student(unmerged_kd, teacher, kd_train_loader_um, opt_st, device, kd_cfg)
            val_auc, _, _ = evaluate_kd(unmerged_kd, kd_val_loader_um, device)
            if val_auc > best_auc_st:
                best_auc_st = val_auc
                best_state_ukd = {k: v.detach().cpu().clone() for k, v in unmerged_kd.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= kd_cfg["self_train_patience"]:
                break
        if best_state_ukd is not None:
            unmerged_kd.load_state_dict(best_state_ukd)
    auc_unmerged_kd, preds_unmerged_kd, _ = evaluate_kd(unmerged_kd, kd_test_loader_um, device)

    # DualView / DualView+MF
    train_loader_dual = torch.utils.data.DataLoader(
        DualViewKDDataset(
            features_hlt_std[train_idx],
            hlt_mask[train_idx],
            features_unmerged_std[train_idx],
            unmerged_mask[train_idx],
            features_off_std[train_idx],
            masks_off[train_idx],
            labels[train_idx],
        ),
        batch_size=BS,
        shuffle=True,
        drop_last=True,
        **DL,
    )
    val_loader_dual = torch.utils.data.DataLoader(
        DualViewKDDataset(
            features_hlt_std[val_idx],
            hlt_mask[val_idx],
            features_unmerged_std[val_idx],
            unmerged_mask[val_idx],
            features_off_std[val_idx],
            masks_off[val_idx],
            labels[val_idx],
        ),
        batch_size=BS,
        shuffle=False,
        **DL,
    )
    test_loader_dual = torch.utils.data.DataLoader(
        DualViewKDDataset(
            features_hlt_std[test_idx],
            hlt_mask[test_idx],
            features_unmerged_std[test_idx],
            unmerged_mask[test_idx],
            features_off_std[test_idx],
            masks_off[test_idx],
            labels[test_idx],
        ),
        batch_size=BS,
        shuffle=False,
        **DL,
    )

    dual_cls = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **CONFIG["model"]).to(device)
    opt_dv = torch.optim.AdamW(dual_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_dv = get_scheduler(opt_dv, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_dv, best_state_dv, no_improve = 0.0, None, 0
    kd_cfg_zero = dict(kd_cfg)
    kd_cfg_zero["alpha_kd"] = 0.0
    for ep in range(CONFIG["training"]["epochs"]):
        _, train_auc = train_kd_epoch_dual(dual_cls, teacher, train_loader_dual, opt_dv, device, kd_cfg_zero)
        val_auc, _, _ = evaluate_kd_dual(dual_cls, val_loader_dual, device)
        sch_dv.step()
        if val_auc > best_auc_dv:
            best_auc_dv = val_auc
            best_state_dv = {k: v.detach().cpu().clone() for k, v in dual_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"DualView ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_dv:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            break
    if best_state_dv is not None:
        dual_cls.load_state_dict(best_state_dv)
    auc_dual, preds_dual, _ = evaluate_kd_dual(dual_cls, test_loader_dual, device)

    train_loader_dual_flag = torch.utils.data.DataLoader(
        DualViewKDDataset(
            features_hlt_std[train_idx],
            hlt_mask[train_idx],
            features_unmerged_flag[train_idx],
            unmerged_mask[train_idx],
            features_off_std[train_idx],
            masks_off[train_idx],
            labels[train_idx],
        ),
        batch_size=BS,
        shuffle=True,
        drop_last=True,
        **DL,
    )
    val_loader_dual_flag = torch.utils.data.DataLoader(
        DualViewKDDataset(
            features_hlt_std[val_idx],
            hlt_mask[val_idx],
            features_unmerged_flag[val_idx],
            unmerged_mask[val_idx],
            features_off_std[val_idx],
            masks_off[val_idx],
            labels[val_idx],
        ),
        batch_size=BS,
        shuffle=False,
        **DL,
    )
    test_loader_dual_flag = torch.utils.data.DataLoader(
        DualViewKDDataset(
            features_hlt_std[test_idx],
            hlt_mask[test_idx],
            features_unmerged_flag[test_idx],
            unmerged_mask[test_idx],
            features_off_std[test_idx],
            masks_off[test_idx],
            labels[test_idx],
        ),
        batch_size=BS,
        shuffle=False,
        **DL,
    )

    dual_flag_cls = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=8, **CONFIG["model"]).to(device)
    opt_dvf = torch.optim.AdamW(dual_flag_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_dvf = get_scheduler(opt_dvf, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_dvf, best_state_dvf, no_improve = 0.0, None, 0
    for ep in range(CONFIG["training"]["epochs"]):
        _, train_auc = train_kd_epoch_dual(dual_flag_cls, teacher, train_loader_dual_flag, opt_dvf, device, kd_cfg_zero)
        val_auc, _, _ = evaluate_kd_dual(dual_flag_cls, val_loader_dual_flag, device)
        sch_dvf.step()
        if val_auc > best_auc_dvf:
            best_auc_dvf = val_auc
            best_state_dvf = {k: v.detach().cpu().clone() for k, v in dual_flag_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"DualView+MF ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_dvf:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            break
    if best_state_dvf is not None:
        dual_flag_cls.load_state_dict(best_state_dvf)
    auc_dual_flag, preds_dual_flag, _ = evaluate_kd_dual(dual_flag_cls, test_loader_dual_flag, device)

    # DualView+KD and DualView+MF+KD
    kd_student = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **CONFIG["model"]).to(device)
    opt_kd = torch.optim.AdamW(kd_student.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_kd = get_scheduler(opt_kd, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None
    for ep in range(CONFIG["training"]["epochs"]):
        current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
        kd_cfg_ep = dict(kd_cfg)
        kd_cfg_ep["alpha_kd"] = current_alpha
        _, train_auc = train_kd_epoch_dual(kd_student, teacher, train_loader_dual, opt_kd, device, kd_cfg_ep)
        val_auc, _, _ = evaluate_kd_dual(kd_student, val_loader_dual, device)
        sch_kd.step()
        if not kd_active and kd_cfg["adaptive_alpha"]:
            val_loss = evaluate_bce_loss_dual(kd_student, val_loader_dual, device)
            if prev_val_loss is not None and abs(prev_val_loss - val_loss) < kd_cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = val_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in kd_student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"DualView+KD ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f} | alpha_kd={current_alpha:.2f}")
        if no_improve >= CONFIG["training"]["patience"]:
            break
    if best_state is not None:
        kd_student.load_state_dict(best_state)
    if kd_cfg["self_train"]:
        opt_st = torch.optim.AdamW(kd_student.parameters(), lr=kd_cfg["self_train_lr"])
        best_auc_st = best_auc
        no_improve = 0
        for ep in range(kd_cfg["self_train_epochs"]):
            _ = self_train_student_dual(kd_student, teacher, train_loader_dual, opt_st, device, kd_cfg)
            val_auc, _, _ = evaluate_kd_dual(kd_student, val_loader_dual, device)
            if val_auc > best_auc_st:
                best_auc_st = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in kd_student.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= kd_cfg["self_train_patience"]:
                break
        if best_state is not None:
            kd_student.load_state_dict(best_state)
    auc_dual_kd, preds_dual_kd, _ = evaluate_kd_dual(kd_student, test_loader_dual, device)

    kd_student_flag = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=8, **CONFIG["model"]).to(device)
    opt_kd_flag = torch.optim.AdamW(kd_student_flag.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_kd_flag = get_scheduler(opt_kd_flag, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None
    for ep in range(CONFIG["training"]["epochs"]):
        current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
        kd_cfg_ep = dict(kd_cfg)
        kd_cfg_ep["alpha_kd"] = current_alpha
        _, train_auc = train_kd_epoch_dual(kd_student_flag, teacher, train_loader_dual_flag, opt_kd_flag, device, kd_cfg_ep)
        val_auc, _, _ = evaluate_kd_dual(kd_student_flag, val_loader_dual_flag, device)
        sch_kd_flag.step()
        if not kd_active and kd_cfg["adaptive_alpha"]:
            val_loss = evaluate_bce_loss_dual(kd_student_flag, val_loader_dual_flag, device)
            if prev_val_loss is not None and abs(prev_val_loss - val_loss) < kd_cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = val_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in kd_student_flag.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"DualView+MF+KD ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f} | alpha_kd={current_alpha:.2f}")
        if no_improve >= CONFIG["training"]["patience"]:
            break
    if best_state is not None:
        kd_student_flag.load_state_dict(best_state)
    if kd_cfg["self_train"]:
        opt_st = torch.optim.AdamW(kd_student_flag.parameters(), lr=kd_cfg["self_train_lr"])
        best_auc_st = best_auc
        no_improve = 0
        for ep in range(kd_cfg["self_train_epochs"]):
            _ = self_train_student_dual(kd_student_flag, teacher, train_loader_dual_flag, opt_st, device, kd_cfg)
            val_auc, _, _ = evaluate_kd_dual(kd_student_flag, val_loader_dual_flag, device)
            if val_auc > best_auc_st:
                best_auc_st = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in kd_student_flag.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= kd_cfg["self_train_patience"]:
                break
        if best_state is not None:
            kd_student_flag.load_state_dict(best_state)
    auc_dual_flag_kd, preds_dual_flag_kd, _ = evaluate_kd_dual(kd_student_flag, test_loader_dual_flag, device)

    print("\nFINAL TEST EVALUATION")
    print(f"Teacher         AUC: {auc_teacher:.4f}")
    print(f"Baseline        AUC: {auc_baseline:.4f}")
    print(f"Baseline+KD     AUC: {auc_baseline_kd:.4f}")
    print(f"Unmerged        AUC: {auc_unmerged:.4f}")
    print(f"Unmerged+KD     AUC: {auc_unmerged_kd:.4f}")
    print(f"DualView        AUC: {auc_dual:.4f}")
    print(f"DualView+MF     AUC: {auc_dual_flag:.4f}")
    print(f"DualView+KD     AUC: {auc_dual_kd:.4f}")
    print(f"DualView+MF+KD  AUC: {auc_dual_flag_kd:.4f}")

    # Plots
    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_bk, tpr_bk, _ = roc_curve(labs, preds_baseline_kd)
    fpr_u, tpr_u, _ = roc_curve(labs, preds_unmerged)
    fpr_uk, tpr_uk, _ = roc_curve(labs, preds_unmerged_kd)
    fpr_dv, tpr_dv, _ = roc_curve(labs, preds_dual)
    fpr_dvf, tpr_dvf, _ = roc_curve(labs, preds_dual_flag)
    fpr_dvk, tpr_dvk, _ = roc_curve(labs, preds_dual_kd)
    fpr_dvfk, tpr_dvfk, _ = roc_curve(labs, preds_dual_flag_kd)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={auc_teacher:.3f})", color="crimson", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline (AUC={auc_baseline:.3f})", color="steelblue", linewidth=2)
    plt.plot(tpr_bk, fpr_bk, "--", label=f"Baseline+KD (AUC={auc_baseline_kd:.3f})", color="royalblue", linewidth=2)
    plt.plot(tpr_u, fpr_u, ":", label=f"Unmerged (AUC={auc_unmerged:.3f})", color="forestgreen", linewidth=2)
    plt.plot(tpr_uk, fpr_uk, ":", label=f"Unmerged+KD (AUC={auc_unmerged_kd:.3f})", color="darkgreen", linewidth=2)
    plt.plot(tpr_dv, fpr_dv, "-", label=f"DualView (AUC={auc_dual:.3f})", color="teal", linewidth=2)
    plt.plot(tpr_dvf, fpr_dvf, "-", label=f"DualView+MF (AUC={auc_dual_flag:.3f})", color="orchid", linewidth=2)
    plt.plot(tpr_dvk, fpr_dvk, "-.", label=f"DualView+KD (AUC={auc_dual_kd:.3f})", color="slateblue", linewidth=2)
    plt.plot(tpr_dvfk, fpr_dvfk, "-.", label=f"DualView+MF+KD (AUC={auc_dual_flag_kd:.3f})", color="darkslateblue", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_root / "results_all.png", dpi=300)
    plt.close()

    np.savez(
        save_root / "results.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_baseline_kd=auc_baseline_kd,
        auc_unmerged=auc_unmerged,
        auc_unmerged_kd=auc_unmerged_kd,
        auc_dual=auc_dual,
        auc_dual_flag=auc_dual_flag,
        auc_dual_kd=auc_dual_kd,
        auc_dual_flag_kd=auc_dual_flag_kd,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_baseline_kd=fpr_bk,
        tpr_baseline_kd=tpr_bk,
        fpr_unmerged=fpr_u,
        tpr_unmerged=tpr_u,
        fpr_unmerged_kd=fpr_uk,
        tpr_unmerged_kd=tpr_uk,
        fpr_dual=fpr_dv,
        tpr_dual=tpr_dv,
        fpr_dual_flag=fpr_dvf,
        tpr_dual_flag=tpr_dvf,
        fpr_dual_kd=fpr_dvk,
        tpr_dual_kd=tpr_dvk,
        fpr_dual_flag_kd=fpr_dvfk,
        tpr_dual_flag_kd=tpr_dvfk,
    )

    torch.save({"model": teacher.state_dict(), "auc": auc_teacher}, save_root / "teacher.pt")
    torch.save({"model": baseline.state_dict(), "auc": auc_baseline}, save_root / "baseline.pt")
    torch.save({"model": hlt_kd.state_dict(), "auc": auc_baseline_kd}, save_root / "baseline_kd.pt")
    torch.save({"model": unmerge_cls.state_dict(), "auc": auc_unmerged}, save_root / "unmerged_classifier.pt")
    torch.save({"model": unmerged_kd.state_dict(), "auc": auc_unmerged_kd}, save_root / "unmerged_kd.pt")
    torch.save({"model": dual_cls.state_dict(), "auc": auc_dual}, save_root / "dual_view_classifier.pt")
    torch.save({"model": dual_flag_cls.state_dict(), "auc": auc_dual_flag}, save_root / "dual_view_mergeflag_classifier.pt")
    torch.save({"model": kd_student.state_dict(), "auc": auc_dual_kd}, save_root / "dual_view_kd.pt")
    torch.save({"model": kd_student_flag.state_dict(), "auc": auc_dual_flag_kd}, save_root / "dual_view_mergeflag_kd.pt")

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()

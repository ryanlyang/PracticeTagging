#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict how many offline constituents each HLT token represents.

Training uses paired offline/HLT jets to label merge counts, but the model only
consumes HLT features at inference.
"""

from pathlib import Path
import argparse
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils


# ----------------------------- Reproducibility ----------------------------- #
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------------------- Column order (EDIT if needed) ----------------------------- #
ETA_IDX = 0
PHI_IDX = 1
PT_IDX = 2


CONFIG = {
    "hlt_effects": {
        "pt_resolution": 0.0,  # 0.10 before
        "eta_resolution": 0.0,  # 0.03 before
        "phi_resolution": 0.0,  # 0.03 before
        "pt_threshold_offline": 0.5,
        "pt_threshold_hlt": 1.5,
        "merge_enabled": True,
        "merge_radius": 0.01,
        "efficiency_loss": 0.03,  # 0.03 before
        "noise_enabled": False,
        "noise_fraction": 0.0,
    },
    "model": {
        "embed_dim": 128,
        "num_heads": 8,
        "num_layers": 6,
        "ff_dim": 512,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 512,
        "epochs": 80,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 3,
        "patience": 15,
    },
}


def apply_hlt_effects_with_labels(const, mask, cfg, seed=42):
    """
    const: (N, M, 4) [pt, eta, phi, E]
    mask:  (N, M) bool
    Returns:
      hlt, hlt_mask, merged_label, stats
    """
    np.random.seed(seed)
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()
    origin_counts = hlt_mask.astype(np.int32)

    n_initial = int(hlt_mask.sum())

    # Effect 1: Higher pT threshold
    pt_threshold = hcfg["pt_threshold_hlt"]
    below_threshold = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0
    origin_counts[~hlt_mask] = 0
    n_lost_threshold = int(below_threshold.sum())

    # Effect 2: Cluster merging
    n_merged = 0
    if hcfg["merge_enabled"] and hcfg["merge_radius"] > 0:
        merge_radius = hcfg["merge_radius"]
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            if len(valid_idx) < 2:
                continue

            to_remove = set()
            for i in range(len(valid_idx)):
                idx_i = valid_idx[i]
                if idx_i in to_remove:
                    continue

                for j in range(i + 1, len(valid_idx)):
                    idx_j = valid_idx[j]
                    if idx_j in to_remove:
                        continue

                    deta = hlt[jet_idx, idx_i, 1] - hlt[jet_idx, idx_j, 1]
                    dphi = hlt[jet_idx, idx_i, 2] - hlt[jet_idx, idx_j, 2]
                    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
                    dR = np.sqrt(deta**2 + dphi**2)

                    if dR < merge_radius:
                        pt_i = hlt[jet_idx, idx_i, 0]
                        pt_j = hlt[jet_idx, idx_j, 0]
                        pt_sum = pt_i + pt_j
                        if pt_sum < 1e-6:
                            continue

                        w_i = pt_i / pt_sum
                        w_j = pt_j / pt_sum

                        hlt[jet_idx, idx_i, 0] = pt_sum
                        hlt[jet_idx, idx_i, 1] = w_i * hlt[jet_idx, idx_i, 1] + w_j * hlt[jet_idx, idx_j, 1]

                        phi_i = hlt[jet_idx, idx_i, 2]
                        phi_j = hlt[jet_idx, idx_j, 2]
                        hlt[jet_idx, idx_i, 2] = np.arctan2(
                            w_i * np.sin(phi_i) + w_j * np.sin(phi_j),
                            w_i * np.cos(phi_i) + w_j * np.cos(phi_j),
                        )

                        hlt[jet_idx, idx_i, 3] = hlt[jet_idx, idx_i, 3] + hlt[jet_idx, idx_j, 3]
                        origin_counts[jet_idx, idx_i] += origin_counts[jet_idx, idx_j]

                        to_remove.add(idx_j)
                        n_merged += 1

            for idx in to_remove:
                hlt_mask[jet_idx, idx] = False
                hlt[jet_idx, idx] = 0
                origin_counts[jet_idx, idx] = 0

    # Effect 3: Resolution smearing
    valid = hlt_mask
    pt_noise = np.random.normal(1.0, hcfg["pt_resolution"], (n_jets, max_part))
    pt_noise = np.clip(pt_noise, 0.5, 1.5)
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)

    eta_noise = np.random.normal(0, hcfg["eta_resolution"], (n_jets, max_part))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)

    phi_noise = np.random.normal(0, hcfg["phi_resolution"], (n_jets, max_part))
    new_phi = hlt[:, :, 2] + phi_noise
    hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)

    # Recalculate E (massless approx)
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)

    # Effect 4: Random efficiency loss
    n_lost_eff = 0
    if hcfg["efficiency_loss"] > 0:
        random_loss = np.random.random((n_jets, max_part)) < hcfg["efficiency_loss"]
        lost = random_loss & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0
        origin_counts[lost] = 0
        n_lost_eff = int(lost.sum())

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0

    merged_label = (origin_counts > 1) & hlt_mask

    n_final = int(hlt_mask.sum())
    stats = {
        "n_initial": n_initial,
        "n_lost_threshold": n_lost_threshold,
        "n_merged": n_merged,
        "n_lost_eff": n_lost_eff,
        "n_final": n_final,
    }
    return hlt, hlt_mask, merged_label, origin_counts, stats


def compute_features(const, mask):
    pt = np.maximum(const[:, :, 0], 1e-8)
    eta = np.clip(const[:, :, 1], -5, 5)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], 1e-8)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    mask_float = mask.astype(float)
    jet_px = (px * mask_float).sum(axis=1, keepdims=True)
    jet_py = (py * mask_float).sum(axis=1, keepdims=True)
    jet_pz = (pz * mask_float).sum(axis=1, keepdims=True)
    jet_E = (E * mask_float).sum(axis=1, keepdims=True)

    jet_pt = np.sqrt(jet_px**2 + jet_py**2) + 1e-8
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)

    delta_eta = eta - jet_eta
    delta_phi = np.arctan2(np.sin(phi - jet_phi), np.cos(phi - jet_phi))

    log_pt = np.log(pt + 1e-8)
    log_E = np.log(E + 1e-8)

    log_pt_rel = np.log(pt / jet_pt + 1e-8)
    log_E_rel = np.log(E / (jet_E + 1e-8) + 1e-8)

    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

    features = np.stack([delta_eta, delta_phi, log_pt, log_E, log_pt_rel, log_E_rel, delta_R], axis=-1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -20, 20)
    features[~mask] = 0
    return features.astype(np.float32)


def get_stats(feat, mask, idx):
    means, stds = np.zeros(7), np.zeros(7)
    for i in range(7):
        vals = feat[idx][:, :, i][mask[idx]]
        means[i] = np.nanmean(vals)
        stds[i] = np.nanstd(vals) + 1e-8
    return means, stds


def standardize(feat, mask, means, stds):
    std = np.clip((feat - means) / stds, -10, 10)
    std = np.nan_to_num(std, 0.0)
    std[~mask] = 0
    return std.astype(np.float32)


class MergeDataset(Dataset):
    def __init__(self, feat_hlt, mask_hlt, count_label):
        self.hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask = torch.tensor(mask_hlt, dtype=torch.bool)
        self.label = torch.tensor(count_label, dtype=torch.long)

    def __len__(self):
        return len(self.hlt)

    def __getitem__(self, i):
        return {
            "hlt": self.hlt[i],
            "mask": self.mask[i],
            "label": self.label[i],
        }


class MergePredictor(nn.Module):
    def __init__(
        self,
        input_dim=7,
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        ff_dim=512,
        dropout=0.1,
        num_classes=6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, max(embed_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(embed_dim // 2, 32), num_classes),
        )

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)
        h = self.transformer(h, src_key_padding_mask=~mask)
        logits = self.head(h)
        return logits


def compute_class_weights(labels, mask, num_classes):
    valid = labels[mask]
    counts = np.bincount(valid, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.ones(num_classes, dtype=np.float64)
    if total > 0:
        weights = total / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
    return weights


def train_epoch(model, loader, opt, device, class_weights):
    model.train()
    total_loss = 0.0
    preds, labs = [], []
    weight = torch.tensor(class_weights, device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch in loader:
        x = batch["hlt"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)

        opt.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits[mask], y[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * mask.sum().item()
        pred_cls = logits[mask].argmax(dim=1)
        preds.extend(pred_cls.detach().cpu().numpy().flatten())
        labs.extend(y[mask].detach().cpu().numpy().flatten())

    acc = (np.array(preds) == np.array(labs)).mean() if len(labs) > 0 else 0.0
    return total_loss / max(len(labs), 1), acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    warned = False
    for batch in loader:
        x = batch["hlt"].to(device)
        mask = batch["mask"].to(device)
        logits = model(x, mask)
        if not warned and not torch.isfinite(logits).all():
            print("Warning: NaN/Inf in logits during evaluation; replacing with 0.0.")
            warned = True
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        pred_cls = logits[mask].argmax(dim=1)
        preds.extend(pred_cls.cpu().numpy().flatten())
        labs.extend(batch["label"][mask.cpu()].numpy().flatten())
    preds = np.array(preds)
    labs = np.array(labs)
    acc = (preds == labs).mean() if labs.size > 0 else 0.0
    return acc, preds, labs


def predict_classes(model, loader, device):
    model.eval()
    preds = []
    warned = False
    with torch.no_grad():
        for batch in loader:
            x = batch["feat"].to(device)
            m = batch["mask"].to(device)
            logits = model(x, m)
            if not warned and not torch.isfinite(logits).all():
                print("Warning: NaN/Inf in logits during evaluation; replacing with 0.0.")
                warned = True
            pred = logits.argmax(dim=2).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def clipped_mae(preds, labs, max_count):
    if preds.size == 0:
        return 0.0
    preds_c = np.clip(preds + 1, 1, max_count)
    labs_c = np.clip(labs + 1, 1, max_count)
    return float(np.abs(preds_c - labs_c).mean())


def confusion_matrix(preds, labs, num_classes):
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labs, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            mat[t, p] += 1
    return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "predict_lost_constituents"))
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=CONFIG["training"]["epochs"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["training"]["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["training"]["lr"])
    parser.add_argument("--weight_decay", type=float, default=CONFIG["training"]["weight_decay"])
    parser.add_argument("--warmup_epochs", type=int, default=CONFIG["training"]["warmup_epochs"])
    parser.add_argument("--patience", type=int, default=CONFIG["training"]["patience"])
    parser.add_argument("--skip_save_models", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=CONFIG["model"]["embed_dim"])
    parser.add_argument("--num_heads", type=int, default=CONFIG["model"]["num_heads"])
    parser.add_argument("--num_layers", type=int, default=CONFIG["model"]["num_layers"])
    parser.add_argument("--ff_dim", type=int, default=CONFIG["model"]["ff_dim"])
    parser.add_argument("--dropout", type=float, default=CONFIG["model"]["dropout"])
    parser.add_argument("--pt_resolution", type=float, default=None)
    parser.add_argument("--eta_resolution", type=float, default=None)
    parser.add_argument("--phi_resolution", type=float, default=None)
    parser.add_argument("--max_merge_count", type=int, default=10)
    parser.add_argument("--k_folds", type=int, default=1, help="K-fold OOF training (K>1).")
    parser.add_argument("--kfold_ensemble_valtest", action="store_true", help="Ensemble K models for val/test.")
    args = parser.parse_args()

    if args.pt_resolution is not None:
        CONFIG["hlt_effects"]["pt_resolution"] = float(args.pt_resolution)
    if args.eta_resolution is not None:
        CONFIG["hlt_effects"]["eta_resolution"] = float(args.eta_resolution)
    if args.phi_resolution is not None:
        CONFIG["hlt_effects"]["phi_resolution"] = float(args.phi_resolution)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    train_path = Path(args.train_path)
    train_files = sorted(list(train_path.glob("*.h5")))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {train_path}")

    print("Loading data via utils.load_from_files...")
    all_data, all_labels, _, _, _ = utils.load_from_files(
        train_files,
        max_jets=args.n_train_jets,
        max_constits=args.max_constits,
        use_train_weights=False,
    )
    all_labels = all_labels.astype(np.int64)
    print(f"Loaded: data={all_data.shape}, labels={all_labels.shape}")

    eta = all_data[:, :, ETA_IDX].astype(np.float32)
    phi = all_data[:, :, PHI_IDX].astype(np.float32)
    pt = all_data[:, :, PT_IDX].astype(np.float32)

    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    constituents_raw = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    print("Applying HLT effects...")
    constituents_hlt, masks_hlt, merged_label, origin_counts, stats = apply_hlt_effects_with_labels(
        constituents_raw, mask_raw, CONFIG, seed=RANDOM_SEED
    )

    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0

    print("HLT Simulation Statistics:")
    print(f"  Offline particles: {stats['n_initial']:,}")
    print(f"  Lost to pT threshold ({CONFIG['hlt_effects']['pt_threshold_hlt']}): {stats['n_lost_threshold']:,}")
    print(f"  Lost to merging (dR<{CONFIG['hlt_effects']['merge_radius']}): {stats['n_merged']:,}")
    print(f"  Lost to efficiency: {stats['n_lost_eff']:,}")
    print(f"  HLT particles: {stats['n_final']:,}")
    print(f"  Avg per jet: Offline={masks_off.sum(axis=1).mean():.1f}, HLT={masks_hlt.sum(axis=1).mean():.1f}")

    print("Computing features...")
    features_off = compute_features(constituents_off, masks_off)
    features_hlt = compute_features(constituents_hlt, masks_hlt)

    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_hlt_std = standardize(features_hlt, masks_hlt, feat_means, feat_stds)

    max_count = max(int(args.max_merge_count), 2)
    count_label = np.clip(origin_counts, 1, max_count) - 1

    train_ds = MergeDataset(features_hlt_std[train_idx], masks_hlt[train_idx], count_label[train_idx])
    val_ds = MergeDataset(features_hlt_std[val_idx], masks_hlt[val_idx], count_label[val_idx])
    test_ds = MergeDataset(features_hlt_std[test_idx], masks_hlt[test_idx], count_label[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    def format_dist(label, mask, name):
        valid = label[mask]
        counts = np.bincount(valid, minlength=max_count).astype(np.int64)
        total = max(counts.sum(), 1)
        frac = counts / total * 100.0
        print(f"{name} merge-count distribution (class=1..{max_count}, {max_count}=cap):")
        for i in range(max_count):
            print(f"  {i+1}: {counts[i]:,} ({frac[i]:.2f}%)")

    format_dist(count_label, masks_hlt, "All HLT")
    format_dist(count_label[train_idx], masks_hlt[train_idx], "Train")

    class_weights = compute_class_weights(count_label[train_idx], masks_hlt[train_idx], max_count)
    print(f"Class weights (1..{max_count}): {np.round(class_weights, 3)}")

    k_folds = max(1, int(args.k_folds))
    models = []

    if k_folds > 1:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
        train_idx_array = np.array(train_idx)
        for fold_id, (train_sub_rel, hold_rel) in enumerate(kf.split(train_idx_array)):
            train_sub = train_idx_array[train_sub_rel]
            hold_sub = train_idx_array[hold_rel]
            print(f"\n--- Fold {fold_id+1}/{k_folds} | train={len(train_sub)} holdout={len(hold_sub)} ---")
            train_ds_f = MergeDataset(features_hlt_std[train_sub], masks_hlt[train_sub], count_label[train_sub])
            val_ds_f = MergeDataset(features_hlt_std[hold_sub], masks_hlt[hold_sub], count_label[hold_sub])
            train_loader_f = DataLoader(train_ds_f, batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader_f = DataLoader(val_ds_f, batch_size=args.batch_size, shuffle=False)

            model = MergePredictor(
                input_dim=7,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                ff_dim=args.ff_dim,
                dropout=args.dropout,
                num_classes=max_count,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            sch = get_scheduler(opt, args.warmup_epochs, args.epochs)
            best_auc, best_state, no_improve = 0.0, None, 0
            for ep in tqdm(range(args.epochs), desc=f"MergePredictor-F{fold_id+1}"):
                train_loss, train_acc = train_epoch(model, train_loader_f, opt, device, class_weights)
                val_acc, _, _ = evaluate(model, val_loader_f, device)
                sch.step()
                if val_acc > best_auc:
                    best_auc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if (ep + 1) % 5 == 0:
                    print(f"Ep {ep+1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, best={best_auc:.4f}")
                if no_improve >= args.patience:
                    print(f"Early stopping at epoch {ep+1}")
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            models.append(model)

        if args.kfold_ensemble_valtest:
            print("Ensembling K models for val/test...")
            # majority vote
            def ensemble_predict(loader):
                preds_stack = []
                for m in models:
                    preds_stack.append(predict_classes(m, loader, device))
                preds_stack = np.stack(preds_stack, axis=0)  # (K, B, L)
                counts = np.zeros((preds_stack.shape[1], preds_stack.shape[2], max_count), dtype=np.int32)
                for c in range(max_count):
                    counts[..., c] = (preds_stack == c).sum(axis=0)
                return counts.argmax(axis=2)

            val_preds = ensemble_predict(val_loader)
            test_preds = ensemble_predict(test_loader)
            val_labs = count_label[val_idx]
            test_labs = count_label[test_idx]
            val_acc = (val_preds[masks_hlt[val_idx]] == val_labs[masks_hlt[val_idx]]).mean()
            test_acc = (test_preds[masks_hlt[test_idx]] == test_labs[masks_hlt[test_idx]]).mean()
            test_mae = clipped_mae(test_preds, test_labs, max_count)
        else:
            model = models[-1]
            val_acc, val_preds, val_labs = evaluate(model, val_loader, device)
            test_acc, test_preds, test_labs = evaluate(model, test_loader, device)
            test_mae = clipped_mae(test_preds, test_labs, max_count)
    else:
        model = MergePredictor(
            input_dim=7,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            num_classes=max_count,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = get_scheduler(opt, args.warmup_epochs, args.epochs)

        best_auc, best_state, no_improve = 0.0, None, 0

        print("\n" + "=" * 70)
        print("TRAINING: MERGE COUNT PREDICTOR (HLT -> count labels)")
        print("=" * 70)
        for ep in tqdm(range(args.epochs), desc="MergePredictor"):
            train_loss, train_acc = train_epoch(model, train_loader, opt, device, class_weights)
            val_acc, _, _ = evaluate(model, val_loader, device)
            sch.step()

            if val_acc > best_auc:
                best_auc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, best={best_auc:.4f}")

            if no_improve >= args.patience:
                print(f"Early stopping at epoch {ep+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        val_acc, val_preds, val_labs = evaluate(model, val_loader, device)
        test_acc, test_preds, test_labs = evaluate(model, test_loader, device)
        test_mae = clipped_mae(test_preds, test_labs, max_count)

    print("\nFinal test accuracy:", f"{test_acc:.4f}")
    print(f"Final test MAE (clipped @ {max_count}): {test_mae:.4f}")

    cm = confusion_matrix(test_preds, test_labs, max_count)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Fraction")
    tick_labels = [str(i) for i in range(1, max_count + 1)]
    plt.xticks(np.arange(max_count), tick_labels)
    plt.yticks(np.arange(max_count), tick_labels)
    plt.xlabel("Predicted merge count")
    plt.ylabel("True merge count")
    plt.title(f"Confusion (acc={test_acc:.3f})")
    plt.tight_layout()
    plt.savefig(save_root / "results.png", dpi=300)
    plt.close()

    np.savez(
        save_root / "results.npz",
        labs=test_labs,
        preds=test_preds,
        acc=test_acc,
        mae=test_mae,
        confusion=cm,
        confusion_norm=cm_norm,
        class_weights=class_weights,
        val_acc=val_acc,
    )

    if not args.skip_save_models:
        if k_folds > 1:
            torch.save(
                {"models": [m.state_dict() for m in models], "max_merge_count": max_count},
                save_root / "merge_predictor_folds.pt",
            )
        else:
            torch.save(
                {"model": model.state_dict(), "best_val_acc": best_auc, "max_merge_count": max_count},
                save_root / "merge_predictor.pt",
            )

    print(f"Saved results to: {save_root}")


if __name__ == "__main__":
    main()

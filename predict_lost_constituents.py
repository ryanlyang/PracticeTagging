#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict which HLT constituents are merged versions of offline constituents.

Training uses paired offline/HLT jets to label merged HLT tokens, but the model
only consumes HLT features at inference.
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt
from tqdm import tqdm

import utils


def safe_sigmoid(logits):
    probs = torch.sigmoid(logits)
    return torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)


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
    return hlt, hlt_mask, merged_label, stats


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
    def __init__(self, feat_hlt, mask_hlt, merged_label):
        self.hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask = torch.tensor(mask_hlt, dtype=torch.bool)
        self.label = torch.tensor(merged_label, dtype=torch.float32)

    def __len__(self):
        return len(self.hlt)

    def __getitem__(self, i):
        return {
            "hlt": self.hlt[i],
            "mask": self.mask[i],
            "label": self.label[i],
        }


class MergePredictor(nn.Module):
    def __init__(self, input_dim=7, embed_dim=128, num_heads=8, num_layers=6, ff_dim=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

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
            nn.Linear(max(embed_dim // 2, 32), 1),
        )

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)
        h = self.transformer(h, src_key_padding_mask=~mask)
        logits = self.head(h).squeeze(-1)
        return logits


def compute_pos_weight(labels, mask):
    valid = labels[mask]
    pos = float(valid.sum())
    neg = float(valid.size - pos)
    if pos <= 0:
        return 1.0
    return max(neg / pos, 1.0)


def train_epoch(model, loader, opt, device, pos_weight):
    model.train()
    total_loss = 0.0
    preds, labs = [], []
    weight = torch.tensor(pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

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
        preds.extend(safe_sigmoid(logits[mask]).detach().cpu().numpy().flatten())
        labs.extend(y[mask].detach().cpu().numpy().flatten())

    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else 0.0
    return total_loss / max(len(labs), 1), auc


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
            print("Warning: NaN/Inf in logits during evaluation; replacing with 0.5.")
            warned = True
        preds.extend(safe_sigmoid(logits[mask]).cpu().numpy().flatten())
        labs.extend(batch["label"][mask.cpu()].numpy().flatten())
    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else 0.0
    return auc, preds, labs


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def best_threshold_accuracy(preds, labs, thresholds=None):
    if preds.size == 0:
        return 0.5, 0.0
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)
    labs_i = labs.astype(np.int64)
    best_acc = 0.0
    best_thr = 0.5
    for thr in thresholds:
        acc = ((preds >= thr).astype(np.int64) == labs_i).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, float(best_acc)


def accuracy_at_threshold(preds, labs, threshold):
    if preds.size == 0:
        return 0.0
    labs_i = labs.astype(np.int64)
    acc = ((preds >= threshold).astype(np.int64) == labs_i).mean()
    return float(acc)


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
    constituents_hlt, masks_hlt, merged_label, stats = apply_hlt_effects_with_labels(
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

    train_ds = MergeDataset(features_hlt_std[train_idx], masks_hlt[train_idx], merged_label[train_idx].astype(np.float32))
    val_ds = MergeDataset(features_hlt_std[val_idx], masks_hlt[val_idx], merged_label[val_idx].astype(np.float32))
    test_ds = MergeDataset(features_hlt_std[test_idx], masks_hlt[test_idx], merged_label[test_idx].astype(np.float32))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    pos_weight = compute_pos_weight(merged_label[train_idx].astype(np.float32), masks_hlt[train_idx])
    print(f"Positive class weight: {pos_weight:.3f}")

    model = MergePredictor(
        input_dim=7,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = get_scheduler(opt, args.warmup_epochs, args.epochs)

    best_auc, best_state, no_improve = 0.0, None, 0

    print("\n" + "=" * 70)
    print("TRAINING: MERGE PREDICTOR (HLT -> merged token labels)")
    print("=" * 70)
    for ep in tqdm(range(args.epochs), desc="MergePredictor"):
        train_loss, train_auc = train_epoch(model, train_loader, opt, device, pos_weight)
        val_auc, _, _ = evaluate(model, val_loader, device)
        sch.step()

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_auc, val_preds, val_labs = evaluate(model, val_loader, device)
    test_auc, test_preds, test_labs = evaluate(model, test_loader, device)
    best_thr, best_val_acc = best_threshold_accuracy(val_preds, val_labs)
    test_acc = accuracy_at_threshold(test_preds, test_labs, best_thr)

    print("\nFinal test AUC:", f"{test_auc:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f} at threshold={best_thr:.3f}")
    print(f"Test accuracy @ val-threshold: {test_acc:.4f}")

    fpr, tpr, _ = roc_curve(test_labs, test_preds) if len(np.unique(test_labs)) > 1 else (np.array([0, 1]), np.array([0, 1]), None)
    plt.figure(figsize=(8, 6))
    plt.plot(tpr, fpr, "-", label=f"Merge predictor (AUC={test_auc:.3f})", color="steelblue", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate", fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_root / "results.png", dpi=300)
    plt.close()

    np.savez(
        save_root / "results.npz",
        labs=test_labs,
        preds=test_preds,
        auc=test_auc,
        fpr=fpr,
        tpr=tpr,
        pos_weight=pos_weight,
        best_val_acc=best_val_acc,
        best_val_threshold=best_thr,
        test_acc=test_acc,
    )

    if not args.skip_save_models:
        torch.save(
            {"model": model.state_dict(), "auc": best_auc},
            save_root / "merge_predictor.pt",
        )

    print(f"Saved results to: {save_root}")


if __name__ == "__main__":
    main()

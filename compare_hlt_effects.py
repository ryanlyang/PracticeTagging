#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare HLT reconstruction effects by retraining an HLT baseline.

Pipeline:
  1) Train a single offline teacher on offline data.
  2) For each HLT config:
     - regenerate HLT view
     - train HLT baseline (no KD)
     - evaluate vs offline teacher
     - save ROC plot + results
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
        "pt_resolution": 0.10,
        "eta_resolution": 0.03,
        "phi_resolution": 0.03,
        "pt_threshold_offline": 0.5,
        "pt_threshold_hlt": 1.5,
        "merge_enabled": True,
        "merge_radius": 0.01,
        "efficiency_loss": 0.03,
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
        "epochs": 50,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 3,
        "patience": 15,
    },
}


def apply_hlt_effects(const, mask, cfg, seed=42):
    np.random.seed(seed)
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    # Effect 1: Higher pT threshold
    pt_threshold = hcfg["pt_threshold_hlt"]
    below_threshold = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0

    # Effect 2: Cluster merging
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
                        to_remove.add(idx_j)

            for idx in to_remove:
                hlt_mask[jet_idx, idx] = False
                hlt[jet_idx, idx] = 0

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
    if hcfg["efficiency_loss"] > 0:
        random_loss = np.random.random((n_jets, max_part)) < hcfg["efficiency_loss"]
        lost = random_loss & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0
    return hlt, hlt_mask


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
    jet_E  = (E  * mask_float).sum(axis=1, keepdims=True)

    jet_pt = np.sqrt(jet_px**2 + jet_py**2) + 1e-8
    jet_p  = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)

    delta_eta = eta - jet_eta
    delta_phi = np.arctan2(np.sin(phi - jet_phi), np.cos(phi - jet_phi))

    log_pt = np.log(pt + 1e-8)
    log_E  = np.log(E  + 1e-8)

    log_pt_rel = np.log(pt / jet_pt + 1e-8)
    log_E_rel  = np.log(E  / (jet_E + 1e-8) + 1e-8)

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


class JetDataset(Dataset):
    def __init__(self, feat_off, feat_hlt, labels, mask_off, mask_hlt):
        self.off = torch.tensor(feat_off, dtype=torch.float32)
        self.hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "off": self.off[i],
            "hlt": self.hlt[i],
            "mask_off": self.mask_off[i],
            "mask_hlt": self.mask_hlt[i],
            "label": self.labels[i],
        }


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ParticleTransformer(nn.Module):
    def __init__(self, input_dim=7, embed_dim=128, num_heads=8, num_layers=6, ff_dim=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim

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

        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(128, dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)
        h = self.transformer(h, src_key_padding_mask=~mask)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(
            query, h, h,
            key_padding_mask=~mask,
            need_weights=False,
        )
        z = self.norm(pooled.squeeze(1))
        logits = self.classifier(z)
        return logits


def train_standard(model, loader, opt, device, feat_key, mask_key):
    model.train()
    total_loss = 0.0
    preds, labs = [], []
    for batch in loader:
        x = batch[feat_key].to(device)
        mask = batch[mask_key].to(device)
        y = batch["label"].to(device)
        opt.zero_grad()
        logits = model(x, mask).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(y)
        preds.extend(safe_sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
    return total_loss / len(preds), roc_auc_score(labs, preds)


@torch.no_grad()
def evaluate(model, loader, device, feat_key, mask_key):
    model.eval()
    preds, labs = [], []
    warned = False
    for batch in loader:
        x = batch[feat_key].to(device)
        mask = batch[mask_key].to(device)
        logits = model(x, mask).squeeze(1)
        if not warned and not torch.isfinite(logits).all():
            print("Warning: NaN/Inf in logits during evaluation; replacing with 0.5.")
            warned = True
        preds.extend(safe_sigmoid(logits).cpu().numpy().flatten())
        labs.extend(batch["label"].cpu().numpy().flatten())
    preds = np.array(preds)
    labs = np.array(labs)
    return roc_auc_score(labs, preds), preds, labs


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "compare_hlt_effects"))
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=CONFIG["training"]["epochs"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["training"]["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["training"]["lr"])
    parser.add_argument("--weight_decay", type=float, default=CONFIG["training"]["weight_decay"])
    parser.add_argument("--warmup_epochs", type=int, default=CONFIG["training"]["warmup_epochs"])
    parser.add_argument("--patience", type=int, default=CONFIG["training"]["patience"])
    parser.add_argument("--skip_save_models", action="store_true")
    args = parser.parse_args()

    CONFIG["training"]["epochs"] = args.epochs
    CONFIG["training"]["batch_size"] = args.batch_size
    CONFIG["training"]["lr"] = args.lr
    CONFIG["training"]["weight_decay"] = args.weight_decay
    CONFIG["training"]["warmup_epochs"] = args.warmup_epochs
    CONFIG["training"]["patience"] = args.patience

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
    all_data, all_labels, _, _, all_pt = utils.load_from_files(
        train_files,
        max_jets=args.n_train_jets,
        max_constits=args.max_constits,
        use_train_weights=False,
    )
    all_labels = all_labels.astype(np.int64)

    eta = all_data[:, :, ETA_IDX].astype(np.float32)
    phi = all_data[:, :, PHI_IDX].astype(np.float32)
    pt = all_data[:, :, PT_IDX].astype(np.float32)

    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    constituents_raw = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0

    print("Computing offline features...")
    features_off = compute_features(constituents_off, masks_off)

    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)

    train_ds_off = JetDataset(features_off_std[train_idx], features_off_std[train_idx], all_labels[train_idx], masks_off[train_idx], masks_off[train_idx])
    val_ds_off = JetDataset(features_off_std[val_idx], features_off_std[val_idx], all_labels[val_idx], masks_off[val_idx], masks_off[val_idx])
    test_ds_off = JetDataset(features_off_std[test_idx], features_off_std[test_idx], all_labels[test_idx], masks_off[test_idx], masks_off[test_idx])

    BS = CONFIG["training"]["batch_size"]
    train_loader_off = DataLoader(train_ds_off, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_off = DataLoader(val_ds_off, batch_size=BS, shuffle=False)
    test_loader_off = DataLoader(test_ds_off, batch_size=BS, shuffle=False)

    print("\n" + "=" * 70)
    print("STEP 1: OFFLINE TEACHER")
    print("=" * 70)
    teacher = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_t = get_scheduler(opt_t, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

    best_auc_t, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Teacher"):
        train_loss, train_auc = train_standard(teacher, train_loader_off, opt_t, device, "off", "mask_off")
        val_auc, _, _ = evaluate(teacher, val_loader_off, device, "off", "mask_off")
        sch_t.step()
        if val_auc > best_auc_t:
            best_auc_t = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_t:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping teacher at epoch {ep+1}")
            break

    if best_state is not None:
        teacher.load_state_dict(best_state)

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": best_auc_t}, save_root / "teacher.pt")

    auc_teacher, preds_teacher, labs = evaluate(teacher, test_loader_off, device, "off", "mask_off")

    hlt_configs = [
        ("default", {
            "pt_resolution": 0.10,
            "eta_resolution": 0.03,
            "phi_resolution": 0.03,
            "pt_threshold_offline": 0.5,
            "pt_threshold_hlt": 1.5,
            "merge_enabled": True,
            "merge_radius": 0.01,
            "efficiency_loss": 0.03,
            "noise_enabled": False,
            "noise_fraction": 0.0,
        }),
        ("no_smear_no_merge_eff0p05", {
            "pt_resolution": 0.0,
            "eta_resolution": 0.0,
            "phi_resolution": 0.0,
            "pt_threshold_offline": 0.5,
            "pt_threshold_hlt": 1.5,
            "merge_enabled": False,
            "merge_radius": 0.01,
            "efficiency_loss": 0.05,
            "noise_enabled": False,
            "noise_fraction": 0.0,
        }),
        ("no_smear_merge_eff0p05", {
            "pt_resolution": 0.0,
            "eta_resolution": 0.0,
            "phi_resolution": 0.0,
            "pt_threshold_offline": 0.5,
            "pt_threshold_hlt": 1.5,
            "merge_enabled": True,
            "merge_radius": 0.01,
            "efficiency_loss": 0.05,
            "noise_enabled": False,
            "noise_fraction": 0.0,
        }),
        ("smear_no_merge_eff0p0", {
            "pt_resolution": 0.10,
            "eta_resolution": 0.03,
            "phi_resolution": 0.03,
            "pt_threshold_offline": 0.5,
            "pt_threshold_hlt": 1.5,
            "merge_enabled": False,
            "merge_radius": 0.01,
            "efficiency_loss": 0.0,
            "noise_enabled": False,
            "noise_fraction": 0.0,
        }),
    ]

    for cfg_name, hcfg in hlt_configs:
        print("\n" + "=" * 70)
        print(f"HLT CONFIG: {cfg_name}")
        print("=" * 70)
        cfg = copy.deepcopy(CONFIG)
        cfg["hlt_effects"] = hcfg

        constituents_hlt, masks_hlt = apply_hlt_effects(constituents_raw, mask_raw, cfg, seed=RANDOM_SEED)
        features_hlt = compute_features(constituents_hlt, masks_hlt)
        features_hlt_std = standardize(features_hlt, masks_hlt, feat_means, feat_stds)

        train_ds = JetDataset(features_off_std[train_idx], features_hlt_std[train_idx], all_labels[train_idx], masks_off[train_idx], masks_hlt[train_idx])
        val_ds = JetDataset(features_off_std[val_idx], features_hlt_std[val_idx], all_labels[val_idx], masks_off[val_idx], masks_hlt[val_idx])
        test_ds = JetDataset(features_off_std[test_idx], features_hlt_std[test_idx], all_labels[test_idx], masks_off[test_idx], masks_hlt[test_idx])

        train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)

        baseline = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
        opt_b = torch.optim.AdamW(baseline.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch_b = get_scheduler(opt_b, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
        best_auc_b, best_state_b, no_improve = 0.0, None, 0

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc=f"Baseline-{cfg_name}"):
            train_loss, train_auc = train_standard(baseline, train_loader, opt_b, device, "hlt", "mask_hlt")
            val_auc, _, _ = evaluate(baseline, val_loader, device, "hlt", "mask_hlt")
            sch_b.step()
            if val_auc > best_auc_b:
                best_auc_b = val_auc
                best_state_b = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_b:.4f}")
            if no_improve >= CONFIG["training"]["patience"] + 5:
                print(f"Early stopping baseline at epoch {ep+1}")
                break

        if best_state_b is not None:
            baseline.load_state_dict(best_state_b)

        auc_baseline, preds_baseline, _ = evaluate(baseline, test_loader, device, "hlt", "mask_hlt")

        cfg_dir = save_root / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
        fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)

        plt.figure(figsize=(8, 6))
        plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={auc_teacher:.3f})", color="crimson", linewidth=2)
        plt.plot(tpr_b, fpr_b, "--", label=f"HLT baseline (AUC={auc_baseline:.3f})", color="steelblue", linewidth=2)
        plt.ylabel("False Positive Rate", fontsize=12)
        plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
        plt.legend(fontsize=12, frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(cfg_dir / "results.png", dpi=300)
        plt.close()

        np.savez(
            cfg_dir / "results.npz",
            labs=labs,
            preds_teacher=preds_teacher,
            preds_baseline=preds_baseline,
            auc_teacher=auc_teacher,
            auc_baseline=auc_baseline,
            fpr_teacher=fpr_t, tpr_teacher=tpr_t,
            fpr_baseline=fpr_b, tpr_baseline=tpr_b,
            hlt_cfg=np.array([cfg_name], dtype=object),
        )

        print(f"Saved results to: {cfg_dir / 'results.npz'} and {cfg_dir / 'results.png'}")


if __name__ == "__main__":
    main()

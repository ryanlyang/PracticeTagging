#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline Teacher + Two-HLT-Views Suite (Augmentation + Consistency)

Trains 4 models:
  1) Teacher (OFFLINE only)                 -> tested on OFFLINE
  2) Baseline (HLT-A only)                  -> tested on HLT-A
  3) Mixed (HLT-A / HLT-B random per sample)-> tested on HLT-A
  4) Consistency (HLT-A + HLT-B, paired)    -> tested on HLT-A

Key idea:
- Create two HLT datasets from the same offline jets using two different random seeds.
- Mixed model treats them as just two noisy versions (no pairing used).
- Consistency model uses pairing and adds confidence-weighted symmetric KL on probabilities.

Notes:
- Keeps your previous pipeline: utils.load_from_files, apply_hlt_effects, compute_features, standardize.
- ROC plot uses your preferred axes: TPR on x, FPR on y.
- Uses matplotlib Agg backend to avoid Qt plugin issues on Linux.

Assumption about utils.load_from_files output:
  all_data: (N, max_constits, 3) with columns [eta, phi, pt]
If your columns differ, edit ETA_IDX/PHI_IDX/PT_IDX below.
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

import matplotlib
matplotlib.use("Agg")
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
PT_IDX  = 2


# ----------------------------- HLT config (matches your professor) ----------------------------- #
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


# ----------------------------- HLT Simulation (same logic as professor) ----------------------------- #
def apply_hlt_effects(const, mask, cfg, seed=42):
    """
    const: (n_jets, max_part, 4) with columns [pt, eta, phi, E]
    mask:  (n_jets, max_part) boolean
    """
    np.random.seed(seed)
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    n_initial = hlt_mask.sum()

    # Effect 1: Higher pT threshold
    pt_threshold = hcfg["pt_threshold_hlt"]
    below_threshold = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0
    n_lost_threshold = below_threshold.sum()

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

                        # Merge into particle i
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
                        n_merged += 1

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
    n_lost_eff = 0
    if hcfg["efficiency_loss"] > 0:
        random_loss = np.random.random((n_jets, max_part)) < hcfg["efficiency_loss"]
        lost = random_loss & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0
        n_lost_eff = lost.sum()

    # Final cleanup
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0

    n_final = hlt_mask.sum()
    retention = 100 * n_final / max(n_initial, 1)

    print("\nHLT Simulation Statistics:")
    print(f"  Offline particles: {n_initial:,}")
    print(f"  Lost to pT threshold ({hcfg['pt_threshold_hlt']}): {n_lost_threshold:,} ({100*n_lost_threshold/max(n_initial,1):.1f}%)")
    print(f"  Lost to merging (dR<{hcfg['merge_radius']}): {n_merged:,} ({100*n_merged/max(n_initial,1):.1f}%)")
    print(f"  Lost to efficiency: {n_lost_eff:,} ({100*n_lost_eff/max(n_initial,1):.1f}%)")
    print(f"  HLT particles: {n_final:,} ({retention:.1f}% of offline)")

    return hlt, hlt_mask


# ----------------------------- Feature computation (same as professor) ----------------------------- #
def compute_features(const, mask):
    """
    const: (N, M, 4) [pt, eta, phi, E]
    mask:  (N, M) bool
    returns: (N, M, 7)
    """
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


# ----------------------------- Dataset (offline + two HLT views) ----------------------------- #
class JetDatasetTwoHLT(Dataset):
    def __init__(self, feat_off, feat_hlt_a, feat_hlt_b, labels, mask_off, mask_a, mask_b):
        self.off = torch.tensor(feat_off, dtype=torch.float32)
        self.hlt_a = torch.tensor(feat_hlt_a, dtype=torch.float32)
        self.hlt_b = torch.tensor(feat_hlt_b, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.mask_a = torch.tensor(mask_a, dtype=torch.bool)
        self.mask_b = torch.tensor(mask_b, dtype=torch.bool)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "off": self.off[i],
            "hlt_a": self.hlt_a[i],
            "hlt_b": self.hlt_b[i],
            "mask_off": self.mask_off[i],
            "mask_a": self.mask_a[i],
            "mask_b": self.mask_b[i],
            "label": self.labels[i],
        }


# ----------------------------- Model (same backbone style) ----------------------------- #
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


# ----------------------------- Consistency loss helpers ----------------------------- #
def binary_kl(p, q, eps=1e-6):
    """
    KL(p || q) for Bernoulli probabilities p and q.
    p, q are tensors in [0,1].
    """
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


def conf_weight(p_a, p_b):
    """
    Confidence weight in [0,1], higher when away from 0.5.
    Uses detached average probability.
    """
    p_bar = 0.5 * (p_a + p_b)
    w = (torch.abs(p_bar - 0.5) * 2.0).detach()
    return torch.clamp(w, 0.0, 1.0)


def ramp_value(epoch, total_epochs, max_value, ramp_frac):
    """
    Linear ramp from 0 to max_value over the first ramp_frac of epochs.
    """
    if ramp_frac is None or ramp_frac <= 0:
        return max_value
    ramp_epochs = max(int(round(total_epochs * ramp_frac)), 1)
    if epoch >= ramp_epochs:
        return max_value
    return max_value * float(epoch + 1) / float(ramp_epochs)


# ----------------------------- Train / eval ----------------------------- #
def train_epoch_standard(model, loader, opt, device, feat_key, mask_key):
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
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / max(len(preds), 1), roc_auc_score(labs, preds)


def train_epoch_mixed(model, loader, opt, device):
    """
    Randomly samples per-sample from HLT-A or HLT-B, no pairing and no consistency.
    """
    model.train()
    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        xa = batch["hlt_a"].to(device)
        xb = batch["hlt_b"].to(device)
        ma = batch["mask_a"].to(device)
        mb = batch["mask_b"].to(device)
        y = batch["label"].to(device)

        # per-sample mix
        choose_a = (torch.rand(y.shape[0], device=device) < 0.5).view(-1, 1, 1)
        x = torch.where(choose_a, xa, xb)

        choose_a_mask = choose_a.view(-1, 1)
        mask = torch.where(choose_a_mask, ma, mb)

        opt.zero_grad()
        logits = model(x, mask).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / max(len(preds), 1), roc_auc_score(labs, preds)


def train_epoch_consistency(model, loader, opt, device, lam_cons, gate_thr=None):
    """
    Best version for this setup:
      - Supervised BCE on BOTH HLT views (average)
      - Confidence-weighted symmetric KL between probs pA and pB
      - Optional hard gating by confidence threshold
    """
    model.train()
    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        xa = batch["hlt_a"].to(device)
        xb = batch["hlt_b"].to(device)
        ma = batch["mask_a"].to(device)
        mb = batch["mask_b"].to(device)
        y = batch["label"].to(device)

        opt.zero_grad()

        za = model(xa, ma).squeeze(1)
        zb = model(xb, mb).squeeze(1)

        # supervised on both views
        loss_hard = 0.5 * (
            F.binary_cross_entropy_with_logits(za, y) +
            F.binary_cross_entropy_with_logits(zb, y)
        )

        pa = torch.sigmoid(za)
        pb = torch.sigmoid(zb)

        w = conf_weight(pa, pb)
        if gate_thr is not None:
            w = w * (w >= gate_thr).float()

        # symmetric KL, with stopgrad on the "teacher side" of each direction
        kl_a_to_b = binary_kl(pa.detach(), pb)
        kl_b_to_a = binary_kl(pb.detach(), pa)
        loss_cons = 0.5 * (kl_a_to_b + kl_b_to_a)

        loss_cons = (w * loss_cons).mean()

        loss = loss_hard + lam_cons * loss_cons

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)

        # track AUC using averaged probability (usually a bit smoother)
        p_avg = 0.5 * (pa + pb)
        preds.extend(p_avg.detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / max(len(preds), 1), roc_auc_score(labs, preds)


@torch.no_grad()
def evaluate(model, loader, device, feat_key, mask_key):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        x = batch[feat_key].to(device)
        mask = batch[mask_key].to(device)
        logits = model(x, mask).squeeze(1)
        preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        labs.extend(batch["label"].cpu().numpy().flatten())
    preds = np.array(preds)
    labs = np.array(labs)
    return roc_auc_score(labs, preds), preds, labs


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=10000)
    parser.add_argument("--max_constits", type=int, default=80)

    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "transformer_twohlt"))
    parser.add_argument("--run_name", type=str, default="default")

    parser.add_argument("--device", type=str, default="cpu")

    # two HLT seeds
    parser.add_argument("--hlt_seed_a", type=int, default=123)
    parser.add_argument("--hlt_seed_b", type=int, default=456)

    # consistency hyperparams
    parser.add_argument("--lam_cons", type=float, default=1.0, help="Max weight for consistency term")
    parser.add_argument("--cons_ramp_frac", type=float, default=0.2, help="Fraction of epochs to ramp lambda from 0 to lam_cons")
    parser.add_argument("--gate_thr", type=float, default=None, help="Optional hard gate on confidence weight, e.g. 0.2")

    # checkpoints (optional)
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--baseline_checkpoint", type=str, default=None)
    parser.add_argument("--mixed_checkpoint", type=str, default=None)
    parser.add_argument("--consistency_checkpoint", type=str, default=None)

    parser.add_argument("--skip_save_models", action="store_true")

    args = parser.parse_args()

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

    # ------------------- Load dataset ------------------- #
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

    print(f"Avg particles per jet (raw mask): {mask_raw.sum(axis=1).mean():.1f}")

    # ------------------- Offline threshold ------------------- #
    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0

    print(f"Offline particles after {pt_threshold_off} threshold: {masks_off.sum():,}")
    print(f"Avg per jet (offline): {masks_off.sum(axis=1).mean():.1f}")

    # ------------------- Two HLT views with different seeds ------------------- #
    print("\n" + "=" * 70)
    print(f"Generating HLT-A with seed={args.hlt_seed_a}")
    print("=" * 70)
    constituents_hlt_a, masks_hlt_a = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=args.hlt_seed_a)

    print("\n" + "=" * 70)
    print(f"Generating HLT-B with seed={args.hlt_seed_b}")
    print("=" * 70)
    constituents_hlt_b, masks_hlt_b = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=args.hlt_seed_b)

    print(f"\nAvg per jet: Offline={masks_off.sum(axis=1).mean():.1f}, HLT-A={masks_hlt_a.sum(axis=1).mean():.1f}, HLT-B={masks_hlt_b.sum(axis=1).mean():.1f}")

    # ------------------- Compute features ------------------- #
    print("\nComputing features...")
    features_off = compute_features(constituents_off, masks_off)
    features_a   = compute_features(constituents_hlt_a, masks_hlt_a)
    features_b   = compute_features(constituents_hlt_b, masks_hlt_b)

    # ------------------- Split indices (70/15/15 stratified) ------------------- #
    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # ------------------- Standardize using training OFFLINE stats ------------------- #
    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_a_std   = standardize(features_a,   masks_hlt_a, feat_means, feat_stds)
    features_b_std   = standardize(features_b,   masks_hlt_b, feat_means, feat_stds)

    # ------------------- Save test split artifacts ------------------- #
    test_data_dir = Path().cwd() / "test_split"
    test_data_dir.mkdir(exist_ok=True)
    np.savez(
        test_data_dir / "test_features_and_masks_twohlt.npz",
        idx_test=test_idx,
        labels=all_labels[test_idx],
        feat_off=features_off_std[test_idx],
        feat_hlt_a=features_a_std[test_idx],
        feat_hlt_b=features_b_std[test_idx],
        mask_off=masks_off[test_idx],
        mask_hlt_a=masks_hlt_a[test_idx],
        mask_hlt_b=masks_hlt_b[test_idx],
        jet_pt=all_pt[test_idx] if all_pt is not None else None,
        feat_means=feat_means,
        feat_stds=feat_stds,
        hlt_seed_a=args.hlt_seed_a,
        hlt_seed_b=args.hlt_seed_b,
    )
    print(f"Saved test artifacts to: {test_data_dir / 'test_features_and_masks_twohlt.npz'}")

    # ------------------- Build datasets/loaders ------------------- #
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

    BS = CONFIG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BS, shuffle=False)

    # ------------------- Checkpoint paths ------------------- #
    teacher_path     = save_dir / "teacher_offline.pt"
    baseline_path    = save_dir / "baseline_hlt_a.pt"
    mixed_path       = save_dir / "mixed_hlt_ab.pt"
    consistency_path = save_dir / "consistency_hlt_ab.pt"

    # ------------------- Model factory ------------------- #
    def make_model():
        return ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)

    # ------------------- STEP 1: Teacher (offline) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 1: TEACHER (Offline only)")
    print("=" * 70)

    teacher = make_model()

    if args.teacher_checkpoint is not None:
        print(f"Loading teacher from: {args.teacher_checkpoint}")
        ckpt = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(ckpt["model"])
        best_auc_teacher = ckpt.get("auc", None)
        print(f"Loaded teacher (val AUC={best_auc_teacher})")
    else:
        opt = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_auc_teacher, best_state, no_improve = 0.0, None, 0
        history_teacher = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Teacher"):
            train_loss, train_auc = train_epoch_standard(teacher, train_loader, opt, device, "off", "mask_off")
            val_auc, _, _ = evaluate(teacher, val_loader, device, "off", "mask_off")
            sch.step()

            history_teacher.append((ep + 1, train_loss, train_auc, val_auc))

            if val_auc > best_auc_teacher:
                best_auc_teacher = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_teacher:.4f}")

            if no_improve >= CONFIG["training"]["patience"]:
                print(f"Early stopping teacher at epoch {ep+1}")
                break

        teacher.load_state_dict(best_state)
        if not args.skip_save_models:
            torch.save({"model": teacher.state_dict(), "auc": best_auc_teacher, "history": history_teacher}, teacher_path)
            print(f"Saved teacher: {teacher_path} (best val AUC={best_auc_teacher:.4f})")

    # ------------------- STEP 2: Baseline (HLT-A only) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE (HLT-A only)")
    print("=" * 70)

    baseline = make_model()

    if args.baseline_checkpoint is not None:
        print(f"Loading baseline from: {args.baseline_checkpoint}")
        ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        baseline.load_state_dict(ckpt["model"])
        best_auc_baseline = ckpt.get("auc", None)
        print(f"Loaded baseline (val AUC={best_auc_baseline})")
    else:
        opt = torch.optim.AdamW(baseline.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_auc_baseline, best_state, no_improve = 0.0, None, 0
        history_baseline = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Baseline"):
            train_loss, train_auc = train_epoch_standard(baseline, train_loader, opt, device, "hlt_a", "mask_a")
            val_auc, _, _ = evaluate(baseline, val_loader, device, "hlt_a", "mask_a")
            sch.step()

            history_baseline.append((ep + 1, train_loss, train_auc, val_auc))

            if val_auc > best_auc_baseline:
                best_auc_baseline = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_baseline:.4f}")

            if no_improve >= CONFIG["training"]["patience"] + 5:
                print(f"Early stopping baseline at epoch {ep+1}")
                break

        baseline.load_state_dict(best_state)
        if not args.skip_save_models:
            torch.save({"model": baseline.state_dict(), "auc": best_auc_baseline, "history": history_baseline}, baseline_path)
            print(f"Saved baseline: {baseline_path} (best val AUC={best_auc_baseline:.4f})")

    # ------------------- STEP 3: Mixed (HLT-A/HLT-B random per sample) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 3: MIXED (HLT-A/HLT-B random per sample, no consistency)")
    print("=" * 70)

    mixed = make_model()

    if args.mixed_checkpoint is not None:
        print(f"Loading mixed from: {args.mixed_checkpoint}")
        ckpt = torch.load(args.mixed_checkpoint, map_location=device)
        mixed.load_state_dict(ckpt["model"])
        best_auc_mixed = ckpt.get("auc", None)
        print(f"Loaded mixed (val AUC={best_auc_mixed})")
    else:
        opt = torch.optim.AdamW(mixed.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_auc_mixed, best_state, no_improve = 0.0, None, 0
        history_mixed = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Mixed"):
            train_loss, train_auc = train_epoch_mixed(mixed, train_loader, opt, device)
            val_auc, _, _ = evaluate(mixed, val_loader, device, "hlt_a", "mask_a")
            sch.step()

            history_mixed.append((ep + 1, train_loss, train_auc, val_auc))

            if val_auc > best_auc_mixed:
                best_auc_mixed = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in mixed.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_mixed:.4f}")

            if no_improve >= CONFIG["training"]["patience"] + 5:
                print(f"Early stopping mixed at epoch {ep+1}")
                break

        mixed.load_state_dict(best_state)
        if not args.skip_save_models:
            torch.save({"model": mixed.state_dict(), "auc": best_auc_mixed, "history": history_mixed}, mixed_path)
            print(f"Saved mixed: {mixed_path} (best val AUC={best_auc_mixed:.4f})")

    # ------------------- STEP 4: Consistency (HLT-A + HLT-B paired) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 4: CONSISTENCY (HLT-A + HLT-B paired + confidence-weighted symmetric KL)")
    print("=" * 70)

    cons = make_model()

    if args.consistency_checkpoint is not None:
        print(f"Loading consistency model from: {args.consistency_checkpoint}")
        ckpt = torch.load(args.consistency_checkpoint, map_location=device)
        cons.load_state_dict(ckpt["model"])
        best_auc_cons = ckpt.get("auc", None)
        print(f"Loaded consistency (val AUC={best_auc_cons})")
    else:
        opt = torch.optim.AdamW(cons.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_auc_cons, best_state, no_improve = 0.0, None, 0
        history_cons = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Consistency"):
            lam = ramp_value(ep, CONFIG["training"]["epochs"], args.lam_cons, args.cons_ramp_frac)
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

            if no_improve >= CONFIG["training"]["patience"] + 5:
                print(f"Early stopping consistency at epoch {ep+1}")
                break

        cons.load_state_dict(best_state)
        if not args.skip_save_models:
            torch.save({"model": cons.state_dict(), "auc": best_auc_cons, "history": history_cons}, consistency_path)
            print(f"Saved consistency: {consistency_path} (best val AUC={best_auc_cons:.4f})")

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

    # Background rejection at 50% signal efficiency for HLT-tested models
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

    # ROC curves (TPR on x, FPR on y)
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
        f.write(f"\nRun: {args.run_name}\n")
        f.write(f"  hlt_seed_a={args.hlt_seed_a}, hlt_seed_b={args.hlt_seed_b}\n")
        f.write(f"  lam_cons={args.lam_cons}, cons_ramp_frac={args.cons_ramp_frac}, gate_thr={args.gate_thr}\n")
        f.write(f"  AUC teacher(off)={auc_teacher:.4f}\n")
        f.write(f"  AUC baseline(hlt_a)={auc_baseline:.4f}\n")
        f.write(f"  AUC mixed(hlt_a)={auc_mixed:.4f}\n")
        f.write(f"  AUC consistency(hlt_a)={auc_cons:.4f}\n")
        f.write(f"  BR@50 baseline={br_baseline:.2f}, mixed={br_mixed:.2f}, consistency={br_cons:.2f}\n")
        f.write(f"  Saved to: {save_dir}\n")
        f.write("=" * 70 + "\n")

    print(f"\nSaved results to: {save_dir / 'results_twohlt.npz'} and {save_dir / 'results_twohlt.png'}")
    print(f"Logged to: {summary_file}")


if __name__ == "__main__":
    main()

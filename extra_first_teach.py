
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer Suite: Offline Teacher + Two-View HLT Augmentation + Consistency Training

What this script trains (4 models):
  1) Teacher (OFFLINE features)               -> evaluated on OFFLINE test
  2) Baseline (HLT view #1 only)             -> evaluated on HLT test (view #1)
  3) Union (HLT view #1 + view #2 as one big dataset, no pairing)
                                             -> evaluated on HLT test (view #1)
  4) Consistency (paired HLT view #1 vs #2):
       - supervised BCE on both views
       - confidence-weighted symmetric KL on probs
       - confidence-weighted cosine embedding alignment on pooled embedding
                                             -> evaluated on HLT test (view #1)

Key idea:
  - Generate TWO independently randomized HLT realizations per jet (different seeds).
  - "Union" ignores pairing (just doubles the dataset).
  - "Consistency" uses pairing and penalizes disagreement between the two HLT views.

Notes:
  - Uses OFFLINE training stats (means/stds computed from OFFLINE train split) to standardize:
      offline, hlt_view1, hlt_view2
  - Matplotlib backend forced to "Agg" to avoid Qt/Wayland plugin errors on Linux.
  - Assumes utils.load_from_files returns:
      all_data: (N, max_constits, 3) columns [eta, phi, pt]
      all_labels: (N,)
      all_pt optional
"""

from pathlib import Path
import argparse
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")  # avoid Qt platform plugin issues
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
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


# ----------------------------- Default Config ----------------------------- #
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
    "consistency": {
        # Default weights; overridden by CLI args
        "lambda_prob": 1.0,
        "lambda_emb": 0.25,
        "rampup_frac": 0.2,    # first 20% epochs ramp from 0 -> lambda
        "conf_power": 1.0,     # optionally sharpen confidence weights
        "conf_min": 0.0,       # clamp confidence weight lower bound
    },
}


# ----------------------------- HLT Simulation ----------------------------- #
def apply_hlt_effects(const, mask, cfg, seed=42, verbose=True):
    """
    const: (n_jets, max_part, 4) columns [pt, eta, phi, E]
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

            for idx_rm in to_remove:
                hlt_mask[jet_idx, idx_rm] = False
                hlt[jet_idx, idx_rm] = 0

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
    retention = 100.0 * n_final / max(n_initial, 1)

    if verbose:
        print("\nHLT Simulation Statistics:")
        print(f"  Seed: {seed}")
        print(f"  Offline particles: {n_initial:,}")
        print(f"  Lost to pT threshold ({hcfg['pt_threshold_hlt']}): {n_lost_threshold:,} ({100*n_lost_threshold/max(n_initial,1):.1f}%)")
        print(f"  Lost to merging (dR<{hcfg['merge_radius']}): {n_merged:,} ({100*n_merged/max(n_initial,1):.1f}%)")
        print(f"  Lost to efficiency: {n_lost_eff:,} ({100*n_lost_eff/max(n_initial,1):.1f}%)")
        print(f"  HLT particles: {n_final:,} ({retention:.1f}% of offline)")

    return hlt, hlt_mask


# ----------------------------- Feature computation ----------------------------- #
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

    features = np.stack(
        [delta_eta, delta_phi, log_pt, log_E, log_pt_rel, log_E_rel, delta_R],
        axis=-1
    )
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


# ----------------------------- Datasets ----------------------------- #
class JetDatasetSingle(Dataset):
    """Single-view dataset: {x, mask, label}"""
    def __init__(self, x, mask, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.x[i], "mask": self.mask[i], "label": self.y[i]}


class JetDatasetPaired(Dataset):
    """Paired two-view dataset: {x1, m1, x2, m2, label}"""
    def __init__(self, x1, m1, x2, m2, y):
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.m1 = torch.tensor(m1, dtype=torch.bool)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.m2 = torch.tensor(m2, dtype=torch.bool)
        self.y  = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x1": self.x1[i], "m1": self.m1[i], "x2": self.x2[i], "m2": self.m2[i], "label": self.y[i]}


# ----------------------------- Model ----------------------------- #
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

    def forward(self, x, mask, return_embedding=False):
        """
        x: (B, M, input_dim)
        mask: (B, M) True for valid particles
        """
        B, M, _ = x.shape

        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(B, M, -1)

        h = self.transformer(h, src_key_padding_mask=~mask)

        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(
            query, h, h,
            key_padding_mask=~mask,
            need_weights=False
        )
        z = self.norm(pooled.squeeze(1))          # pooled embedding
        logits = self.classifier(z)               # (B, 1)

        if return_embedding:
            return logits, z
        return logits


# ----------------------------- Eval helpers ----------------------------- #
@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].to(device)

        logits = model(x, m).squeeze(1)
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return roc_auc_score(labs, preds), preds, labs


def get_scheduler(opt, warmup_epochs, total_epochs):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ----------------------------- Training: standard ----------------------------- #
def train_one_epoch_standard(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].to(device)

        opt.zero_grad()
        logits = model(x, m).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return total_loss / len(labs), roc_auc_score(labs, preds)


# ----------------------------- Consistency losses ----------------------------- #
def symmetric_kl_bernoulli(p, q, eps=1e-6):
    """
    p, q: probabilities in (0,1), shape (B,)
    returns elementwise symmetric KL, shape (B,)
    """
    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)

    kl_pq = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    kl_qp = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
    return 0.5 * (kl_pq + kl_qp)


def confidence_weight(p1, p2, power=1.0, conf_min=0.0):
    """
    Confidence weight in [0,1], based on distance from 0.5.
    Uses max(|p-0.5|) * 2, detached so it doesn't game itself.
    """
    with torch.no_grad():
        c1 = torch.abs(p1 - 0.5) * 2.0
        c2 = torch.abs(p2 - 0.5) * 2.0
        c = torch.maximum(c1, c2)
        c = torch.clamp(c, 0.0, 1.0)
        if power != 1.0:
            c = torch.pow(c, power)
        if conf_min > 0.0:
            c = torch.clamp(c, conf_min, 1.0)
    return c


def cosine_embed_loss(z1, z2, eps=1e-8):
    """
    1 - cosine similarity, shape (B,)
    """
    z1n = z1 / (torch.norm(z1, dim=1, keepdim=True) + eps)
    z2n = z2 / (torch.norm(z2, dim=1, keepdim=True) + eps)
    cos = (z1n * z2n).sum(dim=1)
    return 1.0 - cos


def rampup_weight(epoch, total_epochs, rampup_frac):
    """
    Linear ramp from 0 to 1 over first rampup_frac of epochs.
    """
    ramp_epochs = int(np.ceil(total_epochs * rampup_frac))
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch + 1) / ramp_epochs))


# ----------------------------- Training: paired consistency ----------------------------- #
def train_one_epoch_consistency(model, loader, opt, device, lam_prob, lam_emb, ramp_mult, conf_power, conf_min,
                                epoch=0, attention_epoch=0):
    model.train()
    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x1 = batch["x1"].to(device)
        m1 = batch["m1"].to(device)
        x2 = batch["x2"].to(device)
        m2 = batch["m2"].to(device)
        y  = batch["label"].to(device)

        opt.zero_grad()

        logits1, z1 = model(x1, m1, return_embedding=True)
        logits2, z2 = model(x2, m2, return_embedding=True)
        logits1 = logits1.squeeze(1)
        logits2 = logits2.squeeze(1)

        # supervised on BOTH views (skip during unsupervised warmup)
        if epoch >= attention_epoch:
            loss_sup1 = F.binary_cross_entropy_with_logits(logits1, y)
            loss_sup2 = F.binary_cross_entropy_with_logits(logits2, y)
            loss_sup = 0.5 * (loss_sup1 + loss_sup2)
        else:
            loss_sup = 0.0  # Unsupervised warmup: train only on consistency losses

        # probabilities
        p1 = torch.sigmoid(logits1)
        p2 = torch.sigmoid(logits2)

        # confidence weighting (detached)
        w = confidence_weight(p1, p2, power=conf_power, conf_min=conf_min)

        # prob consistency: symmetric KL on probs
        l_kl = symmetric_kl_bernoulli(p1, p2)          # (B,)
        loss_prob = (w * l_kl).mean()

        # embedding consistency: cosine alignment on pooled embeddings
        l_cos = cosine_embed_loss(z1, z2)              # (B,)
        loss_emb = (w * l_cos).mean()

        loss = loss_sup + (lam_prob * ramp_mult) * loss_prob + (lam_emb * ramp_mult) * loss_emb

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)

        # for train AUC reporting, just use view1
        preds.append(p1.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return total_loss / len(labs), roc_auc_score(labs, preds)


# ----------------------------- Utility: early-stopped training loop ----------------------------- #
def fit_model_standard(model, train_loader, val_loader, device, cfg, desc, save_path=None, skip_save=False):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    sch = get_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["training"]["epochs"])

    best_auc, best_state, no_improve = 0.0, None, 0
    history = []

    for ep in tqdm(range(cfg["training"]["epochs"]), desc=desc):
        tr_loss, tr_auc = train_one_epoch_standard(model, train_loader, opt, device)
        va_auc, _, _ = evaluate_auc(model, val_loader, device)
        sch.step()

        history.append((ep + 1, tr_loss, tr_auc, va_auc))

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_auc:.4f}")

        if no_improve >= cfg["training"]["patience"]:
            print(f"Early stopping at epoch {ep+1} (best val AUC={best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if (save_path is not None) and (not skip_save):
        torch.save({"model": model.state_dict(), "auc": best_auc, "history": history}, save_path)
        print(f"Saved: {save_path} (best val AUC={best_auc:.4f})")
    else:
        print(f"Skip-save={skip_save}. Best val AUC={best_auc:.4f}")

    return best_auc, history


def fit_model_consistency(model, train_loader, val_loader, device, cfg, desc, save_path=None, skip_save=False,
                          lambda_prob=1.0, lambda_emb=0.25, rampup_frac=0.2, conf_power=1.0, conf_min=0.0,
                          attention_epoch=0):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    sch = get_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["training"]["epochs"])

    best_auc, best_state, no_improve = 0.0, None, 0
    history = []

    if attention_epoch > 0:
        print(f"[Unsupervised warmup] Training ONLY on consistency losses for first {attention_epoch} epochs")

    for ep in tqdm(range(cfg["training"]["epochs"]), desc=desc):
        ramp_mult = rampup_weight(ep, cfg["training"]["epochs"], rampup_frac)

        tr_loss, tr_auc = train_one_epoch_consistency(
            model, train_loader, opt, device,
            lam_prob=lambda_prob,
            lam_emb=lambda_emb,
            ramp_mult=ramp_mult,
            conf_power=conf_power,
            conf_min=conf_min,
            epoch=ep,
            attention_epoch=attention_epoch,
        )
        va_auc, _, _ = evaluate_auc(model, val_loader, device)
        sch.step()

        history.append((ep + 1, tr_loss, tr_auc, va_auc, ramp_mult))

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            warmup_tag = " [warmup]" if ep < attention_epoch else ""
            print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_auc:.4f}, ramp={ramp_mult:.2f}{warmup_tag}")

        # Print when transitioning from warmup to full training
        if attention_epoch > 0 and ep + 1 == attention_epoch:
            print(f"[Warmup complete] Switching to full training (supervised + consistency) at epoch {ep+1}")

        if no_improve >= cfg["training"]["patience"]:
            print(f"Early stopping at epoch {ep+1} (best val AUC={best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if (save_path is not None) and (not skip_save):
        torch.save({"model": model.state_dict(), "auc": best_auc, "history": history}, save_path)
        print(f"Saved: {save_path} (best val AUC={best_auc:.4f})")
    else:
        print(f"Skip-save={skip_save}. Best val AUC={best_auc:.4f}")

    return best_auc, history


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="./data",
                        help="Directory containing *.h5 files (default: ./data)")
    parser.add_argument("--n_train_jets", type=int, default=100000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "transformer_twohlt"))
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")

    # Two HLT seeds
    parser.add_argument("--hlt_seed1", type=int, default=123, help="Seed for HLT view #1")
    parser.add_argument("--hlt_seed2", type=int, default=456, help="Seed for HLT view #2")

    # Consistency weights (prob + embedding)
    parser.add_argument("--lambda_prob", type=float, default=1.0, help="Weight for prob symmetric-KL consistency")
    parser.add_argument("--lambda_emb", type=float, default=0.25, help="Weight for embedding cosine consistency")
    parser.add_argument("--rampup_frac", type=float, default=0.2, help="Ramp-up fraction of epochs for consistency weights")

    # Confidence weighting shape
    parser.add_argument("--conf_power", type=float, default=1.0, help="Power applied to confidence weights")
    parser.add_argument("--conf_min", type=float, default=0.0, help="Minimum confidence weight clamp (0 disables)")

    # Unsupervised warmup
    parser.add_argument("--attention_epoch", type=int, default=0, help="Train only on consistency losses (no supervised BCE) for first N epochs (0 disables)")

    # Checkpoint loading
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Load pre-trained teacher and skip teacher training")
    parser.add_argument("--skip_save_models", action="store_true", help="Do not save model weights")

    args = parser.parse_args()

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device:  {device}")
    print(f"Save dir:{save_dir}")

    # ------------------- Load dataset via utils.load_from_files ------------------- #
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

    # Convert -> constituents_raw: [pt, eta, phi, E]
    eta = all_data[:, :, ETA_IDX].astype(np.float32)
    phi = all_data[:, :, PHI_IDX].astype(np.float32)
    pt  = all_data[:, :, PT_IDX].astype(np.float32)

    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    constituents_raw = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    print(f"Avg particles per jet (raw mask): {mask_raw.sum(axis=1).mean():.1f}")

    # Offline threshold (professor style)
    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0
    print(f"Offline particles after {pt_threshold_off} threshold: {masks_off.sum():,}")
    print(f"Avg per jet (offline): {masks_off.sum(axis=1).mean():.1f}")

    # Two HLT realizations with different seeds
    print("\nApplying HLT effects for view #1...")
    constituents_hlt1, masks_hlt1 = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=args.hlt_seed1, verbose=True)

    print("\nApplying HLT effects for view #2...")
    constituents_hlt2, masks_hlt2 = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=args.hlt_seed2, verbose=True)

    print(f"\nAvg per jet: HLT1={masks_hlt1.sum(axis=1).mean():.1f}, HLT2={masks_hlt2.sum(axis=1).mean():.1f}")

    # Compute features
    print("\nComputing features...")
    features_off  = compute_features(constituents_off, masks_off)
    features_hlt1 = compute_features(constituents_hlt1, masks_hlt1)
    features_hlt2 = compute_features(constituents_hlt2, masks_hlt2)

    # Split indices (70/15/15 stratified)
    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])

    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Standardize using OFFLINE train stats
    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)

    features_off_std  = standardize(features_off,  masks_off,  feat_means, feat_stds)
    features_hlt1_std = standardize(features_hlt1, masks_hlt1, feat_means, feat_stds)
    features_hlt2_std = standardize(features_hlt2, masks_hlt2, feat_means, feat_stds)

    # Save test split artifacts
    test_data_dir = Path().cwd() / "test_split"
    test_data_dir.mkdir(exist_ok=True)

    np.savez(
        test_data_dir / "test_features_and_masks_twohlt.npz",
        idx_test=test_idx,
        labels=all_labels[test_idx],
        feat_off=features_off_std[test_idx],
        feat_hlt1=features_hlt1_std[test_idx],
        feat_hlt2=features_hlt2_std[test_idx],
        mask_off=masks_off[test_idx],
        mask_hlt1=masks_hlt1[test_idx],
        mask_hlt2=masks_hlt2[test_idx],
        jet_pt=all_pt[test_idx] if all_pt is not None else None,
        feat_means=feat_means,
        feat_stds=feat_stds,
        hlt_seed1=args.hlt_seed1,
        hlt_seed2=args.hlt_seed2,
    )
    print(f"Saved test artifacts to: {test_data_dir / 'test_features_and_masks_twohlt.npz'}")

    # ------------------- Build datasets/loaders ------------------- #
    BS = CONFIG["training"]["batch_size"]

    # Teacher (offline)
    train_off = JetDatasetSingle(features_off_std[train_idx], masks_off[train_idx], all_labels[train_idx])
    val_off   = JetDatasetSingle(features_off_std[val_idx],   masks_off[val_idx],   all_labels[val_idx])
    test_off  = JetDatasetSingle(features_off_std[test_idx],  masks_off[test_idx],  all_labels[test_idx])

    # HLT view1 single
    train_hlt1 = JetDatasetSingle(features_hlt1_std[train_idx], masks_hlt1[train_idx], all_labels[train_idx])
    val_hlt1   = JetDatasetSingle(features_hlt1_std[val_idx],   masks_hlt1[val_idx],   all_labels[val_idx])
    test_hlt1  = JetDatasetSingle(features_hlt1_std[test_idx],  masks_hlt1[test_idx],  all_labels[test_idx])

    # HLT view2 single
    train_hlt2 = JetDatasetSingle(features_hlt2_std[train_idx], masks_hlt2[train_idx], all_labels[train_idx])

    # Union dataset: concatenate HLT1 and HLT2 (ignore that they correspond)
    union_x = np.concatenate([features_hlt1_std[train_idx], features_hlt2_std[train_idx]], axis=0)
    union_m = np.concatenate([masks_hlt1[train_idx],       masks_hlt2[train_idx]],       axis=0)
    union_y = np.concatenate([all_labels[train_idx],       all_labels[train_idx]],       axis=0)
    train_union = JetDatasetSingle(union_x, union_m, union_y)

    # Paired dataset highlighting correspondence (for consistency)
    train_pair = JetDatasetPaired(
        features_hlt1_std[train_idx], masks_hlt1[train_idx],
        features_hlt2_std[train_idx], masks_hlt2[train_idx],
        all_labels[train_idx]
    )

    # Loaders
    train_off_loader   = DataLoader(train_off,   batch_size=BS, shuffle=True,  drop_last=True)
    val_off_loader     = DataLoader(val_off,     batch_size=BS, shuffle=False)
    test_off_loader    = DataLoader(test_off,    batch_size=BS, shuffle=False)

    train_hlt1_loader  = DataLoader(train_hlt1,  batch_size=BS, shuffle=True,  drop_last=True)
    val_hlt1_loader    = DataLoader(val_hlt1,    batch_size=BS, shuffle=False)
    test_hlt1_loader   = DataLoader(test_hlt1,   batch_size=BS, shuffle=False)

    train_union_loader = DataLoader(train_union, batch_size=BS, shuffle=True,  drop_last=True)
    train_pair_loader  = DataLoader(train_pair,  batch_size=BS, shuffle=True,  drop_last=True)

    # Paths
    teacher_path = save_dir / "teacher.pt"
    baseline_path = save_dir / "baseline_hlt1.pt"
    union_path = save_dir / "union_hlt12.pt"
    cons_path = save_dir / "consistency_hlt12.pt"

    # ------------------- Train models ------------------- #
    print("\n" + "=" * 80)
    print("STEP 1: TEACHER (Offline)")
    print("=" * 80)

    teacher = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)

    if args.teacher_checkpoint is not None:
        print(f"Loading teacher checkpoint: {args.teacher_checkpoint}")
        ckpt = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(ckpt["model"])
        best_teacher_auc = float(ckpt.get("auc", 0.0))
        print(f"Loaded teacher (stored val AUC={best_teacher_auc:.4f})")
    else:
        best_teacher_auc, _ = fit_model_standard(
            teacher, train_off_loader, val_off_loader, device, CONFIG,
            desc="Teacher",
            save_path=teacher_path,
            skip_save=args.skip_save_models,
        )

    print("\n" + "=" * 80)
    print("STEP 2: BASELINE (HLT view #1 only)")
    print("=" * 80)
    baseline = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_baseline_auc, _ = fit_model_standard(
        baseline, train_hlt1_loader, val_hlt1_loader, device, CONFIG,
        desc="Baseline-HLT1",
        save_path=baseline_path,
        skip_save=args.skip_save_models,
    )

    print("\n" + "=" * 80)
    print("STEP 3: UNION (HLT1 + HLT2 concatenated, no pairing, no consistency)")
    print("=" * 80)
    union = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_union_auc, _ = fit_model_standard(
        union, train_union_loader, val_hlt1_loader, device, CONFIG,
        desc="Union-HLT12",
        save_path=union_path,
        skip_save=args.skip_save_models,
    )

    print("\n" + "=" * 80)
    print("STEP 4: CONSISTENCY (paired HLT1 vs HLT2)")
    print("=" * 80)
    cons = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_cons_auc, _ = fit_model_consistency(
        cons, train_pair_loader, val_hlt1_loader, device, CONFIG,
        desc="Consistency-HLT12",
        save_path=cons_path,
        skip_save=args.skip_save_models,
        lambda_prob=args.lambda_prob,
        lambda_emb=args.lambda_emb,
        rampup_frac=args.rampup_frac,
        conf_power=args.conf_power,
        conf_min=args.conf_min,
        attention_epoch=args.attention_epoch,
    )

    # ------------------- Final evaluation on test ------------------- #
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    auc_teacher, preds_teacher, labs_off = evaluate_auc(teacher, test_off_loader, device)
    auc_baseline, preds_baseline, labs_hlt = evaluate_auc(baseline, test_hlt1_loader, device)
    auc_union, preds_union, _ = evaluate_auc(union, test_hlt1_loader, device)
    auc_cons, preds_cons, _ = evaluate_auc(cons, test_hlt1_loader, device)

    print(f"\n{'Model':<40} {'AUC':>10}")
    print("-" * 55)
    print(f"{'Teacher (Offline test)':<40} {auc_teacher:>10.4f}")
    print(f"{'Baseline (HLT1 test)':<40} {auc_baseline:>10.4f}")
    print(f"{'Union (HLT1 test)':<40} {auc_union:>10.4f}")
    print(f"{'Consistency (HLT1 test)':<40} {auc_cons:>10.4f}")
    print("-" * 55)

    # Background rejection @ 50% TPR (HLT-tested)
    fpr_b, tpr_b, _ = roc_curve(labs_hlt, preds_baseline)
    fpr_u, tpr_u, _ = roc_curve(labs_hlt, preds_union)
    fpr_c, tpr_c, _ = roc_curve(labs_hlt, preds_cons)

    wp = 0.5
    idx_b = np.argmax(tpr_b >= wp)
    idx_u = np.argmax(tpr_u >= wp)
    idx_c = np.argmax(tpr_c >= wp)

    br_b = 1.0 / fpr_b[idx_b] if fpr_b[idx_b] > 0 else 0.0
    br_u = 1.0 / fpr_u[idx_u] if fpr_u[idx_u] > 0 else 0.0
    br_c = 1.0 / fpr_c[idx_c] if fpr_c[idx_c] > 0 else 0.0

    print(f"\nBackground Rejection @ {wp*100:.0f}% signal efficiency (HLT-tested, view #1):")
    print(f"  Baseline:    {br_b:.2f}")
    print(f"  Union:       {br_u:.2f}")
    print(f"  Consistency: {br_c:.2f}")

    # ROC curves (axes: TPR on x, FPR on y)
    fpr_t, tpr_t, _ = roc_curve(labs_off, preds_teacher)

    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-",  label=f"Teacher OFF (AUC={auc_teacher:.3f})", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline HLT1 (AUC={auc_baseline:.3f})", linewidth=2)
    plt.plot(tpr_u, fpr_u, "-.", label=f"Union HLT12 (AUC={auc_union:.3f})", linewidth=2)
    plt.plot(tpr_c, fpr_c, ":",  label=f"Consistency HLT12 (AUC={auc_cons:.3f})", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=11, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300)
    plt.close()

    # Save results
    np.savez(
        save_dir / "results.npz",
        labs_off=labs_off,
        preds_teacher=preds_teacher,
        auc_teacher=auc_teacher,

        labs_hlt=labs_hlt,
        preds_baseline=preds_baseline,
        preds_union=preds_union,
        preds_consistency=preds_cons,
        auc_baseline=auc_baseline,
        auc_union=auc_union,
        auc_consistency=auc_cons,

        fpr_teacher=fpr_t, tpr_teacher=tpr_t,
        fpr_baseline=fpr_b, tpr_baseline=tpr_b,
        fpr_union=fpr_u, tpr_union=tpr_u,
        fpr_consistency=fpr_c, tpr_consistency=tpr_c,

        br_baseline=br_b,
        br_union=br_u,
        br_consistency=br_c,

        hlt_seed1=args.hlt_seed1,
        hlt_seed2=args.hlt_seed2,
        lambda_prob=args.lambda_prob,
        lambda_emb=args.lambda_emb,
        rampup_frac=args.rampup_frac,
        conf_power=args.conf_power,
        conf_min=args.conf_min,
    )

    # Append to run summary file
    summary_file = Path(args.save_dir) / "run_summaries.txt"
    with open(summary_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Run: {args.run_name}\n")
        f.write(f"HLT seeds: seed1={args.hlt_seed1}, seed2={args.hlt_seed2}\n")
        f.write(f"Consistency: lambda_prob={args.lambda_prob}, lambda_emb={args.lambda_emb}, rampup_frac={args.rampup_frac}\n")
        f.write(f"Confidence: conf_power={args.conf_power}, conf_min={args.conf_min}\n")
        f.write(f"AUC Teacher (OFF test): {auc_teacher:.4f}\n")
        f.write(f"AUC Baseline (HLT1 test): {auc_baseline:.4f} | BR@50%: {br_b:.2f}\n")
        f.write(f"AUC Union (HLT1 test):    {auc_union:.4f} | BR@50%: {br_u:.2f}\n")
        f.write(f"AUC Consistency (HLT1):   {auc_cons:.4f} | BR@50%: {br_c:.2f}\n")
        f.write(f"Saved: {save_dir / 'results.npz'} and {save_dir / 'results.png'}\n")

    print(f"\nSaved results to: {save_dir / 'results.npz'} and {save_dir / 'results.png'}")
    print(f"Logged to: {summary_file}")


if __name__ == "__main__":
    main()

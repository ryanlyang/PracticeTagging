
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer Suite: Offline Teacher + Multi-View HLT Augmentation + Consistency + KD

What this script trains (6 models):
  1) Teacher (OFFLINE features)               -> evaluated on OFFLINE test
  2) Baseline (HLT view #1 only)             -> evaluated on HLT test (view #1)
  3) Union (HLT views 1..N concatenated, no pairing)
                                             -> evaluated on HLT test (view #1)
  4) Consistency (paired HLT views 1..N):
       - supervised BCE on each view
       - confidence-weighted symmetric KL on probs for all view pairs
       - confidence-weighted cosine embedding alignment on pooled embeddings for all view pairs
                                             -> evaluated on HLT test (view #1)
  5) Student KD (HLT view #1 + teacher distillation)
                                             -> evaluated on HLT test (view #1)
  6) Consistency + KD (paired HLT views 1..N with teacher distillation)
                                             -> evaluated on HLT test (view #1)

Detailed overview:
  This script builds multiple HLT "views" by applying randomized HLT effects to the same
  offline jet constituents. Each view is a different noisy realization of the same jet.
  The models are trained as follows:

  - Teacher (offline): trained on offline features only and evaluated on the offline test split.
    This is the reference model for distillation.

  - Baseline (HLT1): trained only on HLT view #1, evaluated on HLT view #1.
    This measures the raw HLT performance without any multi-view or KD help.

  - Union: concatenates all HLT views into one large dataset, ignoring that they correspond
    to the same jets. This tests whether more HLT statistics alone helps.

  - Consistency: uses paired HLT views from the same jet. It applies supervised BCE on each
    view plus consistency penalties across all view pairs:
      * symmetric KL on prediction probabilities
      * cosine alignment on pooled embeddings
    Consistency encourages the model to be stable under HLT noise (same jet -> same decision).

  - Student KD: trains on HLT view #1 with knowledge distillation from the offline teacher.
    The KD loss includes optional attention KL, embedding cosine alignment, and InfoNCE,
    plus confidence-weighted logit distillation.

  - Consistency + KD: combines both ideas. It uses all HLT views with pairwise consistency
    losses and also distills from the offline teacher into every HLT view. This model is
    the most constrained: it learns from labels, agrees across HLT views, and matches the
    offline teacher's behavior.

  All HLT-based models (baseline/union/consistency/KD) are evaluated on HLT view #1 so that
  performance comparisons stay on a consistent HLT test distribution.

Key idea:
  - Generate N independently randomized HLT realizations per jet (different seeds).
  - "Union" ignores pairing (just concatenates all views).
  - "Consistency" uses pairing and penalizes disagreement across all view pairs.

Notes:
  - Uses OFFLINE training stats (means/stds computed from OFFLINE train split) to standardize:
      offline, all HLT views
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
    "kd": {
        "temperature": 7.0,
        "alpha_kd": 0.5,      # mixes hard vs KD
        "alpha_attn": 0.05,   # masked KL on attention
        "alpha_rep": 0.10,    # cosine alignment on pooled embedding
        "alpha_nce": 0.10,    # InfoNCE on pooled embedding
        "tau_nce": 0.10,      # InfoNCE temperature
        "use_conf_weighted_kd": True,
    },
}

MAX_HLT_VIEWS = 8


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


class JetDatasetMulti(Dataset):
    """Multi-view dataset: {x, mask, label} where x/mask are (n_views, M, F/M)"""
    def __init__(self, x_views, m_views, y):
        # x_views/m_views can be lists of arrays or stacked arrays with shape (V, N, ...)
        if isinstance(x_views, list):
            x_views = np.stack(x_views, axis=0)
        if isinstance(m_views, list):
            m_views = np.stack(m_views, axis=0)
        self.x = torch.tensor(x_views, dtype=torch.float32)
        self.mask = torch.tensor(m_views, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.x[:, i], "mask": self.mask[:, i], "label": self.y[i]}


class JetDatasetOffHlt(Dataset):
    """Offline + single HLT view dataset for KD: {off, hlt, mask_off, mask_hlt, label}"""
    def __init__(self, off, off_mask, hlt, hlt_mask, y):
        self.off = torch.tensor(off, dtype=torch.float32)
        self.off_mask = torch.tensor(off_mask, dtype=torch.bool)
        self.hlt = torch.tensor(hlt, dtype=torch.float32)
        self.hlt_mask = torch.tensor(hlt_mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "off": self.off[i],
            "hlt": self.hlt[i],
            "mask_off": self.off_mask[i],
            "mask_hlt": self.hlt_mask[i],
            "label": self.y[i],
        }


class JetDatasetMultiOff(Dataset):
    """Offline + multi-view HLT dataset for KD+consistency."""
    def __init__(self, x_views, m_views, off, off_mask, y):
        if isinstance(x_views, list):
            x_views = np.stack(x_views, axis=0)
        if isinstance(m_views, list):
            m_views = np.stack(m_views, axis=0)
        self.x = torch.tensor(x_views, dtype=torch.float32)
        self.mask = torch.tensor(m_views, dtype=torch.bool)
        self.off = torch.tensor(off, dtype=torch.float32)
        self.off_mask = torch.tensor(off_mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "x": self.x[:, i],
            "mask": self.mask[:, i],
            "off": self.off[i],
            "mask_off": self.off_mask[i],
            "label": self.y[i],
        }


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

    def forward(self, x, mask, return_embedding=False, return_attention=False):
        """
        x: (B, M, input_dim)
        mask: (B, M) True for valid particles
        """
        B, M, _ = x.shape

        h = x.reshape(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.reshape(B, M, -1)

        h = self.transformer(h, src_key_padding_mask=~mask)

        query = self.pool_query.expand(B, -1, -1)
        if return_attention:
            pooled, attn_weights = self.pool_attn(
                query, h, h,
                key_padding_mask=~mask,
                need_weights=True,
                average_attn_weights=True,
            )
        else:
            pooled, _ = self.pool_attn(
                query, h, h,
                key_padding_mask=~mask,
                need_weights=False,
            )
        z = self.norm(pooled.squeeze(1))          # pooled embedding
        logits = self.classifier(z)               # (B, 1)

        if return_attention and return_embedding:
            return logits, attn_weights.squeeze(1), z
        if return_attention:
            return logits, attn_weights.squeeze(1)
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


def train_one_epoch_kd(student, teacher, loader, opt, device, cfg, temperature, alpha_kd):
    student.train()
    teacher.eval()

    a_attn = cfg["kd"]["alpha_attn"]
    a_rep = cfg["kd"]["alpha_rep"]
    a_nce = cfg["kd"]["alpha_nce"]
    tau_nce = cfg["kd"]["tau_nce"]
    use_conf = cfg["kd"]["use_conf_weighted_kd"]

    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        x_off = batch["off"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            t_logits, t_attn, t_z = teacher(x_off, m_off, return_attention=True, return_embedding=True)
            t_logits = t_logits.squeeze(1)

        opt.zero_grad()
        s_logits, s_attn, s_z = student(x_hlt, m_hlt, return_attention=True, return_embedding=True)
        s_logits = s_logits.squeeze(1)

        loss_hard = F.binary_cross_entropy_with_logits(s_logits, y)
        if use_conf:
            loss_kd = kd_loss_conf_weighted(s_logits, t_logits, temperature)
        else:
            loss_kd = kd_loss_basic(s_logits, t_logits, temperature)

        loss_rep = rep_loss_cosine(s_z, t_z.detach()) if a_rep > 0 else torch.zeros((), device=device)
        loss_nce = info_nce_loss(s_z, t_z.detach(), tau=tau_nce) if a_nce > 0 else torch.zeros((), device=device)
        loss_attn = attn_kl_loss(s_attn, t_attn.detach(), m_hlt, m_off) if a_attn > 0 else torch.zeros((), device=device)

        loss = (1.0 - alpha_kd) * loss_hard + alpha_kd * loss_kd + a_attn * loss_attn + a_rep * loss_rep + a_nce * loss_nce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.append(torch.sigmoid(s_logits).detach().cpu().numpy())
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


# ----------------------------- KD losses + schedules ----------------------------- #
def kd_loss_basic(student_logits, teacher_logits, T):
    s_soft = torch.sigmoid(student_logits / T)
    t_soft = torch.sigmoid(teacher_logits / T)
    return F.binary_cross_entropy(s_soft, t_soft) * (T ** 2)


def kd_loss_conf_weighted(student_logits, teacher_logits, T):
    """
    Confidence-weighted binary KD: higher weight when teacher is far from 0.5.
    """
    s_soft = torch.sigmoid(student_logits / T)
    t_soft = torch.sigmoid(teacher_logits / T)
    w = (torch.abs(torch.sigmoid(teacher_logits) - 0.5) * 2.0).detach()
    per = F.binary_cross_entropy(s_soft, t_soft, reduction="none")
    return (w * per).mean() * (T ** 2)


def rep_loss_cosine(s_z, t_z):
    s = F.normalize(s_z, dim=1)
    t = F.normalize(t_z, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


def info_nce_loss(s_z, t_z, tau=0.1):
    s = F.normalize(s_z, dim=1)
    t = F.normalize(t_z, dim=1)
    logits_st = (s @ t.t()) / tau
    logits_ts = (t @ s.t()) / tau
    labels = torch.arange(s.size(0), device=s.device)
    loss_st = F.cross_entropy(logits_st, labels)
    loss_ts = F.cross_entropy(logits_ts, labels)
    return 0.5 * (loss_st + loss_ts)


def attn_kl_loss(s_attn, t_attn, s_mask, t_mask, eps=1e-8):
    """
    Masked KL on normalized attention distributions over joint-valid tokens.
    """
    joint = (s_mask & t_mask).float()
    denom_s = (s_attn * joint).sum(dim=1, keepdim=True)
    denom_t = (t_attn * joint).sum(dim=1, keepdim=True)
    valid_sample = (denom_s.squeeze(1) > eps) & (denom_t.squeeze(1) > eps)
    if valid_sample.sum().item() == 0:
        return torch.zeros((), device=s_attn.device)

    s = (s_attn * joint) / (denom_s + eps)
    t = (t_attn * joint) / (denom_t + eps)
    s = torch.clamp(s, eps, 1.0)
    t = torch.clamp(t, eps, 1.0)
    kl = (t * (torch.log(t) - torch.log(s))).sum(dim=1)
    return kl[valid_sample].mean()


def get_temperature_schedule(epoch, total_epochs, T_init, T_final):
    if T_final is None:
        return T_init
    return T_init + (T_final - T_init) * (epoch / max(total_epochs - 1, 1))


def get_alpha_schedule(epoch, total_epochs, alpha_init, alpha_final):
    if alpha_final is None:
        return alpha_init
    return alpha_init + (alpha_final - alpha_init) * (epoch / max(total_epochs - 1, 1))


# ----------------------------- Training: paired consistency ----------------------------- #
def train_one_epoch_consistency(model, loader, opt, device, lam_prob, lam_emb, ramp_mult, conf_power, conf_min,
                                epoch=0, attention_epoch=0):
    model.train()
    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x = batch["x"].to(device)       # (B, V, M, F)
        m = batch["mask"].to(device)    # (B, V, M)
        y = batch["label"].to(device)

        opt.zero_grad()

        n_views = x.shape[1]
        logits_list = []
        emb_list = []

        for v in range(n_views):
            logits_v, z_v = model(x[:, v], m[:, v], return_embedding=True)
            logits_list.append(logits_v.squeeze(1))
            emb_list.append(z_v)

        # supervised on ALL views (skip during unsupervised warmup)
        if epoch >= attention_epoch:
            loss_sup = 0.0
            for logits_v in logits_list:
                loss_sup += F.binary_cross_entropy_with_logits(logits_v, y)
            loss_sup = loss_sup / max(n_views, 1)
        else:
            loss_sup = torch.tensor(0.0, device=device)  # Unsupervised warmup: train only on consistency losses

        # probabilities
        probs_list = [torch.sigmoid(l) for l in logits_list]

        # consistency across all view pairs
        pair_count = 0
        loss_prob = torch.tensor(0.0, device=device)
        loss_emb = torch.tensor(0.0, device=device)
        for i in range(n_views):
            for j in range(i + 1, n_views):
                w = confidence_weight(probs_list[i], probs_list[j], power=conf_power, conf_min=conf_min)
                l_kl = symmetric_kl_bernoulli(probs_list[i], probs_list[j])  # (B,)
                l_cos = cosine_embed_loss(emb_list[i], emb_list[j])          # (B,)
                loss_prob += (w * l_kl).mean()
                loss_emb += (w * l_cos).mean()
                pair_count += 1

        if pair_count > 0:
            loss_prob = loss_prob / pair_count
            loss_emb = loss_emb / pair_count

        loss = loss_sup + (lam_prob * ramp_mult) * loss_prob + (lam_emb * ramp_mult) * loss_emb

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)

        # for train AUC reporting, just use view #1 (index 0)
        preds.append(probs_list[0].detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return total_loss / len(labs), roc_auc_score(labs, preds)


def train_one_epoch_consistency_kd(student, teacher, loader, opt, device, lam_prob, lam_emb, ramp_mult,
                                   conf_power, conf_min, cfg, temperature, alpha_kd,
                                   epoch=0, attention_epoch=0):
    student.train()
    teacher.eval()

    a_attn = cfg["kd"]["alpha_attn"]
    a_rep = cfg["kd"]["alpha_rep"]
    a_nce = cfg["kd"]["alpha_nce"]
    tau_nce = cfg["kd"]["tau_nce"]
    use_conf = cfg["kd"]["use_conf_weighted_kd"]

    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x = batch["x"].to(device)       # (B, V, M, F)
        m = batch["mask"].to(device)    # (B, V, M)
        x_off = batch["off"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            t_logits, t_attn, t_z = teacher(x_off, m_off, return_attention=True, return_embedding=True)
            t_logits = t_logits.squeeze(1)

        opt.zero_grad()

        n_views = x.shape[1]
        logits_list = []
        emb_list = []
        attn_list = []

        for v in range(n_views):
            s_logits, s_attn, s_z = student(x[:, v], m[:, v], return_attention=True, return_embedding=True)
            logits_list.append(s_logits.squeeze(1))
            emb_list.append(s_z)
            attn_list.append(s_attn)

        # supervised + KD on ALL views (skip during unsupervised warmup)
        loss_kd = torch.tensor(0.0, device=device)
        if epoch >= attention_epoch:
            for v in range(n_views):
                s_logits = logits_list[v]
                s_z = emb_list[v]
                s_attn = attn_list[v]
                s_mask = m[:, v]

                loss_hard = F.binary_cross_entropy_with_logits(s_logits, y)
                if use_conf:
                    kd_term = kd_loss_conf_weighted(s_logits, t_logits, temperature)
                else:
                    kd_term = kd_loss_basic(s_logits, t_logits, temperature)

                loss_rep = rep_loss_cosine(s_z, t_z.detach()) if a_rep > 0 else torch.zeros((), device=device)
                loss_nce = info_nce_loss(s_z, t_z.detach(), tau=tau_nce) if a_nce > 0 else torch.zeros((), device=device)
                loss_attn = attn_kl_loss(s_attn, t_attn.detach(), s_mask, m_off) if a_attn > 0 else torch.zeros((), device=device)

                loss_kd += (1.0 - alpha_kd) * loss_hard + alpha_kd * kd_term + a_attn * loss_attn + a_rep * loss_rep + a_nce * loss_nce
            loss_kd = loss_kd / max(n_views, 1)

        # probabilities
        probs_list = [torch.sigmoid(l) for l in logits_list]

        # consistency across all view pairs
        pair_count = 0
        loss_prob = torch.tensor(0.0, device=device)
        loss_emb = torch.tensor(0.0, device=device)
        for i in range(n_views):
            for j in range(i + 1, n_views):
                w = confidence_weight(probs_list[i], probs_list[j], power=conf_power, conf_min=conf_min)
                l_kl = symmetric_kl_bernoulli(probs_list[i], probs_list[j])
                l_cos = cosine_embed_loss(emb_list[i], emb_list[j])
                loss_prob += (w * l_kl).mean()
                loss_emb += (w * l_cos).mean()
                pair_count += 1

        if pair_count > 0:
            loss_prob = loss_prob / pair_count
            loss_emb = loss_emb / pair_count

        loss = loss_kd + (lam_prob * ramp_mult) * loss_prob + (lam_emb * ramp_mult) * loss_emb

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.append(probs_list[0].detach().cpu().numpy())
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


def fit_model_kd(student, teacher, train_loader, val_loader, device, cfg, desc, save_path=None, skip_save=False,
                 temp_init=7.0, temp_final=None, alpha_init=0.5, alpha_final=None):
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    sch = get_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["training"]["epochs"])

    best_auc, best_state, no_improve = 0.0, None, 0
    history = []

    for ep in tqdm(range(cfg["training"]["epochs"]), desc=desc):
        temp = get_temperature_schedule(ep, cfg["training"]["epochs"], temp_init, temp_final)
        alpha = get_alpha_schedule(ep, cfg["training"]["epochs"], alpha_init, alpha_final)

        tr_loss, tr_auc = train_one_epoch_kd(student, teacher, train_loader, opt, device, cfg, temp, alpha)
        va_auc, _, _ = evaluate_auc(student, val_loader, device)
        sch.step()

        history.append((ep + 1, tr_loss, tr_auc, va_auc, temp, alpha))

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_auc:.4f}, T={temp:.2f}, alpha={alpha:.2f}")

        if no_improve >= cfg["training"]["patience"]:
            print(f"Early stopping at epoch {ep+1} (best val AUC={best_auc:.4f})")
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    if (save_path is not None) and (not skip_save):
        torch.save({"model": student.state_dict(), "auc": best_auc, "history": history}, save_path)
        print(f"Saved: {save_path} (best val AUC={best_auc:.4f})")
    else:
        print(f"Skip-save={skip_save}. Best val AUC={best_auc:.4f}")

    return best_auc, history


def fit_model_consistency_kd(student, teacher, train_loader, val_loader, device, cfg, desc, save_path=None, skip_save=False,
                             lambda_prob=1.0, lambda_emb=0.25, rampup_frac=0.2, conf_power=1.0, conf_min=0.0,
                             attention_epoch=0, temp_init=7.0, temp_final=None, alpha_init=0.5, alpha_final=None):
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    sch = get_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["training"]["epochs"])

    best_auc, best_state, no_improve = 0.0, None, 0
    history = []

    if attention_epoch > 0:
        print(f"[Unsupervised warmup] Training ONLY on consistency losses for first {attention_epoch} epochs")

    for ep in tqdm(range(cfg["training"]["epochs"]), desc=desc):
        ramp_mult = rampup_weight(ep, cfg["training"]["epochs"], rampup_frac)
        temp = get_temperature_schedule(ep, cfg["training"]["epochs"], temp_init, temp_final)
        alpha = get_alpha_schedule(ep, cfg["training"]["epochs"], alpha_init, alpha_final)

        tr_loss, tr_auc = train_one_epoch_consistency_kd(
            student, teacher, train_loader, opt, device,
            lam_prob=lambda_prob,
            lam_emb=lambda_emb,
            ramp_mult=ramp_mult,
            conf_power=conf_power,
            conf_min=conf_min,
            cfg=cfg,
            temperature=temp,
            alpha_kd=alpha,
            epoch=ep,
            attention_epoch=attention_epoch,
        )
        va_auc, _, _ = evaluate_auc(student, val_loader, device)
        sch.step()

        history.append((ep + 1, tr_loss, tr_auc, va_auc, ramp_mult, temp, alpha))

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            warmup_tag = " [warmup]" if ep < attention_epoch else ""
            print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_auc:.4f}, ramp={ramp_mult:.2f}, T={temp:.2f}, alpha={alpha:.2f}{warmup_tag}")

        if attention_epoch > 0 and ep + 1 == attention_epoch:
            print(f"[Warmup complete] Switching to full training (supervised + KD + consistency) at epoch {ep+1}")

        if no_improve >= cfg["training"]["patience"]:
            print(f"Early stopping at epoch {ep+1} (best val AUC={best_auc:.4f})")
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    if (save_path is not None) and (not skip_save):
        torch.save({"model": student.state_dict(), "auc": best_auc, "history": history}, save_path)
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

    # HLT views and seeds
    parser.add_argument("--n_hlt_views", type=int, default=2,
                        help=f"Number of HLT views to generate (2-{MAX_HLT_VIEWS})")
    parser.add_argument("--hlt_seeds", type=str, default=None,
                        help="Comma-separated seeds for each HLT view (overrides --hlt_seed1/2)")
    parser.add_argument("--hlt_seed_base", type=int, default=123,
                        help="Base seed used to auto-generate HLT seeds when --hlt_seeds is not set")
    parser.add_argument("--hlt_seed_step", type=int, default=333,
                        help="Seed step used with --hlt_seed_base for auto-generated seeds")
    parser.add_argument("--hlt_seed1", type=int, default=123,
                        help="Seed for HLT view #1 (used when n_hlt_views=2 and --hlt_seeds is not set)")
    parser.add_argument("--hlt_seed2", type=int, default=456,
                        help="Seed for HLT view #2 (used when n_hlt_views=2 and --hlt_seeds is not set)")

    # Consistency weights (prob + embedding)
    parser.add_argument("--lambda_prob", type=float, default=1.0, help="Weight for prob symmetric-KL consistency")
    parser.add_argument("--lambda_emb", type=float, default=0.25, help="Weight for embedding cosine consistency")
    parser.add_argument("--rampup_frac", type=float, default=0.2, help="Ramp-up fraction of epochs for consistency weights")

    # Confidence weighting shape
    parser.add_argument("--conf_power", type=float, default=1.0, help="Power applied to confidence weights")
    parser.add_argument("--conf_min", type=float, default=0.0, help="Minimum confidence weight clamp (0 disables)")

    # Unsupervised warmup
    parser.add_argument("--attention_epoch", type=int, default=0, help="Train only on consistency losses (no supervised BCE) for first N epochs (0 disables)")

    # KD schedule and weights
    parser.add_argument("--temp_init", type=float, default=CONFIG["kd"]["temperature"], help="Initial KD temperature")
    parser.add_argument("--temp_final", type=float, default=None, help="Final KD temperature (if annealing)")
    parser.add_argument("--alpha_init", type=float, default=CONFIG["kd"]["alpha_kd"], help="Initial KD alpha mix")
    parser.add_argument("--alpha_final", type=float, default=None, help="Final KD alpha mix (if scheduling)")
    parser.add_argument("--alpha_attn", type=float, default=CONFIG["kd"]["alpha_attn"], help="Weight for attention KL distillation")
    parser.add_argument("--alpha_rep", type=float, default=CONFIG["kd"]["alpha_rep"], help="Weight for embedding cosine alignment")
    parser.add_argument("--alpha_nce", type=float, default=CONFIG["kd"]["alpha_nce"], help="Weight for InfoNCE paired contrastive loss")
    parser.add_argument("--tau_nce", type=float, default=CONFIG["kd"]["tau_nce"], help="InfoNCE temperature")
    parser.add_argument("--no_conf_kd", action="store_true", help="Disable confidence-weighted KD (use plain KD)")

    # Checkpoint loading
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Load pre-trained teacher and skip teacher training")
    parser.add_argument("--skip_save_models", action="store_true", help="Do not save model weights")

    args = parser.parse_args()

    if args.n_hlt_views < 2:
        raise ValueError("n_hlt_views must be >= 2 for multi-view consistency training.")
    if args.n_hlt_views > MAX_HLT_VIEWS:
        raise ValueError(f"n_hlt_views must be <= {MAX_HLT_VIEWS}.")

    if args.hlt_seeds is not None:
        hlt_seeds = [int(s.strip()) for s in args.hlt_seeds.split(",") if s.strip()]
        if len(hlt_seeds) != args.n_hlt_views:
            raise ValueError("Number of --hlt_seeds must match --n_hlt_views.")
    else:
        if args.n_hlt_views == 2:
            hlt_seeds = [args.hlt_seed1, args.hlt_seed2]
        else:
            hlt_seeds = [args.hlt_seed_base + i * args.hlt_seed_step for i in range(args.n_hlt_views)]

    CONFIG["kd"]["alpha_attn"] = float(args.alpha_attn)
    CONFIG["kd"]["alpha_rep"] = float(args.alpha_rep)
    CONFIG["kd"]["alpha_nce"] = float(args.alpha_nce)
    CONFIG["kd"]["tau_nce"] = float(args.tau_nce)
    CONFIG["kd"]["use_conf_weighted_kd"] = (not args.no_conf_kd)

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device:  {device}")
    print(f"Save dir:{save_dir}")
    print("KD settings:")
    print(f"  conf_weighted_kd={CONFIG['kd']['use_conf_weighted_kd']}, alpha_attn={CONFIG['kd']['alpha_attn']}, alpha_rep={CONFIG['kd']['alpha_rep']}, alpha_nce={CONFIG['kd']['alpha_nce']}, tau_nce={CONFIG['kd']['tau_nce']}")

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

    # N HLT realizations with different seeds
    print(f"\nGenerating {args.n_hlt_views} HLT views with seeds: {hlt_seeds}")
    constituents_hlt_list = []
    masks_hlt_list = []
    for i, seed in enumerate(hlt_seeds, start=1):
        print(f"\nApplying HLT effects for view #{i} (seed={seed})...")
        const_hlt, mask_hlt = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=seed, verbose=True)
        constituents_hlt_list.append(const_hlt)
        masks_hlt_list.append(mask_hlt)

    avg_parts = [m.sum(axis=1).mean() for m in masks_hlt_list]
    avg_str = ", ".join([f"HLT{i+1}={avg_parts[i]:.1f}" for i in range(len(avg_parts))])
    print(f"\nAvg per jet: {avg_str}")

    # Compute features
    print("\nComputing features...")
    features_off = compute_features(constituents_off, masks_off)
    features_hlt_list = [
        compute_features(constituents_hlt_list[i], masks_hlt_list[i])
        for i in range(args.n_hlt_views)
    ]

    # Split indices (70/15/15 stratified)
    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])

    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Standardize using OFFLINE train stats
    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)

    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std_list = [
        standardize(features_hlt_list[i], masks_hlt_list[i], feat_means, feat_stds)
        for i in range(args.n_hlt_views)
    ]

    # Save test split artifacts
    test_data_dir = Path().cwd() / "test_split"
    test_data_dir.mkdir(exist_ok=True)

    feat_hlt_views = np.stack(
        [features_hlt_std_list[i][test_idx] for i in range(args.n_hlt_views)],
        axis=0,
    )
    mask_hlt_views = np.stack(
        [masks_hlt_list[i][test_idx] for i in range(args.n_hlt_views)],
        axis=0,
    )

    np.savez(
        test_data_dir / "test_features_and_masks_twohlt.npz",
        idx_test=test_idx,
        labels=all_labels[test_idx],
        feat_off=features_off_std[test_idx],
        feat_hlt1=features_hlt_std_list[0][test_idx],
        feat_hlt2=features_hlt_std_list[1][test_idx],
        feat_hlt_views=feat_hlt_views,
        mask_off=masks_off[test_idx],
        mask_hlt1=masks_hlt_list[0][test_idx],
        mask_hlt2=masks_hlt_list[1][test_idx],
        mask_hlt_views=mask_hlt_views,
        jet_pt=all_pt[test_idx] if all_pt is not None else None,
        feat_means=feat_means,
        feat_stds=feat_stds,
        n_hlt_views=args.n_hlt_views,
        hlt_seeds=np.array(hlt_seeds, dtype=np.int64),
    )
    print(f"Saved test artifacts to: {test_data_dir / 'test_features_and_masks_twohlt.npz'}")

    # ------------------- Build datasets/loaders ------------------- #
    BS = CONFIG["training"]["batch_size"]

    # Teacher (offline)
    train_off = JetDatasetSingle(features_off_std[train_idx], masks_off[train_idx], all_labels[train_idx])
    val_off   = JetDatasetSingle(features_off_std[val_idx],   masks_off[val_idx],   all_labels[val_idx])
    test_off  = JetDatasetSingle(features_off_std[test_idx],  masks_off[test_idx],  all_labels[test_idx])

    # HLT view #1 (index 0)
    features_hlt1_std = features_hlt_std_list[0]
    masks_hlt1 = masks_hlt_list[0]

    # HLT view1 single
    train_hlt1 = JetDatasetSingle(features_hlt1_std[train_idx], masks_hlt1[train_idx], all_labels[train_idx])
    val_hlt1   = JetDatasetSingle(features_hlt1_std[val_idx],   masks_hlt1[val_idx],   all_labels[val_idx])
    test_hlt1  = JetDatasetSingle(features_hlt1_std[test_idx],  masks_hlt1[test_idx],  all_labels[test_idx])

    # Union dataset: concatenate all HLT views (ignore that they correspond)
    union_x = np.concatenate(
        [features_hlt_std_list[i][train_idx] for i in range(args.n_hlt_views)],
        axis=0,
    )
    union_m = np.concatenate(
        [masks_hlt_list[i][train_idx] for i in range(args.n_hlt_views)],
        axis=0,
    )
    union_y = np.concatenate([all_labels[train_idx] for _ in range(args.n_hlt_views)], axis=0)
    train_union = JetDatasetSingle(union_x, union_m, union_y)

    # Multi-view dataset highlighting correspondence (for consistency)
    train_multi = JetDatasetMulti(
        [features_hlt_std_list[i][train_idx] for i in range(args.n_hlt_views)],
        [masks_hlt_list[i][train_idx] for i in range(args.n_hlt_views)],
        all_labels[train_idx],
    )

    # Loaders
    train_off_loader   = DataLoader(train_off,   batch_size=BS, shuffle=True,  drop_last=True)
    val_off_loader     = DataLoader(val_off,     batch_size=BS, shuffle=False)
    test_off_loader    = DataLoader(test_off,    batch_size=BS, shuffle=False)

    train_hlt1_loader  = DataLoader(train_hlt1,  batch_size=BS, shuffle=True,  drop_last=True)
    val_hlt1_loader    = DataLoader(val_hlt1,    batch_size=BS, shuffle=False)
    test_hlt1_loader   = DataLoader(test_hlt1,   batch_size=BS, shuffle=False)

    train_union_loader = DataLoader(train_union, batch_size=BS, shuffle=True,  drop_last=True)
    train_multi_loader = DataLoader(train_multi, batch_size=BS, shuffle=True,  drop_last=True)

    # KD loaders (offline + HLT view #1)
    train_kd = JetDatasetOffHlt(
        features_off_std[train_idx], masks_off[train_idx],
        features_hlt1_std[train_idx], masks_hlt1[train_idx],
        all_labels[train_idx],
    )
    train_kd_loader = DataLoader(train_kd, batch_size=BS, shuffle=True, drop_last=True)

    # KD + consistency loader (offline + HLT multi-view)
    train_multi_off = JetDatasetMultiOff(
        [features_hlt_std_list[i][train_idx] for i in range(args.n_hlt_views)],
        [masks_hlt_list[i][train_idx] for i in range(args.n_hlt_views)],
        features_off_std[train_idx],
        masks_off[train_idx],
        all_labels[train_idx],
    )
    train_multi_off_loader = DataLoader(train_multi_off, batch_size=BS, shuffle=True, drop_last=True)

    # Paths
    teacher_path = save_dir / "teacher.pt"
    baseline_path = save_dir / "baseline_hlt1.pt"
    union_path = save_dir / f"union_hlt{args.n_hlt_views}.pt"
    cons_path = save_dir / f"consistency_hlt{args.n_hlt_views}.pt"
    student_kd_path = save_dir / "student_kd_hlt1.pt"
    cons_kd_path = save_dir / f"consistency_kd_hlt{args.n_hlt_views}.pt"

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

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

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
    print(f"STEP 3: UNION (HLT views 1..{args.n_hlt_views} concatenated, no pairing, no consistency)")
    print("=" * 80)
    union = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_union_auc, _ = fit_model_standard(
        union, train_union_loader, val_hlt1_loader, device, CONFIG,
        desc=f"Union-HLT{args.n_hlt_views}",
        save_path=union_path,
        skip_save=args.skip_save_models,
    )

    print("\n" + "=" * 80)
    print(f"STEP 4: CONSISTENCY (paired HLT views 1..{args.n_hlt_views})")
    print("=" * 80)
    cons = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_cons_auc, _ = fit_model_consistency(
        cons, train_multi_loader, val_hlt1_loader, device, CONFIG,
        desc=f"Consistency-HLT{args.n_hlt_views}",
        save_path=cons_path,
        skip_save=args.skip_save_models,
        lambda_prob=args.lambda_prob,
        lambda_emb=args.lambda_emb,
        rampup_frac=args.rampup_frac,
        conf_power=args.conf_power,
        conf_min=args.conf_min,
        attention_epoch=args.attention_epoch,
    )

    print("\n" + "=" * 80)
    print("STEP 5: STUDENT KD (HLT view #1 + teacher)")
    print("=" * 80)
    student_kd = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_student_kd_auc, _ = fit_model_kd(
        student_kd, teacher, train_kd_loader, val_hlt1_loader, device, CONFIG,
        desc="Student-KD-HLT1",
        save_path=student_kd_path,
        skip_save=args.skip_save_models,
        temp_init=args.temp_init,
        temp_final=args.temp_final,
        alpha_init=args.alpha_init,
        alpha_final=args.alpha_final,
    )

    print("\n" + "=" * 80)
    print(f"STEP 6: CONSISTENCY + KD (HLT views 1..{args.n_hlt_views} + teacher)")
    print("=" * 80)
    cons_kd = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_cons_kd_auc, _ = fit_model_consistency_kd(
        cons_kd, teacher, train_multi_off_loader, val_hlt1_loader, device, CONFIG,
        desc=f"Consistency-KD-HLT{args.n_hlt_views}",
        save_path=cons_kd_path,
        skip_save=args.skip_save_models,
        lambda_prob=args.lambda_prob,
        lambda_emb=args.lambda_emb,
        rampup_frac=args.rampup_frac,
        conf_power=args.conf_power,
        conf_min=args.conf_min,
        attention_epoch=args.attention_epoch,
        temp_init=args.temp_init,
        temp_final=args.temp_final,
        alpha_init=args.alpha_init,
        alpha_final=args.alpha_final,
    )

    # ------------------- Final evaluation on test ------------------- #
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    auc_teacher, preds_teacher, labs_off = evaluate_auc(teacher, test_off_loader, device)
    auc_baseline, preds_baseline, labs_hlt = evaluate_auc(baseline, test_hlt1_loader, device)
    auc_union, preds_union, _ = evaluate_auc(union, test_hlt1_loader, device)
    auc_cons, preds_cons, _ = evaluate_auc(cons, test_hlt1_loader, device)
    auc_student_kd, preds_student_kd, _ = evaluate_auc(student_kd, test_hlt1_loader, device)
    auc_cons_kd, preds_cons_kd, _ = evaluate_auc(cons_kd, test_hlt1_loader, device)

    print(f"\n{'Model':<40} {'AUC':>10}")
    print("-" * 55)
    print(f"{'Teacher (Offline test)':<40} {auc_teacher:>10.4f}")
    print(f"{'Baseline (HLT1 test)':<40} {auc_baseline:>10.4f}")
    print(f"{f'Union (HLT1 test, {args.n_hlt_views} views)':<40} {auc_union:>10.4f}")
    print(f"{f'Consistency (HLT1 test, {args.n_hlt_views} views)':<40} {auc_cons:>10.4f}")
    print(f"{'Student KD (HLT1 test)':<40} {auc_student_kd:>10.4f}")
    print(f"{f'Consistency+KD (HLT1 test, {args.n_hlt_views} views)':<40} {auc_cons_kd:>10.4f}")
    print("-" * 55)

    # Background rejection @ 50% TPR (HLT-tested)
    fpr_b, tpr_b, _ = roc_curve(labs_hlt, preds_baseline)
    fpr_u, tpr_u, _ = roc_curve(labs_hlt, preds_union)
    fpr_c, tpr_c, _ = roc_curve(labs_hlt, preds_cons)
    fpr_s, tpr_s, _ = roc_curve(labs_hlt, preds_student_kd)
    fpr_ck, tpr_ck, _ = roc_curve(labs_hlt, preds_cons_kd)

    wp = 0.5
    idx_b = np.argmax(tpr_b >= wp)
    idx_u = np.argmax(tpr_u >= wp)
    idx_c = np.argmax(tpr_c >= wp)
    idx_s = np.argmax(tpr_s >= wp)
    idx_ck = np.argmax(tpr_ck >= wp)

    br_b = 1.0 / fpr_b[idx_b] if fpr_b[idx_b] > 0 else 0.0
    br_u = 1.0 / fpr_u[idx_u] if fpr_u[idx_u] > 0 else 0.0
    br_c = 1.0 / fpr_c[idx_c] if fpr_c[idx_c] > 0 else 0.0
    br_s = 1.0 / fpr_s[idx_s] if fpr_s[idx_s] > 0 else 0.0
    br_ck = 1.0 / fpr_ck[idx_ck] if fpr_ck[idx_ck] > 0 else 0.0

    print(f"\nBackground Rejection @ {wp*100:.0f}% signal efficiency (HLT-tested, view #1):")
    print(f"  Baseline:    {br_b:.2f}")
    print(f"  Union:       {br_u:.2f}")
    print(f"  Consistency: {br_c:.2f}")
    print(f"  Student KD:  {br_s:.2f}")
    print(f"  Cons+KD:     {br_ck:.2f}")

    # ROC curves (axes: TPR on x, FPR on y)
    fpr_t, tpr_t, _ = roc_curve(labs_off, preds_teacher)

    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-",  label=f"Teacher OFF (AUC={auc_teacher:.3f})", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline HLT1 (AUC={auc_baseline:.3f})", linewidth=2)
    plt.plot(tpr_u, fpr_u, "-.", label=f"Union HLT{args.n_hlt_views} (AUC={auc_union:.3f})", linewidth=2)
    plt.plot(tpr_c, fpr_c, ":",  label=f"Consistency HLT{args.n_hlt_views} (AUC={auc_cons:.3f})", linewidth=2)
    plt.plot(tpr_s, fpr_s, linestyle=(0, (5, 2)), label=f"Student KD HLT1 (AUC={auc_student_kd:.3f})", linewidth=2)
    plt.plot(tpr_ck, fpr_ck, linestyle=(0, (3, 1, 1, 1)), label=f"Consistency+KD HLT{args.n_hlt_views} (AUC={auc_cons_kd:.3f})", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=11, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300)
    plt.close()

    # Focused comparisons
    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-",  label=f"Teacher OFF (AUC={auc_teacher:.3f})", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline HLT1 (AUC={auc_baseline:.3f})", linewidth=2)
    plt.plot(tpr_c, fpr_c, ":",  label=f"Consistency HLT{args.n_hlt_views} (AUC={auc_cons:.3f})", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=11, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "results_teacher_baseline_consistency.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-",  label=f"Teacher OFF (AUC={auc_teacher:.3f})", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline HLT1 (AUC={auc_baseline:.3f})", linewidth=2)
    plt.plot(tpr_s, fpr_s, linestyle=(0, (5, 2)), label=f"Student KD HLT1 (AUC={auc_student_kd:.3f})", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=11, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "results_teacher_baseline_kd.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-",  label=f"Teacher OFF (AUC={auc_teacher:.3f})", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline HLT1 (AUC={auc_baseline:.3f})", linewidth=2)
    plt.plot(tpr_ck, fpr_ck, linestyle=(0, (3, 1, 1, 1)), label=f"Consistency+KD HLT{args.n_hlt_views} (AUC={auc_cons_kd:.3f})", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=11, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "results_teacher_baseline_consistency_kd.png", dpi=300)
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
        preds_student_kd=preds_student_kd,
        preds_consistency_kd=preds_cons_kd,
        auc_baseline=auc_baseline,
        auc_union=auc_union,
        auc_consistency=auc_cons,
        auc_student_kd=auc_student_kd,
        auc_consistency_kd=auc_cons_kd,

        fpr_teacher=fpr_t, tpr_teacher=tpr_t,
        fpr_baseline=fpr_b, tpr_baseline=tpr_b,
        fpr_union=fpr_u, tpr_union=tpr_u,
        fpr_consistency=fpr_c, tpr_consistency=tpr_c,
        fpr_student_kd=fpr_s, tpr_student_kd=tpr_s,
        fpr_consistency_kd=fpr_ck, tpr_consistency_kd=tpr_ck,

        br_baseline=br_b,
        br_union=br_u,
        br_consistency=br_c,
        br_student_kd=br_s,
        br_consistency_kd=br_ck,

        n_hlt_views=args.n_hlt_views,
        hlt_seeds=np.array(hlt_seeds, dtype=np.int64),
        lambda_prob=args.lambda_prob,
        lambda_emb=args.lambda_emb,
        rampup_frac=args.rampup_frac,
        conf_power=args.conf_power,
        conf_min=args.conf_min,
        kd_temp_init=args.temp_init,
        kd_temp_final=args.temp_final if args.temp_final is not None else -1.0,
        kd_alpha_init=args.alpha_init,
        kd_alpha_final=args.alpha_final if args.alpha_final is not None else -1.0,
        kd_alpha_attn=CONFIG["kd"]["alpha_attn"],
        kd_alpha_rep=CONFIG["kd"]["alpha_rep"],
        kd_alpha_nce=CONFIG["kd"]["alpha_nce"],
        kd_tau_nce=CONFIG["kd"]["tau_nce"],
        kd_conf_weighted=CONFIG["kd"]["use_conf_weighted_kd"],
    )

    # Append to run summary file
    summary_file = Path(args.save_dir) / "run_summaries.txt"
    with open(summary_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Run: {args.run_name}\n")
        f.write(f"HLT views: {args.n_hlt_views}\n")
        f.write(f"HLT seeds: {hlt_seeds}\n")
        f.write(f"Consistency: lambda_prob={args.lambda_prob}, lambda_emb={args.lambda_emb}, rampup_frac={args.rampup_frac}\n")
        f.write(f"Confidence: conf_power={args.conf_power}, conf_min={args.conf_min}\n")
        f.write(f"KD: temp_init={args.temp_init}, temp_final={args.temp_final}, alpha_init={args.alpha_init}, alpha_final={args.alpha_final}\n")
        f.write(f"KD weights: alpha_attn={CONFIG['kd']['alpha_attn']}, alpha_rep={CONFIG['kd']['alpha_rep']}, alpha_nce={CONFIG['kd']['alpha_nce']}, tau_nce={CONFIG['kd']['tau_nce']}, conf_weighted={CONFIG['kd']['use_conf_weighted_kd']}\n")
        f.write(f"AUC Teacher (OFF test): {auc_teacher:.4f}\n")
        f.write(f"AUC Baseline (HLT1 test): {auc_baseline:.4f} | BR@50%: {br_b:.2f}\n")
        f.write(f"AUC Union (HLT1 test, {args.n_hlt_views} views): {auc_union:.4f} | BR@50%: {br_u:.2f}\n")
        f.write(f"AUC Consistency (HLT1, {args.n_hlt_views} views): {auc_cons:.4f} | BR@50%: {br_c:.2f}\n")
        f.write(f"AUC Student KD (HLT1 test): {auc_student_kd:.4f} | BR@50%: {br_s:.2f}\n")
        f.write(f"AUC Consistency+KD (HLT1, {args.n_hlt_views} views): {auc_cons_kd:.4f} | BR@50%: {br_ck:.2f}\n")
        f.write(f"Saved: {save_dir / 'results.npz'} and {save_dir / 'results.png'}\n")

    print(f"\nSaved results to: {save_dir / 'results.npz'} and {save_dir / 'results.png'}")
    print(f"Logged to: {summary_file}")

    # Cleanup model checkpoints to save space
    for ckpt in save_dir.glob("*.pt"):
        try:
            ckpt.unlink()
        except OSError as exc:
            print(f"Warning: failed to delete {ckpt}: {exc}")


if __name__ == "__main__":
    main()

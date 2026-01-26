#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-HLT-View Consistency Training for ATLAS Jet Tagging (Transformer)

What this script does (drop-in runnable):
  1) Loads your jets via utils.load_from_files over a directory of .h5 files.
  2) Builds OFFLINE (high-quality) view.
  3) Generates TWO different HLT views (HLT_A, HLT_B) using the same realistic HLT effects
     but with different random seeds (so they are two stochastic reconstructions of the same jets).
  4) Computes professor-style 7 relative features and standardizes using OFFLINE train stats.
  5) Trains models:
      A) Teacher: OFFLINE-only model (trained and tested on OFFLINE)
      B) Baseline: HLT_A-only model (trained and tested on HLT_A)
      C) Mixed: train on a random mixture of HLT_A/HLT_B each batch sample (no consistency)
      D) Consistency: train using BOTH HLT_A and HLT_B for each jet with:
           - supervised BCE on BOTH views (averaged)
           - confidence-weighted symmetric KL on probabilities (with agreement+margin gating)
           - confidence-weighted cosine embedding alignment on pooled embeddings

Evaluation:
  - Teacher is evaluated on OFFLINE test.
  - Baseline / Mixed / Consistency are evaluated on HLT_A test (and also optionally HLT_B test).

Outputs:
  - test_split/test_features_and_masks_two_hlt.npz
      (offline + hltA + hltB standardized features, masks, labels, indices, means/stds)
  - checkpoints/transformer_twohlt_suite/<run_name>/
      teacher.pt, baseline_hltA.pt, mixed.pt, consistency.pt
      results.npz, results.png
      run_summaries.txt (appends a short summary per run)

Assumption about utils.load_from_files output:
  all_data: (N, max_constits, 3) with columns [eta, phi, pt]
If your columns differ, edit ETA_IDX/PHI_IDX/PT_IDX below.
"""

from pathlib import Path
import argparse
import os
import random
import copy
import numpy as np

# Avoid Qt backend issues on headless Linux
import matplotlib
matplotlib.use("Agg")
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # Make dataloader workers deterministic
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------- Column order (EDIT if needed) ----------------------------- #
ETA_IDX = 0
PHI_IDX = 1
PT_IDX = 2


# ----------------------------- Base config (matches professor defaults) ----------------------------- #
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


# ----------------------------- HLT Simulation (professor logic) ----------------------------- #
def apply_hlt_effects(const, mask, cfg, seed=42):
    """
    const: (n_jets, max_part, 4) columns [pt, eta, phi, E]
    mask:  (n_jets, max_part) boolean
    """
    np.random.seed(seed)
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    n_initial = int(hlt_mask.sum())

    # Effect 1: Higher pT threshold
    pt_threshold = hcfg["pt_threshold_hlt"]
    below_threshold = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0
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
        n_lost_eff = int(lost.sum())

    # Final cleanup
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0

    n_final = int(hlt_mask.sum())
    retention = 100 * n_final / max(n_initial, 1)

    print("\nHLT Simulation Statistics (seed={}):".format(seed))
    print(f"  Offline particles: {n_initial:,}")
    print(f"  Lost to pT threshold ({hcfg['pt_threshold_hlt']}): {n_lost_threshold:,} ({100*n_lost_threshold/max(n_initial,1):.1f}%)")
    print(f"  Lost to merging (dR<{hcfg['merge_radius']}): {n_merged:,} ({100*n_merged/max(n_initial,1):.1f}%)")
    print(f"  Lost to efficiency: {n_lost_eff:,} ({100*n_lost_eff/max(n_initial,1):.1f}%)")
    print(f"  HLT particles: {n_final:,} ({retention:.1f}% of offline)")

    return hlt.astype(np.float32), hlt_mask.astype(bool)


# ----------------------------- Feature computation (professor) ----------------------------- #
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
    return means.astype(np.float32), stds.astype(np.float32)


def standardize(feat, mask, means, stds):
    std = np.clip((feat - means) / stds, -10, 10)
    std = np.nan_to_num(std, 0.0)
    std[~mask] = 0
    return std.astype(np.float32)


# ----------------------------- Datasets ----------------------------- #
class SingleViewJetDataset(Dataset):
    def __init__(self, feat, mask, labels):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "x": self.feat[i],
            "mask": self.mask[i],
            "label": self.labels[i],
        }


class MixedHLTDataset(Dataset):
    """
    Returns one of two HLT views at random per sample.
    No explicit acknowledgement that the two views correspond.
    """
    def __init__(self, feat_a, mask_a, feat_b, mask_b, labels, p_a=0.5):
        self.a = torch.tensor(feat_a, dtype=torch.float32)
        self.ma = torch.tensor(mask_a, dtype=torch.bool)
        self.b = torch.tensor(feat_b, dtype=torch.float32)
        self.mb = torch.tensor(mask_b, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.p_a = float(p_a)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if random.random() < self.p_a:
            return {"x": self.a[i], "mask": self.ma[i], "label": self.labels[i]}
        return {"x": self.b[i], "mask": self.mb[i], "label": self.labels[i]}


class PairedHLTDataset(Dataset):
    """
    Returns both HLT views for the SAME jet (paired).
    """
    def __init__(self, feat_a, mask_a, feat_b, mask_b, labels):
        self.a = torch.tensor(feat_a, dtype=torch.float32)
        self.ma = torch.tensor(mask_a, dtype=torch.bool)
        self.b = torch.tensor(feat_b, dtype=torch.float32)
        self.mb = torch.tensor(mask_b, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "x_a": self.a[i],
            "m_a": self.ma[i],
            "x_b": self.b[i],
            "m_b": self.mb[i],
            "label": self.labels[i],
        }


# ----------------------------- Model (professor + embedding output) ----------------------------- #
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
    """
    Same architecture as professor, but can return:
      - logits
      - pooled embedding z (the vector right before classifier)
      - pooling attention weights (optional)
    """
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
        batch_size, seq_len, _ = x.shape

        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)

        h = self.transformer(h, src_key_padding_mask=~mask)

        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, attn_weights = self.pool_attn(
            query, h, h,
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=True,
        )

        z = self.norm(pooled.squeeze(1))
        logits = self.classifier(z)

        if return_embedding and return_attention:
            return logits, z, attn_weights.squeeze(1)
        if return_embedding:
            return logits, z
        if return_attention:
            return logits, attn_weights.squeeze(1)
        return logits


# ----------------------------- Losses for two-view consistency ----------------------------- #
def bce_logits(logits, y):
    return F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1))


def symmetric_kl_bernoulli(p, q, eps=1e-6):
    """
    p, q: probabilities in (0,1), shape (B,)
    returns: KL(p||q) + KL(q||p) averaged over batch
    """
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)

    kl_pq = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    kl_qp = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
    return 0.5 * (kl_pq + kl_qp)


def cosine_embed_loss(z1, z2, eps=1e-8):
    """
    1 - cosine similarity, shape (B,)
    """
    z1n = z1 / (z1.norm(dim=1, keepdim=True) + eps)
    z2n = z2 / (z2.norm(dim=1, keepdim=True) + eps)
    cos = (z1n * z2n).sum(dim=1)
    return 1.0 - cos


def linear_ramp(epoch, ramp_epochs, max_value):
    if ramp_epochs <= 0:
        return max_value
    t = min(max(epoch, 0), ramp_epochs) / float(ramp_epochs)
    return max_value * t


# ----------------------------- Train / Eval ----------------------------- #
@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].cpu().numpy().flatten()
        logits = model(x, m).view(-1)
        p = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds.append(p)
        labs.append(y)
    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return roc_auc_score(labs, preds), preds, labs


def train_standard_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    preds, labs = [], []

    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].to(device)

        opt.zero_grad()
        logits = model(x, m).view(-1)
        loss = bce_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += loss.item() * y.shape[0]
        preds.append(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.append(y.detach().cpu().numpy().flatten())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return total / len(labs), roc_auc_score(labs, preds)


def train_consistency_epoch(
    model,
    loader,
    opt,
    device,
    lambda_prob,
    lambda_emb,
    conf_thr=0.10,
    require_agreement=True,
):
    """
    Best-effort "safe" consistency:
      - supervised BCE on both views
      - confidence-weighted symmetric KL on probabilities
      - confidence-weighted cosine embedding alignment
      - agreement + margin gating (optional)
      - stop-grad style symmetric KL (each direction uses one detached side)
    """
    model.train()
    total = 0.0
    preds, labs = [], []

    for batch in loader:
        x_a = batch["x_a"].to(device)
        m_a = batch["m_a"].to(device)
        x_b = batch["x_b"].to(device)
        m_b = batch["m_b"].to(device)
        y = batch["label"].to(device)

        opt.zero_grad()

        logits_a, z_a = model(x_a, m_a, return_embedding=True)
        logits_b, z_b = model(x_b, m_b, return_embedding=True)

        logits_a = logits_a.view(-1)
        logits_b = logits_b.view(-1)

        # Supervised on BOTH views
        loss_sup = 0.5 * (bce_logits(logits_a, y) + bce_logits(logits_b, y))

        # Probabilities
        p_a = torch.sigmoid(logits_a)
        p_b = torch.sigmoid(logits_b)

        # Detach for weighting / gating
        p_a_d = p_a.detach()
        p_b_d = p_b.detach()

        # Confidence weight in [0,1]
        conf_a = (2.0 * torch.abs(p_a_d - 0.5)).clamp(0.0, 1.0)
        conf_b = (2.0 * torch.abs(p_b_d - 0.5)).clamp(0.0, 1.0)
        w = 0.5 * (conf_a + conf_b)

        # Gating mask
        if require_agreement:
            agree = ((p_a_d >= 0.5) & (p_b_d >= 0.5)) | ((p_a_d < 0.5) & (p_b_d < 0.5))
        else:
            agree = torch.ones_like(w, dtype=torch.bool)

        confident = (torch.maximum(conf_a, conf_b) >= conf_thr)
        gate = (agree & confident).float()

        # Symmetric KL with stop-grad in each direction (stable-ish)
        # term1: KL(p_a_detach || p_b)
        # term2: KL(p_b_detach || p_a)
        kl_1 = symmetric_kl_bernoulli(p_a_d, p_b)
        kl_2 = symmetric_kl_bernoulli(p_b_d, p_a)
        loss_prob_vec = 0.5 * (kl_1 + kl_2)

        # Embedding cosine loss (vector per sample)
        loss_emb_vec = cosine_embed_loss(z_a, z_b)

        # Apply weights + gate
        loss_prob = (w * gate * loss_prob_vec).mean()
        loss_emb = (w * gate * loss_emb_vec).mean()

        loss = loss_sup + lambda_prob * loss_prob + lambda_emb * loss_emb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += loss.item() * y.shape[0]

        # Track AUC on view A
        preds.append(p_a.detach().cpu().numpy().flatten())
        labs.append(y.detach().cpu().numpy().flatten())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return total / len(labs), roc_auc_score(labs, preds)


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / float(max(warmup, 1))
        denom = max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / denom))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="./data", help="Directory containing your *.h5 files")
    parser.add_argument("--n_train_jets", type=int, default=100000)
    parser.add_argument("--max_constits", type=int, default=80)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "transformer_twohlt_suite"))
    parser.add_argument("--run_name", type=str, default="default")

    # Two HLT seeds
    parser.add_argument("--hlt_seed_a", type=int, default=123)
    parser.add_argument("--hlt_seed_b", type=int, default=999)

    # Mixed training
    parser.add_argument("--mix_p_a", type=float, default=0.5, help="Probability of sampling HLT_A in the mixed model")

    # Consistency hyperparams
    parser.add_argument("--lambda_prob", type=float, default=1.0, help="Max weight for probability consistency")
    parser.add_argument("--lambda_emb", type=float, default=0.2, help="Max weight for embedding cosine consistency")
    parser.add_argument("--ramp_epochs", type=int, default=10, help="Linearly ramp consistency weights over this many epochs")
    parser.add_argument("--conf_thr", type=float, default=0.10, help="Confidence threshold for gating (in [0,1])")
    parser.add_argument("--no_agreement_gate", action="store_true", help="If set, do NOT require predicted-class agreement")

    # Optional: skip training some stages if checkpoints exist
    parser.add_argument("--skip_if_exists", action="store_true", help="If set, skip training a model if checkpoint exists")

    args = parser.parse_args()

    seed_everything(RANDOM_SEED)

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

    # ------------------- Load data ------------------- #
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

    # ------------------- Build OFFLINE constituents ------------------- #
    eta = all_data[:, :, ETA_IDX].astype(np.float32)
    phi = all_data[:, :, PHI_IDX].astype(np.float32)
    pt = all_data[:, :, PT_IDX].astype(np.float32)

    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    constituents_raw = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    print(f"Avg particles per jet (raw mask): {mask_raw.sum(axis=1).mean():.1f}")

    # Offline threshold
    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0

    print(f"Offline particles after {pt_threshold_off} threshold: {masks_off.sum():,}")

    # ------------------- Build TWO HLT views (same jets, different randomness) ------------------- #
    print("\nGenerating HLT_A...")
    constituents_hlt_a, masks_hlt_a = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=args.hlt_seed_a)

    print("\nGenerating HLT_B...")
    constituents_hlt_b, masks_hlt_b = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=args.hlt_seed_b)

    print(f"\nAvg per jet: Offline={masks_off.sum(axis=1).mean():.1f}, HLT_A={masks_hlt_a.sum(axis=1).mean():.1f}, HLT_B={masks_hlt_b.sum(axis=1).mean():.1f}")

    # ------------------- Compute features ------------------- #
    print("\nComputing features...")
    feat_off = compute_features(constituents_off, masks_off)
    feat_a = compute_features(constituents_hlt_a, masks_hlt_a)
    feat_b = compute_features(constituents_hlt_b, masks_hlt_b)

    print(f"NaN check: off={np.isnan(feat_off).sum()}, a={np.isnan(feat_a).sum()}, b={np.isnan(feat_b).sum()}")

    # ------------------- Split indices ------------------- #
    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])

    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # ------------------- Standardize (OFFLINE train stats) ------------------- #
    means, stds = get_stats(feat_off, masks_off, train_idx)

    feat_off_s = standardize(feat_off, masks_off, means, stds)
    feat_a_s = standardize(feat_a, masks_hlt_a, means, stds)
    feat_b_s = standardize(feat_b, masks_hlt_b, means, stds)

    print(f"Final NaN check: off={np.isnan(feat_off_s).sum()}, a={np.isnan(feat_a_s).sum()}, b={np.isnan(feat_b_s).sum()}")

    # ------------------- Save test artifacts ------------------- #
    test_split_dir = Path().cwd() / "test_split"
    test_split_dir.mkdir(exist_ok=True)

    np.savez(
        test_split_dir / "test_features_and_masks_two_hlt.npz",
        idx_test=test_idx,
        labels=all_labels[test_idx],
        feat_off=feat_off_s[test_idx],
        mask_off=masks_off[test_idx],
        feat_hlt_a=feat_a_s[test_idx],
        mask_hlt_a=masks_hlt_a[test_idx],
        feat_hlt_b=feat_b_s[test_idx],
        mask_hlt_b=masks_hlt_b[test_idx],
        jet_pt=all_pt[test_idx] if all_pt is not None else None,
        feat_means=means,
        feat_stds=stds,
        hlt_seed_a=args.hlt_seed_a,
        hlt_seed_b=args.hlt_seed_b,
    )
    print(f"Saved test split to: {test_split_dir / 'test_features_and_masks_two_hlt.npz'}")

    # ------------------- DataLoaders ------------------- #
    BS = CONFIG["training"]["batch_size"]
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    # Teacher: OFFLINE
    train_off_ds = SingleViewJetDataset(feat_off_s[train_idx], masks_off[train_idx], all_labels[train_idx])
    val_off_ds = SingleViewJetDataset(feat_off_s[val_idx], masks_off[val_idx], all_labels[val_idx])
    test_off_ds = SingleViewJetDataset(feat_off_s[test_idx], masks_off[test_idx], all_labels[test_idx])

    train_off_loader = DataLoader(
        train_off_ds, batch_size=BS, shuffle=True, drop_last=True,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    val_off_loader = DataLoader(
        val_off_ds, batch_size=BS, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    test_off_loader = DataLoader(
        test_off_ds, batch_size=BS, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )

    # Baseline HLT_A
    train_a_ds = SingleViewJetDataset(feat_a_s[train_idx], masks_hlt_a[train_idx], all_labels[train_idx])
    val_a_ds = SingleViewJetDataset(feat_a_s[val_idx], masks_hlt_a[val_idx], all_labels[val_idx])
    test_a_ds = SingleViewJetDataset(feat_a_s[test_idx], masks_hlt_a[test_idx], all_labels[test_idx])

    train_a_loader = DataLoader(
        train_a_ds, batch_size=BS, shuffle=True, drop_last=True,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    val_a_loader = DataLoader(
        val_a_ds, batch_size=BS, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    test_a_loader = DataLoader(
        test_a_ds, batch_size=BS, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )

    # Optional: Test HLT_B too
    test_b_ds = SingleViewJetDataset(feat_b_s[test_idx], masks_hlt_b[test_idx], all_labels[test_idx])
    test_b_loader = DataLoader(
        test_b_ds, batch_size=BS, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )

    # Mixed: randomly sample view A or B per example
    train_mix_ds = MixedHLTDataset(
        feat_a_s[train_idx], masks_hlt_a[train_idx],
        feat_b_s[train_idx], masks_hlt_b[train_idx],
        all_labels[train_idx], p_a=args.mix_p_a
    )
    val_mix_ds = MixedHLTDataset(
        feat_a_s[val_idx], masks_hlt_a[val_idx],
        feat_b_s[val_idx], masks_hlt_b[val_idx],
        all_labels[val_idx], p_a=args.mix_p_a
    )

    train_mix_loader = DataLoader(
        train_mix_ds, batch_size=BS, shuffle=True, drop_last=True,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    val_mix_loader = DataLoader(
        val_mix_ds, batch_size=BS, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )

    # Consistency: paired views
    train_pair_ds = PairedHLTDataset(
        feat_a_s[train_idx], masks_hlt_a[train_idx],
        feat_b_s[train_idx], masks_hlt_b[train_idx],
        all_labels[train_idx]
    )
    val_pair_ds = PairedHLTDataset(
        feat_a_s[val_idx], masks_hlt_a[val_idx],
        feat_b_s[val_idx], masks_hlt_b[val_idx],
        all_labels[val_idx]
    )

    train_pair_loader = DataLoader(
        train_pair_ds, batch_size=BS, shuffle=True, drop_last=True,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )
    val_pair_loader = DataLoader(
        val_pair_ds, batch_size=BS, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )

    # ------------------- Checkpoints ------------------- #
    teacher_path = save_dir / "teacher.pt"
    baseline_path = save_dir / "baseline_hltA.pt"
    mixed_path = save_dir / "mixed.pt"
    cons_path = save_dir / "consistency.pt"

    # ------------------- Training helpers ------------------- #
    def maybe_load(model, path):
        if path.exists():
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt["model"])
            return ckpt.get("best_val_auc", None), ckpt.get("history", [])
        return None, []

    def save_ckpt(path, model, best_val_auc, history):
        torch.save(
            {
                "model": model.state_dict(),
                "best_val_auc": float(best_val_auc),
                "history": history,
                "config": copy.deepcopy(CONFIG),
                "hlt_seed_a": args.hlt_seed_a,
                "hlt_seed_b": args.hlt_seed_b,
            },
            path,
        )

    # ------------------- Model A: Teacher (OFFLINE) ------------------- #
    print("\n" + "=" * 80)
    print("MODEL A: Teacher (OFFLINE-only)")
    print("=" * 80)

    teacher = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_val_teacher, hist_teacher = (None, [])

    if args.skip_if_exists and teacher_path.exists():
        best_val_teacher, hist_teacher = maybe_load(teacher, teacher_path)
        print(f"Loaded teacher checkpoint: {teacher_path} (best_val_auc={best_val_teacher})")
    else:
        opt = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_val_teacher = 0.0
        best_state = None
        no_improve = 0
        hist_teacher = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Teacher"):
            tr_loss, tr_auc = train_standard_epoch(teacher, train_off_loader, opt, device)
            val_auc, _, _ = evaluate_auc(teacher, val_off_loader, device)
            sch.step()

            hist_teacher.append((ep + 1, tr_loss, tr_auc, val_auc))

            if val_auc > best_val_teacher:
                best_val_teacher = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={val_auc:.4f}, best={best_val_teacher:.4f}")

            if no_improve >= CONFIG["training"]["patience"]:
                print(f"Early stopping teacher at epoch {ep+1}")
                break

        teacher.load_state_dict(best_state)
        save_ckpt(teacher_path, teacher, best_val_teacher, hist_teacher)
        print(f"Saved teacher: {teacher_path} (best_val_auc={best_val_teacher:.4f})")

    # ------------------- Model B: Baseline (HLT_A only) ------------------- #
    print("\n" + "=" * 80)
    print("MODEL B: Baseline (HLT_A only)")
    print("=" * 80)

    baseline = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_val_base, hist_base = (None, [])

    if args.skip_if_exists and baseline_path.exists():
        best_val_base, hist_base = maybe_load(baseline, baseline_path)
        print(f"Loaded baseline checkpoint: {baseline_path} (best_val_auc={best_val_base})")
    else:
        opt = torch.optim.AdamW(baseline.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_val_base = 0.0
        best_state = None
        no_improve = 0
        hist_base = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Baseline HLT_A"):
            tr_loss, tr_auc = train_standard_epoch(baseline, train_a_loader, opt, device)
            val_auc, _, _ = evaluate_auc(baseline, val_a_loader, device)
            sch.step()

            hist_base.append((ep + 1, tr_loss, tr_auc, val_auc))

            if val_auc > best_val_base:
                best_val_base = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={val_auc:.4f}, best={best_val_base:.4f}")

            if no_improve >= CONFIG["training"]["patience"] + 5:
                print(f"Early stopping baseline at epoch {ep+1}")
                break

        baseline.load_state_dict(best_state)
        save_ckpt(baseline_path, baseline, best_val_base, hist_base)
        print(f"Saved baseline: {baseline_path} (best_val_auc={best_val_base:.4f})")

    # ------------------- Model C: Mixed (HLT_A / HLT_B random mix, no consistency) ------------------- #
    print("\n" + "=" * 80)
    print("MODEL C: Mixed (randomly sample HLT_A or HLT_B per example, no consistency)")
    print("=" * 80)

    mixed = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_val_mixed, hist_mixed = (None, [])

    if args.skip_if_exists and mixed_path.exists():
        best_val_mixed, hist_mixed = maybe_load(mixed, mixed_path)
        print(f"Loaded mixed checkpoint: {mixed_path} (best_val_auc={best_val_mixed})")
    else:
        opt = torch.optim.AdamW(mixed.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_val_mixed = 0.0
        best_state = None
        no_improve = 0
        hist_mixed = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Mixed"):
            tr_loss, tr_auc = train_standard_epoch(mixed, train_mix_loader, opt, device)
            # Evaluate on HLT_A val for consistency with test protocol
            val_auc, _, _ = evaluate_auc(mixed, val_a_loader, device)
            sch.step()

            hist_mixed.append((ep + 1, tr_loss, tr_auc, val_auc))

            if val_auc > best_val_mixed:
                best_val_mixed = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in mixed.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={val_auc:.4f}, best={best_val_mixed:.4f}")

            if no_improve >= CONFIG["training"]["patience"] + 5:
                print(f"Early stopping mixed at epoch {ep+1}")
                break

        mixed.load_state_dict(best_state)
        save_ckpt(mixed_path, mixed, best_val_mixed, hist_mixed)
        print(f"Saved mixed: {mixed_path} (best_val_auc={best_val_mixed:.4f})")

    # ------------------- Model D: Consistency (paired HLT_A / HLT_B with best loss) ------------------- #
    print("\n" + "=" * 80)
    print("MODEL D: Consistency (paired HLT_A vs HLT_B with prob KL + embedding cosine)")
    print("=" * 80)
    print(f"  Max lambda_prob={args.lambda_prob}, Max lambda_emb={args.lambda_emb}, ramp_epochs={args.ramp_epochs}")
    print(f"  Gating: conf_thr={args.conf_thr}, require_agreement={not args.no_agreement_gate}")

    cons = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    best_val_cons, hist_cons = (None, [])

    if args.skip_if_exists and cons_path.exists():
        best_val_cons, hist_cons = maybe_load(cons, cons_path)
        print(f"Loaded consistency checkpoint: {cons_path} (best_val_auc={best_val_cons})")
    else:
        opt = torch.optim.AdamW(cons.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_val_cons = 0.0
        best_state = None
        no_improve = 0
        hist_cons = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Consistency"):
            # Ramp up
            lam_p = linear_ramp(ep, args.ramp_epochs, args.lambda_prob)
            lam_e = linear_ramp(ep, args.ramp_epochs, args.lambda_emb)

            tr_loss, tr_auc = train_consistency_epoch(
                cons,
                train_pair_loader,
                opt,
                device,
                lambda_prob=lam_p,
                lambda_emb=lam_e,
                conf_thr=args.conf_thr,
                require_agreement=(not args.no_agreement_gate),
            )

            # Evaluate on HLT_A val
            val_auc, _, _ = evaluate_auc(cons, val_a_loader, device)
            sch.step()

            hist_cons.append((ep + 1, tr_loss, tr_auc, val_auc, lam_p, lam_e))

            if val_auc > best_val_cons:
                best_val_cons = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in cons.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: val_auc={val_auc:.4f}, best={best_val_cons:.4f} | lam_p={lam_p:.3f} lam_e={lam_e:.3f}")

            if no_improve >= CONFIG["training"]["patience"] + 5:
                print(f"Early stopping consistency at epoch {ep+1}")
                break

        cons.load_state_dict(best_state)
        save_ckpt(cons_path, cons, best_val_cons, hist_cons)
        print(f"Saved consistency: {cons_path} (best_val_auc={best_val_cons:.4f})")

    # ------------------- Final evaluation ------------------- #
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    auc_teacher_off, preds_teacher_off, labs_off = evaluate_auc(teacher, test_off_loader, device)

    auc_base_a, preds_base_a, labs_a = evaluate_auc(baseline, test_a_loader, device)
    auc_mixed_a, preds_mixed_a, _ = evaluate_auc(mixed, test_a_loader, device)
    auc_cons_a, preds_cons_a, _ = evaluate_auc(cons, test_a_loader, device)

    # Also evaluate on HLT_B test (optional but useful)
    auc_base_b, preds_base_b, labs_b = evaluate_auc(baseline, test_b_loader, device)
    auc_mixed_b, preds_mixed_b, _ = evaluate_auc(mixed, test_b_loader, device)
    auc_cons_b, preds_cons_b, _ = evaluate_auc(cons, test_b_loader, device)

    print(f"\n{'Model':<35} {'AUC (Offline test)':>18} {'AUC (HLT_A test)':>18} {'AUC (HLT_B test)':>18}")
    print("-" * 92)
    print(f"{'Teacher (OFF)':<35} {auc_teacher_off:>18.4f} {'-':>18} {'-':>18}")
    print(f"{'Baseline (HLT_A only)':<35} {'-':>18} {auc_base_a:>18.4f} {auc_base_b:>18.4f}")
    print(f"{'Mixed (A/B random)':<35} {'-':>18} {auc_mixed_a:>18.4f} {auc_mixed_b:>18.4f}")
    print(f"{'Consistency (A vs B)':<35} {'-':>18} {auc_cons_a:>18.4f} {auc_cons_b:>18.4f}")
    print("-" * 92)

    # ROCs (Teacher on OFFLINE, others on HLT_A for primary figure)
    fpr_t, tpr_t, _ = roc_curve(labs_off, preds_teacher_off)
    fpr_b, tpr_b, _ = roc_curve(labs_a, preds_base_a)
    fpr_m, tpr_m, _ = roc_curve(labs_a, preds_mixed_a)
    fpr_c, tpr_c, _ = roc_curve(labs_a, preds_cons_a)

    # Background rejection @ 50% signal efficiency on HLT_A-tested models
    wp = 0.5
    def br_at_wp(fpr, tpr, wp=0.5):
        idx_wp = np.argmax(tpr >= wp)
        return 1.0 / fpr[idx_wp] if fpr[idx_wp] > 0 else 0.0

    br_base = br_at_wp(fpr_b, tpr_b, wp)
    br_mixed = br_at_wp(fpr_m, tpr_m, wp)
    br_cons = br_at_wp(fpr_c, tpr_c, wp)

    print(f"\nBackground Rejection @ {wp*100:.0f}% signal efficiency (HLT_A-tested):")
    print(f"  Baseline:    {br_base:.2f}")
    print(f"  Mixed:       {br_mixed:.2f}")
    print(f"  Consistency: {br_cons:.2f}")

    # Save results
    np.savez(
        save_dir / "results.npz",
        # labels/preds
        labs_off=labs_off,
        preds_teacher_off=preds_teacher_off,
        labs_hlt_a=labs_a,
        preds_base_a=preds_base_a,
        preds_mixed_a=preds_mixed_a,
        preds_cons_a=preds_cons_a,
        labs_hlt_b=labs_b,
        preds_base_b=preds_base_b,
        preds_mixed_b=preds_mixed_b,
        preds_cons_b=preds_cons_b,
        # AUCs
        auc_teacher_off=auc_teacher_off,
        auc_base_a=auc_base_a,
        auc_mixed_a=auc_mixed_a,
        auc_cons_a=auc_cons_a,
        auc_base_b=auc_base_b,
        auc_mixed_b=auc_mixed_b,
        auc_cons_b=auc_cons_b,
        # ROCs for primary figure
        fpr_teacher_off=fpr_t, tpr_teacher_off=tpr_t,
        fpr_base_a=fpr_b, tpr_base_a=tpr_b,
        fpr_mixed_a=fpr_m, tpr_mixed_a=tpr_m,
        fpr_cons_a=fpr_c, tpr_cons_a=tpr_c,
        # BR @ wp
        br_base=br_base,
        br_mixed=br_mixed,
        br_cons=br_cons,
        wp=wp,
        # meta
        hlt_seed_a=args.hlt_seed_a,
        hlt_seed_b=args.hlt_seed_b,
        lambda_prob=args.lambda_prob,
        lambda_emb=args.lambda_emb,
        ramp_epochs=args.ramp_epochs,
        conf_thr=args.conf_thr,
        require_agreement=(not args.no_agreement_gate),
        mix_p_a=args.mix_p_a,
    )

    # Plot ROC curves with your preferred axes: TPR on x, FPR on y
    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-", label=f"Teacher OFF (AUC={auc_teacher_off:.3f})", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline HLT_A (AUC={auc_base_a:.3f})", linewidth=2)
    plt.plot(tpr_m, fpr_m, "-.", label=f"Mixed HLT_A (AUC={auc_mixed_a:.3f})", linewidth=2)
    plt.plot(tpr_c, fpr_c, ":", label=f"Consistency HLT_A (AUC={auc_cons_a:.3f})", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300)
    plt.close()

    # Append summary
    summary_file = Path(args.save_dir) / "run_summaries.txt"
    with open(summary_file, "a") as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write(f"Run: {args.run_name}\n")
        f.write(f"HLT seeds: A={args.hlt_seed_a}, B={args.hlt_seed_b} | mix_p_a={args.mix_p_a}\n")
        f.write(f"Consistency: lambda_prob={args.lambda_prob}, lambda_emb={args.lambda_emb}, ramp_epochs={args.ramp_epochs}\n")
        f.write(f"Gating: conf_thr={args.conf_thr}, require_agreement={not args.no_agreement_gate}\n")
        f.write(f"AUC Teacher OFF: {auc_teacher_off:.4f}\n")
        f.write(f"AUC Baseline HLT_A: {auc_base_a:.4f} | HLT_B: {auc_base_b:.4f}\n")
        f.write(f"AUC Mixed HLT_A: {auc_mixed_a:.4f} | HLT_B: {auc_mixed_b:.4f}\n")
        f.write(f"AUC Consis HLT_A: {auc_cons_a:.4f} | HLT_B: {auc_cons_b:.4f}\n")
        f.write(f"BR@{wp:.2f} (HLT_A): baseline={br_base:.2f}, mixed={br_mixed:.2f}, cons={br_cons:.2f}\n")
        f.write(f"Artifacts: {save_dir}\n")

    print(f"\nSaved results to: {save_dir / 'results.npz'} and {save_dir / 'results.png'}")
    print(f"Logged to: {summary_file}")


if __name__ == "__main__":
    main()

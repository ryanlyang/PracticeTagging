#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full Unmerge + Unsmeared pipeline.

Goal:
  - Use HLT effects (smearing + merging + efficiency) defined here.
  - Train:
      1) Offline teacher (baseline comparison)
      2) HLT baseline (smeared + merged)
      3) Merge-count predictor (HLT -> count)
      4) Distributional unmerger (HLT token -> offline constituents)
      5) Unsmeared diffusion model (smearing-only, conditional)
      6) Final classifier trained on unmerged->unsmeared constituents
  - At val/test time, use only HLT:
      HLT -> merge-count -> unmerger -> unsmear -> classifier
"""

from pathlib import Path
import argparse
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from tqdm import tqdm
import matplotlib.pyplot as plt

import utils

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


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


# ----------------------------- Column order ----------------------------- #
ETA_IDX = 0
PHI_IDX = 1
PT_IDX = 2


CONFIG = {
    "hlt_effects": {
        # Resolution smearing
        "pt_resolution": 0.10,
        "eta_resolution": 0.03,
        "phi_resolution": 0.03,
        # pT thresholds
        "pt_threshold_offline": 0.5,
        "pt_threshold_hlt": 1.5,
        # Cluster merging
        "merge_enabled": True,
        "merge_radius": 0.01,
        # Efficiency loss
        "efficiency_loss": 0.03,
        # Noise (disabled)
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
    "merge_count_model": {
        "embed_dim": 128,
        "num_heads": 8,
        "num_layers": 6,
        "ff_dim": 512,
        "dropout": 0.1,
    },
    "unmerge_model": {
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 8,
        "decoder_layers": 4,
        "ff_dim": 1024,
        "dropout": 0.1,
        "count_embed_dim": 64,
    },
    "training": {
        "batch_size": 512,
        "epochs": 60,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 3,
        "patience": 15,
    },
    "merge_count_training": {
        "batch_size": 512,
        "epochs": 80,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 3,
        "patience": 15,
    },
    "unmerge_training": {
        "batch_size": 256,
        "epochs": 120,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 5,
        "patience": 20,
        "loss_type": "hungarian",
        "use_true_count": True,
        "curriculum": True,
        "curriculum_start": 2,
        "curriculum_epochs": 20,
        "physics_weight": 0.2,
        "nll_weight": 1.0,
        "distributional": True,
    },
    "diffusion": {
        "timesteps": 1000,
        "schedule": "cosine",
        "pred_type": "v",
        "snr_weight": True,
        "snr_gamma": 5.0,
        "self_cond_prob": 0.5,
        "cond_drop_prob": 0.1,
        "x0_weight": 0.1,
        "jet_loss_weight": 0.1,
        "ema_decay": 0.995,
    },
    "sampling": {
        "method": "ddim",
        "sample_steps": 200,
        "n_samples_eval": 1,
        "guidance_scale": 1.5,
    },
}


# ----------------------------- Utilities ----------------------------- #
def safe_sigmoid(logits):
    probs = torch.sigmoid(logits)
    return torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


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
    means = np.zeros(feat.shape[-1], dtype=np.float32)
    stds = np.zeros(feat.shape[-1], dtype=np.float32)
    for i in range(feat.shape[-1]):
        vals = feat[idx][:, :, i][mask[idx]]
        means[i] = np.nanmean(vals)
        stds[i] = np.nanstd(vals) + 1e-8
    return means, stds


def standardize(feat, mask, means, stds):
    std = np.clip((feat - means) / stds, -10, 10)
    std = np.nan_to_num(std, 0.0)
    std[~mask] = 0
    return std.astype(np.float32)


# ----------------------------- HLT effects with tracking ----------------------------- #
def apply_hlt_effects_with_tracking(const, mask, cfg, seed=42):
    np.random.seed(seed)
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    origin_counts = hlt_mask.astype(np.int32)
    origin_lists = [[([idx] if hlt_mask[j, idx] else []) for idx in range(max_part)]
                    for j in range(n_jets)]

    n_initial = int(hlt_mask.sum())

    pt_threshold = hcfg["pt_threshold_hlt"]
    below_threshold = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0
    origin_counts[~hlt_mask] = 0
    for j in range(n_jets):
        for idx in np.where(below_threshold[j])[0]:
            origin_lists[j][idx] = []
    n_lost_threshold = int(below_threshold.sum())

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

                        origin_lists[jet_idx][idx_i].extend(origin_lists[jet_idx][idx_j])
                        origin_lists[jet_idx][idx_j] = []
                        to_remove.add(idx_j)
                        n_merged += 1

            for idx in to_remove:
                hlt_mask[jet_idx, idx] = False
                hlt[jet_idx, idx] = 0

    # Resolution smearing
    valid = hlt_mask
    if hcfg["pt_resolution"] > 0:
        pt_noise = np.random.normal(1.0, hcfg["pt_resolution"], (n_jets, max_part))
        pt_noise = np.clip(pt_noise, 0.5, 1.5)
        hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)
    if hcfg["eta_resolution"] > 0:
        eta_noise = np.random.normal(0, hcfg["eta_resolution"], (n_jets, max_part))
        hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)
    if hcfg["phi_resolution"] > 0:
        phi_noise = np.random.normal(0, hcfg["phi_resolution"], (n_jets, max_part))
        new_phi = hlt[:, :, 2] + phi_noise
        hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)

    # Recalculate E (massless)
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)

    # Efficiency loss
    n_lost_eff = 0
    if hcfg["efficiency_loss"] > 0:
        random_loss = np.random.random((n_jets, max_part)) < hcfg["efficiency_loss"]
        lost = random_loss & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0
        n_lost_eff = int(lost.sum())
        for j in range(n_jets):
            for idx in np.where(lost[j])[0]:
                origin_lists[j][idx] = []

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0
    origin_counts = np.array([[len(origin_lists[j][i]) for i in range(max_part)] for j in range(n_jets)], dtype=np.int32)
    origin_counts[~hlt_mask] = 0

    n_final = int(hlt_mask.sum())
    stats = {
        "n_initial": n_initial,
        "n_lost_threshold": n_lost_threshold,
        "n_merged": n_merged,
        "n_lost_eff": n_lost_eff,
        "n_final": n_final,
    }
    return hlt, hlt_mask, origin_counts, origin_lists, stats


def apply_smear_only(const, mask, cfg, seed=42):
    """Smearing-only view for diffusion training."""
    np.random.seed(seed)
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape
    hlt = const.copy()
    hlt_mask = mask.copy()
    pt_threshold = hcfg["pt_threshold_hlt"]
    below_threshold = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0

    valid = hlt_mask
    if hcfg["pt_resolution"] > 0:
        pt_noise = np.random.normal(1.0, hcfg["pt_resolution"], (n_jets, max_part))
        pt_noise = np.clip(pt_noise, 0.5, 1.5)
        hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)
    if hcfg["eta_resolution"] > 0:
        eta_noise = np.random.normal(0, hcfg["eta_resolution"], (n_jets, max_part))
        hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)
    if hcfg["phi_resolution"] > 0:
        phi_noise = np.random.normal(0, hcfg["phi_resolution"], (n_jets, max_part))
        new_phi = hlt[:, :, 2] + phi_noise
        hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0
    return hlt, hlt_mask


# ----------------------------- Datasets ----------------------------- #
class JetDataset(Dataset):
    def __init__(self, feat, mask, labels):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"feat": self.feat[i], "mask": self.mask[i], "label": self.labels[i]}


class MergeCountDataset(Dataset):
    def __init__(self, feat, mask, labels):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"feat": self.feat[i], "mask": self.mask[i], "label": self.labels[i]}


class UnmergeDataset(Dataset):
    def __init__(self, feat_hlt, mask_hlt, hlt_const, constituents_off, samples, max_count, tgt_mean, tgt_std):
        self.feat_hlt = feat_hlt
        self.mask_hlt = mask_hlt
        self.hlt_const = hlt_const
        self.constituents_off = constituents_off
        self.samples = samples
        self.max_count = max_count
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        jet_idx, token_idx, origin, pred_count = self.samples[i]
        true_count = min(len(origin), self.max_count)
        origin = origin[:true_count]
        target = self.constituents_off[jet_idx, origin, :4].astype(np.float32)
        target = (target - self.tgt_mean) / self.tgt_std
        target = np.clip(target, -10, 10)
        target_pad = np.zeros((self.max_count, 4), dtype=np.float32)
        target_pad[:true_count] = target
        return {
            "hlt": torch.tensor(self.feat_hlt[jet_idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask_hlt[jet_idx], dtype=torch.bool),
            "token_idx": torch.tensor(token_idx, dtype=torch.long),
            "pred_count": torch.tensor(min(pred_count, self.max_count), dtype=torch.long),
            "true_count": torch.tensor(true_count, dtype=torch.long),
            "target": torch.tensor(target_pad, dtype=torch.float32),
            "hlt_token": torch.tensor(self.hlt_const[jet_idx, token_idx, :4], dtype=torch.float32),
        }


class JetPairDataset(Dataset):
    def __init__(self, off_std, hlt_std, mask_off, mask_hlt):
        self.off = torch.tensor(off_std, dtype=torch.float32)
        self.hlt = torch.tensor(hlt_std, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)

    def __len__(self):
        return len(self.off)

    def __getitem__(self, i):
        return {"off": self.off[i], "hlt": self.hlt[i], "mask": self.mask_hlt[i]}


# ----------------------------- Models ----------------------------- #
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
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
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

    def forward(self, x, mask):
        b, n, _ = x.shape
        h = self.input_proj(x.view(-1, self.input_dim))
        h = h.view(b, n, -1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        query = self.pool_query.expand(b, -1, -1)
        pooled, _ = self.pool_attn(query, h, h, key_padding_mask=~mask, need_weights=False)
        z = self.norm(pooled.squeeze(1))
        return self.classifier(z)


class MergeCountPredictor(nn.Module):
    def __init__(self, input_dim=7, num_classes=10, embed_dim=128, num_heads=8, num_layers=6, ff_dim=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, max(embed_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(embed_dim // 2, 32), num_classes),
        )

    def forward(self, x, mask):
        b, n, _ = x.shape
        h = self.input_proj(x.view(-1, self.input_dim))
        h = h.view(b, n, -1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.head(h)


class UnmergePredictor(nn.Module):
    def __init__(self, input_dim, max_count, embed_dim, num_heads, num_layers, decoder_layers, ff_dim, dropout, count_embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.max_count = max_count
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.count_embed = nn.Embedding(max_count + 1, count_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(embed_dim * 2 + count_embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.query = nn.Parameter(torch.randn(max_count, embed_dim) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        out_dim = 8 if CONFIG["unmerge_training"]["distributional"] else 4
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x, mask, token_idx, count):
        b, n, _ = x.shape
        h = self.input_proj(x.view(-1, self.input_dim))
        h = h.view(b, n, -1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        idx = token_idx.view(-1, 1, 1).expand(-1, 1, self.embed_dim)
        h_t = h.gather(1, idx).squeeze(1)
        h_sum = (h * mask.unsqueeze(-1)).sum(dim=1)
        h_avg = h_sum / mask.sum(dim=1, keepdim=True).clamp(min=1)
        c_emb = self.count_embed(count)
        cond = self.cond_proj(torch.cat([h_t, h_avg, c_emb], dim=1))
        queries = self.query.unsqueeze(0).expand(b, -1, -1) + cond.unsqueeze(1)
        dec = self.decoder(queries, h, memory_key_padding_mask=~mask)
        out = self.out(dec)
        mean = out[..., :4]
        if CONFIG["unmerge_training"]["distributional"]:
            logvar = out[..., 4:]
        else:
            logvar = torch.zeros_like(mean)
        return mean, logvar


# ----------------------------- Matching losses ----------------------------- #
def greedy_match(cost):
    k = cost.shape[0]
    rows, cols = [], []
    used_r, used_c = set(), set()
    for _ in range(k):
        min_val, min_r, min_c = 1e18, -1, -1
        for r in range(k):
            if r in used_r:
                continue
            for c in range(k):
                if c in used_c:
                    continue
                if cost[r, c] < min_val:
                    min_val, min_r, min_c = cost[r, c], r, c
        used_r.add(min_r)
        used_c.add(min_c)
        rows.append(min_r)
        cols.append(min_c)
    return np.array(rows), np.array(cols)


def hungarian_match(cost):
    if _HAS_SCIPY:
        r, c = linear_sum_assignment(cost)
        return r, c
    return greedy_match(cost)


def set_chamfer_loss(preds, targets, true_counts):
    total = 0.0
    for i in range(preds.size(0)):
        k = int(true_counts[i].item())
        pred_i = preds[i, :k]
        tgt_i = targets[i, :k]
        dist = torch.cdist(pred_i, tgt_i, p=1)
        loss_i = dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()
        total += loss_i
    return total / max(preds.size(0), 1)


def matched_nll_loss(mu, logvar, target, true_counts):
    total = 0.0
    for i in range(mu.size(0)):
        k = int(true_counts[i].item())
        mu_i = mu[i, :k]
        lv_i = logvar[i, :k]
        tgt_i = target[i, :k]
        cost = torch.cdist(mu_i, tgt_i, p=1).detach().cpu().numpy()
        r, c = hungarian_match(cost)
        mu_m = mu_i[r]
        lv_m = lv_i[r]
        tgt_m = tgt_i[c]
        var = torch.exp(lv_m)
        nll = 0.5 * (((tgt_m - mu_m) ** 2) / var + lv_m).sum(dim=1)
        total += nll.mean()
    return total / max(mu.size(0), 1)


def matched_l1_loss(mu, target, true_counts):
    total = 0.0
    for i in range(mu.size(0)):
        k = int(true_counts[i].item())
        mu_i = mu[i, :k]
        tgt_i = target[i, :k]
        cost = torch.cdist(mu_i, tgt_i, p=1).detach().cpu().numpy()
        r, c = hungarian_match(cost)
        total += F.l1_loss(mu_i[r], tgt_i[c])
    return total / max(mu.size(0), 1)


def physics_loss(mu, true_counts, hlt_token):
    total = 0.0
    for i in range(mu.size(0)):
        k = int(true_counts[i].item())
        pred = mu[i, :k]
        pt = pred[:, 0]
        eta = pred[:, 1]
        phi = pred[:, 2]
        E = pred[:, 3]
        px = (pt * torch.cos(phi)).sum()
        py = (pt * torch.sin(phi)).sum()
        pz = (pt * torch.sinh(eta)).sum()
        E_sum = E.sum()
        pt_sum = torch.sqrt(px ** 2 + py ** 2 + 1e-8)
        p_sum = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + 1e-8)
        eta_sum = 0.5 * torch.log(torch.clamp((p_sum + pz) / (p_sum - pz + 1e-8), 1e-8, 1e8))
        phi_sum = torch.atan2(py, px)
        pred_vec = torch.stack([pt_sum, eta_sum, phi_sum, E_sum], dim=0)
        total += F.l1_loss(pred_vec, hlt_token[i])
    return total / max(mu.size(0), 1)


# ----------------------------- Diffusion model ----------------------------- #
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def make_beta_schedule(timesteps, schedule):
    if schedule == "cosine":
        return cosine_beta_schedule(timesteps)
    return linear_beta_schedule(timesteps)


def compute_snr(alpha_bar_t):
    return alpha_bar_t / torch.clamp(1.0 - alpha_bar_t, min=1e-8)


def predict_x0_from_eps(x_t, eps, alpha_bar_t):
    return (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)


def predict_eps_from_x0(x_t, x0, alpha_bar_t):
    return (x_t - torch.sqrt(alpha_bar_t) * x0) / torch.sqrt(1.0 - alpha_bar_t)


def predict_v(x0, eps, alpha_bar_t):
    return torch.sqrt(alpha_bar_t) * eps - torch.sqrt(1.0 - alpha_bar_t) * x0


def predict_x0_from_v(x_t, v, alpha_bar_t):
    return torch.sqrt(alpha_bar_t) * x_t - torch.sqrt(1.0 - alpha_bar_t) * v


def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / (half - 1)).to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ConditionalDenoiser(nn.Module):
    def __init__(self, input_dim=4, embed_dim=256, num_heads=8, num_layers=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.x_proj = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.GELU(), nn.Dropout(dropout))
        self.c_proj = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.GELU(), nn.Dropout(dropout))
        self.sc_proj = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.GELU(), nn.Dropout(dropout))
        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        dec = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cond_encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_layers)
        self.out = nn.Linear(embed_dim, input_dim)

    def forward(self, x_t, cond, mask, t, self_cond=None):
        if cond is None:
            cond = torch.zeros_like(x_t)
        h = self.x_proj(x_t) + self.c_proj(cond)
        if self_cond is not None:
            h = h + self.sc_proj(self_cond)
        t_emb = self.t_mlp(get_timestep_embedding(t, self.embed_dim))
        h = h + t_emb.unsqueeze(1)
        mem = self.cond_encoder(self.c_proj(cond), src_key_padding_mask=~mask)
        h = self.decoder(h, mem, tgt_key_padding_mask=~mask, memory_key_padding_mask=~mask)
        return self.out(h)


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.detach()

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


# ----------------------------- Diffusion training/sampling ----------------------------- #
def q_sample(x0, t, noise, alpha_bar):
    a_bar = alpha_bar[t].view(-1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise


def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask.unsqueeze(-1)
    denom = mask.sum() * pred.shape[-1]
    return diff.sum() / torch.clamp(denom, min=1.0)


def jet_summary(x, mask):
    pt = x[:, :, 0]
    eta = x[:, :, 1]
    phi = x[:, :, 2]
    E = x[:, :, 3]
    m = mask.float()
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    jet_px = (px * m).sum(dim=1)
    jet_py = (py * m).sum(dim=1)
    jet_pz = (pz * m).sum(dim=1)
    jet_E = (E * m).sum(dim=1)
    jet_pt = torch.sqrt(jet_px ** 2 + jet_py ** 2 + 1e-8)
    jet_p = torch.sqrt(jet_px ** 2 + jet_py ** 2 + jet_pz ** 2 + 1e-8)
    jet_eta = 0.5 * torch.log(torch.clamp((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = torch.atan2(jet_py, jet_px)
    return torch.stack([jet_pt, jet_eta, jet_phi, jet_E], dim=1)


def train_diffusion_epoch(model, ema, loader, opt, device, alpha_bar):
    model.train()
    total = 0.0
    count = 0
    T = alpha_bar.shape[0]
    for batch in loader:
        x0 = batch["off"].to(device)
        cond = batch["hlt"].to(device)
        mask = batch["mask"].to(device)
        x0 = torch.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
        cond = torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)

        if CONFIG["diffusion"]["cond_drop_prob"] > 0:
            drop = torch.rand(x0.size(0), device=device) < CONFIG["diffusion"]["cond_drop_prob"]
            if drop.any():
                cond = cond.clone()
                cond[drop] = 0.0

        t = torch.randint(0, T, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, noise, alpha_bar)

        self_cond = None
        if CONFIG["diffusion"]["self_cond_prob"] > 0:
            if torch.rand(()) < CONFIG["diffusion"]["self_cond_prob"]:
                with torch.no_grad():
                    pred0 = model(x_t, cond, mask, t, self_cond=None)
                    a_bar = alpha_bar[t].view(-1, 1, 1)
                    if CONFIG["diffusion"]["pred_type"] == "x0":
                        x0_sc = pred0
                    elif CONFIG["diffusion"]["pred_type"] == "v":
                        x0_sc = predict_x0_from_v(x_t, pred0, a_bar)
                    else:
                        x0_sc = predict_x0_from_eps(x_t, pred0, a_bar)
                    self_cond = x0_sc.detach()

        opt.zero_grad()
        pred = model(x_t, cond, mask, t, self_cond=self_cond)
        a_bar = alpha_bar[t].view(-1, 1, 1)
        if CONFIG["diffusion"]["pred_type"] == "x0":
            target = x0
            pred_x0 = pred
        elif CONFIG["diffusion"]["pred_type"] == "v":
            target = predict_v(x0, noise, a_bar)
            pred_x0 = predict_x0_from_v(x_t, pred, a_bar)
        else:
            target = noise
            pred_x0 = predict_x0_from_eps(x_t, pred, a_bar)

        loss_noise = masked_mse(pred, target, mask)
        if CONFIG["diffusion"]["snr_weight"]:
            snr = compute_snr(a_bar)
            w = torch.clamp(snr, max=CONFIG["diffusion"]["snr_gamma"]) / (snr + 1.0)
            loss_noise = loss_noise * w.mean()

        loss = loss_noise
        if CONFIG["diffusion"]["x0_weight"] > 0:
            loss = loss + CONFIG["diffusion"]["x0_weight"] * masked_mse(pred_x0, x0, mask)
        if CONFIG["diffusion"]["jet_loss_weight"] > 0:
            loss = loss + CONFIG["diffusion"]["jet_loss_weight"] * F.l1_loss(jet_summary(pred_x0, mask), jet_summary(x0, mask))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        total += loss.item() * x0.size(0)
        count += x0.size(0)
    return total / max(count, 1)


@torch.no_grad()
def model_pred_eps(model, x, cond, mask, t, alpha_bar):
    if CONFIG["sampling"]["guidance_scale"] != 1.0:
        eps_cond = model(x, cond, mask, t)
        eps_uncond = model(x, torch.zeros_like(cond), mask, t)
        eps = eps_uncond + CONFIG["sampling"]["guidance_scale"] * (eps_cond - eps_uncond)
    else:
        eps = model(x, cond, mask, t)
    a_bar = alpha_bar[t].view(-1, 1, 1)
    if CONFIG["diffusion"]["pred_type"] == "x0":
        eps = predict_eps_from_x0(x, eps, a_bar)
    elif CONFIG["diffusion"]["pred_type"] == "v":
        x0 = predict_x0_from_v(x, eps, a_bar)
        eps = predict_eps_from_x0(x, x0, a_bar)
    return eps


@torch.no_grad()
def sample_ddim(model, cond, mask, betas, alpha, alpha_bar, steps):
    T = betas.shape[0]
    if steps < T:
        idx = torch.linspace(T - 1, 0, steps, device=cond.device).long()
    else:
        idx = torch.arange(T - 1, -1, -1, device=cond.device)
    x = torch.randn_like(cond)
    for i, t in enumerate(idx):
        t_batch = torch.full((x.size(0),), t, device=cond.device, dtype=torch.long)
        eps = model_pred_eps(model, x, cond, mask, t_batch, alpha_bar)
        a_bar_t = alpha_bar[t]
        x0 = predict_x0_from_eps(x, eps, a_bar_t)
        if i == len(idx) - 1:
            x = x0
            break
        t_next = idx[i + 1]
        a_bar_next = alpha_bar[t_next]
        x = torch.sqrt(a_bar_next) * x0 + torch.sqrt(1 - a_bar_next) * eps
        x = x * mask.unsqueeze(-1)
    return x


@torch.no_grad()
def generate_unsmeared(model, hlt_std, mask_hlt, betas, alpha, alpha_bar, device):
    model.eval()
    out = np.zeros_like(hlt_std)
    ds = JetDataset(hlt_std, mask_hlt, np.zeros(hlt_std.shape[0], dtype=np.float32))
    loader = DataLoader(ds, batch_size=CONFIG["training"]["batch_size"], shuffle=False)
    idx = 0
    for batch in tqdm(loader, desc="UnsmearedGen"):
        x = batch["feat"].to(device)
        m = batch["mask"].to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        preds = []
        for _ in range(CONFIG["sampling"]["n_samples_eval"]):
            if CONFIG["sampling"]["method"] == "ddim":
                x0 = sample_ddim(model, x, m, betas, alpha, alpha_bar, CONFIG["sampling"]["sample_steps"])
            else:
                x0 = sample_ddim(model, x, m, betas, alpha, alpha_bar, CONFIG["sampling"]["sample_steps"])
            preds.append(x0)
        x0 = torch.stack(preds, dim=0).mean(dim=0)
        bs = x0.size(0)
        out[idx:idx + bs] = x0.cpu().numpy()
        idx += bs
    return out


# ----------------------------- Training helpers ----------------------------- #
def train_classifier(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    preds, labs = [], []
    for batch in loader:
        x = batch["feat"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].to(device)
        opt.zero_grad()
        logits = model(x, m).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(y)
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else 0.0
    return total_loss / len(preds), auc


@torch.no_grad()
def eval_classifier(model, loader, device):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        x = batch["feat"].to(device)
        m = batch["mask"].to(device)
        logits = model(x, m).squeeze(1)
        preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        labs.extend(batch["label"].cpu().numpy().flatten())
    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else 0.0
    return auc, preds, labs


def compute_class_weights(labels, mask, num_classes):
    valid = labels[mask]
    counts = np.bincount(valid, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.ones(num_classes, dtype=np.float64)
    if total > 0:
        weights = total / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
    return weights


def train_merge_count(model, loader, opt, device, class_weights):
    model.train()
    total_loss = 0.0
    preds, labs = [], []
    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    for batch in loader:
        x = batch["feat"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].to(device)
        opt.zero_grad()
        logits = model(x, m)
        loss = F.cross_entropy(logits[m], y[m], weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=2)
        preds.extend(pred[m].detach().cpu().numpy().flatten())
        labs.extend(y[m].detach().cpu().numpy().flatten())
    acc = (np.array(preds) == np.array(labs)).mean() if len(labs) else 0.0
    return total_loss / max(len(loader), 1), acc


@torch.no_grad()
def eval_merge_count(model, loader, device):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        x = batch["feat"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].to(device)
        logits = model(x, m)
        pred = logits.argmax(dim=2)
        preds.extend(pred[m].detach().cpu().numpy().flatten())
        labs.extend(y[m].detach().cpu().numpy().flatten())
    acc = (np.array(preds) == np.array(labs)).mean() if len(labs) else 0.0
    return acc


@torch.no_grad()
def predict_counts(model, feat, mask, batch_size, device, max_count):
    model.eval()
    preds = np.zeros(mask.shape, dtype=np.int64)
    loader = DataLoader(MergeCountDataset(feat, mask, np.zeros(mask.shape, dtype=np.int64)),
                        batch_size=batch_size, shuffle=False)
    idx = 0
    for batch in loader:
        x = batch["feat"].to(device)
        m = batch["mask"].to(device)
        logits = model(x, m)
        pred_cls = logits.argmax(dim=2).cpu().numpy()
        bs = pred_cls.shape[0]
        preds[idx:idx + bs] = pred_cls + 1
        preds[idx:idx + bs][~mask[idx:idx + bs]] = 0
        idx += bs
    preds = np.clip(preds, 0, max_count)
    return preds


def build_unmerged_view(feat_hlt_std, hlt_mask, hlt_const, counts, unmerge_model, tgt_mean, tgt_std, max_count, max_constits, device, batch_size):
    n_jets, max_part, _ = hlt_const.shape
    pred_map = {}
    samples = []
    for j in range(n_jets):
        for idx in range(max_part):
            if hlt_mask[j, idx] and counts[j, idx] > 1:
                samples.append((j, idx, int(counts[j, idx])))
    if len(samples) > 0:
        unmerge_model.eval()
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                chunk = samples[i:i + batch_size]
                jet_idx = [c[0] for c in chunk]
                tok_idx = [c[1] for c in chunk]
                cnt = [c[2] for c in chunk]
                x = torch.tensor(feat_hlt_std[jet_idx], dtype=torch.float32, device=device)
                m = torch.tensor(hlt_mask[jet_idx], dtype=torch.bool, device=device)
                token_idx = torch.tensor(tok_idx, dtype=torch.long, device=device)
                count = torch.tensor(cnt, dtype=torch.long, device=device)
                mu, _ = unmerge_model(x, m, token_idx, count)
                preds = mu.cpu().numpy()
                for k in range(len(chunk)):
                    c = cnt[k]
                    pred = preds[k, :c]
                    pred = pred * tgt_std + tgt_mean
                    pred[:, 0] = np.clip(pred[:, 0], 0.0, None)
                    pred[:, 1] = np.clip(pred[:, 1], -5.0, 5.0)
                    pred[:, 2] = np.arctan2(np.sin(pred[:, 2]), np.cos(pred[:, 2]))
                    pred[:, 3] = pred[:, 0] * np.cosh(np.clip(pred[:, 1], -5.0, 5.0))
                    pred_map[(chunk[k][0], chunk[k][1])] = pred

    new_const = np.zeros((n_jets, max_constits, 4), dtype=np.float32)
    new_mask = np.zeros((n_jets, max_constits), dtype=bool)
    for j in range(n_jets):
        parts = []
        for idx in range(max_part):
            if not hlt_mask[j, idx]:
                continue
            if counts[j, idx] <= 1:
                parts.append(hlt_const[j, idx])
            else:
                pred = pred_map.get((j, idx))
                if pred is not None:
                    parts.extend(list(pred))
        if len(parts) == 0:
            continue
        parts = np.array(parts, dtype=np.float32)
        order = np.argsort(parts[:, 0])[::-1]
        parts = parts[order]
        n_keep = min(len(parts), max_constits)
        new_const[j, :n_keep] = parts[:n_keep]
        new_mask[j, :n_keep] = True
    return new_const, new_mask


def save_roc_plots(save_dir, labs, curves):
    if len(np.unique(labs)) <= 1:
        print("Warning: ROC plots skipped (only one class present).")
        return
    # All models
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(tpr, fpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "roc_all.png", dpi=200)
    plt.close()

    # Pairwise plots
    pairs = [("Teacher", "Baseline"), ("Teacher", "Final"), ("Baseline", "Final")]
    for a, b in pairs:
        if a not in curves or b not in curves:
            continue
        plt.figure(figsize=(7, 6))
        for name in (a, b):
            fpr, tpr, auc = curves[name]
            plt.plot(tpr, fpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
        plt.xlabel("True Positive Rate")
        plt.ylabel("False Positive Rate")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"roc_{a.lower()}_{b.lower()}.png", dpi=200)
        plt.close()


def compute_unmerge_loss(mu, logvar, target, true_count, hlt_token):
    if CONFIG["unmerge_training"]["loss_type"] == "hungarian":
        loss_nll = matched_nll_loss(mu, logvar, target, true_count)
        loss_l1 = matched_l1_loss(mu, target, true_count)
        loss = CONFIG["unmerge_training"]["nll_weight"] * loss_nll + (1.0 - CONFIG["unmerge_training"]["nll_weight"]) * loss_l1
    else:
        loss = set_chamfer_loss(mu, target, true_count)
    if CONFIG["unmerge_training"]["physics_weight"] > 0:
        loss = loss + CONFIG["unmerge_training"]["physics_weight"] * physics_loss(mu, true_count, hlt_token)
    return loss


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--max_merge_count", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "full_unmerge_unsmear"))
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_save_models", action="store_true")
    # Unmerge knobs
    parser.add_argument("--no_distributional", action="store_true")
    parser.add_argument("--no_curriculum", action="store_true")
    parser.add_argument("--no_true_count", action="store_true")
    parser.add_argument("--unmerge_loss", type=str, choices=["hungarian", "chamfer"], default=None)
    # Diffusion knobs
    parser.add_argument("--pred_type", type=str, choices=["eps", "x0", "v"], default=None)
    parser.add_argument("--no_snr_weight", action="store_true")
    parser.add_argument("--self_cond_prob", type=float, default=None)
    parser.add_argument("--cond_drop_prob", type=float, default=None)
    parser.add_argument("--jet_loss_weight", type=float, default=None)
    # Sampling knobs
    parser.add_argument("--sampling_method", type=str, choices=["ddim", "ddpm"], default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    args = parser.parse_args()

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    # Apply CLI overrides
    if args.no_distributional:
        CONFIG["unmerge_training"]["distributional"] = False
        CONFIG["unmerge_training"]["nll_weight"] = 0.0
    if args.no_curriculum:
        CONFIG["unmerge_training"]["curriculum"] = False
    if args.no_true_count:
        CONFIG["unmerge_training"]["use_true_count"] = False
    if args.unmerge_loss is not None:
        CONFIG["unmerge_training"]["loss_type"] = args.unmerge_loss
    if args.pred_type is not None:
        CONFIG["diffusion"]["pred_type"] = args.pred_type
    if args.no_snr_weight:
        CONFIG["diffusion"]["snr_weight"] = False
    if args.self_cond_prob is not None:
        CONFIG["diffusion"]["self_cond_prob"] = args.self_cond_prob
    if args.cond_drop_prob is not None:
        CONFIG["diffusion"]["cond_drop_prob"] = args.cond_drop_prob
    if args.jet_loss_weight is not None:
        CONFIG["diffusion"]["jet_loss_weight"] = args.jet_loss_weight
    if args.sampling_method is not None:
        CONFIG["sampling"]["method"] = args.sampling_method
    if args.guidance_scale is not None:
        CONFIG["sampling"]["guidance_scale"] = args.guidance_scale

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
    const_raw = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    print("Applying HLT effects (full)...")
    hlt_const, hlt_mask, origin_counts, origin_lists, stats = apply_hlt_effects_with_tracking(
        const_raw, mask_raw, CONFIG, seed=RANDOM_SEED
    )
    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (const_raw[:, :, 0] >= pt_threshold_off)
    const_off = const_raw.copy()
    const_off[~masks_off] = 0

    print("HLT Simulation Statistics:")
    print(f"  Offline particles: {stats['n_initial']:,}")
    print(f"  Lost to pT threshold ({CONFIG['hlt_effects']['pt_threshold_hlt']}): {stats['n_lost_threshold']:,}")
    print(f"  Lost to merging (dR<{CONFIG['hlt_effects']['merge_radius']}): {stats['n_merged']:,}")
    print(f"  Lost to efficiency: {stats['n_lost_eff']:,}")
    print(f"  HLT particles: {stats['n_final']:,}")
    print(f"  Avg per jet: Offline={masks_off.sum(axis=1).mean():.1f}, HLT={hlt_mask.sum(axis=1).mean():.1f}")

    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Offline stats
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)
    feat_means, feat_stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, feat_means, feat_stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, feat_means, feat_stds)

    # ------------------- Teacher ------------------- #
    print("\n" + "=" * 70)
    print("STEP 1: TEACHER (Offline)")
    print("=" * 70)
    BS = CONFIG["training"]["batch_size"]
    train_off = JetDataset(feat_off_std[train_idx], masks_off[train_idx], all_labels[train_idx])
    val_off = JetDataset(feat_off_std[val_idx], masks_off[val_idx], all_labels[val_idx])
    test_off = JetDataset(feat_off_std[test_idx], masks_off[test_idx], all_labels[test_idx])
    train_off_loader = DataLoader(train_off, batch_size=BS, shuffle=True, drop_last=True)
    val_off_loader = DataLoader(val_off, batch_size=BS, shuffle=False)
    test_off_loader = DataLoader(test_off, batch_size=BS, shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_t = get_scheduler(opt_t, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Teacher"):
        _, train_auc = train_classifier(teacher, train_off_loader, opt_t, device)
        val_auc, _, _ = eval_classifier(teacher, val_off_loader, device)
        sch_t.step()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping teacher at epoch {ep+1}")
            break
    if best_state is not None:
        teacher.load_state_dict(best_state)
    auc_teacher, preds_teacher, labs = eval_classifier(teacher, test_off_loader, device)

    # ------------------- Baseline ------------------- #
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE (HLT)")
    print("=" * 70)
    train_hlt = JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], all_labels[train_idx])
    val_hlt = JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], all_labels[val_idx])
    test_hlt = JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], all_labels[test_idx])
    train_hlt_loader = DataLoader(train_hlt, batch_size=BS, shuffle=True, drop_last=True)
    val_hlt_loader = DataLoader(val_hlt, batch_size=BS, shuffle=False)
    test_hlt_loader = DataLoader(test_hlt, batch_size=BS, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_b = get_scheduler(opt_b, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Baseline"):
        _, train_auc = train_classifier(baseline, train_hlt_loader, opt_b, device)
        val_auc, _, _ = eval_classifier(baseline, val_hlt_loader, device)
        sch_b.step()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping baseline at epoch {ep+1}")
            break
    if best_state is not None:
        baseline.load_state_dict(best_state)
    auc_baseline, preds_baseline, _ = eval_classifier(baseline, test_hlt_loader, device)

    # ------------------- Merge count predictor ------------------- #
    print("\n" + "=" * 70)
    print("STEP 3: MERGE COUNT PREDICTOR")
    print("=" * 70)
    max_count = max(int(args.max_merge_count), 2)
    count_label = np.clip(origin_counts, 1, max_count) - 1
    train_cnt = MergeCountDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], count_label[train_idx])
    val_cnt = MergeCountDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], count_label[val_idx])
    BS_cnt = CONFIG["merge_count_training"]["batch_size"]
    train_cnt_loader = DataLoader(train_cnt, batch_size=BS_cnt, shuffle=True, drop_last=True)
    val_cnt_loader = DataLoader(val_cnt, batch_size=BS_cnt, shuffle=False)
    count_model = MergeCountPredictor(input_dim=7, num_classes=max_count, **CONFIG["merge_count_model"]).to(device)
    opt_c = torch.optim.AdamW(count_model.parameters(), lr=CONFIG["merge_count_training"]["lr"], weight_decay=CONFIG["merge_count_training"]["weight_decay"])
    sch_c = get_scheduler(opt_c, CONFIG["merge_count_training"]["warmup_epochs"], CONFIG["merge_count_training"]["epochs"])
    class_weights = compute_class_weights(count_label[train_idx], hlt_mask[train_idx], max_count)
    best_acc, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["merge_count_training"]["epochs"]), desc="MergeCount"):
        _, train_acc = train_merge_count(count_model, train_cnt_loader, opt_c, device, class_weights)
        val_acc = eval_merge_count(count_model, val_cnt_loader, device)
        sch_c.step()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in count_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, best={best_acc:.4f}")
        if no_improve >= CONFIG["merge_count_training"]["patience"]:
            print(f"Early stopping merge-count at epoch {ep+1}")
            break
    if best_state is not None:
        count_model.load_state_dict(best_state)

    pred_counts = predict_counts(count_model, feat_hlt_std, hlt_mask, BS_cnt, device, max_count)

    # ------------------- Unmerger training ------------------- #
    print("\n" + "=" * 70)
    print("STEP 4: UNMERGER (distributional)")
    print("=" * 70)
    samples = []
    for j in range(len(all_labels)):
        for idx in range(args.max_constits):
            origin = origin_lists[j][idx]
            if hlt_mask[j, idx] and len(origin) > 1:
                if len(origin) > max_count:
                    continue
                pc = int(pred_counts[j, idx])
                if pc < 2:
                    pc = 2
                if pc > max_count:
                    pc = max_count
                samples.append((j, idx, origin, pc))

    train_idx_set = set(train_idx)
    val_idx_set = set(val_idx)
    test_idx_set = set(test_idx)
    train_samples = [s for s in samples if s[0] in train_idx_set]
    val_samples = [s for s in samples if s[0] in val_idx_set]
    test_samples = [s for s in samples if s[0] in test_idx_set]
    print(f"Merged samples: train={len(train_samples):,}, val={len(val_samples):,}, test={len(test_samples):,}")
    if len(train_samples) == 0:
        raise RuntimeError("No merged samples in training split.")

    train_targets = [const_off[s[0], s[2], :4] for s in train_samples]
    flat_train = np.concatenate(train_targets, axis=0)
    tgt_mean = flat_train.mean(axis=0)
    tgt_std = flat_train.std(axis=0) + 1e-8

    BS_un = CONFIG["unmerge_training"]["batch_size"]
    train_un = UnmergeDataset(feat_hlt_std, hlt_mask, hlt_const, const_off, train_samples, max_count, tgt_mean, tgt_std)
    val_un = UnmergeDataset(feat_hlt_std, hlt_mask, hlt_const, const_off, val_samples, max_count, tgt_mean, tgt_std)
    test_un = UnmergeDataset(feat_hlt_std, hlt_mask, hlt_const, const_off, test_samples, max_count, tgt_mean, tgt_std)
    train_un_loader = DataLoader(train_un, batch_size=BS_un, shuffle=True, drop_last=True)
    val_un_loader = DataLoader(val_un, batch_size=BS_un, shuffle=False)
    test_un_loader = DataLoader(test_un, batch_size=BS_un, shuffle=False)

    unmerge_model = UnmergePredictor(input_dim=7, max_count=max_count, **CONFIG["unmerge_model"]).to(device)
    opt_u = torch.optim.AdamW(unmerge_model.parameters(), lr=CONFIG["unmerge_training"]["lr"], weight_decay=CONFIG["unmerge_training"]["weight_decay"])
    sch_u = get_scheduler(opt_u, CONFIG["unmerge_training"]["warmup_epochs"], CONFIG["unmerge_training"]["epochs"])
    best_val, best_state, no_improve = 1e9, None, 0

    for ep in tqdm(range(CONFIG["unmerge_training"]["epochs"]), desc="Unmerge"):
        unmerge_model.train()
        total_loss = 0.0
        n_batches = 0
        if CONFIG["unmerge_training"]["curriculum"]:
            frac = min(1.0, (ep + 1) / max(CONFIG["unmerge_training"]["curriculum_epochs"], 1))
            curr_max = int(CONFIG["unmerge_training"]["curriculum_start"] + frac * (max_count - CONFIG["unmerge_training"]["curriculum_start"]))
        else:
            curr_max = max_count
        for batch in train_un_loader:
            x = batch["hlt"].to(device)
            mask = batch["mask"].to(device)
            token_idx = batch["token_idx"].to(device)
            true_count = batch["true_count"].to(device)
            target = batch["target"].to(device)
            hlt_token = batch["hlt_token"].to(device)
            count_in = true_count.clamp(min=2, max=max_count) if CONFIG["unmerge_training"]["use_true_count"] else batch["pred_count"].to(device)
            if curr_max < max_count:
                keep = true_count <= curr_max
                if keep.sum().item() == 0:
                    continue
                x = x[keep]
                mask = mask[keep]
                token_idx = token_idx[keep]
                count_in = count_in[keep]
                true_count = true_count[keep]
                target = target[keep]
                hlt_token = hlt_token[keep]
            opt_u.zero_grad()
            mu, logvar = unmerge_model(x, mask, token_idx, count_in)
            loss = compute_unmerge_loss(mu, logvar, target, true_count, hlt_token)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unmerge_model.parameters(), 1.0)
            opt_u.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        val_loss = 0.0
        n_batches = 0
        unmerge_model.eval()
        with torch.no_grad():
            for batch in val_un_loader:
                x = batch["hlt"].to(device)
                mask = batch["mask"].to(device)
                token_idx = batch["token_idx"].to(device)
                true_count = batch["true_count"].to(device)
                target = batch["target"].to(device)
                hlt_token = batch["hlt_token"].to(device)
                # Use predicted counts for val/test
                count_in = batch["pred_count"].to(device)
                mu, logvar = unmerge_model(x, mask, token_idx, count_in)
                loss = compute_unmerge_loss(mu, logvar, target, true_count, hlt_token)
                val_loss += loss.item()
                n_batches += 1
        val_loss = val_loss / max(n_batches, 1)
        sch_u.step()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in unmerge_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best={best_val:.4f}")
        if no_improve >= CONFIG["unmerge_training"]["patience"]:
            print(f"Early stopping unmerge at epoch {ep+1}")
            break
    if best_state is not None:
        unmerge_model.load_state_dict(best_state)

    # ------------------- Build unmerged dataset ------------------- #
    print("\n" + "=" * 70)
    print("STEP 5: BUILD UNMERGED DATASETS")
    print("=" * 70)
    # Train: use true counts; Val/Test: use predicted counts
    counts_train = origin_counts.copy()
    counts_train = np.clip(counts_train, 0, max_count)
    counts_valtest = pred_counts

    unmerged_const_train, unmerged_mask_train = build_unmerged_view(
        feat_hlt_std, hlt_mask, hlt_const, counts_train,
        unmerge_model, tgt_mean, tgt_std, max_count, args.max_constits, device, BS_un
    )
    unmerged_const_val, unmerged_mask_val = build_unmerged_view(
        feat_hlt_std, hlt_mask, hlt_const, counts_valtest,
        unmerge_model, tgt_mean, tgt_std, max_count, args.max_constits, device, BS_un
    )

    # ------------------- Train unsmear diffusion ------------------- #
    print("\n" + "=" * 70)
    print("STEP 6: UNSMEAR DIFFUSION")
    print("=" * 70)
    smear_only, smear_mask = apply_smear_only(const_off, masks_off, CONFIG, seed=RANDOM_SEED + 123)
    const_means, const_stds = get_stats(const_off, masks_off, train_idx)
    off_std = standardize(const_off, masks_off, const_means, const_stds)
    smear_std = standardize(smear_only, smear_mask, const_means, const_stds)
    train_pair = JetPairDataset(off_std[train_idx], smear_std[train_idx], masks_off[train_idx], smear_mask[train_idx])
    val_pair = JetPairDataset(off_std[val_idx], smear_std[val_idx], masks_off[val_idx], smear_mask[val_idx])
    train_pair_loader = DataLoader(train_pair, batch_size=CONFIG["training"]["batch_size"], shuffle=True, drop_last=True)
    val_pair_loader = DataLoader(val_pair, batch_size=CONFIG["training"]["batch_size"], shuffle=False)

    diff_model = ConditionalDenoiser(input_dim=4, embed_dim=256, num_heads=8, num_layers=8, ff_dim=1024, dropout=0.1).to(device)
    ema = EMA(diff_model, decay=CONFIG["diffusion"]["ema_decay"])
    betas = torch.tensor(make_beta_schedule(CONFIG["diffusion"]["timesteps"], CONFIG["diffusion"]["schedule"]), dtype=torch.float32, device=device)
    alpha = 1.0 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)
    opt_d = torch.optim.AdamW(diff_model.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_d = get_scheduler(opt_d, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_val = 1e9
    best_state = None
    no_improve = 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Diffusion"):
        loss = train_diffusion_epoch(diff_model, ema, train_pair_loader, opt_d, device, alpha_bar)
        sch_d.step()
        if (ep + 1) % 5 == 0:
            # quick val L1
            ema_model = ConditionalDenoiser(input_dim=4, embed_dim=256, num_heads=8, num_layers=8, ff_dim=1024, dropout=0.1).to(device)
            ema.apply_to(ema_model)
            val_loss = 0.0
            count = 0
            for batch in val_pair_loader:
                x0 = batch["off"].to(device)
                cond = batch["hlt"].to(device)
                mask = batch["mask"].to(device)
                x0_pred = sample_ddim(ema_model, cond, mask, betas, alpha, alpha_bar, CONFIG["sampling"]["sample_steps"])
                val_loss += F.l1_loss(x0_pred, x0, reduction="sum").item()
                count += x0.numel()
            val_loss = val_loss / max(count, 1)
            print(f"Ep {ep+1}: train_loss={loss:.6f}, val_l1={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in diff_model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= CONFIG["training"]["patience"]:
                print(f"Early stopping diffusion at epoch {ep+1}")
                break
    if best_state is not None:
        diff_model.load_state_dict(best_state)
        ema.shadow = {k: v.detach().clone() for k, v in diff_model.state_dict().items()}

    ema_model = ConditionalDenoiser(input_dim=4, embed_dim=256, num_heads=8, num_layers=8, ff_dim=1024, dropout=0.1).to(device)
    ema.apply_to(ema_model)

    # ------------------- Unsmeared datasets ------------------- #
    print("\n" + "=" * 70)
    print("STEP 7: UNSMEAR UNMERGED DATA")
    print("=" * 70)
    unmerge_std = standardize(unmerged_const_train, unmerged_mask_train, const_means, const_stds)
    unsmeared_std_train = generate_unsmeared(ema_model, unmerge_std, unmerged_mask_train, betas, alpha, alpha_bar, device)
    unsmeared_const_train = unsmeared_std_train * const_stds + const_means

    unmerge_std_val = standardize(unmerged_const_val, unmerged_mask_val, const_means, const_stds)
    unsmeared_std_val = generate_unsmeared(ema_model, unmerge_std_val, unmerged_mask_val, betas, alpha, alpha_bar, device)
    unsmeared_const_val = unsmeared_std_val * const_stds + const_means

    # Final classifier
    print("\n" + "=" * 70)
    print("STEP 8: FINAL CLASSIFIER (Unmerged+Unsmeared)")
    print("=" * 70)
    feat_uns_train = compute_features(unsmeared_const_train, unmerged_mask_train)
    feat_uns_val = compute_features(unsmeared_const_val, unmerged_mask_val)
    feat_uns_train_std = standardize(feat_uns_train, unmerged_mask_train, feat_means, feat_stds)
    feat_uns_val_std = standardize(feat_uns_val, unmerged_mask_val, feat_means, feat_stds)

    train_uns = JetDataset(feat_uns_train_std[train_idx], unmerged_mask_train[train_idx], all_labels[train_idx])
    val_uns = JetDataset(feat_uns_val_std[val_idx], unmerged_mask_val[val_idx], all_labels[val_idx])
    test_uns = JetDataset(feat_uns_val_std[test_idx], unmerged_mask_val[test_idx], all_labels[test_idx])
    train_uns_loader = DataLoader(train_uns, batch_size=BS, shuffle=True, drop_last=True)
    val_uns_loader = DataLoader(val_uns, batch_size=BS, shuffle=False)
    test_uns_loader = DataLoader(test_uns, batch_size=BS, shuffle=False)

    final_model = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_f = torch.optim.AdamW(final_model.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_f = get_scheduler(opt_f, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Final"):
        _, train_auc = train_classifier(final_model, train_uns_loader, opt_f, device)
        val_auc, _, _ = eval_classifier(final_model, val_uns_loader, device)
        sch_f.step()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping final at epoch {ep+1}")
            break
    if best_state is not None:
        final_model.load_state_dict(best_state)
    auc_final, preds_final, _ = eval_classifier(final_model, test_uns_loader, device)

    print("\nFinal test AUCs:")
    print(f"  Teacher (offline): {auc_teacher:.4f}")
    print(f"  Baseline (HLT):    {auc_baseline:.4f}")
    print(f"  Final (unmerge+unsmear): {auc_final:.4f}")

    # ROC plots
    if len(np.unique(labs)) > 1:
        fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
        fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
        fpr_f, tpr_f, _ = roc_curve(labs, preds_final)
        curves = {
            "Teacher": (fpr_t, tpr_t, auc_teacher),
            "Baseline": (fpr_b, tpr_b, auc_baseline),
            "Final": (fpr_f, tpr_f, auc_final),
        }
        save_roc_plots(save_root, labs, curves)
        print(f"Saved ROC plots to: {save_root}")
    else:
        print("Warning: ROC plots skipped (only one class present).")


if __name__ == "__main__":
    main()

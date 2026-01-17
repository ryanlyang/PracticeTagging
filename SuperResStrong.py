#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Super-Resolution (SR) front-end for HLT jets.

Pipeline:
  x_hlt -> G (super-res generator) -> x_hat_off -> C (offline-style classifier) -> y_hat

Stages:
  1) Train offline teacher T on true offline (optional but recommended)
  2) Train HLT baseline (reference)
  3) Pretrain G to map HLT -> offline-like features
  4) Pretrain C on true offline (warm start)
  5) End-to-end SR training:
     - SR-only: supervised + reconstruction + consistency
     - SR+KD: add teacher KD + teacher embedding alignment

Saves:
  - test_split/test_features_and_masks.npz (offline + hlt standardized features, masks, labels, indices)
  - checkpoints/superres/<run_name>/*.pt  (best checkpoints, if enabled)
  - checkpoints/superres/<run_name>/results.npz + results.png (preds + ROC plot)

Assumption about utils.load_from_files output:
  all_data: (N, max_constits, 3) with columns [eta, phi, pt]
If your columns differ, edit ETA_IDX/PHI_IDX/PT_IDX below.
"""

from pathlib import Path
import argparse
import os
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
# Your EFN code used:
#   angular = data[:,:,0:2]
#   pt      = data[:,:,2]
# So we assume:
ETA_IDX = 0
PHI_IDX = 1
PT_IDX  = 2


# ----------------------------- HLT config (matches your professor) ----------------------------- #
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

        # Noise (disabled like his notebook)
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

    "generator": {
        "embed_dim": 128,
        "num_heads": 8,
        "num_layers": 4,
        "dec_layers": 2,
        "ff_dim": 512,
        "dropout": 0.1,
        "latent_dim": 32,
        "mix_components": 8,
        "min_sigma": 1e-3,
        "lr": 3e-4,
        "epochs": 60,
        "patience": 20,
        "lambda_feat": 1.0,
        "lambda_mask": 0.5,
        "lambda_mult": 0.1,
        "lambda_stats": 0.1,
        "lambda_perc": 0.2,
        "lambda_logit": 0.1,
    },

    "superres": {
        "epochs": 50,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "n_samples": 4,
        "n_samples_eval": 4,
        "mask_threshold": 0.5,
    },

    "consistency": {
        "conf_power": 2.0,
        "conf_min": 0.0,
    },

    "kd": {
        "temperature": 7.0,
        "alpha_kd": 0.5,
        "alpha_attn": 0.05,
        "alpha_rep": 0.10,
        "alpha_nce": 0.10,
        "tau_nce": 0.10,
        "use_conf_weighted_kd": True,
    },

    "loss": {
        "sup": 1.0,
        "cons_prob": 0.1,
        "cons_emb": 0.05,
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

    # pT smearing
    pt_noise = np.random.normal(1.0, hcfg["pt_resolution"], (n_jets, max_part))
    pt_noise = np.clip(pt_noise, 0.5, 1.5)
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)

    # eta smearing
    eta_noise = np.random.normal(0, hcfg["eta_resolution"], (n_jets, max_part))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)

    # phi smearing
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


# ----------------------------- Dataset ----------------------------- #
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


# ----------------------------- Model (same as professor) ----------------------------- #
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


class ParticleTransformerKD(nn.Module):
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

    def forward(self, x, mask, return_attention=False, return_embedding=False):
        batch_size, seq_len, _ = x.shape

        h = x.reshape(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.reshape(batch_size, seq_len, -1)

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

        if return_attention and return_embedding:
            return logits, attn_weights.squeeze(1), z
        if return_attention:
            return logits, attn_weights.squeeze(1)
        if return_embedding:
            return logits, z
        return logits


# ----------------------------- Super-Resolution Generator ----------------------------- #
class SuperResGenerator(nn.Module):
    def __init__(self, input_dim=7, embed_dim=128, num_heads=8, num_layers=4, dec_layers=2,
                 ff_dim=512, dropout=0.1, latent_dim=32, n_components=8, min_sigma=1e-3,
                 max_constits=80):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.min_sigma = min_sigma
        self.max_constits = max_constits

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
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

        self.query_embed = nn.Parameter(torch.randn(1, max_constits, embed_dim) * 0.02)
        self.query_dropout = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refine = nn.TransformerEncoder(dec_layer, num_layers=max(1, dec_layers))

        self.comp_logits = nn.Linear(embed_dim, n_components)
        self.comp_proj = nn.Linear(embed_dim, n_components * embed_dim)
        self.out_mu = nn.Linear(embed_dim, input_dim)
        self.out_logsigma = nn.Linear(embed_dim, input_dim)
        self.out_mask = nn.Linear(embed_dim, 1)

    def forward(self, x, mask):
        bsz, seq_len, _ = x.shape
        h = self.input_proj(x.reshape(-1, self.input_dim)).reshape(bsz, seq_len, -1)
        h = self.encoder(h, src_key_padding_mask=~mask)

        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask_f).sum(dim=1) / denom.squeeze(1)

        queries = self.query_embed[:, :seq_len].expand(bsz, -1, -1)
        queries = self.query_dropout(queries)
        q, _ = self.cross_attn(queries, h, h, key_padding_mask=~mask)
        h_dec = self.refine(q)

        mix_logits = self.comp_logits(pooled)
        comp_embed = self.comp_proj(pooled).view(bsz, self.n_components, self.embed_dim)

        mu_list = []
        log_sigma_list = []
        for k in range(self.n_components):
            h_k = h_dec + comp_embed[:, k].unsqueeze(1)
            mu_list.append(self.out_mu(h_k))
            log_sigma_list.append(self.out_logsigma(h_k))

        mu = torch.stack(mu_list, dim=1)
        log_sigma = torch.stack(log_sigma_list, dim=1)
        mask_logits = self.out_mask(h_dec).squeeze(-1)
        return mu, log_sigma, mask_logits, mix_logits

    def sample_from_params(self, mu, log_sigma, mix_logits):
        sigma = F.softplus(log_sigma) + self.min_sigma
        comp = torch.distributions.Categorical(logits=mix_logits).sample()
        idx = comp.view(-1, 1, 1, 1).expand(-1, 1, mu.size(2), mu.size(3))
        mu_sel = torch.gather(mu, 1, idx).squeeze(1)
        sigma_sel = torch.gather(sigma, 1, idx).squeeze(1)
        eps = torch.randn_like(mu_sel)
        return mu_sel + sigma_sel * eps

    def sample(self, x, mask, n_samples=1, mask_threshold=0.5):
        views = []
        masks = []
        probs = []
        mu, log_sigma, mask_logits, mix_logits = self(x, mask)
        for _ in range(n_samples):
            x_hat = self.sample_from_params(mu, log_sigma, mix_logits)
            p_exist = torch.sigmoid(mask_logits)
            x_hat = x_hat * p_exist.unsqueeze(-1)
            m_hat = p_exist > mask_threshold
            x_hat, m_hat = ensure_nonempty_mask(x_hat, m_hat)
            views.append(x_hat)
            masks.append(m_hat)
            probs.append(p_exist)
        return views, masks, probs


# ----------------------------- KD losses (same as professor) ----------------------------- #
def ensure_nonempty_mask(x, mask):
    if mask.dim() != 2:
        return x, mask
    empty = mask.sum(dim=1) == 0
    if empty.any():
        mask = mask.clone()
        x = x.clone()
        mask[empty, 0] = True
        x[empty, 0] = 0.0
    return x, mask


def safe_sigmoid(logits, temp=1.0, eps=1e-6):
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    probs = torch.sigmoid(logits / temp)
    return torch.clamp(probs, eps, 1.0 - eps)


def kd_loss(student_logits, teacher_logits, T):
    s_soft = safe_sigmoid(student_logits, temp=T)
    t_soft = safe_sigmoid(teacher_logits, temp=T)
    return F.binary_cross_entropy(s_soft, t_soft) * (T ** 2)


def kd_loss_conf_weighted(student_logits, teacher_logits, T):
    s_soft = safe_sigmoid(student_logits, temp=T)
    t_soft = safe_sigmoid(teacher_logits, temp=T)
    w = (torch.abs(safe_sigmoid(teacher_logits) - 0.5) * 2.0).detach()
    per = F.binary_cross_entropy(s_soft, t_soft, reduction="none")
    return (w * per).mean() * (T ** 2)


def info_nce_loss(s_z, t_z, tau=0.1):
    s = F.normalize(s_z, dim=1)
    t = F.normalize(t_z, dim=1)
    logits_st = (s @ t.t()) / tau
    logits_ts = (t @ s.t()) / tau
    labels = torch.arange(s.size(0), device=s.device)
    loss_st = F.cross_entropy(logits_st, labels)
    loss_ts = F.cross_entropy(logits_ts, labels)
    return 0.5 * (loss_st + loss_ts)


def mixture_nll(mu, log_sigma, mix_logits, x, mask, min_sigma=1e-3):
    sigma = F.softplus(log_sigma) + min_sigma
    log_sigma_val = torch.log(sigma)
    diff = (x.unsqueeze(1) - mu) / sigma
    log_prob = -0.5 * diff ** 2 - log_sigma_val - 0.5 * np.log(2.0 * np.pi)
    log_prob = log_prob.sum(dim=-1)
    mask_f = mask.float().unsqueeze(1)
    log_prob = (log_prob * mask_f).sum(dim=-1)
    denom = mask_f.sum(dim=-1).clamp(min=1.0)
    log_prob = log_prob / denom
    log_pi = F.log_softmax(mix_logits, dim=1)
    log_mix = torch.logsumexp(log_pi + log_prob, dim=1)
    return -log_mix.mean()


def mixture_mean(mu, mix_logits):
    weights = F.softmax(mix_logits, dim=1).unsqueeze(-1).unsqueeze(-1)
    return (weights * mu).sum(dim=1)


def attn_kl_loss(s_attn, t_attn, s_mask, t_mask, eps=1e-8):
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


def symmetric_kl_bernoulli(p, q, eps=1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)
    kl_pq = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    kl_qp = q * torch.log(q / p) + (1 - q) * torch.log((1 - p) / (1 - q))
    return 0.5 * (kl_pq + kl_qp)


def confidence_weight_min(p_i, p_j, power=2.0, conf_min=0.0):
    conf_i = torch.abs(p_i - 0.5) * 2.0
    conf_j = torch.abs(p_j - 0.5) * 2.0
    conf = torch.minimum(conf_i, conf_j)
    conf = torch.clamp(conf ** power, min=conf_min, max=1.0)
    return conf


def cosine_embed_loss(z1, z2, eps=1e-8):
    z1n = z1 / (torch.norm(z1, dim=1, keepdim=True) + eps)
    z2n = z2 / (torch.norm(z2, dim=1, keepdim=True) + eps)
    cos = (z1n * z2n).sum(dim=1)
    return 1.0 - cos


def masked_mean_std(x, mask, eps=1e-8):
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x * mask_f).sum(dim=1) / denom.squeeze(1)
    var = ((x - mean.unsqueeze(1)) ** 2 * mask_f).sum(dim=1) / denom.squeeze(1)
    std = torch.sqrt(var + eps)
    return mean, std


def generator_stats_loss(x_hat, x_true, mask):
    mu_hat, std_hat = masked_mean_std(x_hat, mask)
    mu_true, std_true = masked_mean_std(x_true, mask)
    return F.l1_loss(mu_hat, mu_true) + F.l1_loss(std_hat, std_true)


def kl_divergence(mu, logvar):
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=1))


# ----------------------------- Train / eval (no weights) ----------------------------- #
def train_standard(model, loader, opt, device, feat_key, mask_key):
    model.train()
    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x = batch[feat_key].to(device)
        mask = batch[mask_key].to(device)
        y = batch["label"].to(device)
        x, mask = ensure_nonempty_mask(x, mask)

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


def fit_classifier(model, train_loader, val_loader, device, cfg, feat_key, mask_key, label):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    sch = get_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["training"]["epochs"])

    best_auc, best_state, no_improve = 0.0, None, 0
    history = []

    for ep in tqdm(range(cfg["training"]["epochs"]), desc=label):
        train_loss, train_auc = train_standard(model, train_loader, opt, device, feat_key, mask_key)
        val_auc, _, _ = evaluate(model, val_loader, device, feat_key, mask_key)
        sch.step()

        history.append((ep + 1, train_loss, train_auc, val_auc))

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")

        if no_improve >= cfg["training"]["patience"]:
            print(f"Early stopping {label} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc, history


def train_generator_epoch(generator, loader, opt, device, cfg, teacher=None):
    generator.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total_feat = 0.0
    total_mask = 0.0
    total_mult = 0.0
    total_stats = 0.0
    n_batches = 0

    for batch in loader:
        x_off = batch["off"].to(device)
        m_off = batch["mask_off"].to(device)
        x_hlt = batch["hlt"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        x_off, m_off = ensure_nonempty_mask(x_off, m_off)
        x_hlt, m_hlt = ensure_nonempty_mask(x_hlt, m_hlt)

        opt.zero_grad()
        mu, log_sigma, mask_logits, mix_logits = generator(x_hlt, m_hlt)
        p_exist = torch.sigmoid(mask_logits)
        x_mean = mixture_mean(mu, mix_logits) * p_exist.unsqueeze(-1)

        loss_feat = mixture_nll(mu, log_sigma, mix_logits, x_off, m_off, min_sigma=cfg["generator"]["min_sigma"])
        loss_mask = F.binary_cross_entropy_with_logits(mask_logits, m_off.float())
        loss_mult = F.mse_loss(p_exist.sum(dim=1), m_off.float().sum(dim=1))
        loss_stats = generator_stats_loss(x_mean, x_off, m_off)

        loss_perc = torch.tensor(0.0, device=device)
        loss_logit = torch.tensor(0.0, device=device)
        if teacher is not None:
            with torch.no_grad():
                t_logits_true, t_emb_true = teacher(x_off, m_off, return_embedding=True)
                t_logits_true = t_logits_true.squeeze(1)
            t_logits_hat, t_emb_hat = teacher(x_mean, m_off, return_embedding=True)
            t_logits_hat = t_logits_hat.squeeze(1)
            loss_perc = cosine_embed_loss(t_emb_hat, t_emb_true).mean()
            loss_logit = F.mse_loss(t_logits_hat, t_logits_true)

        loss = (
            cfg["generator"]["lambda_feat"] * loss_feat
            + cfg["generator"]["lambda_mask"] * loss_mask
            + cfg["generator"]["lambda_mult"] * loss_mult
            + cfg["generator"]["lambda_stats"] * loss_stats
            + cfg["generator"]["lambda_perc"] * loss_perc
            + cfg["generator"]["lambda_logit"] * loss_logit
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        total_feat += loss_feat.item()
        total_mask += loss_mask.item()
        total_mult += loss_mult.item()
        total_stats += loss_stats.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / n_batches,
        total_feat / n_batches,
        total_mask / n_batches,
        total_mult / n_batches,
        total_stats / n_batches,
    )


@torch.no_grad()
def eval_generator(generator, loader, device, cfg, teacher=None):
    generator.eval()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total_feat = 0.0
    total_mask = 0.0
    total_mult = 0.0
    total_stats = 0.0
    n_batches = 0

    for batch in loader:
        x_off = batch["off"].to(device)
        m_off = batch["mask_off"].to(device)
        x_hlt = batch["hlt"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        x_off, m_off = ensure_nonempty_mask(x_off, m_off)
        x_hlt, m_hlt = ensure_nonempty_mask(x_hlt, m_hlt)

        mu, log_sigma, mask_logits, mix_logits = generator(x_hlt, m_hlt)
        p_exist = torch.sigmoid(mask_logits)
        x_mean = mixture_mean(mu, mix_logits) * p_exist.unsqueeze(-1)

        loss_feat = mixture_nll(mu, log_sigma, mix_logits, x_off, m_off, min_sigma=cfg["generator"]["min_sigma"])
        loss_mask = F.binary_cross_entropy_with_logits(mask_logits, m_off.float())
        loss_mult = F.mse_loss(p_exist.sum(dim=1), m_off.float().sum(dim=1))
        loss_stats = generator_stats_loss(x_mean, x_off, m_off)

        loss_perc = torch.tensor(0.0, device=device)
        loss_logit = torch.tensor(0.0, device=device)
        if teacher is not None:
            t_logits_true, t_emb_true = teacher(x_off, m_off, return_embedding=True)
            t_logits_true = t_logits_true.squeeze(1)
            t_logits_hat, t_emb_hat = teacher(x_mean, m_off, return_embedding=True)
            t_logits_hat = t_logits_hat.squeeze(1)
            loss_perc = cosine_embed_loss(t_emb_hat, t_emb_true).mean()
            loss_logit = F.mse_loss(t_logits_hat, t_logits_true)

        loss = (
            cfg["generator"]["lambda_feat"] * loss_feat
            + cfg["generator"]["lambda_mask"] * loss_mask
            + cfg["generator"]["lambda_mult"] * loss_mult
            + cfg["generator"]["lambda_stats"] * loss_stats
            + cfg["generator"]["lambda_perc"] * loss_perc
            + cfg["generator"]["lambda_logit"] * loss_logit
        )

        total_loss += loss.item()
        total_feat += loss_feat.item()
        total_mask += loss_mask.item()
        total_mult += loss_mult.item()
        total_stats += loss_stats.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / n_batches,
        total_feat / n_batches,
        total_mask / n_batches,
        total_mult / n_batches,
        total_stats / n_batches,
    )


def fit_generator(generator, train_loader, val_loader, device, cfg, teacher=None, save_path=None, skip_save=False):
    opt = torch.optim.AdamW(generator.parameters(), lr=cfg["generator"]["lr"])
    best_val, best_state, no_improve = 1e9, None, 0
    history = []

    for ep in tqdm(range(cfg["generator"]["epochs"]), desc="Generator"):
        tr = train_generator_epoch(generator, train_loader, opt, device, cfg, teacher=teacher)
        va = eval_generator(generator, val_loader, device, cfg, teacher=teacher)
        history.append((ep + 1, *tr, *va))

        if va[0] < best_val:
            best_val = va[0]
            best_state = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: val_loss={va[0]:.4f}, nll={va[1]:.4f}, mask={va[2]:.4f}, mult={va[3]:.4f}")

        patience = cfg["generator"].get("patience", cfg["training"]["patience"])
        if no_improve >= patience:
            print(f"Early stopping generator at epoch {ep+1}")
            break

    if best_state is not None:
        generator.load_state_dict(best_state)
    if save_path is not None and not skip_save:
        torch.save({"model": generator.state_dict(), "val": best_val, "history": history}, save_path)
    return best_val, history


def train_superres_epoch(generator, classifier, teacher, loader, opt, device, cfg,
                         use_kd=False, temperature=7.0, alpha_kd=0.5):
    generator.train()
    classifier.train()
    if teacher is not None:
        teacher.eval()

    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x_off = batch["off"].to(device)
        m_off = batch["mask_off"].to(device)
        x_hlt = batch["hlt"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        y = batch["label"].to(device)
        x_off, m_off = ensure_nonempty_mask(x_off, m_off)
        x_hlt, m_hlt = ensure_nonempty_mask(x_hlt, m_hlt)

        if use_kd and teacher is not None:
            with torch.no_grad():
                t_logits_true, t_attn_true, t_emb_true = teacher(
                    x_off, m_off, return_attention=True, return_embedding=True
                )
                t_logits_true = t_logits_true.squeeze(1)

        opt.zero_grad()

        views_probs = []
        views_emb = []
        loss_mask = torch.tensor(0.0, device=device)
        loss_mult = torch.tensor(0.0, device=device)
        loss_stats = torch.tensor(0.0, device=device)
        loss_kd = torch.tensor(0.0, device=device)
        loss_rep = torch.tensor(0.0, device=device)
        loss_nce = torch.tensor(0.0, device=device)
        loss_attn = torch.tensor(0.0, device=device)
        use_conf = cfg["kd"]["use_conf_weighted_kd"]
        a_attn = cfg["kd"]["alpha_attn"]
        a_rep = cfg["kd"]["alpha_rep"]
        a_nce = cfg["kd"]["alpha_nce"]

        mu, log_sigma, mask_logits, mix_logits = generator(x_hlt, m_hlt)
        p_exist = torch.sigmoid(mask_logits)
        x_mean = mixture_mean(mu, mix_logits) * p_exist.unsqueeze(-1)

        loss_feat = mixture_nll(mu, log_sigma, mix_logits, x_off, m_off, min_sigma=cfg["generator"]["min_sigma"])
        loss_mask = F.binary_cross_entropy_with_logits(mask_logits, m_off.float())
        loss_mult = F.mse_loss(p_exist.sum(dim=1), m_off.float().sum(dim=1))
        loss_stats = generator_stats_loss(x_mean, x_off, m_off)

        for _ in range(cfg["superres"]["n_samples"]):
            x_hat = generator.sample_from_params(mu, log_sigma, mix_logits)
            x_hat = x_hat * p_exist.unsqueeze(-1)
            m_hat = p_exist > cfg["superres"]["mask_threshold"]
            x_hat, m_hat = ensure_nonempty_mask(x_hat, m_hat)

            c_logits, c_attn, c_emb = classifier(
                x_hat, m_hat, return_attention=True, return_embedding=True
            )
            c_logits = c_logits.squeeze(1)
            p = safe_sigmoid(c_logits)
            views_probs.append(p)
            views_emb.append(c_emb)

            if use_kd and teacher is not None:
                t_logits_hat, t_attn_hat, t_emb_hat = teacher(
                    x_hat, m_hat, return_attention=True, return_embedding=True
                )
                t_logits_hat = t_logits_hat.squeeze(1)
                if use_conf:
                    loss_kd = loss_kd + kd_loss_conf_weighted(t_logits_hat, t_logits_true, temperature)
                else:
                    loss_kd = loss_kd + kd_loss(t_logits_hat, t_logits_true, temperature)
                if a_rep > 0:
                    loss_rep = loss_rep + cosine_embed_loss(t_emb_hat, t_emb_true).mean()
                if a_nce > 0:
                    loss_nce = loss_nce + info_nce_loss(t_emb_hat, t_emb_true, tau=cfg["kd"]["tau_nce"])
                if a_attn > 0:
                    loss_attn = loss_attn + attn_kl_loss(t_attn_hat, t_attn_true, m_hat, m_off)

        n_samples = max(1, cfg["superres"]["n_samples"])
        loss_kd = loss_kd / n_samples
        loss_rep = loss_rep / n_samples
        loss_nce = loss_nce / n_samples
        loss_attn = loss_attn / max(1, n_samples)

        p_mean = torch.stack(views_probs, dim=0).mean(dim=0)
        loss_sup = F.binary_cross_entropy(p_mean, y)

        loss_cons_prob = torch.tensor(0.0, device=device)
        loss_cons_emb = torch.tensor(0.0, device=device)
        pair_count = 0
        for i in range(len(views_probs)):
            for j in range(i + 1, len(views_probs)):
                w = confidence_weight_min(
                    views_probs[i], views_probs[j],
                    power=cfg["consistency"]["conf_power"],
                    conf_min=cfg["consistency"]["conf_min"],
                )
                loss_cons_prob = loss_cons_prob + (w * symmetric_kl_bernoulli(views_probs[i], views_probs[j])).mean()
                loss_cons_emb = loss_cons_emb + (w * cosine_embed_loss(views_emb[i], views_emb[j])).mean()
                pair_count += 1
        if pair_count > 0:
            loss_cons_prob = loss_cons_prob / pair_count
            loss_cons_emb = loss_cons_emb / pair_count

        loss = (
            cfg["loss"]["sup"] * loss_sup
            + cfg["generator"]["lambda_feat"] * loss_feat
            + cfg["generator"]["lambda_mask"] * loss_mask
            + cfg["generator"]["lambda_mult"] * loss_mult
            + cfg["generator"]["lambda_stats"] * loss_stats
            + cfg["loss"]["cons_prob"] * loss_cons_prob
            + cfg["loss"]["cons_emb"] * loss_cons_emb
        )

        if use_kd and teacher is not None:
            loss = loss + alpha_kd * loss_kd
            if a_rep > 0:
                loss = loss + a_rep * loss_rep
            if a_nce > 0:
                loss = loss + a_nce * loss_nce
            if a_attn > 0:
                loss = loss + a_attn * loss_attn

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(generator.parameters()) + list(classifier.parameters()), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.append(p_mean.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return total_loss / len(labs), roc_auc_score(labs, preds)


def fit_superres(generator, classifier, teacher, train_loader, val_loader, device, cfg,
                 use_kd=False, save_path=None, skip_save=False, label="SR",
                 temp_init=7.0, temp_final=None, alpha_init=0.5, alpha_final=None):
    params = list(generator.parameters()) + list(classifier.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["superres"]["lr"], weight_decay=cfg["superres"]["weight_decay"])
    sch = get_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["superres"]["epochs"])

    best_auc, best_state, no_improve = 0.0, None, 0
    history = []

    for ep in tqdm(range(cfg["superres"]["epochs"]), desc=label):
        temp = get_temperature_schedule(ep, cfg["superres"]["epochs"], temp_init, temp_final)
        alpha = get_alpha_schedule(ep, cfg["superres"]["epochs"], alpha_init, alpha_final)
        tr_loss, tr_auc = train_superres_epoch(
            generator, classifier, teacher, train_loader, opt, device, cfg,
            use_kd=use_kd, temperature=temp, alpha_kd=alpha
        )
        val_auc, _ = evaluate_superres(generator, classifier, val_loader, device, cfg)
        sch.step()

        history.append((ep + 1, tr_loss, tr_auc, val_auc, temp, alpha))

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {
                "generator": {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()},
                "classifier": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")

        if no_improve >= cfg["training"]["patience"]:
            print(f"Early stopping {label} at epoch {ep+1}")
            break

    if best_state is not None:
        generator.load_state_dict(best_state["generator"])
        classifier.load_state_dict(best_state["classifier"])
    if save_path is not None and not skip_save:
        torch.save(
            {"generator": generator.state_dict(), "classifier": classifier.state_dict(), "auc": best_auc, "history": history},
            save_path,
        )
    return best_auc, history


@torch.no_grad()
def evaluate_superres(generator, classifier, loader, device, cfg):
    generator.eval()
    classifier.eval()
    preds, labs = [], []
    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        y = batch["label"].to(device)
        x_hlt, m_hlt = ensure_nonempty_mask(x_hlt, m_hlt)

        views, masks, _ = generator.sample(
            x_hlt, m_hlt, n_samples=cfg["superres"]["n_samples_eval"],
            mask_threshold=cfg["superres"]["mask_threshold"],
        )
        probs = []
        for x_hat, m_hat in zip(views, masks):
            logits = classifier(x_hat, m_hat).squeeze(1)
            probs.append(safe_sigmoid(logits))
        p_mean = torch.stack(probs, dim=0).mean(dim=0)

        preds.append(p_mean.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return roc_auc_score(labs, preds), preds


@torch.no_grad()
def evaluate(model, loader, device, feat_key, mask_key):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        x = batch[feat_key].to(device)
        mask = batch[mask_key].to(device)
        x, mask = ensure_nonempty_mask(x, mask)
        logits = model(x, mask).squeeze(1)
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


# ----------------------------- Hyperparameter Scheduling ----------------------------- #
def get_temperature_schedule(epoch, total_epochs, T_init, T_final):
    """Linear temperature annealing from T_init to T_final over training"""
    if T_final is None:
        return T_init
    return T_init + (T_final - T_init) * (epoch / max(total_epochs - 1, 1))


def get_alpha_schedule(epoch, total_epochs, alpha_init, alpha_final):
    """Linear alpha scheduling from alpha_init to alpha_final over training"""
    if alpha_final is None:
        return alpha_init
    return alpha_init + (alpha_final - alpha_init) * (epoch / max(total_epochs - 1, 1))


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default="./data",
        help="Directory containing your *.h5 files (default: ./data relative to project root)",
    )
    parser.add_argument("--n_train_jets", type=int, default=100000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "superres"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_name", type=str, default="default", help="Unique name for this run")
    parser.add_argument("--sr_mode", type=str, choices=["sr_only", "sr_kd", "both"], default="both")

    # Generator architecture + training
    parser.add_argument("--gen_embed_dim", type=int, default=CONFIG["generator"]["embed_dim"])
    parser.add_argument("--gen_num_heads", type=int, default=CONFIG["generator"]["num_heads"])
    parser.add_argument("--gen_num_layers", type=int, default=CONFIG["generator"]["num_layers"])
    parser.add_argument("--gen_dec_layers", type=int, default=CONFIG["generator"]["dec_layers"])
    parser.add_argument("--gen_ff_dim", type=int, default=CONFIG["generator"]["ff_dim"])
    parser.add_argument("--gen_dropout", type=float, default=CONFIG["generator"]["dropout"])
    parser.add_argument("--gen_latent_dim", type=int, default=CONFIG["generator"]["latent_dim"])
    parser.add_argument("--gen_mix_components", type=int, default=CONFIG["generator"]["mix_components"])
    parser.add_argument("--gen_min_sigma", type=float, default=CONFIG["generator"]["min_sigma"])
    parser.add_argument("--gen_lr", type=float, default=CONFIG["generator"]["lr"])
    parser.add_argument("--gen_epochs", type=int, default=CONFIG["generator"]["epochs"])
    parser.add_argument("--gen_patience", type=int, default=CONFIG["generator"]["patience"])
    parser.add_argument("--gen_lambda_feat", type=float, default=CONFIG["generator"]["lambda_feat"])
    parser.add_argument("--gen_lambda_mask", type=float, default=CONFIG["generator"]["lambda_mask"])
    parser.add_argument("--gen_lambda_mult", type=float, default=CONFIG["generator"]["lambda_mult"])
    parser.add_argument("--gen_lambda_stats", type=float, default=CONFIG["generator"]["lambda_stats"])
    parser.add_argument("--gen_lambda_perc", type=float, default=CONFIG["generator"]["lambda_perc"])
    parser.add_argument("--gen_lambda_logit", type=float, default=CONFIG["generator"]["lambda_logit"])

    # SR training
    parser.add_argument("--sr_epochs", type=int, default=CONFIG["superres"]["epochs"])
    parser.add_argument("--sr_lr", type=float, default=CONFIG["superres"]["lr"])
    parser.add_argument("--sr_samples", type=int, default=CONFIG["superres"]["n_samples"])
    parser.add_argument("--sr_eval_samples", type=int, default=CONFIG["superres"]["n_samples_eval"])
    parser.add_argument("--sr_mask_threshold", type=float, default=CONFIG["superres"]["mask_threshold"])

    # Consistency / KD hyperparameters
    parser.add_argument("--conf_power", type=float, default=CONFIG["consistency"]["conf_power"])
    parser.add_argument("--conf_min", type=float, default=CONFIG["consistency"]["conf_min"])
    parser.add_argument("--temp_init", type=float, default=CONFIG["kd"]["temperature"])
    parser.add_argument("--temp_final", type=float, default=None)
    parser.add_argument("--alpha_init", type=float, default=CONFIG["kd"]["alpha_kd"])
    parser.add_argument("--alpha_final", type=float, default=None)
    parser.add_argument("--alpha_attn", type=float, default=CONFIG["kd"]["alpha_attn"])
    parser.add_argument("--alpha_rep", type=float, default=CONFIG["kd"]["alpha_rep"])
    parser.add_argument("--alpha_nce", type=float, default=CONFIG["kd"]["alpha_nce"])
    parser.add_argument("--tau_nce", type=float, default=CONFIG["kd"]["tau_nce"])
    parser.add_argument("--no_conf_kd", action="store_true")

    # Loss weights (non-KD)
    parser.add_argument("--loss_sup", type=float, default=CONFIG["loss"]["sup"])
    parser.add_argument("--loss_cons_prob", type=float, default=CONFIG["loss"]["cons_prob"])
    parser.add_argument("--loss_cons_emb", type=float, default=CONFIG["loss"]["cons_emb"])

    # Deprecated KD args (kept for compatibility with older sweep scripts)
    parser.add_argument("--loss_kd", type=float, default=None)
    parser.add_argument("--loss_emb", type=float, default=None)
    parser.add_argument("--loss_attn", type=float, default=None)
    parser.add_argument("--kd_temp", type=float, default=None)

    # Pre-trained model loading (for hyperparameter search efficiency)
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Path to pre-trained teacher model (skips teacher training)")
    parser.add_argument("--baseline_checkpoint", type=str, default=None, help="Path to pre-trained baseline model (skips baseline training)")
    parser.add_argument("--skip_save_models", action="store_true", help="Skip saving model weights (save space during hyperparameter search)")

    args = parser.parse_args()

    # Apply overrides
    CONFIG["generator"]["embed_dim"] = args.gen_embed_dim
    CONFIG["generator"]["num_heads"] = args.gen_num_heads
    CONFIG["generator"]["num_layers"] = args.gen_num_layers
    CONFIG["generator"]["dec_layers"] = args.gen_dec_layers
    CONFIG["generator"]["ff_dim"] = args.gen_ff_dim
    CONFIG["generator"]["dropout"] = args.gen_dropout
    CONFIG["generator"]["latent_dim"] = args.gen_latent_dim
    CONFIG["generator"]["mix_components"] = args.gen_mix_components
    CONFIG["generator"]["min_sigma"] = args.gen_min_sigma
    CONFIG["generator"]["lr"] = args.gen_lr
    CONFIG["generator"]["epochs"] = args.gen_epochs
    CONFIG["generator"]["patience"] = args.gen_patience
    CONFIG["generator"]["lambda_feat"] = args.gen_lambda_feat
    CONFIG["generator"]["lambda_mask"] = args.gen_lambda_mask
    CONFIG["generator"]["lambda_mult"] = args.gen_lambda_mult
    CONFIG["generator"]["lambda_stats"] = args.gen_lambda_stats
    CONFIG["generator"]["lambda_perc"] = args.gen_lambda_perc
    CONFIG["generator"]["lambda_logit"] = args.gen_lambda_logit

    CONFIG["superres"]["epochs"] = args.sr_epochs
    CONFIG["superres"]["lr"] = args.sr_lr
    CONFIG["superres"]["n_samples"] = args.sr_samples
    CONFIG["superres"]["n_samples_eval"] = args.sr_eval_samples
    CONFIG["superres"]["mask_threshold"] = args.sr_mask_threshold

    if args.kd_temp is not None:
        args.temp_init = args.kd_temp
    if args.loss_kd is not None:
        args.alpha_init = args.loss_kd
    if args.loss_emb is not None:
        args.alpha_rep = args.loss_emb
    if args.loss_attn is not None:
        args.alpha_attn = args.loss_attn

    CONFIG["consistency"]["conf_power"] = args.conf_power
    CONFIG["consistency"]["conf_min"] = args.conf_min
    CONFIG["kd"]["temperature"] = args.temp_init
    CONFIG["kd"]["alpha_kd"] = args.alpha_init
    CONFIG["kd"]["alpha_attn"] = args.alpha_attn
    CONFIG["kd"]["alpha_rep"] = args.alpha_rep
    CONFIG["kd"]["alpha_nce"] = args.alpha_nce
    CONFIG["kd"]["tau_nce"] = args.tau_nce
    CONFIG["kd"]["use_conf_weighted_kd"] = (not args.no_conf_kd)

    CONFIG["loss"]["sup"] = args.loss_sup
    CONFIG["loss"]["cons_prob"] = args.loss_cons_prob
    CONFIG["loss"]["cons_emb"] = args.loss_cons_emb

    # Create unique save directory for this run
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

    # ------------------- Load your dataset (same style as your code) ------------------- #
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

    # ------------------- Convert your data -> (pt, eta, phi, E) for professor pipeline ------------------- #
    eta = all_data[:, :, ETA_IDX].astype(np.float32)
    phi = all_data[:, :, PHI_IDX].astype(np.float32)
    pt  = all_data[:, :, PT_IDX].astype(np.float32)

    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))

    constituents_raw = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    print(f"Avg particles per jet (raw mask): {mask_raw.sum(axis=1).mean():.1f}")

    # ------------------- Apply HLT effects (professor smearing strategy) ------------------- #
    print("Applying HLT effects...")
    constituents_hlt, masks_hlt = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=RANDOM_SEED)

    # ------------------- Apply offline threshold (professor style) ------------------- #
    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0

    print(f"Offline particles after {pt_threshold_off} threshold: {masks_off.sum():,}")
    print(f"Avg per jet: Offline={masks_off.sum(axis=1).mean():.1f}, HLT={masks_hlt.sum(axis=1).mean():.1f}")

    # ------------------- Compute features ------------------- #
    print("Computing features...")
    features_off = compute_features(constituents_off, masks_off)
    features_hlt = compute_features(constituents_hlt, masks_hlt)

    print(f"NaN check: Offline={np.isnan(features_off).sum()}, HLT={np.isnan(features_hlt).sum()}")

    # ------------------- Split indices (70/15/15, stratified, professor style) ------------------- #
    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])

    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # ------------------- Standardize using training OFFLINE stats ------------------- #
    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)

    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = standardize(features_hlt, masks_hlt, feat_means, feat_stds)

    print(f"Final NaN check: Offline={np.isnan(features_off_std).sum()}, HLT={np.isnan(features_hlt_std).sum()}")

    # ------------------- Save test split artifacts (for later curve comparisons) ------------------- #
    test_data_dir = Path().cwd() / "test_split"
    test_data_dir.mkdir(exist_ok=True)

    np.savez(
        test_data_dir / "test_features_and_masks.npz",
        idx_test=test_idx,
        labels=all_labels[test_idx],
        feat_off=features_off_std[test_idx],
        feat_hlt=features_hlt_std[test_idx],
        mask_off=masks_off[test_idx],
        mask_hlt=masks_hlt[test_idx],
        jet_pt=all_pt[test_idx] if all_pt is not None else None,
        feat_means=feat_means,
        feat_stds=feat_stds,
    )
    print(f"Saved test features/masks to: {test_data_dir / 'test_features_and_masks.npz'}")

    # ------------------- Build datasets/loaders ------------------- #
    train_ds = JetDataset(features_off_std[train_idx], features_hlt_std[train_idx], all_labels[train_idx], masks_off[train_idx], masks_hlt[train_idx])
    val_ds   = JetDataset(features_off_std[val_idx],   features_hlt_std[val_idx],   all_labels[val_idx],   masks_off[val_idx],   masks_hlt[val_idx])
    test_ds  = JetDataset(features_off_std[test_idx],  features_hlt_std[test_idx],  all_labels[test_idx],  masks_off[test_idx],  masks_hlt[test_idx])

    BS = CONFIG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BS, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BS, shuffle=False)

    # ------------------- Checkpoint paths ------------------- #
    teacher_path = save_dir / "teacher.pt"
    baseline_path = save_dir / "baseline.pt"
    generator_path = save_dir / "generator.pt"
    classifier_path = save_dir / "classifier_off.pt"
    sr_path = save_dir / "superres.pt"
    sr_kd_path = save_dir / "superres_kd.pt"

    # ------------------- STEP 1: Teacher (offline) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 1: TEACHER (Offline / high-quality view)")
    print("=" * 70)

    teacher = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)

    if args.teacher_checkpoint is not None:
        # Load pre-trained teacher
        print(f"Loading pre-trained teacher from: {args.teacher_checkpoint}")
        ckpt = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(ckpt["model"])
        best_auc_teacher = ckpt["auc"]
        history_teacher = ckpt.get("history", [])
        print(f"Loaded teacher with AUC={best_auc_teacher:.4f}")
    else:
        best_auc_teacher, history_teacher = fit_classifier(
            teacher, train_loader, val_loader, device, CONFIG, "off", "mask_off", "Teacher"
        )
        if not args.skip_save_models:
            torch.save({"model": teacher.state_dict(), "auc": best_auc_teacher, "history": history_teacher}, teacher_path)
            print(f"Saved teacher: {teacher_path} (best val AUC={best_auc_teacher:.4f})")
        else:
            print(f"Skipped saving teacher model (best val AUC={best_auc_teacher:.4f})")

    # Freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ------------------- STEP 2: Baseline HLT (no KD) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE HLT (Low-quality view, no KD)")
    print("=" * 70)

    baseline = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)

    if args.baseline_checkpoint is not None:
        # Load pre-trained baseline
        print(f"Loading pre-trained baseline from: {args.baseline_checkpoint}")
        ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        baseline.load_state_dict(ckpt["model"])
        best_auc_baseline = ckpt["auc"]
        history_baseline = ckpt.get("history", [])
        print(f"Loaded baseline with AUC={best_auc_baseline:.4f}")
    else:
        best_auc_baseline, history_baseline = fit_classifier(
            baseline, train_loader, val_loader, device, CONFIG, "hlt", "mask_hlt", "Baseline"
        )
        if not args.skip_save_models:
            torch.save({"model": baseline.state_dict(), "auc": best_auc_baseline, "history": history_baseline}, baseline_path)
            print(f"Saved baseline: {baseline_path} (best val AUC={best_auc_baseline:.4f})")
        else:
            print(f"Skipped saving baseline model (best val AUC={best_auc_baseline:.4f})")

    # ------------------- STEP 3: Super-Resolution Generator Pretrain ------------------- #
    print("\n" + "=" * 70)
    print("STEP 3: GENERATOR PRETRAIN (HLT -> OFFLINE)")
    print("=" * 70)

    generator = SuperResGenerator(
        input_dim=7,
        embed_dim=CONFIG["generator"]["embed_dim"],
        num_heads=CONFIG["generator"]["num_heads"],
        num_layers=CONFIG["generator"]["num_layers"],
        dec_layers=CONFIG["generator"]["dec_layers"],
        ff_dim=CONFIG["generator"]["ff_dim"],
        dropout=CONFIG["generator"]["dropout"],
        latent_dim=CONFIG["generator"]["latent_dim"],
        n_components=CONFIG["generator"]["mix_components"],
        min_sigma=CONFIG["generator"]["min_sigma"],
        max_constits=args.max_constits,
    ).to(device)

    best_gen_val, gen_history = fit_generator(
        generator, train_loader, val_loader, device, CONFIG, teacher=teacher,
        save_path=generator_path, skip_save=args.skip_save_models
    )
    print(f"Generator best val loss: {best_gen_val:.4f}")

    # ------------------- STEP 4: Offline Classifier Pretrain ------------------- #
    print("\n" + "=" * 70)
    print("STEP 4: CLASSIFIER PRETRAIN (Offline)")
    print("=" * 70)

    classifier_off = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
    best_auc_classifier, history_classifier = fit_classifier(
        classifier_off, train_loader, val_loader, device, CONFIG, "off", "mask_off", "Offline-Classifier"
    )
    if not args.skip_save_models:
        torch.save(
            {"model": classifier_off.state_dict(), "auc": best_auc_classifier, "history": history_classifier},
            classifier_path,
        )

    # ------------------- STEP 5: End-to-end SR Training ------------------- #
    run_sr_only = args.sr_mode in ("sr_only", "both")
    run_sr_kd = args.sr_mode in ("sr_kd", "both")

    sr_only_gen = None
    sr_only_clf = None
    sr_kd_gen = None
    sr_kd_clf = None
    best_auc_sr = None
    best_auc_sr_kd = None

    if run_sr_only:
        print("\n" + "=" * 70)
        print("STEP 5A: SR ONLY (HLT -> SR -> C)")
        print("=" * 70)
        sr_only_gen = copy.deepcopy(generator)
        sr_only_clf = copy.deepcopy(classifier_off)
        best_auc_sr, _ = fit_superres(
            sr_only_gen, sr_only_clf, None, train_loader, val_loader, device, CONFIG,
            use_kd=False, save_path=sr_path, skip_save=args.skip_save_models, label="SR-Only",
            temp_init=args.temp_init, temp_final=args.temp_final,
            alpha_init=args.alpha_init, alpha_final=args.alpha_final,
        )
        print(f"Saved SR-only model: {sr_path} (best val AUC={best_auc_sr:.4f})")

    if run_sr_kd:
        print("\n" + "=" * 70)
        print("STEP 5B: SR + KD (HLT -> SR -> C + teacher alignment)")
        print("=" * 70)
        sr_kd_gen = copy.deepcopy(generator)
        sr_kd_clf = copy.deepcopy(classifier_off)
        best_auc_sr_kd, _ = fit_superres(
            sr_kd_gen, sr_kd_clf, teacher, train_loader, val_loader, device, CONFIG,
            use_kd=True, save_path=sr_kd_path, skip_save=args.skip_save_models, label="SR+KD",
            temp_init=args.temp_init, temp_final=args.temp_final,
            alpha_init=args.alpha_init, alpha_final=args.alpha_final,
        )
        print(f"Saved SR+KD model: {sr_kd_path} (best val AUC={best_auc_sr_kd:.4f})")

    # ------------------- Final evaluation on TEST ------------------- #
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    auc_teacher, preds_teacher, labs = evaluate(teacher, test_loader, device, "off", "mask_off")
    auc_baseline, preds_baseline, _ = evaluate(baseline, test_loader, device, "hlt", "mask_hlt")

    auc_sr_only, preds_sr_only = (None, None)
    auc_sr_kd, preds_sr_kd = (None, None)

    if run_sr_only and sr_only_gen is not None and sr_only_clf is not None:
        auc_sr_only, preds_sr_only = evaluate_superres(sr_only_gen, sr_only_clf, test_loader, device, CONFIG)

    if run_sr_kd and sr_kd_gen is not None and sr_kd_clf is not None:
        auc_sr_kd, preds_sr_kd = evaluate_superres(sr_kd_gen, sr_kd_clf, test_loader, device, CONFIG)

    print(f"\n{'Model':<40} {'AUC':>10}")
    print("-" * 52)
    print(f"{'Teacher (Offline)':<40} {auc_teacher:>10.4f}")
    print(f"{'Baseline HLT':<40} {auc_baseline:>10.4f}")
    if auc_sr_only is not None:
        print(f"{'SR Only (HLT)':<40} {auc_sr_only:>10.4f}")
    if auc_sr_kd is not None:
        print(f"{'SR + KD (HLT)':<40} {auc_sr_kd:>10.4f}")
    print("-" * 52)

    # Save results for later plotting
    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_sr, tpr_sr, _ = (None, None, None)
    fpr_sr_kd, tpr_sr_kd, _ = (None, None, None)
    if preds_sr_only is not None:
        fpr_sr, tpr_sr, _ = roc_curve(labs, preds_sr_only)
    if preds_sr_kd is not None:
        fpr_sr_kd, tpr_sr_kd, _ = roc_curve(labs, preds_sr_kd)

    # Calculate Background Rejection at 50% signal efficiency (working point)
    wp = 0.5  # Working point (signal efficiency = TPR)
    idx_t = np.argmax(tpr_t >= wp)
    idx_b = np.argmax(tpr_b >= wp)
    br_teacher = 1.0 / fpr_t[idx_t] if fpr_t[idx_t] > 0 else 0
    br_baseline = 1.0 / fpr_b[idx_b] if fpr_b[idx_b] > 0 else 0
    br_sr = None
    br_sr_kd = None
    if fpr_sr is not None:
        idx_sr = np.argmax(tpr_sr >= wp)
        br_sr = 1.0 / fpr_sr[idx_sr] if fpr_sr[idx_sr] > 0 else 0
    if fpr_sr_kd is not None:
        idx_sr_kd = np.argmax(tpr_sr_kd >= wp)
        br_sr_kd = 1.0 / fpr_sr_kd[idx_sr_kd] if fpr_sr_kd[idx_sr_kd] > 0 else 0

    print(f"\nBackground Rejection at {wp*100:.0f}% signal efficiency:")
    print(f"  Teacher:  {br_teacher:.2f}")
    print(f"  Baseline: {br_baseline:.2f}")
    if br_sr is not None:
        print(f"  SR Only:  {br_sr:.2f}")
    if br_sr_kd is not None:
        print(f"  SR + KD:  {br_sr_kd:.2f}")

    np.savez(
        save_dir / "results.npz",
        labs=labs,
        preds_teacher=preds_teacher,
        preds_baseline=preds_baseline,
        preds_sr_only=preds_sr_only,
        preds_sr_kd=preds_sr_kd,
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_sr_only=auc_sr_only,
        auc_sr_kd=auc_sr_kd,
        br_teacher=br_teacher,
        br_baseline=br_baseline,
        br_sr_only=br_sr,
        br_sr_kd=br_sr_kd,
        fpr_teacher=fpr_t, tpr_teacher=tpr_t,
        fpr_baseline=fpr_b, tpr_baseline=tpr_b,
        fpr_sr_only=fpr_sr, tpr_sr_only=tpr_sr,
        fpr_sr_kd=fpr_sr_kd, tpr_sr_kd=tpr_sr_kd,
    )

    # Plot ROC curves (swapped axes: TPR on x, FPR on y)
    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={auc_teacher:.3f})", color='crimson', linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline (AUC={auc_baseline:.3f})", color='steelblue', linewidth=2)
    if preds_sr_only is not None:
        plt.plot(tpr_sr, fpr_sr, ":", label=f"SR Only (AUC={auc_sr_only:.3f})", color='forestgreen', linewidth=2)
    if preds_sr_kd is not None:
        plt.plot(tpr_sr_kd, fpr_sr_kd, "-.", label=f"SR + KD (AUC={auc_sr_kd:.3f})", color='darkorange', linewidth=2)
    plt.ylabel(r"False Positive Rate", fontsize=12)
    plt.xlabel(r"True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300)
    plt.close()

    # Log hyperparameter run results to a summary file
    summary_file = Path(args.save_dir) / "hyperparameter_search_results.txt"
    with open(summary_file, "a") as f:
        f.write(f"\nRun: {args.run_name}\n")
        f.write(f"  SR mode: {args.sr_mode}\n")
        f.write(f"  SR samples: train={CONFIG['superres']['n_samples']}, eval={CONFIG['superres']['n_samples_eval']}\n")
        f.write(f"  KD temp: {args.temp_init:.2f}")
        if args.temp_final is not None:
            f.write(f" -> {args.temp_final:.2f} (annealing)\n")
        else:
            f.write(" (constant)\n")
        f.write(f"  KD alpha: {args.alpha_init:.2f}")
        if args.alpha_final is not None:
            f.write(f" -> {args.alpha_final:.2f} (scheduling)\n")
        else:
            f.write(" (constant)\n")
        f.write(f"  KD weights: attn={CONFIG['kd']['alpha_attn']}, rep={CONFIG['kd']['alpha_rep']}, nce={CONFIG['kd']['alpha_nce']}, tau_nce={CONFIG['kd']['tau_nce']}, conf_weighted={CONFIG['kd']['use_conf_weighted_kd']}\n")
        f.write(f"  Consistency: conf_power={CONFIG['consistency']['conf_power']}, conf_min={CONFIG['consistency']['conf_min']}\n")
        if br_sr is not None:
            f.write(f"  BR @ 50% eff (SR only): {br_sr:.2f}\n")
        if br_sr_kd is not None:
            f.write(f"  BR @ 50% eff (SR+KD): {br_sr_kd:.2f}\n")
        if auc_sr_only is not None:
            f.write(f"  AUC (SR only): {auc_sr_only:.4f}\n")
        if auc_sr_kd is not None:
            f.write(f"  AUC (SR+KD): {auc_sr_kd:.4f}\n")
        f.write(f"  Saved to: {save_dir}\n")
        f.write("=" * 70 + "\n")

    print(f"\nSaved results to: {save_dir / 'results.npz'} and {save_dir / 'results.png'}")
    print(f"Logged to: {summary_file}")


if __name__ == "__main__":
    main()

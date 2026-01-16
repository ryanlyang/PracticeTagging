#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A2: Offline Teacher + Stochastic Offline Sampler + Multi-sample KD

This script trains:
  1) Teacher on OFFLINE (high-quality) view
  2) Baseline on HLT (low-quality) view, no KD
  3) Stochastic generator G: HLT -> offline-like samples
  4) Student on HLT with:
     - supervised loss on labels
     - KD to teacher ensemble mean from K offline samples
     - uncertainty-weighted KD (downweight teacher disagreement)
     - embedding alignment to teacher embedding mean
     - optional KD to all teacher samples

Notes:
  - "Real HLT" is simulated from offline via apply_hlt_effects unless you supply
    paired HLT features externally.
  - Offline statistics (train split) are used to standardize offline + HLT.
  - Teacher is evaluated on offline test; baseline/student on HLT test.
  - Matplotlib backend forced to Agg for headless runs.

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

    "kd": {
        "temperature": 4.0,
        "alpha_kd": 0.5,
        "alpha_emb": 0.3,
        "beta_all": 0.1,
        "tau_u": 0.05,
        "w_min": 0.1,
    },
    "generator": {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "ff_dim": 256,
        "dropout": 0.1,
        "epochs": 30,
        "lr": 1e-3,
        "lambda_mask": 1.0,
        "lambda_perc": 1.0,
        "lambda_logit": 0.5,
        "min_sigma": 0.02,
        "global_noise": 0.05,
    },
    "student": {
        "k_samples": 16,
        "a_sup": 1.0,
        "a_kd": 1.0,
        "a_emb": 1.0,
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
        if return_attention:
            pooled, attn_weights = self.pool_attn(
                query, h, h,
                key_padding_mask=~mask,
                need_weights=True,
                average_attn_weights=True,
            )
        else:
            pooled, attn_weights = self.pool_attn(
                query, h, h,
                key_padding_mask=~mask,
                need_weights=False,
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


# ----------------------------- Generator + KD helpers ----------------------------- #
class OfflineGenerator(nn.Module):
    def __init__(self, input_dim=7, embed_dim=64, num_heads=4, num_layers=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
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

        self.mu_head = nn.Linear(embed_dim, input_dim)
        self.log_sigma_head = nn.Linear(embed_dim, input_dim)
        self.mask_head = nn.Linear(embed_dim, 1)
        self.global_head = nn.Linear(embed_dim, input_dim)

    def forward(self, x, mask):
        B, M, _ = x.shape
        h = self.input_proj(x.reshape(-1, self.input_dim)).reshape(B, M, -1)
        h = self.transformer(h, src_key_padding_mask=~mask)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        mask_logits = self.mask_head(h).squeeze(-1)

        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask_f).sum(dim=1) / denom.squeeze(1)
        global_scale = self.global_head(pooled)
        return mu, log_sigma, mask_logits, global_scale

    def sample(self, x, mask, n_views=1, min_sigma=0.02, global_noise=0.05):
        mu, log_sigma, mask_logits, global_scale = self(x, mask)
        sigma = F.softplus(log_sigma) + min_sigma
        mask_prob = torch.sigmoid(mask_logits)

        views = []
        masks = []
        for _ in range(n_views):
            eps_local = torch.randn_like(mu)
            eps_global = torch.randn(mu.size(0), 1, mu.size(2), device=mu.device)
            noise = sigma * eps_local + global_scale.unsqueeze(1) * eps_global * global_noise
            x_gen = mu + noise

            u = torch.rand_like(mask_prob)
            m_gen = (u < mask_prob).bool() & mask
            x_gen = x_gen * m_gen.unsqueeze(-1).float()

            views.append(x_gen)
            masks.append(m_gen)
        return views, masks


def kd_loss_to_probs(student_logits, target_probs, T):
    p_s = torch.sigmoid(student_logits / T)
    return F.binary_cross_entropy(p_s, target_probs, reduction="none") * (T ** 2)


def rep_loss_cosine(s_z, t_z):
    s = F.normalize(s_z, dim=1)
    t = F.normalize(t_z, dim=1)
    return 1.0 - (s * t).sum(dim=1)


def masked_huber_loss(pred, target, mask, delta=1.0):
    mask_f = mask.float().unsqueeze(-1)
    diff = pred - target
    abs_diff = torch.abs(diff)
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    lin = abs_diff - quad
    loss = 0.5 * quad ** 2 + delta * lin
    denom = mask_f.sum().clamp(min=1.0)
    return (loss * mask_f).sum() / denom


# ----------------------------- Train / eval (no weights) ----------------------------- #
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
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / len(preds), roc_auc_score(labs, preds)


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
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def teacher_ensemble_stats(logits_list, T, tau_u, w_min):
    # logits_list: list of (B,) tensors
    probs = torch.stack([torch.sigmoid(z / T) for z in logits_list], dim=0)  # (K, B)
    p_bar = probs.mean(dim=0)
    var_p = ((probs - p_bar.unsqueeze(0)) ** 2).mean(dim=0)
    w_u = torch.exp(-var_p / tau_u)
    if w_min > 0.0:
        w_u = torch.clamp(w_u, w_min, 1.0)
    return probs, p_bar, w_u


def fit_generator(generator, teacher, train_loader, val_loader, device, cfg, save_path=None, skip_save=False):
    opt = torch.optim.AdamW(generator.parameters(), lr=cfg["generator"]["lr"])
    best_val, best_state, no_improve = 1e9, None, 0
    history = []

    for ep in tqdm(range(cfg["generator"]["epochs"]), desc="Generator"):
        generator.train()
        teacher.eval()
        tr_loss = 0.0
        tr_batches = 0

        for batch in train_loader:
            x_hlt = batch["hlt"].to(device)
            m_hlt = batch["mask_hlt"].to(device)
            x_off = batch["off"].to(device)
            m_off = batch["mask_off"].to(device)

            opt.zero_grad()
            mu, log_sigma, mask_logits, _ = generator(x_hlt, m_hlt)

            recon = masked_huber_loss(mu, x_off, m_off)
            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m_off.float())

            with torch.no_grad():
                t_logits_true, t_emb_true = teacher(x_off, m_off, return_embedding=True)
            t_logits_hat, t_emb_hat = teacher(mu * m_off.unsqueeze(-1).float(), m_off, return_embedding=True)

            perc = F.mse_loss(t_emb_hat, t_emb_true)
            logit_loss = F.mse_loss(t_logits_hat.squeeze(1), t_logits_true.squeeze(1))

            loss = recon
            loss = loss + cfg["generator"]["lambda_mask"] * mask_loss
            loss = loss + cfg["generator"]["lambda_perc"] * perc
            loss = loss + cfg["generator"]["lambda_logit"] * logit_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item()
            tr_batches += 1

        # Validation
        generator.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x_hlt = batch["hlt"].to(device)
                m_hlt = batch["mask_hlt"].to(device)
                x_off = batch["off"].to(device)
                m_off = batch["mask_off"].to(device)

                mu, log_sigma, mask_logits, _ = generator(x_hlt, m_hlt)
                recon = masked_huber_loss(mu, x_off, m_off)
                mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m_off.float())

                t_logits_true, t_emb_true = teacher(x_off, m_off, return_embedding=True)
                t_logits_hat, t_emb_hat = teacher(mu * m_off.unsqueeze(-1).float(), m_off, return_embedding=True)
                perc = F.mse_loss(t_emb_hat, t_emb_true)
                logit_loss = F.mse_loss(t_logits_hat.squeeze(1), t_logits_true.squeeze(1))

                loss = recon
                loss = loss + cfg["generator"]["lambda_mask"] * mask_loss
                loss = loss + cfg["generator"]["lambda_perc"] * perc
                loss = loss + cfg["generator"]["lambda_logit"] * logit_loss

                val_loss += loss.item()
                val_batches += 1

        tr_loss = tr_loss / max(tr_batches, 1)
        val_loss = val_loss / max(val_batches, 1)
        history.append((ep + 1, tr_loss, val_loss))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: val_loss={val_loss:.4f}")

        if no_improve >= cfg["training"]["patience"]:
            print(f"Early stopping generator at epoch {ep+1}")
            break

    if best_state is not None:
        generator.load_state_dict(best_state)

    if (save_path is not None) and (not skip_save):
        torch.save({"model": generator.state_dict(), "val_loss": best_val, "history": history}, save_path)
        print(f"Saved generator: {save_path} (best val loss={best_val:.4f})")
    else:
        print(f"Skip-save={skip_save}. Best generator val loss={best_val:.4f}")

    return best_val, history


def train_student_epoch(student, teacher, generator, loader, opt, device, cfg, temp):
    student.train()
    teacher.eval()
    generator.eval()

    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        y = batch["label"].to(device)

        opt.zero_grad()
        z_s, e_s = student(x_hlt, m_hlt, return_embedding=True)
        z_s = z_s.squeeze(1)

        # Sample offline candidates from generator and run teacher
        x_off_list, m_off_list = generator.sample(
            x_hlt, m_hlt,
            n_views=cfg["student"]["k_samples"],
            min_sigma=cfg["generator"]["min_sigma"],
            global_noise=cfg["generator"]["global_noise"],
        )

        logits_list = []
        emb_list = []
        with torch.no_grad():
            for x_off_k, m_off_k in zip(x_off_list, m_off_list):
                z_t, e_t = teacher(x_off_k, m_off_k, return_embedding=True)
                logits_list.append(z_t.squeeze(1))
                emb_list.append(e_t)

        probs, p_bar, w_u = teacher_ensemble_stats(
            logits_list, temp, cfg["kd"]["tau_u"], cfg["kd"]["w_min"]
        )

        # Supervised hard label
        loss_sup = F.binary_cross_entropy_with_logits(z_s, y)

        # KD to ensemble mean
        loss_kd = kd_loss_to_probs(z_s, p_bar, temp)
        loss_kd = (w_u * loss_kd).mean()

        # Optional KD to all samples
        loss_kd_all = kd_loss_to_probs(z_s.unsqueeze(0).expand_as(probs), probs, temp)
        loss_kd_all = (w_u.unsqueeze(0) * loss_kd_all).mean()

        # Embedding alignment to teacher mean
        emb_stack = torch.stack([F.normalize(e, dim=1) for e in emb_list], dim=0)
        e_bar = emb_stack.mean(dim=0)
        e_s_n = F.normalize(e_s, dim=1)
        loss_emb = rep_loss_cosine(e_s_n, e_bar)
        loss_emb = (w_u * loss_emb).mean()

        loss = cfg["student"]["a_sup"] * loss_sup
        loss = loss + cfg["student"]["a_kd"] * (loss_kd + cfg["kd"]["beta_all"] * loss_kd_all)
        loss = loss + cfg["student"]["a_emb"] * loss_emb

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.append(torch.sigmoid(z_s).detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return total_loss / len(labs), roc_auc_score(labs, preds)


def fit_student(student, teacher, generator, train_loader, val_loader, device, cfg, save_path=None, skip_save=False):
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    sch = get_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["training"]["epochs"])

    best_auc, best_state, no_improve = 0.0, None, 0
    history = []

    for ep in tqdm(range(cfg["training"]["epochs"]), desc="Student A2"):
        tr_loss, tr_auc = train_student_epoch(
            student, teacher, generator, train_loader, opt, device, cfg,
            temp=cfg["kd"]["temperature"],
        )
        va_auc, _, _ = evaluate(student, val_loader, device, "hlt", "mask_hlt")
        sch.step()

        history.append((ep + 1, tr_loss, tr_auc, va_auc))

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_auc:.4f}")

        if no_improve >= cfg["training"]["patience"]:
            print(f"Early stopping student at epoch {ep+1} (best val AUC={best_auc:.4f})")
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    if (save_path is not None) and (not skip_save):
        torch.save({"model": student.state_dict(), "auc": best_auc, "history": history}, save_path)
        print(f"Saved: {save_path} (best val AUC={best_auc:.4f})")
    else:
        print(f"Skip-save={skip_save}. Best val AUC={best_auc:.4f}")

    return best_auc, history


# ----------------------------- Hyperparameter Scheduling ----------------------------- #
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
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "offline_multi"))
    parser.add_argument("--device", type=str, default="cpu")

    # KD + A2 hyperparameters
    parser.add_argument("--kd_temp", type=float, default=CONFIG["kd"]["temperature"], help="KD temperature")
    parser.add_argument("--a_kd", type=float, default=CONFIG["student"]["a_kd"], help="Weight on KD term")
    parser.add_argument("--a_sup", type=float, default=CONFIG["student"]["a_sup"], help="Weight on supervised term")
    parser.add_argument("--a_emb", type=float, default=CONFIG["student"]["a_emb"], help="Weight on embedding term")
    parser.add_argument("--beta_all", type=float, default=CONFIG["kd"]["beta_all"], help="Weight on KD-to-all term")
    parser.add_argument("--tau_u", type=float, default=CONFIG["kd"]["tau_u"], help="Uncertainty temperature")
    parser.add_argument("--w_min", type=float, default=CONFIG["kd"]["w_min"], help="Min uncertainty weight clamp")
    parser.add_argument("--k_samples", type=int, default=CONFIG["student"]["k_samples"], help="Offline samples per HLT")

    # Generator knobs
    parser.add_argument("--gen_epochs", type=int, default=CONFIG["generator"]["epochs"], help="Generator epochs")
    parser.add_argument("--gen_lr", type=float, default=CONFIG["generator"]["lr"], help="Generator learning rate")
    parser.add_argument("--gen_lambda_mask", type=float, default=CONFIG["generator"]["lambda_mask"], help="Generator mask loss weight")
    parser.add_argument("--gen_lambda_perc", type=float, default=CONFIG["generator"]["lambda_perc"], help="Generator perceptual loss weight")
    parser.add_argument("--gen_lambda_logit", type=float, default=CONFIG["generator"]["lambda_logit"], help="Generator logit loss weight")
    parser.add_argument("--gen_min_sigma", type=float, default=CONFIG["generator"]["min_sigma"], help="Generator min sigma")
    parser.add_argument("--gen_global_noise", type=float, default=CONFIG["generator"]["global_noise"], help="Generator global noise scale")
    parser.add_argument("--generator_checkpoint", type=str, default=None, help="Load generator checkpoint (skip training)")
    parser.add_argument("--skip_generator", action="store_true", help="Skip generator training (requires checkpoint)")
    parser.add_argument("--run_name", type=str, default="default", help="Unique name for this hyperparameter run")

    # Pre-trained model loading (for hyperparameter search efficiency)
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Path to pre-trained teacher model (skips teacher training)")
    parser.add_argument("--baseline_checkpoint", type=str, default=None, help="Path to pre-trained baseline model (skips baseline training)")
    parser.add_argument("--skip_save_models", action="store_true", help="Skip saving model weights (save space during hyperparameter search)")

    args = parser.parse_args()

    CONFIG["kd"]["temperature"] = float(args.kd_temp)
    CONFIG["kd"]["beta_all"] = float(args.beta_all)
    CONFIG["kd"]["tau_u"] = float(args.tau_u)
    CONFIG["kd"]["w_min"] = float(args.w_min)

    CONFIG["student"]["a_kd"] = float(args.a_kd)
    CONFIG["student"]["a_sup"] = float(args.a_sup)
    CONFIG["student"]["a_emb"] = float(args.a_emb)
    CONFIG["student"]["k_samples"] = int(args.k_samples)

    CONFIG["generator"]["epochs"] = int(args.gen_epochs)
    CONFIG["generator"]["lr"] = float(args.gen_lr)
    CONFIG["generator"]["lambda_mask"] = float(args.gen_lambda_mask)
    CONFIG["generator"]["lambda_perc"] = float(args.gen_lambda_perc)
    CONFIG["generator"]["lambda_logit"] = float(args.gen_lambda_logit)
    CONFIG["generator"]["min_sigma"] = float(args.gen_min_sigma)
    CONFIG["generator"]["global_noise"] = float(args.gen_global_noise)

    # Create unique save directory for this hyperparameter run
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
    teacher_path  = save_dir / "teacher.pt"
    baseline_path = save_dir / "baseline.pt"
    generator_path = save_dir / "generator.pt"
    student_path  = save_dir / "student.pt"

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
        # Train teacher from scratch
        opt = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_auc_teacher, best_state, no_improve = 0.0, None, 0
        history_teacher = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Teacher"):
            train_loss, train_auc = train_standard(teacher, train_loader, opt, device, "off", "mask_off")
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
        # Train baseline from scratch
        opt = torch.optim.AdamW(baseline.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_auc_baseline, best_state, no_improve = 0.0, None, 0
        history_baseline = []

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Baseline"):
            train_loss, train_auc = train_standard(baseline, train_loader, opt, device, "hlt", "mask_hlt")
            val_auc, _, _ = evaluate(baseline, val_loader, device, "hlt", "mask_hlt")
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
        else:
            print(f"Skipped saving baseline model (best val AUC={best_auc_baseline:.4f})")

    # ------------------- STEP 3: Generator (HLT -> offline) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 3: GENERATOR (HLT -> offline samples)")
    print("=" * 70)

    generator = OfflineGenerator(
        input_dim=7,
        embed_dim=CONFIG["generator"]["embed_dim"],
        num_heads=CONFIG["generator"]["num_heads"],
        num_layers=CONFIG["generator"]["num_layers"],
        ff_dim=CONFIG["generator"]["ff_dim"],
        dropout=CONFIG["generator"]["dropout"],
    ).to(device)

    if args.generator_checkpoint is not None:
        print(f"Loading generator checkpoint: {args.generator_checkpoint}")
        ckpt = torch.load(args.generator_checkpoint, map_location=device)
        generator.load_state_dict(ckpt["model"])
    elif args.skip_generator:
        raise ValueError("skip_generator set but no generator_checkpoint provided.")
    else:
        fit_generator(
            generator, teacher, train_loader, val_loader, device, CONFIG,
            save_path=generator_path,
            skip_save=args.skip_save_models,
        )

    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False

    # ------------------- STEP 4: Student A2 (HLT + stochastic offline KD) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 4: STUDENT A2 (HLT + offline sampler KD)")
    print("=" * 70)

    student = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
    best_auc_student, _ = fit_student(
        student, teacher, generator, train_loader, val_loader, device, CONFIG,
        save_path=student_path,
        skip_save=args.skip_save_models,
    )

    # ------------------- Final evaluation on TEST ------------------- #
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    auc_teacher, preds_teacher, labs = evaluate(teacher, test_loader, device, "off", "mask_off")
    auc_baseline, preds_baseline, _ = evaluate(baseline, test_loader, device, "hlt", "mask_hlt")
    auc_student, preds_student, _ = evaluate(student, test_loader, device, "hlt", "mask_hlt")

    print(f"\n{'Model':<40} {'AUC':>10}")
    print("-" * 52)
    print(f"{'Teacher (Offline)':<40} {auc_teacher:>10.4f}")
    print(f"{'Baseline HLT (no KD)':<40} {auc_baseline:>10.4f}")
    print(f"{'Student A2 (offline sampler)':<40} {auc_student:>10.4f}")
    print("-" * 52)

    degradation = auc_teacher - auc_baseline
    improvement = auc_student - auc_baseline
    recovery = 100 * improvement / degradation if degradation > 0 else 0.0

    print("\nAnalysis:")
    print(f"  HLT Degradation: {degradation:.4f}")
    print(f"  Student Improvement:  {improvement:+.4f}")
    print(f"  Recovery:        {recovery:.1f}%")

    # Save results for later plotting
    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_s, tpr_s, _ = roc_curve(labs, preds_student)

    # Calculate Background Rejection at 50% signal efficiency (working point)
    wp = 0.5  # Working point (signal efficiency = TPR)
    idx_t = np.argmax(tpr_t >= wp)
    idx_b = np.argmax(tpr_b >= wp)
    idx_s = np.argmax(tpr_s >= wp)
    br_teacher = 1.0 / fpr_t[idx_t] if fpr_t[idx_t] > 0 else 0
    br_baseline = 1.0 / fpr_b[idx_b] if fpr_b[idx_b] > 0 else 0
    br_student = 1.0 / fpr_s[idx_s] if fpr_s[idx_s] > 0 else 0

    print(f"\nBackground Rejection at {wp*100:.0f}% signal efficiency:")
    print(f"  Teacher:  {br_teacher:.2f}")
    print(f"  Baseline: {br_baseline:.2f}")
    print(f"  Student:  {br_student:.2f}")

    np.savez(
        save_dir / "results.npz",
        labs=labs,
        preds_teacher=preds_teacher,
        preds_baseline=preds_baseline,
        preds_student=preds_student,
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_student=auc_student,
        br_teacher=br_teacher,
        br_baseline=br_baseline,
        br_student=br_student,
        fpr_teacher=fpr_t, tpr_teacher=tpr_t,
        fpr_baseline=fpr_b, tpr_baseline=tpr_b,
        fpr_student=fpr_s, tpr_student=tpr_s,
        k_samples=CONFIG["student"]["k_samples"],
        kd_temp=CONFIG["kd"]["temperature"],
        a_sup=CONFIG["student"]["a_sup"],
        a_kd=CONFIG["student"]["a_kd"],
        a_emb=CONFIG["student"]["a_emb"],
        beta_all=CONFIG["kd"]["beta_all"],
        tau_u=CONFIG["kd"]["tau_u"],
        w_min=CONFIG["kd"]["w_min"],
        gen_lambda_mask=CONFIG["generator"]["lambda_mask"],
        gen_lambda_perc=CONFIG["generator"]["lambda_perc"],
        gen_lambda_logit=CONFIG["generator"]["lambda_logit"],
        gen_min_sigma=CONFIG["generator"]["min_sigma"],
        gen_global_noise=CONFIG["generator"]["global_noise"],
    )

    # Plot ROC curves (swapped axes: TPR on x, FPR on y)
    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={auc_teacher:.3f})", color='crimson', linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline (AUC={auc_baseline:.3f})", color='steelblue', linewidth=2)
    plt.plot(tpr_s, fpr_s, ":", label=f"Student A2 (AUC={auc_student:.3f})", color='forestgreen', linewidth=2)
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
        f.write(f"  KD: temp={CONFIG['kd']['temperature']:.2f}, beta_all={CONFIG['kd']['beta_all']}, tau_u={CONFIG['kd']['tau_u']}, w_min={CONFIG['kd']['w_min']}\n")
        f.write(f"  Weights: a_sup={CONFIG['student']['a_sup']}, a_kd={CONFIG['student']['a_kd']}, a_emb={CONFIG['student']['a_emb']}\n")
        f.write(f"  K samples: {CONFIG['student']['k_samples']}\n")
        f.write(f"  Generator: lambda_mask={CONFIG['generator']['lambda_mask']}, lambda_perc={CONFIG['generator']['lambda_perc']}, lambda_logit={CONFIG['generator']['lambda_logit']}, min_sigma={CONFIG['generator']['min_sigma']}, global_noise={CONFIG['generator']['global_noise']}\n")
        f.write(f"  Background Rejection @ 50% efficiency: {br_student:.2f}\n")
        f.write(f"  AUC (Student): {auc_student:.4f}\n")
        f.write(f"  Saved to: {save_dir}\n")
        f.write("=" * 70 + "\n")

    print(f"\nSaved results to: {save_dir / 'results.npz'} and {save_dir / 'results.png'}")
    print(f"Logged to: {summary_file}")


if __name__ == "__main__":
    main()

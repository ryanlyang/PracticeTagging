#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge-based inverse HLT -> Offline sampler.

Idea:
  We know the forward HLT reconstruction effects. Use that knowledge to
  construct a probability distribution of plausible offline candidates
  for a given HLT jet, and train a classifier on those samples.

Pipeline:
  x_hlt -> inverse sampler -> {x_off^k} -> classifier -> y_hat

Optional:
  - Offline teacher for KD guidance
  - Baseline HLT classifier

The inverse sampler is not trained; it uses the known HLT effects and
offline priors (from training data) to generate plausible offline views.
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
    "knowledge": {
        "n_samples": 4,
        "n_samples_eval": 4,
        "inv_noise_scale": 1.0,
        "extra_count_scale": 1.0,
        "extra_lowpt_only": True,
        "split_frac": 0.5,
        "split_radius": 0.01,
        "split_min_frac": 0.05,
        "split_max_frac": 0.3,
        "conserve_pt": True,
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


# ----------------------------- HLT Simulation (forward) ----------------------------- #
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


# ----------------------------- Feature computation (numpy) ----------------------------- #
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


# ----------------------------- Feature computation (torch) ----------------------------- #
def compute_features_torch(const, mask):
    pt = torch.clamp(const[:, :, 0], min=1e-8)
    eta = torch.clamp(const[:, :, 1], min=-5, max=5)
    phi = const[:, :, 2]
    E = torch.clamp(const[:, :, 3], min=1e-8)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    mask_f = mask.float()
    jet_px = (px * mask_f).sum(dim=1, keepdim=True)
    jet_py = (py * mask_f).sum(dim=1, keepdim=True)
    jet_pz = (pz * mask_f).sum(dim=1, keepdim=True)
    jet_E = (E * mask_f).sum(dim=1, keepdim=True)

    jet_pt = torch.sqrt(jet_px**2 + jet_py**2) + 1e-8
    jet_p = torch.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * torch.log(torch.clamp((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = torch.atan2(jet_py, jet_px)

    delta_eta = eta - jet_eta
    delta_phi = torch.atan2(torch.sin(phi - jet_phi), torch.cos(phi - jet_phi))

    log_pt = torch.log(pt + 1e-8)
    log_E = torch.log(E + 1e-8)

    log_pt_rel = torch.log(pt / jet_pt + 1e-8)
    log_E_rel = torch.log(E / (jet_E + 1e-8) + 1e-8)

    delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

    features = torch.stack([delta_eta, delta_phi, log_pt, log_E, log_pt_rel, log_E_rel, delta_R], dim=-1)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = torch.clamp(features, -20, 20)
    features = features * mask.unsqueeze(-1)
    return features


def standardize_torch(feat, mask, means, stds):
    std = (feat - means) / stds
    std = torch.clamp(std, -10, 10)
    std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    std = std * mask.unsqueeze(-1)
    return std


# ----------------------------- Dataset ----------------------------- #
class JetDataset(Dataset):
    def __init__(self, feat_off, feat_hlt, labels, mask_off, mask_hlt, off_const, hlt_const):
        self.off = torch.tensor(feat_off, dtype=torch.float32)
        self.hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.off_const = torch.tensor(off_const, dtype=torch.float32)
        self.hlt_const = torch.tensor(hlt_const, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "off": self.off[i],
            "hlt": self.hlt[i],
            "mask_off": self.mask_off[i],
            "mask_hlt": self.mask_hlt[i],
            "off_const": self.off_const[i],
            "hlt_const": self.hlt_const[i],
            "label": self.labels[i],
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

        if return_attention and return_embedding:
            return logits, attn_weights.squeeze(1), z
        if return_attention:
            return logits, attn_weights.squeeze(1)
        if return_embedding:
            return logits, z
        return logits


# ----------------------------- KD + consistency utils ----------------------------- #
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


def kd_loss_basic(student_logits, teacher_logits, T):
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


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def get_temperature_schedule(epoch, total_epochs, T_init, T_final):
    if T_final is None:
        return T_init
    return T_init + (T_final - T_init) * (epoch / max(total_epochs - 1, 1))


def get_alpha_schedule(epoch, total_epochs, alpha_init, alpha_final):
    if alpha_final is None:
        return alpha_init
    return alpha_init + (alpha_final - alpha_init) * (epoch / max(total_epochs - 1, 1))


# ----------------------------- Knowledge sampler ----------------------------- #
class KnowledgeInverseSampler:
    def __init__(self, off_const, off_mask, hlt_mask, cfg):
        hcfg = cfg["hlt_effects"]
        kcfg = cfg["knowledge"]
        self.pt_res = hcfg["pt_resolution"]
        self.eta_res = hcfg["eta_resolution"]
        self.phi_res = hcfg["phi_resolution"]
        self.pt_thr_off = hcfg["pt_threshold_offline"]
        self.pt_thr_hlt = hcfg["pt_threshold_hlt"]
        self.merge_radius = hcfg["merge_radius"]
        self.inv_noise_scale = kcfg["inv_noise_scale"]
        self.extra_count_scale = kcfg["extra_count_scale"]
        self.extra_lowpt_only = kcfg["extra_lowpt_only"]
        self.split_frac = kcfg["split_frac"]
        self.split_radius = kcfg["split_radius"]
        self.split_min_frac = kcfg["split_min_frac"]
        self.split_max_frac = kcfg["split_max_frac"]
        self.conserve_pt = kcfg["conserve_pt"]

        off_mask_np = off_mask.astype(np.bool_)
        hlt_mask_np = hlt_mask.astype(np.bool_)
        extra_counts = off_mask_np.sum(axis=1) - hlt_mask_np.sum(axis=1)
        extra_counts = np.clip(extra_counts, 0, None).astype(int)
        self.extra_counts = extra_counts[extra_counts >= 0]

        pt = off_const[:, :, 0][off_mask_np]
        eta = off_const[:, :, 1][off_mask_np]
        phi = off_const[:, :, 2][off_mask_np]
        prior_all = np.stack([pt, eta, phi], axis=-1).astype(np.float32)

        lowpt = (pt >= self.pt_thr_off) & (pt < self.pt_thr_hlt)
        prior_low = np.stack([pt[lowpt], eta[lowpt], phi[lowpt]], axis=-1).astype(np.float32)

        self.prior_all = torch.tensor(prior_all, dtype=torch.float32)
        self.prior_low = torch.tensor(prior_low, dtype=torch.float32)

    def _wrap_phi(self, phi):
        return torch.atan2(torch.sin(phi), torch.cos(phi))

    def _sample_prior(self, num, device):
        if self.extra_lowpt_only and self.prior_low.numel() > 0:
            src = self.prior_low
        else:
            src = self.prior_all
        if src.numel() == 0:
            return torch.zeros((num, 3), device=device)
        idx = torch.randint(0, src.size(0), (num,), device=src.device)
        return src[idx].to(device)

    def _sample_extra_counts(self, batch_size):
        if self.extra_counts.size == 0:
            return np.zeros(batch_size, dtype=int)
        counts = np.random.choice(self.extra_counts, size=batch_size, replace=True)
        counts = np.maximum(0, np.round(counts * self.extra_count_scale).astype(int))
        return counts

    def sample_once(self, hlt_const, hlt_mask):
        device = hlt_const.device
        bsz, seq_len, _ = hlt_const.shape

        pt_noise = torch.randn(bsz, seq_len, device=device) * (self.pt_res * self.inv_noise_scale) + 1.0
        pt_noise = torch.clamp(pt_noise, 0.5, 1.5)
        eta_noise = torch.randn(bsz, seq_len, device=device) * (self.eta_res * self.inv_noise_scale)
        phi_noise = torch.randn(bsz, seq_len, device=device) * (self.phi_res * self.inv_noise_scale)

        off_pt = hlt_const[:, :, 0] / pt_noise
        off_pt = torch.clamp(off_pt, min=self.pt_thr_off)
        off_eta = torch.clamp(hlt_const[:, :, 1] - eta_noise, min=-5, max=5)
        off_phi = self._wrap_phi(hlt_const[:, :, 2] - phi_noise)
        off_E = off_pt * torch.cosh(off_eta)

        off_const = torch.stack([off_pt, off_eta, off_phi, off_E], dim=-1)
        off_mask = hlt_mask.clone()

        extra_counts = self._sample_extra_counts(bsz)
        for i in range(bsz):
            n_extra = int(extra_counts[i])
            if n_extra <= 0:
                continue

            avail_idx = torch.nonzero(~off_mask[i], as_tuple=False).flatten()
            if avail_idx.numel() == 0:
                continue

            n_extra = min(n_extra, avail_idx.numel())
            if n_extra <= 0:
                continue

            valid_idx = torch.nonzero(off_mask[i], as_tuple=False).flatten()
            n_split = 0
            if valid_idx.numel() > 0 and self.split_frac > 0:
                n_split = int(round(n_extra * self.split_frac))
                n_split = min(n_split, valid_idx.numel(), n_extra)

            idx_ptr = 0
            if n_split > 0:
                perm_valid = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
                perm_avail = avail_idx[torch.randperm(avail_idx.numel(), device=device)]
                for j in range(n_split):
                    base_idx = perm_valid[j].item()
                    slot_idx = perm_avail[j].item()
                    frac = torch.empty(1, device=device).uniform_(self.split_min_frac, self.split_max_frac).item()
                    pt_base = off_const[i, base_idx, 0]
                    pt_split = pt_base * frac
                    if self.conserve_pt:
                        off_const[i, base_idx, 0] = torch.clamp(pt_base - pt_split, min=self.pt_thr_off)
                        off_const[i, base_idx, 3] = off_const[i, base_idx, 0] * torch.cosh(off_const[i, base_idx, 1])
                    eta_split = off_const[i, base_idx, 1] + torch.randn((), device=device) * self.split_radius
                    phi_split = self._wrap_phi(off_const[i, base_idx, 2] + torch.randn((), device=device) * self.split_radius)
                    eta_split = torch.clamp(eta_split, min=-5, max=5)
                    pt_split = torch.clamp(pt_split, min=self.pt_thr_off)
                    E_split = pt_split * torch.cosh(eta_split)
                    off_const[i, slot_idx] = torch.stack([pt_split, eta_split, phi_split, E_split])
                    off_mask[i, slot_idx] = True
                idx_ptr = n_split

            remaining = n_extra - idx_ptr
            if remaining > 0:
                remain_slots = avail_idx[idx_ptr:idx_ptr + remaining]
                extras = self._sample_prior(remaining, device)
                pt_e = torch.clamp(extras[:, 0], min=self.pt_thr_off)
                eta_e = torch.clamp(extras[:, 1], min=-5, max=5)
                phi_e = self._wrap_phi(extras[:, 2])
                E_e = pt_e * torch.cosh(eta_e)
                off_const[i, remain_slots] = torch.stack([pt_e, eta_e, phi_e, E_e], dim=-1)
                off_mask[i, remain_slots] = True

        off_const = off_const * off_mask.unsqueeze(-1)
        return off_const, off_mask

    def sample(self, hlt_const, hlt_mask, n_samples):
        views, masks = [], []
        for _ in range(n_samples):
            off_const, off_mask = self.sample_once(hlt_const, hlt_mask)
            views.append(off_const)
            masks.append(off_mask)
        return views, masks


# ----------------------------- Train / eval ----------------------------- #
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


def train_knowledge_epoch(model, teacher, loader, sampler, device, cfg, feat_means, feat_stds,
                          opt, use_kd=False, temperature=7.0, alpha_kd=0.5):
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss = 0.0
    preds, labs = [], []
    a_attn = cfg["kd"]["alpha_attn"]
    a_rep = cfg["kd"]["alpha_rep"]
    a_nce = cfg["kd"]["alpha_nce"]
    use_conf = cfg["kd"]["use_conf_weighted_kd"]

    for batch in loader:
        hlt_const = batch["hlt_const"].to(device)
        hlt_mask = batch["mask_hlt"].to(device)
        y = batch["label"].to(device)
        hlt_const, hlt_mask = ensure_nonempty_mask(hlt_const, hlt_mask)

        opt.zero_grad()
        opt_views, opt_masks = sampler.sample(hlt_const, hlt_mask, cfg["knowledge"]["n_samples"])

        views_probs = []
        views_emb = []
        loss_kd = torch.tensor(0.0, device=device)
        loss_rep = torch.tensor(0.0, device=device)
        loss_nce = torch.tensor(0.0, device=device)
        loss_attn = torch.tensor(0.0, device=device)

        for off_const, off_mask in zip(opt_views, opt_masks):
            feat = compute_features_torch(off_const, off_mask)
            feat = standardize_torch(feat, off_mask, feat_means, feat_stds)
            logits, attn, emb = model(feat, off_mask, return_attention=True, return_embedding=True)
            logits = logits.squeeze(1)
            views_probs.append(safe_sigmoid(logits))
            views_emb.append(emb)

            if use_kd and teacher is not None:
                t_logits, t_attn, t_emb = teacher(feat, off_mask, return_attention=True, return_embedding=True)
                t_logits = t_logits.squeeze(1)
                if use_conf:
                    loss_kd = loss_kd + kd_loss_conf_weighted(logits, t_logits, temperature)
                else:
                    loss_kd = loss_kd + kd_loss_basic(logits, t_logits, temperature)
                if a_rep > 0:
                    loss_rep = loss_rep + cosine_embed_loss(emb, t_emb).mean()
                if a_nce > 0:
                    loss_nce = loss_nce + info_nce_loss(emb, t_emb, tau=cfg["kd"]["tau_nce"])
                if a_attn > 0:
                    loss_attn = loss_attn + attn_kl_loss(attn, t_attn, off_mask, off_mask)

        n_samples = max(1, cfg["knowledge"]["n_samples"])
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.append(p_mean.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labs = np.concatenate(labs, axis=0)
    return total_loss / len(labs), roc_auc_score(labs, preds)


def fit_knowledge(model, teacher, train_loader, val_loader, sampler, device, cfg, feat_means, feat_stds,
                  use_kd=False, temp_init=7.0, temp_final=None, alpha_init=0.5, alpha_final=None,
                  save_path=None, skip_save=False):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    sch = get_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["training"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    history = []

    for ep in tqdm(range(cfg["training"]["epochs"]), desc="Knowledge-Res"):
        temp = get_temperature_schedule(ep, cfg["training"]["epochs"], temp_init, temp_final)
        alpha = get_alpha_schedule(ep, cfg["training"]["epochs"], alpha_init, alpha_final)
        tr_loss, tr_auc = train_knowledge_epoch(
            model, teacher, train_loader, sampler, device, cfg, feat_means, feat_stds,
            opt=opt, use_kd=use_kd, temperature=temp, alpha_kd=alpha
        )
        val_auc, _ = evaluate_knowledge(model, val_loader, sampler, device, cfg, feat_means, feat_stds)
        sch.step()

        history.append((ep + 1, tr_loss, tr_auc, val_auc, temp, alpha))

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")

        if no_improve >= cfg["training"]["patience"]:
            print(f"Early stopping Knowledge-Res at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path is not None and not skip_save:
        torch.save({"model": model.state_dict(), "auc": best_auc, "history": history}, save_path)
    return best_auc, history


@torch.no_grad()
def evaluate_knowledge(model, loader, sampler, device, cfg, feat_means, feat_stds):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        hlt_const = batch["hlt_const"].to(device)
        hlt_mask = batch["mask_hlt"].to(device)
        y = batch["label"].to(device)
        hlt_const, hlt_mask = ensure_nonempty_mask(hlt_const, hlt_mask)

        views, masks = sampler.sample(hlt_const, hlt_mask, cfg["knowledge"]["n_samples_eval"])
        probs = []
        for off_const, off_mask in zip(views, masks):
            feat = compute_features_torch(off_const, off_mask)
            feat = standardize_torch(feat, off_mask, feat_means, feat_stds)
            logits = model(feat, off_mask).squeeze(1)
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


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=100000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "knowledge_res"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_name", type=str, default="default")

    parser.add_argument("--run_teacher", action="store_true")
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--run_student", action="store_true")
    parser.add_argument("--use_kd", action="store_true")

    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--baseline_checkpoint", type=str, default=None)
    parser.add_argument("--student_checkpoint", type=str, default=None)
    parser.add_argument("--compare_teacher_checkpoint", type=str, default=None)
    parser.add_argument("--compare_baseline_checkpoint", type=str, default=None)
    parser.add_argument("--compare_student_checkpoint", type=str, default=None)
    parser.add_argument("--skip_save_models", action="store_true")

    parser.add_argument("--knowledge_samples", type=int, default=CONFIG["knowledge"]["n_samples"])
    parser.add_argument("--knowledge_eval_samples", type=int, default=CONFIG["knowledge"]["n_samples_eval"])
    parser.add_argument("--inv_noise_scale", type=float, default=CONFIG["knowledge"]["inv_noise_scale"])
    parser.add_argument("--extra_count_scale", type=float, default=CONFIG["knowledge"]["extra_count_scale"])
    parser.add_argument("--extra_lowpt_only", action="store_true", default=CONFIG["knowledge"]["extra_lowpt_only"])
    parser.add_argument("--split_frac", type=float, default=CONFIG["knowledge"]["split_frac"])
    parser.add_argument("--split_radius", type=float, default=CONFIG["knowledge"]["split_radius"])
    parser.add_argument("--split_min_frac", type=float, default=CONFIG["knowledge"]["split_min_frac"])
    parser.add_argument("--split_max_frac", type=float, default=CONFIG["knowledge"]["split_max_frac"])
    parser.add_argument("--no_conserve_pt", action="store_true")

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

    args = parser.parse_args()

    CONFIG["knowledge"]["n_samples"] = args.knowledge_samples
    CONFIG["knowledge"]["n_samples_eval"] = args.knowledge_eval_samples
    CONFIG["knowledge"]["inv_noise_scale"] = args.inv_noise_scale
    CONFIG["knowledge"]["extra_count_scale"] = args.extra_count_scale
    CONFIG["knowledge"]["extra_lowpt_only"] = args.extra_lowpt_only
    CONFIG["knowledge"]["split_frac"] = args.split_frac
    CONFIG["knowledge"]["split_radius"] = args.split_radius
    CONFIG["knowledge"]["split_min_frac"] = args.split_min_frac
    CONFIG["knowledge"]["split_max_frac"] = args.split_max_frac
    CONFIG["knowledge"]["conserve_pt"] = not args.no_conserve_pt

    CONFIG["consistency"]["conf_power"] = args.conf_power
    CONFIG["consistency"]["conf_min"] = args.conf_min
    CONFIG["kd"]["temperature"] = args.temp_init
    CONFIG["kd"]["alpha_kd"] = args.alpha_init
    CONFIG["kd"]["alpha_attn"] = args.alpha_attn
    CONFIG["kd"]["alpha_rep"] = args.alpha_rep
    CONFIG["kd"]["alpha_nce"] = args.alpha_nce
    CONFIG["kd"]["tau_nce"] = args.tau_nce
    CONFIG["kd"]["use_conf_weighted_kd"] = (not args.no_conf_kd)

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

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

    print("Applying HLT effects...")
    constituents_hlt, masks_hlt = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=RANDOM_SEED)

    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0

    print("Computing features...")
    features_off = compute_features(constituents_off, masks_off)
    features_hlt = compute_features(constituents_hlt, masks_hlt)

    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = standardize(features_hlt, masks_hlt, feat_means, feat_stds)

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

    train_ds = JetDataset(
        features_off_std[train_idx], features_hlt_std[train_idx], all_labels[train_idx],
        masks_off[train_idx], masks_hlt[train_idx],
        constituents_off[train_idx], constituents_hlt[train_idx]
    )
    val_ds = JetDataset(
        features_off_std[val_idx], features_hlt_std[val_idx], all_labels[val_idx],
        masks_off[val_idx], masks_hlt[val_idx],
        constituents_off[val_idx], constituents_hlt[val_idx]
    )
    test_ds = JetDataset(
        features_off_std[test_idx], features_hlt_std[test_idx], all_labels[test_idx],
        masks_off[test_idx], masks_hlt[test_idx],
        constituents_off[test_idx], constituents_hlt[test_idx]
    )

    BS = CONFIG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)

    sampler = KnowledgeInverseSampler(
        constituents_off[train_idx], masks_off[train_idx], masks_hlt[train_idx], CONFIG
    )

    feat_means_t = torch.tensor(feat_means, dtype=torch.float32, device=device)
    feat_stds_t = torch.tensor(feat_stds, dtype=torch.float32, device=device)

    teacher = None
    baseline = None
    student = None
    compare_models = {}

    teacher_path = save_dir / "teacher.pt"
    baseline_path = save_dir / "baseline.pt"
    student_path = save_dir / "knowledge_student.pt"

    if args.run_teacher:
        print("\nSTEP 1: TEACHER (Offline)")
        teacher = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
        if args.teacher_checkpoint is not None:
            ckpt = torch.load(args.teacher_checkpoint, map_location=device)
            teacher.load_state_dict(ckpt["model"])
        else:
            fit_classifier(teacher, train_loader, val_loader, device, CONFIG, "off", "mask_off", "Teacher")
            if not args.skip_save_models:
                torch.save({"model": teacher.state_dict()}, teacher_path)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
    elif args.teacher_checkpoint is not None:
        teacher = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
        ckpt = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(ckpt["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    if args.compare_teacher_checkpoint is not None:
        compare_teacher = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
        ckpt = torch.load(args.compare_teacher_checkpoint, map_location=device)
        compare_teacher.load_state_dict(ckpt["model"])
        compare_teacher.eval()
        for p in compare_teacher.parameters():
            p.requires_grad = False
        compare_models["teacher"] = compare_teacher

    if args.run_baseline:
        print("\nSTEP 2: BASELINE HLT")
        baseline = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
        if args.baseline_checkpoint is not None:
            ckpt = torch.load(args.baseline_checkpoint, map_location=device)
            baseline.load_state_dict(ckpt["model"])
        else:
            fit_classifier(baseline, train_loader, val_loader, device, CONFIG, "hlt", "mask_hlt", "Baseline")
            if not args.skip_save_models:
                torch.save({"model": baseline.state_dict()}, baseline_path)

    if args.compare_baseline_checkpoint is not None:
        compare_baseline = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
        ckpt = torch.load(args.compare_baseline_checkpoint, map_location=device)
        compare_baseline.load_state_dict(ckpt["model"])
        compare_baseline.eval()
        compare_models["baseline"] = compare_baseline

    if args.run_student:
        print("\nSTEP 3: KNOWLEDGE-RES STUDENT")
        student = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
        if args.student_checkpoint is not None:
            ckpt = torch.load(args.student_checkpoint, map_location=device)
            student.load_state_dict(ckpt["model"])
        else:
            fit_knowledge(
                student, teacher, train_loader, val_loader, sampler, device, CONFIG,
                feat_means_t, feat_stds_t, use_kd=args.use_kd,
                temp_init=args.temp_init, temp_final=args.temp_final,
                alpha_init=args.alpha_init, alpha_final=args.alpha_final,
                save_path=student_path, skip_save=args.skip_save_models,
            )

    if args.compare_student_checkpoint is not None:
        compare_student = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
        ckpt = torch.load(args.compare_student_checkpoint, map_location=device)
        compare_student.load_state_dict(ckpt["model"])
        compare_student.eval()
        compare_models["student"] = compare_student

    print("\nFINAL TEST EVALUATION")
    results = {}
    if teacher is not None:
        auc_teacher, preds_teacher, labs = evaluate(teacher, test_loader, device, "off", "mask_off")
        results["teacher"] = (auc_teacher, preds_teacher)
    else:
        labs = all_labels[test_idx]

    if baseline is not None:
        auc_baseline, preds_baseline, _ = evaluate(baseline, test_loader, device, "hlt", "mask_hlt")
        results["baseline"] = (auc_baseline, preds_baseline)

    if student is not None:
        auc_student, preds_student = evaluate_knowledge(student, test_loader, sampler, device, CONFIG, feat_means_t, feat_stds_t)
        results["student"] = (auc_student, preds_student)

    if "teacher" in compare_models and "teacher" not in results:
        auc_teacher, preds_teacher, _ = evaluate(compare_models["teacher"], test_loader, device, "off", "mask_off")
        results["teacher"] = (auc_teacher, preds_teacher)
    if "baseline" in compare_models and "baseline" not in results:
        auc_baseline, preds_baseline, _ = evaluate(compare_models["baseline"], test_loader, device, "hlt", "mask_hlt")
        results["baseline"] = (auc_baseline, preds_baseline)
    if "student" in compare_models and "student" not in results:
        auc_student, preds_student = evaluate_knowledge(
            compare_models["student"], test_loader, sampler, device, CONFIG, feat_means_t, feat_stds_t
        )
        results["student"] = (auc_student, preds_student)

    print(f"\n{'Model':<40} {'AUC':>10}")
    print("-" * 52)
    if "teacher" in results:
        print(f"{'Teacher (Offline)':<40} {results['teacher'][0]:>10.4f}")
    if "baseline" in results:
        print(f"{'Baseline HLT':<40} {results['baseline'][0]:>10.4f}")
    if "student" in results:
        print(f"{'Knowledge-Res (HLT)':<40} {results['student'][0]:>10.4f}")
    print("-" * 52)

    if results:
        plt.figure(figsize=(8, 6))
        if "teacher" in results:
            fpr_t, tpr_t, _ = roc_curve(labs, results["teacher"][1])
            plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={results['teacher'][0]:.3f})", color="crimson", linewidth=2)
        if "baseline" in results:
            fpr_b, tpr_b, _ = roc_curve(labs, results["baseline"][1])
            plt.plot(tpr_b, fpr_b, "--", label=f"Baseline (AUC={results['baseline'][0]:.3f})", color="steelblue", linewidth=2)
        if "student" in results:
            fpr_s, tpr_s, _ = roc_curve(labs, results["student"][1])
            plt.plot(tpr_s, fpr_s, "-", label=f"Knowledge-Res (AUC={results['student'][0]:.3f})", color="forestgreen", linewidth=2)
        plt.ylabel("False Positive Rate")
        plt.xlabel("True Positive Rate (Signal efficiency)")
        plt.legend(frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "results.png", dpi=300)
        plt.close()

    summary_file = Path(args.save_dir) / "hyperparameter_search_results.txt"
    with open(summary_file, "a") as f:
        f.write(f"\nRun: {args.run_name}\n")
        f.write(f"  Knowledge samples: train={CONFIG['knowledge']['n_samples']}, eval={CONFIG['knowledge']['n_samples_eval']}\n")
        f.write(f"  inv_noise_scale={CONFIG['knowledge']['inv_noise_scale']}, extra_scale={CONFIG['knowledge']['extra_count_scale']}, split_frac={CONFIG['knowledge']['split_frac']}\n")
        f.write(f"  KD temp: {args.temp_init:.2f}")
        if args.temp_final is not None:
            f.write(f" -> {args.temp_final:.2f}\n")
        else:
            f.write(" (constant)\n")
        f.write(f"  KD alpha: {args.alpha_init:.2f}")
        if args.alpha_final is not None:
            f.write(f" -> {args.alpha_final:.2f}\n")
        else:
            f.write(" (constant)\n")
        f.write(f"  KD weights: attn={CONFIG['kd']['alpha_attn']}, rep={CONFIG['kd']['alpha_rep']}, nce={CONFIG['kd']['alpha_nce']}, tau_nce={CONFIG['kd']['tau_nce']}, conf_weighted={CONFIG['kd']['use_conf_weighted_kd']}\n")
        f.write(f"  Consistency: conf_power={CONFIG['consistency']['conf_power']}, conf_min={CONFIG['consistency']['conf_min']}\n")
        if "student" in results:
            f.write(f"  AUC (Knowledge-Res): {results['student'][0]:.4f}\n")
        f.write(f"  Saved to: {save_dir}\n")
        f.write("=" * 70 + "\n")

    print(f"\nSaved results to: {save_dir / 'results.png'}")
    print(f"Logged to: {summary_file}")


if __name__ == "__main__":
    main()

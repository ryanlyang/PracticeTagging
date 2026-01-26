#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conditional diffusion denoiser for HLT smearing-only setup.

Goal:
  Learn p(x_off | x_hlt) where HLT effects are only resolution smearing
  (no merging, no efficiency loss). This learns to "unsmear" constituents.

Workflow:
  1) Load paired offline/HLT (HLT derived from offline via smearing).
  2) Standardize features using OFFLINE train stats.
  3) Train conditional diffusion model to predict noise epsilon.
  4) Sample unsmeared jets from HLT at val/test and report reconstruction metrics.

Notes:
  - Uses full-jet conditioning (better accuracy than token-only).
  - Supports cross-attention conditioning, self-conditioning, CFG, DDIM.
  - Uses EMA for sampling (higher quality).
  - Designed for accuracy over speed.
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


# ----------------------------- Column order ----------------------------- #
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
        "merge_enabled": False,
        "merge_radius": 0.0,
        "efficiency_loss": 0.0,
        "noise_enabled": False,
        "noise_fraction": 0.0,
    },
    "model": {
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 8,
        "ff_dim": 1024,
        "dropout": 0.1,
        "use_cross_attn": True,
        "self_cond": True,
    },
    "diffusion": {
        "timesteps": 1000,
        "schedule": "cosine",  # cosine or linear
        "pred_type": "eps",    # eps | x0 | v
        "x0_weight": 0.1,      # auxiliary x0 reconstruction loss
        "snr_weight": True,
        "snr_gamma": 5.0,
        "self_cond_prob": 0.5,
        "cond_drop_prob": 0.1,  # classifier-free guidance dropout
        "ema_decay": 0.995,
        "jet_loss_weight": 0.1,  # jet-level summary loss
    },
    "training": {
        "batch_size": 256,
        "epochs": 80,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 5,
        "patience": 15,
        "grad_clip": 1.0,
    },
    "classifier": {
        "embed_dim": 128,
        "num_heads": 8,
        "num_layers": 6,
        "ff_dim": 512,
        "dropout": 0.1,
        "epochs": 50,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 3,
        "patience": 15,
    },
    "sampling": {
        "sample_steps": 200,   # fewer than full steps for speed; set to timesteps for max accuracy
        "n_samples_eval": 1,   # number of diffusion samples to average at eval
        "method": "ddim",      # ddpm | ddim
        "guidance_scale": 1.5,
    },
}


# ----------------------------- HLT smearing only ----------------------------- #
def apply_hlt_effects_smear_only(const, mask, cfg, seed=42):
    np.random.seed(seed)
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    n_initial = int(hlt_mask.sum())

    # pT threshold (HLT)
    pt_threshold = hcfg["pt_threshold_hlt"]
    below_threshold = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0
    n_lost_threshold = int(below_threshold.sum())

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

    # Recalculate E (massless approx)
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)

    # No merging / efficiency loss
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0

    n_final = int(hlt_mask.sum())
    stats = {
        "n_initial": n_initial,
        "n_lost_threshold": n_lost_threshold,
        "n_final": n_final,
    }
    return hlt, hlt_mask, stats


# ----------------------------- Standardization ----------------------------- #
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


def unstandardize(feat_std, means, stds):
    return feat_std * stds + means


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


# ----------------------------- Dataset ----------------------------- #
class JetPairDataset(Dataset):
    def __init__(self, off_std, hlt_std, mask_off, mask_hlt):
        self.off = torch.tensor(off_std, dtype=torch.float32)
        self.hlt = torch.tensor(hlt_std, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)

    def __len__(self):
        return len(self.off)

    def __getitem__(self, i):
        # Use joint mask (tokens present in HLT); this is the reliable supervision
        joint = self.mask_hlt[i]
        return {
            "off": self.off[i],
            "hlt": self.hlt[i],
            "mask": joint,
        }


class JetDataset(Dataset):
    def __init__(self, feat, mask, labels):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "feat": self.feat[i],
            "mask": self.mask[i],
            "label": self.labels[i],
        }


# ----------------------------- Diffusion utils ----------------------------- #
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
    # sinusoidal embedding
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / (half - 1)
    ).to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


# ----------------------------- Model ----------------------------- #
class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        input_dim=4,
        embed_dim=256,
        num_heads=8,
        num_layers=8,
        ff_dim=1024,
        dropout=0.1,
        use_cross_attn=True,
        self_cond=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_cross_attn = use_cross_attn
        self.self_cond = self_cond
        self.x_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.c_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.self_cond:
            self.sc_proj = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        if self.use_cross_attn:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            dec_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cond_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        else:
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
        self.out = nn.Linear(embed_dim, input_dim)

    def forward(self, x_t, cond, mask, t, self_cond=None):
        # x_t, cond: (B, N, F) ; mask: (B, N)
        if cond is None:
            cond = torch.zeros_like(x_t)
        h = self.x_proj(x_t) + self.c_proj(cond)
        if self.self_cond and self_cond is not None:
            h = h + self.sc_proj(self_cond)
        t_emb = self.t_mlp(get_timestep_embedding(t, self.embed_dim))
        h = h + t_emb.unsqueeze(1)
        if self.use_cross_attn:
            mem = self.cond_encoder(self.c_proj(cond), src_key_padding_mask=~mask)
            h = self.decoder(h, mem, tgt_key_padding_mask=~mask, memory_key_padding_mask=~mask)
        else:
            h = self.encoder(h, src_key_padding_mask=~mask)
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


# ----------------------------- Training ----------------------------- #
def q_sample(x0, t, noise, alpha_bar):
    # x0: (B,N,F), t: (B,), alpha_bar: (T,)
    a_bar = alpha_bar[t].view(-1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise


def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask.unsqueeze(-1)
    denom = mask.sum() * pred.shape[-1]
    return diff.sum() / torch.clamp(denom, min=1.0)


def jet_summary(x, mask):
    # x: (B, N, 4) with [pt, eta, phi, E]
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


def train_epoch(model, ema, loader, opt, device, alpha_bar, cfg):
    model.train()
    total = 0.0
    count = 0
    T = alpha_bar.shape[0]
    for batch in loader:
        x0 = batch["off"].to(device)
        cond = batch["hlt"].to(device)
        mask = batch["mask"].to(device)

        # classifier-free guidance conditioning dropout
        if cfg["cond_drop_prob"] > 0:
            drop = torch.rand(x0.size(0), device=device) < cfg["cond_drop_prob"]
            if drop.any():
                cond = cond.clone()
                cond[drop] = 0.0

        t = torch.randint(0, T, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, noise, alpha_bar)

        # self-conditioning
        self_cond = None
        if cfg["self_cond_prob"] > 0 and model.self_cond:
            if torch.rand(()) < cfg["self_cond_prob"]:
                with torch.no_grad():
                    pred0 = model(x_t, cond, mask, t, self_cond=None)
                    a_bar = alpha_bar[t].view(-1, 1, 1)
                    if cfg["pred_type"] == "x0":
                        x0_sc = pred0
                    elif cfg["pred_type"] == "v":
                        x0_sc = predict_x0_from_v(x_t, pred0, a_bar)
                    else:
                        x0_sc = predict_x0_from_eps(x_t, pred0, a_bar)
                    self_cond = x0_sc.detach()

        opt.zero_grad()
        pred = model(x_t, cond, mask, t, self_cond=self_cond)

        a_bar = alpha_bar[t].view(-1, 1, 1)
        if cfg["pred_type"] == "x0":
            target = x0
            pred_x0 = pred
            pred_eps = predict_eps_from_x0(x_t, pred, a_bar)
        elif cfg["pred_type"] == "v":
            target = predict_v(x0, noise, a_bar)
            pred_x0 = predict_x0_from_v(x_t, pred, a_bar)
            pred_eps = predict_eps_from_x0(x_t, pred_x0, a_bar)
        else:
            target = noise
            pred_eps = pred
            pred_x0 = predict_x0_from_eps(x_t, pred, a_bar)

        loss_noise = masked_mse(pred, target, mask)
        if cfg["snr_weight"]:
            snr = compute_snr(a_bar)
            w = torch.clamp(snr, max=cfg["snr_gamma"]) / (snr + 1.0)
            loss_noise = loss_noise * w.mean()

        loss = loss_noise
        if cfg["x0_weight"] > 0:
            loss_x0 = masked_mse(pred_x0, x0, mask)
            loss = loss + cfg["x0_weight"] * loss_x0

        if cfg["jet_loss_weight"] > 0:
            jet_true = jet_summary(x0, mask)
            jet_pred = jet_summary(pred_x0, mask)
            jet_loss = F.l1_loss(jet_pred, jet_true)
            loss = loss + cfg["jet_loss_weight"] * jet_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["training"]["grad_clip"])
        opt.step()

        if ema is not None:
            ema.update(model)

        total += loss.item() * x0.size(0)
        count += x0.size(0)
    return total / max(count, 1)


@torch.no_grad()
def model_pred_eps(model, x, cond, mask, t, pred_type, alpha_bar, guidance_scale=1.0):
    # CFG: do conditional + unconditional
    if guidance_scale != 1.0:
        eps_cond = model(x, cond, mask, t)
        eps_uncond = model(x, torch.zeros_like(cond), mask, t)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    else:
        eps = model(x, cond, mask, t)

    a_bar = alpha_bar[t].view(-1, 1, 1)
    if pred_type == "x0":
        eps = predict_eps_from_x0(x, eps, a_bar)
    elif pred_type == "v":
        x0 = predict_x0_from_v(x, eps, a_bar)
        eps = predict_eps_from_x0(x, x0, a_bar)
    return eps


@torch.no_grad()
def sample_ddpm(model, cond, mask, betas, alpha, alpha_bar, steps, pred_type, guidance_scale):
    model.eval()
    device = cond.device
    T = betas.shape[0]
    if steps is None or steps > T:
        steps = T

    if steps < T:
        idx = torch.linspace(T - 1, 0, steps, device=device).long()
    else:
        idx = torch.arange(T - 1, -1, -1, device=device)

    x = torch.randn_like(cond)
    for t in idx:
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        eps = model_pred_eps(model, x, cond, mask, t_batch, pred_type, alpha_bar, guidance_scale)
        a_t = alpha[t]
        a_bar_t = alpha_bar[t]
        beta_t = betas[t]

        coef1 = 1.0 / torch.sqrt(a_t)
        coef2 = beta_t / torch.sqrt(1.0 - a_bar_t)
        mean = coef1 * (x - coef2 * eps)

        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean

        x = x * mask.unsqueeze(-1)
    return x


@torch.no_grad()
def sample_ddim(model, cond, mask, betas, alpha, alpha_bar, steps, pred_type, guidance_scale, eta=0.0):
    model.eval()
    device = cond.device
    T = betas.shape[0]
    if steps is None or steps > T:
        steps = T

    if steps < T:
        idx = torch.linspace(T - 1, 0, steps, device=device).long()
    else:
        idx = torch.arange(T - 1, -1, -1, device=device)

    x = torch.randn_like(cond)
    for i, t in enumerate(idx):
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        eps = model_pred_eps(model, x, cond, mask, t_batch, pred_type, alpha_bar, guidance_scale)
        a_bar_t = alpha_bar[t]
        x0 = predict_x0_from_eps(x, eps, a_bar_t)

        if i == len(idx) - 1:
            x = x0
            break

        t_next = idx[i + 1]
        a_bar_next = alpha_bar[t_next]
        sigma = eta * torch.sqrt((1 - a_bar_next) / (1 - a_bar_t) * (1 - a_bar_t / a_bar_next))
        noise = torch.randn_like(x)
        x = torch.sqrt(a_bar_next) * x0 + torch.sqrt(1 - a_bar_next - sigma ** 2) * eps + sigma * noise
        x = x * mask.unsqueeze(-1)
    return x


@torch.no_grad()
def eval_reconstruction(model, loader, device, betas, alpha, alpha_bar, means, stds, cfg, pred_type):
    model.eval()
    total_l1, total_l2, count = 0.0, 0.0, 0
    for batch in loader:
        x0 = batch["off"].to(device)
        cond = batch["hlt"].to(device)
        mask = batch["mask"].to(device)

        preds = []
        for _ in range(cfg["n_samples_eval"]):
            if cfg["method"] == "ddim":
                x0_pred = sample_ddim(
                    model, cond, mask, betas, alpha, alpha_bar,
                    steps=cfg["sample_steps"], pred_type=pred_type,
                    guidance_scale=cfg["guidance_scale"]
                )
            else:
                x0_pred = sample_ddpm(
                    model, cond, mask, betas, alpha, alpha_bar,
                    steps=cfg["sample_steps"], pred_type=pred_type,
                    guidance_scale=cfg["guidance_scale"]
                )
            preds.append(x0_pred)
        x0_pred = torch.stack(preds, dim=0).mean(dim=0)

        diff = (x0_pred - x0) * mask.unsqueeze(-1)
        l1 = diff.abs().sum()
        l2 = (diff ** 2).sum()
        denom = mask.sum() * x0.shape[-1]
        total_l1 += l1.item()
        total_l2 += l2.item()
        count += denom.item()

    l1 = total_l1 / max(count, 1.0)
    l2 = total_l2 / max(count, 1.0)
    return l1, l2


@torch.no_grad()
def generate_unsmeared_constituents(model, hlt_std, mask_hlt, betas, alpha, alpha_bar, cfg, pred_type, device):
    model.eval()
    n = hlt_std.shape[0]
    out = np.zeros_like(hlt_std)
    loader = DataLoader(
        JetDataset(hlt_std, mask_hlt, np.zeros(n, dtype=np.float32)),
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
    )
    idx = 0
    for batch in tqdm(loader, desc="UnsmearedGen"):
        x = batch["feat"].to(device)
        m = batch["mask"].to(device)
        preds = []
        for _ in range(cfg["n_samples_eval"]):
            if cfg["method"] == "ddim":
                x0_pred = sample_ddim(
                    model, x, m, betas, alpha, alpha_bar,
                    steps=cfg["sample_steps"], pred_type=pred_type,
                    guidance_scale=cfg["guidance_scale"]
                )
            else:
                x0_pred = sample_ddpm(
                    model, x, m, betas, alpha, alpha_bar,
                    steps=cfg["sample_steps"], pred_type=pred_type,
                    guidance_scale=cfg["guidance_scale"]
                )
            preds.append(x0_pred)
        x0_pred = torch.stack(preds, dim=0).mean(dim=0)
        bs = x0_pred.size(0)
        out[idx:idx + bs] = x0_pred.cpu().numpy()
        idx += bs
    return out


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unsmear"))
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=CONFIG["training"]["epochs"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["training"]["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["training"]["lr"])
    parser.add_argument("--weight_decay", type=float, default=CONFIG["training"]["weight_decay"])
    parser.add_argument("--timesteps", type=int, default=CONFIG["diffusion"]["timesteps"])
    parser.add_argument("--sample_steps", type=int, default=CONFIG["sampling"]["sample_steps"])
    parser.add_argument("--n_samples_eval", type=int, default=CONFIG["sampling"]["n_samples_eval"])
    parser.add_argument("--ema_decay", type=float, default=CONFIG["diffusion"]["ema_decay"])
    parser.add_argument("--schedule", type=str, default=CONFIG["diffusion"]["schedule"])
    parser.add_argument("--x0_weight", type=float, default=CONFIG["diffusion"]["x0_weight"])
    parser.add_argument("--pred_type", type=str, default=CONFIG["diffusion"]["pred_type"], choices=["eps", "x0", "v"])
    parser.add_argument("--snr_weight", action="store_true", default=CONFIG["diffusion"]["snr_weight"])
    parser.add_argument("--snr_gamma", type=float, default=CONFIG["diffusion"]["snr_gamma"])
    parser.add_argument("--self_cond_prob", type=float, default=CONFIG["diffusion"]["self_cond_prob"])
    parser.add_argument("--cond_drop_prob", type=float, default=CONFIG["diffusion"]["cond_drop_prob"])
    parser.add_argument("--jet_loss_weight", type=float, default=CONFIG["diffusion"]["jet_loss_weight"])
    parser.add_argument("--use_cross_attn", action="store_true", default=CONFIG["model"]["use_cross_attn"])
    parser.add_argument("--no_self_cond", action="store_true", help="Disable self-conditioning")
    parser.add_argument("--sampling_method", type=str, default=CONFIG["sampling"]["method"], choices=["ddpm", "ddim"])
    parser.add_argument("--guidance_scale", type=float, default=CONFIG["sampling"]["guidance_scale"])
    parser.add_argument("--skip_classifiers", action="store_true", help="Skip teacher/baseline/unsmear classifiers")
    args = parser.parse_args()

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

    print("Applying HLT effects (smearing only)...")
    constituents_hlt, masks_hlt, stats = apply_hlt_effects_smear_only(
        constituents_raw, mask_raw, CONFIG, seed=RANDOM_SEED
    )

    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0

    print("HLT Simulation Statistics:")
    print(f"  Offline particles: {stats['n_initial']:,}")
    print(f"  Lost to pT threshold ({CONFIG['hlt_effects']['pt_threshold_hlt']}): {stats['n_lost_threshold']:,}")
    print(f"  HLT particles: {stats['n_final']:,}")
    print(f"  Avg per jet: Offline={masks_off.sum(axis=1).mean():.1f}, HLT={masks_hlt.sum(axis=1).mean():.1f}")

    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Standardize in raw constituent space (pt, eta, phi, E)
    const_means, const_stds = get_stats(constituents_off, masks_off, train_idx)
    off_std = standardize(constituents_off, masks_off, const_means, const_stds)
    hlt_std = standardize(constituents_hlt, masks_hlt, const_means, const_stds)

    train_ds = JetPairDataset(off_std[train_idx], hlt_std[train_idx], masks_off[train_idx], masks_hlt[train_idx])
    val_ds = JetPairDataset(off_std[val_idx], hlt_std[val_idx], masks_off[val_idx], masks_hlt[val_idx])
    test_ds = JetPairDataset(off_std[test_idx], hlt_std[test_idx], masks_off[test_idx], masks_hlt[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ConditionalDenoiser(
        input_dim=4,
        embed_dim=CONFIG["model"]["embed_dim"],
        num_heads=CONFIG["model"]["num_heads"],
        num_layers=CONFIG["model"]["num_layers"],
        ff_dim=CONFIG["model"]["ff_dim"],
        dropout=CONFIG["model"]["dropout"],
        use_cross_attn=args.use_cross_attn,
        self_cond=not args.no_self_cond,
    ).to(device)
    ema = EMA(model, decay=args.ema_decay)

    betas = torch.tensor(make_beta_schedule(args.timesteps, args.schedule), dtype=torch.float32, device=device)
    alpha = 1.0 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], args.epochs)

    best_val = 1e9
    best_state = None
    best_state_ema = None
    no_improve = 0

    print("\n" + "=" * 70)
    print("TRAINING: CONDITIONAL DIFFUSION UNSMEAR")
    print("=" * 70)
    diff_cfg = {
        "pred_type": args.pred_type,
        "x0_weight": args.x0_weight,
        "snr_weight": args.snr_weight,
        "snr_gamma": args.snr_gamma,
        "self_cond_prob": args.self_cond_prob,
        "cond_drop_prob": args.cond_drop_prob,
        "jet_loss_weight": args.jet_loss_weight,
    }
    samp_cfg = {
        "sample_steps": args.sample_steps,
        "n_samples_eval": args.n_samples_eval,
        "method": args.sampling_method,
        "guidance_scale": args.guidance_scale,
    }

    for ep in tqdm(range(args.epochs), desc="Diffusion"):
        loss = train_epoch(model, ema, train_loader, opt, device, alpha_bar, diff_cfg)
        sch.step()

        if (ep + 1) % 5 == 0:
            # eval on val (EMA)
            eval_model = ConditionalDenoiser(
                input_dim=4,
                embed_dim=CONFIG["model"]["embed_dim"],
                num_heads=CONFIG["model"]["num_heads"],
                num_layers=CONFIG["model"]["num_layers"],
                ff_dim=CONFIG["model"]["ff_dim"],
                dropout=CONFIG["model"]["dropout"],
                use_cross_attn=args.use_cross_attn,
                self_cond=not args.no_self_cond,
            ).to(device)
            ema.apply_to(eval_model)
            val_l1, val_l2 = eval_reconstruction(
                eval_model, val_loader, device, betas, alpha, alpha_bar,
                const_means, const_stds, samp_cfg, args.pred_type
            )
            print(f"Ep {ep+1}: train_loss={loss:.6f}, val_l1={val_l1:.6f}, val_l2={val_l2:.6f}")
            if val_l1 < best_val:
                best_val = val_l1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_state_ema = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= CONFIG["training"]["patience"]:
                print(f"Early stopping at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        ema.shadow = best_state_ema

    # Final evaluation using EMA weights
    eval_model = ConditionalDenoiser(
        input_dim=4,
        embed_dim=CONFIG["model"]["embed_dim"],
        num_heads=CONFIG["model"]["num_heads"],
        num_layers=CONFIG["model"]["num_layers"],
        ff_dim=CONFIG["model"]["ff_dim"],
        dropout=CONFIG["model"]["dropout"],
        use_cross_attn=args.use_cross_attn,
        self_cond=not args.no_self_cond,
    ).to(device)
    ema.apply_to(eval_model)
    val_l1, val_l2 = eval_reconstruction(
        eval_model, val_loader, device, betas, alpha, alpha_bar,
        const_means, const_stds, samp_cfg, args.pred_type
    )
    test_l1, test_l2 = eval_reconstruction(
        eval_model, test_loader, device, betas, alpha, alpha_bar,
        const_means, const_stds, samp_cfg, args.pred_type
    )

    print("\nFinal reconstruction:")
    print(f"  Val L1: {val_l1:.6f} | Val L2: {val_l2:.6f}")
    print(f"  Test L1: {test_l1:.6f} | Test L2: {test_l2:.6f}")

    np.savez(
        save_root / "results.npz",
        val_l1=val_l1,
        val_l2=val_l2,
        test_l1=test_l1,
        test_l2=test_l2,
        feat_means=const_means,
        feat_stds=const_stds,
        timesteps=args.timesteps,
        sample_steps=args.sample_steps,
        n_samples_eval=args.n_samples_eval,
        pred_type=args.pred_type,
        sampling_method=args.sampling_method,
        guidance_scale=args.guidance_scale,
    )
    torch.save({"model": eval_model.state_dict()}, save_root / "unsmear_diffusion_ema.pt")
    torch.save({"model": model.state_dict()}, save_root / "unsmear_diffusion.pt")

    print(f"Saved results to: {save_root}")

    if args.skip_classifiers:
        return

    print("\n" + "=" * 70)
    print("CLASSIFIERS: Offline Teacher / HLT Baseline / Unsmeared Student")
    print("=" * 70)

    # Build 7-feature representations
    feat_off = compute_features(constituents_off, masks_off)
    feat_hlt = compute_features(constituents_hlt, masks_hlt)

    feat_means7, feat_stds7 = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, feat_means7, feat_stds7)
    feat_hlt_std = standardize(feat_hlt, masks_hlt, feat_means7, feat_stds7)

    # Generate unsmeared constituents (EMA model)
    unsmear_std = generate_unsmeared_constituents(
        eval_model, hlt_std, masks_hlt, betas, alpha, alpha_bar, samp_cfg, args.pred_type, device
    )
    unsmear_const = unstandardize(unsmear_std, const_means, const_stds)
    unsmear_const[:, :, 0] = np.clip(unsmear_const[:, :, 0], 0.0, None)
    unsmear_const[:, :, 1] = np.clip(unsmear_const[:, :, 1], -5.0, 5.0)
    unsmear_const[:, :, 2] = np.arctan2(np.sin(unsmear_const[:, :, 2]), np.cos(unsmear_const[:, :, 2]))
    unsmear_const[:, :, 3] = np.where(
        masks_hlt, unsmear_const[:, :, 0] * np.cosh(np.clip(unsmear_const[:, :, 1], -5.0, 5.0)), 0.0
    )
    feat_unsmear = compute_features(unsmear_const, masks_hlt)
    feat_unsmear_std = standardize(feat_unsmear, masks_hlt, feat_means7, feat_stds7)

    # Datasets / loaders
    BS = CONFIG["classifier"]["batch_size"]
    train_off = JetDataset(feat_off_std[train_idx], masks_off[train_idx], all_labels[train_idx])
    val_off = JetDataset(feat_off_std[val_idx], masks_off[val_idx], all_labels[val_idx])
    test_off = JetDataset(feat_off_std[test_idx], masks_off[test_idx], all_labels[test_idx])

    train_hlt = JetDataset(feat_hlt_std[train_idx], masks_hlt[train_idx], all_labels[train_idx])
    val_hlt = JetDataset(feat_hlt_std[val_idx], masks_hlt[val_idx], all_labels[val_idx])
    test_hlt = JetDataset(feat_hlt_std[test_idx], masks_hlt[test_idx], all_labels[test_idx])

    train_uns = JetDataset(feat_unsmear_std[train_idx], masks_hlt[train_idx], all_labels[train_idx])
    val_uns = JetDataset(feat_unsmear_std[val_idx], masks_hlt[val_idx], all_labels[val_idx])
    test_uns = JetDataset(feat_unsmear_std[test_idx], masks_hlt[test_idx], all_labels[test_idx])

    train_off_loader = DataLoader(train_off, batch_size=BS, shuffle=True, drop_last=True)
    val_off_loader = DataLoader(val_off, batch_size=BS, shuffle=False)
    test_off_loader = DataLoader(test_off, batch_size=BS, shuffle=False)

    train_hlt_loader = DataLoader(train_hlt, batch_size=BS, shuffle=True, drop_last=True)
    val_hlt_loader = DataLoader(val_hlt, batch_size=BS, shuffle=False)
    test_hlt_loader = DataLoader(test_hlt, batch_size=BS, shuffle=False)

    train_uns_loader = DataLoader(train_uns, batch_size=BS, shuffle=True, drop_last=True)
    val_uns_loader = DataLoader(val_uns, batch_size=BS, shuffle=False)
    test_uns_loader = DataLoader(test_uns, batch_size=BS, shuffle=False)

    # Teacher (offline)
    print("\n" + "=" * 70)
    print("Teacher (Offline)")
    print("=" * 70)
    cls_cfg = CONFIG["classifier"]
    teacher = ParticleTransformer(
        input_dim=7,
        embed_dim=cls_cfg["embed_dim"],
        num_heads=cls_cfg["num_heads"],
        num_layers=cls_cfg["num_layers"],
        ff_dim=cls_cfg["ff_dim"],
        dropout=cls_cfg["dropout"],
    ).to(device)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["classifier"]["lr"], weight_decay=CONFIG["classifier"]["weight_decay"])
    sch_t = get_scheduler(opt_t, CONFIG["classifier"]["warmup_epochs"], CONFIG["classifier"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["classifier"]["epochs"]), desc="Teacher"):
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
        if no_improve >= CONFIG["classifier"]["patience"]:
            print(f"Early stopping teacher at epoch {ep+1}")
            break
    if best_state is not None:
        teacher.load_state_dict(best_state)
    auc_teacher, preds_teacher, labs = eval_classifier(teacher, test_off_loader, device)

    # Baseline (HLT smeared)
    print("\n" + "=" * 70)
    print("Baseline (HLT smeared)")
    print("=" * 70)
    baseline = ParticleTransformer(
        input_dim=7,
        embed_dim=cls_cfg["embed_dim"],
        num_heads=cls_cfg["num_heads"],
        num_layers=cls_cfg["num_layers"],
        ff_dim=cls_cfg["ff_dim"],
        dropout=cls_cfg["dropout"],
    ).to(device)
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=CONFIG["classifier"]["lr"], weight_decay=CONFIG["classifier"]["weight_decay"])
    sch_b = get_scheduler(opt_b, CONFIG["classifier"]["warmup_epochs"], CONFIG["classifier"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["classifier"]["epochs"]), desc="Baseline"):
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
        if no_improve >= CONFIG["classifier"]["patience"]:
            print(f"Early stopping baseline at epoch {ep+1}")
            break
    if best_state is not None:
        baseline.load_state_dict(best_state)
    auc_baseline, preds_baseline, _ = eval_classifier(baseline, test_hlt_loader, device)

    # Student (unsmeared)
    print("\n" + "=" * 70)
    print("Student (Unsmeared)")
    print("=" * 70)
    student = ParticleTransformer(
        input_dim=7,
        embed_dim=cls_cfg["embed_dim"],
        num_heads=cls_cfg["num_heads"],
        num_layers=cls_cfg["num_layers"],
        ff_dim=cls_cfg["ff_dim"],
        dropout=cls_cfg["dropout"],
    ).to(device)
    opt_s = torch.optim.AdamW(student.parameters(), lr=CONFIG["classifier"]["lr"], weight_decay=CONFIG["classifier"]["weight_decay"])
    sch_s = get_scheduler(opt_s, CONFIG["classifier"]["warmup_epochs"], CONFIG["classifier"]["epochs"])
    best_auc, best_state, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["classifier"]["epochs"]), desc="Unsmeared"):
        _, train_auc = train_classifier(student, train_uns_loader, opt_s, device)
        val_auc, _, _ = eval_classifier(student, val_uns_loader, device)
        sch_s.step()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")
        if no_improve >= CONFIG["classifier"]["patience"]:
            print(f"Early stopping unsmeared at epoch {ep+1}")
            break
    if best_state is not None:
        student.load_state_dict(best_state)
    auc_uns, preds_uns, _ = eval_classifier(student, test_uns_loader, device)

    print("\nTest AUCs:")
    print(f"  Teacher (offline):  {auc_teacher:.4f}")
    print(f"  Baseline (HLT):     {auc_baseline:.4f}")
    print(f"  Unsmeared student:  {auc_uns:.4f}")

    np.savez(
        save_root / "classifier_results.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_unsmear=auc_uns,
        preds_teacher=preds_teacher,
        preds_baseline=preds_baseline,
        preds_unsmear=preds_uns,
        labs=labs,
    )


if __name__ == "__main__":
    main()

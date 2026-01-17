#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer Teacher-Student KD with Realistic HLT Effects
Adapted to your dataset loading (utils.load_from_files over a directory of .h5 files)
and your jet counts.

Trains 3 models:
  1) Teacher on OFFLINE (high-quality) view
  2) Baseline on HLT (low-quality) view, no KD
  3) Student on HLT with KD from teacher (now with stronger KD strategies)

Saves:
  - test_split/test_features_and_masks.npz   (offline + hlt standardized features, masks, labels, indices)
  - checkpoints/transformer_kd/<run_name>/{teacher,baseline,student}.pt  (best checkpoints)
  - checkpoints/transformer_kd/<run_name>/results.npz, results.png       (preds + ROC plot)

Assumption about utils.load_from_files output:
  all_data: (N, max_constits, 3) with columns [eta, phi, pt]
If your columns differ, edit ETA_IDX/PHI_IDX/PT_IDX below.

KD upgrades included (picked for “still helps at large data”):
  - Confidence-weighted logit KD (KD emphasized where teacher is informative)
  - Representation alignment on pooled embedding z (cosine loss)
  - Paired contrastive alignment (InfoNCE on z) using batch negatives
  - Attention distillation via masked KL on normalized attention (more principled than entropy MSE)
  - Optional alpha/T scheduling already supported via CLI args
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


# ----------------------------- HLT config (matches professor) ----------------------------- #
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

    # Defaults are intentionally conservative; tune via CLI if you want.
    "kd": {
        "temperature": 7.0,
        "alpha_kd": 0.5,          # mixes hard vs KD
        "alpha_attn": 0.05,       # masked KL on attention
        "alpha_rep": 0.10,        # cosine alignment on pooled embedding z
        "alpha_nce": 0.10,        # InfoNCE on z (paired off<->hlt)
        "alpha_rel": 0.0,         # relational KD on batch pairwise similarities
        "tau_nce": 0.10,          # InfoNCE temperature
        "use_conf_weighted_kd": True,
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


# ----------------------------- Model (same as professor, with embedding return) ----------------------------- #
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


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.state_dict().items():
            self.shadow[name] = param.detach().clone()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if not torch.is_floating_point(param):
                    self.shadow[name] = param.detach().clone()
                    continue
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


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

        z = self.norm(pooled.squeeze(1))   # (B, D)
        logits = self.classifier(z)        # (B, 1)

        if return_attention and return_embedding:
            return logits, attn_weights.squeeze(1), z
        if return_attention:
            return logits, attn_weights.squeeze(1)
        if return_embedding:
            return logits, z
        return logits


class TeacherEnsemble(nn.Module):
    def __init__(self, teachers, temps=None):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        if temps is None:
            temps = [1.0] * len(teachers)
        self.temps = temps

    def forward(self, x, mask, return_attention=False, return_embedding=False):
        logits_list, attn_list, emb_list = [], [], []
        for idx, t in enumerate(self.teachers):
            out = t(x, mask, return_attention=return_attention, return_embedding=return_embedding)
            if return_attention and return_embedding:
                logits, attn, emb = out
            elif return_attention:
                logits, attn = out
                emb = None
            elif return_embedding:
                logits, emb = out
                attn = None
            else:
                logits = out
                attn = None
                emb = None

            temp = float(self.temps[idx]) if idx < len(self.temps) else 1.0
            if temp != 1.0:
                logits = logits / temp
            logits_list.append(logits)
            if attn is not None:
                attn_list.append(attn)
            if emb is not None:
                emb_list.append(emb)

        mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        if return_attention and return_embedding:
            mean_attn = torch.stack(attn_list, dim=0).mean(dim=0)
            mean_emb = torch.stack(emb_list, dim=0).mean(dim=0)
            return mean_logits, mean_attn, mean_emb
        if return_attention:
            mean_attn = torch.stack(attn_list, dim=0).mean(dim=0)
            return mean_logits, mean_attn
        if return_embedding:
            mean_emb = torch.stack(emb_list, dim=0).mean(dim=0)
            return mean_logits, mean_emb
        return mean_logits


# ----------------------------- KD Losses (upgraded) ----------------------------- #
def kd_loss_basic(student_logits, teacher_logits, T):
    s_soft = torch.sigmoid(student_logits / T)
    t_soft = torch.sigmoid(teacher_logits / T)
    return F.binary_cross_entropy(s_soft, t_soft) * (T ** 2)


def kd_loss_conf_weighted(student_logits, teacher_logits, T, eps=1e-8):
    """
    Confidence-weighted binary KD.
    Weight is high when teacher is far from 0.5 (more informative “shape”), low near 0.5.
    """
    s_soft = torch.sigmoid(student_logits / T)
    t_soft = torch.sigmoid(teacher_logits / T)

    # w in [0,1]
    w = (torch.abs(torch.sigmoid(teacher_logits) - 0.5) * 2.0).detach()

    per = F.binary_cross_entropy(s_soft, t_soft, reduction="none")  # (B,)
    loss = (w * per).mean() * (T ** 2)
    return loss


def rep_loss_cosine(s_z, t_z):
    """
    1 - cosine similarity, averaged over batch.
    """
    s = F.normalize(s_z, dim=1)
    t = F.normalize(t_z, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


def info_nce_loss(s_z, t_z, tau=0.1):
    """
    Paired contrastive alignment:
      positives: (s_i, t_i)
      negatives: (s_i, t_j) for j != i
    Symmetric (s->t and t->s).
    """
    s = F.normalize(s_z, dim=1)
    t = F.normalize(t_z, dim=1)

    logits_st = (s @ t.t()) / tau
    logits_ts = (t @ s.t()) / tau

    labels = torch.arange(s.size(0), device=s.device)
    loss_st = F.cross_entropy(logits_st, labels)
    loss_ts = F.cross_entropy(logits_ts, labels)
    return 0.5 * (loss_st + loss_ts)


def relational_kd_loss(s_z, t_z):
    """
    Match pairwise similarities between student and teacher embeddings within a batch.
    """
    s = F.normalize(s_z, dim=1)
    t = F.normalize(t_z, dim=1)
    sim_s = s @ s.t()
    sim_t = t @ t.t()
    mask = ~torch.eye(sim_s.size(0), dtype=torch.bool, device=sim_s.device)
    diff = (sim_s - sim_t)[mask]
    return (diff ** 2).mean()


def attn_kl_loss(s_attn, t_attn, s_mask, t_mask, eps=1e-8):
    """
    Masked KL on normalized attention distributions.
    We compare only on joint-valid tokens so distributions live on the same support.
    """
    joint = (s_mask & t_mask).float()  # (B, L)

    # If a sample has no joint-valid tokens, it contributes 0.
    denom_s = (s_attn * joint).sum(dim=1, keepdim=True)
    denom_t = (t_attn * joint).sum(dim=1, keepdim=True)

    valid_sample = (denom_s.squeeze(1) > eps) & (denom_t.squeeze(1) > eps)
    if valid_sample.sum().item() == 0:
        return torch.zeros((), device=s_attn.device)

    s = (s_attn * joint) / (denom_s + eps)
    t = (t_attn * joint) / (denom_t + eps)

    s = torch.clamp(s, eps, 1.0)
    t = torch.clamp(t, eps, 1.0)

    kl = (t * (torch.log(t) - torch.log(s))).sum(dim=1)  # (B,)
    return kl[valid_sample].mean()


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


def train_standard_ema(model, loader, opt, device, feat_key, mask_key, ema=None):
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
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * len(y)
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / len(preds), roc_auc_score(labs, preds)


def train_kd(student, teacher, loader, opt, device, cfg, temperature=None, alpha_kd=None,
             teacher_temp=1.0, enable_kd=True):
    student.train()
    teacher.eval()

    # Use provided values or fall back to config
    T = temperature if temperature is not None else cfg["kd"]["temperature"]
    a_kd = alpha_kd if alpha_kd is not None else cfg["kd"]["alpha_kd"]

    a_attn = cfg["kd"].get("alpha_attn", 0.0)
    a_rep  = cfg["kd"].get("alpha_rep", 0.0)
    a_nce  = cfg["kd"].get("alpha_nce", 0.0)
    a_rel  = cfg["kd"].get("alpha_rel", 0.0)
    tau_nce = cfg["kd"].get("tau_nce", 0.1)
    use_conf = cfg["kd"].get("use_conf_weighted_kd", True)

    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        x_off = batch["off"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)

        if enable_kd:
            with torch.no_grad():
                t_logits, t_attn, t_z = teacher(x_off, m_off, return_attention=True, return_embedding=True)
                t_logits = t_logits.squeeze(1)
                if teacher_temp != 1.0:
                    t_logits = t_logits / teacher_temp

        opt.zero_grad()
        s_logits, s_attn, s_z = student(x_hlt, m_hlt, return_attention=True, return_embedding=True)
        s_logits = s_logits.squeeze(1)

        # Hard label loss (HLT)
        loss_hard = F.binary_cross_entropy_with_logits(s_logits, y)

        # Logit KD (weighted)
        if enable_kd:
            if use_conf:
                loss_kd = kd_loss_conf_weighted(s_logits, t_logits, T)
            else:
                loss_kd = kd_loss_basic(s_logits, t_logits, T)

            # Embedding alignment + contrastive
            loss_rep = rep_loss_cosine(s_z, t_z.detach()) if a_rep > 0 else torch.zeros((), device=device)
            loss_nce = info_nce_loss(s_z, t_z.detach(), tau=tau_nce) if a_nce > 0 else torch.zeros((), device=device)
            loss_rel = relational_kd_loss(s_z, t_z.detach()) if a_rel > 0 else torch.zeros((), device=device)

            # Attention distillation (masked KL on joint support)
            loss_attn = attn_kl_loss(s_attn, t_attn.detach(), m_hlt, m_off) if a_attn > 0 else torch.zeros((), device=device)
        else:
            loss_kd = torch.zeros((), device=device)
            loss_rep = torch.zeros((), device=device)
            loss_nce = torch.zeros((), device=device)
            loss_rel = torch.zeros((), device=device)
            loss_attn = torch.zeros((), device=device)

        # Combine
        loss = (1.0 - a_kd) * loss_hard + a_kd * loss_kd + a_attn * loss_attn + a_rep * loss_rep + a_nce * loss_nce + a_rel * loss_rel
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / len(preds), roc_auc_score(labs, preds)


@torch.no_grad()
def evaluate(model, loader, device, feat_key, mask_key):
    model.eval()
    preds, labs = [], []
    if not hasattr(evaluate, "_warned"):
        evaluate._warned = False
    for batch in loader:
        x = batch[feat_key].to(device)
        mask = batch[mask_key].to(device)
        logits = model(x, mask).squeeze(1)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            if not evaluate._warned:
                print("Warning: NaN/Inf in logits during evaluation; replacing with 0.0.")
                evaluate._warned = True
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        labs.extend(batch["label"].cpu().numpy().flatten())
    preds = np.array(preds)
    labs = np.array(labs)
    return roc_auc_score(labs, preds), preds, labs


@torch.no_grad()
def evaluate_hlt_loss(model, loader, device, feat_key, mask_key):
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        x = batch[feat_key].to(device)
        mask = batch[mask_key].to(device)
        y = batch["label"].to(device)
        logits = model(x, mask).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        total_loss += loss.item()
        count += len(y)
    if count == 0:
        return 0.0
    return total_loss / count


def train_self_train_epoch(student, source_model, loader, opt, device,
                           source_feat_key, source_mask_key, teacher_temp=1.0,
                           conf_min=0.0, conf_power=1.0, hard_labels=False):
    student.train()
    source_model.eval()
    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x_src = batch[source_feat_key].to(device)
        m_src = batch[source_mask_key].to(device)
        x_hlt = batch["hlt"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            src_logits = source_model(x_src, m_src).squeeze(1)
            if teacher_temp != 1.0:
                src_logits = src_logits / teacher_temp
            p = torch.sigmoid(src_logits)
            if hard_labels:
                y_pseudo = (p >= 0.5).float()
            else:
                y_pseudo = p
            conf = torch.abs(p - 0.5) * 2.0
            w = conf.clamp(0.0, 1.0) ** conf_power
            if conf_min > 0.0:
                w = w * (conf >= conf_min).float()

        opt.zero_grad()
        s_logits = student(x_hlt, m_hlt).squeeze(1)
        loss_vec = F.binary_cross_entropy_with_logits(s_logits, y_pseudo, reduction="none")
        denom = w.sum().clamp(min=1.0)
        loss = (loss_vec * w).sum() / denom
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / max(1, len(labs)), roc_auc_score(labs, preds)


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


@torch.no_grad()
def collect_logits(model, loader, device, feat_key, mask_key):
    model.eval()
    logits_list = []
    labels_list = []
    for batch in loader:
        x = batch[feat_key].to(device)
        mask = batch[mask_key].to(device)
        logits = model(x, mask).squeeze(1)
        logits_list.append(logits.detach())
        labels_list.append(batch["label"].to(device).float().detach())
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def calibrate_temperature(logits, labels, max_iter=50):
    # Use log-temperature to keep it positive.
    log_temp = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_temp)
        loss = F.binary_cross_entropy_with_logits(logits / temp, labels)
        loss.backward()
        return loss

    opt.step(closure)
    temp = torch.exp(log_temp).detach().clamp(min=1e-3).item()
    return temp


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default="./data",
        help="Directory containing your *.h5 files (default: ./data relative to project root)",
    )
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "transformer_kd"))
    parser.add_argument("--device", type=str, default="cpu")

    # KD schedule args (already in your file)
    parser.add_argument("--temp_init", type=float, default=7.0, help="Initial temperature for KD")
    parser.add_argument("--temp_final", type=float, default=None, help="Final temperature (if annealing)")
    parser.add_argument("--alpha_init", type=float, default=0.5, help="Initial alpha_kd weight")
    parser.add_argument("--alpha_final", type=float, default=None, help="Final alpha_kd weight (if scheduling)")
    parser.add_argument("--run_name", type=str, default="default", help="Unique name for this hyperparameter run")
    parser.add_argument("--adaptive_alpha", action="store_true", help="Enable adaptive KD ramp after HLT loss stabilizes")
    parser.add_argument("--alpha_warmup", type=float, default=0.0, help="KD weight before stabilization (adaptive only)")
    parser.add_argument("--alpha_stable_patience", type=int, default=3, help="Epochs with no HLT-loss improvement before KD ramp")
    parser.add_argument("--alpha_stable_delta", type=float, default=1e-4, help="Min HLT-loss improvement to reset patience")
    parser.add_argument("--alpha_warmup_min_epochs", type=int, default=3, help="Minimum epochs before KD ramp can start")
    parser.add_argument("--self_train", action="store_true", help="Enable pseudo-label fine-tuning after KD")
    parser.add_argument("--self_train_source", type=str, default="teacher", choices=["teacher", "student"], help="Pseudo-label source model")
    parser.add_argument("--self_train_epochs", type=int, default=5, help="Self-training epochs")
    parser.add_argument("--self_train_lr", type=float, default=1e-4, help="Self-training learning rate")
    parser.add_argument("--self_train_conf_min", type=float, default=0.0, help="Min confidence to use pseudo-label")
    parser.add_argument("--self_train_conf_power", type=float, default=1.0, help="Confidence weight power for pseudo-label loss")
    parser.add_argument("--self_train_hard", action="store_true", help="Use hard pseudo-labels instead of soft probs")
    parser.add_argument("--self_train_patience", type=int, default=5, help="Early stop patience for self-training")

    # New KD knobs (optional)
    parser.add_argument("--alpha_attn", type=float, default=CONFIG["kd"]["alpha_attn"], help="Weight for attention KL distillation")
    parser.add_argument("--alpha_rep", type=float, default=CONFIG["kd"]["alpha_rep"], help="Weight for embedding cosine alignment")
    parser.add_argument("--alpha_nce", type=float, default=CONFIG["kd"]["alpha_nce"], help="Weight for InfoNCE paired contrastive loss")
    parser.add_argument("--alpha_rel", type=float, default=CONFIG["kd"]["alpha_rel"], help="Weight for relational KD similarity loss")
    parser.add_argument("--tau_nce", type=float, default=CONFIG["kd"]["tau_nce"], help="InfoNCE temperature (smaller = harder)")
    parser.add_argument("--no_conf_kd", action="store_true", help="Disable confidence-weighted KD (use plain KD)")
    parser.add_argument("--calibrate_teacher", action="store_true", help="Calibrate teacher logits via temperature scaling on val split")
    parser.add_argument("--teacher_calib_max_iter", type=int, default=50, help="LBFGS iterations for teacher calibration")
    parser.add_argument("--teacher_ensemble_checkpoints", type=str, default=None, help="Comma-separated teacher checkpoints for ensemble KD")
    parser.add_argument("--ema_teacher", action="store_true", help="Use EMA-smoothed teacher weights for KD")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay for teacher weights")

    # Pre-trained model loading
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Path to pre-trained teacher model (skips teacher training)")
    parser.add_argument("--baseline_checkpoint", type=str, default=None, help="Path to pre-trained baseline model (skips baseline training)")
    parser.add_argument("--skip_save_models", action="store_true", help="Skip saving model weights (save space during hyperparameter search)")

    args = parser.parse_args()

    # Apply KD knobs into CONFIG
    CONFIG["kd"]["alpha_attn"] = float(args.alpha_attn)
    CONFIG["kd"]["alpha_rep"]  = float(args.alpha_rep)
    CONFIG["kd"]["alpha_nce"]  = float(args.alpha_nce)
    CONFIG["kd"]["alpha_rel"]  = float(args.alpha_rel)
    CONFIG["kd"]["tau_nce"]    = float(args.tau_nce)
    CONFIG["kd"]["use_conf_weighted_kd"] = (not args.no_conf_kd)

    # Create unique save directory for this hyperparameter run
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")
    print("KD settings:")
    print(f"  conf_weighted_kd={CONFIG['kd']['use_conf_weighted_kd']}, alpha_attn={CONFIG['kd']['alpha_attn']}, alpha_rep={CONFIG['kd']['alpha_rep']}, alpha_nce={CONFIG['kd']['alpha_nce']}, alpha_rel={CONFIG['kd']['alpha_rel']}, tau_nce={CONFIG['kd']['tau_nce']}")

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

    # ------------------- Split indices (70/15/15, stratified) ------------------- #
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
    student_path  = save_dir / "student.pt"

    # ------------------- STEP 1: Teacher (offline) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 1: TEACHER (Offline / high-quality view)")
    print("=" * 70)

    teacher = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)

    teacher_temp = 1.0
    teacher_ensemble = []
    if args.teacher_ensemble_checkpoints:
        teacher_ensemble = [p.strip() for p in args.teacher_ensemble_checkpoints.split(",") if p.strip()]

    if teacher_ensemble:
        print("Loading teacher ensemble:")
        teachers = []
        temps = []
        for path in teacher_ensemble:
            print(f"  - {path}")
            t = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
            ckpt = torch.load(path, map_location=device)
            t.load_state_dict(ckpt["model"])
            teachers.append(t)
            temps.append(float(ckpt.get("temp", 1.0)))
        teacher = TeacherEnsemble(teachers, temps=temps).to(device)
        best_auc_teacher = None
        history_teacher = []
    elif args.teacher_checkpoint is not None:
        print(f"Loading pre-trained teacher from: {args.teacher_checkpoint}")
        ckpt = torch.load(args.teacher_checkpoint, map_location=device)
        if args.ema_teacher and "ema" in ckpt:
            teacher.load_state_dict(ckpt["ema"])
        else:
            teacher.load_state_dict(ckpt["model"])
        best_auc_teacher = ckpt["auc"]
        history_teacher = ckpt.get("history", [])
        teacher_temp = float(ckpt.get("temp", 1.0))
        print(f"Loaded teacher with AUC={best_auc_teacher:.4f}")
    else:
        opt = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
        sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

        best_auc_teacher, best_state, no_improve = 0.0, None, 0
        best_ema_state = None
        history_teacher = []
        ema = EMA(teacher, decay=args.ema_decay) if args.ema_teacher else None

        for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Teacher"):
            if ema is None:
                train_loss, train_auc = train_standard(teacher, train_loader, opt, device, "off", "mask_off")
            else:
                train_loss, train_auc = train_standard_ema(teacher, train_loader, opt, device, "off", "mask_off", ema=ema)
            val_auc, _, _ = evaluate(teacher, val_loader, device, "off", "mask_off")
            sch.step()

            history_teacher.append((ep + 1, train_loss, train_auc, val_auc))

            if val_auc > best_auc_teacher:
                best_auc_teacher = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
                if ema is not None:
                    best_ema_state = ema.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_teacher:.4f}")

            if no_improve >= CONFIG["training"]["patience"]:
                print(f"Early stopping teacher at epoch {ep+1}")
                break

        teacher.load_state_dict(best_state)
        if args.ema_teacher and best_ema_state is not None:
            teacher.load_state_dict(best_ema_state)
        if not args.skip_save_models:
            torch.save(
                {
                    "model": best_state,
                    "ema": best_ema_state,
                    "auc": best_auc_teacher,
                    "history": history_teacher,
                    "temp": teacher_temp,
                    "ema_decay": args.ema_decay if args.ema_teacher else None,
                },
                teacher_path,
            )
            print(f"Saved teacher: {teacher_path} (best val AUC={best_auc_teacher:.4f})")
        else:
            print(f"Skipped saving teacher model (best val AUC={best_auc_teacher:.4f})")

    if args.calibrate_teacher:
        print("\nCalibrating teacher (temperature scaling on OFFLINE val)...")
        val_logits, val_labels = collect_logits(teacher, val_loader, device, "off", "mask_off")
        teacher_temp = calibrate_temperature(val_logits, val_labels, max_iter=args.teacher_calib_max_iter)
        print(f"Calibrated teacher temperature: {teacher_temp:.4f}")
        if args.teacher_checkpoint is None and not args.skip_save_models and not teacher_ensemble:
            torch.save(
                {"model": teacher.state_dict(), "auc": best_auc_teacher, "history": history_teacher, "temp": teacher_temp},
                teacher_path,
            )

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
        print(f"Loading pre-trained baseline from: {args.baseline_checkpoint}")
        ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        baseline.load_state_dict(ckpt["model"])
        best_auc_baseline = ckpt["auc"]
        history_baseline = ckpt.get("history", [])
        print(f"Loaded baseline with AUC={best_auc_baseline:.4f}")
    else:
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

    # ------------------- STEP 3: Student KD ------------------- #
    print("\n" + "=" * 70)
    print("STEP 3: STUDENT with KD (HLT view + teacher guidance)")
    print("=" * 70)

    student = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch = get_scheduler(opt, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

    best_auc_student, best_state, no_improve = 0.0, None, 0
    history_student = []
    kd_active = not args.adaptive_alpha
    kd_start_epoch = 0
    best_hlt_loss = None
    stable_count = 0

    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Student KD"):
        current_temp = get_temperature_schedule(ep, CONFIG["training"]["epochs"], args.temp_init, args.temp_final)
        if args.adaptive_alpha and not kd_active:
            current_alpha = args.alpha_warmup
            enable_kd = False
        else:
            enable_kd = True
            if args.adaptive_alpha:
                alpha_ep = max(0, ep - kd_start_epoch)
                total_alpha_epochs = max(1, CONFIG["training"]["epochs"] - kd_start_epoch)
                current_alpha = get_alpha_schedule(alpha_ep, total_alpha_epochs, args.alpha_init, args.alpha_final)
            else:
                current_alpha = get_alpha_schedule(ep, CONFIG["training"]["epochs"], args.alpha_init, args.alpha_final)

        train_loss, train_auc = train_kd(
            student, teacher, train_loader, opt, device, CONFIG,
            temperature=current_temp, alpha_kd=current_alpha, teacher_temp=teacher_temp, enable_kd=enable_kd
        )
        val_auc, _, _ = evaluate(student, val_loader, device, "hlt", "mask_hlt")
        sch.step()

        history_student.append((ep + 1, train_loss, train_auc, val_auc))

        if args.adaptive_alpha and not kd_active:
            val_hlt_loss = evaluate_hlt_loss(student, val_loader, device, "hlt", "mask_hlt")
            if best_hlt_loss is None or (best_hlt_loss - val_hlt_loss) > args.alpha_stable_delta:
                best_hlt_loss = val_hlt_loss
                stable_count = 0
            else:
                stable_count += 1
            if (ep + 1) >= args.alpha_warmup_min_epochs and stable_count >= args.alpha_stable_patience:
                kd_active = True
                kd_start_epoch = ep + 1
                print(f"Activating KD ramp at epoch {kd_start_epoch} (val_hlt_loss={val_hlt_loss:.4f})")

        if val_auc > best_auc_student:
            best_auc_student = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: loss={train_loss:.4f}, val_auc={val_auc:.4f}, best={best_auc_student:.4f} | T={current_temp:.2f}, alpha_kd={current_alpha:.2f}")

        if no_improve >= CONFIG["training"]["patience"] + 5:
            print(f"Early stopping student at epoch {ep+1}")
            break

    student.load_state_dict(best_state)
    if args.self_train:
        print("\n" + "=" * 70)
        print("STEP 3B: SELF-TRAIN (pseudo-label fine-tune)")
        print("=" * 70)
        source_model = teacher if args.self_train_source == "teacher" else student
        source_feat_key = "off" if args.self_train_source == "teacher" else "hlt"
        source_mask_key = "mask_off" if args.self_train_source == "teacher" else "mask_hlt"
        opt_st = torch.optim.AdamW(student.parameters(), lr=args.self_train_lr, weight_decay=CONFIG["training"]["weight_decay"])
        best_st_auc, best_st_state, no_improve = 0.0, None, 0
        for ep in tqdm(range(args.self_train_epochs), desc="Self-Train"):
            train_loss, train_auc = train_self_train_epoch(
                student, source_model, train_loader, opt_st, device,
                source_feat_key, source_mask_key, teacher_temp=teacher_temp,
                conf_min=args.self_train_conf_min, conf_power=args.self_train_conf_power,
                hard_labels=args.self_train_hard,
            )
            val_auc, _, _ = evaluate(student, val_loader, device, "hlt", "mask_hlt")
            history_student.append(("self_train", ep + 1, train_loss, train_auc, val_auc))

            if val_auc > best_st_auc:
                best_st_auc = val_auc
                best_st_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 2 == 0:
                print(f"Self ep {ep+1}: loss={train_loss:.4f}, val_auc={val_auc:.4f}, best={best_st_auc:.4f}")

            if no_improve >= args.self_train_patience:
                print(f"Early stopping self-train at epoch {ep+1}")
                break

        if best_st_state is not None:
            student.load_state_dict(best_st_state)
            best_auc_student = max(best_auc_student, best_st_auc)

    if not args.skip_save_models:
        torch.save({"model": student.state_dict(), "auc": best_auc_student, "history": history_student}, student_path)
        print(f"Saved student: {student_path} (best val AUC={best_auc_student:.4f})")
    else:
        print(f"Skipped saving student model (best val AUC={best_auc_student:.4f})")

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
    print(f"{'Student with KD':<40} {auc_student:>10.4f}")
    print("-" * 52)

    degradation = auc_teacher - auc_baseline
    improvement = auc_student - auc_baseline
    recovery = 100 * improvement / degradation if degradation > 0 else 0.0

    print("\nAnalysis:")
    print(f"  HLT Degradation: {degradation:.4f}")
    print(f"  KD Improvement:  {improvement:+.4f}")
    print(f"  Recovery:        {recovery:.1f}%")

    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_s, tpr_s, _ = roc_curve(labs, preds_student)

    # Background Rejection at 50% signal efficiency
    wp = 0.5
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
        kd_cfg=np.array([CONFIG["kd"]["alpha_attn"], CONFIG["kd"]["alpha_rep"], CONFIG["kd"]["alpha_nce"], CONFIG["kd"]["tau_nce"]], dtype=np.float32),
    )

    # Plot ROC curves (TPR on x, FPR on y)
    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-",  label=f"Teacher (AUC={auc_teacher:.3f})",  color="crimson",     linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline (AUC={auc_baseline:.3f})", color="steelblue",  linewidth=2)
    plt.plot(tpr_s, fpr_s, ":",  label=f"Student KD (AUC={auc_student:.3f})", color="forestgreen", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300)
    plt.close()

    # Log hyperparameter run results
    summary_file = Path(args.save_dir) / "hyperparameter_search_results.txt"
    with open(summary_file, "a") as f:
        f.write(f"\nRun: {args.run_name}\n")
        f.write(f"  Temperature: {args.temp_init:.2f}")
        if args.temp_final is not None:
            f.write(f" -> {args.temp_final:.2f} (annealing)\n")
        else:
            f.write(" (constant)\n")
        f.write(f"  Alpha_KD: {args.alpha_init:.2f}")
        if args.alpha_final is not None:
            f.write(f" -> {args.alpha_final:.2f} (scheduling)\n")
        else:
            f.write(" (constant)\n")
        f.write(f"  alpha_attn={CONFIG['kd']['alpha_attn']:.3f}, alpha_rep={CONFIG['kd']['alpha_rep']:.3f}, alpha_nce={CONFIG['kd']['alpha_nce']:.3f}, tau_nce={CONFIG['kd']['tau_nce']:.3f}, conf_kd={CONFIG['kd']['use_conf_weighted_kd']}\n")
        f.write(f"  Background Rejection @ 50% efficiency: {br_student:.2f}\n")
        f.write(f"  AUC (Student): {auc_student:.4f}\n")
        f.write(f"  Saved to: {save_dir}\n")
        f.write("=" * 70 + "\n")

    print(f"\nSaved results to: {save_dir / 'results.npz'} and {save_dir / 'results.png'}")
    print(f"Logged to: {summary_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full unmerge pipeline:
  1) Build HLT view (no smearing, merge + efficiency loss).
  2) Train merge-count predictor (HLT -> count).
  3) Train unmerger (HLT token + predicted count -> offline constituents).
  4) Build unmerged dataset by replacing merged tokens with predicted constituents.
  5) Train teacher (offline), baseline (HLT), and unmerge-model (unmerged view).
  6) Evaluate all on test.
"""

from pathlib import Path
import argparse
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
RANDOM_SEED = 48
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
        "pt_resolution": 0.0,
        "eta_resolution": 0.0,
        "phi_resolution": 0.0,
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
    "merge_count_model": {
        "embed_dim": 128,
        "num_heads": 8,
        "num_layers": 6,
        "ff_dim": 512,
        "dropout": 0.1,
    },
    "unmerge_model": {
        "embed_dim": 192,
        "num_heads": 8,
        "num_layers": 6,
        "decoder_layers": 3,
        "ff_dim": 768,
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
    },
    "kd": {
        "temperature": 7.0,
        "alpha_kd": 0.5,
        "alpha_attn": 0.05,
        "alpha_rep": 0.10,
        "alpha_nce": 0.10,
        "tau_nce": 0.10,
        "conf_weighted": True,
        "adaptive_alpha": True,
        "alpha_warmup": 0.0,
        "alpha_stable_patience": 2,
        "alpha_stable_delta": 1e-4,
        "alpha_warmup_min_epochs": 3,
        "ema_teacher": True,
        "ema_decay": 0.995,
        "self_train": True,
        "self_train_source": "teacher",
        "self_train_epochs": 5,
        "self_train_lr": 1e-4,
        "self_train_conf_min": 0.0,
        "self_train_conf_power": 1.0,
        "self_train_patience": 5,
    },
}


def safe_sigmoid(logits):
    probs = torch.sigmoid(logits)
    return torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.detach()

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


def kd_loss_conf_weighted(student_logits, teacher_logits, T):
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


@torch.no_grad()
def evaluate_bce_loss(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch["feat"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)
        logits = model(x, mask).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total += loss.item() * len(y)
        count += len(y)
    return total / max(count, 1)


@torch.no_grad()
def evaluate_bce_loss_unmerged(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch["unmerged"].to(device)
        mask = batch["mask_unmerged"].to(device)
        y = batch["label"].to(device)
        logits = model(x, mask).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total += loss.item() * len(y)
        count += len(y)
    return total / max(count, 1)

def apply_hlt_effects_with_tracking(const, mask, cfg, seed=42):
    """
    Returns HLT view with per-token origin tracking.
    """
    np.random.seed(seed)
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    origin_counts = hlt_mask.astype(np.int32)
    origin_lists = [[([idx] if hlt_mask[j, idx] else []) for idx in range(max_part)]
                    for j in range(n_jets)]

    n_initial = int(hlt_mask.sum())

    # Effect 1: Higher pT threshold
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
                        origin_counts[jet_idx, idx_i] += origin_counts[jet_idx, idx_j]
                        origin_lists[jet_idx][idx_i].extend(origin_lists[jet_idx][idx_j])

                        to_remove.add(idx_j)
                        n_merged += 1

            for idx in to_remove:
                hlt_mask[jet_idx, idx] = False
                hlt[jet_idx, idx] = 0
                origin_counts[jet_idx, idx] = 0
                origin_lists[jet_idx][idx] = []

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
        origin_counts[lost] = 0
        n_lost_eff = int(lost.sum())
        for j in range(n_jets):
            for idx in np.where(lost[j])[0]:
                origin_lists[j][idx] = []

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0

    n_final = int(hlt_mask.sum())
    stats = {
        "n_initial": n_initial,
        "n_lost_threshold": n_lost_threshold,
        "n_merged": n_merged,
        "n_lost_eff": n_lost_eff,
        "n_final": n_final,
    }
    return hlt, hlt_mask, origin_counts, origin_lists, stats


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
    def __init__(self, feat, mask, count_label):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.label = torch.tensor(count_label, dtype=torch.long)

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, i):
        return {"feat": self.feat[i], "mask": self.mask[i], "label": self.label[i]}


class UnmergeKDDataset(Dataset):
    def __init__(self, feat_unmerged, mask_unmerged, feat_off, mask_off, labels):
        self.unmerged = torch.tensor(feat_unmerged, dtype=torch.float32)
        self.mask_unmerged = torch.tensor(mask_unmerged, dtype=torch.bool)
        self.off = torch.tensor(feat_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "unmerged": self.unmerged[i],
            "mask_unmerged": self.mask_unmerged[i],
            "off": self.off[i],
            "mask_off": self.mask_off[i],
            "label": self.labels[i],
        }


class UnmergeDataset(Dataset):
    def __init__(self, feat_hlt, mask_hlt, constituents_off, samples, max_count, tgt_mean, tgt_std):
        self.feat_hlt = feat_hlt
        self.mask_hlt = mask_hlt
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
        }


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
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, mask, return_attention=False, return_embedding=False):
        batch_size, seq_len, _ = x.shape
        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, attn_weights = self.pool_attn(
            query, h, h, key_padding_mask=~mask, need_weights=True, average_attn_weights=True
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


class MergeCountPredictor(nn.Module):
    def __init__(self, input_dim=7, embed_dim=128, num_heads=8, num_layers=6, ff_dim=512, dropout=0.1, num_classes=6):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
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
        batch_size, seq_len, _ = x.shape
        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        logits = self.head(h)
        return logits


class UnmergePredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        max_count,
        embed_dim,
        num_heads,
        num_layers,
        decoder_layers,
        ff_dim,
        dropout,
        count_embed_dim,
    ):
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

        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 4),
        )

    def forward(self, x, mask, token_idx, count):
        batch_size, seq_len, _ = x.shape
        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)
        h = self.encoder(h, src_key_padding_mask=~mask)

        idx = token_idx.view(-1, 1, 1).expand(-1, 1, self.embed_dim)
        h_t = h.gather(1, idx).squeeze(1)
        h_sum = (h * mask.unsqueeze(-1)).sum(dim=1)
        h_avg = h_sum / mask.sum(dim=1, keepdim=True).clamp(min=1)

        c_emb = self.count_embed(count)
        cond = self.cond_proj(torch.cat([h_t, h_avg, c_emb], dim=1))

        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + cond.unsqueeze(1)

        dec = self.decoder(queries, h, memory_key_padding_mask=~mask)
        out = self.out(dec)
        return out


def compute_class_weights(labels, mask, num_classes):
    valid = labels[mask]
    counts = np.bincount(valid, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.ones(num_classes, dtype=np.float64)
    if total > 0:
        weights = total / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
    return weights


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


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_classifier(model, loader, opt, device, ema=None):
    model.train()
    total_loss = 0.0
    preds, labs = [], []
    for batch in loader:
        x = batch["feat"].to(device)
        mask = batch["mask"].to(device)
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
        preds.extend(safe_sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else 0.0
    return total_loss / len(preds), auc


@torch.no_grad()
def eval_classifier(model, loader, device):
    model.eval()
    preds, labs = [], []
    warned = False
    for batch in loader:
        x = batch["feat"].to(device)
        mask = batch["mask"].to(device)
        logits = model(x, mask).squeeze(1)
        if not warned and not torch.isfinite(logits).all():
            print("Warning: NaN/Inf in logits during evaluation; replacing with 0.5.")
            warned = True
        preds.extend(safe_sigmoid(logits).cpu().numpy().flatten())
        labs.extend(batch["label"].cpu().numpy().flatten())
    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else 0.0
    return auc, preds, labs


def train_kd_epoch(student, teacher, loader, opt, device, kd_cfg):
    student.train()
    teacher.eval()

    total_loss = 0.0
    preds, labs = [], []

    T = kd_cfg["temperature"]
    a_kd = kd_cfg["alpha_kd"]
    a_attn = kd_cfg["alpha_attn"]
    a_rep = kd_cfg["alpha_rep"]
    a_nce = kd_cfg["alpha_nce"]
    tau_nce = kd_cfg["tau_nce"]

    for batch in loader:
        x_u = batch["unmerged"].to(device)
        m_u = batch["mask_unmerged"].to(device)
        x_o = batch["off"].to(device)
        m_o = batch["mask_off"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            t_logits, t_attn, t_z = teacher(x_o, m_o, return_attention=True, return_embedding=True)
            t_logits = t_logits.squeeze(1)

        opt.zero_grad()
        s_logits, s_attn, s_z = student(x_u, m_u, return_attention=True, return_embedding=True)
        s_logits = s_logits.squeeze(1)

        loss_hard = F.binary_cross_entropy_with_logits(s_logits, y)
        if kd_cfg["conf_weighted"]:
            loss_kd = kd_loss_conf_weighted(s_logits, t_logits, T)
        else:
            s_soft = torch.sigmoid(s_logits / T)
            t_soft = torch.sigmoid(t_logits / T)
            loss_kd = F.binary_cross_entropy(s_soft, t_soft) * (T ** 2)

        loss_rep = rep_loss_cosine(s_z, t_z.detach()) if a_rep > 0 else torch.zeros((), device=device)
        loss_nce = info_nce_loss(s_z, t_z.detach(), tau=tau_nce) if a_nce > 0 else torch.zeros((), device=device)
        loss_attn = attn_kl_loss(s_attn, t_attn.detach(), m_u, m_o) if a_attn > 0 else torch.zeros((), device=device)

        loss = (1.0 - a_kd) * loss_hard + a_kd * loss_kd + a_rep * loss_rep + a_nce * loss_nce + a_attn * loss_attn
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.extend(safe_sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else 0.0
    return total_loss / len(preds), auc


@torch.no_grad()
def evaluate_kd(student, loader, device):
    student.eval()
    preds, labs = [], []
    for batch in loader:
        x_u = batch["unmerged"].to(device)
        m_u = batch["mask_unmerged"].to(device)
        y = batch["label"].to(device)
        logits = student(x_u, m_u).squeeze(1)
        preds.extend(safe_sigmoid(logits).cpu().numpy().flatten())
        labs.extend(y.cpu().numpy().flatten())
    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else 0.0
    return auc, preds, labs


def self_train_student(student, teacher, loader, opt, device, cfg):
    student.train()
    teacher.eval()
    conf_min = cfg["self_train_conf_min"]
    conf_power = cfg["self_train_conf_power"]
    total_loss = 0.0
    count = 0
    for batch in loader:
        x_u = batch["unmerged"].to(device)
        m_u = batch["mask_unmerged"].to(device)
        x_o = batch["off"].to(device)
        m_o = batch["mask_off"].to(device)

        with torch.no_grad():
            if cfg["self_train_source"] == "teacher":
                t_logits = teacher(x_o, m_o).squeeze(1)
                probs = torch.sigmoid(t_logits)
            else:
                s_logits = student(x_u, m_u).squeeze(1)
                probs = torch.sigmoid(s_logits)

            conf = torch.clamp(2 * torch.abs(probs - 0.5), 0.0, 1.0)
            conf = torch.clamp(conf, min=conf_min) ** conf_power

        opt.zero_grad()
        logits = student(x_u, m_u).squeeze(1)
        loss = (F.binary_cross_entropy_with_logits(logits, probs, reduction="none") * conf).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(probs)
        count += len(probs)
    return total_loss / max(count, 1)
def train_merge_count(model, loader, opt, device, class_weights):
    model.train()
    total_loss = 0.0
    preds, labs = [], []
    weight = torch.tensor(class_weights, device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch in loader:
        x = batch["feat"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)
        opt.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits[mask], y[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * mask.sum().item()
        pred_cls = logits[mask].argmax(dim=1)
        preds.extend(pred_cls.detach().cpu().numpy().flatten())
        labs.extend(y[mask].detach().cpu().numpy().flatten())
    acc = (np.array(preds) == np.array(labs)).mean() if len(labs) > 0 else 0.0
    return total_loss / max(len(labs), 1), acc


@torch.no_grad()
def eval_merge_count(model, loader, device):
    model.eval()
    preds, labs = [], []
    warned = False
    for batch in loader:
        x = batch["feat"].to(device)
        mask = batch["mask"].to(device)
        logits = model(x, mask)
        if not warned and not torch.isfinite(logits).all():
            print("Warning: NaN/Inf in logits during evaluation; replacing with 0.0.")
            warned = True
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        pred_cls = logits[mask].argmax(dim=1)
        preds.extend(pred_cls.cpu().numpy().flatten())
        labs.extend(batch["label"][mask.cpu()].numpy().flatten())
    preds = np.array(preds)
    labs = np.array(labs)
    acc = (preds == labs).mean() if labs.size > 0 else 0.0
    return acc, preds, labs


def predict_counts(model, feat, mask, batch_size, device, max_count):
    model.eval()
    preds = np.zeros(mask.shape, dtype=np.int64)
    loader = DataLoader(MergeCountDataset(feat, mask, np.zeros(mask.shape, dtype=np.int64)),
                        batch_size=batch_size, shuffle=False)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["feat"].to(device)
            m = batch["mask"].to(device)
            logits = model(x, m)
            pred_cls = logits.argmax(dim=2).cpu().numpy()
            batch_size_curr = pred_cls.shape[0]
            preds[idx:idx + batch_size_curr] = pred_cls + 1
            preds[idx:idx + batch_size_curr][~mask[idx:idx + batch_size_curr]] = 0
            idx += batch_size_curr
    preds = np.clip(preds, 0, max_count)
    return preds


def build_unmerged_dataset(
    feat_hlt_std,
    mask_hlt,
    hlt_const,
    pred_counts,
    unmerge_model,
    tgt_mean,
    tgt_std,
    max_count,
    max_constits,
    device,
    batch_size,
):
    n_jets, max_part, _ = hlt_const.shape
    pred_map = {}
    samples = []
    for j in range(n_jets):
        for idx in range(max_part):
            if mask_hlt[j, idx] and pred_counts[j, idx] > 1:
                samples.append((j, idx, int(pred_counts[j, idx])))

    if len(samples) > 0:
        unmerge_model.eval()
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                chunk = samples[i:i + batch_size]
                jet_idx = [c[0] for c in chunk]
                tok_idx = [c[1] for c in chunk]
                counts = [c[2] for c in chunk]
                x = torch.tensor(feat_hlt_std[jet_idx], dtype=torch.float32, device=device)
                m = torch.tensor(mask_hlt[jet_idx], dtype=torch.bool, device=device)
                token_idx = torch.tensor(tok_idx, dtype=torch.long, device=device)
                count = torch.tensor(counts, dtype=torch.long, device=device)
                preds = unmerge_model(x, m, token_idx, count).cpu().numpy()
                for k in range(len(chunk)):
                    c = counts[k]
                    pred = preds[k, :c]
                    pred = pred * tgt_std + tgt_mean
                    pred[:, 0] = np.clip(pred[:, 0], 0.0, None)
                    pred[:, 1] = np.clip(pred[:, 1], -5.0, 5.0)
                    pred[:, 2] = np.arctan2(np.sin(pred[:, 2]), np.cos(pred[:, 2]))
                    pred[:, 3] = np.clip(pred[:, 3], 0.0, None)
                    pred_map[(chunk[k][0], chunk[k][1])] = pred

    new_const = np.zeros((n_jets, max_constits, 4), dtype=np.float32)
    new_mask = np.zeros((n_jets, max_constits), dtype=bool)

    for j in range(n_jets):
        parts = []
        for idx in range(max_part):
            if not mask_hlt[j, idx]:
                continue
            if pred_counts[j, idx] <= 1:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--max_merge_count", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unmerge_model"))
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_save_models", action="store_true")
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

    print("Applying HLT effects...")
    hlt_const, hlt_mask, origin_counts, origin_lists, stats = apply_hlt_effects_with_tracking(
        constituents_raw, mask_raw, CONFIG, seed=RANDOM_SEED
    )

    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    const_off = constituents_raw.copy()
    const_off[~masks_off] = 0

    print("HLT Simulation Statistics:")
    print(f"  Offline particles: {stats['n_initial']:,}")
    print(f"  Lost to pT threshold ({CONFIG['hlt_effects']['pt_threshold_hlt']}): {stats['n_lost_threshold']:,}")
    print(f"  Lost to merging (dR<{CONFIG['hlt_effects']['merge_radius']}): {stats['n_merged']:,}")
    print(f"  Lost to efficiency: {stats['n_lost_eff']:,}")
    print(f"  HLT particles: {stats['n_final']:,}")
    print(f"  Avg per jet: Offline={masks_off.sum(axis=1).mean():.1f}, HLT={hlt_mask.sum(axis=1).mean():.1f}")

    print("Computing features...")
    features_off = compute_features(const_off, masks_off)
    features_hlt = compute_features(hlt_const, hlt_mask)

    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = standardize(features_hlt, hlt_mask, feat_means, feat_stds)

    max_count = max(int(args.max_merge_count), 2)
    count_label = np.clip(origin_counts, 1, max_count) - 1

    # ------------------- Train teacher (offline) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 1: TEACHER (Offline)")
    print("=" * 70)
    train_ds_off = JetDataset(features_off_std[train_idx], masks_off[train_idx], all_labels[train_idx])
    val_ds_off = JetDataset(features_off_std[val_idx], masks_off[val_idx], all_labels[val_idx])
    test_ds_off = JetDataset(features_off_std[test_idx], masks_off[test_idx], all_labels[test_idx])
    BS = CONFIG["training"]["batch_size"]
    train_loader_off = DataLoader(train_ds_off, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_off = DataLoader(val_ds_off, batch_size=BS, shuffle=False)
    test_loader_off = DataLoader(test_ds_off, batch_size=BS, shuffle=False)

    kd_cfg = CONFIG["kd"]
    teacher = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_t = get_scheduler(opt_t, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_t, best_state_t, no_improve = 0.0, None, 0
    ema = EMA(teacher, decay=kd_cfg["ema_decay"]) if kd_cfg["ema_teacher"] else None
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Teacher"):
        _, train_auc = train_classifier(teacher, train_loader_off, opt_t, device, ema=ema)
        val_auc, _, _ = eval_classifier(teacher, val_loader_off, device)
        sch_t.step()
        if val_auc > best_auc_t:
            best_auc_t = val_auc
            best_state_t = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_t:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping teacher at epoch {ep+1}")
            break
    if best_state_t is not None:
        teacher.load_state_dict(best_state_t)
    if ema is not None:
        ema.apply_to(teacher)

    auc_teacher, preds_teacher, labs = eval_classifier(teacher, test_loader_off, device)

    # ------------------- Train baseline (HLT) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE HLT")
    print("=" * 70)
    train_ds_hlt = JetDataset(features_hlt_std[train_idx], hlt_mask[train_idx], all_labels[train_idx])
    val_ds_hlt = JetDataset(features_hlt_std[val_idx], hlt_mask[val_idx], all_labels[val_idx])
    test_ds_hlt = JetDataset(features_hlt_std[test_idx], hlt_mask[test_idx], all_labels[test_idx])
    train_loader_hlt = DataLoader(train_ds_hlt, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_hlt = DataLoader(val_ds_hlt, batch_size=BS, shuffle=False)
    test_loader_hlt = DataLoader(test_ds_hlt, batch_size=BS, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_b = get_scheduler(opt_b, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_b, best_state_b, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Baseline"):
        _, train_auc = train_classifier(baseline, train_loader_hlt, opt_b, device)
        val_auc, _, _ = eval_classifier(baseline, val_loader_hlt, device)
        sch_b.step()
        if val_auc > best_auc_b:
            best_auc_b = val_auc
            best_state_b = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_b:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping baseline at epoch {ep+1}")
            break
    if best_state_b is not None:
        baseline.load_state_dict(best_state_b)

    auc_baseline, preds_baseline, _ = eval_classifier(baseline, test_loader_hlt, device)

    # ------------------- Train merge-count predictor ------------------- #
    print("\n" + "=" * 70)
    print("STEP 3: MERGE COUNT PREDICTOR")
    print("=" * 70)
    train_ds_cnt = MergeCountDataset(features_hlt_std[train_idx], hlt_mask[train_idx], count_label[train_idx])
    val_ds_cnt = MergeCountDataset(features_hlt_std[val_idx], hlt_mask[val_idx], count_label[val_idx])
    test_ds_cnt = MergeCountDataset(features_hlt_std[test_idx], hlt_mask[test_idx], count_label[test_idx])
    BS_cnt = CONFIG["merge_count_training"]["batch_size"]
    train_loader_cnt = DataLoader(train_ds_cnt, batch_size=BS_cnt, shuffle=True, drop_last=True)
    val_loader_cnt = DataLoader(val_ds_cnt, batch_size=BS_cnt, shuffle=False)

    count_model = MergeCountPredictor(input_dim=7, num_classes=max_count, **CONFIG["merge_count_model"]).to(device)
    opt_c = torch.optim.AdamW(count_model.parameters(), lr=CONFIG["merge_count_training"]["lr"], weight_decay=CONFIG["merge_count_training"]["weight_decay"])
    sch_c = get_scheduler(opt_c, CONFIG["merge_count_training"]["warmup_epochs"], CONFIG["merge_count_training"]["epochs"])
    class_weights = compute_class_weights(count_label[train_idx], hlt_mask[train_idx], max_count)
    best_acc, best_state_c, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["merge_count_training"]["epochs"]), desc="MergeCount"):
        _, train_acc = train_merge_count(count_model, train_loader_cnt, opt_c, device, class_weights)
        val_acc, _, _ = eval_merge_count(count_model, val_loader_cnt, device)
        sch_c.step()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_c = {k: v.detach().cpu().clone() for k, v in count_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, best={best_acc:.4f}")
        if no_improve >= CONFIG["merge_count_training"]["patience"]:
            print(f"Early stopping merge-count at epoch {ep+1}")
            break
    if best_state_c is not None:
        count_model.load_state_dict(best_state_c)

    # Predict counts for all jets
    pred_counts = predict_counts(count_model, features_hlt_std, hlt_mask, BS_cnt, device, max_count)

    # ------------------- Train unmerger ------------------- #
    print("\n" + "=" * 70)
    print("STEP 4: UNMERGER")
    print("=" * 70)
    samples = []
    print("Building merged-token sample list...")
    for j in tqdm(range(len(all_labels)), desc="CollectMerged"):
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

    print("Building unmerge targets (train split)...")
    train_targets = []
    for s in tqdm(train_samples, desc="UnmergeTargets"):
        train_targets.append(const_off[s[0], s[2], :4])
    flat_train = np.concatenate(train_targets, axis=0)
    tgt_mean = flat_train.mean(axis=0)
    tgt_std = flat_train.std(axis=0) + 1e-8

    BS_un = CONFIG["unmerge_training"]["batch_size"]
    train_ds_un = UnmergeDataset(features_hlt_std, hlt_mask, const_off, train_samples, max_count, tgt_mean, tgt_std)
    val_ds_un = UnmergeDataset(features_hlt_std, hlt_mask, const_off, val_samples, max_count, tgt_mean, tgt_std)
    test_ds_un = UnmergeDataset(features_hlt_std, hlt_mask, const_off, test_samples, max_count, tgt_mean, tgt_std)
    train_loader_un = DataLoader(train_ds_un, batch_size=BS_un, shuffle=True, drop_last=True)
    val_loader_un = DataLoader(val_ds_un, batch_size=BS_un, shuffle=False)
    test_loader_un = DataLoader(test_ds_un, batch_size=BS_un, shuffle=False)

    unmerge_model = UnmergePredictor(
        input_dim=7,
        max_count=max_count,
        **CONFIG["unmerge_model"],
    ).to(device)
    opt_u = torch.optim.AdamW(unmerge_model.parameters(), lr=CONFIG["unmerge_training"]["lr"], weight_decay=CONFIG["unmerge_training"]["weight_decay"])
    sch_u = get_scheduler(opt_u, CONFIG["unmerge_training"]["warmup_epochs"], CONFIG["unmerge_training"]["epochs"])
    best_val_loss, best_state_u, no_improve = 1e9, None, 0
    for ep in tqdm(range(CONFIG["unmerge_training"]["epochs"]), desc="Unmerge"):
        unmerge_model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader_un:
            x = batch["hlt"].to(device)
            mask = batch["mask"].to(device)
            token_idx = batch["token_idx"].to(device)
            pred_count = batch["pred_count"].to(device)
            true_count = batch["true_count"].to(device)
            target = batch["target"].to(device)
            opt_u.zero_grad()
            preds = unmerge_model(x, mask, token_idx, pred_count)
            loss = set_chamfer_loss(preds, target, true_count)
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
            for batch in val_loader_un:
                x = batch["hlt"].to(device)
                mask = batch["mask"].to(device)
                token_idx = batch["token_idx"].to(device)
                pred_count = batch["pred_count"].to(device)
                true_count = batch["true_count"].to(device)
                target = batch["target"].to(device)
                preds = unmerge_model(x, mask, token_idx, pred_count)
                loss = set_chamfer_loss(preds, target, true_count)
                val_loss += loss.item()
                n_batches += 1
        val_loss = val_loss / max(n_batches, 1)
        sch_u.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_u = {k: v.detach().cpu().clone() for k, v in unmerge_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best={best_val_loss:.4f}")
        if no_improve >= CONFIG["unmerge_training"]["patience"]:
            print(f"Early stopping unmerge at epoch {ep+1}")
            break
    if best_state_u is not None:
        unmerge_model.load_state_dict(best_state_u)

    test_loss = 0.0
    n_batches = 0
    unmerge_model.eval()
    with torch.no_grad():
        for batch in test_loader_un:
            x = batch["hlt"].to(device)
            mask = batch["mask"].to(device)
            token_idx = batch["token_idx"].to(device)
            pred_count = batch["pred_count"].to(device)
            true_count = batch["true_count"].to(device)
            target = batch["target"].to(device)
            preds = unmerge_model(x, mask, token_idx, pred_count)
            loss = set_chamfer_loss(preds, target, true_count)
            test_loss += loss.item()
            n_batches += 1
    test_loss = test_loss / max(n_batches, 1)
    print(f"Unmerge test loss: {test_loss:.4f}")

    # ------------------- Build unmerged dataset ------------------- #
    print("\n" + "=" * 70)
    print("STEP 5: BUILD UNMERGED DATASET")
    print("=" * 70)
    unmerged_const, unmerged_mask = build_unmerged_dataset(
        features_hlt_std,
        hlt_mask,
        hlt_const,
        pred_counts,
        unmerge_model,
        tgt_mean,
        tgt_std,
        max_count,
        args.max_constits,
        device,
        BS_un,
    )
    features_unmerged = compute_features(unmerged_const, unmerged_mask)
    features_unmerged_std = standardize(features_unmerged, unmerged_mask, feat_means, feat_stds)

    train_ds_unmerged = JetDataset(features_unmerged_std[train_idx], unmerged_mask[train_idx], all_labels[train_idx])
    val_ds_unmerged = JetDataset(features_unmerged_std[val_idx], unmerged_mask[val_idx], all_labels[val_idx])
    test_ds_unmerged = JetDataset(features_unmerged_std[test_idx], unmerged_mask[test_idx], all_labels[test_idx])

    train_loader_um = DataLoader(train_ds_unmerged, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_um = DataLoader(val_ds_unmerged, batch_size=BS, shuffle=False)
    test_loader_um = DataLoader(test_ds_unmerged, batch_size=BS, shuffle=False)

    # ------------------- Train unmerge-model classifier ------------------- #
    print("\n" + "=" * 70)
    print("STEP 6: UNMERGE MODEL CLASSIFIER")
    print("=" * 70)
    unmerge_cls = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_ucls = torch.optim.AdamW(unmerge_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_ucls = get_scheduler(opt_ucls, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_u, best_state_ucls, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="UnmergeCls"):
        _, train_auc = train_classifier(unmerge_cls, train_loader_um, opt_ucls, device)
        val_auc, _, _ = eval_classifier(unmerge_cls, val_loader_um, device)
        sch_ucls.step()
        if val_auc > best_auc_u:
            best_auc_u = val_auc
            best_state_ucls = {k: v.detach().cpu().clone() for k, v in unmerge_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_u:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping unmerge classifier at epoch {ep+1}")
            break
    if best_state_ucls is not None:
        unmerge_cls.load_state_dict(best_state_ucls)

    auc_unmerge, preds_unmerge, _ = eval_classifier(unmerge_cls, test_loader_um, device)

    # ------------------- Train unmerge-model classifier + KD ------------------- #
    print("\n" + "=" * 70)
    print("STEP 7: UNMERGE MODEL + KD")
    print("=" * 70)
    kd_train_ds = UnmergeKDDataset(
        features_unmerged_std[train_idx],
        unmerged_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        all_labels[train_idx],
    )
    kd_val_ds = UnmergeKDDataset(
        features_unmerged_std[val_idx],
        unmerged_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        all_labels[val_idx],
    )
    kd_test_ds = UnmergeKDDataset(
        features_unmerged_std[test_idx],
        unmerged_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        all_labels[test_idx],
    )
    kd_train_loader = DataLoader(kd_train_ds, batch_size=BS, shuffle=True, drop_last=True)
    kd_val_loader = DataLoader(kd_val_ds, batch_size=BS, shuffle=False)
    kd_test_loader = DataLoader(kd_test_ds, batch_size=BS, shuffle=False)

    kd_student = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_kd = torch.optim.AdamW(kd_student.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_kd = get_scheduler(opt_kd, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

    best_auc_kd, best_state_kd, no_improve = 0.0, None, 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None

    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="Unmerge+KD"):
        current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
        kd_cfg_ep = dict(kd_cfg)
        kd_cfg_ep["alpha_kd"] = current_alpha

        train_loss, train_auc = train_kd_epoch(kd_student, teacher, kd_train_loader, opt_kd, device, kd_cfg_ep)
        val_auc, _, _ = evaluate_kd(kd_student, kd_val_loader, device)
        sch_kd.step()

        if not kd_active and kd_cfg["adaptive_alpha"]:
            val_loss = evaluate_bce_loss_unmerged(kd_student, kd_val_loader, device)
            if prev_val_loss is not None and abs(prev_val_loss - val_loss) < kd_cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = val_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
                print(f"Activating KD ramp at epoch {ep+1} (val_loss={val_loss:.4f})")

        if val_auc > best_auc_kd:
            best_auc_kd = val_auc
            best_state_kd = {k: v.detach().cpu().clone() for k, v in kd_student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_kd:.4f} | alpha_kd={current_alpha:.2f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping KD student at epoch {ep+1}")
            break

    if best_state_kd is not None:
        kd_student.load_state_dict(best_state_kd)

    if kd_cfg["self_train"]:
        print("\nSTEP 7B: SELF-TRAIN (pseudo-label fine-tune)")
        opt_st = torch.optim.AdamW(kd_student.parameters(), lr=kd_cfg["self_train_lr"])
        best_auc_st = best_auc_kd
        no_improve = 0
        for ep in range(kd_cfg["self_train_epochs"]):
            st_loss = self_train_student(kd_student, teacher, kd_train_loader, opt_st, device, kd_cfg)
            val_auc, _, _ = evaluate_kd(kd_student, kd_val_loader, device)
            if val_auc > best_auc_st:
                best_auc_st = val_auc
                best_state_kd = {k: v.detach().cpu().clone() for k, v in kd_student.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if (ep + 1) % 2 == 0:
                print(f"Self ep {ep+1}: loss={st_loss:.4f}, val_auc={val_auc:.4f}, best={best_auc_st:.4f}")
            if no_improve >= kd_cfg["self_train_patience"]:
                break
        if best_state_kd is not None:
            kd_student.load_state_dict(best_state_kd)

    auc_unmerge_kd, preds_unmerge_kd, _ = evaluate_kd(kd_student, kd_test_loader, device)

    # ------------------- Final evaluation ------------------- #
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"Unmerge Model    AUC: {auc_unmerge:.4f}")
    print(f"Unmerge + KD     AUC: {auc_unmerge_kd:.4f}")

    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_u, tpr_u, _ = roc_curve(labs, preds_unmerge)
    fpr_k, tpr_k, _ = roc_curve(labs, preds_unmerge_kd)

    def plot_roc(lines, out_name):
        plt.figure(figsize=(8, 6))
        for tpr, fpr, style, label, color in lines:
            plt.plot(tpr, fpr, style, label=label, color=color, linewidth=2)
        plt.ylabel("False Positive Rate", fontsize=12)
        plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
        plt.legend(fontsize=12, frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_root / out_name, dpi=300)
        plt.close()

    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_u, fpr_u, ":", f"Unmerge Model (AUC={auc_unmerge:.3f})", "forestgreen"),
            (tpr_k, fpr_k, "-.", f"Unmerge+KD (AUC={auc_unmerge_kd:.3f})", "darkorange"),
        ],
        "results_all.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
        ],
        "results_teacher_baseline.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_u, fpr_u, ":", f"Unmerge Model (AUC={auc_unmerge:.3f})", "forestgreen"),
        ],
        "results_teacher_unmerge.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_k, fpr_k, "-.", f"Unmerge+KD (AUC={auc_unmerge_kd:.3f})", "darkorange"),
        ],
        "results_teacher_unmerge_kd.png",
    )

    np.savez(
        save_root / "results.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_unmerge=auc_unmerge,
        auc_unmerge_kd=auc_unmerge_kd,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_unmerge=fpr_u,
        tpr_unmerge=tpr_u,
        fpr_unmerge_kd=fpr_k,
        tpr_unmerge_kd=tpr_k,
        unmerge_test_loss=test_loss,
        max_merge_count=max_count,
    )

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": auc_teacher}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": auc_baseline}, save_root / "baseline.pt")
        torch.save({"model": count_model.state_dict(), "acc": best_acc}, save_root / "merge_count.pt")
        torch.save({"model": unmerge_model.state_dict(), "loss": best_val_loss}, save_root / "unmerge_predictor.pt")
        torch.save({"model": unmerge_cls.state_dict(), "auc": auc_unmerge}, save_root / "unmerge_classifier.pt")
        torch.save({"model": kd_student.state_dict(), "auc": auc_unmerge_kd}, save_root / "unmerge_kd.pt")

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()

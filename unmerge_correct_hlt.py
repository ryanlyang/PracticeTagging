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
import json
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt
from tqdm import tqdm


# ----------------------------- Reproducibility ----------------------------- #
RANDOM_SEED = 52
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


CONFIG = {
    "hlt_effects": {
        "pt_resolution": 0.0,
        "eta_resolution": 0.0,
        "phi_resolution": 0.0,
        "pt_threshold_offline": 0.5,
        "pt_threshold_hlt": 1.5,
        "merge_enabled": True,
        "merge_radius": 0.01,
        "efficiency_loss": 0.0,
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
        "physics_weight": 0.0,
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


def load_raw_constituents_from_h5(files, max_jets, max_constits):
    """Load raw constituent four-vectors directly from HDF5 files."""
    const_list = []
    label_list = []
    jets_read = 0

    for fname in files:
        if jets_read >= max_jets:
            break

        with h5py.File(fname, "r") as f:
            n_file = int(f["labels"].shape[0])
            take = min(n_file, max_jets - jets_read)

            pt = f["fjet_clus_pt"][:take, :max_constits].astype(np.float32)
            eta = f["fjet_clus_eta"][:take, :max_constits].astype(np.float32)
            phi = f["fjet_clus_phi"][:take, :max_constits].astype(np.float32)
            ene = f["fjet_clus_E"][:take, :max_constits].astype(np.float32)
            const = np.stack([pt, eta, phi, ene], axis=-1).astype(np.float32)

            labels = f["labels"][:take]
            if labels.ndim > 1:
                if labels.shape[1] == 1:
                    labels = labels[:, 0]
                else:
                    labels = np.argmax(labels, axis=1)
            labels = labels.astype(np.int64)

            const_list.append(const)
            label_list.append(labels)
            jets_read += take

    if not const_list:
        raise RuntimeError("No jets were loaded from input files.")

    all_const = np.concatenate(const_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)
    return all_const, all_labels


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


@torch.no_grad()
def evaluate_bce_loss_dual(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        xa = batch["feat_a"].to(device)
        ma = batch["mask_a"].to(device)
        xb = batch["feat_b"].to(device)
        mb = batch["mask_b"].to(device)
        y = batch["label"].to(device)
        logits = model(xa, ma, xb, mb).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total += loss.item() * len(y)
        count += len(y)
    return total / max(count, 1)

def apply_hlt_effects_with_tracking(const, mask, cfg, seed=42):
    """
    Returns HLT view with per-token origin tracking.
    """
    rs = np.random.RandomState(int(seed))
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
    pt_noise = rs.normal(1.0, hcfg["pt_resolution"], (n_jets, max_part))
    pt_noise = np.clip(pt_noise, 0.5, 1.5)
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)

    eta_noise = rs.normal(0, hcfg["eta_resolution"], (n_jets, max_part))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)

    phi_noise = rs.normal(0, hcfg["phi_resolution"], (n_jets, max_part))
    new_phi = hlt[:, :, 2] + phi_noise
    hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)

    # Recalculate E (massless approx)
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)

    # Effect 4: Random efficiency loss
    n_lost_eff = 0
    if hcfg["efficiency_loss"] > 0:
        random_loss = rs.random_sample((n_jets, max_part)) < hcfg["efficiency_loss"]
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


def compute_jet_pt(const, mask):
    pt = np.maximum(const[:, :, 0], 0.0)
    phi = const[:, :, 2]
    w = mask.astype(np.float32)
    px = (pt * np.cos(phi) * w).sum(axis=1)
    py = (pt * np.sin(phi) * w).sum(axis=1)
    return np.sqrt(px ** 2 + py ** 2)


def build_pt_edges(pt_truth, n_bins):
    valid = np.isfinite(pt_truth) & (pt_truth > 1e-8)
    pt = pt_truth[valid]
    if pt.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)

    q = np.linspace(0.0, 1.0, int(max(n_bins, 1)) + 1)
    edges = np.quantile(pt, q)
    edges = np.unique(edges)
    if edges.size < 2:
        center = float(np.median(pt))
        edges = np.array([max(center * 0.9, 0.0), center * 1.1 + 1e-6], dtype=np.float64)
    return edges.astype(np.float64)


def jet_response_resolution(pt_truth, pt_reco, edges, min_count):
    records = []
    valid = np.isfinite(pt_truth) & np.isfinite(pt_reco) & (pt_truth > 1e-8)
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        in_bin = valid & (pt_truth >= lo)
        if i < len(edges) - 2:
            in_bin = in_bin & (pt_truth < hi)
        else:
            in_bin = in_bin & (pt_truth <= hi)
        n = int(in_bin.sum())
        if n < int(min_count):
            continue
        ratio = pt_reco[in_bin] / pt_truth[in_bin]
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size == 0:
            continue
        records.append(
            {
                "pt_low": lo,
                "pt_high": hi,
                "pt_center": 0.5 * (lo + hi),
                "count": int(ratio.size),
                "response": float(np.mean(ratio)),
                "resolution": float(np.std(ratio)),
            }
        )
    return records


def plot_response_resolution(records_a, records_b, label_a, label_b, out_path):
    centers_a = np.array([r["pt_center"] for r in records_a], dtype=np.float64)
    resp_a = np.array([r["response"] for r in records_a], dtype=np.float64)
    reso_a = np.array([r["resolution"] for r in records_a], dtype=np.float64)
    centers_b = np.array([r["pt_center"] for r in records_b], dtype=np.float64)
    resp_b = np.array([r["response"] for r in records_b], dtype=np.float64)
    reso_b = np.array([r["resolution"] for r in records_b], dtype=np.float64)

    plt.figure(figsize=(10, 4.2))
    plt.subplot(1, 2, 1)
    if centers_a.size > 0:
        plt.plot(centers_a, resp_a, "o-", label=label_a, color="steelblue")
    if centers_b.size > 0:
        plt.plot(centers_b, resp_b, "s--", label=label_b, color="forestgreen")
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Jet pT truth (offline)")
    plt.ylabel("Response: pT_reco / pT_truth")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)
    if centers_a.size > 0:
        plt.plot(centers_a, reso_a, "o-", label=label_a, color="steelblue")
    if centers_b.size > 0:
        plt.plot(centers_b, reso_b, "s--", label=label_b, color="forestgreen")
    plt.xlabel("Jet pT truth (offline)")
    plt.ylabel("Resolution: std(pT_reco / pT_truth)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


class JetDataset(Dataset):
    def __init__(self, feat, mask, labels):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"feat": self.feat[i], "mask": self.mask[i], "label": self.labels[i]}


class DualViewJetDataset(Dataset):
    def __init__(self, feat_a, mask_a, feat_b, mask_b, labels):
        self.feat_a = torch.tensor(feat_a, dtype=torch.float32)
        self.mask_a = torch.tensor(mask_a, dtype=torch.bool)
        self.feat_b = torch.tensor(feat_b, dtype=torch.float32)
        self.mask_b = torch.tensor(mask_b, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "feat_a": self.feat_a[i],
            "mask_a": self.mask_a[i],
            "feat_b": self.feat_b[i],
            "mask_b": self.mask_b[i],
            "label": self.labels[i],
        }


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


class DualViewKDDataset(Dataset):
    def __init__(self, feat_a, mask_a, feat_b, mask_b, feat_off, mask_off, labels):
        self.feat_a = torch.tensor(feat_a, dtype=torch.float32)
        self.mask_a = torch.tensor(mask_a, dtype=torch.bool)
        self.feat_b = torch.tensor(feat_b, dtype=torch.float32)
        self.mask_b = torch.tensor(mask_b, dtype=torch.bool)
        self.off = torch.tensor(feat_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "feat_a": self.feat_a[i],
            "mask_a": self.mask_a[i],
            "feat_b": self.feat_b[i],
            "mask_b": self.mask_b[i],
            "off": self.off[i],
            "mask_off": self.mask_off[i],
            "label": self.labels[i],
        }


class UnmergeDataset(Dataset):
    def __init__(
        self,
        feat_hlt,
        mask_hlt,
        constituents_hlt,
        constituents_off,
        samples,
        max_count,
        tgt_mean,
        tgt_std,
        target_mode,
    ):
        self.feat_hlt = feat_hlt
        self.mask_hlt = mask_hlt
        self.constituents_hlt = constituents_hlt
        self.constituents_off = constituents_off
        self.samples = samples
        self.max_count = max_count
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std
        self.target_mode = target_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        jet_idx, token_idx, origin, pred_count = self.samples[i]
        true_count = min(len(origin), self.max_count)
        origin = origin[:true_count]
        target_abs = self.constituents_off[jet_idx, origin, :4].astype(np.float32)
        parent = self.constituents_hlt[jet_idx, token_idx, :4].astype(np.float32)
        if self.target_mode == "normalized":
            pt_p = max(parent[0], 1e-8)
            eta_p = parent[1]
            phi_p = parent[2]
            e_p = max(parent[3], 1e-8)
            pt_frac = target_abs[:, 0] / pt_p
            e_frac = target_abs[:, 3] / e_p
            deta = target_abs[:, 1] - eta_p
            dphi = np.arctan2(np.sin(target_abs[:, 2] - phi_p), np.cos(target_abs[:, 2] - phi_p))
            target = np.stack([pt_frac, deta, dphi, e_frac], axis=-1).astype(np.float32)
        else:
            target = target_abs

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
            "parent": torch.tensor(parent, dtype=torch.float32),
        }

    def get_true_counts(self):
        counts = []
        for s in self.samples:
            true_count = min(len(s[2]), self.max_count)
            counts.append(true_count)
        return np.array(counts, dtype=np.int64)


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
        mask_safe = mask.clone()
        empty = ~mask_safe.any(dim=1)
        if empty.any():
            mask_safe[empty, 0] = True
        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)
        h = self.encoder(h, src_key_padding_mask=~mask_safe)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, attn_weights = self.pool_attn(
            query, h, h, key_padding_mask=~mask_safe, need_weights=True, average_attn_weights=True
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


class DualViewCrossAttnClassifier(nn.Module):
    def __init__(self, input_dim_a=7, input_dim_b=7, embed_dim=128, num_heads=8, num_layers=6, ff_dim=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj_a = nn.Sequential(
            nn.Linear(input_dim_a, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_proj_b = nn.Sequential(
            nn.Linear(input_dim_b, embed_dim),
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
        self.encoder_a = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.encoder_b = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn_a = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.pool_attn_b = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_a_to_b = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_b_to_a = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim * 4)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, feat_a, mask_a, feat_b, mask_b):
        bsz, seq_len, _ = feat_a.shape
        mask_a_safe = mask_a.clone()
        mask_b_safe = mask_b.clone()
        empty_a = ~mask_a_safe.any(dim=1)
        empty_b = ~mask_b_safe.any(dim=1)
        if empty_a.any():
            mask_a_safe[empty_a, 0] = True
        if empty_b.any():
            mask_b_safe[empty_b, 0] = True
        h_a = self.input_proj_a(feat_a.view(-1, feat_a.size(-1))).view(bsz, seq_len, -1)
        h_b = self.input_proj_b(feat_b.view(-1, feat_b.size(-1))).view(bsz, seq_len, -1)
        h_a = self.encoder_a(h_a, src_key_padding_mask=~mask_a_safe)
        h_b = self.encoder_b(h_b, src_key_padding_mask=~mask_b_safe)
        query = self.pool_query.expand(bsz, -1, -1)
        pooled_a, _ = self.pool_attn_a(query, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)
        pooled_b, _ = self.pool_attn_b(query, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
        cross_a, _ = self.cross_a_to_b(pooled_a, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
        cross_b, _ = self.cross_b_to_a(pooled_b, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)
        fused = torch.cat([pooled_a, pooled_b, cross_a, cross_b], dim=-1).squeeze(1)
        fused = self.norm(fused)
        logits = self.classifier(fused)
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
        mask_safe = mask.clone()
        empty = ~mask_safe.any(dim=1)
        if empty.any():
            mask_safe[empty, 0] = True
        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)
        h = self.encoder(h, src_key_padding_mask=~mask_safe)
        logits = self.head(h)
        return logits


class UnmergePredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        max_count,
        head_mode,
        parent_mode,
        relpos_mode,
        local_attn_mode,
        local_attn_radius,
        local_attn_scale,
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
        self.head_mode = head_mode
        self.parent_mode = parent_mode
        self.relpos_mode = relpos_mode
        self.local_attn_mode = local_attn_mode
        self.local_attn_radius = local_attn_radius
        self.local_attn_scale = local_attn_scale
        self.num_heads = num_heads

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if relpos_mode not in {"none", "attn"}:
            raise ValueError(f"Unsupported relpos_mode: {relpos_mode}")
        if local_attn_mode not in {"none", "soft", "hard"}:
            raise ValueError(f"Unsupported local_attn_mode: {local_attn_mode}")

        if relpos_mode == "none" and local_attn_mode == "none":
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
        else:
            if relpos_mode == "attn":
                self.relpos_mlp = nn.Sequential(
                    nn.Linear(3, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, num_heads),
                )
            else:
                self.relpos_mlp = None
            self.encoder_layers = nn.ModuleList(
                [RelPosEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
            )

        self.count_embed = nn.Embedding(max_count + 1, count_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(embed_dim * 2 + count_embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        if parent_mode not in {"none", "query", "cross"}:
            raise ValueError(f"Unsupported parent_mode: {parent_mode}")
        if parent_mode == "query":
            self.parent_query_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
        elif parent_mode == "cross":
            self.parent_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)

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

        if head_mode not in {"single", "two", "four"}:
            raise ValueError(f"Unsupported head_mode: {head_mode}")

        def make_head(out_dim):
            return nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, out_dim),
            )

        if head_mode == "single":
            self.out = make_head(4)
        elif head_mode == "two":
            self.out_pe = make_head(2)   # pT, E
            self.out_ang = make_head(2)  # eta, phi
        else:  # "four"
            self.out_pt = make_head(1)
            self.out_eta = make_head(1)
            self.out_phi = make_head(1)
            self.out_e = make_head(1)

    def forward(self, x, mask, token_idx, count):
        batch_size, seq_len, _ = x.shape
        mask_safe = mask.clone()
        empty = ~mask_safe.any(dim=1)
        if empty.any():
            mask_safe[empty, 0] = True
        h = x.view(-1, self.input_dim)
        h = self.input_proj(h)
        h = h.view(batch_size, seq_len, -1)
        if self.relpos_mode == "none" and self.local_attn_mode == "none":
            h = self.encoder(h, src_key_padding_mask=~mask_safe)
        else:
            rel_bias = self._build_relpos_bias(x)
            for layer in self.encoder_layers:
                h = layer(h, mask_safe, rel_bias)

        idx = token_idx.view(-1, 1, 1).expand(-1, 1, self.embed_dim)
        h_t = h.gather(1, idx).squeeze(1)
        h_sum = (h * mask.unsqueeze(-1)).sum(dim=1)
        h_avg = h_sum / mask.sum(dim=1, keepdim=True).clamp(min=1)

        c_emb = self.count_embed(count)
        cond = self.cond_proj(torch.cat([h_t, h_avg, c_emb], dim=1))

        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + cond.unsqueeze(1)
        if self.parent_mode == "query":
            parent_bias = self.parent_query_proj(h_t).unsqueeze(1)
            queries = queries + parent_bias
        elif self.parent_mode == "cross":
            parent_kv = h_t.unsqueeze(1)
            parent_ctx, _ = self.parent_attn(queries, parent_kv, parent_kv, need_weights=False)
            queries = queries + parent_ctx

        dec = self.decoder(queries, h, memory_key_padding_mask=~mask)
        if self.head_mode == "single":
            out = self.out(dec)
        elif self.head_mode == "two":
            pe = self.out_pe(dec)
            ang = self.out_ang(dec)
            out = torch.zeros(
                dec.size(0), dec.size(1), 4, device=dec.device, dtype=dec.dtype
            )
            out[:, :, 0] = pe[:, :, 0]   # pT
            out[:, :, 3] = pe[:, :, 1]   # E
            out[:, :, 1] = ang[:, :, 0]  # eta
            out[:, :, 2] = ang[:, :, 1]  # phi
        else:  # "four"
            out = torch.zeros(
                dec.size(0), dec.size(1), 4, device=dec.device, dtype=dec.dtype
            )
            out[:, :, 0] = self.out_pt(dec).squeeze(-1)
            out[:, :, 1] = self.out_eta(dec).squeeze(-1)
            out[:, :, 2] = self.out_phi(dec).squeeze(-1)
            out[:, :, 3] = self.out_e(dec).squeeze(-1)
        return out

    def _build_relpos_bias(self, x):
        # x: [B, L, input_dim]; use delta_eta/delta_phi from features
        eta = x[:, :, 0]
        phi = x[:, :, 1]
        deta = eta[:, :, None] - eta[:, None, :]
        dphi = torch.atan2(torch.sin(phi[:, :, None] - phi[:, None, :]),
                           torch.cos(phi[:, :, None] - phi[:, None, :]))
        dR = torch.sqrt(deta ** 2 + dphi ** 2 + 1e-8)
        bias = 0.0
        if self.relpos_mode == "attn" and self.relpos_mlp is not None:
            rel = torch.stack([deta, dphi, dR], dim=-1)
            rel_bias = self.relpos_mlp(rel)  # [B, L, L, H]
            rel_bias = rel_bias.permute(0, 3, 1, 2).contiguous()  # [B, H, L, L]
            bias = rel_bias
        if self.local_attn_mode != "none":
            if self.local_attn_radius <= 0:
                raise ValueError("local_attn_radius must be > 0 when local_attn_mode is enabled.")
            if self.local_attn_mode == "soft":
                local_bias = -self.local_attn_scale * (dR / self.local_attn_radius) ** 2
            else:  # hard
                local_bias = torch.zeros_like(dR)
                local_bias[dR > self.local_attn_radius] = -1e4
            local_bias = local_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            bias = local_bias if isinstance(bias, float) else bias + local_bias
        if isinstance(bias, float):
            bias = torch.zeros((x.size(0), self.num_heads, x.size(1), x.size(1)), device=x.device)
        return bias


class RelPosEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask, rel_bias):
        # rel_bias: [B, H, L, L] -> [B*H, L, L] for MHA
        B, L, _ = x.shape
        attn_mask = rel_bias.reshape(B * rel_bias.size(1), L, L)
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask, key_padding_mask=~mask, need_weights=False)
        x = x + attn_out
        h2 = self.norm2(x)
        x = x + self.ff(h2)
        return x


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


def physics_loss(pred_abs, true_counts, hlt_token):
    # Compare sum of predicted constituents to merged HLT token (4-vector)
    # hlt_token: (B, 4) [pt, eta, phi, E]
    total = 0.0
    for i in range(pred_abs.size(0)):
        k = int(true_counts[i].item())
        pred = pred_abs[i, :k]
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
    return total / max(pred_abs.size(0), 1)


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


def train_classifier_dual(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    preds, labs = [], []
    for batch in loader:
        xa = batch["feat_a"].to(device)
        ma = batch["mask_a"].to(device)
        xb = batch["feat_b"].to(device)
        mb = batch["mask_b"].to(device)
        y = batch["label"].to(device)
        opt.zero_grad()
        logits = model(xa, ma, xb, mb).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
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


@torch.no_grad()
def eval_classifier_dual(model, loader, device):
    model.eval()
    preds, labs = [], []
    warned = False
    for batch in loader:
        xa = batch["feat_a"].to(device)
        ma = batch["mask_a"].to(device)
        xb = batch["feat_b"].to(device)
        mb = batch["mask_b"].to(device)
        logits = model(xa, ma, xb, mb).squeeze(1)
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


def train_kd_epoch_dual(student, teacher, loader, opt, device, kd_cfg):
    student.train()
    teacher.eval()

    total_loss = 0.0
    preds, labs = [], []

    T = kd_cfg["temperature"]
    a_kd = kd_cfg["alpha_kd"]

    for batch in loader:
        xa = batch["feat_a"].to(device)
        ma = batch["mask_a"].to(device)
        xb = batch["feat_b"].to(device)
        mb = batch["mask_b"].to(device)
        x_o = batch["off"].to(device)
        m_o = batch["mask_off"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            t_logits = teacher(x_o, m_o).squeeze(1)

        opt.zero_grad()
        s_logits = student(xa, ma, xb, mb).squeeze(1)

        loss_hard = F.binary_cross_entropy_with_logits(s_logits, y)
        if kd_cfg["conf_weighted"]:
            loss_kd = kd_loss_conf_weighted(s_logits, t_logits, T)
        else:
            s_soft = torch.sigmoid(s_logits / T)
            t_soft = torch.sigmoid(t_logits / T)
            loss_kd = F.binary_cross_entropy(s_soft, t_soft) * (T ** 2)

        loss = (1.0 - a_kd) * loss_hard + a_kd * loss_kd
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


@torch.no_grad()
def evaluate_kd_dual(student, loader, device):
    student.eval()
    preds, labs = [], []
    for batch in loader:
        xa = batch["feat_a"].to(device)
        ma = batch["mask_a"].to(device)
        xb = batch["feat_b"].to(device)
        mb = batch["mask_b"].to(device)
        y = batch["label"].to(device)
        logits = student(xa, ma, xb, mb).squeeze(1)
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


def self_train_student_dual(student, teacher, loader, opt, device, cfg):
    student.train()
    teacher.eval()
    conf_min = cfg["self_train_conf_min"]
    conf_power = cfg["self_train_conf_power"]
    total_loss = 0.0
    count = 0
    for batch in loader:
        xa = batch["feat_a"].to(device)
        ma = batch["mask_a"].to(device)
        xb = batch["feat_b"].to(device)
        mb = batch["mask_b"].to(device)
        x_o = batch["off"].to(device)
        m_o = batch["mask_off"].to(device)

        with torch.no_grad():
            if cfg["self_train_source"] == "teacher":
                t_logits = teacher(x_o, m_o).squeeze(1)
                probs = torch.sigmoid(t_logits)
            else:
                s_logits = student(xa, ma, xb, mb).squeeze(1)
                probs = torch.sigmoid(s_logits)

            conf = torch.clamp(2 * torch.abs(probs - 0.5), 0.0, 1.0)
            conf = torch.clamp(conf, min=conf_min) ** conf_power

        opt.zero_grad()
        logits = student(xa, ma, xb, mb).squeeze(1)
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


def summarize_merge_count_metrics(preds, labs):
    """Return diagnostics beyond exact multiclass accuracy."""
    if labs.size == 0:
        return {
            "exact_acc": 0.0,
            "within1_acc": 0.0,
            "binary_acc": 0.0,
            "merged_precision": 0.0,
            "merged_recall": 0.0,
            "merged_exact_acc": 0.0,
            "majority_baseline_acc": 0.0,
            "majority_class": -1,
        }

    exact_acc = float((preds == labs).mean())
    within1_acc = float((np.abs(preds - labs) <= 1).mean())

    merged_true = labs > 0
    merged_pred = preds > 0
    binary_acc = float((merged_pred == merged_true).mean())

    tp = int(np.logical_and(merged_pred, merged_true).sum())
    fp = int(np.logical_and(merged_pred, ~merged_true).sum())
    fn = int(np.logical_and(~merged_pred, merged_true).sum())
    merged_precision = float(tp / max(tp + fp, 1))
    merged_recall = float(tp / max(tp + fn, 1))

    if merged_true.any():
        merged_exact_acc = float((preds[merged_true] == labs[merged_true]).mean())
    else:
        merged_exact_acc = 0.0

    binc = np.bincount(labs.astype(np.int64))
    majority_class = int(np.argmax(binc))
    majority_baseline_acc = float((labs == majority_class).mean())

    return {
        "exact_acc": exact_acc,
        "within1_acc": within1_acc,
        "binary_acc": binary_acc,
        "merged_precision": merged_precision,
        "merged_recall": merged_recall,
        "merged_exact_acc": merged_exact_acc,
        "majority_baseline_acc": majority_baseline_acc,
        "majority_class": majority_class,
    }


def print_merge_count_distribution(labels, mask, num_classes, split_name):
    valid = labels[mask].astype(np.int64)
    if valid.size == 0:
        print(f"{split_name} merge-count distribution: no valid tokens.")
        return

    counts = np.bincount(valid, minlength=num_classes)
    total = int(counts.sum())
    parts = []
    for cls in range(num_classes):
        c = int(counts[cls])
        if c == 0:
            continue
        frac = 100.0 * c / max(total, 1)
        true_count = cls + 1
        parts.append(f"{true_count}:{c} ({frac:.1f}%)")

    print(f"{split_name} merge-count distribution (true constituents per active HLT token):")
    print("  " + ", ".join(parts))


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
    target_mode,
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
                    if target_mode == "normalized":
                        parent = hlt_const[chunk[k][0], chunk[k][1], :4].astype(np.float32)
                        pt_p = max(parent[0], 1e-8)
                        eta_p = parent[1]
                        phi_p = parent[2]
                        e_p = max(parent[3], 1e-8)
                        pt = np.clip(pred[:, 0] * pt_p, 0.0, None)
                        eta = np.clip(pred[:, 1] + eta_p, -5.0, 5.0)
                        phi = np.arctan2(np.sin(pred[:, 2] + phi_p), np.cos(pred[:, 2] + phi_p))
                        E = np.clip(pred[:, 3] * e_p, 0.0, None)
                        pred = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)
                    else:
                        pred[:, 0] = np.clip(pred[:, 0], 0.0, None)
                        pred[:, 1] = np.clip(pred[:, 1], -5.0, 5.0)
                        pred[:, 2] = np.arctan2(np.sin(pred[:, 2]), np.cos(pred[:, 2]))
                        pred[:, 3] = np.clip(pred[:, 3], 0.0, None)
                    pred_map[(chunk[k][0], chunk[k][1])] = pred

    new_const = np.zeros((n_jets, max_constits, 4), dtype=np.float32)
    new_mask = np.zeros((n_jets, max_constits), dtype=bool)
    new_flag = np.zeros((n_jets, max_constits), dtype=np.float32)

    for j in range(n_jets):
        parts = []
        for idx in range(max_part):
            if not mask_hlt[j, idx]:
                continue
            if pred_counts[j, idx] <= 1:
                parts.append((hlt_const[j, idx], 0.0))
            else:
                pred = pred_map.get((j, idx))
                if pred is not None:
                    parts.extend([(p, 1.0) for p in pred])
        if len(parts) == 0:
            continue
        parts_arr = np.array([p[0] for p in parts], dtype=np.float32)
        flags_arr = np.array([p[1] for p in parts], dtype=np.float32)
        order = np.argsort(parts_arr[:, 0])[::-1]
        parts_arr = parts_arr[order]
        flags_arr = flags_arr[order]
        n_keep = min(len(parts_arr), max_constits)
        new_const[j, :n_keep] = parts_arr[:n_keep]
        new_mask[j, :n_keep] = True
        new_flag[j, :n_keep] = flags_arr[:n_keep]

    return new_const, new_mask, new_flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--offset_jets", type=int, default=0, help="Skip this many jets to get a new slice.")
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--max_merge_count", type=int, default=10)
    parser.add_argument("--pt_threshold_offline", type=float, default=CONFIG["hlt_effects"]["pt_threshold_offline"])
    parser.add_argument("--pt_threshold_hlt", type=float, default=CONFIG["hlt_effects"]["pt_threshold_hlt"])
    parser.add_argument("--merge_radius", type=float, default=CONFIG["hlt_effects"]["merge_radius"])
    parser.add_argument("--disable_merge", action="store_true")
    parser.add_argument("--pt_resolution", type=float, default=CONFIG["hlt_effects"]["pt_resolution"])
    parser.add_argument("--eta_resolution", type=float, default=CONFIG["hlt_effects"]["eta_resolution"])
    parser.add_argument("--phi_resolution", type=float, default=CONFIG["hlt_effects"]["phi_resolution"])
    parser.add_argument("--efficiency_loss", type=float, default=CONFIG["hlt_effects"]["efficiency_loss"])
    parser.add_argument("--physics_weight", type=float, default=CONFIG["unmerge_training"]["physics_weight"])
    parser.add_argument(
        "--unmerge_head_mode",
        type=str,
        default="single",
        choices=["single", "two", "four"],
        help="Output head mode for unmerger: single (4-d), two (pT/E + eta/phi), four (pT, eta, phi, E).",
    )
    parser.add_argument(
        "--unmerge_parent_mode",
        type=str,
        default="none",
        choices=["none", "query", "cross"],
        help="Parent-token conditioning for unmerger: none, query (add parent bias to queries), cross (cross-attend queries to parent).",
    )
    parser.add_argument(
        "--unmerge_relpos_mode",
        type=str,
        default="none",
        choices=["none", "attn"],
        help="Relative-position encoding mode for unmerger encoder: none or attn (add learned relpos bias to attention).",
    )
    parser.add_argument(
        "--unmerge_local_attn_mode",
        type=str,
        default="none",
        choices=["none", "soft", "hard"],
        help="Local attention bias for unmerger encoder: none, soft (Gaussian bias), hard (mask beyond radius).",
    )
    parser.add_argument(
        "--unmerge_local_attn_radius",
        type=float,
        default=0.2,
        help="Local attention radius in delta-R for unmerger encoder.",
    )
    parser.add_argument(
        "--unmerge_local_attn_scale",
        type=float,
        default=2.0,
        help="Scale for soft local attention bias (larger = stronger local focus).",
    )
    parser.add_argument(
        "--unmerge_target_mode",
        type=str,
        default="absolute",
        choices=["absolute", "normalized"],
        help="Unmerger target mode: absolute (pT,eta,phi,E) or normalized (pt/E fractions + dEta/dPhi).",
    )
    parser.add_argument(
        "--unmerge_count_balanced",
        action="store_true",
        help="Use count-balanced sampling for unmerger training (oversample higher merge counts).",
    )
    parser.add_argument(
        "--merge_count_unweighted",
        action="store_true",
        help="Disable inverse-frequency class weighting for merge-count training.",
    )
    parser.add_argument(
        "--roc_fpr_min",
        type=float,
        default=1e-4,
        help="Minimum FPR shown on log-scale ROC y-axis.",
    )
    parser.add_argument(
        "--response_n_bins",
        type=int,
        default=8,
        help="Number of quantile bins for jet pT response/resolution.",
    )
    parser.add_argument(
        "--response_min_count",
        type=int,
        default=30,
        help="Minimum jets per pT bin to report response/resolution.",
    )
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unmerge_new"))
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_save_models", action="store_true")
    args = parser.parse_args()

    CONFIG["hlt_effects"]["pt_threshold_offline"] = float(args.pt_threshold_offline)
    CONFIG["hlt_effects"]["pt_threshold_hlt"] = float(args.pt_threshold_hlt)
    CONFIG["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    CONFIG["hlt_effects"]["merge_enabled"] = not bool(args.disable_merge)
    CONFIG["hlt_effects"]["pt_resolution"] = float(args.pt_resolution)
    CONFIG["hlt_effects"]["eta_resolution"] = float(args.eta_resolution)
    CONFIG["hlt_effects"]["phi_resolution"] = float(args.phi_resolution)
    CONFIG["hlt_effects"]["efficiency_loss"] = float(args.efficiency_loss)
    CONFIG["unmerge_training"]["physics_weight"] = float(args.physics_weight)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")
    print("HLT config:")
    print(
        f"  pt_threshold_offline={CONFIG['hlt_effects']['pt_threshold_offline']}, "
        f"pt_threshold_hlt={CONFIG['hlt_effects']['pt_threshold_hlt']}, "
        f"merge_enabled={CONFIG['hlt_effects']['merge_enabled']}, "
        f"merge_radius={CONFIG['hlt_effects']['merge_radius']}, "
        f"pt/eta/phi_resolution=({CONFIG['hlt_effects']['pt_resolution']}, "
        f"{CONFIG['hlt_effects']['eta_resolution']}, {CONFIG['hlt_effects']['phi_resolution']}), "
        f"efficiency_loss={CONFIG['hlt_effects']['efficiency_loss']}"
    )

    train_path = Path(args.train_path)
    train_files = sorted(list(train_path.glob("*.h5")))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {train_path}")

    print("Loading raw constituents directly from HDF5...")
    max_jets_needed = args.offset_jets + args.n_train_jets
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets for offset {args.offset_jets} + n_train_jets {args.n_train_jets}. "
            f"Got {all_const_full.shape[0]}."
        )
    constituents_raw = all_const_full[args.offset_jets:args.offset_jets + args.n_train_jets]
    all_labels = all_labels_full[args.offset_jets:args.offset_jets + args.n_train_jets]
    all_labels = all_labels.astype(np.int64)
    print(f"Loaded: raw_constituents={constituents_raw.shape}, labels={all_labels.shape}")

    mask_raw = constituents_raw[:, :, 0] > 0

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
    test_loader_cnt = DataLoader(test_ds_cnt, batch_size=BS_cnt, shuffle=False)

    print_merge_count_distribution(count_label[train_idx], hlt_mask[train_idx], max_count, "Train")
    print_merge_count_distribution(count_label[val_idx], hlt_mask[val_idx], max_count, "Val")
    print_merge_count_distribution(count_label[test_idx], hlt_mask[test_idx], max_count, "Test")

    count_model = MergeCountPredictor(input_dim=7, num_classes=max_count, **CONFIG["merge_count_model"]).to(device)
    opt_c = torch.optim.AdamW(count_model.parameters(), lr=CONFIG["merge_count_training"]["lr"], weight_decay=CONFIG["merge_count_training"]["weight_decay"])
    sch_c = get_scheduler(opt_c, CONFIG["merge_count_training"]["warmup_epochs"], CONFIG["merge_count_training"]["epochs"])
    if args.merge_count_unweighted:
        class_weights = np.ones(max_count, dtype=np.float64)
        print("Merge-count class weights: disabled (all ones).")
    else:
        class_weights = compute_class_weights(count_label[train_idx], hlt_mask[train_idx], max_count)
        print(f"Merge-count class weights: {np.array2string(class_weights, precision=3)}")
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

    train_acc_cnt, train_preds_cnt, train_labs_cnt = eval_merge_count(count_model, train_loader_cnt, device)
    val_acc_cnt, val_preds_cnt, val_labs_cnt = eval_merge_count(count_model, val_loader_cnt, device)
    test_acc_cnt, test_preds_cnt, test_labs_cnt = eval_merge_count(count_model, test_loader_cnt, device)
    split_metrics = {
        "Train": summarize_merge_count_metrics(train_preds_cnt, train_labs_cnt),
        "Val": summarize_merge_count_metrics(val_preds_cnt, val_labs_cnt),
        "Test": summarize_merge_count_metrics(test_preds_cnt, test_labs_cnt),
    }
    print("Merge-count diagnostics:")
    for split_name, split_acc in [("Train", train_acc_cnt), ("Val", val_acc_cnt), ("Test", test_acc_cnt)]:
        m = split_metrics[split_name]
        majority_true_count = m["majority_class"] + 1 if m["majority_class"] >= 0 else -1
        print(
            f"  {split_name}: "
            f"exact={split_acc:.4f}, "
            f"within1={m['within1_acc']:.4f}, "
            f"binary(merged-vs-single)={m['binary_acc']:.4f}, "
            f"merged_exact={m['merged_exact_acc']:.4f}, "
            f"merged_P/R=({m['merged_precision']:.4f}/{m['merged_recall']:.4f}), "
            f"majority_baseline={m['majority_baseline_acc']:.4f} (count={majority_true_count})"
        )

    # Predict counts for all jets
    pred_counts = predict_counts(count_model, features_hlt_std, hlt_mask, BS_cnt, device, max_count)

    # ------------------- Train unmerger ------------------- #
    print("\n" + "=" * 70)
    print("STEP 4: UNMERGER")
    print("=" * 70)
    print(f"Unmerge head mode: {args.unmerge_head_mode}")
    print(f"Unmerge parent mode: {args.unmerge_parent_mode}")
    print(f"Unmerge relpos mode: {args.unmerge_relpos_mode}")
    print(f"Unmerge local attn: mode={args.unmerge_local_attn_mode}, radius={args.unmerge_local_attn_radius}, scale={args.unmerge_local_attn_scale}")
    print(f"Unmerge target mode: {args.unmerge_target_mode}")
    print(f"Unmerge count-balanced: {args.unmerge_count_balanced}")
    print(f"Unmerge physics weight: {CONFIG['unmerge_training']['physics_weight']}")
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
        target_abs = const_off[s[0], s[2], :4]
        if args.unmerge_target_mode == "normalized":
            parent = hlt_const[s[0], s[1], :4].astype(np.float32)
            pt_p = max(parent[0], 1e-8)
            eta_p = parent[1]
            phi_p = parent[2]
            e_p = max(parent[3], 1e-8)
            pt_frac = target_abs[:, 0] / pt_p
            e_frac = target_abs[:, 3] / e_p
            deta = target_abs[:, 1] - eta_p
            dphi = np.arctan2(np.sin(target_abs[:, 2] - phi_p), np.cos(target_abs[:, 2] - phi_p))
            target = np.stack([pt_frac, deta, dphi, e_frac], axis=-1)
            train_targets.append(target.astype(np.float32))
        else:
            train_targets.append(target_abs.astype(np.float32))
    flat_train = np.concatenate(train_targets, axis=0)
    tgt_mean = flat_train.mean(axis=0)
    tgt_std = flat_train.std(axis=0) + 1e-8
    tgt_mean_t = torch.tensor(tgt_mean, dtype=torch.float32, device=device)
    tgt_std_t = torch.tensor(tgt_std, dtype=torch.float32, device=device)

    def preds_to_abs(preds, parent):
        preds_unstd = preds * tgt_std_t + tgt_mean_t
        if args.unmerge_target_mode == "normalized":
            pt_p = parent[:, 0].clamp(min=1e-8)
            eta_p = parent[:, 1]
            phi_p = parent[:, 2]
            e_p = parent[:, 3].clamp(min=1e-8)
            pt = torch.clamp(preds_unstd[..., 0] * pt_p[:, None], min=0.0)
            eta = torch.clamp(preds_unstd[..., 1] + eta_p[:, None], min=-5.0, max=5.0)
            phi = torch.atan2(
                torch.sin(preds_unstd[..., 2] + phi_p[:, None]),
                torch.cos(preds_unstd[..., 2] + phi_p[:, None]),
            )
            E = torch.clamp(preds_unstd[..., 3] * e_p[:, None], min=0.0)
            return torch.stack([pt, eta, phi, E], dim=-1)
        pt = torch.clamp(preds_unstd[..., 0], min=0.0)
        eta = torch.clamp(preds_unstd[..., 1], min=-5.0, max=5.0)
        phi = torch.atan2(torch.sin(preds_unstd[..., 2]), torch.cos(preds_unstd[..., 2]))
        E = torch.clamp(preds_unstd[..., 3], min=0.0)
        return torch.stack([pt, eta, phi, E], dim=-1)

    BS_un = CONFIG["unmerge_training"]["batch_size"]
    train_ds_un = UnmergeDataset(features_hlt_std, hlt_mask, hlt_const, const_off, train_samples, max_count, tgt_mean, tgt_std, args.unmerge_target_mode)
    val_ds_un = UnmergeDataset(features_hlt_std, hlt_mask, hlt_const, const_off, val_samples, max_count, tgt_mean, tgt_std, args.unmerge_target_mode)
    test_ds_un = UnmergeDataset(features_hlt_std, hlt_mask, hlt_const, const_off, test_samples, max_count, tgt_mean, tgt_std, args.unmerge_target_mode)
    if args.unmerge_count_balanced:
        counts = train_ds_un.get_true_counts()
        max_c = max_count
        class_counts = np.bincount(counts, minlength=max_c + 1).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)
        weights = 1.0 / class_counts
        sample_weights = weights[counts]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader_un = DataLoader(train_ds_un, batch_size=BS_un, sampler=sampler, drop_last=True)
    else:
        train_loader_un = DataLoader(train_ds_un, batch_size=BS_un, shuffle=True, drop_last=True)
    val_loader_un = DataLoader(val_ds_un, batch_size=BS_un, shuffle=False)
    test_loader_un = DataLoader(test_ds_un, batch_size=BS_un, shuffle=False)

    unmerge_model = UnmergePredictor(
        input_dim=7,
        max_count=max_count,
        head_mode=args.unmerge_head_mode,
        parent_mode=args.unmerge_parent_mode,
        relpos_mode=args.unmerge_relpos_mode,
        local_attn_mode=args.unmerge_local_attn_mode,
        local_attn_radius=args.unmerge_local_attn_radius,
        local_attn_scale=args.unmerge_local_attn_scale,
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
            parent = batch["parent"].to(device)
            opt_u.zero_grad()
            preds = unmerge_model(x, mask, token_idx, pred_count)
            loss = set_chamfer_loss(preds, target, true_count)
            if CONFIG["unmerge_training"]["physics_weight"] > 0:
                pred_abs = preds_to_abs(preds, parent)
                loss = loss + CONFIG["unmerge_training"]["physics_weight"] * physics_loss(pred_abs, true_count, parent)
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
                parent = batch["parent"].to(device)
                preds = unmerge_model(x, mask, token_idx, pred_count)
                loss = set_chamfer_loss(preds, target, true_count)
                if CONFIG["unmerge_training"]["physics_weight"] > 0:
                    pred_abs = preds_to_abs(preds, parent)
                    loss = loss + CONFIG["unmerge_training"]["physics_weight"] * physics_loss(pred_abs, true_count, parent)
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
    unmerged_const, unmerged_mask, unmerged_flag = build_unmerged_dataset(
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
        args.unmerge_target_mode,
    )
    features_unmerged = compute_features(unmerged_const, unmerged_mask)
    features_unmerged_std = standardize(features_unmerged, unmerged_mask, feat_means, feat_stds)
    features_unmerged_flag = np.concatenate(
        [features_unmerged_std, unmerged_flag[..., None]], axis=-1
    ).astype(np.float32)

    train_ds_unmerged = JetDataset(features_unmerged_std[train_idx], unmerged_mask[train_idx], all_labels[train_idx])
    val_ds_unmerged = JetDataset(features_unmerged_std[val_idx], unmerged_mask[val_idx], all_labels[val_idx])
    test_ds_unmerged = JetDataset(features_unmerged_std[test_idx], unmerged_mask[test_idx], all_labels[test_idx])

    train_ds_unmerged_flag = JetDataset(features_unmerged_flag[train_idx], unmerged_mask[train_idx], all_labels[train_idx])
    val_ds_unmerged_flag = JetDataset(features_unmerged_flag[val_idx], unmerged_mask[val_idx], all_labels[val_idx])
    test_ds_unmerged_flag = JetDataset(features_unmerged_flag[test_idx], unmerged_mask[test_idx], all_labels[test_idx])

    train_ds_dual = DualViewJetDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_unmerged_std[train_idx],
        unmerged_mask[train_idx],
        all_labels[train_idx],
    )
    val_ds_dual = DualViewJetDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_unmerged_std[val_idx],
        unmerged_mask[val_idx],
        all_labels[val_idx],
    )
    test_ds_dual = DualViewJetDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_unmerged_std[test_idx],
        unmerged_mask[test_idx],
        all_labels[test_idx],
    )

    train_ds_dual_flag = DualViewJetDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_unmerged_flag[train_idx],
        unmerged_mask[train_idx],
        all_labels[train_idx],
    )
    val_ds_dual_flag = DualViewJetDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_unmerged_flag[val_idx],
        unmerged_mask[val_idx],
        all_labels[val_idx],
    )
    test_ds_dual_flag = DualViewJetDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_unmerged_flag[test_idx],
        unmerged_mask[test_idx],
        all_labels[test_idx],
    )

    train_loader_um = DataLoader(train_ds_unmerged, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_um = DataLoader(val_ds_unmerged, batch_size=BS, shuffle=False)
    test_loader_um = DataLoader(test_ds_unmerged, batch_size=BS, shuffle=False)

    train_loader_um_flag = DataLoader(train_ds_unmerged_flag, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_um_flag = DataLoader(val_ds_unmerged_flag, batch_size=BS, shuffle=False)
    test_loader_um_flag = DataLoader(test_ds_unmerged_flag, batch_size=BS, shuffle=False)

    train_loader_dual = DataLoader(train_ds_dual, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_dual = DataLoader(val_ds_dual, batch_size=BS, shuffle=False)
    test_loader_dual = DataLoader(test_ds_dual, batch_size=BS, shuffle=False)

    train_loader_dual_flag = DataLoader(train_ds_dual_flag, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_dual_flag = DataLoader(val_ds_dual_flag, batch_size=BS, shuffle=False)
    test_loader_dual_flag = DataLoader(test_ds_dual_flag, batch_size=BS, shuffle=False)

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

    # ------------------- Train unmerge-model classifier + MergeFlag ------------------- #
    print("\n" + "=" * 70)
    print("STEP 6B: UNMERGE MODEL + MERGEFLAG")
    print("=" * 70)
    unmerge_flag_cls = ParticleTransformer(input_dim=8, **CONFIG["model"]).to(device)
    opt_uf = torch.optim.AdamW(unmerge_flag_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_uf = get_scheduler(opt_uf, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_uf, best_state_uf, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="UnmergeFlag"):
        _, train_auc = train_classifier(unmerge_flag_cls, train_loader_um_flag, opt_uf, device)
        val_auc, _, _ = eval_classifier(unmerge_flag_cls, val_loader_um_flag, device)
        sch_uf.step()
        if val_auc > best_auc_uf:
            best_auc_uf = val_auc
            best_state_uf = {k: v.detach().cpu().clone() for k, v in unmerge_flag_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_uf:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping unmerge+flag classifier at epoch {ep+1}")
            break
    if best_state_uf is not None:
        unmerge_flag_cls.load_state_dict(best_state_uf)
    auc_unmerge_flag, preds_unmerge_flag, _ = eval_classifier(unmerge_flag_cls, test_loader_um_flag, device)

    # ------------------- Dual-view classifier (HLT + Unmerged) ------------------- #
    print("\n" + "=" * 70)
    print("STEP 6C: DUAL-VIEW (HLT + UNMERGED)")
    print("=" * 70)
    dual_cls = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **CONFIG["model"]).to(device)
    opt_dv = torch.optim.AdamW(dual_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_dv = get_scheduler(opt_dv, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_dv, best_state_dv, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="DualView"):
        _, train_auc = train_classifier_dual(dual_cls, train_loader_dual, opt_dv, device)
        val_auc, _, _ = eval_classifier_dual(dual_cls, val_loader_dual, device)
        sch_dv.step()
        if val_auc > best_auc_dv:
            best_auc_dv = val_auc
            best_state_dv = {k: v.detach().cpu().clone() for k, v in dual_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_dv:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping dual-view classifier at epoch {ep+1}")
            break
    if best_state_dv is not None:
        dual_cls.load_state_dict(best_state_dv)
    auc_dual, preds_dual, _ = eval_classifier_dual(dual_cls, test_loader_dual, device)

    # ------------------- Dual-view + MergeFlag classifier ------------------- #
    print("\n" + "=" * 70)
    print("STEP 6D: DUAL-VIEW + MERGEFLAG")
    print("=" * 70)
    dual_flag_cls = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=8, **CONFIG["model"]).to(device)
    opt_dvf = torch.optim.AdamW(dual_flag_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_dvf = get_scheduler(opt_dvf, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_dvf, best_state_dvf, no_improve = 0.0, None, 0
    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="DualViewFlag"):
        _, train_auc = train_classifier_dual(dual_flag_cls, train_loader_dual_flag, opt_dvf, device)
        val_auc, _, _ = eval_classifier_dual(dual_flag_cls, val_loader_dual_flag, device)
        sch_dvf.step()
        if val_auc > best_auc_dvf:
            best_auc_dvf = val_auc
            best_state_dvf = {k: v.detach().cpu().clone() for k, v in dual_flag_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_dvf:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping dual-view+flag classifier at epoch {ep+1}")
            break
    if best_state_dvf is not None:
        dual_flag_cls.load_state_dict(best_state_dvf)
    auc_dual_flag, preds_dual_flag, _ = eval_classifier_dual(dual_flag_cls, test_loader_dual_flag, device)

    # ------------------- Dual-view + KD ------------------- #
    print("\n" + "=" * 70)
    print("STEP 6E: DUAL-VIEW + KD")
    print("=" * 70)
    kd_train_ds_dv_nf = DualViewKDDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_unmerged_std[train_idx],
        unmerged_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        all_labels[train_idx],
    )
    kd_val_ds_dv_nf = DualViewKDDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_unmerged_std[val_idx],
        unmerged_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        all_labels[val_idx],
    )
    kd_test_ds_dv_nf = DualViewKDDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_unmerged_std[test_idx],
        unmerged_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        all_labels[test_idx],
    )
    kd_train_loader_dv_nf = DataLoader(kd_train_ds_dv_nf, batch_size=BS, shuffle=True, drop_last=True)
    kd_val_loader_dv_nf = DataLoader(kd_val_ds_dv_nf, batch_size=BS, shuffle=False)
    kd_test_loader_dv_nf = DataLoader(kd_test_ds_dv_nf, batch_size=BS, shuffle=False)

    kd_student_dv_nf = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **CONFIG["model"]).to(device)
    opt_kd_dv_nf = torch.optim.AdamW(kd_student_dv_nf.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_kd_dv_nf = get_scheduler(opt_kd_dv_nf, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

    best_auc_kd_dv_nf, best_state_kd_dv_nf, no_improve = 0.0, None, 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None

    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="DualView+KD"):
        current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
        kd_cfg_ep = dict(kd_cfg)
        kd_cfg_ep["alpha_kd"] = current_alpha

        train_loss, train_auc = train_kd_epoch_dual(
            kd_student_dv_nf, teacher, kd_train_loader_dv_nf, opt_kd_dv_nf, device, kd_cfg_ep
        )
        val_auc, _, _ = evaluate_kd_dual(kd_student_dv_nf, kd_val_loader_dv_nf, device)
        sch_kd_dv_nf.step()

        if not kd_active and kd_cfg["adaptive_alpha"]:
            val_loss = evaluate_bce_loss_dual(kd_student_dv_nf, kd_val_loader_dv_nf, device)
            if prev_val_loss is not None and abs(prev_val_loss - val_loss) < kd_cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = val_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
                print(f"Activating KD ramp at epoch {ep+1} (val_loss={val_loss:.4f})")

        if val_auc > best_auc_kd_dv_nf:
            best_auc_kd_dv_nf = val_auc
            best_state_kd_dv_nf = {k: v.detach().cpu().clone() for k, v in kd_student_dv_nf.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_kd_dv_nf:.4f} | alpha_kd={current_alpha:.2f}"
            )
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping dual-view KD student at epoch {ep+1}")
            break

    if best_state_kd_dv_nf is not None:
        kd_student_dv_nf.load_state_dict(best_state_kd_dv_nf)

    if kd_cfg["self_train"]:
        print("\nSTEP 6E2: SELF-TRAIN (pseudo-label fine-tune, dual-view)")
        opt_st_dv_nf = torch.optim.AdamW(kd_student_dv_nf.parameters(), lr=kd_cfg["self_train_lr"])
        best_auc_st_dv_nf = best_auc_kd_dv_nf
        no_improve = 0
        for ep in range(kd_cfg["self_train_epochs"]):
            st_loss = self_train_student_dual(kd_student_dv_nf, teacher, kd_train_loader_dv_nf, opt_st_dv_nf, device, kd_cfg)
            val_auc, _, _ = evaluate_kd_dual(kd_student_dv_nf, kd_val_loader_dv_nf, device)
            if val_auc > best_auc_st_dv_nf:
                best_auc_st_dv_nf = val_auc
                best_state_kd_dv_nf = {k: v.detach().cpu().clone() for k, v in kd_student_dv_nf.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if (ep + 1) % 2 == 0:
                print(f"Self ep {ep+1}: loss={st_loss:.4f}, val_auc={val_auc:.4f}, best={best_auc_st_dv_nf:.4f}")
            if no_improve >= kd_cfg["self_train_patience"]:
                break
        if best_state_kd_dv_nf is not None:
            kd_student_dv_nf.load_state_dict(best_state_kd_dv_nf)

    auc_dual_kd, preds_dual_kd, _ = evaluate_kd_dual(kd_student_dv_nf, kd_test_loader_dv_nf, device)

    # ------------------- Dual-view + MergeFlag + KD ------------------- #
    print("\n" + "=" * 70)
    print("STEP 6F: DUAL-VIEW + MERGEFLAG + KD")
    print("=" * 70)
    kd_train_ds_dv = DualViewKDDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_unmerged_flag[train_idx],
        unmerged_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        all_labels[train_idx],
    )
    kd_val_ds_dv = DualViewKDDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_unmerged_flag[val_idx],
        unmerged_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        all_labels[val_idx],
    )
    kd_test_ds_dv = DualViewKDDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_unmerged_flag[test_idx],
        unmerged_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        all_labels[test_idx],
    )
    kd_train_loader_dv = DataLoader(kd_train_ds_dv, batch_size=BS, shuffle=True, drop_last=True)
    kd_val_loader_dv = DataLoader(kd_val_ds_dv, batch_size=BS, shuffle=False)
    kd_test_loader_dv = DataLoader(kd_test_ds_dv, batch_size=BS, shuffle=False)

    kd_student_dv = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=8, **CONFIG["model"]).to(device)
    opt_kd_dv = torch.optim.AdamW(kd_student_dv.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_kd_dv = get_scheduler(opt_kd_dv, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

    best_auc_kd_dv, best_state_kd_dv, no_improve = 0.0, None, 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None

    for ep in tqdm(range(CONFIG["training"]["epochs"]), desc="DualView+Flag+KD"):
        current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
        kd_cfg_ep = dict(kd_cfg)
        kd_cfg_ep["alpha_kd"] = current_alpha

        train_loss, train_auc = train_kd_epoch_dual(kd_student_dv, teacher, kd_train_loader_dv, opt_kd_dv, device, kd_cfg_ep)
        val_auc, _, _ = evaluate_kd_dual(kd_student_dv, kd_val_loader_dv, device)
        sch_kd_dv.step()

        if not kd_active and kd_cfg["adaptive_alpha"]:
            val_loss = evaluate_bce_loss_dual(kd_student_dv, kd_val_loader_dv, device)
            if prev_val_loss is not None and abs(prev_val_loss - val_loss) < kd_cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = val_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
                print(f"Activating KD ramp at epoch {ep+1} (val_loss={val_loss:.4f})")

        if val_auc > best_auc_kd_dv:
            best_auc_kd_dv = val_auc
            best_state_kd_dv = {k: v.detach().cpu().clone() for k, v in kd_student_dv.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_kd_dv:.4f} | alpha_kd={current_alpha:.2f}"
            )
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping dual-view+flag KD student at epoch {ep+1}")
            break

    if best_state_kd_dv is not None:
        kd_student_dv.load_state_dict(best_state_kd_dv)

    if kd_cfg["self_train"]:
        print("\nSTEP 6G: SELF-TRAIN (pseudo-label fine-tune, dual-view merge-flag)")
        opt_st_dv = torch.optim.AdamW(kd_student_dv.parameters(), lr=kd_cfg["self_train_lr"])
        best_auc_st_dv = best_auc_kd_dv
        no_improve = 0
        for ep in range(kd_cfg["self_train_epochs"]):
            st_loss = self_train_student_dual(kd_student_dv, teacher, kd_train_loader_dv, opt_st_dv, device, kd_cfg)
            val_auc, _, _ = evaluate_kd_dual(kd_student_dv, kd_val_loader_dv, device)
            if val_auc > best_auc_st_dv:
                best_auc_st_dv = val_auc
                best_state_kd_dv = {k: v.detach().cpu().clone() for k, v in kd_student_dv.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if (ep + 1) % 2 == 0:
                print(f"Self ep {ep+1}: loss={st_loss:.4f}, val_auc={val_auc:.4f}, best={best_auc_st_dv:.4f}")
            if no_improve >= kd_cfg["self_train_patience"]:
                break
        if best_state_kd_dv is not None:
            kd_student_dv.load_state_dict(best_state_kd_dv)

    auc_dual_flag_kd, preds_dual_flag_kd, _ = evaluate_kd_dual(kd_student_dv, kd_test_loader_dv, device)

    # ------------------- Final evaluation ------------------- #
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"Unmerge Model    AUC: {auc_unmerge:.4f}")
    print(f"Unmerge+MF       AUC: {auc_unmerge_flag:.4f}")
    print(f"Dual-View        AUC: {auc_dual:.4f}")
    print(f"Dual-View+MF     AUC: {auc_dual_flag:.4f}")
    print(f"Dual-View+KD     AUC: {auc_dual_kd:.4f}")
    print(f"Dual-View+MF+KD  AUC: {auc_dual_flag_kd:.4f}")

    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_u, tpr_u, _ = roc_curve(labs, preds_unmerge)
    fpr_uf, tpr_uf, _ = roc_curve(labs, preds_unmerge_flag)
    fpr_dv, tpr_dv, _ = roc_curve(labs, preds_dual)
    fpr_dvf, tpr_dvf, _ = roc_curve(labs, preds_dual_flag)
    fpr_dv_k, tpr_dv_k, _ = roc_curve(labs, preds_dual_kd)
    fpr_dvf_k, tpr_dvf_k, _ = roc_curve(labs, preds_dual_flag_kd)

    def plot_roc(lines, out_name):
        min_fpr = max(float(args.roc_fpr_min), 1e-8)
        plt.figure(figsize=(8, 6))
        for tpr, fpr, style, label, color in lines:
            fpr_plot = np.clip(fpr, min_fpr, 1.0)
            plt.plot(tpr, fpr_plot, style, label=label, color=color, linewidth=2)
        plt.ylabel("False Positive Rate", fontsize=12)
        plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
        plt.yscale("log")
        plt.ylim(min_fpr, 1.0)
        plt.xlim(0.0, 1.0)
        plt.legend(fontsize=12, frameon=False)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_root / out_name, dpi=300)
        plt.close()

    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_u, fpr_u, ":", f"Unmerge (AUC={auc_unmerge:.3f})", "forestgreen"),
            (tpr_uf, fpr_uf, "-.", f"Unmerge+MF (AUC={auc_unmerge_flag:.3f})", "darkorange"),
            (tpr_dv, fpr_dv, "-", f"Dual-View (AUC={auc_dual:.3f})", "teal"),
            (tpr_dvf, fpr_dvf, "--", f"DualView+MF (AUC={auc_dual_flag:.3f})", "orchid"),
            (tpr_dv_k, fpr_dv_k, ":", f"DualView+KD (AUC={auc_dual_kd:.3f})", "slateblue"),
            (tpr_dvf_k, fpr_dvf_k, "-.", f"DualView+MF+KD (AUC={auc_dual_flag_kd:.3f})", "darkslateblue"),
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
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_u, fpr_u, ":", f"Unmerge (AUC={auc_unmerge:.3f})", "forestgreen"),
        ],
        "results_teacher_baseline_unmerge.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_uf, fpr_uf, "-.", f"Unmerge+MF (AUC={auc_unmerge_flag:.3f})", "darkorange"),
        ],
        "results_teacher_baseline_unmerge_flag.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_dv, fpr_dv, "-", f"Dual-View (AUC={auc_dual:.3f})", "teal"),
        ],
        "results_teacher_baseline_dualview.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_dv_k, fpr_dv_k, ":", f"DualView+KD (AUC={auc_dual_kd:.3f})", "slateblue"),
        ],
        "results_teacher_baseline_dualview_kd.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_dvf, fpr_dvf, "--", f"DualView+MF (AUC={auc_dual_flag:.3f})", "orchid"),
        ],
        "results_teacher_baseline_dualview_flag.png",
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_dvf_k, fpr_dvf_k, "-.", f"DualView+MF+KD (AUC={auc_dual_flag_kd:.3f})", "darkslateblue"),
        ],
        "results_teacher_baseline_dualview_flag_kd.png",
    )

    # Jet pT response/resolution on test split:
    # truth = offline jet pT, reco = HLT or corrected-HLT (unmerged) jet pT.
    pt_truth_test = compute_jet_pt(const_off[test_idx], masks_off[test_idx])
    pt_hlt_test = compute_jet_pt(hlt_const[test_idx], hlt_mask[test_idx])
    pt_unmerged_test = compute_jet_pt(unmerged_const[test_idx], unmerged_mask[test_idx])

    pt_edges = build_pt_edges(pt_truth_test, args.response_n_bins)
    rr_hlt = jet_response_resolution(pt_truth_test, pt_hlt_test, pt_edges, args.response_min_count)
    rr_unmerged = jet_response_resolution(pt_truth_test, pt_unmerged_test, pt_edges, args.response_min_count)

    plot_response_resolution(
        rr_hlt,
        rr_unmerged,
        "HLT (reco)",
        "Corrected HLT / Unmerged (reco)",
        save_root / "jet_response_resolution.png",
    )

    rr_hlt_map = {(r["pt_low"], r["pt_high"]): r for r in rr_hlt}
    rr_unmerged_map = {(r["pt_low"], r["pt_high"]): r for r in rr_unmerged}
    keys = sorted(set(rr_hlt_map.keys()) & set(rr_unmerged_map.keys()))
    rr_hlt_common = [rr_hlt_map[k] for k in keys]
    rr_unmerged_common = [rr_unmerged_map[k] for k in keys]
    print("\nJet pT response/resolution by truth pT bin (test split):")
    print("  pT_low - pT_high | N | HLT resp | HLT reso | Corrected resp | Corrected reso")
    for h, u in zip(rr_hlt_common, rr_unmerged_common):
        print(
            f"  {h['pt_low']:.1f} - {h['pt_high']:.1f} | {h['count']:5d} | "
            f"{h['response']:.4f} | {h['resolution']:.4f} | "
            f"{u['response']:.4f} | {u['resolution']:.4f}"
        )

    response_payload = {
        "definition": {
            "response": "mean(pT_reco / pT_truth)",
            "resolution": "std(pT_reco / pT_truth)",
            "truth": "offline jet pT (vector sum of constituents)",
            "reco_hlt": "HLT jet pT",
            "reco_corrected_hlt": "unmerged/corrected HLT jet pT",
            "split": "test",
        },
        "pt_bins_edges": pt_edges.tolist(),
        "n_bins_requested": int(args.response_n_bins),
        "min_count_per_bin": int(args.response_min_count),
        "hlt_vs_offline": rr_hlt,
        "corrected_hlt_vs_offline": rr_unmerged,
        "common_bins_for_comparison": [list(k) for k in keys],
    }
    with open(save_root / "jet_response_resolution.json", "w", encoding="utf-8") as f:
        json.dump(response_payload, f, indent=2)

    def rr_field(records, key):
        return np.array([r[key] for r in records], dtype=np.float64)

    np.savez(
        save_root / "results.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_unmerge=auc_unmerge,
        auc_unmerge_flag=auc_unmerge_flag,
        auc_dual=auc_dual,
        auc_dual_flag=auc_dual_flag,
        auc_dual_kd=auc_dual_kd,
        auc_dual_flag_kd=auc_dual_flag_kd,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_unmerge=fpr_u,
        tpr_unmerge=tpr_u,
        fpr_unmerge_flag=fpr_uf,
        tpr_unmerge_flag=tpr_uf,
        fpr_dual=fpr_dv,
        tpr_dual=tpr_dv,
        fpr_dual_flag=fpr_dvf,
        tpr_dual_flag=tpr_dvf,
        fpr_dual_kd=fpr_dv_k,
        tpr_dual_kd=tpr_dv_k,
        fpr_dual_flag_kd=fpr_dvf_k,
        tpr_dual_flag_kd=tpr_dvf_k,
        jet_response_pt_low=rr_field(rr_hlt_common, "pt_low"),
        jet_response_pt_high=rr_field(rr_hlt_common, "pt_high"),
        jet_response_count=rr_field(rr_hlt_common, "count"),
        jet_response_hlt_mean=rr_field(rr_hlt_common, "response"),
        jet_response_hlt_std=rr_field(rr_hlt_common, "resolution"),
        jet_response_corrected_mean=rr_field(rr_unmerged_common, "response"),
        jet_response_corrected_std=rr_field(rr_unmerged_common, "resolution"),
        unmerge_test_loss=test_loss,
        max_merge_count=max_count,
    )

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": auc_teacher}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": auc_baseline}, save_root / "baseline.pt")
        torch.save({"model": count_model.state_dict(), "acc": best_acc}, save_root / "merge_count.pt")
        torch.save({"model": unmerge_model.state_dict(), "loss": best_val_loss}, save_root / "unmerge_predictor.pt")
        torch.save({"model": unmerge_cls.state_dict(), "auc": auc_unmerge}, save_root / "unmerge_classifier.pt")
        torch.save({"model": unmerge_flag_cls.state_dict(), "auc": auc_unmerge_flag}, save_root / "unmerge_mergeflag_classifier.pt")
        torch.save({"model": dual_cls.state_dict(), "auc": auc_dual}, save_root / "dual_view_classifier.pt")
        torch.save({"model": dual_flag_cls.state_dict(), "auc": auc_dual_flag}, save_root / "dual_view_mergeflag_classifier.pt")
        torch.save({"model": kd_student_dv_nf.state_dict(), "auc": auc_dual_kd}, save_root / "dual_view_kd.pt")
        torch.save({"model": kd_student_dv.state_dict(), "auc": auc_dual_flag_kd}, save_root / "dual_view_mergeflag_kd.pt")

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()

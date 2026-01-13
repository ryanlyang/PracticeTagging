
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer Training Suite with Realistic HLT Effects

This is a drop-in replacement script that:
  - Loads your dataset via utils.load_from_files over a directory of .h5 files
  - Builds paired OFFLINE and HLT views per jet (same jets, different recon)
  - Computes professor-style 7 relative features + standardizes using OFFLINE(train) stats
  - Trains 4 models:

    (1) Teacher (OFFLINE only)
        - trained and validated on OFFLINE view

    (2) Baseline Student (HLT only)
        - trained and validated on HLT view

    (3) Mixed-View Model (randomly mixed OFFLINE/HLT during training)
        - trained on a random mix of OFFLINE and HLT inputs, but validated on HLT view only
        - idea: learn a representation robust to both views without explicit consistency

    (4) Consistency Model (your idea, best implementation)
        - single model used at test time on HLT
        - training uses:
            L = BCE(logits_hlt, y)
              + lambda_logit * ConsistencyOnLogits(logits_hlt vs stopgrad(logits_off))
              + lambda_embed * ConsistencyOnEmbedding(z_hlt vs stopgrad(z_off))
          plus optional confidence gating and ramp-up schedule

  - Evaluates:
      Teacher on OFFLINE test view
      All others on HLT test view

  - Saves:
      test_split/test_features_and_masks.npz
      checkpoints/<run_name>/{teacher.pt, baseline.pt, mixed.pt, consistency.pt}
      checkpoints/<run_name>/results.npz, results.png

Assumption about utils.load_from_files output:
  all_data: (N, max_constits, 3) with columns [eta, phi, pt]
If your columns differ, edit ETA_IDX/PHI_IDX/PT_IDX below.
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


# ----------------------------- Default config (professor-style) ----------------------------- #
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
    const: (n_jets, max_part, 4) with columns [pt, eta, phi, E]
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
        merge_radius = float(hcfg["merge_radius"])

        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            if len(valid_idx) < 2:
                continue

            to_remove = set()

            for i in range(len(valid_idx)):
                idx_i = int(valid_idx[i])
                if idx_i in to_remove:
                    continue

                for j in range(i + 1, len(valid_idx)):
                    idx_j = int(valid_idx[j])
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
        n_lost_eff = int(lost.sum())

    # Final cleanup
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0

    n_final = int(hlt_mask.sum())
    retention = 100.0 * n_final / max(n_initial, 1)

    print("\nHLT Simulation Statistics:")
    print(f"  Offline particles: {n_initial:,}")
    print(f"  Lost to pT threshold ({hcfg['pt_threshold_hlt']}): {n_lost_threshold:,} ({100*n_lost_threshold/max(n_initial,1):.1f}%)")
    print(f"  Lost to merging (dR<{hcfg['merge_radius']}): {n_merged:,} ({100*n_merged/max(n_initial,1):.1f}%)")
    print(f"  Lost to efficiency: {n_lost_eff:,} ({100*n_lost_eff/max(n_initial,1):.1f}%)")
    print(f"  HLT particles: {n_final:,} ({retention:.1f}% of offline)")

    return hlt.astype(np.float32), hlt_mask


# ----------------------------- Feature computation (professor style) ----------------------------- #
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

    features = np.stack(
        [delta_eta, delta_phi, log_pt, log_E, log_pt_rel, log_E_rel, delta_R], axis=-1
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


# ----------------------------- Model (professor architecture, with repr output) ----------------------------- #
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
    """
    Adds optional return_repr=True to return the pooled embedding z
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

    def forward(self, x, mask, return_attention=False, return_repr=False):
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

        z = self.norm(pooled.squeeze(1))          # (B, embed_dim)
        logits = self.classifier(z)               # (B, 1)

        out = (logits,)
        if return_attention:
            out = out + (attn_weights.squeeze(1),)
        if return_repr:
            out = out + (z,)
        if len(out) == 1:
            return out[0]
        return out


# ----------------------------- Schedules ----------------------------- #
def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        denom = max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / denom))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def linear_ramp(epoch, start_epoch, end_epoch):
    """
    Returns 0 before start_epoch, 1 after end_epoch, linear in between.
    """
    if epoch < start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    return (epoch - start_epoch) / max(end_epoch - start_epoch, 1)


# ----------------------------- Metrics helpers ----------------------------- #
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


def train_standard_epoch(model, loader, opt, device, feat_key, mask_key):
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

    return total_loss / max(len(preds), 1), roc_auc_score(labs, preds)


def train_mixed_epoch(model, loader, opt, device, mix_off_prob=0.5):
    """
    One model, but for each batch we choose OFFLINE or HLT view at random.
    Validation is still done on HLT.
    """
    model.train()
    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        use_off = (random.random() < mix_off_prob)
        feat_key = "off" if use_off else "hlt"
        mask_key = "mask_off" if use_off else "mask_hlt"

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

    return total_loss / max(len(preds), 1), roc_auc_score(labs, preds)


# ----------------------------- Consistency losses (your idea, best version) ----------------------------- #
def binary_logit_consistency(student_logits, target_logits_detached, T):
    """
    Teacher-free logit consistency:
      BCE(sigmoid(z_s/T), sigmoid(z_t/T)) * T^2
    Here target_logits_detached should already be stopgrad/ detached.
    """
    s = student_logits
    t = target_logits_detached
    s_soft = torch.sigmoid(s / T)
    t_soft = torch.sigmoid(t / T)
    return F.binary_cross_entropy(s_soft, t_soft) * (T ** 2)


def embedding_consistency(z_hlt, z_off_detached, mode="cosine"):
    """
    z_*: (B, D)
    """
    if mode == "mse":
        return F.mse_loss(z_hlt, z_off_detached)
    # cosine distance (1 - cosine similarity)
    z1 = F.normalize(z_hlt, dim=1)
    z2 = F.normalize(z_off_detached, dim=1)
    return (1.0 - (z1 * z2).sum(dim=1)).mean()


def confidence_gate_from_logits(logits_off_detached, gamma=2.0):
    """
    Returns per-sample weights in [0,1], higher when offline branch is confident.
    For binary: confidence = 2*|p-0.5|
    """
    p = torch.sigmoid(logits_off_detached)
    conf = (p - 0.5).abs() * 2.0
    conf = conf.clamp(0.0, 1.0)
    if gamma is None or gamma == 1.0:
        return conf
    return conf.pow(gamma)


def train_consistency_epoch(
    model,
    loader,
    opt,
    device,
    T=4.0,
    lambda_logit=0.5,
    lambda_embed=0.5,
    embed_mode="cosine",
    gate_gamma=2.0,
    ramp_factor=1.0,
):
    """
    Best-practice implementation choices:
      - hard supervision ONLY on HLT branch
      - offline branch provides a stop-grad target (asymmetric)
      - optional confidence gating based on offline confidence
      - ramp_factor lets you ramp consistency terms over epochs
    """
    model.train()
    total_loss = 0.0
    preds, labs = [], []

    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        x_off = batch["off"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)

        opt.zero_grad()

        # HLT forward (trainable path)
        s_logits, _, z_hlt = model(x_hlt, m_hlt, return_attention=True, return_repr=True)
        s_logits = s_logits.squeeze(1)

        # OFF forward (target path)
        with torch.no_grad():
            off_logits, _, z_off = model(x_off, m_off, return_attention=True, return_repr=True)
            off_logits = off_logits.squeeze(1)
            z_off = z_off.detach()
            off_logits = off_logits.detach()

        # Hard label loss on HLT only
        hard = F.binary_cross_entropy_with_logits(s_logits, y)

        # Gating weight per sample
        gate = confidence_gate_from_logits(off_logits, gamma=gate_gamma)  # (B,)
        gate_mean = gate.mean().clamp(min=1e-6)

        # Consistency on logits (KD-style)
        if lambda_logit > 0:
            logit_cons = binary_logit_consistency(s_logits, off_logits, T)
        else:
            logit_cons = torch.zeros((), device=device)

        # Consistency on embedding (pooled representation)
        if lambda_embed > 0:
            emb_cons = embedding_consistency(z_hlt, z_off, mode=embed_mode)
        else:
            emb_cons = torch.zeros((), device=device)

        # Apply gating to the consistency terms.
        # logit_cons and emb_cons are scalars, so we apply mean gate as a global stabilizer.
        cons = (lambda_logit * logit_cons + lambda_embed * emb_cons) * (gate_mean * ramp_factor)

        loss = hard + cons
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / max(len(preds), 1), roc_auc_score(labs, preds)


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="./data", help="Directory containing *.h5 files")
    parser.add_argument("--n_train_jets", type=int, default=15000)
    parser.add_argument("--max_constits", type=int, default=80)

    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "transformer_suite"))
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")

    # Optional: skip training if you want to load pretrained weights
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--baseline_checkpoint", type=str, default=None)
    parser.add_argument("--mixed_checkpoint", type=str, default=None)
    parser.add_argument("--consistency_checkpoint", type=str, default=None)

    parser.add_argument("--skip_save_models", action="store_true")

    # Mixed model knobs
    parser.add_argument("--mix_off_prob", type=float, default=0.5, help="Prob a training batch uses OFFLINE for the mixed model")

    # Consistency model knobs (recommended defaults)
    parser.add_argument("--cons_T", type=float, default=4.0)
    parser.add_argument("--lambda_logit", type=float, default=0.7)
    parser.add_argument("--lambda_embed", type=float, default=0.5)
    parser.add_argument("--embed_mode", type=str, default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--gate_gamma", type=float, default=2.0)

    # Consistency ramp schedule
    parser.add_argument("--cons_ramp_start", type=int, default=2, help="Epoch index (0-based) to start ramping consistency")
    parser.add_argument("--cons_ramp_end", type=int, default=12, help="Epoch index (0-based) to finish ramping consistency")

    args = parser.parse_args()

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

    # ------------------- Load your dataset ------------------- #
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

    # ------------------- Convert your data -> (pt, eta, phi, E) ------------------- #
    eta = all_data[:, :, ETA_IDX].astype(np.float32)
    phi = all_data[:, :, PHI_IDX].astype(np.float32)
    pt = all_data[:, :, PT_IDX].astype(np.float32)

    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    constituents_raw = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    print(f"Avg particles per jet (raw mask): {mask_raw.sum(axis=1).mean():.1f}")

    # ------------------- Apply HLT effects ------------------- #
    print("Applying HLT effects...")
    constituents_hlt, masks_hlt = apply_hlt_effects(constituents_raw, mask_raw, CONFIG, seed=RANDOM_SEED)

    # ------------------- Apply offline threshold ------------------- #
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
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx]
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # ------------------- Standardize using OFFLINE train stats ------------------- #
    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = standardize(features_hlt, masks_hlt, feat_means, feat_stds)

    print(f"Final NaN check: Offline={np.isnan(features_off_std).sum()}, HLT={np.isnan(features_hlt_std).sum()}")

    # ------------------- Save test split artifacts ------------------- #
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
    train_ds = JetDataset(
        features_off_std[train_idx], features_hlt_std[train_idx],
        all_labels[train_idx], masks_off[train_idx], masks_hlt[train_idx]
    )
    val_ds = JetDataset(
        features_off_std[val_idx], features_hlt_std[val_idx],
        all_labels[val_idx], masks_off[val_idx], masks_hlt[val_idx]
    )
    test_ds = JetDataset(
        features_off_std[test_idx], features_hlt_std[test_idx],
        all_labels[test_idx], masks_off[test_idx], masks_hlt[test_idx]
    )

    BS = int(CONFIG["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)

    # ------------------- Checkpoint paths ------------------- #
    teacher_path = save_dir / "teacher.pt"
    baseline_path = save_dir / "baseline.pt"
    mixed_path = save_dir / "mixed.pt"
    consistency_path = save_dir / "consistency.pt"

    # ------------------- Common training config ------------------- #
    epochs = int(CONFIG["training"]["epochs"])
    lr = float(CONFIG["training"]["lr"])
    wd = float(CONFIG["training"]["weight_decay"])
    warmup = int(CONFIG["training"]["warmup_epochs"])
    patience = int(CONFIG["training"]["patience"])

    # =======================================================================
    # 1) TEACHER (OFFLINE)
    # =======================================================================
    print("\n" + "=" * 80)
    print("MODEL 1: TEACHER (train OFFLINE, eval OFFLINE)")
    print("=" * 80)

    teacher = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
    history_teacher = []
    best_teacher_auc = 0.0

    if args.teacher_checkpoint is not None:
        ckpt = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(ckpt["model"])
        best_teacher_auc = ckpt.get("auc", 0.0)
        history_teacher = ckpt.get("history", [])
        print(f"Loaded teacher from {args.teacher_checkpoint} (auc={best_teacher_auc:.4f})")
    else:
        opt = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=wd)
        sch = get_scheduler(opt, warmup, epochs)

        best_state = None
        no_improve = 0

        for ep in tqdm(range(epochs), desc="Teacher"):
            tr_loss, tr_auc = train_standard_epoch(teacher, train_loader, opt, device, "off", "mask_off")
            va_auc, _, _ = evaluate(teacher, val_loader, device, "off", "mask_off")
            sch.step()

            history_teacher.append((ep + 1, tr_loss, tr_auc, va_auc))

            if va_auc > best_teacher_auc:
                best_teacher_auc = va_auc
                best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_teacher_auc:.4f}")

            if no_improve >= patience:
                print(f"Early stopping teacher at epoch {ep+1}")
                break

        if best_state is not None:
            teacher.load_state_dict(best_state)

        if not args.skip_save_models:
            torch.save({"model": teacher.state_dict(), "auc": best_teacher_auc, "history": history_teacher}, teacher_path)
            print(f"Saved teacher: {teacher_path} (best val AUC={best_teacher_auc:.4f})")
        else:
            print(f"Skipped saving teacher (best val AUC={best_teacher_auc:.4f})")

    # =======================================================================
    # 2) BASELINE (HLT-only)
    # =======================================================================
    print("\n" + "=" * 80)
    print("MODEL 2: BASELINE (train HLT, eval HLT)")
    print("=" * 80)

    baseline = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
    history_baseline = []
    best_baseline_auc = 0.0

    if args.baseline_checkpoint is not None:
        ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        baseline.load_state_dict(ckpt["model"])
        best_baseline_auc = ckpt.get("auc", 0.0)
        history_baseline = ckpt.get("history", [])
        print(f"Loaded baseline from {args.baseline_checkpoint} (auc={best_baseline_auc:.4f})")
    else:
        opt = torch.optim.AdamW(baseline.parameters(), lr=lr, weight_decay=wd)
        sch = get_scheduler(opt, warmup, epochs)

        best_state = None
        no_improve = 0

        for ep in tqdm(range(epochs), desc="Baseline"):
            tr_loss, tr_auc = train_standard_epoch(baseline, train_loader, opt, device, "hlt", "mask_hlt")
            va_auc, _, _ = evaluate(baseline, val_loader, device, "hlt", "mask_hlt")
            sch.step()

            history_baseline.append((ep + 1, tr_loss, tr_auc, va_auc))

            if va_auc > best_baseline_auc:
                best_baseline_auc = va_auc
                best_state = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_baseline_auc:.4f}")

            if no_improve >= (patience + 5):
                print(f"Early stopping baseline at epoch {ep+1}")
                break

        if best_state is not None:
            baseline.load_state_dict(best_state)

        if not args.skip_save_models:
            torch.save({"model": baseline.state_dict(), "auc": best_baseline_auc, "history": history_baseline}, baseline_path)
            print(f"Saved baseline: {baseline_path} (best val AUC={best_baseline_auc:.4f})")
        else:
            print(f"Skipped saving baseline (best val AUC={best_baseline_auc:.4f})")

    # =======================================================================
    # 3) MIXED-VIEW MODEL (random OFF/HIT batches, eval HLT)
    # =======================================================================
    print("\n" + "=" * 80)
    print("MODEL 3: MIXED-VIEW (train random OFF/HLT, eval HLT)")
    print("=" * 80)

    mixed = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
    history_mixed = []
    best_mixed_auc = 0.0

    if args.mixed_checkpoint is not None:
        ckpt = torch.load(args.mixed_checkpoint, map_location=device)
        mixed.load_state_dict(ckpt["model"])
        best_mixed_auc = ckpt.get("auc", 0.0)
        history_mixed = ckpt.get("history", [])
        print(f"Loaded mixed from {args.mixed_checkpoint} (auc={best_mixed_auc:.4f})")
    else:
        opt = torch.optim.AdamW(mixed.parameters(), lr=lr, weight_decay=wd)
        sch = get_scheduler(opt, warmup, epochs)

        best_state = None
        no_improve = 0

        for ep in tqdm(range(epochs), desc="Mixed"):
            tr_loss, tr_auc = train_mixed_epoch(mixed, train_loader, opt, device, mix_off_prob=float(args.mix_off_prob))
            va_auc, _, _ = evaluate(mixed, val_loader, device, "hlt", "mask_hlt")
            sch.step()

            history_mixed.append((ep + 1, tr_loss, tr_auc, va_auc))

            if va_auc > best_mixed_auc:
                best_mixed_auc = va_auc
                best_state = {k: v.detach().cpu().clone() for k, v in mixed.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_mixed_auc:.4f}")

            if no_improve >= (patience + 5):
                print(f"Early stopping mixed at epoch {ep+1}")
                break

        if best_state is not None:
            mixed.load_state_dict(best_state)

        if not args.skip_save_models:
            torch.save({"model": mixed.state_dict(), "auc": best_mixed_auc, "history": history_mixed}, mixed_path)
            print(f"Saved mixed: {mixed_path} (best val AUC={best_mixed_auc:.4f})")
        else:
            print(f"Skipped saving mixed (best val AUC={best_mixed_auc:.4f})")

    # =======================================================================
    # 4) CONSISTENCY MODEL (your idea, best version, eval HLT)
    # =======================================================================
    print("\n" + "=" * 80)
    print("MODEL 4: CONSISTENCY (train HLT + OFF alignment, eval HLT)")
    print("=" * 80)
    print("Consistency settings:")
    print(f"  T={args.cons_T}")
    print(f"  lambda_logit={args.lambda_logit}")
    print(f"  lambda_embed={args.lambda_embed} (mode={args.embed_mode})")
    print(f"  gate_gamma={args.gate_gamma}")
    print(f"  ramp: start={args.cons_ramp_start}, end={args.cons_ramp_end}")

    consistency = ParticleTransformerKD(input_dim=7, **CONFIG["model"]).to(device)
    history_cons = []
    best_cons_auc = 0.0

    if args.consistency_checkpoint is not None:
        ckpt = torch.load(args.consistency_checkpoint, map_location=device)
        consistency.load_state_dict(ckpt["model"])
        best_cons_auc = ckpt.get("auc", 0.0)
        history_cons = ckpt.get("history", [])
        print(f"Loaded consistency model from {args.consistency_checkpoint} (auc={best_cons_auc:.4f})")
    else:
        opt = torch.optim.AdamW(consistency.parameters(), lr=lr, weight_decay=wd)
        sch = get_scheduler(opt, warmup, epochs)

        best_state = None
        no_improve = 0

        for ep in tqdm(range(epochs), desc="Consistency"):
            ramp = linear_ramp(ep, int(args.cons_ramp_start), int(args.cons_ramp_end))

            tr_loss, tr_auc = train_consistency_epoch(
                consistency,
                train_loader,
                opt,
                device,
                T=float(args.cons_T),
                lambda_logit=float(args.lambda_logit),
                lambda_embed=float(args.lambda_embed),
                embed_mode=str(args.embed_mode),
                gate_gamma=float(args.gate_gamma),
                ramp_factor=float(ramp),
            )
            va_auc, _, _ = evaluate(consistency, val_loader, device, "hlt", "mask_hlt")
            sch.step()

            history_cons.append((ep + 1, tr_loss, tr_auc, va_auc, ramp))

            if va_auc > best_cons_auc:
                best_cons_auc = va_auc
                best_state = {k: v.detach().cpu().clone() for k, v in consistency.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(f"Ep {ep+1}: ramp={ramp:.2f}, train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_cons_auc:.4f}")

            if no_improve >= (patience + 5):
                print(f"Early stopping consistency at epoch {ep+1}")
                break

        if best_state is not None:
            consistency.load_state_dict(best_state)

        if not args.skip_save_models:
            torch.save({"model": consistency.state_dict(), "auc": best_cons_auc, "history": history_cons}, consistency_path)
            print(f"Saved consistency: {consistency_path} (best val AUC={best_cons_auc:.4f})")
        else:
            print(f"Skipped saving consistency (best val AUC={best_cons_auc:.4f})")

    # =======================================================================
    # FINAL TEST EVALUATION + PLOT
    # =======================================================================
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    auc_teacher, preds_teacher, labs = evaluate(teacher, test_loader, device, "off", "mask_off")
    auc_baseline, preds_baseline, _ = evaluate(baseline, test_loader, device, "hlt", "mask_hlt")
    auc_mixed, preds_mixed, _ = evaluate(mixed, test_loader, device, "hlt", "mask_hlt")
    auc_cons, preds_cons, _ = evaluate(consistency, test_loader, device, "hlt", "mask_hlt")

    print(f"\n{'Model':<40} {'AUC':>10}")
    print("-" * 55)
    print(f"{'Teacher (Offline test)':<40} {auc_teacher:>10.4f}")
    print(f"{'Baseline (HLT test)':<40} {auc_baseline:>10.4f}")
    print(f"{'Mixed (HLT test)':<40} {auc_mixed:>10.4f}")
    print(f"{'Consistency (HLT test)':<40} {auc_cons:>10.4f}")
    print("-" * 55)

    # Curves
    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_m, tpr_m, _ = roc_curve(labs, preds_mixed)
    fpr_c, tpr_c, _ = roc_curve(labs, preds_cons)

    # Background rejection at 50% signal efficiency for the HLT-tested models
    wp = 0.5
    idx_b = np.argmax(tpr_b >= wp)
    idx_m = np.argmax(tpr_m >= wp)
    idx_c = np.argmax(tpr_c >= wp)
    br_baseline = 1.0 / fpr_b[idx_b] if fpr_b[idx_b] > 0 else 0.0
    br_mixed = 1.0 / fpr_m[idx_m] if fpr_m[idx_m] > 0 else 0.0
    br_cons = 1.0 / fpr_c[idx_c] if fpr_c[idx_c] > 0 else 0.0

    print(f"\nBackground Rejection @ {wp*100:.0f}% signal efficiency (HLT-tested):")
    print(f"  Baseline:    {br_baseline:.2f}")
    print(f"  Mixed:       {br_mixed:.2f}")
    print(f"  Consistency: {br_cons:.2f}")

    # Save results
    np.savez(
        save_dir / "results.npz",
        labs=labs,
        preds_teacher=preds_teacher,
        preds_baseline=preds_baseline,
        preds_mixed=preds_mixed,
        preds_consistency=preds_cons,
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_mixed=auc_mixed,
        auc_consistency=auc_cons,
        fpr_teacher=fpr_t, tpr_teacher=tpr_t,
        fpr_baseline=fpr_b, tpr_baseline=tpr_b,
        fpr_mixed=fpr_m, tpr_mixed=tpr_m,
        fpr_consistency=fpr_c, tpr_consistency=tpr_c,
        br_baseline=br_baseline,
        br_mixed=br_mixed,
        br_consistency=br_cons,
    )

    # Plot (keep your preferred axes: TPR on x, FPR on y)
    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-",  label=f"Teacher OFF test (AUC={auc_teacher:.3f})", color="crimson", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"Baseline HLT test (AUC={auc_baseline:.3f})", color="steelblue", linewidth=2)
    plt.plot(tpr_m, fpr_m, "-.", label=f"Mixed HLT test (AUC={auc_mixed:.3f})", color="darkorange", linewidth=2)
    plt.plot(tpr_c, fpr_c, ":",  label=f"Consistency HLT test (AUC={auc_cons:.3f})", color="forestgreen", linewidth=2)

    plt.ylabel(r"False Positive Rate", fontsize=12)
    plt.xlabel(r"True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=11, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300)
    plt.close()

    # Append summary
    summary_file = Path(args.save_dir) / "run_summaries.txt"
    with open(summary_file, "a") as f:
        f.write(f"\nRun: {args.run_name}\n")
        f.write(f"  Teacher AUC (OFF test): {auc_teacher:.4f}\n")
        f.write(f"  Baseline AUC (HLT test): {auc_baseline:.4f}\n")
        f.write(f"  Mixed AUC (HLT test): {auc_mixed:.4f}\n")
        f.write(f"  Consistency AUC (HLT test): {auc_cons:.4f}\n")
        f.write(f"  Mix off prob: {args.mix_off_prob}\n")
        f.write(f"  Consistency: T={args.cons_T}, lambda_logit={args.lambda_logit}, lambda_embed={args.lambda_embed}, embed_mode={args.embed_mode}, gate_gamma={args.gate_gamma}\n")
        f.write(f"  Consistency ramp: start={args.cons_ramp_start}, end={args.cons_ramp_end}\n")
        f.write(f"  BR@50 (baseline/mixed/cons): {br_baseline:.2f} / {br_mixed:.2f} / {br_cons:.2f}\n")
        f.write(f"  Saved to: {save_dir}\n")
        f.write("=" * 80 + "\n")

    print(f"\nSaved results to: {save_dir / 'results.npz'} and {save_dir / 'results.png'}")
    print(f"Logged to: {summary_file}")


if __name__ == "__main__":
    main()

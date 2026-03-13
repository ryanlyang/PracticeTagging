#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train transformer taggers using raw HDF5 constituents with collaborator-style HLT simulation.

This script intentionally avoids utils.load_from_files(...) and instead loads:
  - fjet_clus_pt
  - fjet_clus_eta
  - fjet_clus_phi
  - fjet_clus_E

It trains two models:
  1) Offline model: train on offline view, test on offline view
  2) HLT model: train on HLT view, test on HLT view

By default, merge_radius is set to 1.5 as requested.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEFAULT_CFG = {
    "hlt_effects": {
        "pt_threshold_offline": 0.5,
        "pt_threshold_hlt": 1.5,
        "merge_enabled": True,
        "merge_radius": 0.15,  # requested
        "pt_resolution": 0.00, #0.10,
        "eta_resolution": 0.00, #0.03,
        "phi_resolution": 0.00, #0.03,
        "efficiency_loss": 0.00, #0.03,
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
        "patience": 12,
    },
}


def safe_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)


def wrap_dphi(dphi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def load_raw_from_h5(files, max_jets: int, max_constits: int) -> Tuple[np.ndarray, np.ndarray]:
    const_list = []
    labels_list = []
    read = 0

    for fpath in files:
        if read >= max_jets:
            break
        with h5py.File(fpath, "r") as f:
            n_file = int(f["labels"].shape[0])
            take = min(n_file, max_jets - read)

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
            labels_list.append(labels)
            read += take

    if not const_list:
        raise RuntimeError("No jets were loaded from input files.")

    all_const = np.concatenate(const_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    return all_const, all_labels


def apply_hlt_effects_collab_style(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: Dict,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Collaborator-style HLT effects: threshold -> merge -> smear -> efficiency."""
    rs = np.random.RandomState(int(seed))
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()
    stats = {
        "n_initial": int(hlt_mask.sum()),
        "n_lost_threshold": 0,
        "n_merge_ops": 0,
        "n_lost_efficiency": 0,
        "n_final": 0,
    }

    # Threshold
    below_threshold = (hlt[:, :, 0] < float(hcfg["pt_threshold_hlt"])) & hlt_mask
    hlt_mask[below_threshold] = False
    hlt[~hlt_mask] = 0.0
    stats["n_lost_threshold"] = int(below_threshold.sum())

    # Merge
    if bool(hcfg["merge_enabled"]) and float(hcfg["merge_radius"]) > 0.0:
        r = float(hcfg["merge_radius"])
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            if valid_idx.size < 2:
                continue
            to_remove = set()
            for ii in range(len(valid_idx)):
                i = int(valid_idx[ii])
                if i in to_remove:
                    continue
                for jj in range(ii + 1, len(valid_idx)):
                    j = int(valid_idx[jj])
                    if j in to_remove:
                        continue

                    deta = float(hlt[jet_idx, i, 1] - hlt[jet_idx, j, 1])
                    dphi = float(wrap_dphi(hlt[jet_idx, i, 2] - hlt[jet_idx, j, 2]))
                    dR = float(np.sqrt(deta * deta + dphi * dphi))
                    if dR >= r:
                        continue

                    pt_i = float(hlt[jet_idx, i, 0])
                    pt_j = float(hlt[jet_idx, j, 0])
                    pt_sum = pt_i + pt_j
                    if pt_sum < 1e-6:
                        continue

                    w_i = pt_i / pt_sum
                    w_j = pt_j / pt_sum
                    hlt[jet_idx, i, 0] = pt_sum
                    hlt[jet_idx, i, 1] = w_i * hlt[jet_idx, i, 1] + w_j * hlt[jet_idx, j, 1]

                    phi_i = float(hlt[jet_idx, i, 2])
                    phi_j = float(hlt[jet_idx, j, 2])
                    hlt[jet_idx, i, 2] = np.arctan2(
                        w_i * np.sin(phi_i) + w_j * np.sin(phi_j),
                        w_i * np.cos(phi_i) + w_j * np.cos(phi_j),
                    )
                    hlt[jet_idx, i, 3] = float(hlt[jet_idx, i, 3]) + float(hlt[jet_idx, j, 3])
                    to_remove.add(j)
                    stats["n_merge_ops"] += 1

            for j in to_remove:
                hlt_mask[jet_idx, j] = False
                hlt[jet_idx, j] = 0.0

    # Smearing
    valid = hlt_mask.copy()
    pt_noise = np.clip(
        rs.normal(1.0, float(hcfg["pt_resolution"]), (n_jets, max_part)),
        0.5,
        1.5,
    )
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0.0)

    eta_noise = rs.normal(0.0, float(hcfg["eta_resolution"]), (n_jets, max_part))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5.0, 5.0), 0.0)

    phi_noise = rs.normal(0.0, float(hcfg["phi_resolution"]), (n_jets, max_part))
    new_phi = hlt[:, :, 2] + phi_noise
    hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0.0)

    # Recompute E from (pt, eta) massless approx
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5.0, 5.0)), 0.0)

    # Efficiency
    eff = float(hcfg["efficiency_loss"])
    if eff > 0.0:
        lost = (rs.random_sample((n_jets, max_part)) < eff) & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0.0
        stats["n_lost_efficiency"] = int(lost.sum())

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0.0
    stats["n_final"] = int(hlt_mask.sum())
    stats["avg_constits_per_jet"] = float(hlt_mask.sum(axis=1).mean())
    return hlt.astype(np.float32), hlt_mask.astype(bool), stats


def compute_features(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pt = np.maximum(const[:, :, 0], 1e-8)
    eta = np.clip(const[:, :, 1], -5.0, 5.0)
    phi = const[:, :, 2]
    ene = np.maximum(const[:, :, 3], 1e-8)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    m = mask.astype(float)
    jet_px = (px * m).sum(axis=1, keepdims=True)
    jet_py = (py * m).sum(axis=1, keepdims=True)
    jet_pz = (pz * m).sum(axis=1, keepdims=True)
    jet_ene = (ene * m).sum(axis=1, keepdims=True)

    jet_pt = np.sqrt(jet_px**2 + jet_py**2) + 1e-8
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)

    d_eta = eta - jet_eta
    d_phi = np.arctan2(np.sin(phi - jet_phi), np.cos(phi - jet_phi))
    log_pt = np.log(pt + 1e-8)
    log_e = np.log(ene + 1e-8)
    log_pt_rel = np.log(pt / jet_pt + 1e-8)
    log_e_rel = np.log(ene / (jet_ene + 1e-8) + 1e-8)
    d_r = np.sqrt(d_eta**2 + d_phi**2)

    feat = np.stack([d_eta, d_phi, log_pt, log_e, log_pt_rel, log_e_rel, d_r], axis=-1)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = np.clip(feat, -20.0, 20.0)
    feat[~mask] = 0.0
    return feat.astype(np.float32)


def get_stats(feat: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.zeros(feat.shape[-1], dtype=np.float32)
    stds = np.zeros(feat.shape[-1], dtype=np.float32)
    for i in range(feat.shape[-1]):
        vals = feat[idx][:, :, i][mask[idx]]
        means[i] = float(np.nanmean(vals))
        stds[i] = float(np.nanstd(vals) + 1e-8)
    return means, stds


def standardize(feat: np.ndarray, mask: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    out = np.clip((feat - means[None, None, :]) / stds[None, None, :], -10.0, 10.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


class SingleViewDataset(Dataset):
    def __init__(self, feat: np.ndarray, mask: np.ndarray, labels: np.ndarray):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return {
            "x": self.feat[idx],
            "mask": self.mask[idx],
            "label": self.labels[idx],
        }


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            ResidualBlock(128, dropout=float(dropout)),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = x.reshape(b * s, self.input_dim)
        h = self.input_proj(h)
        h = h.reshape(b, s, self.embed_dim)
        h = self.encoder(h, src_key_padding_mask=~mask)

        q = self.pool_query.expand(b, -1, -1)
        pooled, _ = self.pool_attn(q, h, h, key_padding_mask=~mask, need_weights=False)
        z = self.norm(pooled.squeeze(1))
        logits = self.classifier(z).squeeze(1)
        return logits


def get_scheduler(opt, warmup_epochs: int, total_epochs: int):
    def lr_lambda(ep):
        if ep < int(warmup_epochs):
            return float(ep + 1) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * (ep - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    preds, labs = [], []
    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x, m)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += float(loss.item()) * int(y.shape[0])
        preds.extend(safe_sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
    return total_loss / max(1, len(preds)), float(roc_auc_score(labs, preds))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].to(device)
        logits = model(x, m)
        preds.extend(safe_sigmoid(logits).cpu().numpy().flatten())
        labs.extend(y.cpu().numpy().flatten())
    preds = np.asarray(preds, dtype=np.float32)
    labs = np.asarray(labs, dtype=np.float32)
    return float(roc_auc_score(labs, preds)), preds, labs


def fpr_at_fixed_tpr(labs: np.ndarray, preds: np.ndarray, tpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(labs, preds)
    return float(np.interp(float(tpr_target), tpr, fpr))


def print_fpr_table(
    labs: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    tpr_targets=(0.50, 0.30),
) -> Dict[str, Dict[str, float]]:
    """Print FPR@TPR table and return values as percentages."""
    model_names = list(model_preds.keys())
    col_width = max(18, max(len(n) for n in model_names) + 2)

    print("\nFPR at fixed TPRs (percent)")
    header = f"{'TPR (%)':>8} | " + " | ".join(f"{name + ' %':>{col_width}}" for name in model_names)
    print(header)
    print("-" * len(header))

    out: Dict[str, Dict[str, float]] = {}
    for tpr_tar in tpr_targets:
        row_vals: Dict[str, float] = {}
        row = f"{100.0 * tpr_tar:8.1f} | "
        cells = []
        for name in model_names:
            fpr_pct = 100.0 * fpr_at_fixed_tpr(labs, model_preds[name], tpr_tar)
            row_vals[name] = float(fpr_pct)
            cells.append(f"{fpr_pct:>{col_width}.3f}")
        print(row + " | ".join(cells))
        out[f"{int(round(100.0 * tpr_tar))}"] = row_vals

    return out


def train_with_early_stopping(model, train_loader, val_loader, device, cfg):
    tcfg = cfg["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    sch = get_scheduler(opt, int(tcfg["warmup_epochs"]), int(tcfg["epochs"]))

    best_auc = -1.0
    best_state = None
    no_improve = 0

    for ep in tqdm(range(int(tcfg["epochs"])), leave=False):
        train_loss, train_auc = train_one_epoch(model, train_loader, opt, device)
        val_auc, _, _ = evaluate(model, val_loader, device)
        sch.step()

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"  epoch {ep+1:02d}: train_loss={train_loss:.4f}, "
                f"train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best_val={best_auc:.4f}"
            )
        if no_improve >= int(tcfg["patience"]):
            print(f"  early stop at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def save_plots(
    save_root: Path,
    labs: np.ndarray,
    preds_off: np.ndarray,
    preds_hlt: np.ndarray,
    mask_off: np.ndarray,
    mask_hlt: np.ndarray,
    test_idx: np.ndarray,
):
    fpr_off, tpr_off, _ = roc_curve(labs, preds_off)
    fpr_hlt, tpr_hlt, _ = roc_curve(labs, preds_hlt)
    auc_off = float(roc_auc_score(labs, preds_off))
    auc_hlt = float(roc_auc_score(labs, preds_hlt))
    eps = 1e-4

    off_counts = mask_off[test_idx].sum(axis=1)
    hlt_counts = mask_hlt[test_idx].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax0, ax1 = axes

    ax0.plot(
        np.clip(tpr_off, eps, 1.0),
        np.clip(fpr_off, eps, 1.0),
        lw=2,
        label=f"Offline model (AUC={auc_off:.4f})",
    )
    ax0.plot(
        np.clip(tpr_hlt, eps, 1.0),
        np.clip(fpr_hlt, eps, 1.0),
        lw=2,
        label=f"HLT model (AUC={auc_hlt:.4f})",
    )
    diag_x = np.linspace(eps, 1.0, 400)
    ax0.plot(diag_x, diag_x, "k--", lw=1)
    ax0.set_title("ROC: Offline vs HLT")
    ax0.set_xlabel("True Positive Rate")
    ax0.set_ylabel("False Positive Rate (log scale)")
    ax0.set_yscale("log")
    ax0.set_ylim(eps, 1.0)
    ax0.set_xlim(eps, 1.0)
    ax0.grid(alpha=0.3)
    ax0.legend(loc="lower right")

    bins = np.arange(0, int(max(off_counts.max(), hlt_counts.max())) + 2)
    ax1.hist(off_counts, bins=bins, alpha=0.6, label=f"Offline (mean={off_counts.mean():.1f})")
    ax1.hist(hlt_counts, bins=bins, alpha=0.6, label=f"HLT (mean={hlt_counts.mean():.1f})")
    ax1.set_title("Constituents per Jet (test split)")
    ax1.set_xlabel("Number of valid constituents")
    ax1.set_ylabel("Jets")
    ax1.grid(alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    out_png = save_root / "offline_vs_hlt_effects.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "hlt_raw_transformer"))
    parser.add_argument("--run_name", type=str, default="merge_radius_1p5")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument("--epochs", type=int, default=DEFAULT_CFG["training"]["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CFG["training"]["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CFG["training"]["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CFG["training"]["weight_decay"])
    parser.add_argument("--warmup_epochs", type=int, default=DEFAULT_CFG["training"]["warmup_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CFG["training"]["patience"])

    parser.add_argument("--pt_threshold_offline", type=float, default=DEFAULT_CFG["hlt_effects"]["pt_threshold_offline"])
    parser.add_argument("--pt_threshold_hlt", type=float, default=DEFAULT_CFG["hlt_effects"]["pt_threshold_hlt"])
    parser.add_argument("--merge_radius", type=float, default=1.5)
    parser.add_argument("--pt_resolution", type=float, default=DEFAULT_CFG["hlt_effects"]["pt_resolution"])
    parser.add_argument("--eta_resolution", type=float, default=DEFAULT_CFG["hlt_effects"]["eta_resolution"])
    parser.add_argument("--phi_resolution", type=float, default=DEFAULT_CFG["hlt_effects"]["phi_resolution"])
    parser.add_argument("--efficiency_loss", type=float, default=DEFAULT_CFG["hlt_effects"]["efficiency_loss"])
    parser.add_argument("--disable_merge", action="store_true")
    parser.add_argument("--skip_save_models", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg["training"]["epochs"] = int(args.epochs)
    cfg["training"]["batch_size"] = int(args.batch_size)
    cfg["training"]["lr"] = float(args.lr)
    cfg["training"]["weight_decay"] = float(args.weight_decay)
    cfg["training"]["warmup_epochs"] = int(args.warmup_epochs)
    cfg["training"]["patience"] = int(args.patience)

    cfg["hlt_effects"]["pt_threshold_offline"] = float(args.pt_threshold_offline)
    cfg["hlt_effects"]["pt_threshold_hlt"] = float(args.pt_threshold_hlt)
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["merge_enabled"] = not bool(args.disable_merge)
    cfg["hlt_effects"]["pt_resolution"] = float(args.pt_resolution)
    cfg["hlt_effects"]["eta_resolution"] = float(args.eta_resolution)
    cfg["hlt_effects"]["phi_resolution"] = float(args.phi_resolution)
    cfg["hlt_effects"]["efficiency_loss"] = float(args.efficiency_loss)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")
    print(f"HLT config: {json.dumps(cfg['hlt_effects'], indent=2)}")

    train_path = Path(args.train_path)
    train_files = sorted(train_path.glob("*.h5"))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {train_path}")

    print("Loading raw constituents directly from HDF5...")
    all_const, all_labels = load_raw_from_h5(
        train_files,
        max_jets=int(args.n_train_jets),
        max_constits=int(args.max_constits),
    )
    print(f"Loaded raw const shape={all_const.shape}, labels shape={all_labels.shape}")

    raw_mask = all_const[:, :, 0] > 0.0

    # Offline view
    off_thr = float(cfg["hlt_effects"]["pt_threshold_offline"])
    mask_off = raw_mask & (all_const[:, :, 0] >= off_thr)
    const_off = all_const.copy()
    const_off[~mask_off] = 0.0

    # HLT view
    print("Applying collaborator-style HLT effects...")
    const_hlt, mask_hlt, hlt_stats = apply_hlt_effects_collab_style(all_const, raw_mask, cfg, seed=RANDOM_SEED)
    print("HLT stats:")
    for k, v in hlt_stats.items():
        print(f"  {k}: {v}")
    print(f"  avg_constits_offline: {float(mask_off.sum(axis=1).mean()):.3f}")

    # Features
    print("Computing engineered features...")
    feat_off = compute_features(const_off, mask_off)
    feat_hlt = compute_features(const_hlt, mask_hlt)

    # Splits
    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx]
    )

    # Standardize using OFFLINE train distribution
    means, stds = get_stats(feat_off, mask_off, train_idx)
    feat_off_std = standardize(feat_off, mask_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, mask_hlt, means, stds)

    # Datasets and loaders
    bs = int(cfg["training"]["batch_size"])
    train_ds_off = SingleViewDataset(feat_off_std[train_idx], mask_off[train_idx], all_labels[train_idx])
    val_ds_off = SingleViewDataset(feat_off_std[val_idx], mask_off[val_idx], all_labels[val_idx])
    test_ds_off = SingleViewDataset(feat_off_std[test_idx], mask_off[test_idx], all_labels[test_idx])

    train_ds_hlt = SingleViewDataset(feat_hlt_std[train_idx], mask_hlt[train_idx], all_labels[train_idx])
    val_ds_hlt = SingleViewDataset(feat_hlt_std[val_idx], mask_hlt[val_idx], all_labels[val_idx])
    test_ds_hlt = SingleViewDataset(feat_hlt_std[test_idx], mask_hlt[test_idx], all_labels[test_idx])

    train_loader_off = DataLoader(train_ds_off, batch_size=bs, shuffle=True, drop_last=True)
    val_loader_off = DataLoader(val_ds_off, batch_size=bs, shuffle=False)
    test_loader_off = DataLoader(test_ds_off, batch_size=bs, shuffle=False)

    train_loader_hlt = DataLoader(train_ds_hlt, batch_size=bs, shuffle=True, drop_last=True)
    val_loader_hlt = DataLoader(val_ds_hlt, batch_size=bs, shuffle=False)
    test_loader_hlt = DataLoader(test_ds_hlt, batch_size=bs, shuffle=False)

    # Offline model
    print("\n" + "=" * 70)
    print("Training OFFLINE model (train: offline, test: offline)")
    print("=" * 70)
    model_off = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    model_off, best_val_off = train_with_early_stopping(model_off, train_loader_off, val_loader_off, device, cfg)
    auc_off, preds_off, labs_off = evaluate(model_off, test_loader_off, device)
    print(f"Offline model: best_val_auc={best_val_off:.4f}, test_auc={auc_off:.4f}")

    # HLT model
    print("\n" + "=" * 70)
    print("Training HLT model (train: HLT, test: HLT)")
    print("=" * 70)
    model_hlt = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    model_hlt, best_val_hlt = train_with_early_stopping(model_hlt, train_loader_hlt, val_loader_hlt, device, cfg)
    auc_hlt, preds_hlt, labs_hlt = evaluate(model_hlt, test_loader_hlt, device)
    print(f"HLT model: best_val_auc={best_val_hlt:.4f}, test_auc={auc_hlt:.4f}")

    if not np.array_equal(labs_off.astype(np.int64), labs_hlt.astype(np.int64)):
        raise RuntimeError("Offline and HLT test labels differ unexpectedly.")

    fpr_table = print_fpr_table(
        labs=labs_off,
        model_preds={
            "HLT (baseline)": preds_hlt,
            "Offline (teacher)": preds_off,
        },
        tpr_targets=(0.50, 0.30),
    )

    # Save outputs
    if not args.skip_save_models:
        torch.save({"model": model_off.state_dict(), "test_auc": auc_off}, save_root / "model_offline.pt")
        torch.save({"model": model_hlt.state_dict(), "test_auc": auc_hlt}, save_root / "model_hlt.pt")

    summary = {
        "n_jets": int(all_const.shape[0]),
        "max_constits": int(all_const.shape[1]),
        "auc_offline_test": float(auc_off),
        "auc_hlt_test": float(auc_hlt),
        "auc_delta_off_minus_hlt": float(auc_off - auc_hlt),
        "hlt_effects": cfg["hlt_effects"],
        "hlt_stats": hlt_stats,
        "avg_constits_offline": float(mask_off.sum(axis=1).mean()),
        "avg_constits_hlt": float(mask_hlt.sum(axis=1).mean()),
        "fpr_percent_at_fixed_tpr_percent": {
            "50": fpr_table["50"],
            "30": fpr_table["30"],
        },
    }
    with open(save_root / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_plots(
        save_root=save_root,
        labs=labs_off,
        preds_off=preds_off,
        preds_hlt=preds_hlt,
        mask_off=mask_off,
        mask_hlt=mask_hlt,
        test_idx=test_idx,
    )

    print("\nDone.")
    print(f"Results JSON: {save_root / 'results_summary.json'}")


if __name__ == "__main__":
    main()

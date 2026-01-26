#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unmerge HLT constituents: predict the original offline constituents that merged
into each HLT token (for merged tokens only).
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

import matplotlib.pyplot as plt
from tqdm import tqdm

import utils


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


# ----------------------------- Column order (EDIT if needed) ----------------------------- #
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
        "embed_dim": 192,
        "num_heads": 8,
        "num_layers": 6,
        "decoder_layers": 3,
        "ff_dim": 768,
        "dropout": 0.1,
        "count_embed_dim": 64,
    },
    "training": {
        "batch_size": 256,
        "epochs": 120,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 5,
        "patience": 20,
    },
}


def apply_hlt_effects_with_groups(const, mask, cfg, seed=42):
    """
    const: (N, M, 4) [pt, eta, phi, E]
    mask:  (N, M) bool

    Returns:
      hlt, hlt_mask, samples, stats

    samples: list of (jet_idx, token_idx, origin_indices_list)
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

    n_merged = 0
    n_lost_eff = 0
    samples = []

    for jet_idx in tqdm(range(n_jets), desc="HLTEffects"):
        origin_lists = []
        for idx in range(max_part):
            if hlt_mask[jet_idx, idx]:
                origin_lists.append([idx])
            else:
                origin_lists.append([])

        # Effect 2: Cluster merging
        if hcfg["merge_enabled"] and hcfg["merge_radius"] > 0:
            merge_radius = hcfg["merge_radius"]
            valid_idx = np.where(hlt_mask[jet_idx])[0]

            if len(valid_idx) >= 2:
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
                            origin_lists[idx_i].extend(origin_lists[idx_j])

                            to_remove.add(idx_j)
                            n_merged += 1

                for idx in to_remove:
                    hlt_mask[jet_idx, idx] = False
                    hlt[jet_idx, idx] = 0
                    origin_lists[idx] = []

        # Effect 3: Resolution smearing
        valid = hlt_mask[jet_idx]
        pt_noise = np.random.normal(1.0, hcfg["pt_resolution"], max_part)
        pt_noise = np.clip(pt_noise, 0.5, 1.5)
        hlt[jet_idx, :, 0] = np.where(valid, hlt[jet_idx, :, 0] * pt_noise, 0)

        eta_noise = np.random.normal(0, hcfg["eta_resolution"], max_part)
        hlt[jet_idx, :, 1] = np.where(valid, np.clip(hlt[jet_idx, :, 1] + eta_noise, -5, 5), 0)

        phi_noise = np.random.normal(0, hcfg["phi_resolution"], max_part)
        new_phi = hlt[jet_idx, :, 2] + phi_noise
        hlt[jet_idx, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)

        # Recalculate E (massless approx)
        hlt[jet_idx, :, 3] = np.where(valid, hlt[jet_idx, :, 0] * np.cosh(np.clip(hlt[jet_idx, :, 1], -5, 5)), 0)

        # Effect 4: Random efficiency loss
        if hcfg["efficiency_loss"] > 0:
            random_loss = np.random.random(max_part) < hcfg["efficiency_loss"]
            lost = random_loss & hlt_mask[jet_idx]
            if lost.any():
                hlt_mask[jet_idx, lost] = False
                hlt[jet_idx, lost] = 0
                for idx in np.where(lost)[0]:
                    origin_lists[idx] = []
                n_lost_eff += int(lost.sum())

        # Collect merged-token samples
        for idx in range(max_part):
            if hlt_mask[jet_idx, idx] and len(origin_lists[idx]) > 1:
                samples.append((jet_idx, idx, origin_lists[idx]))

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
    return hlt, hlt_mask, samples, stats


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


def get_stats(feat, mask, idx, dim):
    means, stds = np.zeros(dim), np.zeros(dim)
    for i in range(dim):
        vals = feat[idx][:, :, i][mask[idx]]
        means[i] = np.nanmean(vals)
        stds[i] = np.nanstd(vals) + 1e-8
    return means, stds


def standardize(feat, mask, means, stds):
    std = np.clip((feat - means) / stds, -10, 10)
    std = np.nan_to_num(std, 0.0)
    std[~mask] = 0
    return std.astype(np.float32)


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
        jet_idx, token_idx, origin = self.samples[i]
        count = len(origin)
        target = self.constituents_off[jet_idx, origin, :4].astype(np.float32)

        target = (target - self.tgt_mean) / self.tgt_std
        target = np.clip(target, -10, 10)

        target_pad = np.zeros((self.max_count, 4), dtype=np.float32)
        target_pad[:count] = target

        return {
            "hlt": torch.tensor(self.feat_hlt[jet_idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask_hlt[jet_idx], dtype=torch.bool),
            "token_idx": torch.tensor(token_idx, dtype=torch.long),
            "count": torch.tensor(count, dtype=torch.long),
            "target": torch.tensor(target_pad, dtype=torch.float32),
        }


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


def set_chamfer_loss(preds, targets, counts):
    total = 0.0
    for i in range(preds.size(0)):
        k = int(counts[i].item())
        pred_i = preds[i, :k]
        tgt_i = targets[i, :k]
        dist = torch.cdist(pred_i, tgt_i, p=1)
        loss_i = dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()
        total += loss_i
    return total / max(preds.size(0), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["hlt"].to(device)
        mask = batch["mask"].to(device)
        token_idx = batch["token_idx"].to(device)
        count = batch["count"].to(device)
        target = batch["target"].to(device)
        preds = model(x, mask, token_idx, count)
        loss = set_chamfer_loss(preds, target, count)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--max_merge_count", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "unmerging"))
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=CONFIG["training"]["epochs"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["training"]["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["training"]["lr"])
    parser.add_argument("--weight_decay", type=float, default=CONFIG["training"]["weight_decay"])
    parser.add_argument("--warmup_epochs", type=int, default=CONFIG["training"]["warmup_epochs"])
    parser.add_argument("--patience", type=int, default=CONFIG["training"]["patience"])
    parser.add_argument("--skip_save_models", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=CONFIG["model"]["embed_dim"])
    parser.add_argument("--num_heads", type=int, default=CONFIG["model"]["num_heads"])
    parser.add_argument("--num_layers", type=int, default=CONFIG["model"]["num_layers"])
    parser.add_argument("--decoder_layers", type=int, default=CONFIG["model"]["decoder_layers"])
    parser.add_argument("--ff_dim", type=int, default=CONFIG["model"]["ff_dim"])
    parser.add_argument("--dropout", type=float, default=CONFIG["model"]["dropout"])
    parser.add_argument("--count_embed_dim", type=int, default=CONFIG["model"]["count_embed_dim"])
    parser.add_argument("--pt_resolution", type=float, default=None)
    parser.add_argument("--eta_resolution", type=float, default=None)
    parser.add_argument("--phi_resolution", type=float, default=None)
    parser.add_argument("--n_print_examples", type=int, default=5)
    args = parser.parse_args()

    if args.pt_resolution is not None:
        CONFIG["hlt_effects"]["pt_resolution"] = float(args.pt_resolution)
    if args.eta_resolution is not None:
        CONFIG["hlt_effects"]["eta_resolution"] = float(args.eta_resolution)
    if args.phi_resolution is not None:
        CONFIG["hlt_effects"]["phi_resolution"] = float(args.phi_resolution)

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
    constituents_hlt, masks_hlt, samples, stats = apply_hlt_effects_with_groups(
        constituents_raw, mask_raw, CONFIG, seed=RANDOM_SEED
    )

    pt_threshold_off = CONFIG["hlt_effects"]["pt_threshold_offline"]
    masks_off = mask_raw & (constituents_raw[:, :, 0] >= pt_threshold_off)
    constituents_off = constituents_raw.copy()
    constituents_off[~masks_off] = 0

    print("HLT Simulation Statistics:")
    print(f"  Offline particles: {stats['n_initial']:,}")
    print(f"  Lost to pT threshold ({CONFIG['hlt_effects']['pt_threshold_hlt']}): {stats['n_lost_threshold']:,}")
    print(f"  Lost to merging (dR<{CONFIG['hlt_effects']['merge_radius']}): {stats['n_merged']:,}")
    print(f"  Lost to efficiency: {stats['n_lost_eff']:,}")
    print(f"  HLT particles: {stats['n_final']:,}")
    print(f"  Avg per jet: Offline={masks_off.sum(axis=1).mean():.1f}, HLT={masks_hlt.sum(axis=1).mean():.1f}")
    print(f"Merged-token samples: {len(samples):,}")

    if len(samples) == 0:
        raise RuntimeError("No merged-token samples found. Check HLT merge settings.")

    print("Computing features...")
    features_off = compute_features(constituents_off, masks_off)
    features_hlt = compute_features(constituents_hlt, masks_hlt)

    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx, 7)
    features_hlt_std = standardize(features_hlt, masks_hlt, feat_means, feat_stds)

    max_count = max(int(args.max_merge_count), 2)
    samples = [s for s in samples if len(s[2]) <= max_count]

    train_idx_set = set(train_idx)
    val_idx_set = set(val_idx)
    test_idx_set = set(test_idx)
    train_samples = [s for s in samples if s[0] in train_idx_set]
    val_samples = [s for s in samples if s[0] in val_idx_set]
    test_samples = [s for s in samples if s[0] in test_idx_set]
    print(f"Merged samples: train={len(train_samples):,}, val={len(val_samples):,}, test={len(test_samples):,}")

    def gather_targets(sample_list, label):
        out = []
        for jet_idx, _, origin in tqdm(sample_list, desc=f"Targets-{label}"):
            out.append(constituents_off[jet_idx, origin, :4])
        return out

    print("Building unmerge targets (train split)...")
    train_targets = gather_targets(train_samples, "train")
    flat_train = np.concatenate(train_targets, axis=0)
    tgt_mean = flat_train.mean(axis=0)
    tgt_std = flat_train.std(axis=0) + 1e-8

    train_ds = UnmergeDataset(features_hlt_std, masks_hlt, constituents_off, train_samples, max_count, tgt_mean, tgt_std)
    val_ds = UnmergeDataset(features_hlt_std, masks_hlt, constituents_off, val_samples, max_count, tgt_mean, tgt_std)
    test_ds = UnmergeDataset(features_hlt_std, masks_hlt, constituents_off, test_samples, max_count, tgt_mean, tgt_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = UnmergePredictor(
        input_dim=7,
        max_count=max_count,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        decoder_layers=args.decoder_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        count_embed_dim=args.count_embed_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = get_scheduler(opt, args.warmup_epochs, args.epochs)

    best_val, best_state, no_improve = 1e9, None, 0

    print("\n" + "=" * 70)
    print("TRAINING: UNMERGE PREDICTOR (HLT token -> offline constituents)")
    print("=" * 70)
    for ep in tqdm(range(args.epochs), desc="Unmerge"):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["hlt"].to(device)
            mask = batch["mask"].to(device)
            token_idx = batch["token_idx"].to(device)
            count = batch["count"].to(device)
            target = batch["target"].to(device)

            opt.zero_grad()
            preds = model(x, mask, token_idx, count)
            loss = set_chamfer_loss(preds, target, count)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_loss = evaluate(model, val_loader, device)
        sch.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best={best_val:.4f}")

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate(model, test_loader, device)
    print(f"\nFinal test loss: {test_loss:.4f}")

    if args.n_print_examples > 0:
        print("\nSample predictions vs ground truth (pt, eta, phi, E):")
        shown = 0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch["hlt"].to(device)
                mask = batch["mask"].to(device)
                token_idx = batch["token_idx"].to(device)
                count = batch["count"].to(device)
                target = batch["target"].to(device)
                preds = model(x, mask, token_idx, count)

                preds_np = preds.cpu().numpy()
                target_np = target.cpu().numpy()
                count_np = count.cpu().numpy()

                for i in range(preds_np.shape[0]):
                    k = int(count_np[i])
                    pred = preds_np[i, :k]
                    tgt = target_np[i, :k]

                    pred = pred * tgt_std + tgt_mean
                    tgt = tgt * tgt_std + tgt_mean

                    print(f"Example {shown + 1} | count={k}")
                    print("  pred:")
                    print(np.round(pred, 4))
                    print("  true:")
                    print(np.round(tgt, 4))
                    shown += 1
                    if shown >= args.n_print_examples:
                        break
                if shown >= args.n_print_examples:
                    break

    if not args.skip_save_models:
        torch.save(
            {"model": model.state_dict(), "best_val_loss": best_val, "max_merge_count": max_count},
            save_root / "unmerge_predictor.pt",
        )

    np.savez(
        save_root / "results.npz",
        best_val_loss=best_val,
        test_loss=test_loss,
        max_merge_count=max_count,
        target_mean=tgt_mean,
        target_std=tgt_std,
    )

    print(f"Saved results to: {save_root}")


if __name__ == "__main__":
    main()

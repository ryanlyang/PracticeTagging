#!/usr/bin/env python3
"""
Weakly-supervised per-constituent merge-state predictor (no token-level GT in training).

What this script does:
1) Build pseudo-HLT from offline jets with merge-only + light smearing.
2) Keep origin mappings for evaluation ONLY (never used in training loss).
3) Train a token model to predict merged vs unmerged with:
   - jet-level count consistency (offline_count - hlt_count),
   - pseudo labels from geometric/kinematic nearest-neighbor heuristics,
   - weak regularization/bounds.
4) Evaluate against hidden token-level GT and save plots/results.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils
from unmerge_new_ideas import (
    CONFIG,
    RANDOM_SEED,
    ETA_IDX,
    PHI_IDX,
    PT_IDX,
    apply_hlt_effects_with_tracking,
    compute_features,
    get_stats,
    standardize,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeakMergeDataset(Dataset):
    def __init__(
        self,
        feat: np.ndarray,
        mask: np.ndarray,
        jet_extra_count: np.ndarray,
        pseudo_y: np.ndarray,
        pseudo_w: np.ndarray,
    ) -> None:
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.extra = torch.tensor(jet_extra_count, dtype=torch.float32)
        self.pseudo_y = torch.tensor(pseudo_y, dtype=torch.float32)
        self.pseudo_w = torch.tensor(pseudo_w, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat.shape[0]

    def __getitem__(self, i: int):
        return {
            "feat": self.feat[i],
            "mask": self.mask[i],
            "extra": self.extra[i],
            "pseudo_y": self.pseudo_y[i],
            "pseudo_w": self.pseudo_w[i],
        }


class TokenMergeWeakModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 192,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
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
        self.merge_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.extra_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=~mask)
        logits = self.merge_head(h).squeeze(-1)        # merged vs not merged logits
        extra_raw = self.extra_head(h).squeeze(-1)     # nonnegative latent extra-count signal
        return logits, extra_raw


def build_pseudo_labels(
    hlt_const: np.ndarray,
    hlt_mask: np.ndarray,
    off_const: np.ndarray,
    off_mask: np.ndarray,
    jet_extra_count: np.ndarray,
    dr_unmerged_thr: float,
    relpt_unmerged_thr: float,
    dr_merged_thr: float,
    relpt_merged_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pseudo labels without ancestry GT:
      0 -> likely unmerged
      1 -> likely merged
     -1 -> unknown (ignored in supervised token loss)
    """
    n_jets, max_part, _ = hlt_const.shape
    pseudo_y = -np.ones((n_jets, max_part), dtype=np.float32)
    pseudo_w = np.zeros((n_jets, max_part), dtype=np.float32)

    for j in tqdm(range(n_jets), desc="PseudoLabels", leave=False):
        h_idx = np.where(hlt_mask[j])[0]
        o_idx = np.where(off_mask[j])[0]
        if len(h_idx) == 0 or len(o_idx) == 0:
            continue

        h = hlt_const[j, h_idx, :4]
        o = off_const[j, o_idx, :4]
        h_eta = h[:, 1][:, None]
        o_eta = o[:, 1][None, :]
        h_phi = h[:, 2][:, None]
        o_phi = o[:, 2][None, :]
        dphi = np.arctan2(np.sin(h_phi - o_phi), np.cos(h_phi - o_phi))
        deta = h_eta - o_eta
        dr = np.sqrt(deta * deta + dphi * dphi)

        h_pt = np.maximum(h[:, 0][:, None], 1e-8)
        o_pt = np.maximum(o[:, 0][None, :], 1e-8)
        relpt = np.abs(np.log(h_pt / o_pt))

        # For each HLT token, best offline candidate.
        best = np.argmin(dr + 0.25 * relpt, axis=1)
        best_dr = dr[np.arange(len(h_idx)), best]
        best_relpt = relpt[np.arange(len(h_idx)), best]

        # High-confidence unmerged.
        um = (best_dr < dr_unmerged_thr) & (best_relpt < relpt_unmerged_thr)
        if np.any(um):
            idx_um = h_idx[um]
            pseudo_y[j, idx_um] = 0.0
            # Closer match -> higher confidence.
            conf = np.exp(-best_dr[um] / max(dr_unmerged_thr, 1e-6))
            conf *= np.exp(-best_relpt[um] / max(relpt_unmerged_thr, 1e-6))
            pseudo_w[j, idx_um] = np.clip(conf, 0.2, 1.0)

        # High-confidence merged (only if jet has missing constituents from merging).
        if jet_extra_count[j] > 0:
            mg = (best_dr > dr_merged_thr) | (best_relpt > relpt_merged_thr)
            if np.any(mg):
                idx_mg = h_idx[mg]
                pseudo_y[j, idx_mg] = 1.0
                # farther mismatch -> higher confidence for merged class
                conf = 1.0 - np.exp(-best_dr[mg] / max(dr_merged_thr, 1e-6))
                conf *= 1.0 - np.exp(-best_relpt[mg] / max(relpt_merged_thr, 1e-6))
                # cap by jet-level signal strength
                jet_sig = min(1.0, float(jet_extra_count[j]) / max(len(h_idx), 1))
                pseudo_w[j, idx_mg] = np.clip(0.2 + 0.8 * conf * (0.5 + jet_sig), 0.2, 1.0)

    return pseudo_y, pseudo_w


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    max_merge_count: int,
    w_count: float,
    w_bounds: float,
    w_pseudo: float,
    w_entropy: float,
    w_sparse: float,
) -> float:
    model.train()
    total = 0.0
    n = 0
    max_extra = float(max(1, max_merge_count - 1))

    for batch in loader:
        x = batch["feat"].to(device)
        m = batch["mask"].to(device)
        d_true = batch["extra"].to(device)
        py = batch["pseudo_y"].to(device)
        pw = batch["pseudo_w"].to(device)

        opt.zero_grad()
        logits, extra_raw = model(x, m)
        p = torch.sigmoid(logits)
        m_f = m.float()

        # Expected extra count contribution per token.
        expected_extra = F.softplus(extra_raw) * p * m_f
        d_pred = expected_extra.sum(dim=1)
        loss_count = F.smooth_l1_loss(d_pred, d_true)

        # Bounds: number of merged tokens should be between D/max_extra and D.
        p_sum = (p * m_f).sum(dim=1)
        lower = d_true / max_extra
        upper = d_true
        loss_bounds = (F.relu(lower - p_sum) ** 2 + F.relu(p_sum - upper) ** 2).mean()

        # Pseudo-supervised token loss on confident subset.
        pseudo_mask = (py >= 0.0) & m
        if pseudo_mask.any():
            bce = F.binary_cross_entropy_with_logits(
                logits[pseudo_mask], py[pseudo_mask], reduction="none"
            )
            loss_pseudo = (bce * pw[pseudo_mask]).sum() / (pw[pseudo_mask].sum() + 1e-8)
        else:
            loss_pseudo = torch.zeros((), device=device)

        # Mild regularization.
        p_clip = torch.clamp(p, 1e-6, 1 - 1e-6)
        entropy = -(p_clip * torch.log(p_clip) + (1 - p_clip) * torch.log(1 - p_clip))
        loss_entropy = (entropy * m_f).sum() / (m_f.sum() + 1e-8)
        loss_sparse = (p * m_f).sum() / (m_f.sum() + 1e-8)

        loss = (
            w_count * loss_count
            + w_bounds * loss_bounds
            + w_pseudo * loss_pseudo
            + w_entropy * loss_entropy
            + w_sparse * loss_sparse
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    jet_extra_count: np.ndarray,
    true_merged: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    probs_all = []
    mask_all = []
    gt_all = []
    extra_pred = []
    extra_true = []

    for start in range(0, len(idx), batch_size):
        ii = idx[start:start + batch_size]
        x = torch.tensor(feat[ii], dtype=torch.float32, device=device)
        m = torch.tensor(mask[ii], dtype=torch.bool, device=device)
        logits, extra_raw = model(x, m)
        p = torch.sigmoid(logits)
        exp_extra = (F.softplus(extra_raw) * p * m.float()).sum(dim=1)

        probs_all.append(p.detach().cpu().numpy())
        mask_all.append(m.detach().cpu().numpy())
        gt_all.append(true_merged[ii].astype(np.float32))
        extra_pred.append(exp_extra.detach().cpu().numpy())
        extra_true.append(jet_extra_count[ii].astype(np.float32))

    probs = np.concatenate(probs_all, axis=0)
    msk = np.concatenate(mask_all, axis=0)
    gt = np.concatenate(gt_all, axis=0)
    e_pred = np.concatenate(extra_pred, axis=0)
    e_true = np.concatenate(extra_true, axis=0)

    flat_mask = msk.reshape(-1).astype(bool)
    y_true = gt.reshape(-1)[flat_mask]
    y_prob = probs.reshape(-1)[flat_mask]
    y_pred = (y_prob >= 0.5).astype(np.int64)
    y_true_i = y_true.astype(np.int64)

    acc = float((y_pred == y_true_i).mean())
    tp = int(((y_pred == 1) & (y_true_i == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_i == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_i == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    auc = roc_auc_score(y_true_i, y_prob) if len(np.unique(y_true_i)) > 1 else 0.0
    ap = average_precision_score(y_true_i, y_prob) if len(np.unique(y_true_i)) > 1 else 0.0
    mae_extra = float(np.mean(np.abs(e_pred - e_true)))

    # High-confidence subset metrics.
    conf_hi = y_prob >= 0.9
    conf_lo = y_prob <= 0.1
    conf_mask = conf_hi | conf_lo
    cov = float(conf_mask.mean())
    if conf_mask.any():
        conf_pred = (y_prob[conf_mask] >= 0.5).astype(np.int64)
        conf_true = y_true_i[conf_mask]
        conf_acc = float((conf_pred == conf_true).mean())
    else:
        conf_acc = 0.0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "auc": float(auc),
        "ap": float(ap),
        "mae_extra": mae_extra,
        "conf_cov": cov,
        "conf_acc": conf_acc,
        "y_true": y_true_i,
        "y_prob": y_prob,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--n_train_jets", type=int, default=200000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=80)
    ap.add_argument("--max_merge_count", type=int, default=10)
    ap.add_argument("--save_dir", type=str, default="checkpoints/weak_merge_state")
    ap.add_argument("--run_name", type=str, default="merge_binary_no_gt")
    ap.add_argument("--batch_size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--device", type=str, default="cuda")
    # HLT config (merge-only + light smearing)
    ap.add_argument("--pt_resolution", type=float, default=0.02)
    ap.add_argument("--eta_resolution", type=float, default=0.005)
    ap.add_argument("--phi_resolution", type=float, default=0.005)
    ap.add_argument("--merge_radius", type=float, default=0.01)
    ap.add_argument("--efficiency_loss", type=float, default=0.0)
    ap.add_argument("--pt_threshold_hlt", type=float, default=0.0)
    # Pseudo labeling thresholds
    ap.add_argument("--dr_unmerged_thr", type=float, default=0.003)
    ap.add_argument("--relpt_unmerged_thr", type=float, default=0.03)
    ap.add_argument("--dr_merged_thr", type=float, default=0.01)
    ap.add_argument("--relpt_merged_thr", type=float, default=0.08)
    # Loss weights
    ap.add_argument("--w_count", type=float, default=2.0)
    ap.add_argument("--w_bounds", type=float, default=0.5)
    ap.add_argument("--w_pseudo", type=float, default=1.0)
    ap.add_argument("--w_entropy", type=float, default=0.01)
    ap.add_argument("--w_sparse", type=float, default=0.01)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    # Load data
    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(train_path.glob("*.h5"))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if not train_files:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = args.offset_jets + args.n_train_jets
    data_full, labels_full, _, _, _ = utils.load_from_files(
        [str(p) for p in train_files],
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
        use_train_weights=False,
    )
    if data_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Requested offset+n_train_jets={max_jets_needed}, got {data_full.shape[0]}.")

    data = data_full[args.offset_jets:args.offset_jets + args.n_train_jets]
    labels = labels_full[args.offset_jets:args.offset_jets + args.n_train_jets].astype(np.int64)

    # Offline constituents (pt, eta, phi, E)
    eta = data[:, :, ETA_IDX].astype(np.float32)
    phi = data[:, :, PHI_IDX].astype(np.float32)
    pt = data[:, :, PT_IDX].astype(np.float32)
    off_mask = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    off_const = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)

    # Build HLT with light smearing + merging and zero efficiency loss.
    cfg = copy.deepcopy(CONFIG)
    cfg["hlt_effects"]["pt_resolution"] = float(args.pt_resolution)
    cfg["hlt_effects"]["eta_resolution"] = float(args.eta_resolution)
    cfg["hlt_effects"]["phi_resolution"] = float(args.phi_resolution)
    cfg["hlt_effects"]["merge_enabled"] = True
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["efficiency_loss"] = float(args.efficiency_loss)
    cfg["hlt_effects"]["pt_threshold_hlt"] = float(args.pt_threshold_hlt)
    cfg["hlt_effects"]["pt_threshold_offline"] = 0.5

    hlt_const, hlt_mask, origin_counts, _, hlt_stats = apply_hlt_effects_with_tracking(
        off_const, off_mask, cfg, seed=args.seed
    )
    true_merged = ((origin_counts > 1) & hlt_mask).astype(np.int64)  # eval only

    off_count = off_mask.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    extra_count = np.maximum(0.0, off_count - hlt_count).astype(np.float32)

    # Features and split
    feat_hlt = compute_features(hlt_const, hlt_mask)
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=args.seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=args.seed, stratify=labels[temp_idx]
    )
    means, stds = get_stats(feat_hlt, hlt_mask, train_idx)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    # Pseudo labels (no ancestry usage)
    pseudo_y, pseudo_w = build_pseudo_labels(
        hlt_const,
        hlt_mask,
        off_const,
        off_mask,
        extra_count,
        dr_unmerged_thr=args.dr_unmerged_thr,
        relpt_unmerged_thr=args.relpt_unmerged_thr,
        dr_merged_thr=args.dr_merged_thr,
        relpt_merged_thr=args.relpt_merged_thr,
    )

    # Datasets/loaders
    ds_train = WeakMergeDataset(
        feat_hlt_std[train_idx],
        hlt_mask[train_idx],
        extra_count[train_idx],
        pseudo_y[train_idx],
        pseudo_w[train_idx],
    )
    ds_val = WeakMergeDataset(
        feat_hlt_std[val_idx],
        hlt_mask[val_idx],
        extra_count[val_idx],
        pseudo_y[val_idx],
        pseudo_w[val_idx],
    )
    dl_cfg = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(ds_train, shuffle=True, drop_last=True, **dl_cfg)
    _ = DataLoader(ds_val, shuffle=False, **dl_cfg)  # reserved for future extension

    model = TokenMergeWeakModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    best_val_auc = -1.0
    best_state = None
    no_improve = 0
    history = []
    for ep in range(args.epochs):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            opt=opt,
            device=device,
            max_merge_count=args.max_merge_count,
            w_count=args.w_count,
            w_bounds=args.w_bounds,
            w_pseudo=args.w_pseudo,
            w_entropy=args.w_entropy,
            w_sparse=args.w_sparse,
        )
        scheduler.step()
        val_metrics = evaluate(
            model=model,
            feat=feat_hlt_std,
            mask=hlt_mask,
            jet_extra_count=extra_count,
            true_merged=true_merged,
            idx=val_idx,
            batch_size=args.batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": train_loss,
                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "val_acc": val_metrics["acc"],
                "val_mae_extra": val_metrics["mae_extra"],
            }
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Ep {ep+1}: train_loss={train_loss:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | val_ap={val_metrics['ap']:.4f} | "
                f"val_acc={val_metrics['acc']:.4f} | val_extra_mae={val_metrics['mae_extra']:.3f}"
            )
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(
        model=model,
        feat=feat_hlt_std,
        mask=hlt_mask,
        jet_extra_count=extra_count,
        true_merged=true_merged,
        idx=test_idx,
        batch_size=args.batch_size,
        device=device,
    )
    print("\nTest (hidden token GT, eval-only):")
    print(f"  Token AUC:      {test_metrics['auc']:.4f}")
    print(f"  Token AP:       {test_metrics['ap']:.4f}")
    print(f"  Token Acc@0.5:  {test_metrics['acc']:.4f}")
    print(f"  Precision@0.5:  {test_metrics['precision']:.4f}")
    print(f"  Recall@0.5:     {test_metrics['recall']:.4f}")
    print(f"  Jet extra MAE:  {test_metrics['mae_extra']:.4f}")
    print(f"  High-conf cov:  {test_metrics['conf_cov']:.4f}")
    print(f"  High-conf acc:  {test_metrics['conf_acc']:.4f}")

    # ROC plot
    fpr, tpr, _ = roc_curve(test_metrics["y_true"], test_metrics["y_prob"])
    plt.figure(figsize=(8, 6))
    plt.plot(tpr, fpr, label=f"Weak merge predictor (AUC={test_metrics['auc']:.3f})", color="darkslateblue", linewidth=2)
    plt.yscale("log")
    plt.ylim(1e-4, 1.0)
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_root / "results_token_merge_roc.png", dpi=300)
    plt.close()

    # Save artifacts
    torch.save(
        {
            "model": model.state_dict(),
            "val_auc": best_val_auc,
            "config": vars(args),
            "hlt_stats": hlt_stats,
            "feature_means": means,
            "feature_stds": stds,
        },
        save_root / "weak_merge_predictor.pt",
    )
    np.savez(
        save_root / "results.npz",
        val_auc=best_val_auc,
        test_auc=test_metrics["auc"],
        test_ap=test_metrics["ap"],
        test_acc=test_metrics["acc"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        test_extra_mae=test_metrics["mae_extra"],
        test_conf_cov=test_metrics["conf_cov"],
        test_conf_acc=test_metrics["conf_acc"],
        fpr=fpr,
        tpr=tpr,
        test_y_true=test_metrics["y_true"],
        test_y_prob=test_metrics["y_prob"],
        test_idx=test_idx.astype(np.int32),
    )
    with open(save_root / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save mapping for future experiments (explicitly not used in training).
    np.savez_compressed(
        save_root / "hlt_with_mapping_eval_only.npz",
        hlt_const=hlt_const.astype(np.float32),
        hlt_mask=hlt_mask.astype(bool),
        origin_counts=origin_counts.astype(np.int16),
        true_merged=true_merged.astype(np.int8),
        off_const=off_const.astype(np.float32),
        off_mask=off_mask.astype(bool),
        labels=labels.astype(np.int8),
        train_idx=train_idx.astype(np.int32),
        val_idx=val_idx.astype(np.int32),
        test_idx=test_idx.astype(np.int32),
    )

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()

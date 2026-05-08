#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a top-tagging offline fused-student teacher on fused soft targets.

This is the Strategy-2 precursor:
  fused blend scores -> distilled offline teacher checkpoint (teacher.pt)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from unmerge_correct_hlt import (
    RANDOM_SEED,
    compute_features,
    get_stats,
    standardize,
    ParticleTransformer,
    get_scheduler,
)

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as tkd


def _fpr_at_target_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    """Numerically stable FPR@target-TPR helper (no external module dependency)."""
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    target = float(np.clip(target_tpr, 0.0, 1.0))
    idx = int(np.searchsorted(tpr, target, side="left"))
    if idx <= 0:
        return float(fpr[0])
    if idx >= len(tpr):
        return float(fpr[-1])
    t0, t1 = float(tpr[idx - 1]), float(tpr[idx])
    f0, f1 = float(fpr[idx - 1]), float(fpr[idx])
    if t1 <= t0:
        return float(f1)
    w = (target - t0) / (t1 - t0)
    return float(f0 + w * (f1 - f0))


class SoftTargetJetDataset(Dataset):
    def __init__(self, feat: np.ndarray, mask: np.ndarray, y_hard: np.ndarray, y_soft: np.ndarray):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y_hard = torch.tensor(y_hard.astype(np.float32), dtype=torch.float32)
        self.y_soft = torch.tensor(y_soft.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.feat.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat": self.feat[i],
            "mask": self.mask[i],
            "y_hard": self.y_hard[i],
            "y_soft": self.y_soft[i],
        }


def _build_weighted_sampler(sample_weight: np.ndarray | None) -> WeightedRandomSampler | None:
    if sample_weight is None:
        return None
    sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if sw.size == 0:
        return None
    sw = np.clip(sw, 1e-8, None)
    sw = sw / float(sw.mean())
    return WeightedRandomSampler(
        weights=torch.as_tensor(sw, dtype=torch.double),
        num_samples=int(sw.shape[0]),
        replacement=True,
    )


@torch.no_grad()
def _eval_soft(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    n = 0
    probs_all = []
    y_all = []
    for batch in loader:
        x = batch["feat"].to(device)
        m = batch["mask"].to(device)
        y_h = batch["y_hard"].to(device)
        y_s = batch["y_soft"].to(device)
        logits = model(x, m).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_s, reduction="mean")
        bs = int(x.shape[0])
        loss_sum += float(loss.item()) * bs
        n += bs
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
        y_all.append(y_h.detach().cpu().numpy().astype(np.int64))

    probs = np.concatenate(probs_all, axis=0) if len(probs_all) > 0 else np.zeros((0,), dtype=np.float32)
    y = np.concatenate(y_all, axis=0) if len(y_all) > 0 else np.zeros((0,), dtype=np.int64)
    if probs.size > 0 and np.unique(y).size > 1:
        auc = float(roc_auc_score(y, probs))
        fpr, tpr, _ = roc_curve(y, probs)
        fpr50 = float(_fpr_at_target_tpr(fpr, tpr, 0.50))
    else:
        auc = float("nan")
        fpr50 = float("nan")
    return (loss_sum / max(n, 1), auc, fpr50, probs, y)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train top-tagging offline fused student")
    ap.add_argument("--train_path", type=str, default="./data/train_quarter.h5")
    ap.add_argument("--save_dir", type=Path, required=True)
    ap.add_argument("--run_name", type=str, default="toptag_offline_fused_student")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    ap.add_argument("--n_train_jets", type=int, default=600000)
    ap.add_argument("--n_train_split", type=int, default=150000)
    ap.add_argument("--n_val_split", type=int, default=150000)
    ap.add_argument("--n_test_split", type=int, default=300000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--use_train_weights", action="store_true")

    ap.add_argument("--fused_targets_npz", type=Path, required=True)
    ap.add_argument("--target_key", type=str, default="probs_fused_overall")
    ap.add_argument(
        "--target_split_scheme",
        type=str,
        default="train_val_test",
        choices=["train_val_test", "fit_ref_test"],
    )
    args = ap.parse_args()

    tkd.set_seed(int(args.seed))
    save_root = args.save_dir.expanduser().resolve() / str(args.run_name)
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (torch.cuda.is_available() or str(args.device).startswith("cpu")) else "cpu")

    # Load raw inputs
    train_input = Path(args.train_path)
    if train_input.is_dir():
        train_files = sorted(train_input.glob("*.h5"))
    else:
        train_files = [train_input]
    all_const_full, all_labels_full, all_train_w_full = tkd.load_raw_constituents_labels_weights_from_h5(
        files=train_files,
        max_jets=int(args.n_train_jets),
        max_constits=int(args.max_constits),
        use_train_weights=bool(args.use_train_weights),
    )
    if int(args.offset_jets) > 0:
        off = int(args.offset_jets)
        all_const_full = all_const_full[off:]
        all_labels_full = all_labels_full[off:]
        all_train_w_full = all_train_w_full[off:]

    labels = all_labels_full.astype(np.int64)
    const_off = all_const_full.astype(np.float32)
    masks_off = (const_off[:, :, 0] > 0).astype(bool)
    train_weight = np.clip(all_train_w_full.astype(np.float32), 1e-8, None)

    feat_off = compute_features(const_off, masks_off)
    idx = np.arange(len(labels))
    total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > len(idx):
        raise ValueError(
            f"Requested split exceeds available jets: {total_need} > {len(idx)}"
        )
    if total_need < len(idx):
        idx_use, _ = train_test_split(
            idx,
            train_size=total_need,
            random_state=int(args.seed),
            stratify=labels[idx],
        )
    else:
        idx_use = idx
    train_idx, rem_idx = train_test_split(
        idx_use,
        train_size=int(args.n_train_split),
        random_state=int(args.seed),
        stratify=labels[idx_use],
    )
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=int(args.n_val_split),
        test_size=int(args.n_test_split),
        random_state=int(args.seed),
        stratify=labels[rem_idx],
    )

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)

    # Load fused targets
    fused_npz = args.fused_targets_npz.expanduser().resolve()
    arr = np.load(fused_npz)
    kprefix = str(args.target_key).strip()
    if str(args.target_split_scheme).lower() == "fit_ref_test":
        ktr = f"{kprefix}_fit"
        kva = f"{kprefix}_ref"
        kte = f"{kprefix}_test"
    else:
        ktr = f"{kprefix}_train"
        kva = f"{kprefix}_val"
        kte = f"{kprefix}_test"
    for k in (ktr, kva, kte):
        if k not in arr:
            raise KeyError(f"Missing target key `{k}` in {fused_npz}")
    y_soft_tr = np.asarray(arr[ktr], dtype=np.float32).reshape(-1)
    y_soft_va = np.asarray(arr[kva], dtype=np.float32).reshape(-1)
    y_soft_te = np.asarray(arr[kte], dtype=np.float32).reshape(-1)
    if int(y_soft_tr.shape[0]) != int(len(train_idx)):
        raise ValueError(f"Train target length mismatch: {y_soft_tr.shape[0]} vs {len(train_idx)}")
    if int(y_soft_va.shape[0]) != int(len(val_idx)):
        raise ValueError(f"Val target length mismatch: {y_soft_va.shape[0]} vs {len(val_idx)}")
    if int(y_soft_te.shape[0]) != int(len(test_idx)):
        raise ValueError(f"Test target length mismatch: {y_soft_te.shape[0]} vs {len(test_idx)}")

    ds_tr = SoftTargetJetDataset(
        feat_off_std[train_idx], masks_off[train_idx], labels[train_idx], y_soft_tr
    )
    ds_va = SoftTargetJetDataset(
        feat_off_std[val_idx], masks_off[val_idx], labels[val_idx], y_soft_va
    )
    ds_te = SoftTargetJetDataset(
        feat_off_std[test_idx], masks_off[test_idx], labels[test_idx], y_soft_te
    )

    tr_w = train_weight[train_idx].astype(np.float32)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(args.batch_size),
        sampler=_build_weighted_sampler(tr_w) if bool(args.use_train_weights) else None,
        shuffle=False if bool(args.use_train_weights) else True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    cfg = tkd._deepcopy_config()
    model = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    sch = get_scheduler(opt, int(args.warmup_epochs), int(args.epochs))

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for ep in range(int(args.epochs)):
        model.train()
        tr_loss_sum = 0.0
        tr_n = 0
        for batch in dl_tr:
            x = batch["feat"].to(device)
            m = batch["mask"].to(device)
            y_s = batch["y_soft"].to(device)
            logits = model(x, m).view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_s)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = int(x.shape[0])
            tr_loss_sum += float(loss.item()) * bs
            tr_n += bs
        sch.step()

        tr_loss = tr_loss_sum / max(tr_n, 1)
        va_loss, va_auc, va_fpr50, _, _ = _eval_soft(model, dl_va, device)

        if va_loss < best_val_loss:
            best_val_loss = float(va_loss)
            best_epoch = int(ep + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"FusedStudent ep {ep+1}: train_loss={tr_loss:.6f}, "
                f"val_loss={va_loss:.6f}, val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, "
                f"best_val_loss={best_val_loss:.6f}@{best_epoch}"
            )
        if no_improve >= int(args.patience):
            print(f"Early stopping fused student at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_auc, te_fpr50, te_probs, te_labels = _eval_soft(model, dl_te, device)
    va_loss, va_auc, va_fpr50, va_probs, va_labels = _eval_soft(model, dl_va, device)

    torch.save(
        {
            "model": model.state_dict(),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "val_auc": float(va_auc),
            "val_fpr50": float(va_fpr50),
            "test_auc": float(te_auc),
            "test_fpr50": float(te_fpr50),
        },
        save_root / "offline_fused_student.pt",
    )
    # Save teacher.pt alias for drop-in Stage-A teacher override.
    torch.save(
        {
            "model": model.state_dict(),
            "val_auc": float(va_auc),
            "val_fpr50": float(va_fpr50),
            "best_val_loss": float(best_val_loss),
        },
        save_root / "teacher.pt",
    )
    np.savez_compressed(
        save_root / "offline_fused_student_scores.npz",
        labels_val=va_labels.astype(np.float32),
        probs_val=va_probs.astype(np.float32),
        labels_test=te_labels.astype(np.float32),
        probs_test=te_probs.astype(np.float32),
        y_soft_val=y_soft_va.astype(np.float32),
        y_soft_test=y_soft_te.astype(np.float32),
    )

    summary = {
        "run_name": str(args.run_name),
        "save_root": str(save_root),
        "seed": int(args.seed),
        "fused_targets_npz": str(fused_npz),
        "target_key": str(args.target_key),
        "target_split_scheme": str(args.target_split_scheme),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "val_loss": float(va_loss),
        "val_auc": float(va_auc),
        "val_fpr50": float(va_fpr50),
        "test_loss": float(te_loss),
        "test_auc": float(te_auc),
        "test_fpr50": float(te_fpr50),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    (save_root / "offline_fused_student_summary.json").write_text(json.dumps(summary, indent=2))
    (save_root / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    print("============================================================")
    print("Top-tagging offline fused student done")
    print("============================================================")
    print(f"Run: {save_root}")
    print(f"Val AUC/FPR50:  {va_auc:.6f} / {va_fpr50:.6f}")
    print(f"Test AUC/FPR50: {te_auc:.6f} / {te_fpr50:.6f}")
    print(f"Saved: {save_root / 'teacher.pt'}")


if __name__ == "__main__":
    main()

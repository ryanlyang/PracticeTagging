#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oracle ablation for sparse additive recovery:
- Train teacher (offline) and baseline (HLT)
- Build oracle-added datasets with three token-selection methods:
  1) Integrated Gradients (IG)
  2) Leave-One-Token-Out delta logit (LOTO)
  3) Greedy insertion (approximate, feature-space)
- Train/evaluate a top tagger on each added dataset and compare recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
)
from unmerge_correct_hlt import (
    JetDataset,
    ParticleTransformer,
    compute_features,
    eval_classifier,
    get_scheduler,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
    train_classifier,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fpr_at_target_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    target = float(np.clip(target_tpr, 0.0, 1.0))
    idx = int(np.argmin(np.abs(tpr - target)))
    return float(fpr[idx])


def _model_logits(model: torch.nn.Module, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = model(feat, mask)
    if isinstance(out, tuple):
        out = out[0]
    return out.squeeze(-1)


def _train_single_view_classifier_auc(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> torch.nn.Module:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(
        opt,
        int(train_cfg["warmup_epochs"]),
        int(train_cfg["epochs"]),
    )

    best_val_auc = float("-inf")
    best_state = None
    no_improve = 0

    for ep in tqdm(range(int(train_cfg["epochs"])), desc=name):
        _, tr_auc = train_classifier(model, train_loader, opt, device)
        va_auc, va_preds, va_labs = eval_classifier(model, val_loader, device)
        va_fpr, va_tpr, _ = roc_curve(va_labs, va_preds)
        va_fpr50 = fpr_at_target_tpr(va_fpr, va_tpr, 0.50)
        sch.step()

        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, best_auc={best_val_auc:.4f}"
            )
        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _wrap_phi_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def _compute_novel_mask(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    dr_match: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    n, t, _ = const_off.shape
    novel = np.zeros((n, t), dtype=bool)
    dr_match = float(dr_match)
    for s in tqdm(range(0, n, chunk_size), desc="NovelMask"):
        e = min(n, s + chunk_size)
        off_eta = const_off[s:e, :, 1]
        off_phi = const_off[s:e, :, 2]
        hlt_eta = const_hlt[s:e, :, 1]
        hlt_phi = const_hlt[s:e, :, 2]

        deta = off_eta[:, :, None] - hlt_eta[:, None, :]
        dphi = _wrap_phi_np(off_phi[:, :, None] - hlt_phi[:, None, :])
        dr = np.sqrt(deta * deta + dphi * dphi)

        pair_valid = mask_off[s:e, :, None] & mask_hlt[s:e, None, :]
        dr = np.where(pair_valid, dr, np.inf)
        min_dr = np.min(dr, axis=2)

        matched = min_dr < dr_match
        novel[s:e] = mask_off[s:e] & (~matched)
    return novel


def _build_candidate_pool_by_pt(
    const_off: np.ndarray,
    novel_mask: np.ndarray,
    pool_size: int,
) -> np.ndarray:
    n, t, _ = const_off.shape
    out = np.full((n, pool_size), -1, dtype=np.int64)
    pt = const_off[:, :, 0]
    for i in range(n):
        idx = np.where(novel_mask[i])[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(-pt[i, idx])]
        take = min(pool_size, order.size)
        out[i, :take] = order[:take]
    return out


def _build_top_order_from_scores(
    scores: np.ndarray,
    novel_mask: np.ndarray,
    max_k: int,
) -> np.ndarray:
    n, t = scores.shape
    order = np.full((n, max_k), -1, dtype=np.int64)
    for i in range(n):
        idx = np.where(novel_mask[i])[0]
        if idx.size == 0:
            continue
        sc = scores[i, idx]
        sidx = idx[np.argsort(-sc)]
        take = min(max_k, sidx.size)
        order[i, :take] = sidx[:take]
    return order


def _order_to_mask(order: np.ndarray, k: int, t: int) -> np.ndarray:
    n = order.shape[0]
    out = np.zeros((n, t), dtype=bool)
    kk = min(k, order.shape[1])
    if kk <= 0:
        return out
    idx = order[:, :kk]
    rows, cols = np.where(idx >= 0)
    tok = idx[rows, cols]
    out[rows, tok] = True
    return out


def _build_augmented_constituents(
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_off: np.ndarray,
    add_mask_off: np.ndarray,
    aug_max_constits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = const_hlt.shape[0]
    out_const = np.zeros((n, aug_max_constits, 4), dtype=np.float32)
    out_mask = np.zeros((n, aug_max_constits), dtype=bool)

    for i in range(n):
        h = const_hlt[i, mask_hlt[i]]
        a = const_off[i, add_mask_off[i]]
        if h.size == 0 and a.size == 0:
            continue
        merged = np.concatenate([h, a], axis=0)
        if merged.shape[0] > 1:
            ord_idx = np.argsort(-merged[:, 0])
            merged = merged[ord_idx]
        keep = min(aug_max_constits, merged.shape[0])
        out_const[i, :keep] = merged[:keep]
        out_mask[i, :keep] = True
    return out_const, out_mask


def _compute_ig_scores(
    model: torch.nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    ig_steps: int,
) -> np.ndarray:
    model.eval()
    n, t, f = feat.shape
    out = np.zeros((n, t), dtype=np.float32)
    steps = max(1, int(ig_steps))

    for s in tqdm(range(0, n, batch_size), desc="IG"):
        e = min(n, s + batch_size)
        xb = torch.tensor(feat[s:e], dtype=torch.float32, device=device)
        mb = torch.tensor(mask[s:e], dtype=torch.bool, device=device)
        yb = torch.tensor(labels[s:e], dtype=torch.float32, device=device)
        sign = (2.0 * yb - 1.0).view(-1)

        baseline = torch.zeros_like(xb)
        total_grad = torch.zeros_like(xb)

        for st in range(1, steps + 1):
            alpha = float(st) / float(steps)
            z = (baseline + alpha * (xb - baseline)).detach().requires_grad_(True)
            logits = _model_logits(model, z, mb)
            objective = (logits * sign).sum()
            model.zero_grad(set_to_none=True)
            objective.backward()
            total_grad += z.grad.detach()

        ig = (xb - baseline) * (total_grad / float(steps))
        tok = ig.abs().sum(dim=2)
        tok = torch.where(mb, tok, torch.zeros_like(tok))
        out[s:e] = tok.detach().cpu().numpy().astype(np.float32)

    return out


def _compute_loto_scores(
    model: torch.nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    candidate_pool_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    n, t, _ = feat.shape
    pool = candidate_pool_idx.shape[1]
    scores = np.full((n, t), -1e9, dtype=np.float32)

    for s in tqdm(range(0, n, batch_size), desc="LOTO"):
        e = min(n, s + batch_size)
        bsz = e - s
        xb = torch.tensor(feat[s:e], dtype=torch.float32, device=device)
        mb = torch.tensor(mask[s:e], dtype=torch.bool, device=device)
        yb = torch.tensor(labels[s:e], dtype=torch.float32, device=device)
        sign = (2.0 * yb - 1.0).view(-1)

        with torch.no_grad():
            base_obj = _model_logits(model, xb, mb) * sign

        cand = candidate_pool_idx[s:e]  # [B, pool]

        for p in range(pool):
            tok_idx = cand[:, p]
            valid = tok_idx >= 0
            if not np.any(valid):
                continue
            rows = np.where(valid)[0]
            cols = tok_idx[valid]

            x_mod = xb.clone()
            m_mod = mb.clone()
            rows_t = torch.tensor(rows, dtype=torch.long, device=device)
            cols_t = torch.tensor(cols, dtype=torch.long, device=device)
            x_mod[rows_t, cols_t, :] = 0.0
            m_mod[rows_t, cols_t] = False

            with torch.no_grad():
                obj_mod = _model_logits(model, x_mod, m_mod) * sign
            delta = (base_obj - obj_mod).detach().cpu().numpy()

            scores[s + rows, cols] = delta[rows]

    return scores


def _compute_greedy_insertion_order(
    model: torch.nn.Module,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    feat_off: np.ndarray,
    labels: np.ndarray,
    candidate_pool_idx: np.ndarray,
    k_max: int,
    gain_min: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Approximate greedy insertion in feature-space:
    - Start from HLT feature set.
    - Candidate tokens are offline token features from candidate_pool_idx.
    - Iteratively add candidate with largest immediate gain in true-class margin.
    """
    model.eval()
    n, t, f = feat_hlt.shape
    pool = candidate_pool_idx.shape[1]
    out_order = np.full((n, k_max), -1, dtype=np.int64)

    for s in tqdm(range(0, n, batch_size), desc="Greedy"):
        e = min(n, s + batch_size)
        bsz = e - s
        x_cur = torch.tensor(feat_hlt[s:e], dtype=torch.float32, device=device)
        m_cur = torch.tensor(mask_hlt[s:e], dtype=torch.bool, device=device)
        yb = torch.tensor(labels[s:e], dtype=torch.float32, device=device)
        sign = (2.0 * yb - 1.0).view(-1)

        cand_idx = candidate_pool_idx[s:e].copy()  # [B,pool]
        cand_avail = cand_idx >= 0

        feat_off_b = feat_off[s:e]
        cand_feat = np.zeros((bsz, pool, f), dtype=np.float32)
        vr, vp = np.where(cand_avail)
        cand_feat[vr, vp] = feat_off_b[vr, cand_idx[vr, vp]]

        for kk in range(k_max):
            empty = (~m_cur).any(dim=1)
            empty_np = empty.detach().cpu().numpy()
            if not np.any(empty_np):
                break

            ins_pos = (~m_cur).to(torch.int64).argmax(dim=1)
            with torch.no_grad():
                base_obj = _model_logits(model, x_cur, m_cur) * sign

            gains = np.full((bsz, pool), -1e9, dtype=np.float32)

            for p in range(pool):
                valid_np = cand_avail[:, p] & empty_np
                if not np.any(valid_np):
                    continue

                rows = np.where(valid_np)[0]
                rows_t = torch.tensor(rows, dtype=torch.long, device=device)
                cols_t = ins_pos[rows_t]

                x_mod = x_cur.clone()
                m_mod = m_cur.clone()
                cf_t = torch.tensor(cand_feat[rows, p], dtype=torch.float32, device=device)
                x_mod[rows_t, cols_t, :] = cf_t
                m_mod[rows_t, cols_t] = True

                with torch.no_grad():
                    obj_mod = _model_logits(model, x_mod, m_mod) * sign
                gain = (obj_mod - base_obj).detach().cpu().numpy()
                gains[rows, p] = gain[rows]

            best_p = np.argmax(gains, axis=1)
            best_gain = gains[np.arange(bsz), best_p]
            choose = (
                empty_np
                & (best_gain > float(gain_min))
                & cand_avail[np.arange(bsz), best_p]
            )
            if not np.any(choose):
                break

            choose_rows = np.where(choose)[0]
            for r in choose_rows:
                p = int(best_p[r])
                tok = int(cand_idx[r, p])
                out_order[s + r, kk] = tok
                # apply insertion to current state
                pos = int(ins_pos[r].item())
                x_cur[r, pos, :] = torch.tensor(cand_feat[r, p], dtype=torch.float32, device=device)
                m_cur[r, pos] = True
                cand_avail[r, p] = False

    return out_order


def _build_loaders(
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pin = torch.cuda.is_available()
    ds_tr = JetDataset(feat[train_idx], mask[train_idx], labels[train_idx])
    ds_va = JetDataset(feat[val_idx], mask[val_idx], labels[val_idx])
    ds_te = JetDataset(feat[test_idx], mask[test_idx], labels[test_idx])

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return dl_tr, dl_va, dl_te


def _train_eval_added(
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg_training: Dict,
    cfg_model: Dict,
    device: torch.device,
    num_workers: int,
    name: str,
) -> Dict[str, float]:
    bs = int(cfg_training["batch_size"])
    dl_tr, dl_va, dl_te = _build_loaders(
        feat, mask, labels, train_idx, val_idx, test_idx, bs, num_workers
    )
    model = ParticleTransformer(input_dim=7, **cfg_model).to(device)
    model = _train_single_view_classifier_auc(
        model, dl_tr, dl_va, device, cfg_training, name=name
    )
    auc, preds, labs = eval_classifier(model, dl_te, device)
    fpr, tpr, _ = roc_curve(labs, preds)
    return {
        "auc": float(auc),
        "fpr30": float(fpr_at_target_tpr(fpr, tpr, 0.30)),
        "fpr50": float(fpr_at_target_tpr(fpr, tpr, 0.50)),
    }


def _parse_int_list(x: str) -> List[int]:
    vals = []
    for part in str(x).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    vals = sorted(list(set(vals)))
    if len(vals) == 0:
        raise ValueError("No valid integer values parsed.")
    return vals


def _parse_str_list(x: str) -> List[str]:
    vals = []
    for part in str(x).split(","):
        p = part.strip().lower()
        if p:
            vals.append(p)
    vals = list(dict.fromkeys(vals))
    if len(vals) == 0:
        raise ValueError("No valid methods parsed.")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle sparse-addition method comparison")
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="checkpoints/oracle_added_methods")
    parser.add_argument("--run_name", type=str, default="oracle_added_methods_run")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--n_train_jets", type=int, default=95000)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--aug_max_constits", type=int, default=-1)
    parser.add_argument("--n_train_split", type=int, default=35000)
    parser.add_argument("--n_val_split", type=int, default=10000)
    parser.add_argument("--n_test_split", type=int, default=50000)

    parser.add_argument("--merge_radius", type=float, default=0.01)
    parser.add_argument("--eff_plateau_barrel", type=float, default=0.98)
    parser.add_argument("--eff_plateau_endcap", type=float, default=0.94)
    parser.add_argument("--smear_a", type=float, default=0.35)
    parser.add_argument("--smear_b", type=float, default=0.012)
    parser.add_argument("--smear_c", type=float, default=0.08)

    parser.add_argument("--methods", type=str, default="ig,loto,greedy")
    parser.add_argument("--k_values", type=str, default="8,12,16,20")

    parser.add_argument("--novel_dr_match", type=float, default=0.02)
    parser.add_argument("--ig_steps", type=int, default=8)
    parser.add_argument("--loto_pool", type=int, default=24)
    parser.add_argument("--greedy_pool", type=int, default=12)
    parser.add_argument("--greedy_gain_min", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=-1.0)
    parser.add_argument("--weight_decay", type=float, default=-1.0)
    parser.add_argument("--warmup_epochs", type=int, default=-1)

    args = parser.parse_args()
    set_seed(int(args.seed))

    methods = _parse_str_list(args.methods)
    valid_methods = {"ig", "loto", "greedy"}
    bad = [m for m in methods if m not in valid_methods]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}. Valid: {sorted(valid_methods)}")

    k_values = _parse_int_list(args.k_values)
    k_max_global = int(max(k_values))

    cfg = deepcopy(BASE_CONFIG)
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    if int(args.batch_size) > 0:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if int(args.epochs) > 0:
        cfg["training"]["epochs"] = int(args.epochs)
    if int(args.patience) > 0:
        cfg["training"]["patience"] = int(args.patience)
    if float(args.lr) > 0:
        cfg["training"]["lr"] = float(args.lr)
    if float(args.weight_decay) > 0:
        cfg["training"]["weight_decay"] = float(args.weight_decay)
    if int(args.warmup_epochs) >= 0:
        cfg["training"]["warmup_epochs"] = int(args.warmup_epochs)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Save dir: {save_root}")
    print(f"Methods: {methods}")
    print(f"K values: {k_values}")

    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = int(args.offset_jets + args.n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=int(args.max_constits),
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}"
        )

    const_raw = all_const_full[args.offset_jets : args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets : args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (
        const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"])
    )
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, _, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )

    print("Computing base features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    n = len(labels)
    idx = np.arange(n)
    n_train_split = int(args.n_train_split)
    n_val_split = int(args.n_val_split)
    n_test_split = int(args.n_test_split)
    total_need = n_train_split + n_val_split + n_test_split
    if total_need > n:
        raise ValueError(f"Split counts exceed available jets: {total_need} > {n}")
    if total_need < n:
        idx_use, _ = train_test_split(
            idx,
            train_size=total_need,
            random_state=int(args.seed),
            stratify=labels,
        )
    else:
        idx_use = idx

    train_idx, rem_idx = train_test_split(
        idx_use,
        train_size=n_train_split,
        random_state=int(args.seed),
        stratify=labels[idx_use],
    )
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=n_val_split,
        test_size=n_test_split,
        random_state=int(args.seed),
        stratify=labels[rem_idx],
    )

    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    print("Computing novelty mask (offline tokens not represented by HLT)...")
    novel_mask = _compute_novel_mask(
        const_off,
        masks_off,
        hlt_const,
        hlt_mask,
        dr_match=float(args.novel_dr_match),
        chunk_size=2048,
    )

    bs = int(cfg["training"]["batch_size"])
    nw = int(args.num_workers)

    # Train teacher / baseline once.
    print("\n" + "=" * 70)
    print("Training Teacher (Offline)")
    print("=" * 70)
    dl_train_off, dl_val_off, dl_test_off = _build_loaders(
        feat_off_std, masks_off, labels, train_idx, val_idx, test_idx, bs, nw
    )
    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = _train_single_view_classifier_auc(
        teacher, dl_train_off, dl_val_off, device, cfg["training"], name="Teacher"
    )
    auc_teacher, preds_teacher, labs_test = eval_classifier(teacher, dl_test_off, device)
    fpr_t, tpr_t, _ = roc_curve(labs_test, preds_teacher)
    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.30)
    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)

    print("\n" + "=" * 70)
    print("Training Baseline (HLT)")
    print("=" * 70)
    dl_train_hlt, dl_val_hlt, dl_test_hlt = _build_loaders(
        feat_hlt_std, hlt_mask, labels, train_idx, val_idx, test_idx, bs, nw
    )
    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = _train_single_view_classifier_auc(
        baseline, dl_train_hlt, dl_val_hlt, device, cfg["training"], name="Baseline-HLT"
    )
    auc_baseline, preds_baseline, labs_b = eval_classifier(baseline, dl_test_hlt, device)
    assert np.array_equal(labs_test.astype(np.float32), labs_b.astype(np.float32))
    fpr_b, tpr_b, _ = roc_curve(labs_b, preds_baseline)
    fpr30_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.30)
    fpr50_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.50)

    # Precompute scores/orders per method.
    score_cache: Dict[str, np.ndarray] = {}
    order_cache: Dict[str, np.ndarray] = {}

    if "ig" in methods or "greedy" in methods:
        print("\nComputing IG scores...")
        ig_scores = _compute_ig_scores(
            model=teacher,
            feat=feat_off_std,
            mask=masks_off,
            labels=labels,
            device=device,
            batch_size=bs,
            ig_steps=int(args.ig_steps),
        )
        score_cache["ig"] = ig_scores
        order_cache["ig"] = _build_top_order_from_scores(ig_scores, novel_mask, k_max_global)

    if "loto" in methods:
        print("\nComputing LOTO scores...")
        loto_pool_idx = _build_candidate_pool_by_pt(
            const_off=const_off,
            novel_mask=novel_mask,
            pool_size=int(args.loto_pool),
        )
        loto_scores = _compute_loto_scores(
            model=teacher,
            feat=feat_off_std,
            mask=masks_off,
            labels=labels,
            candidate_pool_idx=loto_pool_idx,
            device=device,
            batch_size=bs,
        )
        score_cache["loto"] = loto_scores
        order_cache["loto"] = _build_top_order_from_scores(loto_scores, novel_mask, k_max_global)

    if "greedy" in methods:
        print("\nComputing Greedy insertion order...")
        # Greedy candidate pool from strongest IG candidates among novel tokens.
        ig_scores = score_cache.get("ig")
        if ig_scores is None:
            raise RuntimeError("IG scores missing for greedy candidate pool.")
        greedy_pool = int(args.greedy_pool)
        greedy_cand_idx = np.full((n, greedy_pool), -1, dtype=np.int64)
        for i in range(n):
            idxv = np.where(novel_mask[i])[0]
            if idxv.size == 0:
                continue
            sc = ig_scores[i, idxv]
            ordv = idxv[np.argsort(-sc)]
            take = min(greedy_pool, ordv.size)
            greedy_cand_idx[i, :take] = ordv[:take]

        order_cache["greedy"] = _compute_greedy_insertion_order(
            model=teacher,
            feat_hlt=feat_hlt_std,
            mask_hlt=hlt_mask,
            feat_off=feat_off_std,
            labels=labels,
            candidate_pool_idx=greedy_cand_idx,
            k_max=k_max_global,
            gain_min=float(args.greedy_gain_min),
            device=device,
            batch_size=bs,
        )

    if int(args.aug_max_constits) > 0:
        aug_max_constits = int(args.aug_max_constits)
    else:
        aug_max_constits = int(args.max_constits) + int(k_max_global)

    results_rows: List[Dict[str, float]] = []

    print("\n" + "=" * 70)
    print("Training oracle-added models")
    print("=" * 70)
    for method in methods:
        if method not in order_cache:
            raise RuntimeError(f"Missing token order for method: {method}")
        token_order = order_cache[method]
        for k in k_values:
            print(f"\n[Method={method}] K={k}")
            add_mask = _order_to_mask(token_order, k=k, t=const_off.shape[1])
            aug_const, aug_mask = _build_augmented_constituents(
                const_hlt=hlt_const,
                mask_hlt=hlt_mask,
                const_off=const_off,
                add_mask_off=add_mask,
                aug_max_constits=aug_max_constits,
            )
            feat_aug = compute_features(aug_const, aug_mask)
            feat_aug_std = standardize(feat_aug, aug_mask, means, stds)

            m = _train_eval_added(
                feat=feat_aug_std,
                mask=aug_mask,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                cfg_training=cfg["training"],
                cfg_model=cfg["model"],
                device=device,
                num_workers=nw,
                name=f"Added-{method}-K{k}",
            )

            rec_auc = float("nan")
            rec_fpr50 = float("nan")
            if abs(auc_teacher - auc_baseline) > 1e-12:
                rec_auc = (m["auc"] - auc_baseline) / (auc_teacher - auc_baseline)
            if abs(fpr50_baseline - fpr50_teacher) > 1e-12:
                rec_fpr50 = (fpr50_baseline - m["fpr50"]) / (fpr50_baseline - fpr50_teacher)

            row = {
                "method": method,
                "k": int(k),
                "auc": float(m["auc"]),
                "fpr30": float(m["fpr30"]),
                "fpr50": float(m["fpr50"]),
                "recovery_auc": float(rec_auc),
                "recovery_fpr50": float(rec_fpr50),
                "mean_added_tokens": float(add_mask.sum(axis=1).mean()),
            }
            results_rows.append(row)
            print(
                f"AUC={row['auc']:.4f} FPR30={row['fpr30']:.6f} FPR50={row['fpr50']:.6f} "
                f"Recovery(AUC)={row['recovery_auc']:.3f} Recovery(FPR50)={row['recovery_fpr50']:.3f}"
            )

    summary = {
        "teacher": {
            "auc": float(auc_teacher),
            "fpr30": float(fpr30_teacher),
            "fpr50": float(fpr50_teacher),
        },
        "baseline_hlt": {
            "auc": float(auc_baseline),
            "fpr30": float(fpr30_baseline),
            "fpr50": float(fpr50_baseline),
        },
        "results": results_rows,
    }

    with open(save_root / "oracle_method_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(save_root / "oracle_method_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method",
            "k",
            "auc",
            "fpr30",
            "fpr50",
            "recovery_auc",
            "recovery_fpr50",
            "mean_added_tokens",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results_rows:
            w.writerow(r)

    # Simple recovery-vs-K plots.
    if len(results_rows) > 0:
        plt.figure(figsize=(7, 5))
        for method in methods:
            xs = [r["k"] for r in results_rows if r["method"] == method]
            ys = [r["recovery_auc"] for r in results_rows if r["method"] == method]
            if xs:
                plt.plot(xs, ys, marker="o", label=method)
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("K (max added tokens)")
        plt.ylabel("Recovery (AUC)")
        plt.title("AUC Recovery vs K")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_root / "recovery_auc_vs_k.png", dpi=220)
        plt.close()

        plt.figure(figsize=(7, 5))
        for method in methods:
            xs = [r["k"] for r in results_rows if r["method"] == method]
            ys = [r["recovery_fpr50"] for r in results_rows if r["method"] == method]
            if xs:
                plt.plot(xs, ys, marker="o", label=method)
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("K (max added tokens)")
        plt.ylabel("Recovery (FPR@50)")
        plt.title("FPR@50 Recovery vs K")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_root / "recovery_fpr50_vs_k.png", dpi=220)
        plt.close()

    run_cfg = {
        "args": vars(args),
        "training": cfg["training"],
        "model": cfg["model"],
        "hlt_effects": cfg["hlt_effects"],
        "split": {
            "n_train_split": int(len(train_idx)),
            "n_val_split": int(len(val_idx)),
            "n_test_split": int(len(test_idx)),
        },
    }
    with open(save_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Teacher AUC={auc_teacher:.4f} FPR50={fpr50_teacher:.6f}")
    print(f"Baseline AUC={auc_baseline:.4f} FPR50={fpr50_baseline:.6f}")
    for r in results_rows:
        print(
            f"[{r['method']}] K={int(r['k'])}: AUC={r['auc']:.4f}, "
            f"FPR50={r['fpr50']:.6f}, RecAUC={r['recovery_auc']:.3f}, "
            f"RecFPR50={r['recovery_fpr50']:.3f}"
        )
    print(f"Saved outputs to: {save_root}")


if __name__ == "__main__":
    main()

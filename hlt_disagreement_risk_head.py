#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone HLT disagreement-risk experiment (no reconstructor training).

Pipeline:
1) Build pseudo-HLT from offline jets.
2) Train teacher (offline features) and baseline (HLT features).
3) Define risk target:
      risk = 1{ y=0 AND p_hlt > t_hlt@TPR AND p_off < t_off@TPR }
   where thresholds are derived from validation positives.
4) Train HLT-only transformer risk head to predict this target.
5) Evaluate baseline top-tagging with:
   - hard veto: if risk>=tau, force low score
   - soft penalty: score = sigmoid(logit - alpha * relu(risk-tau)/(1-tau))
   across a sweep of tau.

Outputs:
- checkpoints/<save_dir>/<run_name>/risk_head_metrics.json
- checkpoints/<save_dir>/<run_name>/risk_head_summary.tsv
- teacher.pt / baseline.pt / risk_head.pt (unless --skip_save_models)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from unmerge_correct_hlt import (
    RANDOM_SEED,
    load_raw_constituents_from_h5,
    compute_features,
    get_stats,
    standardize,
    JetDataset,
    ParticleTransformer,
    get_scheduler,
    train_classifier,
    eval_classifier,
)
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
)


def _deepcopy_config() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


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
    # First point where TPR crosses target.
    idx = np.where(tpr >= target)[0]
    if idx.size == 0:
        return float(fpr[-1])
    i = int(idx[0])
    if i == 0:
        return float(fpr[0])
    # Linear interpolation between i-1 and i.
    t0, t1 = float(tpr[i - 1]), float(tpr[i])
    f0, f1 = float(fpr[i - 1]), float(fpr[i])
    if abs(t1 - t0) < 1e-12:
        return float(f1)
    a = (target - t0) / (t1 - t0)
    return float(f0 + a * (f1 - f0))


def prob_threshold_at_target_tpr(pos_probs: np.ndarray, target_tpr: float) -> float:
    """TPR=0.5 => median positive score; in general quantile q=1-target_tpr."""
    if pos_probs.size == 0:
        return 0.5
    q = float(np.clip(1.0 - target_tpr, 0.0, 1.0))
    return float(np.quantile(pos_probs, q))


def parse_thresholds(s: str) -> List[float]:
    vals = []
    for x in s.replace(";", ",").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    out = sorted(set(float(np.clip(v, 0.0, 1.0)) for v in vals))
    if len(out) == 0:
        out = [0.5]
    return out


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = JetDataset(feat, mask, np.zeros(len(feat), dtype=np.float32))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    logits_all = []
    probs_all = []
    for b in dl:
        x = b["feat"].to(device)
        m = b["mask"].to(device)
        logits = model(x, m).squeeze(1)
        probs = torch.sigmoid(logits)
        logits_all.append(logits.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
    logits_np = np.concatenate(logits_all) if logits_all else np.zeros(0, dtype=np.float32)
    probs_np = np.concatenate(probs_all) if probs_all else np.zeros(0, dtype=np.float32)
    return logits_np.astype(np.float32), probs_np.astype(np.float32)


def train_single_view_classifier_auc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> nn.Module:
    """Train top-tagger and select checkpoint by best val AUC."""
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    best_val_auc = float("-inf")
    fpr50_at_best = float("nan")
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
            fpr50_at_best = float(va_fpr50)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, "
                f"val_fpr50={va_fpr50:.6f}, best_auc={best_val_auc:.4f}, "
                f"fpr50@best={fpr50_at_best:.6f}"
            )
        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_risk_head_auc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    pos_weight: float,
    name: str = "RiskHead",
) -> nn.Module:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    pos_weight_t = torch.tensor(float(max(pos_weight, 1.0)), device=device)

    best_val_auc = float("-inf")
    best_val_ap = float("-inf")
    best_state = None
    no_improve = 0

    for ep in tqdm(range(int(train_cfg["epochs"])), desc=name):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for b in train_loader:
            x = b["feat"].to(device)
            m = b["mask"].to(device)
            y = b["label"].to(device)
            opt.zero_grad()
            logits = model(x, m).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = x.size(0)
            tr_loss += loss.item() * bs
            n_tr += bs
        sch.step()
        tr_loss /= max(n_tr, 1)

        va_auc, va_preds, va_labs = eval_classifier(model, val_loader, device)
        if len(np.unique(va_labs)) > 1:
            va_ap = float(average_precision_score(va_labs, va_preds))
        else:
            va_ap = float("nan")

        improved = np.isfinite(va_auc) and (float(va_auc) > best_val_auc)
        if improved:
            best_val_auc = float(va_auc)
            best_val_ap = float(va_ap)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_loss={tr_loss:.4f}, val_auc={va_auc:.4f}, "
                f"val_ap={va_ap:.4f}, best_auc={best_val_auc:.4f}, best_ap={best_val_ap:.4f}"
            )
        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    out = {"auc": float("nan"), "fpr30": float("nan"), "fpr50": float("nan")}
    if scores.size == 0 or y_true.size == 0:
        return out
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, scores))
        fpr, tpr, _ = roc_curve(y_true, scores)
        out["fpr30"] = float(fpr_at_target_tpr(fpr, tpr, 0.30))
        out["fpr50"] = float(fpr_at_target_tpr(fpr, tpr, 0.50))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=500000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=100)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path().cwd() / "checkpoints" / "hlt_disagreement_risk"),
    )
    parser.add_argument("--run_name", type=str, default="risk_head_default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip_save_models", action="store_true")

    # HLT controls (same defaults as main pipelines).
    parser.add_argument("--merge_radius", type=float, default=BASE_CONFIG["hlt_effects"]["merge_radius"])
    parser.add_argument("--eff_plateau_barrel", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    parser.add_argument("--eff_plateau_endcap", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    parser.add_argument("--smear_a", type=float, default=BASE_CONFIG["hlt_effects"]["smear_a"])
    parser.add_argument("--smear_b", type=float, default=BASE_CONFIG["hlt_effects"]["smear_b"])
    parser.add_argument("--smear_c", type=float, default=BASE_CONFIG["hlt_effects"]["smear_c"])

    # Teacher / baseline config.
    parser.add_argument("--top_epochs", type=int, default=60)
    parser.add_argument("--top_patience", type=int, default=15)
    parser.add_argument("--top_lr", type=float, default=5e-4)

    # Risk-head config.
    parser.add_argument("--risk_epochs", type=int, default=40)
    parser.add_argument("--risk_patience", type=int, default=12)
    parser.add_argument("--risk_lr", type=float, default=4e-4)

    # Disagreement target.
    parser.add_argument("--target_tpr", type=float, default=0.50)
    parser.add_argument("--risk_thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--soft_alpha", type=float, default=2.0)

    args = parser.parse_args()

    set_seed(int(args.seed))

    cfg = _deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    cfg["training"]["epochs"] = int(args.top_epochs)
    cfg["training"]["patience"] = int(args.top_patience)
    cfg["training"]["lr"] = float(args.top_lr)

    risk_cfg = json.loads(json.dumps(cfg["training"]))
    risk_cfg["epochs"] = int(args.risk_epochs)
    risk_cfg["patience"] = int(args.risk_patience)
    risk_cfg["lr"] = float(args.risk_lr)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    risk_thresholds = parse_thresholds(args.risk_thresholds)
    soft_alpha = float(args.soft_alpha)

    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = int(args.offset_jets) + int(args.n_train_jets)
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

    const_raw = all_const_full[int(args.offset_jets): int(args.offset_jets) + int(args.n_train_jets)]
    labels = all_labels_full[int(args.offset_jets): int(args.offset_jets) + int(args.n_train_jets)].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )

    print("Computing features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=int(args.seed), stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=int(args.seed), stratify=labels[temp_idx]
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    bs = int(cfg["training"]["batch_size"])
    dl_teacher_tr = DataLoader(
        JetDataset(feat_off_std[train_idx], masks_off[train_idx], labels[train_idx]),
        batch_size=bs, shuffle=True, drop_last=True,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_teacher_va = DataLoader(
        JetDataset(feat_off_std[val_idx], masks_off[val_idx], labels[val_idx]),
        batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_base_tr = DataLoader(
        JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx]),
        batch_size=bs, shuffle=True, drop_last=True,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_base_va = DataLoader(
        JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx]),
        batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_teacher_te = DataLoader(
        JetDataset(feat_off_std[test_idx], masks_off[test_idx], labels[test_idx]),
        batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_base_te = DataLoader(
        JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx]),
        batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )

    print("\n" + "=" * 70)
    print("STEP 1: TRAIN TEACHER (OFFLINE) + BASELINE (HLT)")
    print("=" * 70)
    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = train_single_view_classifier_auc(
        teacher, dl_teacher_tr, dl_teacher_va, device, cfg["training"], "Teacher"
    )
    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = train_single_view_classifier_auc(
        baseline, dl_base_tr, dl_base_va, device, cfg["training"], "Baseline"
    )

    # Validation scores for target thresholds.
    _, pred_teacher_val = predict_scores(
        teacher, feat_off_std[val_idx], masks_off[val_idx], device, bs, int(args.num_workers)
    )
    base_logit_val, pred_base_val = predict_scores(
        baseline, feat_hlt_std[val_idx], hlt_mask[val_idx], device, bs, int(args.num_workers)
    )
    del base_logit_val

    y_val = labels[val_idx].astype(np.int64)
    pos_teacher = pred_teacher_val[y_val == 1]
    pos_baseline = pred_base_val[y_val == 1]
    t_off = prob_threshold_at_target_tpr(pos_teacher, float(args.target_tpr))
    t_hlt = prob_threshold_at_target_tpr(pos_baseline, float(args.target_tpr))
    print(
        f"Target thresholds from val @TPR={float(args.target_tpr):.2f}: "
        f"t_hlt={t_hlt:.6f}, t_off={t_off:.6f}"
    )

    print("\n" + "=" * 70)
    print("STEP 2: BUILD DISAGREEMENT-RISK TARGET")
    print("=" * 70)
    # Get teacher / baseline scores over all jets.
    _, pred_teacher_all = predict_scores(
        teacher, feat_off_std, masks_off, device, bs, int(args.num_workers)
    )
    base_logit_all, pred_base_all = predict_scores(
        baseline, feat_hlt_std, hlt_mask, device, bs, int(args.num_workers)
    )
    y_all = labels.astype(np.int64)

    risk_target = (
        (y_all == 0)
        & (pred_base_all > t_hlt)
        & (pred_teacher_all < t_off)
    ).astype(np.float32)

    def split_pos_rate(name: str, ii: np.ndarray) -> None:
        r = risk_target[ii]
        print(f"{name}: N={len(ii)}, risk_pos={int(r.sum())}, rate={float(r.mean()):.6f}")

    split_pos_rate("Train", train_idx)
    split_pos_rate("Val", val_idx)
    split_pos_rate("Test", test_idx)

    print("\n" + "=" * 70)
    print("STEP 3: TRAIN HLT-ONLY RISK HEAD")
    print("=" * 70)
    dl_risk_tr = DataLoader(
        JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], risk_target[train_idx]),
        batch_size=bs, shuffle=True, drop_last=True,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_risk_va = DataLoader(
        JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], risk_target[val_idx]),
        batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_risk_te = DataLoader(
        JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], risk_target[test_idx]),
        batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    n_pos = float(risk_target[train_idx].sum())
    n_tot = float(len(train_idx))
    n_neg = max(n_tot - n_pos, 1.0)
    pos_weight = float(n_neg / max(n_pos, 1.0))
    print(f"Risk-head pos_weight={pos_weight:.4f}")

    risk_head = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    risk_head = train_risk_head_auc(
        risk_head, dl_risk_tr, dl_risk_va, device, risk_cfg, pos_weight=pos_weight
    )

    # Risk head quality on risk target.
    risk_auc_te, risk_pred_te, risk_lab_te = eval_classifier(risk_head, dl_risk_te, device)
    if len(np.unique(risk_lab_te)) > 1:
        risk_ap_te = float(average_precision_score(risk_lab_te, risk_pred_te))
    else:
        risk_ap_te = float("nan")
    print(f"Risk-head test target metrics: AUC={risk_auc_te:.4f}, AP={risk_ap_te:.4f}")

    print("\n" + "=" * 70)
    print("STEP 4: TOP-TAGGING EVAL WITH RISK SWEEPS (TEST)")
    print("=" * 70)
    y_test = labels[test_idx].astype(np.int64)
    base_logits_test = base_logit_all[test_idx].astype(np.float32)
    base_probs_test = pred_base_all[test_idx].astype(np.float32)
    teacher_probs_test = pred_teacher_all[test_idx].astype(np.float32)

    # Recompute risk probs on test in aligned order.
    risk_logits_test, risk_probs_test = predict_scores(
        risk_head, feat_hlt_std[test_idx], hlt_mask[test_idx], device, bs, int(args.num_workers)
    )
    del risk_logits_test

    teacher_m = metrics_from_scores(y_test, teacher_probs_test)
    baseline_m = metrics_from_scores(y_test, base_probs_test)

    print(
        f"Teacher test:  AUC={teacher_m['auc']:.4f}, FPR30={teacher_m['fpr30']:.6f}, FPR50={teacher_m['fpr50']:.6f}"
    )
    print(
        f"Baseline test: AUC={baseline_m['auc']:.4f}, FPR30={baseline_m['fpr30']:.6f}, FPR50={baseline_m['fpr50']:.6f}"
    )
    print(f"Risk thresholds: {risk_thresholds}")
    print(f"Soft alpha: {soft_alpha:.4f}")

    sweep_rows = []
    for tau in risk_thresholds:
        tau = float(np.clip(tau, 0.0, 1.0))

        # Hard veto: force very low top score.
        hard_scores = base_probs_test.copy()
        hard_scores[risk_probs_test >= tau] = 0.0
        hard_m = metrics_from_scores(y_test, hard_scores)

        # Soft penalty in logit space, activated above tau.
        if tau >= 1.0:
            soft_scores = base_probs_test.copy()
        else:
            gate = np.clip((risk_probs_test - tau) / max(1.0 - tau, 1e-6), 0.0, 1.0)
            soft_logits = base_logits_test - float(soft_alpha) * gate
            soft_scores = 1.0 / (1.0 + np.exp(-soft_logits))
        soft_m = metrics_from_scores(y_test, soft_scores)

        sweep_rows.append(
            {
                "tau": tau,
                "hard": hard_m,
                "soft": soft_m,
                "veto_rate": float((risk_probs_test >= tau).mean()),
            }
        )
        print(
            f"tau={tau:.3f} | veto_rate={((risk_probs_test >= tau).mean()):.4f} | "
            f"HARD AUC={hard_m['auc']:.4f} FPR30={hard_m['fpr30']:.6f} FPR50={hard_m['fpr50']:.6f} | "
            f"SOFT AUC={soft_m['auc']:.4f} FPR30={soft_m['fpr30']:.6f} FPR50={soft_m['fpr50']:.6f}"
        )

    # Best rows by low-FPR metrics.
    def _best(method: str, key: str) -> Dict:
        cand = [r for r in sweep_rows if np.isfinite(r[method][key])]
        if not cand:
            return {}
        return min(cand, key=lambda r: r[method][key])

    best_hard_fpr30 = _best("hard", "fpr30")
    best_hard_fpr50 = _best("hard", "fpr50")
    best_soft_fpr30 = _best("soft", "fpr30")
    best_soft_fpr50 = _best("soft", "fpr50")

    out = {
        "run_name": str(args.run_name),
        "data_setup": {
            "train_path_arg": str(args.train_path),
            "train_files": [str(p.resolve()) for p in train_files],
            "n_train_jets": int(args.n_train_jets),
            "offset_jets": int(args.offset_jets),
            "max_constits": int(args.max_constits),
            "seed": int(args.seed),
            "split": {"train_frac": 0.70, "val_frac": 0.15, "test_frac": 0.15},
        },
        "hlt_effects": cfg["hlt_effects"],
        "target_definition": {
            "target_tpr": float(args.target_tpr),
            "t_hlt_val": float(t_hlt),
            "t_off_val": float(t_off),
            "rule": "risk=1{y=0 and p_hlt>t_hlt and p_off<t_off}",
        },
        "risk_target_rate": {
            "train": float(risk_target[train_idx].mean()),
            "val": float(risk_target[val_idx].mean()),
            "test": float(risk_target[test_idx].mean()),
        },
        "risk_head_test_target_metrics": {
            "auc": float(risk_auc_te),
            "ap": float(risk_ap_te),
        },
        "top_tag_test": {
            "teacher": teacher_m,
            "baseline": baseline_m,
            "risk_sweep": sweep_rows,
            "best_hard_by_fpr30": best_hard_fpr30,
            "best_hard_by_fpr50": best_hard_fpr50,
            "best_soft_by_fpr30": best_soft_fpr30,
            "best_soft_by_fpr50": best_soft_fpr50,
        },
        "soft_alpha": float(soft_alpha),
        "risk_thresholds": risk_thresholds,
    }

    with open(save_root / "risk_head_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Save compact TSV.
    tsv_path = save_root / "risk_head_summary.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("method\ttau\tveto_rate\tauc\tfpr30\tfpr50\n")
        f.write(
            f"teacher\t-\t-\t{teacher_m['auc']:.6f}\t{teacher_m['fpr30']:.6f}\t{teacher_m['fpr50']:.6f}\n"
        )
        f.write(
            f"baseline\t-\t-\t{baseline_m['auc']:.6f}\t{baseline_m['fpr30']:.6f}\t{baseline_m['fpr50']:.6f}\n"
        )
        for r in sweep_rows:
            f.write(
                f"hard\t{r['tau']:.6f}\t{r['veto_rate']:.6f}\t{r['hard']['auc']:.6f}\t"
                f"{r['hard']['fpr30']:.6f}\t{r['hard']['fpr50']:.6f}\n"
            )
            f.write(
                f"soft\t{r['tau']:.6f}\t{r['veto_rate']:.6f}\t{r['soft']['auc']:.6f}\t"
                f"{r['soft']['fpr30']:.6f}\t{r['soft']['fpr50']:.6f}\n"
            )

    if not bool(args.skip_save_models):
        torch.save({"model": teacher.state_dict(), "val_threshold": t_off}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "val_threshold": t_hlt}, save_root / "baseline.pt")
        torch.save({"model": risk_head.state_dict()}, save_root / "risk_head.pt")

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg["hlt_effects"], "stats": hlt_stats}, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Saved metrics to: {save_root / 'risk_head_metrics.json'}")
    print(f"Saved summary to: {tsv_path}")
    print(f"Save dir: {save_root}")


if __name__ == "__main__":
    main()


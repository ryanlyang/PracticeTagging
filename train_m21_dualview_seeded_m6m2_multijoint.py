#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model-21: Seeded dualview from two pretrained reconstructors (m6 + m2 Hungarian),
with one frozen pre-joint stage and multiple progressive-unfreeze joint modes.

Requested flow:
1) Load split/config from m2 run dir.
2) Load pretrained reconstructors:
   - Reco-A from m6 Stage-A checkpoint (pre-joint)
   - Reco-B from m2 Hungarian Stage2 checkpoint (pre-joint)
3) Train dual-view tagger on corrected views with both reconstructors frozen (pre-joint).
4) Reload pre-joint states and run joint finetune in modes:
   - both      : unfreeze A+B progressively
   - m6_only   : unfreeze A progressively, keep B frozen
   - m2_only   : unfreeze B progressively, keep A frozen
5) Save per-mode metrics, scores, and checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as b
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as m2mod


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def safe_load_state(model: nn.Module, state: Dict[str, torch.Tensor], name: str) -> None:
    miss, unexp = model.load_state_dict(state, strict=False)
    if len(miss) > 0 or len(unexp) > 0:
        print(f"[{name}] strict=False load: missing={len(miss)}, unexpected={len(unexp)}")


def sanitize_numpy_scores(s: np.ndarray) -> np.ndarray:
    return np.nan_to_num(s.astype(np.float64), nan=0.5, posinf=1.0, neginf=0.0)


def auc_and_fpr(labels: np.ndarray, probs: np.ndarray, target_tpr: float) -> Tuple[float, float]:
    labels = labels.astype(np.float32)
    probs = sanitize_numpy_scores(probs)
    finite = np.isfinite(labels) & np.isfinite(probs)
    if not np.any(finite):
        return float("nan"), float("nan")
    labels = labels[finite]
    probs = probs[finite]
    auc = float(roc_auc_score(labels, probs)) if np.unique(labels).size > 1 else float("nan")
    if np.unique(labels).size < 2:
        return auc, float("nan")
    fpr, tpr, _ = roc_curve(labels, probs)
    return auc, float(b.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))


def threshold_at_target_tpr(labels: np.ndarray, probs: np.ndarray, target_tpr: float) -> float:
    fpr, tpr, thr = roc_curve(labels.astype(np.float32), probs.astype(np.float64))
    if len(thr) == 0:
        return 0.5
    valid = np.isfinite(thr)
    if not np.any(valid):
        return float(np.median(probs))
    idx_valid = np.where(valid)[0]
    idx = int(idx_valid[np.argmin(np.abs(tpr[idx_valid] - float(target_tpr)))])
    return float(thr[idx])


def fpr_from_val_threshold(labels_test: np.ndarray, probs_test: np.ndarray, thr_val: float) -> float:
    y = labels_test.astype(np.float32) > 0.5
    neg = ~y
    pred = probs_test >= float(thr_val)
    fp = int((pred & neg).sum())
    nneg = int(neg.sum())
    return float(fp / max(nneg, 1))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def predict_single_view_logits(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    split_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    idx = split_idx.astype(np.int64)
    out = np.zeros(idx.shape[0], dtype=np.float64)
    ptr = 0
    with torch.no_grad():
        for start in range(0, len(idx), int(batch_size)):
            end = min(start + int(batch_size), len(idx))
            sl = idx[start:end]
            x = torch.tensor(feat[sl], dtype=torch.float32, device=device)
            m = torch.tensor(mask[sl], dtype=torch.bool, device=device)
            z = model(x, m).squeeze(1)
            k = end - start
            out[ptr: ptr + k] = z.detach().cpu().numpy().astype(np.float64)
            ptr += k
    return out


class FrozenDualDataset(Dataset):
    def __init__(
        self,
        feat_a: np.ndarray,
        mask_a: np.ndarray,
        feat_b: np.ndarray,
        mask_b: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_a = feat_a.astype(np.float32)
        self.mask_a = mask_a.astype(bool)
        self.feat_b = feat_b.astype(np.float32)
        self.mask_b = mask_b.astype(bool)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return {
            "feat_a": self.feat_a[i],
            "mask_a": self.mask_a[i],
            "feat_b": self.feat_b[i],
            "mask_b": self.mask_b[i],
            "label": self.labels[i],
        }


class JointDualDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt = feat_hlt.astype(np.float32)
        self.mask_hlt = mask_hlt.astype(bool)
        self.const_hlt = const_hlt.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "label": self.labels[i],
        }


@torch.no_grad()
def eval_dual_loader(
    dual: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_tpr: float,
) -> Dict[str, np.ndarray | float]:
    dual.eval()
    preds = []
    labs = []
    for batch in loader:
        xa = batch["feat_a"].to(device=device, dtype=torch.float32)
        ma = batch["mask_a"].to(device=device, dtype=torch.bool)
        xb = batch["feat_b"].to(device=device, dtype=torch.float32)
        mb = batch["mask_b"].to(device=device, dtype=torch.bool)
        y = batch["label"].to(device=device, dtype=torch.float32)
        logits = dual(xa, ma, xb, mb).squeeze(1)
        preds.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64))
        labs.append(y.detach().cpu().numpy().astype(np.float32))

    probs = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.float64)
    labels = np.concatenate(labs) if labs else np.zeros((0,), dtype=np.float32)
    auc, fpr50 = auc_and_fpr(labels, probs, float(target_tpr))
    return {"probs": probs, "labels": labels, "auc": auc, "fpr50": fpr50}


def train_dual_frozen(
    dual: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_rank: float,
    rank_tau: float,
    target_tpr: float,
    select_metric: str,
) -> Tuple[nn.Module, Dict[str, float]]:
    opt = torch.optim.AdamW(dual.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for ep in range(int(epochs)):
        dual.train()
        run_tot = run_cls = run_rank = 0.0
        n_seen = 0

        for batch in train_loader:
            xa = batch["feat_a"].to(device=device, dtype=torch.float32)
            ma = batch["mask_a"].to(device=device, dtype=torch.bool)
            xb = batch["feat_b"].to(device=device, dtype=torch.float32)
            mb = batch["mask_b"].to(device=device, dtype=torch.bool)
            y = batch["label"].to(device=device, dtype=torch.float32)

            opt.zero_grad()
            logits = dual(xa, ma, xb, mb).squeeze(1)
            l_cls = F.binary_cross_entropy_with_logits(logits, y)
            l_rank = b.low_fpr_surrogate_loss(logits, y, target_tpr=float(target_tpr), tau=float(rank_tau))
            loss = l_cls + float(lambda_rank) * l_rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            run_tot += float(loss.item()) * bs
            run_cls += float(l_cls.item()) * bs
            run_rank += float(l_rank.item()) * bs
            n_seen += bs

        sch.step()

        val_pack = eval_dual_loader(dual, val_loader, device, float(target_tpr))
        auc_v = float(val_pack["auc"])
        fpr50_v = float(val_pack["fpr50"])

        if str(select_metric).lower() == "fpr50":
            sel = fpr50_v
            improved = np.isfinite(sel) and (sel < best_sel)
        else:
            sel = auc_v
            improved = np.isfinite(sel) and (sel > best_sel)

        if improved:
            best_sel = float(sel)
            best_state = {k: v.detach().cpu().clone() for k, v in dual.state_dict().items()}
            best_metrics = {
                "best_epoch": int(ep + 1),
                "best_select_metric": str(select_metric).lower(),
                "best_sel": float(best_sel),
                "best_val_auc": float(auc_v),
                "best_val_fpr50": float(fpr50_v),
                "best_train_loss": float(run_tot / max(n_seen, 1)),
                "best_train_loss_cls": float(run_cls / max(n_seen, 1)),
                "best_train_loss_rank": float(run_rank / max(n_seen, 1)),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"DualFrozen ep {ep+1}: train_loss={run_tot/max(n_seen,1):.5f} "
                f"(cls={run_cls/max(n_seen,1):.5f}, rank={run_rank/max(n_seen,1):.5f}) | "
                f"val_auc={auc_v:.4f}, val_fpr50={fpr50_v:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping DualFrozen at epoch {ep+1}")
            break

    if best_state is not None:
        dual.load_state_dict(best_state)
    return dual, best_metrics


def build_two_views_numpy(
    split_name: str,
    split_idx: np.ndarray,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    reco_a: nn.Module,
    reco_b: nn.Module,
    device: torch.device,
    batch_size: int,
    corrected_weight_floor: float,
) -> Dict[str, np.ndarray]:
    print(f"Building corrected views ({split_name})...")
    feat_a, mask_a = b.build_corrected_view_numpy(
        reconstructor=reco_a,
        feat_hlt=feat_hlt_std[split_idx],
        mask_hlt=hlt_mask[split_idx],
        const_hlt=hlt_const[split_idx],
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_b, mask_b = m2mod.build_corrected_view_numpy(
        reconstructor=reco_b,
        feat_hlt=feat_hlt_std[split_idx],
        mask_hlt=hlt_mask[split_idx],
        const_hlt=hlt_const[split_idx],
        device=device,
        batch_size=int(batch_size),
        corrected_weight_floor=float(corrected_weight_floor),
        corrected_use_flags=False,
    )
    feat_a = np.nan_to_num(feat_a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    feat_b = np.nan_to_num(feat_b.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "feat_a": feat_a,
        "mask_a": mask_a.astype(bool),
        "feat_b": feat_b,
        "mask_b": mask_b.astype(bool),
    }


def l2_anchor_to_state(model: nn.Module, ref_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    vals = []
    for n, p in model.named_parameters():
        if n in ref_state:
            ref = ref_state[n].to(device=p.device, dtype=p.dtype)
            vals.append((p - ref).pow(2).mean())
    if len(vals) == 0:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(vals).mean()


def _set_reco_trainability(reco: nn.Module, stage: str, allow_unfreeze: bool) -> int:
    for p in reco.parameters():
        p.requires_grad_(False)

    if not allow_unfreeze or stage == "frozen":
        return sum(int(p.numel()) for p in reco.parameters() if p.requires_grad)

    if stage == "heads":
        head_tokens = (
            "head",
            "mlp",
            "input_proj",
            "ctx_mod",
            "action",
            "split_exist",
            "split_delta",
            "cand_proj",
            "router",
            "gate",
        )
        for n, p in reco.named_parameters():
            ln = n.lower()
            if any(tok in ln for tok in head_tokens):
                p.requires_grad_(True)
    else:  # full
        for p in reco.parameters():
            p.requires_grad_(True)

    return sum(int(p.numel()) for p in reco.parameters() if p.requires_grad)


def _epoch_stage(ep_idx: int, phase1_epochs: int, phase2_epochs: int) -> str:
    ep1 = int(ep_idx) + 1
    if ep1 <= int(phase1_epochs):
        return "frozen"
    if ep1 <= int(phase1_epochs + phase2_epochs):
        return "heads"
    return "full"


@torch.no_grad()
def eval_dual_joint_dynamic(
    reco_a: nn.Module,
    reco_b: nn.Module,
    dual: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    target_tpr: float,
) -> Dict[str, np.ndarray | float]:
    reco_a.eval()
    reco_b.eval()
    dual.eval()

    preds = []
    labs = []
    for batch in loader:
        feat_hlt = batch["feat_hlt"].to(device=device, dtype=torch.float32)
        mask_hlt = batch["mask_hlt"].to(device=device, dtype=torch.bool)
        const_hlt = batch["const_hlt"].to(device=device, dtype=torch.float32)
        y = batch["label"].to(device=device, dtype=torch.float32)

        out_a = reco_a(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
        feat_a, mask_a = b.build_soft_corrected_view(
            out_a,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=False,
        )
        out_b = reco_b(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = m2mod.build_soft_corrected_view(
            out_b,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=False,
        )
        logits = dual(feat_a, mask_a, feat_b, mask_b).squeeze(1)
        preds.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64))
        labs.append(y.detach().cpu().numpy().astype(np.float32))

    probs = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.float64)
    labels = np.concatenate(labs) if labs else np.zeros((0,), dtype=np.float32)
    auc, fpr50 = auc_and_fpr(labels, probs, float(target_tpr))
    return {"probs": probs, "labels": labels, "auc": auc, "fpr50": fpr50}


def train_joint_mode(
    mode_name: str,
    allow_unfreeze_a: bool,
    allow_unfreeze_b: bool,
    reco_a: nn.Module,
    reco_b: nn.Module,
    dual: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr_dual: float,
    lr_reco_a: float,
    lr_reco_b: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_rank: float,
    rank_tau: float,
    lambda_anchor_a: float,
    lambda_anchor_b: float,
    corrected_weight_floor: float,
    target_tpr: float,
    select_metric: str,
    unfreeze_phase1_epochs: int,
    unfreeze_phase2_epochs: int,
) -> Dict[str, float]:
    init_a = {k: v.detach().cpu().clone() for k, v in reco_a.state_dict().items()}
    init_b = {k: v.detach().cpu().clone() for k, v in reco_b.state_dict().items()}

    for p in dual.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [
            {"params": dual.parameters(), "lr": float(lr_dual)},
            {"params": reco_a.parameters(), "lr": float(lr_reco_a)},
            {"params": reco_b.parameters(), "lr": float(lr_reco_b)},
        ],
        lr=float(lr_dual),
        weight_decay=float(weight_decay),
    )
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_sel = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for ep in range(int(epochs)):
        stage = _epoch_stage(ep, int(unfreeze_phase1_epochs), int(unfreeze_phase2_epochs))
        n_train_a = _set_reco_trainability(reco_a, stage=stage, allow_unfreeze=bool(allow_unfreeze_a))
        n_train_b = _set_reco_trainability(reco_b, stage=stage, allow_unfreeze=bool(allow_unfreeze_b))

        dual.train()
        reco_a.train()
        reco_b.train()

        run_tot = run_cls = run_rank = run_anc_a = run_anc_b = 0.0
        n_seen = 0

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device=device, dtype=torch.float32)
            mask_hlt = batch["mask_hlt"].to(device=device, dtype=torch.bool)
            const_hlt = batch["const_hlt"].to(device=device, dtype=torch.float32)
            y = batch["label"].to(device=device, dtype=torch.float32)

            opt.zero_grad()

            out_a = reco_a(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
            feat_a, mask_a = b.build_soft_corrected_view(
                out_a,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            out_b = reco_b(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
            feat_b, mask_b = m2mod.build_soft_corrected_view(
                out_b,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )

            logits = dual(feat_a, mask_a, feat_b, mask_b).squeeze(1)
            l_cls = F.binary_cross_entropy_with_logits(logits, y)
            l_rank = b.low_fpr_surrogate_loss(logits, y, target_tpr=float(target_tpr), tau=float(rank_tau))

            if bool(allow_unfreeze_a) and float(lambda_anchor_a) > 0.0:
                l_anc_a = l2_anchor_to_state(reco_a, init_a)
            else:
                l_anc_a = torch.zeros((), device=device)

            if bool(allow_unfreeze_b) and float(lambda_anchor_b) > 0.0:
                l_anc_b = l2_anchor_to_state(reco_b, init_b)
            else:
                l_anc_b = torch.zeros((), device=device)

            loss = l_cls + float(lambda_rank) * l_rank + float(lambda_anchor_a) * l_anc_a + float(lambda_anchor_b) * l_anc_b
            loss.backward()

            torch.nn.utils.clip_grad_norm_(dual.parameters(), 1.0)
            if n_train_a > 0:
                torch.nn.utils.clip_grad_norm_([p for p in reco_a.parameters() if p.requires_grad], 1.0)
            if n_train_b > 0:
                torch.nn.utils.clip_grad_norm_([p for p in reco_b.parameters() if p.requires_grad], 1.0)

            opt.step()

            bs = y.size(0)
            run_tot += float(loss.item()) * bs
            run_cls += float(l_cls.item()) * bs
            run_rank += float(l_rank.item()) * bs
            run_anc_a += float(l_anc_a.item()) * bs
            run_anc_b += float(l_anc_b.item()) * bs
            n_seen += bs

        sch.step()

        val_pack = eval_dual_joint_dynamic(
            reco_a=reco_a,
            reco_b=reco_b,
            dual=dual,
            loader=val_loader,
            device=device,
            corrected_weight_floor=float(corrected_weight_floor),
            target_tpr=float(target_tpr),
        )
        auc_v = float(val_pack["auc"])
        fpr50_v = float(val_pack["fpr50"])

        if str(select_metric).lower() == "fpr50":
            sel = fpr50_v
            improved = np.isfinite(sel) and (sel < best_sel)
        else:
            sel = auc_v
            improved = np.isfinite(sel) and (sel > best_sel)

        if improved:
            best_sel = float(sel)
            best_state = {
                "dual": {k: v.detach().cpu().clone() for k, v in dual.state_dict().items()},
                "reco_a": {k: v.detach().cpu().clone() for k, v in reco_a.state_dict().items()},
                "reco_b": {k: v.detach().cpu().clone() for k, v in reco_b.state_dict().items()},
            }
            best_metrics = {
                "mode": str(mode_name),
                "best_epoch": int(ep + 1),
                "best_select_metric": str(select_metric).lower(),
                "best_sel": float(best_sel),
                "best_val_auc": float(auc_v),
                "best_val_fpr50": float(fpr50_v),
                "best_train_loss": float(run_tot / max(n_seen, 1)),
                "best_train_loss_cls": float(run_cls / max(n_seen, 1)),
                "best_train_loss_rank": float(run_rank / max(n_seen, 1)),
                "best_train_anchor_a": float(run_anc_a / max(n_seen, 1)),
                "best_train_anchor_b": float(run_anc_b / max(n_seen, 1)),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 2 == 0:
            print(
                f"Joint[{mode_name}] ep {ep+1}: stage={stage}, trainableA={n_train_a}, trainableB={n_train_b}, "
                f"train_loss={run_tot/max(n_seen,1):.5f} (cls={run_cls/max(n_seen,1):.5f}, rank={run_rank/max(n_seen,1):.5f}, "
                f"ancA={run_anc_a/max(n_seen,1):.5f}, ancB={run_anc_b/max(n_seen,1):.5f}) | "
                f"val_auc={auc_v:.4f}, val_fpr50={fpr50_v:.6f}, best_sel={best_sel:.6f}"
            )

        if no_improve >= int(patience):
            print(f"Early stopping Joint[{mode_name}] at epoch {ep+1}")
            break

    if best_state is not None:
        dual.load_state_dict(best_state["dual"])
        reco_a.load_state_dict(best_state["reco_a"])
        reco_b.load_state_dict(best_state["reco_b"])

    return best_metrics


def parse_joint_modes(s: str) -> List[str]:
    allowed = {"both", "m6_only", "m2_only"}
    out = []
    for tok in str(s).split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in allowed:
            raise ValueError(f"Unknown joint mode: {t}; allowed={sorted(allowed)}")
        if t not in out:
            out.append(t)
    if not out:
        out = ["both", "m6_only", "m2_only"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2_run_dir", type=str, required=True)
    ap.add_argument("--m6_run_dir", type=str, required=True)
    ap.add_argument("--m2_reco_ckpt", type=str, default="offline_reconstructor_stage2.pt")
    ap.add_argument("--m6_reco_ckpt", type=str, default="offline_reconstructor_stageA.pt")
    ap.add_argument("--m2_baseline_ckpt", type=str, default="baseline.pt")
    ap.add_argument("--teacher_ckpt", type=str, default="teacher.pt")

    ap.add_argument(
        "--save_dir",
        type=str,
        default=str(Path().cwd() / "checkpoints" / "reco_teacher_joint_fusion_6model_150k75k150k" / "model21_dualview_seeded_m6m2"),
    )
    ap.add_argument("--run_name", type=str, default="model21_dualview_seeded_m6m2_150k75k150k_seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--skip_save_models", action="store_true")

    ap.add_argument("--reco_eval_batch_size", type=int, default=256)
    ap.add_argument("--corrected_weight_floor", type=float, default=0.03)
    ap.add_argument("--target_tpr", type=float, default=0.50)

    ap.add_argument("--frozen_epochs", type=int, default=45)
    ap.add_argument("--frozen_patience", type=int, default=12)
    ap.add_argument("--frozen_batch_size", type=int, default=256)
    ap.add_argument("--frozen_lr", type=float, default=3e-4)
    ap.add_argument("--frozen_weight_decay", type=float, default=1e-4)
    ap.add_argument("--frozen_warmup_epochs", type=int, default=5)
    ap.add_argument("--frozen_lambda_rank", type=float, default=0.2)
    ap.add_argument("--frozen_rank_tau", type=float, default=0.05)

    ap.add_argument("--joint_modes", type=str, default="both,m6_only,m2_only")
    ap.add_argument("--joint_epochs", type=int, default=25)
    ap.add_argument("--joint_patience", type=int, default=8)
    ap.add_argument("--joint_batch_size", type=int, default=128)
    ap.add_argument("--joint_lr_dual", type=float, default=1e-4)
    ap.add_argument("--joint_lr_reco_a", type=float, default=2e-6)
    ap.add_argument("--joint_lr_reco_b", type=float, default=2e-6)
    ap.add_argument("--joint_weight_decay", type=float, default=1e-4)
    ap.add_argument("--joint_warmup_epochs", type=int, default=3)
    ap.add_argument("--joint_lambda_rank", type=float, default=0.2)
    ap.add_argument("--joint_rank_tau", type=float, default=0.05)
    ap.add_argument("--joint_lambda_anchor_a", type=float, default=0.02)
    ap.add_argument("--joint_lambda_anchor_b", type=float, default=0.02)
    ap.add_argument("--joint_unfreeze_phase1_epochs", type=int, default=3)
    ap.add_argument("--joint_unfreeze_phase2_epochs", type=int, default=7)

    ap.add_argument("--select_metric", type=str, choices=["auc", "fpr50"], default="auc")

    args = ap.parse_args()

    set_seed(int(args.seed))

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    m2_run = Path(args.m2_run_dir)
    m6_run = Path(args.m6_run_dir)

    setup_path = m2_run / "data_setup.json"
    split_path = m2_run / "data_splits.npz"
    if not setup_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Missing m2 setup/splits in {m2_run}")

    with open(setup_path, "r", encoding="utf-8") as f:
        data_setup = json.load(f)
    split_npz = np.load(split_path)

    train_idx = split_npz["train_idx"].astype(np.int64)
    val_idx = split_npz["val_idx"].astype(np.int64)
    test_idx = split_npz["test_idx"].astype(np.int64)
    means = split_npz["means"].astype(np.float32)
    stds = split_npz["stds"].astype(np.float32)

    train_files = [Path(p) for p in data_setup.get("train_files", [])]
    if len(train_files) == 0:
        raise RuntimeError("data_setup.json has no train_files")

    n_train_jets = int(data_setup.get("n_train_jets"))
    offset_jets = int(data_setup.get("offset_jets", 0))
    max_constits = int(data_setup.get("max_constits", 100))
    hlt_cfg = dict(data_setup.get("hlt_effects", {}))

    cfg = b._deepcopy_config()
    cfg["hlt_effects"].update(hlt_cfg)

    max_jets_needed = int(offset_jets + n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = b.load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}")

    const_raw = all_const_full[offset_jets: offset_jets + n_train_jets]
    labels = all_labels_full[offset_jets: offset_jets + n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, _hlt_stats, _ = b.apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(data_setup.get("seed", args.seed)),
    )

    print("Computing standardized HLT features...")
    feat_hlt = b.compute_features(hlt_const, hlt_mask)
    feat_hlt_std = b.standardize(feat_hlt, hlt_mask, means, stds)
    feat_hlt_std = np.nan_to_num(feat_hlt_std.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    baseline = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)

    baseline_path = m2_run / args.m2_baseline_ckpt
    teacher_path = m2_run / args.teacher_ckpt
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_path}")
    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_path}")

    safe_load_state(baseline, load_model_state(baseline_path, device), "baseline")
    safe_load_state(teacher, load_model_state(teacher_path, device), "teacher")

    hlt_logits_val = predict_single_view_logits(baseline, feat_hlt_std, hlt_mask, val_idx, device, int(args.reco_eval_batch_size))
    hlt_logits_test = predict_single_view_logits(baseline, feat_hlt_std, hlt_mask, test_idx, device, int(args.reco_eval_batch_size))

    teacher_logits_val = predict_single_view_logits(teacher, feat_hlt_std, hlt_mask, val_idx, device, int(args.reco_eval_batch_size))
    teacher_logits_test = predict_single_view_logits(teacher, feat_hlt_std, hlt_mask, test_idx, device, int(args.reco_eval_batch_size))

    reco_a = b.OfflineReconstructor(input_dim=7, **b.BASE_CONFIG["reconstructor_model"]).to(device)
    reco_b = m2mod.OfflineReconstructor(input_dim=7, **m2mod.BASE_CONFIG["reconstructor_model"]).to(device)

    reco_a_path = m6_run / args.m6_reco_ckpt
    reco_b_path = m2_run / args.m2_reco_ckpt
    if not reco_a_path.exists():
        raise FileNotFoundError(f"Missing m6 reco checkpoint: {reco_a_path}")
    if not reco_b_path.exists():
        raise FileNotFoundError(f"Missing m2 reco checkpoint: {reco_b_path}")

    safe_load_state(reco_a, load_model_state(reco_a_path, device), "reco_a_m6")
    safe_load_state(reco_b, load_model_state(reco_b_path, device), "reco_b_m2")

    for p in reco_a.parameters():
        p.requires_grad_(False)
    for p in reco_b.parameters():
        p.requires_grad_(False)

    print("\n" + "=" * 70)
    print("STEP 1: BUILD FROZEN TWO-VIEW TENSORS")
    print("=" * 70)
    train_views = build_two_views_numpy(
        split_name="train",
        split_idx=train_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        reco_a=reco_a,
        reco_b=reco_b,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    val_views = build_two_views_numpy(
        split_name="val",
        split_idx=val_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        reco_a=reco_a,
        reco_b=reco_b,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    test_views = build_two_views_numpy(
        split_name="test",
        split_idx=test_idx,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        reco_a=reco_a,
        reco_b=reco_b,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )

    ds_train_f = FrozenDualDataset(
        feat_a=train_views["feat_a"],
        mask_a=train_views["mask_a"],
        feat_b=train_views["feat_b"],
        mask_b=train_views["mask_b"],
        labels=labels[train_idx],
    )
    ds_val_f = FrozenDualDataset(
        feat_a=val_views["feat_a"],
        mask_a=val_views["mask_a"],
        feat_b=val_views["feat_b"],
        mask_b=val_views["mask_b"],
        labels=labels[val_idx],
    )
    ds_test_f = FrozenDualDataset(
        feat_a=test_views["feat_a"],
        mask_a=test_views["mask_a"],
        feat_b=test_views["feat_b"],
        mask_b=test_views["mask_b"],
        labels=labels[test_idx],
    )

    dl_train_f = DataLoader(ds_train_f, batch_size=int(args.frozen_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_f = DataLoader(ds_val_f, batch_size=int(args.frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_f = DataLoader(ds_test_f, batch_size=int(args.frozen_batch_size), shuffle=False, num_workers=int(args.num_workers))

    dual = b.DualViewCrossAttnClassifier(input_dim_a=10, input_dim_b=10, **cfg["model"]).to(device)

    print("\n" + "=" * 70)
    print("STEP 2: DUALVIEW PRE-JOINT (RECO A/B FROZEN)")
    print("=" * 70)
    dual, frozen_metrics = train_dual_frozen(
        dual=dual,
        train_loader=dl_train_f,
        val_loader=dl_val_f,
        device=device,
        epochs=int(args.frozen_epochs),
        patience=int(args.frozen_patience),
        lr=float(args.frozen_lr),
        weight_decay=float(args.frozen_weight_decay),
        warmup_epochs=int(args.frozen_warmup_epochs),
        lambda_rank=float(args.frozen_lambda_rank),
        rank_tau=float(args.frozen_rank_tau),
        target_tpr=float(args.target_tpr),
        select_metric=str(args.select_metric),
    )

    frozen_val = eval_dual_loader(dual, dl_val_f, device, float(args.target_tpr))
    frozen_test = eval_dual_loader(dual, dl_test_f, device, float(args.target_tpr))

    print(
        f"DualPreJoint: val_auc={float(frozen_val['auc']):.4f}, val_fpr50={float(frozen_val['fpr50']):.6f} | "
        f"test_auc={float(frozen_test['auc']):.4f}, test_fpr50={float(frozen_test['fpr50']):.6f}"
    )

    dual_prejoint_state = {k: v.detach().cpu().clone() for k, v in dual.state_dict().items()}
    reco_a_prejoint_state = {k: v.detach().cpu().clone() for k, v in reco_a.state_dict().items()}
    reco_b_prejoint_state = {k: v.detach().cpu().clone() for k, v in reco_b.state_dict().items()}

    ds_train_j = JointDualDataset(
        feat_hlt=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        labels=labels[train_idx],
    )
    ds_val_j = JointDualDataset(
        feat_hlt=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        labels=labels[val_idx],
    )
    ds_test_j = JointDualDataset(
        feat_hlt=feat_hlt_std[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_hlt=hlt_const[test_idx],
        labels=labels[test_idx],
    )

    dl_train_j = DataLoader(ds_train_j, batch_size=int(args.joint_batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_j = DataLoader(ds_val_j, batch_size=int(args.joint_batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_j = DataLoader(ds_test_j, batch_size=int(args.joint_batch_size), shuffle=False, num_workers=int(args.num_workers))

    mode_names = parse_joint_modes(str(args.joint_modes))
    mode_map = {
        "both": (True, True),
        "m6_only": (True, False),
        "m2_only": (False, True),
    }

    mode_train_metrics: Dict[str, Dict[str, float]] = {}
    mode_eval_val: Dict[str, Dict[str, float | np.ndarray]] = {}
    mode_eval_test: Dict[str, Dict[str, float | np.ndarray]] = {}

    print("\n" + "=" * 70)
    print("STEP 3: JOINT MODES (RELOAD PRE-JOINT EACH MODE)")
    print("=" * 70)

    for mode_name in mode_names:
        allow_a, allow_b = mode_map[mode_name]
        print("-" * 70)
        print(f"Joint mode: {mode_name} (unfreeze_m6={allow_a}, unfreeze_m2={allow_b})")
        print("-" * 70)

        dual.load_state_dict(dual_prejoint_state)
        reco_a.load_state_dict(reco_a_prejoint_state)
        reco_b.load_state_dict(reco_b_prejoint_state)

        for p in reco_a.parameters():
            p.requires_grad_(False)
        for p in reco_b.parameters():
            p.requires_grad_(False)

        if int(args.joint_epochs) > 0:
            train_metrics = train_joint_mode(
                mode_name=mode_name,
                allow_unfreeze_a=allow_a,
                allow_unfreeze_b=allow_b,
                reco_a=reco_a,
                reco_b=reco_b,
                dual=dual,
                train_loader=dl_train_j,
                val_loader=dl_val_j,
                device=device,
                epochs=int(args.joint_epochs),
                patience=int(args.joint_patience),
                lr_dual=float(args.joint_lr_dual),
                lr_reco_a=float(args.joint_lr_reco_a),
                lr_reco_b=float(args.joint_lr_reco_b),
                weight_decay=float(args.joint_weight_decay),
                warmup_epochs=int(args.joint_warmup_epochs),
                lambda_rank=float(args.joint_lambda_rank),
                rank_tau=float(args.joint_rank_tau),
                lambda_anchor_a=float(args.joint_lambda_anchor_a),
                lambda_anchor_b=float(args.joint_lambda_anchor_b),
                corrected_weight_floor=float(args.corrected_weight_floor),
                target_tpr=float(args.target_tpr),
                select_metric=str(args.select_metric),
                unfreeze_phase1_epochs=int(args.joint_unfreeze_phase1_epochs),
                unfreeze_phase2_epochs=int(args.joint_unfreeze_phase2_epochs),
            )
        else:
            train_metrics = {}

        val_pack = eval_dual_joint_dynamic(
            reco_a=reco_a,
            reco_b=reco_b,
            dual=dual,
            loader=dl_val_j,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            target_tpr=float(args.target_tpr),
        )
        test_pack = eval_dual_joint_dynamic(
            reco_a=reco_a,
            reco_b=reco_b,
            dual=dual,
            loader=dl_test_j,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            target_tpr=float(args.target_tpr),
        )

        thr_val = threshold_at_target_tpr(val_pack["labels"], val_pack["probs"], float(args.target_tpr))
        fpr_test_at_val_thr = fpr_from_val_threshold(test_pack["labels"], test_pack["probs"], thr_val)

        train_metrics = dict(train_metrics)
        train_metrics.update(
            {
                "mode": mode_name,
                "val_auc": float(val_pack["auc"]),
                "val_fpr50": float(val_pack["fpr50"]),
                "test_auc": float(test_pack["auc"]),
                "test_fpr50": float(test_pack["fpr50"]),
                "val_threshold": float(thr_val),
                "test_fpr_at_val_thr": float(fpr_test_at_val_thr),
            }
        )

        mode_train_metrics[mode_name] = train_metrics
        mode_eval_val[mode_name] = val_pack
        mode_eval_test[mode_name] = test_pack

        print(
            f"Joint[{mode_name}] final: val_auc={float(val_pack['auc']):.4f}, val_fpr50={float(val_pack['fpr50']):.6f} | "
            f"test_auc={float(test_pack['auc']):.4f}, test_fpr50={float(test_pack['fpr50']):.6f}, "
            f"test_fpr@valthr={float(fpr_test_at_val_thr):.6f}"
        )

        if not args.skip_save_models:
            torch.save({"model": dual.state_dict(), "mode": mode_name, "metrics": train_metrics}, save_root / f"dual_joint_{mode_name}.pt")
            torch.save({"model": reco_a.state_dict(), "mode": mode_name}, save_root / f"offline_reconstructor_A_{mode_name}.pt")
            torch.save({"model": reco_b.state_dict(), "mode": mode_name}, save_root / f"offline_reconstructor_B_{mode_name}.pt")

    # Reference metrics.
    probs_hlt_val = sigmoid_np(hlt_logits_val)
    probs_hlt_test = sigmoid_np(hlt_logits_test)
    probs_teacher_val = sigmoid_np(teacher_logits_val)
    probs_teacher_test = sigmoid_np(teacher_logits_test)

    hlt_auc_val, hlt_fpr50_val = auc_and_fpr(labels[val_idx], probs_hlt_val, float(args.target_tpr))
    hlt_auc_test, hlt_fpr50_test = auc_and_fpr(labels[test_idx], probs_hlt_test, float(args.target_tpr))
    teacher_auc_val, teacher_fpr50_val = auc_and_fpr(labels[val_idx], probs_teacher_val, float(args.target_tpr))
    teacher_auc_test, teacher_fpr50_test = auc_and_fpr(labels[test_idx], probs_teacher_test, float(args.target_tpr))

    # Save scores.
    save_arrays: Dict[str, np.ndarray] = {
        "labels_val": labels[val_idx].astype(np.float32),
        "labels_test": labels[test_idx].astype(np.float32),
        "preds_hlt_val": probs_hlt_val.astype(np.float64),
        "preds_hlt_test": probs_hlt_test.astype(np.float64),
        "preds_teacher_val": probs_teacher_val.astype(np.float64),
        "preds_teacher_test": probs_teacher_test.astype(np.float64),
        "preds_dual_prejoint_val": frozen_val["probs"].astype(np.float64),
        "preds_dual_prejoint_test": frozen_test["probs"].astype(np.float64),
    }
    for mode_name in mode_names:
        save_arrays[f"preds_dual_joint_{mode_name}_val"] = mode_eval_val[mode_name]["probs"].astype(np.float64)
        save_arrays[f"preds_dual_joint_{mode_name}_test"] = mode_eval_test[mode_name]["probs"].astype(np.float64)

    np.savez_compressed(save_root / "dualview_seeded_multijoint_scores.npz", **save_arrays)

    out_json = {
        "variant": "model21_dualview_seeded_m6m2_multijoint",
        "seed": int(args.seed),
        "select_metric": str(args.select_metric).lower(),
        "target_tpr": float(args.target_tpr),
        "joint_modes": mode_names,
        "m2": {
            "run_dir": str(m2_run),
            "reco_ckpt": str(args.m2_reco_ckpt),
            "baseline_ckpt": str(args.m2_baseline_ckpt),
            "teacher_ckpt": str(args.teacher_ckpt),
        },
        "m6": {
            "run_dir": str(m6_run),
            "reco_ckpt": str(args.m6_reco_ckpt),
        },
        "hlt": {
            "auc_val": float(hlt_auc_val),
            "auc_test": float(hlt_auc_test),
            "fpr50_val": float(hlt_fpr50_val),
            "fpr50_test": float(hlt_fpr50_test),
        },
        "teacher": {
            "auc_val": float(teacher_auc_val),
            "auc_test": float(teacher_auc_test),
            "fpr50_val": float(teacher_fpr50_val),
            "fpr50_test": float(teacher_fpr50_test),
        },
        "dual_prejoint_train": frozen_metrics,
        "dual_prejoint_eval": {
            "auc_val": float(frozen_val["auc"]),
            "fpr50_val": float(frozen_val["fpr50"]),
            "auc_test": float(frozen_test["auc"]),
            "fpr50_test": float(frozen_test["fpr50"]),
            "val_threshold": float(threshold_at_target_tpr(frozen_val["labels"], frozen_val["probs"], float(args.target_tpr))),
            "test_fpr_at_val_thr": float(
                fpr_from_val_threshold(
                    frozen_test["labels"],
                    frozen_test["probs"],
                    threshold_at_target_tpr(frozen_val["labels"], frozen_val["probs"], float(args.target_tpr)),
                )
            ),
        },
        "joint_modes_metrics": mode_train_metrics,
    }

    with open(save_root / "dualview_seeded_multijoint_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    if not args.skip_save_models:
        torch.save({"model": dual_prejoint_state, "stage": "prejoint"}, save_root / "dual_prejoint.pt")
        torch.save({"model": reco_a_prejoint_state, "stage": "prejoint"}, save_root / "offline_reconstructor_A_prejoint.pt")
        torch.save({"model": reco_b_prejoint_state, "stage": "prejoint"}, save_root / "offline_reconstructor_B_prejoint.pt")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"HLT AUC (val/test): {hlt_auc_val:.4f} / {hlt_auc_test:.4f}\n"
        f"Teacher AUC (val/test): {teacher_auc_val:.4f} / {teacher_auc_test:.4f}\n"
        f"DualPreJoint AUC (val/test): {float(frozen_val['auc']):.4f} / {float(frozen_test['auc']):.4f}"
    )
    for mode_name in mode_names:
        mv = mode_eval_val[mode_name]
        mt = mode_eval_test[mode_name]
        print(
            f"DualJoint[{mode_name}] AUC (val/test): {float(mv['auc']):.4f} / {float(mt['auc']):.4f} | "
            f"FPR@50(test)={float(mt['fpr50']):.6f}"
        )
    print(f"\nSaved model-21 seeded dualview results to: {save_root}")


if __name__ == "__main__":
    main()

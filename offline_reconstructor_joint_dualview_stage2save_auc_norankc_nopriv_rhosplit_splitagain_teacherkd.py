#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-driven Offline Reconstructor + Dual-View tagger joint training.

Pipeline:
1) Build pseudo-HLT from offline jets (same realistic generator family).
2) Train teacher (offline) and baseline (HLT).
3) Stage A: pretrain reconstructor (reconstruction losses).
4) Stage B: pretrain dual-view classifier with reconstructor frozen.
5) Stage C: joint finetune reconstructor + dual-view classifier.
6) Select checkpoints by lowest val FPR@50% TPR.

Notes:
- Uses soft candidate view from reconstructor outputs for differentiable joint training.
- Includes a differentiable low-FPR surrogate targeting TPR=0.5 behavior.
- Variant: non-privileged budgeting with rho split targets.
  * Supervises via true_added = (offline_count - hlt_count), non-privileged.
  * Uses merge_target = rho * true_added and eff_target = (1-rho) * true_added.
  * Efficiency-generation branch is enabled.
  * Adds a conservative "split-again" mechanism in corrected-view construction:
    second-stage split mass is inferred from first-stage split signals and constrained
    by remaining merge budget.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import matplotlib.pyplot as plt
from tqdm import tqdm

from unmerge_correct_hlt import (
    RANDOM_SEED,
    load_raw_constituents_from_h5,
    compute_features,
    compute_jet_pt,
    build_pt_edges,
    jet_response_resolution,
    plot_response_resolution,
    get_stats,
    standardize,
    ParticleTransformer,
    DualViewCrossAttnClassifier,
    DualViewKDDataset,
    JetDataset,
    get_scheduler,
    eval_classifier,
    eval_classifier_dual,
    train_classifier,
)

from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
    OfflineReconstructor,
    compute_reconstruction_losses,
    reconstruct_dataset,
    plot_roc,
    fpr_at_target_tpr,
    plot_constituent_count_diagnostics,
    plot_budget_diagnostics,
    train_dual_kd_student,
)


# ----------------------------- Reproducibility ----------------------------- #
def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


def _deepcopy_config() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _clamp_target_scale(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def load_raw_constituents_labels_weights_from_h5(
    files: list[Path],
    max_jets: int,
    max_constits: int,
    use_train_weights: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw constituents + labels and (optionally) per-jet training weights.
    Weight key preference: training_weights, then weights, else ones fallback.
    """
    const_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    jets_read = 0

    for fname in files:
        if jets_read >= int(max_jets):
            break
        with h5py.File(fname, "r") as f:
            n_file = int(f["labels"].shape[0])
            take = min(n_file, int(max_jets) - jets_read)

            pt = f["fjet_clus_pt"][:take, : int(max_constits)].astype(np.float32)
            eta = f["fjet_clus_eta"][:take, : int(max_constits)].astype(np.float32)
            phi = f["fjet_clus_phi"][:take, : int(max_constits)].astype(np.float32)
            ene = f["fjet_clus_E"][:take, : int(max_constits)].astype(np.float32)
            const = np.stack([pt, eta, phi, ene], axis=-1).astype(np.float32)

            labels = f["labels"][:take]
            if labels.ndim > 1:
                if labels.shape[1] == 1:
                    labels = labels[:, 0]
                else:
                    labels = np.argmax(labels, axis=1)
            labels = labels.astype(np.int64)

            if bool(use_train_weights):
                if "training_weights" in f:
                    w = np.asarray(f["training_weights"][:take], dtype=np.float32)
                elif "weights" in f:
                    w = np.asarray(f["weights"][:take], dtype=np.float32)
                    print(f"Info: using 'weights' from {fname} as training weights.")
                else:
                    print(f"Warning: missing training weight key in {fname}; using ones.")
                    w = np.ones((take,), dtype=np.float32)
            else:
                w = np.ones((take,), dtype=np.float32)

            if w.ndim > 1:
                w = w.reshape(w.shape[0], -1)[:, 0]
            w = np.asarray(w, dtype=np.float32)
            if w.shape[0] != labels.shape[0]:
                raise RuntimeError(
                    f"training_weights length mismatch in {fname}: {w.shape[0]} vs labels {labels.shape[0]}"
                )

            const_list.append(const)
            label_list.append(labels)
            w_list.append(w)
            jets_read += int(take)

    if len(const_list) == 0:
        raise RuntimeError("No jets were loaded from input HDF5 files.")

    all_const = np.concatenate(const_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)
    all_w = np.concatenate(w_list, axis=0)
    return all_const, all_labels, all_w


# Conservative split-again controls for corrected-view construction.
SPLIT_AGAIN_CFG = {
    "enabled": True,
    "exist_thr": 0.30,
    "psplit_thr": 0.20,
    "dr_thr": 0.10,
    "alpha": 8.0,
    "beta": 4.0,
    "gamma": 3.0,
    "score_power": 1.25,
    "budget_frac": 1.0,
    "max_parent_added": 2.0,
}


def train_single_view_classifier_auc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> nn.Module:
    """Train single-view top-tagger and select checkpoint by best val AUC."""
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


def reconstruct_and_standardize_features(
    reconstructor: nn.Module,
    feat_hlt_std: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    max_constits: int,
    device: torch.device,
    batch_size: int,
    weight_threshold: float,
    use_budget_topk: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct one split and return standardized 7D features + mask."""
    reco_const, reco_mask, *_ = reconstruct_dataset(
        model=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=mask_hlt,
        const_hlt=const_hlt,
        max_constits=int(max_constits),
        device=device,
        batch_size=int(batch_size),
        weight_threshold=float(weight_threshold),
        use_budget_topk=bool(use_budget_topk),
    )
    feat_reco = compute_features(reco_const, reco_mask)
    feat_reco_std = standardize(feat_reco, reco_mask, means, stds).astype(np.float32, copy=False)
    return feat_reco_std, reco_mask


class JointDualDataset(Dataset):
    def __init__(
        self,
        feat_hlt_reco: np.ndarray,
        feat_hlt_dual: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
        labels: np.ndarray,
        joint_fused_target: np.ndarray | None = None,
    ):
        self.feat_hlt_reco = torch.tensor(feat_hlt_reco, dtype=torch.float32)
        self.feat_hlt_dual = torch.tensor(feat_hlt_dual, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        self.joint_fused_target = None
        if joint_fused_target is not None:
            tgt = np.asarray(joint_fused_target, dtype=np.float32).reshape(-1)
            if int(tgt.shape[0]) != int(self.labels.shape[0]):
                raise ValueError(
                    "joint_fused_target length mismatch: "
                    f"{int(tgt.shape[0])} vs labels {int(self.labels.shape[0])}"
                )
            self.joint_fused_target = torch.tensor(tgt, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt_reco.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        out = {
            "feat_hlt_reco": self.feat_hlt_reco[i],
            "feat_hlt_dual": self.feat_hlt_dual[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
            "label": self.labels[i],
        }
        if self.joint_fused_target is not None:
            out["joint_fused_target"] = self.joint_fused_target[i]
        return out


class StageAReconstructionDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        labels: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
        stagea_fused_target: Optional[np.ndarray] = None,
        stagea_anchor_target: Optional[np.ndarray] = None,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)
        self.stagea_fused_target = None
        if stagea_fused_target is not None:
            tgt = np.asarray(stagea_fused_target, dtype=np.float32).reshape(-1)
            if int(tgt.shape[0]) != int(self.labels.shape[0]):
                raise ValueError(
                    "stagea_fused_target length mismatch: "
                    f"{int(tgt.shape[0])} vs labels {int(self.labels.shape[0])}"
                )
            self.stagea_fused_target = torch.tensor(tgt, dtype=torch.float32)
        self.stagea_anchor_target = None
        if stagea_anchor_target is not None:
            tgt_anchor = np.asarray(stagea_anchor_target, dtype=np.float32).reshape(-1)
            if int(tgt_anchor.shape[0]) != int(self.labels.shape[0]):
                raise ValueError(
                    "stagea_anchor_target length mismatch: "
                    f"{int(tgt_anchor.shape[0])} vs labels {int(self.labels.shape[0])}"
                )
            self.stagea_anchor_target = torch.tensor(tgt_anchor, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        out = {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "label": self.labels[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
        }
        if self.stagea_fused_target is not None:
            out["stagea_fused_target"] = self.stagea_fused_target[i]
        if self.stagea_anchor_target is not None:
            out["stagea_anchor_target"] = self.stagea_anchor_target[i]
        return out


def _weighted_batch_mean(vec: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return vec.mean()
    sw = sample_weight.float().clamp(min=0.0)
    denom = sw.sum().clamp(min=1e-6)
    return (vec * sw).sum() / denom


def _build_advantage_weights(
    target_soft: torch.Tensor,
    anchor_soft: torch.Tensor,
    adv_weight: float,
    adv_power: float,
    uncert_weight: float,
    use_abs_delta: bool,
    w_min: float,
    w_max: float,
) -> torch.Tensor:
    w = torch.ones_like(target_soft, dtype=torch.float32)
    if float(adv_weight) > 0.0:
        delta = target_soft - anchor_soft
        delta_mag = delta.abs() if bool(use_abs_delta) else F.relu(delta)
        delta_mag = torch.pow(delta_mag.clamp(min=0.0), float(max(adv_power, 1e-6)))
        w = w * (1.0 + float(adv_weight) * delta_mag)
    if float(uncert_weight) > 0.0:
        uncert = (1.0 - 2.0 * torch.abs(target_soft - 0.5)).clamp(min=0.0, max=1.0)
        w = w * (1.0 + float(uncert_weight) * uncert)
    w = w.clamp(min=float(max(w_min, 1e-6)), max=float(max(w_max, max(w_min, 1e-6))))
    return w


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


def stage_scale_local(epoch: int, cfg: Dict) -> float:
    s1 = int(cfg.get("stage1_epochs", 0))
    s2 = int(cfg.get("stage2_epochs", 0))
    if epoch < s1:
        return 0.35
    if epoch < s2:
        return 0.70
    return 1.0


def _standardize_features_torch(
    feat: torch.Tensor,
    mask: torch.Tensor,
    means_t: torch.Tensor,
    stds_t: torch.Tensor,
) -> torch.Tensor:
    feat_std = (feat - means_t.view(1, 1, -1)) / stds_t.view(1, 1, -1)
    feat_std = torch.nan_to_num(feat_std, nan=0.0, posinf=0.0, neginf=0.0)
    feat_std = feat_std * mask.unsqueeze(-1).float()
    return feat_std


def _build_teacher_reco_features_from_output(
    reco_out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    weight_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    L = int(const_hlt.shape[1])
    tok_tokens = reco_out["cand_tokens"][:, :L, :]
    tok_w = reco_out["cand_weights"][:, :L].clamp(0.0, 1.0)

    mask_b = (tok_w > float(weight_floor)) & mask_hlt
    none_valid = ~mask_b.any(dim=1)
    if none_valid.any():
        # Fall back to HLT support when reco weights are too sparse.
        mask_b = torch.where(mask_hlt, mask_hlt, mask_b)
        none_valid = ~mask_b.any(dim=1)
        if none_valid.any():
            mask_b = mask_b.clone()
            mask_b[none_valid, 0] = True

    feat7 = compute_features_torch(tok_tokens, mask_b)
    return feat7, mask_b


def _sorted_edit_budget_vec(
    reco_tokens: torch.Tensor,
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
) -> torch.Tensor:
    # Permutation-robust edit magnitude proxy via pT-sorted matching.
    very_low = -1e9
    pt_pred = torch.where(mask_hlt, reco_tokens[..., 0], torch.full_like(reco_tokens[..., 0], very_low))
    pt_hlt = torch.where(mask_hlt, const_hlt[..., 0], torch.full_like(const_hlt[..., 0], very_low))

    idx_pred = torch.argsort(pt_pred, dim=1, descending=True)
    idx_hlt = torch.argsort(pt_hlt, dim=1, descending=True)

    gather4_pred = idx_pred.unsqueeze(-1).expand(-1, -1, reco_tokens.shape[-1])
    gather4_hlt = idx_hlt.unsqueeze(-1).expand(-1, -1, const_hlt.shape[-1])

    pred_sorted = torch.gather(reco_tokens, 1, gather4_pred)
    hlt_sorted = torch.gather(const_hlt, 1, gather4_hlt)
    mask_sorted = torch.gather(mask_hlt, 1, idx_hlt)

    abs_diff_tok = (pred_sorted - hlt_sorted).abs().mean(dim=-1)
    denom = mask_sorted.float().sum(dim=1).clamp(min=1.0)
    mean_edit = (abs_diff_tok * mask_sorted.float()).sum(dim=1) / denom
    return mean_edit


def _attention_kl_loss_masked(
    attn_pred: torch.Tensor,
    attn_target: torch.Tensor,
    mask_pred: torch.Tensor,
    mask_target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    # attn_*: [B, L], masks: [B, L]
    joint = (mask_pred | mask_target).float()
    denom_p = (attn_pred * joint).sum(dim=1, keepdim=True)
    denom_t = (attn_target * joint).sum(dim=1, keepdim=True)
    valid = (denom_p.squeeze(1) > eps) & (denom_t.squeeze(1) > eps)
    if valid.sum().item() == 0:
        return torch.zeros((), device=attn_pred.device)

    p = (attn_pred * joint) / (denom_p + eps)
    t = (attn_target * joint) / (denom_t + eps)
    p = torch.clamp(p, eps, 1.0)
    t = torch.clamp(t, eps, 1.0)

    kl_t_p = (t * (torch.log(t) - torch.log(p))).sum(dim=1)
    kl_p_t = (p * (torch.log(p) - torch.log(t))).sum(dim=1)
    return 0.5 * (kl_t_p[valid].mean() + kl_p_t[valid].mean())


def _compose_teacher_guided_reco_total(
    losses_raw: Dict[str, torch.Tensor],
    ema_state: Dict[str, float] | None,
    normalize_terms: bool,
    ema_decay: float,
    norm_eps: float,
    w_logit: float,
    w_emb: float,
    w_tok: float,
    w_phys: float,
    w_budget: float,
    update_ema: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float] | None]:
    terms = {
        "kd": losses_raw["kd"],
        "emb": losses_raw["emb"],
        "tok": losses_raw["tok"],
        "phys": losses_raw["phys"],
        "budget": losses_raw["budget_hinge"],
    }
    if not bool(normalize_terms):
        total = (
            float(w_logit) * terms["kd"]
            + float(w_emb) * terms["emb"]
            + float(w_tok) * terms["tok"]
            + float(w_phys) * terms["phys"]
            + float(w_budget) * terms["budget"]
        )
        return total, terms, ema_state

    if ema_state is None:
        ema_state = {k: float(max(v.detach().item(), float(norm_eps))) for k, v in terms.items()}

    norm_terms: Dict[str, torch.Tensor] = {}
    for k, v in terms.items():
        cur = float(v.detach().item())
        prev = float(ema_state.get(k, max(cur, float(norm_eps))))
        nxt = float(ema_decay) * prev + (1.0 - float(ema_decay)) * cur if bool(update_ema) else prev
        if bool(update_ema):
            ema_state[k] = max(nxt, float(norm_eps))
        denom = max(float(ema_state[k]), float(norm_eps))
        norm_terms[k] = v / denom

    total = (
        float(w_logit) * norm_terms["kd"]
        + float(w_emb) * norm_terms["emb"]
        + float(w_tok) * norm_terms["tok"]
        + float(w_phys) * norm_terms["phys"]
        + float(w_budget) * norm_terms["budget"]
    )
    return total, norm_terms, ema_state


def _compute_teacher_guided_reco_losses(
    reco_out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    teacher_model: nn.Module,
    means_t: torch.Tensor,
    stds_t: torch.Tensor,
    loss_cfg: Dict,
    kd_temperature: float,
    budget_eps: float,
    budget_weight_floor: float,
    stagea_fused_target: Optional[torch.Tensor] = None,
    stagea_anchor_target: Optional[torch.Tensor] = None,
    stagea_fused_adv_weight: float = 0.0,
    stagea_fused_adv_power: float = 1.0,
    stagea_fused_uncert_weight: float = 0.0,
    stagea_fused_adv_use_abs: bool = False,
    stagea_fused_kd_w_min: float = 0.25,
    stagea_fused_kd_w_max: float = 4.0,
    stagea_lambda_delta_aux: float = 0.0,
) -> Dict[str, torch.Tensor]:
    aux_losses = compute_reconstruction_losses(
        reco_out,
        const_hlt,
        mask_hlt,
        const_off,
        mask_off,
        budget_merge_true,
        budget_eff_true,
        loss_cfg,
    )
    loss_phys = aux_losses["phys"]

    with torch.no_grad():
        feat_off_raw = compute_features_torch(const_off, mask_off)
        feat_off_std = _standardize_features_torch(feat_off_raw, mask_off, means_t, stds_t)
        off_pack = teacher_model(feat_off_std, mask_off, return_attention=True, return_embedding=True)
        logits_teacher_off = off_pack[0].view(-1)
        attn_teacher_off = off_pack[1]
        emb_teacher_off = off_pack[2]

    feat_reco_raw, mask_reco = _build_teacher_reco_features_from_output(
        reco_out,
        const_hlt,
        mask_hlt,
        weight_floor=budget_weight_floor,
    )
    feat_reco_std = _standardize_features_torch(feat_reco_raw, mask_reco, means_t, stds_t)
    reco_pack = teacher_model(feat_reco_std, mask_reco, return_attention=True, return_embedding=True)
    logits_teacher_reco = reco_pack[0].view(-1)
    attn_teacher_reco = reco_pack[1]
    emb_teacher_reco = reco_pack[2]

    anchor_soft_teacher = torch.sigmoid(logits_teacher_off / kd_temperature)
    if stagea_anchor_target is not None:
        anchor_soft = torch.clamp(stagea_anchor_target.view(-1), min=0.0, max=1.0)
    else:
        anchor_soft = anchor_soft_teacher
    if stagea_fused_target is not None:
        target_soft = torch.clamp(stagea_fused_target.view(-1), min=0.0, max=1.0)
    else:
        target_soft = anchor_soft

    kd_sample_w = _build_advantage_weights(
        target_soft=target_soft,
        anchor_soft=anchor_soft,
        adv_weight=float(max(stagea_fused_adv_weight, 0.0)),
        adv_power=float(max(stagea_fused_adv_power, 1e-6)),
        uncert_weight=float(max(stagea_fused_uncert_weight, 0.0)),
        use_abs_delta=bool(stagea_fused_adv_use_abs),
        w_min=float(max(stagea_fused_kd_w_min, 1e-6)),
        w_max=float(max(stagea_fused_kd_w_max, max(stagea_fused_kd_w_min, 1e-6))),
    )

    kd_vec = (
        F.binary_cross_entropy_with_logits(
            logits_teacher_reco / kd_temperature,
            target_soft,
            reduction="none",
        )
        * (kd_temperature * kd_temperature)
    )
    loss_kd = _weighted_batch_mean(kd_vec, kd_sample_w)

    if float(stagea_lambda_delta_aux) > 0.0:
        reco_soft = torch.sigmoid(logits_teacher_reco / kd_temperature)
        delta_pred = reco_soft - anchor_soft
        delta_tgt = target_soft - anchor_soft
        delta_aux_vec = (delta_pred - delta_tgt).pow(2)
        delta_aux = _weighted_batch_mean(delta_aux_vec, kd_sample_w)
        loss_kd = loss_kd + float(stagea_lambda_delta_aux) * delta_aux

    # Embedding alignment (jet-level representation consistency).
    emb_off_n = F.normalize(emb_teacher_off, dim=1)
    emb_reco_n = F.normalize(emb_teacher_reco, dim=1)
    loss_emb = (1.0 - (emb_off_n * emb_reco_n).sum(dim=1)).mean()

    # Token-level teacher alignment via pooled-attention distributions.
    loss_tok = _attention_kl_loss_masked(
        attn_pred=attn_teacher_reco,
        attn_target=attn_teacher_off,
        mask_pred=mask_reco,
        mask_target=mask_off,
    )

    reco_tokens = reco_out["cand_tokens"][:, : const_hlt.shape[1], :]
    mean_edit_vec = _sorted_edit_budget_vec(reco_tokens, const_hlt, mask_hlt)
    budget_hinge_vec = F.relu(mean_edit_vec - budget_eps)
    loss_budget_hinge = _weighted_batch_mean(budget_hinge_vec, None)

    return {
        "kd": loss_kd,
        "emb": loss_emb,
        "tok": loss_tok,
        "phys": loss_phys,
        "budget_hinge": loss_budget_hinge,
        "logits_teacher_reco": logits_teacher_reco,
        "kd_weight_mean": kd_sample_w.mean(),
        "target_soft_mean": target_soft.mean(),
        "anchor_soft_mean": anchor_soft.mean(),
    }


def train_reconstructor_teacher_guided(
    model: OfflineReconstructor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
    teacher_model: nn.Module,
    feat_means: np.ndarray,
    feat_stds: np.ndarray,
    kd_temperature: float,
    lambda_kd: float,
    lambda_emb: float,
    lambda_tok: float,
    lambda_phys: float,
    lambda_budget_hinge: float,
    budget_eps: float,
    budget_weight_floor: float,
    target_tpr_for_fpr: float,
    normalize_loss_terms: bool,
    loss_norm_ema_decay: float,
    loss_norm_eps: float,
    reload_best_at_stage_transition: bool,
    stagea_fused_adv_weight: float = 0.0,
    stagea_fused_adv_power: float = 1.0,
    stagea_fused_uncert_weight: float = 0.0,
    stagea_fused_adv_use_abs: bool = False,
    stagea_fused_kd_w_min: float = 0.25,
    stagea_fused_kd_w_max: float = 4.0,
    stagea_lambda_delta_aux: float = 0.0,
) -> Tuple[OfflineReconstructor, Dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    kd_temperature = max(float(kd_temperature), 1e-3)
    lambda_kd = float(max(lambda_kd, 0.0))
    lambda_emb = float(max(lambda_emb, 0.0))
    lambda_tok = float(max(lambda_tok, 0.0))
    lambda_phys = float(max(lambda_phys, 0.0))
    lambda_budget_hinge = float(max(lambda_budget_hinge, 0.0))
    budget_eps = float(max(budget_eps, 0.0))
    budget_weight_floor = float(max(budget_weight_floor, 0.0))
    loss_norm_ema_decay = float(np.clip(loss_norm_ema_decay, 0.0, 0.9999))
    loss_norm_eps = float(max(loss_norm_eps, 1e-12))
    stagea_fused_adv_weight = float(max(stagea_fused_adv_weight, 0.0))
    stagea_fused_adv_power = float(max(stagea_fused_adv_power, 1e-6))
    stagea_fused_uncert_weight = float(max(stagea_fused_uncert_weight, 0.0))
    stagea_fused_kd_w_min = float(max(stagea_fused_kd_w_min, 1e-6))
    stagea_fused_kd_w_max = float(max(stagea_fused_kd_w_max, stagea_fused_kd_w_min))
    stagea_lambda_delta_aux = float(max(stagea_lambda_delta_aux, 0.0))

    means_t = torch.tensor(feat_means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(np.clip(feat_stds, 1e-6, None), dtype=torch.float32, device=device)

    teacher_model.eval()
    for p_t in teacher_model.parameters():
        p_t.requires_grad_(False)

    best_state = None
    best_val_auc = float("-inf")
    no_improve = 0
    best_metrics: Dict[str, float] = {}
    min_stop_epoch = int(train_cfg.get("stage2_epochs", 0)) + int(train_cfg.get("min_full_scale_epochs", 5))

    reco_loss_ema_state = {
        "kd": 1.0,
        "emb": 1.0,
        "tok": 1.0,
        "phys": 1.0,
        "budget": 1.0,
    }

    total_epochs = int(train_cfg["epochs"])
    stage1_end = int(max(int(train_cfg.get("stage1_epochs", 0)), 0))
    stage2_end = int(max(int(train_cfg.get("stage2_epochs", stage1_end)), stage1_end))
    stage1_end = min(stage1_end, total_epochs)
    stage2_end = min(max(stage2_end, stage1_end), total_epochs)

    def _phase_idx(epoch: int) -> int:
        if epoch < stage1_end:
            return 0
        if epoch < stage2_end:
            return 1
        return 2

    phase_names = {
        0: "phase_035",
        1: "phase_070",
        2: "phase_100",
    }
    current_phase = _phase_idx(0)
    phase_best: Dict[int, Dict[str, object]] = {}

    for ep in tqdm(range(int(train_cfg["epochs"])), desc="Reconstructor"):
        phase_idx = _phase_idx(ep)
        if ep > 0 and phase_idx != current_phase:
            if bool(reload_best_at_stage_transition):
                prev_best = phase_best.get(current_phase, None)
                if prev_best is not None:
                    state_pack = prev_best["state"]
                    model.load_state_dict(state_pack["model"])
                    opt.load_state_dict(state_pack["opt"])
                    sch.load_state_dict(state_pack["sch"])
                    reco_loss_ema_state = {k: float(v) for k, v in state_pack["ema"].items()}
                    no_improve = 0
                    print(
                        f"Stage-A transition -> reloaded best from {phase_names[current_phase]} "
                        f"(epoch={int(prev_best['epoch'])+1}, val_teacher_auc={float(prev_best['val_auc']):.4f})"
                    )
                else:
                    print(
                        f"Stage-A transition -> no stored best for {phase_names[current_phase]}, "
                        "continuing with current weights."
                    )
            current_phase = phase_idx

        model.train()
        sc = stage_scale_local(ep, train_cfg)

        tr_total = tr_kd = tr_emb = tr_tok = tr_phys = tr_budget_hinge = 0.0
        tr_kd_w = 0.0
        n_tr = 0
        tr_probs_all = []
        tr_labels_all = []

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            labels_batch = batch["label"].to(device)
            budget_merge_true = batch["budget_merge_true"].to(device)
            budget_eff_true = batch["budget_eff_true"].to(device)
            fused_target_batch = batch.get("stagea_fused_target", None)
            if fused_target_batch is not None:
                fused_target_batch = fused_target_batch.to(device)
            anchor_target_batch = batch.get("stagea_anchor_target", None)
            if anchor_target_batch is not None:
                anchor_target_batch = anchor_target_batch.to(device)

            opt.zero_grad()
            out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=sc)

            losses = _compute_teacher_guided_reco_losses(
                reco_out=out,
                const_hlt=const_hlt,
                mask_hlt=mask_hlt,
                const_off=const_off,
                mask_off=mask_off,
                budget_merge_true=budget_merge_true,
                budget_eff_true=budget_eff_true,
                teacher_model=teacher_model,
                means_t=means_t,
                stds_t=stds_t,
                loss_cfg=loss_cfg,
                kd_temperature=kd_temperature,
                budget_eps=budget_eps,
                budget_weight_floor=budget_weight_floor,
                stagea_fused_target=fused_target_batch,
                stagea_anchor_target=anchor_target_batch,
                stagea_fused_adv_weight=stagea_fused_adv_weight,
                stagea_fused_adv_power=stagea_fused_adv_power,
                stagea_fused_uncert_weight=stagea_fused_uncert_weight,
                stagea_fused_adv_use_abs=bool(stagea_fused_adv_use_abs),
                stagea_fused_kd_w_min=stagea_fused_kd_w_min,
                stagea_fused_kd_w_max=stagea_fused_kd_w_max,
                stagea_lambda_delta_aux=stagea_lambda_delta_aux,
            )
            loss_total, _, reco_loss_ema_state = _compose_teacher_guided_reco_total(
                losses_raw=losses,
                ema_state=reco_loss_ema_state,
                normalize_terms=bool(normalize_loss_terms),
                ema_decay=loss_norm_ema_decay,
                norm_eps=loss_norm_eps,
                w_logit=lambda_kd,
                w_emb=lambda_emb,
                w_tok=lambda_tok,
                w_phys=lambda_phys,
                w_budget=lambda_budget_hinge,
                update_ema=True,
            )
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            probs_reco = torch.sigmoid(losses["logits_teacher_reco"]).detach().cpu().numpy()
            tr_probs_all.append(probs_reco)
            tr_labels_all.append(labels_batch.detach().cpu().numpy().astype(np.int64))

            bs = feat_hlt.size(0)
            tr_total += loss_total.item() * bs
            tr_kd += losses["kd"].item() * bs
            tr_emb += losses["emb"].item() * bs
            tr_tok += losses["tok"].item() * bs
            tr_phys += losses["phys"].item() * bs
            tr_budget_hinge += losses["budget_hinge"].item() * bs
            tr_kd_w += losses["kd_weight_mean"].item() * bs
            n_tr += bs

        model.eval()
        va_total = va_kd = va_emb = va_tok = va_phys = va_budget_hinge = 0.0
        va_kd_w = 0.0
        n_va = 0
        va_probs_all = []
        va_labels_all = []

        with torch.no_grad():
            for batch in val_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                const_off = batch["const_off"].to(device)
                mask_off = batch["mask_off"].to(device)
                labels_batch = batch["label"].to(device)
                budget_merge_true = batch["budget_merge_true"].to(device)
                budget_eff_true = batch["budget_eff_true"].to(device)
                fused_target_batch = batch.get("stagea_fused_target", None)
                if fused_target_batch is not None:
                    fused_target_batch = fused_target_batch.to(device)
                anchor_target_batch = batch.get("stagea_anchor_target", None)
                if anchor_target_batch is not None:
                    anchor_target_batch = anchor_target_batch.to(device)

                out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
                losses = _compute_teacher_guided_reco_losses(
                    reco_out=out,
                    const_hlt=const_hlt,
                    mask_hlt=mask_hlt,
                    const_off=const_off,
                    mask_off=mask_off,
                    budget_merge_true=budget_merge_true,
                    budget_eff_true=budget_eff_true,
                    teacher_model=teacher_model,
                    means_t=means_t,
                    stds_t=stds_t,
                    loss_cfg=loss_cfg,
                    kd_temperature=kd_temperature,
                    budget_eps=budget_eps,
                    budget_weight_floor=budget_weight_floor,
                    stagea_fused_target=fused_target_batch,
                    stagea_anchor_target=anchor_target_batch,
                    stagea_fused_adv_weight=stagea_fused_adv_weight,
                    stagea_fused_adv_power=stagea_fused_adv_power,
                    stagea_fused_uncert_weight=stagea_fused_uncert_weight,
                    stagea_fused_adv_use_abs=bool(stagea_fused_adv_use_abs),
                    stagea_fused_kd_w_min=stagea_fused_kd_w_min,
                    stagea_fused_kd_w_max=stagea_fused_kd_w_max,
                    stagea_lambda_delta_aux=stagea_lambda_delta_aux,
                )
                loss_total, _, _ = _compose_teacher_guided_reco_total(
                    losses_raw=losses,
                    ema_state=reco_loss_ema_state,
                    normalize_terms=bool(normalize_loss_terms),
                    ema_decay=loss_norm_ema_decay,
                    norm_eps=loss_norm_eps,
                    w_logit=lambda_kd,
                    w_emb=lambda_emb,
                    w_tok=lambda_tok,
                    w_phys=lambda_phys,
                    w_budget=lambda_budget_hinge,
                    update_ema=False,
                )
                probs_reco = torch.sigmoid(losses["logits_teacher_reco"]).detach().cpu().numpy()
                va_probs_all.append(probs_reco)
                va_labels_all.append(labels_batch.detach().cpu().numpy().astype(np.int64))

                bs = feat_hlt.size(0)
                va_total += loss_total.item() * bs
                va_kd += losses["kd"].item() * bs
                va_emb += losses["emb"].item() * bs
                va_tok += losses["tok"].item() * bs
                va_phys += losses["phys"].item() * bs
                va_budget_hinge += losses["budget_hinge"].item() * bs
                va_kd_w += losses["kd_weight_mean"].item() * bs
                n_va += bs

        sch.step()

        tr_total /= max(n_tr, 1)
        tr_kd /= max(n_tr, 1)
        tr_emb /= max(n_tr, 1)
        tr_tok /= max(n_tr, 1)
        tr_phys /= max(n_tr, 1)
        tr_budget_hinge /= max(n_tr, 1)
        tr_kd_w /= max(n_tr, 1)

        va_total /= max(n_va, 1)
        va_kd /= max(n_va, 1)
        va_emb /= max(n_va, 1)
        va_tok /= max(n_va, 1)
        va_phys /= max(n_va, 1)
        va_budget_hinge /= max(n_va, 1)
        va_kd_w /= max(n_va, 1)

        tr_probs = np.concatenate(tr_probs_all, axis=0) if len(tr_probs_all) > 0 else np.zeros((0,), dtype=np.float32)
        tr_labels = np.concatenate(tr_labels_all, axis=0) if len(tr_labels_all) > 0 else np.zeros((0,), dtype=np.int64)
        va_probs = np.concatenate(va_probs_all, axis=0) if len(va_probs_all) > 0 else np.zeros((0,), dtype=np.float32)
        va_labels = np.concatenate(va_labels_all, axis=0) if len(va_labels_all) > 0 else np.zeros((0,), dtype=np.int64)

        if np.unique(tr_labels).size > 1 and tr_probs.size > 0:
            tr_auc = float(roc_auc_score(tr_labels, tr_probs))
            tr_fpr, tr_tpr, _ = roc_curve(tr_labels, tr_probs)
            tr_fpr50 = float(fpr_at_target_tpr(tr_fpr, tr_tpr, float(target_tpr_for_fpr)))
        else:
            tr_auc, tr_fpr50 = float("nan"), float("nan")

        if np.unique(va_labels).size > 1 and va_probs.size > 0:
            va_auc = float(roc_auc_score(va_labels, va_probs))
            va_fpr, va_tpr, _ = roc_curve(va_labels, va_probs)
            va_fpr50 = float(fpr_at_target_tpr(va_fpr, va_tpr, float(target_tpr_for_fpr)))
        else:
            va_auc, va_fpr50 = float("nan"), float("nan")

        if np.isfinite(va_auc):
            phase_entry = phase_best.get(phase_idx, None)
            if (phase_entry is None) or (va_auc > float(phase_entry["val_auc"])):
                phase_best[phase_idx] = {
                    "val_auc": float(va_auc),
                    "val_fpr50": float(va_fpr50),
                    "epoch": int(ep),
                    "state": {
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "opt": copy.deepcopy(opt.state_dict()),
                        "sch": copy.deepcopy(sch.state_dict()),
                        "ema": {k: float(v) for k, v in reco_loss_ema_state.items()},
                    },
                }

        if np.isfinite(va_auc) and (va_auc > best_val_auc):
            best_val_auc = float(va_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_metrics = {
                "selected_metric": "teacher_on_reco_val_auc",
                "selected_val_auc": float(va_auc),
                "selected_val_fpr50": float(va_fpr50),
                "selected_val_total_loss": float(va_total),
                "val_total": float(va_total),
                "val_kd": float(va_kd),
                "val_emb": float(va_emb),
                "val_tok": float(va_tok),
                "val_phys": float(va_phys),
                "val_budget_hinge": float(va_budget_hinge),
                "train_total": float(tr_total),
                "train_kd": float(tr_kd),
                "train_emb": float(tr_emb),
                "train_tok": float(tr_tok),
                "train_phys": float(tr_phys),
                "train_budget_hinge": float(tr_budget_hinge),
                "train_teacher_auc": float(tr_auc),
                "train_teacher_fpr50": float(tr_fpr50),
                "val_teacher_auc": float(va_auc),
                "val_teacher_fpr50": float(va_fpr50),
                "loss_normalized": bool(normalize_loss_terms),
                "loss_norm_ema_decay": float(loss_norm_ema_decay),
                "train_kd_weight_mean": float(tr_kd_w),
                "val_kd_weight_mean": float(va_kd_w),
            }
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Ep {ep+1}: train_total={tr_total:.4f}, val_total={va_total:.4f}, "
                f"train_teacher_auc={tr_auc:.4f}, val_teacher_auc={va_auc:.4f}, "
                f"val_teacher_fpr50={va_fpr50:.6f}, best_teacher_auc={best_val_auc:.4f} | "
                f"kd={va_kd:.4f}, emb={va_emb:.4f}, tok={va_tok:.4f}, "
                f"phys={va_phys:.4f}, budget_hinge={va_budget_hinge:.4f}, kd_w={va_kd_w:.3f}, "
                f"stage_scale={sc:.2f}"
            )
        if (ep + 1) >= min_stop_epoch and no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping reconstructor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if len(best_metrics) == 0:
        best_metrics = {}
    best_metrics["stageA_stagewise_best_reload"] = bool(reload_best_at_stage_transition)
    best_metrics["stageA_phase_best"] = {
        phase_names[int(k)]: {
            "epoch": int(v["epoch"]) + 1,
            "val_teacher_auc": float(v["val_auc"]),
            "val_teacher_fpr50": float(v["val_fpr50"]),
        }
        for k, v in sorted(phase_best.items(), key=lambda kv: int(kv[0]))
    }
    return model, best_metrics


def compute_features_torch(const: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Torch equivalent of compute_features(...) from unmerge_correct_hlt.py.
    const: [B, L, 4] => [pt, eta, phi, E]
    mask:  [B, L] bool
    returns: [B, L, 7]
    """
    eps = 1e-8
    pt = const[..., 0].clamp(min=eps)
    eta = const[..., 1].clamp(min=-5.0, max=5.0)
    phi = const[..., 2]
    E = const[..., 3].clamp(min=eps)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    w = mask.float()
    jet_px = (px * w).sum(dim=1, keepdim=True)
    jet_py = (py * w).sum(dim=1, keepdim=True)
    jet_pz = (pz * w).sum(dim=1, keepdim=True)
    jet_E = (E * w).sum(dim=1, keepdim=True)

    jet_pt = torch.sqrt(jet_px.pow(2) + jet_py.pow(2) + eps)
    jet_p = torch.sqrt(jet_px.pow(2) + jet_py.pow(2) + jet_pz.pow(2) + eps)
    ratio = (jet_p + jet_pz) / (jet_p - jet_pz + eps)
    ratio = torch.clamp(ratio, min=1e-8, max=1e8)
    jet_eta = 0.5 * torch.log(ratio)
    jet_phi = torch.atan2(jet_py, jet_px)

    delta_eta = eta - jet_eta
    delta_phi = torch.atan2(torch.sin(phi - jet_phi), torch.cos(phi - jet_phi))

    log_pt = torch.log(pt + eps)
    log_E = torch.log(E + eps)
    log_pt_rel = torch.log(pt / (jet_pt + eps) + eps)
    log_E_rel = torch.log(E / (jet_E + eps) + eps)
    delta_R = torch.sqrt(delta_eta.pow(2) + delta_phi.pow(2) + eps)

    feat = torch.stack(
        [delta_eta, delta_phi, log_pt, log_E, log_pt_rel, log_E_rel, delta_R],
        dim=-1,
    )
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = torch.clamp(feat, min=-20.0, max=20.0)
    feat = feat * w.unsqueeze(-1)
    return feat


def build_soft_corrected_view(
    reco_out: Dict[str, torch.Tensor],
    weight_floor: float = 1e-4,
    scale_features_by_weight: bool = True,
    include_flags: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds a differentiable fixed-length corrected view.
    The DualViewCrossAttnClassifier expects both views to share the same sequence length.
    We therefore map reconstructor outputs back to L token slots (L = HLT token count):
      - corrected token kinematics from tok branch
      - parent-level split mass summary
      - per-token share of efficiency budget

    Output feature dims:
      - default: 7 base kinematic features + 3 channels
        [tok_weight, parent_added_weight, eff_share] = 10.
      - with include_flags: +2 channels
        [merge_flag, eff_flag] => 12.
    """
    eps = 1e-8
    L = reco_out["action_prob"].shape[1]
    tok_tokens = reco_out["cand_tokens"][:, :L, :]
    tok_w = reco_out["cand_weights"][:, :L].clamp(0.0, 1.0)
    mask_b = tok_w > float(weight_floor)
    none_valid = ~mask_b.any(dim=1)
    if none_valid.any():
        mask_b = mask_b.clone()
        mask_b[none_valid, 0] = True

    feat7 = compute_features_torch(tok_tokens, mask_b)
    if scale_features_by_weight:
        feat7 = feat7 * tok_w.unsqueeze(-1)

    # Parent-level merge-added mass per token from split branch.
    child_w = reco_out["child_weight"]
    K = max(int(child_w.shape[1] // max(L, 1)), 1)
    parent_added = child_w.reshape(child_w.shape[0], L, K).sum(dim=2).clamp(0.0, 1.0)

    # Optional second-stage split mass (split-again), aggregated back to parent slots.
    if bool(SPLIT_AGAIN_CFG.get("enabled", False)):
        cw = child_w.reshape(child_w.shape[0], L, K).clamp(0.0, 1.0)
        split_delta = reco_out["split_delta"]
        dR = torch.sqrt(split_delta[..., 1].pow(2) + split_delta[..., 2].pow(2) + 1e-8)
        p_split = reco_out["action_prob"][..., 2].unsqueeze(-1).clamp(0.0, 1.0)

        exist_thr = float(SPLIT_AGAIN_CFG.get("exist_thr", 0.30))
        psplit_thr = float(SPLIT_AGAIN_CFG.get("psplit_thr", 0.20))
        dr_thr = float(SPLIT_AGAIN_CFG.get("dr_thr", 0.10))
        alpha = float(SPLIT_AGAIN_CFG.get("alpha", 8.0))
        beta = float(SPLIT_AGAIN_CFG.get("beta", 4.0))
        gamma = float(SPLIT_AGAIN_CFG.get("gamma", 3.0))
        score_power = max(float(SPLIT_AGAIN_CFG.get("score_power", 1.25)), 1.0)
        budget_frac = max(float(SPLIT_AGAIN_CFG.get("budget_frac", 1.0)), 0.0)
        max_parent_added = max(float(SPLIT_AGAIN_CFG.get("max_parent_added", 2.0)), 1.0)

        # Score: high when child exists strongly, parent is split-prone, and local split displacement is large.
        split_again_logit = (
            alpha * (cw - exist_thr)
            + beta * (p_split - psplit_thr)
            + gamma * (dR - dr_thr)
        )
        split_again_score = torch.sigmoid(split_again_logit).pow(score_power)
        split_again_raw = (cw * split_again_score).clamp(0.0, 1.0)

        # Budget-aware cap: only use remaining merge budget fraction after first-stage children.
        merge_budget = reco_out["budget_merge"].clamp(min=0.0)
        used_lvl1 = cw.sum(dim=(1, 2))
        remain = (budget_frac * merge_budget - used_lvl1).clamp(min=0.0)
        raw_sum = split_again_raw.sum(dim=(1, 2))
        scale = torch.where(raw_sum > 1e-8, (remain / (raw_sum + 1e-8)).clamp(0.0, 1.0), torch.zeros_like(raw_sum))
        split_again = split_again_raw * scale.view(-1, 1, 1)

        parent_added_lvl2 = split_again.sum(dim=2)
        parent_added = (parent_added + parent_added_lvl2).clamp(0.0, max_parent_added)

    # Distribute efficiency budget as a smooth per-token share signal.
    valid_count = mask_b.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    eff_share = (reco_out["budget_eff"].unsqueeze(1) / valid_count).clamp(0.0, 1.0)
    eff_share = eff_share * mask_b.float()

    extra = torch.stack([tok_w, parent_added, eff_share], dim=-1)
    if include_flags:
        tok_merge_flag = reco_out["cand_merge_flags"][:, :L].clamp(0.0, 1.0)
        tok_eff_flag = reco_out["cand_eff_flags"][:, :L].clamp(0.0, 1.0)
        extra = torch.cat([extra, tok_merge_flag.unsqueeze(-1), tok_eff_flag.unsqueeze(-1)], dim=-1)
    feat_b = torch.cat([feat7, extra], dim=-1)
    feat_b = torch.nan_to_num(feat_b, nan=0.0, posinf=0.0, neginf=0.0)
    feat_b = feat_b * mask_b.unsqueeze(-1).float()
    return feat_b, mask_b


def low_fpr_surrogate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_tpr: float = 0.50,
    tau: float = 0.05,
) -> torch.Tensor:
    """
    Differentiable proxy that targets low FPR around fixed TPR operating point.
    """
    probs = torch.sigmoid(logits)
    pos = probs[labels > 0.5]
    neg = probs[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.zeros((), device=logits.device)

    # TPR=0.5 implies threshold near median positive score.
    q = float(max(0.0, min(1.0, 1.0 - target_tpr)))
    thr = torch.quantile(pos.detach(), q=q)

    # Soft FPR proxy: negatives above threshold.
    neg_term = torch.sigmoid((neg - thr) / max(float(tau), 1e-4)).mean()
    # Keep positive side calibrated around threshold.
    pos_term = torch.sigmoid((thr - pos) / max(float(tau), 1e-4)).mean()
    return neg_term + 0.5 * pos_term


@torch.no_grad()
def eval_joint_model(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    dual_model.eval()
    reconstructor.eval()

    preds = []
    labs = []
    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        feat_hlt_dual = batch["feat_hlt_dual"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = build_soft_corrected_view(
            reco_out,
            weight_floor=corrected_weight_floor,
            scale_features_by_weight=True,
            include_flags=corrected_use_flags,
        )
        logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)
        p = torch.sigmoid(logits)
        preds.append(p.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    labs = np.concatenate(labs) if labs else np.zeros(0, dtype=np.float32)
    if preds.size == 0:
        return float("nan"), preds, labs, float("nan")
    auc = roc_auc_score(labs, preds) if len(np.unique(labs)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(labs, preds)
    fpr50 = fpr_at_target_tpr(fpr, tpr, 0.50)
    return float(auc), preds, labs, float(fpr50)


@torch.no_grad()
def build_corrected_view_numpy(
    reconstructor: OfflineReconstructor,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    device: torch.device,
    batch_size: int,
    corrected_weight_floor: float,
    corrected_use_flags: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    n_jets, seq_len, _ = feat_hlt.shape
    out_dim_b = 12 if corrected_use_flags else 10
    feat_b = np.zeros((n_jets, seq_len, out_dim_b), dtype=np.float32)
    mask_b = np.zeros((n_jets, seq_len), dtype=bool)

    reconstructor.eval()
    for start in range(0, n_jets, int(batch_size)):
        end = min(start + int(batch_size), n_jets)
        x = torch.tensor(feat_hlt[start:end], dtype=torch.float32, device=device)
        m = torch.tensor(mask_hlt[start:end], dtype=torch.bool, device=device)
        c = torch.tensor(const_hlt[start:end], dtype=torch.float32, device=device)
        reco_out = reconstructor(x, m, c, stage_scale=1.0)
        fb, mb = build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=bool(corrected_use_flags),
        )
        feat_b[start:end] = fb.detach().cpu().numpy()
        mask_b[start:end] = mb.detach().cpu().numpy()
    return feat_b, mask_b


def summarize_soft_corrected_view(
    feat_b: np.ndarray,
    mask_b: np.ndarray,
) -> Dict[str, float]:
    # Extra channels: [tok_weight, parent_added_weight, eff_share] (+ optional merge/eff flags)
    tok_w = feat_b[..., 7]
    parent_added = feat_b[..., 8]
    eff_share = feat_b[..., 9]
    has_flags = feat_b.shape[-1] >= 12
    merge_flag_soft = feat_b[..., 10] if has_flags else np.zeros_like(tok_w)
    eff_flag_soft = feat_b[..., 11] if has_flags else np.zeros_like(tok_w)
    valid = mask_b.astype(bool)
    if not np.any(valid):
        return {
            "mean_tokens_active_per_jet": 0.0,
            "mean_tok_weight_valid": 0.0,
            "mean_parent_added_valid": 0.0,
            "mean_eff_share_valid": 0.0,
            "mean_merge_flag_soft_valid": 0.0,
            "mean_eff_flag_soft_valid": 0.0,
            "p95_tok_weight_valid": 0.0,
            "p95_parent_added_valid": 0.0,
            "p95_eff_share_valid": 0.0,
            "p95_merge_flag_soft_valid": 0.0,
            "p95_eff_flag_soft_valid": 0.0,
        }
    tok_cnt = valid.sum(axis=1).astype(np.float64)
    return {
        "mean_tokens_active_per_jet": float(tok_cnt.mean()),
        "mean_tok_weight_valid": float(tok_w[valid].mean()),
        "mean_parent_added_valid": float(parent_added[valid].mean()),
        "mean_eff_share_valid": float(eff_share[valid].mean()),
        "mean_merge_flag_soft_valid": float(merge_flag_soft[valid].mean()) if has_flags else 0.0,
        "mean_eff_flag_soft_valid": float(eff_flag_soft[valid].mean()) if has_flags else 0.0,
        "p95_tok_weight_valid": float(np.percentile(tok_w[valid], 95.0)),
        "p95_parent_added_valid": float(np.percentile(parent_added[valid], 95.0)),
        "p95_eff_share_valid": float(np.percentile(eff_share[valid], 95.0)),
        "p95_merge_flag_soft_valid": float(np.percentile(merge_flag_soft[valid], 95.0)) if has_flags else 0.0,
        "p95_eff_flag_soft_valid": float(np.percentile(eff_flag_soft[valid], 95.0)) if has_flags else 0.0,
    }



def _threshold_for_target_tpr(preds: np.ndarray, labels: np.ndarray, target_tpr: float) -> float:
    target_tpr = float(np.clip(target_tpr, 0.0, 1.0))
    pos = preds[labels > 0.5]
    if pos.size == 0:
        return float("inf")
    q = float(np.clip(1.0 - target_tpr, 0.0, 1.0))
    return float(np.quantile(pos, q=q))


def _binary_rates_from_mask(pred_mask: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    labels_b = labels > 0.5
    neg_b = ~labels_b
    n_pos = int(labels_b.sum())
    n_neg = int(neg_b.sum())
    tp = int((pred_mask & labels_b).sum())
    fp = int((pred_mask & neg_b).sum())
    tpr = float(tp / max(n_pos, 1))
    fpr = float(fp / max(n_neg, 1))
    return {
        "tp": tp,
        "fp": fp,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tpr": tpr,
        "fpr": fpr,
    }


def build_overlap_report_at_tpr(
    labels: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    target_tpr: float,
) -> Dict[str, object]:
    labels = labels.astype(np.float32)
    selections: Dict[str, Dict[str, object]] = {}
    model_names = list(model_preds.keys())
    for name in model_names:
        preds = np.asarray(model_preds[name], dtype=np.float64)
        thr = _threshold_for_target_tpr(preds, labels, target_tpr)
        sel = preds >= thr
        rates = _binary_rates_from_mask(sel, labels)
        selections[name] = {
            "threshold": float(thr),
            "selected_count": int(sel.sum()),
            "selected_fraction": float(sel.mean()) if sel.size > 0 else 0.0,
            "tp_count": int(rates["tp"]),
            "fp_count": int(rates["fp"]),
            "tpr": float(rates["tpr"]),
            "fpr": float(rates["fpr"]),
            "pred_mask": sel,
        }

    pairs: Dict[str, Dict[str, float]] = {}
    labels_pos = labels > 0.5
    labels_neg = ~labels_pos
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a = model_names[i]
            b = model_names[j]
            sa = selections[a]["pred_mask"]
            sb = selections[b]["pred_mask"]

            inter = sa & sb
            union = sa | sb
            a_only = sa & (~sb)
            b_only = sb & (~sa)

            inter_tp = inter & labels_pos
            union_tp = union & labels_pos
            a_tp = sa & labels_pos
            b_tp = sb & labels_pos
            a_only_tp = a_only & labels_pos
            b_only_tp = b_only & labels_pos

            pair_key = f"{a}__{b}"
            pairs[pair_key] = {
                "overlap_selected_count": int(inter.sum()),
                "overlap_selected_frac_of_a": float(inter.sum() / max(sa.sum(), 1)),
                "overlap_selected_frac_of_b": float(inter.sum() / max(sb.sum(), 1)),
                "overlap_selected_jaccard": float(inter.sum() / max(union.sum(), 1)),
                "a_only_selected_count": int(a_only.sum()),
                "b_only_selected_count": int(b_only.sum()),
                "overlap_tp_count": int(inter_tp.sum()),
                "overlap_tp_frac_of_a_tp": float(inter_tp.sum() / max(a_tp.sum(), 1)),
                "overlap_tp_frac_of_b_tp": float(inter_tp.sum() / max(b_tp.sum(), 1)),
                "overlap_tp_jaccard": float(inter_tp.sum() / max(union_tp.sum(), 1)),
                "a_only_tp_count": int(a_only_tp.sum()),
                "b_only_tp_count": int(b_only_tp.sum()),
                "a_only_fp_count": int((a_only & labels_neg).sum()),
                "b_only_fp_count": int((b_only & labels_neg).sum()),
            }

    clean_models: Dict[str, Dict[str, float]] = {}
    for name, info in selections.items():
        clean_models[name] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in info.items() if k != "pred_mask"}

    return {
        "target_tpr": float(target_tpr),
        "models": clean_models,
        "pairs": pairs,
    }


def search_best_weighted_combo_at_tpr(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    name_a: str,
    name_b: str,
    target_tpr: float,
    weight_step: float,
) -> Dict[str, float]:
    labels = labels.astype(np.float32)
    preds_a = np.asarray(preds_a, dtype=np.float64)
    preds_b = np.asarray(preds_b, dtype=np.float64)
    step = float(max(weight_step, 1e-4))

    best = {
        "name_a": name_a,
        "name_b": name_b,
        "target_tpr": float(target_tpr),
        "weight_step": float(step),
        "w_a": float("nan"),
        "w_b": float("nan"),
        "threshold": float("nan"),
        "tpr": float("nan"),
        "fpr": float("inf"),
        "tp": 0,
        "fp": 0,
    }

    w_vals = np.arange(0.0, 1.0 + 0.5 * step, step, dtype=np.float64)
    for w_a in w_vals:
        w_b = 1.0 - w_a
        score = w_a * preds_a + w_b * preds_b
        thr = _threshold_for_target_tpr(score, labels, target_tpr)
        pred_mask = score >= thr
        rates = _binary_rates_from_mask(pred_mask, labels)
        fpr = float(rates["fpr"])
        tpr = float(rates["tpr"])

        replace = False
        if fpr < float(best["fpr"]):
            replace = True
        elif np.isfinite(fpr) and np.isclose(fpr, float(best["fpr"])):
            # Tie-break toward closer achieved TPR to target.
            if abs(tpr - float(target_tpr)) < abs(float(best["tpr"]) - float(target_tpr)):
                replace = True

        if replace:
            best = {
                "name_a": name_a,
                "name_b": name_b,
                "target_tpr": float(target_tpr),
                "weight_step": float(step),
                "w_a": float(w_a),
                "w_b": float(w_b),
                "threshold": float(thr),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "tp": int(rates["tp"]),
                "fp": int(rates["fp"]),
            }

    return best



def select_weighted_combo_on_val_and_eval_test(
    labels_val: np.ndarray,
    preds_a_val: np.ndarray,
    preds_b_val: np.ndarray,
    labels_test: np.ndarray,
    preds_a_test: np.ndarray,
    preds_b_test: np.ndarray,
    name_a: str,
    name_b: str,
    target_tpr: float,
    weight_step: float,
) -> Dict[str, object]:
    best_val = search_best_weighted_combo_at_tpr(
        labels=labels_val,
        preds_a=preds_a_val,
        preds_b=preds_b_val,
        name_a=name_a,
        name_b=name_b,
        target_tpr=target_tpr,
        weight_step=weight_step,
    )
    w_a = float(best_val.get("w_a", float("nan")))
    w_b = float(best_val.get("w_b", float("nan")))
    thr = float(best_val.get("threshold", float("nan")))
    if not (np.isfinite(w_a) and np.isfinite(w_b) and np.isfinite(thr)):
        return {
            "selection": {"source": "val", "best": best_val},
            "test_eval": {
                "name_a": name_a,
                "name_b": name_b,
                "target_tpr": float(target_tpr),
                "w_a": float("nan"),
                "w_b": float("nan"),
                "threshold_from_val": float("nan"),
                "tpr": float("nan"),
                "fpr": float("nan"),
                "tp": 0,
                "fp": 0,
            },
        }

    score_test = w_a * np.asarray(preds_a_test, dtype=np.float64) + w_b * np.asarray(preds_b_test, dtype=np.float64)
    pred_test = score_test >= thr
    test_rates = _binary_rates_from_mask(pred_test, labels_test.astype(np.float32))
    return {
        "selection": {"source": "val", "best": best_val},
        "test_eval": {
            "name_a": name_a,
            "name_b": name_b,
            "target_tpr": float(target_tpr),
            "w_a": float(w_a),
            "w_b": float(w_b),
            "threshold_from_val": float(thr),
            "tpr": float(test_rates["tpr"]),
            "fpr": float(test_rates["fpr"]),
            "tp": int(test_rates["tp"]),
            "fp": int(test_rates["fp"]),
        },
    }


def _delta_phi_np(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    d = phi_a - phi_b
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def _pairwise_token_cost(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # a, b have shape [N,4] and [M,4] with [pt, eta, phi, E].
    eta_a = a[:, 1:2]
    eta_b = b[:, 1][None, :]
    phi_a = a[:, 2:3]
    phi_b = b[:, 2][None, :]
    pt_a = np.log(np.clip(a[:, 0:1], eps, None))
    pt_b = np.log(np.clip(b[:, 0], eps, None))[None, :]

    deta = eta_a - eta_b
    dphi = _delta_phi_np(phi_a, phi_b)
    dlogpt = pt_a - pt_b

    dr2 = deta * deta + dphi * dphi
    cost = dr2 + 0.25 * (dlogpt * dlogpt)
    return cost, np.sqrt(np.maximum(dr2, 0.0)), np.abs(dlogpt)


def compute_reco_set_matching_diagnostics(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_reco: np.ndarray,
    mask_reco: np.ndarray,
    max_jets: int,
    seed: int,
) -> Dict[str, float]:
    n = int(const_off.shape[0])
    rng = np.random.default_rng(int(seed))
    if max_jets > 0 and max_jets < n:
        idx = rng.choice(n, size=max_jets, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n, dtype=np.int64)

    ch_hlt = []
    ch_reco = []
    dr_hlt_to_off = []
    dr_reco_to_off = []
    dr_reco_to_hlt = []
    dlogpt_hlt_to_off = []
    dlogpt_reco_to_off = []

    n_off_vec = mask_off[idx].sum(axis=1).astype(np.float64)
    n_hlt_vec = mask_hlt[idx].sum(axis=1).astype(np.float64)
    n_reco_vec = mask_reco[idx].sum(axis=1).astype(np.float64)

    for i in idx:
        off = const_off[i][mask_off[i]]
        hlt = const_hlt[i][mask_hlt[i]]
        reco = const_reco[i][mask_reco[i]]

        if off.shape[0] > 0 and hlt.shape[0] > 0:
            c_ho, dr_ho, dlpt_ho = _pairwise_token_cost(hlt, off)
            min_ho = c_ho.min(axis=1)
            min_oh = c_ho.min(axis=0)
            ch_hlt.append(float(0.5 * (min_ho.mean() + min_oh.mean())))
            dr_hlt_to_off.append(float(dr_ho.min(axis=1).mean()))
            dlogpt_hlt_to_off.append(float(dlpt_ho.min(axis=1).mean()))

        if off.shape[0] > 0 and reco.shape[0] > 0:
            c_ro, dr_ro, dlpt_ro = _pairwise_token_cost(reco, off)
            min_ro = c_ro.min(axis=1)
            min_or = c_ro.min(axis=0)
            ch_reco.append(float(0.5 * (min_ro.mean() + min_or.mean())))
            dr_reco_to_off.append(float(dr_ro.min(axis=1).mean()))
            dlogpt_reco_to_off.append(float(dlpt_ro.min(axis=1).mean()))

        if reco.shape[0] > 0 and hlt.shape[0] > 0:
            _, dr_rh, _ = _pairwise_token_cost(reco, hlt)
            dr_reco_to_hlt.append(float(dr_rh.min(axis=1).mean()))

    ch_hlt_arr = np.asarray(ch_hlt, dtype=np.float64)
    ch_reco_arr = np.asarray(ch_reco, dtype=np.float64)
    dr_hlt_arr = np.asarray(dr_hlt_to_off, dtype=np.float64)
    dr_reco_arr = np.asarray(dr_reco_to_off, dtype=np.float64)
    dlpt_hlt_arr = np.asarray(dlogpt_hlt_to_off, dtype=np.float64)
    dlpt_reco_arr = np.asarray(dlogpt_reco_to_off, dtype=np.float64)
    dr_reco_hlt_arr = np.asarray(dr_reco_to_hlt, dtype=np.float64)

    true_added = np.maximum(n_off_vec - n_hlt_vec, 0.0)
    pred_added = np.maximum(n_reco_vec - n_hlt_vec, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recovery = np.where(true_added > 0.0, pred_added / np.maximum(true_added, 1e-8), np.nan)

    diag = {
        "n_jets_total": int(n),
        "n_jets_eval": int(idx.size),
        "mean_count_offline": float(n_off_vec.mean()) if n_off_vec.size else 0.0,
        "mean_count_hlt": float(n_hlt_vec.mean()) if n_hlt_vec.size else 0.0,
        "mean_count_reco": float(n_reco_vec.mean()) if n_reco_vec.size else 0.0,
        "mae_count_hlt_vs_offline": float(np.abs(n_hlt_vec - n_off_vec).mean()) if n_off_vec.size else 0.0,
        "mae_count_reco_vs_offline": float(np.abs(n_reco_vec - n_off_vec).mean()) if n_off_vec.size else 0.0,
        "frac_exact_count_match_hlt": float((n_hlt_vec == n_off_vec).mean()) if n_off_vec.size else 0.0,
        "frac_exact_count_match_reco": float((n_reco_vec == n_off_vec).mean()) if n_off_vec.size else 0.0,
        "mean_chamfer_hlt_to_offline": float(ch_hlt_arr.mean()) if ch_hlt_arr.size else float("nan"),
        "mean_chamfer_reco_to_offline": float(ch_reco_arr.mean()) if ch_reco_arr.size else float("nan"),
        "median_chamfer_hlt_to_offline": float(np.median(ch_hlt_arr)) if ch_hlt_arr.size else float("nan"),
        "median_chamfer_reco_to_offline": float(np.median(ch_reco_arr)) if ch_reco_arr.size else float("nan"),
        "frac_reco_better_chamfer": float((ch_reco_arr < ch_hlt_arr).mean()) if (ch_hlt_arr.size and ch_reco_arr.size and ch_hlt_arr.size == ch_reco_arr.size) else float("nan"),
        "mean_min_dr_hlt_to_offline": float(dr_hlt_arr.mean()) if dr_hlt_arr.size else float("nan"),
        "mean_min_dr_reco_to_offline": float(dr_reco_arr.mean()) if dr_reco_arr.size else float("nan"),
        "mean_min_dlogpt_hlt_to_offline": float(dlpt_hlt_arr.mean()) if dlpt_hlt_arr.size else float("nan"),
        "mean_min_dlogpt_reco_to_offline": float(dlpt_reco_arr.mean()) if dlpt_reco_arr.size else float("nan"),
        "mean_min_dr_reco_to_hlt": float(dr_reco_hlt_arr.mean()) if dr_reco_hlt_arr.size else float("nan"),
        "n_added_jets": int(np.isfinite(recovery).sum()),
        "mean_added_recovery_ratio": float(np.nanmean(recovery)) if np.isfinite(recovery).any() else float("nan"),
        "median_added_recovery_ratio": float(np.nanmedian(recovery)) if np.isfinite(recovery).any() else float("nan"),
        "frac_added_overrecover_gt1p25": float(np.nanmean(recovery > 1.25)) if np.isfinite(recovery).any() else float("nan"),
        "frac_added_underrecover_lt0p50": float(np.nanmean(recovery < 0.50)) if np.isfinite(recovery).any() else float("nan"),
    }
    return diag


class JetRegressionDataset(Dataset):
    def __init__(self, feat: np.ndarray, mask: np.ndarray, target_log: np.ndarray):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.target_log = torch.tensor(target_log, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat": self.feat[i],
            "mask": self.mask[i],
            "target_log": self.target_log[i],
        }


class JetLevelRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(embed_dim))
        self.head = nn.Sequential(
            nn.Linear(int(embed_dim), int(embed_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(embed_dim), int(output_dim)),
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(feat)
        x = self.encoder(x, src_key_padding_mask=~mask)
        w = mask.float().unsqueeze(-1)
        pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        pooled = self.norm(pooled)
        return self.head(pooled)


def _jet_mass_np(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    eps = 1e-8
    pt = const[..., 0]
    eta = const[..., 1]
    phi = const[..., 2]
    energy = const[..., 3]
    w = mask.astype(np.float32)
    px = (pt * np.cos(phi) * w).sum(axis=1)
    py = (pt * np.sin(phi) * w).sum(axis=1)
    pz = (pt * np.sinh(eta) * w).sum(axis=1)
    e = (energy * w).sum(axis=1)
    p2 = px * px + py * py + pz * pz
    m2 = np.maximum(e * e - p2, eps)
    return np.sqrt(m2).astype(np.float32)


def _tau_n(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, n_axes: int, r0: float) -> float:
    eps = 1e-8
    n = int(pt.shape[0])
    if n == 0:
        return 0.0
    n_axes = max(1, min(int(n_axes), n))
    # Use hardest constituents as simple deterministic axes.
    axis_idx = np.argsort(-pt)[:n_axes]
    a_eta = eta[axis_idx]
    a_phi = phi[axis_idx]
    deta = eta[:, None] - a_eta[None, :]
    dphi = np.arctan2(np.sin(phi[:, None] - a_phi[None, :]), np.cos(phi[:, None] - a_phi[None, :]))
    dr = np.sqrt(deta * deta + dphi * dphi)
    min_dr = dr.min(axis=1)
    d0 = np.sum(pt) * float(r0) + eps
    tau = float(np.sum(pt * min_dr) / d0)
    return tau


def _d2_topk(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, topk: int, beta: float) -> float:
    eps = 1e-8
    if pt.shape[0] < 3:
        return 0.0
    order = np.argsort(-pt)[: min(int(topk), int(pt.shape[0]))]
    pt_k = pt[order]
    eta_k = eta[order]
    phi_k = phi[order]
    m = int(pt_k.shape[0])
    if m < 3:
        return 0.0
    z = pt_k / (np.sum(pt_k) + eps)
    deta = eta_k[:, None] - eta_k[None, :]
    dphi = np.arctan2(np.sin(phi_k[:, None] - phi_k[None, :]), np.cos(phi_k[:, None] - phi_k[None, :]))
    dr = np.sqrt(deta * deta + dphi * dphi) + eps

    e2 = 0.0
    for i in range(m):
        zi = z[i]
        for j in range(i + 1, m):
            e2 += zi * z[j] * (dr[i, j] ** beta)

    e3 = 0.0
    for i in range(m):
        zi = z[i]
        for j in range(i + 1, m):
            zij = zi * z[j]
            dij = dr[i, j] ** beta
            for k in range(j + 1, m):
                e3 += zij * z[k] * dij * (dr[i, k] ** beta) * (dr[j, k] ** beta)
    d2 = float(e3 / ((e2 ** 3) + eps))
    return d2


def _compute_substructure_np(
    const: np.ndarray,
    mask: np.ndarray,
    tau_r0: float = 0.8,
    tau_topk: int = 24,
    d2_topk: int = 10,
    d2_beta: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_jets = int(const.shape[0])
    tau21 = np.zeros(n_jets, dtype=np.float32)
    tau32 = np.zeros(n_jets, dtype=np.float32)
    d2 = np.zeros(n_jets, dtype=np.float32)
    eps = 1e-8
    for j in range(n_jets):
        m = mask[j]
        if not np.any(m):
            continue
        c = const[j, m]
        order = np.argsort(-c[:, 0])[: min(int(tau_topk), c.shape[0])]
        c = c[order]
        pt = c[:, 0].astype(np.float64)
        eta = c[:, 1].astype(np.float64)
        phi = c[:, 2].astype(np.float64)
        t1 = _tau_n(pt, eta, phi, 1, tau_r0)
        t2 = _tau_n(pt, eta, phi, 2, tau_r0)
        t3 = _tau_n(pt, eta, phi, 3, tau_r0)
        tau21[j] = np.float32(t2 / (t1 + eps))
        tau32[j] = np.float32(t3 / (t2 + eps))
        d2[j] = np.float32(_d2_topk(pt, eta, phi, topk=int(d2_topk), beta=float(d2_beta)))
    tau21 = np.nan_to_num(tau21, nan=0.0, posinf=0.0, neginf=0.0)
    tau32 = np.nan_to_num(tau32, nan=0.0, posinf=0.0, neginf=0.0)
    d2 = np.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0)
    tau21 = np.clip(tau21, 0.0, 5.0)
    tau32 = np.clip(tau32, 0.0, 5.0)
    d2 = np.clip(d2, 0.0, 1e3)
    return tau21, tau32, d2


def compute_jet_regression_targets(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Returns transformed target vectors:
      [log_pt, log_E, log_m, tau21, tau32, log1p_d2, log1p_n_off, log1p_n_added]
    for offline target and HLT reference (last channel set to 0 for HLT side).
    """
    eps = 1e-8
    idx = {
        "log_pt": 0,
        "log_e": 1,
        "log_m": 2,
        "tau21": 3,
        "tau32": 4,
        "log1p_d2": 5,
        "log1p_n_off": 6,
        "log1p_n_added": 7,
    }
    n_jets = int(const_off.shape[0])
    off = np.zeros((n_jets, 8), dtype=np.float32)
    hlt = np.zeros((n_jets, 8), dtype=np.float32)

    pt_off = compute_jet_pt(const_off, mask_off).astype(np.float32)
    pt_hlt = compute_jet_pt(const_hlt, mask_hlt).astype(np.float32)
    e_off = (const_off[:, :, 3] * mask_off.astype(np.float32)).sum(axis=1).astype(np.float32)
    e_hlt = (const_hlt[:, :, 3] * mask_hlt.astype(np.float32)).sum(axis=1).astype(np.float32)
    m_off = _jet_mass_np(const_off, mask_off)
    m_hlt = _jet_mass_np(const_hlt, mask_hlt)
    tau21_off, tau32_off, d2_off = _compute_substructure_np(const_off, mask_off)
    tau21_hlt, tau32_hlt, d2_hlt = _compute_substructure_np(const_hlt, mask_hlt)
    n_off = mask_off.sum(axis=1).astype(np.float32)
    n_hlt = mask_hlt.sum(axis=1).astype(np.float32)
    n_added = np.maximum(n_off - n_hlt, 0.0).astype(np.float32)

    off[:, idx["log_pt"]] = np.log(np.clip(pt_off, eps, None))
    off[:, idx["log_e"]] = np.log(np.clip(e_off, eps, None))
    off[:, idx["log_m"]] = np.log(np.clip(m_off, eps, None))
    off[:, idx["tau21"]] = tau21_off
    off[:, idx["tau32"]] = tau32_off
    off[:, idx["log1p_d2"]] = np.log1p(np.clip(d2_off, 0.0, None))
    off[:, idx["log1p_n_off"]] = np.log1p(np.clip(n_off, 0.0, None))
    off[:, idx["log1p_n_added"]] = np.log1p(np.clip(n_added, 0.0, None))

    hlt[:, idx["log_pt"]] = np.log(np.clip(pt_hlt, eps, None))
    hlt[:, idx["log_e"]] = np.log(np.clip(e_hlt, eps, None))
    hlt[:, idx["log_m"]] = np.log(np.clip(m_hlt, eps, None))
    hlt[:, idx["tau21"]] = tau21_hlt
    hlt[:, idx["tau32"]] = tau32_hlt
    hlt[:, idx["log1p_d2"]] = np.log1p(np.clip(d2_hlt, 0.0, None))
    hlt[:, idx["log1p_n_off"]] = np.log1p(np.clip(n_hlt, 0.0, None))
    hlt[:, idx["log1p_n_added"]] = 0.0

    off = np.nan_to_num(off, nan=0.0, posinf=0.0, neginf=0.0)
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    return off.astype(np.float32), hlt.astype(np.float32), idx


def _jet_reg_metric_dict(pred: np.ndarray, true: np.ndarray, idx: Dict[str, int]) -> Dict[str, float]:
    m: Dict[str, float] = {}
    m["mae_log_pt"] = float(np.mean(np.abs(pred[:, idx["log_pt"]] - true[:, idx["log_pt"]])))
    m["mae_log_e"] = float(np.mean(np.abs(pred[:, idx["log_e"]] - true[:, idx["log_e"]])))
    m["mae_log_m"] = float(np.mean(np.abs(pred[:, idx["log_m"]] - true[:, idx["log_m"]])))
    m["mae_tau21"] = float(np.mean(np.abs(pred[:, idx["tau21"]] - true[:, idx["tau21"]])))
    m["mae_tau32"] = float(np.mean(np.abs(pred[:, idx["tau32"]] - true[:, idx["tau32"]])))
    m["mae_log1p_d2"] = float(np.mean(np.abs(pred[:, idx["log1p_d2"]] - true[:, idx["log1p_d2"]])))
    m["mae_log1p_n_off"] = float(np.mean(np.abs(pred[:, idx["log1p_n_off"]] - true[:, idx["log1p_n_off"]])))
    m["mae_log1p_n_added"] = float(np.mean(np.abs(pred[:, idx["log1p_n_added"]] - true[:, idx["log1p_n_added"]])))

    m["mae_pt"] = float(
        np.mean(np.abs(np.exp(pred[:, idx["log_pt"]]) - np.exp(true[:, idx["log_pt"]])))
    )
    m["mae_e"] = float(
        np.mean(np.abs(np.exp(pred[:, idx["log_e"]]) - np.exp(true[:, idx["log_e"]])))
    )
    m["mae_m"] = float(
        np.mean(np.abs(np.exp(pred[:, idx["log_m"]]) - np.exp(true[:, idx["log_m"]])))
    )
    m["mae_d2"] = float(
        np.mean(np.abs(np.expm1(pred[:, idx["log1p_d2"]]) - np.expm1(true[:, idx["log1p_d2"]])))
    )
    m["mae_n_off"] = float(
        np.mean(np.abs(np.expm1(pred[:, idx["log1p_n_off"]]) - np.expm1(true[:, idx["log1p_n_off"]])))
    )
    m["mae_n_added"] = float(
        np.mean(np.abs(np.expm1(pred[:, idx["log1p_n_added"]]) - np.expm1(true[:, idx["log1p_n_added"]])))
    )
    return m


def train_jet_regressor(
    model: JetLevelRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
) -> Tuple[JetLevelRegressor, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))
    best_state = None
    best_val = float("inf")
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for ep in tqdm(range(int(epochs)), desc="JetRegressor"):
        model.train()
        tr = 0.0
        n_tr = 0
        for batch in train_loader:
            feat = batch["feat"].to(device)
            mask = batch["mask"].to(device)
            tgt = batch["target_log"].to(device)
            opt.zero_grad()
            pred = model(feat, mask)
            loss = F.smooth_l1_loss(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = feat.size(0)
            tr += loss.item() * bs
            n_tr += bs
        sch.step()
        tr /= max(n_tr, 1)

        model.eval()
        va = 0.0
        n_va = 0
        all_pred = []
        all_tgt = []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["feat"].to(device)
                mask = batch["mask"].to(device)
                tgt = batch["target_log"].to(device)
                pred = model(feat, mask)
                loss = F.smooth_l1_loss(pred, tgt)
                bs = feat.size(0)
                va += loss.item() * bs
                n_va += bs
                all_pred.append(pred.detach().cpu().numpy())
                all_tgt.append(tgt.detach().cpu().numpy())
        va /= max(n_va, 1)
        pred_val = np.concatenate(all_pred, axis=0) if len(all_pred) > 0 else np.zeros((0, 8), dtype=np.float32)
        tgt_val = np.concatenate(all_tgt, axis=0) if len(all_tgt) > 0 else np.zeros((0, 8), dtype=np.float32)
        # default indices for compact reporting (log_pt/log_e are 0/1 in target layout)
        mae_pt_val = float(np.mean(np.abs(np.exp(pred_val[:, 0]) - np.exp(tgt_val[:, 0])))) if pred_val.size else float("nan")
        mae_e_val = float(np.mean(np.abs(np.exp(pred_val[:, 1]) - np.exp(tgt_val[:, 1])))) if pred_val.size else float("nan")

        if va < best_val:
            best_val = float(va)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "best_val_loss": float(va),
                "best_val_mae_pt": float(mae_pt_val),
                "best_val_mae_e": float(mae_e_val),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"JetReg ep {ep+1}: train_loss={tr:.5f}, val_loss={va:.5f}, "
                f"val_mae_pt={mae_pt_val:.3f}, val_mae_e={mae_e_val:.3f}, best={best_val:.5f}"
            )
        if no_improve >= int(patience):
            print(f"Early stopping JetRegressor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


@torch.no_grad()
def predict_jet_regressor(
    model: JetLevelRegressor,
    feat: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out_dim = int(model.head[-1].out_features)
    out = np.zeros((feat.shape[0], out_dim), dtype=np.float32)
    for start in range(0, feat.shape[0], int(batch_size)):
        end = min(start + int(batch_size), feat.shape[0])
        x = torch.tensor(feat[start:end], dtype=torch.float32, device=device)
        m = torch.tensor(mask[start:end], dtype=torch.bool, device=device)
        pred = model(x, m)
        out[start:end] = pred.detach().cpu().numpy().astype(np.float32)
    return out


def train_joint_dual(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    freeze_reconstructor: bool,
    epochs: int,
    patience: int,
    lr_dual: float,
    lr_reco: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_reco: float,
    lambda_rank: float,
    lambda_cons: float,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    min_epochs: int,
    select_metric: str = "auc",
    teacher_model: nn.Module | None = None,
    feat_means: np.ndarray | None = None,
    feat_stds: np.ndarray | None = None,
    reco_kd_temperature: float = 2.5,
    reco_lambda_kd: float = 1.0,
    reco_lambda_emb: float = 1.2,
    reco_lambda_tok: float = 0.6,
    reco_lambda_phys: float = 0.2,
    reco_lambda_budget_hinge: float = 0.03,
    reco_budget_eps: float = 0.015,
    reco_budget_weight_floor: float = 1e-4,
    reco_normalize_loss_terms: bool = True,
    reco_loss_norm_ema_decay: float = 0.98,
    reco_loss_norm_eps: float = 1e-6,
    reco_stagea_fused_adv_weight: float = 0.0,
    reco_stagea_fused_adv_power: float = 1.0,
    reco_stagea_fused_uncert_weight: float = 0.0,
    reco_stagea_fused_adv_use_abs: bool = False,
    reco_stagea_fused_kd_w_min: float = 0.25,
    reco_stagea_fused_kd_w_max: float = 4.0,
    reco_stagea_lambda_delta_aux: float = 0.0,
    joint_fused_kd_lambda: float = 0.0,
    joint_fused_kd_temp: float = 1.0,
    joint_fused_adv_weight: float = 0.0,
    joint_fused_adv_power: float = 1.0,
    joint_fused_uncert_weight: float = 0.0,
    joint_fused_adv_use_abs: bool = False,
    joint_fused_kd_w_min: float = 0.25,
    joint_fused_kd_w_max: float = 4.0,
) -> Tuple[OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    for p in reconstructor.parameters():
        p.requires_grad = not freeze_reconstructor

    params = [{"params": dual_model.parameters(), "lr": float(lr_dual)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_dual), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))

    means_t = None
    stds_t = None
    reco_loss_ema_state = None
    if teacher_model is not None and feat_means is not None and feat_stds is not None:
        teacher_model.eval()
        for p_t in teacher_model.parameters():
            p_t.requires_grad_(False)
        means_t = torch.tensor(feat_means, dtype=torch.float32, device=device)
        stds_t = torch.tensor(np.clip(feat_stds, 1e-6, None), dtype=torch.float32, device=device)
        reco_loss_ema_state = {
            "kd": 1.0,
            "emb": 1.0,
            "tok": 1.0,
            "phys": 1.0,
            "budget": 1.0,
        }
    joint_fused_kd_lambda = float(max(joint_fused_kd_lambda, 0.0))
    joint_fused_kd_temp = float(max(joint_fused_kd_temp, 1e-3))
    joint_fused_adv_weight = float(max(joint_fused_adv_weight, 0.0))
    joint_fused_adv_power = float(max(joint_fused_adv_power, 1e-6))
    joint_fused_uncert_weight = float(max(joint_fused_uncert_weight, 0.0))
    joint_fused_kd_w_min = float(max(joint_fused_kd_w_min, 1e-6))
    joint_fused_kd_w_max = float(max(joint_fused_kd_w_max, joint_fused_kd_w_min))

    best_state_dual_sel = None
    best_state_reco_sel = None
    best_state_dual_auc = None
    best_state_reco_auc = None
    best_state_dual_fpr = None
    best_state_reco_fpr = None

    best_val_fpr50 = float("inf")  # best observed across epochs
    best_val_auc = float("-inf")   # best observed across epochs
    best_sel_score = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    sel_val_fpr50 = float("nan")
    sel_val_auc = float("nan")
    no_improve = 0

    for ep in tqdm(range(int(epochs)), desc=stage_name):
        dual_model.train()
        if freeze_reconstructor:
            reconstructor.eval()
        else:
            reconstructor.train()

        tr_loss = 0.0
        tr_cls = 0.0
        tr_rank = 0.0
        tr_reco = 0.0
        tr_cons = 0.0
        tr_fused_kd = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt_reco = batch["feat_hlt_reco"].to(device)
            feat_hlt_dual = batch["feat_hlt_dual"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            b_merge = batch["budget_merge_true"].to(device)
            b_eff = batch["budget_eff_true"].to(device)
            y = batch["label"].to(device)
            joint_fused_target = batch.get("joint_fused_target", None)
            if joint_fused_target is not None:
                joint_fused_target = joint_fused_target.to(device)

            opt.zero_grad()

            if freeze_reconstructor:
                with torch.no_grad():
                    reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            else:
                reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)

            feat_b, mask_b = build_soft_corrected_view(
                reco_out,
                weight_floor=corrected_weight_floor,
                scale_features_by_weight=True,
                include_flags=corrected_use_flags,
            )
            logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)

            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()
            if joint_fused_target is not None and joint_fused_kd_lambda > 0.0:
                target_soft_joint = torch.clamp(joint_fused_target.view(-1), min=0.0, max=1.0)
                anchor_soft_joint = torch.sigmoid(logits.detach() / joint_fused_kd_temp)
                w_joint = _build_advantage_weights(
                    target_soft=target_soft_joint,
                    anchor_soft=anchor_soft_joint,
                    adv_weight=joint_fused_adv_weight,
                    adv_power=joint_fused_adv_power,
                    uncert_weight=joint_fused_uncert_weight,
                    use_abs_delta=bool(joint_fused_adv_use_abs),
                    w_min=joint_fused_kd_w_min,
                    w_max=joint_fused_kd_w_max,
                )
                kd_joint_vec = (
                    F.binary_cross_entropy_with_logits(
                        logits / joint_fused_kd_temp,
                        target_soft_joint,
                        reduction="none",
                    )
                    * (joint_fused_kd_temp * joint_fused_kd_temp)
                )
                loss_joint_fused_kd = _weighted_batch_mean(kd_joint_vec, w_joint)
            else:
                loss_joint_fused_kd = torch.zeros((), device=device)

            if float(lambda_reco) > 0.0:
                if teacher_model is not None and means_t is not None and stds_t is not None:
                    reco_losses = _compute_teacher_guided_reco_losses(
                        reco_out=reco_out,
                        const_hlt=const_hlt,
                        mask_hlt=mask_hlt,
                        const_off=const_off,
                        mask_off=mask_off,
                        budget_merge_true=b_merge,
                        budget_eff_true=b_eff,
                        teacher_model=teacher_model,
                        means_t=means_t,
                        stds_t=stds_t,
                        loss_cfg=BASE_CONFIG["loss"],
                        kd_temperature=float(max(reco_kd_temperature, 1e-3)),
                        budget_eps=float(max(reco_budget_eps, 0.0)),
                        budget_weight_floor=float(max(reco_budget_weight_floor, 0.0)),
                        stagea_fused_target=joint_fused_target,
                        stagea_anchor_target=None,
                        stagea_fused_adv_weight=float(max(reco_stagea_fused_adv_weight, 0.0)),
                        stagea_fused_adv_power=float(max(reco_stagea_fused_adv_power, 1e-6)),
                        stagea_fused_uncert_weight=float(max(reco_stagea_fused_uncert_weight, 0.0)),
                        stagea_fused_adv_use_abs=bool(reco_stagea_fused_adv_use_abs),
                        stagea_fused_kd_w_min=float(max(reco_stagea_fused_kd_w_min, 1e-6)),
                        stagea_fused_kd_w_max=float(max(reco_stagea_fused_kd_w_max, reco_stagea_fused_kd_w_min)),
                        stagea_lambda_delta_aux=float(max(reco_stagea_lambda_delta_aux, 0.0)),
                    )
                    loss_reco, _, reco_loss_ema_state = _compose_teacher_guided_reco_total(
                        losses_raw=reco_losses,
                        ema_state=reco_loss_ema_state,
                        normalize_terms=bool(reco_normalize_loss_terms),
                        ema_decay=float(np.clip(reco_loss_norm_ema_decay, 0.0, 0.9999)),
                        norm_eps=float(max(reco_loss_norm_eps, 1e-12)),
                        w_logit=float(max(reco_lambda_kd, 0.0)),
                        w_emb=float(max(reco_lambda_emb, 0.0)),
                        w_tok=float(max(reco_lambda_tok, 0.0)),
                        w_phys=float(max(reco_lambda_phys, 0.0)),
                        w_budget=float(max(reco_lambda_budget_hinge, 0.0)),
                        update_ema=True,
                    )
                else:
                    reco_losses = compute_reconstruction_losses(
                        reco_out,
                        const_hlt,
                        mask_hlt,
                        const_off,
                        mask_off,
                        b_merge,
                        b_eff,
                        BASE_CONFIG["loss"],
                    )
                    loss_reco = reco_losses["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            loss = (
                loss_cls
                + float(lambda_rank) * loss_rank
                + float(lambda_reco) * loss_reco
                + float(lambda_cons) * loss_cons
                + float(joint_fused_kd_lambda) * loss_joint_fused_kd
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual_model.parameters(), 1.0)
            if not freeze_reconstructor:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = feat_hlt_reco.size(0)
            tr_loss += loss.item() * bs
            tr_cls += loss_cls.item() * bs
            tr_rank += loss_rank.item() * bs
            tr_reco += loss_reco.item() * bs
            tr_cons += loss_cons.item() * bs
            tr_fused_kd += loss_joint_fused_kd.item() * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)
        tr_fused_kd /= max(n_tr, 1)

        va_auc, _, _, va_fpr50 = eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_model,
            loader=val_loader,
            device=device,
            corrected_weight_floor=corrected_weight_floor,
            corrected_use_flags=corrected_use_flags,
        )

        # Track best by each metric.
        if np.isfinite(va_fpr50) and float(va_fpr50) < best_val_fpr50:
            best_val_fpr50 = float(va_fpr50)
            best_state_dual_fpr = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco_fpr = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state_dual_auc = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco_auc = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}

        if str(select_metric).lower() == "auc":
            improved = np.isfinite(va_auc) and (float(va_auc) > best_sel_score)
            current_score = float(va_auc) if np.isfinite(va_auc) else float("-inf")
        else:
            improved = np.isfinite(va_fpr50) and (float(va_fpr50) < best_sel_score)
            current_score = float(va_fpr50) if np.isfinite(va_fpr50) else float("inf")

        if improved:
            best_sel_score = current_score
            sel_val_fpr50 = float(va_fpr50)
            sel_val_auc = float(va_auc)
            best_state_dual_sel = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco_sel = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # Print every epoch for Stage C variants; keep every 5 for earlier stages.
        print_every = 1 if str(stage_name).startswith("StageC") else 5
        if (ep + 1) % print_every == 0:
            print(
                f"{stage_name} ep {ep+1}: train_loss={tr_loss:.4f} "
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}, fused_kd={tr_fused_kd:.4f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, "
                f"select={str(select_metric).lower()}, best_sel={best_sel_score:.6f}"
            )

        if (ep + 1) >= int(min_epochs) and no_improve >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_dual_sel is not None:
        dual_model.load_state_dict(best_state_dual_sel)
    if best_state_reco_sel is not None:
        reconstructor.load_state_dict(best_state_reco_sel)

    metrics = {
        "selection_metric": str(select_metric).lower(),
        "selected_val_fpr50": float(sel_val_fpr50),
        "selected_val_auc": float(sel_val_auc),
        "best_val_fpr50_seen": float(best_val_fpr50),
        "best_val_auc_seen": float(best_val_auc),
    }
    state_pack = {
        "selected": {"dual": best_state_dual_sel, "reco": best_state_reco_sel},
        "auc": {"dual": best_state_dual_auc, "reco": best_state_reco_auc},
        "fpr50": {"dual": best_state_dual_fpr, "reco": best_state_reco_fpr},
    }
    return reconstructor, dual_model, metrics, state_pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=50000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument(
        "--n_train_split",
        type=int,
        default=-1,
        help="If >0 and paired with --n_val_split/--n_test_split, use exact split counts instead of 70/15/15.",
    )
    parser.add_argument("--n_val_split", type=int, default=-1)
    parser.add_argument("--n_test_split", type=int, default=-1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_joint"),
    )
    parser.add_argument("--run_name", type=str, default="joint_default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--skip_save_models", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--use_train_weights", action="store_true")
    parser.add_argument(
        "--step1_load_dir",
        type=str,
        default="",
        help="If set to a directory containing teacher.pt and baseline.pt, load them and skip STEP 1 training.",
    )

    # HLT controls
    parser.add_argument("--merge_radius", type=float, default=BASE_CONFIG["hlt_effects"]["merge_radius"])
    parser.add_argument("--eff_plateau_barrel", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    parser.add_argument("--eff_plateau_endcap", type=float, default=BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    parser.add_argument("--smear_a", type=float, default=BASE_CONFIG["hlt_effects"]["smear_a"])
    parser.add_argument("--smear_b", type=float, default=BASE_CONFIG["hlt_effects"]["smear_b"])
    parser.add_argument("--smear_c", type=float, default=BASE_CONFIG["hlt_effects"]["smear_c"])

    # Stage A (reconstructor pretrain)
    parser.add_argument("--stageA_epochs", type=int, default=90)
    parser.add_argument("--stageA_patience", type=int, default=18)
    parser.add_argument("--stageA_kd_temp", type=float, default=2.5)
    parser.add_argument("--stageA_lambda_kd", type=float, default=1.0)
    parser.add_argument("--stageA_lambda_emb", type=float, default=1.2)
    parser.add_argument("--stageA_lambda_tok", type=float, default=0.6)
    parser.add_argument("--stageA_lambda_phys", type=float, default=0.2)
    parser.add_argument("--stageA_lambda_budget_hinge", type=float, default=0.03)
    parser.add_argument("--stageA_budget_eps", type=float, default=0.015)
    parser.add_argument("--stageA_budget_weight_floor", type=float, default=1e-4)
    parser.add_argument("--stageA_target_tpr", type=float, default=0.50)
    parser.add_argument(
        "--stageA_teacher_ckpt",
        type=str,
        default="",
        help="Optional external teacher checkpoint for Stage-A guidance only (overrides STEP-1 teacher in Stage-A).",
    )
    parser.add_argument(
        "--stageA_fused_targets_npz",
        type=str,
        default="",
        help="Optional fused-target NPZ for direct Stage-A logit distillation.",
    )
    parser.add_argument(
        "--stageA_fused_targets_key",
        type=str,
        default="probs_fused_overall",
        help="Fused-target key prefix in NPZ (e.g. probs_fused_overall, probs_fused_bin).",
    )
    parser.add_argument(
        "--stageA_fused_split_scheme",
        type=str,
        default="train_val_test",
        choices=["train_val_test", "fit_ref_test"],
        help="How to read Stage-A fused targets from NPZ suffixes.",
    )
    parser.add_argument(
        "--stageA_fused_source_splits_npz",
        type=str,
        default="",
        help="Optional source data_splits.npz for exact Stage-A fused target index alignment.",
    )
    parser.add_argument("--stageA_fused_source_val_key", type=str, default="val_idx")
    parser.add_argument("--stageA_fused_source_test_key", type=str, default="test_idx")
    parser.add_argument(
        "--stageA_fused_train_from",
        type=str,
        default="train",
        choices=["train", "fit", "ref"],
        help="Which fused partition to use for Stage-A train targets.",
    )
    parser.add_argument(
        "--stageA_fused_val_from",
        type=str,
        default="val",
        choices=["val", "ref", "test"],
        help="Which fused partition to use for Stage-A val targets.",
    )
    parser.add_argument("--stageA_fused_adv_weight", type=float, default=0.0)
    parser.add_argument("--stageA_fused_adv_power", type=float, default=1.0)
    parser.add_argument("--stageA_fused_uncert_weight", type=float, default=0.0)
    parser.add_argument("--stageA_fused_adv_use_abs", action="store_true")
    parser.add_argument("--stageA_fused_kd_w_min", type=float, default=0.25)
    parser.add_argument("--stageA_fused_kd_w_max", type=float, default=4.0)
    parser.add_argument("--stageA_lambda_delta_aux", type=float, default=0.0)
    parser.add_argument(
        "--stageA_anchor_targets_npz",
        type=str,
        default="",
        help="Optional NPZ providing anchor targets for Stage-A fused-advantage weighting.",
    )
    parser.add_argument(
        "--stageA_anchor_targets_key",
        type=str,
        default="probs_anchor_overall",
        help="Anchor-target key prefix in NPZ.",
    )
    parser.add_argument(
        "--stageA_anchor_split_scheme",
        type=str,
        default="train_val_test",
        choices=["train_val_test", "fit_ref_test"],
        help="How to read Stage-A anchor targets from NPZ suffixes.",
    )
    parser.add_argument("--disable_stageA_loss_normalization", action="store_true")
    parser.add_argument("--stageA_loss_norm_ema_decay", type=float, default=0.98)
    parser.add_argument("--stageA_loss_norm_eps", type=float, default=1e-6)
    parser.add_argument(
        "--disable_stageA_stagewise_best_reload",
        action="store_true",
        help="Disable reloading the best Stage-A validation checkpoint at each stage-scale transition.",
    )
    parser.add_argument(
        "--train_reco_only_after_stageA",
        action="store_true",
        help="After Stage-A, train/evaluate a reco-only top-tagger before Stage-B dual-view training.",
    )
    parser.add_argument("--reco_only_epochs", type=int, default=60)
    parser.add_argument("--reco_only_patience", type=int, default=15)
    parser.add_argument("--reco_only_lr", type=float, default=4e-4)
    parser.add_argument("--reco_only_weight_decay", type=float, default=1e-5)
    parser.add_argument("--reco_only_warmup_epochs", type=int, default=3)
    parser.add_argument("--reco_only_batch_size", type=int, default=512)

    # Stage B (tagger pretrain, reconstructor frozen)
    parser.add_argument("--stageB_epochs", type=int, default=45)
    parser.add_argument("--stageB_patience", type=int, default=12)
    parser.add_argument("--stageB_min_epochs", type=int, default=12)
    parser.add_argument("--stageB_lr_dual", type=float, default=4e-4)
    parser.add_argument("--stageB_lambda_rank", type=float, default=0.0)
    parser.add_argument("--stageB_lambda_cons", type=float, default=0.0)
    parser.add_argument("--stageB_joint_fused_kd_lambda", type=float, default=0.0)

    # Kept for CLI compatibility; this script always selects by val_auc.
    parser.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])

    # Stage C (joint finetune)
    parser.add_argument("--stageC_epochs", type=int, default=65)
    parser.add_argument("--stageC_patience", type=int, default=14)
    parser.add_argument("--stageC_min_epochs", type=int, default=25)
    parser.add_argument("--stageC_lr_dual", type=float, default=2e-4)
    parser.add_argument("--stageC_lr_reco", type=float, default=1e-4)
    parser.add_argument("--lambda_reco", type=float, default=0.35)
    # Stage C rank term is disabled in this variant.
    parser.add_argument("--lambda_rank", type=float, default=0.0)
    parser.add_argument("--lambda_cons", type=float, default=0.06)
    parser.add_argument("--stageC_joint_fused_kd_lambda", type=float, default=0.0)
    parser.add_argument("--joint_fused_targets_npz", type=str, default="")
    parser.add_argument("--joint_fused_targets_key", type=str, default="probs_fused_overall")
    parser.add_argument(
        "--joint_fused_split_scheme",
        type=str,
        default="train_val_test",
        choices=["train_val_test", "fit_ref_test"],
    )
    parser.add_argument(
        "--joint_fused_source_splits_npz",
        type=str,
        default="",
        help="Optional source data_splits.npz for exact Joint fused-train target index alignment.",
    )
    parser.add_argument("--joint_fused_source_val_key", type=str, default="val_idx")
    parser.add_argument("--joint_fused_source_test_key", type=str, default="test_idx")
    parser.add_argument(
        "--joint_fused_train_from",
        type=str,
        default="train",
        choices=["train", "fit", "ref"],
        help="Which fused partition to use for Joint train targets.",
    )
    parser.add_argument(
        "--joint_fused_val_from",
        type=str,
        default="val",
        choices=["val", "ref", "test"],
        help="Which fused partition to use for Joint val targets.",
    )
    parser.add_argument("--joint_fused_kd_temp", type=float, default=1.0)
    parser.add_argument("--joint_fused_adv_weight", type=float, default=0.0)
    parser.add_argument("--joint_fused_adv_power", type=float, default=1.0)
    parser.add_argument("--joint_fused_uncert_weight", type=float, default=0.0)
    parser.add_argument("--joint_fused_adv_use_abs", action="store_true")
    parser.add_argument("--joint_fused_kd_w_min", type=float, default=0.25)
    parser.add_argument("--joint_fused_kd_w_max", type=float, default=4.0)
    parser.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    parser.add_argument("--use_corrected_flags", action="store_true")
    parser.add_argument("--disable_split_again", action="store_true")
    parser.add_argument("--split_again_exist_thr", type=float, default=0.30)
    parser.add_argument("--split_again_psplit_thr", type=float, default=0.20)
    parser.add_argument("--split_again_dr_thr", type=float, default=0.10)
    parser.add_argument("--split_again_alpha", type=float, default=8.0)
    parser.add_argument("--split_again_beta", type=float, default=4.0)
    parser.add_argument("--split_again_gamma", type=float, default=3.0)
    parser.add_argument("--split_again_score_power", type=float, default=1.25)
    parser.add_argument("--split_again_budget_frac", type=float, default=1.0)
    parser.add_argument("--split_again_max_parent_added", type=float, default=2.0)
    parser.add_argument("--loss_w_pt_ratio", type=float, default=BASE_CONFIG["loss"]["w_pt_ratio"])
    parser.add_argument("--loss_w_e_ratio", type=float, default=BASE_CONFIG["loss"]["w_e_ratio"])
    parser.add_argument("--loss_w_budget", type=float, default=BASE_CONFIG["loss"]["w_budget"])
    parser.add_argument("--loss_w_sparse", type=float, default=BASE_CONFIG["loss"]["w_sparse"])
    parser.add_argument("--loss_w_local", type=float, default=BASE_CONFIG["loss"]["w_local"])
    parser.add_argument(
        "--added_target_scale",
        type=float,
        default=0.90,
        help="rho in [0,1] for non-privileged split targets: merge=rho*true_added, eff=(1-rho)*true_added.",
    )

    # Reconstructor decode controls (used for diagnostics and KD set build).
    parser.add_argument("--reco_weight_threshold", type=float, default=0.03)
    parser.add_argument("--reco_disable_budget_topk", action="store_true")

    # Overlap/disagreement and score-fusion reporting.
    parser.add_argument("--report_target_tpr", type=float, default=0.50)
    parser.add_argument("--combo_weight_step", type=float, default=0.01)

    # Response/resolution diagnostics.
    parser.add_argument("--response_n_bins", type=int, default=8)
    parser.add_argument("--response_min_count", type=int, default=30)
    parser.add_argument("--diag_match_max_jets", type=int, default=20000)
    parser.add_argument("--diag_match_seed", type=int, default=-1)

    # Stage D (final KD with frozen reconstructor and fixed corrected view)
    parser.add_argument("--disable_final_kd", action="store_true")
    parser.add_argument("--stageD_kd_epochs", type=int, default=-1)
    parser.add_argument("--stageD_kd_patience", type=int, default=-1)
    parser.add_argument("--stageD_kd_lr", type=float, default=-1.0)

    # Optional frozen jet-level regressor -> additional dual-view input channels.
    parser.add_argument("--enable_jet_regressor", action="store_true")
    parser.add_argument("--jet_reg_epochs", type=int, default=40)
    parser.add_argument("--jet_reg_patience", type=int, default=10)
    parser.add_argument("--jet_reg_lr", type=float, default=3e-4)
    parser.add_argument("--jet_reg_weight_decay", type=float, default=1e-5)
    parser.add_argument("--jet_reg_warmup_epochs", type=int, default=3)
    parser.add_argument("--jet_reg_embed_dim", type=int, default=128)
    parser.add_argument("--jet_reg_num_heads", type=int, default=8)
    parser.add_argument("--jet_reg_num_layers", type=int, default=4)
    parser.add_argument("--jet_reg_ff_dim", type=int, default=512)
    parser.add_argument("--jet_reg_dropout", type=float, default=0.1)

    args = parser.parse_args()
    set_seed(int(args.seed))

    # This variant always selects checkpoints by validation AUC.
    selection_metric = "auc"
    if str(args.selection_metric).lower() != "auc":
        print(f"Note: overriding --selection_metric={args.selection_metric} to 'auc' in this script.")

    cfg = _deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)
    cfg["loss"]["w_pt_ratio"] = float(args.loss_w_pt_ratio)
    cfg["loss"]["w_e_ratio"] = float(args.loss_w_e_ratio)
    cfg["loss"]["w_budget"] = float(args.loss_w_budget)
    cfg["loss"]["w_sparse"] = float(args.loss_w_sparse)
    cfg["loss"]["w_local"] = float(args.loss_w_local)

    # Global split-again settings used by corrected-view builder.
    SPLIT_AGAIN_CFG["enabled"] = not bool(args.disable_split_again)
    SPLIT_AGAIN_CFG["exist_thr"] = float(args.split_again_exist_thr)
    SPLIT_AGAIN_CFG["psplit_thr"] = float(args.split_again_psplit_thr)
    SPLIT_AGAIN_CFG["dr_thr"] = float(args.split_again_dr_thr)
    SPLIT_AGAIN_CFG["alpha"] = float(args.split_again_alpha)
    SPLIT_AGAIN_CFG["beta"] = float(args.split_again_beta)
    SPLIT_AGAIN_CFG["gamma"] = float(args.split_again_gamma)
    SPLIT_AGAIN_CFG["score_power"] = float(args.split_again_score_power)
    SPLIT_AGAIN_CFG["budget_frac"] = float(args.split_again_budget_frac)
    SPLIT_AGAIN_CFG["max_parent_added"] = float(args.split_again_max_parent_added)

    cfg["reconstructor_training"]["epochs"] = int(args.stageA_epochs)
    cfg["reconstructor_training"]["patience"] = int(args.stageA_patience)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = args.offset_jets + args.n_train_jets
    print("Loading offline constituents...")
    all_const_full, all_labels_full, all_train_w_full = load_raw_constituents_labels_weights_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
        use_train_weights=bool(args.use_train_weights),
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}"
        )

    const_raw = all_const_full[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)
    train_weight = all_train_w_full[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.float32)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(args.seed),
    )
    # Non-privileged target: supervise split counts from true_added only.
    # true_added_raw is offline-vs-HLT count gap and is available during training.
    # We allocate it with rho: merge=rho*true_added, eff=(1-rho)*true_added.
    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    rho = _clamp_target_scale(float(args.added_target_scale))
    budget_merge_true_raw = (rho * true_added_raw).astype(np.float32)
    budget_eff_true_raw = ((1.0 - rho) * true_added_raw).astype(np.float32)
    budget_merge_true = budget_merge_true_raw.copy()
    budget_eff_true = budget_eff_true_raw.copy()
    print(
        f"Non-priv rho split setup: rho={rho:.3f}, "
        f"mean_true_added_raw={float(true_added_raw.mean()):.3f}, "
        f"mean_target_merge={float(budget_merge_true.mean()):.3f}, "
        f"mean_target_eff={float(budget_eff_true.mean()):.3f}"
    )

    print("Computing features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    n_train_split = int(args.n_train_split)
    n_val_split = int(args.n_val_split)
    n_test_split = int(args.n_test_split)
    custom_split = (n_train_split > 0 and n_val_split > 0 and n_test_split > 0)

    idx = np.arange(len(labels))
    if custom_split:
        total_need = int(n_train_split + n_val_split + n_test_split)
        if total_need > len(idx):
            raise ValueError(
                f"Requested split counts exceed available jets: "
                f"{n_train_split}+{n_val_split}+{n_test_split} > {len(idx)}"
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
            train_size=int(n_train_split),
            random_state=int(args.seed),
            stratify=labels[idx_use],
        )
        val_idx, test_idx = train_test_split(
            rem_idx,
            train_size=int(n_val_split),
            test_size=int(n_test_split),
            random_state=int(args.seed),
            stratify=labels[rem_idx],
        )
    else:
        train_idx, temp_idx = train_test_split(
            idx, test_size=0.30, random_state=int(args.seed), stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=int(args.seed), stratify=labels[temp_idx]
        )
    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} "
        f"(custom_counts={custom_split})"
    )
    train_w_train = train_weight[train_idx].astype(np.float32)

    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    # Persist exact data setup/splits so Stage-C-only reruns can faithfully reload.
    data_setup = {
        "train_path_arg": str(args.train_path),
        "train_files": [str(p.resolve()) for p in train_files],
        "n_train_jets": int(args.n_train_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "seed": int(args.seed),
        "use_train_weights": bool(args.use_train_weights),
        "split": (
            {
                "mode": "custom_counts",
                "n_train_split": int(len(train_idx)),
                "n_val_split": int(len(val_idx)),
                "n_test_split": int(len(test_idx)),
            }
            if custom_split
            else {"mode": "fractions", "train_frac": 0.70, "val_frac": 0.15, "test_frac": 0.15}
        ),
        "hlt_effects": cfg["hlt_effects"],
        "variant": "nopriv_rhosplit",
        "split_again": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in SPLIT_AGAIN_CFG.items()},
        "rho": float(rho),
        "mean_true_added_raw": float(true_added_raw.mean()),
        "mean_target_merge": float(budget_merge_true.mean()),
        "mean_target_eff": float(budget_eff_true.mean()),
    }
    with open(save_root / "data_setup.json", "w", encoding="utf-8") as f:
        json.dump(data_setup, f, indent=2)
    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
    )

    # Teacher / baseline
    print("\n" + "=" * 70)
    print("STEP 1: TEACHER + BASELINE")
    print("=" * 70)
    BS = int(cfg["training"]["batch_size"])

    ds_train_off = JetDataset(feat_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    ds_val_off = JetDataset(feat_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    ds_test_off = JetDataset(feat_off_std[test_idx], masks_off[test_idx], labels[test_idx])
    dl_train_off = DataLoader(
        ds_train_off,
        batch_size=BS,
        sampler=_build_weighted_sampler(train_w_train) if bool(args.use_train_weights) else None,
        shuffle=False if bool(args.use_train_weights) else True,
        drop_last=True,
    )
    dl_val_off = DataLoader(ds_val_off, batch_size=BS, shuffle=False)
    dl_test_off = DataLoader(ds_test_off, batch_size=BS, shuffle=False)

    step1_load_dir = str(getattr(args, "step1_load_dir", "")).strip()
    step1_teacher_ckpt = None
    step1_baseline_ckpt = None
    if step1_load_dir:
        step1_dir = Path(step1_load_dir).expanduser().resolve()
        step1_teacher_ckpt = step1_dir / "teacher.pt"
        step1_baseline_ckpt = step1_dir / "baseline.pt"

    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    if step1_teacher_ckpt is not None and step1_teacher_ckpt.is_file():
        teacher_pack = torch.load(step1_teacher_ckpt, map_location=device)
        teacher_sd = teacher_pack["model"] if isinstance(teacher_pack, dict) and "model" in teacher_pack else teacher_pack
        teacher.load_state_dict(teacher_sd, strict=True)
        print(f"STEP 1: loaded teacher checkpoint from {step1_teacher_ckpt}")
    else:
        teacher = train_single_view_classifier_auc(
            teacher, dl_train_off, dl_val_off, device, cfg["training"], name="Teacher"
        )
    auc_teacher, preds_teacher, labs = eval_classifier(teacher, dl_test_off, device)
    auc_teacher_val, preds_teacher_val, labs_val_teacher = eval_classifier(teacher, dl_val_off, device)

    ds_train_hlt = JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx])
    ds_val_hlt = JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    ds_test_hlt = JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_train_hlt = DataLoader(
        ds_train_hlt,
        batch_size=BS,
        sampler=_build_weighted_sampler(train_w_train) if bool(args.use_train_weights) else None,
        shuffle=False if bool(args.use_train_weights) else True,
        drop_last=True,
    )
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=BS, shuffle=False)
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=BS, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    if step1_baseline_ckpt is not None and step1_baseline_ckpt.is_file():
        baseline_pack = torch.load(step1_baseline_ckpt, map_location=device)
        baseline_sd = baseline_pack["model"] if isinstance(baseline_pack, dict) and "model" in baseline_pack else baseline_pack
        baseline.load_state_dict(baseline_sd, strict=True)
        print(f"STEP 1: loaded baseline checkpoint from {step1_baseline_ckpt}")
    else:
        baseline = train_single_view_classifier_auc(
            baseline, dl_train_hlt, dl_val_hlt, device, cfg["training"], name="Baseline"
        )
    auc_baseline, preds_baseline, _ = eval_classifier(baseline, dl_test_hlt, device)
    auc_baseline_val, preds_baseline_val, labs_val_baseline = eval_classifier(baseline, dl_val_hlt, device)
    assert np.array_equal(labs_val_teacher.astype(np.float32), labs_val_baseline.astype(np.float32))

    # Optional jet-level regressor to provide frozen global calibration features to dual-view tagger.
    jet_regressor = None
    jet_reg_metrics: Dict[str, object] = {"enabled": bool(args.enable_jet_regressor)}
    feat_hlt_dual = feat_hlt_std.astype(np.float32, copy=True)
    if bool(args.enable_jet_regressor):
        print("\n" + "=" * 70)
        print("STEP 1B: JET-LEVEL REGRESSOR (HLT -> offline global jet targets)")
        print("=" * 70)
        # Targets:
        # [log_pt, log_e, log_m, tau21, tau32, log1p_d2, log1p_n_off, log1p_n_added]
        target_off, target_hlt_ref, target_idx = compute_jet_regression_targets(
            const_off=const_off,
            mask_off=masks_off,
            const_hlt=hlt_const,
            mask_hlt=hlt_mask,
        )
        target_dim = int(target_off.shape[1])

        jet_reg_train_ds = JetRegressionDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], target_off[train_idx])
        jet_reg_val_ds = JetRegressionDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], target_off[val_idx])
        jet_reg_train_loader = DataLoader(
            jet_reg_train_ds,
            batch_size=int(cfg["training"]["batch_size"]),
            sampler=_build_weighted_sampler(train_w_train) if bool(args.use_train_weights) else None,
            shuffle=False if bool(args.use_train_weights) else True,
            drop_last=True,
        )
        jet_reg_val_loader = DataLoader(
            jet_reg_val_ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
        )

        jet_regressor = JetLevelRegressor(
            input_dim=7,
            output_dim=target_dim,
            embed_dim=int(args.jet_reg_embed_dim),
            num_heads=int(args.jet_reg_num_heads),
            num_layers=int(args.jet_reg_num_layers),
            ff_dim=int(args.jet_reg_ff_dim),
            dropout=float(args.jet_reg_dropout),
        ).to(device)
        jet_regressor, jet_reg_best = train_jet_regressor(
            model=jet_regressor,
            train_loader=jet_reg_train_loader,
            val_loader=jet_reg_val_loader,
            device=device,
            epochs=int(args.jet_reg_epochs),
            patience=int(args.jet_reg_patience),
            lr=float(args.jet_reg_lr),
            weight_decay=float(args.jet_reg_weight_decay),
            warmup_epochs=int(args.jet_reg_warmup_epochs),
        )
        pred_log_all = predict_jet_regressor(
            model=jet_regressor,
            feat=feat_hlt_std,
            mask=hlt_mask,
            device=device,
            batch_size=int(cfg["training"]["batch_size"]),
        )
        delta_vs_hlt = pred_log_all - target_hlt_ref
        extra_global = np.concatenate([pred_log_all, delta_vs_hlt], axis=-1).astype(np.float32)
        extra_global = np.repeat(extra_global[:, None, :], feat_hlt_std.shape[1], axis=1)
        feat_hlt_dual = np.concatenate([feat_hlt_std, extra_global], axis=-1).astype(np.float32)
        feat_hlt_dual[~hlt_mask] = 0.0

        # Eval metrics on val/test.
        jet_reg_val_pred = pred_log_all[val_idx]
        jet_reg_test_pred = pred_log_all[test_idx]
        jet_reg_val_true = target_off[val_idx]
        jet_reg_test_true = target_off[test_idx]
        val_m = _jet_reg_metric_dict(jet_reg_val_pred, jet_reg_val_true, target_idx)
        test_m = _jet_reg_metric_dict(jet_reg_test_pred, jet_reg_test_true, target_idx)
        jet_reg_metrics = {
            "enabled": True,
            "best": jet_reg_best,
            "target_index": target_idx,
            "val": val_m,
            "test": test_m,
        }
        print(
            "Jet regressor test MAE: "
            f"pT={test_m['mae_pt']:.3f}, E={test_m['mae_e']:.3f}, "
            f"mass={test_m['mae_m']:.3f}, tau21={test_m['mae_tau21']:.4f}, "
            f"tau32={test_m['mae_tau32']:.4f}, n_added={test_m['mae_n_added']:.3f}"
        )

    def _load_split_targets_from_npz(
        npz_path: str,
        key_prefix: str,
        scheme: str,
        require_test: bool,
        expected_train_len: int,
        expected_val_len: int,
        expected_test_len: int,
        desc: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        p = Path(npz_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Missing {desc} NPZ: {p}")
        arr = np.load(p)
        scheme_l = str(scheme).strip().lower()
        if scheme_l == "fit_ref_test":
            ktr = f"{key_prefix}_fit"
            kva = f"{key_prefix}_ref"
            kte = f"{key_prefix}_test"
        else:
            ktr = f"{key_prefix}_train"
            kva = f"{key_prefix}_val"
            kte = f"{key_prefix}_test"
        need = [ktr, kva] + ([kte] if bool(require_test) else [])
        miss = [k for k in need if k not in arr]
        if len(miss) > 0:
            raise KeyError(
                f"Missing keys for {desc} in {p}: {miss}. Available keys: {sorted(arr.files)}"
            )
        tr = np.asarray(arr[ktr], dtype=np.float32).reshape(-1)
        va = np.asarray(arr[kva], dtype=np.float32).reshape(-1)
        te = np.asarray(arr[kte], dtype=np.float32).reshape(-1) if bool(require_test) else None
        if int(tr.shape[0]) != int(expected_train_len):
            raise ValueError(
                f"{desc} train length mismatch: {int(tr.shape[0])} vs expected {int(expected_train_len)}."
            )
        if int(va.shape[0]) != int(expected_val_len):
            raise ValueError(
                f"{desc} val length mismatch: {int(va.shape[0])} vs expected {int(expected_val_len)}."
            )
        if te is not None and int(te.shape[0]) != int(expected_test_len):
            raise ValueError(
                f"{desc} test length mismatch: {int(te.shape[0])} vs expected {int(expected_test_len)}."
            )
        return tr, va, te

    # Optional direct fused targets for Stage-A logit distillation.
    stagea_train_idx_sel = train_idx
    stagea_val_idx_sel = val_idx
    stagea_fused_tr = None
    stagea_fused_va = None
    stagea_anchor_tr = None
    stagea_anchor_va = None
    stagea_fused_npz_path = str(getattr(args, "stageA_fused_targets_npz", "")).strip()
    if len(stagea_fused_npz_path) > 0:
        fused_npz = Path(stagea_fused_npz_path).expanduser().resolve()
        arr_fused = np.load(fused_npz)
        key_prefix = str(getattr(args, "stageA_fused_targets_key", "probs_fused_overall")).strip()
        scheme = str(getattr(args, "stageA_fused_split_scheme", "train_val_test")).strip().lower()
        if scheme == "fit_ref_test":
            k_fit = f"{key_prefix}_fit"
            k_ref = f"{key_prefix}_ref"
            k_test = f"{key_prefix}_test"
            k_train = k_fit
            k_val = k_ref
        else:
            k_fit = f"{key_prefix}_fit" if f"{key_prefix}_fit" in arr_fused else f"{key_prefix}_train"
            k_ref = f"{key_prefix}_ref" if f"{key_prefix}_ref" in arr_fused else f"{key_prefix}_val"
            k_test = f"{key_prefix}_test"
            k_train = f"{key_prefix}_train"
            k_val = f"{key_prefix}_val"

        if k_fit not in arr_fused or k_ref not in arr_fused or k_test not in arr_fused:
            raise KeyError(
                f"Missing fused keys in {fused_npz}. "
                f"Need `{k_fit}`, `{k_ref}`, `{k_test}`. Available keys: {sorted(arr_fused.files)}"
            )
        y_fit = np.asarray(arr_fused[k_fit], dtype=np.float32).reshape(-1)
        y_ref = np.asarray(arr_fused[k_ref], dtype=np.float32).reshape(-1)
        y_test = np.asarray(arr_fused[k_test], dtype=np.float32).reshape(-1)

        source_splits_npz = str(getattr(args, "stageA_fused_source_splits_npz", "")).strip()
        if len(source_splits_npz) > 0:
            src_npz = Path(source_splits_npz).expanduser().resolve()
            src = np.load(src_npz)
            src_val_key = str(getattr(args, "stageA_fused_source_val_key", "val_idx")).strip()
            src_test_key = str(getattr(args, "stageA_fused_source_test_key", "test_idx")).strip()
            if src_val_key not in src or src_test_key not in src:
                raise KeyError(
                    f"Missing `{src_val_key}` or `{src_test_key}` in {src_npz}. "
                    f"Available keys: {sorted(src.files)}"
                )
            if "idx_fit" not in arr_fused or "idx_ref" not in arr_fused:
                raise KeyError(
                    f"Missing idx_fit/idx_ref in fused NPZ for source alignment: {fused_npz}"
                )
            src_val = np.asarray(src[src_val_key], dtype=np.int64).reshape(-1)
            src_test = np.asarray(src[src_test_key], dtype=np.int64).reshape(-1)
            idx_fit_local = np.asarray(arr_fused["idx_fit"], dtype=np.int64).reshape(-1)
            idx_ref_local = np.asarray(arr_fused["idx_ref"], dtype=np.int64).reshape(-1)
            if idx_fit_local.size == 0 or idx_ref_local.size == 0:
                raise ValueError("idx_fit/idx_ref empty in fused NPZ.")
            if int(idx_fit_local.max()) >= int(src_val.shape[0]) or int(idx_ref_local.max()) >= int(src_val.shape[0]):
                raise ValueError(
                    "idx_fit/idx_ref out of bounds for source val split: "
                    f"fit_max={int(idx_fit_local.max())}, ref_max={int(idx_ref_local.max())}, val_len={int(src_val.shape[0])}"
                )

            train_from = str(getattr(args, "stageA_fused_train_from", "train")).strip().lower()
            val_from = str(getattr(args, "stageA_fused_val_from", "val")).strip().lower()

            if train_from == "fit":
                stagea_train_idx_sel = src_val[idx_fit_local]
                stagea_fused_tr = y_fit
            elif train_from == "ref":
                stagea_train_idx_sel = src_val[idx_ref_local]
                stagea_fused_tr = y_ref
            else:
                if k_train not in arr_fused:
                    raise KeyError(f"Missing `{k_train}` in {fused_npz} for train_from=train")
                stagea_train_idx_sel = train_idx
                stagea_fused_tr = np.asarray(arr_fused[k_train], dtype=np.float32).reshape(-1)

            if val_from == "ref":
                stagea_val_idx_sel = src_val[idx_ref_local]
                stagea_fused_va = y_ref
            elif val_from == "test":
                stagea_val_idx_sel = src_test
                stagea_fused_va = y_test
            else:
                if k_val not in arr_fused:
                    raise KeyError(f"Missing `{k_val}` in {fused_npz} for val_from=val")
                stagea_val_idx_sel = val_idx
                stagea_fused_va = np.asarray(arr_fused[k_val], dtype=np.float32).reshape(-1)

            print(
                "Stage-A fused source alignment enabled: "
                f"src={src_npz}, train_from={train_from}, val_from={val_from} | "
                f"train={len(stagea_train_idx_sel)} val={len(stagea_val_idx_sel)}"
            )
        else:
            stagea_fused_tr, stagea_fused_va, _ = _load_split_targets_from_npz(
                npz_path=stagea_fused_npz_path,
                key_prefix=key_prefix,
                scheme=scheme,
                require_test=False,
                expected_train_len=int(len(train_idx)),
                expected_val_len=int(len(val_idx)),
                expected_test_len=int(len(test_idx)),
                desc="Stage-A fused targets",
            )
        if stagea_fused_tr is not None and int(stagea_fused_tr.shape[0]) != int(len(stagea_train_idx_sel)):
            raise ValueError(
                f"Stage-A fused train length mismatch: {int(stagea_fused_tr.shape[0])} vs train idx {int(len(stagea_train_idx_sel))}"
            )
        if stagea_fused_va is not None and int(stagea_fused_va.shape[0]) != int(len(stagea_val_idx_sel)):
            raise ValueError(
                f"Stage-A fused val length mismatch: {int(stagea_fused_va.shape[0])} vs val idx {int(len(stagea_val_idx_sel))}"
            )
        print(
            "Stage-A fused targets loaded: "
            f"{fused_npz} | key={key_prefix} | scheme={scheme} | "
            f"train={None if stagea_fused_tr is None else stagea_fused_tr.shape} "
            f"val={None if stagea_fused_va is None else stagea_fused_va.shape}"
        )
        try:
            if stagea_fused_tr is not None and np.unique(labels[stagea_train_idx_sel]).size > 1:
                auc_t = float(roc_auc_score(labels[stagea_train_idx_sel].astype(np.int64), stagea_fused_tr.astype(np.float64)))
            else:
                auc_t = float("nan")
            if stagea_fused_va is not None and np.unique(labels[stagea_val_idx_sel]).size > 1:
                auc_v = float(roc_auc_score(labels[stagea_val_idx_sel].astype(np.int64), stagea_fused_va.astype(np.float64)))
            else:
                auc_v = float("nan")
            print(f"Stage-A fused target sanity AUC: train={auc_t:.4f}, val={auc_v:.4f}")
        except Exception as e:
            print(f"Warning: unable to compute Stage-A fused target sanity AUC ({e})")

    stagea_anchor_npz_path = str(getattr(args, "stageA_anchor_targets_npz", "")).strip()
    if len(stagea_anchor_npz_path) > 0:
        anchor_key_prefix = str(getattr(args, "stageA_anchor_targets_key", "probs_anchor_overall")).strip()
        anchor_scheme = str(getattr(args, "stageA_anchor_split_scheme", "train_val_test")).strip().lower()
        stagea_anchor_tr, stagea_anchor_va, _ = _load_split_targets_from_npz(
            npz_path=stagea_anchor_npz_path,
            key_prefix=anchor_key_prefix,
            scheme=anchor_scheme,
            require_test=False,
            expected_train_len=int(len(stagea_train_idx_sel)),
            expected_val_len=int(len(stagea_val_idx_sel)),
            expected_test_len=int(len(test_idx)),
            desc="Stage-A anchor targets",
        )
        print(
            "Stage-A anchor targets loaded: "
            f"{Path(stagea_anchor_npz_path).expanduser().resolve()} | key={anchor_key_prefix} | scheme={anchor_scheme} | "
            f"train={stagea_anchor_tr.shape} val={stagea_anchor_va.shape}"
        )

    joint_train_idx_sel = train_idx
    joint_val_idx_sel = val_idx
    joint_test_idx_sel = test_idx
    joint_fused_tr = None
    joint_fused_va = None
    joint_fused_te = None
    joint_fused_npz_path = str(getattr(args, "joint_fused_targets_npz", "")).strip()
    if len(joint_fused_npz_path) == 0 and len(stagea_fused_npz_path) > 0:
        joint_fused_npz_path = stagea_fused_npz_path
    if len(joint_fused_npz_path) > 0:
        joint_fused_npz = Path(joint_fused_npz_path).expanduser().resolve()
        arr_joint_fused = np.load(joint_fused_npz)
        joint_key_prefix = str(getattr(args, "joint_fused_targets_key", "probs_fused_overall")).strip()
        joint_scheme = str(getattr(args, "joint_fused_split_scheme", "train_val_test")).strip().lower()
        if joint_scheme == "fit_ref_test":
            k_fit = f"{joint_key_prefix}_fit"
            k_ref = f"{joint_key_prefix}_ref"
            k_test = f"{joint_key_prefix}_test"
            k_train = k_fit
            k_val = k_ref
        else:
            k_fit = f"{joint_key_prefix}_fit" if f"{joint_key_prefix}_fit" in arr_joint_fused else f"{joint_key_prefix}_train"
            k_ref = f"{joint_key_prefix}_ref" if f"{joint_key_prefix}_ref" in arr_joint_fused else f"{joint_key_prefix}_val"
            k_test = f"{joint_key_prefix}_test"
            k_train = f"{joint_key_prefix}_train"
            k_val = f"{joint_key_prefix}_val"

        if k_fit not in arr_joint_fused or k_ref not in arr_joint_fused or k_test not in arr_joint_fused:
            raise KeyError(
                f"Missing fused keys in {joint_fused_npz}. "
                f"Need `{k_fit}`, `{k_ref}`, `{k_test}`. Available keys: {sorted(arr_joint_fused.files)}"
            )

        source_splits_npz = str(getattr(args, "joint_fused_source_splits_npz", "")).strip()
        if len(source_splits_npz) > 0:
            src_npz = Path(source_splits_npz).expanduser().resolve()
            src = np.load(src_npz)
            src_val_key = str(getattr(args, "joint_fused_source_val_key", "val_idx")).strip()
            src_test_key = str(getattr(args, "joint_fused_source_test_key", "test_idx")).strip()
            if src_val_key not in src or src_test_key not in src:
                raise KeyError(
                    f"Missing `{src_val_key}` or `{src_test_key}` in {src_npz}. "
                    f"Available keys: {sorted(src.files)}"
                )
            if "idx_fit" not in arr_joint_fused or "idx_ref" not in arr_joint_fused:
                raise KeyError(
                    f"Missing idx_fit/idx_ref in fused NPZ for source alignment: {joint_fused_npz}"
                )

            src_val = np.asarray(src[src_val_key], dtype=np.int64).reshape(-1)
            src_test = np.asarray(src[src_test_key], dtype=np.int64).reshape(-1)
            idx_fit_local = np.asarray(arr_joint_fused["idx_fit"], dtype=np.int64).reshape(-1)
            idx_ref_local = np.asarray(arr_joint_fused["idx_ref"], dtype=np.int64).reshape(-1)
            if idx_fit_local.size == 0 or idx_ref_local.size == 0:
                raise ValueError("idx_fit/idx_ref empty in fused NPZ.")
            if int(idx_fit_local.max()) >= int(src_val.shape[0]) or int(idx_ref_local.max()) >= int(src_val.shape[0]):
                raise ValueError(
                    "idx_fit/idx_ref out of bounds for source val split: "
                    f"fit_max={int(idx_fit_local.max())}, ref_max={int(idx_ref_local.max())}, val_len={int(src_val.shape[0])}"
                )

            y_fit = np.asarray(arr_joint_fused[k_fit], dtype=np.float32).reshape(-1)
            y_ref = np.asarray(arr_joint_fused[k_ref], dtype=np.float32).reshape(-1)
            y_test = np.asarray(arr_joint_fused[k_test], dtype=np.float32).reshape(-1)
            train_from = str(getattr(args, "joint_fused_train_from", "train")).strip().lower()
            val_from = str(getattr(args, "joint_fused_val_from", "val")).strip().lower()

            if train_from == "fit":
                joint_train_idx_sel = src_val[idx_fit_local]
                joint_fused_tr = y_fit
            elif train_from == "ref":
                joint_train_idx_sel = src_val[idx_ref_local]
                joint_fused_tr = y_ref
            else:
                if k_train not in arr_joint_fused:
                    raise KeyError(f"Missing `{k_train}` in {joint_fused_npz} for joint_fused_train_from=train")
                joint_train_idx_sel = train_idx
                joint_fused_tr = np.asarray(arr_joint_fused[k_train], dtype=np.float32).reshape(-1)

            if val_from == "ref":
                joint_val_idx_sel = src_val[idx_ref_local]
                joint_fused_va = y_ref
            elif val_from == "test":
                joint_val_idx_sel = src_test
                joint_fused_va = y_test
            else:
                if k_val not in arr_joint_fused:
                    raise KeyError(f"Missing `{k_val}` in {joint_fused_npz} for joint_fused_val_from=val")
                joint_val_idx_sel = val_idx
                joint_fused_va = np.asarray(arr_joint_fused[k_val], dtype=np.float32).reshape(-1)

            # Keep evaluation split fixed to this run's test split for metric comparability.
            if k_test in arr_joint_fused and int(np.asarray(arr_joint_fused[k_test]).reshape(-1).shape[0]) == int(len(test_idx)):
                joint_test_idx_sel = test_idx
                joint_fused_te = np.asarray(arr_joint_fused[k_test], dtype=np.float32).reshape(-1)
            else:
                joint_test_idx_sel = test_idx
                joint_fused_te = None

            print(
                "Joint fused source alignment enabled: "
                f"src={src_npz}, train_from={train_from}, val_from={val_from} | "
                f"train={len(joint_train_idx_sel)} val={len(joint_val_idx_sel)} test={len(joint_test_idx_sel)}"
            )
        else:
            joint_fused_tr, joint_fused_va, joint_fused_te = _load_split_targets_from_npz(
                npz_path=joint_fused_npz_path,
                key_prefix=joint_key_prefix,
                scheme=joint_scheme,
                require_test=True,
                expected_train_len=int(len(train_idx)),
                expected_val_len=int(len(val_idx)),
                expected_test_len=int(len(test_idx)),
                desc="Joint fused targets",
            )

        if joint_fused_tr is not None and int(joint_fused_tr.shape[0]) != int(len(joint_train_idx_sel)):
            raise ValueError(
                f"Joint fused train length mismatch: {int(joint_fused_tr.shape[0])} vs train idx {int(len(joint_train_idx_sel))}"
            )
        if joint_fused_va is not None and int(joint_fused_va.shape[0]) != int(len(joint_val_idx_sel)):
            raise ValueError(
                f"Joint fused val length mismatch: {int(joint_fused_va.shape[0])} vs val idx {int(len(joint_val_idx_sel))}"
            )
        if joint_fused_te is not None and int(joint_fused_te.shape[0]) != int(len(joint_test_idx_sel)):
            raise ValueError(
                f"Joint fused test length mismatch: {int(joint_fused_te.shape[0])} vs test idx {int(len(joint_test_idx_sel))}"
            )
        print(
            "Joint fused targets loaded: "
            f"{joint_fused_npz} | key={joint_key_prefix} | scheme={joint_scheme} | "
            f"train={None if joint_fused_tr is None else joint_fused_tr.shape} "
            f"val={None if joint_fused_va is None else joint_fused_va.shape} "
            f"test={None if joint_fused_te is None else joint_fused_te.shape}"
        )

    # Stage-A teacher model can be overridden independently from STEP-1 teacher.
    stagea_teacher_model = teacher
    stagea_teacher_ckpt = str(getattr(args, "stageA_teacher_ckpt", "")).strip()
    if len(stagea_teacher_ckpt) > 0:
        teacher_override_path = Path(stagea_teacher_ckpt).expanduser().resolve()
        if not teacher_override_path.is_file():
            raise FileNotFoundError(f"Missing --stageA_teacher_ckpt: {teacher_override_path}")
        stagea_teacher_model = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
        teacher_override_pack = torch.load(teacher_override_path, map_location=device)
        teacher_override_sd = (
            teacher_override_pack["model"]
            if isinstance(teacher_override_pack, dict) and ("model" in teacher_override_pack)
            else teacher_override_pack
        )
        stagea_teacher_model.load_state_dict(teacher_override_sd, strict=True)
        print(f"Stage-A teacher override loaded from: {teacher_override_path}")

    # Stage A: reconstructor pretrain
    print("\n" + "=" * 70)
    print("STEP 2: STAGE A (RECONSTRUCTOR PRETRAIN)")
    print("=" * 70)
    ds_train_reco = StageAReconstructionDataset(
        feat_hlt_std[stagea_train_idx_sel], hlt_mask[stagea_train_idx_sel], hlt_const[stagea_train_idx_sel],
        const_off[stagea_train_idx_sel], masks_off[stagea_train_idx_sel], labels[stagea_train_idx_sel],
        budget_merge_true[stagea_train_idx_sel], budget_eff_true[stagea_train_idx_sel],
        stagea_fused_target=stagea_fused_tr,
        stagea_anchor_target=stagea_anchor_tr,
    )
    ds_val_reco = StageAReconstructionDataset(
        feat_hlt_std[stagea_val_idx_sel], hlt_mask[stagea_val_idx_sel], hlt_const[stagea_val_idx_sel],
        const_off[stagea_val_idx_sel], masks_off[stagea_val_idx_sel], labels[stagea_val_idx_sel],
        budget_merge_true[stagea_val_idx_sel], budget_eff_true[stagea_val_idx_sel],
        stagea_fused_target=stagea_fused_va,
        stagea_anchor_target=stagea_anchor_va,
    )
    stagea_train_weight = train_weight[stagea_train_idx_sel].astype(np.float32)
    dl_train_reco = DataLoader(
        ds_train_reco,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        sampler=_build_weighted_sampler(stagea_train_weight) if bool(args.use_train_weights) else None,
        shuffle=False if bool(args.use_train_weights) else True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_reco = DataLoader(
        ds_val_reco,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    BASE_CONFIG["loss"] = cfg["loss"]
    reconstructor, reco_val_metrics = train_reconstructor_teacher_guided(
        model=reconstructor,
        train_loader=dl_train_reco,
        val_loader=dl_val_reco,
        device=device,
        train_cfg=cfg["reconstructor_training"],
        loss_cfg=cfg["loss"],
        teacher_model=stagea_teacher_model,
        feat_means=means.astype(np.float32),
        feat_stds=stds.astype(np.float32),
        kd_temperature=float(args.stageA_kd_temp),
        lambda_kd=float(args.stageA_lambda_kd),
        lambda_emb=float(args.stageA_lambda_emb),
        lambda_tok=float(args.stageA_lambda_tok),
        lambda_phys=float(args.stageA_lambda_phys),
        lambda_budget_hinge=float(args.stageA_lambda_budget_hinge),
        budget_eps=float(args.stageA_budget_eps),
        budget_weight_floor=float(args.stageA_budget_weight_floor),
        target_tpr_for_fpr=float(args.stageA_target_tpr),
        normalize_loss_terms=not bool(args.disable_stageA_loss_normalization),
        loss_norm_ema_decay=float(args.stageA_loss_norm_ema_decay),
        loss_norm_eps=float(args.stageA_loss_norm_eps),
        reload_best_at_stage_transition=not bool(args.disable_stageA_stagewise_best_reload),
        stagea_fused_adv_weight=float(args.stageA_fused_adv_weight),
        stagea_fused_adv_power=float(args.stageA_fused_adv_power),
        stagea_fused_uncert_weight=float(args.stageA_fused_uncert_weight),
        stagea_fused_adv_use_abs=bool(args.stageA_fused_adv_use_abs),
        stagea_fused_kd_w_min=float(args.stageA_fused_kd_w_min),
        stagea_fused_kd_w_max=float(args.stageA_fused_kd_w_max),
        stagea_lambda_delta_aux=float(args.stageA_lambda_delta_aux),
    )

    reco_only_model = None
    auc_reco_only = float("nan")
    auc_reco_only_val = float("nan")
    preds_reco_only = None
    preds_reco_only_val = None
    reco_only_metrics: Dict[str, object] = {"enabled": bool(args.train_reco_only_after_stageA)}
    if bool(args.train_reco_only_after_stageA):
        print("\n" + "=" * 70)
        print("STEP 2B: RECO-ONLY TOP-TAGGER (FROZEN STAGE-A RECONSTRUCTOR)")
        print("=" * 70)

        reco_bs = int(max(1, int(args.reco_only_batch_size)))
        reco_feat_tr, reco_mask_tr = reconstruct_and_standardize_features(
            reconstructor=reconstructor,
            feat_hlt_std=feat_hlt_std[train_idx],
            mask_hlt=hlt_mask[train_idx],
            const_hlt=hlt_const[train_idx],
            means=means,
            stds=stds,
            max_constits=int(args.max_constits),
            device=device,
            batch_size=int(cfg["reconstructor_training"]["batch_size"]),
            weight_threshold=float(args.reco_weight_threshold),
            use_budget_topk=not bool(args.reco_disable_budget_topk),
        )
        reco_feat_va, reco_mask_va = reconstruct_and_standardize_features(
            reconstructor=reconstructor,
            feat_hlt_std=feat_hlt_std[val_idx],
            mask_hlt=hlt_mask[val_idx],
            const_hlt=hlt_const[val_idx],
            means=means,
            stds=stds,
            max_constits=int(args.max_constits),
            device=device,
            batch_size=int(cfg["reconstructor_training"]["batch_size"]),
            weight_threshold=float(args.reco_weight_threshold),
            use_budget_topk=not bool(args.reco_disable_budget_topk),
        )
        reco_feat_te, reco_mask_te = reconstruct_and_standardize_features(
            reconstructor=reconstructor,
            feat_hlt_std=feat_hlt_std[test_idx],
            mask_hlt=hlt_mask[test_idx],
            const_hlt=hlt_const[test_idx],
            means=means,
            stds=stds,
            max_constits=int(args.max_constits),
            device=device,
            batch_size=int(cfg["reconstructor_training"]["batch_size"]),
            weight_threshold=float(args.reco_weight_threshold),
            use_budget_topk=not bool(args.reco_disable_budget_topk),
        )

        ds_train_reco_only = JetDataset(reco_feat_tr, reco_mask_tr, labels[train_idx])
        ds_val_reco_only = JetDataset(reco_feat_va, reco_mask_va, labels[val_idx])
        ds_test_reco_only = JetDataset(reco_feat_te, reco_mask_te, labels[test_idx])
        dl_train_reco_only = DataLoader(
            ds_train_reco_only,
            batch_size=reco_bs,
            sampler=_build_weighted_sampler(train_w_train) if bool(args.use_train_weights) else None,
            shuffle=False if bool(args.use_train_weights) else True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        dl_val_reco_only = DataLoader(
            ds_val_reco_only,
            batch_size=reco_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        dl_test_reco_only = DataLoader(
            ds_test_reco_only,
            batch_size=reco_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        reco_only_cfg = {
            "epochs": int(args.reco_only_epochs),
            "patience": int(args.reco_only_patience),
            "lr": float(args.reco_only_lr),
            "weight_decay": float(args.reco_only_weight_decay),
            "warmup_epochs": int(args.reco_only_warmup_epochs),
        }
        reco_only_model = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
        reco_only_model = train_single_view_classifier_auc(
            reco_only_model,
            dl_train_reco_only,
            dl_val_reco_only,
            device,
            reco_only_cfg,
            name="RecoOnly",
        )
        auc_reco_only, preds_reco_only, labs_reco_only = eval_classifier(reco_only_model, dl_test_reco_only, device)
        auc_reco_only_val, preds_reco_only_val, labs_reco_only_val = eval_classifier(reco_only_model, dl_val_reco_only, device)
        assert np.array_equal(labs.astype(np.float32), labs_reco_only.astype(np.float32))
        assert np.array_equal(labs_val_teacher.astype(np.float32), labs_reco_only_val.astype(np.float32))

        fpr_ro, tpr_ro, _ = roc_curve(labs_reco_only, preds_reco_only)
        fpr30_ro = fpr_at_target_tpr(fpr_ro, tpr_ro, 0.30)
        fpr50_ro = fpr_at_target_tpr(fpr_ro, tpr_ro, 0.50)
        print(
            f"RecoOnly probe: AUC_test={auc_reco_only:.4f}, "
            f"FPR@30={fpr30_ro:.6f}, FPR@50={fpr50_ro:.6f}"
        )
        reco_only_metrics = {
            "enabled": True,
            "train_cfg": reco_only_cfg,
            "auc_val": float(auc_reco_only_val),
            "auc_test": float(auc_reco_only),
            "fpr30_test": float(fpr30_ro),
            "fpr50_test": float(fpr50_ro),
        }

    # Joint datasets
    ds_train_joint = JointDualDataset(
        feat_hlt_std[joint_train_idx_sel], feat_hlt_dual[joint_train_idx_sel], hlt_mask[joint_train_idx_sel], hlt_const[joint_train_idx_sel],
        const_off[joint_train_idx_sel], masks_off[joint_train_idx_sel],
        budget_merge_true[joint_train_idx_sel], budget_eff_true[joint_train_idx_sel],
        labels[joint_train_idx_sel],
        joint_fused_target=joint_fused_tr,
    )
    ds_val_joint = JointDualDataset(
        feat_hlt_std[joint_val_idx_sel], feat_hlt_dual[joint_val_idx_sel], hlt_mask[joint_val_idx_sel], hlt_const[joint_val_idx_sel],
        const_off[joint_val_idx_sel], masks_off[joint_val_idx_sel],
        budget_merge_true[joint_val_idx_sel], budget_eff_true[joint_val_idx_sel],
        labels[joint_val_idx_sel],
        joint_fused_target=joint_fused_va,
    )
    ds_test_joint = JointDualDataset(
        feat_hlt_std[joint_test_idx_sel], feat_hlt_dual[joint_test_idx_sel], hlt_mask[joint_test_idx_sel], hlt_const[joint_test_idx_sel],
        const_off[joint_test_idx_sel], masks_off[joint_test_idx_sel],
        budget_merge_true[joint_test_idx_sel], budget_eff_true[joint_test_idx_sel],
        labels[joint_test_idx_sel],
        joint_fused_target=joint_fused_te,
    )

    joint_train_weight = train_weight[joint_train_idx_sel].astype(np.float32)
    dl_train_joint = DataLoader(
        ds_train_joint,
        batch_size=BS,
        sampler=_build_weighted_sampler(joint_train_weight) if bool(args.use_train_weights) else None,
        shuffle=False if bool(args.use_train_weights) else True,
        drop_last=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint = DataLoader(
        ds_val_joint, batch_size=BS, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    dl_test_joint = DataLoader(
        ds_test_joint, batch_size=BS, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # Stage B + C: task-driven classifier/reconstructor coupling
    print("\n" + "=" * 70)
    print("STEP 3: STAGE B (DUAL PRETRAIN, FROZEN RECONSTRUCTOR)")
    print("=" * 70)
    dual_input_dim_a = int(feat_hlt_dual.shape[-1])
    dual_input_dim_b = 12 if bool(args.use_corrected_flags) else 10
    dual_joint = DualViewCrossAttnClassifier(input_dim_a=dual_input_dim_a, input_dim_b=dual_input_dim_b, **cfg["model"]).to(device)
    reconstructor, dual_joint, stageB_metrics, stageB_states = train_joint_dual(
        reconstructor=reconstructor,
        dual_model=dual_joint,
        train_loader=dl_train_joint,
        val_loader=dl_val_joint,
        device=device,
        stage_name="StageB-DualPretrain",
        freeze_reconstructor=True,
        epochs=int(args.stageB_epochs),
        patience=int(args.stageB_patience),
        lr_dual=float(args.stageB_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        lambda_reco=0.0,
        lambda_rank=float(args.stageB_lambda_rank),
        lambda_cons=float(args.stageB_lambda_cons),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
        min_epochs=int(args.stageB_min_epochs),
        select_metric=selection_metric,
        joint_fused_kd_lambda=float(args.stageB_joint_fused_kd_lambda),
        joint_fused_kd_temp=float(args.joint_fused_kd_temp),
        joint_fused_adv_weight=float(args.joint_fused_adv_weight),
        joint_fused_adv_power=float(args.joint_fused_adv_power),
        joint_fused_uncert_weight=float(args.joint_fused_uncert_weight),
        joint_fused_adv_use_abs=bool(args.joint_fused_adv_use_abs),
        joint_fused_kd_w_min=float(args.joint_fused_kd_w_min),
        joint_fused_kd_w_max=float(args.joint_fused_kd_w_max),
    )

    # Stage B test evaluation + checkpoint snapshot (before Stage C joint finetune).
    auc_stage2, preds_stage2, labs_stage2, _ = eval_joint_model(
        reconstructor,
        dual_joint,
        dl_test_joint,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    auc_stage2_val, preds_stage2_val, labs_stage2_val, _ = eval_joint_model(
        reconstructor,
        dual_joint,
        dl_val_joint,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    assert np.array_equal(labs.astype(np.float32), labs_stage2.astype(np.float32))
    assert np.array_equal(labs_val_teacher.astype(np.float32), labs_stage2_val.astype(np.float32))
    stage2_reco_state = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
    stage2_dual_state = {k: v.detach().cpu().clone() for k, v in dual_joint.state_dict().items()}

    # Also evaluate Stage-B best-val_fpr50 checkpoint on test for direct comparison.
    auc_stage2_fprsel = float("nan")
    preds_stage2_fprsel = None
    if stageB_states.get("fpr50", {}).get("dual") is not None and stageB_states.get("fpr50", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageB_states["fpr50"]["reco"])
        dual_joint.load_state_dict(stageB_states["fpr50"]["dual"])
        auc_stage2_fprsel, preds_stage2_fprsel, labs_stage2_fprsel, _ = eval_joint_model(
            reconstructor,
            dual_joint,
            dl_test_joint,
            device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        assert np.array_equal(labs.astype(np.float32), labs_stage2_fprsel.astype(np.float32))

    # Restore Stage-B selected state before entering Stage C.
    reconstructor.load_state_dict(stage2_reco_state)
    dual_joint.load_state_dict(stage2_dual_state)

    print("\n" + "=" * 70)
    print("STEP 4: STAGE C (JOINT FINETUNE)")
    print("=" * 70)
    reconstructor, dual_joint, stageC_metrics, stageC_states = train_joint_dual(
        reconstructor=reconstructor,
        dual_model=dual_joint,
        train_loader=dl_train_joint,
        val_loader=dl_val_joint,
        device=device,
        stage_name="StageC-Joint",
        freeze_reconstructor=False,
        epochs=int(args.stageC_epochs),
        patience=int(args.stageC_patience),
        lr_dual=float(args.stageC_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_epochs=int(cfg["training"]["warmup_epochs"]),
        lambda_reco=float(args.lambda_reco),
        lambda_rank=0.0,
        lambda_cons=float(args.lambda_cons),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
        min_epochs=int(args.stageC_min_epochs),
        select_metric=selection_metric,
        teacher_model=teacher,
        feat_means=means.astype(np.float32),
        feat_stds=stds.astype(np.float32),
        reco_kd_temperature=float(args.stageA_kd_temp),
        reco_lambda_kd=float(args.stageA_lambda_kd),
        reco_lambda_emb=float(args.stageA_lambda_emb),
        reco_lambda_tok=float(args.stageA_lambda_tok),
        reco_lambda_phys=float(args.stageA_lambda_phys),
        reco_lambda_budget_hinge=float(args.stageA_lambda_budget_hinge),
        reco_budget_eps=float(args.stageA_budget_eps),
        reco_budget_weight_floor=float(args.stageA_budget_weight_floor),
        reco_normalize_loss_terms=not bool(args.disable_stageA_loss_normalization),
        reco_loss_norm_ema_decay=float(args.stageA_loss_norm_ema_decay),
        reco_loss_norm_eps=float(args.stageA_loss_norm_eps),
        reco_stagea_fused_adv_weight=float(args.stageA_fused_adv_weight),
        reco_stagea_fused_adv_power=float(args.stageA_fused_adv_power),
        reco_stagea_fused_uncert_weight=float(args.stageA_fused_uncert_weight),
        reco_stagea_fused_adv_use_abs=bool(args.stageA_fused_adv_use_abs),
        reco_stagea_fused_kd_w_min=float(args.stageA_fused_kd_w_min),
        reco_stagea_fused_kd_w_max=float(args.stageA_fused_kd_w_max),
        reco_stagea_lambda_delta_aux=float(args.stageA_lambda_delta_aux),
        joint_fused_kd_lambda=float(args.stageC_joint_fused_kd_lambda),
        joint_fused_kd_temp=float(args.joint_fused_kd_temp),
        joint_fused_adv_weight=float(args.joint_fused_adv_weight),
        joint_fused_adv_power=float(args.joint_fused_adv_power),
        joint_fused_uncert_weight=float(args.joint_fused_uncert_weight),
        joint_fused_adv_use_abs=bool(args.joint_fused_adv_use_abs),
        joint_fused_kd_w_min=float(args.joint_fused_kd_w_min),
        joint_fused_kd_w_max=float(args.joint_fused_kd_w_max),
    )

    auc_joint, preds_joint, labs_joint, _ = eval_joint_model(
        reconstructor,
        dual_joint,
        dl_test_joint,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    assert np.array_equal(labs.astype(np.float32), labs_joint.astype(np.float32))

    # Evaluate Stage-C best-val_fpr50 checkpoint on test too.
    auc_joint_fprsel = float("nan")
    preds_joint_fprsel = None
    if stageC_states.get("fpr50", {}).get("dual") is not None and stageC_states.get("fpr50", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["fpr50"]["reco"])
        dual_joint.load_state_dict(stageC_states["fpr50"]["dual"])
        auc_joint_fprsel, preds_joint_fprsel, labs_joint_fprsel, _ = eval_joint_model(
            reconstructor,
            dual_joint,
            dl_test_joint,
            device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        assert np.array_equal(labs.astype(np.float32), labs_joint_fprsel.astype(np.float32))

    # Restore Stage-C selected state for downstream diagnostics/KD.
    if stageC_states.get("selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["selected"]["reco"])
    if stageC_states.get("selected", {}).get("dual") is not None:
        dual_joint.load_state_dict(stageC_states["selected"]["dual"])

    auc_joint_val, preds_joint_val, labs_joint_val, _ = eval_joint_model(
        reconstructor,
        dual_joint,
        dl_val_joint,
        device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    assert np.array_equal(labs_val_teacher.astype(np.float32), labs_joint_val.astype(np.float32))

    # Build hard reconstructed view for diagnostics.
    print("\n" + "=" * 70)
    print("STEP 5: RECONSTRUCTION DIAGNOSTICS")
    print("=" * 70)
    (
        reco_const,
        reco_mask,
        reco_merge_flag,
        reco_eff_flag,
        created_merge_count,
        created_eff_count,
        pred_budget_total,
        pred_budget_merge,
        pred_budget_eff,
    ) = reconstruct_dataset(
        model=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        max_constits=args.max_constits,
        device=device,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        weight_threshold=float(args.reco_weight_threshold),
        use_budget_topk=not bool(args.reco_disable_budget_topk),
    )

    reco_set_match_diag = compute_reco_set_matching_diagnostics(
        const_off=const_off[test_idx],
        mask_off=masks_off[test_idx],
        const_hlt=hlt_const[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_reco=reco_const[test_idx],
        mask_reco=reco_mask[test_idx],
        max_jets=int(args.diag_match_max_jets),
        seed=int(args.seed if int(args.diag_match_seed) < 0 else args.diag_match_seed),
    )
    print("\nReco set-matching diagnostics (test split sample):")
    print(
        f"  mean_chamfer hlt->offline={reco_set_match_diag['mean_chamfer_hlt_to_offline']:.5f}, "
        f"reco->offline={reco_set_match_diag['mean_chamfer_reco_to_offline']:.5f}, "
        f"frac_reco_better={reco_set_match_diag['frac_reco_better_chamfer']:.3f}"
    )
    print(
        f"  count MAE hlt={reco_set_match_diag['mae_count_hlt_vs_offline']:.3f}, "
        f"reco={reco_set_match_diag['mae_count_reco_vs_offline']:.3f}, "
        f"added recovery mean={reco_set_match_diag['mean_added_recovery_ratio']:.3f}"
    )

    # Jet pT response/resolution diagnostics (test split).
    pt_truth_test = compute_jet_pt(const_off[test_idx], masks_off[test_idx])
    pt_hlt_test = compute_jet_pt(hlt_const[test_idx], hlt_mask[test_idx])
    pt_reco_test = compute_jet_pt(reco_const[test_idx], reco_mask[test_idx])
    pt_edges = build_pt_edges(pt_truth_test, int(args.response_n_bins))
    rr_hlt = jet_response_resolution(pt_truth_test, pt_hlt_test, pt_edges, int(args.response_min_count))
    rr_reco = jet_response_resolution(pt_truth_test, pt_reco_test, pt_edges, int(args.response_min_count))
    plot_response_resolution(
        rr_hlt,
        rr_reco,
        "HLT (reco)",
        "Joint-corrected HLT (reco)",
        save_root / "jet_response_resolution.png",
    )
    rr_hlt_map = {(r["pt_low"], r["pt_high"]): r for r in rr_hlt}
    rr_reco_map = {(r["pt_low"], r["pt_high"]): r for r in rr_reco}
    rr_keys = sorted(set(rr_hlt_map.keys()) & set(rr_reco_map.keys()))
    rr_hlt_common = [rr_hlt_map[k] for k in rr_keys]
    rr_reco_common = [rr_reco_map[k] for k in rr_keys]

    print("\nJet pT response/resolution by truth pT bin (test split):")
    print("  pT_low - pT_high | N | HLT resp | HLT reso | Corrected resp | Corrected reso")
    for h, r in zip(rr_hlt_common, rr_reco_common):
        print(
            f"  {h['pt_low']:.1f} - {h['pt_high']:.1f} | {h['count']:5d} | "
            f"{h['response']:.4f} | {h['resolution']:.4f} | "
            f"{r['response']:.4f} | {r['resolution']:.4f}"
        )

    count_summary = plot_constituent_count_diagnostics(
        save_root=save_root,
        mask_off=masks_off,
        hlt_mask=hlt_mask,
        reco_mask=reco_mask,
        created_merge_count=created_merge_count,
        created_eff_count=created_eff_count,
        hlt_stats=hlt_stats,
    )
    budget_summary = plot_budget_diagnostics(
        save_root=save_root,
        true_merge=budget_merge_true_raw[test_idx],
        true_eff=budget_eff_true_raw[test_idx],
        pred_merge=pred_budget_merge[test_idx],
        pred_eff=pred_budget_eff[test_idx],
    )
    print("\nConstituent-count diagnostics:")
    print(
        f"  Means: offline={count_summary['offline_count_mean']:.3f}, "
        f"hlt={count_summary['hlt_count_mean']:.3f}, reco={count_summary['reco_count_mean']:.3f}"
    )
    print(
        f"  MAE vs offline: hlt={count_summary['hlt_count_mae_vs_offline']:.3f}, "
        f"reco={count_summary['reco_count_mae_vs_offline']:.3f}"
    )
    print("\nBudget diagnostics (test split):")
    print(
        f"  MAE: merge={budget_summary['merge_mae']:.3f}, "
        f"eff={budget_summary['eff_mae']:.3f}, total={budget_summary['total_mae']:.3f}"
    )
    print(
        f"  Bias: merge={budget_summary['merge_bias']:.3f}, "
        f"eff={budget_summary['eff_bias']:.3f}, total={budget_summary['total_bias']:.3f}"
    )

    # Build fixed corrected view tensors for final KD stage and additional diagnostics.
    feat_b_all, mask_b_all = build_corrected_view_numpy(
        reconstructor=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        device=device,
        batch_size=BS,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    soft_view_summary_test = summarize_soft_corrected_view(
        feat_b_all[test_idx],
        mask_b_all[test_idx],
    )

    # Stage D: final KD with reconstructor frozen.
    stageD_metrics: Dict[str, object] = {}
    kd_student = None
    auc_joint_kd = float("nan")
    preds_joint_kd = None
    tpr_j_kd = np.array([], dtype=np.float64)
    fpr_j_kd = np.array([], dtype=np.float64)
    fpr30_joint_kd = float("nan")
    fpr50_joint_kd = float("nan")
    if not bool(args.disable_final_kd):
        print("\n" + "=" * 70)
        print("STEP 6: STAGE D (FINAL KD, FROZEN RECONSTRUCTOR)")
        print("=" * 70)
        kd_train_cfg = json.loads(json.dumps(cfg["training"]))
        kd_cfg = json.loads(json.dumps(cfg["kd"]))
        if int(args.stageD_kd_epochs) > 0:
            kd_train_cfg["epochs"] = int(args.stageD_kd_epochs)
        if int(args.stageD_kd_patience) > 0:
            kd_train_cfg["patience"] = int(args.stageD_kd_patience)
        if float(args.stageD_kd_lr) > 0:
            kd_train_cfg["lr"] = float(args.stageD_kd_lr)

        kd_train_ds = DualViewKDDataset(
            feat_hlt_dual[train_idx], hlt_mask[train_idx],
            feat_b_all[train_idx], mask_b_all[train_idx],
            feat_off_std[train_idx], masks_off[train_idx],
            labels[train_idx],
        )
        kd_val_ds = DualViewKDDataset(
            feat_hlt_dual[val_idx], hlt_mask[val_idx],
            feat_b_all[val_idx], mask_b_all[val_idx],
            feat_off_std[val_idx], masks_off[val_idx],
            labels[val_idx],
        )
        kd_test_ds = DualViewKDDataset(
            feat_hlt_dual[test_idx], hlt_mask[test_idx],
            feat_b_all[test_idx], mask_b_all[test_idx],
            feat_off_std[test_idx], masks_off[test_idx],
            labels[test_idx],
        )
        kd_train_loader = DataLoader(
            kd_train_ds,
            batch_size=BS,
            sampler=_build_weighted_sampler(train_w_train) if bool(args.use_train_weights) else None,
            shuffle=False if bool(args.use_train_weights) else True,
            drop_last=True,
        )
        kd_val_loader = DataLoader(kd_val_ds, batch_size=BS, shuffle=False)
        kd_test_loader = DataLoader(kd_test_ds, batch_size=BS, shuffle=False)

        kd_student = DualViewCrossAttnClassifier(input_dim_a=dual_input_dim_a, input_dim_b=dual_input_dim_b, **cfg["model"]).to(device)
        kd_student.load_state_dict(dual_joint.state_dict())
        kd_student = train_dual_kd_student(
            student=kd_student,
            teacher=teacher,
            kd_train_loader=kd_train_loader,
            kd_val_loader=kd_val_loader,
            device=device,
            train_cfg=kd_train_cfg,
            kd_cfg=kd_cfg,
            name="StageD-Joint+KD",
            run_self_train=bool(kd_cfg.get("self_train", True)),
        )
        auc_joint_kd, preds_joint_kd, labs_joint_kd = eval_classifier_dual(kd_student, kd_test_loader, device)
        assert np.array_equal(labs.astype(np.float32), labs_joint_kd.astype(np.float32))
        fpr_j_kd, tpr_j_kd, _ = roc_curve(labs, preds_joint_kd)
        fpr30_joint_kd = fpr_at_target_tpr(fpr_j_kd, tpr_j_kd, 0.30)
        fpr50_joint_kd = fpr_at_target_tpr(fpr_j_kd, tpr_j_kd, 0.50)
        stageD_metrics = {
            "enabled": 1.0,
            "train_cfg": kd_train_cfg,
            "kd_cfg": kd_cfg,
        }
    else:
        stageD_metrics = {"enabled": 0.0}

    # Final metrics
    fpr_ro = np.array([], dtype=np.float64)
    tpr_ro = np.array([], dtype=np.float64)
    fpr30_reco_only = float("nan")
    fpr50_reco_only = float("nan")
    if preds_reco_only is not None:
        fpr_ro, tpr_ro, _ = roc_curve(labs, preds_reco_only)
        fpr30_reco_only = fpr_at_target_tpr(fpr_ro, tpr_ro, 0.30)
        fpr50_reco_only = fpr_at_target_tpr(fpr_ro, tpr_ro, 0.50)

    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_s2, tpr_s2, _ = roc_curve(labs, preds_stage2)
    fpr_j, tpr_j, _ = roc_curve(labs, preds_joint)
    if preds_stage2_fprsel is not None:
        fpr_s2_fprsel, tpr_s2_fprsel, _ = roc_curve(labs, preds_stage2_fprsel)
    else:
        fpr_s2_fprsel, tpr_s2_fprsel = np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if preds_joint_fprsel is not None:
        fpr_j_fprsel, tpr_j_fprsel, _ = roc_curve(labs, preds_joint_fprsel)
    else:
        fpr_j_fprsel, tpr_j_fprsel = np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.30)
    fpr30_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.30)
    fpr30_stage2 = fpr_at_target_tpr(fpr_s2, tpr_s2, 0.30)
    fpr30_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.30)
    fpr30_stage2_fprsel = fpr_at_target_tpr(fpr_s2_fprsel, tpr_s2_fprsel, 0.30) if preds_stage2_fprsel is not None else float("nan")
    fpr30_joint_fprsel = fpr_at_target_tpr(fpr_j_fprsel, tpr_j_fprsel, 0.30) if preds_joint_fprsel is not None else float("nan")
    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_baseline = fpr_at_target_tpr(fpr_b, tpr_b, 0.50)
    fpr50_stage2 = fpr_at_target_tpr(fpr_s2, tpr_s2, 0.50)
    fpr50_joint = fpr_at_target_tpr(fpr_j, tpr_j, 0.50)
    fpr50_stage2_fprsel = fpr_at_target_tpr(fpr_s2_fprsel, tpr_s2_fprsel, 0.50) if preds_stage2_fprsel is not None else float("nan")
    fpr50_joint_fprsel = fpr_at_target_tpr(fpr_j_fprsel, tpr_j_fprsel, 0.50) if preds_joint_fprsel is not None else float("nan")

    overlap_models = {
        "teacher": preds_teacher,
        "hlt": preds_baseline,
        "stage2": preds_stage2,
        "joint": preds_joint,
    }
    overlap_report = build_overlap_report_at_tpr(
        labels=labs.astype(np.float32),
        model_preds=overlap_models,
        target_tpr=float(args.report_target_tpr),
    )
    best_combo_hlt_joint_test_posthoc = search_best_weighted_combo_at_tpr(
        labels=labs.astype(np.float32),
        preds_a=preds_baseline,
        preds_b=preds_joint,
        name_a="hlt",
        name_b="joint",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )
    best_combo_hlt_stage2_test_posthoc = search_best_weighted_combo_at_tpr(
        labels=labs.astype(np.float32),
        preds_a=preds_baseline,
        preds_b=preds_stage2,
        name_a="hlt",
        name_b="stage2",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )
    best_combo_hlt_joint_valsel = select_weighted_combo_on_val_and_eval_test(
        labels_val=labs_val_teacher.astype(np.float32),
        preds_a_val=preds_baseline_val,
        preds_b_val=preds_joint_val,
        labels_test=labs.astype(np.float32),
        preds_a_test=preds_baseline,
        preds_b_test=preds_joint,
        name_a="hlt",
        name_b="joint",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )
    best_combo_hlt_stage2_valsel = select_weighted_combo_on_val_and_eval_test(
        labels_val=labs_val_teacher.astype(np.float32),
        preds_a_val=preds_baseline_val,
        preds_b_val=preds_stage2_val,
        labels_test=labs.astype(np.float32),
        preds_a_test=preds_baseline,
        preds_b_test=preds_stage2,
        name_a="hlt",
        name_b="stage2",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    if preds_reco_only is not None:
        print(f"RecoOnly (StageA probe) AUC: {auc_reco_only:.4f}")
    print(f"Stage2 (PreJoint) AUC: {auc_stage2:.4f}")
    if preds_stage2_fprsel is not None:
        print(f"Stage2 (BestValFPR50) AUC: {auc_stage2_fprsel:.4f}")
    print(f"Joint Dual-View  AUC: {auc_joint:.4f}")
    if preds_joint_fprsel is not None:
        print(f"Joint Dual-View (BestValFPR50) AUC: {auc_joint_fprsel:.4f}")
    if preds_joint_kd is not None:
        print(f"Joint Dual-View+KD AUC: {auc_joint_kd:.4f}")
    print()
    print(
        f"FPR@30 Teacher/Baseline/Stage2/Joint: "
        f"{fpr30_teacher:.6f} / {fpr30_baseline:.6f} / {fpr30_stage2:.6f} / {fpr30_joint:.6f}"
    )
    if preds_reco_only is not None:
        print(f"FPR@30 RecoOnly: {fpr30_reco_only:.6f}")
    if preds_stage2_fprsel is not None or preds_joint_fprsel is not None:
        print(
            f"FPR@30 Stage2BestFPR / JointBestFPR: "
            f"{fpr30_stage2_fprsel:.6f} / {fpr30_joint_fprsel:.6f}"
        )
    print(
        f"FPR@50 Teacher/Baseline/Stage2/Joint: "
        f"{fpr50_teacher:.6f} / {fpr50_baseline:.6f} / {fpr50_stage2:.6f} / {fpr50_joint:.6f}"
    )
    if preds_reco_only is not None:
        print(f"FPR@50 RecoOnly: {fpr50_reco_only:.6f}")
    if preds_stage2_fprsel is not None or preds_joint_fprsel is not None:
        print(
            f"FPR@50 Stage2BestFPR / JointBestFPR: "
            f"{fpr50_stage2_fprsel:.6f} / {fpr50_joint_fprsel:.6f}"
        )
    if preds_joint_kd is not None:
        print(f"FPR@30 Joint+KD: {fpr30_joint_kd:.6f}")
        print(f"FPR@50 Joint+KD: {fpr50_joint_kd:.6f}")

    pair_hj = overlap_report["pairs"].get("hlt__joint", {})
    pair_tj = overlap_report["pairs"].get("teacher__joint", {})
    print(
        f"TP overlap @TPR={float(args.report_target_tpr):.2f} (HLT vs Joint): "
        f"{int(pair_hj.get('overlap_tp_count', 0))} shared TP | "
        f"overlap frac of HLT TP={float(pair_hj.get('overlap_tp_frac_of_a_tp', float('nan'))):.3f}, "
        f"of Joint TP={float(pair_hj.get('overlap_tp_frac_of_b_tp', float('nan'))):.3f}"
    )
    print(
        f"TP overlap @TPR={float(args.report_target_tpr):.2f} (Teacher vs Joint): "
        f"{int(pair_tj.get('overlap_tp_count', 0))} shared TP | "
        f"overlap frac of Teacher TP={float(pair_tj.get('overlap_tp_frac_of_a_tp', float('nan'))):.3f}, "
        f"of Joint TP={float(pair_tj.get('overlap_tp_frac_of_b_tp', float('nan'))):.3f}"
    )
    print(
        f"Best weighted combo @TPR={float(args.report_target_tpr):.2f} (HLT+Joint, VAL-selected -> TEST): "
        f"w_hlt={best_combo_hlt_joint_valsel['test_eval']['w_a']:.3f}, "
        f"w_joint={best_combo_hlt_joint_valsel['test_eval']['w_b']:.3f}, "
        f"FPR_test={best_combo_hlt_joint_valsel['test_eval']['fpr']:.6f}"
    )
    print(
        f"Best weighted combo @TPR={float(args.report_target_tpr):.2f} (HLT+Joint, TEST post-hoc): "
        f"w_hlt={best_combo_hlt_joint_test_posthoc['w_a']:.3f}, "
        f"w_joint={best_combo_hlt_joint_test_posthoc['w_b']:.3f}, "
        f"FPR={best_combo_hlt_joint_test_posthoc['fpr']:.6f}"
    )

    # Save val/test score arrays for downstream fusion studies.
    fusion_dump = {
        "labels_val": labs_val_teacher.astype(np.float32),
        "labels_test": labs.astype(np.float32),
        "preds_teacher_val": preds_teacher_val.astype(np.float64),
        "preds_teacher_test": preds_teacher.astype(np.float64),
        "preds_hlt_val": preds_baseline_val.astype(np.float64),
        "preds_hlt_test": preds_baseline.astype(np.float64),
        "preds_stage2_val": preds_stage2_val.astype(np.float64),
        "preds_stage2_test": preds_stage2.astype(np.float64),
        "preds_joint_val": preds_joint_val.astype(np.float64),
        "preds_joint_test": preds_joint.astype(np.float64),
        "hlt_nconst_test": hlt_mask[test_idx].sum(axis=1).astype(np.float32),
        "target_tpr": np.array(float(args.report_target_tpr), dtype=np.float64),
    }
    if preds_reco_only_val is not None and preds_reco_only is not None:
        fusion_dump["preds_reco_only_val"] = preds_reco_only_val.astype(np.float64)
        fusion_dump["preds_reco_only_test"] = preds_reco_only.astype(np.float64)
    np.savez_compressed(save_root / "fusion_scores_val_test.npz", **fusion_dump)
    print(f"Saved fusion score arrays to: {save_root / 'fusion_scores_val_test.npz'}")

    plot_lines = [
        (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
        (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
        (tpr_s2, fpr_s2, "-.", f"Stage2 PreJoint (AUC={auc_stage2:.3f})", "darkorange"),
        (tpr_j, fpr_j, "-.", f"Joint Dual (AUC={auc_joint:.3f})", "darkslateblue"),
    ]
    if preds_reco_only is not None:
        plot_lines.append((tpr_ro, fpr_ro, "-.", f"RecoOnly (AUC={auc_reco_only:.3f})", "teal"))
    if preds_stage2_fprsel is not None:
        plot_lines.append(
            (tpr_s2_fprsel, fpr_s2_fprsel, ":", f"Stage2 BestValFPR (AUC={auc_stage2_fprsel:.3f})", "peru")
        )
    if preds_joint_fprsel is not None:
        plot_lines.append(
            (tpr_j_fprsel, fpr_j_fprsel, "--", f"Joint BestValFPR (AUC={auc_joint_fprsel:.3f})", "indigo")
        )
    if preds_joint_kd is not None:
        plot_lines.append((tpr_j_kd, fpr_j_kd, ":", f"Joint Dual+KD (AUC={auc_joint_kd:.3f})", "darkgreen"))
    plot_roc(
        plot_lines,
        save_root / "results_teacher_baseline_joint.png",
        min_fpr=1e-4,
    )

    def rr_field(records, key):
        return np.array([r[key] for r in records], dtype=np.float64)

    np.savez(
        save_root / "results.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_reco_only=auc_reco_only,
        auc_stage2=auc_stage2,
        auc_stage2_fprsel=auc_stage2_fprsel,
        auc_joint=auc_joint,
        auc_joint_fprsel=auc_joint_fprsel,
        auc_joint_kd=auc_joint_kd,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_stage2=fpr_s2,
        tpr_stage2=tpr_s2,
        fpr_reco_only=fpr_ro,
        tpr_reco_only=tpr_ro,
        fpr_stage2_fprsel=fpr_s2_fprsel,
        tpr_stage2_fprsel=tpr_s2_fprsel,
        fpr_joint=fpr_j,
        tpr_joint=tpr_j,
        fpr_joint_fprsel=fpr_j_fprsel,
        tpr_joint_fprsel=tpr_j_fprsel,
        fpr_joint_kd=fpr_j_kd,
        tpr_joint_kd=tpr_j_kd,
        fpr30_teacher=fpr30_teacher,
        fpr30_baseline=fpr30_baseline,
        fpr30_reco_only=fpr30_reco_only,
        fpr30_stage2=fpr30_stage2,
        fpr30_stage2_fprsel=fpr30_stage2_fprsel,
        fpr30_joint=fpr30_joint,
        fpr30_joint_fprsel=fpr30_joint_fprsel,
        fpr30_joint_kd=fpr30_joint_kd,
        fpr50_teacher=fpr50_teacher,
        fpr50_baseline=fpr50_baseline,
        fpr50_reco_only=fpr50_reco_only,
        fpr50_stage2=fpr50_stage2,
        fpr50_stage2_fprsel=fpr50_stage2_fprsel,
        fpr50_joint=fpr50_joint,
        fpr50_joint_fprsel=fpr50_joint_fprsel,
        fpr50_joint_kd=fpr50_joint_kd,
        jet_response_pt_low=rr_field(rr_hlt_common, "pt_low"),
        jet_response_pt_high=rr_field(rr_hlt_common, "pt_high"),
        jet_response_count=rr_field(rr_hlt_common, "count"),
        jet_response_hlt_mean=rr_field(rr_hlt_common, "response"),
        jet_response_hlt_std=rr_field(rr_hlt_common, "resolution"),
        jet_response_corrected_mean=rr_field(rr_reco_common, "response"),
        jet_response_corrected_std=rr_field(rr_reco_common, "resolution"),
        rho=float(rho),
    )

    with open(save_root / "constituent_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(count_summary, f, indent=2)
    with open(save_root / "budget_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(budget_summary, f, indent=2)
    with open(save_root / "soft_corrected_view_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(soft_view_summary_test, f, indent=2)
    with open(save_root / "reco_set_matching_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(reco_set_match_diag, f, indent=2)
    with open(save_root / "overlap_report_tpr50.json", "w", encoding="utf-8") as f:
        json.dump(overlap_report, f, indent=2)
    with open(save_root / "best_combo_hlt_joint_tpr50.json", "w", encoding="utf-8") as f:
        json.dump({"hlt_joint_val_selected_eval_test": best_combo_hlt_joint_valsel, "hlt_stage2_val_selected_eval_test": best_combo_hlt_stage2_valsel, "hlt_joint_test_posthoc": best_combo_hlt_joint_test_posthoc, "hlt_stage2_test_posthoc": best_combo_hlt_stage2_test_posthoc}, f, indent=2)
    with open(save_root / "jet_regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(jet_reg_metrics, f, indent=2)

    with open(save_root / "joint_stage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": {
                    "mode": "nopriv_rhosplit_splitagain",
                    "rho": float(rho),
                    "split_again": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in SPLIT_AGAIN_CFG.items()},
                    "mean_true_added_raw": float(true_added_raw.mean()),
                    "mean_target_merge": float(budget_merge_true.mean()),
                    "mean_target_eff": float(budget_eff_true.mean()),
                },
                "jet_regressor": jet_reg_metrics,
                "stageA_reconstructor": reco_val_metrics,
                "stageA_reco_only_probe": reco_only_metrics,
                "stageB_joint": stageB_metrics,
                "stageC_joint": stageC_metrics,
                "stageD_kd": stageD_metrics,
                "overlap_report_tpr": overlap_report,
                "best_combo_hlt_joint_val_selected_eval_test": best_combo_hlt_joint_valsel,
                "best_combo_hlt_stage2_val_selected_eval_test": best_combo_hlt_stage2_valsel,
                "best_combo_hlt_joint_test_posthoc": best_combo_hlt_joint_test_posthoc,
                "best_combo_hlt_stage2_test_posthoc": best_combo_hlt_stage2_test_posthoc,
                "reco_set_matching_diagnostics": reco_set_match_diag,
                "test_stage2": {
                    "auc_stage2": float(auc_stage2),
                    "auc_stage2_fprsel": float(auc_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr30_stage2": float(fpr30_stage2),
                    "fpr30_stage2_fprsel": float(fpr30_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr50_stage2": float(fpr50_stage2),
                    "fpr50_stage2_fprsel": float(fpr50_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                },
                "test": {
                    "auc_teacher": float(auc_teacher),
                    "auc_baseline": float(auc_baseline),
                    "auc_reco_only": float(auc_reco_only) if preds_reco_only is not None else None,
                    "auc_stage2": float(auc_stage2),
                    "auc_stage2_fprsel": float(auc_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "auc_joint": float(auc_joint),
                    "auc_joint_fprsel": float(auc_joint_fprsel) if preds_joint_fprsel is not None else None,
                    "auc_joint_kd": float(auc_joint_kd) if preds_joint_kd is not None else None,
                    "fpr30_teacher": float(fpr30_teacher),
                    "fpr30_baseline": float(fpr30_baseline),
                    "fpr30_reco_only": float(fpr30_reco_only) if preds_reco_only is not None else None,
                    "fpr30_stage2": float(fpr30_stage2),
                    "fpr30_stage2_fprsel": float(fpr30_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr30_joint": float(fpr30_joint),
                    "fpr30_joint_fprsel": float(fpr30_joint_fprsel) if preds_joint_fprsel is not None else None,
                    "fpr30_joint_kd": float(fpr30_joint_kd) if preds_joint_kd is not None else None,
                    "fpr50_teacher": float(fpr50_teacher),
                    "fpr50_baseline": float(fpr50_baseline),
                    "fpr50_reco_only": float(fpr50_reco_only) if preds_reco_only is not None else None,
                    "fpr50_stage2": float(fpr50_stage2),
                    "fpr50_stage2_fprsel": float(fpr50_stage2_fprsel) if preds_stage2_fprsel is not None else None,
                    "fpr50_joint": float(fpr50_joint),
                    "fpr50_joint_fprsel": float(fpr50_joint_fprsel) if preds_joint_fprsel is not None else None,
                    "fpr50_joint_kd": float(fpr50_joint_kd) if preds_joint_kd is not None else None,
                },
            },
            f,
            indent=2,
        )

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg["hlt_effects"], "stats": hlt_stats}, f, indent=2)

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": auc_teacher}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": auc_baseline}, save_root / "baseline.pt")
        if reco_only_model is not None:
            torch.save(
                {
                    "model": reco_only_model.state_dict(),
                    "auc": float(auc_reco_only),
                    "fpr30": float(fpr30_reco_only),
                    "fpr50": float(fpr50_reco_only),
                },
                save_root / "reco_only_stagea_probe.pt",
            )
        if jet_regressor is not None:
            torch.save({"model": jet_regressor.state_dict(), "metrics": jet_reg_metrics}, save_root / "jet_regressor.pt")
        torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor.pt")
        torch.save({"model": stage2_reco_state, "val": reco_val_metrics}, save_root / "offline_reconstructor_stage2.pt")
        torch.save(
            {
                "model": stage2_dual_state,
                "auc": float(auc_stage2),
                "fpr30": float(fpr30_stage2),
                "fpr50": float(fpr50_stage2),
            },
            save_root / "dual_joint_stage2.pt",
        )
        if stageB_states.get("fpr50", {}).get("reco") is not None:
            torch.save({"model": stageB_states["fpr50"]["reco"], "val": reco_val_metrics}, save_root / "offline_reconstructor_stage2_bestfpr50.pt")
        if stageB_states.get("fpr50", {}).get("dual") is not None:
            torch.save(
                {
                    "model": stageB_states["fpr50"]["dual"],
                    "auc": float(auc_stage2_fprsel) if preds_stage2_fprsel is not None else float("nan"),
                    "fpr30": float(fpr30_stage2_fprsel) if preds_stage2_fprsel is not None else float("nan"),
                    "fpr50": float(fpr50_stage2_fprsel) if preds_stage2_fprsel is not None else float("nan"),
                },
                save_root / "dual_joint_stage2_bestfpr50.pt",
            )
        torch.save({"model": dual_joint.state_dict(), "auc": auc_joint}, save_root / "dual_joint.pt")
        if stageC_states.get("fpr50", {}).get("reco") is not None:
            torch.save({"model": stageC_states["fpr50"]["reco"], "val": reco_val_metrics}, save_root / "offline_reconstructor_bestfpr50.pt")
        if stageC_states.get("fpr50", {}).get("dual") is not None:
            torch.save(
                {
                    "model": stageC_states["fpr50"]["dual"],
                    "auc": float(auc_joint_fprsel) if preds_joint_fprsel is not None else float("nan"),
                    "fpr30": float(fpr30_joint_fprsel) if preds_joint_fprsel is not None else float("nan"),
                    "fpr50": float(fpr50_joint_fprsel) if preds_joint_fprsel is not None else float("nan"),
                },
                save_root / "dual_joint_bestfpr50.pt",
            )
        if kd_student is not None:
            torch.save({"model": kd_student.state_dict(), "auc": auc_joint_kd}, save_root / "dual_joint_kd.pt")

    print(f"\nSaved joint results to: {save_root}")


if __name__ == "__main__":
    main()

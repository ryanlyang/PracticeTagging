#!/usr/bin/env python3
"""
Fast Stage-C finetuning starting from a saved Stage2 checkpoint.

This script re-creates data/splits deterministically, loads:
  - offline_reconstructor_stage2.pt
  - dual_joint_stage2.pt
from a previous run folder, then runs Stage C only.
If available, it consumes run_dir/data_setup.json and run_dir/data_splits.npz
to reproduce the exact original data setup and split indices.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc as joint
import offline_reconstructor_no_gt_local30kv2 as reco_base
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as LOCAL30K_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
    fpr_at_target_tpr,
)
from unmerge_correct_hlt import (
    RANDOM_SEED,
    DualViewCrossAttnClassifier,
    JetDataset,
    ParticleTransformer,
    compute_features,
    eval_classifier,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
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


def _load_checkpoint_state(path: Path, device: torch.device, tag: str) -> Dict[str, torch.Tensor]:
    """
    Load checkpoint and return a plain state_dict.

    Supports:
    - plain state_dict files
    - wrapped dicts like {"model": state_dict, ...}
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    keys = list(ckpt.keys())[:8] if isinstance(ckpt, dict) else [type(ckpt).__name__]
    raise RuntimeError(
        f"Unsupported checkpoint format for {tag}: {path}. "
        f"Top-level keys/type preview: {keys}"
    )


def load_cfg_from_run(run_dir: Path) -> Dict:
    cfg = joint._deepcopy_config()
    hlt_stats_path = run_dir / "hlt_stats.json"
    if hlt_stats_path.exists():
        h = json.load(open(hlt_stats_path, "r", encoding="utf-8"))
        hcfg = h.get("config", {})
        for k, v in hcfg.items():
            if k in cfg["hlt_effects"]:
                cfg["hlt_effects"][k] = v
    return cfg


def load_saved_data_setup(run_dir: Path) -> Dict:
    path = run_dir / "data_setup.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            out = json.load(f)
        return out if isinstance(out, dict) else {}
    except Exception as e:
        print(f"Warning: failed to read saved data setup {path}: {e}")
        return {}


def load_saved_splits(run_dir: Path) -> Dict[str, np.ndarray]:
    path = run_dir / "data_splits.npz"
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:
        print(f"Warning: failed to read saved splits {path}: {e}")
        return {}


def maybe_build_jetreg_features(
    run_dir: Path,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    const_off: np.ndarray,
    masks_off: np.ndarray,
    hlt_const: np.ndarray,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> np.ndarray:
    jet_reg_ckpt = run_dir / "jet_regressor.pt"
    jet_meta_path = run_dir / "jet_regression_metrics.json"

    enabled_in_run = False
    if jet_meta_path.exists():
        try:
            enabled_in_run = bool(json.load(open(jet_meta_path, "r", encoding="utf-8")).get("enabled", False))
        except Exception:
            enabled_in_run = False

    if (not jet_reg_ckpt.exists()) or (not enabled_in_run):
        return feat_hlt_std.astype(np.float32, copy=True)

    target_off, target_hlt_ref, _ = joint.compute_jet_regression_targets(
        const_off=const_off,
        mask_off=masks_off,
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
    )
    target_dim = int(target_off.shape[1])

    model = joint.JetLevelRegressor(
        input_dim=7,
        output_dim=target_dim,
        embed_dim=int(args.jet_reg_embed_dim),
        num_heads=int(args.jet_reg_num_heads),
        num_layers=int(args.jet_reg_num_layers),
        ff_dim=int(args.jet_reg_ff_dim),
        dropout=float(args.jet_reg_dropout),
    ).to(device)
    model.load_state_dict(torch.load(jet_reg_ckpt, map_location=device))

    pred_log_all = joint.predict_jet_regressor(
        model=model,
        feat=feat_hlt_std,
        mask=hlt_mask,
        device=device,
        batch_size=int(batch_size),
    )
    delta_vs_hlt = pred_log_all - target_hlt_ref
    extra_global = np.concatenate([pred_log_all, delta_vs_hlt], axis=-1).astype(np.float32)
    extra_global = np.repeat(extra_global[:, None, :], feat_hlt_std.shape[1], axis=1)
    feat_hlt_dual = np.concatenate([feat_hlt_std, extra_global], axis=-1).astype(np.float32)
    feat_hlt_dual[~hlt_mask] = 0.0
    return feat_hlt_dual


def maybe_eval_single_view_checkpoint(
    ckpt_path: Path,
    tag: str,
    feat_test: np.ndarray,
    mask_test: np.ndarray,
    labels_test: np.ndarray,
    model_cfg: Dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Load a single-view classifier checkpoint (teacher/baseline), evaluate on test split,
    and return AUC/FPR@30/FPR@50 metrics. Returns {} if ckpt is missing.
    """
    if not ckpt_path.exists():
        print(f"Warning: {tag} checkpoint not found: {ckpt_path}")
        return {}

    model = ParticleTransformer(input_dim=7, **model_cfg).to(device)
    state = _load_checkpoint_state(ckpt_path, device, tag)
    model.load_state_dict(state)

    ds = JetDataset(feat_test, mask_test, labels_test)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    auc, preds, labs = eval_classifier(model, dl, device)
    fpr, tpr, _ = roc_curve(labs, preds)
    return {
        "auc": float(auc),
        "fpr30": float(fpr_at_target_tpr(fpr, tpr, 0.30)),
        "fpr50": float(fpr_at_target_tpr(fpr, tpr, 0.50)),
    }


def predict_single_view_scores(
    model: nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = JetDataset(feat, mask, labels)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    _, preds, labs = eval_classifier(model, dl, device)
    return preds.astype(np.float32), labs.astype(np.float32)


def prob_threshold_at_target_tpr(pos_probs: np.ndarray, target_tpr: float) -> float:
    if pos_probs.size == 0:
        return 0.5
    q = float(np.clip(1.0 - float(target_tpr), 0.0, 1.0))
    return float(np.quantile(pos_probs, q))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _weighted_batch_mean(vec: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return vec.mean()
    denom = sample_weight.sum().clamp(min=1e-6)
    return (vec * sample_weight).sum() / denom


def compute_reconstruction_losses_weighted(
    out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    loss_cfg: Dict,
    sample_weight: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Stage-C helper mirroring offline_reconstructor_no_gt_local30kv2.compute_reconstruction_losses,
    but with optional per-jet sample weighting on the final batch reduction.
    """
    eps = 1e-8

    if sample_weight is not None:
        if sample_weight.dim() != 1 or sample_weight.shape[0] != const_hlt.shape[0]:
            raise ValueError(
                f"sample_weight shape mismatch for reconstruction weighting: "
                f"{tuple(sample_weight.shape)} vs batch {const_hlt.shape[0]}"
            )
        sw = sample_weight.float().clamp(min=0.0)
    else:
        sw = None

    pred = out["cand_tokens"]
    w = out["cand_weights"].clamp(0.0, 1.0)

    cost = reco_base._token_cost_matrix(pred, const_off)
    valid_tgt = mask_off.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))

    pred_to_tgt = cost.min(dim=2).values
    loss_pred_to_tgt = (w * pred_to_tgt).sum(dim=1) / (w.sum(dim=1) + eps)

    penalty = float(loss_cfg["unselected_penalty"]) * (1.0 - w).unsqueeze(2)
    tgt_to_pred = (cost + penalty).min(dim=1).values
    tgt_mask_f = mask_off.float()
    loss_tgt_to_pred = (tgt_to_pred * tgt_mask_f).sum(dim=1) / (tgt_mask_f.sum(dim=1) + eps)
    loss_set_vec = loss_pred_to_tgt + loss_tgt_to_pred

    pred_px, pred_py, pred_pz, pred_E = reco_base._weighted_fourvec_sums(pred, w)
    true_px, true_py, true_pz, true_E = reco_base._weighted_fourvec_sums(const_off, mask_off.float())

    norm = true_px.abs() + true_py.abs() + true_pz.abs() + true_E.abs() + 1.0
    loss_phys_vec = (
        (pred_px - true_px).abs()
        + (pred_py - true_py).abs()
        + (pred_pz - true_pz).abs()
        + (pred_E - true_E).abs()
    ) / norm

    pred_pt = torch.sqrt(pred_px.pow(2) + pred_py.pow(2) + eps)
    true_pt = torch.sqrt(true_px.pow(2) + true_py.pow(2) + eps)
    pt_ratio = pred_pt / (true_pt + eps)
    loss_pt_ratio_vec = F.smooth_l1_loss(pt_ratio, torch.ones_like(pt_ratio), reduction="none")

    e_ratio = pred_E / (true_E + eps)
    loss_e_ratio_vec = F.smooth_l1_loss(e_ratio, torch.ones_like(e_ratio), reduction="none")

    true_count = mask_off.float().sum(dim=1)
    hlt_count = mask_hlt.float().sum(dim=1)
    true_added = (true_count - hlt_count).clamp(min=0.0)

    pred_count = w.sum(dim=1)
    pred_added = out["child_weight"].sum(dim=1) + out["gen_weight"].sum(dim=1)
    pred_added_merge = out["child_weight"].sum(dim=1)
    pred_added_eff = out["gen_weight"].sum(dim=1)

    budget_total = out["budget_total"]
    budget_merge = out["budget_merge"]
    budget_eff = out["budget_eff"]
    budget_true_total = budget_merge_true + budget_eff_true

    loss_budget_vec = (
        F.smooth_l1_loss(pred_count, true_count, reduction="none")
        + F.smooth_l1_loss(budget_total, true_count, reduction="none")
        + F.smooth_l1_loss(pred_added, true_added, reduction="none")
        + F.smooth_l1_loss(budget_merge + budget_eff, true_added, reduction="none")
        + F.smooth_l1_loss(budget_merge, budget_merge_true, reduction="none")
        + F.smooth_l1_loss(budget_eff, budget_eff_true, reduction="none")
        + F.smooth_l1_loss(pred_added_merge, budget_merge_true, reduction="none")
        + F.smooth_l1_loss(pred_added_eff, budget_eff_true, reduction="none")
        + F.smooth_l1_loss(budget_merge + budget_eff, budget_true_total, reduction="none")
    )

    loss_sparse_vec = out["child_weight"].mean(dim=1) + out["gen_weight"].mean(dim=1)

    split_delta = out["split_delta"]
    split_step = torch.sqrt(split_delta[..., 1].pow(2) + split_delta[..., 2].pow(2) + eps)
    split_w = out["child_weight"].reshape(split_step.shape)
    loss_local_split_vec = (split_w * split_step).sum(dim=(1, 2)) / (split_w.sum(dim=(1, 2)) + eps)

    gen_tokens = out["gen_tokens"]
    gen_w = out["gen_weight"]
    h_eta = const_hlt[:, :, 1]
    h_phi = const_hlt[:, :, 2]
    g_eta = gen_tokens[:, :, 1].unsqueeze(2)
    g_phi = gen_tokens[:, :, 2].unsqueeze(2)
    d_eta = g_eta - h_eta.unsqueeze(1)
    d_phi = torch.atan2(torch.sin(g_phi - h_phi.unsqueeze(1)), torch.cos(g_phi - h_phi.unsqueeze(1)))
    dR = torch.sqrt(d_eta.pow(2) + d_phi.pow(2) + eps)
    dR = torch.where(mask_hlt.unsqueeze(1), dR, torch.full_like(dR, 1e4))
    nearest = dR.min(dim=2).values
    excess = F.relu(nearest - float(loss_cfg["gen_local_radius"]))
    loss_local_gen_vec = (gen_w * excess).sum(dim=1) / (gen_w.sum(dim=1) + eps)
    loss_local_vec = loss_local_split_vec + loss_local_gen_vec

    total_vec = (
        float(loss_cfg["w_set"]) * loss_set_vec
        + float(loss_cfg["w_phys"]) * loss_phys_vec
        + float(loss_cfg["w_pt_ratio"]) * loss_pt_ratio_vec
        + float(loss_cfg["w_e_ratio"]) * loss_e_ratio_vec
        + float(loss_cfg["w_budget"]) * loss_budget_vec
        + float(loss_cfg["w_sparse"]) * loss_sparse_vec
        + float(loss_cfg["w_local"]) * loss_local_vec
    )

    return {
        "total": _weighted_batch_mean(total_vec, sw),
        "set": _weighted_batch_mean(loss_set_vec, sw),
        "phys": _weighted_batch_mean(loss_phys_vec, sw),
        "pt_ratio": _weighted_batch_mean(loss_pt_ratio_vec, sw),
        "e_ratio": _weighted_batch_mean(loss_e_ratio_vec, sw),
        "budget": _weighted_batch_mean(loss_budget_vec, sw),
        "sparse": _weighted_batch_mean(loss_sparse_vec, sw),
        "local": _weighted_batch_mean(loss_local_vec, sw),
    }


def build_discrepancy_weights(
    y_train: np.ndarray,
    p_teacher_train: np.ndarray,
    p_baseline_train: np.ndarray,
    p_teacher_val: np.ndarray,
    p_baseline_val: np.ndarray,
    y_val: np.ndarray,
    target_tpr: float,
    tau: float,
    lambda_disc: float,
    max_mult: float,
    include_pos: bool,
    pos_scale: float,
    normalize_mean_one: bool,
    teacher_conf_min: float,
    correctness_tau: float,
    use_teacher_hard_correct_gate: bool,
    use_teacher_conf_gate: bool,
    use_teacher_better_gate: bool,
    weight_mode: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    tau = max(float(tau), 1e-6)
    correctness_tau = max(float(correctness_tau), 1e-6)
    target_tpr = float(np.clip(target_tpr, 0.0, 1.0))
    max_mult = max(float(max_mult), 1.0)
    lambda_disc = max(float(lambda_disc), 0.0)
    teacher_conf_min = float(np.clip(teacher_conf_min, 0.0, 1.0))

    t_off = prob_threshold_at_target_tpr(p_teacher_val[y_val == 1], target_tpr)
    t_hlt = prob_threshold_at_target_tpr(p_baseline_val[y_val == 1], target_tpr)

    y_train = y_train.astype(np.int64)
    p_teacher_true = np.where(y_train == 1, p_teacher_train, 1.0 - p_teacher_train).astype(np.float32)
    p_hlt_true = np.where(y_train == 1, p_baseline_train, 1.0 - p_baseline_train).astype(np.float32)

    teacher_hard_correct = ((p_teacher_train >= 0.5).astype(np.int64) == y_train).astype(np.float32)
    gate = np.ones_like(p_teacher_true, dtype=np.float32)
    if bool(use_teacher_hard_correct_gate):
        gate *= teacher_hard_correct
    if bool(use_teacher_conf_gate):
        gate *= _sigmoid_np((p_teacher_true - teacher_conf_min) / correctness_tau).astype(np.float32)
    if bool(use_teacher_better_gate):
        gate *= _sigmoid_np((p_teacher_true - p_hlt_true) / correctness_tau).astype(np.float32)

    mode = str(weight_mode).strip().lower()
    if mode not in ("tail_disagreement", "smooth_delta"):
        raise ValueError(f"Unsupported --disc_weight_mode: {weight_mode}")

    if mode == "smooth_delta":
        # Smooth discrepancy: emphasize jets where HLT is more top-like than teacher (for y=0),
        # gated by teacher-right confidence checks to avoid noisy disagreements.
        r_neg = (
            np.maximum(p_baseline_train - p_teacher_train, 0.0).astype(np.float32)
            * (y_train == 0).astype(np.float32)
            * gate
        ).astype(np.float32)
        r = r_neg.astype(np.float32, copy=True)

        r_pos = np.zeros_like(r, dtype=np.float32)
        if bool(include_pos):
            r_pos = (
                np.maximum(p_teacher_train - p_baseline_train, 0.0).astype(np.float32)
                * (y_train == 1).astype(np.float32)
                * gate
            ).astype(np.float32)
            r += float(pos_scale) * r_pos
    else:
        # Original tail-focused discrepancy weighting.
        r_neg = (
            _sigmoid_np((p_baseline_train - t_hlt) / tau)
            * _sigmoid_np((t_off - p_teacher_train) / tau)
            * (y_train == 0).astype(np.float32)
            * gate
        ).astype(np.float32)
        r = r_neg.astype(np.float32, copy=True)

        r_pos = np.zeros_like(r, dtype=np.float32)
        if bool(include_pos):
            r_pos = (
                _sigmoid_np((p_teacher_train - t_off) / tau)
                * _sigmoid_np((t_hlt - p_baseline_train) / tau)
                * (y_train == 1).astype(np.float32)
                * gate
            ).astype(np.float32)
            r += float(pos_scale) * r_pos

    w = (1.0 + float(lambda_disc) * r).astype(np.float32)
    w = np.clip(w, 1.0, float(max_mult)).astype(np.float32)
    if bool(normalize_mean_one):
        m = float(np.mean(w))
        if m > 0:
            w = (w / m).astype(np.float32)

    summary = {
        "weight_mode": mode,
        "target_tpr": float(target_tpr),
        "tau": float(tau),
        "lambda_disc": float(lambda_disc),
        "max_mult": float(max_mult),
        "include_pos": bool(include_pos),
        "pos_scale": float(pos_scale),
        "normalize_mean_one": bool(normalize_mean_one),
        "teacher_conf_min": float(teacher_conf_min),
        "correctness_tau": float(correctness_tau),
        "use_teacher_hard_correct_gate": bool(use_teacher_hard_correct_gate),
        "use_teacher_conf_gate": bool(use_teacher_conf_gate),
        "use_teacher_better_gate": bool(use_teacher_better_gate),
        "t_off_val": float(t_off),
        "t_hlt_val": float(t_hlt),
        "teacher_hard_correct_rate": float(np.mean(teacher_hard_correct)),
        "mean_r_neg": float(np.mean(r_neg)),
        "mean_r_pos": float(np.mean(r_pos)),
        "mean_weight": float(np.mean(w)),
        "p95_weight": float(np.percentile(w, 95.0)),
        "max_weight": float(np.max(w)),
        "fraction_w_gt_1p1": float(np.mean(w > 1.1)),
        "fraction_w_gt_1p5": float(np.mean(w > 1.5)),
    }
    return w, summary


class WeightedJointDualDataset(Dataset):
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
        sample_weight: np.ndarray | None = None,
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
        if sample_weight is None:
            sw = np.ones((labels.shape[0],), dtype=np.float32)
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.shape[0] != labels.shape[0]:
                raise ValueError(f"sample_weight length mismatch: {sw.shape[0]} vs {labels.shape[0]}")
        self.sample_weight = torch.tensor(sw, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt_reco.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt_reco": self.feat_hlt_reco[i],
            "feat_hlt_dual": self.feat_hlt_dual[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
            "label": self.labels[i],
            "sample_weight": self.sample_weight[i],
        }


def train_joint_dual_weighted(
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
    select_metric: str,
    use_discrepancy_weights_cls: bool,
    use_discrepancy_weights_reco: bool,
) -> Tuple[OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    for p in reconstructor.parameters():
        p.requires_grad = not freeze_reconstructor

    params = [{"params": dual_model.parameters(), "lr": float(lr_dual)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_dual), weight_decay=float(weight_decay))
    sch = joint.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_state_dual_sel = None
    best_state_reco_sel = None
    best_state_dual_auc = None
    best_state_reco_auc = None
    best_state_dual_fpr = None
    best_state_reco_fpr = None

    best_val_fpr50 = float("inf")
    best_val_auc = float("-inf")
    best_sel_score = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    sel_val_fpr50 = float("nan")
    sel_val_auc = float("nan")
    no_improve = 0

    for ep in range(int(epochs)):
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
            sw = batch.get("sample_weight", None)
            if sw is not None:
                sw = sw.to(device)

            opt.zero_grad()

            if freeze_reconstructor:
                with torch.no_grad():
                    reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            else:
                reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)

            feat_b, mask_b = joint.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=bool(corrected_use_flags),
            )
            logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)

            if bool(use_discrepancy_weights_cls) and sw is not None:
                loss_cls_raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                denom = sw.sum().clamp(min=1e-6)
                loss_cls = (loss_cls_raw * sw).sum() / denom
            else:
                loss_cls = F.binary_cross_entropy_with_logits(logits, y)

            loss_rank = joint.low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()

            if float(lambda_reco) > 0.0:
                if bool(use_discrepancy_weights_reco) and sw is not None:
                    reco_losses = compute_reconstruction_losses_weighted(
                        reco_out,
                        const_hlt,
                        mask_hlt,
                        const_off,
                        mask_off,
                        b_merge,
                        b_eff,
                        LOCAL30K_CONFIG["loss"],
                        sample_weight=sw,
                    )
                else:
                    reco_losses = joint.compute_reconstruction_losses(
                        reco_out,
                        const_hlt,
                        mask_hlt,
                        const_off,
                        mask_off,
                        b_merge,
                        b_eff,
                        LOCAL30K_CONFIG["loss"],
                    )
                loss_reco = reco_losses["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            loss = (
                loss_cls
                + float(lambda_rank) * loss_rank
                + float(lambda_reco) * loss_reco
                + float(lambda_cons) * loss_cons
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual_model.parameters(), 1.0)
            if not freeze_reconstructor:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = feat_hlt_reco.size(0)
            tr_loss += float(loss.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            tr_rank += float(loss_rank.item()) * bs
            tr_reco += float(loss_reco.item()) * bs
            tr_cons += float(loss_cons.item()) * bs
            n_tr += bs

        sch.step()
        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)

        va_auc, _, _, va_fpr50 = joint.eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_model,
            loader=val_loader,
            device=device,
            corrected_weight_floor=float(corrected_weight_floor),
            corrected_use_flags=bool(corrected_use_flags),
        )

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

        print_every = 1 if str(stage_name).startswith("StageC") else 5
        if (ep + 1) % print_every == 0:
            print(
                f"{stage_name} ep {ep+1}: train_loss={tr_loss:.4f} "
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}) | "
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
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Previous run folder with Stage2 checkpoints")
    p.add_argument("--save_dir", type=str, default="", help="If empty, defaults to <run_dir>/stagec_refine")
    p.add_argument("--run_name", type=str, default="stagec_refine")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--n_train_jets", type=int, default=100000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=80)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=-1)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--ignore_saved_data_setup",
        action="store_true",
        help="Ignore run_dir/data_setup.json and run_dir/data_splits.npz; rebuild from CLI args.",
    )

    p.add_argument("--reco_ckpt", type=str, default="")
    p.add_argument("--dual_ckpt", type=str, default="")
    p.add_argument(
        "--fresh_dual_init",
        action="store_true",
        help="Initialize dual-view classifier from scratch instead of loading dual Stage2 checkpoint.",
    )

    # Stage C knobs for fast iteration.
    p.add_argument("--stageC_epochs", type=int, default=35)
    p.add_argument("--stageC_patience", type=int, default=8)
    p.add_argument("--stageC_min_epochs", type=int, default=8)
    p.add_argument(
        "--stageC_freeze_reco_epochs",
        type=int,
        default=0,
        help=(
            "Freeze reconstructor for the first N Stage-C epochs, then unfreeze for the remaining epochs. "
            "0 means never frozen."
        ),
    )
    p.add_argument("--stageC_lr_dual", type=float, default=2e-5)
    p.add_argument("--stageC_lr_reco", type=float, default=1e-5)
    p.add_argument("--stageC_lambda_rank", type=float, default=0.0)
    p.add_argument("--lambda_reco", type=float, default=0.35)
    p.add_argument("--lambda_cons", type=float, default=0.0)
    p.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])
    p.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    p.add_argument("--use_corrected_flags", action="store_true")
    p.add_argument("--disc_weight_enable", action="store_true")
    p.add_argument("--disc_weight_mode", type=str, default="tail_disagreement", choices=["tail_disagreement", "smooth_delta"])
    p.add_argument("--disc_target_tpr", type=float, default=0.50)
    p.add_argument("--disc_tau", type=float, default=0.05)
    p.add_argument("--disc_lambda", type=float, default=1.0)
    p.add_argument("--disc_max_mult", type=float, default=3.0)
    p.add_argument("--disc_include_pos", action="store_true")
    p.add_argument("--disc_pos_scale", type=float, default=0.25)
    p.add_argument("--disc_no_mean_normalize", action="store_true")
    p.add_argument(
        "--disc_apply_to_reco",
        action="store_true",
        help="Apply discrepancy sample weights to reconstruction loss reduction (lambda_reco branch).",
    )
    p.add_argument(
        "--disc_disable_cls_weight",
        action="store_true",
        help="Do not apply discrepancy sample weights to BCE classification loss.",
    )
    p.add_argument("--disc_teacher_conf_min", type=float, default=0.60)
    p.add_argument("--disc_correctness_tau", type=float, default=0.05)
    p.add_argument("--disc_disable_teacher_hard_correct_gate", action="store_true")
    p.add_argument("--disc_disable_teacher_conf_gate", action="store_true")
    p.add_argument("--disc_disable_teacher_better_gate", action="store_true")

    # Optional overrides for reconstruction loss weights during Stage C.
    p.add_argument("--loss_w_pt_ratio", type=float, default=-1.0)
    p.add_argument("--loss_w_e_ratio", type=float, default=-1.0)
    p.add_argument("--loss_w_budget", type=float, default=-1.0)
    p.add_argument("--loss_w_sparse", type=float, default=-1.0)
    p.add_argument("--loss_w_local", type=float, default=-1.0)

    # Jet reg model architecture defaults (used only if a jet reg ckpt exists in run_dir).
    p.add_argument("--jet_reg_embed_dim", type=int, default=128)
    p.add_argument("--jet_reg_num_heads", type=int, default=8)
    p.add_argument("--jet_reg_num_layers", type=int, default=4)
    p.add_argument("--jet_reg_ff_dim", type=int, default=512)
    p.add_argument("--jet_reg_dropout", type=float, default=0.1)

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    saved_setup = {}
    saved_splits = {}
    use_saved_data_setup = False
    if not bool(args.ignore_saved_data_setup):
        saved_setup = load_saved_data_setup(run_dir)
        saved_splits = load_saved_splits(run_dir)
        use_saved_data_setup = len(saved_setup) > 0

    eff_seed = int(saved_setup.get("seed", args.seed)) if use_saved_data_setup else int(args.seed)
    eff_n_train_jets = int(saved_setup.get("n_train_jets", args.n_train_jets)) if use_saved_data_setup else int(args.n_train_jets)
    eff_offset_jets = int(saved_setup.get("offset_jets", args.offset_jets)) if use_saved_data_setup else int(args.offset_jets)
    eff_max_constits = int(saved_setup.get("max_constits", args.max_constits)) if use_saved_data_setup else int(args.max_constits)
    set_seed(eff_seed)

    out_root = Path(args.save_dir) if str(args.save_dir).strip() else (run_dir / "stagec_refine")
    save_root = out_root / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg = load_cfg_from_run(run_dir)

    if args.loss_w_pt_ratio >= 0:
        cfg["loss"]["w_pt_ratio"] = float(args.loss_w_pt_ratio)
    if args.loss_w_e_ratio >= 0:
        cfg["loss"]["w_e_ratio"] = float(args.loss_w_e_ratio)
    if args.loss_w_budget >= 0:
        cfg["loss"]["w_budget"] = float(args.loss_w_budget)
    if args.loss_w_sparse >= 0:
        cfg["loss"]["w_sparse"] = float(args.loss_w_sparse)
    if args.loss_w_local >= 0:
        cfg["loss"]["w_local"] = float(args.loss_w_local)

    print(f"Device: {device}")
    print(f"Load run dir: {run_dir}")
    print(f"Save dir: {save_root}")

    if use_saved_data_setup:
        train_files = [Path(p) for p in saved_setup.get("train_files", [])]
        train_files = [p for p in train_files if p.exists()]
        if len(train_files) == 0:
            print("Warning: saved train_files unavailable; falling back to --train_path")
    else:
        train_files = []

    if len(train_files) == 0:
        train_path = Path(args.train_path)
        if train_path.is_dir():
            train_files = sorted(list(train_path.glob("*.h5")))
        else:
            train_files = [Path(x) for x in str(args.train_path).split(",") if x.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    print(
        f"Data setup source: {'saved data_setup.json' if use_saved_data_setup else 'CLI args'} | "
        f"seed={eff_seed}, n_train_jets={eff_n_train_jets}, offset_jets={eff_offset_jets}, max_constits={eff_max_constits}"
    )

    max_jets_needed = int(eff_offset_jets) + int(eff_n_train_jets)
    print("Loading offline constituents...")
    all_const, all_labels = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=int(eff_max_constits),
    )
    if all_const.shape[0] < max_jets_needed:
        raise RuntimeError(f"Requested {max_jets_needed} jets but got {all_const.shape[0]}")

    const_raw = all_const[int(eff_offset_jets): int(eff_offset_jets) + int(eff_n_train_jets)]
    labels = all_labels[int(eff_offset_jets): int(eff_offset_jets) + int(eff_n_train_jets)].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT deterministically...")
    hlt_const, hlt_mask, _, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=int(eff_seed),
    )
    budget_merge_true = budget_truth["merge_lost_per_jet"].astype(np.float32)
    budget_eff_true = budget_truth["eff_lost_per_jet"].astype(np.float32)

    print("Computing features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    splits_source = "recomputed"
    has_saved_split_idx = (
        isinstance(saved_splits, dict)
        and "train_idx" in saved_splits
        and "val_idx" in saved_splits
        and "test_idx" in saved_splits
    )
    if use_saved_data_setup and has_saved_split_idx:
        train_idx = np.asarray(saved_splits["train_idx"], dtype=np.int64)
        val_idx = np.asarray(saved_splits["val_idx"], dtype=np.int64)
        test_idx = np.asarray(saved_splits["test_idx"], dtype=np.int64)
        all_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)
        max_idx = int(np.max(all_idx)) if all_idx.size > 0 else -1
        if max_idx >= len(labels):
            raise RuntimeError(
                f"Saved split indices exceed available labels: max_idx={max_idx}, n_labels={len(labels)}"
            )
        splits_source = "saved data_splits.npz"
    else:
        idx = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(
            idx, test_size=0.30, random_state=int(eff_seed), stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=int(eff_seed), stratify=labels[temp_idx]
        )
    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} "
        f"(source: {splits_source})"
    )

    if use_saved_data_setup and isinstance(saved_splits, dict) and "means" in saved_splits and "stds" in saved_splits:
        means = np.asarray(saved_splits["means"], dtype=np.float32)
        stds = np.asarray(saved_splits["stds"], dtype=np.float32)
        if means.shape[-1] != feat_off.shape[-1] or stds.shape[-1] != feat_off.shape[-1]:
            print("Warning: saved means/stds shape mismatch; recomputing from train split.")
            means, stds = get_stats(feat_off, masks_off, train_idx)
    else:
        means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)

    bs = int(cfg["training"]["batch_size"]) if int(args.batch_size) <= 0 else int(args.batch_size)
    feat_hlt_dual = maybe_build_jetreg_features(
        run_dir=run_dir,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        const_off=const_off,
        masks_off=masks_off,
        hlt_const=hlt_const,
        device=device,
        batch_size=bs,
        args=args,
    )

    train_sample_weight = np.ones((len(train_idx),), dtype=np.float32)
    discrepancy_summary: Dict[str, float] = {"enabled": bool(args.disc_weight_enable)}
    if bool(args.disc_weight_enable):
        teacher_ckpt = run_dir / "teacher.pt"
        baseline_ckpt = run_dir / "baseline.pt"
        if not teacher_ckpt.exists():
            raise FileNotFoundError(f"Discrepancy weighting needs teacher checkpoint: {teacher_ckpt}")
        if not baseline_ckpt.exists():
            raise FileNotFoundError(f"Discrepancy weighting needs baseline checkpoint: {baseline_ckpt}")

        teacher_model = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
        teacher_model.load_state_dict(_load_checkpoint_state(teacher_ckpt, device, "teacher"))
        baseline_model = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
        baseline_model.load_state_dict(_load_checkpoint_state(baseline_ckpt, device, "baseline"))

        p_teacher_train, y_teacher_train = predict_single_view_scores(
            teacher_model, feat_off_std[train_idx], masks_off[train_idx], labels[train_idx],
            batch_size=bs, num_workers=int(args.num_workers), device=device,
        )
        p_teacher_val, y_teacher_val = predict_single_view_scores(
            teacher_model, feat_off_std[val_idx], masks_off[val_idx], labels[val_idx],
            batch_size=bs, num_workers=int(args.num_workers), device=device,
        )
        p_baseline_train, y_baseline_train = predict_single_view_scores(
            baseline_model, feat_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx],
            batch_size=bs, num_workers=int(args.num_workers), device=device,
        )
        p_baseline_val, y_baseline_val = predict_single_view_scores(
            baseline_model, feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx],
            batch_size=bs, num_workers=int(args.num_workers), device=device,
        )
        if not np.array_equal(y_teacher_train.astype(np.int64), y_baseline_train.astype(np.int64)):
            raise RuntimeError("Teacher/baseline train label mismatch while building discrepancy weights.")
        if not np.array_equal(y_teacher_val.astype(np.int64), y_baseline_val.astype(np.int64)):
            raise RuntimeError("Teacher/baseline val label mismatch while building discrepancy weights.")

        train_sample_weight, discrepancy_summary = build_discrepancy_weights(
            y_train=y_teacher_train.astype(np.int64),
            p_teacher_train=p_teacher_train,
            p_baseline_train=p_baseline_train,
            p_teacher_val=p_teacher_val,
            p_baseline_val=p_baseline_val,
            y_val=y_teacher_val.astype(np.int64),
            target_tpr=float(args.disc_target_tpr),
            tau=float(args.disc_tau),
            lambda_disc=float(args.disc_lambda),
            max_mult=float(args.disc_max_mult),
            include_pos=bool(args.disc_include_pos),
            pos_scale=float(args.disc_pos_scale),
            normalize_mean_one=(not bool(args.disc_no_mean_normalize)),
            teacher_conf_min=float(args.disc_teacher_conf_min),
            correctness_tau=float(args.disc_correctness_tau),
            use_teacher_hard_correct_gate=(not bool(args.disc_disable_teacher_hard_correct_gate)),
            use_teacher_conf_gate=(not bool(args.disc_disable_teacher_conf_gate)),
            use_teacher_better_gate=(not bool(args.disc_disable_teacher_better_gate)),
            weight_mode=str(args.disc_weight_mode),
        )
        discrepancy_summary["enabled"] = True
        print(
            "Discrepancy weighting enabled: "
            f"mean_w={discrepancy_summary.get('mean_weight', float('nan')):.4f}, "
            f"p95_w={discrepancy_summary.get('p95_weight', float('nan')):.4f}, "
            f"w>1.5={discrepancy_summary.get('fraction_w_gt_1p5', float('nan')):.4f}, "
            f"t_hlt={discrepancy_summary.get('t_hlt_val', float('nan')):.6f}, "
            f"t_off={discrepancy_summary.get('t_off_val', float('nan')):.6f}"
        )
        print(
            "Discrepancy application: "
            f"cls_weighted={bool(not args.disc_disable_cls_weight)}, "
            f"reco_weighted={bool(args.disc_apply_to_reco)}"
        )
        np.savez_compressed(
            save_root / "discrepancy_weights_train.npz",
            train_idx=train_idx.astype(np.int64),
            sample_weight=train_sample_weight.astype(np.float32),
        )
    else:
        print("Discrepancy weighting disabled (uniform Stage-C classification weights).")

    ds_train_joint = WeightedJointDualDataset(
        feat_hlt_std[train_idx], feat_hlt_dual[train_idx], hlt_mask[train_idx], hlt_const[train_idx],
        const_off[train_idx], masks_off[train_idx], budget_merge_true[train_idx], budget_eff_true[train_idx],
        labels[train_idx], sample_weight=train_sample_weight,
    )
    ds_val_joint = WeightedJointDualDataset(
        feat_hlt_std[val_idx], feat_hlt_dual[val_idx], hlt_mask[val_idx], hlt_const[val_idx],
        const_off[val_idx], masks_off[val_idx], budget_merge_true[val_idx], budget_eff_true[val_idx],
        labels[val_idx], sample_weight=np.ones((len(val_idx),), dtype=np.float32),
    )
    ds_test_joint = WeightedJointDualDataset(
        feat_hlt_std[test_idx], feat_hlt_dual[test_idx], hlt_mask[test_idx], hlt_const[test_idx],
        const_off[test_idx], masks_off[test_idx], budget_merge_true[test_idx], budget_eff_true[test_idx],
        labels[test_idx], sample_weight=np.ones((len(test_idx),), dtype=np.float32),
    )

    dl_train_joint = DataLoader(
        ds_train_joint, batch_size=bs, shuffle=True, drop_last=True,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_val_joint = DataLoader(
        ds_val_joint, batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    dl_test_joint = DataLoader(
        ds_test_joint, batch_size=bs, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    dual_input_dim_a = int(feat_hlt_dual.shape[-1])
    dual_input_dim_b = 12 if bool(args.use_corrected_flags) else 10
    dual_joint = DualViewCrossAttnClassifier(
        input_dim_a=dual_input_dim_a,
        input_dim_b=dual_input_dim_b,
        **cfg["model"],
    ).to(device)

    reco_ckpt = Path(args.reco_ckpt) if str(args.reco_ckpt).strip() else (run_dir / "offline_reconstructor_stage2.pt")
    dual_ckpt = Path(args.dual_ckpt) if str(args.dual_ckpt).strip() else (run_dir / "dual_joint_stage2.pt")
    if not reco_ckpt.exists():
        raise FileNotFoundError(f"Missing reconstructor checkpoint: {reco_ckpt}")
    if (not bool(args.fresh_dual_init)) and (not dual_ckpt.exists()):
        raise FileNotFoundError(f"Missing dual checkpoint: {dual_ckpt}")
    if bool(args.fresh_dual_init) and (not dual_ckpt.exists()):
        raise FileNotFoundError(
            f"Missing dual checkpoint for Stage2 reference metrics (fresh-dual mode): {dual_ckpt}"
        )

    reco_state = _load_checkpoint_state(reco_ckpt, device, "reconstructor")
    reconstructor.load_state_dict(reco_state)
    if bool(args.fresh_dual_init):
        # Still evaluate loaded Stage2 model for reference metrics.
        dual_ref = DualViewCrossAttnClassifier(
            input_dim_a=dual_input_dim_a,
            input_dim_b=dual_input_dim_b,
            **cfg["model"],
        ).to(device)
        dual_ref.load_state_dict(_load_checkpoint_state(dual_ckpt, device, "dual"))
        print(f"Dual init mode: FRESH (training model randomly initialized), reference Stage2 dual loaded from {dual_ckpt}")
        auc_stage2, preds_stage2, labs_stage2, _ = joint.eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_ref,
            loader=dl_test_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        del dual_ref
    else:
        dual_state = _load_checkpoint_state(dual_ckpt, device, "dual")
        dual_joint.load_state_dict(dual_state)
        print(f"Dual init mode: LOADED from {dual_ckpt}")
        auc_stage2, preds_stage2, labs_stage2, _ = joint.eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_joint,
            loader=dl_test_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
    fpr_s2, tpr_s2, _ = roc_curve(labs_stage2, preds_stage2)
    fpr30_stage2 = float(fpr_at_target_tpr(fpr_s2, tpr_s2, 0.30))
    fpr50_stage2 = float(fpr_at_target_tpr(fpr_s2, tpr_s2, 0.50))

    # Evaluate teacher/baseline from source run on the exact same rebuilt test split.
    teacher_metrics = maybe_eval_single_view_checkpoint(
        ckpt_path=run_dir / "teacher.pt",
        tag="teacher",
        feat_test=feat_off_std[test_idx],
        mask_test=masks_off[test_idx],
        labels_test=labels[test_idx],
        model_cfg=cfg["model"],
        batch_size=bs,
        num_workers=int(args.num_workers),
        device=device,
    )
    baseline_metrics = maybe_eval_single_view_checkpoint(
        ckpt_path=run_dir / "baseline.pt",
        tag="baseline",
        feat_test=feat_hlt_std[test_idx],
        mask_test=hlt_mask[test_idx],
        labels_test=labels[test_idx],
        model_cfg=cfg["model"],
        batch_size=bs,
        num_workers=int(args.num_workers),
        device=device,
    )

    print("\n" + "=" * 70)
    print("FAST STAGE C: JOINT FINETUNE FROM SAVED STAGE2")
    print("=" * 70)

    LOCAL30K_CONFIG["loss"] = cfg["loss"]
    total_stagec_epochs = int(args.stageC_epochs)
    freeze_epochs = max(0, min(int(args.stageC_freeze_reco_epochs), total_stagec_epochs))
    unfreeze_epochs = max(0, total_stagec_epochs - freeze_epochs)
    if freeze_epochs > 0:
        print(
            f"Stage-C schedule: freeze reconstructor for {freeze_epochs} epoch(s), "
            f"then unfreeze for {unfreeze_epochs} epoch(s)."
        )
    else:
        print("Stage-C schedule: reconstructor unfrozen from epoch 1.")

    def _is_auc_mode() -> bool:
        return str(args.selection_metric).lower() == "auc"

    def _better_selected(new_m: Dict[str, float], cur_m: Dict[str, float] | None) -> bool:
        if cur_m is None:
            return True
        if _is_auc_mode():
            return float(new_m.get("selected_val_auc", float("-inf"))) > float(cur_m.get("selected_val_auc", float("-inf")))
        return float(new_m.get("selected_val_fpr50", float("inf"))) < float(cur_m.get("selected_val_fpr50", float("inf")))

    def _better_auc(new_m: Dict[str, float], cur_m: Dict[str, float] | None) -> bool:
        if cur_m is None:
            return True
        return float(new_m.get("best_val_auc_seen", float("-inf"))) > float(cur_m.get("best_val_auc_seen", float("-inf")))

    def _better_fpr(new_m: Dict[str, float], cur_m: Dict[str, float] | None) -> bool:
        if cur_m is None:
            return True
        return float(new_m.get("best_val_fpr50_seen", float("inf"))) < float(cur_m.get("best_val_fpr50_seen", float("inf")))

    selected_metrics = None
    auc_metrics = None
    fpr_metrics = None
    selected_states = None
    auc_states = None
    fpr_states = None
    phase_reports = []

    def _run_phase(phase_name: str, freeze_reco: bool, epochs: int, patience: int, min_epochs: int) -> None:
        nonlocal reconstructor, dual_joint
        nonlocal selected_metrics, auc_metrics, fpr_metrics
        nonlocal selected_states, auc_states, fpr_states
        if int(epochs) <= 0:
            return
        reconstructor, dual_joint, ph_metrics, ph_states = train_joint_dual_weighted(
            reconstructor=reconstructor,
            dual_model=dual_joint,
            train_loader=dl_train_joint,
            val_loader=dl_val_joint,
            device=device,
            stage_name=phase_name,
            freeze_reconstructor=bool(freeze_reco),
            epochs=int(epochs),
            patience=int(patience),
            lr_dual=float(args.stageC_lr_dual),
            lr_reco=float(args.stageC_lr_reco),
            weight_decay=float(cfg["training"]["weight_decay"]),
            warmup_epochs=int(cfg["training"]["warmup_epochs"]),
            lambda_reco=float(args.lambda_reco),
            lambda_rank=float(args.stageC_lambda_rank),
            lambda_cons=float(args.lambda_cons),
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
            min_epochs=int(min_epochs),
            select_metric=str(args.selection_metric),
            use_discrepancy_weights_cls=(bool(args.disc_weight_enable) and (not bool(args.disc_disable_cls_weight))),
            use_discrepancy_weights_reco=(bool(args.disc_weight_enable) and bool(args.disc_apply_to_reco)),
        )
        phase_reports.append(
            {
                "phase_name": phase_name,
                "freeze_reconstructor": bool(freeze_reco),
                "epochs": int(epochs),
                "metrics": ph_metrics,
            }
        )
        if _better_selected(ph_metrics, selected_metrics):
            selected_metrics = ph_metrics
            selected_states = ph_states.get("selected", {})
        if _better_auc(ph_metrics, auc_metrics):
            auc_metrics = ph_metrics
            auc_states = ph_states.get("auc", {})
        if _better_fpr(ph_metrics, fpr_metrics):
            fpr_metrics = ph_metrics
            fpr_states = ph_states.get("fpr50", {})

    if freeze_epochs > 0:
        _run_phase(
            phase_name="StageC-FromSavedStage2-FrozenReco",
            freeze_reco=True,
            epochs=int(freeze_epochs),
            patience=max(int(freeze_epochs) + 1, int(args.stageC_patience)),
            min_epochs=int(freeze_epochs),
        )
    _run_phase(
        phase_name="StageC-FromSavedStage2",
        freeze_reco=False,
        epochs=int(unfreeze_epochs if freeze_epochs > 0 else total_stagec_epochs),
        patience=int(args.stageC_patience),
        min_epochs=min(int(args.stageC_min_epochs), int(unfreeze_epochs if freeze_epochs > 0 else total_stagec_epochs)),
    )

    stageC_metrics = {
        "selection_metric": str(args.selection_metric).lower(),
        "selected_val_fpr50": float(selected_metrics.get("selected_val_fpr50", float("nan"))) if selected_metrics else float("nan"),
        "selected_val_auc": float(selected_metrics.get("selected_val_auc", float("nan"))) if selected_metrics else float("nan"),
        "best_val_fpr50_seen": float(fpr_metrics.get("best_val_fpr50_seen", float("nan"))) if fpr_metrics else float("nan"),
        "best_val_auc_seen": float(auc_metrics.get("best_val_auc_seen", float("nan"))) if auc_metrics else float("nan"),
    }
    stageC_states = {
        "selected": {"dual": (selected_states or {}).get("dual"), "reco": (selected_states or {}).get("reco")},
        "auc": {"dual": (auc_states or {}).get("dual"), "reco": (auc_states or {}).get("reco")},
        "fpr50": {"dual": (fpr_states or {}).get("dual"), "reco": (fpr_states or {}).get("reco")},
        "phase_reports": phase_reports,
    }

    auc_joint, preds_joint, labs_joint, _ = joint.eval_joint_model(
        reconstructor=reconstructor,
        dual_model=dual_joint,
        loader=dl_test_joint,
        device=device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    fpr_j, tpr_j, _ = roc_curve(labs_joint, preds_joint)
    fpr30_joint = float(fpr_at_target_tpr(fpr_j, tpr_j, 0.30))
    fpr50_joint = float(fpr_at_target_tpr(fpr_j, tpr_j, 0.50))

    # Also evaluate Stage-C best-val_fpr50 checkpoint for comparison.
    auc_joint_fprsel = float("nan")
    fpr30_joint_fprsel = float("nan")
    fpr50_joint_fprsel = float("nan")
    if stageC_states.get("fpr50", {}).get("dual") is not None and stageC_states.get("fpr50", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["fpr50"]["reco"])
        dual_joint.load_state_dict(stageC_states["fpr50"]["dual"])
        auc_joint_fprsel, preds_joint_fprsel, labs_joint_fprsel, _ = joint.eval_joint_model(
            reconstructor=reconstructor,
            dual_model=dual_joint,
            loader=dl_test_joint,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        fpr_f, tpr_f, _ = roc_curve(labs_joint_fprsel, preds_joint_fprsel)
        fpr30_joint_fprsel = float(fpr_at_target_tpr(fpr_f, tpr_f, 0.30))
        fpr50_joint_fprsel = float(fpr_at_target_tpr(fpr_f, tpr_f, 0.50))

    # Restore selected state before saving.
    if stageC_states.get("selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["selected"]["reco"])
    if stageC_states.get("selected", {}).get("dual") is not None:
        dual_joint.load_state_dict(stageC_states["selected"]["dual"])

    torch.save(reconstructor.state_dict(), save_root / "offline_reconstructor.pt")
    torch.save(dual_joint.state_dict(), save_root / "dual_joint.pt")

    base_test = {}
    base_path = run_dir / "joint_stage_metrics.json"
    if base_path.exists():
        try:
            base_test = json.load(open(base_path, "r", encoding="utf-8")).get("test", {})
        except Exception:
            base_test = {}

    out_metrics = {
        "source_run_dir": str(run_dir),
        "source_reco_ckpt": str(reco_ckpt),
        "source_dual_ckpt": str(dual_ckpt),
        "stageC_args": {
            "stageC_epochs": int(args.stageC_epochs),
            "stageC_patience": int(args.stageC_patience),
            "stageC_min_epochs": int(args.stageC_min_epochs),
            "stageC_freeze_reco_epochs": int(args.stageC_freeze_reco_epochs),
            "stageC_lr_dual": float(args.stageC_lr_dual),
            "stageC_lr_reco": float(args.stageC_lr_reco),
            "stageC_lambda_rank": float(args.stageC_lambda_rank),
            "lambda_reco": float(args.lambda_reco),
            "lambda_cons": float(args.lambda_cons),
            "selection_metric": str(args.selection_metric),
            "corrected_weight_floor": float(args.corrected_weight_floor),
            "use_corrected_flags": bool(args.use_corrected_flags),
            "fresh_dual_init": bool(args.fresh_dual_init),
            "disc_weight_enable": bool(args.disc_weight_enable),
            "disc_weight_mode": str(args.disc_weight_mode),
            "disc_target_tpr": float(args.disc_target_tpr),
            "disc_tau": float(args.disc_tau),
            "disc_lambda": float(args.disc_lambda),
            "disc_max_mult": float(args.disc_max_mult),
            "disc_include_pos": bool(args.disc_include_pos),
            "disc_pos_scale": float(args.disc_pos_scale),
            "disc_no_mean_normalize": bool(args.disc_no_mean_normalize),
            "disc_apply_to_reco": bool(args.disc_apply_to_reco),
            "disc_disable_cls_weight": bool(args.disc_disable_cls_weight),
            "disc_teacher_conf_min": float(args.disc_teacher_conf_min),
            "disc_correctness_tau": float(args.disc_correctness_tau),
            "disc_disable_teacher_hard_correct_gate": bool(args.disc_disable_teacher_hard_correct_gate),
            "disc_disable_teacher_conf_gate": bool(args.disc_disable_teacher_conf_gate),
            "disc_disable_teacher_better_gate": bool(args.disc_disable_teacher_better_gate),
        },
        "data_reload": {
            "setup_source": "saved data_setup.json" if use_saved_data_setup else "cli args",
            "splits_source": splits_source,
            "seed_effective": int(eff_seed),
            "n_train_jets_effective": int(eff_n_train_jets),
            "offset_jets_effective": int(eff_offset_jets),
            "max_constits_effective": int(eff_max_constits),
            "train_files_used": [str(p) for p in train_files],
            "ignore_saved_data_setup": bool(args.ignore_saved_data_setup),
        },
        "stageC_metrics": stageC_metrics,
        "stageC_discrepancy_weighting": discrepancy_summary,
        "stageC_phase_reports": phase_reports,
        "test_stage2_loaded": {
            "auc": float(auc_stage2),
            "fpr30": float(fpr30_stage2),
            "fpr50": float(fpr50_stage2),
        },
        "test_stageC_selected": {
            "auc": float(auc_joint),
            "fpr30": float(fpr30_joint),
            "fpr50": float(fpr50_joint),
        },
        "test_stageC_bestfpr50": {
            "auc": float(auc_joint_fprsel),
            "fpr30": float(fpr30_joint_fprsel),
            "fpr50": float(fpr50_joint_fprsel),
        },
        "test_teacher_loaded": teacher_metrics,
        "test_baseline_loaded": baseline_metrics,
        "baseline_teacher_from_source": base_test,
    }

    with open(save_root / "stagec_refine_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    np.savez_compressed(
        save_root / "results.npz",
        labels=labs_joint.astype(np.float32),
        preds_stage2=preds_stage2.astype(np.float32),
        preds_stagec=preds_joint.astype(np.float32),
    )

    print("\n" + "=" * 70)
    print("FAST STAGE C RESULTS")
    print("=" * 70)
    if len(teacher_metrics) > 0:
        print(
            f"Teacher (loaded): AUC={teacher_metrics['auc']:.4f}, "
            f"FPR30={teacher_metrics['fpr30']:.6f}, FPR50={teacher_metrics['fpr50']:.6f}"
        )
    if len(baseline_metrics) > 0:
        print(
            f"Baseline (loaded): AUC={baseline_metrics['auc']:.4f}, "
            f"FPR30={baseline_metrics['fpr30']:.6f}, FPR50={baseline_metrics['fpr50']:.6f}"
        )
    print(f"Loaded Stage2: AUC={auc_stage2:.4f}, FPR30={fpr30_stage2:.6f}, FPR50={fpr50_stage2:.6f}")
    print(f"StageC Selected: AUC={auc_joint:.4f}, FPR30={fpr30_joint:.6f}, FPR50={fpr50_joint:.6f}")
    if np.isfinite(auc_joint_fprsel):
        print(
            f"StageC BestValFPR50: AUC={auc_joint_fprsel:.4f}, "
            f"FPR30={fpr30_joint_fprsel:.6f}, FPR50={fpr50_joint_fprsel:.6f}"
        )
    print(f"Saved to: {save_root}")


if __name__ == "__main__":
    main()

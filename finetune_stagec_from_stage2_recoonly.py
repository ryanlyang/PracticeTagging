#!/usr/bin/env python3
"""
Stage-C continuation from saved Stage2 checkpoints, but with a reco-only top tagger.

What this script does:
1) Reloads the original data setup/splits deterministically from a saved run dir.
2) Loads Stage2 reconstructor checkpoint (typically offline_reconstructor_stage2.pt).
3) Trains a reco-only top tagger on soft corrected outputs with reconstructor frozen.
4) Saves frozen-phase selected checkpoint.
5) Continues joint finetuning by unfreezing reconstructor (same reco-only tagger).
6) Saves final selected checkpoint.

The reco-only classifier consumes only the reconstructed corrected view
(and optional merge/eff flags) -- no HLT branch / no dual-view classifier.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import offline_reconstructor_joint_dualview_stage2save_auc_norankc as joint
import offline_reconstructor_no_gt_local30kv2 as reco_base
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as LOCAL30K_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
    compute_reconstruction_losses,
    fpr_at_target_tpr,
)
from unmerge_correct_hlt import (
    RANDOM_SEED,
    JetDataset,
    ParticleTransformer,
    compute_features,
    eval_classifier,
    get_scheduler,
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


class RecoOnlyStageCDataset(Dataset):
    def __init__(
        self,
        feat_hlt_reco: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
        labels: np.ndarray,
        sample_weight_cls: np.ndarray | None = None,
        sample_weight_reco: np.ndarray | None = None,
        specialist_bucket_mask: np.ndarray | None = None,
    ):
        self.feat_hlt_reco = torch.tensor(feat_hlt_reco, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        n = labels.shape[0]
        if sample_weight_cls is None:
            sw_cls = np.ones((n,), dtype=np.float32)
        else:
            sw_cls = np.asarray(sample_weight_cls, dtype=np.float32)
            if sw_cls.shape[0] != n:
                raise ValueError(f"sample_weight_cls length mismatch: {sw_cls.shape[0]} vs {n}")
        if sample_weight_reco is None:
            sw_reco = np.ones((n,), dtype=np.float32)
        else:
            sw_reco = np.asarray(sample_weight_reco, dtype=np.float32)
            if sw_reco.shape[0] != n:
                raise ValueError(f"sample_weight_reco length mismatch: {sw_reco.shape[0]} vs {n}")
        if specialist_bucket_mask is None:
            spec_mask = np.zeros((n,), dtype=np.uint8)
        else:
            spec_mask = np.asarray(specialist_bucket_mask, dtype=np.uint8)
            if spec_mask.shape[0] != n:
                raise ValueError(f"specialist_bucket_mask length mismatch: {spec_mask.shape[0]} vs {n}")
        self.sample_weight_cls = torch.tensor(sw_cls, dtype=torch.float32)
        self.sample_weight_reco = torch.tensor(sw_reco, dtype=torch.float32)
        self.specialist_bucket_mask = torch.tensor(spec_mask, dtype=torch.uint8)

    def __len__(self) -> int:
        return int(self.feat_hlt_reco.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt_reco": self.feat_hlt_reco[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
            "label": self.labels[i],
            "sample_weight_cls": self.sample_weight_cls[i],
            "sample_weight_reco": self.sample_weight_reco[i],
            "specialist_bucket_mask": self.specialist_bucket_mask[i],
        }


@torch.no_grad()
def eval_recoonly_joint_model(
    reconstructor: OfflineReconstructor,
    reco_clf: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    reconstructor.eval()
    reco_clf.eval()
    preds: List[np.ndarray] = []
    labs: List[np.ndarray] = []

    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].detach().cpu().numpy().astype(np.float32)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = joint.build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=bool(corrected_use_flags),
        )
        logits = reco_clf(feat_b, mask_b).squeeze(1)
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        preds.append(p)
        labs.append(y)

    if len(preds) == 0:
        return float("nan"), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64), float("nan")

    pred = np.concatenate(preds).astype(np.float32)
    lab = np.concatenate(labs).astype(np.int64)
    if len(np.unique(lab)) < 2:
        return float("nan"), pred, lab, float("nan")

    fpr, tpr, _ = roc_curve(lab, pred)
    auc = float(np.trapz(tpr, fpr))
    fpr50 = float(fpr_at_target_tpr(fpr, tpr, 0.50))
    return float(auc), pred, lab, float(fpr50)


@torch.no_grad()
def eval_recoonly_joint_model_both_metrics(
    reconstructor: OfflineReconstructor,
    reco_clf: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    weighted_key: Optional[str] = None,
) -> Dict[str, object]:
    reconstructor.eval()
    reco_clf.eval()
    preds_list: List[np.ndarray] = []
    labs_list: List[np.ndarray] = []
    w_list: List[np.ndarray] = []
    has_weights = weighted_key is not None

    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = joint.build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=bool(corrected_use_flags),
        )
        logits = reco_clf(feat_b, mask_b).squeeze(1)
        p = torch.sigmoid(logits)

        preds_list.append(p.detach().cpu().numpy())
        labs_list.append(y.detach().cpu().numpy())
        if has_weights:
            if weighted_key in batch:
                w_list.append(batch[weighted_key].detach().cpu().numpy())
            else:
                has_weights = False
                w_list = []

    preds = np.concatenate(preds_list) if preds_list else np.zeros(0, dtype=np.float32)
    labs = np.concatenate(labs_list) if labs_list else np.zeros(0, dtype=np.float32)
    if preds.size == 0:
        return {
            "preds": preds,
            "labs": labs,
            "weights": np.zeros(0, dtype=np.float32),
            "auc_unweighted": float("nan"),
            "fpr50_unweighted": float("nan"),
            "auc_weighted": float("nan"),
            "fpr50_weighted": float("nan"),
        }

    if len(np.unique(labs)) > 1:
        auc_unw = roc_auc_score(labs, preds)
        fpr_unw, tpr_unw, _ = roc_curve(labs, preds)
        fpr50_unw = fpr_at_target_tpr(fpr_unw, tpr_unw, 0.50)
    else:
        auc_unw = float("nan")
        fpr50_unw = float("nan")

    weights = np.concatenate(w_list).astype(np.float32) if (has_weights and w_list) else np.zeros(0, dtype=np.float32)
    if weights.size == preds.size and float(np.sum(weights)) > 0.0 and len(np.unique(labs)) > 1:
        auc_w = roc_auc_score(labs, preds, sample_weight=weights)
        fpr_w, tpr_w, _ = roc_curve(labs, preds, sample_weight=weights)
        fpr50_w = fpr_at_target_tpr(fpr_w, tpr_w, 0.50)
    else:
        auc_w = float("nan")
        fpr50_w = float("nan")

    return {
        "preds": preds.astype(np.float32),
        "labs": labs.astype(np.float32),
        "weights": weights.astype(np.float32),
        "auc_unweighted": float(auc_unw),
        "fpr50_unweighted": float(fpr50_unw),
        "auc_weighted": float(auc_w),
        "fpr50_weighted": float(fpr50_w),
    }


def build_specialist_bucket_weights(
    labels: np.ndarray,
    hlt_count: np.ndarray,
    p_hlt: np.ndarray,
    hlt_jet_pt: np.ndarray,
    count_low: int,
    count_high: int,
    p_hlt_threshold: float,
    jet_pt_hlt_min: float,
    w_neg: float,
    w_pos: float,
    w_other: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    y = labels.astype(np.int64)
    c = hlt_count.astype(np.int32)
    p = p_hlt.astype(np.float32)
    pt = hlt_jet_pt.astype(np.float32)
    use_pt_gate = float(jet_pt_hlt_min) > 0.0
    pt_gate = np.ones_like(p, dtype=bool) if not use_pt_gate else (pt >= float(jet_pt_hlt_min))
    in_bucket = (
        (c > int(count_low))
        & (c <= int(count_high))
        & (p >= float(p_hlt_threshold))
        & pt_gate
    )
    w = np.full((y.shape[0],), float(w_other), dtype=np.float32)
    w[in_bucket & (y == 0)] = float(w_neg)
    w[in_bucket & (y == 1)] = float(w_pos)
    summary = {
        "count_low": int(count_low),
        "count_high": int(count_high),
        "p_hlt_threshold": float(p_hlt_threshold),
        "jet_pt_hlt_min": float(jet_pt_hlt_min),
        "use_jet_pt_gate": bool(use_pt_gate),
        "w_neg": float(w_neg),
        "w_pos": float(w_pos),
        "w_other": float(w_other),
        "traffic": float(np.mean(in_bucket)),
        "traffic_count": int(np.sum(in_bucket)),
        "mean_weight": float(np.mean(w)),
        "p95_weight": float(np.percentile(w, 95.0)),
        "fraction_w_gt_1p1": float(np.mean(w > 1.1)),
        "fraction_w_gt_1p5": float(np.mean(w > 1.5)),
        "fraction_w_gt_5": float(np.mean(w > 5.0)),
    }
    return w.astype(np.float32), in_bucket.astype(np.uint8), summary


@torch.no_grad()
def _predict_hlt_probs(
    baseline_ckpt: Path,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    labels: np.ndarray,
    model_cfg: Dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    model = ParticleTransformer(input_dim=7, **model_cfg).to(device)
    model.load_state_dict(_load_checkpoint_state(baseline_ckpt, device, "baseline_for_spec_bucket"))
    ds = JetDataset(feat_hlt_std, hlt_mask, labels)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    _, preds, _ = eval_classifier(model, dl, device)
    return np.asarray(preds, dtype=np.float32)


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
    eps = 1e-8
    sw = None
    if sample_weight is not None:
        sw = sample_weight.float().clamp(min=0.0)

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

    pred_pt = torch.sqrt(pred_px.pow(2) + pred_py.pow(2) + eps)
    true_pt = torch.sqrt(true_px.pow(2) + true_py.pow(2) + eps)
    pred_p = torch.sqrt(pred_px.pow(2) + pred_py.pow(2) + pred_pz.pow(2) + eps)
    true_p = torch.sqrt(true_px.pow(2) + true_py.pow(2) + true_pz.pow(2) + eps)
    pred_m2 = torch.clamp(pred_E.pow(2) - pred_p.pow(2), min=0.0)
    true_m2 = torch.clamp(true_E.pow(2) - true_p.pow(2), min=0.0)
    pred_m = torch.sqrt(pred_m2 + eps)
    true_m = torch.sqrt(true_m2 + eps)

    loss_pt_ratio_vec = torch.abs(pred_pt / (true_pt + eps) - 1.0)
    loss_e_ratio_vec = torch.abs(pred_E / (true_E + eps) - 1.0)
    loss_m_ratio_vec = torch.abs(pred_m / (true_m + eps) - 1.0)
    loss_phys_vec = (
        float(loss_cfg["w_pt"]) * loss_pt_ratio_vec
        + float(loss_cfg["w_E"]) * loss_e_ratio_vec
        + float(loss_cfg["w_m"]) * loss_m_ratio_vec
    )

    merge_pred = out["merge_budget"]
    eff_pred = out["eff_budget"]
    loss_budget_vec = (
        torch.abs(merge_pred - budget_merge_true)
        + float(loss_cfg.get("eff_budget_weight", 1.0)) * torch.abs(eff_pred - budget_eff_true)
    )
    loss_sparse_vec = out["cand_weights"].mean(dim=1)
    loss_local_vec = out.get("locality_penalty", torch.zeros_like(loss_sparse_vec)).float()

    total_vec = (
        float(loss_cfg["lambda_set"]) * loss_set_vec
        + float(loss_cfg["lambda_phys"]) * loss_phys_vec
        + float(loss_cfg["lambda_budget"]) * loss_budget_vec
        + float(loss_cfg["lambda_sparse"]) * loss_sparse_vec
        + float(loss_cfg.get("lambda_local", 0.0)) * loss_local_vec
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


def train_recoonly_joint(
    reconstructor: OfflineReconstructor,
    reco_clf: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    freeze_reconstructor: bool,
    epochs: int,
    patience: int,
    lr_cls: float,
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
    apply_cls_weight: bool = False,
    apply_reco_weight: bool = False,
    val_weight_key: Optional[str] = None,
    use_weighted_val_selection: bool = False,
) -> Tuple[OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    for p in reconstructor.parameters():
        p.requires_grad = not freeze_reconstructor

    params = [{"params": reco_clf.parameters(), "lr": float(lr_cls)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_cls), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_state_cls_sel = None
    best_state_reco_sel = None
    best_state_cls_auc = None
    best_state_reco_auc = None
    best_state_cls_fpr = None
    best_state_reco_fpr = None

    best_val_fpr50 = float("inf")
    best_val_auc = float("-inf")
    best_val_fpr50_unw = float("inf")
    best_val_auc_unw = float("-inf")
    best_val_fpr50_w = float("inf")
    best_val_auc_w = float("-inf")
    best_sel_score = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    sel_val_fpr50 = float("nan")
    sel_val_auc = float("nan")
    sel_val_fpr50_unw = float("nan")
    sel_val_auc_unw = float("nan")
    sel_val_fpr50_w = float("nan")
    sel_val_auc_w = float("nan")
    val_metric_source = "weighted" if bool(use_weighted_val_selection) else "unweighted"
    no_improve = 0

    for ep in tqdm(range(int(epochs)), desc=stage_name):
        reco_clf.train()
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
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            b_merge = batch["budget_merge_true"].to(device)
            b_eff = batch["budget_eff_true"].to(device)
            y = batch["label"].to(device)
            sw_cls = batch.get("sample_weight_cls", None)
            sw_reco = batch.get("sample_weight_reco", None)
            if sw_cls is not None:
                sw_cls = sw_cls.to(device)
            if sw_reco is not None:
                sw_reco = sw_reco.to(device)

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
            logits = reco_clf(feat_b, mask_b).squeeze(1)

            if bool(apply_cls_weight) and sw_cls is not None:
                loss_cls_raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                denom = sw_cls.sum().clamp(min=1e-6)
                loss_cls = (loss_cls_raw * sw_cls).sum() / denom
            else:
                loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = joint.low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()

            if float(lambda_reco) > 0.0:
                if bool(apply_reco_weight) and sw_reco is not None:
                    reco_losses = compute_reconstruction_losses_weighted(
                        reco_out,
                        const_hlt,
                        mask_hlt,
                        const_off,
                        mask_off,
                        b_merge,
                        b_eff,
                        LOCAL30K_CONFIG["loss"],
                        sample_weight=sw_reco,
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
            torch.nn.utils.clip_grad_norm_(reco_clf.parameters(), 1.0)
            if not freeze_reconstructor:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = int(feat_hlt_reco.size(0))
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

        va_pack = eval_recoonly_joint_model_both_metrics(
            reconstructor=reconstructor,
            reco_clf=reco_clf,
            loader=val_loader,
            device=device,
            corrected_weight_floor=float(corrected_weight_floor),
            corrected_use_flags=bool(corrected_use_flags),
            weighted_key=val_weight_key if bool(use_weighted_val_selection) else None,
        )
        va_auc_unw = float(va_pack["auc_unweighted"])
        va_fpr50_unw = float(va_pack["fpr50_unweighted"])
        va_auc_w = float(va_pack["auc_weighted"])
        va_fpr50_w = float(va_pack["fpr50_weighted"])
        has_weighted_val = bool(use_weighted_val_selection) and np.isfinite(va_auc_w) and np.isfinite(va_fpr50_w)
        va_auc = float(va_auc_w) if has_weighted_val else float(va_auc_unw)
        va_fpr50 = float(va_fpr50_w) if has_weighted_val else float(va_fpr50_unw)
        metric_source_epoch = "weighted" if has_weighted_val else "unweighted"

        if np.isfinite(va_fpr50_unw) and float(va_fpr50_unw) < best_val_fpr50_unw:
            best_val_fpr50_unw = float(va_fpr50_unw)
        if np.isfinite(va_auc_unw) and float(va_auc_unw) > best_val_auc_unw:
            best_val_auc_unw = float(va_auc_unw)
        if np.isfinite(va_fpr50_w) and float(va_fpr50_w) < best_val_fpr50_w:
            best_val_fpr50_w = float(va_fpr50_w)
        if np.isfinite(va_auc_w) and float(va_auc_w) > best_val_auc_w:
            best_val_auc_w = float(va_auc_w)

        if np.isfinite(va_fpr50) and float(va_fpr50) < best_val_fpr50:
            best_val_fpr50 = float(va_fpr50)
            best_state_cls_fpr = {k: v.detach().cpu().clone() for k, v in reco_clf.state_dict().items()}
            best_state_reco_fpr = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state_cls_auc = {k: v.detach().cpu().clone() for k, v in reco_clf.state_dict().items()}
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
            sel_val_fpr50_unw = float(va_fpr50_unw)
            sel_val_auc_unw = float(va_auc_unw)
            sel_val_fpr50_w = float(va_fpr50_w)
            sel_val_auc_w = float(va_auc_w)
            val_metric_source = str(metric_source_epoch)
            best_state_cls_sel = {k: v.detach().cpu().clone() for k, v in reco_clf.state_dict().items()}
            best_state_reco_sel = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print_every = 1 if str(stage_name).startswith("StageC") else 5
        if (ep + 1) % print_every == 0:
            print(
                f"{stage_name} ep {ep+1}: train_loss={tr_loss:.4f} "
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}) | "
                f"val_auc_unw={va_auc_unw:.4f}, val_fpr50_unw={va_fpr50_unw:.6f}, "
                f"val_auc_w={va_auc_w:.4f}, val_fpr50_w={va_fpr50_w:.6f}, "
                f"val_metric_source={metric_source_epoch}, "
                f"select={str(select_metric).lower()}, best_sel={best_sel_score:.6f}"
            )

        if (ep + 1) >= int(min_epochs) and no_improve >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_cls_sel is not None:
        reco_clf.load_state_dict(best_state_cls_sel)
    if best_state_reco_sel is not None:
        reconstructor.load_state_dict(best_state_reco_sel)

    metrics = {
        "val_metric_source": str(val_metric_source),
        "selection_metric": str(select_metric).lower(),
        "selected_val_fpr50": float(sel_val_fpr50),
        "selected_val_auc": float(sel_val_auc),
        "selected_val_fpr50_unweighted": float(sel_val_fpr50_unw),
        "selected_val_auc_unweighted": float(sel_val_auc_unw),
        "selected_val_fpr50_weighted": float(sel_val_fpr50_w),
        "selected_val_auc_weighted": float(sel_val_auc_w),
        "best_val_fpr50_seen": float(best_val_fpr50),
        "best_val_auc_seen": float(best_val_auc),
        "best_val_fpr50_seen_unweighted": float(best_val_fpr50_unw),
        "best_val_auc_seen_unweighted": float(best_val_auc_unw),
        "best_val_fpr50_seen_weighted": float(best_val_fpr50_w),
        "best_val_auc_seen_weighted": float(best_val_auc_w),
    }
    state_pack = {
        "selected": {"clf": best_state_cls_sel, "reco": best_state_reco_sel},
        "auc": {"clf": best_state_cls_auc, "reco": best_state_reco_auc},
        "fpr50": {"clf": best_state_cls_fpr, "reco": best_state_reco_fpr},
    }
    return reconstructor, reco_clf, metrics, state_pack


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Previous run folder with Stage2 checkpoints")
    p.add_argument("--save_dir", type=str, default="", help="If empty, defaults to <run_dir>/stagec_refine")
    p.add_argument("--run_name", type=str, default="stagec_recoonly_refine")

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

    # Stage C schedule and loss settings.
    p.add_argument("--stageC_epochs", type=int, default=70)
    p.add_argument("--stageC_patience", type=int, default=12)
    p.add_argument("--stageC_min_epochs", type=int, default=25)
    p.add_argument("--stageC_freeze_reco_epochs", type=int, default=20)
    p.add_argument("--stageC_lr_cls", type=float, default=1e-5)
    p.add_argument("--stageC_lr_reco", type=float, default=5e-6)
    p.add_argument("--stageC_lambda_rank", type=float, default=0.0)
    p.add_argument("--lambda_reco", type=float, default=0.4)
    p.add_argument("--lambda_cons", type=float, default=0.06)
    p.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])
    p.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    p.add_argument("--use_corrected_flags", action="store_true")

    # Optional specialist bucket weighting (same style as spec-bucket runs).
    p.add_argument("--spec_bucket_enable", action="store_true")
    p.add_argument("--spec_bucket_count_low", type=int, default=0)
    p.add_argument("--spec_bucket_count_high", type=int, default=15)
    p.add_argument("--spec_bucket_p_hlt_threshold", type=float, default=0.242)
    p.add_argument("--spec_bucket_jet_pt_hlt_min", type=float, default=0.0)
    p.add_argument("--spec_bucket_w_neg", type=float, default=10.0)
    p.add_argument("--spec_bucket_w_pos", type=float, default=4.0)
    p.add_argument("--spec_bucket_w_other", type=float, default=1.0)

    args = p.parse_args()
    if bool(args.spec_bucket_enable):
        if int(args.spec_bucket_count_high) <= int(args.spec_bucket_count_low):
            raise ValueError("spec_bucket_count_high must be > spec_bucket_count_low")
        if float(args.spec_bucket_w_neg) < 0.0 or float(args.spec_bucket_w_pos) < 0.0 or float(args.spec_bucket_w_other) < 0.0:
            raise ValueError("specialist bucket weights must be non-negative")

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

    cfg = load_cfg_from_run(run_dir)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Load run dir: {run_dir}")
    print(f"Save dir: {save_root}")

    train_path = Path(args.train_path)
    train_files_saved = saved_setup.get("train_files", None) if use_saved_data_setup else None
    if isinstance(train_files_saved, list) and len(train_files_saved) > 0:
        train_files = [Path(x) for x in train_files_saved]
    else:
        train_files = [train_path / "test.h5"]

    source_tag = "saved data_setup.json" if use_saved_data_setup else "CLI args"
    print(
        f"Data setup source: {source_tag} | "
        f"seed={eff_seed}, n_train_jets={eff_n_train_jets}, offset_jets={eff_offset_jets}, max_constits={eff_max_constits}"
    )

    print("Loading offline constituents...")
    if int(eff_offset_jets) <= 0:
        const_raw, labels = load_raw_constituents_from_h5(
            train_files,
            max_jets=eff_n_train_jets,
            max_constits=eff_max_constits,
        )
    else:
        # Compatibility path for helper versions that do not support offset loading.
        need = int(eff_n_train_jets) + int(eff_offset_jets)
        const_all, labels_all = load_raw_constituents_from_h5(
            train_files,
            max_jets=need,
            max_constits=eff_max_constits,
        )
        start = int(eff_offset_jets)
        end = start + int(eff_n_train_jets)
        if const_all.shape[0] < end:
            raise RuntimeError(
                f"Requested n_train_jets={eff_n_train_jets} with offset={eff_offset_jets}, "
                f"but only loaded {const_all.shape[0]} jets."
            )
        const_raw = const_all[start:end]
        labels = labels_all[start:end]

    masks_off = const_raw[:, :, 0] > 0.0
    const_off = const_raw.copy()

    print("Generating pseudo-HLT deterministically...")
    hlt_const, hlt_mask, hlt_stats, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=eff_seed,
    )

    print("Computing features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    saved_train_idx = saved_splits.get("train_idx") if isinstance(saved_splits, dict) else None
    saved_val_idx = saved_splits.get("val_idx") if isinstance(saved_splits, dict) else None
    saved_test_idx = saved_splits.get("test_idx") if isinstance(saved_splits, dict) else None
    has_saved_split_idx = (
        saved_train_idx is not None
        and saved_val_idx is not None
        and saved_test_idx is not None
    )

    if use_saved_data_setup and has_saved_split_idx:
        train_idx = np.asarray(saved_train_idx, dtype=np.int64)
        val_idx = np.asarray(saved_val_idx, dtype=np.int64)
        test_idx = np.asarray(saved_test_idx, dtype=np.int64)
        all_idx = np.arange(labels.shape[0], dtype=np.int64)
        used = np.concatenate([train_idx, val_idx, test_idx], axis=0)
        if np.any(used < 0) or np.any(used >= labels.shape[0]):
            raise ValueError("Saved split indices are out of bounds for currently loaded data.")
        if np.unique(used).shape[0] != used.shape[0]:
            raise ValueError("Saved split indices contain duplicates across train/val/test.")
        missing = np.setdiff1d(all_idx, used)
        if missing.size > 0:
            print(f"Warning: saved splits do not cover {missing.size} samples; they will be ignored.")
        splits_source = "saved data_splits.npz"
    else:
        idx = np.arange(labels.shape[0])
        train_idx, tmp_idx = train_test_split(
            idx,
            test_size=0.30,
            random_state=eff_seed,
            stratify=labels,
        )
        val_idx, test_idx = train_test_split(
            tmp_idx,
            test_size=0.50,
            random_state=eff_seed,
            stratify=labels[tmp_idx],
        )
        splits_source = "fresh train_test_split"

    if use_saved_data_setup and isinstance(saved_splits, dict) and "means" in saved_splits and "stds" in saved_splits:
        means = np.asarray(saved_splits["means"], dtype=np.float32)
        stds = np.asarray(saved_splits["stds"], dtype=np.float32)
    else:
        means, stds = get_stats(feat_hlt[train_idx], hlt_mask[train_idx])

    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)
    feat_off_std = standardize(feat_off, masks_off, means, stds)

    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)

    stage_metrics_path = run_dir / "joint_stage_metrics.json"
    added_target_scale = 1.0
    if stage_metrics_path.exists():
        try:
            m = json.load(open(stage_metrics_path, "r", encoding="utf-8"))
            added_target_scale = float(m.get("variant", {}).get("added_target_scale", 1.0))
        except Exception:
            pass

    budget_merge_true = (added_target_scale * true_added_raw).astype(np.float32)
    budget_eff_true = np.zeros_like(true_added_raw, dtype=np.float32)

    bs = int(cfg["training"]["batch_size"]) if int(args.batch_size) <= 0 else int(args.batch_size)

    sample_weight_cls = np.ones((labels.shape[0],), dtype=np.float32)
    sample_weight_reco = np.ones((labels.shape[0],), dtype=np.float32)
    specialist_bucket_mask = np.zeros((labels.shape[0],), dtype=np.uint8)
    specialist_summary_train: Dict[str, object] = {"enabled": False}
    specialist_summary_val: Dict[str, object] = {"enabled": False}
    specialist_summary_test: Dict[str, object] = {"enabled": False}
    use_weighted_val_selection = False
    apply_cls_weight = False
    apply_reco_weight = False

    if bool(args.spec_bucket_enable):
        baseline_ckpt = run_dir / "baseline.pt"
        if not baseline_ckpt.exists():
            raise FileNotFoundError(
                f"Specialist bucket requested, but baseline checkpoint missing: {baseline_ckpt}"
            )
        p_hlt = _predict_hlt_probs(
            baseline_ckpt=baseline_ckpt,
            feat_hlt_std=feat_hlt_std,
            hlt_mask=hlt_mask,
            labels=labels,
            model_cfg=cfg["model"],
            batch_size=bs,
            num_workers=int(args.num_workers),
            device=device,
        )
        if p_hlt.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"Baseline probability length mismatch: {p_hlt.shape[0]} vs {labels.shape[0]}"
            )
        pt = hlt_const[:, :, 0].astype(np.float32)
        phi = hlt_const[:, :, 2].astype(np.float32)
        m = hlt_mask.astype(np.float32)
        jet_px_hlt = (pt * np.cos(phi) * m).sum(axis=1)
        jet_py_hlt = (pt * np.sin(phi) * m).sum(axis=1)
        hlt_jet_pt = np.sqrt(np.maximum(jet_px_hlt**2 + jet_py_hlt**2, 0.0)).astype(np.float32)

        sample_weight_cls, specialist_bucket_mask, specialist_summary_all = build_specialist_bucket_weights(
            labels=labels,
            hlt_count=hlt_count,
            p_hlt=p_hlt,
            hlt_jet_pt=hlt_jet_pt,
            count_low=int(args.spec_bucket_count_low),
            count_high=int(args.spec_bucket_count_high),
            p_hlt_threshold=float(args.spec_bucket_p_hlt_threshold),
            jet_pt_hlt_min=float(args.spec_bucket_jet_pt_hlt_min),
            w_neg=float(args.spec_bucket_w_neg),
            w_pos=float(args.spec_bucket_w_pos),
            w_other=float(args.spec_bucket_w_other),
        )
        sample_weight_reco = sample_weight_cls.copy()
        use_weighted_val_selection = True
        apply_cls_weight = True
        apply_reco_weight = True
        np.savez_compressed(
            save_root / "specialist_bucket_weights.npz",
            sample_weight_cls=sample_weight_cls.astype(np.float32),
            sample_weight_reco=sample_weight_reco.astype(np.float32),
            specialist_bucket_mask=specialist_bucket_mask.astype(np.uint8),
            p_hlt=p_hlt.astype(np.float32),
            hlt_jet_pt=hlt_jet_pt.astype(np.float32),
            hlt_count=hlt_count.astype(np.float32),
        )
        print(
            "Specialist bucket rule: "
            f"{int(args.spec_bucket_count_low)} < n_const_hlt <= {int(args.spec_bucket_count_high)} "
            f"and p_hlt >= {float(args.spec_bucket_p_hlt_threshold):.6f}"
            + (
                f" and jet_pt_hlt >= {float(args.spec_bucket_jet_pt_hlt_min):.1f}"
                if float(args.spec_bucket_jet_pt_hlt_min) > 0.0
                else ""
            )
        )
        print(
            "Specialist bucket weights: "
            f"w_neg={float(args.spec_bucket_w_neg):.3f}, "
            f"w_pos={float(args.spec_bucket_w_pos):.3f}, "
            f"w_other={float(args.spec_bucket_w_other):.3f}"
        )
        print(
            "Specialist bucket weights (all): "
            f"traffic={float(specialist_summary_all.get('traffic', float('nan'))):.4f}, "
            f"mean={float(specialist_summary_all.get('mean_weight', float('nan'))):.4f}, "
            f"p95={float(specialist_summary_all.get('p95_weight', float('nan'))):.4f}"
        )

    ds_train = RecoOnlyStageCDataset(
        feat_hlt_reco=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        budget_merge_true=budget_merge_true[train_idx],
        budget_eff_true=budget_eff_true[train_idx],
        labels=labels[train_idx],
        sample_weight_cls=sample_weight_cls[train_idx],
        sample_weight_reco=sample_weight_reco[train_idx],
        specialist_bucket_mask=specialist_bucket_mask[train_idx],
    )
    ds_val = RecoOnlyStageCDataset(
        feat_hlt_reco=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        budget_merge_true=budget_merge_true[val_idx],
        budget_eff_true=budget_eff_true[val_idx],
        labels=labels[val_idx],
        sample_weight_cls=sample_weight_cls[val_idx],
        sample_weight_reco=sample_weight_reco[val_idx],
        specialist_bucket_mask=specialist_bucket_mask[val_idx],
    )
    ds_test = RecoOnlyStageCDataset(
        feat_hlt_reco=feat_hlt_std[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_hlt=hlt_const[test_idx],
        const_off=const_off[test_idx],
        mask_off=masks_off[test_idx],
        budget_merge_true=budget_merge_true[test_idx],
        budget_eff_true=budget_eff_true[test_idx],
        labels=labels[test_idx],
        sample_weight_cls=np.ones((len(test_idx),), dtype=np.float32),
        sample_weight_reco=np.ones((len(test_idx),), dtype=np.float32),
        specialist_bucket_mask=specialist_bucket_mask[test_idx],
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=bs,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=bs,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} "
        f"(source: {splits_source})"
    )
    if bool(args.spec_bucket_enable):
        def _split_summary(name: str, idx_arr: np.ndarray) -> Dict[str, object]:
            w = sample_weight_cls[idx_arr].astype(np.float32)
            m = specialist_bucket_mask[idx_arr].astype(np.uint8)
            return {
                "enabled": True,
                "split": name,
                "n": int(len(idx_arr)),
                "traffic": float(np.mean(m > 0)),
                "traffic_count": int(np.sum(m > 0)),
                "mean_weight": float(np.mean(w)),
                "p95_weight": float(np.percentile(w, 95.0)),
                "fraction_w_gt_1p1": float(np.mean(w > 1.1)),
                "fraction_w_gt_1p5": float(np.mean(w > 1.5)),
                "fraction_w_gt_5": float(np.mean(w > 5.0)),
            }

        specialist_summary_train = _split_summary("train", train_idx)
        specialist_summary_val = _split_summary("val", val_idx)
        specialist_summary_test = _split_summary("test", test_idx)
        print(
            "Specialist bucket split stats: "
            f"train traffic={specialist_summary_train['traffic']:.4f}, "
            f"val traffic={specialist_summary_val['traffic']:.4f}, "
            f"test traffic={specialist_summary_test['traffic']:.4f}"
        )

    reco_ckpt = Path(args.reco_ckpt) if str(args.reco_ckpt).strip() else (run_dir / "offline_reconstructor_stage2.pt")
    if not reco_ckpt.exists():
        raise FileNotFoundError(
            f"Stage2 reconstructor checkpoint not found: {reco_ckpt}. "
            "Expected offline_reconstructor_stage2.pt in run_dir or pass --reco_ckpt."
        )

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(_load_checkpoint_state(reco_ckpt, device, "stage2_reconstructor"))

    reco_input_dim = 12 if bool(args.use_corrected_flags) else 10
    reco_clf = ParticleTransformer(input_dim=reco_input_dim, **cfg["model"]).to(device)

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
    print("FAST STAGE C (RECO-ONLY): FROZEN -> UNFREEZE")
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
    print(
        "Specialist weighting: "
        f"enabled={bool(args.spec_bucket_enable)} cls={bool(apply_cls_weight)} reco={bool(apply_reco_weight)} "
        f"weighted_val={bool(use_weighted_val_selection)}"
    )

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
    frozen_selected_metrics = None
    frozen_selected_states = None
    phase_reports = []

    def _run_phase(phase_name: str, freeze_reco: bool, epochs: int, patience: int, min_epochs: int) -> None:
        nonlocal reconstructor, reco_clf
        nonlocal selected_metrics, auc_metrics, fpr_metrics
        nonlocal selected_states, auc_states, fpr_states
        nonlocal frozen_selected_metrics, frozen_selected_states
        if int(epochs) <= 0:
            return
        reconstructor, reco_clf, ph_metrics, ph_states = train_recoonly_joint(
            reconstructor=reconstructor,
            reco_clf=reco_clf,
            train_loader=dl_train,
            val_loader=dl_val,
            device=device,
            stage_name=phase_name,
            freeze_reconstructor=bool(freeze_reco),
            epochs=int(epochs),
            patience=int(patience),
            lr_cls=float(args.stageC_lr_cls),
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
            apply_cls_weight=bool(apply_cls_weight),
            apply_reco_weight=bool(apply_reco_weight),
            val_weight_key="sample_weight_cls",
            use_weighted_val_selection=bool(use_weighted_val_selection),
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
        if bool(freeze_reco):
            frozen_selected_metrics = ph_metrics
            frozen_selected_states = ph_states.get("selected", {})

    if freeze_epochs > 0:
        _run_phase(
            phase_name="StageC-RecoOnly-FrozenReco",
            freeze_reco=True,
            epochs=int(freeze_epochs),
            patience=max(int(freeze_epochs) + 1, int(args.stageC_patience)),
            min_epochs=int(freeze_epochs),
        )
        if (frozen_selected_states or {}).get("reco") is not None:
            reconstructor.load_state_dict(frozen_selected_states["reco"])
        if (frozen_selected_states or {}).get("clf") is not None:
            reco_clf.load_state_dict(frozen_selected_states["clf"])

    _run_phase(
        phase_name="StageC-RecoOnly",
        freeze_reco=False,
        epochs=int(unfreeze_epochs if freeze_epochs > 0 else total_stagec_epochs),
        patience=int(args.stageC_patience),
        min_epochs=min(int(args.stageC_min_epochs), int(unfreeze_epochs if freeze_epochs > 0 else total_stagec_epochs)),
    )

    stageC_metrics = {
        "selection_metric": str(args.selection_metric).lower(),
        "val_metric_source": str(selected_metrics.get("val_metric_source", "unweighted")) if selected_metrics else "unweighted",
        "selected_val_fpr50": float(selected_metrics.get("selected_val_fpr50", float("nan"))) if selected_metrics else float("nan"),
        "selected_val_auc": float(selected_metrics.get("selected_val_auc", float("nan"))) if selected_metrics else float("nan"),
        "selected_val_fpr50_unweighted": float(selected_metrics.get("selected_val_fpr50_unweighted", float("nan"))) if selected_metrics else float("nan"),
        "selected_val_auc_unweighted": float(selected_metrics.get("selected_val_auc_unweighted", float("nan"))) if selected_metrics else float("nan"),
        "selected_val_fpr50_weighted": float(selected_metrics.get("selected_val_fpr50_weighted", float("nan"))) if selected_metrics else float("nan"),
        "selected_val_auc_weighted": float(selected_metrics.get("selected_val_auc_weighted", float("nan"))) if selected_metrics else float("nan"),
        "best_val_fpr50_seen": float(fpr_metrics.get("best_val_fpr50_seen", float("nan"))) if fpr_metrics else float("nan"),
        "best_val_auc_seen": float(auc_metrics.get("best_val_auc_seen", float("nan"))) if auc_metrics else float("nan"),
        "best_val_fpr50_seen_unweighted": float(fpr_metrics.get("best_val_fpr50_seen_unweighted", float("nan"))) if fpr_metrics else float("nan"),
        "best_val_auc_seen_unweighted": float(auc_metrics.get("best_val_auc_seen_unweighted", float("nan"))) if auc_metrics else float("nan"),
        "best_val_fpr50_seen_weighted": float(fpr_metrics.get("best_val_fpr50_seen_weighted", float("nan"))) if fpr_metrics else float("nan"),
        "best_val_auc_seen_weighted": float(auc_metrics.get("best_val_auc_seen_weighted", float("nan"))) if auc_metrics else float("nan"),
    }
    stageC_states = {
        "selected": {"clf": (selected_states or {}).get("clf"), "reco": (selected_states or {}).get("reco")},
        "auc": {"clf": (auc_states or {}).get("clf"), "reco": (auc_states or {}).get("reco")},
        "fpr50": {"clf": (fpr_states or {}).get("clf"), "reco": (fpr_states or {}).get("reco")},
        "frozen_selected": {"clf": (frozen_selected_states or {}).get("clf"), "reco": (frozen_selected_states or {}).get("reco")},
        "phase_reports": phase_reports,
    }

    # Evaluate frozen selected (if present).
    auc_frozen = float("nan")
    fpr30_frozen = float("nan")
    fpr50_frozen = float("nan")
    preds_frozen = np.array([], dtype=np.float32)
    labs_ref = None
    if stageC_states.get("frozen_selected", {}).get("clf") is not None and stageC_states.get("frozen_selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["frozen_selected"]["reco"])
        reco_clf.load_state_dict(stageC_states["frozen_selected"]["clf"])
        torch.save({"model": reconstructor.state_dict()}, save_root / "offline_reconstructor_recoonly_stagec_frozen_ckpt.pt")
        torch.save({"model": reco_clf.state_dict()}, save_root / "recoonly_classifier_stagec_frozen_ckpt.pt")
        auc_frozen, preds_frozen, labs_frozen, _ = eval_recoonly_joint_model(
            reconstructor=reconstructor,
            reco_clf=reco_clf,
            loader=dl_test,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        if len(labs_frozen) > 0 and np.isfinite(auc_frozen):
            fpr_fr, tpr_fr, _ = roc_curve(labs_frozen, preds_frozen)
            fpr30_frozen = float(fpr_at_target_tpr(fpr_fr, tpr_fr, 0.30))
            fpr50_frozen = float(fpr_at_target_tpr(fpr_fr, tpr_fr, 0.50))
        labs_ref = labs_frozen

    # Evaluate selected checkpoint.
    if stageC_states.get("selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["selected"]["reco"])
    if stageC_states.get("selected", {}).get("clf") is not None:
        reco_clf.load_state_dict(stageC_states["selected"]["clf"])

    auc_selected, preds_selected, labs_selected, _ = eval_recoonly_joint_model(
        reconstructor=reconstructor,
        reco_clf=reco_clf,
        loader=dl_test,
        device=device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    fpr30_selected = float("nan")
    fpr50_selected = float("nan")
    if len(labs_selected) > 0 and np.isfinite(auc_selected):
        fpr_sel, tpr_sel, _ = roc_curve(labs_selected, preds_selected)
        fpr30_selected = float(fpr_at_target_tpr(fpr_sel, tpr_sel, 0.30))
        fpr50_selected = float(fpr_at_target_tpr(fpr_sel, tpr_sel, 0.50))
    labs_ref = labs_selected if labs_ref is None else labs_ref

    # Evaluate best-val-fpr50 checkpoint for reference.
    auc_bestfpr = float("nan")
    fpr30_bestfpr = float("nan")
    fpr50_bestfpr = float("nan")
    if stageC_states.get("fpr50", {}).get("clf") is not None and stageC_states.get("fpr50", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["fpr50"]["reco"])
        reco_clf.load_state_dict(stageC_states["fpr50"]["clf"])
        auc_bestfpr, preds_bestfpr, labs_bestfpr, _ = eval_recoonly_joint_model(
            reconstructor=reconstructor,
            reco_clf=reco_clf,
            loader=dl_test,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        if len(labs_bestfpr) > 0 and np.isfinite(auc_bestfpr):
            fpr_bf, tpr_bf, _ = roc_curve(labs_bestfpr, preds_bestfpr)
            fpr30_bestfpr = float(fpr_at_target_tpr(fpr_bf, tpr_bf, 0.30))
            fpr50_bestfpr = float(fpr_at_target_tpr(fpr_bf, tpr_bf, 0.50))

    # Restore selected for saving.
    if stageC_states.get("selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["selected"]["reco"])
    if stageC_states.get("selected", {}).get("clf") is not None:
        reco_clf.load_state_dict(stageC_states["selected"]["clf"])

    torch.save({"model": reconstructor.state_dict()}, save_root / "offline_reconstructor_recoonly_stagec_selected_ckpt.pt")
    torch.save({"model": reco_clf.state_dict()}, save_root / "recoonly_classifier_stagec_selected_ckpt.pt")
    torch.save(reconstructor.state_dict(), save_root / "offline_reconstructor.pt")
    torch.save(reco_clf.state_dict(), save_root / "recoonly_classifier.pt")

    for fname in ["data_setup.json", "data_splits.npz", "teacher.pt", "baseline.pt", "hlt_stats.json"]:
        src = run_dir / fname
        if src.exists():
            try:
                shutil.copy2(src, save_root / fname)
            except Exception as e:
                print(f"Warning: failed to copy {src} -> {save_root / fname}: {e}")

    out_metrics = {
        "source_run_dir": str(run_dir),
        "source_reco_ckpt": str(reco_ckpt),
        "stageC_args": {
            "stageC_epochs": int(args.stageC_epochs),
            "stageC_patience": int(args.stageC_patience),
            "stageC_min_epochs": int(args.stageC_min_epochs),
            "stageC_freeze_reco_epochs": int(args.stageC_freeze_reco_epochs),
            "stageC_lr_cls": float(args.stageC_lr_cls),
            "stageC_lr_reco": float(args.stageC_lr_reco),
            "stageC_lambda_rank": float(args.stageC_lambda_rank),
            "lambda_reco": float(args.lambda_reco),
            "lambda_cons": float(args.lambda_cons),
            "selection_metric": str(args.selection_metric),
            "corrected_weight_floor": float(args.corrected_weight_floor),
            "use_corrected_flags": bool(args.use_corrected_flags),
            "recoonly_input_dim": int(reco_input_dim),
            "spec_bucket_enable": bool(args.spec_bucket_enable),
            "spec_bucket_count_low": int(args.spec_bucket_count_low),
            "spec_bucket_count_high": int(args.spec_bucket_count_high),
            "spec_bucket_p_hlt_threshold": float(args.spec_bucket_p_hlt_threshold),
            "spec_bucket_jet_pt_hlt_min": float(args.spec_bucket_jet_pt_hlt_min),
            "spec_bucket_w_neg": float(args.spec_bucket_w_neg),
            "spec_bucket_w_pos": float(args.spec_bucket_w_pos),
            "spec_bucket_w_other": float(args.spec_bucket_w_other),
            "spec_bucket_apply_cls_weight": bool(apply_cls_weight),
            "spec_bucket_apply_reco_weight": bool(apply_reco_weight),
            "spec_bucket_weighted_val_selection": bool(use_weighted_val_selection),
        },
        "data_reload": {
            "setup_source": "saved data_setup.json" if use_saved_data_setup else "cli args",
            "splits_source": splits_source,
            "seed_effective": int(eff_seed),
            "n_train_jets_effective": int(eff_n_train_jets),
            "offset_jets_effective": int(eff_offset_jets),
            "max_constits_effective": int(eff_max_constits),
            "train_files_used": [str(pth) for pth in train_files],
            "ignore_saved_data_setup": bool(args.ignore_saved_data_setup),
        },
        "stageC_metrics": stageC_metrics,
        "stageC_phase_reports": phase_reports,
        "test_stageC_frozen_selected": {
            "auc": float(auc_frozen),
            "fpr30": float(fpr30_frozen),
            "fpr50": float(fpr50_frozen),
        },
        "test_stageC_selected": {
            "auc": float(auc_selected),
            "fpr30": float(fpr30_selected),
            "fpr50": float(fpr50_selected),
        },
        "test_stageC_bestfpr50": {
            "auc": float(auc_bestfpr),
            "fpr30": float(fpr30_bestfpr),
            "fpr50": float(fpr50_bestfpr),
        },
        "specialist_bucket": {
            "enabled": bool(args.spec_bucket_enable),
            "train_summary": specialist_summary_train,
            "val_summary": specialist_summary_val,
            "test_summary": specialist_summary_test,
        },
        "test_teacher_loaded": teacher_metrics,
        "test_baseline_loaded": baseline_metrics,
    }

    with open(save_root / "stagec_recoonly_refine_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    np.savez_compressed(
        save_root / "results_recoonly.npz",
        labels=(labs_ref.astype(np.float32) if labs_ref is not None else np.array([], dtype=np.float32)),
        preds_stagec_frozen=preds_frozen.astype(np.float32),
        preds_stagec_selected=preds_selected.astype(np.float32),
        specialist_bucket_mask_test=specialist_bucket_mask[test_idx].astype(np.uint8),
        specialist_weight_test=sample_weight_cls[test_idx].astype(np.float32),
    )

    print("\n" + "=" * 70)
    print("FAST STAGE C RECO-ONLY RESULTS")
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
    if np.isfinite(auc_frozen):
        print(f"RecoOnly FrozenSelected: AUC={auc_frozen:.4f}, FPR30={fpr30_frozen:.6f}, FPR50={fpr50_frozen:.6f}")
    print(f"RecoOnly Selected: AUC={auc_selected:.4f}, FPR30={fpr30_selected:.6f}, FPR50={fpr50_selected:.6f}")
    if np.isfinite(auc_bestfpr):
        print(
            f"RecoOnly BestValFPR50: AUC={auc_bestfpr:.4f}, "
            f"FPR30={fpr30_bestfpr:.6f}, FPR50={fpr50_bestfpr:.6f}"
        )
    print(f"Saved to: {save_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JetClass V2: m2-style joint pipeline with constrained unmerge-attribute heads.

Design:
1) Load JetClass ROOT data and build train/val/test splits.
2) Build HLT-like corrupted view with the same controls used in
   run_train_jetclass_HLT_150k75k150k_stronger.sh.
3) Train teacher (offline) and baseline (HLT) with JetClassTransformer.
4) Stage A: pretrain unmerge-only reconstructor (predicts kinematics/actions/budgets).
   Reconstructor consumes richer per-token features (kin + pid + track fields),
   but target tokens are 4D kinematics.
5) Stage B/C: train multiclass dual-view classifier (JetClass-style dual transformer),
   first with frozen reconstructor (B), then joint finetune (C).

V2 adds constrained non-kinematic supervision for reconstructed (unmerged) tokens:
- split merge-mode logits (none / same-type / e+gamma / ch+nh),
- child type logits (constrained by merge policy),
- child charge and track-parameter regression for child slots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_jetclass_hlt_teacher_baseline import (
    HLTParams,
    JetClassTransformer,
    JetDataset,
    MERGE_MODE_NONE,
    N_MERGE_MODES,
    N_TYPES,
    TYPE_CH,
    TYPE_ELE,
    TYPE_MU,
    build_hlt_view,
    collect_files_by_class,
    compute_features,
    eval_epoch,
    eval_metrics,
    fit_model,
    get_mean_std,
    load_split,
    make_loader,
    set_seed,
    split_files_by_class,
    standardize,
    summarize_hlt_diagnostics,
)
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_RECO_CONFIG,
    OfflineReconstructor,
)
from offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly import (
    WeightedReconstructionDataset,
    build_soft_corrected_view,
    compute_reconstruction_losses_weighted,
    train_reconstructor_weighted,
    wrap_reconstructor_unmerge_only,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JetClass m2-style joint dual-view (V2 attr heads)")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"),
    )
    p.add_argument("--save_dir", type=Path, default=Path("checkpoints/jetclass_joint_dualview"))
    p.add_argument("--run_name", type=str, default="jetclass_joint_v2attr_150k75k150k_stronger")
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=2)

    # Data/feature
    p.add_argument("--feature_mode", type=str, default="full", choices=["kin", "kinpid", "full"])
    p.add_argument("--max_constits", type=int, default=128)
    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--shuffle_files", action="store_true", default=False)
    p.add_argument("--n_train_jets", type=int, default=150000)
    p.add_argument("--n_val_jets", type=int, default=75000)
    p.add_argument("--n_test_jets", type=int, default=150000)

    # Shared classifier (teacher/baseline/dual) architecture
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # Teacher/baseline optimization
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=3)

    # Binary-style reporting from multiclass probabilities
    p.add_argument("--target_class", type=str, default="HToBB")
    p.add_argument("--background_class", type=str, default="ZJetsToNuNu")

    # HLT corruption knobs (stronger profile defaults)
    p.add_argument("--hlt_pt_threshold", type=float, default=1.5)
    p.add_argument("--merge_prob_scale", type=float, default=1.20)
    p.add_argument("--reassign_scale", type=float, default=1.25)
    p.add_argument("--smear_scale", type=float, default=1.25)
    p.add_argument("--eff_plateau_barrel", type=float, default=0.95)
    p.add_argument("--eff_plateau_endcap", type=float, default=0.88)
    p.add_argument("--eff_turnon_pt", type=float, default=1.2)
    p.add_argument("--eff_width_pt", type=float, default=0.45)

    # Reconstructor (Stage A)
    p.add_argument("--reco_batch_size", type=int, default=96)
    p.add_argument("--stageA_epochs", type=int, default=90)
    p.add_argument("--stageA_patience", type=int, default=18)
    p.add_argument("--stageA_lr", type=float, default=2e-4)
    p.add_argument("--stageA_warmup_epochs", type=int, default=5)
    p.add_argument("--stageA_weight_decay", type=float, default=1e-5)
    p.add_argument("--stageA_stage1_epochs", type=int, default=20)
    p.add_argument("--stageA_stage2_epochs", type=int, default=55)
    p.add_argument("--stageA_min_full_scale_epochs", type=int, default=5)
    p.add_argument("--disable_stageA_stagewise_best_reload", action="store_true")

    # Reconstructor architecture
    p.add_argument("--reco_embed_dim", type=int, default=256)
    p.add_argument("--reco_num_heads", type=int, default=8)
    p.add_argument("--reco_num_layers", type=int, default=8)
    p.add_argument("--reco_ff_dim", type=int, default=1024)
    p.add_argument("--reco_dropout", type=float, default=0.1)
    p.add_argument("--reco_max_split_children", type=int, default=2)
    p.add_argument("--reco_max_generated_tokens", type=int, default=48)

    # V2: constrained non-kin heads for reconstructed (unmerged) children.
    p.add_argument("--v2_attr_hidden_dim", type=int, default=128)
    p.add_argument("--v2_attr_slots", type=int, default=2)
    p.add_argument("--v2_mode_none_weight", type=float, default=0.20)
    p.add_argument("--v2_mode_label_smoothing", type=float, default=0.0)
    p.add_argument("--v2_track_weight", type=float, default=1.0)

    # Stage-A extra calibration for attr heads.
    p.add_argument("--stageA_attr_epochs", type=int, default=12)
    p.add_argument("--stageA_attr_patience", type=int, default=4)
    p.add_argument("--stageA_attr_lr", type=float, default=2e-4)
    p.add_argument("--stageA_attr_weight_decay", type=float, default=1e-5)
    p.add_argument("--stageA_attr_unfreeze_base", action="store_true")

    # Stage B/C
    p.add_argument("--stageB_epochs", type=int, default=35)
    p.add_argument("--stageB_patience", type=int, default=10)
    p.add_argument("--stageB_min_epochs", type=int, default=10)
    p.add_argument("--stageB_lr_dual", type=float, default=4e-4)
    p.add_argument("--stageC_epochs", type=int, default=45)
    p.add_argument("--stageC_patience", type=int, default=12)
    p.add_argument("--stageC_min_epochs", type=int, default=15)
    p.add_argument("--stageC_lr_dual", type=float, default=2e-4)
    p.add_argument("--stageC_lr_reco", type=float, default=1e-4)
    p.add_argument("--lambda_reco", type=float, default=0.4)
    p.add_argument("--lambda_cons", type=float, default=0.06)
    p.add_argument("--lambda_attr_mode", type=float, default=0.10)
    p.add_argument("--lambda_attr_type", type=float, default=0.15)
    p.add_argument("--lambda_attr_charge", type=float, default=0.03)
    p.add_argument("--lambda_attr_track", type=float, default=0.03)
    p.add_argument("--corrected_weight_floor", type=float, default=1e-4)

    # Non-privileged budget target
    p.add_argument("--added_target_scale", type=float, default=0.90)

    # Set/loss config (m2-style knobs)
    p.add_argument(
        "--loss_set_mode",
        type=str,
        default="hungarian",
        choices=["chamfer", "chamfer_sinkhorn", "sinkhorn", "ot", "hungarian", "combo"],
    )
    p.add_argument("--loss_w_set", type=float, default=1.0)
    p.add_argument("--loss_w_phys", type=float, default=0.0)
    p.add_argument("--loss_w_pt_ratio", type=float, default=0.0)
    p.add_argument("--loss_w_m_ratio", type=float, default=0.0)
    p.add_argument("--loss_w_e_ratio", type=float, default=0.0)
    p.add_argument("--loss_w_budget", type=float, default=0.65)
    p.add_argument("--loss_w_sparse", type=float, default=0.02)
    p.add_argument("--loss_w_local", type=float, default=0.05)
    p.add_argument("--loss_unselected_penalty", type=float, default=0.0)
    p.add_argument("--loss_gen_local_radius", type=float, default=0.0)
    p.add_argument("--loss_set_sinkhorn_eps", type=float, default=0.08)
    p.add_argument("--loss_set_sinkhorn_iters", type=int, default=25)
    p.add_argument("--loss_set_ot_eps", type=float, default=0.02)
    p.add_argument("--loss_set_ot_iters", type=int, default=50)
    p.add_argument("--loss_set_combo_w_chamfer", type=float, default=1.0)
    p.add_argument("--loss_set_combo_w_sinkhorn", type=float, default=0.0)
    p.add_argument("--loss_set_combo_w_ot", type=float, default=0.0)
    p.add_argument("--loss_set_combo_w_hungarian", type=float, default=0.0)

    p.add_argument("--skip_save_models", action="store_true")
    return p.parse_args()


class ReconstructorWithAttrHeads(nn.Module):
    """
    V2 wrapper: keep original kinematic reconstructor outputs and add constrained
    per-parent split-attribute heads.
    """

    def __init__(
        self,
        base: nn.Module,
        input_dim: int,
        hidden_dim: int = 128,
        attr_slots: int = 2,
    ):
        super().__init__()
        self.base = base
        self.attr_slots = int(max(1, attr_slots))
        self.trunk = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.mode_head = nn.Linear(int(hidden_dim), int(N_MERGE_MODES))
        self.child_type_head = nn.Linear(int(hidden_dim), self.attr_slots * int(N_TYPES))
        self.child_charge_head = nn.Linear(int(hidden_dim), self.attr_slots)
        self.child_track_head = nn.Linear(int(hidden_dim), self.attr_slots * 4)

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        stage_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        out = self.base(feat_hlt, mask_hlt, const_hlt, stage_scale=stage_scale)
        h = self.trunk(feat_hlt)
        bsz, seq_len, _ = h.shape
        out["split_mode_logits"] = self.mode_head(h)
        out["child_type_logits"] = self.child_type_head(h).view(bsz, seq_len, self.attr_slots, int(N_TYPES))
        out["child_charge_pred"] = torch.tanh(self.child_charge_head(h)).view(bsz, seq_len, self.attr_slots)
        out["child_track_pred"] = self.child_track_head(h).view(bsz, seq_len, self.attr_slots, 4)
        return out


def compute_v2_attr_losses(
    reco_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    mode_none_weight: float,
    mode_label_smoothing: float,
    track_weight: float,
) -> Dict[str, torch.Tensor]:
    """
    Constrained attr losses:
      - mode supervised on all valid HLT parent tokens (none vs specific split mode),
      - child-type/charge/track supervised only on split-target parents.
    """
    device = reco_out["cand_tokens"].device
    zero = torch.zeros((), device=device)
    required = ("split_mode_logits", "child_type_logits", "child_charge_pred", "child_track_pred")
    if any(k not in reco_out for k in required):
        return {"mode": zero, "type": zero, "charge": zero, "track": zero, "total": zero}

    mask_hlt = batch["mask_hlt"].to(device)
    split_target_mask = batch["split_target_mask"].to(device)
    mode_target = batch["split_mode_target"].to(device)

    # Parent mode classification across all valid slots.
    valid = mask_hlt
    if valid.any():
        mode_logits = reco_out["split_mode_logits"][valid]
        mode_gt = mode_target[valid]
        cls_w = torch.tensor(
            [float(mode_none_weight), 1.0, 1.0, 1.0],
            dtype=torch.float32,
            device=device,
        )
        loss_mode = F.cross_entropy(
            mode_logits,
            mode_gt,
            weight=cls_w,
            label_smoothing=float(max(0.0, mode_label_smoothing)),
        )
    else:
        loss_mode = zero

    # Child losses only where true split target exists.
    sel = split_target_mask & mask_hlt
    if not sel.any():
        return {
            "mode": loss_mode,
            "type": zero,
            "charge": zero,
            "track": zero,
            "total": loss_mode,
        }

    child_type_logits = reco_out["child_type_logits"][:, :, :2, :]
    child_charge_pred = reco_out["child_charge_pred"][:, :, :2]
    child_track_pred = reco_out["child_track_pred"][:, :, :2, :]

    child_type_tgt = torch.stack(
        [
            batch["child_type_a_target"].to(device),
            batch["child_type_b_target"].to(device),
        ],
        dim=-1,
    )
    child_charge_tgt = torch.stack(
        [
            batch["child_charge_a_target"].to(device),
            batch["child_charge_b_target"].to(device),
        ],
        dim=-1,
    )
    child_track_tgt = torch.stack(
        [
            batch["child_track_a_target"].to(device),
            batch["child_track_b_target"].to(device),
        ],
        dim=2,
    )

    sel2 = sel.unsqueeze(-1).expand(-1, -1, 2)
    type_logits_sel = child_type_logits[sel2].reshape(-1, int(N_TYPES))
    type_tgt_sel = child_type_tgt[sel2].reshape(-1)
    loss_type = F.cross_entropy(type_logits_sel, type_tgt_sel)

    # Charge/track only meaningful for track-like child targets.
    track_like = (child_type_tgt == TYPE_CH) | (child_type_tgt == TYPE_ELE) | (child_type_tgt == TYPE_MU)
    charge_mask = sel2 & track_like
    if charge_mask.any():
        loss_charge = F.smooth_l1_loss(child_charge_pred[charge_mask], child_charge_tgt[charge_mask])
    else:
        loss_charge = zero

    track_mask = charge_mask.unsqueeze(-1).expand(-1, -1, -1, 4)
    if track_mask.any():
        loss_track = F.smooth_l1_loss(child_track_pred[track_mask], child_track_tgt[track_mask])
    else:
        loss_track = zero

    total = loss_mode + loss_type + loss_charge + float(track_weight) * loss_track
    return {
        "mode": loss_mode,
        "type": loss_type,
        "charge": loss_charge,
        "track": loss_track,
        "total": total,
    }


def train_stageA_attr_heads(
    reconstructor: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    freeze_base: bool,
    mode_none_weight: float,
    mode_label_smoothing: float,
    track_weight: float,
) -> Tuple[nn.Module, Dict[str, float]]:
    # Optionally freeze base reconstructor and train only attr heads.
    if hasattr(reconstructor, "base"):
        for p in reconstructor.base.parameters():
            p.requires_grad_(not bool(freeze_base))

    params = [p for p in reconstructor.parameters() if p.requires_grad]
    if not params:
        return reconstructor, {"best_val_attr": float("nan")}
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    best_val = float("inf")
    best_state = None
    wait = 0
    for ep in range(int(epochs)):
        reconstructor.train()
        tr_total = 0.0
        n_tr = 0
        for batch in train_loader:
            feat_hlt_reco = batch["feat_hlt_reco"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            opt.zero_grad()
            out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            losses = compute_v2_attr_losses(
                out,
                batch,
                mode_none_weight=float(mode_none_weight),
                mode_label_smoothing=float(mode_label_smoothing),
                track_weight=float(track_weight),
            )
            loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            bs = int(mask_hlt.shape[0])
            tr_total += float(loss.item()) * bs
            n_tr += bs
        tr_total /= max(n_tr, 1)

        reconstructor.eval()
        va_total = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                feat_hlt_reco = batch["feat_hlt_reco"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
                losses = compute_v2_attr_losses(
                    out,
                    batch,
                    mode_none_weight=float(mode_none_weight),
                    mode_label_smoothing=float(mode_label_smoothing),
                    track_weight=float(track_weight),
                )
                bs = int(mask_hlt.shape[0])
                va_total += float(losses["total"].item()) * bs
                n_va += bs
        va_total /= max(n_va, 1)

        if va_total < best_val:
            best_val = float(va_total)
            best_state = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep + 1) % 2 == 0:
            print(f"StageA-Attr ep {ep+1}: train_attr={tr_total:.4f}, val_attr={va_total:.4f}, best={best_val:.4f}")
        if wait >= int(patience):
            print(f"Early stopping StageA-Attr at epoch {ep+1}")
            break

    if best_state is not None:
        reconstructor.load_state_dict(best_state)
    # Restore base trainability for downstream stages.
    if hasattr(reconstructor, "base"):
        for p in reconstructor.base.parameters():
            p.requires_grad_(True)
    return reconstructor, {"best_val_attr": float(best_val)}


class JointDualDatasetMulti(Dataset):
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
        split_target_mask: np.ndarray,
        split_mode_target: np.ndarray,
        child_type_a_target: np.ndarray,
        child_type_b_target: np.ndarray,
        child_attr_a_target: np.ndarray,
        child_attr_b_target: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt_reco = torch.tensor(feat_hlt_reco, dtype=torch.float32)
        self.feat_hlt_dual = torch.tensor(feat_hlt_dual, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)
        self.split_target_mask = torch.tensor(split_target_mask, dtype=torch.bool)
        self.split_mode_target = torch.tensor(split_mode_target.astype(np.int64), dtype=torch.long)
        self.child_type_a_target = torch.tensor(child_type_a_target.astype(np.int64), dtype=torch.long)
        self.child_type_b_target = torch.tensor(child_type_b_target.astype(np.int64), dtype=torch.long)
        self.child_attr_a_target = torch.tensor(child_attr_a_target, dtype=torch.float32)
        self.child_attr_b_target = torch.tensor(child_attr_b_target, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

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
            "split_target_mask": self.split_target_mask[i],
            "split_mode_target": self.split_mode_target[i],
            "child_type_a_target": self.child_type_a_target[i],
            "child_type_b_target": self.child_type_b_target[i],
            "child_charge_a_target": self.child_attr_a_target[i, :, 0],
            "child_charge_b_target": self.child_attr_b_target[i, :, 0],
            "child_track_a_target": self.child_attr_a_target[i, :, 1:5],
            "child_track_b_target": self.child_attr_b_target[i, :, 1:5],
            "label": self.labels[i],
        }


class JetClassDualViewTransformer(nn.Module):
    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        n_classes: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim_a = int(input_dim_a)
        self.input_dim_b = int(input_dim_b)
        self.input_proj_a = nn.Sequential(
            nn.Linear(self.input_dim_a, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_proj_b = nn.Sequential(
            nn.Linear(self.input_dim_b, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        enc_layer_a = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        enc_layer_b = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder_a = nn.TransformerEncoder(enc_layer_a, num_layers=num_layers)
        self.encoder_b = nn.TransformerEncoder(enc_layer_b, num_layers=num_layers)

        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn_a = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.pool_attn_b = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_a_to_b = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_b_to_a = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim * 4)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, feat_a: torch.Tensor, mask_a: torch.Tensor, feat_b: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
        bsz, seq_len_a, _ = feat_a.shape
        _, seq_len_b, _ = feat_b.shape
        mask_a_safe = mask_a.clone()
        mask_b_safe = mask_b.clone()
        empty_a = ~mask_a_safe.any(dim=1)
        empty_b = ~mask_b_safe.any(dim=1)
        if empty_a.any():
            mask_a_safe[empty_a, 0] = True
        if empty_b.any():
            mask_b_safe[empty_b, 0] = True

        h_a = self.input_proj_a(feat_a.reshape(-1, self.input_dim_a)).view(bsz, seq_len_a, -1)
        h_b = self.input_proj_b(feat_b.reshape(-1, self.input_dim_b)).view(bsz, seq_len_b, -1)

        h_a = self.encoder_a(h_a, src_key_padding_mask=~mask_a_safe)
        h_b = self.encoder_b(h_b, src_key_padding_mask=~mask_b_safe)

        q = self.pool_query.expand(bsz, -1, -1)
        pooled_a, _ = self.pool_attn_a(q, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)
        pooled_b, _ = self.pool_attn_b(q, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
        cross_a, _ = self.cross_a_to_b(pooled_a, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
        cross_b, _ = self.cross_b_to_a(pooled_b, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)
        fused = torch.cat([pooled_a, pooled_b, cross_a, cross_b], dim=-1).squeeze(1)
        fused = self.norm(fused)
        return self.head(fused)


@torch.no_grad()
def eval_joint_multiclass(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    background_class: str,
    target_class: str,
    corrected_weight_floor: float,
) -> Dict[str, object]:
    reconstructor.eval()
    dual_model.eval()
    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    total = 0.0
    n = 0
    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        feat_hlt_dual = batch["feat_hlt_dual"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=False,
        )
        logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b)
        loss = F.cross_entropy(logits, y)
        bs = int(y.shape[0])
        total += float(loss.item()) * bs
        n += bs
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(class_names)), dtype=np.float32)
    ys = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,), dtype=np.int64)
    out: Dict[str, object] = {"loss": total / max(n, 1), "probs": probs, "labels": ys}
    if ys.size == 0:
        out.update({"acc": float("nan"), "auc_macro_ovr": float("nan"), "signal_vs_bg_fpr50": float("nan")})
        return out
    out.update(eval_metrics(ys, probs, class_names, background_class, target_class))
    return out


def train_joint_dual_multiclass(
    reconstructor: OfflineReconstructor,
    dual_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    freeze_reconstructor: bool,
    epochs: int,
    patience: int,
    min_epochs: int,
    lr_dual: float,
    lr_reco: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_reco: float,
    lambda_cons: float,
    lambda_attr_mode: float,
    lambda_attr_type: float,
    lambda_attr_charge: float,
    lambda_attr_track: float,
    loss_cfg: Dict,
    mode_none_weight: float,
    mode_label_smoothing: float,
    track_weight: float,
    class_names: Sequence[str],
    background_class: str,
    target_class: str,
    corrected_weight_floor: float,
) -> Tuple[OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, torch.Tensor]]]:
    for p in reconstructor.parameters():
        p.requires_grad_(not freeze_reconstructor)
    for p in dual_model.parameters():
        p.requires_grad_(True)

    params = [{"params": dual_model.parameters(), "lr": float(lr_dual)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_dual), weight_decay=float(weight_decay))

    def _lr_lambda(ep: int) -> float:
        if ep < int(warmup_epochs):
            return (ep + 1) / max(int(warmup_epochs), 1)
        x = (ep - int(warmup_epochs)) / max(int(epochs) - int(warmup_epochs), 1)
        return 0.5 * (1.0 + math.cos(math.pi * x))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    best_metric = float("-inf")
    best_val_auc = float("-inf")
    best_val_acc = float("-inf")
    best_state_dual = None
    best_state_reco = None
    wait = 0

    for ep in tqdm(range(int(epochs)), desc=stage_name):
        if freeze_reconstructor:
            reconstructor.eval()
        else:
            reconstructor.train()
        dual_model.train()

        tr_total = tr_cls = tr_reco = tr_cons = tr_attr = 0.0
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

            opt.zero_grad()

            if freeze_reconstructor:
                with torch.no_grad():
                    reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            else:
                reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)

            feat_b, mask_b = build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b)
            loss_cls = F.cross_entropy(logits, y)

            if float(lambda_reco) > 0.0:
                losses_reco = compute_reconstruction_losses_weighted(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_merge,
                    b_eff,
                    loss_cfg,
                    sample_weight=None,
                )
                loss_reco = losses_reco["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()
            losses_attr = compute_v2_attr_losses(
                reco_out,
                batch,
                mode_none_weight=float(mode_none_weight),
                mode_label_smoothing=float(mode_label_smoothing),
                track_weight=float(track_weight),
            )
            loss_attr = (
                float(lambda_attr_mode) * losses_attr["mode"]
                + float(lambda_attr_type) * losses_attr["type"]
                + float(lambda_attr_charge) * losses_attr["charge"]
                + float(lambda_attr_track) * losses_attr["track"]
            )
            loss = (
                loss_cls
                + float(lambda_reco) * loss_reco
                + float(lambda_cons) * loss_cons
                + loss_attr
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual_model.parameters(), 1.0)
            if not freeze_reconstructor:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = int(y.shape[0])
            tr_total += float(loss.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            tr_reco += float(loss_reco.item()) * bs
            tr_cons += float(loss_cons.item()) * bs
            tr_attr += float(loss_attr.item()) * bs
            n_tr += bs

        sch.step()

        tr_total /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)
        tr_attr /= max(n_tr, 1)

        va = eval_joint_multiclass(
            reconstructor=reconstructor,
            dual_model=dual_model,
            loader=val_loader,
            device=device,
            class_names=class_names,
            background_class=background_class,
            target_class=target_class,
            corrected_weight_floor=float(corrected_weight_floor),
        )

        va_auc = float(va["auc_macro_ovr"]) if np.isfinite(float(va["auc_macro_ovr"])) else float("nan")
        va_acc = float(va["acc"]) if np.isfinite(float(va["acc"])) else float("nan")
        metric = va_auc if np.isfinite(va_auc) else va_acc
        if np.isfinite(va_auc) and va_auc > best_val_auc:
            best_val_auc = va_auc
        if np.isfinite(va_acc) and va_acc > best_val_acc:
            best_val_acc = va_acc

        if np.isfinite(metric) and metric > best_metric:
            best_metric = float(metric)
            best_state_dual = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            wait = 0
        else:
            wait += 1

        print_every = 1 if stage_name.startswith("StageC") else 5
        if (ep + 1) % print_every == 0:
            print(
                f"{stage_name} ep {ep+1}: train(total/cls/reco/cons/attr)="
                f"{tr_total:.4f}/{tr_cls:.4f}/{tr_reco:.4f}/{tr_cons:.4f}/{tr_attr:.4f} | "
                f"val(loss/acc/auc/fpr50_sigbg/fpr50_ratio)="
                f"{float(va['loss']):.4f}/{float(va['acc']):.4f}/{float(va['auc_macro_ovr']):.4f}/"
                f"{float(va['signal_vs_bg_fpr50']):.6f}/{float(va['target_vs_bg_ratio_fpr50']):.6f} "
                f"best_metric={best_metric:.4f}"
            )

        if (ep + 1) >= int(min_epochs) and wait >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_dual is not None:
        dual_model.load_state_dict(best_state_dual)
    if best_state_reco is not None:
        reconstructor.load_state_dict(best_state_reco)

    metrics = {
        "best_val_metric": float(best_metric),
        "best_val_auc_macro_ovr": float(best_val_auc),
        "best_val_acc": float(best_val_acc),
    }
    states = {"dual": best_state_dual, "reco": best_state_reco}
    return reconstructor, dual_model, metrics, states


def run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(int(args.seed))
    save_root = (args.save_dir / args.run_name).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")

    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    files_by_class = collect_files_by_class(args.data_dir.resolve())
    class_names = sorted(files_by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    n_classes = len(class_names)
    print("Classes:")
    for c in class_names:
        print(f"  {c:12s} : {len(files_by_class[c])} files")

    tr_files, va_files, te_files = split_files_by_class(
        files_by_class,
        n_train=int(args.train_files_per_class),
        n_val=int(args.val_files_per_class),
        n_test=int(args.test_files_per_class),
        shuffle=bool(args.shuffle_files),
        seed=int(args.seed),
    )

    print("Loading train split...")
    tr_tok_raw, tr_mask_raw, tr_y = load_split(
        tr_files,
        n_total=int(args.n_train_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.seed) + 101,
    )
    print("Loading val split...")
    va_tok_raw, va_mask_raw, va_y = load_split(
        va_files,
        n_total=int(args.n_val_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.seed) + 202,
    )
    print("Loading test split...")
    te_tok_raw, te_mask_raw, te_y = load_split(
        te_files,
        n_total=int(args.n_test_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.seed) + 303,
    )

    print(
        f"Loaded jets: train={len(tr_y)}, val={len(va_y)}, test={len(te_y)} | "
        f"mean constituents train={tr_mask_raw.sum(axis=1).mean():.2f}"
    )

    hlt_params = HLTParams(
        hlt_pt_threshold=float(args.hlt_pt_threshold),
        merge_prob_scale=float(args.merge_prob_scale),
        reassign_scale=float(args.reassign_scale),
        smear_scale=float(args.smear_scale),
        eff_plateau_barrel=float(args.eff_plateau_barrel),
        eff_plateau_endcap=float(args.eff_plateau_endcap),
        eff_turnon_pt=float(args.eff_turnon_pt),
        eff_width_pt=float(args.eff_width_pt),
    )
    print("Building HLT-like corrupted splits...")
    tr_hlt_tok_raw, tr_hlt_mask_raw, tr_hlt_diag, tr_hlt_prov = build_hlt_view(
        tr_tok_raw,
        tr_mask_raw,
        params=hlt_params,
        seed=int(args.seed) + 1001,
        return_provenance=True,
    )
    va_hlt_tok_raw, va_hlt_mask_raw, va_hlt_diag, va_hlt_prov = build_hlt_view(
        va_tok_raw,
        va_mask_raw,
        params=hlt_params,
        seed=int(args.seed) + 1002,
        return_provenance=True,
    )
    te_hlt_tok_raw, te_hlt_mask_raw, te_hlt_diag, te_hlt_prov = build_hlt_view(
        te_tok_raw,
        te_mask_raw,
        params=hlt_params,
        seed=int(args.seed) + 1003,
        return_provenance=True,
    )

    tr_hlt_diag_summary = summarize_hlt_diagnostics(tr_hlt_diag)
    va_hlt_diag_summary = summarize_hlt_diagnostics(va_hlt_diag)
    te_hlt_diag_summary = summarize_hlt_diagnostics(te_hlt_diag)

    print(
        "Constituent means | offline/hlt: "
        f"train={tr_mask_raw.sum(axis=1).mean():.2f}/{tr_hlt_mask_raw.sum(axis=1).mean():.2f} "
        f"val={va_mask_raw.sum(axis=1).mean():.2f}/{va_hlt_mask_raw.sum(axis=1).mean():.2f} "
        f"test={te_mask_raw.sum(axis=1).mean():.2f}/{te_hlt_mask_raw.sum(axis=1).mean():.2f}"
    )
    print(
        "HLT drop decomposition (train): "
        f"eff={tr_hlt_diag_summary.get('drop_eff_share', float('nan')):.3f}, "
        f"thr={tr_hlt_diag_summary.get('drop_threshold_share', float('nan')):.3f}, "
        f"merge={tr_hlt_diag_summary.get('drop_merge_share', float('nan')):.3f}, "
        f"mean_merges/jet={tr_hlt_diag_summary.get('mean_merges_per_jet', float('nan')):.3f}"
    )
    tr_split_target_frac = float(np.mean(tr_hlt_prov["split_target_mask"]))
    va_split_target_frac = float(np.mean(va_hlt_prov["split_target_mask"]))
    te_split_target_frac = float(np.mean(te_hlt_prov["split_target_mask"]))
    print(
        "V2 split-target coverage: "
        f"train={tr_split_target_frac:.4f}, val={va_split_target_frac:.4f}, test={te_split_target_frac:.4f}"
    )

    # 4D kinematic views for reconstruction targets/inputs.
    tr_off_const4 = tr_tok_raw[:, :, :4].astype(np.float32)
    va_off_const4 = va_tok_raw[:, :, :4].astype(np.float32)
    te_off_const4 = te_tok_raw[:, :, :4].astype(np.float32)
    tr_hlt_const4 = tr_hlt_tok_raw[:, :, :4].astype(np.float32)
    va_hlt_const4 = va_hlt_tok_raw[:, :, :4].astype(np.float32)
    te_hlt_const4 = te_hlt_tok_raw[:, :, :4].astype(np.float32)

    # Features (reconstructor and tagger share the same feature tensor in V1).
    tr_feat_off = compute_features(tr_tok_raw, tr_mask_raw, feature_mode=args.feature_mode)
    va_feat_off = compute_features(va_tok_raw, va_mask_raw, feature_mode=args.feature_mode)
    te_feat_off = compute_features(te_tok_raw, te_mask_raw, feature_mode=args.feature_mode)
    tr_feat_hlt = compute_features(tr_hlt_tok_raw, tr_hlt_mask_raw, feature_mode=args.feature_mode)
    va_feat_hlt = compute_features(va_hlt_tok_raw, va_hlt_mask_raw, feature_mode=args.feature_mode)
    te_feat_hlt = compute_features(te_hlt_tok_raw, te_hlt_mask_raw, feature_mode=args.feature_mode)

    idx_all = np.arange(len(tr_y))
    mean, std = get_mean_std(tr_feat_off, tr_mask_raw, idx_all)
    tr_feat_off = standardize(tr_feat_off, tr_mask_raw, mean, std)
    va_feat_off = standardize(va_feat_off, va_mask_raw, mean, std)
    te_feat_off = standardize(te_feat_off, te_mask_raw, mean, std)
    tr_feat_hlt = standardize(tr_feat_hlt, tr_hlt_mask_raw, mean, std)
    va_feat_hlt = standardize(va_feat_hlt, va_hlt_mask_raw, mean, std)
    te_feat_hlt = standardize(te_feat_hlt, te_hlt_mask_raw, mean, std)

    # Non-privileged budget supervision from count gaps.
    added_scale = float(np.clip(float(args.added_target_scale), 0.0, 1.0))
    tr_true_added = np.maximum(tr_mask_raw.sum(axis=1) - tr_hlt_mask_raw.sum(axis=1), 0.0).astype(np.float32)
    va_true_added = np.maximum(va_mask_raw.sum(axis=1) - va_hlt_mask_raw.sum(axis=1), 0.0).astype(np.float32)
    te_true_added = np.maximum(te_mask_raw.sum(axis=1) - te_hlt_mask_raw.sum(axis=1), 0.0).astype(np.float32)
    tr_budget_merge = (added_scale * tr_true_added).astype(np.float32)
    va_budget_merge = (added_scale * va_true_added).astype(np.float32)
    te_budget_merge = (added_scale * te_true_added).astype(np.float32)
    tr_budget_eff = np.zeros_like(tr_budget_merge, dtype=np.float32)
    va_budget_eff = np.zeros_like(va_budget_merge, dtype=np.float32)
    te_budget_eff = np.zeros_like(te_budget_merge, dtype=np.float32)
    print(
        f"Non-priv target setup: added_target_scale={added_scale:.3f}, "
        f"mean_true_added_raw(train)={float(tr_true_added.mean()):.3f}, "
        f"mean_target_added(train)={float(tr_budget_merge.mean()):.3f}"
    )

    # Teacher/Baseline datasets.
    ds_tr_off = JetDataset(tr_feat_off, tr_mask_raw, tr_y)
    ds_va_off = JetDataset(va_feat_off, va_mask_raw, va_y)
    ds_te_off = JetDataset(te_feat_off, te_mask_raw, te_y)
    ds_tr_hlt = JetDataset(tr_feat_hlt, tr_hlt_mask_raw, tr_y)
    ds_va_hlt = JetDataset(va_feat_hlt, va_hlt_mask_raw, va_y)
    ds_te_hlt = JetDataset(te_feat_hlt, te_hlt_mask_raw, te_y)

    dl_tr_off = make_loader(ds_tr_off, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    dl_va_off = make_loader(ds_va_off, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_te_off = make_loader(ds_te_off, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_tr_hlt = make_loader(ds_tr_hlt, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    dl_va_hlt = make_loader(ds_va_hlt, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_te_hlt = make_loader(ds_te_hlt, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    input_dim = int(tr_feat_off.shape[-1])

    print("\n" + "=" * 70)
    print("STEP 1: TEACHER + BASELINE (JETCLASS TRANSFORMER)")
    print("=" * 70)
    teacher, teacher_val_best, hist_teacher = fit_model(
        train_loader=dl_tr_off,
        val_loader=dl_va_off,
        input_dim=input_dim,
        n_classes=n_classes,
        class_names=class_names,
        background_class=str(args.background_class),
        target_class=str(args.target_class),
        args=args,
        tag="teacher_offline",
        save_dir=save_root,
    )
    baseline, baseline_val_best, hist_baseline = fit_model(
        train_loader=dl_tr_hlt,
        val_loader=dl_va_hlt,
        input_dim=input_dim,
        n_classes=n_classes,
        class_names=class_names,
        background_class=str(args.background_class),
        target_class=str(args.target_class),
        args=args,
        tag="baseline_hlt",
        save_dir=save_root,
    )

    teacher = teacher.to(device)
    baseline = baseline.to(device)
    teacher_test = eval_epoch(
        teacher,
        dl_te_off,
        device=device,
        class_names=class_names,
        background_class=str(args.background_class),
        target_class=str(args.target_class),
    )
    baseline_test = eval_epoch(
        baseline,
        dl_te_hlt,
        device=device,
        class_names=class_names,
        background_class=str(args.background_class),
        target_class=str(args.target_class),
    )

    print("\n" + "=" * 70)
    print("STEP 2: STAGE A (RECONSTRUCTOR PRETRAIN)")
    print("=" * 70)

    reco_cfg = json.loads(json.dumps(BASE_RECO_CONFIG))
    reco_cfg["reconstructor_model"]["embed_dim"] = int(args.reco_embed_dim)
    reco_cfg["reconstructor_model"]["num_heads"] = int(args.reco_num_heads)
    reco_cfg["reconstructor_model"]["num_layers"] = int(args.reco_num_layers)
    reco_cfg["reconstructor_model"]["ff_dim"] = int(args.reco_ff_dim)
    reco_cfg["reconstructor_model"]["dropout"] = float(args.reco_dropout)
    reco_cfg["reconstructor_model"]["max_split_children"] = int(args.reco_max_split_children)
    reco_cfg["reconstructor_model"]["max_generated_tokens"] = int(args.reco_max_generated_tokens)

    reco_cfg["reconstructor_training"]["batch_size"] = int(args.reco_batch_size)
    reco_cfg["reconstructor_training"]["epochs"] = int(args.stageA_epochs)
    reco_cfg["reconstructor_training"]["patience"] = int(args.stageA_patience)
    reco_cfg["reconstructor_training"]["lr"] = float(args.stageA_lr)
    reco_cfg["reconstructor_training"]["weight_decay"] = float(args.stageA_weight_decay)
    reco_cfg["reconstructor_training"]["warmup_epochs"] = int(args.stageA_warmup_epochs)
    reco_cfg["reconstructor_training"]["stage1_epochs"] = int(args.stageA_stage1_epochs)
    reco_cfg["reconstructor_training"]["stage2_epochs"] = int(args.stageA_stage2_epochs)
    reco_cfg["reconstructor_training"]["min_full_scale_epochs"] = int(args.stageA_min_full_scale_epochs)

    reco_cfg["loss"]["set_loss_mode"] = str(args.loss_set_mode).strip().lower()
    reco_cfg["loss"]["w_set"] = float(args.loss_w_set)
    reco_cfg["loss"]["w_phys"] = float(args.loss_w_phys)
    reco_cfg["loss"]["w_pt_ratio"] = float(args.loss_w_pt_ratio)
    reco_cfg["loss"]["w_m_ratio"] = float(args.loss_w_m_ratio)
    reco_cfg["loss"]["w_e_ratio"] = float(args.loss_w_e_ratio)
    reco_cfg["loss"]["w_budget"] = float(args.loss_w_budget)
    reco_cfg["loss"]["w_sparse"] = float(args.loss_w_sparse)
    reco_cfg["loss"]["w_local"] = float(args.loss_w_local)
    reco_cfg["loss"]["unselected_penalty"] = float(args.loss_unselected_penalty)
    reco_cfg["loss"]["gen_local_radius"] = float(args.loss_gen_local_radius)
    reco_cfg["loss"]["sinkhorn_eps"] = float(args.loss_set_sinkhorn_eps)
    reco_cfg["loss"]["sinkhorn_iters"] = int(args.loss_set_sinkhorn_iters)
    reco_cfg["loss"]["ot_eps"] = float(args.loss_set_ot_eps)
    reco_cfg["loss"]["ot_iters"] = int(args.loss_set_ot_iters)
    reco_cfg["loss"]["combo_w_chamfer"] = float(args.loss_set_combo_w_chamfer)
    reco_cfg["loss"]["combo_w_sinkhorn"] = float(args.loss_set_combo_w_sinkhorn)
    reco_cfg["loss"]["combo_w_ot"] = float(args.loss_set_combo_w_ot)
    reco_cfg["loss"]["combo_w_hungarian"] = float(args.loss_set_combo_w_hungarian)

    base_reconstructor = OfflineReconstructor(
        input_dim=input_dim,
        **reco_cfg["reconstructor_model"],
    ).to(device)
    base_reconstructor = wrap_reconstructor_unmerge_only(base_reconstructor)
    reconstructor = ReconstructorWithAttrHeads(
        base=base_reconstructor,
        input_dim=input_dim,
        hidden_dim=int(args.v2_attr_hidden_dim),
        attr_slots=int(args.v2_attr_slots),
    ).to(device)

    ds_tr_reco = WeightedReconstructionDataset(
        tr_feat_hlt,
        tr_hlt_mask_raw,
        tr_hlt_const4,
        tr_off_const4,
        tr_mask_raw,
        tr_budget_merge,
        tr_budget_eff,
        sample_weight_reco=None,
    )
    ds_va_reco = WeightedReconstructionDataset(
        va_feat_hlt,
        va_hlt_mask_raw,
        va_hlt_const4,
        va_off_const4,
        va_mask_raw,
        va_budget_merge,
        va_budget_eff,
        sample_weight_reco=None,
    )
    dl_tr_reco = DataLoader(
        ds_tr_reco,
        batch_size=int(reco_cfg["reconstructor_training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_va_reco = DataLoader(
        ds_va_reco,
        batch_size=int(reco_cfg["reconstructor_training"]["batch_size"]),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    reconstructor, reco_val_metrics = train_reconstructor_weighted(
        model=reconstructor,
        train_loader=dl_tr_reco,
        val_loader=dl_va_reco,
        device=device,
        train_cfg=reco_cfg["reconstructor_training"],
        loss_cfg=reco_cfg["loss"],
        apply_reco_weight=False,
        reload_best_at_stage_transition=not bool(args.disable_stageA_stagewise_best_reload),
    )

    # Joint datasets.
    ds_tr_joint = JointDualDatasetMulti(
        tr_feat_hlt,
        tr_feat_hlt,
        tr_hlt_mask_raw,
        tr_hlt_const4,
        tr_off_const4,
        tr_mask_raw,
        tr_budget_merge,
        tr_budget_eff,
        tr_hlt_prov["split_target_mask"],
        tr_hlt_prov["split_mode_target"],
        tr_hlt_prov["child_type_a_target"],
        tr_hlt_prov["child_type_b_target"],
        tr_hlt_prov["child_attr_a_target"],
        tr_hlt_prov["child_attr_b_target"],
        tr_y,
    )
    ds_va_joint = JointDualDatasetMulti(
        va_feat_hlt,
        va_feat_hlt,
        va_hlt_mask_raw,
        va_hlt_const4,
        va_off_const4,
        va_mask_raw,
        va_budget_merge,
        va_budget_eff,
        va_hlt_prov["split_target_mask"],
        va_hlt_prov["split_mode_target"],
        va_hlt_prov["child_type_a_target"],
        va_hlt_prov["child_type_b_target"],
        va_hlt_prov["child_attr_a_target"],
        va_hlt_prov["child_attr_b_target"],
        va_y,
    )
    ds_te_joint = JointDualDatasetMulti(
        te_feat_hlt,
        te_feat_hlt,
        te_hlt_mask_raw,
        te_hlt_const4,
        te_off_const4,
        te_mask_raw,
        te_budget_merge,
        te_budget_eff,
        te_hlt_prov["split_target_mask"],
        te_hlt_prov["split_mode_target"],
        te_hlt_prov["child_type_a_target"],
        te_hlt_prov["child_type_b_target"],
        te_hlt_prov["child_attr_a_target"],
        te_hlt_prov["child_attr_b_target"],
        te_y,
    )
    dl_tr_joint = DataLoader(
        ds_tr_joint,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_va_joint = DataLoader(
        ds_va_joint,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_te_joint = DataLoader(
        ds_te_joint,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    # Optional Stage-A attr-head calibration using exact synthetic merge provenance.
    freeze_base = not bool(args.stageA_attr_unfreeze_base)
    reconstructor, reco_attr_metrics = train_stageA_attr_heads(
        reconstructor=reconstructor,
        train_loader=dl_tr_joint,
        val_loader=dl_va_joint,
        device=device,
        epochs=int(args.stageA_attr_epochs),
        patience=int(args.stageA_attr_patience),
        lr=float(args.stageA_attr_lr),
        weight_decay=float(args.stageA_attr_weight_decay),
        freeze_base=freeze_base,
        mode_none_weight=float(args.v2_mode_none_weight),
        mode_label_smoothing=float(args.v2_mode_label_smoothing),
        track_weight=float(args.v2_track_weight),
    )

    print("\n" + "=" * 70)
    print("STEP 3: STAGE B (DUAL PRETRAIN, FROZEN RECONSTRUCTOR)")
    print("=" * 70)
    dual_model = JetClassDualViewTransformer(
        input_dim_a=input_dim,
        input_dim_b=10,
        n_classes=n_classes,
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    reconstructor, dual_model, stageB_metrics, stageB_states = train_joint_dual_multiclass(
        reconstructor=reconstructor,
        dual_model=dual_model,
        train_loader=dl_tr_joint,
        val_loader=dl_va_joint,
        device=device,
        stage_name="StageB-DualPretrain",
        freeze_reconstructor=True,
        epochs=int(args.stageB_epochs),
        patience=int(args.stageB_patience),
        min_epochs=int(args.stageB_min_epochs),
        lr_dual=float(args.stageB_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(args.weight_decay),
        warmup_epochs=int(args.warmup_epochs),
        lambda_reco=0.0,
        lambda_cons=0.0,
        lambda_attr_mode=0.0,
        lambda_attr_type=0.0,
        lambda_attr_charge=0.0,
        lambda_attr_track=0.0,
        loss_cfg=reco_cfg["loss"],
        mode_none_weight=float(args.v2_mode_none_weight),
        mode_label_smoothing=float(args.v2_mode_label_smoothing),
        track_weight=float(args.v2_track_weight),
        class_names=class_names,
        background_class=str(args.background_class),
        target_class=str(args.target_class),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    stage2_test = eval_joint_multiclass(
        reconstructor,
        dual_model,
        dl_te_joint,
        device,
        class_names,
        str(args.background_class),
        str(args.target_class),
        float(args.corrected_weight_floor),
    )
    stage2_reco_state = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
    stage2_dual_state = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}

    print("\n" + "=" * 70)
    print("STEP 4: STAGE C (JOINT FINETUNE)")
    print("=" * 70)
    reconstructor, dual_model, stageC_metrics, stageC_states = train_joint_dual_multiclass(
        reconstructor=reconstructor,
        dual_model=dual_model,
        train_loader=dl_tr_joint,
        val_loader=dl_va_joint,
        device=device,
        stage_name="StageC-Joint",
        freeze_reconstructor=False,
        epochs=int(args.stageC_epochs),
        patience=int(args.stageC_patience),
        min_epochs=int(args.stageC_min_epochs),
        lr_dual=float(args.stageC_lr_dual),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(args.weight_decay),
        warmup_epochs=int(args.warmup_epochs),
        lambda_reco=float(args.lambda_reco),
        lambda_cons=float(args.lambda_cons),
        lambda_attr_mode=float(args.lambda_attr_mode),
        lambda_attr_type=float(args.lambda_attr_type),
        lambda_attr_charge=float(args.lambda_attr_charge),
        lambda_attr_track=float(args.lambda_attr_track),
        loss_cfg=reco_cfg["loss"],
        mode_none_weight=float(args.v2_mode_none_weight),
        mode_label_smoothing=float(args.v2_mode_label_smoothing),
        track_weight=float(args.v2_track_weight),
        class_names=class_names,
        background_class=str(args.background_class),
        target_class=str(args.target_class),
        corrected_weight_floor=float(args.corrected_weight_floor),
    )
    joint_test = eval_joint_multiclass(
        reconstructor,
        dual_model,
        dl_te_joint,
        device,
        class_names,
        str(args.background_class),
        str(args.target_class),
        float(args.corrected_weight_floor),
    )

    summary = {
        "class_names": class_names,
        "n_classes": int(n_classes),
        "split_sizes": {"train": int(len(tr_y)), "val": int(len(va_y)), "test": int(len(te_y))},
        "feature_mode": str(args.feature_mode),
        "hlt_params": {
            "hlt_pt_threshold": float(args.hlt_pt_threshold),
            "merge_prob_scale": float(args.merge_prob_scale),
            "reassign_scale": float(args.reassign_scale),
            "smear_scale": float(args.smear_scale),
            "eff_plateau_barrel": float(args.eff_plateau_barrel),
            "eff_plateau_endcap": float(args.eff_plateau_endcap),
            "eff_turnon_pt": float(args.eff_turnon_pt),
            "eff_width_pt": float(args.eff_width_pt),
        },
        "binary_metric_config": {
            "target_class": str(args.target_class),
            "background_class": str(args.background_class),
        },
        "v2_attr_config": {
            "attr_hidden_dim": int(args.v2_attr_hidden_dim),
            "attr_slots": int(args.v2_attr_slots),
            "mode_none_weight": float(args.v2_mode_none_weight),
            "mode_label_smoothing": float(args.v2_mode_label_smoothing),
            "stageA_attr_epochs": int(args.stageA_attr_epochs),
            "stageA_attr_patience": int(args.stageA_attr_patience),
            "stageA_attr_lr": float(args.stageA_attr_lr),
            "stageA_attr_weight_decay": float(args.stageA_attr_weight_decay),
            "stageA_attr_freeze_base": bool(freeze_base),
            "lambda_attr_mode": float(args.lambda_attr_mode),
            "lambda_attr_type": float(args.lambda_attr_type),
            "lambda_attr_charge": float(args.lambda_attr_charge),
            "lambda_attr_track": float(args.lambda_attr_track),
        },
        "constituent_stats": {
            "train_offline_mean": float(tr_mask_raw.sum(axis=1).mean()),
            "train_hlt_mean": float(tr_hlt_mask_raw.sum(axis=1).mean()),
            "val_offline_mean": float(va_mask_raw.sum(axis=1).mean()),
            "val_hlt_mean": float(va_hlt_mask_raw.sum(axis=1).mean()),
            "test_offline_mean": float(te_mask_raw.sum(axis=1).mean()),
            "test_hlt_mean": float(te_hlt_mask_raw.sum(axis=1).mean()),
        },
        "split_target_coverage": {
            "train": tr_split_target_frac,
            "val": va_split_target_frac,
            "test": te_split_target_frac,
        },
        "hlt_diagnostics": {
            "train": tr_hlt_diag_summary,
            "val": va_hlt_diag_summary,
            "test": te_hlt_diag_summary,
        },
        "teacher_val_best": teacher_val_best,
        "baseline_val_best": baseline_val_best,
        "stageA_reconstructor": reco_val_metrics,
        "stageA_reconstructor_attr": reco_attr_metrics,
        "stageB_joint": stageB_metrics,
        "stageC_joint": stageC_metrics,
        "test_metrics": {
            "teacher_on_offline": teacher_test,
            "baseline_on_hlt": baseline_test,
            "stage2_on_hlt": {k: v for k, v in stage2_test.items() if k not in ("probs", "labels")},
            "joint_on_hlt": {k: v for k, v in joint_test.items() if k not in ("probs", "labels")},
        },
        "added_target_scale": float(added_scale),
        "mean_true_added_raw_train": float(tr_true_added.mean()),
        "mean_target_added_train": float(tr_budget_merge.mean()),
    }

    with (save_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (save_root / "teacher_history.json").open("w", encoding="utf-8") as f:
        json.dump(hist_teacher, f, indent=2)
    with (save_root / "baseline_history.json").open("w", encoding="utf-8") as f:
        json.dump(hist_baseline, f, indent=2)
    with (save_root / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    np.savez_compressed(save_root / "hlt_diagnostics_train_perjet.npz", **tr_hlt_diag)
    np.savez_compressed(save_root / "hlt_diagnostics_val_perjet.npz", **va_hlt_diag)
    np.savez_compressed(save_root / "hlt_diagnostics_test_perjet.npz", **te_hlt_diag)
    np.savez_compressed(save_root / "hlt_unmerge_targets_train.npz", **tr_hlt_prov)
    np.savez_compressed(save_root / "hlt_unmerge_targets_val.npz", **va_hlt_prov)
    np.savez_compressed(save_root / "hlt_unmerge_targets_test.npz", **te_hlt_prov)
    np.savez_compressed(
        save_root / "joint_test_scores.npz",
        labels=np.asarray(joint_test["labels"], dtype=np.int64),
        probs=np.asarray(joint_test["probs"], dtype=np.float32),
    )

    if not bool(args.skip_save_models):
        torch.save({"model": teacher.state_dict()}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict()}, save_root / "baseline.pt")
        torch.save({"model": stage2_reco_state}, save_root / "offline_reconstructor_stage2.pt")
        torch.save({"model": stage2_dual_state}, save_root / "dual_joint_stage2.pt")
        torch.save({"model": reconstructor.state_dict()}, save_root / "offline_reconstructor.pt")
        torch.save({"model": dual_model.state_dict()}, save_root / "dual_joint.pt")

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(
        f"Teacher (Offline): acc={teacher_test['acc']:.4f}, auc_macro={teacher_test['auc_macro_ovr']:.4f}, "
        f"fpr50(sig-vs-bg)={teacher_test['signal_vs_bg_fpr50']:.6f}, "
        f"fpr50({args.target_class}/({args.target_class}+{args.background_class}))="
        f"{teacher_test['target_vs_bg_ratio_fpr50']:.6f}"
    )
    print(
        f"Baseline (HLT):   acc={baseline_test['acc']:.4f}, auc_macro={baseline_test['auc_macro_ovr']:.4f}, "
        f"fpr50(sig-vs-bg)={baseline_test['signal_vs_bg_fpr50']:.6f}, "
        f"fpr50({args.target_class}/({args.target_class}+{args.background_class}))="
        f"{baseline_test['target_vs_bg_ratio_fpr50']:.6f}"
    )
    print(
        f"Stage2 (PreJoint): acc={float(stage2_test['acc']):.4f}, auc_macro={float(stage2_test['auc_macro_ovr']):.4f}, "
        f"fpr50(sig-vs-bg)={float(stage2_test['signal_vs_bg_fpr50']):.6f}, "
        f"fpr50({args.target_class}/({args.target_class}+{args.background_class}))="
        f"{float(stage2_test['target_vs_bg_ratio_fpr50']):.6f}"
    )
    print(
        f"Joint (StageC):    acc={float(joint_test['acc']):.4f}, auc_macro={float(joint_test['auc_macro_ovr']):.4f}, "
        f"fpr50(sig-vs-bg)={float(joint_test['signal_vs_bg_fpr50']):.6f}, "
        f"fpr50({args.target_class}/({args.target_class}+{args.background_class}))="
        f"{float(joint_test['target_vs_bg_ratio_fpr50']):.6f}"
    )
    print(f"StageA Attr best val loss: {float(reco_attr_metrics.get('best_val_attr', float('nan'))):.4f}")
    print(f"Saved outputs to: {save_root}")
    return summary


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

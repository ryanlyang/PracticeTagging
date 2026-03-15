#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge-only (no-smearing) HLT -> Offline reconstruction pipeline
without constituent-to-constituent mapping.

What this script does:
1) Load offline jets from HDF5 (jet-level pairing only).
2) Generate pseudo-HLT jets (merge + efficiency only; smearing/reassignment disabled),
   but DO NOT keep direct constituent ancestry mapping.
3) Train a heavy structured Offline Reconstructor with shared relpos transformer backbone and
   multi-head outputs (actions, split, generation, budget).
   In this variant, unsmear/reassign actions are disabled so non-merged tokens stay unchanged.
4) Build reconstructed jets from model predictions.
5) Train/evaluate the same top tagger suite as unmerge_correct_hlt.py:
   Teacher, Baseline, Unmerge, Unmerge+MF, DualView, DualView+MF, DualView+KD, DualView+MF+KD.
6) Save checkpoints, ROC plots, response plots, and results.npz in the same style.
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
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from tqdm import tqdm

from unmerge_correct_hlt import (
    RANDOM_SEED,
    RelPosEncoderLayer,
    load_raw_constituents_from_h5,
    compute_features,
    get_stats,
    standardize,
    compute_jet_pt,
    build_pt_edges,
    jet_response_resolution,
    plot_response_resolution,
    JetDataset,
    DualViewJetDataset,
    DualViewKDDataset,
    ParticleTransformer,
    DualViewCrossAttnClassifier,
    get_scheduler,
    train_classifier,
    train_classifier_dual,
    eval_classifier,
    eval_classifier_dual,
    train_kd_epoch_dual,
    evaluate_kd_dual,
    evaluate_bce_loss_dual,
    self_train_student_dual,
)


# ----------------------------- Reproducibility ----------------------------- #
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


CONFIG = {
    "hlt_effects": {
        "pt_threshold_hlt": 1.5,
        "pt_threshold_offline": 0.5,
        "merge_enabled": True,
        "merge_radius": 0.01,
        "eta_break": 1.5,
        "eff_plateau_barrel": 0.98,
        "eff_plateau_endcap": 0.94,
        "eff_pt50_barrel": 1.6,
        "eff_pt50_endcap": 1.9,
        "eff_width_barrel": 0.20,
        "eff_width_endcap": 0.25,
        "eff_density_alpha": 0.055,
        "eff_quality_min": 0.90,
        "eff_quality_max": 1.06,
        "eff_floor": 0.02,
        "eff_ceil": 0.995,
        "smearing_enabled": False,
        "smear_a": 0.0,
        "smear_b": 0.0,
        "smear_c": 0.0,
        "smear_eta_scale": 0.0,
        "smear_sigma_min": 0.0,
        "smear_sigma_max": 0.0,
        "eta_smear_const": 0.0,
        "eta_smear_inv_sqrt": 0.0,
        "phi_smear_const": 0.0,
        "phi_smear_inv_sqrt": 0.0,
        "tail_base": 0.0,
        "tail_eta_coeff": 0.0,
        "tail_density_coeff": 0.0,
        "tail_prob_max": 0.0,
        "tail_mu": 0.98,
        "tail_sigma_scale": 2.5,
        "tail_sigma_add": 0.015,
        "pt_resp_min": 0.40,
        "pt_resp_max": 1.60,
        "density_radius": 0.04,
        "reassign_prob_base": 0.0,
        "reassign_density_coeff": 0.0,
        "reassign_prob_max": 0.0,
        "reassign_radius": 0.08,
        "reassign_strength_min": 0.20,
        "reassign_strength_max": 0.65,
        "jet_quality_sigma": 0.08,
        "jet_quality_min": 0.75,
        "jet_quality_max": 1.35,
        "post_smear_pt_threshold": 0.0,
    },
    "reconstructor_model": {
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 8,
        "ff_dim": 1024,
        "dropout": 0.1,
        "max_split_children": 2,
        "max_generated_tokens": 48,
    },
    "reconstructor_training": {
        "batch_size": 96,
        "epochs": 110,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 5,
        "patience": 20,
        "stage1_epochs": 20,
        "stage2_epochs": 55,
    },
    "loss": {
        "w_set": 1.0,
        "w_phys": 0.35,
        "w_pt_ratio": 0.70,
        "w_e_ratio": 0.35,
        "w_budget": 0.65,
        "w_sparse": 0.02,
        "w_local": 0.03,
        "unselected_penalty": 0.35,
        "gen_local_radius": 0.30,
    },
    "model": {
        "embed_dim": 128,
        "num_heads": 8,
        "num_layers": 6,
        "ff_dim": 512,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 512,
        "epochs": 60,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 3,
        "patience": 15,
    },
    "kd": {
        "temperature": 7.0,
        "alpha_kd": 0.5,
        "alpha_attn": 0.05,
        "alpha_rep": 0.10,
        "alpha_nce": 0.10,
        "tau_nce": 0.10,
        "conf_weighted": True,
        "adaptive_alpha": True,
        "alpha_warmup": 0.0,
        "alpha_stable_patience": 2,
        "alpha_stable_delta": 1e-4,
        "alpha_warmup_min_epochs": 3,
        "ema_teacher": True,
        "ema_decay": 0.995,
        "self_train": True,
        "self_train_source": "teacher",
        "self_train_epochs": 5,
        "self_train_lr": 1e-4,
        "self_train_conf_min": 0.0,
        "self_train_conf_power": 1.0,
        "self_train_patience": 5,
    },
}


def wrap_phi_np(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def wrap_phi_t(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _compute_local_density_np(
    eta: np.ndarray,
    phi: np.ndarray,
    valid_idx: np.ndarray,
    radius: float,
) -> np.ndarray:
    if len(valid_idx) <= 1:
        return np.zeros(len(valid_idx), dtype=np.float32)
    eta_v = eta[valid_idx]
    phi_v = phi[valid_idx]
    deta = eta_v[:, None] - eta_v[None, :]
    dphi = wrap_phi_np(phi_v[:, None] - phi_v[None, :])
    dR = np.sqrt(deta * deta + dphi * dphi)
    neigh = (dR < radius).astype(np.int32)
    np.fill_diagonal(neigh, 0)
    return neigh.sum(axis=1).astype(np.float32)


def apply_hlt_effects_realistic_nomap(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: Dict,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict[str, np.ndarray]]:
    """Realistic pseudo-HLT generation without constituent ancestry tracking."""
    rs = np.random.RandomState(int(seed))
    hcfg = cfg["hlt_effects"]
    n_jets, max_part, _ = const.shape

    hlt = const.copy()
    hlt_mask = mask.copy()

    n_initial = int(hlt_mask.sum())
    merge_lost_per_jet = np.zeros(n_jets, dtype=np.float32)
    eff_lost_per_jet = np.zeros(n_jets, dtype=np.float32)

    # 1) Pre-threshold
    pt_threshold = float(hcfg["pt_threshold_hlt"])
    below = (hlt[:, :, 0] < pt_threshold) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0
    n_lost_threshold_pre = int(below.sum())

    # 2) Local merging
    n_merged = 0
    merge_radius = float(hcfg["merge_radius"])
    if hcfg["merge_enabled"] and merge_radius > 0:
        for j in range(n_jets):
            valid_idx = np.where(hlt_mask[j])[0]
            if len(valid_idx) < 2:
                continue
            to_remove = set()
            for ii in range(len(valid_idx)):
                a = valid_idx[ii]
                if a in to_remove:
                    continue
                for jj in range(ii + 1, len(valid_idx)):
                    b = valid_idx[jj]
                    if b in to_remove:
                        continue
                    deta = hlt[j, a, 1] - hlt[j, b, 1]
                    dphi = wrap_phi_np(hlt[j, a, 2] - hlt[j, b, 2])
                    dR = float(np.sqrt(deta * deta + dphi * dphi))
                    if dR >= merge_radius:
                        continue

                    pt_a = float(hlt[j, a, 0])
                    pt_b = float(hlt[j, b, 0])
                    pt_sum = pt_a + pt_b
                    if pt_sum <= 1e-8:
                        continue
                    wa = pt_a / pt_sum
                    wb = pt_b / pt_sum

                    hlt[j, a, 0] = pt_sum
                    hlt[j, a, 1] = wa * hlt[j, a, 1] + wb * hlt[j, b, 1]
                    phi_a = hlt[j, a, 2]
                    phi_b = hlt[j, b, 2]
                    hlt[j, a, 2] = np.arctan2(
                        wa * np.sin(phi_a) + wb * np.sin(phi_b),
                        wa * np.cos(phi_a) + wb * np.cos(phi_b),
                    )
                    hlt[j, a, 3] = hlt[j, a, 3] + hlt[j, b, 3]
                    to_remove.add(b)
                    n_merged += 1
                    merge_lost_per_jet[j] += 1.0

            for idx in to_remove:
                hlt_mask[j, idx] = False
                hlt[j, idx] = 0

    # 3) Efficiency model
    jet_q = rs.lognormal(mean=0.0, sigma=float(hcfg["jet_quality_sigma"]), size=n_jets).astype(np.float32)
    jet_q = np.clip(jet_q, float(hcfg["jet_quality_min"]), float(hcfg["jet_quality_max"]))

    density = np.zeros((n_jets, max_part), dtype=np.float32)
    for j in range(n_jets):
        valid = np.where(hlt_mask[j])[0]
        if len(valid) == 0:
            continue
        density_j = _compute_local_density_np(
            eta=hlt[j, :, 1],
            phi=hlt[j, :, 2],
            valid_idx=valid,
            radius=float(hcfg["density_radius"]),
        )
        density[j, valid] = density_j

    abs_eta = np.abs(hlt[:, :, 1])
    pt = np.maximum(hlt[:, :, 0], 1e-8)
    eta_plateau = np.where(
        abs_eta < hcfg["eta_break"],
        hcfg["eff_plateau_barrel"],
        hcfg["eff_plateau_endcap"],
    ).astype(np.float32)
    pt50 = np.where(
        abs_eta < hcfg["eta_break"],
        hcfg["eff_pt50_barrel"],
        hcfg["eff_pt50_endcap"],
    ).astype(np.float32)
    width = np.where(
        abs_eta < hcfg["eta_break"],
        hcfg["eff_width_barrel"],
        hcfg["eff_width_endcap"],
    ).astype(np.float32)
    turn_on = 1.0 / (1.0 + np.exp(-(pt - pt50) / np.maximum(width, 1e-6)))
    density_term = np.exp(-float(hcfg["eff_density_alpha"]) * density)
    q_eff = np.clip(jet_q[:, None], float(hcfg["eff_quality_min"]), float(hcfg["eff_quality_max"]))
    eps = eta_plateau * turn_on * density_term * q_eff
    eps = np.clip(eps, float(hcfg["eff_floor"]), float(hcfg["eff_ceil"]))

    u = rs.random_sample((n_jets, max_part))
    lost_eff = (u > eps) & hlt_mask
    hlt_mask[lost_eff] = False
    hlt[lost_eff] = 0
    eff_lost_per_jet = lost_eff.sum(axis=1).astype(np.float32)
    n_lost_eff = int(lost_eff.sum())

    # 4) Optional smearing + tails + local reassignment
    n_reassigned = 0
    if bool(hcfg.get("smearing_enabled", True)):
        for j in range(n_jets):
            valid = np.where(hlt_mask[j])[0]
            if len(valid) == 0:
                continue

            pt_j = np.maximum(hlt[j, valid, 0], 1e-8)
            eta_j = hlt[j, valid, 1]
            phi_j = hlt[j, valid, 2]
            abs_eta_j = np.abs(eta_j)
            dens_j = density[j, valid]

            eta_scale = 1.0 + float(hcfg["smear_eta_scale"]) * abs_eta_j
            q = float(jet_q[j])

            sigma_rel = np.sqrt(
                (float(hcfg["smear_a"]) / np.sqrt(pt_j)) ** 2
                + float(hcfg["smear_b"]) ** 2
                + (float(hcfg["smear_c"]) / pt_j) ** 2
            )
            sigma_rel = sigma_rel * eta_scale * q
            sigma_rel = np.clip(sigma_rel, float(hcfg["smear_sigma_min"]), float(hcfg["smear_sigma_max"]))

            tail_prob = (
                float(hcfg["tail_base"])
                + float(hcfg["tail_eta_coeff"]) * abs_eta_j
                + float(hcfg["tail_density_coeff"]) * dens_j
            )
            tail_prob = np.clip(tail_prob, 0.0, float(hcfg["tail_prob_max"]))
            is_tail = rs.random_sample(len(valid)) < tail_prob

            ratio = rs.normal(loc=1.0, scale=sigma_rel)
            tail_sigma = float(hcfg["tail_sigma_scale"]) * sigma_rel + float(hcfg["tail_sigma_add"])
            ratio_tail = rs.normal(loc=float(hcfg["tail_mu"]), scale=tail_sigma)
            ratio[is_tail] = ratio_tail[is_tail]
            ratio = np.clip(ratio, float(hcfg["pt_resp_min"]), float(hcfg["pt_resp_max"]))
            pt_new = np.clip(pt_j * ratio, 1e-8, None)

            sigma_eta = (
                float(hcfg["eta_smear_const"]) + float(hcfg["eta_smear_inv_sqrt"]) / np.sqrt(pt_j)
            ) * eta_scale * q
            sigma_phi = (
                float(hcfg["phi_smear_const"]) + float(hcfg["phi_smear_inv_sqrt"]) / np.sqrt(pt_j)
            ) * eta_scale * q

            eta_new = eta_j + rs.normal(loc=0.0, scale=sigma_eta)
            phi_new = wrap_phi_np(phi_j + rs.normal(loc=0.0, scale=sigma_phi))

            if float(hcfg["reassign_prob_base"]) > 0.0 and len(valid) > 1:
                p_reassign = float(hcfg["reassign_prob_base"]) + float(hcfg["reassign_density_coeff"]) * dens_j
                p_reassign = np.clip(p_reassign, 0.0, float(hcfg["reassign_prob_max"]))
                do_reassign = rs.random_sample(len(valid)) < p_reassign
                for ii in np.where(do_reassign)[0]:
                    deta = eta_new[ii] - eta_new
                    dphi = wrap_phi_np(phi_new[ii] - phi_new)
                    dR = np.sqrt(deta * deta + dphi * dphi)
                    dR[ii] = 1e9
                    nn = int(np.argmin(dR))
                    if dR[nn] > float(hcfg["reassign_radius"]):
                        continue
                    lam = rs.uniform(
                        float(hcfg["reassign_strength_min"]),
                        float(hcfg["reassign_strength_max"]),
                    )
                    eta_new[ii] = (1.0 - lam) * eta_new[ii] + lam * eta_new[nn]
                    phi_new[ii] = np.arctan2(
                        (1.0 - lam) * np.sin(phi_new[ii]) + lam * np.sin(phi_new[nn]),
                        (1.0 - lam) * np.cos(phi_new[ii]) + lam * np.cos(phi_new[nn]),
                    )
                    n_reassigned += 1

            eta_new = np.clip(eta_new, -5.0, 5.0)
            phi_new = wrap_phi_np(phi_new)
            e_new = pt_new * np.cosh(eta_new)

            hlt[j, valid, 0] = pt_new
            hlt[j, valid, 1] = eta_new
            hlt[j, valid, 2] = phi_new
            hlt[j, valid, 3] = e_new

    # 5) Optional post-smear threshold
    post_thr = float(hcfg["post_smear_pt_threshold"])
    n_lost_threshold_post = 0
    if post_thr > 0:
        below_post = (hlt[:, :, 0] < post_thr) & hlt_mask
        hlt_mask[below_post] = False
        hlt[below_post] = 0
        n_lost_threshold_post = int(below_post.sum())

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    hlt[~hlt_mask] = 0.0

    stats = {
        "n_jets": int(n_jets),
        "n_initial": int(n_initial),
        "n_lost_threshold_pre": int(n_lost_threshold_pre),
        "n_merged_pairs": int(n_merged),
        "n_lost_eff": int(n_lost_eff),
        "n_reassigned": int(n_reassigned),
        "n_lost_threshold_post": int(n_lost_threshold_post),
        "n_final": int(hlt_mask.sum()),
        "avg_offline_per_jet": float(mask.sum(axis=1).mean()),
        "avg_hlt_per_jet": float(hlt_mask.sum(axis=1).mean()),
    }
    budget_truth = {
        "merge_lost_per_jet": merge_lost_per_jet.astype(np.float32),
        "eff_lost_per_jet": eff_lost_per_jet.astype(np.float32),
    }
    return hlt.astype(np.float32), hlt_mask.astype(bool), stats, budget_truth


class ReconstructionDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
        }


class ReconstructInputDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray, const_hlt: np.ndarray):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)

    def __len__(self) -> int:
        return self.feat_hlt.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
        }


class OfflineReconstructor(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_split_children: int = 2,
        max_generated_tokens: int = 48,
    ):
        super().__init__()
        self.max_split_children = int(max_split_children)
        self.max_generated_tokens = int(max_generated_tokens)
        self.num_heads = int(num_heads)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.relpos_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, num_heads),
        )

        self.encoder_layers = nn.ModuleList(
            [RelPosEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.token_norm = nn.LayerNorm(embed_dim)

        # Token action heads
        self.action_head = nn.Linear(embed_dim, 4)  # keep, unsmear, split, reassign
        self.unsmear_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
        )
        self.reassign_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),
        )

        # Split heads
        self.split_exist_head = nn.Linear(embed_dim, self.max_split_children)
        self.split_delta_head = nn.Linear(embed_dim, self.max_split_children * 3)

        # Jet context + budget
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.budget_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )  # total, merge-added, eff-added

        # Generation head
        self.gen_queries = nn.Parameter(torch.randn(1, self.max_generated_tokens, embed_dim) * 0.02)
        self.gen_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.gen_norm = nn.LayerNorm(embed_dim)
        self.gen_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
        )
        self.gen_exist_head = nn.Linear(embed_dim, 1)

    def _build_relpos_bias(self, const_raw: torch.Tensor) -> torch.Tensor:
        # const_raw: [B, L, 4] with [pt, eta, phi, E]
        eta = const_raw[:, :, 1]
        phi = const_raw[:, :, 2]
        deta = eta[:, :, None] - eta[:, None, :]
        dphi = torch.atan2(
            torch.sin(phi[:, :, None] - phi[:, None, :]),
            torch.cos(phi[:, :, None] - phi[:, None, :]),
        )
        dR = torch.sqrt(deta.pow(2) + dphi.pow(2) + 1e-8)
        rel = torch.stack([deta, dphi, dR], dim=-1)
        bias = self.relpos_mlp(rel)  # [B, L, L, H]
        bias = bias.permute(0, 3, 1, 2).contiguous()  # [B, H, L, L]
        return bias

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        stage_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        B, L, _ = feat_hlt.shape
        mask_safe = mask_hlt.clone()
        empty = ~mask_safe.any(dim=1)
        if empty.any():
            mask_safe[empty, 0] = True

        x = self.input_proj(feat_hlt)
        rel_bias = self._build_relpos_bias(const_hlt)
        for layer in self.encoder_layers:
            x = layer(x, mask_safe, rel_bias)
        x = self.token_norm(x)

        action_logits = self.action_head(x)
        action_prob = torch.softmax(action_logits, dim=-1)
        p_keep_raw = action_prob[..., 0]
        p_split_raw = action_prob[..., 2]
        keep_split_norm = (p_keep_raw + p_split_raw + eps)
        p_keep = p_keep_raw / keep_split_norm
        p_split = p_split_raw / keep_split_norm
        p_unsmear = torch.zeros_like(p_keep)
        p_reassign = torch.zeros_like(p_keep)

        pt = const_hlt[..., 0].clamp(min=eps)
        eta = const_hlt[..., 1].clamp(min=-5.0, max=5.0)
        phi = const_hlt[..., 2]
        E = const_hlt[..., 3].clamp(min=eps)

        tok_pt = pt
        tok_eta = eta
        tok_phi = phi
        tok_E = E

        tok_tokens = torch.stack([tok_pt, tok_eta, tok_phi, tok_E], dim=-1)
        tok_weight = (p_keep + 0.15 * p_split).clamp(0.0, 1.0)
        tok_weight = tok_weight * mask_hlt.float()
        tok_merge_flag = torch.zeros_like(tok_weight)
        tok_eff_flag = torch.zeros_like(tok_weight)

        # Split children from each HLT token
        K = self.max_split_children
        split_exist = torch.sigmoid(self.split_exist_head(x))
        split_exist = split_exist * (p_split.unsqueeze(-1) * stage_scale) * mask_hlt.float().unsqueeze(-1)

        split_delta = self.split_delta_head(x).view(B, L, K, 3)
        split_frac = torch.sigmoid(split_delta[..., 0]) / float(K + 1)
        child_pt = pt.unsqueeze(-1) * split_frac
        child_eta = (eta.unsqueeze(-1) + 0.25 * torch.tanh(split_delta[..., 1])).clamp(min=-5.0, max=5.0)
        child_phi = wrap_phi_t(phi.unsqueeze(-1) + 0.25 * torch.tanh(split_delta[..., 2]))
        child_E = child_pt * torch.cosh(child_eta)

        child_tokens = torch.stack([child_pt, child_eta, child_phi, child_E], dim=-1).reshape(B, L * K, 4)
        child_weight = split_exist.reshape(B, L * K).clamp(0.0, 1.0)
        child_merge_flag = torch.ones_like(child_weight)
        child_eff_flag = torch.zeros_like(child_weight)

        # Jet context and budget heads
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=~mask_safe, need_weights=False)
        ctx = pooled.squeeze(1)
        budget_raw = self.budget_head(ctx)
        budget_total = F.softplus(budget_raw[:, 0])
        budget_merge = F.softplus(budget_raw[:, 1])
        budget_eff = F.softplus(budget_raw[:, 2])

        # Generation slots (efficiency-loss recovery)
        gq = self.gen_queries.expand(B, -1, -1)
        gen_dec, _ = self.gen_attn(gq, x, x, key_padding_mask=~mask_safe, need_weights=False)
        gen_dec = self.gen_norm(gen_dec)
        gen_raw = self.gen_head(gen_dec)
        gen_exist = torch.sigmoid(self.gen_exist_head(gen_dec).squeeze(-1)) * stage_scale

        gen_pt = torch.exp(torch.clamp(gen_raw[..., 0], min=-8.0, max=6.0))
        gen_eta = gen_raw[..., 1].clamp(min=-5.0, max=5.0)
        gen_phi = wrap_phi_t(gen_raw[..., 2])
        gen_E = torch.exp(torch.clamp(gen_raw[..., 3], min=-8.0, max=10.0))
        gen_E = torch.maximum(gen_E, gen_pt * torch.cosh(gen_eta))
        gen_tokens = torch.stack([gen_pt, gen_eta, gen_phi, gen_E], dim=-1)
        gen_merge_flag = torch.zeros_like(gen_exist)
        gen_eff_flag = torch.ones_like(gen_exist)

        # Budget-informed calibration of generated/split weights.
        child_sum = child_weight.sum(dim=1, keepdim=True) + eps
        gen_sum = gen_exist.sum(dim=1, keepdim=True) + eps
        child_scale = (budget_merge.unsqueeze(1) / child_sum).clamp(min=0.25, max=4.0)
        gen_scale = (budget_eff.unsqueeze(1) / gen_sum).clamp(min=0.25, max=4.0)
        child_weight = (child_weight * child_scale).clamp(0.0, 1.0)
        gen_exist = (gen_exist * gen_scale).clamp(0.0, 1.0)

        cand_tokens = torch.cat([tok_tokens, child_tokens, gen_tokens], dim=1)
        cand_weights = torch.cat([tok_weight, child_weight, gen_exist], dim=1)
        cand_merge_flags = torch.cat([tok_merge_flag, child_merge_flag, gen_merge_flag], dim=1)
        cand_eff_flags = torch.cat([tok_eff_flag, child_eff_flag, gen_eff_flag], dim=1)

        return {
            "cand_tokens": cand_tokens,
            "cand_weights": cand_weights,
            "cand_merge_flags": cand_merge_flags,
            "cand_eff_flags": cand_eff_flags,
            "action_prob": action_prob,
            "child_weight": child_weight,
            "gen_weight": gen_exist,
            "budget_total": budget_total,
            "budget_merge": budget_merge,
            "budget_eff": budget_eff,
            "split_delta": split_delta,
            "gen_tokens": gen_tokens,
        }


def _token_cost_matrix(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Return pairwise token cost matrix: [B, N_pred, N_tgt]."""
    eps = 1e-8
    p_pt = pred[:, :, 0].clamp(min=eps).unsqueeze(2)
    t_pt = tgt[:, :, 0].clamp(min=eps).unsqueeze(1)

    p_eta = pred[:, :, 1].unsqueeze(2)
    t_eta = tgt[:, :, 1].unsqueeze(1)

    p_phi = pred[:, :, 2].unsqueeze(2)
    t_phi = tgt[:, :, 2].unsqueeze(1)

    p_E = pred[:, :, 3].clamp(min=eps).unsqueeze(2)
    t_E = tgt[:, :, 3].clamp(min=eps).unsqueeze(1)

    d_logpt = torch.abs(torch.log(p_pt) - torch.log(t_pt))
    d_eta = torch.abs(p_eta - t_eta)
    d_phi = torch.abs(torch.atan2(torch.sin(p_phi - t_phi), torch.cos(p_phi - t_phi)))
    d_logE = torch.abs(torch.log(p_E) - torch.log(t_E))

    return d_logpt + 0.60 * d_eta + 0.60 * d_phi + 0.25 * d_logE


def _weighted_fourvec_sums(const: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    pt = const[:, :, 0]
    eta = const[:, :, 1]
    phi = const[:, :, 2]
    E = const[:, :, 3]

    px = (w * pt * torch.cos(phi)).sum(dim=1)
    py = (w * pt * torch.sin(phi)).sum(dim=1)
    pz = (w * pt * torch.sinh(eta)).sum(dim=1)
    Es = (w * E).sum(dim=1)
    return px, py, pz, Es


def compute_reconstruction_losses(
    out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    eps = 1e-8

    pred = out["cand_tokens"]
    w = out["cand_weights"].clamp(0.0, 1.0)

    # Set-level weighted Chamfer-like loss.
    cost = _token_cost_matrix(pred, const_off)
    valid_tgt = mask_off.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))

    pred_to_tgt = cost.min(dim=2).values
    loss_pred_to_tgt = (w * pred_to_tgt).sum(dim=1) / (w.sum(dim=1) + eps)

    penalty = float(loss_cfg["unselected_penalty"]) * (1.0 - w).unsqueeze(2)
    tgt_to_pred = (cost + penalty).min(dim=1).values
    tgt_mask_f = mask_off.float()
    loss_tgt_to_pred = (tgt_to_pred * tgt_mask_f).sum(dim=1) / (tgt_mask_f.sum(dim=1) + eps)

    loss_set = (loss_pred_to_tgt + loss_tgt_to_pred).mean()

    # Physics consistency on full jet 4-vector.
    pred_px, pred_py, pred_pz, pred_E = _weighted_fourvec_sums(pred, w)
    true_px, true_py, true_pz, true_E = _weighted_fourvec_sums(const_off, mask_off.float())

    norm = (
        true_px.abs()
        + true_py.abs()
        + true_pz.abs()
        + true_E.abs()
        + 1.0
    )
    loss_phys = (
        (pred_px - true_px).abs()
        + (pred_py - true_py).abs()
        + (pred_pz - true_pz).abs()
        + (pred_E - true_E).abs()
    ) / norm
    loss_phys = loss_phys.mean()
    pred_pt = torch.sqrt(pred_px.pow(2) + pred_py.pow(2) + eps)
    true_pt = torch.sqrt(true_px.pow(2) + true_py.pow(2) + eps)
    pt_ratio = pred_pt / (true_pt + eps)
    loss_pt_ratio = F.smooth_l1_loss(pt_ratio, torch.ones_like(pt_ratio))
    e_ratio = pred_E / (true_E + eps)
    loss_e_ratio = F.smooth_l1_loss(e_ratio, torch.ones_like(e_ratio))

    # Budget/count losses.
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

    loss_budget = (
        F.smooth_l1_loss(pred_count, true_count)
        + F.smooth_l1_loss(budget_total, true_count)
        + F.smooth_l1_loss(pred_added, true_added)
        + F.smooth_l1_loss(budget_merge + budget_eff, true_added)
        + F.smooth_l1_loss(budget_merge, budget_merge_true)
        + F.smooth_l1_loss(budget_eff, budget_eff_true)
        + F.smooth_l1_loss(pred_added_merge, budget_merge_true)
        + F.smooth_l1_loss(pred_added_eff, budget_eff_true)
        + F.smooth_l1_loss(budget_merge + budget_eff, budget_true_total)
    )

    # Sparsity regularization (avoid gratuitous split/gen actions).
    loss_sparse = out["child_weight"].mean() + out["gen_weight"].mean()

    # Locality regularization.
    split_delta = out["split_delta"]
    split_step = torch.sqrt(split_delta[..., 1].pow(2) + split_delta[..., 2].pow(2) + 1e-8)
    split_w = out["child_weight"].reshape(split_step.shape)
    loss_local_split = (split_w * split_step).sum() / (split_w.sum() + eps)

    gen_tokens = out["gen_tokens"]
    gen_w = out["gen_weight"]
    h_eta = const_hlt[:, :, 1]
    h_phi = const_hlt[:, :, 2]
    g_eta = gen_tokens[:, :, 1].unsqueeze(2)
    g_phi = gen_tokens[:, :, 2].unsqueeze(2)
    d_eta = g_eta - h_eta.unsqueeze(1)
    d_phi = torch.atan2(torch.sin(g_phi - h_phi.unsqueeze(1)), torch.cos(g_phi - h_phi.unsqueeze(1)))
    dR = torch.sqrt(d_eta.pow(2) + d_phi.pow(2) + 1e-8)
    dR = torch.where(mask_hlt.unsqueeze(1), dR, torch.full_like(dR, 1e4))
    nearest = dR.min(dim=2).values
    excess = F.relu(nearest - float(loss_cfg["gen_local_radius"]))
    loss_local_gen = (gen_w * excess).sum() / (gen_w.sum() + eps)
    loss_local = loss_local_split + loss_local_gen

    total = (
        float(loss_cfg["w_set"]) * loss_set
        + float(loss_cfg["w_phys"]) * loss_phys
        + float(loss_cfg["w_pt_ratio"]) * loss_pt_ratio
        + float(loss_cfg["w_e_ratio"]) * loss_e_ratio
        + float(loss_cfg["w_budget"]) * loss_budget
        + float(loss_cfg["w_sparse"]) * loss_sparse
        + float(loss_cfg["w_local"]) * loss_local
    )

    return {
        "total": total,
        "set": loss_set,
        "phys": loss_phys,
        "pt_ratio": loss_pt_ratio,
        "e_ratio": loss_e_ratio,
        "budget": loss_budget,
        "sparse": loss_sparse,
        "local": loss_local,
    }


def stage_scale(epoch: int, cfg: Dict) -> float:
    s1 = int(cfg["stage1_epochs"])
    s2 = int(cfg["stage2_epochs"])
    if epoch < s1:
        return 0.35
    if epoch < s2:
        return 0.70
    return 1.0


def train_reconstructor(
    model: OfflineReconstructor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
) -> Tuple[OfflineReconstructor, Dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    sch = get_scheduler(opt, int(train_cfg["warmup_epochs"]), int(train_cfg["epochs"]))

    best_state = None
    best_val = 1e9
    no_improve = 0
    best_metrics: Dict[str, float] = {}

    for ep in tqdm(range(int(train_cfg["epochs"])), desc="Reconstructor"):
        model.train()
        sc = stage_scale(ep, train_cfg)

        tr_total = 0.0
        tr_set = 0.0
        tr_phys = 0.0
        tr_pt_ratio = 0.0
        tr_e_ratio = 0.0
        tr_budget = 0.0
        tr_sparse = 0.0
        tr_local = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            budget_merge_true = batch["budget_merge_true"].to(device)
            budget_eff_true = batch["budget_eff_true"].to(device)

            opt.zero_grad()
            out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=sc)
            losses = compute_reconstruction_losses(
                out,
                const_hlt,
                mask_hlt,
                const_off,
                mask_off,
                budget_merge_true,
                budget_eff_true,
                loss_cfg,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = feat_hlt.size(0)
            tr_total += losses["total"].item() * bs
            tr_set += losses["set"].item() * bs
            tr_phys += losses["phys"].item() * bs
            tr_pt_ratio += losses["pt_ratio"].item() * bs
            tr_e_ratio += losses["e_ratio"].item() * bs
            tr_budget += losses["budget"].item() * bs
            tr_sparse += losses["sparse"].item() * bs
            tr_local += losses["local"].item() * bs
            n_tr += bs

        model.eval()
        va_total = 0.0
        va_set = 0.0
        va_phys = 0.0
        va_pt_ratio = 0.0
        va_e_ratio = 0.0
        va_budget = 0.0
        va_sparse = 0.0
        va_local = 0.0
        n_va = 0

        with torch.no_grad():
            for batch in val_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                const_off = batch["const_off"].to(device)
                mask_off = batch["mask_off"].to(device)
                budget_merge_true = batch["budget_merge_true"].to(device)
                budget_eff_true = batch["budget_eff_true"].to(device)

                out = model(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
                losses = compute_reconstruction_losses(
                    out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    budget_merge_true,
                    budget_eff_true,
                    loss_cfg,
                )

                bs = feat_hlt.size(0)
                va_total += losses["total"].item() * bs
                va_set += losses["set"].item() * bs
                va_phys += losses["phys"].item() * bs
                va_pt_ratio += losses["pt_ratio"].item() * bs
                va_e_ratio += losses["e_ratio"].item() * bs
                va_budget += losses["budget"].item() * bs
                va_sparse += losses["sparse"].item() * bs
                va_local += losses["local"].item() * bs
                n_va += bs

        sch.step()

        tr_total /= max(n_tr, 1)
        tr_set /= max(n_tr, 1)
        tr_phys /= max(n_tr, 1)
        tr_pt_ratio /= max(n_tr, 1)
        tr_e_ratio /= max(n_tr, 1)
        tr_budget /= max(n_tr, 1)
        tr_sparse /= max(n_tr, 1)
        tr_local /= max(n_tr, 1)

        va_total /= max(n_va, 1)
        va_set /= max(n_va, 1)
        va_phys /= max(n_va, 1)
        va_pt_ratio /= max(n_va, 1)
        va_e_ratio /= max(n_va, 1)
        va_budget /= max(n_va, 1)
        va_sparse /= max(n_va, 1)
        va_local /= max(n_va, 1)

        if va_total < best_val:
            best_val = va_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_metrics = {
                "val_total": va_total,
                "val_set": va_set,
                "val_phys": va_phys,
                "val_pt_ratio": va_pt_ratio,
                "val_e_ratio": va_e_ratio,
                "val_budget": va_budget,
                "val_sparse": va_sparse,
                "val_local": va_local,
            }
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"Ep {ep+1}: "
                f"train_total={tr_total:.4f}, val_total={va_total:.4f}, best={best_val:.4f} | "
                f"set={va_set:.4f}, phys={va_phys:.4f}, pt_ratio={va_pt_ratio:.4f}, e_ratio={va_e_ratio:.4f}, budget={va_budget:.4f}, "
                f"sparse={va_sparse:.4f}, local={va_local:.4f}, stage_scale={sc:.2f}"
            )

        if no_improve >= int(train_cfg["patience"]):
            print(f"Early stopping reconstructor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


def reconstruct_dataset(
    model: OfflineReconstructor,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    max_constits: int,
    device: torch.device,
    batch_size: int,
    weight_threshold: float = 0.03,
    use_budget_topk: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = ReconstructInputDataset(feat_hlt, mask_hlt, const_hlt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    reco_const = np.zeros((feat_hlt.shape[0], max_constits, 4), dtype=np.float32)
    reco_mask = np.zeros((feat_hlt.shape[0], max_constits), dtype=bool)
    reco_merge_flag = np.zeros((feat_hlt.shape[0], max_constits), dtype=np.float32)
    reco_eff_flag = np.zeros((feat_hlt.shape[0], max_constits), dtype=np.float32)
    created_merge_count = np.zeros(feat_hlt.shape[0], dtype=np.int32)
    created_eff_count = np.zeros(feat_hlt.shape[0], dtype=np.int32)
    pred_budget_total = np.zeros(feat_hlt.shape[0], dtype=np.float32)
    pred_budget_merge = np.zeros(feat_hlt.shape[0], dtype=np.float32)
    pred_budget_eff = np.zeros(feat_hlt.shape[0], dtype=np.float32)

    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="ReconstructAll"):
            x = batch["feat_hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            c = batch["const_hlt"].to(device)
            out = model(x, m, c, stage_scale=1.0)

            cand = out["cand_tokens"].cpu().numpy()
            w = out["cand_weights"].cpu().numpy()
            merge_flags = out["cand_merge_flags"].cpu().numpy()
            eff_flags = out["cand_eff_flags"].cpu().numpy()
            budget_total = out["budget_total"].cpu().numpy()
            budget_merge = out["budget_merge"].cpu().numpy()
            budget_eff = out["budget_eff"].cpu().numpy()
            n_tok = x.shape[1]
            n_child = out["child_weight"].shape[1]
            gen_start = int(n_tok + n_child)

            bsz = cand.shape[0]
            for i in range(bsz):
                order = np.argsort(-w[i])
                if use_budget_topk:
                    k_budget = int(np.clip(np.rint(float(budget_total[i])), 1, max_constits))
                    picked = [int(idx) for idx in order[:k_budget]]
                    if len(picked) == 0:
                        picked = [int(order[0])]
                else:
                    picked = []
                    for idx in order:
                        if len(picked) >= max_constits:
                            break
                        if w[i, idx] < weight_threshold and len(picked) > 0:
                            break
                        picked.append(int(idx))
                    if len(picked) == 0:
                        picked = [int(order[0])]

                n = min(len(picked), max_constits)
                sel = picked[:n]
                reco_const[offset + i, :n] = cand[i, sel]
                reco_mask[offset + i, :n] = True
                reco_merge_flag[offset + i, :n] = np.clip(merge_flags[i, sel], 0.0, 1.0)
                reco_eff_flag[offset + i, :n] = np.clip(eff_flags[i, sel], 0.0, 1.0)
                sel_arr = np.array(sel, dtype=np.int64)
                created_merge_count[offset + i] = int(np.sum((sel_arr >= n_tok) & (sel_arr < gen_start)))
                created_eff_count[offset + i] = int(np.sum(sel_arr >= gen_start))
                pred_budget_total[offset + i] = float(budget_total[i])
                pred_budget_merge[offset + i] = float(budget_merge[i])
                pred_budget_eff[offset + i] = float(budget_eff[i])

            offset += bsz

    reco_const = np.nan_to_num(reco_const, nan=0.0, posinf=0.0, neginf=0.0)
    reco_const[~reco_mask] = 0.0
    return (
        reco_const,
        reco_mask,
        reco_merge_flag,
        reco_eff_flag,
        created_merge_count,
        created_eff_count,
        pred_budget_total,
        pred_budget_merge,
        pred_budget_eff,
    )


def train_single_view_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    sch = get_scheduler(opt, train_cfg["warmup_epochs"], train_cfg["epochs"])
    best_auc = 0.0
    best_state = None
    no_improve = 0

    for ep in tqdm(range(train_cfg["epochs"]), desc=name):
        _, tr_auc = train_classifier(model, train_loader, opt, device)
        va_auc, _, _ = eval_classifier(model, val_loader, device)
        sch.step()

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_auc:.4f}")
        if no_improve >= train_cfg["patience"]:
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_dual_view_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    name: str,
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    sch = get_scheduler(opt, train_cfg["warmup_epochs"], train_cfg["epochs"])
    best_auc = 0.0
    best_state = None
    no_improve = 0

    for ep in tqdm(range(train_cfg["epochs"]), desc=name):
        _, tr_auc = train_classifier_dual(model, train_loader, opt, device)
        va_auc, _, _ = eval_classifier_dual(model, val_loader, device)
        sch.step()

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, best={best_auc:.4f}")
        if no_improve >= train_cfg["patience"]:
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_dual_kd_student(
    student: nn.Module,
    teacher: nn.Module,
    kd_train_loader: DataLoader,
    kd_val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    kd_cfg: Dict,
    name: str,
    run_self_train: bool,
) -> nn.Module:
    opt = torch.optim.AdamW(student.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    sch = get_scheduler(opt, train_cfg["warmup_epochs"], train_cfg["epochs"])

    best_auc = 0.0
    best_state = None
    no_improve = 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None

    for ep in tqdm(range(train_cfg["epochs"]), desc=name):
        current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
        kd_cfg_ep = dict(kd_cfg)
        kd_cfg_ep["alpha_kd"] = current_alpha

        _, tr_auc = train_kd_epoch_dual(student, teacher, kd_train_loader, opt, device, kd_cfg_ep)
        va_auc, _, _ = evaluate_kd_dual(student, kd_val_loader, device)
        sch.step()

        if not kd_active and kd_cfg["adaptive_alpha"]:
            va_loss = evaluate_bce_loss_dual(student, kd_val_loader, device)
            if prev_val_loss is not None and abs(prev_val_loss - va_loss) < kd_cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = va_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
                print(f"Activating KD ramp at epoch {ep+1} (val_loss={va_loss:.4f})")

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"{name} ep {ep+1}: train_auc={tr_auc:.4f}, val_auc={va_auc:.4f}, "
                f"best={best_auc:.4f} | alpha_kd={current_alpha:.2f}"
            )
        if no_improve >= train_cfg["patience"]:
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    if run_self_train:
        print(f"\nSelf-train {name}...")
        opt_st = torch.optim.AdamW(student.parameters(), lr=kd_cfg["self_train_lr"])
        best_auc_st = best_auc
        no_improve = 0
        for ep in range(kd_cfg["self_train_epochs"]):
            st_loss = self_train_student_dual(student, teacher, kd_train_loader, opt_st, device, kd_cfg)
            va_auc, _, _ = evaluate_kd_dual(student, kd_val_loader, device)
            if va_auc > best_auc_st:
                best_auc_st = va_auc
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if (ep + 1) % 2 == 0:
                print(f"Self ep {ep+1}: loss={st_loss:.4f}, val_auc={va_auc:.4f}, best={best_auc_st:.4f}")
            if no_improve >= kd_cfg["self_train_patience"]:
                break
        if best_state is not None:
            student.load_state_dict(best_state)

    return student


def plot_roc(lines, out_path: Path, min_fpr: float):
    min_fpr = max(float(min_fpr), 1e-8)
    plt.figure(figsize=(8, 6))
    for tpr, fpr, style, label, color in lines:
        fpr_plot = np.clip(fpr, min_fpr, 1.0)
        plt.plot(tpr, fpr_plot, style, label=label, color=color, linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.yscale("log")
    plt.ylim(min_fpr, 1.0)
    plt.xlim(0.0, 1.0)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def fpr_at_target_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    order = np.argsort(tpr)
    t = tpr[order]
    f = fpr[order]
    # Collapse duplicate TPR values by taking minimum FPR at each TPR.
    t_unique = np.unique(t)
    f_unique = np.empty_like(t_unique)
    for i, tv in enumerate(t_unique):
        f_unique[i] = np.min(f[t == tv])
    if target_tpr <= t_unique[0]:
        return float(f_unique[0])
    if target_tpr >= t_unique[-1]:
        return float(f_unique[-1])
    return float(np.interp(target_tpr, t_unique, f_unique))


def plot_constituent_count_diagnostics(
    save_root: Path,
    mask_off: np.ndarray,
    hlt_mask: np.ndarray,
    reco_mask: np.ndarray,
    created_merge_count: np.ndarray,
    created_eff_count: np.ndarray,
    hlt_stats: Dict,
) -> Dict[str, float]:
    off_n = mask_off.sum(axis=1).astype(np.int32)
    hlt_n = hlt_mask.sum(axis=1).astype(np.int32)
    reco_n = reco_mask.sum(axis=1).astype(np.int32)

    max_n = int(max(off_n.max(), hlt_n.max(), reco_n.max()))
    bins = np.arange(0, max_n + 2) - 0.5

    # 1) Count distributions.
    plt.figure(figsize=(8, 5))
    plt.hist(off_n, bins=bins, density=True, histtype="step", linewidth=2, color="crimson", label="Offline")
    plt.hist(hlt_n, bins=bins, density=True, histtype="step", linewidth=2, color="steelblue", label="HLT")
    plt.hist(reco_n, bins=bins, density=True, histtype="step", linewidth=2, color="forestgreen", label="Reconstructed")
    plt.xlabel("Constituent count per jet")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_root / "constituent_count_hist.png", dpi=300)
    plt.close()

    # 2) Count error distributions vs offline.
    err_hlt = hlt_n - off_n
    err_reco = reco_n - off_n
    e_min = int(min(err_hlt.min(), err_reco.min()))
    e_max = int(max(err_hlt.max(), err_reco.max()))
    bins_e = np.arange(e_min, e_max + 2) - 0.5

    plt.figure(figsize=(8, 5))
    plt.hist(err_hlt, bins=bins_e, density=True, histtype="step", linewidth=2, color="steelblue", label="HLT - Offline")
    plt.hist(err_reco, bins=bins_e, density=True, histtype="step", linewidth=2, color="forestgreen", label="Reco - Offline")
    plt.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Count error per jet")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_root / "constituent_count_error_hist.png", dpi=300)
    plt.close()

    # 3) Mean predicted count vs offline-count bins.
    x_vals = []
    hlt_mean = []
    reco_mean = []
    for n in np.unique(off_n):
        sel = off_n == n
        if sel.sum() < 10:
            continue
        x_vals.append(int(n))
        hlt_mean.append(float(hlt_n[sel].mean()))
        reco_mean.append(float(reco_n[sel].mean()))

    if len(x_vals) > 0:
        x_arr = np.array(x_vals, dtype=np.float64)
        plt.figure(figsize=(8, 5))
        plt.plot(x_arr, x_arr, "k:", linewidth=1.5, label="Ideal y=x")
        plt.plot(x_arr, np.array(hlt_mean), "o-", color="steelblue", label="HLT mean")
        plt.plot(x_arr, np.array(reco_mean), "s--", color="forestgreen", label="Reconstructed mean")
        plt.xlabel("Offline constituent count")
        plt.ylabel("Mean predicted constituent count")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_root / "constituent_count_profile.png", dpi=300)
        plt.close()

    # 3B) Created constituents per-jet distributions by source.
    cmax = int(max(created_merge_count.max(), created_eff_count.max(), 1))
    bins_c = np.arange(0, cmax + 2) - 0.5
    plt.figure(figsize=(8, 5))
    plt.hist(
        created_merge_count,
        bins=bins_c,
        density=True,
        histtype="step",
        linewidth=2,
        color="darkorange",
        label="Created by unmerge",
    )
    plt.hist(
        created_eff_count,
        bins=bins_c,
        density=True,
        histtype="step",
        linewidth=2,
        color="mediumseagreen",
        label="Created by efficiency generation",
    )
    plt.xlabel("Created constituents per jet")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_root / "constituent_created_count_hist.png", dpi=300)
    plt.close()

    # 4) Created-vs-lost means (per jet).
    n_jets = max(int(off_n.shape[0]), 1)
    lost_merge_mean = float(hlt_stats.get("n_merged_pairs", 0)) / float(n_jets)
    lost_eff_mean = float(hlt_stats.get("n_lost_eff", 0)) / float(n_jets)
    needed_add_mean = float(np.maximum(off_n - hlt_n, 0).mean())
    created_merge_mean = float(created_merge_count.mean())
    created_eff_mean = float(created_eff_count.mean())
    total_created_mean = created_merge_mean + created_eff_mean

    labels = [
        "Lost: merging",
        "Lost: efficiency",
        "Need add (Offline-HLT)",
        "Created: unmerge",
        "Created: efficiency",
        "Created: total",
    ]
    values = [
        lost_merge_mean,
        lost_eff_mean,
        needed_add_mean,
        created_merge_mean,
        created_eff_mean,
        total_created_mean,
    ]
    colors = ["steelblue", "cornflowerblue", "slategray", "darkorange", "mediumseagreen", "forestgreen"]

    plt.figure(figsize=(9.5, 5))
    x = np.arange(len(labels))
    plt.bar(x, values, color=colors, alpha=0.9)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Mean constituents per jet")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_root / "constituent_created_vs_lost_mean.png", dpi=300)
    plt.close()

    summary = {
        "offline_count_mean": float(off_n.mean()),
        "hlt_count_mean": float(hlt_n.mean()),
        "reco_count_mean": float(reco_n.mean()),
        "hlt_count_bias_vs_offline": float((hlt_n - off_n).mean()),
        "reco_count_bias_vs_offline": float((reco_n - off_n).mean()),
        "hlt_count_mae_vs_offline": float(np.abs(hlt_n - off_n).mean()),
        "reco_count_mae_vs_offline": float(np.abs(reco_n - off_n).mean()),
        "lost_merge_mean_per_jet": lost_merge_mean,
        "lost_eff_mean_per_jet": lost_eff_mean,
        "needed_add_mean_per_jet": needed_add_mean,
        "created_merge_mean_per_jet": created_merge_mean,
        "created_eff_mean_per_jet": created_eff_mean,
        "created_total_mean_per_jet": total_created_mean,
    }
    return summary


def plot_budget_diagnostics(
    save_root: Path,
    true_merge: np.ndarray,
    true_eff: np.ndarray,
    pred_merge: np.ndarray,
    pred_eff: np.ndarray,
) -> Dict[str, float]:
    true_merge = true_merge.astype(np.float64)
    true_eff = true_eff.astype(np.float64)
    pred_merge = pred_merge.astype(np.float64)
    pred_eff = pred_eff.astype(np.float64)
    true_total = true_merge + true_eff
    pred_total = pred_merge + pred_eff

    err_merge = pred_merge - true_merge
    err_eff = pred_eff - true_eff
    err_total = pred_total - true_total

    e_min = int(np.floor(min(err_merge.min(), err_eff.min(), err_total.min())))
    e_max = int(np.ceil(max(err_merge.max(), err_eff.max(), err_total.max())))
    bins = np.linspace(e_min - 0.5, e_max + 0.5, num=max(25, e_max - e_min + 2))

    plt.figure(figsize=(8.5, 5))
    plt.hist(err_merge, bins=bins, density=True, histtype="step", linewidth=2, color="darkorange", label="merge budget error")
    plt.hist(err_eff, bins=bins, density=True, histtype="step", linewidth=2, color="mediumseagreen", label="eff budget error")
    plt.hist(err_total, bins=bins, density=True, histtype="step", linewidth=2, color="slateblue", label="total budget error")
    plt.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Predicted budget - true budget")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_root / "budget_error_hist.png", dpi=300)
    plt.close()

    # Profile by true budget quantiles.
    q_edges = np.quantile(true_total, np.linspace(0.0, 1.0, 9))
    q_edges = np.unique(q_edges)
    x = []
    y_true = []
    y_pred_total = []
    y_pred_merge = []
    y_pred_eff = []
    for i in range(max(len(q_edges) - 1, 0)):
        lo, hi = q_edges[i], q_edges[i + 1]
        if i < len(q_edges) - 2:
            sel = (true_total >= lo) & (true_total < hi)
        else:
            sel = (true_total >= lo) & (true_total <= hi)
        if sel.sum() < 10:
            continue
        x.append(float(true_total[sel].mean()))
        y_true.append(float(true_total[sel].mean()))
        y_pred_total.append(float(pred_total[sel].mean()))
        y_pred_merge.append(float(pred_merge[sel].mean()))
        y_pred_eff.append(float(pred_eff[sel].mean()))

    if len(x) > 0:
        x = np.array(x, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)
        y_pred_total = np.array(y_pred_total, dtype=np.float64)
        y_pred_merge = np.array(y_pred_merge, dtype=np.float64)
        y_pred_eff = np.array(y_pred_eff, dtype=np.float64)

        plt.figure(figsize=(8.5, 5))
        plt.plot(x, y_true, "k:", linewidth=1.5, label="ideal y=x")
        plt.plot(x, y_pred_total, "o-", color="slateblue", label="pred total")
        plt.plot(x, y_pred_merge, "s--", color="darkorange", label="pred merge")
        plt.plot(x, y_pred_eff, "d--", color="mediumseagreen", label="pred eff")
        plt.xlabel("Mean true total budget in bin")
        plt.ylabel("Mean predicted budget")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_root / "budget_profile.png", dpi=300)
        plt.close()

    summary = {
        "merge_mae": float(np.abs(err_merge).mean()),
        "eff_mae": float(np.abs(err_eff).mean()),
        "total_mae": float(np.abs(err_total).mean()),
        "merge_bias": float(err_merge.mean()),
        "eff_bias": float(err_eff.mean()),
        "total_bias": float(err_total.mean()),
        "merge_rmse": float(np.sqrt((err_merge ** 2).mean())),
        "eff_rmse": float(np.sqrt((err_eff ** 2).mean())),
        "total_rmse": float(np.sqrt((err_total ** 2).mean())),
        "true_merge_mean": float(true_merge.mean()),
        "pred_merge_mean": float(pred_merge.mean()),
        "true_eff_mean": float(true_eff.mean()),
        "pred_eff_mean": float(pred_eff.mean()),
        "true_total_mean": float(true_total.mean()),
        "pred_total_mean": float(pred_total.mean()),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--offset_jets", type=int, default=0)
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default=str(Path().cwd() / "checkpoints" / "offline_reconstructor_no_gt"))
    parser.add_argument("--run_name", type=str, default="merge_only_nominal")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--roc_fpr_min", type=float, default=1e-4)
    parser.add_argument("--response_n_bins", type=int, default=8)
    parser.add_argument("--response_min_count", type=int, default=30)

    # HLT knobs
    parser.add_argument("--merge_radius", type=float, default=CONFIG["hlt_effects"]["merge_radius"])
    parser.add_argument("--pt_threshold_hlt", type=float, default=CONFIG["hlt_effects"]["pt_threshold_hlt"])
    parser.add_argument("--pt_threshold_offline", type=float, default=CONFIG["hlt_effects"]["pt_threshold_offline"])
    parser.add_argument("--post_smear_pt_threshold", type=float, default=CONFIG["hlt_effects"]["post_smear_pt_threshold"])
    parser.add_argument("--eff_plateau_barrel", type=float, default=CONFIG["hlt_effects"]["eff_plateau_barrel"])
    parser.add_argument("--eff_plateau_endcap", type=float, default=CONFIG["hlt_effects"]["eff_plateau_endcap"])
    parser.add_argument("--smear_a", type=float, default=CONFIG["hlt_effects"]["smear_a"])
    parser.add_argument("--smear_b", type=float, default=CONFIG["hlt_effects"]["smear_b"])
    parser.add_argument("--smear_c", type=float, default=CONFIG["hlt_effects"]["smear_c"])

    # Reconstructor knobs
    parser.add_argument("--reco_epochs", type=int, default=CONFIG["reconstructor_training"]["epochs"])
    parser.add_argument("--reco_batch_size", type=int, default=CONFIG["reconstructor_training"]["batch_size"])
    parser.add_argument("--reco_lr", type=float, default=CONFIG["reconstructor_training"]["lr"])
    parser.add_argument("--reco_patience", type=int, default=CONFIG["reconstructor_training"]["patience"])
    parser.add_argument("--reco_weight_threshold", type=float, default=0.03)
    parser.add_argument("--reco_disable_budget_topk", action="store_true")

    parser.add_argument("--skip_save_models", action="store_true")
    args = parser.parse_args()

    # Apply user overrides.
    CONFIG["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    CONFIG["hlt_effects"]["pt_threshold_hlt"] = float(args.pt_threshold_hlt)
    CONFIG["hlt_effects"]["pt_threshold_offline"] = float(args.pt_threshold_offline)
    CONFIG["hlt_effects"]["post_smear_pt_threshold"] = float(args.post_smear_pt_threshold)
    CONFIG["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    CONFIG["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    CONFIG["hlt_effects"]["smear_a"] = float(args.smear_a)
    CONFIG["hlt_effects"]["smear_b"] = float(args.smear_b)
    CONFIG["hlt_effects"]["smear_c"] = float(args.smear_c)

    CONFIG["reconstructor_training"]["epochs"] = int(args.reco_epochs)
    CONFIG["reconstructor_training"]["batch_size"] = int(args.reco_batch_size)
    CONFIG["reconstructor_training"]["lr"] = float(args.reco_lr)
    CONFIG["reconstructor_training"]["patience"] = int(args.reco_patience)

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
    print("Loading offline constituents from HDF5...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets for offset {args.offset_jets} + n_train_jets {args.n_train_jets}. "
            f"Got {all_const_full.shape[0]}"
        )

    const_raw = all_const_full[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(CONFIG["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating realistic pseudo-HLT (no constituent mapping)...")
    hlt_const, hlt_mask, hlt_stats, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        CONFIG,
        seed=RANDOM_SEED,
    )
    budget_merge_true = budget_truth["merge_lost_per_jet"].astype(np.float32)
    budget_eff_true = budget_truth["eff_lost_per_jet"].astype(np.float32)

    print("HLT Simulation Statistics:")
    print(f"  Jets: {hlt_stats['n_jets']:,}")
    print(f"  Offline particles: {hlt_stats['n_initial']:,}")
    print(f"  Lost to pre-threshold: {hlt_stats['n_lost_threshold_pre']:,}")
    print(f"  Merging operations: {hlt_stats['n_merged_pairs']:,}")
    print(f"  Lost to efficiency: {hlt_stats['n_lost_eff']:,}")
    print(f"  Reassigned tokens: {hlt_stats['n_reassigned']:,}")
    print(f"  Lost to post-threshold: {hlt_stats['n_lost_threshold_post']:,}")
    print(f"  HLT particles: {hlt_stats['n_final']:,}")
    print(
        f"  Avg per jet: Offline={hlt_stats['avg_offline_per_jet']:.2f}, "
        f"HLT={hlt_stats['avg_hlt_per_jet']:.2f}"
    )

    print("Computing features...")
    features_off = compute_features(const_off, masks_off)
    features_hlt = compute_features(hlt_const, hlt_mask)

    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=labels[temp_idx],
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = standardize(features_hlt, hlt_mask, feat_means, feat_stds)

    # ====================================================================== #
    # STEP 1: Teacher
    # ====================================================================== #
    print("\n" + "=" * 70)
    print("STEP 1: TEACHER (Offline)")
    print("=" * 70)
    BS = CONFIG["training"]["batch_size"]

    train_ds_off = JetDataset(features_off_std[train_idx], masks_off[train_idx], labels[train_idx])
    val_ds_off = JetDataset(features_off_std[val_idx], masks_off[val_idx], labels[val_idx])
    test_ds_off = JetDataset(features_off_std[test_idx], masks_off[test_idx], labels[test_idx])

    train_loader_off = DataLoader(train_ds_off, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_off = DataLoader(val_ds_off, batch_size=BS, shuffle=False)
    test_loader_off = DataLoader(test_ds_off, batch_size=BS, shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    teacher = train_single_view_classifier(
        teacher,
        train_loader_off,
        val_loader_off,
        device,
        CONFIG["training"],
        name="Teacher",
    )
    auc_teacher, preds_teacher, labs = eval_classifier(teacher, test_loader_off, device)

    # ====================================================================== #
    # STEP 2: Baseline HLT
    # ====================================================================== #
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE HLT")
    print("=" * 70)

    train_ds_hlt = JetDataset(features_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx])
    val_ds_hlt = JetDataset(features_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    test_ds_hlt = JetDataset(features_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])

    train_loader_hlt = DataLoader(train_ds_hlt, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_hlt = DataLoader(val_ds_hlt, batch_size=BS, shuffle=False)
    test_loader_hlt = DataLoader(test_ds_hlt, batch_size=BS, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    baseline = train_single_view_classifier(
        baseline,
        train_loader_hlt,
        val_loader_hlt,
        device,
        CONFIG["training"],
        name="Baseline",
    )
    auc_baseline, preds_baseline, _ = eval_classifier(baseline, test_loader_hlt, device)

    # ====================================================================== #
    # STEP 3: Offline Reconstructor (no constituent mapping)
    # ====================================================================== #
    print("\n" + "=" * 70)
    print("STEP 3: OFFLINE RECONSTRUCTOR (NO CONST MAP)")
    print("=" * 70)

    train_ds_reco = ReconstructionDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        hlt_const[train_idx],
        const_off[train_idx],
        masks_off[train_idx],
        budget_merge_true[train_idx],
        budget_eff_true[train_idx],
    )
    val_ds_reco = ReconstructionDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        hlt_const[val_idx],
        const_off[val_idx],
        masks_off[val_idx],
        budget_merge_true[val_idx],
        budget_eff_true[val_idx],
    )

    train_loader_reco = DataLoader(
        train_ds_reco,
        batch_size=CONFIG["reconstructor_training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader_reco = DataLoader(
        val_ds_reco,
        batch_size=CONFIG["reconstructor_training"]["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    reconstructor = OfflineReconstructor(
        input_dim=7,
        **CONFIG["reconstructor_model"],
    ).to(device)

    reconstructor, reco_val_metrics = train_reconstructor(
        reconstructor,
        train_loader_reco,
        val_loader_reco,
        device,
        CONFIG["reconstructor_training"],
        CONFIG["loss"],
    )

    print("Best reconstructor val metrics:")
    for k, v in reco_val_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("Building reconstructed dataset for all jets...")
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
        feat_hlt=features_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        max_constits=args.max_constits,
        device=device,
        batch_size=CONFIG["reconstructor_training"]["batch_size"],
        weight_threshold=float(args.reco_weight_threshold),
        use_budget_topk=not bool(args.reco_disable_budget_topk),
    )

    features_reco = compute_features(reco_const, reco_mask)
    features_reco_std = standardize(features_reco, reco_mask, feat_means, feat_stds)
    features_reco_flag = np.concatenate(
        [features_reco_std, reco_merge_flag[..., None], reco_eff_flag[..., None]],
        axis=-1,
    ).astype(np.float32)

    # ====================================================================== #
    # STEP 4: Taggers on reconstructed view
    # ====================================================================== #
    train_ds_reco_cls = JetDataset(features_reco_std[train_idx], reco_mask[train_idx], labels[train_idx])
    val_ds_reco_cls = JetDataset(features_reco_std[val_idx], reco_mask[val_idx], labels[val_idx])
    test_ds_reco_cls = JetDataset(features_reco_std[test_idx], reco_mask[test_idx], labels[test_idx])

    train_ds_reco_flag = JetDataset(features_reco_flag[train_idx], reco_mask[train_idx], labels[train_idx])
    val_ds_reco_flag = JetDataset(features_reco_flag[val_idx], reco_mask[val_idx], labels[val_idx])
    test_ds_reco_flag = JetDataset(features_reco_flag[test_idx], reco_mask[test_idx], labels[test_idx])

    train_loader_reco_cls = DataLoader(train_ds_reco_cls, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_reco_cls = DataLoader(val_ds_reco_cls, batch_size=BS, shuffle=False)
    test_loader_reco_cls = DataLoader(test_ds_reco_cls, batch_size=BS, shuffle=False)

    train_loader_reco_flag = DataLoader(train_ds_reco_flag, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_reco_flag = DataLoader(val_ds_reco_flag, batch_size=BS, shuffle=False)
    test_loader_reco_flag = DataLoader(test_ds_reco_flag, batch_size=BS, shuffle=False)

    print("\n" + "=" * 70)
    print("STEP 4A: UNMERGE MODEL CLASSIFIER (Reconstructed)")
    print("=" * 70)
    unmerge_cls = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    unmerge_cls = train_single_view_classifier(
        unmerge_cls,
        train_loader_reco_cls,
        val_loader_reco_cls,
        device,
        CONFIG["training"],
        name="Unmerge",
    )
    auc_unmerge, preds_unmerge, _ = eval_classifier(unmerge_cls, test_loader_reco_cls, device)

    print("\n" + "=" * 70)
    print("STEP 4B: UNMERGE MODEL + MERGEFLAG")
    print("=" * 70)
    unmerge_flag_cls = ParticleTransformer(input_dim=9, **CONFIG["model"]).to(device)
    unmerge_flag_cls = train_single_view_classifier(
        unmerge_flag_cls,
        train_loader_reco_flag,
        val_loader_reco_flag,
        device,
        CONFIG["training"],
        name="Unmerge+MF",
    )
    auc_unmerge_flag, preds_unmerge_flag, _ = eval_classifier(unmerge_flag_cls, test_loader_reco_flag, device)

    # Dual-view datasets
    train_ds_dual = DualViewJetDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_reco_std[train_idx],
        reco_mask[train_idx],
        labels[train_idx],
    )
    val_ds_dual = DualViewJetDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_reco_std[val_idx],
        reco_mask[val_idx],
        labels[val_idx],
    )
    test_ds_dual = DualViewJetDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_reco_std[test_idx],
        reco_mask[test_idx],
        labels[test_idx],
    )

    train_ds_dual_flag = DualViewJetDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_reco_flag[train_idx],
        reco_mask[train_idx],
        labels[train_idx],
    )
    val_ds_dual_flag = DualViewJetDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_reco_flag[val_idx],
        reco_mask[val_idx],
        labels[val_idx],
    )
    test_ds_dual_flag = DualViewJetDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_reco_flag[test_idx],
        reco_mask[test_idx],
        labels[test_idx],
    )

    train_loader_dual = DataLoader(train_ds_dual, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_dual = DataLoader(val_ds_dual, batch_size=BS, shuffle=False)
    test_loader_dual = DataLoader(test_ds_dual, batch_size=BS, shuffle=False)

    train_loader_dual_flag = DataLoader(train_ds_dual_flag, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_dual_flag = DataLoader(val_ds_dual_flag, batch_size=BS, shuffle=False)
    test_loader_dual_flag = DataLoader(test_ds_dual_flag, batch_size=BS, shuffle=False)

    print("\n" + "=" * 70)
    print("STEP 4C: DUAL-VIEW CLASSIFIER (HLT + Reconstructed)")
    print("=" * 70)
    dual_cls = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **CONFIG["model"]).to(device)
    dual_cls = train_dual_view_classifier(
        dual_cls,
        train_loader_dual,
        val_loader_dual,
        device,
        CONFIG["training"],
        name="DualView",
    )
    auc_dual, preds_dual, _ = eval_classifier_dual(dual_cls, test_loader_dual, device)

    print("\n" + "=" * 70)
    print("STEP 4D: DUAL-VIEW + MERGEFLAG CLASSIFIER")
    print("=" * 70)
    dual_flag_cls = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=9, **CONFIG["model"]).to(device)
    dual_flag_cls = train_dual_view_classifier(
        dual_flag_cls,
        train_loader_dual_flag,
        val_loader_dual_flag,
        device,
        CONFIG["training"],
        name="DualView+MF",
    )
    auc_dual_flag, preds_dual_flag, _ = eval_classifier_dual(dual_flag_cls, test_loader_dual_flag, device)

    # KD datasets (dual)
    kd_train_ds_dual = DualViewKDDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_reco_std[train_idx],
        reco_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        labels[train_idx],
    )
    kd_val_ds_dual = DualViewKDDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_reco_std[val_idx],
        reco_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        labels[val_idx],
    )
    kd_test_ds_dual = DualViewKDDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_reco_std[test_idx],
        reco_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        labels[test_idx],
    )

    kd_train_loader_dual = DataLoader(kd_train_ds_dual, batch_size=BS, shuffle=True, drop_last=True)
    kd_val_loader_dual = DataLoader(kd_val_ds_dual, batch_size=BS, shuffle=False)
    kd_test_loader_dual = DataLoader(kd_test_ds_dual, batch_size=BS, shuffle=False)

    print("\n" + "=" * 70)
    print("STEP 4E: DUAL-VIEW + KD")
    print("=" * 70)
    kd_student_dual = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **CONFIG["model"]).to(device)
    kd_student_dual = train_dual_kd_student(
        kd_student_dual,
        teacher,
        kd_train_loader_dual,
        kd_val_loader_dual,
        device,
        CONFIG["training"],
        CONFIG["kd"],
        name="DualView+KD",
        run_self_train=bool(CONFIG["kd"]["self_train"]),
    )
    auc_dual_kd, preds_dual_kd, _ = evaluate_kd_dual(kd_student_dual, kd_test_loader_dual, device)

    kd_train_ds_dual_flag = DualViewKDDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_reco_flag[train_idx],
        reco_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        labels[train_idx],
    )
    kd_val_ds_dual_flag = DualViewKDDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_reco_flag[val_idx],
        reco_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        labels[val_idx],
    )
    kd_test_ds_dual_flag = DualViewKDDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_reco_flag[test_idx],
        reco_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        labels[test_idx],
    )

    kd_train_loader_dual_flag = DataLoader(kd_train_ds_dual_flag, batch_size=BS, shuffle=True, drop_last=True)
    kd_val_loader_dual_flag = DataLoader(kd_val_ds_dual_flag, batch_size=BS, shuffle=False)
    kd_test_loader_dual_flag = DataLoader(kd_test_ds_dual_flag, batch_size=BS, shuffle=False)

    print("\n" + "=" * 70)
    print("STEP 4F: DUAL-VIEW + MERGEFLAG + KD")
    print("=" * 70)
    kd_student_dual_flag = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=9, **CONFIG["model"]).to(device)
    kd_student_dual_flag = train_dual_kd_student(
        kd_student_dual_flag,
        teacher,
        kd_train_loader_dual_flag,
        kd_val_loader_dual_flag,
        device,
        CONFIG["training"],
        CONFIG["kd"],
        name="DualView+MF+KD",
        run_self_train=bool(CONFIG["kd"]["self_train"]),
    )
    auc_dual_flag_kd, preds_dual_flag_kd, _ = evaluate_kd_dual(kd_student_dual_flag, kd_test_loader_dual_flag, device)

    # ====================================================================== #
    # Final evaluation and outputs
    # ====================================================================== #
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"Unmerge Model    AUC: {auc_unmerge:.4f}")
    print(f"Unmerge+MF       AUC: {auc_unmerge_flag:.4f}")
    print(f"Dual-View        AUC: {auc_dual:.4f}")
    print(f"Dual-View+MF     AUC: {auc_dual_flag:.4f}")
    print(f"Dual-View+KD     AUC: {auc_dual_kd:.4f}")
    print(f"Dual-View+MF+KD  AUC: {auc_dual_flag_kd:.4f}")

    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_u, tpr_u, _ = roc_curve(labs, preds_unmerge)
    fpr_uf, tpr_uf, _ = roc_curve(labs, preds_unmerge_flag)
    fpr_dv, tpr_dv, _ = roc_curve(labs, preds_dual)
    fpr_dvf, tpr_dvf, _ = roc_curve(labs, preds_dual_flag)
    fpr_dv_k, tpr_dv_k, _ = roc_curve(labs, preds_dual_kd)
    fpr_dvf_k, tpr_dvf_k, _ = roc_curve(labs, preds_dual_flag_kd)

    target_tpr = 0.30
    fpr30_teacher = fpr_at_target_tpr(fpr_t, tpr_t, target_tpr)
    fpr30_baseline = fpr_at_target_tpr(fpr_b, tpr_b, target_tpr)
    fpr30_unmerge = fpr_at_target_tpr(fpr_u, tpr_u, target_tpr)
    fpr30_unmerge_flag = fpr_at_target_tpr(fpr_uf, tpr_uf, target_tpr)
    fpr30_dual = fpr_at_target_tpr(fpr_dv, tpr_dv, target_tpr)
    fpr30_dual_flag = fpr_at_target_tpr(fpr_dvf, tpr_dvf, target_tpr)
    fpr30_dual_kd = fpr_at_target_tpr(fpr_dv_k, tpr_dv_k, target_tpr)
    fpr30_dual_flag_kd = fpr_at_target_tpr(fpr_dvf_k, tpr_dvf_k, target_tpr)

    print(f"\nFPR at fixed TPR={target_tpr:.2f}")
    print("  Teacher (Offline):      " + f"{fpr30_teacher:.6f} ({100.0*fpr30_teacher:.3f}%)")
    print("  Baseline (HLT):         " + f"{fpr30_baseline:.6f} ({100.0*fpr30_baseline:.3f}%)")
    print("  Unmerge Model:          " + f"{fpr30_unmerge:.6f} ({100.0*fpr30_unmerge:.3f}%)")
    print("  Unmerge+MF:             " + f"{fpr30_unmerge_flag:.6f} ({100.0*fpr30_unmerge_flag:.3f}%)")
    print("  Dual-View:              " + f"{fpr30_dual:.6f} ({100.0*fpr30_dual:.3f}%)")
    print("  Dual-View+MF:           " + f"{fpr30_dual_flag:.6f} ({100.0*fpr30_dual_flag:.3f}%)")
    print("  Dual-View+KD:           " + f"{fpr30_dual_kd:.6f} ({100.0*fpr30_dual_kd:.3f}%)")
    print("  Dual-View+MF+KD:        " + f"{fpr30_dual_flag_kd:.6f} ({100.0*fpr30_dual_flag_kd:.3f}%)")

    target_tpr = 0.50
    fpr50_teacher = fpr_at_target_tpr(fpr_t, tpr_t, target_tpr)
    fpr50_baseline = fpr_at_target_tpr(fpr_b, tpr_b, target_tpr)
    fpr50_unmerge = fpr_at_target_tpr(fpr_u, tpr_u, target_tpr)
    fpr50_unmerge_flag = fpr_at_target_tpr(fpr_uf, tpr_uf, target_tpr)
    fpr50_dual = fpr_at_target_tpr(fpr_dv, tpr_dv, target_tpr)
    fpr50_dual_flag = fpr_at_target_tpr(fpr_dvf, tpr_dvf, target_tpr)
    fpr50_dual_kd = fpr_at_target_tpr(fpr_dv_k, tpr_dv_k, target_tpr)
    fpr50_dual_flag_kd = fpr_at_target_tpr(fpr_dvf_k, tpr_dvf_k, target_tpr)

    print(f"\nFPR at fixed TPR={target_tpr:.2f}")
    print("  Teacher (Offline):      " + f"{fpr50_teacher:.6f} ({100.0*fpr50_teacher:.3f}%)")
    print("  Baseline (HLT):         " + f"{fpr50_baseline:.6f} ({100.0*fpr50_baseline:.3f}%)")
    print("  Unmerge Model:          " + f"{fpr50_unmerge:.6f} ({100.0*fpr50_unmerge:.3f}%)")
    print("  Unmerge+MF:             " + f"{fpr50_unmerge_flag:.6f} ({100.0*fpr50_unmerge_flag:.3f}%)")
    print("  Dual-View:              " + f"{fpr50_dual:.6f} ({100.0*fpr50_dual:.3f}%)")
    print("  Dual-View+MF:           " + f"{fpr50_dual_flag:.6f} ({100.0*fpr50_dual_flag:.3f}%)")
    print("  Dual-View+KD:           " + f"{fpr50_dual_kd:.6f} ({100.0*fpr50_dual_kd:.3f}%)")
    print("  Dual-View+MF+KD:        " + f"{fpr50_dual_flag_kd:.6f} ({100.0*fpr50_dual_flag_kd:.3f}%)")

    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_u, fpr_u, ":", f"Unmerge (AUC={auc_unmerge:.3f})", "forestgreen"),
            (tpr_uf, fpr_uf, "-.", f"Unmerge+MF (AUC={auc_unmerge_flag:.3f})", "darkorange"),
            (tpr_dv, fpr_dv, "-", f"Dual-View (AUC={auc_dual:.3f})", "teal"),
            (tpr_dvf, fpr_dvf, "--", f"DualView+MF (AUC={auc_dual_flag:.3f})", "orchid"),
            (tpr_dv_k, fpr_dv_k, ":", f"DualView+KD (AUC={auc_dual_kd:.3f})", "slateblue"),
            (tpr_dvf_k, fpr_dvf_k, "-.", f"DualView+MF+KD (AUC={auc_dual_flag_kd:.3f})", "darkslateblue"),
        ],
        save_root / "results_all.png",
        args.roc_fpr_min,
    )

    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
        ],
        save_root / "results_teacher_baseline.png",
        args.roc_fpr_min,
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_u, fpr_u, ":", f"Unmerge (AUC={auc_unmerge:.3f})", "forestgreen"),
        ],
        save_root / "results_teacher_baseline_unmerge.png",
        args.roc_fpr_min,
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_uf, fpr_uf, "-.", f"Unmerge+MF (AUC={auc_unmerge_flag:.3f})", "darkorange"),
        ],
        save_root / "results_teacher_baseline_unmerge_flag.png",
        args.roc_fpr_min,
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_dv, fpr_dv, "-", f"Dual-View (AUC={auc_dual:.3f})", "teal"),
        ],
        save_root / "results_teacher_baseline_dualview.png",
        args.roc_fpr_min,
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_dv_k, fpr_dv_k, ":", f"DualView+KD (AUC={auc_dual_kd:.3f})", "slateblue"),
        ],
        save_root / "results_teacher_baseline_dualview_kd.png",
        args.roc_fpr_min,
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_dvf, fpr_dvf, "--", f"DualView+MF (AUC={auc_dual_flag:.3f})", "orchid"),
        ],
        save_root / "results_teacher_baseline_dualview_flag.png",
        args.roc_fpr_min,
    )
    plot_roc(
        [
            (tpr_t, fpr_t, "-", f"Teacher (AUC={auc_teacher:.3f})", "crimson"),
            (tpr_b, fpr_b, "--", f"HLT Baseline (AUC={auc_baseline:.3f})", "steelblue"),
            (tpr_dvf_k, fpr_dvf_k, "-.", f"DualView+MF+KD (AUC={auc_dual_flag_kd:.3f})", "darkslateblue"),
        ],
        save_root / "results_teacher_baseline_dualview_flag_kd.png",
        args.roc_fpr_min,
    )

    # Jet pT response/resolution on test split.
    pt_truth_test = compute_jet_pt(const_off[test_idx], masks_off[test_idx])
    pt_hlt_test = compute_jet_pt(hlt_const[test_idx], hlt_mask[test_idx])
    pt_reco_test = compute_jet_pt(reco_const[test_idx], reco_mask[test_idx])

    pt_edges = build_pt_edges(pt_truth_test, args.response_n_bins)
    rr_hlt = jet_response_resolution(pt_truth_test, pt_hlt_test, pt_edges, args.response_min_count)
    rr_reco = jet_response_resolution(pt_truth_test, pt_reco_test, pt_edges, args.response_min_count)
    plot_response_resolution(
        rr_hlt,
        rr_reco,
        "HLT (reco)",
        "Corrected HLT / Reconstructed (reco)",
        save_root / "jet_response_resolution.png",
    )

    rr_hlt_map = {(r["pt_low"], r["pt_high"]): r for r in rr_hlt}
    rr_reco_map = {(r["pt_low"], r["pt_high"]): r for r in rr_reco}
    keys = sorted(set(rr_hlt_map.keys()) & set(rr_reco_map.keys()))
    rr_hlt_common = [rr_hlt_map[k] for k in keys]
    rr_reco_common = [rr_reco_map[k] for k in keys]

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
    print("\nConstituent-count diagnostics:")
    print(
        f"  Means: offline={count_summary['offline_count_mean']:.3f}, "
        f"hlt={count_summary['hlt_count_mean']:.3f}, reco={count_summary['reco_count_mean']:.3f}"
    )
    print(
        f"  MAE vs offline: hlt={count_summary['hlt_count_mae_vs_offline']:.3f}, "
        f"reco={count_summary['reco_count_mae_vs_offline']:.3f}"
    )
    print(
        f"  Created means (per jet): unmerge={count_summary['created_merge_mean_per_jet']:.3f}, "
        f"efficiency={count_summary['created_eff_mean_per_jet']:.3f}, "
        f"total={count_summary['created_total_mean_per_jet']:.3f}"
    )

    budget_summary = plot_budget_diagnostics(
        save_root=save_root,
        true_merge=budget_merge_true[test_idx],
        true_eff=budget_eff_true[test_idx],
        pred_merge=pred_budget_merge[test_idx],
        pred_eff=pred_budget_eff[test_idx],
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

    def rr_field(records, key):
        return np.array([r[key] for r in records], dtype=np.float64)

    np.savez(
        save_root / "results.npz",
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_unmerge=auc_unmerge,
        auc_unmerge_flag=auc_unmerge_flag,
        auc_dual=auc_dual,
        auc_dual_flag=auc_dual_flag,
        auc_dual_kd=auc_dual_kd,
        auc_dual_flag_kd=auc_dual_flag_kd,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_unmerge=fpr_u,
        tpr_unmerge=tpr_u,
        fpr_unmerge_flag=fpr_uf,
        tpr_unmerge_flag=tpr_uf,
        fpr_dual=fpr_dv,
        tpr_dual=tpr_dv,
        fpr_dual_flag=fpr_dvf,
        tpr_dual_flag=tpr_dvf,
        fpr_dual_kd=fpr_dv_k,
        tpr_dual_kd=tpr_dv_k,
        fpr_dual_flag_kd=fpr_dvf_k,
        tpr_dual_flag_kd=tpr_dvf_k,
        fpr30_teacher=fpr30_teacher,
        fpr30_baseline=fpr30_baseline,
        fpr30_unmerge=fpr30_unmerge,
        fpr30_unmerge_flag=fpr30_unmerge_flag,
        fpr30_dual=fpr30_dual,
        fpr30_dual_flag=fpr30_dual_flag,
        fpr30_dual_kd=fpr30_dual_kd,
        fpr30_dual_flag_kd=fpr30_dual_flag_kd,
        fpr50_teacher=fpr50_teacher,
        fpr50_baseline=fpr50_baseline,
        fpr50_unmerge=fpr50_unmerge,
        fpr50_unmerge_flag=fpr50_unmerge_flag,
        fpr50_dual=fpr50_dual,
        fpr50_dual_flag=fpr50_dual_flag,
        fpr50_dual_kd=fpr50_dual_kd,
        fpr50_dual_flag_kd=fpr50_dual_flag_kd,
        jet_response_pt_low=rr_field(rr_hlt_common, "pt_low"),
        jet_response_pt_high=rr_field(rr_hlt_common, "pt_high"),
        jet_response_count=rr_field(rr_hlt_common, "count"),
        jet_response_hlt_mean=rr_field(rr_hlt_common, "response"),
        jet_response_hlt_std=rr_field(rr_hlt_common, "resolution"),
        jet_response_corrected_mean=rr_field(rr_reco_common, "response"),
        jet_response_corrected_std=rr_field(rr_reco_common, "resolution"),
        count_offline_mean=count_summary["offline_count_mean"],
        count_hlt_mean=count_summary["hlt_count_mean"],
        count_reco_mean=count_summary["reco_count_mean"],
        count_hlt_mae=count_summary["hlt_count_mae_vs_offline"],
        count_reco_mae=count_summary["reco_count_mae_vs_offline"],
        created_merge_mean=count_summary["created_merge_mean_per_jet"],
        created_eff_mean=count_summary["created_eff_mean_per_jet"],
        created_total_mean=count_summary["created_total_mean_per_jet"],
        budget_merge_mae=budget_summary["merge_mae"],
        budget_eff_mae=budget_summary["eff_mae"],
        budget_total_mae=budget_summary["total_mae"],
        budget_merge_bias=budget_summary["merge_bias"],
        budget_eff_bias=budget_summary["eff_bias"],
        budget_total_bias=budget_summary["total_bias"],
        unmerge_test_loss=float(reco_val_metrics.get("val_total", np.nan)),
        max_merge_count=0,
    )

    with open(save_root / "hlt_stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": CONFIG["hlt_effects"], "stats": hlt_stats}, f, indent=2)

    with open(save_root / "reconstructor_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(reco_val_metrics, f, indent=2)

    with open(save_root / "constituent_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(count_summary, f, indent=2)

    with open(save_root / "budget_summary_test.json", "w", encoding="utf-8") as f:
        json.dump(budget_summary, f, indent=2)

    if not args.skip_save_models:
        torch.save({"model": teacher.state_dict(), "auc": auc_teacher}, save_root / "teacher.pt")
        torch.save({"model": baseline.state_dict(), "auc": auc_baseline}, save_root / "baseline.pt")
        torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor.pt")
        torch.save({"model": unmerge_cls.state_dict(), "auc": auc_unmerge}, save_root / "unmerge_classifier.pt")
        torch.save({"model": unmerge_flag_cls.state_dict(), "auc": auc_unmerge_flag}, save_root / "unmerge_mergeflag_classifier.pt")
        torch.save({"model": dual_cls.state_dict(), "auc": auc_dual}, save_root / "dual_view_classifier.pt")
        torch.save({"model": dual_flag_cls.state_dict(), "auc": auc_dual_flag}, save_root / "dual_view_mergeflag_classifier.pt")
        torch.save({"model": kd_student_dual.state_dict(), "auc": auc_dual_kd}, save_root / "dual_view_kd.pt")
        torch.save({"model": kd_student_dual_flag.state_dict(), "auc": auc_dual_flag_kd}, save_root / "dual_view_mergeflag_kd.pt")

    np.savez_compressed(
        save_root / "reconstructed_dataset.npz",
        const_off=const_off.astype(np.float32),
        mask_off=masks_off.astype(bool),
        hlt_const=hlt_const.astype(np.float32),
        hlt_mask=hlt_mask.astype(bool),
        reco_const=reco_const.astype(np.float32),
        reco_mask=reco_mask.astype(bool),
        reco_merge_flag=reco_merge_flag.astype(np.float32),
        reco_eff_flag=reco_eff_flag.astype(np.float32),
        created_merge_count=created_merge_count.astype(np.int32),
        created_eff_count=created_eff_count.astype(np.int32),
        budget_merge_true=budget_merge_true.astype(np.float32),
        budget_eff_true=budget_eff_true.astype(np.float32),
        budget_total_pred=pred_budget_total.astype(np.float32),
        budget_merge_pred=pred_budget_merge.astype(np.float32),
        budget_eff_pred=pred_budget_eff.astype(np.float32),
        labels=labels.astype(np.int64),
    )

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m35 hybrid top tagger:
- Load trained non-top-tagger stacks from m33 and m34 runs.
- Rebuild candidates:
    * m33: best top-like + best bg-like
    * m34: top-N global candidates (default N=2)
- Train a hybrid top tagger on HLT + fused candidates + summary features.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_detfeas_dualview as m33
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m34_globalcand_multiview as m34
from unmerge_correct_hlt import ParticleTransformer, compute_features, standardize


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HybridDataset(Dataset):
    def __init__(
        self,
        const_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        cand_const: np.ndarray,
        cand_mask: np.ndarray,
        cand_feat: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.cand_const = torch.tensor(cand_const, dtype=torch.float32)
        self.cand_mask = torch.tensor(cand_mask, dtype=torch.bool)
        self.cand_feat = torch.tensor(cand_feat, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        n = int(self.labels.shape[0])
        if sample_weight is None:
            sw = np.ones((n,), dtype=np.float32)
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.shape[0] != n:
                raise ValueError(f"sample_weight mismatch: {sw.shape[0]} vs {n}")
        self.sample_weight = torch.tensor(sw, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "const_hlt": self.const_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "cand_const": self.cand_const[i],
            "cand_mask": self.cand_mask[i],
            "cand_feat": self.cand_feat[i],
            "label": self.labels[i],
            "sample_weight": self.sample_weight[i],
        }


class HybridGatedClassifier(nn.Module):
    def __init__(
        self,
        cand_feat_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        num_cands: int = 4,
    ):
        super().__init__()
        self.num_cands = int(num_cands)
        self.hlt_encoder = m33.SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.cand_encoder = m33.SetEncoder(
            input_dim=5,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=max(2, num_layers // 2),
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(cand_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hlt_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.cand_head = nn.Sequential(
            nn.Linear(embed_dim * (1 + self.num_cands), 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * (2 + self.num_cands), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        const_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_const: torch.Tensor,
        cand_mask: torch.Tensor,
        cand_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.hlt_encoder(m33._const_to_token5(const_hlt), mask_hlt)
        c_embs = []
        for i in range(self.num_cands):
            c_i = self.cand_encoder(m33._const_to_token5(cand_const[:, i]), cand_mask[:, i])
            c_embs.append(c_i)
        c_cat = torch.cat(c_embs, dim=-1)
        f = self.feat_proj(cand_feat)

        logit_h = self.hlt_head(h).squeeze(-1)
        logit_c = self.cand_head(torch.cat(c_embs + [f], dim=-1)).squeeze(-1)
        g = self.gate(torch.cat([h, c_cat, f], dim=-1)).squeeze(-1)
        return (1.0 - g) * logit_h + g * logit_c


def _weighted_mean(vec: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.float().clamp(min=0.0)
    return (vec * w).sum() / w.sum().clamp(min=1e-6)


def _candidate_disagreement_features(cand_const: np.ndarray, cand_mask: np.ndarray) -> np.ndarray:
    # cand_const: [N, C, L, 4], cand_mask: [N, C, L]
    pt = cand_const[..., 0] * cand_mask
    eta = cand_const[..., 1]
    phi = cand_const[..., 2]
    mass = cand_const[..., 3]
    cnt = cand_mask.sum(axis=2).astype(np.float32)

    jet_pt = pt.sum(axis=2).astype(np.float32)
    px = (pt * np.cos(phi)).sum(axis=2)
    py = (pt * np.sin(phi)).sum(axis=2)
    pz = (pt * np.sinh(eta)).sum(axis=2)
    e = (np.sqrt(np.maximum((pt * np.cosh(eta)) ** 2 + (mass * mass) * cand_mask, 0.0))).sum(axis=2)
    jet_mass = np.sqrt(np.maximum(e * e - px * px - py * py - pz * pz, 0.0)).astype(np.float32)

    def _pairwise_abs(x: np.ndarray) -> np.ndarray:
        c = int(x.shape[1])
        diffs = []
        for i in range(c):
            for j in range(i + 1, c):
                diffs.append(np.abs(x[:, i] - x[:, j]))
        if not diffs:
            return np.zeros((x.shape[0], 1), dtype=np.float32)
        return np.stack(diffs, axis=1).astype(np.float32)

    d_cnt = _pairwise_abs(cnt)
    d_pt = _pairwise_abs(jet_pt)
    d_mass = _pairwise_abs(jet_mass)

    feat = np.stack(
        [
            np.std(cnt, axis=1),
            np.std(jet_pt, axis=1),
            np.std(jet_mass, axis=1),
            (np.max(cnt, axis=1) - np.min(cnt, axis=1)),
            (np.max(jet_pt, axis=1) - np.min(jet_pt, axis=1)),
            (np.max(jet_mass, axis=1) - np.min(jet_mass, axis=1)),
            np.mean(d_cnt, axis=1),
            np.max(d_cnt, axis=1),
            np.mean(d_pt, axis=1),
            np.max(d_pt, axis=1),
            np.mean(d_mass, axis=1),
            np.max(d_mass, axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    return feat


def _train_model(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[nn.Module, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_auc = float("-inf")
    best_epoch = 0
    no_imp = 0

    for ep in range(int(epochs)):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for batch in dl_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch["label"]
            sw = batch["sample_weight"]
            logit = model(
                const_hlt=batch["const_hlt"],
                mask_hlt=batch["mask_hlt"],
                cand_const=batch["cand_const"],
                cand_mask=batch["cand_mask"],
                cand_feat=batch["cand_feat"],
            )
            lv = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            loss = _weighted_mean(lv, sw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            bs = int(y.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_n += bs

        model.eval()
        vp = []
        vy = []
        vw = []
        with torch.no_grad():
            for batch in dl_va:
                batch = {k: v.to(device) for k, v in batch.items()}
                y = batch["label"]
                sw = batch["sample_weight"]
                p = torch.sigmoid(model(
                    const_hlt=batch["const_hlt"],
                    mask_hlt=batch["mask_hlt"],
                    cand_const=batch["cand_const"],
                    cand_mask=batch["cand_mask"],
                    cand_feat=batch["cand_feat"],
                ))
                vp.append(p.detach().cpu().numpy().astype(np.float64))
                vy.append(y.detach().cpu().numpy().astype(np.float64))
                vw.append(sw.detach().cpu().numpy().astype(np.float64))
        vp_np = np.concatenate(vp, axis=0) if vp else np.array([], dtype=np.float64)
        vy_np = np.concatenate(vy, axis=0) if vy else np.array([], dtype=np.float64)
        vw_np = np.concatenate(vw, axis=0) if vw else None
        va_auc = float(roc_auc_score(vy_np, vp_np, sample_weight=vw_np)) if len(np.unique(vy_np)) > 1 else 0.0

        if va_auc > best_auc:
            best_auc = float(va_auc)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"Hybrid ep {ep+1:03d}: train_loss={tr_loss/max(1,tr_n):.5f} val_auc={va_auc:.4f} best={best_auc:.4f}@{best_epoch}")
        if no_imp >= int(patience):
            print(f"Hybrid early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_auc": float(best_auc), "best_epoch": int(best_epoch)}


@torch.no_grad()
def _eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pp, yy, ww = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["label"]
        sw = batch["sample_weight"]
        p = torch.sigmoid(model(
            const_hlt=batch["const_hlt"],
            mask_hlt=batch["mask_hlt"],
            cand_const=batch["cand_const"],
            cand_mask=batch["cand_mask"],
            cand_feat=batch["cand_feat"],
        ))
        pp.append(p.detach().cpu().numpy().astype(np.float64))
        yy.append(y.detach().cpu().numpy().astype(np.float64))
        ww.append(sw.detach().cpu().numpy().astype(np.float64))
    p_np = np.concatenate(pp, axis=0) if pp else np.array([], dtype=np.float64)
    y_np = np.concatenate(yy, axis=0) if yy else np.array([], dtype=np.float64)
    w_np = np.concatenate(ww, axis=0) if ww else np.array([], dtype=np.float64)
    auc = float(roc_auc_score(y_np, p_np, sample_weight=w_np)) if len(np.unique(y_np)) > 1 else 0.0
    fpr, tpr, _ = roc_curve(y_np, p_np, sample_weight=w_np)
    fpr50 = m33.fpr_at_target_tpr(fpr, tpr, target_tpr=0.50)
    return auc, float(fpr50), p_np, y_np, w_np


def _load_stats_npz(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    p = run_dir / "data_splits.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    z = np.load(p)
    return z["means"].astype(np.float32), z["stds"].astype(np.float32)


def _load_m33_stack(args, cfg: Dict, run_dir: Path, device: torch.device):
    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline.load_state_dict(torch.load(run_dir / "baseline_hlt.pt", map_location=device)["model"])
    baseline.eval()

    ae = m33.OfflineLatentAE(
        latent_dim=int(args.latent_dim),
        slots=int(args.max_constits),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    ae.load_state_dict(torch.load(run_dir / "offline_prior_ae.pt", map_location=device)["model"])
    ae.eval()

    ps = torch.load(run_dir / "offline_prior_stats.pt", map_location=device)
    prior = m33.PriorStats(mean=ps["prior_mean"].to(device), logvar=ps["prior_logvar"].to(device))

    degrader = m33.OfflineToHLTDegrader(
        latent_dim=int(args.latent_dim),
        slots=int(args.max_constits),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    degrader.load_state_dict(torch.load(run_dir / "degrader.pt", map_location=device)["model"])
    degrader.eval()

    proposer = m33.HLTLatentProposer(
        latent_dim=int(args.latent_dim),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    proposer.load_state_dict(torch.load(run_dir / "proposer.pt", map_location=device)["model"])
    proposer.eval()

    selector = m33.CandidateRealismSelector(
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers // 2)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    selector.load_state_dict(torch.load(run_dir / "selector.pt", map_location=device)["model"])
    selector.eval()
    return baseline, ae, prior, degrader, proposer, selector


def _load_m34_stack(args, cfg: Dict, run_dir: Path, device: torch.device):
    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline.load_state_dict(torch.load(run_dir / "baseline_hlt.pt", map_location=device)["model"])
    baseline.eval()

    ae = m34.OfflineLatentAE(
        latent_dim=int(args.latent_dim),
        slots=int(args.max_constits),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    ae.load_state_dict(torch.load(run_dir / "offline_prior_ae.pt", map_location=device)["model"])
    ae.eval()

    ps = torch.load(run_dir / "offline_prior_stats.pt", map_location=device)
    prior = m34.PriorStats(mean=ps["prior_mean"].to(device), logvar=ps["prior_logvar"].to(device))

    degrader = m34.OfflineToHLTDegrader(
        latent_dim=int(args.latent_dim),
        slots=int(args.max_constits),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    degrader.load_state_dict(torch.load(run_dir / "degrader.pt", map_location=device)["model"])
    degrader.eval()

    proposer = m34.HLTLatentProposer(
        latent_dim=int(args.latent_dim),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    proposer.load_state_dict(torch.load(run_dir / "proposer.pt", map_location=device)["model"])
    proposer.eval()

    selector = m34.CandidateRealismSelector(
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers // 2)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    selector.load_state_dict(torch.load(run_dir / "selector.pt", map_location=device)["model"])
    selector.eval()
    return baseline, ae, prior, degrader, proposer, selector


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m35 hybrid from m33+m34 candidate stacks")
    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model35_hybrid_m33m34")
    p.add_argument("--run_name", type=str, default="model35_hybrid_m33m34_150k75k150k_seed0")

    p.add_argument("--m33_run_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model33_detfeas_dualview_postrefine/model33_k6_detfeas_dualview_postrefine_150k75k150k_seed0")
    p.add_argument("--m34_run_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model34_globalcand_multiview/model34_k12_globalcand_multiview3_150k75k150k_seed0")

    p.add_argument("--n_train_jets", type=int, default=375000)
    p.add_argument("--n_train_split", type=int, default=100000)
    p.add_argument("--n_val_split", type=int, default=75000)
    p.add_argument("--n_test_split", type=int, default=150000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--use_train_weights", action="store_true")

    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)

    # deterministic HLT / search controls (match m33/m34 postrefine runners by default)
    p.add_argument("--dhard_seed_offset", type=int, default=1337)
    p.add_argument("--pred_exist_threshold", type=float, default=0.08)
    p.add_argument("--unmatched_penalty", type=float, default=0.0)
    p.add_argument("--search_w_chamfer", type=float, default=1.0)
    p.add_argument("--search_w_count", type=float, default=0.30)
    p.add_argument("--search_w_pt", type=float, default=0.12)
    p.add_argument("--search_w_mass", type=float, default=0.06)

    # m33 candidate generation
    p.add_argument("--m33_k0_infer", type=int, default=160)
    p.add_argument("--m33_top_m_infer", type=int, default=24)
    p.add_argument("--m33_infer_refine_steps", type=int, default=3)
    p.add_argument("--m33_infer_refine_lr", type=float, default=0.035)
    p.add_argument("--m33_infer_refine_max_step_norm", type=float, default=0.20)
    p.add_argument("--m33_search_target_k", type=int, default=6)
    p.add_argument("--m33_search_batch_size", type=int, default=40)
    p.add_argument("--m33_search_chunk_k0", type=int, default=100)
    p.add_argument("--m33_search_shortlist_m", type=int, default=20)
    p.add_argument("--m33_search_max_rounds", type=int, default=20)
    p.add_argument("--m33_search_keep_per_round", type=int, default=10)
    p.add_argument("--m33_search_max_pool_size", type=int, default=120)
    p.add_argument("--m33_search_eps_total", type=float, default=0.20)
    p.add_argument("--m33_search_eps_count", type=float, default=0.25)
    p.add_argument("--m33_post_refine_steps", type=int, default=6)
    p.add_argument("--m33_post_refine_lr", type=float, default=0.03)
    p.add_argument("--m33_post_refine_max_step_norm", type=float, default=0.20)
    p.add_argument("--m33_post_refine_anchor_lambda", type=float, default=0.10)
    p.add_argument("--m33_post_refine_accept_margin", type=float, default=0.00)
    p.add_argument("--m33_post_refine_batch_size", type=int, default=16)
    p.add_argument("--m33_selector_score_alpha", type=float, default=1.40)
    p.add_argument("--m33_prescore_w_prior", type=float, default=0.70)
    p.add_argument("--m33_prescore_w_q", type=float, default=0.30)

    # m34 candidate generation
    p.add_argument("--m34_k0_infer", type=int, default=160)
    p.add_argument("--m34_top_m_infer", type=int, default=24)
    p.add_argument("--m34_infer_refine_steps", type=int, default=3)
    p.add_argument("--m34_infer_refine_lr", type=float, default=0.035)
    p.add_argument("--m34_infer_refine_max_step_norm", type=float, default=0.20)
    p.add_argument("--m34_search_target_k", type=int, default=12)
    p.add_argument("--m34_search_batch_size", type=int, default=40)
    p.add_argument("--m34_search_chunk_k0", type=int, default=100)
    p.add_argument("--m34_search_shortlist_m", type=int, default=20)
    p.add_argument("--m34_search_max_rounds", type=int, default=20)
    p.add_argument("--m34_search_keep_per_round", type=int, default=10)
    p.add_argument("--m34_search_max_pool_size", type=int, default=120)
    p.add_argument("--m34_search_eps_total", type=float, default=0.20)
    p.add_argument("--m34_search_eps_count", type=float, default=0.25)
    p.add_argument("--m34_post_refine_steps", type=int, default=6)
    p.add_argument("--m34_post_refine_lr", type=float, default=0.03)
    p.add_argument("--m34_post_refine_max_step_norm", type=float, default=0.20)
    p.add_argument("--m34_post_refine_anchor_lambda", type=float, default=0.10)
    p.add_argument("--m34_post_refine_accept_margin", type=float, default=0.00)
    p.add_argument("--m34_post_refine_batch_size", type=int, default=16)
    p.add_argument("--m34_selector_score_alpha", type=float, default=1.40)
    p.add_argument("--m34_prescore_w_prior", type=float, default=0.70)
    p.add_argument("--m34_prescore_w_q", type=float, default=0.30)
    p.add_argument("--m34_mv_n_select", type=int, default=2)

    # hybrid head training
    p.add_argument("--hybrid_epochs", type=int, default=80)
    p.add_argument("--hybrid_patience", type=int, default=14)
    p.add_argument("--hybrid_lr", type=float, default=1.2e-4)
    p.add_argument("--hybrid_weight_decay", type=float, default=1e-4)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    m33_run = Path(args.m33_run_dir)
    m34_run = Path(args.m34_run_dir)
    print("=" * 72)
    print("Model-35 Hybrid (m33 + m34)")
    print(f"Run: {save_root}")
    print(f"m33 run: {m33_run}")
    print(f"m34 run: {m34_run}")
    print("=" * 72)

    cfg = base._deepcopy_config()
    files = base._parse_h5_path_arg(str(args.train_path))
    max_needed = int(args.offset_jets + args.n_train_jets)
    all_const, all_labels, all_w = base.load_raw_constituents_labels_weights_from_h5(
        files=files,
        max_jets=max_needed,
        max_constits=int(args.max_constits),
        use_train_weights=bool(args.use_train_weights),
    )
    if all_const.shape[0] < max_needed:
        raise RuntimeError(f"Requested {max_needed} jets but found {all_const.shape[0]}")

    const_raw = all_const[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)
    train_w = all_w[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.float32)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0
    jet_keys = (np.arange(len(const_off), dtype=np.int64) + int(args.offset_jets)).astype(np.int64)

    const_hlt, mask_hlt, hlt_stats = m33._apply_hlt_effects_deterministic_keyed(
        const=const_off,
        mask=masks_off,
        cfg=cfg,
        jet_keys=jet_keys,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    print(
        f"HLT stats: avg_offline={hlt_stats.get('avg_offline_per_jet', float('nan')):.2f}, "
        f"avg_hlt={hlt_stats.get('avg_hlt_per_jet', float('nan')):.2f}"
    )

    feat_hlt = compute_features(const_hlt, mask_hlt)
    m33_means, m33_stds = _load_stats_npz(m33_run)
    m34_means, m34_stds = _load_stats_npz(m34_run)
    feat_hlt_std_m33 = standardize(feat_hlt, mask_hlt, m33_means, m33_stds)
    feat_hlt_std_m34 = standardize(feat_hlt, mask_hlt, m34_means, m34_stds)

    idx_all = np.arange(len(labels))
    total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > len(idx_all):
        raise ValueError(f"Split sum {total_need} exceeds dataset size {len(idx_all)}")
    if total_need < len(idx_all):
        idx_use, _ = train_test_split(
            idx_all,
            train_size=total_need,
            random_state=int(args.seed),
            stratify=labels[idx_all],
        )
    else:
        idx_use = idx_all

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
    print(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    sw_train = train_w[train_idx] if bool(args.use_train_weights) else np.ones((len(train_idx),), dtype=np.float32)
    sw_val = train_w[val_idx] if bool(args.use_train_weights) else np.ones((len(val_idx),), dtype=np.float32)
    sw_test = train_w[test_idx] if bool(args.use_train_weights) else np.ones((len(test_idx),), dtype=np.float32)

    # Load m33/m34 non-top-tagger stacks.
    m33_baseline, m33_ae, m33_prior, m33_degrader, m33_prop, m33_sel = _load_m33_stack(args, cfg, m33_run, device)
    m34_baseline, m34_ae, m34_prior, m34_degrader, m34_prop, m34_sel = _load_m34_stack(args, cfg, m34_run, device)

    # Baseline probs.
    p33_train, _ = m33._predict_probs(m33_baseline, feat_hlt_std_m33[train_idx], mask_hlt[train_idx], labels[train_idx], int(args.batch_size), int(args.num_workers), device)
    p33_val, _ = m33._predict_probs(m33_baseline, feat_hlt_std_m33[val_idx], mask_hlt[val_idx], labels[val_idx], int(args.batch_size), int(args.num_workers), device)
    p33_test, _ = m33._predict_probs(m33_baseline, feat_hlt_std_m33[test_idx], mask_hlt[test_idx], labels[test_idx], int(args.batch_size), int(args.num_workers), device)

    p34_train, _ = m34._predict_probs(m34_baseline, feat_hlt_std_m34[train_idx], mask_hlt[train_idx], labels[train_idx], int(args.batch_size), int(args.num_workers), device)
    p34_val, _ = m34._predict_probs(m34_baseline, feat_hlt_std_m34[val_idx], mask_hlt[val_idx], labels[val_idx], int(args.batch_size), int(args.num_workers), device)
    p34_test, _ = m34._predict_probs(m34_baseline, feat_hlt_std_m34[test_idx], mask_hlt[test_idx], labels[test_idx], int(args.batch_size), int(args.num_workers), device)

    # Build m33 pools/features.
    def _build_m33(split_idx: np.ndarray, p_base: np.ndarray):
        pools = m33._chunked_search_candidates_split(
            proposer=m33_prop,
            ae=m33_ae,
            degrader=m33_degrader,
            prior=m33_prior,
            const_hlt=const_hlt[split_idx],
            mask_hlt=mask_hlt[split_idx],
            jet_keys=jet_keys[split_idx],
            cfg=cfg,
            base_seed=int(args.seed + args.dhard_seed_offset),
            device=device,
            batch_size=int(args.m33_search_batch_size),
            target_k=int(args.m33_search_target_k),
            chunk_k0=int(args.m33_search_chunk_k0),
            shortlist_m=int(args.m33_search_shortlist_m),
            max_rounds=int(args.m33_search_max_rounds),
            keep_per_round=int(args.m33_search_keep_per_round),
            max_pool_size=int(args.m33_search_max_pool_size),
            eps_total=float(args.m33_search_eps_total),
            eps_count=float(args.m33_search_eps_count),
            pred_exist_threshold=float(args.pred_exist_threshold),
            unmatched_penalty=float(args.unmatched_penalty),
            prescore_w_prior=float(args.m33_prescore_w_prior),
            prescore_w_q=float(args.m33_prescore_w_q),
            refine_steps=int(args.m33_infer_refine_steps),
            refine_step_size=float(args.m33_infer_refine_lr),
            refine_max_step_norm=float(args.m33_infer_refine_max_step_norm),
            post_accept_refine_steps=int(args.m33_post_refine_steps),
            post_accept_refine_step_size=float(args.m33_post_refine_lr),
            post_accept_refine_max_step_norm=float(args.m33_post_refine_max_step_norm),
            post_accept_refine_anchor_lambda=float(args.m33_post_refine_anchor_lambda),
            post_accept_refine_accept_margin=float(args.m33_post_refine_accept_margin),
            post_accept_refine_batch_size=int(args.m33_post_refine_batch_size),
            w_chamfer=float(args.search_w_chamfer),
            w_count=float(args.search_w_count),
            w_pt=float(args.search_w_pt),
            w_mass=float(args.search_w_mass),
        )
        sel_bg = m33._score_selector_candidates(
            m33_sel, const_hlt[split_idx], mask_hlt[split_idx],
            pools["class0"]["const"], pools["class0"]["mask"], pools["class0"]["res_total"],
            cand_class_val=0, batch_size=int(args.batch_size), device=device
        )
        sel_tp = m33._score_selector_candidates(
            m33_sel, const_hlt[split_idx], mask_hlt[split_idx],
            pools["class1"]["const"], pools["class1"]["mask"], pools["class1"]["res_total"],
            cand_class_val=1, batch_size=int(args.batch_size), device=device
        )
        c0 = pools["class0"]
        c1 = pools["class1"]
        row = np.arange(c0["res_total"].shape[0])
        score_alpha = float(args.m33_selector_score_alpha)
        q_bg = sel_bg - score_alpha * c0["res_total"]
        q_tp = sel_tp - score_alpha * c1["res_total"]
        idx_bg = np.argmax(q_bg, axis=1)
        idx_tp = np.argmax(q_tp, axis=1)

        c0_res_pre = c0["res_total_pre"] if "res_total_pre" in c0 else c0["res_total"]
        c1_res_pre = c1["res_total_pre"] if "res_total_pre" in c1 else c1["res_total"]
        c0_round = c0["round_found"] if "round_found" in c0 else np.full_like(c0["res_total"], float(args.m33_search_max_rounds + 1))
        c1_round = c1["round_found"] if "round_found" in c1 else np.full_like(c1["res_total"], float(args.m33_search_max_rounds + 1))
        tgt_k = float(max(1, int(pools.get("stats", {}).get("target_k", c0["res_total"].shape[1]))))

        m33_extra = np.stack(
            [
                c1["feasible_count"].astype(np.float32) / tgt_k,                 # top accept frac
                c0["feasible_count"].astype(np.float32) / tgt_k,                 # bg accept frac
                np.min(c1["res_total"], axis=1).astype(np.float32),              # top best residual
                np.min(c0["res_total"], axis=1).astype(np.float32),              # bg best residual
                (float(args.m33_search_eps_total) - np.min(c1["res_total"], axis=1)).astype(np.float32),  # top gap to threshold
                (float(args.m33_search_eps_total) - np.min(c0["res_total"], axis=1)).astype(np.float32),  # bg gap to threshold
                c1_round[row, idx_tp].astype(np.float32),                        # selected top round
                c0_round[row, idx_bg].astype(np.float32),                        # selected bg round
                (c1_res_pre[row, idx_tp] - c1["res_total"][row, idx_tp]).astype(np.float32),  # top pre->post improve
                (c0_res_pre[row, idx_bg] - c0["res_total"][row, idx_bg]).astype(np.float32),  # bg pre->post improve
            ],
            axis=1,
        ).astype(np.float32)

        dv = m33._build_dualview_features(
            pools=pools,
            sel_score_bg=sel_bg,
            sel_score_top=sel_tp,
            baseline_prob=p_base,
            score_alpha=score_alpha,
        )
        return dv, pools["stats"], m33_extra

    dv_tr, m33_stats_tr, m33x_tr = _build_m33(train_idx, p33_train)
    dv_va, m33_stats_va, m33x_va = _build_m33(val_idx, p33_val)
    dv_te, m33_stats_te, m33x_te = _build_m33(test_idx, p33_test)

    # Build m34 pools/features (global pool).
    def _build_m34(split_idx: np.ndarray, p_base: np.ndarray):
        pools = m34._chunked_search_candidates_split(
            proposer=m34_prop,
            ae=m34_ae,
            degrader=m34_degrader,
            prior=m34_prior,
            const_hlt=const_hlt[split_idx],
            mask_hlt=mask_hlt[split_idx],
            jet_keys=jet_keys[split_idx],
            cfg=cfg,
            base_seed=int(args.seed + args.dhard_seed_offset),
            device=device,
            batch_size=int(args.m34_search_batch_size),
            target_k=int(args.m34_search_target_k),
            chunk_k0=int(args.m34_search_chunk_k0),
            shortlist_m=int(args.m34_search_shortlist_m),
            max_rounds=int(args.m34_search_max_rounds),
            keep_per_round=int(args.m34_search_keep_per_round),
            max_pool_size=int(args.m34_search_max_pool_size),
            eps_total=float(args.m34_search_eps_total),
            eps_count=float(args.m34_search_eps_count),
            pred_exist_threshold=float(args.pred_exist_threshold),
            unmatched_penalty=float(args.unmatched_penalty),
            prescore_w_prior=float(args.m34_prescore_w_prior),
            prescore_w_q=float(args.m34_prescore_w_q),
            refine_steps=int(args.m34_infer_refine_steps),
            refine_step_size=float(args.m34_infer_refine_lr),
            refine_max_step_norm=float(args.m34_infer_refine_max_step_norm),
            post_accept_refine_steps=int(args.m34_post_refine_steps),
            post_accept_refine_step_size=float(args.m34_post_refine_lr),
            post_accept_refine_max_step_norm=float(args.m34_post_refine_max_step_norm),
            post_accept_refine_anchor_lambda=float(args.m34_post_refine_anchor_lambda),
            post_accept_refine_accept_margin=float(args.m34_post_refine_accept_margin),
            post_accept_refine_batch_size=int(args.m34_post_refine_batch_size),
            w_chamfer=float(args.search_w_chamfer),
            w_count=float(args.search_w_count),
            w_pt=float(args.search_w_pt),
            w_mass=float(args.search_w_mass),
            search_cls_values=(0,),
        )
        sel_sc = m34._score_selector_candidates(
            m34_sel, const_hlt[split_idx], mask_hlt[split_idx],
            pools["class0"]["const"], pools["class0"]["mask"], pools["class0"]["res_total"],
            cand_class_val=0, batch_size=int(args.batch_size), device=device
        )
        c0 = pools["class0"]
        score_alpha = float(args.m34_selector_score_alpha)
        q = sel_sc - score_alpha * c0["res_total"]
        s = int(max(1, min(int(args.m34_mv_n_select), int(c0["res_total"].shape[1]))))
        idx = np.argsort(-q, axis=1)[:, :s]
        row = np.arange(c0["res_total"].shape[0])[:, None]

        sel_res_post = c0["res_total"][row, idx]
        c0_res_pre = c0["res_total_pre"] if "res_total_pre" in c0 else c0["res_total"]
        sel_res_pre = c0_res_pre[row, idx]
        c0_round = c0["round_found"] if "round_found" in c0 else np.full_like(c0["res_total"], float(args.m34_search_max_rounds + 1))
        sel_round = c0_round[row, idx]
        tgt_k = float(max(1, int(pools.get("stats", {}).get("target_k", c0["res_total"].shape[1]))))

        # Fixed-size summary for m34 extra signals (pad by repeating first candidate if needed).
        r0 = sel_round[:, 0].astype(np.float32)
        r1 = sel_round[:, 1].astype(np.float32) if s >= 2 else sel_round[:, 0].astype(np.float32)
        d0 = (sel_res_pre[:, 0] - sel_res_post[:, 0]).astype(np.float32)
        d1 = (sel_res_pre[:, 1] - sel_res_post[:, 1]).astype(np.float32) if s >= 2 else d0
        m34_extra = np.stack(
            [
                c0["feasible_count"].astype(np.float32) / tgt_k,                 # accept frac
                np.min(c0["res_total"], axis=1).astype(np.float32),              # best residual
                (float(args.m34_search_eps_total) - np.min(c0["res_total"], axis=1)).astype(np.float32),  # gap to threshold
                r0,                                                                # selected round #1
                r1,                                                                # selected round #2
                d0,                                                                # pre->post improve #1
                d1,                                                                # pre->post improve #2
                np.mean((sel_res_pre - sel_res_post).astype(np.float32), axis=1), # mean pre->post improve across selected
            ],
            axis=1,
        ).astype(np.float32)

        mv = m34._build_multiview3_features(
            pool=pools["class0"],
            sel_score=sel_sc,
            baseline_prob=p_base,
            score_alpha=score_alpha,
            n_select=int(args.m34_mv_n_select),
        )
        return mv, pools["stats"], m34_extra

    mv_tr, m34_stats_tr, m34x_tr = _build_m34(train_idx, p34_train)
    mv_va, m34_stats_va, m34x_va = _build_m34(val_idx, p34_val)
    mv_te, m34_stats_te, m34x_te = _build_m34(test_idx, p34_test)

    def _build_hybrid(
        dv: Dict[str, np.ndarray],
        mv: Dict[str, np.ndarray],
        m33_extra: np.ndarray,
        m34_extra: np.ndarray,
        p33: np.ndarray,
        p34: np.ndarray,
    ):
        cand_const = np.concatenate(
            [
                dv["cand_top_const"][:, None, :, :],
                dv["cand_bg_const"][:, None, :, :],
                mv["cand_const"],
            ],
            axis=1,
        ).astype(np.float32)
        cand_mask = np.concatenate(
            [
                dv["cand_top_mask"][:, None, :],
                dv["cand_bg_mask"][:, None, :],
                mv["cand_mask"],
            ],
            axis=1,
        ).astype(bool)
        extra = np.stack(
            [p33.astype(np.float32), p34.astype(np.float32), (p33 - p34).astype(np.float32)],
            axis=1,
        )
        disagree = _candidate_disagreement_features(cand_const, cand_mask)
        cand_feat = np.concatenate(
            [dv["cand_feat"], mv["cand_feat"], m33_extra, m34_extra, disagree, extra],
            axis=1,
        ).astype(np.float32)
        return cand_const, cand_mask, cand_feat

    hc_tr, hm_tr, hf_tr = _build_hybrid(dv_tr, mv_tr, m33x_tr, m34x_tr, p33_train, p34_train)
    hc_va, hm_va, hf_va = _build_hybrid(dv_va, mv_va, m33x_va, m34x_va, p33_val, p34_val)
    hc_te, hm_te, hf_te = _build_hybrid(dv_te, mv_te, m33x_te, m34x_te, p33_test, p34_test)

    ds_tr = HybridDataset(const_hlt[train_idx], mask_hlt[train_idx], hc_tr, hm_tr, hf_tr, labels[train_idx], sw_train)
    ds_va = HybridDataset(const_hlt[val_idx], mask_hlt[val_idx], hc_va, hm_va, hf_va, labels[val_idx], sw_val)
    ds_te = HybridDataset(const_hlt[test_idx], mask_hlt[test_idx], hc_te, hm_te, hf_te, labels[test_idx], sw_test)
    dl_tr = DataLoader(ds_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_va = DataLoader(ds_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_te = DataLoader(ds_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    model = HybridGatedClassifier(
        cand_feat_dim=int(hf_tr.shape[1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers // 2)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        num_cands=int(hc_tr.shape[1]),
    ).to(device)
    model, metrics = _train_model(
        model=model,
        dl_tr=dl_tr,
        dl_va=dl_va,
        device=device,
        epochs=int(args.hybrid_epochs),
        lr=float(args.hybrid_lr),
        weight_decay=float(args.hybrid_weight_decay),
        patience=int(args.hybrid_patience),
    )
    auc_hyb, fpr50_hyb, pred_hyb, y_test, w_test = _eval_model(model, dl_te, device)

    # Baseline references.
    auc_m33b = float(roc_auc_score(labels[test_idx], p33_test, sample_weight=sw_test))
    fpr_m33, tpr_m33, _ = roc_curve(labels[test_idx], p33_test, sample_weight=sw_test)
    fpr50_m33b = m33.fpr_at_target_tpr(fpr_m33, tpr_m33, 0.50)
    auc_m34b = float(roc_auc_score(labels[test_idx], p34_test, sample_weight=sw_test))
    fpr_m34, tpr_m34, _ = roc_curve(labels[test_idx], p34_test, sample_weight=sw_test)
    fpr50_m34b = m33.fpr_at_target_tpr(fpr_m34, tpr_m34, 0.50)

    print("\n" + "=" * 72)
    print("FINAL TEST")
    print("=" * 72)
    print(
        f"m33-baseline AUC={auc_m33b:.4f} FPR50={fpr50_m33b:.6f} | "
        f"m34-baseline AUC={auc_m34b:.4f} FPR50={fpr50_m34b:.6f} | "
        f"m35-hybrid AUC={auc_hyb:.4f} FPR50={fpr50_hyb:.6f}"
    )

    torch.save({"model": model.state_dict(), "metrics": metrics}, save_root / "hybrid_top_tagger.pt")
    np.savez_compressed(
        save_root / "m35_test_scores.npz",
        labels_test=y_test.astype(np.float32),
        preds_m35_hybrid=pred_hyb.astype(np.float32),
        preds_m33_baseline=p33_test.astype(np.float32),
        preds_m34_baseline=p34_test.astype(np.float32),
        sample_weight=w_test.astype(np.float32),
        auc_m33_baseline=float(auc_m33b),
        auc_m34_baseline=float(auc_m34b),
        auc_m35_hybrid=float(auc_hyb),
        fpr50_m33_baseline=float(fpr50_m33b),
        fpr50_m34_baseline=float(fpr50_m34b),
        fpr50_m35_hybrid=float(fpr50_hyb),
    )
    report = {
        "model": "m35_hybrid_m33m34",
        "seed": int(args.seed),
        "m33_run_dir": str(m33_run),
        "m34_run_dir": str(m34_run),
        "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "search_stats": {
            "m33_train": m33_stats_tr,
            "m33_val": m33_stats_va,
            "m33_test": m33_stats_te,
            "m34_train": m34_stats_tr,
            "m34_val": m34_stats_va,
            "m34_test": m34_stats_te,
        },
        "scores": {
            "m33_baseline": {"auc": float(auc_m33b), "fpr50": float(fpr50_m33b)},
            "m34_baseline": {"auc": float(auc_m34b), "fpr50": float(fpr50_m34b)},
            "m35_hybrid": {"auc": float(auc_hyb), "fpr50": float(fpr50_hyb), "train_metrics": metrics},
        },
    }
    with open(save_root / "m35_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()

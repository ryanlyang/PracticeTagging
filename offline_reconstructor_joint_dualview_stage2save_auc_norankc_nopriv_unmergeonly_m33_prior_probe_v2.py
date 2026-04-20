#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m33 prior probe v2: stronger Offline realism manifold diagnostics.

Upgrades vs v1:
1) Class-conditional CVAE prior instead of AE.
2) Latent class supervision + KL warmup + existence BCE for count stability.
3) Class-conditional GMM sampling (multi-modal) instead of single Gaussian per class.
4) Same diagnostics: fidelity, realism critic, class sanity, D_hard roundtrip.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_detfeas_dualview as m33
from unmerge_correct_hlt import compute_features, get_stats, standardize, ParticleTransformer


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _jet_mass_np(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = mask.astype(np.float32)
    pt = const[..., 0] * w
    eta = const[..., 1]
    phi = const[..., 2]
    e = const[..., 3] * w
    px = (pt * np.cos(phi)).sum(axis=1)
    py = (pt * np.sin(phi)).sum(axis=1)
    pz = (pt * np.sinh(eta)).sum(axis=1)
    et = e.sum(axis=1)
    m2 = et * et - px * px - py * py - pz * pz
    return np.sqrt(np.clip(m2, 0.0, None)).astype(np.float32)


def _summary_vectors(const: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    m = mask.astype(np.float32)
    cnt = m.sum(axis=1).astype(np.float32)
    jet_pt = (const[..., 0] * m).sum(axis=1).astype(np.float32)
    jet_mass = _jet_mass_np(const, mask)
    abs_eta_mean = ((np.abs(const[..., 1]) * m).sum(axis=1) / (cnt + 1e-6)).astype(np.float32)
    pt_mean = ((const[..., 0] * m).sum(axis=1) / (cnt + 1e-6)).astype(np.float32)
    pt2_mean = ((const[..., 0] ** 2 * m).sum(axis=1) / (cnt + 1e-6)).astype(np.float32)
    pt_std = np.sqrt(np.clip(pt2_mean - pt_mean * pt_mean, 0.0, None)).astype(np.float32)
    return {
        "count": cnt,
        "jet_pt": jet_pt,
        "jet_mass": jet_mass,
        "abs_eta_mean": abs_eta_mean,
        "pt_mean": pt_mean,
        "pt_std": pt_std,
    }


def _ks_stat_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdfa = np.searchsorted(a, grid, side="right") / float(max(1, a.size))
    cdfb = np.searchsorted(b, grid, side="right") / float(max(1, b.size))
    return float(np.max(np.abs(cdfa - cdfb)))


def _critic_features(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s = _summary_vectors(const, mask)
    x = np.stack(
        [
            s["count"],
            np.log1p(np.clip(s["jet_pt"], 0.0, None)),
            np.log1p(np.clip(s["jet_mass"], 0.0, None)),
            s["abs_eta_mean"],
            np.log1p(np.clip(s["pt_mean"], 0.0, None)),
            np.log1p(np.clip(s["pt_std"], 0.0, None)),
        ],
        axis=1,
    ).astype(np.float32)
    return x


class _CriticMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_critic_auc(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> Dict[str, float]:
    x_tr, x_va, y_tr, y_va = train_test_split(
        x,
        y,
        train_size=0.8,
        random_state=int(seed),
        stratify=y.astype(np.int64),
    )
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = x_tr.std(axis=0, keepdims=True) + 1e-6
    x_tr = (x_tr - mu) / sd
    x_va = (x_va - mu) / sd

    ds_tr = TensorDataset(torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    ds_va = TensorDataset(torch.tensor(x_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32))
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=int(batch_size), shuffle=False)

    model = _CriticMLP(dim=x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_auc = float("-inf")
    best_epoch = 0
    best_state = None

    for ep in range(int(epochs)):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        pp = []
        yy = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                p = torch.sigmoid(model(xb)).detach().cpu().numpy().astype(np.float64)
                pp.append(p)
                yy.append(yb.numpy().astype(np.float64))
        p_np = np.concatenate(pp, axis=0) if pp else np.array([], dtype=np.float64)
        y_np = np.concatenate(yy, axis=0) if yy else np.array([], dtype=np.float64)
        auc = float(roc_auc_score(y_np, p_np)) if len(np.unique(y_np)) > 1 else 0.0
        if auc > best_auc:
            best_auc = auc
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_auc": float(best_auc), "best_epoch": int(best_epoch)}


class ConditionalOfflineCVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        slots: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        num_classes: int = 2,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)

        self.encoder = m33.SetEncoder(
            input_dim=5,
            embed_dim=self.embed_dim,
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
        )
        self.class_emb = nn.Embedding(self.num_classes, self.embed_dim)
        self.posterior = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )
        self.mu_head = nn.Linear(self.embed_dim, self.latent_dim)
        self.logvar_head = nn.Linear(self.embed_dim, self.latent_dim)

        self.decoder = m33.SetDecoder(
            latent_dim=self.latent_dim,
            slots=int(slots),
            embed_dim=self.embed_dim,
            num_heads=int(num_heads),
            num_layers=max(2, int(num_layers) // 2),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            num_classes=self.num_classes,
            use_class_cond=True,
        )
        self.class_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.num_classes),
        )

    def encode_posterior(self, const: torch.Tensor, mask: torch.Tensor, cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = m33._const_to_token5(const)
        h = self.encoder(x, mask)
        c = self.class_emb(cls.long())
        u = self.posterior(torch.cat([h, c], dim=-1))
        mu = self.mu_head(u)
        logvar = torch.clamp(self.logvar_head(u), min=-8.0, max=6.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor, cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(z, class_idx=cls.long())

    def forward(self, const: torch.Tensor, mask: torch.Tensor, cls: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode_posterior(const, mask, cls)
        z = self.reparameterize(mu, logvar)
        pred_const, pred_exist_logit = self.decode(z, cls)
        cls_logit = self.class_head(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "pred_const": pred_const,
            "pred_exist_logit": pred_exist_logit,
            "cls_logit": cls_logit,
        }


@dataclass
class FidelityMetrics:
    set_mean: float
    count_mean: float
    pt_rel_mean: float
    mass_rel_mean: float
    set_mean_top: float
    set_mean_bg: float
    count_mean_top: float
    count_mean_bg: float


def _eval_prior_fidelity(
    model: ConditionalOfflineCVAE,
    const: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    pred_exist_threshold: float,
    unmatched_penalty: float,
) -> FidelityMetrics:
    model.eval()
    n = int(const.shape[0])
    set_all = []
    cnt_all = []
    pt_all = []
    mass_all = []
    lab_all = []

    with torch.no_grad():
        for s in range(0, n, int(batch_size)):
            e = min(n, s + int(batch_size))
            c = torch.tensor(const[s:e], dtype=torch.float32, device=device)
            m = torch.tensor(mask[s:e], dtype=torch.bool, device=device)
            y = torch.tensor(labels[s:e], dtype=torch.long, device=device)

            mu, logvar = model.encode_posterior(c, m, y)
            pc, pl = model.decode(mu, y)
            pw = torch.sigmoid(pl)
            pm = pw > float(pred_exist_threshold)

            l_set = m33._set_loss_chamfer_vec(
                pred_const=pc,
                pred_w=pw,
                tgt_const=c,
                tgt_mask=m,
                unmatched_penalty=float(unmatched_penalty),
            )
            l_cnt = m33._count_loss_vec(pw, m)

            p_pt = (pc[..., 0] * pm.float()).sum(dim=1)
            t_pt = (c[..., 0] * m.float()).sum(dim=1)
            l_pt = torch.abs(p_pt - t_pt) / (t_pt + 1e-6)

            p_mass = m33._jet_mass_vec(pc, pm)
            t_mass = m33._jet_mass_vec(c, m)
            l_mass = torch.abs(p_mass - t_mass) / (t_mass + 1e-6)

            set_all.append(l_set.detach().cpu().numpy())
            cnt_all.append(l_cnt.detach().cpu().numpy())
            pt_all.append(l_pt.detach().cpu().numpy())
            mass_all.append(l_mass.detach().cpu().numpy())
            lab_all.append(y.detach().cpu().numpy())

    set_np = np.concatenate(set_all, axis=0).astype(np.float64)
    cnt_np = np.concatenate(cnt_all, axis=0).astype(np.float64)
    pt_np = np.concatenate(pt_all, axis=0).astype(np.float64)
    mass_np = np.concatenate(mass_all, axis=0).astype(np.float64)
    y_np = np.concatenate(lab_all, axis=0).astype(np.int64)

    top = y_np == 1
    bg = y_np == 0

    def _m(x: np.ndarray, mm: np.ndarray) -> float:
        return float(x[mm].mean()) if mm.any() else float("nan")

    return FidelityMetrics(
        set_mean=float(set_np.mean()),
        count_mean=float(cnt_np.mean()),
        pt_rel_mean=float(pt_np.mean()),
        mass_rel_mean=float(mass_np.mean()),
        set_mean_top=_m(set_np, top),
        set_mean_bg=_m(set_np, bg),
        count_mean_top=_m(cnt_np, top),
        count_mean_bg=_m(cnt_np, bg),
    )


def _collect_latent_means(
    model: ConditionalOfflineCVAE,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    z_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            c = batch["const_off"].to(device)
            m = batch["mask_off"].to(device)
            y = batch["label"].to(device)
            mu, _ = model.encode_posterior(c, m, y)
            z_all.append(mu.detach().cpu().numpy().astype(np.float32))
            y_all.append(y.detach().cpu().numpy().astype(np.int64))
    return np.concatenate(z_all, axis=0), np.concatenate(y_all, axis=0)


def _fit_class_gmms(
    z_mu: np.ndarray,
    y: np.ndarray,
    n_components: int,
    reg_covar: float,
    seed: int,
) -> Dict[int, GaussianMixture]:
    gmms: Dict[int, GaussianMixture] = {}
    for cls in [0, 1]:
        zc = z_mu[y == cls]
        if zc.shape[0] < 16:
            raise RuntimeError(f"Not enough latent points for class {cls}: {zc.shape[0]}")
        k = int(max(1, min(int(n_components), zc.shape[0] // 8)))
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=float(reg_covar),
            random_state=int(seed) + int(cls) * 17,
            max_iter=300,
            n_init=3,
        )
        gmm.fit(zc.astype(np.float64))
        gmms[cls] = gmm
    return gmms


def _sample_from_class_gmms(
    model: ConditionalOfflineCVAE,
    gmms: Dict[int, GaussianMixture],
    n_per_cls: int,
    pred_exist_threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen_const_l = []
    gen_mask_l = []
    gen_cls_l = []
    model.eval()
    with torch.no_grad():
        for cls in [0, 1]:
            z_np, _ = gmms[cls].sample(n_samples=int(n_per_cls))
            z = torch.tensor(z_np.astype(np.float32), dtype=torch.float32, device=device)
            c = torch.full((int(n_per_cls),), int(cls), dtype=torch.long, device=device)
            gc, gl = model.decode(z, c)
            gm = torch.sigmoid(gl) > float(pred_exist_threshold)
            gen_const_l.append(gc.detach().cpu().numpy().astype(np.float32))
            gen_mask_l.append(gm.detach().cpu().numpy().astype(bool))
            gen_cls_l.append(np.full((int(n_per_cls),), int(cls), dtype=np.int64))

    return (
        np.concatenate(gen_const_l, axis=0),
        np.concatenate(gen_mask_l, axis=0),
        np.concatenate(gen_cls_l, axis=0),
    )


def _train_cvae_prior(
    model: ConditionalOfflineCVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    kl_beta_max: float,
    kl_warmup_epochs: int,
    loss_w_count: float,
    loss_w_exist: float,
    loss_w_pt: float,
    loss_w_mass: float,
    loss_w_cls: float,
    loss_w_sep: float,
    unmatched_penalty: float,
) -> Tuple[ConditionalOfflineCVAE, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    no_imp = 0

    for ep in range(int(epochs)):
        model.train()
        tr_n = 0
        tr_loss = 0.0
        tr_set = 0.0
        tr_cnt = 0.0
        tr_kl = 0.0
        tr_cls = 0.0
        tr_sep = 0.0

        beta = float(kl_beta_max) * min(1.0, float(ep + 1) / float(max(1, kl_warmup_epochs)))

        for batch in train_loader:
            c = batch["const_off"].to(device)
            m = batch["mask_off"].to(device)
            y = batch["label"].to(device)
            sw = batch["sample_weight"].to(device)

            out = model(c, m, y)
            pc = out["pred_const"]
            pl = out["pred_exist_logit"]
            pw = torch.sigmoid(pl)
            pm = pw > 0.08

            l_set = m33._set_loss_chamfer_vec(pc, pw, c, m, unmatched_penalty=float(unmatched_penalty))
            l_cnt = m33._count_loss_vec(pw, m)
            l_exist = F.binary_cross_entropy_with_logits(pl, m.float(), reduction="none").mean(dim=1)

            p_pt = (pc[..., 0] * pm.float()).sum(dim=1)
            t_pt = (c[..., 0] * m.float()).sum(dim=1)
            l_pt = torch.abs(p_pt - t_pt) / (t_pt + 1e-6)

            p_mass = m33._jet_mass_vec(pc, pm)
            t_mass = m33._jet_mass_vec(c, m)
            l_mass = torch.abs(p_mass - t_mass) / (t_mass + 1e-6)

            mu = out["mu"]
            logvar = out["logvar"]
            l_kl = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=1)

            l_cls = F.cross_entropy(out["cls_logit"], y, reduction="none")

            z = out["z"]
            z0 = z[y == 0]
            z1 = z[y == 1]
            if z0.shape[0] > 0 and z1.shape[0] > 0:
                d = torch.norm(z0.mean(dim=0) - z1.mean(dim=0), p=2)
                l_sep = torch.exp(-d).repeat(c.shape[0])
            else:
                l_sep = torch.zeros((c.shape[0],), device=device, dtype=torch.float32)

            total_vec = (
                l_set
                + float(loss_w_count) * l_cnt
                + float(loss_w_exist) * l_exist
                + float(loss_w_pt) * l_pt
                + float(loss_w_mass) * l_mass
                + beta * l_kl
                + float(loss_w_cls) * l_cls
                + float(loss_w_sep) * l_sep
            )
            loss = m33._weighted_mean(total_vec, sw)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bs = int(c.shape[0])
            tr_n += bs
            tr_loss += float(loss.item()) * bs
            tr_set += float(m33._weighted_mean(l_set.detach(), sw).item()) * bs
            tr_cnt += float(m33._weighted_mean(l_cnt.detach(), sw).item()) * bs
            tr_kl += float(m33._weighted_mean(l_kl.detach(), sw).item()) * bs
            tr_cls += float(m33._weighted_mean(l_cls.detach(), sw).item()) * bs
            tr_sep += float(m33._weighted_mean(l_sep.detach(), sw).item()) * bs

        model.eval()
        va_n = 0
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                c = batch["const_off"].to(device)
                m = batch["mask_off"].to(device)
                y = batch["label"].to(device)
                sw = batch["sample_weight"].to(device)

                out = model(c, m, y)
                pc = out["pred_const"]
                pl = out["pred_exist_logit"]
                pw = torch.sigmoid(pl)
                pm = pw > 0.08

                l_set = m33._set_loss_chamfer_vec(pc, pw, c, m, unmatched_penalty=float(unmatched_penalty))
                l_cnt = m33._count_loss_vec(pw, m)
                l_exist = F.binary_cross_entropy_with_logits(pl, m.float(), reduction="none").mean(dim=1)

                p_pt = (pc[..., 0] * pm.float()).sum(dim=1)
                t_pt = (c[..., 0] * m.float()).sum(dim=1)
                l_pt = torch.abs(p_pt - t_pt) / (t_pt + 1e-6)

                p_mass = m33._jet_mass_vec(pc, pm)
                t_mass = m33._jet_mass_vec(c, m)
                l_mass = torch.abs(p_mass - t_mass) / (t_mass + 1e-6)

                mu = out["mu"]
                logvar = out["logvar"]
                l_kl = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=1)
                l_cls = F.cross_entropy(out["cls_logit"], y, reduction="none")

                total_vec = (
                    l_set
                    + float(loss_w_count) * l_cnt
                    + float(loss_w_exist) * l_exist
                    + float(loss_w_pt) * l_pt
                    + float(loss_w_mass) * l_mass
                    + beta * l_kl
                    + float(loss_w_cls) * l_cls
                )
                loss = m33._weighted_mean(total_vec, sw)

                bs = int(c.shape[0])
                va_n += bs
                va_loss += float(loss.item()) * bs

        tr = tr_loss / max(1, tr_n)
        va = va_loss / max(1, va_n)

        if va < best_val:
            best_val = float(va)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"PriorV2 ep {ep+1:03d}: train={tr:.5f} set={tr_set/max(1,tr_n):.5f} cnt={tr_cnt/max(1,tr_n):.5f} "
                f"kl={tr_kl/max(1,tr_n):.5f} cls={tr_cls/max(1,tr_n):.5f} sep={tr_sep/max(1,tr_n):.5f} "
                f"beta={beta:.3f} val={va:.5f} best={best_val:.5f}@{best_epoch}"
            )

        if no_imp >= int(patience):
            print(f"PriorV2 early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val": float(best_val), "best_epoch": int(best_epoch)}


def _gate(value: float, good: float, warn: float, higher_is_better: bool) -> str:
    if not np.isfinite(value):
        return "FAIL"
    if higher_is_better:
        if value >= good:
            return "PASS"
        if value >= warn:
            return "WARN"
        return "FAIL"
    if value <= good:
        return "PASS"
    if value <= warn:
        return "WARN"
    return "FAIL"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m33 prior probe v2 diagnostics")
    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_debug/m33_prior_probe_v2")
    p.add_argument("--run_name", type=str, default="m33_prior_probe_v2_debug_seed0")

    p.add_argument("--n_train_jets", type=int, default=180000)
    p.add_argument("--n_train_split", type=int, default=100000)
    p.add_argument("--n_val_split", type=int, default=25000)
    p.add_argument("--n_test_split", type=int, default=35000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=80)

    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)

    p.add_argument("--prior_epochs", type=int, default=70)
    p.add_argument("--prior_patience", type=int, default=12)
    p.add_argument("--prior_lr", type=float, default=1e-4)
    p.add_argument("--prior_weight_decay", type=float, default=2e-4)
    p.add_argument("--prior_kl_beta_max", type=float, default=0.05)
    p.add_argument("--prior_kl_warmup_epochs", type=int, default=12)
    p.add_argument("--prior_loss_w_count", type=float, default=0.55)
    p.add_argument("--prior_loss_w_exist", type=float, default=0.35)
    p.add_argument("--prior_loss_w_pt", type=float, default=0.10)
    p.add_argument("--prior_loss_w_mass", type=float, default=0.06)
    p.add_argument("--prior_loss_w_cls", type=float, default=0.25)
    p.add_argument("--prior_loss_w_sep", type=float, default=0.04)

    p.add_argument("--gmm_components", type=int, default=16)
    p.add_argument("--gmm_reg_covar", type=float, default=1e-4)

    p.add_argument("--teacher_epochs", type=int, default=20)
    p.add_argument("--teacher_patience", type=int, default=6)
    p.add_argument("--teacher_lr", type=float, default=3e-4)
    p.add_argument("--teacher_weight_decay", type=float, default=1e-4)
    p.add_argument("--teacher_warmup_epochs", type=int, default=2)

    p.add_argument("--pred_exist_threshold", type=float, default=0.08)
    p.add_argument("--unmatched_penalty", type=float, default=0.0)

    p.add_argument("--n_prior_samples_per_class", type=int, default=5000)
    p.add_argument("--critic_epochs", type=int, default=20)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--critic_weight_decay", type=float, default=1e-4)
    p.add_argument("--critic_batch_size", type=int, default=512)

    p.add_argument("--roundtrip_eval_count", type=int, default=6000)
    p.add_argument("--roundtrip_w_chamfer", type=float, default=1.0)
    p.add_argument("--roundtrip_w_count", type=float, default=0.30)
    p.add_argument("--roundtrip_w_pt", type=float, default=0.12)
    p.add_argument("--roundtrip_w_mass", type=float, default=0.06)
    p.add_argument("--dhard_seed_offset", type=int, default=1337)

    p.add_argument("--gate_overfit_ratio_good", type=float, default=1.30)
    p.add_argument("--gate_overfit_ratio_warn", type=float, default=1.55)
    p.add_argument("--gate_critic_auc_good", type=float, default=0.72)
    p.add_argument("--gate_critic_auc_warn", type=float, default=0.82)
    p.add_argument("--gate_class_sep_good", type=float, default=0.25)
    p.add_argument("--gate_class_sep_warn", type=float, default=0.15)
    p.add_argument("--gate_roundtrip_good", type=float, default=0.45)
    p.add_argument("--gate_roundtrip_warn", type=float, default=0.65)
    p.add_argument("--gate_best_epoch_min", type=int, default=3)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    set_seed(int(args.seed))

    device = torch.device(args.device)
    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("m33 Prior Probe v2")
    print(f"Run: {save_root}")
    print("=" * 72)

    files = base._parse_h5_path_arg(str(args.train_path))
    max_needed = int(args.offset_jets + args.n_train_jets)
    all_const, all_labels, _all_w = base.load_raw_constituents_labels_weights_from_h5(
        files=files,
        max_jets=max_needed,
        max_constits=int(args.max_constits),
        use_train_weights=False,
    )
    if all_const.shape[0] < max_needed:
        raise RuntimeError(f"Requested {max_needed} jets but found {all_const.shape[0]}")

    const_raw = all_const[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    cfg = base._deepcopy_config()
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0
    jet_keys = (np.arange(len(const_off), dtype=np.int64) + int(args.offset_jets)).astype(np.int64)

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

    feat_off = compute_features(const_off, masks_off)
    means, stds = get_stats(feat_off, masks_off, train_idx)
    feat_off_std = standardize(feat_off, masks_off, means, stds)

    teacher_cfg = {
        "epochs": int(args.teacher_epochs),
        "patience": int(args.teacher_patience),
        "lr": float(args.teacher_lr),
        "weight_decay": float(args.teacher_weight_decay),
        "warmup_epochs": int(args.teacher_warmup_epochs),
    }
    ds_tr = base.WeightedJetDataset(feat_off_std[train_idx], masks_off[train_idx], labels[train_idx], np.ones((len(train_idx),), dtype=np.float32))
    ds_va = base.WeightedJetDataset(feat_off_std[val_idx], masks_off[val_idx], labels[val_idx], np.ones((len(val_idx),), dtype=np.float32))
    ds_te = base.WeightedJetDataset(feat_off_std[test_idx], masks_off[test_idx], labels[test_idx], np.ones((len(test_idx),), dtype=np.float32))
    dl_tr = DataLoader(ds_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_va = DataLoader(ds_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_te = DataLoader(ds_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = base.train_single_view_classifier_auc(
        model=teacher,
        train_loader=dl_tr,
        val_loader=dl_va,
        device=device,
        train_cfg=teacher_cfg,
        name="ProbeTeacherV2",
    )
    teacher_auc_test, teacher_p_test, teacher_y_test, _tw = base._eval_classifier_with_optional_weights(teacher, dl_te, device)
    print(f"ProbeTeacher test AUC={teacher_auc_test:.4f}")

    ds_prior_tr = m33.OfflineStageDataset(
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        labels=labels[train_idx],
        sample_weight=np.ones((len(train_idx),), dtype=np.float32),
    )
    ds_prior_va = m33.OfflineStageDataset(
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        labels=labels[val_idx],
        sample_weight=np.ones((len(val_idx),), dtype=np.float32),
    )
    dl_prior_tr = DataLoader(ds_prior_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_prior_va = DataLoader(ds_prior_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    prior = ConditionalOfflineCVAE(
        latent_dim=int(args.latent_dim),
        slots=int(args.max_constits),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)

    prior, prior_train_metrics = _train_cvae_prior(
        model=prior,
        train_loader=dl_prior_tr,
        val_loader=dl_prior_va,
        device=device,
        epochs=int(args.prior_epochs),
        lr=float(args.prior_lr),
        weight_decay=float(args.prior_weight_decay),
        patience=int(args.prior_patience),
        kl_beta_max=float(args.prior_kl_beta_max),
        kl_warmup_epochs=int(args.prior_kl_warmup_epochs),
        loss_w_count=float(args.prior_loss_w_count),
        loss_w_exist=float(args.prior_loss_w_exist),
        loss_w_pt=float(args.prior_loss_w_pt),
        loss_w_mass=float(args.prior_loss_w_mass),
        loss_w_cls=float(args.prior_loss_w_cls),
        loss_w_sep=float(args.prior_loss_w_sep),
        unmatched_penalty=float(args.unmatched_penalty),
    )

    z_mu_tr, y_tr = _collect_latent_means(prior, dl_prior_tr, device)
    gmms = _fit_class_gmms(
        z_mu=z_mu_tr,
        y=y_tr,
        n_components=int(args.gmm_components),
        reg_covar=float(args.gmm_reg_covar),
        seed=int(args.seed),
    )

    met_tr = _eval_prior_fidelity(
        model=prior,
        const=const_off[train_idx],
        mask=masks_off[train_idx],
        labels=labels[train_idx],
        batch_size=int(args.batch_size),
        device=device,
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
    )
    met_va = _eval_prior_fidelity(
        model=prior,
        const=const_off[val_idx],
        mask=masks_off[val_idx],
        labels=labels[val_idx],
        batch_size=int(args.batch_size),
        device=device,
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
    )
    met_te = _eval_prior_fidelity(
        model=prior,
        const=const_off[test_idx],
        mask=masks_off[test_idx],
        labels=labels[test_idx],
        batch_size=int(args.batch_size),
        device=device,
        pred_exist_threshold=float(args.pred_exist_threshold),
        unmatched_penalty=float(args.unmatched_penalty),
    )

    n_per_cls = int(max(64, args.n_prior_samples_per_class))
    gen_const, gen_mask, gen_cls = _sample_from_class_gmms(
        model=prior,
        gmms=gmms,
        n_per_cls=n_per_cls,
        pred_exist_threshold=float(args.pred_exist_threshold),
        device=device,
    )

    rng = np.random.RandomState(int(args.seed))
    real_pick = []
    for cls in [0, 1]:
        idx_cls = val_idx[labels[val_idx] == cls]
        if len(idx_cls) == 0:
            continue
        pick = rng.choice(idx_cls, size=min(len(idx_cls), n_per_cls), replace=False)
        real_pick.append(pick)
    real_pick_idx = np.concatenate(real_pick, axis=0) if real_pick else np.array([], dtype=np.int64)

    real_sum = _summary_vectors(const_off[real_pick_idx], masks_off[real_pick_idx])
    gen_sum = _summary_vectors(gen_const, gen_mask)
    realism_stats = {
        "ks_count": _ks_stat_1d(real_sum["count"], gen_sum["count"]),
        "ks_jet_pt": _ks_stat_1d(real_sum["jet_pt"], gen_sum["jet_pt"]),
        "ks_jet_mass": _ks_stat_1d(real_sum["jet_mass"], gen_sum["jet_mass"]),
        "ks_abs_eta_mean": _ks_stat_1d(real_sum["abs_eta_mean"], gen_sum["abs_eta_mean"]),
        "real_count_mean": float(real_sum["count"].mean()),
        "gen_count_mean": float(gen_sum["count"].mean()),
    }

    x_real = _critic_features(const_off[real_pick_idx], masks_off[real_pick_idx])
    x_gen = _critic_features(gen_const, gen_mask)
    y_real = np.ones((x_real.shape[0],), dtype=np.float32)
    y_gen = np.zeros((x_gen.shape[0],), dtype=np.float32)
    x_crit = np.concatenate([x_real, x_gen], axis=0)
    y_crit = np.concatenate([y_real, y_gen], axis=0)
    critic_metrics = _train_critic_auc(
        x=x_crit,
        y=y_crit,
        device=device,
        epochs=int(args.critic_epochs),
        lr=float(args.critic_lr),
        weight_decay=float(args.critic_weight_decay),
        batch_size=int(args.critic_batch_size),
        seed=int(args.seed),
    )

    feat_gen = compute_features(gen_const, gen_mask)
    feat_gen_std = standardize(feat_gen, gen_mask, means, stds)
    p_gen, y_gen_pred = m33._predict_probs(
        model=teacher,
        feat=feat_gen_std,
        mask=gen_mask,
        labels=gen_cls.astype(np.float32),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    top_mask = y_gen_pred.astype(np.int64) == 1
    bg_mask = y_gen_pred.astype(np.int64) == 0
    class_sanity = {
        "mean_prob_top_prior": float(np.mean(p_gen[top_mask])) if top_mask.any() else float("nan"),
        "mean_prob_bg_prior": float(np.mean(p_gen[bg_mask])) if bg_mask.any() else float("nan"),
        "separation_top_minus_bg": float(np.mean(p_gen[top_mask]) - np.mean(p_gen[bg_mask])) if top_mask.any() and bg_mask.any() else float("nan"),
    }

    rt_n = int(min(max(256, args.roundtrip_eval_count), len(val_idx)))
    rt_idx = rng.choice(val_idx, size=rt_n, replace=False)
    rt_const = const_off[rt_idx]
    rt_mask = masks_off[rt_idx]
    rt_keys = jet_keys[rt_idx]

    prior.eval()
    rec_const_l = []
    rec_mask_l = []
    with torch.no_grad():
        for s in range(0, rt_n, int(args.batch_size)):
            e = min(rt_n, s + int(args.batch_size))
            c = torch.tensor(rt_const[s:e], dtype=torch.float32, device=device)
            m = torch.tensor(rt_mask[s:e], dtype=torch.bool, device=device)
            y = torch.tensor(labels[rt_idx][s:e], dtype=torch.long, device=device)
            mu, _ = prior.encode_posterior(c, m, y)
            pc, pl = prior.decode(mu, y)
            pm = torch.sigmoid(pl) > float(args.pred_exist_threshold)
            rec_const_l.append(pc.detach().cpu().numpy().astype(np.float32))
            rec_mask_l.append(pm.detach().cpu().numpy().astype(bool))
    rec_const = np.concatenate(rec_const_l, axis=0)
    rec_mask = np.concatenate(rec_mask_l, axis=0)

    h_true, hm_true, _st_true = m33._apply_hlt_effects_deterministic_keyed(
        const=rt_const,
        mask=rt_mask,
        cfg=cfg,
        jet_keys=rt_keys,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    h_reco, hm_reco, _st_rec = m33._apply_hlt_effects_deterministic_keyed(
        const=rec_const,
        mask=rec_mask,
        cfg=cfg,
        jet_keys=rt_keys,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    resid = m33._residual_fast_vec(
        pred_const=torch.tensor(h_reco, dtype=torch.float32, device=device),
        pred_mask=torch.tensor(hm_reco, dtype=torch.bool, device=device),
        tgt_const=torch.tensor(h_true, dtype=torch.float32, device=device),
        tgt_mask=torch.tensor(hm_true, dtype=torch.bool, device=device),
        w_chamfer=float(args.roundtrip_w_chamfer),
        w_count=float(args.roundtrip_w_count),
        w_pt=float(args.roundtrip_w_pt),
        w_mass=float(args.roundtrip_w_mass),
        unmatched_penalty=float(args.unmatched_penalty),
    )
    rt_total = resid["total"].detach().cpu().numpy().astype(np.float64)
    rt_count = resid["count"].detach().cpu().numpy().astype(np.float64)
    roundtrip_metrics = {
        "n_eval": int(rt_n),
        "total_mean": float(rt_total.mean()),
        "total_p50": float(np.quantile(rt_total, 0.50)),
        "total_p90": float(np.quantile(rt_total, 0.90)),
        "count_mean": float(rt_count.mean()),
    }

    overfit_set_ratio = float(met_va.set_mean / max(1e-8, met_tr.set_mean))
    overfit_count_ratio = float(met_va.count_mean / max(1e-8, met_tr.count_mean))
    gate_best_epoch = "PASS" if int(prior_train_metrics.get("best_epoch", 0)) >= int(args.gate_best_epoch_min) else "WARN"
    gate_set_ratio = _gate(overfit_set_ratio, float(args.gate_overfit_ratio_good), float(args.gate_overfit_ratio_warn), False)
    gate_count_ratio = _gate(overfit_count_ratio, float(args.gate_overfit_ratio_good), float(args.gate_overfit_ratio_warn), False)
    gate_critic = _gate(float(critic_metrics["val_auc"]), float(args.gate_critic_auc_good), float(args.gate_critic_auc_warn), False)
    gate_class_sep = _gate(float(class_sanity["separation_top_minus_bg"]), float(args.gate_class_sep_good), float(args.gate_class_sep_warn), True)
    gate_roundtrip = _gate(float(roundtrip_metrics["total_mean"]), float(args.gate_roundtrip_good), float(args.gate_roundtrip_warn), False)

    gates = {
        "best_epoch_not_too_early": gate_best_epoch,
        "overfit_set_ratio": gate_set_ratio,
        "overfit_count_ratio": gate_count_ratio,
        "critic_real_vs_gen_auc": gate_critic,
        "class_sanity_separation": gate_class_sep,
        "roundtrip_total_mean": gate_roundtrip,
    }
    n_fail = sum(1 for v in gates.values() if v == "FAIL")
    n_warn = sum(1 for v in gates.values() if v == "WARN")
    overall = "PASS" if n_fail == 0 and n_warn <= 1 else ("WARN" if n_fail == 0 else "FAIL")

    report = {
        "model": "m33_prior_probe_v2",
        "seed": int(args.seed),
        "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "teacher": {"auc_test": float(teacher_auc_test)},
        "prior_train_metrics": prior_train_metrics,
        "fidelity": {
            "train": met_tr.__dict__,
            "val": met_va.__dict__,
            "test": met_te.__dict__,
            "overfit_set_ratio": overfit_set_ratio,
            "overfit_count_ratio": overfit_count_ratio,
        },
        "realism_stats": realism_stats,
        "critic": critic_metrics,
        "class_sanity": class_sanity,
        "roundtrip": roundtrip_metrics,
        "gates": gates,
        "overall_status": overall,
        "gmm_components_used": {"class0": int(gmms[0].n_components), "class1": int(gmms[1].n_components)},
    }

    with open(save_root / "m33_prior_probe_v2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        save_root / "m33_prior_probe_v2_arrays.npz",
        teacher_test_probs=np.asarray(teacher_p_test, dtype=np.float32),
        teacher_test_labels=np.asarray(teacher_y_test, dtype=np.float32),
        gen_probs=np.asarray(p_gen, dtype=np.float32),
        gen_labels=np.asarray(y_gen_pred, dtype=np.float32),
        roundtrip_total=np.asarray(rt_total, dtype=np.float32),
        roundtrip_count=np.asarray(rt_count, dtype=np.float32),
    )

    print("=" * 72)
    print("m33 Prior Probe v2 Summary")
    print("=" * 72)
    print(f"Overall status: {overall}")
    print(f"Teacher AUC(test): {teacher_auc_test:.4f}")
    print(f"PriorV2 val set/count: {met_va.set_mean:.4f} / {met_va.count_mean:.4f}")
    print(f"Overfit ratios set/count: {overfit_set_ratio:.3f} / {overfit_count_ratio:.3f}")
    print(f"Critic val AUC(real-vs-gen): {critic_metrics['val_auc']:.4f}")
    print(f"Class sanity separation(top-bg): {class_sanity['separation_top_minus_bg']:.4f}")
    print(f"Roundtrip total mean/p90: {roundtrip_metrics['total_mean']:.4f} / {roundtrip_metrics['total_p90']:.4f}")
    print("Gates:", gates)
    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()

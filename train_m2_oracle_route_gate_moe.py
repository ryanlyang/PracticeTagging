#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oracle-route gate MoE for m2:
  p_fused = g * p_joint + (1 - g) * p_hlt

Gate inputs:
  - score features (prob/logit/conf/entropy/disagreement)
  - reco diagnostics (action/split/budget/correction magnitude)
  - selected router signals
  - pooled HLT embedding + pooled joint embedding

Training:
  Phase A: train gate only (baseline/reco/dual frozen)
  Phase B: optional light reco unfreeze + gate training
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as m2
from unmerge_correct_hlt import (
    ParticleTransformer,
    DualViewCrossAttnClassifier,
    compute_features,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
)
from offline_reconstructor_no_gt_local30kv2 import (
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _clip_probs_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def _logit_t(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p / (1.0 - p))


def _entropy_t(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _fpr_at_target_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    idx = int(np.argmin(np.abs(tpr - float(target_tpr))))
    return float(fpr[idx])


def _threshold_at_target_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    if y.size == 0 or np.unique(y).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    idx = np.where(tpr >= float(target_tpr))[0]
    if idx.size == 0:
        return float(thr[-1])
    return float(thr[idx[0]])


def _deepcopy_cfg() -> Dict:
    return m2._deepcopy_config()


def _load_ckpt_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _infer_baseline_input_dim(sd: Dict[str, torch.Tensor]) -> int:
    if "input_proj.0.weight" in sd:
        return int(sd["input_proj.0.weight"].shape[1])
    if "input_proj.weight" in sd:
        return int(sd["input_proj.weight"].shape[1])
    raise RuntimeError("Could not infer baseline input dim")


def _infer_dual_input_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "input_proj_a.0.weight" in sd:
        da = int(sd["input_proj_a.0.weight"].shape[1])
    elif "input_proj_a.weight" in sd:
        da = int(sd["input_proj_a.weight"].shape[1])
    else:
        raise RuntimeError("Could not infer dual input_dim_a")
    if "input_proj_b.0.weight" in sd:
        db = int(sd["input_proj_b.0.weight"].shape[1])
    elif "input_proj_b.weight" in sd:
        db = int(sd["input_proj_b.weight"].shape[1])
    else:
        raise RuntimeError("Could not infer dual input_dim_b")
    return da, db


def _build_train_files(data_setup: Dict, train_path_arg: str) -> List[Path]:
    tf = [Path(p) for p in data_setup.get("train_files", []) if str(p).strip()]
    if tf and all(p.exists() for p in tf):
        return tf
    train_path = Path(train_path_arg)
    if train_path.is_dir():
        fs = sorted(train_path.glob("*.h5"))
    else:
        fs = [Path(p) for p in str(train_path_arg).split(",") if str(p).strip()]
    if len(fs) == 0:
        raise FileNotFoundError(f"No .h5 files found in --train_path={train_path_arg}")
    return fs


def _offline_mask(const_raw: np.ndarray, pt_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    raw_mask = const_raw[:, :, 0] > 0.0
    mask_off = raw_mask & (const_raw[:, :, 0] >= float(pt_thr))
    const_off = const_raw.copy()
    const_off[~mask_off] = 0.0
    return const_off.astype(np.float32), mask_off.astype(bool)


def _wrap_phi_np(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def _merge_two_tokens(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pa, ea, fa, Ea = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    pb, eb, fb, Eb = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    pxa = pa * math.cos(fa)
    pya = pa * math.sin(fa)
    pza = pa * math.sinh(ea)
    pxb = pb * math.cos(fb)
    pyb = pb * math.sin(fb)
    pzb = pb * math.sinh(eb)
    px = pxa + pxb
    py = pya + pyb
    pz = pza + pzb
    E = max(Ea + Eb, 1e-8)
    pt = max(math.sqrt(px * px + py * py), 1e-8)
    phi = math.atan2(py, px)
    eta = float(np.arcsinh(pz / max(pt, 1e-8)))
    eta = float(np.clip(eta, -5.0, 5.0))
    Emin = pt * math.cosh(eta)
    E = max(E, Emin)
    return np.array([pt, eta, phi, E], dtype=np.float32)


def _apply_corruption_batch(
    const_in: np.ndarray,
    mask_in: np.ndarray,
    kind: str,
    severity: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    const = const_in.copy()
    mask = mask_in.copy()
    B, L, _ = const.shape

    if kind == "pt_noise":
        z = rng.normal(0.0, float(severity), size=(B, L)).astype(np.float32)
        f = np.exp(z).astype(np.float32)
        const[:, :, 0] = np.where(mask, const[:, :, 0] * f, const[:, :, 0])
        const[:, :, 3] = np.where(mask, const[:, :, 3] * f, const[:, :, 3])
    elif kind == "eta_phi_jitter":
        de = rng.normal(0.0, float(severity), size=(B, L)).astype(np.float32)
        dp = rng.normal(0.0, float(severity), size=(B, L)).astype(np.float32)
        const[:, :, 1] = np.where(mask, np.clip(const[:, :, 1] + de, -5.0, 5.0), const[:, :, 1])
        const[:, :, 2] = np.where(mask, _wrap_phi_np(const[:, :, 2] + dp), const[:, :, 2])
    elif kind == "dropout":
        keep = mask.copy()
        drop = (rng.random(size=mask.shape) < float(severity)) & mask
        keep[drop] = False
        empty = ~keep.any(axis=1)
        if np.any(empty):
            for i in np.where(empty)[0]:
                idx = np.where(mask[i])[0]
                if idx.size > 0:
                    j = int(idx[np.argmax(const[i, idx, 0])])
                    keep[i, j] = True
        mask = keep
        const[~mask] = 0.0
    elif kind == "merge":
        frac = float(np.clip(severity, 0.0, 0.95))
        for i in range(B):
            idx = np.where(mask[i])[0]
            n = int(idx.size)
            if n < 2:
                continue
            n_pairs = max(1, int(round(0.5 * frac * n)))
            n_pairs = min(n_pairs, n // 2)
            perm = rng.permutation(idx)
            for k in range(n_pairs):
                a = int(perm[2 * k])
                b = int(perm[2 * k + 1])
                const[i, a] = _merge_two_tokens(const[i, a], const[i, b])
                const[i, b] = 0.0
                mask[i, b] = False
    elif kind == "global_scale":
        z = rng.normal(0.0, float(severity), size=(B, 1)).astype(np.float32)
        f = np.exp(z).astype(np.float32)
        const[:, :, 0] = np.where(mask, const[:, :, 0] * f, const[:, :, 0])
        const[:, :, 3] = np.where(mask, const[:, :, 3] * f, const[:, :, 3])
    else:
        raise ValueError(f"Unknown corruption kind: {kind}")

    const[:, :, 0] = np.where(mask, np.clip(const[:, :, 0], 1e-8, 1e8), 0.0)
    const[:, :, 1] = np.where(mask, np.clip(const[:, :, 1], -5.0, 5.0), 0.0)
    const[:, :, 2] = np.where(mask, _wrap_phi_np(const[:, :, 2]), 0.0)
    e_floor = const[:, :, 0] * np.cosh(const[:, :, 1])
    const[:, :, 3] = np.where(mask, np.maximum(const[:, :, 3], e_floor), 0.0)
    return const.astype(np.float32), mask.astype(bool)


def _parse_corruptions(spec: str) -> List[Tuple[str, float]]:
    out = []
    for tok in [x.strip() for x in str(spec).split(",") if x.strip()]:
        if ":" not in tok:
            raise ValueError(f"Invalid corruption token: {tok}")
        k, v = tok.split(":", 1)
        out.append((k.strip(), float(v)))
    if len(out) == 0:
        raise ValueError("No valid corruptions parsed")
    return out


class RouterDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray, const_hlt: np.ndarray, labels: np.ndarray):
        self.feat_hlt = feat_hlt.astype(np.float32)
        self.mask_hlt = mask_hlt.astype(bool)
        self.const_hlt = const_hlt.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": torch.from_numpy(self.feat_hlt[i]),
            "mask_hlt": torch.from_numpy(self.mask_hlt[i]),
            "const_hlt": torch.from_numpy(self.const_hlt[i]),
            "label": torch.tensor(self.labels[i], dtype=torch.float32),
        }


class GateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def dual_forward_with_embedding(
    dual: DualViewCrossAttnClassifier,
    feat_a: torch.Tensor,
    mask_a: torch.Tensor,
    feat_b: torch.Tensor,
    mask_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, _ = feat_a.shape
    mask_a_safe = mask_a.clone()
    mask_b_safe = mask_b.clone()
    empty_a = ~mask_a_safe.any(dim=1)
    empty_b = ~mask_b_safe.any(dim=1)
    if empty_a.any():
        mask_a_safe[empty_a, 0] = True
    if empty_b.any():
        mask_b_safe[empty_b, 0] = True

    h_a = dual.input_proj_a(feat_a.view(-1, feat_a.size(-1))).view(bsz, seq_len, -1)
    h_b = dual.input_proj_b(feat_b.view(-1, feat_b.size(-1))).view(bsz, seq_len, -1)
    h_a = dual.encoder_a(h_a, src_key_padding_mask=~mask_a_safe)
    h_b = dual.encoder_b(h_b, src_key_padding_mask=~mask_b_safe)
    query = dual.pool_query.expand(bsz, -1, -1)
    pooled_a, _ = dual.pool_attn_a(query, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)
    pooled_b, _ = dual.pool_attn_b(query, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
    cross_a, _ = dual.cross_a_to_b(pooled_a, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
    cross_b, _ = dual.cross_b_to_a(pooled_b, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)
    fused = torch.cat([pooled_a, pooled_b, cross_a, cross_b], dim=-1).squeeze(1)
    fused = dual.norm(fused)
    logits = dual.classifier(fused).squeeze(1)
    return logits, fused


def extract_reco_diag(
    reco_out: Dict[str, torch.Tensor],
    mask_hlt: torch.Tensor,
    const_hlt: torch.Tensor,
    reconstructor: OfflineReconstructor,
) -> Dict[str, torch.Tensor]:
    tok_mask = mask_hlt.float()
    token_den = tok_mask.sum(dim=1).clamp(min=1.0)
    ap = reco_out["action_prob"].clamp(min=1e-8)
    ent = -(ap * torch.log(ap)).sum(dim=-1)
    peak = ap.max(dim=-1).values
    K = max(int(reconstructor.max_split_children), 1)
    B, L = mask_hlt.shape
    child = reco_out["child_weight"].view(B, L, K)
    split_total = child.sum(dim=-1)
    bm = reco_out["budget_merge"]
    be = reco_out["budget_eff"]
    child_sum = reco_out["child_weight"].sum(dim=1).clamp(min=1e-6)
    gen_sum = reco_out["gen_weight"].sum(dim=1).clamp(min=1e-6)
    exp_added = reco_out["child_weight"].sum(dim=1) + reco_out["gen_weight"].sum(dim=1)
    add_unc = (
        (reco_out["child_weight"] * (1.0 - reco_out["child_weight"])).sum(dim=1)
        + (reco_out["gen_weight"] * (1.0 - reco_out["gen_weight"])).sum(dim=1)
    )

    tok_tokens = reco_out["cand_tokens"][:, :L, :]
    pt_h = const_hlt[..., 0].clamp(min=1e-8)
    eta_h = const_hlt[..., 1]
    phi_h = const_hlt[..., 2]
    pt_r = tok_tokens[..., 0].clamp(min=1e-8)
    eta_r = tok_tokens[..., 1]
    phi_r = tok_tokens[..., 2]
    d_logpt = torch.abs(torch.log(pt_r) - torch.log(pt_h))
    d_eta = torch.abs(eta_r - eta_h)
    d_phi = torch.abs(torch.atan2(torch.sin(phi_r - phi_h), torch.cos(phi_r - phi_h)))
    corr_mag_tok = d_logpt + 0.5 * d_eta + 0.5 * d_phi
    corr_mag_mean = (corr_mag_tok * tok_mask).sum(dim=1) / token_den

    return {
        "action_entropy_mean": (ent * tok_mask).sum(dim=1) / token_den,
        "action_peak_mean": (peak * tok_mask).sum(dim=1) / token_den,
        "split_total_mean": (split_total * tok_mask).sum(dim=1) / token_den,
        "split_total_max": split_total.masked_fill(~mask_hlt, 0.0).max(dim=1).values,
        "budget_merge_pressure": bm / child_sum,
        "budget_eff_pressure": be / gen_sum,
        "correction_magnitude_mean": corr_mag_mean,
        "expected_added_count": exp_added,
        "added_count_uncertainty": add_unc,
    }


@dataclass
class BatchForward:
    p_h: torch.Tensor
    p_j: torch.Tensor
    z_h: torch.Tensor
    z_j: torch.Tensor
    diag: Dict[str, torch.Tensor]
    gate_feats: torch.Tensor


def build_gate_batch_features(
    *,
    feat_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_hlt: torch.Tensor,
    baseline: ParticleTransformer,
    reconstructor: OfflineReconstructor,
    dual: DualViewCrossAttnClassifier,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    thr_h50: float,
    thr_j50: float,
    conf_ref_h: torch.Tensor,
    conf_ref_j: torch.Tensor,
    means: np.ndarray,
    stds: np.ndarray,
    corruption_list: List[Tuple[str, float]],
    rng: np.random.Generator,
    grad_through_reco: bool,
    corruption_feature_no_grad: bool = True,
) -> BatchForward:
    # baseline
    with torch.no_grad():
        logit_h, z_h = baseline(feat_hlt, mask_hlt, return_embedding=True)
    logit_h = logit_h.squeeze(1)

    # reco + dual
    if grad_through_reco:
        reco_out = reconstructor(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = m2.build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=bool(corrected_use_flags),
        )
        logit_j, z_j = dual_forward_with_embedding(dual, feat_hlt, mask_hlt, feat_b, mask_b)
    else:
        with torch.no_grad():
            reco_out = reconstructor(feat_hlt, mask_hlt, const_hlt, stage_scale=1.0)
            feat_b, mask_b = m2.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=bool(corrected_use_flags),
            )
            logit_j, z_j = dual_forward_with_embedding(dual, feat_hlt, mask_hlt, feat_b, mask_b)

    p_h = torch.sigmoid(logit_h)
    p_j = torch.sigmoid(logit_j)
    conf_h = torch.abs(p_h - 0.5) * 2.0
    conf_j = torch.abs(p_j - 0.5) * 2.0
    ent_h = _entropy_t(p_h)
    ent_j = _entropy_t(p_j)

    diag = extract_reco_diag(reco_out, mask_hlt, const_hlt, reconstructor)

    # selected core router signals
    entropy_gap = ent_j - ent_h
    conf_gap = conf_j - conf_h
    threshold_dist_gap = torch.abs(p_j - float(thr_j50)) - torch.abs(p_h - float(thr_h50))

    # conformal p-value proxy gap via confidence nonconformity
    nc_h = torch.minimum(p_h, 1.0 - p_h).detach()
    nc_j = torch.minimum(p_j, 1.0 - p_j).detach()
    rank_h = torch.searchsorted(conf_ref_h, nc_h, right=True).float() / float(max(conf_ref_h.numel(), 1))
    rank_j = torch.searchsorted(conf_ref_j, nc_j, right=True).float() / float(max(conf_ref_j.numel(), 1))
    pval_gap = (1.0 - rank_j) - (1.0 - rank_h)

    # Corruption-based stability features.
    with (torch.no_grad() if corruption_feature_no_grad else torch.enable_grad()):
        const_np = const_hlt.detach().cpu().numpy().astype(np.float32)
        mask_np = mask_hlt.detach().cpu().numpy().astype(bool)
        wins_prob = []
        wins_conf = []
        wins_deg = []
        cross_gap_fpr50 = []
        for kind, severity in corruption_list:
            c_np, m_np = _apply_corruption_batch(const_np, mask_np, kind, severity, rng)
            f_np = compute_features(c_np, m_np)
            f_np_std = standardize(f_np, m_np, means, stds).astype(np.float32)
            f_t = torch.from_numpy(f_np_std).to(feat_hlt.device, dtype=torch.float32)
            m_t = torch.from_numpy(m_np).to(feat_hlt.device, dtype=torch.bool)
            c_t = torch.from_numpy(c_np).to(feat_hlt.device, dtype=torch.float32)

            with torch.no_grad():
                logit_hc, _ = baseline(f_t, m_t, return_embedding=True)
                reco_c = reconstructor(f_t, m_t, c_t, stage_scale=1.0)
                fb_c, mb_c = m2.build_soft_corrected_view(
                    reco_c,
                    weight_floor=float(corrected_weight_floor),
                    scale_features_by_weight=True,
                    include_flags=bool(corrected_use_flags),
                )
                logit_jc, _ = dual_forward_with_embedding(dual, f_t, m_t, fb_c, mb_c)
                phc = torch.sigmoid(logit_hc.squeeze(1))
                pjc = torch.sigmoid(logit_jc)

            dprob_h = torch.abs(phc - p_h.detach())
            dprob_j = torch.abs(pjc - p_j.detach())
            conf_hc = torch.abs(phc - 0.5) * 2.0
            conf_jc = torch.abs(pjc - 0.5) * 2.0
            dconf_h = torch.abs(conf_hc - conf_h.detach())
            dconf_j = torch.abs(conf_jc - conf_j.detach())
            drop_h = torch.clamp(p_h.detach() - phc, min=0.0)
            drop_j = torch.clamp(p_j.detach() - pjc, min=0.0)
            cross_h50 = ((p_h.detach() - float(thr_h50)) * (phc - float(thr_h50)) < 0.0).float()
            cross_j50 = ((p_j.detach() - float(thr_j50)) * (pjc - float(thr_j50)) < 0.0).float()

            wins_prob.append((dprob_j < dprob_h).float())
            wins_conf.append((dconf_j < dconf_h).float())
            wins_deg.append((drop_j < drop_h).float())
            cross_gap_fpr50.append(cross_j50 - cross_h50)

        if len(wins_prob) > 0:
            joint_stability_prob_win_frac = torch.stack(wins_prob, dim=0).mean(dim=0)
            joint_stability_conf_win_frac = torch.stack(wins_conf, dim=0).mean(dim=0)
            degrade_win_fraction = torch.stack(wins_deg, dim=0).mean(dim=0)
            threshold_cross_flip_rate_gap_fpr50 = torch.stack(cross_gap_fpr50, dim=0).mean(dim=0)
        else:
            z = torch.zeros_like(p_h)
            joint_stability_prob_win_frac = z
            joint_stability_conf_win_frac = z
            degrade_win_fraction = z
            threshold_cross_flip_rate_gap_fpr50 = z

    basic = [
        p_h,
        p_j,
        _logit_t(p_h),
        _logit_t(p_j),
        conf_h,
        conf_j,
        ent_h,
        ent_j,
        p_j - p_h,
        torch.abs(p_j - p_h),
        entropy_gap,
        conf_gap,
        threshold_dist_gap,
        pval_gap,
        joint_stability_conf_win_frac,
        joint_stability_prob_win_frac,
        degrade_win_fraction,
        threshold_cross_flip_rate_gap_fpr50,
        diag["action_entropy_mean"],
        diag["split_total_mean"],
        diag["split_total_max"],
        diag["budget_merge_pressure"],
        diag["budget_eff_pressure"],
        diag["correction_magnitude_mean"],
    ]
    scalars = torch.stack(basic, dim=1)
    gate_feats = torch.cat([scalars, z_h.detach(), z_j], dim=1)
    return BatchForward(
        p_h=p_h,
        p_j=p_j,
        z_h=z_h.detach(),
        z_j=z_j,
        diag=diag,
        gate_feats=gate_feats,
    )


def lowfpr_cost_t(
    y: torch.Tensor,
    p: torch.Tensor,
    thr30: float,
    thr50: float,
    alpha_neg: float,
    tau: float,
) -> torch.Tensor:
    eps = 1e-6
    p = p.clamp(min=eps, max=1.0 - eps)
    bce = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    tail = torch.sigmoid((p - float(thr50)) / float(tau)) + 0.5 * torch.sigmoid((p - float(thr30)) / float(tau))
    return bce + float(alpha_neg) * (1.0 - y) * tail


def metrics_from_preds(y: np.ndarray, p: np.ndarray, p_hlt_ref: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if y.size == 0 or np.unique(y).size < 2:
        out["auc"] = float("nan")
        out["fpr30"] = float("nan")
        out["fpr50"] = float("nan")
    else:
        out["auc"] = float(roc_auc_score(y, p))
        out["fpr30"] = _fpr_at_target_tpr(y, p, 0.30)
        out["fpr50"] = _fpr_at_target_tpr(y, p, 0.50)
    eps = 1e-6
    q = np.clip(p, eps, 1.0 - eps)
    h = np.clip(p_hlt_ref, eps, 1.0 - eps)
    bce_q = -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))
    bce_h = -(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))
    diff = bce_q - bce_h
    out["harm_bce_frac_all_vs_hlt"] = float(np.mean(diff > 0.0))
    neg = y < 0.5
    out["harm_bce_frac_neg_vs_hlt"] = float(np.mean(diff[neg] > 0.0)) if np.any(neg) else float("nan")
    return out


def unfreeze_reco_tail(reco: OfflineReconstructor, last_n_layers: int) -> List[nn.Parameter]:
    for p in reco.parameters():
        p.requires_grad = False
    n_layers = len(reco.encoder_layers)
    from_idx = max(0, n_layers - int(last_n_layers))
    for i in range(from_idx, n_layers):
        for p in reco.encoder_layers[i].parameters():
            p.requires_grad = True
    # keep heads trainable for adaptation.
    for name in [
        "token_norm",
        "action_head",
        "unsmear_head",
        "reassign_head",
        "split_exist_head",
        "split_delta_head",
        "budget_head",
        "pool_attn",
        "gen_attn",
        "gen_norm",
        "gen_head",
        "gen_exist_head",
    ]:
        mod = getattr(reco, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
    # pool query / gen queries params
    if hasattr(reco, "pool_query"):
        reco.pool_query.requires_grad = True
    if hasattr(reco, "gen_queries"):
        reco.gen_queries.requires_grad = True
    return [p for p in reco.parameters() if p.requires_grad]


def evaluate_split(
    loader: DataLoader,
    *,
    baseline: ParticleTransformer,
    reconstructor: OfflineReconstructor,
    dual: DualViewCrossAttnClassifier,
    gate: GateMLP,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    thr_h30: float,
    thr_h50: float,
    thr_j30: float,
    thr_j50: float,
    conf_ref_h: torch.Tensor,
    conf_ref_j: torch.Tensor,
    means: np.ndarray,
    stds: np.ndarray,
    corruption_list: List[Tuple[str, float]],
    rng: np.random.Generator,
    device: torch.device,
) -> Dict[str, object]:
    gate.eval()
    baseline.eval()
    reconstructor.eval()
    dual.eval()
    ys, phs, pjs, pfs, gs = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["feat_hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            c = batch["const_hlt"].to(device)
            y = batch["label"].to(device)
            bf = build_gate_batch_features(
                feat_hlt=x,
                mask_hlt=m,
                const_hlt=c,
                baseline=baseline,
                reconstructor=reconstructor,
                dual=dual,
                corrected_weight_floor=corrected_weight_floor,
                corrected_use_flags=corrected_use_flags,
                thr_h50=thr_h50,
                thr_j50=thr_j50,
                conf_ref_h=conf_ref_h,
                conf_ref_j=conf_ref_j,
                means=means,
                stds=stds,
                corruption_list=corruption_list,
                rng=rng,
                grad_through_reco=False,
                corruption_feature_no_grad=True,
            )
            g = torch.sigmoid(gate(bf.gate_feats))
            p_f = g * bf.p_j + (1.0 - g) * bf.p_h
            ys.append(y.detach().cpu().numpy())
            phs.append(bf.p_h.detach().cpu().numpy())
            pjs.append(bf.p_j.detach().cpu().numpy())
            pfs.append(p_f.detach().cpu().numpy())
            gs.append(g.detach().cpu().numpy())

    y = np.concatenate(ys).astype(np.float32)
    ph = _clip_probs_np(np.concatenate(phs).astype(np.float32))
    pj = _clip_probs_np(np.concatenate(pjs).astype(np.float32))
    pf = _clip_probs_np(np.concatenate(pfs).astype(np.float32))
    g = np.concatenate(gs).astype(np.float32)
    return {
        "y": y,
        "p_hlt": ph,
        "p_joint": pj,
        "p_fused": pf,
        "g": g,
        "metrics_fused": metrics_from_preds(y, pf, ph),
        "metrics_hlt": metrics_from_preds(y, ph, ph),
        "metrics_joint": metrics_from_preds(y, pj, ph),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="m2 oracle-route gate MoE")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--n_train_jets", type=int, default=375000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--n_train_split", type=int, default=150000)
    ap.add_argument("--n_val_split", type=int, default=75000)
    ap.add_argument("--n_test_split", type=int, default=150000)
    ap.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_oracle_route_gate_moe")
    ap.add_argument("--run_name", type=str, default="model2_oraclegate_moe_150k75k150k_seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hlt_seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])
    ap.add_argument("--phaseA_epochs", type=int, default=8)
    ap.add_argument("--phaseB_epochs", type=int, default=8)
    ap.add_argument("--phaseA_lr_gate", type=float, default=1e-3)
    ap.add_argument("--phaseB_lr_gate", type=float, default=5e-4)
    ap.add_argument("--phaseB_lr_reco", type=float, default=5e-6)
    ap.add_argument("--phaseB_unfreeze_last_n_encoder_layers", type=int, default=2)
    ap.add_argument("--lambda_cls", type=float, default=1.0)
    ap.add_argument("--lambda_route", type=float, default=0.7)
    ap.add_argument("--lambda_reco_anchor", type=float, default=2e-4)
    ap.add_argument("--lambda_gate_balance", type=float, default=0.0)
    ap.add_argument("--gate_target_usage", type=float, default=-1.0)
    ap.add_argument("--gate_hidden", type=int, default=256)
    ap.add_argument("--gate_dropout", type=float, default=0.10)
    ap.add_argument("--cost_alpha_neg", type=float, default=4.0)
    ap.add_argument("--cost_tau", type=float, default=0.02)
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    ap.add_argument(
        "--corruptions",
        type=str,
        default="pt_noise:0.04,eta_phi_jitter:0.03,dropout:0.07,merge:0.12,global_scale:0.04",
    )
    ap.add_argument("--save_fusion_scores", action="store_true")
    args = ap.parse_args()

    set_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    save_root = Path(args.save_dir).expanduser().resolve() / str(args.run_name)
    save_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    cfg = _deepcopy_cfg()
    hlt_stats_path = run_dir / "hlt_stats.json"
    if hlt_stats_path.exists():
        hlt_obj = json.loads(hlt_stats_path.read_text())
        hcfg = hlt_obj.get("config", {})
        for k in list(cfg.get("hlt_effects", {}).keys()):
            if k in hcfg:
                cfg["hlt_effects"][k] = hcfg[k]

    data_setup_path = run_dir / "data_setup.json"
    data_setup = json.loads(data_setup_path.read_text()) if data_setup_path.exists() else {}
    train_files = _build_train_files(data_setup, args.train_path)

    max_need = int(args.offset_jets) + int(args.n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files, max_jets=max_need, max_constits=int(args.max_constits)
    )
    if all_const_full.shape[0] < max_need:
        raise RuntimeError(f"Not enough jets loaded: {all_const_full.shape[0]} < {max_need}")

    const_raw = all_const_full[int(args.offset_jets): int(args.offset_jets) + int(args.n_train_jets)]
    labels = all_labels_full[int(args.offset_jets): int(args.offset_jets) + int(args.n_train_jets)].astype(np.int64)
    const_off, mask_off = _offline_mask(const_raw, float(cfg["hlt_effects"]["pt_threshold_offline"]))
    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, _, _ = apply_hlt_effects_realistic_nomap(
        const_off, mask_off, cfg, seed=int(args.hlt_seed)
    )
    print("Computing HLT features...")
    feat_hlt = compute_features(hlt_const, hlt_mask)

    # Use exact historical split when present.
    split_npz = run_dir / "data_splits.npz"
    if split_npz.exists():
        d = np.load(split_npz)
        train_idx = d["train_idx"].astype(np.int64)
        val_idx = d["val_idx"].astype(np.int64)
        test_idx = d["test_idx"].astype(np.int64)
        if np.max(train_idx) >= len(labels) or np.max(val_idx) >= len(labels) or np.max(test_idx) >= len(labels):
            raise RuntimeError("data_splits.npz indices exceed current loaded jets")
        if "means" in d and "stds" in d:
            means = d["means"].astype(np.float32)
            stds = d["stds"].astype(np.float32)
        else:
            means, stds = get_stats(feat_hlt, hlt_mask, train_idx)
        print(f"Using split from {split_npz} -> train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    else:
        idx = np.arange(len(labels))
        total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
        if total_need > len(idx):
            raise RuntimeError(f"Requested split counts exceed available jets: {total_need} > {len(idx)}")
        if total_need < len(idx):
            idx_use, _ = train_test_split(
                idx, train_size=total_need, random_state=int(args.seed), stratify=labels[idx]
            )
        else:
            idx_use = idx
        train_idx, rem_idx = train_test_split(
            idx_use, train_size=int(args.n_train_split), random_state=int(args.seed), stratify=labels[idx_use]
        )
        val_idx, test_idx = train_test_split(
            rem_idx,
            train_size=int(args.n_val_split),
            test_size=int(args.n_test_split),
            random_state=int(args.seed),
            stratify=labels[rem_idx],
        )
        means, stds = get_stats(feat_hlt, hlt_mask, train_idx)
        print(f"Custom split train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds).astype(np.float32)

    train_ds = RouterDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], hlt_const[train_idx], labels[train_idx])
    val_ds = RouterDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], hlt_const[val_idx], labels[val_idx])
    test_ds = RouterDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], hlt_const[test_idx], labels[test_idx])
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers), pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)

    # Load pretrained models from run_dir.
    baseline_sd = _load_ckpt_state(run_dir / "baseline.pt", device)
    reco_sd = _load_ckpt_state(run_dir / "offline_reconstructor.pt", device)
    dual_sd = _load_ckpt_state(run_dir / "dual_joint.pt", device)
    baseline_dim = _infer_baseline_input_dim(baseline_sd)
    dual_in_a, dual_in_b = _infer_dual_input_dims(dual_sd)
    corrected_use_flags = bool(int(dual_in_b) == 12)
    if int(baseline_dim) != 7 or int(dual_in_a) != 7:
        raise RuntimeError(f"This ablation currently expects input_dim_a=7. got baseline={baseline_dim}, dual_a={dual_in_a}")

    baseline = ParticleTransformer(input_dim=baseline_dim, **cfg["model"]).to(device)
    baseline.load_state_dict(baseline_sd, strict=True)
    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(reco_sd, strict=True)
    dual = DualViewCrossAttnClassifier(input_dim_a=dual_in_a, input_dim_b=dual_in_b, **cfg["model"]).to(device)
    dual.load_state_dict(dual_sd, strict=True)

    for p in baseline.parameters():
        p.requires_grad = False
    for p in dual.parameters():
        p.requires_grad = False
    baseline.eval()
    dual.eval()

    # Precompute thresholds and conformal references on train split with frozen models.
    print("Precomputing train thresholds/conformal refs...")
    ph_train, pj_train, y_train = [], [], []
    infer_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="precompute"):
            x = batch["feat_hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            c = batch["const_hlt"].to(device)
            y = batch["label"].to(device)
            lh, _ = baseline(x, m, return_embedding=True)
            out = reconstructor(x, m, c, stage_scale=1.0)
            fb, mb = m2.build_soft_corrected_view(
                out,
                weight_floor=float(args.corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=bool(corrected_use_flags),
            )
            lj, _ = dual_forward_with_embedding(dual, x, m, fb, mb)
            ph_train.append(torch.sigmoid(lh.squeeze(1)).cpu().numpy())
            pj_train.append(torch.sigmoid(lj).cpu().numpy())
            y_train.append(y.cpu().numpy())
    ph_train = _clip_probs_np(np.concatenate(ph_train).astype(np.float32))
    pj_train = _clip_probs_np(np.concatenate(pj_train).astype(np.float32))
    y_train = np.concatenate(y_train).astype(np.float32)
    thr_h30 = _threshold_at_target_tpr(y_train, ph_train, 0.30)
    thr_h50 = _threshold_at_target_tpr(y_train, ph_train, 0.50)
    thr_j30 = _threshold_at_target_tpr(y_train, pj_train, 0.30)
    thr_j50 = _threshold_at_target_tpr(y_train, pj_train, 0.50)
    conf_ref_h = torch.from_numpy(np.sort(np.minimum(ph_train, 1.0 - ph_train))).to(device=device, dtype=torch.float32)
    conf_ref_j = torch.from_numpy(np.sort(np.minimum(pj_train, 1.0 - pj_train))).to(device=device, dtype=torch.float32)
    print(f"Thresholds @TPR30/50 HLT: {thr_h30:.6f}/{thr_h50:.6f} | Joint: {thr_j30:.6f}/{thr_j50:.6f}")

    corruption_list = _parse_corruptions(args.corruptions)
    # determine gate input dim from one batch
    b0 = next(iter(val_loader))
    with torch.no_grad():
        bf0 = build_gate_batch_features(
            feat_hlt=b0["feat_hlt"].to(device),
            mask_hlt=b0["mask_hlt"].to(device),
            const_hlt=b0["const_hlt"].to(device),
            baseline=baseline,
            reconstructor=reconstructor,
            dual=dual,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=corrected_use_flags,
            thr_h50=thr_h50,
            thr_j50=thr_j50,
            conf_ref_h=conf_ref_h,
            conf_ref_j=conf_ref_j,
            means=means,
            stds=stds,
            corruption_list=corruption_list,
            rng=rng,
            grad_through_reco=False,
            corruption_feature_no_grad=True,
        )
    gate_in_dim = int(bf0.gate_feats.shape[1])
    print(f"Gate input dim: {gate_in_dim}")
    gate = GateMLP(in_dim=gate_in_dim, hidden=int(args.gate_hidden), dropout=float(args.gate_dropout)).to(device)

    best_sel = float("-inf") if str(args.selection_metric) == "auc" else float("inf")
    best_state = None
    history: List[Dict[str, float]] = []

    # keep initial reco params for anchor in phase B
    reco_init = {n: p.detach().clone() for n, p in reconstructor.named_parameters()}

    def run_phase(phase_name: str, epochs: int, phase_b: bool) -> None:
        nonlocal best_sel, best_state
        if epochs <= 0:
            return
        if phase_b:
            reco_params = unfreeze_reco_tail(reconstructor, int(args.phaseB_unfreeze_last_n_encoder_layers))
            optim = torch.optim.AdamW(
                [
                    {"params": gate.parameters(), "lr": float(args.phaseB_lr_gate), "weight_decay": 1e-4},
                    {"params": reco_params, "lr": float(args.phaseB_lr_reco), "weight_decay": 1e-6},
                ]
            )
        else:
            for p in reconstructor.parameters():
                p.requires_grad = False
            optim = torch.optim.AdamW(gate.parameters(), lr=float(args.phaseA_lr_gate), weight_decay=1e-4)

        for ep in range(1, int(epochs) + 1):
            gate.train()
            reconstructor.train(phase_b)
            tr_loss = 0.0
            tr_n = 0
            for batch in tqdm(train_loader, desc=f"{phase_name} ep{ep}"):
                x = batch["feat_hlt"].to(device)
                m = batch["mask_hlt"].to(device)
                c = batch["const_hlt"].to(device)
                y = batch["label"].to(device)
                bf = build_gate_batch_features(
                    feat_hlt=x,
                    mask_hlt=m,
                    const_hlt=c,
                    baseline=baseline,
                    reconstructor=reconstructor,
                    dual=dual,
                    corrected_weight_floor=float(args.corrected_weight_floor),
                    corrected_use_flags=corrected_use_flags,
                    thr_h50=thr_h50,
                    thr_j50=thr_j50,
                    conf_ref_h=conf_ref_h,
                    conf_ref_j=conf_ref_j,
                    means=means,
                    stds=stds,
                    corruption_list=corruption_list,
                    rng=rng,
                    grad_through_reco=phase_b,
                    corruption_feature_no_grad=True,
                )
                # oracle route label
                with torch.no_grad():
                    c_h = lowfpr_cost_t(y, bf.p_h.detach(), thr_h30, thr_h50, float(args.cost_alpha_neg), float(args.cost_tau))
                    c_j = lowfpr_cost_t(y, bf.p_j.detach(), thr_j30, thr_j50, float(args.cost_alpha_neg), float(args.cost_tau))
                    z_oracle = (c_j < c_h).float()

                g_logit = gate(bf.gate_feats)
                g = torch.sigmoid(g_logit)
                p_f = g * bf.p_j + (1.0 - g) * bf.p_h
                loss_cls = F.binary_cross_entropy(p_f.clamp(1e-6, 1.0 - 1e-6), y)
                loss_route = F.binary_cross_entropy(g.clamp(1e-6, 1.0 - 1e-6), z_oracle)
                loss = float(args.lambda_cls) * loss_cls + float(args.lambda_route) * loss_route

                if float(args.lambda_gate_balance) > 0.0 and float(args.gate_target_usage) >= 0.0:
                    bal = (g.mean() - float(args.gate_target_usage)) ** 2
                    loss = loss + float(args.lambda_gate_balance) * bal

                if phase_b and float(args.lambda_reco_anchor) > 0.0:
                    anc = torch.zeros((), device=device)
                    n_anc = 0
                    for n, p in reconstructor.named_parameters():
                        if p.requires_grad:
                            anc = anc + (p - reco_init[n].to(device)).pow(2).mean()
                            n_anc += 1
                    if n_anc > 0:
                        anc = anc / float(n_anc)
                        loss = loss + float(args.lambda_reco_anchor) * anc

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=5.0)
                optim.step()
                tr_loss += float(loss.item()) * int(y.numel())
                tr_n += int(y.numel())

            val_out = evaluate_split(
                val_loader,
                baseline=baseline,
                reconstructor=reconstructor,
                dual=dual,
                gate=gate,
                corrected_weight_floor=float(args.corrected_weight_floor),
                corrected_use_flags=corrected_use_flags,
                thr_h30=thr_h30,
                thr_h50=thr_h50,
                thr_j30=thr_j30,
                thr_j50=thr_j50,
                conf_ref_h=conf_ref_h,
                conf_ref_j=conf_ref_j,
                means=means,
                stds=stds,
                corruption_list=corruption_list,
                rng=rng,
                device=device,
            )
            mval = val_out["metrics_fused"]
            sel = float(mval["auc"]) if str(args.selection_metric) == "auc" else float(mval["fpr50"])
            improved = (sel > best_sel) if str(args.selection_metric) == "auc" else (sel < best_sel)
            if improved:
                best_sel = sel
                best_state = {
                    "gate": {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()},
                    "reco": {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()},
                    "epoch": int(ep),
                    "phase": str(phase_name),
                    "val_metrics": mval,
                }
            rec = {
                "phase": str(phase_name),
                "epoch": int(ep),
                "train_loss": float(tr_loss / max(tr_n, 1)),
                "val_auc_fused": float(mval["auc"]),
                "val_fpr30_fused": float(mval["fpr30"]),
                "val_fpr50_fused": float(mval["fpr50"]),
                "val_auc_joint": float(val_out["metrics_joint"]["auc"]),
                "val_fpr50_joint": float(val_out["metrics_joint"]["fpr50"]),
                "val_auc_hlt": float(val_out["metrics_hlt"]["auc"]),
                "val_fpr50_hlt": float(val_out["metrics_hlt"]["fpr50"]),
                "val_gate_mean": float(np.mean(val_out["g"])),
            }
            history.append(rec)
            print(
                f"{phase_name} ep{ep}: "
                f"train_loss={rec['train_loss']:.4f} | "
                f"val_auc_fused={rec['val_auc_fused']:.4f}, val_fpr50_fused={rec['val_fpr50_fused']:.6f}, "
                f"val_gate_mean={rec['val_gate_mean']:.3f}"
            )

    run_phase("A_gate_only", int(args.phaseA_epochs), phase_b=False)
    run_phase("B_reco_unfreeze", int(args.phaseB_epochs), phase_b=True)

    if best_state is None:
        raise RuntimeError("No best state captured; training failed")
    gate.load_state_dict(best_state["gate"], strict=True)
    reconstructor.load_state_dict(best_state["reco"], strict=True)
    print(f"Loaded best checkpoint from phase={best_state['phase']} epoch={best_state['epoch']}")

    val_out = evaluate_split(
        val_loader,
        baseline=baseline,
        reconstructor=reconstructor,
        dual=dual,
        gate=gate,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=corrected_use_flags,
        thr_h30=thr_h30,
        thr_h50=thr_h50,
        thr_j30=thr_j30,
        thr_j50=thr_j50,
        conf_ref_h=conf_ref_h,
        conf_ref_j=conf_ref_j,
        means=means,
        stds=stds,
        corruption_list=corruption_list,
        rng=rng,
        device=device,
    )
    test_out = evaluate_split(
        test_loader,
        baseline=baseline,
        reconstructor=reconstructor,
        dual=dual,
        gate=gate,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=corrected_use_flags,
        thr_h30=thr_h30,
        thr_h50=thr_h50,
        thr_j30=thr_j30,
        thr_j50=thr_j50,
        conf_ref_h=conf_ref_h,
        conf_ref_j=conf_ref_j,
        means=means,
        stds=stds,
        corruption_list=corruption_list,
        rng=rng,
        device=device,
    )

    report = {
        "run_dir": str(run_dir),
        "save_dir": str(save_root),
        "selection_metric": str(args.selection_metric),
        "best_state": {
            "phase": str(best_state["phase"]),
            "epoch": int(best_state["epoch"]),
            "val_metrics": best_state["val_metrics"],
        },
        "thresholds": {
            "hlt_tpr30": float(thr_h30),
            "hlt_tpr50": float(thr_h50),
            "joint_tpr30": float(thr_j30),
            "joint_tpr50": float(thr_j50),
        },
        "val_metrics": {
            "fused": val_out["metrics_fused"],
            "joint": val_out["metrics_joint"],
            "hlt": val_out["metrics_hlt"],
            "gate_mean": float(np.mean(val_out["g"])),
        },
        "test_metrics": {
            "fused": test_out["metrics_fused"],
            "joint": test_out["metrics_joint"],
            "hlt": test_out["metrics_hlt"],
            "gate_mean": float(np.mean(test_out["g"])),
        },
        "history": history,
        "settings": vars(args),
    }

    with (save_root / "router_gate_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    torch.save(
        {
            "gate": gate.state_dict(),
            "reconstructor": reconstructor.state_dict(),
            "best_state": best_state,
            "settings": vars(args),
        },
        save_root / "router_gate_best.pt",
    )

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "router_gate_scores_val_test.npz",
            val_labels=val_out["y"].astype(np.float32),
            val_hlt=val_out["p_hlt"].astype(np.float32),
            val_joint=val_out["p_joint"].astype(np.float32),
            val_fused=val_out["p_fused"].astype(np.float32),
            val_gate=val_out["g"].astype(np.float32),
            test_labels=test_out["y"].astype(np.float32),
            test_hlt=test_out["p_hlt"].astype(np.float32),
            test_joint=test_out["p_joint"].astype(np.float32),
            test_fused=test_out["p_fused"].astype(np.float32),
            test_gate=test_out["g"].astype(np.float32),
        )

    print("\n======================================================================")
    print("FINAL TEST EVALUATION (Router Gate MoE)")
    print("======================================================================")
    mt = test_out["metrics_hlt"]
    mj = test_out["metrics_joint"]
    mf = test_out["metrics_fused"]
    print(f"HLT   AUC: {mt['auc']:.4f} | FPR30={mt['fpr30']:.6f} | FPR50={mt['fpr50']:.6f}")
    print(f"Joint AUC: {mj['auc']:.4f} | FPR30={mj['fpr30']:.6f} | FPR50={mj['fpr50']:.6f}")
    print(f"Fused AUC: {mf['auc']:.4f} | FPR30={mf['fpr30']:.6f} | FPR50={mf['fpr50']:.6f}")
    print(f"Gate mean(test): {float(np.mean(test_out['g'])):.4f}")
    print(f"Saved results to: {save_root}")


if __name__ == "__main__":
    main()


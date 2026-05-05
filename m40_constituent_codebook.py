#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m40 constituent codebook utilities.

Purpose:
- Fit discrete constituent codebooks (global / pt-stratified / residual2).
- Evaluate codebooks on val/test proxy metrics:
  - token distortion
  - jet-level drift
  - deterministic Offline->HLT residual floor
- Sweep over (strategy, K), shortlist best, and emit launch commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_detfeas_dualview as m33


# -----------------------------------------------------------------------------
# Token / physics helpers
# -----------------------------------------------------------------------------


def _const_to_token5_np(const: np.ndarray) -> np.ndarray:
    eps = 1e-8
    pt = np.clip(const[..., 0], eps, None)
    eta = np.clip(const[..., 1], -5.0, 5.0)
    phi = const[..., 2]
    e = np.clip(const[..., 3], eps, None)
    return np.stack(
        [np.log(pt), eta, np.sin(phi), np.cos(phi), np.log(e)],
        axis=-1,
    ).astype(np.float32)


def _token5_to_const_np(tok: np.ndarray) -> np.ndarray:
    logpt = np.clip(tok[..., 0], -9.0, 9.0)
    eta = np.clip(tok[..., 1], -5.0, 5.0)
    sinphi = tok[..., 2]
    cosphi = tok[..., 3]
    loge = np.clip(tok[..., 4], -9.0, 11.0)

    pt = np.exp(logpt)
    phi = np.arctan2(sinphi, cosphi)
    e = np.exp(loge)
    min_e = pt * np.cosh(eta)
    e = np.maximum(e, min_e)
    return np.stack([pt, eta, phi, e], axis=-1).astype(np.float32)


def _jet_mass_np(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = mask.astype(np.float32)
    pt = const[..., 0] * w
    eta = const[..., 1]
    phi = const[..., 2]
    e = const[..., 3] * w

    px = np.sum(pt * np.cos(phi), axis=1)
    py = np.sum(pt * np.sin(phi), axis=1)
    pz = np.sum(pt * np.sinh(eta), axis=1)
    et = np.sum(e, axis=1)
    m2 = et * et - px * px - py * py - pz * pz
    return np.sqrt(np.clip(m2, 0.0, None)).astype(np.float32)


def _weighted_mean(x: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sum(x * w) / max(np.sum(w), eps))


# -----------------------------------------------------------------------------
# Data prep
# -----------------------------------------------------------------------------


@dataclass
class SplitData:
    const_off: np.ndarray
    mask_off: np.ndarray
    labels: np.ndarray
    train_w: np.ndarray
    jet_keys: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray



def load_split_data(
    train_path: str,
    n_train_jets: int,
    n_train_split: int,
    n_val_split: int,
    n_test_split: int,
    offset_jets: int,
    max_constits: int,
    seed: int,
    use_train_weights: bool,
) -> SplitData:
    files = base._parse_h5_path_arg(str(train_path))
    max_needed = int(offset_jets + n_train_jets)
    all_const, all_labels, all_train_w = base.load_raw_constituents_labels_weights_from_h5(
        files=files,
        max_jets=max_needed,
        max_constits=int(max_constits),
        use_train_weights=bool(use_train_weights),
    )
    if all_const.shape[0] < max_needed:
        raise RuntimeError(f"Requested {max_needed} jets but found {all_const.shape[0]}.")

    const_raw = all_const[offset_jets : offset_jets + n_train_jets].astype(np.float32)
    labels = all_labels[offset_jets : offset_jets + n_train_jets].astype(np.int64)
    train_w = all_train_w[offset_jets : offset_jets + n_train_jets].astype(np.float32)

    cfg = base._deepcopy_config()
    raw_mask = const_raw[:, :, 0] > 0.0
    mask_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~mask_off] = 0.0

    total_need = int(n_train_split + n_val_split + n_test_split)
    if total_need > const_off.shape[0]:
        raise RuntimeError(f"Requested splits sum to {total_need} but only {const_off.shape[0]} jets loaded.")

    idx_all = np.arange(len(labels), dtype=np.int64)
    if total_need < len(idx_all):
        idx_use, _ = train_test_split(
            idx_all,
            train_size=total_need,
            random_state=int(seed),
            stratify=labels[idx_all],
        )
    else:
        idx_use = idx_all

    train_idx, rem_idx = train_test_split(
        idx_use,
        train_size=int(n_train_split),
        random_state=int(seed),
        stratify=labels[idx_use],
    )
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=int(n_val_split),
        test_size=int(n_test_split),
        random_state=int(seed),
        stratify=labels[rem_idx],
    )

    jet_keys = (np.arange(len(const_off), dtype=np.int64) + int(offset_jets)).astype(np.int64)
    return SplitData(
        const_off=const_off,
        mask_off=mask_off,
        labels=labels,
        train_w=train_w,
        jet_keys=jet_keys,
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )


# -----------------------------------------------------------------------------
# Token extraction + sampling
# -----------------------------------------------------------------------------


@dataclass
class TokenPool:
    token: np.ndarray
    token_norm: np.ndarray
    token_mean: np.ndarray
    token_std: np.ndarray
    const: np.ndarray
    jet_w: np.ndarray
    logpt: np.ndarray



def _collect_tokens_from_split(
    const: np.ndarray,
    mask: np.ndarray,
    jet_w: np.ndarray,
    max_jets: int,
    max_tokens: int,
    seed: int,
) -> TokenPool:
    rng = np.random.default_rng(int(seed))
    n = int(const.shape[0])
    if max_jets > 0 and max_jets < n:
        jsel = rng.choice(n, size=int(max_jets), replace=False)
    else:
        jsel = np.arange(n, dtype=np.int64)

    c = const[jsel]
    m = mask[jsel]
    jw = jet_w[jsel]

    tok5 = _const_to_token5_np(c)
    flat_mask = m.reshape(-1)
    flat_tok = tok5.reshape(-1, tok5.shape[-1])[flat_mask]
    flat_const = c.reshape(-1, c.shape[-1])[flat_mask]

    jw_rep = np.repeat(jw, c.shape[1])
    flat_w = jw_rep[flat_mask].astype(np.float64)
    flat_logpt = flat_tok[:, 0].astype(np.float32)

    total = int(flat_tok.shape[0])
    if max_tokens > 0 and total > max_tokens:
        p = flat_w / np.clip(np.sum(flat_w), 1e-12, None)
        idx = rng.choice(total, size=int(max_tokens), replace=False, p=p)
        flat_tok = flat_tok[idx]
        flat_const = flat_const[idx]
        flat_w = flat_w[idx]
        flat_logpt = flat_logpt[idx]

    w_norm = flat_w / np.clip(np.sum(flat_w), 1e-12, None)
    mean = np.sum(flat_tok * w_norm[:, None], axis=0).astype(np.float32)
    var = np.sum(((flat_tok - mean[None, :]) ** 2) * w_norm[:, None], axis=0).astype(np.float32)
    std = np.sqrt(np.clip(var, 1e-8, None)).astype(np.float32)

    tok_norm = ((flat_tok - mean[None, :]) / std[None, :]).astype(np.float32)
    return TokenPool(
        token=flat_tok.astype(np.float32),
        token_norm=tok_norm,
        token_mean=mean,
        token_std=std,
        const=flat_const.astype(np.float32),
        jet_w=flat_w.astype(np.float32),
        logpt=flat_logpt.astype(np.float32),
    )


# -----------------------------------------------------------------------------
# Codebook fit
# -----------------------------------------------------------------------------


@dataclass
class Codebook:
    strategy: str
    k: int
    token_mean: np.ndarray
    token_std: np.ndarray
    data: Dict[str, np.ndarray]



def _fit_minibatch_kmeans(x: np.ndarray, k: int, seed: int, batch_size: int = 8192, n_init: int = 10) -> np.ndarray:
    if x.shape[0] < k:
        raise RuntimeError(f"Need at least k={k} samples, got {x.shape[0]}.")
    km = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=int(seed),
        batch_size=int(batch_size),
        n_init=int(n_init),
        max_iter=200,
        reassignment_ratio=0.01,
    )
    km.fit(x)
    return km.cluster_centers_.astype(np.float32)


def fit_codebook(
    pool: TokenPool,
    strategy: str,
    k: int,
    seed: int,
    pt_n_bands: int = 4,
    residual_coarse_k: int = -1,
) -> Codebook:
    s = str(strategy).strip().lower()
    if s not in {"global", "pt_stratified", "residual2"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    k = int(k)
    x = pool.token_norm

    if s == "global":
        c = _fit_minibatch_kmeans(x, k=k, seed=seed)
        return Codebook(
            strategy=s,
            k=k,
            token_mean=pool.token_mean,
            token_std=pool.token_std,
            data={"centroids_norm": c},
        )

    if s == "pt_stratified":
        b = int(max(2, pt_n_bands))
        q = np.linspace(0.0, 1.0, b + 1)
        edges = np.quantile(pool.logpt, q).astype(np.float32)
        edges[0] = -np.inf
        edges[-1] = np.inf

        band_ids = np.digitize(pool.logpt, edges[1:-1], right=False)
        counts = np.bincount(band_ids, minlength=b).astype(np.float64)
        frac = counts / np.clip(np.sum(counts), 1e-12, None)

        k_band = np.maximum(4, np.floor(frac * k).astype(np.int64))
        while int(np.sum(k_band)) < k:
            j = int(np.argmax(frac - (k_band / np.clip(np.sum(k_band), 1e-12, None))))
            k_band[j] += 1
        while int(np.sum(k_band)) > k:
            j = int(np.argmax(k_band))
            if k_band[j] > 4:
                k_band[j] -= 1
            else:
                break

        centers = []
        k_actual = 0
        for j in range(b):
            ids = np.where(band_ids == j)[0]
            kb = int(k_band[j])
            if ids.size < kb:
                kb = int(max(1, min(ids.size, kb)))
            if kb <= 0 or ids.size == 0:
                continue
            c = _fit_minibatch_kmeans(x[ids], k=kb, seed=int(seed + 17 * (j + 1)))
            centers.append(c)
            k_actual += int(c.shape[0])

        if not centers:
            raise RuntimeError("pt_stratified fit failed: no bands had samples.")

        max_k = max(c.shape[0] for c in centers)
        cent_pad = np.zeros((len(centers), max_k, x.shape[1]), dtype=np.float32)
        mask_pad = np.zeros((len(centers), max_k), dtype=bool)
        k_list = np.zeros((len(centers),), dtype=np.int64)
        for j, c in enumerate(centers):
            kk = c.shape[0]
            cent_pad[j, :kk] = c
            mask_pad[j, :kk] = True
            k_list[j] = kk

        return Codebook(
            strategy=s,
            k=int(k_actual),
            token_mean=pool.token_mean,
            token_std=pool.token_std,
            data={
                "edges_logpt": edges.astype(np.float32),
                "centroids_norm_pad": cent_pad,
                "centroid_mask_pad": mask_pad,
                "k_per_band": k_list,
            },
        )

    # residual2
    if residual_coarse_k is None or int(residual_coarse_k) <= 0:
        kc = int(max(8, round(math.sqrt(max(8, k)))))
    else:
        kc = int(max(2, residual_coarse_k))
    kf = int(max(2, math.ceil(k / max(1, kc))))

    c_coarse = _fit_minibatch_kmeans(x, k=kc, seed=seed)
    idx_c, recon_c = quantize_global(x, c_coarse)
    resid = x - recon_c
    c_fine = _fit_minibatch_kmeans(resid, k=kf, seed=int(seed + 911))

    return Codebook(
        strategy=s,
        k=int(kc * kf),
        token_mean=pool.token_mean,
        token_std=pool.token_std,
        data={
            "coarse_centroids_norm": c_coarse,
            "fine_centroids_norm": c_fine,
            "coarse_k": np.asarray([kc], dtype=np.int64),
            "fine_k": np.asarray([kf], dtype=np.int64),
        },
    )


# -----------------------------------------------------------------------------
# Quantization / decode
# -----------------------------------------------------------------------------


def _batched_argmin_l2(x: np.ndarray, centers: np.ndarray, chunk: int = 200000) -> np.ndarray:
    n = int(x.shape[0])
    out = np.zeros((n,), dtype=np.int64)
    c2 = np.sum(centers * centers, axis=1)[None, :]
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        xx = x[lo:hi]
        x2 = np.sum(xx * xx, axis=1)[:, None]
        d2 = x2 + c2 - 2.0 * (xx @ centers.T)
        out[lo:hi] = np.argmin(d2, axis=1).astype(np.int64)
    return out


def quantize_global(x_norm: np.ndarray, centers_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = _batched_argmin_l2(x_norm, centers_norm)
    recon = centers_norm[idx]
    return idx.astype(np.int64), recon.astype(np.float32)


def _quantize_pt_stratified(
    x_norm: np.ndarray,
    logpt: np.ndarray,
    edges: np.ndarray,
    cent_pad: np.ndarray,
    mask_pad: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    band = np.digitize(logpt, edges[1:-1], right=False).astype(np.int64)
    n = int(x_norm.shape[0])
    idx = np.zeros((n,), dtype=np.int64)
    recon = np.zeros_like(x_norm, dtype=np.float32)

    for b in np.unique(band):
        ids = np.where(band == b)[0]
        centers = cent_pad[b][mask_pad[b]]
        if centers.shape[0] == 0:
            continue
        ii, rr = quantize_global(x_norm[ids], centers)
        idx[ids] = ii
        recon[ids] = rr

    return idx, band, recon


def quantize_with_codebook(token: np.ndarray, codebook: Codebook) -> Dict[str, np.ndarray]:
    mean = codebook.token_mean[None, :]
    std = codebook.token_std[None, :]
    x_norm = ((token - mean) / std).astype(np.float32)

    if codebook.strategy == "global":
        centers = codebook.data["centroids_norm"]
        idx, recon_norm = quantize_global(x_norm, centers)
        recon_tok = recon_norm * std + mean
        return {
            "recon_token": recon_tok.astype(np.float32),
            "index": idx.astype(np.int64),
        }

    if codebook.strategy == "pt_stratified":
        edges = codebook.data["edges_logpt"]
        cent_pad = codebook.data["centroids_norm_pad"]
        mask_pad = codebook.data["centroid_mask_pad"].astype(bool)
        idx, band, recon_norm = _quantize_pt_stratified(x_norm, token[:, 0], edges, cent_pad, mask_pad)
        recon_tok = recon_norm * std + mean
        return {
            "recon_token": recon_tok.astype(np.float32),
            "index": idx.astype(np.int64),
            "band": band.astype(np.int64),
        }

    if codebook.strategy == "residual2":
        cc = codebook.data["coarse_centroids_norm"]
        cf = codebook.data["fine_centroids_norm"]
        idx_c, recon_c = quantize_global(x_norm, cc)
        resid = x_norm - recon_c
        idx_f, recon_f = quantize_global(resid, cf)
        recon_norm = recon_c + recon_f
        recon_tok = recon_norm * std + mean
        return {
            "recon_token": recon_tok.astype(np.float32),
            "index_coarse": idx_c.astype(np.int64),
            "index_fine": idx_f.astype(np.int64),
        }

    raise RuntimeError(f"Unsupported strategy: {codebook.strategy}")


# -----------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------


def save_codebook(codebook: Codebook, out_dir: Path, extra_meta: Optional[Dict] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "codebook.npz"
    meta_path = out_dir / "codebook_meta.json"

    np.savez_compressed(
        npz_path,
        token_mean=codebook.token_mean.astype(np.float32),
        token_std=codebook.token_std.astype(np.float32),
        **{k: np.asarray(v) for k, v in codebook.data.items()},
    )

    meta = {
        "strategy": codebook.strategy,
        "k": int(codebook.k),
        "npz": str(npz_path.name),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return npz_path


def load_codebook(path: str | Path) -> Codebook:
    p = Path(path)
    if p.is_dir():
        meta_p = p / "codebook_meta.json"
        npz_p = p / "codebook.npz"
    elif p.suffix == ".npz":
        meta_p = p.with_name("codebook_meta.json")
        npz_p = p
    else:
        raise RuntimeError(f"Unsupported codebook path: {p}")

    if not npz_p.is_file():
        raise RuntimeError(f"Missing codebook npz: {npz_p}")
    if not meta_p.is_file():
        raise RuntimeError(f"Missing codebook meta: {meta_p}")

    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)

    z = np.load(npz_p, allow_pickle=False)
    strategy = str(meta["strategy"])
    k = int(meta["k"])
    token_mean = z["token_mean"].astype(np.float32)
    token_std = z["token_std"].astype(np.float32)

    data: Dict[str, np.ndarray] = {}
    for kk in z.files:
        if kk in {"token_mean", "token_std"}:
            continue
        data[kk] = z[kk]

    return Codebook(strategy=strategy, k=k, token_mean=token_mean, token_std=token_std, data=data)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def _quantize_const_array(const: np.ndarray, mask: np.ndarray, codebook: Codebook) -> np.ndarray:
    tok = _const_to_token5_np(const)
    flat_mask = mask.reshape(-1)
    flat_tok = tok.reshape(-1, tok.shape[-1])

    active_tok = flat_tok[flat_mask]
    q = quantize_with_codebook(active_tok, codebook)
    recon_tok = q["recon_token"]

    out_tok = flat_tok.copy()
    out_tok[flat_mask] = recon_tok
    out_const = _token5_to_const_np(out_tok.reshape(tok.shape))
    out_const[~mask] = 0.0
    return out_const.astype(np.float32)


def evaluate_codebook(
    codebook: Codebook,
    split: SplitData,
    eval_split: str,
    seed: int,
    dhard_seed_offset: int,
    merge_radius: float,
    eff_plateau_barrel: float,
    eff_plateau_endcap: float,
    smear_a: float,
    smear_b: float,
    smear_c: float,
    residual_unmatched_penalty: float,
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    eps_total: float,
    eps_count: float,
) -> Dict[str, float]:
    if eval_split == "val":
        idx = split.val_idx
    elif eval_split == "test":
        idx = split.test_idx
    elif eval_split == "train":
        idx = split.train_idx
    else:
        raise RuntimeError(f"Unknown eval_split={eval_split}")

    const_ref = split.const_off[idx]
    mask_ref = split.mask_off[idx]
    wj = split.train_w[idx].astype(np.float64)
    keys = split.jet_keys[idx]

    const_q = _quantize_const_array(const_ref, mask_ref, codebook)

    # token distortion on active constituents
    tok_ref = _const_to_token5_np(const_ref)
    tok_q = _const_to_token5_np(const_q)
    flat_mask = mask_ref.reshape(-1)
    tr = tok_ref.reshape(-1, tok_ref.shape[-1])[flat_mask]
    tq = tok_q.reshape(-1, tok_q.shape[-1])[flat_mask]
    dw = np.repeat(wj, const_ref.shape[1])[flat_mask]
    dw = dw / np.clip(np.sum(dw), 1e-12, None)

    d2 = np.sum((tr - tq) ** 2, axis=1)
    token_mse = float(np.sum(d2 * dw))
    token_rmse = float(math.sqrt(max(token_mse, 0.0)))

    # jet drifts
    pt_ref = np.sum(const_ref[..., 0] * mask_ref.astype(np.float32), axis=1)
    pt_q = np.sum(const_q[..., 0] * mask_ref.astype(np.float32), axis=1)
    m_ref = _jet_mass_np(const_ref, mask_ref)
    m_q = _jet_mass_np(const_q, mask_ref)
    cnt = np.sum(mask_ref, axis=1).astype(np.float32)

    pt_rel = np.abs(pt_q - pt_ref) / np.clip(pt_ref, 1e-6, None)
    m_rel = np.abs(m_q - m_ref) / np.clip(m_ref, 1e-6, None)

    pt_rel_mae_w = _weighted_mean(pt_rel.astype(np.float64), wj)
    mass_rel_mae_w = _weighted_mean(m_rel.astype(np.float64), wj)

    # deterministic HLT residual floor
    cfg = base._deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(smear_a)
    cfg["hlt_effects"]["smear_b"] = float(smear_b)
    cfg["hlt_effects"]["smear_c"] = float(smear_c)

    hlt_ref, hm_ref, _st_ref = m33._apply_hlt_effects_deterministic_keyed(
        const=const_ref,
        mask=mask_ref,
        cfg=cfg,
        jet_keys=keys,
        base_seed=int(seed + dhard_seed_offset),
    )
    hlt_q, hm_q, _st_q = m33._apply_hlt_effects_deterministic_keyed(
        const=const_q,
        mask=mask_ref,
        cfg=cfg,
        jet_keys=keys,
        base_seed=int(seed + dhard_seed_offset),
    )

    # Compare HLT(quantized offline) to HLT(reference offline) with m33 residual.
    with torch.no_grad():
        pred_const = torch.from_numpy(hlt_q.astype(np.float32))
        pred_mask = torch.from_numpy(hm_q.astype(bool))
        tgt_const = torch.from_numpy(hlt_ref.astype(np.float32))
        tgt_mask = torch.from_numpy(hm_ref.astype(bool))
        resid = m33._residual_fast_vec(
            pred_const=pred_const,
            pred_mask=pred_mask,
            tgt_const=tgt_const,
            tgt_mask=tgt_mask,
            w_chamfer=float(w_chamfer),
            w_count=float(w_count),
            w_pt=float(w_pt),
            w_mass=float(w_mass),
            unmatched_penalty=float(residual_unmatched_penalty),
        )

    r_tot = resid["total"].cpu().numpy().astype(np.float64)
    r_set = resid["set"].cpu().numpy().astype(np.float64)
    r_count = resid["count"].cpu().numpy().astype(np.float64)
    r_pt = resid["pt"].cpu().numpy().astype(np.float64)
    r_mass = resid["mass"].cpu().numpy().astype(np.float64)

    feasible = (r_tot <= float(eps_total)) & (r_count <= float(eps_count))
    feasible_w = _weighted_mean(feasible.astype(np.float64), wj)

    out: Dict[str, float] = {
        "token_mse": float(token_mse),
        "token_rmse": float(token_rmse),
        "jet_pt_rel_mae_w": float(pt_rel_mae_w),
        "jet_mass_rel_mae_w": float(mass_rel_mae_w),
        "residual_total_mean_w": _weighted_mean(r_tot, wj),
        "residual_set_mean_w": _weighted_mean(r_set, wj),
        "residual_count_mean_w": _weighted_mean(r_count, wj),
        "residual_pt_mean_w": _weighted_mean(r_pt, wj),
        "residual_mass_mean_w": _weighted_mean(r_mass, wj),
        "residual_total_p90": float(np.quantile(r_tot, 0.90)),
        "feasible_rate_w": float(feasible_w),
        "n_eval_jets": int(len(idx)),
    }

    # Scalar for ranking (lower is better)
    out["composite_score"] = (
        float(out["residual_total_mean_w"])
        + 0.35 * float(out["jet_mass_rel_mae_w"])
        + 0.20 * float(out["jet_pt_rel_mae_w"])
        + 0.10 * float(out["token_rmse"])
        - 0.15 * float(out["feasible_rate_w"])
    )
    return out


# -----------------------------------------------------------------------------
# CLI wrappers
# -----------------------------------------------------------------------------


def _common_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--n_train_jets", type=int, default=370000)
    p.add_argument("--n_train_split", type=int, default=50000)
    p.add_argument("--n_val_split", type=int, default=20000)
    p.add_argument("--n_test_split", type=int, default=300000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_train_weights", action="store_true")


def build_fit_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m40 fit constituent codebook")
    _common_data_args(p)
    p.add_argument("--strategy", type=str, choices=["global", "pt_stratified", "residual2"], default="global")
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--pt_n_bands", type=int, default=4)
    p.add_argument("--residual_coarse_k", type=int, default=-1)
    p.add_argument("--max_fit_jets", type=int, default=120000)
    p.add_argument("--max_fit_tokens", type=int, default=2500000)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, required=True)
    return p


def cli_fit(args: argparse.Namespace) -> int:
    split = load_split_data(
        train_path=args.train_path,
        n_train_jets=int(args.n_train_jets),
        n_train_split=int(args.n_train_split),
        n_val_split=int(args.n_val_split),
        n_test_split=int(args.n_test_split),
        offset_jets=int(args.offset_jets),
        max_constits=int(args.max_constits),
        seed=int(args.seed),
        use_train_weights=bool(args.use_train_weights),
    )
    pool = _collect_tokens_from_split(
        const=split.const_off[split.train_idx],
        mask=split.mask_off[split.train_idx],
        jet_w=split.train_w[split.train_idx],
        max_jets=int(args.max_fit_jets),
        max_tokens=int(args.max_fit_tokens),
        seed=int(args.seed),
    )

    codebook = fit_codebook(
        pool=pool,
        strategy=str(args.strategy),
        k=int(args.k),
        seed=int(args.seed),
        pt_n_bands=int(args.pt_n_bands),
        residual_coarse_k=int(args.residual_coarse_k),
    )

    out_dir = Path(args.save_dir) / args.run_name
    meta = {
        "strategy": str(args.strategy),
        "k_requested": int(args.k),
        "k_effective": int(codebook.k),
        "seed": int(args.seed),
        "max_fit_jets": int(args.max_fit_jets),
        "max_fit_tokens": int(args.max_fit_tokens),
        "n_tokens_used": int(pool.token.shape[0]),
    }
    save_codebook(codebook, out_dir=out_dir, extra_meta=meta)
    print(f"Saved codebook: {out_dir}")
    print(json.dumps(meta, indent=2))
    return 0


def build_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m40 evaluate constituent codebook")
    _common_data_args(p)
    p.add_argument("--codebook_path", type=str, required=True)
    p.add_argument("--eval_split", type=str, choices=["train", "val", "test"], default="val")
    p.add_argument("--dhard_seed_offset", type=int, default=1337)

    p.add_argument("--merge_radius", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["merge_radius"]))
    p.add_argument("--eff_plateau_barrel", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"]))
    p.add_argument("--eff_plateau_endcap", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"]))
    p.add_argument("--smear_a", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_a"]))
    p.add_argument("--smear_b", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_b"]))
    p.add_argument("--smear_c", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_c"]))

    p.add_argument("--residual_unmatched_penalty", type=float, default=0.20)
    p.add_argument("--search_w_chamfer", type=float, default=1.00)
    p.add_argument("--search_w_count", type=float, default=0.25)
    p.add_argument("--search_w_pt", type=float, default=0.12)
    p.add_argument("--search_w_mass", type=float, default=0.08)
    p.add_argument("--search_eps_total", type=float, default=0.60)
    p.add_argument("--search_eps_count", type=float, default=0.30)

    p.add_argument("--save_json", type=str, default="")
    return p


def cli_eval(args: argparse.Namespace) -> int:
    codebook = load_codebook(args.codebook_path)
    split = load_split_data(
        train_path=args.train_path,
        n_train_jets=int(args.n_train_jets),
        n_train_split=int(args.n_train_split),
        n_val_split=int(args.n_val_split),
        n_test_split=int(args.n_test_split),
        offset_jets=int(args.offset_jets),
        max_constits=int(args.max_constits),
        seed=int(args.seed),
        use_train_weights=bool(args.use_train_weights),
    )

    out = evaluate_codebook(
        codebook=codebook,
        split=split,
        eval_split=str(args.eval_split),
        seed=int(args.seed),
        dhard_seed_offset=int(args.dhard_seed_offset),
        merge_radius=float(args.merge_radius),
        eff_plateau_barrel=float(args.eff_plateau_barrel),
        eff_plateau_endcap=float(args.eff_plateau_endcap),
        smear_a=float(args.smear_a),
        smear_b=float(args.smear_b),
        smear_c=float(args.smear_c),
        residual_unmatched_penalty=float(args.residual_unmatched_penalty),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
    )

    meta = {
        "codebook_path": str(args.codebook_path),
        "strategy": codebook.strategy,
        "k_effective": int(codebook.k),
        "eval_split": str(args.eval_split),
        "seed": int(args.seed),
    }
    payload = {**meta, **out}
    if args.save_json:
        out_p = Path(args.save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved eval: {out_p}")

    print(json.dumps(payload, indent=2))
    return 0


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------


def _parse_csv_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_csv_list_str(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def build_sweep_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m40 run quantization sweep")
    _common_data_args(p)
    p.add_argument("--strategies", type=str, default="global,pt_stratified,residual2")
    p.add_argument("--k_values", type=str, default="128,256,512,1024")
    p.add_argument("--top_n", type=int, default=3)

    p.add_argument("--pt_n_bands", type=int, default=4)
    p.add_argument("--residual_coarse_k", type=int, default=-1)
    p.add_argument("--max_fit_jets", type=int, default=120000)
    p.add_argument("--max_fit_tokens", type=int, default=2500000)

    p.add_argument("--eval_split", type=str, choices=["train", "val", "test"], default="val")
    p.add_argument("--dhard_seed_offset", type=int, default=1337)
    p.add_argument("--merge_radius", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["merge_radius"]))
    p.add_argument("--eff_plateau_barrel", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"]))
    p.add_argument("--eff_plateau_endcap", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"]))
    p.add_argument("--smear_a", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_a"]))
    p.add_argument("--smear_b", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_b"]))
    p.add_argument("--smear_c", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_c"]))

    p.add_argument("--residual_unmatched_penalty", type=float, default=0.20)
    p.add_argument("--search_w_chamfer", type=float, default=1.00)
    p.add_argument("--search_w_count", type=float, default=0.25)
    p.add_argument("--search_w_pt", type=float, default=0.12)
    p.add_argument("--search_w_mass", type=float, default=0.08)
    p.add_argument("--search_eps_total", type=float, default=0.60)
    p.add_argument("--search_eps_count", type=float, default=0.30)

    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--emit_launcher", action="store_true")
    return p


def _emit_launcher_script(shortlist: List[Dict], out_path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Auto-generated by m40 sweep.",
        "# NOTE: Current m39 does not yet consume codebook path as decoding constraints.",
        "# These commands launch baseline m39 runs with codebook provenance exported.",
        "",
    ]
    for i, row in enumerate(shortlist, start=1):
        cb = row["codebook_dir"]
        label = row["label"]
        lines.append(f"echo \"[{i}] {label} score={row['composite_score']:.6f}\"")
        lines.append(
            f"CODEBOOK_PATH='{cb}' CODEBOOK_LABEL='{label}' "
            "bash sbatch/reco_teacher_joint_fusion_6model_150k75k150k/submit_m39_prefixspecialist_detresid_multicand_150k75k300k.sh"
        )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_path.chmod(0o755)


def cli_sweep(args: argparse.Namespace) -> int:
    strategies = _parse_csv_list_str(args.strategies)
    k_values = _parse_csv_list_int(args.k_values)
    out_root = Path(args.save_dir) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    split = load_split_data(
        train_path=args.train_path,
        n_train_jets=int(args.n_train_jets),
        n_train_split=int(args.n_train_split),
        n_val_split=int(args.n_val_split),
        n_test_split=int(args.n_test_split),
        offset_jets=int(args.offset_jets),
        max_constits=int(args.max_constits),
        seed=int(args.seed),
        use_train_weights=bool(args.use_train_weights),
    )

    pool = _collect_tokens_from_split(
        const=split.const_off[split.train_idx],
        mask=split.mask_off[split.train_idx],
        jet_w=split.train_w[split.train_idx],
        max_jets=int(args.max_fit_jets),
        max_tokens=int(args.max_fit_tokens),
        seed=int(args.seed),
    )

    rows: List[Dict] = []

    for s in strategies:
        for k in k_values:
            label = f"{s}_k{k}"
            cb_dir = out_root / "codebooks" / label
            print("=" * 72)
            print(f"[m40 sweep] fitting {label}")
            cb = fit_codebook(
                pool=pool,
                strategy=s,
                k=int(k),
                seed=int(args.seed),
                pt_n_bands=int(args.pt_n_bands),
                residual_coarse_k=int(args.residual_coarse_k),
            )
            save_codebook(
                cb,
                out_dir=cb_dir,
                extra_meta={
                    "label": label,
                    "k_requested": int(k),
                    "k_effective": int(cb.k),
                    "seed": int(args.seed),
                },
            )

            print(f"[m40 sweep] evaluating {label}")
            ev = evaluate_codebook(
                codebook=cb,
                split=split,
                eval_split=str(args.eval_split),
                seed=int(args.seed),
                dhard_seed_offset=int(args.dhard_seed_offset),
                merge_radius=float(args.merge_radius),
                eff_plateau_barrel=float(args.eff_plateau_barrel),
                eff_plateau_endcap=float(args.eff_plateau_endcap),
                smear_a=float(args.smear_a),
                smear_b=float(args.smear_b),
                smear_c=float(args.smear_c),
                residual_unmatched_penalty=float(args.residual_unmatched_penalty),
                w_chamfer=float(args.search_w_chamfer),
                w_count=float(args.search_w_count),
                w_pt=float(args.search_w_pt),
                w_mass=float(args.search_w_mass),
                eps_total=float(args.search_eps_total),
                eps_count=float(args.search_eps_count),
            )

            row = {
                "label": label,
                "strategy": s,
                "k_requested": int(k),
                "k_effective": int(cb.k),
                "codebook_dir": str(cb_dir.resolve()),
                **ev,
            }
            rows.append(row)
            with open(cb_dir / "eval_report.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

    rows_sorted = sorted(rows, key=lambda x: float(x["composite_score"]))

    # Leaderboard CSV
    csv_path = out_root / "leaderboard.csv"
    fields = [
        "label",
        "strategy",
        "k_requested",
        "k_effective",
        "composite_score",
        "residual_total_mean_w",
        "feasible_rate_w",
        "jet_mass_rel_mae_w",
        "jet_pt_rel_mae_w",
        "token_rmse",
        "residual_total_p90",
        "n_eval_jets",
        "codebook_dir",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({k: r.get(k, "") for k in fields})

    top_n = int(max(1, args.top_n))
    shortlist = rows_sorted[:top_n]
    short_json = out_root / "shortlist.json"
    with open(short_json, "w", encoding="utf-8") as f:
        json.dump(shortlist, f, indent=2)

    print("=" * 72)
    print("m40 sweep leaderboard (best first):")
    for i, r in enumerate(shortlist, start=1):
        print(
            f"  {i}. {r['label']} | score={r['composite_score']:.6f} | "
            f"R={r['residual_total_mean_w']:.5f} | feas={r['feasible_rate_w']:.4f} | "
            f"mass={r['jet_mass_rel_mae_w']:.5f} | token={r['token_rmse']:.5f}"
        )
    print(f"Saved leaderboard: {csv_path}")
    print(f"Saved shortlist:   {short_json}")

    if bool(args.emit_launcher):
        launcher = out_root / "launch_m39_from_m40_shortlist.sh"
        _emit_launcher_script(shortlist, launcher)
        print(f"Saved launcher:    {launcher}")

    return 0


# -----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="m40 constituent codebook toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", parents=[build_fit_parser()], add_help=False)
    p_fit.set_defaults(_fn=cli_fit)

    p_eval = sub.add_parser("eval", parents=[build_eval_parser()], add_help=False)
    p_eval.set_defaults(_fn=cli_eval)

    p_sw = sub.add_parser("sweep", parents=[build_sweep_parser()], add_help=False)
    p_sw.set_defaults(_fn=cli_sweep)

    args = p.parse_args()
    return int(args._fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

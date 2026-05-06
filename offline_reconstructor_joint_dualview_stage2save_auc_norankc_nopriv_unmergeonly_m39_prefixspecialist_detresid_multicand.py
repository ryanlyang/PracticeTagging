#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m39: Prefix-specialist m28 completion + deterministic D_hard residual selection + multi-candidate dualview.

Pipeline:
1) Load Offline jets and build deterministic pseudo-HLT.
2) Train Teacher (Offline) and HLT baseline classifiers.
3) Train carryover predictor on HLT tokens (token-level carry likelihood).
4) Train m28-style HLT->Offline completer as a specialist for a fixed carryover prefix length.
5) For each jet: build diverse seed prefixes, force-prefix decode K candidates,
   run deterministic D_hard on each candidate, score residuals, keep best M.
6) Train final multi-candidate dualview heads (NoGate/Gated).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_detfeas_dualview as m33
import offline_reconstructor_joint_dualview_seq2seq_nexttoken_m28_sinkset_noar as m28
from unmerge_correct_hlt import ParticleTransformer, compute_features, get_stats, standardize


# -----------------------------------------------------------------------------
# Data / utils
# -----------------------------------------------------------------------------


@dataclass
class TokenCodebookQuantizer:
    strategy: str
    mean: np.ndarray  # [5]
    std: np.ndarray   # [5]
    data: Dict[str, np.ndarray]


def _load_token_codebook_quantizer(path: str) -> TokenCodebookQuantizer:
    p = Path(path)
    if p.is_dir():
        meta_p = p / "codebook_meta.json"
        npz_p = p / "codebook.npz"
    elif p.suffix == ".npz":
        meta_p = p.with_name("codebook_meta.json")
        npz_p = p
    else:
        raise RuntimeError(f"Unsupported codebook path: {path}")

    if not npz_p.is_file():
        raise RuntimeError(f"Missing codebook npz: {npz_p}")
    if not meta_p.is_file():
        raise RuntimeError(f"Missing codebook meta: {meta_p}")

    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    strategy = str(meta.get("strategy", "")).strip().lower()
    if strategy not in {"global", "pt_stratified", "residual2"}:
        raise RuntimeError(f"Unsupported codebook strategy in meta: {strategy}")

    z = np.load(npz_p, allow_pickle=False)
    mean = z["token_mean"].astype(np.float32)
    std = z["token_std"].astype(np.float32)
    data: Dict[str, np.ndarray] = {}
    for k in z.files:
        if k in {"token_mean", "token_std"}:
            continue
        data[k] = z[k]
    return TokenCodebookQuantizer(strategy=strategy, mean=mean, std=std, data=data)


def _batched_argmin_l2_np(x: np.ndarray, centers: np.ndarray, chunk: int = 200000) -> np.ndarray:
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


def _quantize_token5_np(tok: np.ndarray, q: Optional[TokenCodebookQuantizer]) -> np.ndarray:
    if q is None:
        return tok
    shp = tok.shape
    x = tok.reshape(-1, shp[-1]).astype(np.float32)
    mean = q.mean[None, :]
    std = q.std[None, :]
    xn = (x - mean) / std

    if q.strategy == "global":
        c = q.data["centroids_norm"].astype(np.float32)
        idx = _batched_argmin_l2_np(xn, c)
        xq = c[idx]
    elif q.strategy == "pt_stratified":
        edges = q.data["edges_logpt"].astype(np.float32)
        cent_pad = q.data["centroids_norm_pad"].astype(np.float32)
        mask_pad = q.data["centroid_mask_pad"].astype(bool)
        band = np.digitize(x[:, 0], edges[1:-1], right=False).astype(np.int64)
        xq = np.zeros_like(xn, dtype=np.float32)
        for b in np.unique(band):
            ids = np.where(band == b)[0]
            cc = cent_pad[b][mask_pad[b]]
            if cc.shape[0] <= 0:
                xq[ids] = xn[ids]
                continue
            ii = _batched_argmin_l2_np(xn[ids], cc)
            xq[ids] = cc[ii]
    elif q.strategy == "residual2":
        cc = q.data["coarse_centroids_norm"].astype(np.float32)
        cf = q.data["fine_centroids_norm"].astype(np.float32)
        ic = _batched_argmin_l2_np(xn, cc)
        rc = cc[ic]
        rf = xn - rc
        iff = _batched_argmin_l2_np(rf, cf)
        xq = rc + cf[iff]
    else:
        xq = xn

    out = xq * std + mean
    return out.reshape(shp).astype(np.float32)


def _quantize_token5_torch(tok: torch.Tensor, q: Optional[TokenCodebookQuantizer]) -> torch.Tensor:
    if q is None:
        return tok
    dev = tok.device
    dt = tok.dtype
    shp = tok.shape
    x = tok.reshape(-1, shp[-1])
    mean = torch.tensor(q.mean, dtype=dt, device=dev).view(1, -1)
    std = torch.tensor(q.std, dtype=dt, device=dev).view(1, -1)
    xn = (x - mean) / std

    def _nn(xn_part: torch.Tensor, centers_np: np.ndarray) -> torch.Tensor:
        c = torch.tensor(centers_np, dtype=dt, device=dev)
        x2 = (xn_part * xn_part).sum(dim=1, keepdim=True)
        c2 = (c * c).sum(dim=1).view(1, -1)
        d2 = x2 + c2 - 2.0 * (xn_part @ c.t())
        idx = torch.argmin(d2, dim=1)
        return c[idx]

    if q.strategy == "global":
        xq = _nn(xn, q.data["centroids_norm"].astype(np.float32))
    elif q.strategy == "pt_stratified":
        edges = q.data["edges_logpt"].astype(np.float32)
        cent_pad = q.data["centroids_norm_pad"].astype(np.float32)
        mask_pad = q.data["centroid_mask_pad"].astype(bool)
        band = np.digitize(x[:, 0].detach().cpu().numpy().astype(np.float32), edges[1:-1], right=False).astype(np.int64)
        xq = torch.zeros_like(xn)
        for b in np.unique(band):
            ids = np.where(band == b)[0]
            if ids.size == 0:
                continue
            cc = cent_pad[b][mask_pad[b]]
            if cc.shape[0] <= 0:
                xq[torch.tensor(ids, dtype=torch.long, device=dev)] = xn[torch.tensor(ids, dtype=torch.long, device=dev)]
                continue
            ids_t = torch.tensor(ids, dtype=torch.long, device=dev)
            xq[ids_t] = _nn(xn[ids_t], cc)
    elif q.strategy == "residual2":
        cc = q.data["coarse_centroids_norm"].astype(np.float32)
        cf = q.data["fine_centroids_norm"].astype(np.float32)
        rc = _nn(xn, cc)
        rf = xn - rc
        xq = rc + _nn(rf, cf)
    else:
        xq = xn

    out = xq * std + mean
    return out.reshape(shp)


def _const_to_token5_np(const: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pt = np.clip(const[..., 0], 1e-8, None)
    eta = np.clip(const[..., 1], -5.0, 5.0)
    phi = const[..., 2]
    e = np.clip(const[..., 3], 1e-8, None)
    tok = np.stack(
        [
            np.log(pt),
            eta,
            np.sin(phi),
            np.cos(phi),
            np.log(e),
        ],
        axis=-1,
    ).astype(np.float32)
    tok[~mask] = 0.0
    return tok


def _token5_to_const_np(tok: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pt = np.exp(np.clip(tok[..., 0], -20.0, 20.0)).astype(np.float32)
    eta = np.clip(tok[..., 1], -5.0, 5.0).astype(np.float32)
    sphi = tok[..., 2]
    cphi = tok[..., 3]
    phi = np.arctan2(sphi, cphi).astype(np.float32)
    e = np.exp(np.clip(tok[..., 4], -20.0, 20.0)).astype(np.float32)
    out = np.stack([pt, eta, phi, e], axis=-1).astype(np.float32)
    out[~mask] = 0.0
    return out


def _build_candidate_token12_np(
    off_const: np.ndarray,
    off_mask: np.ndarray,
    hlt_const: np.ndarray,
    hlt_mask: np.ndarray,
) -> np.ndarray:
    off5 = _const_to_token5_np(off_const, off_mask)
    hlt5 = _const_to_token5_np(hlt_const, hlt_mask)
    om = off_mask.astype(np.float32)[..., None]
    hm = hlt_mask.astype(np.float32)[..., None]
    tok = np.concatenate([off5, hlt5, om, hm], axis=-1).astype(np.float32)
    valid = (off_mask | hlt_mask)
    tok[~valid] = 0.0
    return tok


def _fpr50(y_true: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, p, sample_weight=w)
    return float(m33.fpr_at_target_tpr(fpr, tpr, 0.50))


class RecoCarryDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray, carry_tgt: np.ndarray):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.carry_tgt = torch.tensor(carry_tgt, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "carry_tgt": self.carry_tgt[i],
        }


class FeatMaskDataset(Dataset):
    def __init__(self, feat_hlt: np.ndarray, mask_hlt: np.ndarray):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat": self.feat_hlt[i],
            "mask": self.mask_hlt[i],
        }


class RecoSpecialistDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        tgt_tok: np.ndarray,
        tgt_mask: np.ndarray,
        labels: np.ndarray,
        prefix_tok: np.ndarray,
        prefix_len: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.tgt_tok = torch.tensor(tgt_tok, dtype=torch.float32)
        self.tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.prefix_tok = torch.tensor(prefix_tok, dtype=torch.float32)
        self.prefix_len = torch.tensor(prefix_len, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.feat_hlt.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "tgt_tok": self.tgt_tok[i],
            "tgt_mask": self.tgt_mask[i],
            "label": self.labels[i],
            "prefix_tok": self.prefix_tok[i],
            "prefix_len": self.prefix_len[i],
        }


def _build_specialist_prefix_tokens(
    carry_probs: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    prefix_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(const_hlt.shape[0])
    pmax = int(max(0, prefix_max))
    if pmax <= 0:
        return np.zeros((n, 0, 5), dtype=np.float32), np.zeros((n,), dtype=np.int64)
    hlt_tok = _const_to_token5_np(const_hlt, mask_hlt)
    pref_tok = np.zeros((n, pmax, 5), dtype=np.float32)
    pref_len = np.zeros((n,), dtype=np.int64)

    for i in range(n):
        ids = np.where(mask_hlt[i])[0]
        if ids.size == 0:
            continue
        nn = int(min(pmax, ids.size))
        scores = carry_probs[i, ids]
        pt_vals = const_hlt[i, ids, 0]
        ord_idx = np.lexsort((-pt_vals, -scores))
        pick = ids[ord_idx[:nn]]
        pick = pick[np.argsort(-const_hlt[i, pick, 0])]
        pref_tok[i, :nn] = hlt_tok[i, pick]
        pref_len[i] = nn
    return pref_tok, pref_len


def _build_continuation_targets_from_prefix(
    off_const: np.ndarray,
    off_mask: np.ndarray,
    prefix_tok: np.ndarray,
    prefix_len: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build continuation targets by greedily removing Offline tokens that match prefix tokens.
    """
    n, l, _ = off_const.shape
    off_tok = _const_to_token5_np(off_const, off_mask)
    out_tok = np.zeros((n, l, 5), dtype=np.float32)
    out_mask = np.zeros((n, l), dtype=bool)
    w = np.asarray([1.0, 0.35, 0.25, 0.25, 1.0], dtype=np.float32)

    for i in range(n):
        off_ids = np.where(off_mask[i])[0]
        if off_ids.size == 0:
            continue
        keep_ids = off_ids.tolist()
        npref = int(min(prefix_len[i], prefix_tok.shape[1]))
        if npref > 0 and len(keep_ids) > 0:
            pref_i = prefix_tok[i, :npref]
            taken = set()
            for j in range(npref):
                rem = [idx for idx in keep_ids if idx not in taken]
                if not rem:
                    break
                rem_arr = np.asarray(rem, dtype=np.int64)
                d = np.abs(off_tok[i, rem_arr] - pref_i[j][None, :]) * w[None, :]
                d = d.sum(axis=1)
                k = int(np.argmin(d))
                taken.add(int(rem_arr[k]))
            keep_ids = [idx for idx in keep_ids if idx not in taken]

        kk = len(keep_ids)
        if kk > 0:
            out_tok[i, :kk] = off_tok[i, keep_ids]
            out_mask[i, :kk] = True

    return out_tok, out_mask


def _gather_contiguous_from_full(
    out_full: Dict[str, torch.Tensor],
    prefix_len: torch.Tensor,
    cont_steps: int,
) -> Dict[str, torch.Tensor]:
    b = int(prefix_len.shape[0])
    if cont_steps <= 0:
        cont_steps = 1
    idx = prefix_len.view(b, 1) + torch.arange(cont_steps, device=prefix_len.device).view(1, cont_steps)

    pred = out_full["pred_tok"]
    stop = out_full["stop_logits"]
    conf = out_full["conf_logits"]
    attn = out_full["attn"]

    dd = pred.shape[-1]
    ll = attn.shape[-1]
    pred_cont = pred.gather(1, idx.unsqueeze(-1).expand(-1, -1, dd))
    stop_cont = stop.gather(1, idx)
    conf_cont = conf.gather(1, idx)
    attn_cont = attn.gather(1, idx.unsqueeze(-1).expand(-1, -1, ll))

    return {
        "pred_tok": pred_cont,
        "stop_logits": stop_cont,
        "conf_logits": conf_cont,
        "count_pred": out_full["count_pred"],
        "attn": attn_cont,
        "gate": torch.zeros_like(stop_cont),
    }


def _forward_free_running_forced_prefix_single(
    model: m28.HLT2OfflineSeq2Seq,
    feat_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_hlt: torch.Tensor,
    prefix_tok: torch.Tensor,
    prefix_len: torch.Tensor,
    max_steps: int,
    token_quantizer: Optional[TokenCodebookQuantizer] = None,
) -> Dict[str, torch.Tensor]:
    b = int(feat_hlt.shape[0])
    t = int(max(1, max_steps))
    mem, hlt_tok, count_pred = model.encode(feat_hlt, mask_hlt, const_hlt)
    mem_pad = ~mask_hlt

    in_tok = model.bos_token.expand(b, 1, model.token_dim)
    n_layers = len(model.decoder.layers)
    layer_cache: List[torch.Tensor | None] = [None] * n_layers

    pred_seq = []
    stop_seq = []
    conf_seq = []
    attn_seq = []
    gate_seq = []

    for step in range(t):
        x_step = model.dec_in(in_tok) + model.dec_pos[:, step : step + 1, :]
        h_last, layer_cache = model._decoder_step_cached(x_step, mem, mem_pad, layer_cache)
        pred_tok, stop_logits, conf_logits, attn, gate = model._predict_from_hidden(
            h_last,
            mem,
            mask_hlt,
            hlt_tok,
            hyp_idx=0,
        )
        next_tok = pred_tok[:, 0, :]
        force_mask = prefix_len > step
        if bool(force_mask.any()):
            forced = prefix_tok[:, step, :]
            next_tok = torch.where(force_mask.unsqueeze(1), forced, next_tok)
        if token_quantizer is not None:
            next_tok = _quantize_token5_torch(next_tok, token_quantizer)
        pred_seq.append(next_tok)
        stop_seq.append(stop_logits[:, 0])
        conf_seq.append(conf_logits[:, 0])
        attn_seq.append(attn[:, 0, :])
        gate_seq.append(gate[:, 0])
        in_tok = next_tok.unsqueeze(1)

    return {
        "pred_tok": torch.stack(pred_seq, dim=1),
        "stop_logits": torch.stack(stop_seq, dim=1),
        "conf_logits": torch.stack(conf_seq, dim=1),
        "count_pred": count_pred,
        "attn": torch.stack(attn_seq, dim=1),
        "gate": torch.stack(gate_seq, dim=1),
    }


def _train_reco_specialist_with_prefix(
    model: m28.HLT2OfflineSeq2Seq,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: Dict,
    loss_cfg: Dict,
    token_quantizer: Optional[TokenCodebookQuantizer] = None,
) -> Tuple[m28.HLT2OfflineSeq2Seq, Dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_state = None
    best_val = float("inf")
    best_ep = -1
    no_improve = 0
    min_epochs = int(train_cfg.get("min_epochs", 1))
    patience = int(train_cfg.get("patience", 20))
    total_epochs = int(train_cfg["epochs"])

    phase_rewind = bool(loss_cfg.get("phase_rewind", True))
    phase_reset_optimizer = bool(loss_cfg.get("phase_reset_optimizer", True))
    phase_lr_decay = float(loss_cfg.get("phase_lr_decay", 0.80))
    physics_warmup_epochs = int(max(loss_cfg.get("physics_warmup_epochs", 0), 0))
    current_phase_idx = -1
    phase_best_state = None
    phase_best_val = float("inf")
    phase_best_ep = -1
    ep_times: List[float] = []

    for ep in range(total_epochs):
        t0 = time.perf_counter()
        phase_sched = m28.phased_curriculum_schedule(ep, total_epochs, loss_cfg)
        phase_idx = int(phase_sched["phase_idx"])
        phase_name = str(phase_sched["phase_name"])
        ss_prob = float(phase_sched["ss_prob"])
        fr_mix_alpha = float(phase_sched["fr_mix_alpha"])
        fr_every_n = int(phase_sched["fr_every_n"])
        physics_scale = 1.0 if physics_warmup_epochs <= 0 else min(1.0, float(ep + 1) / float(physics_warmup_epochs))

        if current_phase_idx < 0:
            current_phase_idx = phase_idx
            print(f"Entering phase {phase_idx} ({phase_name}) at epoch {ep+1}")
        elif phase_idx != current_phase_idx:
            if phase_rewind and phase_best_state is not None:
                model.load_state_dict(phase_best_state)
                print(
                    f"Phase transition {current_phase_idx}->{phase_idx}: rewound to "
                    f"phase-best epoch {phase_best_ep} (valFR={phase_best_val:.6f})"
                )
            if phase_reset_optimizer:
                old_lr = float(opt.param_groups[0]["lr"])
                new_lr = max(old_lr * float(phase_lr_decay), float(train_cfg["lr"]) * 0.20)
                opt = torch.optim.AdamW(model.parameters(), lr=new_lr, weight_decay=float(train_cfg["weight_decay"]))
                print(f"Phase {phase_idx} optimizer reset: lr {old_lr:.3e} -> {new_lr:.3e}")
            current_phase_idx = phase_idx
            phase_best_state = None
            phase_best_val = float("inf")
            phase_best_ep = -1
            no_improve = 0

        model.train()
        tr_total = tr_set = tr_eos = tr_cnt = tr_jpt = tr_j4 = tr_cfr = tr_cfp = tr_fr = 0.0
        ntr = 0

        for bi, batch in enumerate(train_loader):
            feat_hlt = batch["feat_hlt"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            tgt_tok_cont = batch["tgt_tok"].to(device)
            tgt_mask_cont = batch["tgt_mask"].to(device)
            pref_tok = batch["prefix_tok"].to(device)
            pref_len = batch["prefix_len"].to(device)

            cont_steps = int(max(tgt_mask_cont.float().sum(dim=1).max().item(), 1))
            tgt_tok_trim = tgt_tok_cont[:, :cont_steps, :]
            tgt_mask_trim = tgt_mask_cont[:, :cont_steps]
            if token_quantizer is not None:
                tgt_tok_trim = _quantize_token5_torch(tgt_tok_trim, token_quantizer)
            t_full = int(pref_tok.shape[1] + cont_steps)

            full_tok = torch.zeros((feat_hlt.shape[0], t_full, 5), dtype=tgt_tok_cont.dtype, device=device)
            for ib in range(int(feat_hlt.shape[0])):
                pp = int(min(pref_len[ib].item(), pref_tok.shape[1]))
                cc = int(tgt_mask_trim[ib].sum().item())
                if pp > 0:
                    full_tok[ib, :pp] = pref_tok[ib, :pp]
                if cc > 0:
                    full_tok[ib, pp : pp + cc] = tgt_tok_trim[ib, :cc]

            valid_prev_mask = torch.zeros((feat_hlt.shape[0], t_full), dtype=torch.bool, device=device)
            for ib in range(int(feat_hlt.shape[0])):
                pp = int(min(pref_len[ib].item(), pref_tok.shape[1]))
                cc = int(tgt_mask_trim[ib].sum().item())
                if cc > 0:
                    valid_prev_mask[ib, pp : pp + cc] = True

            out_first = model.forward_teacher(feat_hlt, mask_hlt, const_hlt, full_tok)
            prev_tok = out_first["pred_tok"]
            if token_quantizer is not None:
                prev_tok = _quantize_token5_torch(prev_tok, token_quantizer)
            dec_in = m28.build_decoder_input_tokens(
                model.bos_token,
                full_tok,
                model_prev_tokens=prev_tok,
                model_prob=float(ss_prob),
                valid_prev_mask=valid_prev_mask,
            )
            out_full = model.forward_teacher_with_inputs(feat_hlt, mask_hlt, const_hlt, dec_in)
            out_cont = _gather_contiguous_from_full(out_full, pref_len.long(), cont_steps)
            if token_quantizer is not None:
                out_cont = dict(out_cont)
                out_cont["pred_tok"] = _quantize_token5_torch(out_cont["pred_tok"], token_quantizer)
            losses = m28.compute_reco_losses(out_cont, tgt_tok_trim, tgt_mask_trim, loss_cfg, physics_scale=physics_scale)

            tf_total = losses["total"]
            apply_fr = (fr_every_n > 0) and (fr_mix_alpha > 0.0) and ((bi + 1) % fr_every_n == 0)
            if apply_fr:
                out_fr_full = _forward_free_running_forced_prefix_single(
                    model=model,
                    feat_hlt=feat_hlt,
                    mask_hlt=mask_hlt,
                    const_hlt=const_hlt,
                    prefix_tok=pref_tok,
                    prefix_len=pref_len.long(),
                    max_steps=t_full,
                    token_quantizer=token_quantizer,
                )
                out_fr_cont = _gather_contiguous_from_full(out_fr_full, pref_len.long(), cont_steps)
                if token_quantizer is not None:
                    out_fr_cont = dict(out_fr_cont)
                    out_fr_cont["pred_tok"] = _quantize_token5_torch(out_fr_cont["pred_tok"], token_quantizer)
                fr_losses = m28.compute_reco_losses(out_fr_cont, tgt_tok_trim, tgt_mask_trim, loss_cfg, physics_scale=physics_scale)
                total = (1.0 - float(fr_mix_alpha)) * tf_total + float(fr_mix_alpha) * fr_losses["total"]
                tr_fr += float(fr_losses["total"].item()) * int(feat_hlt.shape[0])
            else:
                total = tf_total

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = int(feat_hlt.shape[0])
            tr_total += float(total.item()) * bs
            tr_set += float(losses["set"].item()) * bs
            tr_eos += float(losses["eos"].item()) * bs
            tr_cnt += float(losses["count"].item()) * bs
            tr_jpt += float(losses["jetpt"].item()) * bs
            tr_j4 += float(losses["fourvec"].item()) * bs
            tr_cfr += float(losses["conf_rank"].item()) * bs
            tr_cfp += float(losses["conf_prefix"].item()) * bs
            ntr += bs

        model.eval()
        va_tf_total = va_fr_total = 0.0
        va_tf_set = va_tf_eos = va_tf_cnt = va_tf_jpt = va_tf_j4 = va_tf_cfr = va_tf_cfp = 0.0
        nva = 0
        with torch.no_grad():
            for batch in val_loader:
                feat_hlt = batch["feat_hlt"].to(device)
                mask_hlt = batch["mask_hlt"].to(device)
                const_hlt = batch["const_hlt"].to(device)
                tgt_tok_cont = batch["tgt_tok"].to(device)
                tgt_mask_cont = batch["tgt_mask"].to(device)
                pref_tok = batch["prefix_tok"].to(device)
                pref_len = batch["prefix_len"].to(device)

                cont_steps = int(max(tgt_mask_cont.float().sum(dim=1).max().item(), 1))
                tgt_tok_trim = tgt_tok_cont[:, :cont_steps, :]
                tgt_mask_trim = tgt_mask_cont[:, :cont_steps]
                if token_quantizer is not None:
                    tgt_tok_trim = _quantize_token5_torch(tgt_tok_trim, token_quantizer)
                t_full = int(pref_tok.shape[1] + cont_steps)

                full_tok = torch.zeros((feat_hlt.shape[0], t_full, 5), dtype=tgt_tok_cont.dtype, device=device)
                valid_prev_mask = torch.zeros((feat_hlt.shape[0], t_full), dtype=torch.bool, device=device)
                for ib in range(int(feat_hlt.shape[0])):
                    pp = int(min(pref_len[ib].item(), pref_tok.shape[1]))
                    cc = int(tgt_mask_trim[ib].sum().item())
                    if pp > 0:
                        full_tok[ib, :pp] = pref_tok[ib, :pp]
                    if cc > 0:
                        full_tok[ib, pp : pp + cc] = tgt_tok_trim[ib, :cc]
                        valid_prev_mask[ib, pp : pp + cc] = True

                out_first = model.forward_teacher(feat_hlt, mask_hlt, const_hlt, full_tok)
                prev_tok = out_first["pred_tok"]
                if token_quantizer is not None:
                    prev_tok = _quantize_token5_torch(prev_tok, token_quantizer)
                dec_in = m28.build_decoder_input_tokens(
                    model.bos_token,
                    full_tok,
                    model_prev_tokens=prev_tok,
                    model_prob=float(ss_prob),
                    valid_prev_mask=valid_prev_mask,
                )
                out_full = model.forward_teacher_with_inputs(feat_hlt, mask_hlt, const_hlt, dec_in)
                out_cont = _gather_contiguous_from_full(out_full, pref_len.long(), cont_steps)
                if token_quantizer is not None:
                    out_cont = dict(out_cont)
                    out_cont["pred_tok"] = _quantize_token5_torch(out_cont["pred_tok"], token_quantizer)
                losses_tf = m28.compute_reco_losses(out_cont, tgt_tok_trim, tgt_mask_trim, loss_cfg, physics_scale=physics_scale)

                out_fr_full = _forward_free_running_forced_prefix_single(
                    model=model,
                    feat_hlt=feat_hlt,
                    mask_hlt=mask_hlt,
                    const_hlt=const_hlt,
                    prefix_tok=pref_tok,
                    prefix_len=pref_len.long(),
                    max_steps=t_full,
                    token_quantizer=token_quantizer,
                )
                out_fr_cont = _gather_contiguous_from_full(out_fr_full, pref_len.long(), cont_steps)
                if token_quantizer is not None:
                    out_fr_cont = dict(out_fr_cont)
                    out_fr_cont["pred_tok"] = _quantize_token5_torch(out_fr_cont["pred_tok"], token_quantizer)
                losses_fr = m28.compute_reco_losses(out_fr_cont, tgt_tok_trim, tgt_mask_trim, loss_cfg, physics_scale=physics_scale)

                bs = int(feat_hlt.shape[0])
                va_tf_total += float(losses_tf["total"].item()) * bs
                va_fr_total += float(losses_fr["total"].item()) * bs
                va_tf_set += float(losses_tf["set"].item()) * bs
                va_tf_eos += float(losses_tf["eos"].item()) * bs
                va_tf_cnt += float(losses_tf["count"].item()) * bs
                va_tf_jpt += float(losses_tf["jetpt"].item()) * bs
                va_tf_j4 += float(losses_tf["fourvec"].item()) * bs
                va_tf_cfr += float(losses_tf["conf_rank"].item()) * bs
                va_tf_cfp += float(losses_tf["conf_prefix"].item()) * bs
                nva += bs

        tr_total_m = tr_total / max(1, ntr)
        va_tf_m = va_tf_total / max(1, nva)
        va_fr_m = va_fr_total / max(1, nva)

        if va_fr_m < phase_best_val:
            phase_best_val = va_fr_m
            phase_best_ep = ep + 1
            phase_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        improved = va_fr_m < best_val
        if improved:
            best_val = va_fr_m
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        ep_dt = float(time.perf_counter() - t0)
        ep_times.append(ep_dt)
        mean_dt = float(np.mean(ep_times[-10:])) if ep_times else ep_dt
        eta_s = max(total_epochs - (ep + 1), 0) * mean_dt
        print(
            f"RecoSpec ep {ep+1:03d} | t={ep_dt/60.0:05.2f}m eta={eta_s/3600.0:05.2f}h | "
            f"phase={phase_idx}:{phase_name} ss={ss_prob:.2f} mixFR={fr_mix_alpha:.2f}@{fr_every_n} | "
            f"train total={tr_total_m:.5f} set={tr_set/max(1,ntr):.5f} eos={tr_eos/max(1,ntr):.5f} "
            f"cnt={tr_cnt/max(1,ntr):.5f} jpt={tr_jpt/max(1,ntr):.5f} j4={tr_j4/max(1,ntr):.5f} "
            f"crk={tr_cfr/max(1,ntr):.5f} cpre={tr_cfp/max(1,ntr):.5f} fr={tr_fr/max(1,ntr):.5f} | "
            f"valTF total={va_tf_m:.5f} set={va_tf_set/max(1,nva):.5f} eos={va_tf_eos/max(1,nva):.5f} "
            f"cnt={va_tf_cnt/max(1,nva):.5f} jpt={va_tf_jpt/max(1,nva):.5f} j4={va_tf_j4/max(1,nva):.5f} "
            f"crk={va_tf_cfr/max(1,nva):.5f} cpre={va_tf_cfp/max(1,nva):.5f} | valFR total={va_fr_m:.5f}"
        )

        if (ep + 1) >= min_epochs and no_improve >= patience:
            print(f"Early stopping RecoSpecialist at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_fr_total": float(best_val),
        "best_epoch": int(best_ep),
    }


class DualViewM38Dataset(Dataset):
    """Numpy-backed dataset to avoid duplicating large candidate tensors in memory."""

    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        cand_tokens: np.ndarray,
        cand_masks: np.ndarray,
        cand_meta: np.ndarray,
        summary_feat: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray,
    ):
        self.feat_hlt = feat_hlt.astype(np.float32, copy=False)
        self.mask_hlt = mask_hlt.astype(bool, copy=False)
        self.cand_tokens = cand_tokens.astype(np.float32, copy=False)
        self.cand_masks = cand_masks.astype(bool, copy=False)
        self.cand_meta = cand_meta.astype(np.float32, copy=False)
        self.summary_feat = summary_feat.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.sample_weight = sample_weight.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": torch.from_numpy(self.feat_hlt[i]),
            "mask_hlt": torch.from_numpy(self.mask_hlt[i]),
            "cand_tokens": torch.from_numpy(self.cand_tokens[i]),
            "cand_masks": torch.from_numpy(self.cand_masks[i]),
            "cand_meta": torch.from_numpy(self.cand_meta[i]),
            "summary_feat": torch.from_numpy(self.summary_feat[i]),
            "label": torch.tensor(self.labels[i], dtype=torch.float32),
            "sample_weight": torch.tensor(self.sample_weight[i], dtype=torch.float32),
        }


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class CarryoverTokenPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 384,
        dropout: float = 0.10,
        max_tokens: int = 100,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(embed_dim)),
            nn.LayerNorm(int(embed_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.pos = nn.Parameter(torch.zeros(1, int(max_tokens), int(embed_dim)))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.head = nn.Linear(int(embed_dim), 1)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, feat_hlt: torch.Tensor, mask_hlt: torch.Tensor) -> torch.Tensor:
        b, l, _ = feat_hlt.shape
        x = self.in_proj(feat_hlt) + self.pos[:, :l, :]
        h = self.encoder(x, src_key_padding_mask=~mask_hlt)
        return self.head(h).squeeze(-1)


class MultiCandidateM38NoGate(nn.Module):
    def __init__(
        self,
        cand_meta_dim: int,
        summary_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hlt_encoder = ParticleTransformer(
            input_dim=7,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.cand_encoder = ParticleTransformer(
            input_dim=12,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=max(2, num_layers // 2),
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(int(cand_meta_dim), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(int(summary_dim), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_score = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.cand_value = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.embed_dim = int(embed_dim)

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_tokens: torch.Tensor,
        cand_masks: torch.Tensor,
        cand_meta: torch.Tensor,
        summary_feat: torch.Tensor,
    ) -> torch.Tensor:
        _, hlt_z = self.hlt_encoder(feat_hlt, mask_hlt, return_embedding=True)
        b, m, l, d = cand_tokens.shape
        cand_tok = cand_tokens.reshape(b * m, l, d)
        cand_m = cand_masks.reshape(b * m, l)
        _, cand_z_flat = self.cand_encoder(cand_tok, cand_m, return_embedding=True)
        cand_z = cand_z_flat.reshape(b, m, self.embed_dim)

        meta_z = self.meta_proj(cand_meta)
        hz = hlt_z.unsqueeze(1).expand(b, m, self.embed_dim)
        score_in = torch.cat([cand_z, meta_z, hz], dim=-1)
        attn = torch.softmax(self.attn_score(score_in).squeeze(-1), dim=1)

        value = self.cand_value(torch.cat([cand_z, meta_z], dim=-1))
        agg = (attn.unsqueeze(-1) * value).sum(dim=1)
        top1 = value[:, 0, :]
        s = self.summary_proj(summary_feat)

        x = torch.cat([hlt_z, agg, top1, s], dim=-1)
        return self.fuse(x).squeeze(-1)


class MultiCandidateM38Gated(nn.Module):
    def __init__(
        self,
        cand_meta_dim: int,
        summary_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hlt_encoder = ParticleTransformer(
            input_dim=7,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.cand_encoder = ParticleTransformer(
            input_dim=12,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=max(2, num_layers // 2),
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(int(cand_meta_dim), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(int(summary_dim), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_score = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.cand_value = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
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
            nn.Linear(embed_dim * 3, 256),
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
            nn.Linear(embed_dim * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.embed_dim = int(embed_dim)

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        cand_tokens: torch.Tensor,
        cand_masks: torch.Tensor,
        cand_meta: torch.Tensor,
        summary_feat: torch.Tensor,
    ) -> torch.Tensor:
        _, hlt_z = self.hlt_encoder(feat_hlt, mask_hlt, return_embedding=True)
        b, m, l, d = cand_tokens.shape
        cand_tok = cand_tokens.reshape(b * m, l, d)
        cand_m = cand_masks.reshape(b * m, l)
        _, cand_z_flat = self.cand_encoder(cand_tok, cand_m, return_embedding=True)
        cand_z = cand_z_flat.reshape(b, m, self.embed_dim)

        meta_z = self.meta_proj(cand_meta)
        hz = hlt_z.unsqueeze(1).expand(b, m, self.embed_dim)
        score_in = torch.cat([cand_z, meta_z, hz], dim=-1)
        attn = torch.softmax(self.attn_score(score_in).squeeze(-1), dim=1)

        value = self.cand_value(torch.cat([cand_z, meta_z], dim=-1))
        agg = (attn.unsqueeze(-1) * value).sum(dim=1)
        top1 = value[:, 0, :]
        s = self.summary_proj(summary_feat)

        cand_pack = torch.cat([agg, top1, s], dim=-1)
        logit_h = self.hlt_head(hlt_z).squeeze(-1)
        logit_c = self.cand_head(cand_pack).squeeze(-1)
        g = self.gate(torch.cat([hlt_z, cand_pack], dim=-1)).squeeze(-1)
        return (1.0 - g) * logit_h + g * logit_c


# -----------------------------------------------------------------------------
# Training / eval helpers
# -----------------------------------------------------------------------------


def _forward_m38(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        feat_hlt=batch["feat_hlt"],
        mask_hlt=batch["mask_hlt"],
        cand_tokens=batch["cand_tokens"],
        cand_masks=batch["cand_masks"],
        cand_meta=batch["cand_meta"],
        summary_feat=batch["summary_feat"],
    )


def _train_m38_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    name: str,
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
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch["label"]
            sw = batch["sample_weight"]
            logit = _forward_m38(model, batch)
            lv = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            loss = m33._weighted_mean(lv, sw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            bs = int(y.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_n += bs

        model.eval()
        vp, vy, vw = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                y = batch["label"]
                sw = batch["sample_weight"]
                p = torch.sigmoid(_forward_m38(model, batch))
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
            print(
                f"{name} ep {ep+1:03d}: train_loss={tr_loss/max(1,tr_n):.5f} "
                f"val_auc={va_auc:.4f} best={best_auc:.4f}@{best_epoch}"
            )

        if no_imp >= int(patience):
            print(f"Early stopping {name} at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
    }


@torch.no_grad()
def _eval_m38_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pp, yy, ww = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["label"]
        sw = batch["sample_weight"]
        p = torch.sigmoid(_forward_m38(model, batch))
        pp.append(p.detach().cpu().numpy().astype(np.float64))
        yy.append(y.detach().cpu().numpy().astype(np.float64))
        ww.append(sw.detach().cpu().numpy().astype(np.float64))
    p_np = np.concatenate(pp, axis=0)
    y_np = np.concatenate(yy, axis=0)
    w_np = np.concatenate(ww, axis=0)
    auc = float(roc_auc_score(y_np, p_np, sample_weight=w_np)) if len(np.unique(y_np)) > 1 else 0.0
    fpr, tpr, _ = roc_curve(y_np, p_np, sample_weight=w_np)
    fpr50 = float(m33.fpr_at_target_tpr(fpr, tpr, 0.50))
    return auc, fpr50, p_np.astype(np.float32), y_np.astype(np.float32), w_np.astype(np.float32)


def _carry_topk_metrics(
    prob: np.ndarray,
    tgt: np.ndarray,
    mask: np.ndarray,
    topks: Tuple[int, ...] = (2, 5, 7, 10, 12),
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = int(prob.shape[0])
    for k in topks:
        k_int = int(max(1, k))
        rec_sum = 0.0
        rec_den = 0
        prec_sum = 0.0
        prec_den = 0
        for i in range(n):
            ids = np.where(mask[i])[0]
            if ids.size == 0:
                continue
            p = prob[i, ids]
            y = (tgt[i, ids] > 0.5)
            kk = int(min(k_int, ids.size))
            if kk <= 0:
                continue
            top_loc = np.argpartition(-p, kth=kk - 1)[:kk]
            tp = int(y[top_loc].sum())
            prec_sum += float(tp) / float(kk)
            prec_den += 1
            pos = int(y.sum())
            if pos > 0:
                rec_sum += float(tp) / float(pos)
                rec_den += 1
        out[f"precision_at_{k_int}"] = float(prec_sum / max(1, prec_den))
        out[f"recall_at_{k_int}"] = float(rec_sum / max(1, rec_den))
    return out


def _train_carry_predictor(
    model: CarryoverTokenPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    pos_weight: float,
    lr_decay_start_epoch: int = 0,
    lr_decay_gamma: float = 1.0,
    min_lr_ratio: float = 0.30,
) -> Tuple[CarryoverTokenPredictor, Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    pos_w = torch.tensor(float(max(pos_weight, 1e-3)), dtype=torch.float32, device=device)
    best_auc = float("-inf")
    best_state = None
    best_epoch = 0
    best_topk_metrics: Dict[str, float] = {}
    no_imp = 0

    for ep in range(int(epochs)):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for batch in train_loader:
            feat = batch["feat_hlt"].to(device)
            mask = batch["mask_hlt"].to(device)
            tgt = batch["carry_tgt"].to(device)
            logit = model(feat, mask)
            lv = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none", pos_weight=pos_w)
            lv = lv * mask.float()
            loss = lv.sum() / mask.float().sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            bs = int(feat.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_n += bs

        model.eval()
        p_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        p_jet: List[np.ndarray] = []
        y_jet: List[np.ndarray] = []
        m_jet: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["feat_hlt"].to(device)
                mask = batch["mask_hlt"].to(device)
                tgt = batch["carry_tgt"].to(device)
                p = torch.sigmoid(model(feat, mask))
                p_all.append(p[mask].detach().cpu().numpy().astype(np.float64))
                y_all.append(tgt[mask].detach().cpu().numpy().astype(np.float64))
                p_jet.append(p.detach().cpu().numpy().astype(np.float32))
                y_jet.append(tgt.detach().cpu().numpy().astype(np.float32))
                m_jet.append(mask.detach().cpu().numpy().astype(bool))

        p_np = np.concatenate(p_all, axis=0) if p_all else np.array([], dtype=np.float64)
        y_np = np.concatenate(y_all, axis=0) if y_all else np.array([], dtype=np.float64)
        p_jet_np = np.concatenate(p_jet, axis=0) if p_jet else np.zeros((0, 0), dtype=np.float32)
        y_jet_np = np.concatenate(y_jet, axis=0) if y_jet else np.zeros((0, 0), dtype=np.float32)
        m_jet_np = np.concatenate(m_jet, axis=0) if m_jet else np.zeros((0, 0), dtype=bool)
        if len(np.unique(y_np)) > 1:
            va_auc = float(roc_auc_score(y_np, p_np))
        else:
            va_auc = 0.0
        topk_metrics = _carry_topk_metrics(p_jet_np, y_jet_np, m_jet_np, topks=(2, 5, 7, 10, 12))

        if va_auc > best_auc:
            best_auc = float(va_auc)
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_topk_metrics = dict(topk_metrics)
            no_imp = 0
        else:
            no_imp += 1

        if int(lr_decay_start_epoch) > 0 and (ep + 1) >= int(lr_decay_start_epoch) and float(lr_decay_gamma) < 1.0:
            floor_lr = float(lr) * float(max(min_lr_ratio, 1e-4))
            for pg in opt.param_groups:
                pg["lr"] = max(float(pg["lr"]) * float(lr_decay_gamma), floor_lr)
        curr_lr = float(opt.param_groups[0]["lr"])

        if (ep + 1) % 2 == 0 or ep == 0:
            print(
                f"CarryPredictor ep {ep+1:03d}: train_loss={tr_loss/max(1,tr_n):.5f} "
                f"val_auc={va_auc:.4f} r@12={topk_metrics.get('recall_at_12', 0.0):.4f} "
                f"p@12={topk_metrics.get('precision_at_12', 0.0):.4f} "
                f"best={best_auc:.4f}@{best_epoch} lr={curr_lr:.3e}"
            )

        if no_imp >= int(patience):
            print(f"Early stopping CarryPredictor at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "best_topk_metrics": best_topk_metrics,
        "lr_decay_start_epoch": int(lr_decay_start_epoch),
        "lr_decay_gamma": float(lr_decay_gamma),
        "min_lr_ratio": float(min_lr_ratio),
    }


@torch.no_grad()
def _predict_carry_probs(
    model: CarryoverTokenPredictor,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    ds = FeatMaskDataset(feat_hlt, mask_hlt)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    out = []
    for batch in dl:
        feat = batch["feat"].to(device)
        mask = batch["mask"].to(device)
        p = torch.sigmoid(model(feat, mask)).detach().cpu().numpy().astype(np.float32)
        p[~mask.detach().cpu().numpy()] = 0.0
        out.append(p)
    return np.concatenate(out, axis=0)


# -----------------------------------------------------------------------------
# Carry target + seeded generation
# -----------------------------------------------------------------------------


def _build_carry_targets_np(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    dist_thresh: float,
    batch_size: int,
) -> np.ndarray:
    """Token-level target: HLT token has a near-identical Offline token in token-5 space."""
    n, l, _ = const_hlt.shape
    off_tok = _const_to_token5_np(const_off, mask_off)
    hlt_tok = _const_to_token5_np(const_hlt, mask_hlt)
    tgt = np.zeros((n, l), dtype=np.float32)

    for s in range(0, n, int(batch_size)):
        e = min(s + int(batch_size), n)
        h = torch.tensor(hlt_tok[s:e], dtype=torch.float32)
        o = torch.tensor(off_tok[s:e], dtype=torch.float32)
        mh = torch.tensor(mask_hlt[s:e], dtype=torch.bool)
        mo = torch.tensor(mask_off[s:e], dtype=torch.bool)

        d = torch.cdist(h, o, p=2)  # [B,L,L]
        valid = mh.unsqueeze(2) & mo.unsqueeze(1)
        d = torch.where(valid, d, torch.full_like(d, 1e6))
        md = d.min(dim=2).values
        t = (md < float(dist_thresh)) & mh
        tgt[s:e] = t.cpu().numpy().astype(np.float32)

    return tgt


def _build_carry_targets_fixed_k_np(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    k_target: int,
    dist_thresh: float,
    batch_size: int,
    thresh_gate: bool = False,
) -> np.ndarray:
    """
    Token-level fixed-k target:
      mark exactly top-k HLT tokens (by nearest Offline-token distance) as carryovers.
      Optionally apply dist threshold gate before picking top-k.
    """
    n, l, _ = const_hlt.shape
    tgt = np.zeros((n, l), dtype=np.float32)
    k_target = int(max(0, k_target))
    if k_target <= 0:
        return tgt

    off_tok = _const_to_token5_np(const_off, mask_off)
    hlt_tok = _const_to_token5_np(const_hlt, mask_hlt)

    for s in range(0, n, int(batch_size)):
        e = min(s + int(batch_size), n)
        h = torch.tensor(hlt_tok[s:e], dtype=torch.float32)
        o = torch.tensor(off_tok[s:e], dtype=torch.float32)
        mh = torch.tensor(mask_hlt[s:e], dtype=torch.bool)
        mo = torch.tensor(mask_off[s:e], dtype=torch.bool)

        d = torch.cdist(h, o, p=2)  # [B,L,L]
        valid = mh.unsqueeze(2) & mo.unsqueeze(1)
        d = torch.where(valid, d, torch.full_like(d, 1e6))
        md = d.min(dim=2).values.cpu().numpy().astype(np.float32)
        mh_np = mh.cpu().numpy().astype(bool)

        for bi in range(e - s):
            ids = np.where(mh_np[bi])[0]
            if ids.size == 0:
                continue
            dvals = md[bi, ids]
            cand_ids = ids
            cand_d = dvals
            if bool(thresh_gate):
                keep = dvals < float(dist_thresh)
                if np.any(keep):
                    cand_ids = ids[keep]
                    cand_d = dvals[keep]
            kk = int(min(k_target, cand_ids.size))
            if kk <= 0:
                continue
            pick_local = np.argpartition(cand_d, kk - 1)[:kk]
            pick_ids = cand_ids[pick_local]
            tgt[s + bi, pick_ids] = 1.0

    return tgt


def _build_prefix_schedule(k: int, max_prefix: int) -> List[int]:
    if k <= 1:
        return [int(max(0, max_prefix))]
    vals = np.linspace(0, int(max_prefix), int(k))
    out = [int(np.round(v)) for v in vals]
    out = [int(np.clip(v, 0, int(max_prefix))) for v in out]
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _prepare_prefixes_batch(
    carry_probs: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    schedule: List[int],
    seed_temp: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b, l = carry_probs.shape
    k = len(schedule)
    pmax = int(max(1, max(schedule)))

    hlt_tok = _const_to_token5_np(const_hlt, mask_hlt)
    prefix_tok = np.zeros((b, k, pmax, 5), dtype=np.float32)
    prefix_len = np.zeros((b, k), dtype=np.int64)
    prefix_carry_mean = np.zeros((b, k), dtype=np.float32)
    prefix_carry_min = np.zeros((b, k), dtype=np.float32)

    pt = const_hlt[..., 0]
    valid_ids_cache = [np.where(mask_hlt[i])[0] for i in range(b)]

    for i in range(b):
        ids = valid_ids_cache[i]
        if ids.size == 0:
            continue
        base = carry_probs[i, ids]
        for kk, n_pref in enumerate(schedule):
            nn = int(min(max(0, n_pref), ids.size, pmax))
            if nn <= 0:
                prefix_len[i, kk] = 0
                continue
            noise = rng.gumbel(loc=0.0, scale=1.0, size=ids.size).astype(np.float32)
            score = base + float(seed_temp) * (1.0 + 0.05 * kk) * noise
            pick_local = np.argpartition(-score, kth=nn - 1)[:nn]
            pick_ids = ids[pick_local]
            ord_pt = np.argsort(-pt[i, pick_ids])
            pick_ids = pick_ids[ord_pt]

            prefix_len[i, kk] = nn
            prefix_tok[i, kk, :nn] = hlt_tok[i, pick_ids]
            cvals = carry_probs[i, pick_ids]
            prefix_carry_mean[i, kk] = float(np.mean(cvals)) if cvals.size > 0 else 0.0
            prefix_carry_min[i, kk] = float(np.min(cvals)) if cvals.size > 0 else 0.0

    return prefix_tok, prefix_len, prefix_carry_mean, prefix_carry_min


@torch.no_grad()
def _decode_forced_prefix_multi(
    model: m28.HLT2OfflineSeq2Seq,
    feat_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_hlt: torch.Tensor,
    prefix_tok_bkpd: torch.Tensor,
    prefix_len_bk: torch.Tensor,
    max_steps: int,
    token_quantizer: Optional[TokenCodebookQuantizer] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forced-prefix autoregressive decoding.

    Returns:
      pred_const_bkt4, pred_mask_bkt, conf_mean_bk, stop_len_bk
    """
    b = int(feat_hlt.shape[0])
    k = int(prefix_tok_bkpd.shape[1])
    t = int(max_steps)

    mem, hlt_tok, count_pred = model.encode(feat_hlt, mask_hlt, const_hlt)
    mem_pad = ~mask_hlt

    all_const = []
    all_mask = []
    all_conf_mean = []
    all_len = []

    for hk in range(k):
        in_tok = model.bos_token.expand(b, 1, model.token_dim)
        n_layers = len(model.decoder.layers)
        layer_cache: List[torch.Tensor | None] = [None] * n_layers

        pred_tok_seq = []
        stop_seq = []
        conf_seq = []

        pref_len = prefix_len_bk[:, hk]
        pref_tok = prefix_tok_bkpd[:, hk]

        for step in range(t):
            x_step = model.dec_in(in_tok) + model.dec_pos[:, step : step + 1, :]
            h_last, layer_cache = model._decoder_step_cached(x_step, mem, mem_pad, layer_cache)
            pred_tok, stop_logits, conf_logits, _attn, _gate = model._predict_from_hidden(
                h_last,
                mem,
                mask_hlt,
                hlt_tok,
                hyp_idx=(hk % int(model.num_hypotheses)),
            )

            next_tok = pred_tok[:, 0, :]
            force_mask = pref_len > step
            if bool(force_mask.any()):
                forced = pref_tok[:, step, :]
                next_tok = torch.where(force_mask.unsqueeze(1), forced, next_tok)
            if token_quantizer is not None:
                next_tok = _quantize_token5_torch(next_tok, token_quantizer)

            pred_tok_seq.append(next_tok)
            stop_seq.append(torch.sigmoid(stop_logits[:, 0]))
            conf_seq.append(torch.sigmoid(conf_logits[:, 0]))
            in_tok = next_tok.unsqueeze(1)

        pred_tok_full = torch.stack(pred_tok_seq, dim=1)  # [B,T,5]
        stop_prob = torch.stack(stop_seq, dim=1)          # [B,T]
        conf_prob = torch.stack(conf_seq, dim=1)          # [B,T]

        pred_const = m28.token_to_const_torch(pred_tok_full)

        cp = torch.clamp(torch.round(count_pred), min=0, max=t).long()
        stop_pos = (stop_prob > 0.5).float()
        first_stop = torch.where(
            stop_pos.any(dim=1),
            stop_pos.argmax(dim=1),
            cp,
        )
        out_len = torch.maximum(first_stop.long(), pref_len.long())
        out_len = torch.clamp(out_len, min=0, max=t)

        steps = torch.arange(t, device=feat_hlt.device).view(1, -1)
        out_mask = steps < out_len.unsqueeze(1)

        conf_mean = (conf_prob * out_mask.float()).sum(dim=1) / out_mask.float().sum(dim=1).clamp(min=1.0)

        # sort valid part by pT descending for classifier consistency
        pred_np = pred_const.detach().cpu().numpy().astype(np.float32)
        mask_np = out_mask.detach().cpu().numpy().astype(bool)
        conf_np = conf_mean.detach().cpu().numpy().astype(np.float32)
        len_np = out_len.detach().cpu().numpy().astype(np.int64)
        for i in range(b):
            ll = int(len_np[i])
            if ll > 0:
                ord_i = np.argsort(-pred_np[i, :ll, 0])
                pred_np[i, :ll] = pred_np[i, :ll][ord_i]

        pred_t = torch.tensor(pred_np, dtype=torch.float32, device=feat_hlt.device)
        mask_t = torch.tensor(mask_np, dtype=torch.bool, device=feat_hlt.device)

        all_const.append(pred_t)
        all_mask.append(mask_t)
        all_conf_mean.append(torch.tensor(conf_np, dtype=torch.float32, device=feat_hlt.device))
        all_len.append(torch.tensor(len_np, dtype=torch.float32, device=feat_hlt.device))

    pred_const_bkt4 = torch.stack(all_const, dim=1)
    pred_mask_bkt = torch.stack(all_mask, dim=1)
    conf_mean_bk = torch.stack(all_conf_mean, dim=1)
    stop_len_bk = torch.stack(all_len, dim=1)
    pred_const_bkt4 = torch.nan_to_num(pred_const_bkt4, nan=0.0, posinf=0.0, neginf=0.0)
    pred_const_bkt4 = torch.where(pred_mask_bkt.unsqueeze(-1), pred_const_bkt4, torch.zeros_like(pred_const_bkt4))
    return pred_const_bkt4, pred_mask_bkt, conf_mean_bk, stop_len_bk


@dataclass
class CandidateSplitOutput:
    off_const: np.ndarray
    off_mask: np.ndarray
    hlt_const: np.ndarray
    hlt_mask: np.ndarray
    res_total: np.ndarray
    res_set: np.ndarray
    res_count: np.ndarray
    res_pt: np.ndarray
    res_mass: np.ndarray
    feasible: np.ndarray
    prefix_len: np.ndarray
    prefix_carry_mean: np.ndarray
    prefix_carry_min: np.ndarray
    conf_mean: np.ndarray
    stop_len: np.ndarray
    rank_norm: np.ndarray
    feasible_count_all: np.ndarray
    best_residual_all: np.ndarray


@torch.no_grad()
def _generate_candidates_split(
    reco_model: m28.HLT2OfflineSeq2Seq,
    carry_model: CarryoverTokenPredictor,
    feat_hlt: np.ndarray,
    const_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    jet_keys: np.ndarray,
    cfg: Dict,
    seed_offset: int,
    candidate_k: int,
    keep_m: int,
    max_prefix: int,
    seed_temp: float,
    eps_total: float,
    eps_count: float,
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    batch_size: int,
    device: torch.device,
    seed: int,
    token_quantizer: Optional[TokenCodebookQuantizer] = None,
) -> CandidateSplitOutput:
    reco_model.eval()
    carry_model.eval()

    n, t, _ = const_hlt.shape
    k = int(candidate_k)
    m = int(keep_m)

    schedule = _build_prefix_schedule(k=k, max_prefix=max_prefix)
    rng = np.random.default_rng(int(seed))

    off_const_out = np.zeros((n, m, t, 4), dtype=np.float32)
    off_mask_out = np.zeros((n, m, t), dtype=bool)
    hlt_const_out = np.zeros((n, m, t, 4), dtype=np.float32)
    hlt_mask_out = np.zeros((n, m, t), dtype=bool)

    res_total_out = np.full((n, m), np.inf, dtype=np.float32)
    res_set_out = np.full((n, m), np.inf, dtype=np.float32)
    res_count_out = np.full((n, m), np.inf, dtype=np.float32)
    res_pt_out = np.full((n, m), np.inf, dtype=np.float32)
    res_mass_out = np.full((n, m), np.inf, dtype=np.float32)
    feasible_out = np.zeros((n, m), dtype=np.float32)

    prefix_len_out = np.zeros((n, m), dtype=np.float32)
    prefix_mean_out = np.zeros((n, m), dtype=np.float32)
    prefix_min_out = np.zeros((n, m), dtype=np.float32)
    conf_mean_out = np.zeros((n, m), dtype=np.float32)
    stop_len_out = np.zeros((n, m), dtype=np.float32)
    rank_norm_out = np.zeros((n, m), dtype=np.float32)

    feasible_count_all = np.zeros((n,), dtype=np.float32)
    best_res_all = np.full((n,), np.inf, dtype=np.float32)

    ds = FeatMaskDataset(feat_hlt, mask_hlt)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    offset = 0
    for batch in dl:
        feat_b = batch["feat"].to(device)
        mask_b = batch["mask"].to(device)
        bsz = int(feat_b.shape[0])

        const_b_np = const_hlt[offset : offset + bsz]
        mask_b_np = mask_hlt[offset : offset + bsz]
        keys_b = jet_keys[offset : offset + bsz]

        const_b = torch.tensor(const_b_np, dtype=torch.float32, device=device)

        carry_probs_t = torch.sigmoid(carry_model(feat_b, mask_b))
        carry_probs_np = carry_probs_t.detach().cpu().numpy().astype(np.float32)
        carry_probs_np[~mask_b_np] = 0.0

        pref_tok_np, pref_len_np, pref_mean_np, pref_min_np = _prepare_prefixes_batch(
            carry_probs=carry_probs_np,
            const_hlt=const_b_np,
            mask_hlt=mask_b_np,
            schedule=schedule,
            seed_temp=float(seed_temp),
            rng=rng,
        )
        if token_quantizer is not None:
            pref_tok_np = _quantize_token5_np(pref_tok_np, token_quantizer)
        pref_tok_t = torch.tensor(pref_tok_np, dtype=torch.float32, device=device)
        pref_len_t = torch.tensor(pref_len_np, dtype=torch.long, device=device)

        pred_const_bkt4, pred_mask_bkt, conf_mean_bk, stop_len_bk = _decode_forced_prefix_multi(
            model=reco_model,
            feat_hlt=feat_b,
            mask_hlt=mask_b,
            const_hlt=const_b,
            prefix_tok_bkpd=pref_tok_t,
            prefix_len_bk=pref_len_t,
            max_steps=int(t),
            token_quantizer=token_quantizer,
        )

        pred_const_np = pred_const_bkt4.detach().cpu().numpy().astype(np.float32)
        pred_mask_np = pred_mask_bkt.detach().cpu().numpy().astype(bool)
        conf_mean_np = conf_mean_bk.detach().cpu().numpy().astype(np.float32)
        stop_len_np = stop_len_bk.detach().cpu().numpy().astype(np.float32)

        flat_const = pred_const_np.reshape(bsz * k, t, 4)
        flat_mask = pred_mask_np.reshape(bsz * k, t)
        flat_keys = np.repeat(keys_b, k).astype(np.int64)

        hlt_pred_const, hlt_pred_mask, _ = m33._apply_hlt_effects_deterministic_keyed(
            const=flat_const,
            mask=flat_mask,
            cfg=cfg,
            jet_keys=flat_keys,
            base_seed=int(seed_offset),
        )

        tgt_const = np.repeat(const_b_np[:, None, :, :], k, axis=1).reshape(bsz * k, t, 4)
        tgt_mask = np.repeat(mask_b_np[:, None, :], k, axis=1).reshape(bsz * k, t)

        resid = m33._residual_fast_vec(
            pred_const=torch.tensor(hlt_pred_const, dtype=torch.float32),
            pred_mask=torch.tensor(hlt_pred_mask, dtype=torch.bool),
            tgt_const=torch.tensor(tgt_const, dtype=torch.float32),
            tgt_mask=torch.tensor(tgt_mask, dtype=torch.bool),
            w_chamfer=float(w_chamfer),
            w_count=float(w_count),
            w_pt=float(w_pt),
            w_mass=float(w_mass),
        )

        r_tot = resid["total"].cpu().numpy().reshape(bsz, k).astype(np.float32)
        r_set = resid["set"].cpu().numpy().reshape(bsz, k).astype(np.float32)
        r_cnt = resid["count"].cpu().numpy().reshape(bsz, k).astype(np.float32)
        r_pt = resid["pt"].cpu().numpy().reshape(bsz, k).astype(np.float32)
        r_mass = resid["mass"].cpu().numpy().reshape(bsz, k).astype(np.float32)

        hlt_pred_const_b = hlt_pred_const.reshape(bsz, k, t, 4)
        hlt_pred_mask_b = hlt_pred_mask.reshape(bsz, k, t)

        feas_all = (r_tot <= float(eps_total)) & (r_cnt <= float(eps_count))
        feasible_count_all[offset : offset + bsz] = feas_all.sum(axis=1).astype(np.float32)
        best_res_all[offset : offset + bsz] = np.min(r_tot, axis=1).astype(np.float32)

        order = np.argsort(r_tot, axis=1)
        pick = order[:, :m]

        for i in range(bsz):
            gi = offset + i
            for j in range(m):
                kj = int(pick[i, j])
                off_const_out[gi, j] = pred_const_np[i, kj]
                off_mask_out[gi, j] = pred_mask_np[i, kj]
                hlt_const_out[gi, j] = hlt_pred_const_b[i, kj]
                hlt_mask_out[gi, j] = hlt_pred_mask_b[i, kj]

                res_total_out[gi, j] = r_tot[i, kj]
                res_set_out[gi, j] = r_set[i, kj]
                res_count_out[gi, j] = r_cnt[i, kj]
                res_pt_out[gi, j] = r_pt[i, kj]
                res_mass_out[gi, j] = r_mass[i, kj]
                feasible_out[gi, j] = float(feas_all[i, kj])

                prefix_len_out[gi, j] = float(pref_len_np[i, kj])
                prefix_mean_out[gi, j] = float(pref_mean_np[i, kj])
                prefix_min_out[gi, j] = float(pref_min_np[i, kj])
                conf_mean_out[gi, j] = float(conf_mean_np[i, kj])
                stop_len_out[gi, j] = float(stop_len_np[i, kj])
                rank_norm_out[gi, j] = float(j) / float(max(1, m - 1))

        offset += bsz

    return CandidateSplitOutput(
        off_const=off_const_out,
        off_mask=off_mask_out,
        hlt_const=hlt_const_out,
        hlt_mask=hlt_mask_out,
        res_total=res_total_out,
        res_set=res_set_out,
        res_count=res_count_out,
        res_pt=res_pt_out,
        res_mass=res_mass_out,
        feasible=feasible_out,
        prefix_len=prefix_len_out,
        prefix_carry_mean=prefix_mean_out,
        prefix_carry_min=prefix_min_out,
        conf_mean=conf_mean_out,
        stop_len=stop_len_out,
        rank_norm=rank_norm_out,
        feasible_count_all=feasible_count_all,
        best_residual_all=best_res_all,
    )


# -----------------------------------------------------------------------------
# Candidate -> model arrays
# -----------------------------------------------------------------------------


def _build_m38_multicandidate_arrays(c: CandidateSplitOutput, max_constits: int, candidate_k: int) -> Dict[str, np.ndarray]:
    n, m, _, _ = c.off_const.shape

    cand_tokens = _build_candidate_token12_np(
        off_const=c.off_const,
        off_mask=c.off_mask,
        hlt_const=c.hlt_const,
        hlt_mask=c.hlt_mask,
    )
    cand_masks = (c.off_mask | c.hlt_mask).astype(bool)
    empty = ~cand_masks.any(axis=2)
    if np.any(empty):
        cand_masks[empty, 0] = True
    cand_tokens[~cand_masks] = 0.0

    prefix_len_norm = c.prefix_len / float(max(1, max_constits))
    stop_len_norm = c.stop_len / float(max(1, max_constits))

    cand_meta = np.stack(
        [
            c.res_total,
            c.res_set,
            c.res_count,
            c.res_pt,
            c.res_mass,
            c.feasible,
            prefix_len_norm,
            c.prefix_carry_mean,
            c.prefix_carry_min,
            c.conf_mean,
            stop_len_norm,
            c.rank_norm,
        ],
        axis=-1,
    ).astype(np.float32)

    best = c.res_total[:, 0]
    second = c.res_total[:, np.minimum(1, m - 1)]
    third = c.res_total[:, np.minimum(2, m - 1)]
    mean = c.res_total.mean(axis=1)
    std = c.res_total.std(axis=1)
    feasible_frac_topm = c.feasible.mean(axis=1)
    feasible_frac_all = np.clip(c.feasible_count_all / float(max(1, candidate_k)), 0.0, 1.0)
    pref_mean = c.prefix_len.mean(axis=1) / float(max(1, max_constits))
    carry_mean = c.prefix_carry_mean.mean(axis=1)
    conf_mean = c.conf_mean.mean(axis=1)
    gap12 = second - best
    gap23 = third - second

    summary_feat = np.stack(
        [
            best,
            second,
            third,
            mean,
            std,
            gap12,
            gap23,
            feasible_frac_topm.astype(np.float32),
            feasible_frac_all.astype(np.float32),
            pref_mean.astype(np.float32),
            carry_mean.astype(np.float32),
            conf_mean.astype(np.float32),
            c.best_residual_all.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "cand_tokens": cand_tokens.astype(np.float32),
        "cand_masks": cand_masks.astype(bool),
        "cand_meta": cand_meta.astype(np.float32),
        "summary_feat": summary_feat.astype(np.float32),
    }


# -----------------------------------------------------------------------------
# Parser / main
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m39 prefix-specialist m28 + deterministic residual multicandidate dualview")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model39_prefixspecialist_detresid_multicand")
    p.add_argument("--run_name", type=str, default="model39_prefixspecialist_detresid_multicand_150k75k300k_seed0")
    p.add_argument("--codebook_path", type=str, default="", help="Optional m40 codebook path (provenance hook).")
    p.add_argument("--codebook_label", type=str, default="", help="Optional m40 codebook label (provenance hook).")
    p.add_argument(
        "--step1_quantize_teacher_offline",
        action="store_true",
        help="Quantize Offline constituents with the provided codebook before STEP-1 Teacher feature building.",
    )

    p.add_argument("--n_train_jets", type=int, default=525000)
    p.add_argument("--n_train_split", type=int, default=150000)
    p.add_argument("--n_val_split", type=int, default=75000)
    p.add_argument("--n_test_split", type=int, default=300000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--use_train_weights", action="store_true")

    # deterministic D_hard
    p.add_argument("--merge_radius", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["merge_radius"]))
    p.add_argument("--eff_plateau_barrel", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"]))
    p.add_argument("--eff_plateau_endcap", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"]))
    p.add_argument("--smear_a", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_a"]))
    p.add_argument("--smear_b", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_b"]))
    p.add_argument("--smear_c", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_c"]))
    p.add_argument("--dhard_seed_offset", type=int, default=1337)

    # teacher / baseline
    p.add_argument("--cls_epochs", type=int, default=60)
    p.add_argument("--cls_patience", type=int, default=12)
    p.add_argument("--cls_lr", type=float, default=3e-4)
    p.add_argument("--cls_weight_decay", type=float, default=1e-4)
    p.add_argument("--cls_warmup_epochs", type=int, default=3)

    # carry predictor
    p.add_argument("--carry_epochs", type=int, default=24)
    p.add_argument("--carry_patience", type=int, default=6)
    p.add_argument("--carry_lr", type=float, default=2e-4)
    p.add_argument("--carry_weight_decay", type=float, default=1e-4)
    p.add_argument("--carry_embed_dim", type=int, default=128)
    p.add_argument("--carry_num_heads", type=int, default=4)
    p.add_argument("--carry_num_layers", type=int, default=3)
    p.add_argument("--carry_ff_dim", type=int, default=384)
    p.add_argument("--carry_dropout", type=float, default=0.10)
    p.add_argument("--carry_dist_thresh", type=float, default=0.22)
    p.add_argument("--carry_target_mode", type=str, choices=["threshold", "fixed_k"], default="threshold")
    p.add_argument("--carry_target_k", type=int, default=-1, help="Used in fixed_k mode; -1 => specialist prefix.")
    p.add_argument("--carry_target_thresh_gate", type=int, default=0, help="If 1 with fixed_k: gate candidates by carry_dist_thresh.")
    p.add_argument("--carry_lr_decay_start_epoch", type=int, default=0)
    p.add_argument("--carry_lr_decay_gamma", type=float, default=1.0)
    p.add_argument("--carry_min_lr_ratio", type=float, default=0.30)

    # m28 completer
    p.add_argument("--reco_batch_size", type=int, default=96)
    p.add_argument("--reco_epochs", type=int, default=140)
    p.add_argument("--reco_patience", type=int, default=20)
    p.add_argument("--reco_min_epochs", type=int, default=35)
    p.add_argument("--reco_lr", type=float, default=2e-4)
    p.add_argument("--reco_weight_decay", type=float, default=1e-5)
    p.add_argument("--reco_embed_dim", type=int, default=384)
    p.add_argument("--reco_num_heads", type=int, default=8)
    p.add_argument("--reco_num_enc_layers", type=int, default=6)
    p.add_argument("--reco_num_dec_layers", type=int, default=6)
    p.add_argument("--reco_ff_dim", type=int, default=1024)
    p.add_argument("--reco_dropout", type=float, default=0.10)
    p.add_argument("--reco_set_loss_mode", type=str, default="hungarian", choices=["chamfer", "hungarian", "sinkhorn"])
    p.add_argument("--reco_loss_w_eos", type=float, default=float(m28.LOSS_CFG["w_eos"]))
    p.add_argument("--reco_loss_w_count", type=float, default=float(m28.LOSS_CFG["w_count"]))
    p.add_argument("--reco_loss_w_jetpt", type=float, default=float(m28.LOSS_CFG["w_jetpt"]))
    p.add_argument("--reco_loss_w_4vec", type=float, default=float(m28.LOSS_CFG["w_4vec"]))
    p.add_argument("--reco_loss_w_conf_rank", type=float, default=float(m28.LOSS_CFG["w_conf_rank"]))
    p.add_argument("--reco_loss_w_conf_prefix", type=float, default=float(m28.LOSS_CFG["w_conf_prefix"]))
    p.add_argument("--reco_conf_margin", type=float, default=float(m28.LOSS_CFG["conf_margin"]))
    p.add_argument("--reco_conf_prefix_tau", type=float, default=float(m28.LOSS_CFG["prefix_tau"]))
    p.add_argument("--reco_physics_warmup_epochs", type=int, default=int(m28.LOSS_CFG["physics_warmup_epochs"]))
    p.add_argument("--reco_phase1_end_epoch", type=int, default=int(m28.LOSS_CFG["phase1_end_epoch"]))
    p.add_argument("--reco_phase2_end_epoch", type=int, default=int(m28.LOSS_CFG["phase2_end_epoch"]))
    p.add_argument("--reco_phase3_end_epoch", type=int, default=int(m28.LOSS_CFG["phase3_end_epoch"]))
    p.add_argument("--reco_phase2_alpha_fr_end", type=float, default=float(m28.LOSS_CFG["phase2_alpha_fr_end"]))
    p.add_argument("--reco_phase3_alpha_fr_end", type=float, default=float(m28.LOSS_CFG["phase3_alpha_fr_end"]))
    p.add_argument("--reco_phase4_alpha_fr", type=float, default=float(m28.LOSS_CFG["phase4_alpha_fr"]))
    p.add_argument("--reco_phase2_ss_end", type=float, default=float(m28.LOSS_CFG["phase2_ss_end"]))
    p.add_argument("--reco_phase3_ss_end", type=float, default=float(m28.LOSS_CFG["phase3_ss_end"]))
    p.add_argument("--reco_phase4_ss", type=float, default=float(m28.LOSS_CFG["phase4_ss"]))
    p.add_argument("--reco_phase2_free_run_every_n", type=int, default=int(m28.LOSS_CFG["phase2_free_run_every_n"]))
    p.add_argument("--reco_phase3_free_run_every_n", type=int, default=int(m28.LOSS_CFG["phase3_free_run_every_n"]))
    p.add_argument("--reco_phase4_free_run_every_n", type=int, default=int(m28.LOSS_CFG["phase4_free_run_every_n"]))
    p.add_argument("--reco_phase_lr_decay", type=float, default=float(m28.LOSS_CFG["phase_lr_decay"]))

    # seeded candidate generation
    p.add_argument("--seed_candidate_k", type=int, default=1)
    p.add_argument("--seed_keep_m", type=int, default=1)
    p.add_argument("--seed_max_prefix", type=int, default=12)
    p.add_argument("--seed_temp", type=float, default=0.35)
    p.add_argument("--candidate_gen_batch", type=int, default=64)
    p.add_argument("--train_specialist_prefix", type=int, default=-1, help="Fixed carryover prefix count used for specialist reco training; -1 uses seed_max_prefix.")

    # residual acceptance / ranking
    p.add_argument("--search_eps_total", type=float, default=0.60)
    p.add_argument("--search_eps_count", type=float, default=0.30)
    p.add_argument("--search_w_chamfer", type=float, default=1.0)
    p.add_argument("--search_w_count", type=float, default=0.25)
    p.add_argument("--search_w_pt", type=float, default=0.12)
    p.add_argument("--search_w_mass", type=float, default=0.08)

    # final dualview
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--dual_epochs", type=int, default=80)
    p.add_argument("--dual_patience", type=int, default=14)
    p.add_argument("--dual_lr", type=float, default=1.2e-4)
    p.add_argument("--dual_weight_decay", type=float, default=1e-4)

    p.add_argument("--save_fusion_scores", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    m33.set_seed(int(args.seed))

    device = torch.device(args.device)
    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Model-39 Prefix-Specialist m28 + Deterministic Residual MultiCandidate DualView")
    print(f"Run: {save_root}")
    print(
        f"Split train/val/test = {int(args.n_train_split)}/{int(args.n_val_split)}/{int(args.n_test_split)} | "
        f"K={int(args.seed_candidate_k)} keepM={int(args.seed_keep_m)} max_prefix={int(args.seed_max_prefix)} "
        f"train_specialist_prefix={int(args.train_specialist_prefix)}"
    )
    token_quantizer: Optional[TokenCodebookQuantizer] = None
    if str(args.codebook_path).strip():
        token_quantizer = _load_token_codebook_quantizer(str(args.codebook_path).strip())
        print(
            f"Quantized decode enabled: strategy={token_quantizer.strategy} "
            f"codebook={str(args.codebook_path)} label={str(args.codebook_label)}"
        )
    print("=" * 72)

    # ---------------------------------------------------------------------
    # Data + deterministic pseudo-HLT
    # ---------------------------------------------------------------------
    train_files = base._parse_h5_path_arg(str(args.train_path))
    max_needed = int(args.offset_jets + args.n_train_jets)
    all_const, all_labels, all_train_w = base.load_raw_constituents_labels_weights_from_h5(
        files=train_files,
        max_jets=max_needed,
        max_constits=int(args.max_constits),
        use_train_weights=bool(args.use_train_weights),
    )
    if all_const.shape[0] < max_needed:
        raise RuntimeError(f"Requested {max_needed} jets but found {all_const.shape[0]}.")

    const_raw = all_const[args.offset_jets : args.offset_jets + args.n_train_jets]
    labels = all_labels[args.offset_jets : args.offset_jets + args.n_train_jets].astype(np.int64)
    train_w = all_train_w[args.offset_jets : args.offset_jets + args.n_train_jets].astype(np.float32)

    total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > const_raw.shape[0]:
        raise RuntimeError(f"Requested splits sum to {total_need} but loaded jets are only {const_raw.shape[0]}")

    idx_all = np.arange(len(labels), dtype=np.int64)
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

    cfg = base._deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    raw_mask = const_raw[:, :, 0] > 0.0
    mask_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~mask_off] = 0.0

    print("Generating pseudo-HLT (deterministic keyed D_hard)...")
    jet_keys = (np.arange(len(const_off), dtype=np.int64) + int(args.offset_jets)).astype(np.int64)
    const_hlt, mask_hlt, hlt_stats = m33._apply_hlt_effects_deterministic_keyed(
        const=const_off,
        mask=mask_off,
        cfg=cfg,
        jet_keys=jet_keys,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    # Match m28 training regime: sort both Offline and HLT tokens by descending pT.
    const_off, mask_off, _ = m28.sort_constituents_by_pt_np(const_off, mask_off)
    const_hlt, mask_hlt, _ = m28.sort_constituents_by_pt_np(const_hlt, mask_hlt)

    const_off_teacher = const_off
    if bool(args.step1_quantize_teacher_offline):
        if token_quantizer is None:
            raise RuntimeError("--step1_quantize_teacher_offline requires --codebook_path.")
        off_tok_full = _const_to_token5_np(const_off, mask_off)
        off_tok_quant = _quantize_token5_np(off_tok_full, token_quantizer)
        const_off_teacher = _token5_to_const_np(off_tok_quant, mask_off)
        print(
            "STEP 1 teacher offline quantization enabled: "
            f"strategy={token_quantizer.strategy} label={str(args.codebook_label)}"
        )
    print(
        "HLT stats: "
        f"avg_offline={hlt_stats.get('avg_offline_per_jet', float('nan')):.2f}, "
        f"avg_hlt={hlt_stats.get('avg_hlt_per_jet', float('nan')):.2f}, "
        f"merged={hlt_stats.get('n_merged_pairs', 0)}, eff_lost={hlt_stats.get('n_lost_eff', 0)}"
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # standardized features
    print("Computing standardized features for train/val/test...")
    feat_off_tr = compute_features(const_off_teacher[train_idx], mask_off[train_idx])
    feat_off_va = compute_features(const_off_teacher[val_idx], mask_off[val_idx])
    feat_off_te = compute_features(const_off_teacher[test_idx], mask_off[test_idx])
    feat_hlt_tr = compute_features(const_hlt[train_idx], mask_hlt[train_idx])
    feat_hlt_va = compute_features(const_hlt[val_idx], mask_hlt[val_idx])
    feat_hlt_te = compute_features(const_hlt[test_idx], mask_hlt[test_idx])

    means, stds = get_stats(feat_off_tr, mask_off[train_idx], np.arange(feat_off_tr.shape[0], dtype=np.int64))
    feat_off_tr = standardize(feat_off_tr, mask_off[train_idx], means, stds)
    feat_off_va = standardize(feat_off_va, mask_off[val_idx], means, stds)
    feat_off_te = standardize(feat_off_te, mask_off[test_idx], means, stds)
    feat_hlt_tr = standardize(feat_hlt_tr, mask_hlt[train_idx], means, stds)
    feat_hlt_va = standardize(feat_hlt_va, mask_hlt[val_idx], means, stds)
    feat_hlt_te = standardize(feat_hlt_te, mask_hlt[test_idx], means, stds)

    sw_train = train_w[train_idx] if bool(args.use_train_weights) else np.ones((len(train_idx),), dtype=np.float32)
    sw_val = train_w[val_idx] if bool(args.use_train_weights) else np.ones((len(val_idx),), dtype=np.float32)
    sw_test = train_w[test_idx] if bool(args.use_train_weights) else np.ones((len(test_idx),), dtype=np.float32)

    # ---------------------------------------------------------------------
    # STEP 1: Teacher + baseline
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 1: Teacher + HLT baseline")
    print("=" * 72)

    cls_cfg = {
        "epochs": int(args.cls_epochs),
        "patience": int(args.cls_patience),
        "lr": float(args.cls_lr),
        "weight_decay": float(args.cls_weight_decay),
        "warmup_epochs": int(args.cls_warmup_epochs),
    }

    ds_tr_off = base.WeightedJetDataset(feat_off_tr, mask_off[train_idx], labels[train_idx], sw_train)
    ds_va_off = base.WeightedJetDataset(feat_off_va, mask_off[val_idx], labels[val_idx], sw_val)
    ds_te_off = base.WeightedJetDataset(feat_off_te, mask_off[test_idx], labels[test_idx], sw_test)

    dl_tr_off = DataLoader(ds_tr_off, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    dl_va_off = DataLoader(ds_va_off, batch_size=int(args.batch_size), shuffle=False)
    dl_te_off = DataLoader(ds_te_off, batch_size=int(args.batch_size), shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    teacher = base.train_single_view_classifier_auc(
        model=teacher,
        train_loader=dl_tr_off,
        val_loader=dl_va_off,
        device=device,
        train_cfg=cls_cfg,
        name="Teacher",
    )
    teacher_auc_test, teacher_p_test, teacher_y_test, teacher_w_test = base._eval_classifier_with_optional_weights(teacher, dl_te_off, device)

    ds_tr_hlt = base.WeightedJetDataset(feat_hlt_tr, mask_hlt[train_idx], labels[train_idx], sw_train)
    ds_va_hlt = base.WeightedJetDataset(feat_hlt_va, mask_hlt[val_idx], labels[val_idx], sw_val)
    ds_te_hlt = base.WeightedJetDataset(feat_hlt_te, mask_hlt[test_idx], labels[test_idx], sw_test)

    dl_tr_hlt = DataLoader(ds_tr_hlt, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    dl_va_hlt = DataLoader(ds_va_hlt, batch_size=int(args.batch_size), shuffle=False)
    dl_te_hlt = DataLoader(ds_te_hlt, batch_size=int(args.batch_size), shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = base.train_single_view_classifier_auc(
        model=baseline,
        train_loader=dl_tr_hlt,
        val_loader=dl_va_hlt,
        device=device,
        train_cfg=cls_cfg,
        name="Baseline",
    )
    baseline_auc_test, baseline_p_test, baseline_y_test, baseline_w_test = base._eval_classifier_with_optional_weights(baseline, dl_te_hlt, device)

    print(f"Teacher test AUC={teacher_auc_test:.4f} | Baseline test AUC={baseline_auc_test:.4f}")

    specialist_prefix = int(args.train_specialist_prefix if int(args.train_specialist_prefix) >= 0 else args.seed_max_prefix)
    specialist_prefix = int(np.clip(specialist_prefix, 0, int(args.max_constits)))

    # ---------------------------------------------------------------------
    # STEP 2: Carryover predictor
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2: Carryover token predictor")
    print("=" * 72)

    carry_mode = str(args.carry_target_mode).lower()
    k_carry_effective = -1
    if carry_mode == "fixed_k":
        k_carry = int(args.carry_target_k)
        if k_carry < 0:
            k_carry = int(specialist_prefix)
        k_carry = int(np.clip(k_carry, 0, int(args.max_constits)))
        k_carry_effective = int(k_carry)
        print(f"Carry target mode: fixed_k (k={k_carry_effective}, thresh_gate={bool(int(args.carry_target_thresh_gate))})")
        carry_tgt_tr = _build_carry_targets_fixed_k_np(
            const_off=const_off[train_idx],
            mask_off=mask_off[train_idx],
            const_hlt=const_hlt[train_idx],
            mask_hlt=mask_hlt[train_idx],
            k_target=int(k_carry_effective),
            dist_thresh=float(args.carry_dist_thresh),
            batch_size=256,
            thresh_gate=bool(int(args.carry_target_thresh_gate)),
        )
        carry_tgt_va = _build_carry_targets_fixed_k_np(
            const_off=const_off[val_idx],
            mask_off=mask_off[val_idx],
            const_hlt=const_hlt[val_idx],
            mask_hlt=mask_hlt[val_idx],
            k_target=int(k_carry_effective),
            dist_thresh=float(args.carry_dist_thresh),
            batch_size=256,
            thresh_gate=bool(int(args.carry_target_thresh_gate)),
        )
    else:
        print("Carry target mode: threshold")
        carry_tgt_tr = _build_carry_targets_np(
            const_off=const_off[train_idx],
            mask_off=mask_off[train_idx],
            const_hlt=const_hlt[train_idx],
            mask_hlt=mask_hlt[train_idx],
            dist_thresh=float(args.carry_dist_thresh),
            batch_size=256,
        )
        carry_tgt_va = _build_carry_targets_np(
            const_off=const_off[val_idx],
            mask_off=mask_off[val_idx],
            const_hlt=const_hlt[val_idx],
            mask_hlt=mask_hlt[val_idx],
            dist_thresh=float(args.carry_dist_thresh),
            batch_size=256,
        )

    pos_frac = float(carry_tgt_tr[mask_hlt[train_idx]].mean()) if np.any(mask_hlt[train_idx]) else 0.5
    pos_w = float((1.0 - pos_frac) / max(pos_frac, 1e-6))

    ds_c_tr = RecoCarryDataset(feat_hlt_tr, mask_hlt[train_idx], carry_tgt_tr)
    ds_c_va = RecoCarryDataset(feat_hlt_va, mask_hlt[val_idx], carry_tgt_va)
    dl_c_tr = DataLoader(ds_c_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_c_va = DataLoader(ds_c_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    carry_model = CarryoverTokenPredictor(
        input_dim=7,
        embed_dim=int(args.carry_embed_dim),
        num_heads=int(args.carry_num_heads),
        num_layers=int(args.carry_num_layers),
        ff_dim=int(args.carry_ff_dim),
        dropout=float(args.carry_dropout),
        max_tokens=int(args.max_constits),
    ).to(device)

    carry_model, carry_metrics = _train_carry_predictor(
        model=carry_model,
        train_loader=dl_c_tr,
        val_loader=dl_c_va,
        device=device,
        epochs=int(args.carry_epochs),
        lr=float(args.carry_lr),
        weight_decay=float(args.carry_weight_decay),
        patience=int(args.carry_patience),
        pos_weight=pos_w,
        lr_decay_start_epoch=int(args.carry_lr_decay_start_epoch),
        lr_decay_gamma=float(args.carry_lr_decay_gamma),
        min_lr_ratio=float(args.carry_min_lr_ratio),
    )

    # ---------------------------------------------------------------------
    # STEP 3: m28 completer (prefix specialist)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 3: Train m28-style completer (prefix specialist)")
    print("=" * 72)

    print(f"Specialist prefix length (train-time): {specialist_prefix}")

    carry_probs_tr = _predict_carry_probs(
        model=carry_model,
        feat_hlt=feat_hlt_tr,
        mask_hlt=mask_hlt[train_idx],
        device=device,
        batch_size=int(args.batch_size),
    )
    carry_probs_va = _predict_carry_probs(
        model=carry_model,
        feat_hlt=feat_hlt_va,
        mask_hlt=mask_hlt[val_idx],
        device=device,
        batch_size=int(args.batch_size),
    )

    pref_tok_tr, pref_len_tr = _build_specialist_prefix_tokens(
        carry_probs=carry_probs_tr,
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        prefix_max=int(specialist_prefix),
    )
    pref_tok_va, pref_len_va = _build_specialist_prefix_tokens(
        carry_probs=carry_probs_va,
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        prefix_max=int(specialist_prefix),
    )
    if token_quantizer is not None:
        pref_tok_tr = _quantize_token5_np(pref_tok_tr, token_quantizer)
        pref_tok_va = _quantize_token5_np(pref_tok_va, token_quantizer)

    tgt_tok_tr, tgt_mask_tr = _build_continuation_targets_from_prefix(
        off_const=const_off[train_idx],
        off_mask=mask_off[train_idx],
        prefix_tok=pref_tok_tr,
        prefix_len=pref_len_tr,
    )
    tgt_tok_va, tgt_mask_va = _build_continuation_targets_from_prefix(
        off_const=const_off[val_idx],
        off_mask=mask_off[val_idx],
        prefix_tok=pref_tok_va,
        prefix_len=pref_len_va,
    )

    ds_reco_tr = RecoSpecialistDataset(
        feat_hlt=feat_hlt_tr,
        mask_hlt=mask_hlt[train_idx],
        const_hlt=const_hlt[train_idx],
        tgt_tok=tgt_tok_tr,
        tgt_mask=tgt_mask_tr,
        labels=labels[train_idx].astype(np.float32),
        prefix_tok=pref_tok_tr,
        prefix_len=pref_len_tr,
    )
    ds_reco_va = RecoSpecialistDataset(
        feat_hlt=feat_hlt_va,
        mask_hlt=mask_hlt[val_idx],
        const_hlt=const_hlt[val_idx],
        tgt_tok=tgt_tok_va,
        tgt_mask=tgt_mask_va,
        labels=labels[val_idx].astype(np.float32),
        prefix_tok=pref_tok_va,
        prefix_len=pref_len_va,
    )
    dl_reco_tr = DataLoader(
        ds_reco_tr,
        batch_size=int(args.reco_batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
    )
    dl_reco_va = DataLoader(
        ds_reco_va,
        batch_size=int(args.reco_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    reco_model = m28.HLT2OfflineSeq2Seq(
        input_dim_hlt=7,
        token_dim=5,
        embed_dim=int(args.reco_embed_dim),
        num_heads=int(args.reco_num_heads),
        num_enc_layers=int(args.reco_num_enc_layers),
        num_dec_layers=int(args.reco_num_dec_layers),
        ff_dim=int(args.reco_ff_dim),
        dropout=float(args.reco_dropout),
        max_hlt_tokens=int(args.max_constits),
        max_decode_tokens=int(args.max_constits + specialist_prefix),
        use_coord_residual_param=False,
        num_hypotheses=1,
    ).to(device)

    reco_train_cfg = {
        "batch_size": int(args.reco_batch_size),
        "epochs": int(args.reco_epochs),
        "lr": float(args.reco_lr),
        "weight_decay": float(args.reco_weight_decay),
        "patience": int(args.reco_patience),
        "min_epochs": int(args.reco_min_epochs),
    }
    reco_loss_cfg = dict(m28.LOSS_CFG)
    reco_loss_cfg["set_loss_mode"] = str(args.reco_set_loss_mode)
    reco_loss_cfg["w_eos"] = float(args.reco_loss_w_eos)
    reco_loss_cfg["w_count"] = float(args.reco_loss_w_count)
    reco_loss_cfg["w_jetpt"] = float(args.reco_loss_w_jetpt)
    reco_loss_cfg["w_4vec"] = float(args.reco_loss_w_4vec)
    reco_loss_cfg["w_conf_rank"] = float(args.reco_loss_w_conf_rank)
    reco_loss_cfg["w_conf_prefix"] = float(args.reco_loss_w_conf_prefix)
    reco_loss_cfg["conf_margin"] = float(args.reco_conf_margin)
    reco_loss_cfg["prefix_tau"] = float(args.reco_conf_prefix_tau)
    reco_loss_cfg["physics_warmup_epochs"] = int(args.reco_physics_warmup_epochs)
    reco_loss_cfg["phase1_end_epoch"] = int(args.reco_phase1_end_epoch)
    reco_loss_cfg["phase2_end_epoch"] = int(args.reco_phase2_end_epoch)
    reco_loss_cfg["phase3_end_epoch"] = int(args.reco_phase3_end_epoch)
    reco_loss_cfg["phase2_alpha_fr_end"] = float(args.reco_phase2_alpha_fr_end)
    reco_loss_cfg["phase3_alpha_fr_end"] = float(args.reco_phase3_alpha_fr_end)
    reco_loss_cfg["phase4_alpha_fr"] = float(args.reco_phase4_alpha_fr)
    reco_loss_cfg["phase2_ss_end"] = float(args.reco_phase2_ss_end)
    reco_loss_cfg["phase3_ss_end"] = float(args.reco_phase3_ss_end)
    reco_loss_cfg["phase4_ss"] = float(args.reco_phase4_ss)
    reco_loss_cfg["phase2_free_run_every_n"] = int(args.reco_phase2_free_run_every_n)
    reco_loss_cfg["phase3_free_run_every_n"] = int(args.reco_phase3_free_run_every_n)
    reco_loss_cfg["phase4_free_run_every_n"] = int(args.reco_phase4_free_run_every_n)
    reco_loss_cfg["phase_lr_decay"] = float(args.reco_phase_lr_decay)
    reco_loss_cfg["winner_mode"] = "reco"

    reco_model, reco_metrics = _train_reco_specialist_with_prefix(
        model=reco_model,
        train_loader=dl_reco_tr,
        val_loader=dl_reco_va,
        device=device,
        train_cfg=reco_train_cfg,
        loss_cfg=reco_loss_cfg,
        token_quantizer=token_quantizer,
    )

    # ---------------------------------------------------------------------
    # STEP 4: Seeded candidate generation (train/val)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 4: Seeded candidate generation + deterministic residual ranking")
    print("=" * 72)

    c_tr = _generate_candidates_split(
        reco_model=reco_model,
        carry_model=carry_model,
        feat_hlt=feat_hlt_tr,
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        jet_keys=jet_keys[train_idx],
        cfg=cfg,
        seed_offset=int(args.seed + args.dhard_seed_offset),
        candidate_k=int(args.seed_candidate_k),
        keep_m=int(args.seed_keep_m),
        max_prefix=int(args.seed_max_prefix),
        seed_temp=float(args.seed_temp),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
        batch_size=int(args.candidate_gen_batch),
        device=device,
        seed=int(args.seed) + 101,
        token_quantizer=token_quantizer,
    )
    c_va = _generate_candidates_split(
        reco_model=reco_model,
        carry_model=carry_model,
        feat_hlt=feat_hlt_va,
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        jet_keys=jet_keys[val_idx],
        cfg=cfg,
        seed_offset=int(args.seed + args.dhard_seed_offset),
        candidate_k=int(args.seed_candidate_k),
        keep_m=int(args.seed_keep_m),
        max_prefix=int(args.seed_max_prefix),
        seed_temp=float(args.seed_temp),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
        batch_size=int(args.candidate_gen_batch),
        device=device,
        seed=int(args.seed) + 202,
        token_quantizer=token_quantizer,
    )

    print(
        "Candidate stats: "
        f"train(feasible_all={np.mean(c_tr.feasible_count_all):.2f}/{int(args.seed_candidate_k)}, bestR={np.mean(c_tr.best_residual_all):.4f}) "
        f"val(feasible_all={np.mean(c_va.feasible_count_all):.2f}/{int(args.seed_candidate_k)}, bestR={np.mean(c_va.best_residual_all):.4f})"
    )

    # ---------------------------------------------------------------------
    # STEP 5: Final dualview heads
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 5: Final Multi-Candidate DualView Classifiers (NoGate + Gated)")
    print("=" * 72)

    mv_tr = _build_m38_multicandidate_arrays(c_tr, max_constits=int(args.max_constits), candidate_k=int(args.seed_candidate_k))
    mv_va = _build_m38_multicandidate_arrays(c_va, max_constits=int(args.max_constits), candidate_k=int(args.seed_candidate_k))

    meta_tr = mv_tr["cand_meta"]
    meta_mean = meta_tr.reshape(-1, meta_tr.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    meta_std = (meta_tr.reshape(-1, meta_tr.shape[-1]).std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    summary_mean = mv_tr["summary_feat"].mean(axis=0, keepdims=True).astype(np.float32)
    summary_std = (mv_tr["summary_feat"].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    mm = meta_mean.reshape(1, 1, -1)
    ms = meta_std.reshape(1, 1, -1)
    sm = summary_mean.reshape(1, -1)
    ss = summary_std.reshape(1, -1)
    mv_tr["cand_meta"] = ((mv_tr["cand_meta"] - mm) / ms).astype(np.float32)
    mv_va["cand_meta"] = ((mv_va["cand_meta"] - mm) / ms).astype(np.float32)
    mv_tr["summary_feat"] = ((mv_tr["summary_feat"] - sm) / ss).astype(np.float32)
    mv_va["summary_feat"] = ((mv_va["summary_feat"] - sm) / ss).astype(np.float32)

    ds_dv_tr = DualViewM38Dataset(
        feat_hlt=feat_hlt_tr,
        mask_hlt=mask_hlt[train_idx],
        cand_tokens=mv_tr["cand_tokens"],
        cand_masks=mv_tr["cand_masks"],
        cand_meta=mv_tr["cand_meta"],
        summary_feat=mv_tr["summary_feat"],
        labels=labels[train_idx],
        sample_weight=sw_train,
    )
    ds_dv_va = DualViewM38Dataset(
        feat_hlt=feat_hlt_va,
        mask_hlt=mask_hlt[val_idx],
        cand_tokens=mv_va["cand_tokens"],
        cand_masks=mv_va["cand_masks"],
        cand_meta=mv_va["cand_meta"],
        summary_feat=mv_va["summary_feat"],
        labels=labels[val_idx],
        sample_weight=sw_val,
    )

    dl_dv_tr = DataLoader(ds_dv_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_dv_va = DataLoader(ds_dv_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    m38_nogate = MultiCandidateM38NoGate(
        cand_meta_dim=int(mv_tr["cand_meta"].shape[-1]),
        summary_dim=int(mv_tr["summary_feat"].shape[-1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    m38_nogate, m38_nogate_metrics = _train_m38_model(
        model=m38_nogate,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="M39NoGate",
    )

    m38_gated = MultiCandidateM38Gated(
        cand_meta_dim=int(mv_tr["cand_meta"].shape[-1]),
        summary_dim=int(mv_tr["summary_feat"].shape[-1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    m38_gated, m38_gated_metrics = _train_m38_model(
        model=m38_gated,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="M39Gated",
    )

    # ---------------------------------------------------------------------
    # STEP 6: Test candidates + final eval
    # ---------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 6: Test seeded candidates + final evaluation")
    print("=" * 72)

    c_te = _generate_candidates_split(
        reco_model=reco_model,
        carry_model=carry_model,
        feat_hlt=feat_hlt_te,
        const_hlt=const_hlt[test_idx],
        mask_hlt=mask_hlt[test_idx],
        jet_keys=jet_keys[test_idx],
        cfg=cfg,
        seed_offset=int(args.seed + args.dhard_seed_offset),
        candidate_k=int(args.seed_candidate_k),
        keep_m=int(args.seed_keep_m),
        max_prefix=int(args.seed_max_prefix),
        seed_temp=float(args.seed_temp),
        eps_total=float(args.search_eps_total),
        eps_count=float(args.search_eps_count),
        w_chamfer=float(args.search_w_chamfer),
        w_count=float(args.search_w_count),
        w_pt=float(args.search_w_pt),
        w_mass=float(args.search_w_mass),
        batch_size=int(args.candidate_gen_batch),
        device=device,
        seed=int(args.seed) + 303,
        token_quantizer=token_quantizer,
    )
    mv_te = _build_m38_multicandidate_arrays(c_te, max_constits=int(args.max_constits), candidate_k=int(args.seed_candidate_k))
    mv_te["cand_meta"] = ((mv_te["cand_meta"] - mm) / ms).astype(np.float32)
    mv_te["summary_feat"] = ((mv_te["summary_feat"] - sm) / ss).astype(np.float32)

    ds_dv_te = DualViewM38Dataset(
        feat_hlt=feat_hlt_te,
        mask_hlt=mask_hlt[test_idx],
        cand_tokens=mv_te["cand_tokens"],
        cand_masks=mv_te["cand_masks"],
        cand_meta=mv_te["cand_meta"],
        summary_feat=mv_te["summary_feat"],
        labels=labels[test_idx],
        sample_weight=sw_test,
    )
    dl_dv_te = DataLoader(ds_dv_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    auc_nog, fpr50_nog, pred_nog, lab_final, w_final = _eval_m38_model(m38_nogate, dl_dv_te, device)
    auc_gat, fpr50_gat, pred_gat, _lab2, _w2 = _eval_m38_model(m38_gated, dl_dv_te, device)

    fpr50_teacher = _fpr50(teacher_y_test, teacher_p_test, teacher_w_test)
    fpr50_baseline = _fpr50(baseline_y_test, baseline_p_test, baseline_w_test)

    print("\n" + "=" * 72)
    print("FINAL TEST")
    print("=" * 72)
    print(
        f"Teacher AUC={teacher_auc_test:.4f} FPR50={fpr50_teacher:.6f} | "
        f"HLT baseline AUC={baseline_auc_test:.4f} FPR50={fpr50_baseline:.6f} | "
        f"m39 NoGate AUC={auc_nog:.4f} FPR50={fpr50_nog:.6f} | "
        f"m39 Gated AUC={auc_gat:.4f} FPR50={fpr50_gat:.6f}"
    )

    # save artifacts
    torch.save({"model": teacher.state_dict(), "auc_test": float(teacher_auc_test)}, save_root / "teacher.pt")
    torch.save({"model": baseline.state_dict(), "auc_test": float(baseline_auc_test)}, save_root / "baseline_hlt.pt")
    torch.save({"model": carry_model.state_dict(), "metrics": carry_metrics}, save_root / "carry_predictor.pt")
    torch.save({"model": reco_model.state_dict(), "metrics": reco_metrics}, save_root / "reco_completer_m28style.pt")
    torch.save({"model": m38_nogate.state_dict(), "metrics": m38_nogate_metrics}, save_root / "m39_multicand_nogate.pt")
    torch.save({"model": m38_gated.state_dict(), "metrics": m38_gated_metrics}, save_root / "m39_multicand_gated.pt")

    np.savez_compressed(
        save_root / "m39_test_scores.npz",
        labels_test=lab_final.astype(np.float32),
        preds_m39_nogate=pred_nog.astype(np.float32),
        preds_m39_gated=pred_gat.astype(np.float32),
        preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
        preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
        sample_weight=np.asarray(w_final, dtype=np.float32),
        auc_teacher=float(teacher_auc_test),
        auc_hlt=float(baseline_auc_test),
        auc_m39_nogate=float(auc_nog),
        auc_m39_gated=float(auc_gat),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_hlt=float(fpr50_baseline),
        fpr50_m39_nogate=float(fpr50_nog),
        fpr50_m39_gated=float(fpr50_gat),
    )

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "fusion_scores_test.npz",
            labels_test=lab_final.astype(np.float32),
            preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
            preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
            preds_m39_nogate=np.asarray(pred_nog, dtype=np.float32),
            preds_m39_gated=np.asarray(pred_gat, dtype=np.float32),
            sample_weight=np.asarray(w_final, dtype=np.float32),
        )

    report = {
        "model": "m39_prefixspecialist_detresid_multicand",
        "seed": int(args.seed),
        "n_train_jets": int(args.n_train_jets),
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "teacher": {
            "auc_test": float(teacher_auc_test),
            "fpr50_test": float(fpr50_teacher),
        },
        "hlt_baseline": {
            "auc_test": float(baseline_auc_test),
            "fpr50_test": float(fpr50_baseline),
        },
        "carry_predictor": carry_metrics,
        "quantization": {
            "codebook_path": str(args.codebook_path),
            "codebook_label": str(args.codebook_label),
            "step1_quantize_teacher_offline": bool(args.step1_quantize_teacher_offline),
        },
        "carry_targeting": {
            "mode": str(args.carry_target_mode),
            "k": int(args.carry_target_k),
            "k_effective": int(k_carry_effective),
            "dist_thresh": float(args.carry_dist_thresh),
            "thresh_gate": bool(int(args.carry_target_thresh_gate)),
        },
        "reco_completer": reco_metrics,
        "candidate_generation": {
            "candidate_k": int(args.seed_candidate_k),
            "keep_m": int(args.seed_keep_m),
            "max_prefix": int(args.seed_max_prefix),
            "eps_total": float(args.search_eps_total),
            "eps_count": float(args.search_eps_count),
            "weights": {
                "chamfer": float(args.search_w_chamfer),
                "count": float(args.search_w_count),
                "pt": float(args.search_w_pt),
                "mass": float(args.search_w_mass),
            },
            "train": {
                "mean_feasible_all": float(np.mean(c_tr.feasible_count_all)),
                "mean_best_residual": float(np.mean(c_tr.best_residual_all)),
            },
            "val": {
                "mean_feasible_all": float(np.mean(c_va.feasible_count_all)),
                "mean_best_residual": float(np.mean(c_va.best_residual_all)),
            },
            "test": {
                "mean_feasible_all": float(np.mean(c_te.feasible_count_all)),
                "mean_best_residual": float(np.mean(c_te.best_residual_all)),
            },
        },
        "m39_nogate": {
            "auc_test": float(auc_nog),
            "fpr50_test": float(fpr50_nog),
            "metrics": m38_nogate_metrics,
        },
        "m39_gated": {
            "auc_test": float(auc_gat),
            "fpr50_test": float(fpr50_gat),
            "metrics": m38_gated_metrics,
        },
    }
    with open(save_root / "m39_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
    )

    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()

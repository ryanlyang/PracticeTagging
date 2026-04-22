#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m36: Dictionary retrieval dualview top-tagging pipeline.

Key idea:
- Hold out a large dictionary split of paired (offline, HLT, class).
- For each query HLT jet, retrieve nearest HLT dictionary entries per true class.
- Use paired offline jets as candidate views (no bounded repair in v1).
- Train compatibility selector + final dualview top taggers.

Compared to m33/m34:
- Replaces latent prior/proposer/degrader candidate generation with retrieval.
- Keeps teacher/baseline, selector, and final dualview heads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_detfeas_dualview as m33
from unmerge_correct_hlt import ParticleTransformer, compute_features, get_stats, standardize


@dataclass
class JetMeta:
    count: np.ndarray
    jet_pt: np.ndarray
    jet_mass: np.ndarray


@dataclass
class RetrievalClassIndex:
    class_val: int
    local_ids: np.ndarray
    nn: NearestNeighbors


class RetrievalEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(embed_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (b + eps)


def _jet_meta_np(const: np.ndarray, mask: np.ndarray) -> JetMeta:
    m = mask.astype(np.float32)
    pt = const[..., 0] * m
    eta = const[..., 1]
    phi = const[..., 2]
    e = const[..., 3] * m

    px = pt * np.cos(phi) * m
    py = pt * np.sin(phi) * m
    pz = pt * np.sinh(eta) * m

    px_sum = px.sum(axis=1)
    py_sum = py.sum(axis=1)
    pz_sum = pz.sum(axis=1)
    e_sum = e.sum(axis=1)

    jet_pt = np.sqrt(np.maximum(px_sum ** 2 + py_sum ** 2, 0.0))
    p2 = px_sum ** 2 + py_sum ** 2 + pz_sum ** 2
    m2 = np.maximum(e_sum ** 2 - p2, 0.0)
    jet_mass = np.sqrt(m2)
    count = mask.sum(axis=1).astype(np.float32)
    return JetMeta(count=count.astype(np.float32), jet_pt=jet_pt.astype(np.float32), jet_mass=jet_mass.astype(np.float32))


def _build_retrieval_desc(
    const: np.ndarray,
    mask: np.ndarray,
    meta: JetMeta,
    max_constits: int,
) -> np.ndarray:
    m = mask.astype(bool)
    pt = np.where(m, const[..., 0], 0.0)
    eta = np.where(m, const[..., 1], 0.0)

    pt_sum = pt.sum(axis=1)
    topk = min(3, pt.shape[1])
    if topk > 0:
        part = np.partition(pt, kth=pt.shape[1] - topk, axis=1)[:, -topk:]
        part = np.sort(part, axis=1)[:, ::-1]
        if topk < 3:
            pad = np.zeros((pt.shape[0], 3 - topk), dtype=np.float32)
            top3 = np.concatenate([part, pad], axis=1)
        else:
            top3 = part
    else:
        top3 = np.zeros((pt.shape[0], 3), dtype=np.float32)

    frac = _safe_div(top3, pt_sum[:, None])
    eta_w = _safe_div((eta * pt).sum(axis=1), pt_sum)
    eta_abs_w = _safe_div((np.abs(eta) * pt).sum(axis=1), pt_sum)

    desc = np.stack(
        [
            meta.count / float(max_constits),
            np.log1p(meta.jet_pt),
            np.log1p(meta.jet_mass),
            frac[:, 0],
            frac[:, 1],
            frac[:, 2],
            eta_w / 5.0,
            eta_abs_w / 5.0,
        ],
        axis=1,
    ).astype(np.float32)
    return desc


def _select_landmark_indices(desc: np.ndarray, n_landmarks: int, seed: int) -> np.ndarray:
    n = int(desc.shape[0])
    k = int(max(1, min(n_landmarks, n)))
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, n))
    ids = [first]
    min_d2 = np.sum((desc - desc[first:first + 1]) ** 2, axis=1)
    min_d2[first] = -1.0
    for _ in range(1, k):
        nxt = int(np.argmax(min_d2))
        ids.append(nxt)
        d2 = np.sum((desc - desc[nxt:nxt + 1]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[ids] = -1.0
    return np.asarray(ids, dtype=np.int64)


def _build_landmark11_desc(
    desc: np.ndarray,
    meta: JetMeta,
    landmark_desc: np.ndarray,
    max_constits: int,
) -> np.ndarray:
    # 11D = 8 landmark-distance channels + count + log(1+pt) + log(1+mass)
    # Distances are computed in standardized physics-descriptor space.
    k = int(landmark_desc.shape[0])
    diff = desc[:, None, :] - landmark_desc[None, :, :]
    d = np.sqrt(np.maximum((diff * diff).sum(axis=2), 0.0)).astype(np.float32)
    if k < 8:
        d = np.concatenate([d, np.zeros((d.shape[0], 8 - k), dtype=np.float32)], axis=1)
    elif k > 8:
        d = d[:, :8]
    c = (meta.count / float(max_constits)).astype(np.float32)[:, None]
    p = np.log1p(meta.jet_pt).astype(np.float32)[:, None]
    m = np.log1p(meta.jet_mass).astype(np.float32)[:, None]
    return np.concatenate([d, c, p, m], axis=1).astype(np.float32)


def _train_retrieval_embedder(
    desc_train: np.ndarray,
    meta_train: JetMeta,
    labels_train: np.ndarray,
    desc_dict: np.ndarray,
    meta_dict: JetMeta,
    dict_labels: np.ndarray,
    max_constits: int,
    w_desc: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    seed: int,
    device: torch.device,
    embed_dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    pool_size: int,
    train_anchors: int,
    lr: float,
    weight_decay: float,
    margin: float,
) -> Tuple[RetrievalEmbedder, Dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    n_all = int(desc_train.shape[0])
    n_use = int(min(max(1, train_anchors), n_all))
    if n_use < n_all:
        use_idx = rng.choice(n_all, size=n_use, replace=False)
    else:
        use_idx = np.arange(n_all, dtype=np.int64)

    dtr = torch.tensor(desc_train[use_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(labels_train[use_idx].astype(np.int64), dtype=torch.long, device=device)
    ctr = torch.tensor(meta_train.count[use_idx], dtype=torch.float32, device=device)
    ptr = torch.tensor(meta_train.jet_pt[use_idx], dtype=torch.float32, device=device)
    mtr = torch.tensor(meta_train.jet_mass[use_idx], dtype=torch.float32, device=device)

    dd = torch.tensor(desc_dict, dtype=torch.float32, device=device)
    cd = torch.tensor(meta_dict.count, dtype=torch.float32, device=device)
    pd = torch.tensor(meta_dict.jet_pt, dtype=torch.float32, device=device)
    md = torch.tensor(meta_dict.jet_mass, dtype=torch.float32, device=device)

    class0 = torch.tensor(np.where(dict_labels.astype(np.int64) == 0)[0], dtype=torch.long, device=device)
    class1 = torch.tensor(np.where(dict_labels.astype(np.int64) == 1)[0], dtype=torch.long, device=device)
    if int(class0.numel()) == 0 or int(class1.numel()) == 0:
        raise RuntimeError("Dictionary must have both classes for learned retrieval embedder.")

    model = RetrievalEmbedder(
        input_dim=int(desc_train.shape[1]),
        hidden_dim=int(hidden_dim),
        embed_dim=int(embed_dim),
        dropout=float(dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    eps = 1e-6
    bs = int(max(16, batch_size))
    psize = int(max(32, pool_size))
    maxc = float(max(1, max_constits))
    loss_hist = []

    for ep in range(int(max(1, epochs))):
        model.train()
        perm = torch.randperm(dtr.shape[0], device=device)
        ep_loss = 0.0
        ep_n = 0

        for s in range(0, int(perm.numel()), bs):
            idx = perm[s:s + bs]
            if idx.numel() == 0:
                continue
            ad = dtr[idx]
            ay = ytr[idx]
            ac = ctr[idx]
            ap = ptr[idx]
            am = mtr[idx]

            b = int(idx.numel())
            cand_idx = torch.empty((b, psize), dtype=torch.long, device=device)
            is_bg = ay == 0
            is_tp = ~is_bg
            if bool(is_bg.any()):
                nbg = int(is_bg.sum().item())
                rid = torch.randint(0, int(class0.numel()), (nbg, psize), device=device)
                cand_idx[is_bg] = class0[rid]
            if bool(is_tp.any()):
                ntp = int(is_tp.sum().item())
                rid = torch.randint(0, int(class1.numel()), (ntp, psize), device=device)
                cand_idx[is_tp] = class1[rid]

            cd_desc = dd[cand_idx]  # [B, P, D]
            rs = torch.linalg.norm(cd_desc - ad.unsqueeze(1), dim=-1)

            cc = cd[cand_idx]
            cp = pd[cand_idx]
            cm = md[cand_idx]
            rc = torch.abs(cc - ac.unsqueeze(1)) / maxc
            rpt = torch.abs(torch.log1p(cp.clamp_min(0.0)) - torch.log1p(ap.clamp_min(0.0).unsqueeze(1)))
            rms = torch.abs(torch.log1p(cm.clamp_min(0.0)) - torch.log1p(am.clamp_min(0.0).unsqueeze(1)))
            rt = float(w_desc) * rs + float(w_count) * rc + float(w_pt) * rpt + float(w_mass) * rms

            pos_j = torch.argmin(rt, dim=1)
            neg_j = torch.argmax(rt, dim=1)
            row = torch.arange(b, device=device)
            pos_desc = cd_desc[row, pos_j]
            neg_desc = cd_desc[row, neg_j]

            z_a = model(ad)
            z_p = model(pos_desc)
            z_n = model(neg_desc)

            loss_trip = F.triplet_margin_loss(z_a, z_p, z_n, margin=float(margin), p=2.0, reduction="mean")
            d_ap = torch.linalg.norm(z_a - z_p, dim=-1)
            d_an = torch.linalg.norm(z_a - z_n, dim=-1)
            loss_rank = F.softplus((d_ap - d_an) / 0.1).mean()
            pos_target = rt[row, pos_j].detach()
            neg_target = rt[row, neg_j].detach()
            pos_target = pos_target / (pos_target.mean() + eps)
            neg_target = neg_target / (neg_target.mean() + eps)
            loss_scale = 0.5 * F.mse_loss(d_ap, pos_target) + 0.5 * F.mse_loss(d_an, neg_target)
            loss = loss_trip + 0.25 * loss_rank + 0.10 * loss_scale

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ep_loss += float(loss.item()) * b
            ep_n += b

        mean_loss = ep_loss / max(1, ep_n)
        loss_hist.append(mean_loss)
        print(f"RetrievalEmbed ep {ep+1:03d}: loss={mean_loss:.6f}")

    metrics = {
        "epochs": int(max(1, epochs)),
        "anchor_count": int(n_use),
        "pool_size": int(psize),
        "loss_final": float(loss_hist[-1]) if loss_hist else float("nan"),
        "loss_best": float(min(loss_hist)) if loss_hist else float("nan"),
    }
    return model, metrics


@torch.no_grad()
def _encode_retrieval_desc(
    model: RetrievalEmbedder,
    desc: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    n = int(desc.shape[0])
    out = np.zeros((n, model.net[-1].out_features), dtype=np.float32)
    bs = int(max(64, batch_size))
    for s in range(0, n, bs):
        e = min(n, s + bs)
        x = torch.tensor(desc[s:e], dtype=torch.float32, device=device)
        z = model(x).detach().cpu().numpy().astype(np.float32)
        out[s:e] = z
    return out


def _topk_smallest_idx(values: np.ndarray, k: int) -> np.ndarray:
    k = int(max(0, min(k, values.shape[0])))
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    if values.shape[0] <= k:
        return np.argsort(values).astype(np.int64)
    part = np.argpartition(values, k - 1)[:k]
    return part[np.argsort(values[part])].astype(np.int64)


def _build_class_index(
    dict_desc: np.ndarray,
    dict_labels: np.ndarray,
    class_val: int,
) -> RetrievalClassIndex:
    local_ids = np.where(dict_labels.astype(np.int64) == int(class_val))[0].astype(np.int64)
    if local_ids.size == 0:
        raise RuntimeError(f"Dictionary has no entries for class {class_val}")
    nn = NearestNeighbors(algorithm="ball_tree", leaf_size=40, metric="euclidean")
    nn.fit(dict_desc[local_ids])
    return RetrievalClassIndex(class_val=int(class_val), local_ids=local_ids, nn=nn)


def _retrieve_pool_for_class(
    query_index_desc: np.ndarray,
    query_score_desc: np.ndarray,
    query_meta: JetMeta,
    query_const_hlt: np.ndarray,
    query_mask_hlt: np.ndarray,
    class_index: RetrievalClassIndex,
    dict_score_desc: np.ndarray,
    dict_const_off: np.ndarray,
    dict_mask_off: np.ndarray,
    dict_meta: JetMeta,
    target_k: int,
    per_round: int,
    max_rounds: int,
    eps_total: float,
    eps_count: float,
    w_desc: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    max_constits: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    n, l, _ = query_const_hlt.shape
    k_total = int(max(1, per_round * max_rounds))
    k_nn = int(max(1, min(k_total, class_index.local_ids.shape[0])))

    cand_const = np.zeros((n, target_k, l, 4), dtype=np.float32)
    cand_mask = np.zeros((n, target_k, l), dtype=bool)
    resid_total = np.full((n, target_k), np.inf, dtype=np.float32)
    resid_set = np.full((n, target_k), np.inf, dtype=np.float32)
    resid_count = np.full((n, target_k), np.inf, dtype=np.float32)
    resid_pt = np.full((n, target_k), np.inf, dtype=np.float32)
    resid_mass = np.full((n, target_k), np.inf, dtype=np.float32)
    round_found = np.full((n, target_k), float(max_rounds + 1), dtype=np.float32)
    feasible = np.zeros((n, target_k), dtype=bool)
    feasible_count = np.zeros((n,), dtype=np.float32)

    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        qidx_b = query_index_desc[s:e]
        _dists_b, local_b = class_index.nn.kneighbors(qidx_b, n_neighbors=k_nn, return_distance=True)

        for i in range(e - s):
            local_ids = class_index.local_ids[local_b[i].astype(np.int64)]
            qscore = query_score_desc[s + i]
            cscore = dict_score_desc[local_ids]

            q_cnt = float(query_meta.count[s + i])
            q_pt = float(query_meta.jet_pt[s + i])
            q_mass = float(query_meta.jet_mass[s + i])

            c_cnt = dict_meta.count[local_ids]
            c_pt = dict_meta.jet_pt[local_ids]
            c_mass = dict_meta.jet_mass[local_ids]

            rs = np.sqrt(np.sum((cscore - qscore[None, :]) ** 2, axis=1, dtype=np.float32)).astype(np.float32)
            rc = np.abs(c_cnt - q_cnt) / max(1.0, float(max_constits))
            rpt = np.abs(np.log1p(c_pt) - np.log1p(max(q_pt, 0.0)))
            rms = np.abs(np.log1p(c_mass) - np.log1p(max(q_mass, 0.0)))
            rt = (
                float(w_desc) * rs
                + float(w_count) * rc
                + float(w_pt) * rpt
                + float(w_mass) * rms
            )

            feas_mask = (rt <= float(eps_total)) & (rc <= float(eps_count))
            feasible_count[s + i] = float(feas_mask.sum())

            feas_idx = np.where(feas_mask)[0].astype(np.int64)
            if feas_idx.size > 0:
                ord_feas_local = _topk_smallest_idx(rt[feas_idx], target_k)
                pick_feas = feas_idx[ord_feas_local]
            else:
                pick_feas = np.zeros((0,), dtype=np.int64)

            need = int(target_k - pick_feas.size)
            if need > 0:
                non_idx = np.where(~feas_mask)[0].astype(np.int64)
                if non_idx.size > 0:
                    ord_non_local = _topk_smallest_idx(rt[non_idx], need)
                    pick_non = non_idx[ord_non_local]
                else:
                    pick_non = np.zeros((0,), dtype=np.int64)
                pick = np.concatenate([pick_feas, pick_non], axis=0)
            else:
                pick = pick_feas[:target_k]

            if pick.size == 0:
                pick = np.array([int(np.argmin(rt))], dtype=np.int64)

            if pick.size < target_k:
                pad = np.full((target_k - pick.size,), int(pick[0]), dtype=np.int64)
                pick = np.concatenate([pick, pad], axis=0)

            pick = pick[:target_k]
            gids = local_ids[pick]

            cand_const[s + i] = dict_const_off[gids]
            cand_mask[s + i] = dict_mask_off[gids]
            resid_total[s + i] = rt[pick]
            resid_set[s + i] = rs[pick]
            resid_count[s + i] = rc[pick]
            resid_pt[s + i] = rpt[pick]
            resid_mass[s + i] = rms[pick]
            round_found[s + i] = (pick // max(1, int(per_round)) + 1).astype(np.float32)
            feasible[s + i] = feas_mask[pick]

    return {
        "const": cand_const,
        "mask": cand_mask,
        "res_total": resid_total,
        "res_total_pre": resid_total.copy(),
        "res_set": resid_set,
        "res_set_pre": resid_set.copy(),
        "res_count": resid_count,
        "res_count_pre": resid_count.copy(),
        "res_pt": resid_pt,
        "res_pt_pre": resid_pt.copy(),
        "res_mass": resid_mass,
        "res_mass_pre": resid_mass.copy(),
        "round_found": round_found,
        "feasible": feasible,
        "feasible_count": feasible_count,
    }


def _retrieve_pools_split(
    query_index_desc: np.ndarray,
    query_score_desc: np.ndarray,
    query_meta: JetMeta,
    query_const_hlt: np.ndarray,
    query_mask_hlt: np.ndarray,
    idx_bg: RetrievalClassIndex,
    idx_top: RetrievalClassIndex,
    dict_score_desc: np.ndarray,
    dict_const_off: np.ndarray,
    dict_mask_off: np.ndarray,
    dict_meta: JetMeta,
    target_k: int,
    per_round: int,
    max_rounds: int,
    eps_total: float,
    eps_count: float,
    w_desc: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    max_constits: int,
    batch_size: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    c0 = _retrieve_pool_for_class(
        query_index_desc=query_index_desc,
        query_score_desc=query_score_desc,
        query_meta=query_meta,
        query_const_hlt=query_const_hlt,
        query_mask_hlt=query_mask_hlt,
        class_index=idx_bg,
        dict_score_desc=dict_score_desc,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_meta=dict_meta,
        target_k=int(target_k),
        per_round=int(per_round),
        max_rounds=int(max_rounds),
        eps_total=float(eps_total),
        eps_count=float(eps_count),
        w_desc=float(w_desc),
        w_count=float(w_count),
        w_pt=float(w_pt),
        w_mass=float(w_mass),
        max_constits=int(max_constits),
        batch_size=int(batch_size),
    )
    c1 = _retrieve_pool_for_class(
        query_index_desc=query_index_desc,
        query_score_desc=query_score_desc,
        query_meta=query_meta,
        query_const_hlt=query_const_hlt,
        query_mask_hlt=query_mask_hlt,
        class_index=idx_top,
        dict_score_desc=dict_score_desc,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_meta=dict_meta,
        target_k=int(target_k),
        per_round=int(per_round),
        max_rounds=int(max_rounds),
        eps_total=float(eps_total),
        eps_count=float(eps_count),
        w_desc=float(w_desc),
        w_count=float(w_count),
        w_pt=float(w_pt),
        w_mass=float(w_mass),
        max_constits=int(max_constits),
        batch_size=int(batch_size),
    )

    return {
        "class0": c0,
        "class1": c1,
        "stats": {
            "target_k": int(target_k),
            "class0_mean_feasible": float(np.mean(np.minimum(c0["feasible_count"], int(target_k)))),
            "class1_mean_feasible": float(np.mean(np.minimum(c1["feasible_count"], int(target_k)))),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m36 dictionary-retrieval dualview top tagging")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model36_dictretrieval_dualview")
    p.add_argument("--run_name", type=str, default="model36_dictretrieval_dualview_1m150k75k300k_seed0")

    p.add_argument("--n_train_jets", type=int, default=1525000)
    p.add_argument("--n_dict_split", type=int, default=1000000)
    p.add_argument("--n_train_split", type=int, default=150000)
    p.add_argument("--n_val_split", type=int, default=75000)
    p.add_argument("--n_test_split", type=int, default=300000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--use_train_weights", action="store_true")

    # HLT effects (deterministic keyed D_hard)
    p.add_argument("--merge_radius", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["merge_radius"]))
    p.add_argument("--eff_plateau_barrel", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"]))
    p.add_argument("--eff_plateau_endcap", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"]))
    p.add_argument("--smear_a", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_a"]))
    p.add_argument("--smear_b", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_b"]))
    p.add_argument("--smear_c", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_c"]))
    p.add_argument("--dhard_seed_offset", type=int, default=1337)

    # Teacher/baseline
    p.add_argument("--cls_epochs", type=int, default=60)
    p.add_argument("--cls_patience", type=int, default=12)
    p.add_argument("--cls_lr", type=float, default=3e-4)
    p.add_argument("--cls_weight_decay", type=float, default=1e-4)
    p.add_argument("--cls_warmup_epochs", type=int, default=3)

    # Retrieval candidate stage
    p.add_argument("--retrieval_target_k", type=int, default=3)
    p.add_argument("--retrieval_per_round", type=int, default=256)
    p.add_argument("--retrieval_max_rounds", type=int, default=10)
    p.add_argument("--retrieval_batch_size", type=int, default=256)
    p.add_argument("--retrieval_eps_total", type=float, default=0.90)
    p.add_argument("--retrieval_eps_count", type=float, default=0.50)
    p.add_argument("--retrieval_w_desc", type=float, default=1.00)
    p.add_argument("--retrieval_w_count", type=float, default=0.25)
    p.add_argument("--retrieval_w_pt", type=float, default=0.12)
    p.add_argument("--retrieval_w_mass", type=float, default=0.08)
    p.add_argument("--retrieval_descriptor_variant", type=str, default="physics", choices=["physics", "landmark11"])
    p.add_argument("--retrieval_landmarks", type=int, default=8)
    p.add_argument("--retrieval_index_mode", type=str, default="descriptor", choices=["descriptor", "learned"])
    p.add_argument("--retrieval_embed_dim", type=int, default=16)
    p.add_argument("--retrieval_embed_hidden", type=int, default=96)
    p.add_argument("--retrieval_embed_dropout", type=float, default=0.10)
    p.add_argument("--retrieval_embed_epochs", type=int, default=6)
    p.add_argument("--retrieval_embed_batch_size", type=int, default=512)
    p.add_argument("--retrieval_embed_pool_size", type=int, default=256)
    p.add_argument("--retrieval_embed_train_anchors", type=int, default=60000)
    p.add_argument("--retrieval_embed_lr", type=float, default=3e-4)
    p.add_argument("--retrieval_embed_weight_decay", type=float, default=1e-4)
    p.add_argument("--retrieval_embed_margin", type=float, default=0.20)

    # Selector
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--selector_epochs", type=int, default=40)
    p.add_argument("--selector_patience", type=int, default=8)
    p.add_argument("--selector_lr", type=float, default=2e-4)
    p.add_argument("--selector_weight_decay", type=float, default=1e-4)
    p.add_argument("--selector_neg_per_class", type=int, default=3)
    p.add_argument("--selector_score_alpha", type=float, default=1.25)
    p.add_argument("--selector_mode", type=str, default="compat_selector", choices=["compat_selector", "residual_only"])

    # Final dualview classifiers
    p.add_argument("--dual_epochs", type=int, default=60)
    p.add_argument("--dual_patience", type=int, default=12)
    p.add_argument("--dual_lr", type=float, default=1.5e-4)
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
    print("Model-36 Dictionary Retrieval + DualView Pipeline")
    print(f"Run: {save_root}")
    print(
        f"Split dict/train/val/test = {int(args.n_dict_split)}/{int(args.n_train_split)}/{int(args.n_val_split)}/{int(args.n_test_split)} | "
        f"retrieve per-round={int(args.retrieval_per_round)} rounds={int(args.retrieval_max_rounds)} targetK={int(args.retrieval_target_k)} | "
        f"desc_variant={str(args.retrieval_descriptor_variant)} index_mode={str(args.retrieval_index_mode)}"
    )
    print("=" * 72)

    # Load offline jets.
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

    const_raw = all_const[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)
    train_w = all_train_w[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.float32)

    total_need = int(args.n_dict_split + args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > const_raw.shape[0]:
        raise RuntimeError(
            f"Requested splits sum to {total_need} but loaded jets are only {const_raw.shape[0]}"
        )

    raw_mask = const_raw[:, :, 0] > 0.0
    cfg = base._deepcopy_config()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT (deterministic keyed D_hard)...")
    jet_keys = (np.arange(len(const_off), dtype=np.int64) + int(args.offset_jets)).astype(np.int64)
    const_hlt, mask_hlt, hlt_stats = m33._apply_hlt_effects_deterministic_keyed(
        const=const_off,
        mask=masks_off,
        cfg=cfg,
        jet_keys=jet_keys,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    print(
        "HLT stats: "
        f"avg_offline={hlt_stats.get('avg_offline_per_jet', float('nan')):.2f}, "
        f"avg_hlt={hlt_stats.get('avg_hlt_per_jet', float('nan')):.2f}, "
        f"merged={hlt_stats.get('n_merged_pairs', 0)}, eff_lost={hlt_stats.get('n_lost_eff', 0)}"
    )

    # Split: dictionary disjoint from train/val/test.
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

    dict_idx, rem_idx = train_test_split(
        idx_use,
        train_size=int(args.n_dict_split),
        random_state=int(args.seed),
        stratify=labels[idx_use],
    )
    train_idx, rem2_idx = train_test_split(
        rem_idx,
        train_size=int(args.n_train_split),
        random_state=int(args.seed),
        stratify=labels[rem_idx],
    )
    val_idx, test_idx = train_test_split(
        rem2_idx,
        train_size=int(args.n_val_split),
        test_size=int(args.n_test_split),
        random_state=int(args.seed),
        stratify=labels[rem2_idx],
    )

    print(
        f"Split sizes: Dict={len(dict_idx)}, Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )

    # Build features only for train/val/test (dictionary is retrieval only).
    print("Computing standardized features for teacher/baseline (train/val/test only)...")
    feat_off_tr = compute_features(const_off[train_idx], masks_off[train_idx])
    feat_off_va = compute_features(const_off[val_idx], masks_off[val_idx])
    feat_off_te = compute_features(const_off[test_idx], masks_off[test_idx])
    feat_hlt_tr = compute_features(const_hlt[train_idx], mask_hlt[train_idx])
    feat_hlt_va = compute_features(const_hlt[val_idx], mask_hlt[val_idx])
    feat_hlt_te = compute_features(const_hlt[test_idx], mask_hlt[test_idx])

    tr_local_idx = np.arange(feat_off_tr.shape[0], dtype=np.int64)
    means, stds = get_stats(feat_off_tr, masks_off[train_idx], tr_local_idx)

    feat_off_tr = standardize(feat_off_tr, masks_off[train_idx], means, stds)
    feat_off_va = standardize(feat_off_va, masks_off[val_idx], means, stds)
    feat_off_te = standardize(feat_off_te, masks_off[test_idx], means, stds)
    feat_hlt_tr = standardize(feat_hlt_tr, mask_hlt[train_idx], means, stds)
    feat_hlt_va = standardize(feat_hlt_va, mask_hlt[val_idx], means, stds)
    feat_hlt_te = standardize(feat_hlt_te, mask_hlt[test_idx], means, stds)

    # STEP 1: teacher and baseline classifiers.
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

    sw_train = train_w[train_idx] if bool(args.use_train_weights) else np.ones((len(train_idx),), dtype=np.float32)
    sw_val = train_w[val_idx] if bool(args.use_train_weights) else np.ones((len(val_idx),), dtype=np.float32)
    sw_test = train_w[test_idx] if bool(args.use_train_weights) else np.ones((len(test_idx),), dtype=np.float32)

    ds_tr_off = base.WeightedJetDataset(feat_off_tr, masks_off[train_idx], labels[train_idx], sw_train)
    ds_va_off = base.WeightedJetDataset(feat_off_va, masks_off[val_idx], labels[val_idx], sw_val)
    ds_te_off = base.WeightedJetDataset(feat_off_te, masks_off[test_idx], labels[test_idx], sw_test)

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

    # STEP 2: Build dictionary retrieval index (class-conditional true labels).
    print("\n" + "=" * 72)
    print("STEP 2: Dictionary HLT retrieval index")
    print("=" * 72)

    dict_const_off = const_off[dict_idx].astype(np.float32)
    dict_mask_off = masks_off[dict_idx].astype(bool)
    dict_const_hlt = const_hlt[dict_idx].astype(np.float32)
    dict_mask_hlt = mask_hlt[dict_idx].astype(bool)
    dict_labels = labels[dict_idx].astype(np.int64)

    dict_meta = _jet_meta_np(dict_const_hlt, dict_mask_hlt)
    tr_meta = _jet_meta_np(const_hlt[train_idx], mask_hlt[train_idx])
    va_meta = _jet_meta_np(const_hlt[val_idx], mask_hlt[val_idx])
    te_meta = _jet_meta_np(const_hlt[test_idx], mask_hlt[test_idx])

    desc_dict_raw = _build_retrieval_desc(dict_const_hlt, dict_mask_hlt, dict_meta, int(args.max_constits))
    desc_tr_raw = _build_retrieval_desc(const_hlt[train_idx], mask_hlt[train_idx], tr_meta, int(args.max_constits))
    desc_va_raw = _build_retrieval_desc(const_hlt[val_idx], mask_hlt[val_idx], va_meta, int(args.max_constits))
    desc_te_raw = _build_retrieval_desc(const_hlt[test_idx], mask_hlt[test_idx], te_meta, int(args.max_constits))

    desc_mean = desc_dict_raw.mean(axis=0, keepdims=True)
    desc_std = desc_dict_raw.std(axis=0, keepdims=True) + 1e-6

    desc_dict = ((desc_dict_raw - desc_mean) / desc_std).astype(np.float32)
    desc_tr = ((desc_tr_raw - desc_mean) / desc_std).astype(np.float32)
    desc_va = ((desc_va_raw - desc_mean) / desc_std).astype(np.float32)
    desc_te = ((desc_te_raw - desc_mean) / desc_std).astype(np.float32)

    retrieval_variant = str(args.retrieval_descriptor_variant).lower()
    landmark_ids = np.zeros((0,), dtype=np.int64)
    landmark_desc = np.zeros((0, desc_dict.shape[1]), dtype=np.float32)
    landmark_meta: Dict[str, object] = {"variant": retrieval_variant}
    if retrieval_variant == "landmark11":
        n_landmarks = int(max(1, args.retrieval_landmarks))
        landmark_ids = _select_landmark_indices(desc_dict, n_landmarks=n_landmarks, seed=int(args.seed) + 173)
        landmark_desc = desc_dict[landmark_ids].astype(np.float32)
        retrieval_score_dict = _build_landmark11_desc(desc_dict, dict_meta, landmark_desc, int(args.max_constits))
        retrieval_score_tr = _build_landmark11_desc(desc_tr, tr_meta, landmark_desc, int(args.max_constits))
        retrieval_score_va = _build_landmark11_desc(desc_va, va_meta, landmark_desc, int(args.max_constits))
        retrieval_score_te = _build_landmark11_desc(desc_te, te_meta, landmark_desc, int(args.max_constits))
        landmark_meta = {
            "variant": "landmark11",
            "n_landmarks": int(landmark_desc.shape[0]),
            "landmark_ids": landmark_ids.astype(np.int64).tolist(),
        }
        print(f"Landmark11 descriptor enabled: n_landmarks={int(landmark_desc.shape[0])}")
    else:
        retrieval_score_dict = desc_dict
        retrieval_score_tr = desc_tr
        retrieval_score_va = desc_va
        retrieval_score_te = desc_te

    score_mean = retrieval_score_dict.mean(axis=0, keepdims=True).astype(np.float32)
    score_std = (retrieval_score_dict.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    retrieval_embedder = None
    retrieval_embed_metrics: Dict[str, object] = {}
    idx_dict_desc = retrieval_score_dict
    idx_tr_desc = retrieval_score_tr
    idx_va_desc = retrieval_score_va
    idx_te_desc = retrieval_score_te

    if str(args.retrieval_index_mode).lower() == "learned":
        print("Training learned retrieval embedding index...")
        retrieval_embedder, retrieval_embed_metrics = _train_retrieval_embedder(
            desc_train=retrieval_score_tr,
            meta_train=tr_meta,
            labels_train=labels[train_idx],
            desc_dict=retrieval_score_dict,
            meta_dict=dict_meta,
            dict_labels=dict_labels,
            max_constits=int(args.max_constits),
            w_desc=float(args.retrieval_w_desc),
            w_count=float(args.retrieval_w_count),
            w_pt=float(args.retrieval_w_pt),
            w_mass=float(args.retrieval_w_mass),
            seed=int(args.seed),
            device=device,
            embed_dim=int(args.retrieval_embed_dim),
            hidden_dim=int(args.retrieval_embed_hidden),
            dropout=float(args.retrieval_embed_dropout),
            epochs=int(args.retrieval_embed_epochs),
            batch_size=int(args.retrieval_embed_batch_size),
            pool_size=int(args.retrieval_embed_pool_size),
            train_anchors=int(args.retrieval_embed_train_anchors),
            lr=float(args.retrieval_embed_lr),
            weight_decay=float(args.retrieval_embed_weight_decay),
            margin=float(args.retrieval_embed_margin),
        )
        idx_dict_desc = _encode_retrieval_desc(
            retrieval_embedder,
            retrieval_score_dict,
            batch_size=int(args.retrieval_batch_size),
            device=device,
        )
        idx_tr_desc = _encode_retrieval_desc(
            retrieval_embedder,
            retrieval_score_tr,
            batch_size=int(args.retrieval_batch_size),
            device=device,
        )
        idx_va_desc = _encode_retrieval_desc(
            retrieval_embedder,
            retrieval_score_va,
            batch_size=int(args.retrieval_batch_size),
            device=device,
        )
        idx_te_desc = _encode_retrieval_desc(
            retrieval_embedder,
            retrieval_score_te,
            batch_size=int(args.retrieval_batch_size),
            device=device,
        )
    else:
        retrieval_embed_metrics = {"mode": "descriptor", "epochs": 0, "descriptor_variant": retrieval_variant}

    idx_bg = _build_class_index(idx_dict_desc, dict_labels, class_val=0)
    idx_top = _build_class_index(idx_dict_desc, dict_labels, class_val=1)
    print(
        f"Dictionary class sizes: bg={idx_bg.local_ids.shape[0]} top={idx_top.local_ids.shape[0]} "
        f"| index_dim={idx_dict_desc.shape[1]}"
    )

    # STEP 3: Retrieve candidates for train/val.
    print("\n" + "=" * 72)
    print("STEP 3: Retrieve candidates for train/val (no repair)")
    print("=" * 72)

    pools_train = _retrieve_pools_split(
        query_index_desc=idx_tr_desc,
        query_score_desc=retrieval_score_tr,
        query_meta=tr_meta,
        query_const_hlt=const_hlt[train_idx],
        query_mask_hlt=mask_hlt[train_idx],
        idx_bg=idx_bg,
        idx_top=idx_top,
        dict_score_desc=retrieval_score_dict,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_meta=dict_meta,
        target_k=int(args.retrieval_target_k),
        per_round=int(args.retrieval_per_round),
        max_rounds=int(args.retrieval_max_rounds),
        eps_total=float(args.retrieval_eps_total),
        eps_count=float(args.retrieval_eps_count),
        w_desc=float(args.retrieval_w_desc),
        w_count=float(args.retrieval_w_count),
        w_pt=float(args.retrieval_w_pt),
        w_mass=float(args.retrieval_w_mass),
        max_constits=int(args.max_constits),
        batch_size=int(args.retrieval_batch_size),
    )
    pools_val = _retrieve_pools_split(
        query_index_desc=idx_va_desc,
        query_score_desc=retrieval_score_va,
        query_meta=va_meta,
        query_const_hlt=const_hlt[val_idx],
        query_mask_hlt=mask_hlt[val_idx],
        idx_bg=idx_bg,
        idx_top=idx_top,
        dict_score_desc=retrieval_score_dict,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_meta=dict_meta,
        target_k=int(args.retrieval_target_k),
        per_round=int(args.retrieval_per_round),
        max_rounds=int(args.retrieval_max_rounds),
        eps_total=float(args.retrieval_eps_total),
        eps_count=float(args.retrieval_eps_count),
        w_desc=float(args.retrieval_w_desc),
        w_count=float(args.retrieval_w_count),
        w_pt=float(args.retrieval_w_pt),
        w_mass=float(args.retrieval_w_mass),
        max_constits=int(args.max_constits),
        batch_size=int(args.retrieval_batch_size),
    )

    print(
        "Retrieval stats: "
        f"train(feasC0={pools_train['stats']['class0_mean_feasible']:.2f}, feasC1={pools_train['stats']['class1_mean_feasible']:.2f}) "
        f"val(feasC0={pools_val['stats']['class0_mean_feasible']:.2f}, feasC1={pools_val['stats']['class1_mean_feasible']:.2f})"
    )

    # STEP 4: Candidate compatibility selector (optional).
    print("\n" + "=" * 72)
    if str(args.selector_mode).lower() == "compat_selector":
        print("STEP 4: Train HLT-candidate compatibility selector")
    else:
        print("STEP 4: Residual-only candidate quality (selector skipped)")
    print("=" * 72)

    selector = None
    if str(args.selector_mode).lower() == "compat_selector":
        sel_tr = m33._build_selector_arrays(
            const_hlt=const_hlt[train_idx],
            mask_hlt=mask_hlt[train_idx],
            const_off=const_off[train_idx],
            mask_off=masks_off[train_idx],
            pools=pools_train,
            neg_per_class=int(args.selector_neg_per_class),
        )
        sel_va = m33._build_selector_arrays(
            const_hlt=const_hlt[val_idx],
            mask_hlt=mask_hlt[val_idx],
            const_off=const_off[val_idx],
            mask_off=masks_off[val_idx],
            pools=pools_val,
            neg_per_class=int(args.selector_neg_per_class),
        )

        ds_sel_tr = m33.SelectorDataset(**sel_tr)
        ds_sel_va = m33.SelectorDataset(**sel_va)
        dl_sel_tr = DataLoader(ds_sel_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
        dl_sel_va = DataLoader(ds_sel_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

        selector = m33.CandidateRealismSelector(
            embed_dim=int(args.embed_dim),
            num_heads=int(args.num_heads),
            num_layers=max(2, int(args.num_layers // 2)),
            ff_dim=int(args.ff_dim),
            dropout=float(args.dropout),
        ).to(device)
        selector, selector_metrics = m33._train_selector(
            model=selector,
            train_loader=dl_sel_tr,
            val_loader=dl_sel_va,
            device=device,
            epochs=int(args.selector_epochs),
            lr=float(args.selector_lr),
            weight_decay=float(args.selector_weight_decay),
            patience=int(args.selector_patience),
        )

        sel_bg_train = m33._score_selector_candidates(
            selector,
            const_hlt[train_idx],
            mask_hlt[train_idx],
            pools_train["class0"]["const"],
            pools_train["class0"]["mask"],
            pools_train["class0"]["res_total"],
            cand_class_val=0,
            batch_size=int(args.batch_size),
            device=device,
        )
        sel_tp_train = m33._score_selector_candidates(
            selector,
            const_hlt[train_idx],
            mask_hlt[train_idx],
            pools_train["class1"]["const"],
            pools_train["class1"]["mask"],
            pools_train["class1"]["res_total"],
            cand_class_val=1,
            batch_size=int(args.batch_size),
            device=device,
        )
        sel_bg_val = m33._score_selector_candidates(
            selector,
            const_hlt[val_idx],
            mask_hlt[val_idx],
            pools_val["class0"]["const"],
            pools_val["class0"]["mask"],
            pools_val["class0"]["res_total"],
            cand_class_val=0,
            batch_size=int(args.batch_size),
            device=device,
        )
        sel_tp_val = m33._score_selector_candidates(
            selector,
            const_hlt[val_idx],
            mask_hlt[val_idx],
            pools_val["class1"]["const"],
            pools_val["class1"]["mask"],
            pools_val["class1"]["res_total"],
            cand_class_val=1,
            batch_size=int(args.batch_size),
            device=device,
        )
    else:
        selector_metrics = {
            "mode": "residual_only",
            "skipped": 1,
            "best_val_auc": float("nan"),
            "best_epoch": 0,
        }
        sel_bg_train = np.zeros_like(pools_train["class0"]["res_total"], dtype=np.float32)
        sel_tp_train = np.zeros_like(pools_train["class1"]["res_total"], dtype=np.float32)
        sel_bg_val = np.zeros_like(pools_val["class0"]["res_total"], dtype=np.float32)
        sel_tp_val = np.zeros_like(pools_val["class1"]["res_total"], dtype=np.float32)

    dv_tr = m33._build_dualview_features(
        pools=pools_train,
        sel_score_bg=sel_bg_train,
        sel_score_top=sel_tp_train,
        baseline_prob=np.zeros((len(train_idx),), dtype=np.float32),
        score_alpha=float(args.selector_score_alpha),
    )
    dv_va = m33._build_dualview_features(
        pools=pools_val,
        sel_score_bg=sel_bg_val,
        sel_score_top=sel_tp_val,
        baseline_prob=np.zeros((len(val_idx),), dtype=np.float32),
        score_alpha=float(args.selector_score_alpha),
    )

    # STEP 5: Final dualview classifiers (no-gate and gated).
    print("\n" + "=" * 72)
    print("STEP 5: Final DualView Classifiers (NoGate + Gated)")
    print("=" * 72)

    ds_dv_tr = m33.DualViewCandidateDataset(
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        cand_top_const=dv_tr["cand_top_const"],
        cand_top_mask=dv_tr["cand_top_mask"],
        cand_bg_const=dv_tr["cand_bg_const"],
        cand_bg_mask=dv_tr["cand_bg_mask"],
        cand_feat=dv_tr["cand_feat"],
        labels=labels[train_idx],
        sample_weight=sw_train,
    )
    ds_dv_va = m33.DualViewCandidateDataset(
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        cand_top_const=dv_va["cand_top_const"],
        cand_top_mask=dv_va["cand_top_mask"],
        cand_bg_const=dv_va["cand_bg_const"],
        cand_bg_mask=dv_va["cand_bg_mask"],
        cand_feat=dv_va["cand_feat"],
        labels=labels[val_idx],
        sample_weight=sw_val,
    )

    dl_dv_tr = DataLoader(ds_dv_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_dv_va = DataLoader(ds_dv_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    dv_nogate = m33.DualViewNoGateClassifier(
        cand_feat_dim=int(dv_tr["cand_feat"].shape[1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers // 2)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    dv_nogate, dv_nogate_metrics = m33._train_dualview_model(
        model=dv_nogate,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="DualViewNoGate",
    )

    dv_gated = m33.DualViewGatedClassifier(
        cand_feat_dim=int(dv_tr["cand_feat"].shape[1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=max(2, int(args.num_layers // 2)),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    dv_gated, dv_gated_metrics = m33._train_dualview_model(
        model=dv_gated,
        train_loader=dl_dv_tr,
        val_loader=dl_dv_va,
        device=device,
        epochs=int(args.dual_epochs),
        lr=float(args.dual_lr),
        weight_decay=float(args.dual_weight_decay),
        patience=int(args.dual_patience),
        name="DualViewGated",
    )

    # STEP 6: Build test pools and evaluate final models.
    print("\n" + "=" * 72)
    print("STEP 6: Test retrieval + final evaluation")
    print("=" * 72)

    pools_test = _retrieve_pools_split(
        query_index_desc=idx_te_desc,
        query_score_desc=retrieval_score_te,
        query_meta=te_meta,
        query_const_hlt=const_hlt[test_idx],
        query_mask_hlt=mask_hlt[test_idx],
        idx_bg=idx_bg,
        idx_top=idx_top,
        dict_score_desc=retrieval_score_dict,
        dict_const_off=dict_const_off,
        dict_mask_off=dict_mask_off,
        dict_meta=dict_meta,
        target_k=int(args.retrieval_target_k),
        per_round=int(args.retrieval_per_round),
        max_rounds=int(args.retrieval_max_rounds),
        eps_total=float(args.retrieval_eps_total),
        eps_count=float(args.retrieval_eps_count),
        w_desc=float(args.retrieval_w_desc),
        w_count=float(args.retrieval_w_count),
        w_pt=float(args.retrieval_w_pt),
        w_mass=float(args.retrieval_w_mass),
        max_constits=int(args.max_constits),
        batch_size=int(args.retrieval_batch_size),
    )

    if selector is not None:
        sel_bg_test = m33._score_selector_candidates(
            selector,
            const_hlt[test_idx],
            mask_hlt[test_idx],
            pools_test["class0"]["const"],
            pools_test["class0"]["mask"],
            pools_test["class0"]["res_total"],
            cand_class_val=0,
            batch_size=int(args.batch_size),
            device=device,
        )
        sel_tp_test = m33._score_selector_candidates(
            selector,
            const_hlt[test_idx],
            mask_hlt[test_idx],
            pools_test["class1"]["const"],
            pools_test["class1"]["mask"],
            pools_test["class1"]["res_total"],
            cand_class_val=1,
            batch_size=int(args.batch_size),
            device=device,
        )
    else:
        sel_bg_test = np.zeros_like(pools_test["class0"]["res_total"], dtype=np.float32)
        sel_tp_test = np.zeros_like(pools_test["class1"]["res_total"], dtype=np.float32)

    dv_te = m33._build_dualview_features(
        pools=pools_test,
        sel_score_bg=sel_bg_test,
        sel_score_top=sel_tp_test,
        baseline_prob=np.zeros((len(test_idx),), dtype=np.float32),
        score_alpha=float(args.selector_score_alpha),
    )

    ds_dv_te = m33.DualViewCandidateDataset(
        const_hlt=const_hlt[test_idx],
        mask_hlt=mask_hlt[test_idx],
        cand_top_const=dv_te["cand_top_const"],
        cand_top_mask=dv_te["cand_top_mask"],
        cand_bg_const=dv_te["cand_bg_const"],
        cand_bg_mask=dv_te["cand_bg_mask"],
        cand_feat=dv_te["cand_feat"],
        labels=labels[test_idx],
        sample_weight=sw_test,
    )
    dl_dv_te = DataLoader(ds_dv_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    auc_nog, fpr50_nog, pred_nog, lab_final, w_final = m33._eval_dualview_model(dv_nogate, dl_dv_te, device)
    auc_gat, fpr50_gat, pred_gat, _lab2, _w2 = m33._eval_dualview_model(dv_gated, dl_dv_te, device)

    fpr_t, tpr_t, _ = roc_curve(teacher_y_test, teacher_p_test, sample_weight=teacher_w_test)
    fpr_b, tpr_b, _ = roc_curve(baseline_y_test, baseline_p_test, sample_weight=baseline_w_test)
    fpr50_teacher = m33.fpr_at_target_tpr(fpr_t, tpr_t, 0.50)
    fpr50_baseline = m33.fpr_at_target_tpr(fpr_b, tpr_b, 0.50)

    print("\n" + "=" * 72)
    print("FINAL TEST")
    print("=" * 72)
    print(
        f"Teacher AUC={teacher_auc_test:.4f} FPR50={fpr50_teacher:.6f} | "
        f"HLT baseline AUC={baseline_auc_test:.4f} FPR50={fpr50_baseline:.6f} | "
        f"m36 NoGate AUC={auc_nog:.4f} FPR50={fpr50_nog:.6f} | "
        f"m36 Gated AUC={auc_gat:.4f} FPR50={fpr50_gat:.6f}"
    )

    # Save artifacts.
    torch.save({"model": teacher.state_dict(), "auc_test": float(teacher_auc_test)}, save_root / "teacher.pt")
    torch.save({"model": baseline.state_dict(), "auc_test": float(baseline_auc_test)}, save_root / "baseline_hlt.pt")
    if selector is not None:
        torch.save({"model": selector.state_dict(), "metrics": selector_metrics}, save_root / "selector.pt")
    torch.save({"model": dv_nogate.state_dict(), "metrics": dv_nogate_metrics}, save_root / "dualview_nogate.pt")
    torch.save({"model": dv_gated.state_dict(), "metrics": dv_gated_metrics}, save_root / "dualview_gated.pt")
    if retrieval_embedder is not None:
        torch.save(
            {
                "model": retrieval_embedder.state_dict(),
                "metrics": retrieval_embed_metrics,
                "index_mode": str(args.retrieval_index_mode),
                "descriptor_variant": str(args.retrieval_descriptor_variant),
            },
            save_root / "retrieval_embedder.pt",
        )

    np.savez_compressed(
        save_root / "m36_test_scores.npz",
        labels_test=lab_final.astype(np.float32),
        preds_m36_nogate=pred_nog.astype(np.float32),
        preds_m36_gated=pred_gat.astype(np.float32),
        preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
        preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
        sample_weight=np.asarray(w_final, dtype=np.float32),
        auc_teacher=float(teacher_auc_test),
        auc_hlt=float(baseline_auc_test),
        auc_m36_nogate=float(auc_nog),
        auc_m36_gated=float(auc_gat),
        fpr50_teacher=float(fpr50_teacher),
        fpr50_hlt=float(fpr50_baseline),
        fpr50_m36_nogate=float(fpr50_nog),
        fpr50_m36_gated=float(fpr50_gat),
    )

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "fusion_scores_test.npz",
            labels_test=lab_final.astype(np.float32),
            preds_teacher=np.asarray(teacher_p_test, dtype=np.float32),
            preds_hlt=np.asarray(baseline_p_test, dtype=np.float32),
            preds_m36_nogate=np.asarray(pred_nog, dtype=np.float32),
            preds_m36_gated=np.asarray(pred_gat, dtype=np.float32),
            sample_weight=np.asarray(w_final, dtype=np.float32),
        )

    report = {
        "model": "m36_dictretrieval_dualview",
        "seed": int(args.seed),
        "n_train_jets": int(args.n_train_jets),
        "split": {
            "dictionary": int(len(dict_idx)),
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
        "m36_nogate": {
            "auc_test": float(auc_nog),
            "fpr50_test": float(fpr50_nog),
            "metrics": dv_nogate_metrics,
        },
        "m36_gated": {
            "auc_test": float(auc_gat),
            "fpr50_test": float(fpr50_gat),
            "metrics": dv_gated_metrics,
        },
        "selector_metrics": selector_metrics,
        "retrieval": {
            "index_mode": str(args.retrieval_index_mode),
            "descriptor_variant": str(args.retrieval_descriptor_variant),
            "target_k": int(args.retrieval_target_k),
            "per_round": int(args.retrieval_per_round),
            "max_rounds": int(args.retrieval_max_rounds),
            "eps_total": float(args.retrieval_eps_total),
            "eps_count": float(args.retrieval_eps_count),
            "weights": {
                "desc": float(args.retrieval_w_desc),
                "count": float(args.retrieval_w_count),
                "pt": float(args.retrieval_w_pt),
                "mass": float(args.retrieval_w_mass),
            },
            "train": pools_train["stats"],
            "val": pools_val["stats"],
            "test": pools_test["stats"],
            "dict_class_counts": {
                "bg": int(idx_bg.local_ids.shape[0]),
                "top": int(idx_top.local_ids.shape[0]),
            },
            "embedder_metrics": retrieval_embed_metrics,
            "landmark_meta": landmark_meta,
        },
        "selector_mode": str(args.selector_mode),
    }
    with open(save_root / "m36_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        save_root / "data_splits.npz",
        dict_idx=dict_idx.astype(np.int64),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
        retrieval_desc_mean=desc_mean.astype(np.float32),
        retrieval_desc_std=desc_std.astype(np.float32),
        retrieval_score_mean=score_mean.astype(np.float32),
        retrieval_score_std=score_std.astype(np.float32),
        retrieval_landmark_ids=landmark_ids.astype(np.int64),
        retrieval_landmark_desc=landmark_desc.astype(np.float32),
    )

    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()

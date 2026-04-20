#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m33 D_hard / D_soft probe

Goals:
1) Verify deterministic keyed HLT generation (D_hard).
2) Characterize local smoothness/sensitivity of D_hard under small offline perturbations.
3) Train/evaluate D_soft surrogate (offline->HLT degrader) against deterministic D_hard data.
4) Measure D_soft ranking and calibration quality vs D_hard over candidate pools.
5) Evaluate post-acceptance refinement utility:
     refine candidate offline jets with D_soft objective, then score improvement by D_hard.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_detfeas_dualview as m33


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _wrap_phi(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def _raw_from_const(const: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    pt = const[..., 0].clamp(min=eps)
    eta = const[..., 1].clamp(min=-5.0, max=5.0)
    phi = const[..., 2]
    e = const[..., 3].clamp(min=eps)
    eta_scaled = (eta / 5.0).clamp(min=-0.999, max=0.999)
    eta_raw = 0.5 * torch.log((1.0 + eta_scaled) / (1.0 - eta_scaled))
    return torch.stack(
        [torch.log(pt), eta_raw, torch.sin(phi), torch.cos(phi), torch.log(e)],
        dim=-1,
    )


def _const_from_raw(raw: torch.Tensor) -> torch.Tensor:
    logpt = torch.clamp(raw[..., 0], min=-9.0, max=9.0)
    eta = 5.0 * torch.tanh(raw[..., 1])
    sinphi = raw[..., 2]
    cosphi = raw[..., 3]
    loge = torch.clamp(raw[..., 4], min=-9.0, max=11.0)

    pt = torch.exp(logpt)
    phi = torch.atan2(sinphi, cosphi)
    e = torch.exp(loge)
    e = torch.maximum(e, pt * torch.cosh(eta))
    return torch.stack([pt, eta, phi, e], dim=-1)


def _jet_mass_weighted(const: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ww = w.float()
    pt = const[..., 0] * ww
    eta = const[..., 1]
    phi = const[..., 2]
    e = const[..., 3] * ww
    px = (pt * torch.cos(phi)).sum(dim=1)
    py = (pt * torch.sin(phi)).sum(dim=1)
    pz = (pt * torch.sinh(eta)).sum(dim=1)
    et = e.sum(dim=1)
    m2 = et * et - px * px - py * py - pz * pz
    return torch.sqrt(torch.clamp(m2, min=0.0))


def _soft_residual_vec(
    pred_const: torch.Tensor,
    pred_w: torch.Tensor,
    tgt_const: torch.Tensor,
    tgt_mask: torch.Tensor,
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    unmatched_penalty: float,
) -> Dict[str, torch.Tensor]:
    set_v = m33._set_loss_chamfer_vec(
        pred_const=pred_const,
        pred_w=pred_w,
        tgt_const=tgt_const,
        tgt_mask=tgt_mask,
        unmatched_penalty=float(unmatched_penalty),
    )
    cnt_v = m33._count_loss_vec(pred_w, tgt_mask)

    pred_pt = (pred_const[..., 0] * pred_w).sum(dim=1)
    tgt_pt = (tgt_const[..., 0] * tgt_mask.float()).sum(dim=1)
    pt_v = torch.abs(pred_pt - tgt_pt) / (tgt_pt + 1e-6)

    pred_mass = _jet_mass_weighted(pred_const, pred_w)
    tgt_mass = m33._jet_mass_vec(tgt_const, tgt_mask)
    mass_v = torch.abs(pred_mass - tgt_mass) / (tgt_mass + 1e-6)

    total_v = (
        float(w_chamfer) * set_v
        + float(w_count) * cnt_v
        + float(w_pt) * pt_v
        + float(w_mass) * mass_v
    )
    return {"total": total_v, "set": set_v, "count": cnt_v, "pt": pt_v, "mass": mass_v}


def _summary(v: np.ndarray) -> Dict[str, float]:
    if v.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(v)),
        "p50": float(np.quantile(v, 0.50)),
        "p90": float(np.quantile(v, 0.90)),
    }


def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    sx = float(rx.std())
    sy = float(ry.std())
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    c = np.corrcoef(rx, ry)
    return float(c[0, 1])


def _apply_dhard_chunked(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: Dict,
    jet_keys: np.ndarray,
    base_seed: int,
    chunk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(const.shape[0])
    outs_c = []
    outs_m = []
    for s in range(0, n, int(chunk)):
        e = min(n, s + int(chunk))
        c, m, _ = m33._apply_hlt_effects_deterministic_keyed(
            const=const[s:e],
            mask=mask[s:e],
            cfg=cfg,
            jet_keys=jet_keys[s:e],
            base_seed=int(base_seed),
        )
        outs_c.append(c)
        outs_m.append(m)
    return np.concatenate(outs_c, axis=0), np.concatenate(outs_m, axis=0)


def _build_perturbed_candidates(
    const_off: np.ndarray,
    mask_off: np.ndarray,
    n_cand: int,
    rng: np.random.RandomState,
    pt_sigma: float,
    eta_sigma: float,
    phi_sigma: float,
    e_sigma: float,
    drop_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create candidate offline jets around true offline jets.
    cand[:,:,0] is exactly the original jet.
    """
    b, l, _ = const_off.shape
    k = int(max(2, n_cand))
    cand_c = np.zeros((b, k, l, 4), dtype=np.float32)
    cand_m = np.zeros((b, k, l), dtype=bool)

    cand_c[:, 0] = const_off
    cand_m[:, 0] = mask_off

    for i in range(b):
        c0 = const_off[i].copy()
        m0 = mask_off[i].copy()
        act_idx = np.where(m0)[0]
        for kk in range(1, k):
            c = c0.copy()
            m = m0.copy()
            if act_idx.size > 0:
                # Jitter active constituents.
                pt_mult = np.exp(rng.normal(0.0, pt_sigma, size=act_idx.size)).astype(np.float32)
                e_mult = np.exp(rng.normal(0.0, e_sigma, size=act_idx.size)).astype(np.float32)
                c[act_idx, 0] = np.maximum(1e-6, c[act_idx, 0] * pt_mult)
                c[act_idx, 1] = np.clip(c[act_idx, 1] + rng.normal(0.0, eta_sigma, size=act_idx.size), -5.0, 5.0)
                c[act_idx, 2] = _wrap_phi(c[act_idx, 2] + rng.normal(0.0, phi_sigma, size=act_idx.size))
                c[act_idx, 3] = np.maximum(1e-6, c[act_idx, 3] * e_mult)
                min_e = c[act_idx, 0] * np.cosh(c[act_idx, 1])
                c[act_idx, 3] = np.maximum(c[act_idx, 3], min_e)

                # Randomly drop some active constituents for diversity.
                drop_mask = rng.rand(act_idx.size) < float(drop_prob)
                if np.any(drop_mask):
                    di = act_idx[drop_mask]
                    m[di] = False
                    c[di] = 0.0

            cand_c[i, kk] = c
            cand_m[i, kk] = m
    return cand_c, cand_m


@torch.no_grad()
def _evaluate_degrader_split(
    model: m33.OfflineToHLTDegrader,
    loader: DataLoader,
    device: torch.device,
    pred_exist_threshold: float,
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    unmatched_penalty: float,
) -> Dict[str, Dict[str, float]]:
    soft_tot = []
    soft_set = []
    soft_cnt = []
    hard_tot = []
    hard_set = []
    hard_cnt = []

    model.eval()
    for batch in loader:
        co = batch["const_off"].to(device)
        mo = batch["mask_off"].to(device)
        ch = batch["const_hlt"].to(device)
        mh = batch["mask_hlt"].to(device)

        ph, pl = model(co, mo)
        pw = torch.sigmoid(pl)
        soft = _soft_residual_vec(
            pred_const=ph,
            pred_w=pw,
            tgt_const=ch,
            tgt_mask=mh,
            w_chamfer=float(w_chamfer),
            w_count=float(w_count),
            w_pt=float(w_pt),
            w_mass=float(w_mass),
            unmatched_penalty=float(unmatched_penalty),
        )
        pm = pw > float(pred_exist_threshold)
        hard = m33._residual_fast_vec(
            pred_const=ph,
            pred_mask=pm,
            tgt_const=ch,
            tgt_mask=mh,
            w_chamfer=float(w_chamfer),
            w_count=float(w_count),
            w_pt=float(w_pt),
            w_mass=float(w_mass),
            unmatched_penalty=float(unmatched_penalty),
        )
        soft_tot.append(soft["total"].detach().cpu().numpy())
        soft_set.append(soft["set"].detach().cpu().numpy())
        soft_cnt.append(soft["count"].detach().cpu().numpy())
        hard_tot.append(hard["total"].detach().cpu().numpy())
        hard_set.append(hard["set"].detach().cpu().numpy())
        hard_cnt.append(hard["count"].detach().cpu().numpy())

    soft_tot_np = np.concatenate(soft_tot, axis=0) if soft_tot else np.array([], dtype=np.float64)
    soft_set_np = np.concatenate(soft_set, axis=0) if soft_set else np.array([], dtype=np.float64)
    soft_cnt_np = np.concatenate(soft_cnt, axis=0) if soft_cnt else np.array([], dtype=np.float64)
    hard_tot_np = np.concatenate(hard_tot, axis=0) if hard_tot else np.array([], dtype=np.float64)
    hard_set_np = np.concatenate(hard_set, axis=0) if hard_set else np.array([], dtype=np.float64)
    hard_cnt_np = np.concatenate(hard_cnt, axis=0) if hard_cnt else np.array([], dtype=np.float64)
    return {
        "soft_total": _summary(soft_tot_np),
        "soft_set": _summary(soft_set_np),
        "soft_count": _summary(soft_cnt_np),
        "hard_total": _summary(hard_tot_np),
        "hard_set": _summary(hard_set_np),
        "hard_count": _summary(hard_cnt_np),
    }


def _refine_post_acceptance(
    degrader: m33.OfflineToHLTDegrader,
    cand_const: np.ndarray,
    cand_mask: np.ndarray,
    tgt_hlt_const: np.ndarray,
    tgt_hlt_mask: np.ndarray,
    device: torch.device,
    steps: int,
    step_size: float,
    max_step_norm: float,
    anchor_lambda: float,
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    unmatched_penalty: float,
) -> np.ndarray:
    """
    Refine offline candidates in constituent-space (mask fixed), guided by D_soft objective.
    """
    n, l, _ = cand_const.shape
    if n == 0 or int(steps) <= 0:
        return cand_const.copy()

    c0 = torch.tensor(cand_const, dtype=torch.float32, device=device)
    m0 = torch.tensor(cand_mask, dtype=torch.bool, device=device)
    th = torch.tensor(tgt_hlt_const, dtype=torch.float32, device=device)
    tm = torch.tensor(tgt_hlt_mask, dtype=torch.bool, device=device)

    raw = _raw_from_const(c0).detach()
    raw = raw.clone().requires_grad_(True)

    degrader.eval()
    for _ in range(int(steps)):
        const_now = _const_from_raw(raw)
        const_now = torch.where(m0.unsqueeze(-1), const_now, torch.zeros_like(const_now))

        ph, pl = degrader(const_now, m0)
        pw = torch.sigmoid(pl)
        soft = _soft_residual_vec(
            pred_const=ph,
            pred_w=pw,
            tgt_const=th,
            tgt_mask=tm,
            w_chamfer=float(w_chamfer),
            w_count=float(w_count),
            w_pt=float(w_pt),
            w_mass=float(w_mass),
            unmatched_penalty=float(unmatched_penalty),
        )
        anchor = m33._set_loss_chamfer_vec(
            pred_const=const_now,
            pred_w=m0.float(),
            tgt_const=c0,
            tgt_mask=m0,
            unmatched_penalty=float(unmatched_penalty),
        )
        loss = soft["total"].mean() + float(anchor_lambda) * anchor.mean()
        grad = torch.autograd.grad(loss, raw, only_inputs=True, create_graph=False)[0]
        grad = torch.where(m0.unsqueeze(-1), grad, torch.zeros_like(grad))

        step = -float(step_size) * grad
        if float(max_step_norm) > 0.0:
            sn = torch.sqrt(step.pow(2).sum(dim=(1, 2), keepdim=True) + 1e-8)
            clip = torch.clamp(float(max_step_norm) / sn, max=1.0)
            step = step * clip
        raw = (raw + step).detach().requires_grad_(True)

    out = _const_from_raw(raw.detach())
    out = torch.where(m0.unsqueeze(-1), out, torch.zeros_like(out))
    return out.detach().cpu().numpy().astype(np.float32)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m33 D_hard / D_soft probe")
    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model33_dhard_dsoft_probe")
    p.add_argument("--run_name", type=str, default="model33_dhard_dsoft_probe_debug_seed0")

    p.add_argument("--n_train_jets", type=int, default=70000)
    p.add_argument("--n_train_split", type=int, default=20000)
    p.add_argument("--n_val_split", type=int, default=8000)
    p.add_argument("--n_test_split", type=int, default=12000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=80)

    # D_hard config
    p.add_argument("--merge_radius", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["merge_radius"]))
    p.add_argument("--eff_plateau_barrel", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"]))
    p.add_argument("--eff_plateau_endcap", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"]))
    p.add_argument("--smear_a", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_a"]))
    p.add_argument("--smear_b", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_b"]))
    p.add_argument("--smear_c", type=float, default=float(base.BASE_CONFIG["hlt_effects"]["smear_c"]))
    p.add_argument("--dhard_seed_offset", type=int, default=1337)

    # Residual weights
    p.add_argument("--unmatched_penalty", type=float, default=0.0)
    p.add_argument("--pred_exist_threshold", type=float, default=0.08)
    p.add_argument("--res_w_chamfer", type=float, default=1.0)
    p.add_argument("--res_w_count", type=float, default=0.30)
    p.add_argument("--res_w_pt", type=float, default=0.12)
    p.add_argument("--res_w_mass", type=float, default=0.06)

    # Determinism / perturb tests
    p.add_argument("--det_eval_count", type=int, default=3000)
    p.add_argument("--det_chunk_size", type=int, default=97)
    p.add_argument("--pert_eval_count", type=int, default=2500)
    p.add_argument("--pert_pt_sigma", type=float, default=0.03)
    p.add_argument("--pert_eta_sigma", type=float, default=0.02)
    p.add_argument("--pert_phi_sigma", type=float, default=0.02)
    p.add_argument("--pert_e_sigma", type=float, default=0.03)
    p.add_argument("--pert_drop_prob", type=float, default=0.02)

    # D_soft degrader training
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--degrader_epochs", type=int, default=40)
    p.add_argument("--degrader_patience", type=int, default=8)
    p.add_argument("--degrader_lr", type=float, default=2e-4)
    p.add_argument("--degrader_weight_decay", type=float, default=1e-4)
    p.add_argument("--degrader_loss_w_count", type=float, default=0.30)

    # Candidate ranking / calibration
    p.add_argument("--cand_eval_count", type=int, default=1800)
    p.add_argument("--cand_per_jet", type=int, default=24)
    p.add_argument("--cand_pt_sigma", type=float, default=0.08)
    p.add_argument("--cand_eta_sigma", type=float, default=0.05)
    p.add_argument("--cand_phi_sigma", type=float, default=0.05)
    p.add_argument("--cand_e_sigma", type=float, default=0.08)
    p.add_argument("--cand_drop_prob", type=float, default=0.08)
    p.add_argument("--rank_top_m", type=int, default=8)
    p.add_argument("--rank_eval_batch_size", type=int, default=512)
    p.add_argument("--eps_total", type=float, default=0.20)
    p.add_argument("--eps_count", type=float, default=0.25)

    # Post-acceptance refinement probe
    p.add_argument("--refine_eval_jets", type=int, default=450)
    p.add_argument("--refine_selected_k", type=int, default=6)
    p.add_argument("--refine_steps", type=int, default=6)
    p.add_argument("--refine_lr", type=float, default=0.03)
    p.add_argument("--refine_max_step_norm", type=float, default=0.20)
    p.add_argument("--refine_anchor_lambda", type=float, default=0.10)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("m33 D_hard / D_soft probe")
    print(f"Run: {save_root}")
    print("=" * 72)

    # Data load.
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
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

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

    print("Building deterministic HLT from offline...")
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

    rng = np.random.RandomState(int(args.seed))

    # ------------------------------------------------------------------
    # Test 1: D_hard determinism
    # ------------------------------------------------------------------
    n_det = int(min(max(64, args.det_eval_count), len(val_idx)))
    det_idx = rng.choice(val_idx, size=n_det, replace=False)
    c_det = const_off[det_idx]
    m_det = masks_off[det_idx]
    k_det = jet_keys[det_idx]

    c1, m1, _ = m33._apply_hlt_effects_deterministic_keyed(
        const=c_det,
        mask=m_det,
        cfg=cfg,
        jet_keys=k_det,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    c2, m2, _ = m33._apply_hlt_effects_deterministic_keyed(
        const=c_det,
        mask=m_det,
        cfg=cfg,
        jet_keys=k_det,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    perm = rng.permutation(n_det)
    c3p, m3p, _ = m33._apply_hlt_effects_deterministic_keyed(
        const=c_det[perm],
        mask=m_det[perm],
        cfg=cfg,
        jet_keys=k_det[perm],
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    inv = np.argsort(perm)
    c3 = c3p[inv]
    m3 = m3p[inv]
    c4, m4 = _apply_dhard_chunked(
        const=c_det,
        mask=m_det,
        cfg=cfg,
        jet_keys=k_det,
        base_seed=int(args.seed + args.dhard_seed_offset),
        chunk=int(args.det_chunk_size),
    )
    det_report = {
        "n_eval": int(n_det),
        "same_order_const_exact": bool(np.array_equal(c1, c2)),
        "same_order_mask_exact": bool(np.array_equal(m1, m2)),
        "shuffled_order_const_exact": bool(np.array_equal(c1, c3)),
        "shuffled_order_mask_exact": bool(np.array_equal(m1, m3)),
        "chunked_const_exact": bool(np.array_equal(c1, c4)),
        "chunked_mask_exact": bool(np.array_equal(m1, m4)),
        "max_abs_diff_same_order": float(np.max(np.abs(c1 - c2))),
        "max_abs_diff_shuffled": float(np.max(np.abs(c1 - c3))),
        "max_abs_diff_chunked": float(np.max(np.abs(c1 - c4))),
    }

    # ------------------------------------------------------------------
    # Test 2: D_hard perturbation behavior
    # ------------------------------------------------------------------
    n_pert = int(min(max(64, args.pert_eval_count), len(val_idx)))
    pert_idx = rng.choice(val_idx, size=n_pert, replace=False)
    c0 = const_off[pert_idx].copy()
    m0 = masks_off[pert_idx].copy()
    k0 = jet_keys[pert_idx].copy()

    c_pert, m_pert = _build_perturbed_candidates(
        const_off=c0,
        mask_off=m0,
        n_cand=2,
        rng=rng,
        pt_sigma=float(args.pert_pt_sigma),
        eta_sigma=float(args.pert_eta_sigma),
        phi_sigma=float(args.pert_phi_sigma),
        e_sigma=float(args.pert_e_sigma),
        drop_prob=float(args.pert_drop_prob),
    )
    c_pert = c_pert[:, 1]
    m_pert = m_pert[:, 1]

    h0_c, h0_m, _ = m33._apply_hlt_effects_deterministic_keyed(
        const=c0, mask=m0, cfg=cfg, jet_keys=k0, base_seed=int(args.seed + args.dhard_seed_offset)
    )
    hp_c, hp_m, _ = m33._apply_hlt_effects_deterministic_keyed(
        const=c_pert, mask=m_pert, cfg=cfg, jet_keys=k0, base_seed=int(args.seed + args.dhard_seed_offset)
    )
    res_pert = m33._residual_fast_vec(
        pred_const=torch.tensor(hp_c, dtype=torch.float32, device=device),
        pred_mask=torch.tensor(hp_m, dtype=torch.bool, device=device),
        tgt_const=torch.tensor(h0_c, dtype=torch.float32, device=device),
        tgt_mask=torch.tensor(h0_m, dtype=torch.bool, device=device),
        w_chamfer=float(args.res_w_chamfer),
        w_count=float(args.res_w_count),
        w_pt=float(args.res_w_pt),
        w_mass=float(args.res_w_mass),
        unmatched_penalty=float(args.unmatched_penalty),
    )
    pert_total = res_pert["total"].detach().cpu().numpy().astype(np.float64)
    pert_count = res_pert["count"].detach().cpu().numpy().astype(np.float64)
    pert_report = {
        "n_eval": int(n_pert),
        "total": _summary(pert_total),
        "count": _summary(pert_count),
        "frac_large_total_gt_0p5": float(np.mean(pert_total > 0.5)),
    }

    # ------------------------------------------------------------------
    # Test 3: Train/eval D_soft surrogate
    # ------------------------------------------------------------------
    print("Training D_soft surrogate (degrader)...")
    ds_tr = m33.PairStageDataset(
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        const_hlt=const_hlt[train_idx],
        mask_hlt=mask_hlt[train_idx],
        labels=labels[train_idx],
        sample_weight=None,
    )
    ds_va = m33.PairStageDataset(
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        const_hlt=const_hlt[val_idx],
        mask_hlt=mask_hlt[val_idx],
        labels=labels[val_idx],
        sample_weight=None,
    )
    ds_te = m33.PairStageDataset(
        const_off=const_off[test_idx],
        mask_off=masks_off[test_idx],
        const_hlt=const_hlt[test_idx],
        mask_hlt=mask_hlt[test_idx],
        labels=labels[test_idx],
        sample_weight=None,
    )
    dl_tr = DataLoader(ds_tr, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_va = DataLoader(ds_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_te = DataLoader(ds_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    degrader = m33.OfflineToHLTDegrader(
        latent_dim=int(args.latent_dim),
        slots=int(args.max_constits),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    degrader, degrader_metrics = m33._train_degrader(
        model=degrader,
        train_loader=dl_tr,
        val_loader=dl_va,
        device=device,
        epochs=int(args.degrader_epochs),
        lr=float(args.degrader_lr),
        weight_decay=float(args.degrader_weight_decay),
        patience=int(args.degrader_patience),
        loss_w_count=float(args.degrader_loss_w_count),
        unmatched_penalty=float(args.unmatched_penalty),
    )

    dsoft_eval = {
        "train": _evaluate_degrader_split(
            model=degrader,
            loader=dl_tr,
            device=device,
            pred_exist_threshold=float(args.pred_exist_threshold),
            w_chamfer=float(args.res_w_chamfer),
            w_count=float(args.res_w_count),
            w_pt=float(args.res_w_pt),
            w_mass=float(args.res_w_mass),
            unmatched_penalty=float(args.unmatched_penalty),
        ),
        "val": _evaluate_degrader_split(
            model=degrader,
            loader=dl_va,
            device=device,
            pred_exist_threshold=float(args.pred_exist_threshold),
            w_chamfer=float(args.res_w_chamfer),
            w_count=float(args.res_w_count),
            w_pt=float(args.res_w_pt),
            w_mass=float(args.res_w_mass),
            unmatched_penalty=float(args.unmatched_penalty),
        ),
        "test": _evaluate_degrader_split(
            model=degrader,
            loader=dl_te,
            device=device,
            pred_exist_threshold=float(args.pred_exist_threshold),
            w_chamfer=float(args.res_w_chamfer),
            w_count=float(args.res_w_count),
            w_pt=float(args.res_w_pt),
            w_mass=float(args.res_w_mass),
            unmatched_penalty=float(args.unmatched_penalty),
        ),
    }

    # ------------------------------------------------------------------
    # Test 4: D_soft ranking + calibration vs D_hard on candidate pools
    # ------------------------------------------------------------------
    n_cand_eval = int(min(max(64, args.cand_eval_count), len(val_idx)))
    cand_idx = rng.choice(val_idx, size=n_cand_eval, replace=False)
    c_base = const_off[cand_idx]
    m_base = masks_off[cand_idx]
    h_tgt = const_hlt[cand_idx]
    hm_tgt = mask_hlt[cand_idx]
    k_base = jet_keys[cand_idx]

    cand_c, cand_m = _build_perturbed_candidates(
        const_off=c_base,
        mask_off=m_base,
        n_cand=int(args.cand_per_jet),
        rng=rng,
        pt_sigma=float(args.cand_pt_sigma),
        eta_sigma=float(args.cand_eta_sigma),
        phi_sigma=float(args.cand_phi_sigma),
        e_sigma=float(args.cand_e_sigma),
        drop_prob=float(args.cand_drop_prob),
    )

    b, k, l, _ = cand_c.shape
    cand_flat = cand_c.reshape(b * k, l, 4).astype(np.float32)
    cand_mask_flat = cand_m.reshape(b * k, l).astype(bool)
    h_tgt_rep = np.repeat(h_tgt, k, axis=0).astype(np.float32)
    hm_tgt_rep = np.repeat(hm_tgt, k, axis=0).astype(bool)
    keys_f = np.repeat(k_base, k).astype(np.int64)

    rank_bs = int(max(32, args.rank_eval_batch_size))
    soft_tot_flat = np.zeros((b * k,), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, b * k, rank_bs):
            e = min(b * k, s + rank_bs)
            cand_cf = torch.tensor(cand_flat[s:e], dtype=torch.float32, device=device)
            cand_mf = torch.tensor(cand_mask_flat[s:e], dtype=torch.bool, device=device)
            h_tgt_f = torch.tensor(h_tgt_rep[s:e], dtype=torch.float32, device=device)
            hm_tgt_f = torch.tensor(hm_tgt_rep[s:e], dtype=torch.bool, device=device)
            ph, pl = degrader(cand_cf, cand_mf)
            pw = torch.sigmoid(pl)
            soft = _soft_residual_vec(
                pred_const=ph,
                pred_w=pw,
                tgt_const=h_tgt_f,
                tgt_mask=hm_tgt_f,
                w_chamfer=float(args.res_w_chamfer),
                w_count=float(args.res_w_count),
                w_pt=float(args.res_w_pt),
                w_mass=float(args.res_w_mass),
                unmatched_penalty=float(args.unmatched_penalty),
            )
            soft_tot_flat[s:e] = soft["total"].detach().cpu().numpy().astype(np.float64)
    soft_tot = soft_tot_flat.reshape(b, k).astype(np.float64)

    h_pred_np, hm_pred_np, _ = m33._apply_hlt_effects_deterministic_keyed(
        const=cand_flat,
        mask=cand_mask_flat,
        cfg=cfg,
        jet_keys=keys_f,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    hard_tot_flat = np.zeros((b * k,), dtype=np.float64)
    hard_cnt_flat = np.zeros((b * k,), dtype=np.float64)
    for s in range(0, b * k, rank_bs):
        e = min(b * k, s + rank_bs)
        hard = m33._residual_fast_vec(
            pred_const=torch.tensor(h_pred_np[s:e], dtype=torch.float32, device=device),
            pred_mask=torch.tensor(hm_pred_np[s:e], dtype=torch.bool, device=device),
            tgt_const=torch.tensor(h_tgt_rep[s:e], dtype=torch.float32, device=device),
            tgt_mask=torch.tensor(hm_tgt_rep[s:e], dtype=torch.bool, device=device),
            w_chamfer=float(args.res_w_chamfer),
            w_count=float(args.res_w_count),
            w_pt=float(args.res_w_pt),
            w_mass=float(args.res_w_mass),
            unmatched_penalty=float(args.unmatched_penalty),
        )
        hard_tot_flat[s:e] = hard["total"].detach().cpu().numpy().astype(np.float64)
        hard_cnt_flat[s:e] = hard["count"].detach().cpu().numpy().astype(np.float64)
    hard_tot = hard_tot_flat.reshape(b, k).astype(np.float64)
    hard_cnt = hard_cnt_flat.reshape(b, k).astype(np.float64)

    topm = int(max(1, min(int(args.rank_top_m), k)))
    spear = []
    top1_hit = []
    hard_best_in_soft_topm = []
    feasible_recall_topm = []
    feasible_lab = (hard_tot <= float(args.eps_total)) & (hard_cnt <= float(args.eps_count))
    for i in range(b):
        s = _spearman_np(soft_tot[i], hard_tot[i])
        if np.isfinite(s):
            spear.append(float(s))
        sb = int(np.argmin(soft_tot[i]))
        hb = int(np.argmin(hard_tot[i]))
        top1_hit.append(float(sb == hb))
        soft_topm = np.argsort(soft_tot[i])[:topm]
        hard_best_in_soft_topm.append(float(hb in soft_topm))
        if feasible_lab[i].any():
            feasible_recall_topm.append(float(np.any(feasible_lab[i, soft_topm])))

    y_flat = feasible_lab.reshape(-1).astype(np.int32)
    score_flat = (-soft_tot.reshape(-1)).astype(np.float64)
    if len(np.unique(y_flat)) > 1:
        feas_auc = float(roc_auc_score(y_flat, score_flat))
        feas_ap = float(average_precision_score(y_flat, score_flat))
    else:
        feas_auc = float("nan")
        feas_ap = float("nan")

    ranking_report = {
        "n_jets": int(b),
        "cand_per_jet": int(k),
        "spearman_mean": float(np.mean(spear)) if len(spear) > 0 else float("nan"),
        "top1_match_rate": float(np.mean(top1_hit)) if len(top1_hit) > 0 else float("nan"),
        "hard_best_in_soft_topm_rate": float(np.mean(hard_best_in_soft_topm)) if len(hard_best_in_soft_topm) > 0 else float("nan"),
        "feasible_recall_in_soft_topm": float(np.mean(feasible_recall_topm)) if len(feasible_recall_topm) > 0 else float("nan"),
        "feasible_fraction_all_candidates": float(np.mean(y_flat)),
        "feasible_auc_from_softscore": feas_auc,
        "feasible_ap_from_softscore": feas_ap,
    }

    # ------------------------------------------------------------------
    # Test 5: Post-acceptance refinement (D_soft-guided) and D_hard delta
    # ------------------------------------------------------------------
    n_ref_jets = int(min(max(32, args.refine_eval_jets), b))
    ref_order = np.arange(b)[:n_ref_jets]
    sel_k = int(max(1, min(int(args.refine_selected_k), k)))

    sel_const = []
    sel_mask = []
    sel_tgt_h = []
    sel_tgt_m = []
    sel_keys = []
    before_total = []
    before_count = []
    for i in ref_order:
        feas_i = feasible_lab[i]
        ord_feas = np.where(feas_i)[0]
        ord_non = np.where(~feas_i)[0]
        ord_feas = ord_feas[np.argsort(hard_tot[i, ord_feas])] if ord_feas.size > 0 else np.array([], dtype=np.int64)
        ord_non = ord_non[np.argsort(hard_tot[i, ord_non])] if ord_non.size > 0 else np.array([], dtype=np.int64)
        pick = np.concatenate([ord_feas, ord_non], axis=0)[:sel_k]
        for kk in pick.tolist():
            sel_const.append(cand_c[i, kk])
            sel_mask.append(cand_m[i, kk])
            sel_tgt_h.append(h_tgt[i])
            sel_tgt_m.append(hm_tgt[i])
            sel_keys.append(int(k_base[i]))
            before_total.append(float(hard_tot[i, kk]))
            before_count.append(float(hard_cnt[i, kk]))

    sel_const_np = np.asarray(sel_const, dtype=np.float32)
    sel_mask_np = np.asarray(sel_mask, dtype=bool)
    sel_tgt_h_np = np.asarray(sel_tgt_h, dtype=np.float32)
    sel_tgt_m_np = np.asarray(sel_tgt_m, dtype=bool)
    sel_keys_np = np.asarray(sel_keys, dtype=np.int64)

    refined_np = _refine_post_acceptance(
        degrader=degrader,
        cand_const=sel_const_np,
        cand_mask=sel_mask_np,
        tgt_hlt_const=sel_tgt_h_np,
        tgt_hlt_mask=sel_tgt_m_np,
        device=device,
        steps=int(args.refine_steps),
        step_size=float(args.refine_lr),
        max_step_norm=float(args.refine_max_step_norm),
        anchor_lambda=float(args.refine_anchor_lambda),
        w_chamfer=float(args.res_w_chamfer),
        w_count=float(args.res_w_count),
        w_pt=float(args.res_w_pt),
        w_mass=float(args.res_w_mass),
        unmatched_penalty=float(args.unmatched_penalty),
    )

    ref_h_np, ref_hm_np, _ = m33._apply_hlt_effects_deterministic_keyed(
        const=refined_np,
        mask=sel_mask_np,
        cfg=cfg,
        jet_keys=sel_keys_np,
        base_seed=int(args.seed + args.dhard_seed_offset),
    )
    ref_res = m33._residual_fast_vec(
        pred_const=torch.tensor(ref_h_np, dtype=torch.float32, device=device),
        pred_mask=torch.tensor(ref_hm_np, dtype=torch.bool, device=device),
        tgt_const=torch.tensor(sel_tgt_h_np, dtype=torch.float32, device=device),
        tgt_mask=torch.tensor(sel_tgt_m_np, dtype=torch.bool, device=device),
        w_chamfer=float(args.res_w_chamfer),
        w_count=float(args.res_w_count),
        w_pt=float(args.res_w_pt),
        w_mass=float(args.res_w_mass),
        unmatched_penalty=float(args.unmatched_penalty),
    )
    after_total = ref_res["total"].detach().cpu().numpy().astype(np.float64)
    after_count = ref_res["count"].detach().cpu().numpy().astype(np.float64)
    before_total_np = np.asarray(before_total, dtype=np.float64)
    before_count_np = np.asarray(before_count, dtype=np.float64)
    delta = after_total - before_total_np

    refine_report = {
        "n_selected_candidates": int(len(before_total_np)),
        "before_total": _summary(before_total_np),
        "after_total": _summary(after_total),
        "delta_total": _summary(delta),
        "frac_improved_total": float(np.mean(delta < 0.0)) if delta.size > 0 else float("nan"),
        "frac_improved_by_gt_0p02": float(np.mean(delta < -0.02)) if delta.size > 0 else float("nan"),
        "before_feasible_frac": float(np.mean((before_total_np <= float(args.eps_total)) & (before_count_np <= float(args.eps_count)))) if before_total_np.size > 0 else float("nan"),
        "after_feasible_frac": float(np.mean((after_total <= float(args.eps_total)) & (after_count <= float(args.eps_count)))) if after_total.size > 0 else float("nan"),
    }

    report = {
        "model": "m33_dhard_dsoft_probe",
        "seed": int(args.seed),
        "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "hlt_stats": hlt_stats,
        "determinism": det_report,
        "dhard_perturbation": pert_report,
        "degrader_train": degrader_metrics,
        "degrader_eval": dsoft_eval,
        "soft_vs_hard_ranking": ranking_report,
        "post_accept_refine": refine_report,
    }

    with open(save_root / "m33_dhard_dsoft_probe_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        save_root / "m33_dhard_dsoft_probe_arrays.npz",
        soft_total=soft_tot.astype(np.float32),
        hard_total=hard_tot.astype(np.float32),
        hard_count=hard_cnt.astype(np.float32),
        feasible=feasible_lab.astype(np.int8),
        refine_before_total=before_total_np.astype(np.float32),
        refine_after_total=after_total.astype(np.float32),
        refine_delta_total=delta.astype(np.float32),
    )

    print("=" * 72)
    print("m33 D_hard / D_soft probe summary")
    print("=" * 72)
    print(f"Determinism exact (order/shuffle/chunk): {det_report['same_order_const_exact']} / {det_report['shuffled_order_const_exact']} / {det_report['chunked_const_exact']}")
    print(f"D_soft best val epoch={degrader_metrics.get('best_epoch', -1)} val={degrader_metrics.get('best_val', float('nan')):.5f}")
    print(f"Ranking Spearman mean={ranking_report['spearman_mean']:.4f}, top1={ranking_report['top1_match_rate']:.4f}, hard-best-in-soft-topM={ranking_report['hard_best_in_soft_topm_rate']:.4f}")
    print(f"Refine delta mean={refine_report['delta_total']['mean']:.5f}, improve_frac={refine_report['frac_improved_total']:.4f}")
    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()

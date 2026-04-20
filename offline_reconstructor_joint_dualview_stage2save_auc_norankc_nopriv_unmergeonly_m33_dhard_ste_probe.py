#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
m33 D_hard STE probe

Purpose:
- Keep exact deterministic D_hard forward behavior.
- Attach gradients from a physics-inspired differentiable twin (soft proxy).
- Evaluate if STE gradients are useful for ranking and post-acceptance refinement.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

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


def _wrap_phi_np(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_phi_t(phi: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phi), torch.cos(phi))


def _raw_from_const(const: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    pt = const[..., 0].clamp(min=eps)
    eta = const[..., 1].clamp(min=-5.0, max=5.0)
    phi = const[..., 2]
    e = const[..., 3].clamp(min=eps)
    eta_scaled = (eta / 5.0).clamp(min=-0.999, max=0.999)
    eta_raw = 0.5 * torch.log((1.0 + eta_scaled) / (1.0 - eta_scaled))
    return torch.stack([torch.log(pt), eta_raw, torch.sin(phi), torch.cos(phi), torch.log(e)], dim=-1)


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

    pred_mass = m33._jet_mass_vec(pred_const, pred_w > 0.5)
    tgt_mass = m33._jet_mass_vec(tgt_const, tgt_mask)
    mass_v = torch.abs(pred_mass - tgt_mass) / (tgt_mass + 1e-6)

    total_v = (
        float(w_chamfer) * set_v
        + float(w_count) * cnt_v
        + float(w_pt) * pt_v
        + float(w_mass) * mass_v
    )
    return {"total": total_v, "set": set_v, "count": cnt_v, "pt": pt_v, "mass": mass_v}


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
                pt_mult = np.exp(rng.normal(0.0, pt_sigma, size=act_idx.size)).astype(np.float32)
                e_mult = np.exp(rng.normal(0.0, e_sigma, size=act_idx.size)).astype(np.float32)
                c[act_idx, 0] = np.maximum(1e-6, c[act_idx, 0] * pt_mult)
                c[act_idx, 1] = np.clip(c[act_idx, 1] + rng.normal(0.0, eta_sigma, size=act_idx.size), -5.0, 5.0)
                c[act_idx, 2] = _wrap_phi_np(c[act_idx, 2] + rng.normal(0.0, phi_sigma, size=act_idx.size))
                c[act_idx, 3] = np.maximum(1e-6, c[act_idx, 3] * e_mult)
                c[act_idx, 3] = np.maximum(c[act_idx, 3], c[act_idx, 0] * np.cosh(c[act_idx, 1]))

                drop_mask = rng.rand(act_idx.size) < float(drop_prob)
                if np.any(drop_mask):
                    di = act_idx[drop_mask]
                    m[di] = False
                    c[di] = 0.0

            cand_c[i, kk] = c
            cand_m[i, kk] = m
    return cand_c, cand_m


def _deterministic_jet_seed(base_seed: int, jet_key: int, stream_id: int = 0) -> int:
    return int(m33._deterministic_jet_seed(base_seed=int(base_seed), jet_key=int(jet_key), stream_id=int(stream_id)))


class DHardSTE:
    """
    Hard-forward / soft-backward operator.
    Forward output exactly equals deterministic D_hard (const + hard mask).
    Backward gradients come from differentiable physics-inspired soft proxy.
    """

    def __init__(self, cfg: Dict, base_seed: int, max_constits: int, device: torch.device):
        self.cfg = cfg
        self.hcfg = cfg["hlt_effects"]
        self.base_seed = int(base_seed)
        self.max_constits = int(max_constits)
        self.device = device

    def _noise_for_keys(self, jet_keys: np.ndarray) -> Dict[str, torch.Tensor]:
        b = int(len(jet_keys))
        l = int(self.max_constits)
        u_keep = np.zeros((b, l), dtype=np.float32)
        z_pt = np.zeros((b, l), dtype=np.float32)
        z_eta = np.zeros((b, l), dtype=np.float32)
        z_phi = np.zeros((b, l), dtype=np.float32)
        q_jet = np.zeros((b,), dtype=np.float32)

        for i in range(b):
            s = _deterministic_jet_seed(self.base_seed, int(jet_keys[i]), stream_id=0)
            rs = np.random.RandomState(int(s))
            q = rs.lognormal(mean=0.0, sigma=float(self.hcfg["jet_quality_sigma"]))
            q = np.clip(q, float(self.hcfg["jet_quality_min"]), float(self.hcfg["jet_quality_max"]))
            q_jet[i] = np.float32(q)
            u_keep[i] = rs.random_sample(l).astype(np.float32)
            z_pt[i] = rs.normal(0.0, 1.0, size=l).astype(np.float32)
            z_eta[i] = rs.normal(0.0, 1.0, size=l).astype(np.float32)
            z_phi[i] = rs.normal(0.0, 1.0, size=l).astype(np.float32)

        return {
            "u_keep": torch.tensor(u_keep, dtype=torch.float32, device=self.device),
            "z_pt": torch.tensor(z_pt, dtype=torch.float32, device=self.device),
            "z_eta": torch.tensor(z_eta, dtype=torch.float32, device=self.device),
            "z_phi": torch.tensor(z_phi, dtype=torch.float32, device=self.device),
            "q_jet": torch.tensor(q_jet, dtype=torch.float32, device=self.device),
        }

    def _soft_proxy(self, const: torch.Tensor, mask: torch.Tensor, jet_keys: np.ndarray, tau: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-8
        h = self.hcfg
        b, l, _ = const.shape
        noise = self._noise_for_keys(jet_keys)

        pt = const[..., 0].clamp(min=1e-8)
        eta = const[..., 1].clamp(min=-5.0, max=5.0)
        phi = const[..., 2]

        m = mask.float()

        # Pre-threshold soft keep.
        keep_pre = m * torch.sigmoid((pt - float(h["pt_threshold_hlt"])) / float(tau["thr"]))

        # Pairwise geometry.
        deta = eta.unsqueeze(2) - eta.unsqueeze(1)
        dphi = _wrap_phi_t(phi.unsqueeze(2) - phi.unsqueeze(1))
        dR = torch.sqrt(torch.clamp(deta * deta + dphi * dphi, min=1e-10))
        eye = torch.eye(l, device=const.device, dtype=torch.bool).unsqueeze(0)

        # Local density (smooth approximation).
        dens_r = float(max(1e-4, h["density_radius"]))
        kern = torch.exp(-0.5 * (dR / dens_r) ** 2)
        kern = kern.masked_fill(eye, 0.0)
        density = (kern * keep_pre.unsqueeze(1)).sum(dim=2)

        # Merge proxy: smooth near-radius interactions.
        merge_aff = torch.sigmoid((float(h["merge_radius"]) - dR) / float(tau["merge"]))
        merge_aff = merge_aff.masked_fill(eye, 0.0)
        merge_aff = merge_aff * keep_pre.unsqueeze(1) * keep_pre.unsqueeze(2)
        close_score = merge_aff.sum(dim=2)

        merge_alpha = float(tau["merge_alpha"])
        merge_keep = torch.exp(-merge_alpha * close_score)

        # Soft pt/eta/phi blend from nearby candidates (merge-like behavior).
        w_nb = merge_aff
        w_sum = w_nb.sum(dim=2) + eps
        eta_nb = (w_nb * eta.unsqueeze(1)).sum(dim=2) / w_sum
        phi_sin_nb = (w_nb * torch.sin(phi).unsqueeze(1)).sum(dim=2) / w_sum
        phi_cos_nb = (w_nb * torch.cos(phi).unsqueeze(1)).sum(dim=2) / w_sum
        phi_nb = torch.atan2(phi_sin_nb, phi_cos_nb)
        pt_nb = (w_nb * pt.unsqueeze(1)).sum(dim=2) / w_sum

        blend = torch.tanh(close_score)
        eta_m = (1.0 - blend) * eta + blend * eta_nb
        phi_m = _wrap_phi_t((1.0 - blend) * phi + blend * phi_nb)
        pt_m = pt + float(tau["merge_gain"]) * blend * pt_nb

        abs_eta = torch.abs(eta_m)
        eta_break = float(h["eta_break"])
        eta_mix = torch.sigmoid((eta_break - abs_eta) / float(tau["eta_break"]))

        plateau = eta_mix * float(h["eff_plateau_barrel"]) + (1.0 - eta_mix) * float(h["eff_plateau_endcap"])
        pt50 = eta_mix * float(h["eff_pt50_barrel"]) + (1.0 - eta_mix) * float(h["eff_pt50_endcap"])
        width = eta_mix * float(h["eff_width_barrel"]) + (1.0 - eta_mix) * float(h["eff_width_endcap"])

        turn_on = torch.sigmoid((pt_m - pt50) / torch.clamp(width, min=1e-4))
        density_term = torch.exp(-float(h["eff_density_alpha"]) * density)
        q_eff = noise["q_jet"].unsqueeze(1)

        eps_keep = plateau * turn_on * density_term * q_eff
        eps_keep = eps_keep.clamp(min=float(h["eff_floor"]), max=float(h["eff_ceil"]))

        keep_eff = torch.sigmoid((eps_keep - noise["u_keep"]) / float(tau["eff"]))

        # Smearing proxy.
        eta_scale = 1.0 + float(h["smear_eta_scale"]) * abs_eta
        q = noise["q_jet"].unsqueeze(1)
        sigma_rel = torch.sqrt(
            (float(h["smear_a"]) / torch.sqrt(pt_m + eps)) ** 2
            + float(h["smear_b"]) ** 2
            + (float(h["smear_c"]) / (pt_m + eps)) ** 2
        )
        sigma_rel = sigma_rel * eta_scale * q
        sigma_rel = sigma_rel.clamp(min=float(h["smear_sigma_min"]), max=float(h["smear_sigma_max"]))

        ratio = 1.0 + noise["z_pt"] * sigma_rel
        ratio = ratio.clamp(min=float(h["pt_resp_min"]), max=float(h["pt_resp_max"]))
        pt_s = (pt_m * ratio).clamp(min=1e-8)

        sigma_eta = (float(h["eta_smear_const"]) + float(h["eta_smear_inv_sqrt"]) / torch.sqrt(pt_m + eps)) * eta_scale * q
        sigma_phi = (float(h["phi_smear_const"]) + float(h["phi_smear_inv_sqrt"]) / torch.sqrt(pt_m + eps)) * eta_scale * q

        eta_s = (eta_m + noise["z_eta"] * sigma_eta).clamp(min=-5.0, max=5.0)
        phi_s = _wrap_phi_t(phi_m + noise["z_phi"] * sigma_phi)
        e_s = torch.maximum(pt_s * torch.cosh(eta_s), torch.full_like(pt_s, 1e-8))

        post_thr = float(h["post_smear_pt_threshold"])
        if post_thr > 0.0:
            keep_post = torch.sigmoid((pt_s - post_thr) / float(tau["post"]))
        else:
            keep_post = torch.ones_like(pt_s)

        w = keep_pre * merge_keep * keep_eff * keep_post
        w = w.clamp(min=0.0, max=1.0)

        soft_const = torch.stack([pt_s, eta_s, phi_s, e_s], dim=-1)
        return soft_const, w

    def forward(self, const: torch.Tensor, mask: torch.Tensor, jet_keys: np.ndarray, tau: Dict[str, float]) -> Dict[str, torch.Tensor]:
        # Hard forward branch (exact D_hard).
        with torch.no_grad():
            c_np = const.detach().cpu().numpy().astype(np.float32)
            m_np = mask.detach().cpu().numpy().astype(bool)
            h_np, hm_np, _ = m33._apply_hlt_effects_deterministic_keyed(
                const=c_np,
                mask=m_np,
                cfg=self.cfg,
                jet_keys=jet_keys.astype(np.int64),
                base_seed=int(self.base_seed),
            )
            hard_const = torch.tensor(h_np, dtype=torch.float32, device=const.device)
            hard_w = torch.tensor(hm_np.astype(np.float32), dtype=torch.float32, device=const.device)

        # Differentiable soft branch.
        soft_const, soft_w = self._soft_proxy(const=const, mask=mask, jet_keys=jet_keys, tau=tau)

        # STE: forward==hard, backward==soft gradients.
        ste_const = hard_const + (soft_const - soft_const.detach())
        ste_w = hard_w + (soft_w - soft_w.detach())

        return {
            "hard_const": hard_const,
            "hard_w": hard_w,
            "soft_const": soft_const,
            "soft_w": soft_w,
            "ste_const": ste_const,
            "ste_w": ste_w,
        }


def _refine_with_ste(
    op: DHardSTE,
    cand_const: np.ndarray,
    cand_mask: np.ndarray,
    tgt_hlt_const: np.ndarray,
    tgt_hlt_mask: np.ndarray,
    jet_keys: np.ndarray,
    device: torch.device,
    steps: int,
    step_size: float,
    max_step_norm: float,
    anchor_lambda: float,
    tau: Dict[str, float],
    w_chamfer: float,
    w_count: float,
    w_pt: float,
    w_mass: float,
    unmatched_penalty: float,
) -> np.ndarray:
    n, _, _ = cand_const.shape
    if n == 0 or int(steps) <= 0:
        return cand_const.copy()

    c0 = torch.tensor(cand_const, dtype=torch.float32, device=device)
    m0 = torch.tensor(cand_mask, dtype=torch.bool, device=device)
    th = torch.tensor(tgt_hlt_const, dtype=torch.float32, device=device)
    tm = torch.tensor(tgt_hlt_mask, dtype=torch.bool, device=device)
    jk = np.asarray(jet_keys, dtype=np.int64)

    raw = _raw_from_const(c0).detach().requires_grad_(True)
    for _ in range(int(steps)):
        const_now = _const_from_raw(raw)
        const_now = torch.where(m0.unsqueeze(-1), const_now, torch.zeros_like(const_now))

        out = op.forward(const=const_now, mask=m0, jet_keys=jk, tau=tau)
        soft = _soft_residual_vec(
            pred_const=out["ste_const"],
            pred_w=out["ste_w"],
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
    p = argparse.ArgumentParser(description="m33 D_hard STE gradient probe")
    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_debug/m33_dhard_ste_probe")
    p.add_argument("--run_name", type=str, default="m33_dhard_ste_probe_debug_seed0")

    p.add_argument("--n_train_jets", type=int, default=70000)
    p.add_argument("--n_train_split", type=int, default=20000)
    p.add_argument("--n_val_split", type=int, default=8000)
    p.add_argument("--n_test_split", type=int, default=12000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
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
    p.add_argument("--res_w_chamfer", type=float, default=1.0)
    p.add_argument("--res_w_count", type=float, default=0.30)
    p.add_argument("--res_w_pt", type=float, default=0.12)
    p.add_argument("--res_w_mass", type=float, default=0.06)

    # Soft proxy temperatures/knobs
    p.add_argument("--tau_thr", type=float, default=0.08)
    p.add_argument("--tau_eff", type=float, default=0.06)
    p.add_argument("--tau_merge", type=float, default=0.02)
    p.add_argument("--tau_post", type=float, default=0.08)
    p.add_argument("--tau_eta_break", type=float, default=0.08)
    p.add_argument("--merge_alpha", type=float, default=0.75)
    p.add_argument("--merge_gain", type=float, default=0.45)

    # Candidate ranking test
    p.add_argument("--cand_eval_count", type=int, default=1200)
    p.add_argument("--cand_per_jet", type=int, default=24)
    p.add_argument("--cand_pt_sigma", type=float, default=0.08)
    p.add_argument("--cand_eta_sigma", type=float, default=0.05)
    p.add_argument("--cand_phi_sigma", type=float, default=0.05)
    p.add_argument("--cand_e_sigma", type=float, default=0.08)
    p.add_argument("--cand_drop_prob", type=float, default=0.08)
    p.add_argument("--rank_top_m", type=int, default=8)

    # Refinement test
    p.add_argument("--refine_eval_jets", type=int, default=350)
    p.add_argument("--refine_selected_k", type=int, default=6)
    p.add_argument("--refine_steps", type=int, default=8)
    p.add_argument("--refine_lr", type=float, default=0.03)
    p.add_argument("--refine_max_step_norm", type=float, default=0.20)
    p.add_argument("--refine_anchor_lambda", type=float, default=0.08)

    # Feasible cut (for reporting)
    p.add_argument("--eps_total", type=float, default=0.20)
    p.add_argument("--eps_count", type=float, default=0.25)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("m33 D_hard STE probe")
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

    tau = {
        "thr": float(args.tau_thr),
        "eff": float(args.tau_eff),
        "merge": float(args.tau_merge),
        "post": float(args.tau_post),
        "eta_break": float(args.tau_eta_break),
        "merge_alpha": float(args.merge_alpha),
        "merge_gain": float(args.merge_gain),
    }

    op = DHardSTE(
        cfg=cfg,
        base_seed=int(args.seed + args.dhard_seed_offset),
        max_constits=int(args.max_constits),
        device=device,
    )

    rng = np.random.RandomState(int(args.seed))

    # Test 1: hard-forward identity.
    n_id = int(min(512, len(val_idx)))
    id_idx = rng.choice(val_idx, size=n_id, replace=False)
    c_id = torch.tensor(const_off[id_idx], dtype=torch.float32, device=device)
    m_id = torch.tensor(masks_off[id_idx], dtype=torch.bool, device=device)
    k_id = jet_keys[id_idx]

    out_id = op.forward(c_id, m_id, k_id, tau=tau)
    id_const_diff = float((out_id["ste_const"].detach() - out_id["hard_const"]).abs().max().item())
    id_w_diff = float((out_id["ste_w"].detach() - out_id["hard_w"]).abs().max().item())

    # Test 2: gradient sanity.
    raw = _raw_from_const(c_id).detach().requires_grad_(True)
    c_var = _const_from_raw(raw)
    c_var = torch.where(m_id.unsqueeze(-1), c_var, torch.zeros_like(c_var))
    out_g = op.forward(c_var, m_id, k_id, tau=tau)
    target_c = torch.tensor(const_hlt[id_idx], dtype=torch.float32, device=device)
    target_m = torch.tensor(mask_hlt[id_idx], dtype=torch.bool, device=device)
    loss_g = _soft_residual_vec(
        pred_const=out_g["ste_const"],
        pred_w=out_g["ste_w"],
        tgt_const=target_c,
        tgt_mask=target_m,
        w_chamfer=float(args.res_w_chamfer),
        w_count=float(args.res_w_count),
        w_pt=float(args.res_w_pt),
        w_mass=float(args.res_w_mass),
        unmatched_penalty=float(args.unmatched_penalty),
    )["total"].mean()
    grad = torch.autograd.grad(loss_g, raw, only_inputs=True, create_graph=False)[0]
    grad_abs = grad.abs().detach().cpu().numpy()
    grad_report = {
        "grad_abs_mean": float(np.mean(grad_abs)),
        "grad_abs_p90": float(np.quantile(grad_abs, 0.90)),
        "grad_nonzero_frac": float(np.mean(grad_abs > 1e-8)),
    }

    # Test 3: ranking proxy vs D_hard.
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
    keys_rep = np.repeat(k_base, k).astype(np.int64)

    # Proxy scores from soft branch.
    cand_cf = torch.tensor(cand_flat, dtype=torch.float32, device=device)
    cand_mf = torch.tensor(cand_mask_flat, dtype=torch.bool, device=device)
    tgt_cf = torch.tensor(h_tgt_rep, dtype=torch.float32, device=device)
    tgt_mf = torch.tensor(hm_tgt_rep, dtype=torch.bool, device=device)

    out_rank = op.forward(cand_cf, cand_mf, keys_rep, tau=tau)
    soft_rank = _soft_residual_vec(
        pred_const=out_rank["soft_const"],
        pred_w=out_rank["soft_w"],
        tgt_const=tgt_cf,
        tgt_mask=tgt_mf,
        w_chamfer=float(args.res_w_chamfer),
        w_count=float(args.res_w_count),
        w_pt=float(args.res_w_pt),
        w_mass=float(args.res_w_mass),
        unmatched_penalty=float(args.unmatched_penalty),
    )["total"].detach().cpu().numpy().reshape(b, k).astype(np.float64)

    hard_rank = m33._residual_fast_vec(
        pred_const=out_rank["hard_const"],
        pred_mask=out_rank["hard_w"] > 0.5,
        tgt_const=tgt_cf,
        tgt_mask=tgt_mf,
        w_chamfer=float(args.res_w_chamfer),
        w_count=float(args.res_w_count),
        w_pt=float(args.res_w_pt),
        w_mass=float(args.res_w_mass),
        unmatched_penalty=float(args.unmatched_penalty),
    )
    hard_total = hard_rank["total"].detach().cpu().numpy().reshape(b, k).astype(np.float64)
    hard_count = hard_rank["count"].detach().cpu().numpy().reshape(b, k).astype(np.float64)

    topm = int(max(1, min(int(args.rank_top_m), k)))
    spear = []
    top1_hit = []
    hard_best_in_soft_topm = []
    feasible_recall_topm = []
    feasible_lab = (hard_total <= float(args.eps_total)) & (hard_count <= float(args.eps_count))

    for i in range(b):
        s = _spearman_np(soft_rank[i], hard_total[i])
        if np.isfinite(s):
            spear.append(float(s))
        sb = int(np.argmin(soft_rank[i]))
        hb = int(np.argmin(hard_total[i]))
        top1_hit.append(float(sb == hb))
        soft_topm = np.argsort(soft_rank[i])[:topm]
        hard_best_in_soft_topm.append(float(hb in soft_topm))
        if feasible_lab[i].any():
            feasible_recall_topm.append(float(np.any(feasible_lab[i, soft_topm])))

    ranking_report = {
        "n_jets": int(b),
        "cand_per_jet": int(k),
        "spearman_mean": float(np.mean(spear)) if spear else float("nan"),
        "top1_match_rate": float(np.mean(top1_hit)) if top1_hit else float("nan"),
        "hard_best_in_soft_topm_rate": float(np.mean(hard_best_in_soft_topm)) if hard_best_in_soft_topm else float("nan"),
        "feasible_recall_in_soft_topm": float(np.mean(feasible_recall_topm)) if feasible_recall_topm else float("nan"),
    }

    # Test 4: refinement utility (STE-guided, measured by D_hard).
    n_ref_jets = int(min(max(32, args.refine_eval_jets), b))
    sel_k = int(max(1, min(int(args.refine_selected_k), k)))

    sel_const = []
    sel_mask = []
    sel_tgt_h = []
    sel_tgt_m = []
    sel_keys = []
    before_total = []
    before_count = []

    for i in range(n_ref_jets):
        ord_all = np.argsort(hard_total[i])[:sel_k]
        for kk in ord_all.tolist():
            sel_const.append(cand_c[i, kk])
            sel_mask.append(cand_m[i, kk])
            sel_tgt_h.append(h_tgt[i])
            sel_tgt_m.append(hm_tgt[i])
            sel_keys.append(int(k_base[i]))
            before_total.append(float(hard_total[i, kk]))
            before_count.append(float(hard_count[i, kk]))

    sel_const_np = np.asarray(sel_const, dtype=np.float32)
    sel_mask_np = np.asarray(sel_mask, dtype=bool)
    sel_tgt_h_np = np.asarray(sel_tgt_h, dtype=np.float32)
    sel_tgt_m_np = np.asarray(sel_tgt_m, dtype=bool)
    sel_keys_np = np.asarray(sel_keys, dtype=np.int64)

    refined_np = _refine_with_ste(
        op=op,
        cand_const=sel_const_np,
        cand_mask=sel_mask_np,
        tgt_hlt_const=sel_tgt_h_np,
        tgt_hlt_mask=sel_tgt_m_np,
        jet_keys=sel_keys_np,
        device=device,
        steps=int(args.refine_steps),
        step_size=float(args.refine_lr),
        max_step_norm=float(args.refine_max_step_norm),
        anchor_lambda=float(args.refine_anchor_lambda),
        tau=tau,
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
        "model": "m33_dhard_ste_probe",
        "seed": int(args.seed),
        "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "hlt_stats": hlt_stats,
        "ste_identity": {
            "max_abs_const_forward_diff": id_const_diff,
            "max_abs_weight_forward_diff": id_w_diff,
            "hard_forward_exact": bool(id_const_diff < 1e-8 and id_w_diff < 1e-8),
        },
        "grad_sanity": grad_report,
        "soft_vs_hard_ranking": ranking_report,
        "post_accept_refine": refine_report,
        "tau": tau,
    }

    with open(save_root / "m33_dhard_ste_probe_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        save_root / "m33_dhard_ste_probe_arrays.npz",
        soft_rank=np.asarray(soft_rank, dtype=np.float32),
        hard_rank=np.asarray(hard_total, dtype=np.float32),
        hard_count=np.asarray(hard_count, dtype=np.float32),
        refine_before_total=np.asarray(before_total_np, dtype=np.float32),
        refine_after_total=np.asarray(after_total, dtype=np.float32),
        refine_delta_total=np.asarray(delta, dtype=np.float32),
    )

    print("=" * 72)
    print("m33 D_hard STE probe summary")
    print("=" * 72)
    print(f"STE hard-forward exact: {report['ste_identity']['hard_forward_exact']} (max const diff={id_const_diff:.3e}, max w diff={id_w_diff:.3e})")
    print(f"Grad sanity mean/p90/nonzero: {grad_report['grad_abs_mean']:.4e} / {grad_report['grad_abs_p90']:.4e} / {grad_report['grad_nonzero_frac']:.3f}")
    print(f"Ranking Spearman/top1/hard-in-topM: {ranking_report['spearman_mean']:.4f} / {ranking_report['top1_match_rate']:.4f} / {ranking_report['hard_best_in_soft_topm_rate']:.4f}")
    print(f"Refine delta mean={refine_report['delta_total']['mean']:.5f}, improve_frac={refine_report['frac_improved_total']:.4f}")
    print(f"Saved: {save_root}")


if __name__ == "__main__":
    main()

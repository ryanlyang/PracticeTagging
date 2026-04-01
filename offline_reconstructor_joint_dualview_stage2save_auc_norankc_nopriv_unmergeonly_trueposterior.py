#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
True-posterior ablation wrapper (isolated variant).

Design goals:
- Explicit K posterior hypotheses (not temperature-only reweighting).
- Hypothesis-specific corrected candidate tokens/weights from learned mode latents.
- Posterior calibration + diversity objectives in reconstruction loss.
- Learned dual aggregation that consumes per-hypothesis logits + posterior probs.

Implementation strategy:
- Reuse full A/B/C pipeline from the base unmerge-only script.
- Swap reconstructor class with posterior-augmented subclass.
- Swap reconstruction loss with posterior-aware aggregation.
- Swap dual model with posterior-aware learned aggregator.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except Exception:
        v = int(default)
    return int(max(lo, min(hi, v)))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.getenv(name, str(default)))
    except Exception:
        v = float(default)
    return float(max(lo, min(hi, v)))


# Posterior hypothesis controls
_POST_K = _env_int("POSTERIOR_K", 3, 2, 8)
_POST_MODE_DIM = _env_int("POSTERIOR_MODE_DIM", 32, 8, 256)
_POST_TOKEN_HIDDEN = _env_int("POSTERIOR_TOKEN_HIDDEN", 192, 32, 1024)
_POST_CTX_HIDDEN = _env_int("POSTERIOR_CTX_HIDDEN", 96, 16, 512)

# Per-mode perturbation bounds
_POST_PT_SHIFT = _env_float("POSTERIOR_PT_SHIFT", 0.35, 0.01, 2.0)      # additive in log-pt
_POST_E_SHIFT = _env_float("POSTERIOR_E_SHIFT", 0.35, 0.01, 2.0)        # additive in log-E
_POST_ETA_SHIFT = _env_float("POSTERIOR_ETA_SHIFT", 0.30, 0.01, 2.0)    # additive in eta
_POST_PHI_SHIFT = _env_float("POSTERIOR_PHI_SHIFT", 0.30, 0.01, 2.0)    # additive in phi
_POST_W_SHIFT = _env_float("POSTERIOR_W_SHIFT", 1.00, 0.01, 6.0)        # additive in cand-weight logit
_POST_W_RENORM_MIN = _env_float("POSTERIOR_W_RENORM_MIN", 0.50, 0.05, 2.0)
_POST_W_RENORM_MAX = _env_float("POSTERIOR_W_RENORM_MAX", 2.00, 0.5, 10.0)

# Posterior objective controls
_POST_ASSIGN_TAU = _env_float("POSTERIOR_ASSIGN_TAU", 0.30, 1e-3, 5.0)
_POST_W_CAL_STAGEA = _env_float("POSTERIOR_W_CAL_STAGEA", 0.03, 0.0, 5.0)
_POST_W_DIV_STAGEA = _env_float("POSTERIOR_W_DIV_STAGEA", 0.02, 0.0, 5.0)
_POST_W_ENT_STAGEA = _env_float("POSTERIOR_W_ENT_STAGEA", 0.01, 0.0, 5.0)
_POST_W_CAL_JOINT = _env_float("POSTERIOR_W_CAL_JOINT", 0.00, 0.0, 5.0)
_POST_W_DIV_JOINT = _env_float("POSTERIOR_W_DIV_JOINT", 0.00, 0.0, 5.0)
_POST_W_ENT_JOINT = _env_float("POSTERIOR_W_ENT_JOINT", 0.00, 0.0, 5.0)
_POST_ENT_TARGET = _env_float("POSTERIOR_ENT_TARGET", 0.35, 0.0, 1.0)   # normalized entropy floor

# Dual learned aggregation controls
_POST_GATE_HIDDEN = _env_int("POSTERIOR_GATE_HIDDEN", 48, 8, 512)
_POST_GATE_DROPOUT = _env_float("POSTERIOR_GATE_DROPOUT", 0.05, 0.0, 0.7)
_POST_GATE_PRIOR_SCALE = _env_float("POSTERIOR_GATE_PRIOR_SCALE", 0.70, 0.0, 5.0)

# Keep this variant aligned with unmerge-only semantics by default.
_POST_ZERO_EFF_BRANCH = bool(_env_int("POSTERIOR_ZERO_EFF_BRANCH", 1, 0, 1))


_LAST_RECO_OUT: Optional[dict] = None
_LAST_BUILD_ARGS = {
    "weight_floor": 1e-4,
    "scale_features_by_weight": True,
    "include_flags": False,
}

_ORIG_BUILD_SOFT_VIEW = base.build_soft_corrected_view
_ORIG_DUAL_CLASS = base.DualViewCrossAttnClassifier
_ORIG_RECO_LOSS_FN = base.compute_reconstruction_losses_weighted
_ORIG_RECO_CLASS = base.OfflineReconstructor
_ORIG_TRAIN_RECO_WEIGHTED = base.train_reconstructor_weighted
_ORIG_TRAIN_JOINT_DUAL = base.train_joint_dual

_STAGE_CONTEXT = "other"  # stageA | joint | other


class OfflineReconstructorTruePosterior(base.reco_base.OfflineReconstructor):
    """
    Posterior-augmented reconstructor:
    - Runs standard reconstructor forward.
    - Builds K hypothesis-specific corrections from learned mode latents.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.posterior_k = int(_POST_K)
        self.posterior_mode_dim = int(_POST_MODE_DIM)

        # Prefixes intentionally start with existing head prefixes so Stage-C
        # progressive unfreeze ("heads") includes these parameters.
        self.action_head_posterior_mode = nn.Sequential(
            nn.Linear(10, _POST_CTX_HIDDEN),
            nn.GELU(),
            nn.Linear(_POST_CTX_HIDDEN, self.posterior_k),
        )
        self.unsmear_head_posterior_token = nn.Sequential(
            nn.Linear(7 + self.posterior_mode_dim, _POST_TOKEN_HIDDEN),
            nn.GELU(),
            nn.Linear(_POST_TOKEN_HIDDEN, 5),  # dlogpt, deta, dphi, dloge, dlogit(w)
        )
        self.posterior_mode_emb = nn.Parameter(
            torch.randn(self.posterior_k, self.posterior_mode_dim) * 0.02
        )

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        stage_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        out = super().forward(feat_hlt, mask_hlt, const_hlt, stage_scale=stage_scale)

        cand_tokens = out["cand_tokens"]  # [B,N,4]
        cand_weights = out["cand_weights"].clamp(0.0, 1.0)  # [B,N]
        cand_merge = out["cand_merge_flags"].clamp(0.0, 1.0)  # [B,N]
        cand_eff = out["cand_eff_flags"].clamp(0.0, 1.0)  # [B,N]

        bsz, n_tok, _ = cand_tokens.shape
        l_hlt = int(out["action_prob"].shape[1])
        n_child = int(out["child_weight"].shape[1])
        n_gen = int(out["gen_weight"].shape[1])

        # Build compact jet context for posterior mode logits.
        mask_f = mask_hlt.float()
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        action_mean = (out["action_prob"] * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # [B,4]
        count_norm = cand_weights.sum(dim=1, keepdim=True) / max(float(n_tok), 1.0)
        child_norm = out["child_weight"].sum(dim=1, keepdim=True) / max(float(n_child), 1.0)
        gen_norm = out["gen_weight"].sum(dim=1, keepdim=True) / max(float(max(n_gen, 1)), 1.0)
        ctx = torch.cat(
            [
                out["budget_total"].unsqueeze(1),
                out["budget_merge"].unsqueeze(1),
                out["budget_eff"].unsqueeze(1),
                action_mean,
                count_norm,
                child_norm,
                gen_norm,
            ],
            dim=1,
        )  # [B,10]
        mode_logits = self.action_head_posterior_mode(ctx)  # [B,K]
        mode_prob = torch.softmax(mode_logits, dim=1)

        token_feat = torch.cat(
            [
                cand_tokens,
                cand_weights.unsqueeze(-1),
                cand_merge.unsqueeze(-1),
                cand_eff.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B,N,7]

        mode_tokens = []
        mode_weights = []
        for k in range(self.posterior_k):
            mode_emb = self.posterior_mode_emb[k].view(1, 1, -1).expand(bsz, n_tok, -1)
            p_in = torch.cat([token_feat, mode_emb], dim=-1)
            d = self.unsmear_head_posterior_token(p_in)

            pt0 = cand_tokens[..., 0].clamp(min=eps)
            eta0 = cand_tokens[..., 1].clamp(min=-5.0, max=5.0)
            phi0 = cand_tokens[..., 2]
            e0 = cand_tokens[..., 3].clamp(min=eps)

            d_logpt = float(stage_scale) * float(_POST_PT_SHIFT) * torch.tanh(d[..., 0])
            d_eta = float(stage_scale) * float(_POST_ETA_SHIFT) * torch.tanh(d[..., 1])
            d_phi = float(stage_scale) * float(_POST_PHI_SHIFT) * torch.tanh(d[..., 2])
            d_loge = float(stage_scale) * float(_POST_E_SHIFT) * torch.tanh(d[..., 3])
            d_wlogit = float(stage_scale) * float(_POST_W_SHIFT) * torch.tanh(d[..., 4])

            pt1 = torch.exp(torch.clamp(torch.log(pt0) + d_logpt, min=-9.0, max=9.0))
            eta1 = (eta0 + d_eta).clamp(min=-5.0, max=5.0)
            phi1 = base.reco_base.wrap_phi_t(phi0 + d_phi)
            e1 = torch.exp(torch.clamp(torch.log(e0) + d_loge, min=-9.0, max=11.0))
            e1 = torch.maximum(e1, pt1 * torch.cosh(eta1))
            tok1 = torch.stack([pt1, eta1, phi1, e1], dim=-1)

            w0 = cand_weights.clamp(min=1e-6, max=1.0 - 1e-6)
            wlog0 = torch.log(w0) - torch.log1p(-w0)
            w1 = torch.sigmoid(wlog0 + d_wlogit)
            # Keep per-jet candidate mass close to baseline.
            w0_sum = cand_weights.sum(dim=1, keepdim=True).clamp(min=eps)
            w1_sum = w1.sum(dim=1, keepdim=True).clamp(min=eps)
            scale = (w0_sum / w1_sum).clamp(
                min=float(_POST_W_RENORM_MIN),
                max=float(_POST_W_RENORM_MAX),
            )
            w1 = (w1 * scale).clamp(0.0, 1.0)

            if _POST_ZERO_EFF_BRANCH and n_gen > 0:
                w1 = w1.clone()
                w1[:, -n_gen:] = 0.0

            mode_tokens.append(tok1)
            mode_weights.append(w1)

        out["posterior_mode_logits"] = mode_logits
        out["posterior_mode_prob"] = mode_prob
        out["posterior_mode_tokens"] = torch.stack(mode_tokens, dim=1)  # [B,K,N,4]
        out["posterior_mode_weights"] = torch.stack(mode_weights, dim=1)  # [B,K,N]
        out["posterior_mode_merge_flags"] = cand_merge.unsqueeze(1).expand(-1, self.posterior_k, -1)
        out["posterior_mode_eff_flags"] = cand_eff.unsqueeze(1).expand(-1, self.posterior_k, -1)
        return out


def _mode_out_from_posterior(out: Dict[str, torch.Tensor], mode_idx: int) -> Dict[str, torch.Tensor]:
    out_h = dict(out)
    mode_tokens = out["posterior_mode_tokens"][:, mode_idx]
    mode_weights = out["posterior_mode_weights"][:, mode_idx].clamp(0.0, 1.0)

    l_hlt = int(out["action_prob"].shape[1])
    n_child = int(out["child_weight"].shape[1])
    n_gen = int(out["gen_weight"].shape[1])

    if _POST_ZERO_EFF_BRANCH and n_gen > 0:
        mode_weights = mode_weights.clone()
        mode_weights[:, -n_gen:] = 0.0

    out_h["cand_tokens"] = mode_tokens
    out_h["cand_weights"] = mode_weights

    if "posterior_mode_merge_flags" in out:
        out_h["cand_merge_flags"] = out["posterior_mode_merge_flags"][:, mode_idx]
    if "posterior_mode_eff_flags" in out:
        out_h["cand_eff_flags"] = out["posterior_mode_eff_flags"][:, mode_idx]

    if n_child > 0:
        out_h["child_weight"] = mode_weights[:, l_hlt : l_hlt + n_child]
    if n_gen > 0:
        out_h["gen_weight"] = mode_weights[:, l_hlt + n_child : l_hlt + n_child + n_gen]
        if _POST_ZERO_EFF_BRANCH:
            out_h["gen_weight"] = torch.zeros_like(out_h["gen_weight"])
            if "cand_eff_flags" in out_h:
                out_h["cand_eff_flags"] = torch.zeros_like(out_h["cand_eff_flags"])

    if _POST_ZERO_EFF_BRANCH and "budget_eff" in out_h:
        out_h["budget_eff"] = out_h["budget_eff"] * 0.0
    return out_h


def _posterior_proxy_vec(
    out_h: Dict[str, torch.Tensor],
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    unselected_penalty: float,
) -> torch.Tensor:
    eps = 1e-8
    pred = out_h["cand_tokens"]
    w = out_h["cand_weights"].clamp(0.0, 1.0)
    cost = base.reco_base._token_cost_matrix(pred, const_off)
    valid_tgt = mask_off.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))

    pred_to_tgt = cost.min(dim=2).values
    loss_pred_to_tgt = (w * pred_to_tgt).sum(dim=1) / (w.sum(dim=1) + eps)

    penalty = float(unselected_penalty) * (1.0 - w).unsqueeze(2)
    tgt_to_pred = (cost + penalty).min(dim=1).values
    tgt_mask_f = mask_off.float()
    loss_tgt_to_pred = (tgt_to_pred * tgt_mask_f).sum(dim=1) / (tgt_mask_f.sum(dim=1) + eps)
    return loss_pred_to_tgt + loss_tgt_to_pred


def _weighted_mode_average(q: torch.Tensor, sw: Optional[torch.Tensor]) -> torch.Tensor:
    # q: [B,K]
    if sw is None:
        return q.mean(dim=0)
    denom = sw.sum().clamp(min=1e-6)
    return (q * sw.unsqueeze(1)).sum(dim=0) / denom


def _compute_reco_losses_trueposterior(
    out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    loss_cfg: Dict,
    sample_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if "posterior_mode_logits" not in out or "posterior_mode_tokens" not in out or "posterior_mode_weights" not in out:
        return _ORIG_RECO_LOSS_FN(
            out,
            const_hlt,
            mask_hlt,
            const_off,
            mask_off,
            budget_merge_true,
            budget_eff_true,
            loss_cfg,
            sample_weight=sample_weight,
        )

    sw = None if sample_weight is None else sample_weight.float().clamp(min=0.0)
    eps = 1e-8
    h = int(out["posterior_mode_tokens"].shape[1])

    loss_dicts = []
    proxy_vecs = []
    mode_pt = []
    mode_count = []
    for k in range(h):
        out_h = _mode_out_from_posterior(out, k)
        lk = _ORIG_RECO_LOSS_FN(
            out_h,
            const_hlt,
            mask_hlt,
            const_off,
            mask_off,
            budget_merge_true,
            budget_eff_true,
            loss_cfg,
            sample_weight=sample_weight,
        )
        loss_dicts.append(lk)

        proxy_vecs.append(
            _posterior_proxy_vec(
                out_h=out_h,
                const_off=const_off,
                mask_off=mask_off,
                unselected_penalty=float(loss_cfg.get("unselected_penalty", 0.0)),
            )
        )

        w = out_h["cand_weights"].clamp(0.0, 1.0)
        t = out_h["cand_tokens"]
        px = (w * t[..., 0] * torch.cos(t[..., 2])).sum(dim=1)
        py = (w * t[..., 0] * torch.sin(t[..., 2])).sum(dim=1)
        pt = torch.sqrt(px.pow(2) + py.pow(2) + eps)
        mode_pt.append(pt)
        mode_count.append(w.sum(dim=1))

    # Per-jet responsibilities from proxy reconstruction fit.
    proxy_stack = torch.stack(proxy_vecs, dim=1)  # [B,H]
    q = torch.softmax(-proxy_stack / float(_POST_ASSIGN_TAU), dim=1)
    pi = torch.softmax(out["posterior_mode_logits"], dim=1)
    q_bar = _weighted_mode_average(q, sw)  # [H]

    # Combine original reconstruction components using mode responsibilities.
    out_loss: Dict[str, torch.Tensor] = {}
    keys = list(loss_dicts[0].keys())
    for key in keys:
        vals = torch.stack([d[key] for d in loss_dicts], dim=0)  # [H]
        out_loss[key] = (q_bar * vals).sum(dim=0)

    # Calibration: predicted mode probs should match responsibility posterior.
    kl_vec = (q * (torch.log(q + eps) - torch.log(pi + eps))).sum(dim=1)
    loss_cal = base._weighted_batch_mean(kl_vec, sw)

    # Entropy floor on pi to avoid early mode collapse.
    ent_pi = -(pi * torch.log(pi + eps)).sum(dim=1) / math.log(max(float(h), 2.0))
    loss_ent = base._weighted_batch_mean(torch.relu(float(_POST_ENT_TARGET) - ent_pi), sw)

    # Diversity: encourage modes to differ on jet-level summaries.
    pt_stack = torch.stack(mode_pt, dim=1)      # [B,H]
    cnt_stack = torch.stack(mode_count, dim=1)  # [B,H]
    pair_d = []
    for i in range(h):
        for j in range(i + 1, h):
            pt_i = pt_stack[:, i]
            pt_j = pt_stack[:, j]
            c_i = cnt_stack[:, i]
            c_j = cnt_stack[:, j]
            rel_pt = (pt_i - pt_j).abs() / (0.5 * (pt_i + pt_j) + 1.0)
            rel_c = (c_i - c_j).abs() / (0.5 * (c_i + c_j) + 1.0)
            pair_d.append(rel_pt + 0.30 * rel_c)
    if pair_d:
        d_mean = torch.stack(pair_d, dim=1).mean(dim=1)
        loss_div = base._weighted_batch_mean(torch.exp(-d_mean), sw)  # smaller when diverse
    else:
        loss_div = torch.zeros((), device=proxy_stack.device)

    if _STAGE_CONTEXT == "stageA":
        w_cal = float(_POST_W_CAL_STAGEA)
        w_div = float(_POST_W_DIV_STAGEA)
        w_ent = float(_POST_W_ENT_STAGEA)
    else:
        w_cal = float(_POST_W_CAL_JOINT)
        w_div = float(_POST_W_DIV_JOINT)
        w_ent = float(_POST_W_ENT_JOINT)

    out_loss["posterior_cal"] = loss_cal
    out_loss["posterior_div"] = loss_div
    out_loss["posterior_ent"] = loss_ent
    out_loss["posterior_q_entropy"] = base._weighted_batch_mean(
        -(q * torch.log(q + eps)).sum(dim=1) / math.log(max(float(h), 2.0),
    ), sw)
    out_loss["posterior_pi_entropy"] = base._weighted_batch_mean(ent_pi, sw)

    out_loss["total"] = out_loss["total"] + w_cal * loss_cal + w_div * loss_div + w_ent * loss_ent
    return out_loss


def _build_soft_corrected_view_cache(
    reco_out,
    weight_floor: float = 1e-4,
    scale_features_by_weight: bool = True,
    include_flags: bool = False,
):
    global _LAST_RECO_OUT, _LAST_BUILD_ARGS
    _LAST_RECO_OUT = reco_out
    _LAST_BUILD_ARGS = {
        "weight_floor": float(weight_floor),
        "scale_features_by_weight": bool(scale_features_by_weight),
        "include_flags": bool(include_flags),
    }
    return _ORIG_BUILD_SOFT_VIEW(
        reco_out,
        weight_floor=weight_floor,
        scale_features_by_weight=scale_features_by_weight,
        include_flags=include_flags,
    )


class DualViewCrossAttnClassifierTruePosterior(nn.Module):
    """
    Shared dual scorer over posterior hypotheses + learned aggregation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base_model = _ORIG_DUAL_CLASS(*args, **kwargs)
        self.gate = None

    def _ensure_gate(self, h: int, device: torch.device):
        gate_in_dim = 3 * h + 2  # logits(H), pi(H), mass(H), score_std, pi_entropy
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, int(_POST_GATE_HIDDEN)),
            nn.GELU(),
            nn.Dropout(float(_POST_GATE_DROPOUT)),
            nn.Linear(int(_POST_GATE_HIDDEN), h),
        ).to(device)

    def forward(self, feat_a, mask_a, feat_b, mask_b):
        if _LAST_RECO_OUT is None:
            return self.base_model(feat_a, mask_a, feat_b, mask_b)

        out = _LAST_RECO_OUT
        if "posterior_mode_tokens" not in out or "posterior_mode_weights" not in out or "posterior_mode_logits" not in out:
            return self.base_model(feat_a, mask_a, feat_b, mask_b)
        if int(out["cand_weights"].shape[0]) != int(feat_a.shape[0]):
            return self.base_model(feat_a, mask_a, feat_b, mask_b)

        build_args = dict(_LAST_BUILD_ARGS)
        h = int(out["posterior_mode_tokens"].shape[1])
        pi = torch.softmax(out["posterior_mode_logits"], dim=1)  # [B,H]

        if self.gate is None:
            self._ensure_gate(h=h, device=feat_a.device)
        if self.gate is None:
            raise RuntimeError("Posterior gate was not created.")

        logits_h = []
        mass_h = []
        for k in range(h):
            out_k = _mode_out_from_posterior(out, k)
            feat_k, mask_k = _ORIG_BUILD_SOFT_VIEW(
                out_k,
                weight_floor=float(build_args["weight_floor"]),
                scale_features_by_weight=bool(build_args["scale_features_by_weight"]),
                include_flags=bool(build_args["include_flags"]),
            )
            lk = self.base_model(feat_a, mask_a, feat_k, mask_k).squeeze(1)
            mk = out_k["cand_weights"].sum(dim=1) / max(float(out_k["cand_weights"].shape[1]), 1.0)
            logits_h.append(lk)
            mass_h.append(mk)

        logits_stack = torch.stack(logits_h, dim=1)  # [B,H]
        mass_stack = torch.stack(mass_h, dim=1)      # [B,H]
        score_std = logits_stack.std(dim=1, unbiased=False, keepdim=True)  # [B,1]
        pi_entropy = (-(pi * torch.log(pi + 1e-8)).sum(dim=1, keepdim=True) / math.log(max(float(h), 2.0)))

        gate_in = torch.cat([logits_stack, pi, mass_stack, score_std, pi_entropy], dim=1)
        gate_logits = self.gate(gate_in) + float(_POST_GATE_PRIOR_SCALE) * torch.log(pi + 1e-8)
        alpha = torch.softmax(gate_logits, dim=1)
        out_logit = (alpha * logits_stack).sum(dim=1, keepdim=True)
        return out_logit


def _train_reconstructor_weighted_ctx(*args, **kwargs):
    global _STAGE_CONTEXT
    prev = _STAGE_CONTEXT
    _STAGE_CONTEXT = "stageA"
    try:
        return _ORIG_TRAIN_RECO_WEIGHTED(*args, **kwargs)
    finally:
        _STAGE_CONTEXT = prev


def _train_joint_dual_ctx(*args, **kwargs):
    global _STAGE_CONTEXT
    prev = _STAGE_CONTEXT
    _STAGE_CONTEXT = "joint"
    try:
        return _ORIG_TRAIN_JOINT_DUAL(*args, **kwargs)
    finally:
        _STAGE_CONTEXT = prev


base.OfflineReconstructor = OfflineReconstructorTruePosterior
base.reco_base.OfflineReconstructor = OfflineReconstructorTruePosterior
base.build_soft_corrected_view = _build_soft_corrected_view_cache
base.DualViewCrossAttnClassifier = DualViewCrossAttnClassifierTruePosterior
base.compute_reconstruction_losses_weighted = _compute_reco_losses_trueposterior
base.train_reconstructor_weighted = _train_reconstructor_weighted_ctx
base.train_joint_dual = _train_joint_dual_ctx


if __name__ == "__main__":
    print(
        "[TruePosterior] enabled "
        f"(K={_POST_K}, mode_dim={_POST_MODE_DIM}, token_hidden={_POST_TOKEN_HIDDEN}, "
        f"assign_tau={_POST_ASSIGN_TAU:.3f}, "
        f"w_cal_stageA={_POST_W_CAL_STAGEA:.4f}, w_div_stageA={_POST_W_DIV_STAGEA:.4f}, w_ent_stageA={_POST_W_ENT_STAGEA:.4f}, "
        f"w_cal_joint={_POST_W_CAL_JOINT:.4f}, w_div_joint={_POST_W_DIV_JOINT:.4f}, w_ent_joint={_POST_W_ENT_JOINT:.4f}, "
        f"gate_hidden={_POST_GATE_HIDDEN}, gate_drop={_POST_GATE_DROPOUT:.3f}, prior_scale={_POST_GATE_PRIOR_SCALE:.3f})"
    )
    base.main()


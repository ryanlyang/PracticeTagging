#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Forces Hungarian set loss as the primary set-level objective.
- Adds matched-pair token auxiliary supervision from the same Hungarian assignment.

Key idea:
- Keep one forward pass and existing global objectives.
- Use Hungarian matches to provide per-token correction supervision
  (eta/phi/logpt/logE) for improved local credit assignment.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


_ORIG_COMPUTE_RECO_LOSSES_WEIGHTED = base.compute_reconstruction_losses_weighted

_FORCE_HUNGARIAN_SET = int(os.environ.get("HUNGARIAN_TOKENAUX_FORCE_SET", "1")) != 0
_HUNGARIAN_TOKEN_AUX_LAMBDA = float(max(float(os.environ.get("HUNGARIAN_TOKEN_AUX_LAMBDA", "0.15")), 0.0))
_HUNGARIAN_TOKEN_AUX_W_LOGPT = float(max(float(os.environ.get("HUNGARIAN_TOKEN_AUX_W_LOGPT", "1.0")), 0.0))
_HUNGARIAN_TOKEN_AUX_W_ETA = float(max(float(os.environ.get("HUNGARIAN_TOKEN_AUX_W_ETA", "0.6")), 0.0))
_HUNGARIAN_TOKEN_AUX_W_PHI = float(max(float(os.environ.get("HUNGARIAN_TOKEN_AUX_W_PHI", "0.6")), 0.0))
_HUNGARIAN_TOKEN_AUX_W_LOGE = float(max(float(os.environ.get("HUNGARIAN_TOKEN_AUX_W_LOGE", "0.25")), 0.0))
_HUNGARIAN_TOKEN_AUX_BETA = float(max(float(os.environ.get("HUNGARIAN_TOKEN_AUX_BETA", "0.08")), 1e-6))
_HUNGARIAN_TOKEN_AUX_MIN_MATCH_WEIGHT = float(max(float(os.environ.get("HUNGARIAN_TOKEN_AUX_MIN_MATCH_WEIGHT", "0.05")), 0.0))

_WARNED_NO_SCIPY = False


def _huber_abs(err: torch.Tensor, beta: float) -> torch.Tensor:
    """Smooth-L1 style robust penalty around 0 for absolute error tensor."""
    a = err.abs()
    b = max(float(beta), 1e-6)
    return torch.where(a < b, 0.5 * (a * a) / b, a - 0.5 * b)


def _hungarian_token_aux_vec(
    out: Dict[str, torch.Tensor],
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns per-jet vectors:
      aux_vec: matched-pair token supervision loss
      match_frac_vec: matched target fraction per jet
      pair_cost_vec: unweighted mean pair loss per jet
    """
    eps = 1e-8
    pred = out["cand_tokens"]
    cand_w = out["cand_weights"].clamp(0.0, 1.0)

    cost = base.reco_base._token_cost_matrix(pred, const_off)
    valid_tgt = mask_off.unsqueeze(1)
    cost = torch.where(valid_tgt, cost, torch.full_like(cost, 1e4))

    bsz = int(cost.shape[0])
    aux_list = []
    match_frac_list = []
    pair_cost_list = []

    for bi in range(bsz):
        n_tgt = int(mask_off[bi].sum().item())
        zero = torch.zeros((), device=cost.device, dtype=cost.dtype)
        if n_tgt <= 0:
            aux_list.append(zero)
            match_frac_list.append(zero)
            pair_cost_list.append(zero)
            continue

        c_bt = cost[bi, :, :n_tgt]
        c_np = c_bt.detach().cpu().numpy()
        row_ind, col_ind = base.linear_sum_assignment(c_np)  # type: ignore[misc]

        if len(row_ind) == 0:
            aux_list.append(zero)
            match_frac_list.append(zero)
            pair_cost_list.append(zero)
            continue

        row_t = torch.as_tensor(row_ind, device=cost.device, dtype=torch.long)
        col_t = torch.as_tensor(col_ind, device=cost.device, dtype=torch.long)

        p = pred[bi, row_t, :4]
        t = const_off[bi, col_t, :4]

        p_pt = p[:, 0].clamp(min=eps)
        p_eta = p[:, 1]
        p_phi = p[:, 2]
        p_E = p[:, 3].clamp(min=eps)

        t_pt = t[:, 0].clamp(min=eps)
        t_eta = t[:, 1]
        t_phi = t[:, 2]
        t_E = t[:, 3].clamp(min=eps)

        d_logpt = torch.log(p_pt) - torch.log(t_pt)
        d_eta = p_eta - t_eta
        d_phi = torch.atan2(torch.sin(p_phi - t_phi), torch.cos(p_phi - t_phi))
        d_logE = torch.log(p_E) - torch.log(t_E)

        l_pair = (
            _HUNGARIAN_TOKEN_AUX_W_LOGPT * _huber_abs(d_logpt, _HUNGARIAN_TOKEN_AUX_BETA)
            + _HUNGARIAN_TOKEN_AUX_W_ETA * _huber_abs(d_eta, _HUNGARIAN_TOKEN_AUX_BETA)
            + _HUNGARIAN_TOKEN_AUX_W_PHI * _huber_abs(d_phi, _HUNGARIAN_TOKEN_AUX_BETA)
            + _HUNGARIAN_TOKEN_AUX_W_LOGE * _huber_abs(d_logE, _HUNGARIAN_TOKEN_AUX_BETA)
        )

        wm = cand_w[bi, row_t]
        if _HUNGARIAN_TOKEN_AUX_MIN_MATCH_WEIGHT > 0.0:
            wm = wm.clamp(min=float(_HUNGARIAN_TOKEN_AUX_MIN_MATCH_WEIGHT))

        aux_i = (wm * l_pair).sum() / (wm.sum() + eps)
        pair_i = l_pair.mean()
        frac_i = torch.tensor(
            float(len(col_ind)) / float(max(n_tgt, 1)),
            device=cost.device,
            dtype=cost.dtype,
        )

        aux_list.append(aux_i)
        match_frac_list.append(frac_i)
        pair_cost_list.append(pair_i)

    return (
        torch.stack(aux_list, dim=0),
        torch.stack(match_frac_list, dim=0),
        torch.stack(pair_cost_list, dim=0),
    )


def compute_reconstruction_losses_weighted_hungarian_tokenaux(
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
    global _WARNED_NO_SCIPY

    # Force Hungarian as primary set objective unless disabled.
    if _FORCE_HUNGARIAN_SET:
        loss_cfg_eff = dict(loss_cfg)
        loss_cfg_eff["set_loss_mode"] = "hungarian"
    else:
        loss_cfg_eff = loss_cfg

    losses = _ORIG_COMPUTE_RECO_LOSSES_WEIGHTED(
        out=out,
        const_hlt=const_hlt,
        mask_hlt=mask_hlt,
        const_off=const_off,
        mask_off=mask_off,
        budget_merge_true=budget_merge_true,
        budget_eff_true=budget_eff_true,
        loss_cfg=loss_cfg_eff,
        sample_weight=sample_weight,
    )

    zero = torch.zeros_like(losses["total"])
    losses["hungarian_token_aux"] = zero
    losses["hungarian_match_frac"] = zero
    losses["hungarian_pair_cost"] = zero

    if _HUNGARIAN_TOKEN_AUX_LAMBDA <= 0.0:
        return losses

    if (not getattr(base, "_HAS_SCIPY_HUNGARIAN", False)) or (getattr(base, "linear_sum_assignment", None) is None):
        if not _WARNED_NO_SCIPY:
            print("[Hungarian-TokenAux] SciPy Hungarian unavailable; token auxiliary disabled.")
            _WARNED_NO_SCIPY = True
        return losses

    sw = None
    if sample_weight is not None:
        sw = sample_weight.float().clamp(min=0.0)

    aux_vec, match_frac_vec, pair_cost_vec = _hungarian_token_aux_vec(
        out=out,
        const_off=const_off,
        mask_off=mask_off,
    )

    aux = base._weighted_batch_mean(aux_vec, sw)
    match_frac = base._weighted_batch_mean(match_frac_vec, sw)
    pair_cost = base._weighted_batch_mean(pair_cost_vec, sw)

    losses["hungarian_token_aux"] = aux
    losses["hungarian_match_frac"] = match_frac
    losses["hungarian_pair_cost"] = pair_cost
    losses["total"] = losses["total"] + float(_HUNGARIAN_TOKEN_AUX_LAMBDA) * aux
    return losses


base.compute_reconstruction_losses_weighted = compute_reconstruction_losses_weighted_hungarian_tokenaux


if __name__ == "__main__":
    print(
        "[Hungarian-TokenAux] "
        f"force_set={int(_FORCE_HUNGARIAN_SET)} "
        f"lambda={_HUNGARIAN_TOKEN_AUX_LAMBDA:.6g} "
        f"weights(logpt,eta,phi,logE)=({ _HUNGARIAN_TOKEN_AUX_W_LOGPT:.3g},"
        f"{ _HUNGARIAN_TOKEN_AUX_W_ETA:.3g},{ _HUNGARIAN_TOKEN_AUX_W_PHI:.3g},{ _HUNGARIAN_TOKEN_AUX_W_LOGE:.3g}) "
        f"beta={_HUNGARIAN_TOKEN_AUX_BETA:.4g}"
    )
    base.main()

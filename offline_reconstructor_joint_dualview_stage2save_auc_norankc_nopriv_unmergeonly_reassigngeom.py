#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper runner: identical to
offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py,
but with reconstructor reassign action constrained to geometry-only updates
(eta/phi), i.e. reassign does not modify token pT/E.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

import offline_reconstructor_no_gt_local30kv2 as reco_base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base_run


class OfflineReconstructorReassignGeomOnly(reco_base.OfflineReconstructor):
    """
    Same architecture/params/state_dict keys as base OfflineReconstructor.
    Only behavior change: reassign no longer contributes to d_logpt/d_logE.
    """

    def forward(
        self,
        feat_hlt: torch.Tensor,
        mask_hlt: torch.Tensor,
        const_hlt: torch.Tensor,
        stage_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        B, L, _ = feat_hlt.shape
        mask_safe = mask_hlt.clone()
        empty = ~mask_safe.any(dim=1)
        if empty.any():
            mask_safe[empty, 0] = True

        x = self.input_proj(feat_hlt)
        rel_bias = self._build_relpos_bias(const_hlt)
        for layer in self.encoder_layers:
            x = layer(x, mask_safe, rel_bias)
        x = self.token_norm(x)

        action_logits = self.action_head(x)
        action_prob = torch.softmax(action_logits, dim=-1)
        p_keep = action_prob[..., 0]
        p_unsmear = action_prob[..., 1]
        p_split = action_prob[..., 2]
        p_reassign = action_prob[..., 3]

        pt = const_hlt[..., 0].clamp(min=eps)
        eta = const_hlt[..., 1].clamp(min=-5.0, max=5.0)
        phi = const_hlt[..., 2]
        E = const_hlt[..., 3].clamp(min=eps)

        unsmear_delta = self.unsmear_head(x)
        reassign_delta = 0.35 * torch.tanh(self.reassign_head(x))

        # Reassign is geometry-only: no pT/E scaling contribution.
        d_logpt = stage_scale * p_unsmear * unsmear_delta[..., 0]
        d_eta = stage_scale * (p_unsmear * unsmear_delta[..., 1] + p_reassign * reassign_delta[..., 0])
        d_phi = stage_scale * (p_unsmear * unsmear_delta[..., 2] + p_reassign * reassign_delta[..., 1])
        d_logE = stage_scale * p_unsmear * unsmear_delta[..., 3]

        tok_pt = torch.exp(torch.clamp(torch.log(pt) + d_logpt, min=-9.0, max=9.0))
        tok_eta = (eta + 0.5 * torch.tanh(d_eta)).clamp(min=-5.0, max=5.0)
        tok_phi = reco_base.wrap_phi_t(phi + d_phi)
        tok_E = torch.exp(torch.clamp(torch.log(E) + d_logE, min=-9.0, max=11.0))
        tok_E = torch.maximum(tok_E, tok_pt * torch.cosh(tok_eta))

        tok_tokens = torch.stack([tok_pt, tok_eta, tok_phi, tok_E], dim=-1)
        tok_weight = (p_keep + p_unsmear + 0.35 * p_reassign + 0.15 * p_split).clamp(0.0, 1.0)
        tok_weight = tok_weight * mask_hlt.float()
        tok_merge_flag = torch.zeros_like(tok_weight)
        tok_eff_flag = torch.zeros_like(tok_weight)

        # Split children from each HLT token
        K = self.max_split_children
        split_exist = torch.sigmoid(self.split_exist_head(x))
        split_exist = split_exist * (p_split.unsqueeze(-1) * stage_scale) * mask_hlt.float().unsqueeze(-1)

        split_delta = self.split_delta_head(x).view(B, L, K, 3)
        split_frac = torch.sigmoid(split_delta[..., 0]) / float(K + 1)
        child_pt = pt.unsqueeze(-1) * split_frac
        child_eta = (eta.unsqueeze(-1) + 0.25 * torch.tanh(split_delta[..., 1])).clamp(min=-5.0, max=5.0)
        child_phi = reco_base.wrap_phi_t(phi.unsqueeze(-1) + 0.25 * torch.tanh(split_delta[..., 2]))
        child_E = child_pt * torch.cosh(child_eta)

        child_tokens = torch.stack([child_pt, child_eta, child_phi, child_E], dim=-1).reshape(B, L * K, 4)
        child_weight = split_exist.reshape(B, L * K).clamp(0.0, 1.0)
        child_merge_flag = torch.ones_like(child_weight)
        child_eff_flag = torch.zeros_like(child_weight)

        # Jet context and budget heads
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=~mask_safe, need_weights=False)
        ctx = pooled.squeeze(1)
        budget_raw = self.budget_head(ctx)
        budget_total = F.softplus(budget_raw[:, 0])
        budget_merge = F.softplus(budget_raw[:, 1])
        budget_eff = F.softplus(budget_raw[:, 2])

        # Generation slots (efficiency-loss recovery)
        gq = self.gen_queries.expand(B, -1, -1)
        gen_dec, _ = self.gen_attn(gq, x, x, key_padding_mask=~mask_safe, need_weights=False)
        gen_dec = self.gen_norm(gen_dec)
        gen_raw = self.gen_head(gen_dec)
        gen_exist = torch.sigmoid(self.gen_exist_head(gen_dec).squeeze(-1)) * stage_scale

        gen_pt = torch.exp(torch.clamp(gen_raw[..., 0], min=-8.0, max=6.0))
        gen_eta = gen_raw[..., 1].clamp(min=-5.0, max=5.0)
        gen_phi = reco_base.wrap_phi_t(gen_raw[..., 2])
        gen_E = torch.exp(torch.clamp(gen_raw[..., 3], min=-8.0, max=10.0))
        gen_E = torch.maximum(gen_E, gen_pt * torch.cosh(gen_eta))
        gen_tokens = torch.stack([gen_pt, gen_eta, gen_phi, gen_E], dim=-1)
        gen_merge_flag = torch.zeros_like(gen_exist)
        gen_eff_flag = torch.ones_like(gen_exist)

        # Budget-informed calibration of generated/split weights.
        child_sum = child_weight.sum(dim=1, keepdim=True) + eps
        gen_sum = gen_exist.sum(dim=1, keepdim=True) + eps
        child_scale = (budget_merge.unsqueeze(1) / child_sum).clamp(min=0.25, max=4.0)
        gen_scale = (budget_eff.unsqueeze(1) / gen_sum).clamp(min=0.25, max=4.0)
        child_weight = (child_weight * child_scale).clamp(0.0, 1.0)
        gen_exist = (gen_exist * gen_scale).clamp(0.0, 1.0)

        cand_tokens = torch.cat([tok_tokens, child_tokens, gen_tokens], dim=1)
        cand_weights = torch.cat([tok_weight, child_weight, gen_exist], dim=1)
        cand_merge_flags = torch.cat([tok_merge_flag, child_merge_flag, gen_merge_flag], dim=1)
        cand_eff_flags = torch.cat([tok_eff_flag, child_eff_flag, gen_eff_flag], dim=1)

        return {
            "cand_tokens": cand_tokens,
            "cand_weights": cand_weights,
            "cand_merge_flags": cand_merge_flags,
            "cand_eff_flags": cand_eff_flags,
            "action_prob": action_prob,
            "child_weight": child_weight,
            "gen_weight": gen_exist,
            "budget_total": budget_total,
            "budget_merge": budget_merge,
            "budget_eff": budget_eff,
            "split_delta": split_delta,
            "gen_tokens": gen_tokens,
        }


def main() -> None:
    # Swap the reconstructor class used inside the base runner.
    base_run.OfflineReconstructor = OfflineReconstructorReassignGeomOnly
    base_run.main()


if __name__ == "__main__":
    main()

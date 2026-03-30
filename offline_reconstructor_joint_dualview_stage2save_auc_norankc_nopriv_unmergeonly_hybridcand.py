#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Adds a hybrid corrected-view ablation:
  * keep original compressed corrected view branch
  * add top-M raw reco-candidate branch into the dual classifier

This is an architecture-only ablation; reconstructor losses/heads are unchanged.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


_CAND_TOPM = int(max(int(os.environ.get("HYBRID_CAND_TOPM", "96")), 8))
_CAND_SCALE_BY_WEIGHT = bool(int(os.environ.get("HYBRID_CAND_SCALE_BY_WEIGHT", "1")))
_CAND_INPUT_DIM = int(max(int(os.environ.get("HYBRID_CAND_INPUT_DIM", "10")), 4))
_CAND_BRANCH_DROPOUT = float(os.environ.get("HYBRID_CAND_BRANCH_DROPOUT", "0.1"))

_ORIG_BUILD_SOFT_VIEW = base.build_soft_corrected_view
_HYBRID_LAST_CAND: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


def _build_candidate_branch_from_reco(
    reco_out: dict,
    weight_floor: float,
    topm: int,
    scale_features_by_weight: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    reco_out = base.enforce_unmerge_only_output(reco_out)

    cand_tokens = reco_out["cand_tokens"]  # [B, N, 4]
    cand_w = reco_out["cand_weights"].clamp(0.0, 1.0)  # [B, N]
    cand_merge = reco_out["cand_merge_flags"].clamp(0.0, 1.0)  # [B, N]
    cand_eff = reco_out["cand_eff_flags"].clamp(0.0, 1.0)  # [B, N]

    bsz, n_cand, _ = cand_tokens.shape
    k = min(int(topm), int(n_cand))

    vals, idx = torch.topk(cand_w, k=k, dim=1, largest=True, sorted=True)  # [B, k], [B, k]
    gather_idx_tok = idx.unsqueeze(-1).expand(-1, -1, cand_tokens.size(-1))
    top_tok = torch.gather(cand_tokens, dim=1, index=gather_idx_tok)  # [B, k, 4]
    top_merge = torch.gather(cand_merge, dim=1, index=idx)  # [B, k]
    top_eff = torch.gather(cand_eff, dim=1, index=idx)  # [B, k]

    if k < int(topm):
        pad = int(topm - k)
        top_tok = torch.cat([top_tok, torch.zeros(bsz, pad, 4, device=top_tok.device, dtype=top_tok.dtype)], dim=1)
        vals = torch.cat([vals, torch.zeros(bsz, pad, device=vals.device, dtype=vals.dtype)], dim=1)
        top_merge = torch.cat(
            [top_merge, torch.zeros(bsz, pad, device=top_merge.device, dtype=top_merge.dtype)], dim=1
        )
        top_eff = torch.cat([top_eff, torch.zeros(bsz, pad, device=top_eff.device, dtype=top_eff.dtype)], dim=1)

    mask_c = vals > float(weight_floor)
    empty = ~mask_c.any(dim=1)
    if empty.any():
        mask_c = mask_c.clone()
        mask_c[empty, 0] = True

    feat7 = base.compute_features_torch(top_tok, mask_c)
    if bool(scale_features_by_weight):
        feat7 = feat7 * vals.unsqueeze(-1)

    extra = torch.stack([vals, top_merge, top_eff], dim=-1)  # [B, M, 3]
    feat_c = torch.cat([feat7, extra], dim=-1)
    feat_c = torch.nan_to_num(feat_c, nan=0.0, posinf=0.0, neginf=0.0)
    feat_c = feat_c * mask_c.unsqueeze(-1).float()

    # Optional truncate/pad to requested input dim if needed.
    if feat_c.size(-1) > _CAND_INPUT_DIM:
        feat_c = feat_c[:, :, :_CAND_INPUT_DIM]
    elif feat_c.size(-1) < _CAND_INPUT_DIM:
        pad_dim = int(_CAND_INPUT_DIM - feat_c.size(-1))
        feat_c = torch.cat(
            [feat_c, torch.zeros(feat_c.size(0), feat_c.size(1), pad_dim, device=feat_c.device, dtype=feat_c.dtype)],
            dim=-1,
        )
    return feat_c, mask_c


def _build_soft_corrected_view_hybrid(
    reco_out,
    weight_floor: float = 1e-4,
    scale_features_by_weight: bool = True,
    include_flags: bool = False,
):
    global _HYBRID_LAST_CAND
    feat_b, mask_b = _ORIG_BUILD_SOFT_VIEW(
        reco_out,
        weight_floor=weight_floor,
        scale_features_by_weight=scale_features_by_weight,
        include_flags=include_flags,
    )
    feat_c, mask_c = _build_candidate_branch_from_reco(
        reco_out,
        weight_floor=float(weight_floor),
        topm=int(_CAND_TOPM),
        scale_features_by_weight=bool(_CAND_SCALE_BY_WEIGHT),
    )
    _HYBRID_LAST_CAND = (feat_c, mask_c)
    return feat_b, mask_b


class DualViewCrossAttnClassifierHybrid(nn.Module):
    def __init__(
        self,
        input_dim_a: int = 7,
        input_dim_b: int = 7,
        input_dim_c: int = 10,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.input_proj_a = nn.Sequential(
            nn.Linear(int(input_dim_a), self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.input_proj_b = nn.Sequential(
            nn.Linear(int(input_dim_b), self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.input_proj_c = nn.Sequential(
            nn.Linear(int(input_dim_c), self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(_CAND_BRANCH_DROPOUT)),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder_a = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.encoder_b = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.encoder_c = nn.TransformerEncoder(enc_layer, num_layers=max(2, int(num_layers // 2)))

        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pool_attn_a = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.pool_attn_b = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.pool_attn_c = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=dropout, batch_first=True)

        self.cross_a_to_b = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_b_to_a = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_a_to_c = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(self.embed_dim * 6)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 6, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
        ms = mask.clone()
        empty = ~ms.any(dim=1)
        if empty.any():
            ms[empty, 0] = True
        return ms

    def _encode_branch(self, feat: torch.Tensor, mask: torch.Tensor, proj: nn.Module, enc: nn.Module) -> torch.Tensor:
        bsz, sl, _ = feat.shape
        h = proj(feat.reshape(-1, feat.size(-1))).reshape(bsz, sl, -1)
        h = enc(h, src_key_padding_mask=~mask)
        return h

    def forward(self, feat_a, mask_a, feat_b, mask_b, feat_c=None, mask_c=None):
        global _HYBRID_LAST_CAND

        if feat_c is None or mask_c is None:
            if _HYBRID_LAST_CAND is not None:
                feat_c, mask_c = _HYBRID_LAST_CAND
            else:
                feat_c, mask_c = None, None

        use_c = feat_c is not None and mask_c is not None

        mask_a_safe = self._safe_mask(mask_a)
        mask_b_safe = self._safe_mask(mask_b)

        h_a = self._encode_branch(feat_a, mask_a_safe, self.input_proj_a, self.encoder_a)
        h_b = self._encode_branch(feat_b, mask_b_safe, self.input_proj_b, self.encoder_b)

        q = self.pool_query.expand(feat_a.size(0), -1, -1)
        pooled_a, _ = self.pool_attn_a(q, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)
        pooled_b, _ = self.pool_attn_b(q, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
        cross_a_b, _ = self.cross_a_to_b(pooled_a, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
        cross_b_a, _ = self.cross_b_to_a(pooled_b, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)

        if use_c:
            mask_c_safe = self._safe_mask(mask_c)
            h_c = self._encode_branch(feat_c, mask_c_safe, self.input_proj_c, self.encoder_c)
            pooled_c, _ = self.pool_attn_c(q, h_c, h_c, key_padding_mask=~mask_c_safe, need_weights=False)
            cross_a_c, _ = self.cross_a_to_c(pooled_a, h_c, h_c, key_padding_mask=~mask_c_safe, need_weights=False)
        else:
            pooled_c = torch.zeros_like(pooled_a)
            cross_a_c = torch.zeros_like(pooled_a)

        fused = torch.cat([pooled_a, pooled_b, cross_a_b, cross_b_a, pooled_c, cross_a_c], dim=-1).squeeze(1)
        fused = self.norm(fused)
        return self.classifier(fused)


base.build_soft_corrected_view = _build_soft_corrected_view_hybrid
base.DualViewCrossAttnClassifier = DualViewCrossAttnClassifierHybrid


if __name__ == "__main__":
    print(
        "[HybridCand] enabled "
        f"(topM={_CAND_TOPM}, cand_input_dim={_CAND_INPUT_DIM}, "
        f"scale_by_weight={int(_CAND_SCALE_BY_WEIGHT)})"
    )
    base.main()


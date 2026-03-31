#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full training pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Swaps dual-view head with an explicit residual-stream variant:
  * Stream A: HLT features
  * Stream B: corrected-view features
  * Stream D: explicit residual features (corrected - HLT) + corrected extras
  * Shared cross-attention fusion with optional adaptive block gating
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


_DELTA_RES_UNSCALE_CORR7 = int(os.environ.get("DELTA_RES_UNSCALE_CORR7", "1")) != 0
_DELTA_RES_USE_BLOCK_GATING = int(os.environ.get("DELTA_RES_USE_BLOCK_GATING", "1")) != 0
_DELTA_RES_GATE_DROPOUT = float(max(min(float(os.environ.get("DELTA_RES_GATE_DROPOUT", "0.05")), 0.5), 0.0))


class DualViewCrossAttnClassifierDeltaResidual(nn.Module):
    """
    Dual-view classifier with explicit residual stream.
    """

    def __init__(
        self,
        input_dim_a=7,
        input_dim_b=7,
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.input_dim_a = int(input_dim_a)
        self.input_dim_b = int(input_dim_b)
        self.input_dim_d = int(input_dim_b)  # residual7 + corrected extras
        self.unscale_corr7 = bool(_DELTA_RES_UNSCALE_CORR7)
        self.use_block_gating = bool(_DELTA_RES_USE_BLOCK_GATING)

        def _proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )

        self.input_proj_a = _proj(self.input_dim_a)
        self.input_proj_b = _proj(self.input_dim_b)
        self.input_proj_d = _proj(self.input_dim_d)

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
        self.encoder_d = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pool_attn_a = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True)
        self.pool_attn_b = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True)
        self.pool_attn_d = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True)

        self.cross_a_to_b = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True)
        self.cross_b_to_a = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True)
        self.cross_d_to_b = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True)
        self.cross_b_to_d = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True)

        self.n_blocks = 7
        if self.use_block_gating:
            self.block_gate = nn.Sequential(
                nn.Linear(self.embed_dim * 3, self.embed_dim),
                nn.GELU(),
                nn.Dropout(_DELTA_RES_GATE_DROPOUT),
                nn.Linear(self.embed_dim, self.n_blocks),
            )
        else:
            self.block_gate = None

        fused_dim = self.embed_dim * self.n_blocks
        self.norm = nn.LayerNorm(fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(192, 1),
        )

    def _safe_mask(self, m: torch.Tensor) -> torch.Tensor:
        out = m.clone()
        empty = ~out.any(dim=1)
        if empty.any():
            out[empty, 0] = True
        return out

    def _project_seq(self, feat: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        b, l, d = feat.shape
        return proj(feat.reshape(-1, d)).reshape(b, l, -1)

    def forward(self, feat_a, mask_a, feat_b, mask_b):
        bsz, seq_len, _ = feat_a.shape
        mask_a_safe = self._safe_mask(mask_a)
        mask_b_safe = self._safe_mask(mask_b)

        # Build explicit residual stream.
        corr7 = feat_b[..., : self.input_dim_a]
        if self.unscale_corr7 and feat_b.size(-1) > self.input_dim_a:
            tok_w = feat_b[..., self.input_dim_a : self.input_dim_a + 1].clamp(min=1e-3)
            corr7 = corr7 / tok_w
        delta7 = corr7 - feat_a[..., : self.input_dim_a]
        if feat_b.size(-1) > self.input_dim_a:
            delta_extra = feat_b[..., self.input_dim_a :]
            feat_d = torch.cat([delta7, delta_extra], dim=-1)
        else:
            feat_d = delta7
        mask_d_safe = mask_b_safe

        h_a = self._project_seq(feat_a, self.input_proj_a)
        h_b = self._project_seq(feat_b, self.input_proj_b)
        h_d = self._project_seq(feat_d, self.input_proj_d)

        h_a = self.encoder_a(h_a, src_key_padding_mask=~mask_a_safe)
        h_b = self.encoder_b(h_b, src_key_padding_mask=~mask_b_safe)
        h_d = self.encoder_d(h_d, src_key_padding_mask=~mask_d_safe)

        q = self.pool_query.expand(bsz, -1, -1)
        pooled_a, _ = self.pool_attn_a(q, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)
        pooled_b, _ = self.pool_attn_b(q, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
        pooled_d, _ = self.pool_attn_d(q, h_d, h_d, key_padding_mask=~mask_d_safe, need_weights=False)

        cross_a, _ = self.cross_a_to_b(pooled_a, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
        cross_b, _ = self.cross_b_to_a(pooled_b, h_a, h_a, key_padding_mask=~mask_a_safe, need_weights=False)
        cross_d, _ = self.cross_d_to_b(pooled_d, h_b, h_b, key_padding_mask=~mask_b_safe, need_weights=False)
        cross_bd, _ = self.cross_b_to_d(pooled_b, h_d, h_d, key_padding_mask=~mask_d_safe, need_weights=False)

        blocks = torch.stack(
            [
                pooled_a.squeeze(1),
                pooled_b.squeeze(1),
                pooled_d.squeeze(1),
                cross_a.squeeze(1),
                cross_b.squeeze(1),
                cross_d.squeeze(1),
                cross_bd.squeeze(1),
            ],
            dim=1,
        )  # [B,7,E]

        if self.block_gate is not None:
            gate_in = torch.cat([pooled_a.squeeze(1), pooled_b.squeeze(1), pooled_d.squeeze(1)], dim=-1)  # [B,3E]
            gate_logits = self.block_gate(gate_in)  # [B,7]
            gate = torch.softmax(gate_logits, dim=1)
            blocks = blocks * gate.unsqueeze(-1)

        fused = blocks.reshape(bsz, self.n_blocks * self.embed_dim)
        fused = self.norm(fused)
        logits = self.classifier(fused)
        return logits


base.DualViewCrossAttnClassifier = DualViewCrossAttnClassifierDeltaResidual


if __name__ == "__main__":
    print(
        "[DeltaResidualDual] enabled "
        f"(unscale_corr7={int(_DELTA_RES_UNSCALE_CORR7)}, "
        f"block_gating={int(_DELTA_RES_USE_BLOCK_GATING)}, "
        f"gate_dropout={_DELTA_RES_GATE_DROPOUT:.3f})"
    )
    base.main()


"""
Parentness + per-parent count models for unmerger pretraining.

Task definition (default):
- Inputs are HLT-view tokens per jet: x_hlt [B,S,D] with mask [B,S].
- Head A (parentness): predict which HLT tokens correspond to merged parents (should be split).
- Head B (count): predict per-token missing count k_i (typically group_size-1) for merged parents.

Design:
- Shared Transformer encoder (token-level).
- Two lightweight heads.
- Gated count: k_pred = sigmoid(parent_logit) * softplus(raw_k),
  so non-parent tokens naturally get near-zero counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ParentCountOutput:
    parent_logit: torch.Tensor  # [B,S]
    k_raw: torch.Tensor  # [B,S]
    k_pred: torch.Tensor  # [B,S]


class ParentnessCountTransformer(nn.Module):
    """Token-level Transformer encoder with parentness + count heads."""

    def __init__(
        self,
        *,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        softplus_beta: float = 0.2,
        init_k_bias: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.softplus_beta = float(softplus_beta)

        # Keep consistent with other experiments: project per-token then encode.
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
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
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.parent_head = nn.Linear(self.embed_dim, 1)
        self.k_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 1),
        )

        self._init_weights()
        if init_k_bias is not None:
            # Initialize the last bias so initial softplus output is near a reasonable prior.
            b = float(max(float(init_k_bias), 1e-6))
            inv = math.log(math.expm1(self.softplus_beta * b)) / self.softplus_beta
            last = self.k_head[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                last.bias.data.fill_(float(inv))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> ParentCountOutput:
        """
        Args:
          x: [B,S,D] engineered features
          mask: [B,S] bool, True for valid tokens
        Returns:
          ParentCountOutput with tensors [B,S]
        """
        B, S, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        h = x.reshape(B * S, self.input_dim)
        h = self.input_proj(h)
        h = h.reshape(B, S, self.embed_dim)

        h = self.transformer(h, src_key_padding_mask=~mask)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        parent_logit = self.parent_head(h).squeeze(-1)  # [B,S]

        k_in = h.reshape(B * S, self.embed_dim)
        k_raw = self.k_head(k_in).reshape(B, S)  # [B,S]
        k_pos = torch.nn.functional.softplus(k_raw, beta=self.softplus_beta)  # [B,S] >= 0

        gate = torch.sigmoid(parent_logit)
        k_pred = gate * k_pos

        return ParentCountOutput(parent_logit=parent_logit, k_raw=k_raw, k_pred=k_pred)


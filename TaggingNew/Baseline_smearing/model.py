"""
Models for the *baseline KD pipeline* extracted from `Previous/Flowmatching.ipynb`.

This module intentionally keeps the same architecture principles as the notebook:
- ParticleTransformerKD: Transformer encoder + learnable pooling query attention
- Optional `return_attention=True` to support attention distillation
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(dim)),
            # LayerNorm over feature dimension per sample (token / pooled vector), independent of batch statistics.
            nn.LayerNorm(int(dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ParticleTransformerKD(nn.Module):
    """Flowmatching-style tagger used as teacher/baseline/student.

    Notes:
    - Input: engineered particle features [B,S,7]
    - Mask: [B,S] with True for valid tokens
    - Returns:
      - logits [B,1] (or [B] after squeeze in training code)
      - optionally attention weights [B,S] from the pooling attention
    """

    def __init__(
        self,
        *,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            # LayerNorm to avoid BatchNorm statistics being polluted by padding/masked tokens.
            nn.LayerNorm(self.embed_dim),
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

        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            ResidualBlock(128, dropout=float(dropout)),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, return_attention: bool = False):
        # x: [B,S,D], mask: [B,S] True for valid tokens
        B, S, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        h = x.reshape(B * S, self.input_dim)
        h = self.input_proj(h)
        h = h.reshape(B, S, self.embed_dim)

        h = self.transformer(h, src_key_padding_mask=~mask)

        q = self.pool_query.expand(B, -1, -1)  # [B,1,E]
        pooled, attn = self.pool_attn(
            q, h, h, key_padding_mask=~mask, need_weights=True, average_attn_weights=True
        )  # pooled: [B,1,E], attn: [B,1,S]

        z = self.norm(pooled.squeeze(1))
        logits = self.classifier(z)  # [B,1]

        if return_attention:
            return logits, attn.squeeze(1)  # [B,1], [B,S]
        return logits

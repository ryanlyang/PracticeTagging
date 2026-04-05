"""
Count-only unmerger models.

Goal:
- Predict jet-level total missing count ΔN (e.g. N_off - N_hlt) from HLT features.

Design principles:
- Keep preprocessing consistent with `unmerge/unmerge.ipynb` (7D engineered features).
- Use a Transformer encoder + learnable pooling query attention (Flowmatching-style),
  so the backbone is consistent with later reconstruction-style components.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class CountTransformer(nn.Module):
    """Transformer encoder + pooling attention -> predict ΔN (non-negative).

    Inputs:
      x: [B,S,D] engineered features (typically D=7)
      mask: [B,S] boolean, True for valid tokens

    Output:
      pred: [B] non-negative (softplus applied)
    """

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
        init_bias: float | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.softplus_beta = float(softplus_beta)

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

        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 1),
        )

        self._init_weights()
        if init_bias is not None:
            # Initialize last bias so initial prediction is near a reasonable prior.
            # pred = softplus(raw, beta)  => raw = inv_softplus(pred, beta)
            b = float(init_bias)
            b = max(b, 1e-6)
            inv = math.log(math.expm1(self.softplus_beta * b)) / self.softplus_beta
            last = self.head[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                last.bias.data.fill_(float(inv))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B,S,D], mask: [B,S]
        B, S, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        h = x.reshape(B * S, self.input_dim)
        h = self.input_proj(h)
        h = h.reshape(B, S, self.embed_dim)

        h = self.transformer(h, src_key_padding_mask=~mask)

        q = self.pool_query.expand(B, -1, -1)  # [B,1,E]
        pooled, _ = self.pool_attn(q, h, h, key_padding_mask=~mask, need_weights=False)
        z = self.norm(pooled.squeeze(1))

        raw = self.head(z).squeeze(-1)  # [B]
        # Smaller beta keeps gradients alive for negative raw values.
        pred = torch.nn.functional.softplus(raw, beta=self.softplus_beta)  # enforce non-negative
        return pred

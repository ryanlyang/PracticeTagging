"""
Joint unsmear models.

This directory keeps only the models actually needed for joint training:
- `ParticleTransformerKD`: offline teacher / HLT baseline / HLT+KD
- `SharedEncoderUnsmearClassifier`: joint model with a shared encoder
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(dim)),
            nn.LayerNorm(int(dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ParticleTransformerKD(nn.Module):
    """Downstream classifier."""

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
        B, S, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        h = x.reshape(B * S, self.input_dim)
        h = self.input_proj(h)
        h = h.reshape(B, S, self.embed_dim)
        h = self.transformer(h, src_key_padding_mask=~mask)

        q = self.pool_query.expand(B, -1, -1)
        pooled, attn = self.pool_attn(
            q, h, h, key_padding_mask=~mask, need_weights=True, average_attn_weights=True
        )
        z = self.norm(pooled.squeeze(1))
        logits = self.classifier(z)

        if return_attention:
            return logits, attn.squeeze(1)
        return logits


class SharedEncoderUnsmearClassifier(nn.Module):
    """Joint model with a shared encoder for unsmear regression and jet classification."""

    def __init__(
        self,
        *,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        unsmear_decoder_layers: int = 2,
        unsmear_decoder_heads: int = 8,
        unsmear_decoder_ff_dim: int = 512,
        unsmear_decoder_dropout: float | None = None,
        return_reco: bool = True,
        add_mask_channel: bool = True,
        mask_output: bool = True,
        use_positional_embedding: bool = True,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.return_reco = bool(return_reco)
        self.add_mask_channel = bool(add_mask_channel)
        self.mask_output = bool(mask_output)
        self.use_positional_embedding = bool(use_positional_embedding)
        self.max_seq_len = int(max_seq_len)
        self.unsmear_decoder_layers = int(unsmear_decoder_layers)
        self.unsmear_decoder_heads = int(unsmear_decoder_heads)
        self.unsmear_decoder_ff_dim = int(unsmear_decoder_ff_dim)
        self.unsmear_decoder_dropout = (
            float(dropout)
            if unsmear_decoder_dropout is None
            else float(unsmear_decoder_dropout)
        )

        in_dim = self.input_dim + (1 if self.add_mask_channel else 0)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        if self.use_positional_embedding:
            self.pos_embed = nn.Embedding(self.max_seq_len, self.embed_dim)
        else:
            self.pos_embed = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.unsmear_decoder_heads,
            dim_feedforward=self.unsmear_decoder_ff_dim,
            dropout=self.unsmear_decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.unsmear_decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=self.unsmear_decoder_layers,
        )
        self.unsmear_decoder_input = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.unsmear_decoder_dropout),
        )
        self.unsmear_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.unsmear_decoder_dropout),
            nn.Linear(self.embed_dim, self.input_dim),
        )

        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True
        )
        self.cls_norm = nn.LayerNorm(self.embed_dim)
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
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the shared encoder."""
        B, S, _ = x.shape
        if self.pos_embed is not None and S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len={self.max_seq_len}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        inp = x
        if self.add_mask_channel:
            inp = torch.cat([inp, mask.to(dtype=x.dtype).unsqueeze(-1)], dim=-1)

        h = self.input_proj(inp.reshape(B * S, -1)).reshape(B, S, self.embed_dim)
        if self.pos_embed is not None:
            # Use learnable positional embeddings to preserve sequence-order priors.
            pos_idx = torch.arange(S, device=x.device)
            h = h + self.pos_embed(pos_idx).unsqueeze(0)
        h = h * mask.to(dtype=h.dtype).unsqueeze(-1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return h

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, return_attention: bool = False):
        """Return `(reco, logits)` and optionally the pooling attention weights."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.encode(x, mask)

        dec_in = self.unsmear_decoder_input(h)
        dec_in = dec_in * mask.to(dtype=dec_in.dtype).unsqueeze(-1)
        dec_out = self.unsmear_decoder(
            tgt=dec_in,
            memory=h,
            tgt_key_padding_mask=~mask,
            memory_key_padding_mask=~mask,
        )
        dec_out = dec_out * mask.to(dtype=dec_out.dtype).unsqueeze(-1)
        delta = self.unsmear_head(dec_out)
        reco = (x + delta) if self.return_reco else delta
        if self.mask_output:
            reco = reco * mask.to(dtype=reco.dtype).unsqueeze(-1)

        q = self.pool_query.expand(x.shape[0], -1, -1)
        pooled, attn = self.pool_attn(
            q, h, h, key_padding_mask=~mask, need_weights=True, average_attn_weights=True
        )
        z = self.cls_norm(pooled.squeeze(1))
        logits = self.classifier(z)

        if return_attention:
            return reco, logits, attn.squeeze(1)
        return reco, logits

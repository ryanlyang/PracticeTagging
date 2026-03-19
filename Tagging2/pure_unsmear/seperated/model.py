"""
Unsmear models (our internal implementation).

Main models are provided for comparison:
- Token-wise MLP regression (fastest local baseline)
- Token-wise Transformer regression (global token interaction baseline)
- UNet regression (fast deterministic regression)
- Flow Matching (conditional vector field + sampling/integration)

Notes:
- We use LayerNorm to avoid BatchNorm statistics being polluted by padding tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mask_conv_features(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a token mask to `[B,C,L]` features, resizing the mask if needed."""
    m = mask.to(dtype=h.dtype).unsqueeze(1)
    if m.size(-1) != h.size(-1):
        m = nn.functional.interpolate(m, size=h.size(-1), mode="nearest")
    return h * m


class _TokenMLPBlock(nn.Module):
    """Per-token MLP block with residual connection."""

    def __init__(self, dim: int, *, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(dim)),
            nn.LayerNorm(int(dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(dim), int(dim)),
            nn.LayerNorm(int(dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TokenMLP(nn.Module):
    """Token-wise MLP regressor for unsmearing."""

    def __init__(
        self,
        *,
        input_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        return_reco: bool = True,
        predict_logvar: bool = False,
        add_mask_channel: bool = True,
        mask_output: bool = True,
        w_dr: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.return_reco = bool(return_reco)
        self.predict_logvar = bool(predict_logvar)
        self.add_mask_channel = bool(add_mask_channel)
        self.mask_output = bool(mask_output)

        in_dim = self.input_dim + (1 if self.add_mask_channel else 0)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.Sequential(
            *[_TokenMLPBlock(self.hidden_dim, dropout=float(dropout)) for _ in range(int(num_layers))]
        )
        out_dim = (2 * self.input_dim) if self.predict_logvar else self.input_dim
        self.out = nn.Linear(self.hidden_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, S, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        inp = x
        if self.add_mask_channel:
            inp = torch.cat([inp, mask.to(dtype=x.dtype).unsqueeze(-1)], dim=-1)

        h = self.in_proj(inp.reshape(B * S, -1))
        h = self.blocks(h)
        out = self.out(h).reshape(B, S, -1)

        if self.predict_logvar:
            delta, log_var = torch.split(out, [self.input_dim, self.input_dim], dim=-1)
            mu = (x + delta) if self.return_reco else delta
            if self.mask_output:
                m = mask.to(dtype=mu.dtype).unsqueeze(-1)
                mu = mu * m
                log_var = log_var * m
            return mu, log_var

        delta = out
        y = (x + delta) if self.return_reco else delta
        if self.mask_output:
            y = y * mask.to(dtype=y.dtype).unsqueeze(-1)
        return y


# class TokenDenoiserTransformer(nn.Module):
#     def __init__(
#         self,
#         *,
#         input_dim: int = 7,
#         embed_dim: int = 128,
#         num_heads: int = 8,
#         num_layers: int = 4,
#         ff_dim: int = 512,
#         dropout: float = 0.1,
#         # True: return reconstructed target features (x + delta); False: return residual delta.
#         return_reco: bool = True,
#     ):
#         super().__init__()
#         self.input_dim = int(input_dim)
#         self.embed_dim = int(embed_dim)
#         self.return_reco = bool(return_reco)

#         self.input_proj = nn.Sequential(
#             nn.Linear(self.input_dim, self.embed_dim),
#             nn.LayerNorm(self.embed_dim),
#             nn.GELU(),
#             nn.Dropout(float(dropout)),
#         )

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim,
#             nhead=int(num_heads),
#             dim_feedforward=int(ff_dim),
#             dropout=float(dropout),
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,
#         )
#         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

#         self.head = nn.Sequential(
#             nn.LayerNorm(self.embed_dim),
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.GELU(),
#             nn.Dropout(float(dropout)),
#             nn.Linear(self.embed_dim, self.input_dim),
#         )

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#           x: [B,S,D]
#           mask: [B,S] bool, True for valid tokens
#         Returns:
#           y_hat: [B,S,D]
#         """
#         B, S, _ = x.shape
#         x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

#         h = self.input_proj(x.view(B * S, self.input_dim)).view(B, S, self.embed_dim)
#         h = self.encoder(h, src_key_padding_mask=~mask)
#         delta = self.head(h)

#         return (x + delta) if self.return_reco else delta


# -----------------------------
# UNet regression (1D conv over sequence length)
# -----------------------------


class TokenTransformerRegressor(nn.Module):
    """Transformer regressor for token-level unsmearing."""

    def __init__(
        self,
        *,
        input_dim: int = 7,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        return_reco: bool = True,
        predict_logvar: bool = False,
        add_mask_channel: bool = True,
        mask_output: bool = True,
        use_positional_embedding: bool = True,
        max_seq_len: int = 128,
        w_dr: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.return_reco = bool(return_reco)
        self.predict_logvar = bool(predict_logvar)
        self.add_mask_channel = bool(add_mask_channel)
        self.mask_output = bool(mask_output)
        self.use_positional_embedding = bool(use_positional_embedding)
        self.max_seq_len = int(max_seq_len)

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

        out_dim = (2 * self.input_dim) if self.predict_logvar else self.input_dim
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.embed_dim, out_dim),
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
          x: [B,S,D]
          mask: [B,S] bool, True for valid tokens
        Returns:
          y_hat: [B,S,D] or (mu, log_var)
        """
        B, S, _ = x.shape
        if self.pos_embed is not None and S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len={self.max_seq_len}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        inp = x
        if self.add_mask_channel:
            inp = torch.cat([inp, mask.to(dtype=x.dtype).unsqueeze(-1)], dim=-1)

        h = self.input_proj(inp.reshape(B * S, -1)).reshape(B, S, self.embed_dim)
        if self.pos_embed is not None:
            # Use learnable rank embeddings to encode token order explicitly and preserve the pt-sorting prior.
            pos_idx = torch.arange(S, device=x.device)
            h = h + self.pos_embed(pos_idx).unsqueeze(0)
        # Zero out padding positions first to reduce drift on invalid tokens in the residual branch.
        h = h * mask.to(dtype=h.dtype).unsqueeze(-1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        out = self.head(h)

        if self.predict_logvar:
            delta, log_var = torch.split(out, [self.input_dim, self.input_dim], dim=-1)
            mu = (x + delta) if self.return_reco else delta
            if self.mask_output:
                m = mask.to(dtype=mu.dtype).unsqueeze(-1)
                mu = mu * m
                log_var = log_var * m
            return mu, log_var

        delta = out
        y = (x + delta) if self.return_reco else delta
        if self.mask_output:
            y = y * mask.to(dtype=y.dtype).unsqueeze(-1)
        return y


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

        self.unsmear_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
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
            pos_idx = torch.arange(S, device=x.device)
            h = h + self.pos_embed(pos_idx).unsqueeze(0)
        h = h * mask.to(dtype=h.dtype).unsqueeze(-1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return h

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, return_attention: bool = False):
        """Return `(reco, logits)` and optionally the pooling attention weights."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.encode(x, mask)

        delta = self.unsmear_head(h)
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


class _ConvBlock1D(nn.Module):
    """Conv1D block with LayerNorm over channel dim (via transpose)."""

    def __init__(self, cin: int, cout: int, *, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(int(cin), int(cout), kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,L]
        h = self.conv(x)
        h = self.act(h)
        h = self.drop(h)
        # LayerNorm expects [...,C]
        h = h.transpose(1, 2)
        h = self.norm(h)
        h = h.transpose(1, 2)
        return h


class TokenUNet1D(nn.Module):
    """
    1D UNet over token sequence length.

    Inputs:
      x: [B,S,D] (standardized post-smear features)
      mask: [B,S] (True for valid tokens)
    Outputs:
      y_hat: [B,S,D] (prediction; by default returns reconstructed target features)
    """

    def __init__(
        self,
        *,
        input_dim: int = 7,
        base_channels: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
        # True: return reconstructed target features (x + delta); False: return residual delta.
        return_reco: bool = True,
        # True: also predict log_var (heteroscedastic regression). In this case forward returns (mu, log_var).
        predict_logvar: bool = False,
        add_mask_channel: bool = True,
        mask_output: bool = True,
        mask_encoder_features: bool = True,
        w_dr: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.return_reco = bool(return_reco)
        self.predict_logvar = bool(predict_logvar)
        self.add_mask_channel = bool(add_mask_channel)
        self.mask_output = bool(mask_output)
        self.mask_encoder_features = bool(mask_encoder_features)

        cin = self.input_dim + (1 if self.add_mask_channel else 0)
        chs = [int(base_channels) * (2**i) for i in range(int(depth) + 1)]

        self.in_proj = nn.Conv1d(int(cin), int(chs[0]), kernel_size=1)

        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(int(depth)):
            self.down.append(nn.Sequential(_ConvBlock1D(chs[i], chs[i], dropout=float(dropout)),
                                           _ConvBlock1D(chs[i], chs[i], dropout=float(dropout))))
            self.downsample.append(nn.Conv1d(chs[i], chs[i + 1], kernel_size=4, stride=2, padding=1))

        self.mid = nn.Sequential(
            _ConvBlock1D(chs[int(depth)], chs[int(depth)], dropout=float(dropout)),
            _ConvBlock1D(chs[int(depth)], chs[int(depth)], dropout=float(dropout)),
        )

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in reversed(range(int(depth))):
            self.upsample.append(nn.ConvTranspose1d(chs[i + 1], chs[i], kernel_size=4, stride=2, padding=1))
            self.up.append(nn.Sequential(_ConvBlock1D(chs[i] * 2, chs[i], dropout=float(dropout)),
                                         _ConvBlock1D(chs[i], chs[i], dropout=float(dropout))))

        out_dim = (2 * self.input_dim) if self.predict_logvar else self.input_dim
        self.out = nn.Conv1d(chs[0], int(out_dim), kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # [B,D,S]
        xt = x.transpose(1, 2)
        if self.add_mask_channel:
            m = mask.to(dtype=xt.dtype).unsqueeze(1)  # [B,1,S]
            xt = torch.cat([xt, m], dim=1)

        h = self.in_proj(xt)
        if self.mask_encoder_features:
            h = _mask_conv_features(h, mask)
        skips = []
        for blk, ds in zip(self.down, self.downsample):
            h = blk(h)
            if self.mask_encoder_features:
                h = _mask_conv_features(h, mask)
            skips.append(h)
            h = ds(h)
            if self.mask_encoder_features:
                h = _mask_conv_features(h, mask)

        h = self.mid(h)
        if self.mask_encoder_features:
            h = _mask_conv_features(h, mask)

        for us, blk in zip(self.upsample, self.up):
            h = us(h)
            if self.mask_encoder_features:
                h = _mask_conv_features(h, mask)
            # Align sequence length (stride / odd length can cause off-by-one).
            if skips:
                s = skips.pop()
                if h.size(-1) != s.size(-1):
                    diff = s.size(-1) - h.size(-1)
                    if diff > 0:
                        h = nn.functional.pad(h, (0, diff))
                    elif diff < 0:
                        h = h[..., : s.size(-1)]
                h = torch.cat([h, s], dim=1)
            h = blk(h)
            if self.mask_encoder_features:
                h = _mask_conv_features(h, mask)

        out = self.out(h).transpose(1, 2)  # [B,S,?]
        if self.predict_logvar:
            delta, log_var = torch.split(out, [self.input_dim, self.input_dim], dim=-1)
            mu = (x + delta) if self.return_reco else delta
            if self.mask_output:
                m = mask.to(dtype=mu.dtype).unsqueeze(-1)
                mu = mu * m
                log_var = log_var * m
            return mu, log_var

        delta = out
        y = (x + delta) if self.return_reco else delta
        if self.mask_output:
            y = y * mask.to(dtype=y.dtype).unsqueeze(-1)
        return y


# -----------------------------
# Flow Matching (conditional vector field)
# -----------------------------


class _TimeEmbed(nn.Module):
    """Time embedding (for Flow Matching).

    The old implementation using sin(2πt), cos(2πt) makes t=0 and t=1 identical (endpoints are indistinguishable).
    Here we concatenate the raw t (non-periodic) with multi-frequency Fourier features, then apply an MLP.
    """

    def __init__(self, dim: int, *, n_freqs: int = 16, max_freq: float = 1000.0, t_eps: float = 1e-4):
        super().__init__()
        self.dim = int(dim)
        self.t_eps = float(t_eps)
        # Log-space frequencies in [1, max_freq] to avoid endpoint periodic collisions.
        freqs = torch.logspace(0.0, float(torch.log10(torch.tensor(float(max_freq)))), steps=int(n_freqs))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * int(n_freqs), dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0,1]
        tt = t.view(-1, 1).clamp(min=self.t_eps, max=1.0 - self.t_eps)
        ang = tt * self.freqs.view(1, -1)
        inp = torch.cat([tt, torch.sin(ang), torch.cos(ang)], dim=1)
        return self.mlp(inp)  # [B,dim]


class CondFlowMatcher(nn.Module):
    """
    Conditional Flow Matching: learn a vector field v(x_t, t; cond).

    During training we use a simple bridge:
      x_t = (1-t)*x_post + t*x_pre
      v*  = x_pre - x_post

    During inference we start from x_post and integrate to t=1 to obtain the unsmeared output.
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
        time_n_freqs: int = 16,
        time_max_freq: float = 200.0,
        time_t_eps: float = 1e-4,
        per_layer_cond: bool = False,
        w_dr: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.per_layer_cond = bool(per_layer_cond)

        in_dim = self.input_dim * (2 if not self.per_layer_cond else 1)
        # By default keep the old input-level conditioning; with per-layer conditioning enabled, only x_t is encoded at the input.
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.time = _TimeEmbed(
            self.embed_dim,
            n_freqs=int(time_n_freqs),
            max_freq=float(time_max_freq),
            t_eps=float(time_t_eps),
        )

        if self.per_layer_cond:
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=self.embed_dim,
                        nhead=int(num_heads),
                        dim_feedforward=int(ff_dim),
                        dropout=float(dropout),
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.layer_cond_mix = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.embed_dim * 2),
                        nn.Linear(self.embed_dim * 2, self.embed_dim),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.time_layers = nn.ModuleList(
                [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(int(num_layers))]
            )
        else:
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

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.embed_dim, self.input_dim),
        )

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x_t:  [B,S,D]
          cond: [B,S,D] (typically x_post)
          mask: [B,S] bool
          t:    [B] float in [0,1]
        Returns:
          v_hat: [B,S,D]
        """
        B, S, _ = x_t.shape
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
        cond = torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
        if self.per_layer_cond:
            inp = x_t
        else:
            inp = torch.cat([x_t, cond], dim=-1)  # [B,S,2D]

        h = self.in_proj(inp.view(B * S, -1)).view(B, S, self.embed_dim)
        cond_h = self.cond_proj(cond.view(B * S, self.input_dim)).view(B, S, self.embed_dim)
        te = self.time(t).view(B, 1, self.embed_dim)
        if self.per_layer_cond:
            # Closer to DiffLense: each layer concatenates the current hidden state with the condition before linear mixing.
            for layer, cond_mix, time_layer in zip(self.layers, self.layer_cond_mix, self.time_layers):
                h = cond_mix(torch.cat([h, cond_h], dim=-1))
                h = h + time_layer(te)
                h = layer(h, src_key_padding_mask=~mask)
        else:
            h = h + te
            h = self.encoder(h, src_key_padding_mask=~mask)
        v = self.head(h)
        return v


# -----------------------------
# Downstream tagger (copied from tagging/Baseline/model.py)
# -----------------------------


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
    """Downstream tagger (teacher / student / KD student).

    Notes:
    - Input: engineered particle features [B,S,D]
    - mask: [B,S], True means valid token
    - Returns: logits [B,1]; optionally returns pooling attention [B,S] (for attention distillation)
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

        # learnable pooling query + attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(self.embed_dim, num_heads=4, dropout=float(dropout), batch_first=True)
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

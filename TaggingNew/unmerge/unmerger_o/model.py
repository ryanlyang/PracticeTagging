"""
Ordered unmerger (objectness-only version).

目标：
- Shared encoder over HLT token features.
- Heads:
  - parentness: per-token logit (该 token 是否为 merged parent)
  - reconstruction: given a parent token index (teacher forcing in training),
    generate Kmax child features in a fixed ordered layout (按 offline pt 降序对齐).
  - objectness: per-slot logit (该 slot 是否对应真实 child)

说明：
- 这版模型不再预测 k（完全用 objectness 决定生成多少个 child）。
- 训练/数据逻辑在 `tool.py`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UnmergerOutputs:
    parent_logit: torch.Tensor  # [B,S]
    child_feat: Optional[torch.Tensor] = None  # [B,K,child_dim] if parent_idx provided
    obj_logit: Optional[torch.Tensor] = None  # [B,K] if parent_idx provided


class OrderedUnmerger(nn.Module):
    """Transformer encoder + conditional query decoder (ordered child generation) + objectness head."""

    def __init__(
        self,
        *,
        input_dim: int = 4,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers_enc: int = 4,
        num_layers_dec: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
        k_max: int = 8,
        child_dim: int = 4,
        # --- dR-biased attention in encoder (optional) ---
        use_dr_attn: bool = False,
        dr_sigma: float = 1.0,
        dr_gamma_init: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.k_max = int(k_max)
        self.child_dim = int(child_dim)
        self.use_dr_attn = bool(use_dr_attn)
        self.dr_sigma = float(dr_sigma)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        # Encoder: either vanilla TransformerEncoder or dR-biased encoder
        if not self.use_dr_attn:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=int(num_heads),
                dim_feedforward=int(ff_dim),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers_enc))
            self.dr_gamma = None
        else:
            self.dr_gamma = nn.Parameter(torch.tensor(float(dr_gamma_init)))
            self.encoder = _DRBiasedEncoder(
                d_model=int(self.embed_dim),
                nhead=int(num_heads),
                dim_feedforward=int(ff_dim),
                dropout=float(dropout),
                activation="gelu",
                norm_first=True,
                num_layers=int(num_layers_enc),
            )

        # Token-level parentness
        self.parent_head = nn.Linear(self.embed_dim, 1)

        # Reconstruction decoder: fixed K queries conditioned on parent+global embeddings
        self.query_embed = nn.Parameter(torch.randn(self.k_max, self.embed_dim) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(num_layers_dec))
        self.cond_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.child_head = nn.Linear(self.embed_dim, self.child_dim)

        # Slot-level objectness (是否真实 child)
        self.obj_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        parent_idx: Optional[torch.Tensor] = None,
    ) -> UnmergerOutputs:
        """
        Args:
          x: [B,S,D] token features (standardized)
          mask: [B,S] bool, True for valid tokens
          parent_idx: optional [B] long indices into S (teacher forcing / selected parent token)
        Returns:
          UnmergerOutputs; child_feat/obj_logit returned only if parent_idx is provided.
        """
        B, S, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 用于 dR attention 的几何坐标（来自输入特征的 dEta/dPhi）
        coords = None
        if bool(self.use_dr_attn):
            coords = _extract_deta_dphi(x, int(self.input_dim))  # [B,S,2]

        h = x.reshape(B * S, self.input_dim)
        h = self.input_proj(h)
        h = h.reshape(B, S, self.embed_dim)

        if isinstance(self.encoder, _DRBiasedEncoder):
            memory = self.encoder(h, mask, coords=coords, dr_sigma=float(self.dr_sigma), dr_gamma=self.dr_gamma)  # [B,S,E]
        else:
            memory = self.encoder(h, src_key_padding_mask=~mask)  # [B,S,E]
        memory = torch.nan_to_num(memory, nan=0.0, posinf=0.0, neginf=0.0)

        parent_logit = self.parent_head(memory).squeeze(-1)  # [B,S]

        if parent_idx is None:
            return UnmergerOutputs(parent_logit=parent_logit, child_feat=None, obj_logit=None)

        # Conditional decoding for child features + objectness
        bi = torch.arange(B, device=memory.device)
        p = memory[bi, parent_idx]  # [B,E]
        denom = mask.to(memory.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        g = (memory * mask.to(memory.dtype).unsqueeze(-1)).sum(dim=1) / denom  # [B,E]
        cond = self.cond_proj(torch.cat([p, g], dim=-1)).unsqueeze(1)  # [B,1,E]

        q = self.query_embed.unsqueeze(0).expand(B, -1, -1) + cond  # [B,K,E]
        dec = self.decoder(tgt=q, memory=memory, memory_key_padding_mask=~mask)  # [B,K,E]
        child_feat = self.child_head(dec)  # [B,K,child_dim]
        obj_logit = self.obj_head(dec).squeeze(-1)  # [B,K]

        return UnmergerOutputs(parent_logit=parent_logit, child_feat=child_feat, obj_logit=obj_logit)


def _extract_deta_dphi(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    """
    从输入 token features 提取 (dEta, dPhi)。
    约定：
    - 4D: (log_pt, dEta, dPhi, log_E) => (1,2)
    - 7D: (dEta, dPhi, log_pt, log_E, ...) => (0,1)
    注意：这里用的是标准化后的坐标，但只用于 attention bias（相对关系），通常够用。
    """
    D = int(input_dim)
    if D == 4:
        deta = x[..., 1]
        dphi = x[..., 2]
    elif D >= 7:
        deta = x[..., 0]
        dphi = x[..., 1]
    else:
        raise ValueError(f"Unsupported input_dim for dR attention: {D}")
    return torch.stack([deta, dphi], dim=-1)


class _DRBiasedEncoderLayer(nn.Module):
    """Encoder layer with optional dR-biased self-attention."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.norm_first = bool(norm_first)
        self.self_attn = nn.MultiheadAttention(self.d_model, self.nhead, dropout=float(dropout), batch_first=True)
        self.linear1 = nn.Linear(self.d_model, int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), self.d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.act = nn.GELU() if str(activation).lower() == "gelu" else nn.ReLU()

    def _sa(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        y, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.dropout1(y)

    def _ff(self, x: torch.Tensor):
        y = self.linear2(self.dropout(self.act(self.linear1(x))))
        return self.dropout2(y)

    def forward(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        if self.norm_first:
            x = x + self._sa(self.norm1(x), key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            x = x + self._ff(self.norm2(x))
            return x
        x = self.norm1(x + self._sa(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask))
        x = self.norm2(x + self._ff(x))
        return x


class _DRBiasedEncoder(nn.Module):
    """Stack of encoder layers. Adds dR Gaussian bias to attention logits."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        num_layers: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DRBiasedEncoderLayer(
                    d_model=int(d_model),
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    activation=str(activation),
                    norm_first=bool(norm_first),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.nhead = int(nhead)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        coords: Optional[torch.Tensor],
        dr_sigma: float,
        dr_gamma: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # x: [B,S,E], mask: [B,S]
        kp = ~mask  # True means pad
        attn_mask = None
        if coords is not None and dr_gamma is not None:
            # coords: [B,S,2] (dEta,dPhi), compute dR^2
            c = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
            d = c.unsqueeze(2) - c.unsqueeze(1)  # [B,S,S,2]
            dr2 = (d[..., 0] ** 2 + d[..., 1] ** 2).clamp_min(0.0)  # [B,S,S]
            sigma2 = float(dr_sigma) ** 2 + 1e-8
            # Gaussian bias: -gamma * dr^2 / (2*sigma^2)；gamma 可学习
            gamma = dr_gamma.to(dtype=dr2.dtype)
            bias = -(gamma * dr2) / (2.0 * sigma2)  # [B,S,S]
            # Mask out padding tokens (queries or keys)
            bias = bias.masked_fill(kp.unsqueeze(1), float("-inf"))
            bias = bias.masked_fill(kp.unsqueeze(2), float("-inf"))
            # Expand to (B*nhead, S, S) for MultiheadAttention
            attn_mask = bias.repeat_interleave(self.nhead, dim=0)

        h = x
        for layer in self.layers:
            h = layer(h, key_padding_mask=kp, attn_mask=attn_mask)
        return h


def obj_prob_from_logits(obj_logit: torch.Tensor) -> torch.Tensor:
    """把 obj_logit 转成概率。"""
    return torch.sigmoid(obj_logit)


# -----------------------------
# Downstream taggers (teacher/baseline/student/dual)
# -----------------------------


class TokenTagger(nn.Module):
    """Transformer encoder tagger with pooling attention (KD-friendly)."""

    def __init__(
        self,
        *,
        input_dim: int = 4,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        E = int(embed_dim)
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)

        self.input_proj = nn.Linear(self.input_dim, E)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=E,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        # Flowmatching-style pooling attention: learnable query attends to tokens.
        self.pool_query = nn.Parameter(torch.randn(1, 1, E) * 0.02)
        pool_heads = int(min(4, int(num_heads)))
        self.pool_attn = nn.MultiheadAttention(E, pool_heads, dropout=float(dropout), batch_first=True)

        self.head = nn.Sequential(
            nn.LayerNorm(E),
            nn.Linear(E, E),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(E, 1),
        )

    def _encode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # x: [B,S,D], mask: [B,S]
        kp = ~mask
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=kp)  # [B,S,E]
        q = self.pool_query.expand(int(h.shape[0]), -1, -1)  # [B,1,E]
        pooled, attn = self.pool_attn(
            q,
            h,
            h,
            key_padding_mask=kp,
            need_weights=bool(return_attention),
            average_attn_weights=True,
        )
        pooled = pooled.squeeze(1)  # [B,E]
        attn_w = attn.squeeze(1) if return_attention and attn is not None else None  # [B,S]
        return h, pooled, attn_w

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attention: bool = False):
        _, pooled, attn = self._encode(x, mask, return_attention=bool(return_attention))
        logit = self.head(pooled).squeeze(-1)  # [B]
        if return_attention:
            return logit, attn
        return logit


class DualViewTagger(nn.Module):
    """Dual view: encode HLT and UNM, cross-attend pooled(HLT) to token(UNM), then classify."""

    def __init__(
        self,
        *,
        input_dim: int = 4,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        fuse_heads: int = 8,
    ):
        super().__init__()
        self.enc_hlt = TokenTagger(
            input_dim=int(input_dim),
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
        )
        self.enc_unm = TokenTagger(
            input_dim=int(input_dim),
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
        )
        E = int(embed_dim)
        self.cross_attn = nn.MultiheadAttention(E, int(fuse_heads), dropout=float(dropout), batch_first=True)
        self.fuse_head = nn.Sequential(
            nn.LayerNorm(E * 2),
            nn.Linear(E * 2, E),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(E, 1),
        )

    def forward(
        self,
        hlt: torch.Tensor,
        hlt_mask: torch.Tensor,
        unm: torch.Tensor,
        unm_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        _, pool_h, attn_h = self.enc_hlt._encode(hlt, hlt_mask, return_attention=bool(return_attention))
        tok_u, _, _ = self.enc_unm._encode(unm, unm_mask, return_attention=False)
        q = pool_h.unsqueeze(1)  # [B,1,E]
        attn_out, _ = self.cross_attn(q, tok_u, tok_u, key_padding_mask=~unm_mask)
        attn_out = attn_out.squeeze(1)
        fused = torch.cat([pool_h, attn_out], dim=-1)
        logit = self.fuse_head(fused).squeeze(-1)
        if return_attention:
            return logit, attn_h
        return logit


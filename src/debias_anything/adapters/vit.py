# Copyright 2026 Théau d'Audiffret, Mariia Vladimirova, Jean-Yves Franceschi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ViT from the h-space to SigLIP 2, with axial 2D RoPE and AdaLN-Zero(σ)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .hspace_to_siglip import _SinusoidalEmbedding


def _axial_rope_tables(
    grid: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` tables ``(grid², head_dim)`` of an axial 2D RoPE."""
    if head_dim % 4 != 0:
        raise ValueError(
            f"head_dim={head_dim} must be divisible by 4 for an axial 2D RoPE "
            "(two axes × sin/cos pairs)"
        )
    per_axis = head_dim // 2  # channels assigned to each axis
    freqs = 1.0 / (
        10000.0 ** (torch.arange(0, per_axis, 2, device=device).float() / per_axis)
    )  # (per_axis/2,)
    pos = torch.arange(grid, device=device).float()
    angles = pos[:, None] * freqs[None, :]  # (grid, per_axis/2)

    rows = angles[:, None, :].expand(grid, grid, angles.shape[1])
    cols = angles[None, :, :].expand(grid, grid, angles.shape[1])
    ang = torch.cat([rows, cols], dim=-1).reshape(grid * grid, -1)  # (N, head_dim/2)
    ang = ang.repeat_interleave(2, dim=-1)  # (N, head_dim) — one angle per pair
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``(x0, x1, x2, x3, …) -> (-x1, x0, -x3, x2, …)`` — the pairwise RoPE rotation."""
    x = x.unflatten(-1, (-1, 2))
    x0, x1 = x.unbind(-1)
    return torch.stack((-x1, x0), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` (B, heads, N, head_dim) rotated by the ``cos``/``sin`` tables."""
    return x * cos + _rotate_half(x) * sin


class _RoPESelfAttention(nn.Module):
    """Multi-head self-attention with 2D RoPE on Q and K."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:  # x (B, N, C)
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, N, head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _Block(nn.Module):
    """Pre-norm attention and MLP block with AdaLN-Zero(σ) gates."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = _RoPESelfAttention(dim, num_heads, attn_drop=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        # 6 outputs: (scale, shift, gate) × (attention, MLP)
        self.ada = nn.Linear(cond_dim, dim * 6)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1.0 + s1[:, None]) + b1[:, None]
        x = x + g1[:, None] * self.attn(h, cos, sin)
        h = self.norm2(x) * (1.0 + s2[:, None]) + b2[:, None]
        return x + g2[:, None] * self.mlp(h)


class _AttentionPool(nn.Module):
    """Attention pooling with a learned query (the pooling head of SigLIP)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        q = self.query.expand(tokens.shape[0], -1, -1)
        z = self.attn(q, tokens, tokens, need_weights=False)[0].squeeze(1)
        return z + self.mlp(self.norm(z))


class HSpaceViT(nn.Module):
    """h-space ``(B, C, H, W)`` and timestep ``(B,)`` → unit SigLIP embedding."""

    def __init__(
        self,
        h_channels: int = 1280,
        grid: int = 8,
        time_dim: int = 128,
        out_dim: int = 768,
        num_heads: int = 16,
        pool_dim: int = 1024,
        n_blocks: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        # No text conditioning in this architecture: σ goes through AdaLN, and the prompt is
        # already in h (read on a conditioned forward). The attribute is exposed because the
        # guidance reads it to decide whether to pass ``prompt_embeds`` to the forward.
        self.text_dim = None
        dim = min(h_channels, pool_dim)
        if dim % num_heads != 0:
            raise ValueError(
                f"the working width min(h_channels, {pool_dim}) = {dim} must be "
                f"divisible by num_heads={num_heads}"
            )
        self.grid = grid
        self.head_dim = dim // num_heads

        self.reduce = nn.Linear(h_channels, dim) if dim != h_channels else nn.Identity()

        # σ conditioning, shared by all blocks: each derives its own AdaLN from it.
        self.t_emb = nn.Sequential(
            _SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.blocks = nn.ModuleList(
            [
                _Block(dim, num_heads, time_dim, mlp_ratio, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.pool = _AttentionPool(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

        # RoPE tables: functions of (grid, head_dim) only, hence constant. Stored as non-persistent
        # buffers — recomputed at construction, never read from a checkpoint, which leaves
        # ``grid`` free to change.
        cos, sin = _axial_rope_tables(
            grid, self.head_dim, torch.device("cpu"), torch.float32
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Signature aligned with HSpaceToSigLIP: the guidance calls proj(h, t, prompt_embeds).
        # This model has no text conditioning (text_dim=None), so the caller passes None; a
        # tensor would be silently ignored, hence the explicit rejection.
        if prompt_embeds is not None:
            raise ValueError(
                "HSpaceViT has no text conditioning: prompt_embeds would be ignored"
            )
        tokens = self.reduce(h.flatten(2).transpose(1, 2))  # (B, N, dim)
        n = tokens.shape[1]
        if n != self.grid * self.grid:
            raise ValueError(
                f"h has {n} tokens, the model is built for grid={self.grid} "
                f"({self.grid * self.grid} tokens)"
            )
        cos = self.rope_cos.to(tokens.dtype)
        sin = self.rope_sin.to(tokens.dtype)

        cond = self.t_emb(t.reshape(-1)).to(tokens.dtype)
        for blk in self.blocks:
            tokens = blk(tokens, cond, cos, sin)

        z = self.head(self.norm(self.pool(tokens)))
        return F.normalize(z, dim=-1)

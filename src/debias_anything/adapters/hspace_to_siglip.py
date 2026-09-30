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

"""Adapter of the CelebA EDM and P2 models: bottleneck features → SigLIP 2 embedding."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# Maximum width of the pooling: a wider ``h`` is first reduced by a 1×1 projection.
POOL_DIM = 1024


class _SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding of the scalar time condition."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time_dim must be even")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=x.device) * -scale)  # (half,)
        emb = x.reshape(-1, 1) * freqs[None, :]  # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


class _TransformerBlock(nn.Module):
    """Pre-norm self-attention and MLP block over the spatial tokens."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        z = self.norm1(tokens)
        tokens = tokens + self.attn(z, z, z, need_weights=False)[0]
        return tokens + self.mlp(self.norm2(tokens))


class _AttentionPool(nn.Module):
    """Attention pooling with a learned query (the pooling head of SigLIP)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # tokens (B, N, C)
        q = self.query.expand(tokens.shape[0], -1, -1)  # (B, 1, C)
        z = self.attn(q, tokens, tokens, need_weights=False)[0]  # (B, 1, C)
        z = z.squeeze(1)  # (B, C)
        return z + self.mlp(self.norm(z))  # residual MLP, as in SigLIP's MAP


class HSpaceToSigLIP(nn.Module):
    """h-space ``(B, C, H, W)`` and time ``(B,)`` → unit SigLIP embedding."""

    def __init__(
        self,
        h_channels: int = 192,
        num_tokens: int = 64,
        time_dim: int = 128,
        hidden: int = 512,
        out_dim: int = 768,
        num_heads: int = 8,
        pool_dim: int = POOL_DIM,
        n_blocks: int = 0,
        mlp_ratio: float = 4.0,
        text_dim: int | None = None,
    ):
        super().__init__()
        dim = min(
            h_channels, pool_dim
        )  # width at which the whole downstream part operates
        if dim % num_heads != 0:
            raise ValueError(
                f"the pooling width min(h_channels, {pool_dim}) = {dim} must be "
                "divisible by num_heads"
            )
        self.reduce = nn.Linear(h_channels, dim) if dim != h_channels else nn.Identity()
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_tokens, dim) * 0.02
        )  # 0.02 is the init std of SigLIP, so that the norm of the embeddings is ~1 at the start
        self.blocks = nn.ModuleList(
            [_TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(n_blocks)]
        )
        self.pool = _AttentionPool(dim, num_heads)

        self.t_emb = nn.Sequential(
            _SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.text_dim = text_dim
        # Projects the prompt into the FiLM conditioning space, where it is added to the
        # embedding of t. Zero-initialized last layer: at initialization the model is exactly
        # the one without text, and the text conditioning only appears once learned.
        if text_dim is not None:
            self.txt_mlp = nn.Sequential(
                nn.Linear(text_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
            nn.init.zeros_(self.txt_mlp[-1].weight)
            nn.init.zeros_(self.txt_mlp[-1].bias)

        self.film = nn.Linear(time_dim, dim * 2)
        # Zero-init of the FiLM: start as a pure LayerNorm (scale=1, shift=0).
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = h.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
        tokens = self.reduce(tokens)  # 1×1 over the channels: (B, N, C) -> (B, N, dim)
        tokens = tokens + self.pos_emb  # learned spatial localization: dim (B, N, C)
        for blk in self.blocks:  # empty if n_blocks=0: original architecture, identical
            tokens = blk(tokens)
        z = self.pool(tokens)  # (B, C) — weighted, not averaged
        z = self.norm(z)
        cond = self.t_emb(t.reshape(-1))  # (B, time_dim)
        if self.text_dim is not None:
            if prompt_embeds is None:
                raise ValueError(
                    "this adapter was built with text_dim="
                    f"{self.text_dim}: prompt_embeds is required"
                )
            # mean over the prompt tokens: the FiLM takes a single vector, and the structure of
            # the prompt has no place in this path.
            cond = cond + self.txt_mlp(prompt_embeds.mean(1).to(cond.dtype))
        elif prompt_embeds is not None:
            raise ValueError(
                "this adapter was built without text_dim: it would ignore prompt_embeds"
            )
        scale, shift = self.film(cond).chunk(
            2, dim=-1
        )  # chunk splits the two halves into scale and shift
        z = z * (1.0 + scale) + shift
        z = F.silu(self.fc1(z))
        z = self.fc2(z)
        return F.normalize(z, dim=-1)  # unit embedding, ready for cosine


def load_celeba_adapter(path, device: torch.device | str) -> HSpaceToSigLIP:
    """Frozen CelebA adapter, default architecture, from a bare ``state_dict``."""
    adapter = HSpaceToSigLIP().to(device)
    adapter.load_state_dict(torch.load(path, map_location=device))
    return adapter.eval()


P2_ADAPTER_FORMAT = "balancing_act_hspace_siglip_direct_ddim_v1"


def load_p2_adapter(
    path, device: torch.device | str
) -> tuple[HSpaceToSigLIP, dict[str, Any]]:
    """Frozen P2 adapter and its checkpoint metadata (DDIM grid, SigLIP id, t0)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adapter checkpoint not found: {path}")
    checkpoint: dict[str, Any] = torch.load(
        path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != P2_ADAPTER_FORMAT:
        raise ValueError(f"{path} is not a P2 adapter checkpoint")
    arch = checkpoint.get("arch")
    if not isinstance(arch, dict):
        raise ValueError(f"{path} has no adapter architecture metadata")
    if checkpoint.get("time_condition") != "t_over_t0":
        raise ValueError("only the adapter time convention t / t0 is supported")
    adapter = HSpaceToSigLIP(**arch).to(device)
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    adapter.eval().requires_grad_(False)
    return adapter, checkpoint

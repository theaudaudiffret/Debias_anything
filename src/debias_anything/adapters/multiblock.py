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

"""Adapter of SD 1.5: U-Net blocks aggregated as in Readout Guidance, then a ViT."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..hspace.sd15 import BLOCK_SHAPES, BLOCKS_DEFAULT
from .hspace_to_siglip import _SinusoidalEmbedding
from .vit import HSpaceViT

__all__ = ["MultiBlockHSpaceToSigLIP", "load_multiblock_projector"]

# Common width of the projected features, before the mixing. 384 is the ``projection_dim`` of
# every config of Readout Guidance *and* of Diffusion Hyperfeatures; kept as is, for lack of a
# reason to depart from it.
PROJECTION_DIM = 384

# Groups of the GroupNorm of the bottlenecks — 32 in detectron2 and in both papers.
_GN_GROUPS = 32


class _BottleneckBlock(nn.Module):
    """σ-conditioned residual bottleneck 1×1 → 3×3 → 1×1 (Readout Guidance)."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        mid = out_channels // 4  # the paper's rule: projection_dim // 4
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.norm1 = nn.GroupNorm(_GN_GROUPS, mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_GN_GROUPS, mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.norm3 = nn.GroupNorm(_GN_GROUPS, out_channels)
        # The σ conditioning, added by broadcasting after conv1 (the paper's rule).
        self.emb = nn.Linear(cond_dim, mid)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = out + F.relu(self.emb(cond))[:, :, None, None]
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return out + self.shortcut(x)


class _AggregationNetwork(nn.Module):
    """Per-block bottlenecks, pooled to a common grid and mixed by softmax weights."""

    def __init__(
        self,
        blocks: Sequence[str],
        grid: int,
        cond_dim: int,
        projection_dim: int = PROJECTION_DIM,
        block_channels: dict[str, int] | None = None,
    ):
        super().__init__()
        if not blocks:
            raise ValueError("blocks is empty: nothing to aggregate")
        if projection_dim % (4 * _GN_GROUPS) != 0:
            raise ValueError(
                f"projection_dim={projection_dim}: the inner width projection_dim//4 must "
                f"be a multiple of {_GN_GROUPS} (GroupNorm groups)"
            )
        self.blocks = tuple(blocks)
        self.grid = grid
        # Input channels per block. By default those of the encoder + bottleneck readable under
        # truncation; ``block_channels`` allows reading others (the ``up_blocks``, see
        # :mod:`debias_anything.hspace.sd15`) without this module having to know their table.
        chans = block_channels or {b: c for b, (c, _) in BLOCK_SHAPES.items()}
        missing = [b for b in self.blocks if b not in chans]
        if missing:
            raise ValueError(
                f"unknown channels for {missing}: pass them via ``block_channels``"
            )
        # ModuleDict does not accept dotted keys ("down_blocks.0"): we index by position, the
        # pairing being given by the order of ``self.blocks``.
        self.bottlenecks = nn.ModuleList(
            [_BottleneckBlock(chans[b], projection_dim, cond_dim) for b in self.blocks]
        )
        self.mixing_weights = nn.Parameter(torch.ones(len(self.blocks)))

    def forward(
        self, feats: dict[str, torch.Tensor], cond: torch.Tensor
    ) -> torch.Tensor:
        missing = [b for b in self.blocks if b not in feats]
        if missing:
            raise ValueError(
                f"missing features for {missing}: the model is built for "
                f"{list(self.blocks)}. Pass the same ``blocks`` to sd15_hspace_multi."
            )
        w = F.softmax(self.mixing_weights, dim=0)
        out = None
        for i, name in enumerate(self.blocks):
            f = self.bottlenecks[i](feats[name].to(cond.dtype), cond)
            if f.shape[-1] != self.grid:
                # Averaging, not strides: the features of down_blocks.0 are dense, keeping 1 in
                # 16 of them would discard most of the signal.
                f = F.adaptive_avg_pool2d(f, self.grid)
            f = w[i] * f
            out = f if out is None else out + f
        assert out is not None  # ``blocks`` non-empty, guaranteed by the constructor
        return out


class MultiBlockHSpaceToSigLIP(nn.Module):
    """``{block: (B, C, H, W)}`` and timestep ``(B,)`` → unit SigLIP embedding."""

    def __init__(
        self,
        blocks: Sequence[str] = BLOCKS_DEFAULT,
        grid: int = 8,
        projection_dim: int = PROJECTION_DIM,
        time_dim: int = 128,
        out_dim: int = 768,
        num_heads: int = 8,
        pool_dim: int = 384,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        block_channels: dict[str, int] | None = None,
    ):
        super().__init__()
        # Read by the guidance to decide whether to pass ``prompt_embeds``: this model, like
        # HSpaceViT, has no text conditioning (the prompt is already in the features).
        self.text_dim = None
        self.blocks = tuple(blocks)
        self.grid = grid

        self.t_emb = nn.Sequential(
            _SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.aggregate = _AggregationNetwork(
            self.blocks, grid, time_dim, projection_dim, block_channels
        )
        # The trunk sees ``projection_dim`` channels instead of the 1280 of mid_block. ``pool_dim``
        # remains the working width: min(projection_dim, pool_dim), as in HSpaceViT.
        self.trunk = HSpaceViT(
            h_channels=projection_dim,
            grid=grid,
            time_dim=time_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            pool_dim=pool_dim,
            n_blocks=n_blocks,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    @property
    def mixing_weights(self) -> torch.Tensor:
        """Softmax mixing weights of the blocks."""
        return F.softmax(self.aggregate.mixing_weights.detach(), dim=0)

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        t: torch.Tensor,
        prompt_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prompt_embeds is not None:
            raise ValueError(
                "MultiBlockHSpaceToSigLIP has no text conditioning: prompt_embeds "
                "would be ignored (the prompt is already in the features, read on a "
                "conditioned forward)"
            )
        cond = self.t_emb(t.reshape(-1))
        h = self.aggregate(feats, cond)
        return self.trunk(h, t)


# Architecture keys written into the checkpoint — exactly the constructor arguments, so that the
# model can be rebuilt without being given its config again.
_ARCH_KEYS = (
    "blocks",
    "grid",
    "projection_dim",
    "time_dim",
    "out_dim",
    "num_heads",
    "pool_dim",
    "n_blocks",
    "mlp_ratio",
    "dropout",
    # Absent from checkpoints predating non-encoder readings: the loader passes it only if it is
    # present, and its absence restores the default table (BLOCK_SHAPES).
    "block_channels",
)


def load_multiblock_projector(path, device: torch.device | str | None = None):
    """Frozen ``(model, meta)`` from an ``arch="hspace_multiblock"`` checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"{path} is not an adapter checkpoint (dict with 'state_dict')"
        )
    if ckpt.get("arch") != "hspace_multiblock":
        raise ValueError(
            f"{path} has arch={ckpt.get('arch')!r}: it is not a multi-block adapter "
            "(arch='hspace_multiblock')."
        )
    model = MultiBlockHSpaceToSigLIP(**{k: ckpt[k] for k in _ARCH_KEYS if k in ckpt})
    model.load_state_dict(ckpt["state_dict"])
    model.eval().requires_grad_(False)
    if device is not None:
        model.to(device)
    return model, {k: v for k, v in ckpt.items() if k != "state_dict"}

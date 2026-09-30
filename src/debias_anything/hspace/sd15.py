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

"""h-space readers of the SD 1.5 U-Net (input ``x = x₀ + σ·ε``, block features out)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..models.sd15 import SigmaBridge
from .hooks import StopForward, hspace_hook, multi_hspace_hook

# Channels of the bottleneck (``block_out_channels[-1]``) and pixels per h-space cell: 8 (VAE) × 8
# (three downsamplings of the U-Net).
H_MID = 1280
H_DOWNSAMPLE = 64

# Blocks readable before the truncation at ``mid_block``, in forward order.
BLOCK_SHAPES: dict[str, tuple[int, int]] = {
    "down_blocks.0": (320, 16),
    "down_blocks.1": (640, 32),
    "down_blocks.2": (1280, 64),
    "down_blocks.3": (1280, 64),
    "mid_block": (H_MID, H_DOWNSAMPLE),
}
BLOCKS_ALL: tuple[str, ...] = tuple(BLOCK_SHAPES)
# The four distinct scales: ``down_blocks.3`` shares the grid and the width of ``down_blocks.2`` and
# ``mid_block``, from which it is only separated by two ResNets without attention.
BLOCKS_DEFAULT: tuple[str, ...] = (
    "down_blocks.0",
    "down_blocks.1",
    "down_blocks.2",
    "mid_block",
)

# Blocks of the decoder side, and the reading of the released adapter.
BLOCK_SHAPES_DECODER: dict[str, tuple[int, int]] = {
    "mid_block": (H_MID, H_DOWNSAMPLE),
    "up_blocks.0": (1280, 32),
    "up_blocks.1": (1280, 16),
    "up_blocks.2": (640, 8),
    "up_blocks.3": (320, 8),
}
BLOCKS_DECODER_DEFAULT: tuple[str, ...] = tuple(BLOCK_SHAPES_DECODER)

# Execution order of the blocks in the U-Net forward pass: nothing after the last requested block
# needs to run.
FORWARD_ORDER: tuple[str, ...] = (
    *(f"down_blocks.{i}" for i in range(4)),
    "mid_block",
    *(f"up_blocks.{i}" for i in range(4)),
)


def _unet_forward(unet, latents_noisy, sigma, prompt_embeds, bridge) -> None:
    dtype = next(unet.parameters()).dtype
    model_in = bridge.scale_model_input(latents_noisy, sigma).to(dtype)
    t = bridge.t_of_sigma(sigma)
    try:
        # No ``added_cond_kwargs``: the U-Net of SD 1.5 has no ``add_embedding``.
        unet(model_in, t, encoder_hidden_states=prompt_embeds)
    except StopForward:
        pass


def sd15_hspace(
    unet: nn.Module,
    latents_noisy: torch.Tensor,
    sigma: torch.Tensor,
    prompt_embeds: torch.Tensor,
    bridge: SigmaBridge,
) -> torch.Tensor:
    """Output of ``unet.mid_block``, (B, 1280, H/64, W/64)."""
    with hspace_hook(unet, stop=True, module="mid_block") as cache:
        _unet_forward(unet, latents_noisy, sigma, prompt_embeds, bridge)
    return cache["h"]


def sd15_hspace_multi(
    unet: nn.Module,
    latents_noisy: torch.Tensor,
    sigma: torch.Tensor,
    prompt_embeds: torch.Tensor,
    bridge: SigmaBridge,
    blocks: Sequence[str] = BLOCKS_DEFAULT,
) -> dict[str, torch.Tensor]:
    """Encoder and bottleneck features ``{block: (B, C, H, W)}``, forward truncated."""
    unknown = [b for b in blocks if b not in BLOCK_SHAPES]
    if unknown:
        raise ValueError(
            f"unknown blocks {unknown}; readable under truncation: {list(BLOCKS_ALL)}. "
            "The up_blocks need the decoder: see sd15_hspace_decoder."
        )
    with multi_hspace_hook(unet, blocks) as cache:
        _unet_forward(unet, latents_noisy, sigma, prompt_embeds, bridge)
    missing = [b for b in blocks if b not in cache]
    if missing:
        raise RuntimeError(
            f"blocks {missing} not captured: forward truncated before them"
        )
    return {b: cache[b] for b in blocks}


def sd15_hspace_decoder(
    unet: nn.Module,
    latents_noisy: torch.Tensor,
    sigma: torch.Tensor,
    prompt_embeds: torch.Tensor,
    bridge: SigmaBridge,
    blocks: Sequence[str] = BLOCKS_DECODER_DEFAULT,
) -> dict[str, torch.Tensor]:
    """Bottleneck and decoder features ``{block: (B, C, H, W)}`` in one forward pass."""
    shapes = {**BLOCK_SHAPES, **BLOCK_SHAPES_DECODER}
    unknown = [b for b in blocks if b not in shapes]
    if unknown:
        raise ValueError(f"unknown blocks {unknown}; known: {list(shapes)}")
    stop_after = max(blocks, key=FORWARD_ORDER.index)
    with multi_hspace_hook(unet, blocks, stop_after=stop_after) as cache:
        _unet_forward(unet, latents_noisy, sigma, prompt_embeds, bridge)
    missing = [b for b in blocks if b not in cache]
    if missing:
        raise RuntimeError(
            f"blocks {missing} not captured: forward truncated before them"
        )
    return {b: cache[b] for b in blocks}

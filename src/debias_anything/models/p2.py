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

"""P2 U-Net for CelebA-HQ (Choi et al., 2022) on guided-diffusion, and its schedule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .third_party.guided_diffusion import unet as guided_unet

P2_UNET_KWARGS: dict[str, Any] = dict(
    image_size=256,
    in_channels=3,
    model_channels=128,
    out_channels=6,  # learn_sigma=True
    num_res_blocks=1,
    attention_resolutions=(256 // 16,),  # attention_resolutions="16"
    dropout=0.0,
    channel_mult=(1, 1, 2, 2, 4, 4),
    num_classes=None,
    use_checkpoint=False,
    use_fp16=False,
    num_heads=4,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_new_attention_order=False,
)


def load_p2(path: str | Path, device: torch.device) -> nn.Module:
    """Frozen P2 U-Net in eval mode, loaded strictly from ``path``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"P2 checkpoint not found: {path}")
    model = guided_unet.UNetModel(**P2_UNET_KWARGS).to(device)
    state: Any = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


T0 = 999  # last DDPM timestep: 1000 steps


def alpha_bar(t0: int, device: torch.device | str) -> torch.Tensor:
    """ᾱ_t of P2's linear schedule, β ∈ [1e-4, 2e-2] over ``t0 + 1`` steps."""
    betas = torch.linspace(1e-4, 2e-2, t0 + 1, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


def ddim_timesteps(t0: int, n_steps: int, device: torch.device | str) -> torch.Tensor:
    """The ``n_steps`` integer states of the DDIM grid ``linspace(0, t0, n_steps)``."""
    if n_steps < 2:
        raise ValueError("a DDIM grid needs at least 2 steps")
    return torch.linspace(0, t0, n_steps, device=device).long()

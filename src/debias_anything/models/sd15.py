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

"""Stable Diffusion 1.5 and the σ ↔ timestep conversions of its Euler scheduler."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

SD15_BASE = "runwayml/stable-diffusion-v1-5"


def load_pipeline(
    device: torch.device | str,
    pipeline_cls=None,
    scheduler: Literal["euler", "ddim"] = "euler",
):
    """SD 1.5 in fp16, without safety checker, with the Euler or the DDIM scheduler."""
    from diffusers import (
        DDIMScheduler,
        EulerDiscreteScheduler,
        StableDiffusionPipeline,
    )

    pipe = (
        (pipeline_cls or StableDiffusionPipeline)
        .from_pretrained(
            SD15_BASE,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        .to(device)
    )
    if scheduler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    elif scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"unknown scheduler {scheduler!r}: 'euler' or 'ddim'")
    return pipe


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """``np.interp`` in torch (``xp`` increasing, ``x`` clamped to its range)."""
    x = x.clamp(xp[0], xp[-1])
    # index of the first element of xp >= x, hence [lo, hi] brackets x
    hi = torch.searchsorted(xp, x.contiguous()).clamp(1, xp.numel() - 1)
    lo = hi - 1
    w = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] + w * (fp[hi] - fp[lo])


class SigmaBridge(nn.Module):
    """σ ↔ timestep conversions of an Euler scheduler, exact on its sampling path."""

    sigmas: torch.Tensor
    timesteps: torch.Tensor

    def __init__(self, alphas_cumprod: torch.Tensor):
        super().__init__()
        a = (
            alphas_cumprod.double()
        )  # double, for an accurate inverse of the sampling path
        sigmas = (((1.0 - a) / a) ** 0.5).float()  # increasing in t
        self.register_buffer("sigmas", sigmas)
        self.register_buffer(
            "timesteps", torch.arange(sigmas.numel(), dtype=torch.float32)
        )

    @classmethod
    def from_scheduler(cls, scheduler) -> SigmaBridge:
        return cls(torch.as_tensor(scheduler.alphas_cumprod))

    @property
    def sigma_min(self) -> float:
        return float(self.sigmas[0])

    @property
    def sigma_max(self) -> float:
        return float(self.sigmas[-1])

    def t_of_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """σ → continuous timestep, the exact inverse of the sampling path."""
        return _interp(sigma.reshape(-1).float(), self.sigmas, self.timesteps)

    def sigma_of_t(self, t: torch.Tensor) -> torch.Tensor:
        """timestep → σ, as ``set_timesteps`` does."""
        return _interp(t.reshape(-1).float(), self.timesteps, self.sigmas)

    @staticmethod
    def scale_model_input(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """``x / √(σ²+1)``, as ``EulerDiscreteScheduler.scale_model_input``."""
        return x / (sigma.reshape(-1, 1, 1, 1) ** 2 + 1) ** 0.5

    def sample_sigma_log_uniform(
        self, n: int, device: torch.device, sigma_min: float, sigma_max: float
    ) -> torch.Tensor:
        """σ log-uniform on [σ_min, σ_max], shape (n, 1, 1, 1)."""
        lo, hi = math.log(sigma_min), math.log(sigma_max)
        return (torch.rand(n, device=device) * (hi - lo) + lo).exp().view(n, 1, 1, 1)

    def sample_sigma_uniform(
        self, n: int, device: torch.device, sigma_min: float, sigma_max: float
    ) -> torch.Tensor:
        """σ uniform on [σ_min, σ_max], shape (n, 1, 1, 1)."""
        return (
            torch.rand(n, device=device) * (sigma_max - sigma_min) + sigma_min
        ).view(n, 1, 1, 1)

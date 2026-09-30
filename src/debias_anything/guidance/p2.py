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

"""Guidance of P2 on CelebA-HQ, sampled with DDIM (Table 2)."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..hspace.hooks import StopForward, hspace_hook
from ..models import p2 as p2_model
from ..siglip import siglip_image
from .assignment import balanced_assignment, capacities


@contextmanager
def no_activation_checkpointing() -> Generator[None, None, None]:
    """Disable P2's attention checkpointing, which breaks a frozen model's backward."""
    unet = p2_model.guided_unet
    original = unet.checkpoint
    unet.checkpoint = lambda func, inputs, params, flag: func(*inputs)
    try:
        yield
    finally:
        unet.checkpoint = original


def _read_h(p2: nn.Module, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """P2's h-space (``middle_block``), from a truncated forward pass."""
    try:
        with hspace_hook(p2, stop=True, module="middle_block") as cache:
            p2(x_t, timestep)
    except StopForward:
        pass
    return cache["h"]


@dataclass
class P2Guidance:
    """Settings of the guided DDIM loop of a run."""

    adapter: nn.Module
    t0: int  # the adapter is conditioned on t / t0
    guided_timesteps: set[int]
    prototypes: (
        torch.Tensor
    )  # (K, D) centred class prototypes, empty for an unguided run
    proportions: list[float]  # target proportion of each class
    guidance_weight: float = 1.0
    diversity_weight: float = 0.0  # SigLIP minority score; needs ``siglip``
    siglip: tuple[Any, ...] | None = None  # (model, processor, mean, std, size)
    perturb_proj_weight: float = 0.0  # diversity term of the paper
    perturb_timestep: dict[int, int] | None = None  # t -> t_s
    perturb_gen: torch.Generator | None = None  # dedicated generator of ε
    minority_t_min: int = 0
    minority_t_max: int = 1000
    minority_every_n: int = 1

    def gradient(
        self,
        p2: nn.Module,
        x: torch.Tensor,
        timestep_value: int,
        alpha_bar_t: torch.Tensor,
        diversity_weight: float,
        diversity_target: torch.Tensor | None,
        perturb_proj_weight: float,
        x0: torch.Tensor,
        alpha_bar_s: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``∇_y`` of the fairness and diversity objectives, with diagnostics."""
        timestep = torch.full(
            (x.shape[0],), timestep_value, device=x.device, dtype=torch.long
        )
        sqrt_alpha = alpha_bar_t.sqrt()
        do_diversity = diversity_target is not None and diversity_weight != 0.0
        do_perturb_proj = perturb_proj_weight != 0.0
        with torch.enable_grad():
            y_in = (x / sqrt_alpha).detach().requires_grad_(True)
            with no_activation_checkpointing():
                h = _read_h(p2, sqrt_alpha * y_in, timestep)
                z = self.adapter(h.float(), timestep.float() / float(self.t0))
                affinity = z @ self.prototypes.T  # (B, K)
                caps = capacities(self.proportions, affinity.shape[0])
                assignment = balanced_assignment(affinity.detach(), caps)
                text_objective = affinity.gather(1, assignment[:, None]).sum()
                (g_text,) = torch.autograd.grad(
                    text_objective, y_in, retain_graph=do_diversity or do_perturb_proj
                )
                grad_y = self.guidance_weight * g_text
                if do_diversity:
                    diversity = 1.0 - (z * diversity_target).sum(-1)
                    (g_div,) = torch.autograd.grad(diversity.sum(), y_in)
                    grad_y = grad_y + diversity_weight * g_div
                if do_perturb_proj:
                    with torch.no_grad():  # stop-gradient on s = proj(h(y_s); t_s)
                        eps = torch.randn(
                            x0.shape,
                            generator=self.perturb_gen,
                            device=x0.device,
                            dtype=x0.dtype,
                        )
                        # x̂₀ re-noised at the exact σ of t_s
                        sigma_s = ((1.0 - alpha_bar_s) / alpha_bar_s).sqrt()
                        y_s = x0 + sigma_s * eps
                        assert self.perturb_timestep is not None
                        timestep_s = torch.full_like(
                            timestep, self.perturb_timestep[timestep_value]
                        )
                        h_s = _read_h(p2, alpha_bar_s.sqrt() * y_s, timestep_s)
                        s = self.adapter(
                            h_s.float(), timestep_s.float() / float(self.t0)
                        )
                    minority = 1.0 - (z * s).sum(-1)
                    (g_min,) = torch.autograd.grad(minority.sum(), y_in)
                    grad_y = grad_y + perturb_proj_weight * g_min
        return grad_y.detach(), {
            "affinity": affinity.detach(),
            "assigned_counts": torch.bincount(assignment, minlength=affinity.shape[1]),
            "capacities": torch.tensor(caps, device=x.device),
            "grad_y_norm": grad_y.flatten(1).norm(dim=1).mean().detach(),
        }

    def predict_x0(
        self,
        p2: nn.Module,
        x: torch.Tensor,
        timestep_value: int,
        step: int,
        alpha_bar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """P2's x̂₀ plus the guidance correction ``σ_vp²·∇_y``."""
        minority_on = (
            self.minority_t_min <= timestep_value <= self.minority_t_max
            and step % self.minority_every_n == 0
        )
        diversity_weight = self.diversity_weight if minority_on else 0.0
        perturb_proj_weight = self.perturb_proj_weight if minority_on else 0.0
        t_s = (
            self.perturb_timestep[timestep_value]
            if self.perturb_timestep is not None
            else timestep_value
        )
        alpha_bar_t = alpha_bar[timestep_value]
        alpha_bar_s = alpha_bar[t_s]

        timestep = torch.full(
            (x.shape[0],), timestep_value, device=x.device, dtype=torch.long
        )
        with torch.no_grad():
            output = p2(x, timestep)
        epsilon = output[:, :3]
        x0 = (x - (1.0 - alpha_bar_t).sqrt() * epsilon) / alpha_bar_t.sqrt()
        do_diversity = self.siglip is not None and diversity_weight != 0.0
        do_perturb_proj = perturb_proj_weight != 0.0
        if timestep_value not in self.guided_timesteps or (
            self.guidance_weight == 0.0 and not do_diversity and not do_perturb_proj
        ):
            return x0, None

        diversity_target = None
        if do_diversity:
            assert self.siglip is not None
            siglip, _proc, mean, std, size = self.siglip
            with torch.no_grad():
                diversity_target = siglip_image(siglip, x0, mean, std, size)

        grad_y, diagnostics = self.gradient(
            p2,
            x,
            timestep_value,
            alpha_bar_t,
            diversity_weight,
            diversity_target,
            perturb_proj_weight,
            x0,
            alpha_bar_s,
        )
        sigma_vp_sq = (1.0 - alpha_bar_t) / alpha_bar_t
        return x0 + sigma_vp_sq * grad_y, diagnostics


def perturbation_timesteps(
    sequence: list[int],
    adapter_timesteps: set[int],
    alpha_bar: torch.Tensor,
    perturb_sigma: float,
) -> dict[int, int]:
    """``t → t_s``: adapter timestep of σ_vp closest to ``perturb_sigma·σ_vp(t)``."""
    sigma_vp = ((1.0 - alpha_bar) / alpha_bar).sqrt()
    grid = torch.tensor(sorted(adapter_timesteps), device=alpha_bar.device)
    return {
        t: int(grid[(sigma_vp[grid] - perturb_sigma * sigma_vp[t]).abs().argmin()])
        for t in sequence
    }


@torch.no_grad()
def ddim_sample(
    p2: nn.Module,
    guidance: P2Guidance,
    x: torch.Tensor,
    sequence: list[int],
    alpha_bar: torch.Tensor,
    eta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """Guided DDIM from ``x`` over ``sequence``: images in [-1, 1] and diagnostics."""
    diagnostics = None
    previous_sequence = [-1] + sequence[:-1]
    for step, (current, previous) in enumerate(
        zip(reversed(sequence), reversed(previous_sequence))
    ):
        alpha_t = alpha_bar[current]
        x0, diagnostics = guidance.predict_x0(p2, x, current, step, alpha_bar)
        alpha_previous = (
            alpha_bar.new_tensor(1.0) if previous < 0 else alpha_bar[previous]
        )
        # the guided x̂₀ converted back to ε, so that DDIM sees a guided denoiser
        epsilon = (x - alpha_t.sqrt() * x0) / (1.0 - alpha_t).sqrt()
        sigma = eta * torch.sqrt(
            (1.0 - alpha_t / alpha_previous) * (1.0 - alpha_previous) / (1.0 - alpha_t)
        )
        direction_scale = torch.sqrt(
            (1.0 - alpha_previous - sigma.square()).clamp_min(0.0)
        )
        x = alpha_previous.sqrt() * x0 + direction_scale * epsilon
        if sigma.item() != 0.0:
            x = x + sigma * torch.randn_like(x)
    return x, diagnostics

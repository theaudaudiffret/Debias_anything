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

"""Guidance terms of the CelebA 64×64 EDM model, each wrapping the denoiser."""

import torch
import torch.nn.functional as F
from torch import nn

from ..hspace.hooks import StopForward, hspace_hook
from ..models.edm import Denoiser, c_noise
from ..siglip import load_siglip, siglip_image
from .assignment import batched_sign
from .gradients import linf_normalize


class _Window:
    """σ window outside of which a term is not applied."""

    sigma_min: float | None
    sigma_max: float | None

    def _outside_window(self, sigma: torch.Tensor) -> bool:
        s = float(sigma.flatten()[0])  # scalar σ of the current step
        return (self.sigma_min is not None and s < self.sigma_min) or (
            self.sigma_max is not None and s > self.sigma_max
        )


class _HSpaceGuidance(_Window, nn.Module):
    """Base of the terms that read the adapter on the truncated U-Net."""

    def __init__(
        self,
        denoiser: nn.Module,
        unet: nn.Module,
        projector: nn.Module,
        direction: torch.Tensor,  # (D,) unit SigLIP direction e_target − e_source
        guidance_weight: float,
        p: float,  # proportion of the batch assigned to the target
        sigma_min: float | None,
        sigma_max: float | None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.unet = unet
        # uncompiled denoiser, for the truncated forward pass
        self._raw_denoiser = Denoiser(unet)
        self.projector = projector.eval()
        self.register_buffer("direction", F.normalize(direction.detach(), dim=-1))
        self.guidance_weight = guidance_weight
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _read_z(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """``z = proj(h(x); σ)``, from a forward pass truncated at the h-space."""
        try:
            with hspace_hook(self.unet, stop=True) as cache:
                self._raw_denoiser(x, sigma)
        except StopForward:
            pass
        return self.projector(cache["h"], c_noise(sigma).view(-1))

    def _fairness_objective(self, z: torch.Tensor) -> torch.Tensor:
        """``(B,)`` per-sample fairness objective ``signᵢ ⟨zᵢ, Δ⟩``."""
        score = (z * self.direction).sum(-1)  # (B,) target-vs-source affinity
        return batched_sign(score, self.p) * score


class _GatedDiversity:
    """Step gating and dedicated noise generator of the diversity terms."""

    guide_every_n: int
    guidance_weight_minority: float
    noise_seed: int

    def _init_gate(self) -> None:
        assert self.guide_every_n >= 1
        self._step = 0  # calls to forward, counted even outside the σ window
        self._gen: torch.Generator | None = None  # created lazily on the input device

    def _next_step_has_diversity(self) -> bool:
        step, self._step = self._step, self._step + 1
        return self.guidance_weight_minority != 0.0 and step % self.guide_every_n == 0

    def _noise_like(self, x: torch.Tensor) -> torch.Tensor:
        """ε from a dedicated generator, independent of the sampler's."""
        if self._gen is None or self._gen.device != x.device:
            self._gen = torch.Generator(device=x.device).manual_seed(self.noise_seed)
        return torch.randn(x.shape, generator=self._gen, device=x.device, dtype=x.dtype)


class BatchedTextGuidance(_HSpaceGuidance):
    """Fairness term alone (Section 4.2)."""

    def __init__(
        self,
        denoiser: nn.Module,
        unet: nn.Module,
        projector: nn.Module,
        direction: torch.Tensor,
        p: float = 0.5,
        guidance_weight: float = 1.0,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
    ):
        super().__init__(
            denoiser,
            unet,
            projector,
            direction,
            guidance_weight,
            p,
            sigma_min,
            sigma_max,
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            d = self.denoiser(x, sigma)
        if self._outside_window(sigma) or self.guidance_weight == 0.0:
            return d

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            z = self._read_z(x_in, sigma)
            obj = self._fairness_objective(z).sum()
            (grad,) = torch.autograd.grad(obj, x_in)

        sigma_sq = sigma.reshape(-1, 1, 1, 1) ** 2
        return d + self.guidance_weight * sigma_sq * grad


class SGMSTextGuidance(_GatedDiversity, _HSpaceGuidance):
    """Fairness term plus the self-guided minority score of Um & Ye (ECCV 2024)."""

    def __init__(
        self,
        denoiser: nn.Module,
        unet: nn.Module,
        projector: nn.Module,
        direction: torch.Tensor,
        guidance_weight: float = 1.0,
        guidance_weight_minority: float = 1.0,  # w of the paper
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        perturb_sigma: float = 0.5,
        noise_seed: int = 42,
        dist: str = "lpips",  # 'lpips' (the paper) or 'l2'
        p: float = 0.5,
        guide_every_n: int = 1,  # n=5 in the experiments of Um & Ye
        # per-term ℓ∞ normalisation (Sehwag 2022, App. A.4); off by default, since the magnitude
        # of ∇L̃ carries the self-regulation of the minority term
        normalize_grad: bool = False,
    ):
        super().__init__(
            denoiser,
            unet,
            projector,
            direction,
            guidance_weight,
            p,
            sigma_min,
            sigma_max,
        )
        self.guidance_weight_minority = guidance_weight_minority
        self.perturb_sigma = perturb_sigma
        self.noise_seed = noise_seed
        self.dist = dist
        self.guide_every_n = guide_every_n
        self.normalize_grad = normalize_grad
        self._init_gate()
        if dist == "lpips":  # frozen LPIPS VGG, on the device of the adapter
            import lpips

            self.lpips_metric = (
                lpips.LPIPS(net="vgg").to(next(projector.parameters()).device).eval()
            )
            self.lpips_metric.requires_grad_(False)
        elif dist != "l2":
            raise ValueError(f"dist={dist!r}: expected 'lpips' or 'l2'")

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            d = self.denoiser(x, sigma)
        do_minority = self._next_step_has_diversity()
        if self._outside_window(sigma) or (
            self.guidance_weight == 0.0 and not do_minority
        ):
            return d

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            grad = None

            if self.guidance_weight != 0.0:
                z = self._read_z(x_in, sigma)
                (g_text,) = torch.autograd.grad(self._fairness_objective(z).sum(), x_in)
                if self.normalize_grad:
                    g_text = linf_normalize(g_text)
                grad = self.guidance_weight * g_text
                del z, g_text

            if do_minority:
                x0_hat = self._raw_denoiser(
                    x_in, sigma
                )  # full, differentiable denoiser
                eps = self._noise_like(x_in)
                i = int((1 - self.perturb_sigma) * 1000)
                sigma_s = (
                    80 ** (1 / 7) + i / (1000 - 1) * (0.002 ** (1 / 7) - 80 ** (1 / 7))
                ) ** 7
                sigma_s = torch.tensor(sigma_s, device=x_in.device, dtype=x_in.dtype)
                ps = torch.full_like(sigma.reshape(-1, 1, 1, 1), sigma_s)
                with torch.no_grad():  # stop-gradient on x̂₀(x̂_s)
                    x0_hat_s = self._raw_denoiser(x0_hat.detach() + sigma_s * eps, ps)
                del eps
                if self.dist == "lpips":
                    minority_score = self.lpips_metric(x0_hat, x0_hat_s).flatten()
                else:
                    minority_score = (x0_hat - x0_hat_s).flatten(1).norm(dim=1)
                del x0_hat_s, x0_hat
                (g_min,) = torch.autograd.grad(minority_score.sum(), x_in)
                if self.normalize_grad:
                    g_min = linf_normalize(g_min)
                term = self.guidance_weight_minority * g_min
                grad = term if grad is None else grad + term
                del minority_score, g_min

            del x_in

        sigma_sq = sigma.reshape(-1, 1, 1, 1) ** 2
        return d + sigma_sq * grad


class SigLIPMSTextGuidance(_GatedDiversity, _HSpaceGuidance):
    """Fairness term plus the SigLIP minority score (Appendix A.3)."""

    def __init__(
        self,
        denoiser: nn.Module,
        unet: nn.Module,
        projector: nn.Module,
        direction: torch.Tensor,
        siglip_path: str,  # Hugging Face id or local directory of SigLIP
        guidance_weight: float = 1.0,
        guidance_weight_minority: float = 1.0,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        p: float = 0.5,
        guide_every_n: int = 1,
        normalize_grad: bool = False,
    ):
        super().__init__(
            denoiser,
            unet,
            projector,
            direction,
            guidance_weight,
            p,
            sigma_min,
            sigma_max,
        )
        siglip, _, mean, std, size = load_siglip(direction.device, siglip_path)
        self.siglip = siglip
        self.register_buffer("siglip_mean", mean)
        self.register_buffer("siglip_std", std)
        self.siglip_size = size
        self.guidance_weight_minority = guidance_weight_minority
        self.guide_every_n = guide_every_n
        self.normalize_grad = normalize_grad
        self._init_gate()

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            d = self.denoiser(x, sigma)  # x̂₀ = D_θ(x; σ)
        do_minority = self._next_step_has_diversity()
        if self._outside_window(sigma) or (
            self.guidance_weight == 0.0 and not do_minority
        ):
            return d

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            grad = None
            z = self._read_z(x_in, sigma)  # shared by both terms

            if self.guidance_weight != 0.0:
                (g_text,) = torch.autograd.grad(
                    self._fairness_objective(z).sum(), x_in, retain_graph=do_minority
                )
                if self.normalize_grad:
                    g_text = linf_normalize(g_text)
                grad = self.guidance_weight * g_text
                del g_text

            if do_minority:
                with torch.no_grad():  # stop-gradient on s = SigLIP(x̂₀)
                    s = siglip_image(
                        self.siglip,
                        d,
                        self.siglip_mean,
                        self.siglip_std,
                        self.siglip_size,
                    )
                minority_score = 1.0 - (z * s).sum(-1)  # (B,) cosine distance
                (g_min,) = torch.autograd.grad(minority_score.sum(), x_in)
                if self.normalize_grad:
                    g_min = linf_normalize(g_min)
                term = self.guidance_weight_minority * g_min
                grad = term if grad is None else grad + term
                del s, minority_score, g_min

            del x_in

        sigma_sq = sigma.reshape(-1, 1, 1, 1) ** 2
        return d + sigma_sq * grad


class PerturbationProjTextGuidance(_GatedDiversity, _HSpaceGuidance):
    """Fairness term plus the diversity term of the paper (PerturbProj, Eq. 12)."""

    def __init__(
        self,
        denoiser: nn.Module,
        unet: nn.Module,
        projector: nn.Module,
        direction: torch.Tensor,
        guidance_weight: float = 1.0,
        guidance_weight_minority: float = 1.0,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        perturb_sigma: float = 0.5,  # σ_s as a fraction of the σ of the current step
        noise_seed: int = 42,
        p: float = 0.5,
        guide_every_n: int = 1,
        normalize_grad: bool = False,
    ):
        super().__init__(
            denoiser,
            unet,
            projector,
            direction,
            guidance_weight,
            p,
            sigma_min,
            sigma_max,
        )
        self.guidance_weight_minority = guidance_weight_minority
        self.perturb_sigma = perturb_sigma
        self.noise_seed = noise_seed
        self.guide_every_n = guide_every_n
        self.normalize_grad = normalize_grad
        self._init_gate()

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            d = self.denoiser(x, sigma)  # x̂₀ = D_θ(x; σ)
        do_minority = self._next_step_has_diversity()
        if self._outside_window(sigma) or (
            self.guidance_weight == 0.0 and not do_minority
        ):
            return d

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            grad = None
            z = self._read_z(x_in, sigma)  # shared by both terms

            if self.guidance_weight != 0.0:
                (g_text,) = torch.autograd.grad(
                    self._fairness_objective(z).sum(), x_in, retain_graph=do_minority
                )
                if self.normalize_grad:
                    g_text = linf_normalize(g_text)
                grad = self.guidance_weight * g_text
                del g_text

            if do_minority:
                with torch.no_grad():  # stop-gradient on s = proj(h(x̂_s); σ_s)
                    eps = self._noise_like(x_in)
                    ps = self.perturb_sigma * sigma.reshape(
                        -1, 1, 1, 1
                    )  # σ_s, (B,1,1,1)
                    s = self._read_z(d + ps * eps, ps)
                    del eps
                minority_score = 1.0 - (z * s).sum(-1)  # (B,) cosine distance
                (g_min,) = torch.autograd.grad(minority_score.sum(), x_in)
                if self.normalize_grad:
                    g_min = linf_normalize(g_min)
                term = self.guidance_weight_minority * g_min
                grad = term if grad is None else grad + term
                del s, minority_score, g_min

            del x_in

        sigma_sq = sigma.reshape(-1, 1, 1, 1) ** 2
        return d + sigma_sq * grad

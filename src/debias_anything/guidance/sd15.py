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

"""Guidance of Stable Diffusion 1.5 (Table 3)."""

from __future__ import annotations

import inspect

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

from ..adapters.multiblock import load_multiblock_projector
from ..hspace.sd15 import H_DOWNSAMPLE, sd15_hspace_decoder
from ..models.sd15 import SigmaBridge
from ..siglip import encode_text, load_siglip, siglip_image
from .assignment import balanced_assignment
from .assignment import capacities as class_capacities
from .gradients import scaled_grad


class DebiasAnythingSD15(StableDiffusionPipeline):
    """``StableDiffusionPipeline`` with the fairness guidance of Debias Anything."""

    def setup(
        self, siglip_path: str, projector_ckpt: str, source: str, targets: list[str]
    ) -> None:
        """Load SigLIP and the adapter, and embed the source and target sentences."""
        device = self._execution_device
        self.siglip, self.siglip_proc, self.mean, self.std, self.size = load_siglip(
            device, siglip_path
        )
        self.source_embed = self._text_embed(source)
        self.target_embeds = torch.stack(
            [self._text_embed(t) for t in self._checked(targets)]
        )
        self.directions = F.normalize(
            self.target_embeds - self.source_embed, dim=-1
        )  # (K, D)
        # "Stay on the source" class, with the direction opposite to the target: the step
        # decreases ⟨z, Δ₀⟩ instead of increasing it, which reproduces the ``sign = −1`` of the
        # binary case. It exists **only with a single target**, where "the" source is unambiguous;
        # at K ≥ 2, ``−Δₖ`` would move backwards along target *k* and "which one" has no answer,
        # so the whole batch goes to the targets. Not ``None`` therefore means "binary attribute",
        # and alone determines the set of assignable classes of :meth:`__call__`.
        self.source_direction = -self.directions[0] if len(targets) == 1 else None
        # The Δ of the binary case, for the reported scalar score: it is the same object as
        # ``directions[0]`` when there is a single target.
        self.direction = self.directions[0] if len(targets) == 1 else None

        self.projector, meta = load_multiblock_projector(projector_ckpt, device)
        # The three conditions whose mismatch gives a wrong score **without raising an error**:
        # features read under the prompt of the run, conditioning on the continuous timestep, and
        # the same SigLIP space as Δ. The fourth, the list of blocks, is replayed as is.
        if meta.get("h_input") != "cond" or meta.get("time_cond") != "timestep":
            raise ValueError(
                f"unexpected checkpoint convention: {meta.get('h_input')=}, {meta.get('time_cond')=}"
            )
        if int(meta["out_dim"]) != int(self.directions.shape[-1]):
            raise ValueError(
                f"adapter outputs {meta['out_dim']}D but the Δₖ are {self.directions.shape[-1]}D"
            )
        if int(meta["grid"]) != int(meta["resolution"]) // H_DOWNSAMPLE:
            raise ValueError(
                f"grid={meta['grid']} incompatible with {meta['resolution']}² (SD 1.5)"
            )
        if not any(b.startswith("up_blocks") for b in meta["blocks"]):
            raise ValueError(
                f"{projector_ckpt} does not read the decoder: blocks={meta['blocks']}"
            )

        self.blocks = tuple(str(b) for b in meta["blocks"])
        self.bridge = SigmaBridge.from_scheduler(self.scheduler).to(device)
        self.unet.enable_gradient_checkpointing()  # otherwise the 64² up_blocks dominate memory
        # image-by-image decoding: batched decoding does not give exactly the same pixels, hence
        # not exactly the same ``siglip_score``
        self.vae.enable_slicing()

    # ------------------------------------------------------------------ SigLIP

    def _text_embed(self, prompt: str) -> torch.Tensor:
        return encode_text(
            self.siglip, self.siglip_proc, [prompt], self._execution_device
        )[0]

    @staticmethod
    def _checked(targets: list[str]) -> list[str]:
        """``targets``, checked to be non-empty and without duplicates."""
        if not targets:
            raise ValueError("at least one target is required")
        if len(set(targets)) != len(targets):
            raise ValueError(f"duplicate targets: {targets}")
        return targets

    @torch.no_grad()
    def siglip_embed(self, images) -> torch.Tensor:
        """Unit SigLIP embeddings ``(N, D)`` of PIL images."""
        x = torch.stack(
            [torch.from_numpy(np.array(im)).permute(2, 0, 1) for im in images]
        )
        return self._embed_pixels(x.float().to(self._execution_device) / 127.5 - 1.0)

    @torch.no_grad()
    def _embed_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Unit SigLIP embeddings of images ``(B, 3, H, W)`` in [-1, 1]."""
        return siglip_image(self.siglip, x, self.mean, self.std, self.size)

    @torch.no_grad()
    def siglip_score(self, images) -> torch.Tensor:
        """``⟨SigLIP(x), Δ⟩`` of PIL images (single target only)."""
        if self.direction is None:
            raise ValueError(
                "scalar score undefined for K ≥ 2 targets; use siglip_class"
            )
        return self.siglip_embed(images) @ self.direction

    @torch.no_grad()
    def siglip_class(self, images) -> torch.Tensor:
        """Index of the target sentence closest to each image in SigLIP space."""
        return (self.siglip_embed(images) @ self.target_embeds.T).argmax(-1)

    # ------------------------------------------------------------------ guidance

    def _assigned_grad(
        self,
        latents,
        sigma,
        cond_embeds,
        capacities,
        n_per,
        directions,
        div_target=None,
    ):
        """Affinities, per-group assignment and ``∇_x`` of the assigned objective."""
        x = latents.detach().requires_grad_(True)
        b = x.shape[0]
        sigma = sigma.reshape(1)
        feats = sd15_hspace_decoder(
            self.unet, x, sigma.expand(b), cond_embeds, self.bridge, blocks=self.blocks
        )
        z = self.projector(
            {name: f.float() for name, f in feats.items()},
            self.bridge.t_of_sigma(sigma).expand(b),
        )
        affinity = z @ directions.T  # (B, K), ⟨z, Δₖ⟩
        assignment = torch.cat(
            [
                balanced_assignment(affinity[j : j + n_per].detach(), capacities)
                for j in range(0, b, n_per)
            ]
        )
        objective = affinity.gather(1, assignment[:, None]).sum()
        # Decomposition of the increased term, to make evacuation *visible*: since ``Δₖ`` is not
        # centred, ``⟨z, Δₖ⟩`` increases just as well by moving away from the source as by moving
        # towards the target. The two separate projections tell them apart (see the log in
        # ``__call__``). ``proj_target`` is indexed by the clamped assignment: with a single target
        # the last class is the source, which has no ``target_embeds``, and the row read for it is
        # that of the target it moves away from, the right reference to read the retreat.
        assigned_target = assignment.clamp(max=len(self.target_embeds) - 1)
        diagnostics = {
            "proj_source": (z.detach() @ self.source_embed),
            "proj_target": (z.detach() * self.target_embeds[assigned_target]).sum(-1),
        }
        grad = scaled_grad(objective, x, retain_graph=div_target is not None)
        grad_div = None
        if div_target is not None:
            grad_div = scaled_grad((1.0 - (z * div_target).sum(-1)).sum(), x)
        return affinity.detach(), grad, assignment, diagnostics, grad_div

    @torch.no_grad()
    def __call__(  # type: ignore[override]  — deliberately reduced signature
        self,
        prompt: list[str],
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str | None = None,
        num_images_per_prompt: int = 4,
        generator: torch.Generator | None = None,
        weight: float = 0.0,
        weight_minority: float = 0.0,  # siglipms
        proportions: list[float] | None = None,
        sigma_min: float | None = 1.0,
        sigma_max: float | None = 6.0,
        eta: float = 0.0,  # DDIM only (``scheduler=ddim``, requires weight=0); ignored by Euler
        log: list | None = None,
    ):
        device = self._execution_device
        # With **a single target** the attribute is binary: the source is then a fully assignable
        # class (direction ``−Δ₀``), added in *last* position, and ``proportions`` is
        # ``[p, 1 − p]``. Without it there would be a single class, hence a quota ``[1.0]``: not a
        # quota but unconditional guidance towards the target, which is never the desired
        # default. To obtain it anyway: ``[1.0, 0.0]``.
        #
        # With ``K ≥ 2`` targets the source is again only the origin of the ``Δₖ`` and receives
        # nothing: ``−Δₖ`` would move backwards along target *k*, and "which one" has no answer.
        if self.source_direction is not None:
            directions = torch.cat([self.directions, self.source_direction[None]])
        else:
            directions = self.directions
        if proportions is None:
            proportions = [1.0 / len(directions)] * len(directions)
        # The quota is imposed *per prompt group*, so the capacities are computed on ``n_per`` and
        # not on the batch: they must sum to ``n_per``.
        #
        # Only difference from a grouped top-k sign assignment: for a half-integer
        # ``nₖ = pₖ·n_per`` (odd n_per at p = 0.5), the largest remainder of ``capacities`` breaks
        # the tie towards the class declared first (n_per = 3 → [2, 1]), whereas the ``+ 0.5`` of
        # the top-k rounded it towards the target class (k = 2). Both are monotone in n_per; for
        # even ``n_per``, the capacities are identical.
        if len(proportions) != len(directions):
            raise ValueError(
                f"{len(proportions)} proportions for {len(directions)} classes"
            )
        capacities = class_capacities(proportions, num_images_per_prompt)
        # SD 1.5's ``encode_prompt`` requires ``negative_prompt`` to have the same type as
        # ``prompt``: a list here, hence the same string repeated.
        negatives = (
            [negative_prompt] * len(prompt) if negative_prompt is not None else None
        )
        cond, uncond = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negatives,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = self.prepare_latents(
            len(prompt) * num_images_per_prompt,
            self.unet.config.in_channels,
            512,
            512,
            cond.dtype,
            device,
            generator,
            None,
        )
        both = torch.cat([uncond, cond])
        self.scheduler.is_scale_input_called = True  # the σ grid is set, no warning
        # ``eta`` only makes sense for DDIM (ignored by Euler); same detection as DiffLens'
        # ``prepare_extra_step_kwargs`` (``SD_model/pipeline_stable_diffusion.py``).
        accepts_eta = "eta" in inspect.signature(self.scheduler.step).parameters

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            model_in = self.scheduler.scale_model_input(torch.cat([latents] * 2), t)
            eps_u, eps_c = self.unet(
                model_in, t, encoder_hidden_states=both, return_dict=False
            )[0].chunk(2)
            eps = eps_u + guidance_scale * (eps_c - eps_u)

            if weight != 0.0 or weight_minority != 0.0:
                # ``.sigmas`` only exists for the Euler family: read it only if the guidance is
                # actually active, otherwise a DDIM run (``weight=0`` only, see the module
                # docstring) crashes here without ever entering the guidance.
                sigma = self.scheduler.sigmas[i].to(device)
                in_window = (sigma_min is None or float(sigma) >= sigma_min) and (
                    sigma_max is None or float(sigma) <= sigma_max
                )
                if in_window:
                    div_target = None
                    if weight_minority != 0.0:
                        # frozen target of the SigLIP minority score:
                        # SigLIP(decode(x̂₀)), x̂₀ = x − σ·ε_CFG
                        x0 = latents - sigma.to(latents.dtype) * eps
                        div_target = self._embed_pixels(
                            self.vae.decode(
                                x0 / self.vae.config.scaling_factor, return_dict=False
                            )[0]
                        )
                    with torch.enable_grad():
                        # the assignment is only known after scoring the whole group, hence the
                        # order: it is solved in ``_assigned_grad``, before the backward
                        affinity, grad, assignment, diag, grad_div = (
                            self._assigned_grad(
                                latents,
                                sigma,
                                cond,
                                capacities,
                                num_images_per_prompt,
                                directions,
                                div_target,
                            )
                        )
                    # ``grad`` already increases the affinity of the assigned class: the step
                    # subtracts it from ε
                    delta = -(weight * sigma) * grad
                    if grad_div is not None:
                        delta = delta - (weight_minority * sigma) * grad_div
                    if log is not None:
                        # denominator = ε **before** correction
                        log.append(
                            {
                                "sigma": float(sigma),
                                "affinity": affinity.tolist(),
                                "assignment": assignment.tolist(),
                                # read together: ``proj_target`` stalling while
                                # ``proj_source`` collapses = ⟨z, Δ⟩ increased by evacuation
                                "proj_source": diag["proj_source"].tolist(),
                                "proj_target": diag["proj_target"].tolist(),
                                "eps_ratio": float(
                                    delta.abs().amax() / eps.abs().amax()
                                ),
                            }
                        )
                    eps = eps + delta.to(eps.dtype)

            step_kwargs = {"eta": eta} if accepts_eta else {}
            latents = self.scheduler.step(
                eps, t, latents, generator=generator, return_dict=False, **step_kwargs
            )[0]

        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return self.image_processor.postprocess(image.float(), output_type="pil")

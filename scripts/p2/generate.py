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

"""Generate CelebA-HQ images with P2 under the guidance of Debias Anything (Table 2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from PIL import Image

from debias_anything.adapters import load_p2_adapter
from debias_anything.guidance.assignment import capacities
from debias_anything.guidance.p2 import P2Guidance, ddim_sample, perturbation_timesteps
from debias_anything.guidance.text import centred_prototypes
from debias_anything.models.p2 import alpha_bar, ddim_timesteps, load_p2
from debias_anything.paths import CONF_DIR, resolve
from debias_anything.siglip import load_siglip


@dataclass
class Config:
    out: str | None = None
    p2_checkpoint: str = "checkpoints/celebahq_p2.pt"
    projector_checkpoint: str = "checkpoints/celebahq_p2_adapter.pt"
    # binary form: the direction is e_target − e_source
    source_prompt: str | None = None
    target_prompt: str | None = None
    target_proportion: float = 0.5  # share of the target in the binary form
    # K-class form, exclusive of the binary one
    class_prompts: list[str] | None = None
    class_proportions: list[float] | None = None  # null: uniform
    n_images: int = 100
    batch_size: int = 8
    guidance_weight: float = 1.0
    n_ddim_steps: int = 50
    eta: float = 0.0
    seed: int = 42
    # image i starts from a CPU generator seeded with seed + i: the initial noises are then the
    # same across methods and batch sizes
    per_sample_noise_seeds: bool = False
    siglip: str | None = None  # null: the SigLIP id recorded in the adapter checkpoint
    # SigLIP minority score (Appendix A.3): 1 − <proj(h(x)), sg[SigLIP(x̂₀)]>
    diversity: bool = False
    diversity_weight: float = 1.0
    # Diversity term of the paper (Eq. 12): 1 − <proj(h(x_t)), sg[proj(h(x_s))]>, with x_s the
    # estimate x̂₀ re-noised at t_s. Exclusive of ``diversity``.
    perturb_proj: bool = False
    perturb_proj_weight: float = 1.0
    # t_s is the adapter timestep whose σ is closest to perturb_sigma · σ(t)
    perturb_sigma: float = 0.5
    perturb_seed: int = (
        42  # dedicated generator of ε, which does not consume the sampler's
    )
    # window and cadence of the diversity term alone (the σ ∈ [1, 20] of CelebA is t ∈ [258, 768])
    minority_t_min: int = 0
    minority_t_max: int = 1000
    minority_every_n: int = 1  # one DDIM step out of n, counted from t = t0
    guidance_timestep_max: int | None = None  # guide only the states t <= this value
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


ConfigStore.instance().store(name="p2_generate_schema", node=Config)


def validate(cfg: Config) -> None:
    if cfg.out is None:
        raise ValueError("out is required")
    if cfg.diversity and cfg.perturb_proj:
        raise ValueError("diversity and perturb_proj are mutually exclusive")
    if cfg.guidance_timestep_max is not None and cfg.guidance_timestep_max < 0:
        raise ValueError("guidance_timestep_max must be non-negative")
    if cfg.minority_every_n < 1:
        raise ValueError("minority_every_n must be >= 1")
    if cfg.minority_t_min > cfg.minority_t_max:
        raise ValueError("minority_t_min must be <= minority_t_max")
    if not 0.0 <= cfg.target_proportion <= 1.0:
        raise ValueError("target_proportion must be in [0, 1]")
    if cfg.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for proportion guidance")
    if cfg.n_images < 1:
        raise ValueError("n_images must be positive")
    if not 0.0 <= cfg.eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    if cfg.n_images % cfg.batch_size:
        # a shorter last batch would have other capacities nₖ, hence another target composition
        raise ValueError(
            f"n_images {cfg.n_images} must be a multiple of batch_size {cfg.batch_size}"
        )


def resolve_prompts(cfg: Config) -> list[str] | None:
    """The class prompts, whichever form is used; ``None`` for an unguided run."""
    legacy = [p for p in (cfg.source_prompt, cfg.target_prompt) if p is not None]
    if cfg.class_prompts and legacy:
        raise ValueError("class_prompts is exclusive of source_prompt/target_prompt")
    if cfg.class_prompts:
        if len(cfg.class_prompts) < 2:
            raise ValueError("class_prompts needs at least two classes")
        if len(set(cfg.class_prompts)) != len(cfg.class_prompts):
            # two identical prompts give a null prototype, silently normalised to zero
            raise ValueError("duplicate class_prompts")
        return list(cfg.class_prompts)
    if len(legacy) == 1:
        raise ValueError("source_prompt and target_prompt go by pairs")
    if legacy:
        # order imposed: the direction of the binary case is e_target − e_source
        return [cfg.source_prompt, cfg.target_prompt]
    if cfg.guidance_weight != 0.0 or cfg.diversity or cfg.perturb_proj:
        raise ValueError("class prompts are required as soon as the guidance is active")
    return None


def resolve_proportions(cfg: Config, n_classes: int) -> list[float]:
    """The target proportion of each class, summing to 1."""
    if cfg.class_proportions is None:
        if n_classes == 2:
            return [1.0 - cfg.target_proportion, cfg.target_proportion]
        return [1.0 / n_classes] * n_classes
    proportions = [float(value) for value in cfg.class_proportions]
    if len(proportions) != n_classes:
        raise ValueError(f"{len(proportions)} proportions for {n_classes} classes")
    if any(value < 0.0 for value in proportions) or abs(sum(proportions) - 1.0) > 1e-6:
        raise ValueError("class_proportions must be non-negative and sum to 1")
    return proportions


def save_images(x_m11: torch.Tensor, output_dir: Path, start_index: int) -> None:
    images = (
        x_m11.add(1).mul(127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    )
    for offset, image in enumerate(images):
        Image.fromarray(image).save(output_dir / f"{start_index + offset:05d}.png")


@hydra.main(
    version_base="1.3", config_path=str(CONF_DIR / "p2"), config_name="generate"
)
def main(cfg: Config) -> None:
    validate(cfg)
    out = resolve(cfg.out)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    p2 = load_p2(resolve(cfg.p2_checkpoint), device)
    adapter, metadata = load_p2_adapter(resolve(cfg.projector_checkpoint), device)
    siglip_id = cfg.siglip or metadata["siglip"]
    prompts = resolve_prompts(cfg)
    if prompts is None:
        prototypes, proportions = torch.empty(0, device=device), []
    else:
        prototypes = centred_prototypes(prompts, siglip_id, device)
        proportions = resolve_proportions(cfg, len(prompts))
        if prototypes.shape[-1] != metadata["arch"]["out_dim"]:
            raise ValueError(
                "SigLIP text embedding dimension does not match the adapter output"
            )

    t0 = int(metadata["t0"])
    sequence = ddim_timesteps(t0, cfg.n_ddim_steps, device).tolist()
    trained_timesteps = {int(t) for t in metadata["ddim_timesteps"].tolist()}
    untrained_steps = set(sequence) - trained_timesteps
    guided_timesteps = trained_timesteps
    if cfg.guidance_timestep_max is not None:
        guided_timesteps = {
            t for t in trained_timesteps if t <= cfg.guidance_timestep_max
        }
        if not guided_timesteps:
            raise ValueError(
                f"guidance_timestep_max {cfg.guidance_timestep_max} leaves no trained timestep"
            )
    # The adapter knows the 49 guided steps of its grid and omits only t=999, the initial noise:
    # do not guide a sampling schedule unseen during its training.
    if not trained_timesteps.issubset(sequence) or untrained_steps - {t0}:
        raise ValueError(
            "The sampling DDIM grid must contain exactly the adapter's trained "
            "timesteps (and optionally the initial t=t0 state)."
        )
    schedule = alpha_bar(t0, device)
    guidance = P2Guidance(
        adapter=adapter,
        t0=t0,
        guided_timesteps=guided_timesteps,
        prototypes=prototypes,
        proportions=proportions,
        guidance_weight=cfg.guidance_weight,
        diversity_weight=cfg.diversity_weight,
        siglip=load_siglip(device, siglip_id) if cfg.diversity else None,
        perturb_proj_weight=cfg.perturb_proj_weight if cfg.perturb_proj else 0.0,
        perturb_timestep=perturbation_timesteps(
            sequence, trained_timesteps, schedule, cfg.perturb_sigma
        ),
        perturb_gen=(
            torch.Generator(device=device).manual_seed(cfg.perturb_seed)
            if cfg.perturb_proj
            else None
        ),
        minority_t_min=cfg.minority_t_min,
        minority_t_max=cfg.minority_t_max,
        minority_every_n=cfg.minority_every_n,
    )

    out.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    config["class_prompts_resolved"] = prompts
    config["class_proportions_resolved"] = proportions
    if prompts is not None:
        config["capacities_per_batch"] = capacities(proportions, cfg.batch_size)
        # Gram of the prototypes: an off-diagonal entry close to 1 flags two classes SigLIP
        # does not separate, and that no guidance weight will
        config["prototype_gram"] = (prototypes @ prototypes.T).tolist()
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    print(
        f"device={device} | DDIM={len(sequence)} states | guided={len(guided_timesteps)} states"
    )
    if prompts is None:
        print(
            "unguided run (guidance weight 0, no diversity term): no SigLIP prototype"
        )
    else:
        for index, (prompt, proportion) in enumerate(zip(prompts, proportions)):
            print(f"  class {index}: {prompt!r}  target {proportion:.4f}")
        print(f"per-batch capacities: {config['capacities_per_batch']}")

    for generated in range(0, cfg.n_images, cfg.batch_size):
        batch = cfg.batch_size
        if cfg.per_sample_noise_seeds:
            x = torch.stack(
                [
                    torch.randn(
                        (3, 256, 256),
                        generator=torch.Generator(device="cpu").manual_seed(
                            cfg.seed + index
                        ),
                    )
                    for index in range(generated, generated + batch)
                ]
            ).to(device)
        else:
            x = torch.randn(batch, 3, 256, 256, device=device)
        x, diagnostics = ddim_sample(p2, guidance, x, sequence, schedule, cfg.eta)
        save_images(x, out, generated)
        message = f"saved {generated + batch}/{cfg.n_images}"
        if diagnostics is not None:
            message += (
                f" | assigned={diagnostics['assigned_counts'].tolist()}"
                f" | mean|affinity|={diagnostics['affinity'].abs().mean().item():.3g}"
                f" | mean ||grad_y||={diagnostics['grad_y_norm'].item():.3g}"
            )
        print(message)


if __name__ == "__main__":
    main()

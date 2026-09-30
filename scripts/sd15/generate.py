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

"""Generate SD 1.5 images under the guidance of Debias Anything (Table 3)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from debias_anything.guidance.assignment import capacities
from debias_anything.guidance.sd15 import DebiasAnythingSD15
from debias_anything.models.sd15 import load_pipeline
from debias_anything.paths import CONF_DIR, model_path, resolve

# The occupations of DiffLens (SD_generate.py:62).
PROMPTS = [
    "A face of a firefighter",
    "A face of a doctor",
    "A face of a receptionist",
    "A face of a nurse",
]
# Negative prompt of every guided run of Table 3; "" to sample without one.
NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, artifacts, watermark, text, signature, "
    "deformed, disfigured, bad anatomy, extra limbs, cropped, out of frame, cartoon"
)


@dataclass
class Config:
    out: str = "outputs/sd15_guided"
    prompts: list[str] = field(default_factory=lambda: list(PROMPTS))
    n_images: int = 16  # per prompt, in groups of ``group``
    group: int = 4  # the group on which the proportions are imposed
    weight: float = 100.0  # fairness weight
    weight_minority: float = 0.0  # SigLIP minority score (Appendix A.3)
    # one sentence per class; with a single target the source is the second class
    targets: list[str] = field(default_factory=lambda: ["a photo of a woman"])
    source: str = "a photo of a man"  # common origin of the directions Δₖ
    # target proportion of each class, summing to 1 (null: uniform); with a single target,
    # [p, 1 − p] for the target and the source
    target_proportions: list[float] | None = None
    sigma_min: float = 0.0
    sigma_max: float = 6.0
    steps: int = 40
    guidance_scale: float = 7.5
    negative_prompt: str = NEGATIVE_PROMPT
    seed: int = 0
    projector: str = "checkpoints/sd15_adapter.pth"
    siglip: str = "google/siglip2-base-patch16-224"
    # 'ddim' is DiffLens' scheduler (DDIMScheduler.from_config of the default config, without
    # trailing), only accepted at weight=0
    scheduler: str = "euler"
    eta: float = 0.0  # DDIM only (DiffLens' default); ignored by Euler


ConfigStore.instance().store(name="sd15_generate_schema", node=Config)


def class_proportions(cfg: Config) -> list[float]:
    """Target proportions of the assignable classes (with the source when binary)."""
    n_classes = len(cfg.targets) + (1 if len(cfg.targets) == 1 else 0)
    if cfg.target_proportions is None:
        return [1.0 / n_classes] * n_classes
    proportions = [float(v) for v in cfg.target_proportions]
    if len(proportions) != n_classes:
        raise SystemExit(f"{len(proportions)} proportions for {n_classes} classes")
    if any(v < 0.0 for v in proportions) or abs(sum(proportions) - 1.0) > 1e-6:
        raise SystemExit("target_proportions must be non-negative and sum to 1")
    return proportions


@hydra.main(
    version_base="1.3", config_path=str(CONF_DIR / "sd15"), config_name="generate"
)
def main(cfg: Config) -> None:
    if cfg.scheduler == "ddim" and (cfg.weight != 0.0 or cfg.weight_minority != 0.0):
        raise SystemExit(
            f"scheduler=ddim with weight={cfg.weight:g}: the guidance assumes x̂₀ = x − σ·ε "
            "(Euler, variance exploding), which does not hold under DDIM. scheduler=ddim is only "
            "accepted at weight=0."
        )
    targets = list(cfg.targets)
    proportions = class_proportions(cfg)

    pipe = load_pipeline("cuda", DebiasAnythingSD15, scheduler=cfg.scheduler)
    scheduler_name = (
        "DDIM (default config, as DiffLens SD_generate.py)"
        if cfg.scheduler == "ddim"
        else "EulerDiscrete trailing"
    )
    pipe.setup(model_path(cfg.siglip), str(resolve(cfg.projector)), cfg.source, targets)

    out_root = resolve(cfg.out) / "splitted_images"
    report: dict = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "scheduler": scheduler_name,
        "source": cfg.source,
        "targets": targets,
        # with a single target, the last entry is the part of the group that goes to the source
        "proportions": proportions,
        "source_is_a_class": len(targets) == 1,
        "capacities_per_group": capacities(proportions, cfg.group),
        # Gram matrix of the Δₖ: an off-diagonal entry close to 1 flags two directions SigLIP
        # does not separate. The Δₖ share the component −e_source, so this Gram is more
        # correlated than that of centred prototypes; only a value close to 1 disqualifies.
        "direction_gram": (pipe.directions @ pipe.directions.T).tolist(),
    }
    print(
        f"capacities per group of {cfg.group}: {report['capacities_per_group']}",
        flush=True,
    )
    for prompt in cfg.prompts:
        out_dir = out_root / prompt.replace(" ", "-")
        out_dir.mkdir(parents=True, exist_ok=True)
        scores: list[float] = []
        labels: list[int] = []
        ratios: list[float] = []
        for i in range(0, cfg.n_images, cfg.group):
            log: list[dict] = []
            images = pipe(
                [prompt],
                num_inference_steps=cfg.steps,
                guidance_scale=cfg.guidance_scale,
                negative_prompt=cfg.negative_prompt or None,
                num_images_per_prompt=cfg.group,
                generator=torch.Generator(device="cuda").manual_seed(cfg.seed + i),
                weight=cfg.weight,
                weight_minority=cfg.weight_minority,
                proportions=proportions,
                sigma_min=cfg.sigma_min,
                sigma_max=cfg.sigma_max,
                eta=cfg.eta,
                log=log,
            )
            if pipe.direction is not None:
                scores += pipe.siglip_score(images).tolist()
            labels += pipe.siglip_class(images).tolist()
            ratios += [r["eps_ratio"] for r in log]
            for k, image in enumerate(images):
                image.save(out_dir / f"{i + k}.png")
            print(f"[{prompt}] {i + len(images)}/{cfg.n_images}", flush=True)

        report[prompt] = {
            "siglip_score_mean": (sum(scores) / len(scores)) if scores else None,
            # histogram of the closest target: informative at K ≥ 2 only (with a single target it
            # is [1.0] by construction, and siglip_score_mean measures). The *realised* share, not
            # the assigned one.
            "realized_proportions": [
                labels.count(k) / len(labels) for k in range(len(targets))
            ],
            "eps_ratio_mean": (sum(ratios) / len(ratios)) if ratios else 0.0,
            "n_images": len(labels),
        }
        print(prompt, json.dumps(report[prompt]), flush=True)

    (resolve(cfg.out) / "score.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()

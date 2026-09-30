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

"""Build the training cache of the SD 1.5 adapter from SD 1.5's own samples (App. C)."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import hydra
import torch
import yaml
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

from debias_anything.log import get_logger
from debias_anything.models.sd15 import load_pipeline
from debias_anything.paths import CONF_DIR, REPO_ROOT, model_path
from debias_anything.siglip import load_siglip, siglip_image


def load_prompt_bank(path: str | pathlib.Path) -> tuple[list[str], list[str]]:
    """``templates × subjects`` as index-aligned generated and recorded prompts."""
    spec = yaml.safe_load(pathlib.Path(path).read_text())
    gen_bank, rec_bank = [], []
    for t in spec["templates"]:
        for s in spec["subjects"]:
            g, r = (s, s) if isinstance(s, str) else (s["gen"], s["record"])
            gen_bank.append(t.format(subject=g))
            rec_bank.append(t.format(subject=r))
    return gen_bank, rec_bank


logger = get_logger("sd15_dataset", log_file="sd15_dataset.json")


def _save_preview(img: torch.Tensor, paths: list[pathlib.Path]) -> None:
    from PIL import Image

    arr = ((img.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()
    for x, path in zip(arr, paths, strict=True):
        Image.fromarray(x.permute(1, 2, 0).numpy()).save(path, quality=90)


@dataclass
class Config:
    out_dir: str = (
        "data/sd15_adapter_cache"  # must be new: an existing cache is resumed
    )
    prompts: str = "conf/sd15/adapter_prompts.yaml"
    n_images: int = 50000
    resolution: int = 512
    # Fixed negative prompt: it sets the domain of the trajectories the adapter reads, and is
    # recorded in prompts.pt. Changing it invalidates a checkpoint.
    negative_prompt: str = ""
    # Sampling path of the guided runs of scripts/sd15/generate.py: it sets the distribution of
    # the x₀ on which the adapter reads the h-space.
    steps: int = 30
    guidance_scale: float = 7.5
    batch_size: int = 192
    shard_size: int = 1000
    seed: int = 0
    siglip: str = "google/siglip2-base-patch16-224"
    n_preview: int = 200
    # null -> cuda if available
    device: str | None = None


ConfigStore.instance().store(name="sd15_adapter_cache_schema", node=Config)


def build_pipe(device: torch.device | str):
    """SD 1.5 with the Euler ``trailing`` scheduler of the guided runs."""
    pipe = load_pipeline(device, scheduler="euler")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def encode_prompt_bank(
    pipe, bank: list[str], negative_prompt: str, device: torch.device | str
) -> dict:
    """SD 1.5 text embeddings of the prompts and of the negative prompt."""
    prompts = list(bank)
    embeds = []
    for s in range(0, len(prompts), 64):
        pe, _ = pipe.encode_prompt(
            prompt=prompts[s : s + 64],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        embeds.append(pe.cpu())
    neg_pe, _ = pipe.encode_prompt(
        prompt=[negative_prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return {
        "prompts": prompts,
        "embeds": torch.cat(embeds).half(),  # (P, 77, 768)
        "negative_prompt": negative_prompt,
        "neg_embeds": neg_pe.cpu().half(),  # (1, 77, 768)
    }


@torch.no_grad()
def build(cfg: Config) -> None:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = REPO_ROOT / cfg.out_dir
    (out / "preview").mkdir(parents=True, exist_ok=True)

    gen_bank, bank = load_prompt_bank(REPO_ROOT / cfg.prompts)
    # Prompt→image assignment fixed by the seed: a resumed run gives the same dataset.
    g = torch.Generator().manual_seed(cfg.seed)
    prompt_id = torch.randint(0, len(bank), (cfg.n_images,), generator=g).to(
        torch.int16
    )

    steps, gs = cfg.steps, cfg.guidance_scale

    n_shards = -(-cfg.n_images // cfg.shard_size)
    todo = [i for i in range(n_shards) if not (out / f"shard_{i:05d}.pt").exists()]
    if not todo:
        logger.info(f"cache already complete: {n_shards} shards in {out}")
        return
    logger.info(
        f"{len(todo)}/{n_shards} shards to produce "
        f"steps={steps} guidance_scale={gs} res={cfg.resolution} | {len(bank)} prompts",
        extra={"event": "start", "shards_todo": len(todo), "n_shards": n_shards},
    )

    pipe = build_pipe(device)
    if not (out / "prompts.pt").exists():
        enc = encode_prompt_bank(pipe, bank, cfg.negative_prompt, device)
        # The prompt that actually generated each entry (ground-truth label); may differ from
        # ``prompts`` for {gen, record} subjects. Never encoded: only the recorded prompt
        # conditions the adapter.
        enc["gen_prompts"] = gen_bank
        torch.save(enc, out / "prompts.pt")
        n_split = sum(g != r for g, r in zip(gen_bank, bank, strict=True))
        logger.info(
            f"encoded bank: {len(bank)} prompts ({n_split} with gen ≠ record) "
            f"+ 1 negative prompt ({cfg.negative_prompt[:60]}…)"
        )

    siglip, _, mean, std, size = load_siglip(device, model_path(cfg.siglip))

    n_todo_images = sum(
        min((shard + 1) * cfg.shard_size, cfg.n_images) - shard * cfg.shard_size
        for shard in todo
    )
    pbar = tqdm(total=n_todo_images, desc="generation", unit="img")

    for shard in todo:
        lo = shard * cfg.shard_size
        hi = min(lo + cfg.shard_size, cfg.n_images)
        lat_buf, emb_buf = [], []
        for s in range(lo, hi, cfg.batch_size):
            ids = prompt_id[s : min(s + cfg.batch_size, hi)]
            latents = pipe(
                prompt=[gen_bank[int(i)] for i in ids],
                negative_prompt=[cfg.negative_prompt] * len(ids),
                num_inference_steps=steps,
                guidance_scale=gs,
                height=cfg.resolution,
                width=cfg.resolution,
                generator=torch.Generator(device=device).manual_seed(cfg.seed + s),
                output_type="latent",
            ).images  # type: ignore[union-attr]  # (B,4,H/8,W/8), sampling-loop latent, *not* divided by scaling_factor
            assert torch.is_tensor(latents)
            img = pipe.vae.decode(
                latents / pipe.vae.config.scaling_factor, return_dict=False
            )[0]  # (B,3,H,W) in [-1,1]
            lat_buf.append(latents.half().cpu())
            emb_buf.append(siglip_image(siglip, img.float(), mean, std, size).cpu())
            if s < cfg.n_preview:
                _save_preview(
                    img[: cfg.n_preview - s],
                    [
                        out / "preview" / f"{s + k:06d}.jpg"
                        for k in range(min(len(ids), cfg.n_preview - s))
                    ],
                )
            pbar.update(len(ids))
        torch.save(
            {
                "latents": torch.cat(lat_buf),
                "siglip": torch.cat(emb_buf).float(),
                "prompt_id": prompt_id[lo:hi],
            },
            out / f"shard_{shard:05d}.pt",
        )
        logger.info(
            f"shard {shard + 1}/{n_shards} written ({hi - lo} images)",
            extra={"event": "shard", "shard": shard, "n": hi - lo},
        )

    pbar.close()
    logger.info(f"cache complete: {cfg.n_images} images in {out}")


@hydra.main(
    version_base="1.3", config_path=str(CONF_DIR / "sd15"), config_name="adapter_cache"
)
def main(cfg: Config) -> None:
    build(cfg)


if __name__ == "__main__":
    main()

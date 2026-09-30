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

"""Sample the CelebA 64×64 EDM model under a guidance term and compute its metrics."""

import hashlib
import inspect
import json
import time
import typing
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import hydra.utils
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from debias_anything.configs import register_guidance_configs, register_sampler_configs
from debias_anything.data import get_dataloader
from debias_anything.guidance.factory import build_guidance, guidance_tag
from debias_anything.log import get_logger
from debias_anything.metrics.metrics import CELEBA_SACS, Metrics
from debias_anything.models.edm import Denoiser, DiffusionUNet, build_unet_for
from debias_anything.paths import CONF_DIR, resolve

logger = get_logger("sample_and_evaluate", log_file="sample_and_evaluate.jsonl")

register_guidance_configs()
register_sampler_configs()


def _timed(name: str, fn):
    t0 = time.perf_counter()
    result = fn()
    logger.info(f"{name} done in {time.perf_counter() - t0:.2f}s")
    return result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    sampler: Any = MISSING
    guidance: Any = MISSING
    model_path: str = "checkpoints/celeba_edm_64.pth"
    n_batches: int = 50
    batch_size: int = 100
    n_real_train: int = 5_000  # size of the real reference set
    n_steps: int = 100
    dataloader_kind: str = "celeba_balanced"
    seed: int | None = 42
    imsize: int = 64
    channels: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    projector_path: str | None = "checkpoints/celeba_adapter.pth"
    target_prompt: str | None = "a photo of a man"
    source_prompt: str | None = "a photo of a woman"
    siglip_path: str = "google/siglip2-base-patch16-224"
    per_class: bool = True
    monge: bool = True
    # Output directory (metrics JSON, .pt samples), relative to the repository. The file names
    # encode the guidance but neither the seed nor the number of images: one directory per
    # seed or per sample size prevents a run from overwriting another.
    output_dir: str = "result_metrics"
    # On CelebA, fairness is computed for EACH trained SAC (gender,
    # eyeglasses): fields fairness_discrepancy_<name> / class_deviation_table_<name>.


ConfigStore.instance().store(name="celeba_sample_and_evaluate_schema", node=Config)


def _build_sampler(cfg: Config, device: torch.device):
    """The configured sampler, given only the arguments its constructor accepts."""
    target_cls = hydra.utils.get_class(cfg.sampler._target_)
    accepted = inspect.signature(target_cls.__init__).parameters
    candidates = dict(
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        channels=cfg.channels,
        imsize=cfg.imsize,
        device=device,
    )
    kwargs = {k: v for k, v in candidates.items() if k in accepted}
    sampler_cfg = {k: v for k, v in cfg.sampler.items() if k != "name"}
    return hydra.utils.instantiate(sampler_cfg, **kwargs)


def _output_dir(cfg: Config) -> Path:
    out = resolve(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


NAME_MAX = 255  # limit of a path component (ext4) — beyond it: ENAMETOOLONG


def _clamp_filename(filename: str) -> str:
    """``filename``, shortened with a hash if it exceeds ``NAME_MAX`` bytes."""
    if len(filename.encode()) <= NAME_MAX:
        return filename
    stem, _, ext = filename.rpartition(".")
    digest = hashlib.sha1(filename.encode()).hexdigest()[:8]
    keep = NAME_MAX - len(f"_{digest}.{ext}".encode())
    return f"{stem.encode()[:keep].decode(errors='ignore')}_{digest}.{ext}"


def _out_path(cfg: Config, classifier_tag: str, prefix: str) -> str:
    filename = f"{prefix}{classifier_tag}_{cfg.dataloader_kind}_{cfg.sampler.name}.json"
    return str(_output_dir(cfg) / _clamp_filename(filename))


@hydra.main(
    version_base="1.3",
    config_path=str(CONF_DIR / "celeba"),
    config_name="sample_and_evaluate",
)
def main(cfg: Config) -> dict:
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    sampler = _build_sampler(
        cfg, torch.device(cfg.device)
    )  # to check that the arguments are valid before loading the model

    device = torch.device(cfg.device)
    logger.info(
        f"setup device={device} sampler={cfg.sampler.name} n_samples={cfg.n_batches * cfg.batch_size}"
    )

    batch_size = cfg.batch_size
    dataloader = get_dataloader(kind=cfg.dataloader_kind, batch_size=batch_size)
    val_dataloader = get_dataloader(
        kind=cfg.dataloader_kind, batch_size=batch_size, split="val"
    )
    n_batches = cfg.n_batches
    # `targets` (real-set side) follows the reference SAC = the first one declared,
    # which also serves as the partition for the per-class metrics.
    reference_attr = next(attr for *_, attr in CELEBA_SACS.values())
    sac_attr_idx = dataloader.dataset.attr_names.index(reference_attr)  # type: ignore[attr-defined]

    # Columns of the real set used as "proportions_réel" for each SAC: the
    # attributes of the trained SACs.
    sac_attr_names = sorted({attr for *_, attr in CELEBA_SACS.values()})
    sac_attr_indices = {
        attr: dataloader.dataset.attr_names.index(attr)  # type: ignore[attr-defined]
        for attr in sac_attr_names
    }

    x_chunks, t_chunks, total = [], [], 0
    sac_chunks: dict[str, list[torch.Tensor]] = {a: [] for a in sac_attr_indices}
    for batch in dataloader:
        labels = batch[1].to(device)
        for attr, idx in sac_attr_indices.items():
            sac_chunks[attr].append(labels[:, idx])
        labels = labels[:, sac_attr_idx]  # (B, 40) → (B,)
        x_chunks.append(batch[0].to(device))
        t_chunks.append(labels)
        total += batch[0].shape[0]
        if total >= cfg.n_real_train:
            break
    x = torch.cat(x_chunks, dim=0)[: cfg.n_real_train]
    targets = torch.cat(t_chunks, dim=0)[: cfg.n_real_train]
    sac_targets = {
        attr: torch.cat(chunks, dim=0)[: cfg.n_real_train]
        for attr, chunks in sac_chunks.items()
        if chunks
    }
    logger.info(
        f"train real set loaded: {x.shape[0]} samples (target {cfg.n_real_train})"
    )

    # MIND requires as many generated as real samples (assert in monge_inception_distance):
    # check it here, before sampling, rather than after.
    n_generated = cfg.n_batches * cfg.batch_size
    if cfg.monge and n_generated != x.shape[0]:
        raise ValueError(
            f"monge=true requires n_batches*batch_size == size of the real set: "
            f"{cfg.n_batches}*{cfg.batch_size}={n_generated} vs {x.shape[0]} real "
            f"(dataloader_kind={cfg.dataloader_kind}, n_real_train={cfg.n_real_train})"
        )

    x_val_chunks, t_val_chunks = [], []
    for batch in val_dataloader:
        labels_val = batch[1].to(device)[:, sac_attr_idx]
        x_val_chunks.append(batch[0].to(device))
        t_val_chunks.append(labels_val)
    x_val = torch.cat(x_val_chunks, dim=0)
    targets_val = torch.cat(t_val_chunks, dim=0)
    logger.info(f"val set loaded: {x_val.shape[0]} samples")

    channels = x.shape[1]
    image_size = int(x.shape[-1])

    # Rebuild the architecture with the same helper as scripts/celeba/train_generator.py
    # so that the state_dict of the checkpoint matches the topology (otherwise
    # load_state_dict fails with strict=True — typically the CelebA 64×64 case,
    # which uses 4 scales).
    unet = build_unet_for(
        image_size=image_size,
        in_channels=channels,
        out_channels=channels,
    ).to(device)
    unet.load_state_dict(torch.load(resolve(cfg.model_path), map_location=device))
    ema_model = unet
    if device.type == "cuda":
        ema_model = typing.cast(
            DiffusionUNet, torch.compile(unet, mode="reduce-overhead")
        )
        logger.info("ema_model compiled (mode=reduce-overhead)")

    denoiser = Denoiser(ema_model).eval()

    denoiser = build_guidance(cfg, denoiser, unet, device, logger)

    generated_chunks: list[torch.Tensor] = []
    t0 = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for _ in range(n_batches):
            generated_chunks.append(typing.cast(torch.Tensor, sampler.sample(denoiser)))
    generated_samples = torch.cat(generated_chunks, dim=0).float()
    generation_time_s = time.perf_counter() - t0
    logger.info(
        f"generation done in {generation_time_s:.2f}s shape={tuple(generated_samples.shape)}"
    )
    classifier_tag = guidance_tag(cfg)

    metrics = Metrics(
        x,
        generated_samples,
        targets,
        device=device,
        n_batches=cfg.n_batches,
        batch_size=cfg.batch_size,
        images_val=x_val,
        targets_val=targets_val,
        sac_targets=sac_targets,
    )
    num_classes = 2  # the SACs are binary (debias_anything.evaluators.celeba)

    precision, recall, density, coverage = _timed(
        "precision_recall_density_coverage", metrics.precision_recall_density_coverage
    )

    metrics_dict = {
        "generation_time_s": generation_time_s,
        "fid": _timed("fid", metrics.fid),
        "fid_val": _timed("fid_val", metrics.fid_val),
        # "fid_clean": _timed("fid_clean", metrics.fid_clean),
        "sfid": _timed("sfid", metrics.sfid),
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
        "knn_distance": _timed("knn_distance", metrics.knn_distance),
        "vendi": _timed("vendi", metrics.vendi),
        "vendi_real": _timed("vendi_real", metrics.vendi_real),
    }

    # CelebA: one pair of fields per SAC, suffixed with its name.
    for name in metrics.sacs:
        metrics_dict[f"fairness_discrepancy_{name}"] = _timed(
            f"fairness_discrepancy_{name}",
            lambda n=name: metrics.fairness_discrepancy_for(n),
        )
        metrics_dict[f"class_deviation_table_{name}"] = _timed(
            f"class_deviation_table_{name}",
            lambda n=name: metrics.class_deviation_table_for(n).to_dict(
                orient="records"
            ),
        )

    if cfg.monge:
        metrics_dict["monge"] = _timed(
            "monge", lambda: metrics.monge(rng_seed=cfg.seed or 42)
        )

    if cfg.per_class:
        prdc_per_class = _timed(
            "precision_recall_density_coverage_per_class",
            lambda: metrics.precision_recall_density_coverage_per_class(
                num_classes=num_classes
            ),
        )
        metrics_dict.update(
            {
                "fid_per_class": _timed(
                    "fid_per_class",
                    lambda: {
                        str(c): v
                        for c, v in metrics.fid_per_class(
                            num_classes=num_classes
                        ).items()
                    },
                ),
                "fid_val_per_class": _timed(
                    "fid_val_per_class",
                    lambda: {
                        str(c): v
                        for c, v in metrics.fid_val_per_class(
                            num_classes=num_classes
                        ).items()
                    },
                ),
                "precision_per_class": {
                    str(c): v[0] for c, v in prdc_per_class.items()
                },
                "recall_per_class": {str(c): v[1] for c, v in prdc_per_class.items()},
                "density_per_class": {str(c): v[2] for c, v in prdc_per_class.items()},
                "coverage_per_class": {str(c): v[3] for c, v in prdc_per_class.items()},
                "vendi_per_class": _timed(
                    "vendi_per_class",
                    lambda: {
                        str(c): v
                        for c, v in metrics.vendi_per_class(
                            num_classes=num_classes
                        ).items()
                    },
                ),
                "vendi_real_per_class": _timed(
                    "vendi_real_per_class",
                    lambda: {
                        str(c): v
                        for c, v in metrics.vendi_real_per_class(
                            num_classes=num_classes
                        ).items()
                    },
                ),
            }
        )

    prefix = "metrics_"
    out_path = _out_path(cfg, classifier_tag, prefix)

    if cfg.per_class:
        samples_prefix = "samples_"
        samples_path = _out_path(cfg, classifier_tag, samples_prefix).replace(
            ".json", ".pt"
        )
        torch.save(generated_samples, samples_path)
    else:
        torch.save(
            generated_samples,
            str(
                _output_dir(cfg)
                / _clamp_filename(
                    f"generated_samples_{classifier_tag}_{cfg.dataloader_kind}_{cfg.sampler.name}.pt"
                )
            ),
        )

    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    logger.info(f"results saved to {out_path}")

    return metrics_dict


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to silence the warnings of the FID library
    main()

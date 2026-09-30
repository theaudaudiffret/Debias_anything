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

"""Build an EDM guidance term from its Hydra config."""

import inspect
from typing import Any

import hydra.utils
from omegaconf import OmegaConf

from ..adapters.hspace_to_siglip import load_celeba_adapter
from ..paths import resolve
from .text import text_direction


def build_guidance(cfg: Any, denoiser, unet, device, logger=None):
    """``denoiser`` wrapped by the configured term, unchanged for ``guidance=none``."""

    def log(msg):
        if logger is not None:
            logger.info(msg)

    target_path = cfg.guidance._target_
    if target_path is None:
        log("no guidance")
        return denoiser

    target_cls = hydra.utils.get_class(target_path)
    accepted = inspect.signature(target_cls.__init__).parameters

    kwargs = {}
    if "denoiser" in accepted:
        kwargs["denoiser"] = denoiser
    if "unet" in accepted:
        kwargs["unet"] = unet
    if "device" in accepted:
        kwargs["device"] = device
    if "siglip_path" in accepted:
        kwargs["siglip_path"] = cfg.siglip_path
    if "projector" in accepted:
        if cfg.projector_path is None:
            raise ValueError(f"guidance {target_path} needs projector_path")
        kwargs["projector"] = load_celeba_adapter(resolve(cfg.projector_path), device)
    if "direction" in accepted:
        if cfg.target_prompt is None:
            raise ValueError(f"guidance {target_path} needs target_prompt")
        kwargs["direction"] = text_direction(
            cfg.target_prompt, cfg.source_prompt, cfg.siglip_path, device
        )

    gd = hydra.utils.instantiate(cfg.guidance, **kwargs)
    log(f"{target_path} guidance ready")
    return gd.to(device)


def guidance_tag(cfg: Any) -> str:
    """Tag of the output file names: term, target sentence and every config field."""
    target_path = cfg.guidance._target_
    if target_path is None:
        return "no_guidance"

    name = target_path.rsplit(".", 1)[-1]
    slug = (cfg.target_prompt or "text").replace(" ", "_")
    params = OmegaConf.to_container(cfg.guidance, resolve=True)
    assert isinstance(params, dict)
    params.pop("_target_", None)
    param_str = "_".join(f"{k}{v}" for k, v in params.items())
    return f"{name}_{slug}_{param_str}"

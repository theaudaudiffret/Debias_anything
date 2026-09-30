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

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class NoGuidanceConfig:
    _target_: str | None


@dataclass
class BatchedTextGuidanceConfig:
    _target_: str
    guidance_weight: float
    sigma_min: float | None
    sigma_max: float | None
    p: float


@dataclass
class SGMSTextGuidanceConfig:
    _target_: str
    guidance_weight: float
    sigma_min: float | None
    sigma_max: float | None
    p: float
    guidance_weight_minority: float
    perturb_sigma: float
    noise_seed: int
    dist: str
    guide_every_n: int
    normalize_grad: bool


@dataclass
class SigLIPMSTextGuidanceConfig:
    _target_: str
    guidance_weight: float
    sigma_min: float | None
    sigma_max: float | None
    p: float
    guidance_weight_minority: float
    guide_every_n: int
    normalize_grad: bool


@dataclass
class PerturbationProjTextGuidanceConfig:
    _target_: str
    guidance_weight: float
    sigma_min: float | None
    sigma_max: float | None
    p: float
    guidance_weight_minority: float
    perturb_sigma: float
    noise_seed: int
    guide_every_n: int
    normalize_grad: bool


def register_guidance_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="guidance", name="base_none", node=NoGuidanceConfig)
    cs.store(group="guidance", name="base_batched_text", node=BatchedTextGuidanceConfig)
    cs.store(group="guidance", name="base_sgms", node=SGMSTextGuidanceConfig)
    cs.store(group="guidance", name="base_siglipms", node=SigLIPMSTextGuidanceConfig)
    cs.store(
        group="guidance",
        name="base_perturbationproj",
        node=PerturbationProjTextGuidanceConfig,
    )

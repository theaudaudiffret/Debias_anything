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
class DeterministicSamplerConfig:
    _target_: str
    name: str


@dataclass
class StochasticSamplerConfig:
    _target_: str
    name: str


def register_sampler_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="sampler", name="base_deterministic", node=DeterministicSamplerConfig
    )
    cs.store(group="sampler", name="base_stochastic", node=StochasticSamplerConfig)

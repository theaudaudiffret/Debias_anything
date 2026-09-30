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
from enum import StrEnum


@dataclass
class MLflowConfig:
    experiment_name: str = "Default"
    run_name: str | None = None


class DataloaderKind(StrEnum):
    celeba = "celeba"
    celeba_balanced = "celeba_balanced"
    celeba_balanced_eyeglasses = "celeba_balanced_eyeglasses"
    celeba_hq = "celeba_hq"
    minority_celeba = "minority_celeba"

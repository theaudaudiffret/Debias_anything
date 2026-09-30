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

"""MLflow helpers of the training scripts (database ``mlflow.db``)."""

from pathlib import Path
from typing import Any, cast

import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from .paths import MLFLOW_TRACKING_URI


def flatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for sub_key, sub_value in flatten_dict(value, sep).items():
                flat[f"{key}{sep}{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def log_cfg_to_mlflow(cfg: Any) -> None:
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    mlflow.log_params(flatten_dict(cast("dict[str, Any]", container)))
    hydra_dir = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
    if hydra_dir.exists():
        mlflow.log_artifacts(str(hydra_dir), artifact_path=".hydra")


def set_experiment(name: str) -> None:
    """Use the repository database ``mlflow.db`` and select experiment ``name``."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(name)

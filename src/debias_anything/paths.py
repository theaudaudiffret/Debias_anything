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

"""Repository locations; relative config paths are relative to ``REPO_ROOT``."""

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"
DATA_ROOT = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
LOG_DIR = REPO_ROOT / "logs"
MLFLOW_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"


def resolve(path: str | Path) -> Path:
    """``path`` if absolute, else ``REPO_ROOT / path``."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def portable(path: str | Path) -> str:
    """``path`` relative to ``REPO_ROOT`` if inside it, else with ``~`` for the home."""
    for candidate in (Path(os.path.abspath(path)), Path(path).resolve()):
        if candidate.is_relative_to(REPO_ROOT):
            return str(candidate.relative_to(REPO_ROOT))
    path = Path(os.path.abspath(path))
    if path.is_relative_to(Path.home()):
        return f"~/{path.relative_to(Path.home())}"
    return str(path)


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard link ``source`` at ``destination``, or copy it across file systems."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def model_path(name: str) -> str:
    """Local copy of a Hugging Face model under ``REPO_ROOT``, else its hub id."""
    local = REPO_ROOT / name
    return str(local) if local.exists() else name

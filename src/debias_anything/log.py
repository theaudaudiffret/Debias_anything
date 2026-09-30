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

"""Loggers writing to stdout and optionally to ``logs/<file>`` as JSON lines."""

import json
import logging
import os
from datetime import UTC, datetime

from .paths import LOG_DIR

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_CONSOLE_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_initialized: set[str] = set()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__ and not k.startswith("_")
        }
        if extra:
            payload["extra"] = extra
        return json.dumps(payload, default=str)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Logger to stdout, and to ``logs/<log_file>`` (JSON lines) if given."""
    logger = logging.getLogger(name)

    if name in _initialized:
        return logger

    logger.setLevel(_LOG_LEVEL)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(_CONSOLE_FORMATTER)
    logger.addHandler(console)

    if log_file is not None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_DIR / log_file)
        fh.setFormatter(_JsonFormatter())
        logger.addHandler(fh)

    _initialized.add(name)
    return logger

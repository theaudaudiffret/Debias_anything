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

"""Forward hooks reading the h-space of a U-Net, optionally stopping the forward."""

from collections.abc import Generator, Sequence
from contextlib import ExitStack, contextmanager
from typing import cast

import torch
from torch import nn


class StopForward(Exception):
    """Raised by a hook to stop the forward pass."""


def get_module(model: nn.Module, name: str) -> nn.Module:
    """``model.down_blocks[2]`` from the dotted name ``"down_blocks.2"``."""
    obj: object = model
    for part in name.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)  # type: ignore[index]
    return cast(nn.Module, obj)


@contextmanager
def hspace_hook(
    unet: nn.Module, stop: bool = False, module: str = "mid_block2"
) -> Generator[dict, None, None]:
    """Expose the output of ``unet.<module>`` as ``cache["h"]``; ``stop`` truncates."""
    cache: dict = {}

    def _hook(_module, _inputs, output):
        cache["h"] = output
        if stop:
            raise StopForward

    handle = get_module(unet, module).register_forward_hook(_hook)
    try:
        yield cache
    finally:
        handle.remove()


@contextmanager
def multi_hspace_hook(
    unet: nn.Module, blocks: Sequence[str], stop_after: str = "mid_block"
) -> Generator[dict[str, torch.Tensor], None, None]:
    """``cache[name]`` = output of each of ``blocks``; stops after ``stop_after``."""
    cache: dict[str, torch.Tensor] = {}
    wanted = set(blocks)

    def _make(name: str):
        def _hook(_module, _inputs, output):
            if name in wanted:
                cache[name] = output[0] if isinstance(output, tuple) else output
            if name == stop_after:
                raise StopForward

        return _hook

    with ExitStack() as stack:
        for name in dict.fromkeys([*blocks, stop_after]):
            handle = get_module(unet, name).register_forward_hook(_make(name))
            stack.callback(handle.remove)
        yield cache

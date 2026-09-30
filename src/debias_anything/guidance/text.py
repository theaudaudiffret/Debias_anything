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

"""The attribute directions, from sentences embedded by SigLIP 2."""

import torch
import torch.nn.functional as F

from ..siglip import encode_text, load_siglip


@torch.no_grad()
def text_direction(
    target: str, source: str | None, siglip_path: str, device: torch.device
) -> torch.Tensor:
    """Unit direction ``normalize(e_target − e_source)``, or ``e_target`` alone."""
    model, proc, *_ = load_siglip(device, siglip_path)
    e_t = encode_text(model, proc, [target], device)[0]
    if source is None:
        return e_t
    return F.normalize(e_t - encode_text(model, proc, [source], device)[0], dim=-1)


@torch.no_grad()
def centred_prototypes(
    prompts: list[str], siglip_path: str, device: torch.device
) -> torch.Tensor:
    """``(K, D)`` centred unit prototypes ``normalize(eₖ − mean_l e_l)``."""
    model, proc, *_ = load_siglip(device, siglip_path)
    embeddings = encode_text(model, proc, prompts, device)
    return F.normalize(embeddings - embeddings.mean(0, keepdim=True), dim=-1)

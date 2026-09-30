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

"""SigLIP 2: the embedding space of the adapters and of the attribute sentences."""

from typing import Literal

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

SIGLIP = "google/siglip2-base-patch16-224"


def _pool(out):
    """Pooled features, whatever the output type of the transformers version."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def load_siglip(device: torch.device | str, path: str = SIGLIP):
    """Frozen SigLIP 2, its processor and ``(mean, std, size)`` of its preprocessing."""
    model = AutoModel.from_pretrained(path, dtype=torch.float32).to(device).eval()
    model.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(path)
    ip = proc.image_processor
    mean = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
    return model, proc, mean, std, ip.size["height"]


def siglip_image(
    siglip,
    img_m11: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    size: int,
    resize: Literal["bilinear", "bicubic"] = "bilinear",
    clamp: bool = True,
) -> torch.Tensor:
    """Unit SigLIP embeddings of [-1, 1] images (bicubic, no clamp for CelebA/P2)."""
    x = img_m11 * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    if clamp:
        x = x.clamp(0, 1)
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    x = F.interpolate(x, size=size, mode=resize, align_corners=False, antialias=True)
    x = (x - mean) / std
    return F.normalize(_pool(siglip.get_image_features(pixel_values=x)), dim=-1)


def encode_text(
    siglip,
    proc,
    prompts: list[str],
    device: torch.device | str,
) -> torch.Tensor:
    """Unit SigLIP embeddings ``(P, D)`` of ``prompts``, padded to 64 tokens."""
    inp = proc(
        text=list(prompts),
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    return F.normalize(_pool(siglip.get_text_features(**inp)).float(), dim=-1)

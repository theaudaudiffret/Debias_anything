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

"""Adapters from the h-space of a frozen denoiser to SigLIP 2 (Section 4.1)."""

from .hspace_to_siglip import HSpaceToSigLIP, load_celeba_adapter, load_p2_adapter
from .losses import cosine_loss
from .multiblock import MultiBlockHSpaceToSigLIP, load_multiblock_projector
from .vit import HSpaceViT

__all__ = [
    "HSpaceToSigLIP",
    "HSpaceViT",
    "MultiBlockHSpaceToSigLIP",
    "cosine_loss",
    "load_celeba_adapter",
    "load_multiblock_projector",
    "load_p2_adapter",
]

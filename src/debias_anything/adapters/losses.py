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

"""Training loss of the adapters (Eq. 4)."""

import torch


def cosine_loss(
    z: torch.Tensor, tgt: torch.Tensor, src_idx: torch.Tensor
) -> tuple[torch.Tensor, float, float]:
    """``1 − mean cos(z, tgt[src_idx])``, and the mean positive and negative cosines."""
    sim = z @ tgt.t()  # (A, B) cosines
    pos_sim = sim[torch.arange(z.shape[0], device=z.device), src_idx]  # (A,)
    loss = 1.0 - pos_sim.mean()
    with torch.no_grad():
        pos_mask = (
            src_idx[:, None] == torch.arange(tgt.shape[0], device=z.device)[None, :]
        )
    return loss, pos_sim.mean().item(), sim[~pos_mask].mean().item()

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

"""Batch assignment of the fairness term under class quotas (Section 4.2, Eq. 9)."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def batched_sign(score: torch.Tensor, p: float) -> torch.Tensor:
    """+1 for the ``round(p·B)`` largest scores, −1 for the others."""
    B = score.shape[0]
    k = int(round(p * B))
    sign = torch.full((B,), -1.0, device=score.device)
    if k > 0:
        sign[torch.topk(score.detach(), k).indices] = 1.0
    return sign


def capacities(proportions: list[float], B: int) -> list[int]:
    """Integer quotas summing to ``B``, closest to ``pₖ·B`` (largest remainder)."""
    if not proportions:
        raise ValueError("empty proportions")
    exact = [p * B for p in proportions]
    caps = [int(np.floor(v)) for v in exact]
    for index in np.argsort([-(v - np.floor(v)) for v in exact])[: B - sum(caps)]:
        caps[int(index)] += 1
    return caps


def balanced_assignment(affinity: torch.Tensor, capacities: list[int]) -> torch.Tensor:
    """Assignment maximising ``Σᵢ affinity[i, π(i)]`` under the quotas."""
    B, K = affinity.shape
    if len(capacities) != K:
        raise ValueError(f"{len(capacities)} capacities for {K} classes")
    if sum(capacities) != B:
        raise ValueError(f"capacities summing to {sum(capacities)} for a batch of {B}")
    columns = np.repeat(np.arange(K), capacities)  # column j -> class columns[j]
    rows, cols = linear_sum_assignment(
        -affinity.detach().float().cpu().numpy()[:, columns]
    )
    assignment = torch.empty(B, dtype=torch.long, device=affinity.device)
    assignment[torch.as_tensor(rows, device=affinity.device)] = torch.as_tensor(
        columns[cols], device=affinity.device
    )
    return assignment

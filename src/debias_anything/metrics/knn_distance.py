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

"""AvgkNN distance of generated to real features (Sehwag et al., CVPR 2022)."""

from __future__ import annotations

import numpy as np
import torch


def compute_knn_distance(
    feats_real: np.ndarray,
    feats_gen: np.ndarray,
    k: int = 5,
    device: torch.device | str = "cpu",
    chunk_size: int = 1024,
) -> float:
    """Mean L2 distance of each generated sample to its ``k`` nearest real ones."""
    device = torch.device(device)
    r = torch.from_numpy(feats_real).to(device)
    g = torch.from_numpy(feats_gen).to(device)
    M = g.shape[0]
    avg_knn = torch.empty(M, device=device)
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        dists = torch.cdist(g[start:end], r)
        topk = torch.topk(dists, k, dim=1, largest=False).values
        avg_knn[start:end] = topk.mean(dim=1)
    return float(avg_knn.mean().cpu())

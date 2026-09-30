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

"""Precision, recall, density and coverage, computed in chunks."""

from __future__ import annotations

import numpy as np
import torch


def _knn_radii(
    features: np.ndarray,
    k: int,
    device: torch.device,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Distance of each point to its k-th nearest neighbour (itself excluded)."""
    f = torch.from_numpy(features).to(device)  # (N, D)
    N = f.shape[0]
    radii = torch.empty(N, device=device)  # (N,)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = f[start:end]  # (B, D)
        dists = torch.cdist(chunk, f)  # (B, N)
        local_idx = torch.arange(end - start, device=device)  # (B,)
        dists[local_idx, start + local_idx] = float("inf")  # mask self-distance
        radii[start:end] = torch.topk(dists, k, dim=1, largest=False).values[
            :, -1
        ]  # (B,)
    return radii.cpu().numpy()  # (N,)


def _count_hits(
    samples: np.ndarray,
    manifold: np.ndarray,
    radii: np.ndarray,
    device: torch.device,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Number of balls ``B(m_j, r_j)`` containing each sample."""
    M = samples.shape[0]
    s = torch.from_numpy(samples).to(device)  # (M, D)
    m = torch.from_numpy(manifold).to(device)  # (N, D)
    r = torch.from_numpy(radii).to(device)  # (N,)
    out = torch.empty(M, dtype=torch.long, device=device)
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        dists = torch.cdist(s[start:end], m)  # (B, N)
        out[start:end] = (dists < r[None, :]).sum(dim=1)  # (B,)
    return out.cpu().numpy()


def _covered(
    centers: np.ndarray,
    others: np.ndarray,
    radii: np.ndarray,
    device: torch.device,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Whether each ball ``B(c_i, r_i)`` contains a point of ``others``."""
    N = centers.shape[0]
    c = torch.from_numpy(centers).to(device)  # (N, D)
    o = torch.from_numpy(others).to(device)  # (M, D)
    r = torch.from_numpy(radii).to(device)  # (N,)
    out = torch.empty(N, dtype=torch.bool, device=device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dists = torch.cdist(c[start:end], o)  # (B, M)
        out[start:end] = (dists < r[start:end, None]).any(dim=1)  # (B,)
    return out.cpu().numpy()


def compute_precision_recall_density_coverage(
    feats_real: np.ndarray,
    feats_gen: np.ndarray,
    k: int = 5,
    device: torch.device | str = "cpu",
    chunk_size: int = 1024,
) -> tuple[float, float, float, float]:
    """Precision, recall (Kynkäänniemi 2019), density, coverage (Naeem 2020)."""
    device = torch.device(device)
    radii_r = _knn_radii(feats_real, k, device, chunk_size)
    radii_g = _knn_radii(feats_gen, k, device, chunk_size)

    hits_gen_in_real = _count_hits(feats_gen, feats_real, radii_r, device, chunk_size)
    hits_real_in_gen = _count_hits(feats_real, feats_gen, radii_g, device, chunk_size)

    precision = float((hits_gen_in_real > 0).mean())
    recall = float((hits_real_in_gen > 0).mean())
    density = float(hits_gen_in_real.mean() / k)
    coverage = float(
        _covered(feats_real, feats_gen, radii_r, device, chunk_size).mean()
    )

    return precision, recall, density, coverage

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

"""Fréchet Inception Distance (Heusel et al., 2017)."""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics.image.fid import _compute_fid


def compute_fid(feats_real: np.ndarray, feats_gen: np.ndarray) -> float:
    """FID between two feature sets."""
    mu_r = torch.from_numpy(feats_real.mean(0)).double()
    mu_g = torch.from_numpy(feats_gen.mean(0)).double()
    sigma_r = torch.from_numpy(np.cov(feats_real, rowvar=False)).double()
    sigma_g = torch.from_numpy(np.cov(feats_gen, rowvar=False)).double()
    return float(_compute_fid(mu_r, sigma_r, mu_g, sigma_g))


def monge_inception_distance(
    x: np.ndarray, y: np.ndarray, rng_seed: int, n_projections: int = 1000
) -> float:
    """Monge Inception Distance: sliced Wasserstein distance of two feature sets."""
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    num_samples, d = x_t.shape
    assert num_samples == y_t.shape[0]

    ALPHA = 3 * d
    generator = torch.Generator(device=x_t.device).manual_seed(
        rng_seed
    )  # Use a fixed seed for reproducibility

    u_proj = torch.randn(
        (n_projections, d), generator=generator, dtype=x_t.dtype, device=x_t.device
    )  # Random projection vectors
    u_proj /= torch.linalg.norm(
        u_proj, dim=-1, keepdim=True
    )  # Normalize to unit vectors

    x_proj = u_proj @ x_t.T  # Project x onto the random vectors
    y_proj = u_proj @ y_t.T  # Project y onto the random vectors
    dists = torch.mean(
        (
            torch.topk(x_proj, num_samples, dim=-1).values
            - torch.topk(y_proj, num_samples, dim=-1).values
        )
        ** 2,
        dim=1,
    )  # Compute the squared distances between the sorted projections

    return ALPHA * torch.mean(dists).item()

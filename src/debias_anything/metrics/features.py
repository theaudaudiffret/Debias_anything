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

"""Feature extractors of the image metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def extract_inception_features(
    images: torch.Tensor,
    inception_model: nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Inception-V3 features (N, 2048) of images (N, C, H, W) in [-1, 1]."""
    feats = []
    for batch in images.split(batch_size):
        batch = (batch.clamp(-1, 1) + 1) / 2
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        feats.append(
            inception_model(batch.to(device))[0].squeeze(-1).squeeze(-1).cpu().numpy()
        )
    return np.concatenate(feats)


@torch.no_grad()
def extract_clean_fid_features(
    images: torch.Tensor,
    feat_model: Callable[[torch.Tensor], torch.Tensor],
    resizer: Callable[[np.ndarray], np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Clean-FID Inception features (Parmar et al., 2022) of images in [-1, 1]."""
    feats = []
    for batch in images.split(batch_size):
        batch = (batch.clamp(-1, 1) + 1) / 2 * 255
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        resized = np.stack(
            [resizer(img.permute(1, 2, 0).cpu().numpy()) for img in batch]
        )
        resized_batch = torch.from_numpy(resized).permute(0, 3, 1, 2).float()
        feats.append(feat_model(resized_batch.to(device)).cpu().numpy())
    return np.concatenate(feats)


@torch.no_grad()
def extract_sfid_features(
    images: torch.Tensor,
    inception_sfid_model: nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """sFID features (first 7 channels of Mixed_6e) of images in [-1, 1]."""
    feats = []
    for batch in images.split(batch_size):
        batch = (batch.clamp(-1, 1) + 1) / 2
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        spatial = inception_sfid_model(batch.to(device))[0][:, :7]
        feats.append(spatial.reshape(spatial.shape[0], -1).cpu().numpy())
    return np.concatenate(feats)

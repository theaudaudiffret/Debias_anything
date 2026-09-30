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

"""Vendi score (Friedman & Dieng, 2023): the effective number of distinct samples."""

from __future__ import annotations

import numpy as np
from vendi_score import vendi


def compute_vendi_score(features: np.ndarray) -> float:
    """Vendi score of features ``(N, D)`` under a cosine kernel."""
    n = features.shape[0]
    if n == 0:
        return float("nan")
    if n == 1:
        return 1.0

    # score_X (n×n Gram) and score_dual (D×D Gram) are mathematically equivalent
    # (shared nonzero eigenvalues) — pick whichever matrix is smaller to compute.
    score_fn = vendi.score_dual if features.shape[1] <= n else vendi.score_X
    return float(score_fn(features, normalize=True))

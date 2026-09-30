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

"""Fairness metrics: predictions, fairness discrepancy, per-class deviation table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .correct_classifier import CLEAM


@torch.no_grad()
def predict_classes(
    classifier: nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
    track_p_hat_0: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Argmax predictions of ``classifier``, and the per-batch ``p̂₀`` if tracked."""
    classifier = classifier.to(device).eval()
    preds: list[torch.Tensor] = []
    p_hat_0_per_batch: list[float] = []
    for batch in images.split(batch_size):
        logits = classifier(batch.to(device))  # (B, num_classes)
        labels = logits.argmax(dim=-1)  # (B,)
        if track_p_hat_0:
            p_hat_0_per_batch.append((labels == 0).float().mean().item())
        preds.append(labels.cpu())
    return torch.cat(preds).numpy(), np.array(p_hat_0_per_batch)


def fairness_discrepancy(
    preds: np.ndarray,
    num_classes: int,
    cleam: CLEAM | None = None,
    target: np.ndarray | None = None,
) -> float:
    """FD = ‖p* − p_target‖₂, CLEAM-corrected if ``cleam`` is given."""
    counts = np.bincount(preds, minlength=num_classes).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    if cleam is not None:
        p_star_0 = cleam.muCLEAM(p[0])
        p = np.array([p_star_0, 1.0 - p_star_0])  # binary case only
    if target is None:
        target = np.full(num_classes, 1.0 / num_classes)
    return float(np.linalg.norm(p - target))  # L2 norm


def class_deviation_table(
    preds: np.ndarray,
    targets_real: torch.Tensor,
    num_classes: int,
    cleam: CLEAM | None = None,
    p_hat_0_per_batch: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-class deviation table, with CLEAM 95 % bounds in the binary case."""
    counts_gen = np.bincount(preds, minlength=num_classes).astype(np.int64)
    mu_gen = counts_gen / max(counts_gen.sum(), 1)

    lower_bound = np.full(num_classes, np.nan)
    upper_bound = np.full(num_classes, np.nan)

    if cleam is not None:
        assert p_hat_0_per_batch is not None, "p_hat_0_per_batch required for CLEAM CI"
        mu_cleam_0, L0, U0 = cleam.confidence_CLEAM(p_hat_0_per_batch)
        proportions_gen = np.array([mu_cleam_0, 1.0 - mu_cleam_0])
        lower_bound = np.array([L0, 1.0 - U0])
        upper_bound = np.array([U0, 1.0 - L0])
    else:
        proportions_gen = mu_gen

    target_counts_real = np.bincount(
        targets_real.long().cpu(), minlength=num_classes
    ).astype(np.int64)
    proportions_target = target_counts_real / max(target_counts_real.sum(), 1)
    ecart = np.where(
        target_counts_real > 0,
        np.round((proportions_gen - proportions_target) / proportions_target * 100, 1),
        np.nan,
    )

    return pd.DataFrame(
        {
            "classe": np.arange(num_classes),
            "nombre_réel": target_counts_real,
            "proportions_réel": proportions_target,
            "nombre_généré": counts_gen,
            "proportions_généré_cleam": proportions_gen,
            "intervalle_confiance_inf_cleam": lower_bound,
            "intervalle_confiance_sup_cleam": upper_bound,
            "écart_gen_real%": ecart,
        }
    )

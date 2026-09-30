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

import numpy as np
from scipy.stats import t


class CLEAM:
    """CLEAM correction of the proportions of a binary classifier (Teo et al., 2023)."""

    def __init__(self, alpha0: float, alpha1: float, n_batches: int = 10):
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha1_prime = 1.0 - alpha1
        self.denom = alpha0 - self.alpha1_prime  # = α_0 + α_1 − 1
        if abs(self.denom) < 1e-3:
            raise ValueError(
                f"SAC near chance level (α_0 + α_1 − 1 = {self.denom:.3g}); "
                "CLEAM correction is ill-conditioned."
            )
        self.n_batches = n_batches

    def muCLEAM(self, p_hat_0: float) -> float:
        """Corrected proportion ``p*₀`` from the raw ``p̂₀`` (Eq. 8)."""
        return (p_hat_0 - self.alpha1_prime) / self.denom

    def confidence_CLEAM(self, p_hat_0_per_batch):
        """Corrected ``p*₀`` and its 95 % interval from the per-batch ``p̂₀``."""
        x = np.asarray(p_hat_0_per_batch, dtype=np.float64)
        s = x.size  # number of batches (Eq. 9)
        mu_hat = x.mean()  # Eq. 6
        sigma_hat = x.std(ddof=1)  # Eq. 7
        mu_cleam = self.muCLEAM(mu_hat)  # Eq. 8
        q = t.ppf(0.975, df=s - 1) if s < 30 else 1.96  #
        half_width = q * sigma_hat / (np.sqrt(s) * self.denom)  # Eq. 10
        return mu_cleam, mu_cleam - half_width, mu_cleam + half_width

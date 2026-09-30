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
import torch
from torch import nn


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeterministicSampler:
    def __init__(
        self,
        n_steps: int = 1000,
        batch_size: int = 64,
        channels: int = 3,
        imsize: int = 64,
        rau: int = 7,
        sigmax: torch.Tensor = torch.tensor(80.0),
        sigmin: torch.Tensor = torch.tensor(0.002),
        device: torch.device | str | None = None,
    ):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.dim = (channels, imsize, imsize)
        self.sigmin = sigmin
        self.sigmax = sigmax
        self.rau = rau
        self.device = torch.device(device) if device is not None else _default_device()
        self.t = self.calculate_time_steps()

    def calculate_time_steps(self) -> torch.Tensor:
        i = torch.arange(self.n_steps, dtype=torch.float32, device=self.device)
        t_steps = (
            self.sigmax ** (1 / self.rau)
            + i
            / (self.n_steps - 1)
            * (self.sigmin ** (1 / self.rau) - self.sigmax ** (1 / self.rau))
        ) ** self.rau

        t_zero = torch.tensor([0.0], device=self.device)
        self.t = torch.cat([t_steps, t_zero])
        return self.t

    def direction(
        self, model: nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        sigma_t = t.expand(x.shape[0]).reshape(-1, *([1] * (x.dim() - 1)))
        return 1 / t * (x - model(x, sigma_t))

    @torch.no_grad()
    def sample(self, model: nn.Module) -> torch.Tensor:
        x = self.t[0] * torch.randn(self.batch_size, *self.dim, device=self.device)
        for i in range(self.n_steps):
            d = self.direction(model, x, self.t[i])
            x_euler = x + (self.t[i + 1] - self.t[i]) * d
            if self.t[i + 1] > 0:
                d_forward = self.direction(model, x_euler, self.t[i + 1])
                x = x + (self.t[i + 1] - self.t[i]) * (0.5 * d + 0.5 * d_forward)
            else:
                x = x_euler
        return x


class StochasticSampler:
    def __init__(
        self,
        n_steps: int = 1000,
        batch_size: int = 64,
        channels: int = 3,
        imsize: int = 64,
        rau: int = 7,
        sigmax: torch.Tensor = torch.tensor(80.0),
        sigmin: torch.Tensor = torch.tensor(0.002),
        S_churn: float = 40.0,
        S_tmin: float = 0.05,
        S_tmax: float = 50.0,
        S_noise: float = 1.007,
        device: torch.device | str | None = None,
        hook: int | None = None,
        offload_hooks: bool = True,
    ):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.dim = (channels, imsize, imsize)
        self.sigmin = sigmin
        self.sigmax = sigmax
        self.rau = rau
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        self.device = torch.device(device) if device is not None else _default_device()
        self.hook_number = hook
        self.offload_hooks = offload_hooks
        self.t = self.calculate_time_steps()

    def calculate_time_steps(self) -> torch.Tensor:
        i = torch.arange(self.n_steps, dtype=torch.float32, device=self.device)
        t_steps = (
            self.sigmax ** (1 / self.rau)
            + i
            / (self.n_steps - 1)
            * (self.sigmin ** (1 / self.rau) - self.sigmax ** (1 / self.rau))
        ) ** self.rau

        t_zero = torch.tensor([0.0], device=self.device)
        self.t = torch.cat([t_steps, t_zero])
        return self.t

    def direction(
        self, model: nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        sigma_t = t.expand(x.shape[0]).reshape(
            -1, *([1] * (x.dim() - 1))
        )  # (B, 1, 1, 1)
        return 1 / t * (x - model(x, sigma_t))

    @torch.no_grad()
    def sample(
        self, model: nn.Module
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Sample a batch; with ``hook``, also the intermediate states and their σ."""
        x = self.t[0] * torch.randn(self.batch_size, *self.dim, device=self.device)
        store = (lambda t: t.detach().cpu()) if self.offload_hooks else (lambda t: t)
        hooks = [store(x)]
        noises = [self.t[0]]
        for i in range(self.n_steps):
            gamma = (
                min(self.S_churn / self.n_steps, np.sqrt(2) - 1)
                if self.S_tmin < self.t[i] < self.S_tmax
                else 0
            )
            t_i_hat = self.t[i] + gamma * self.t[i]
            x = (
                x
                + torch.sqrt(t_i_hat**2 - self.t[i] ** 2)
                * torch.randn_like(x)
                * self.S_noise
            )
            d = self.direction(model, x, t_i_hat)
            x_euler = x + (self.t[i + 1] - t_i_hat) * d
            if self.t[i + 1] > 0:
                d_forward = self.direction(model, x_euler, self.t[i + 1])
                x = x + (self.t[i + 1] - t_i_hat) * (0.5 * d + 0.5 * d_forward)
            else:
                x = x_euler
            if (
                self.hook_number is not None
                and self.hook_number != 0
                and i % (self.n_steps // self.hook_number) == 0
            ):
                hooks.append(store(x))
                noises.append(self.t[i + 1])
        if self.hook_number is not None:
            return x, hooks, noises
        return x

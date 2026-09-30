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

"""Numerics of the guidance gradients."""

import torch


def linf_normalize(grad: torch.Tensor) -> torch.Tensor:
    """Per-sample ℓ∞ normalisation of a gradient (Sehwag et al., CVPR 2022)."""
    denom = grad.flatten(1).abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    return grad / denom.view(-1, 1, 1, 1)


GRAD_SCALE = 2.0**12
"""Loss scale of the fp16 guidance backward passes."""


def scaled_grad(
    scalar: torch.Tensor,
    inputs: torch.Tensor,
    scale: float = GRAD_SCALE,
    retain_graph: bool = False,
) -> torch.Tensor:
    """∇ of ``scalar`` computed at ``scale·scalar`` against fp16 underflow, in fp32."""
    (grad,) = torch.autograd.grad(scalar * scale, inputs, retain_graph=retain_graph)
    grad = grad.float() / scale
    if not torch.isfinite(grad).all():
        raise FloatingPointError(
            f"non-finite gradient with a loss scale of {scale:g}: the fp16 backward pass "
            "saturated. Lower debias_anything.guidance.gradients.GRAD_SCALE and run again."
        )
    return grad

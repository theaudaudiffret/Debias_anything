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

"""EDM U-Net and preconditioning of the CelebA 64×64 generator (Karras et al., 2022)."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# EDM preconditioning constants
# ---------------------------------------------------------------------------

parameters = {
    "sigmin": 0.002,
    "sigmax": 80.0,
    "sigdata": 0.5,
    "rho": 7.0,
    "Pmean": -1.2,
    "Pstd": 1.2,
    "noise_levels": 1000,
}


def c_skip(sigma: torch.Tensor) -> torch.Tensor:
    # skip-connection weight of the EDM preconditioning
    return parameters["sigdata"] ** 2 / (sigma**2 + parameters["sigdata"] ** 2)


def c_out(sigma: torch.Tensor) -> torch.Tensor:
    # output weight of the score network in the EDM preconditioning
    return (
        sigma
        * parameters["sigdata"]
        / torch.sqrt(sigma**2 + parameters["sigdata"] ** 2)
    )


def c_in(sigma: torch.Tensor) -> torch.Tensor:
    # input weight of the score network in the EDM preconditioning
    return 1.0 / torch.sqrt(sigma**2 + (parameters["sigdata"]) ** 2)


def c_noise(sigma: torch.Tensor) -> torch.Tensor:
    # noise conditioning fed to the time embedding
    return 0.25 * torch.log(sigma)


def sample_sigma(
    Pmean: float = parameters["Pmean"], Pstd: float = parameters["Pstd"], size: int = 1
) -> torch.Tensor:
    log_sigma = torch.randn(size) * Pstd + Pmean
    return torch.exp(log_sigma)


# ---------------------------------------------------------------------------
# Time embedding: sinusoidal Fourier features + widened MLP
# ---------------------------------------------------------------------------


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.reshape(x.shape[0], 1) * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ---------------------------------------------------------------------------
# Residual block with AdaGN (scale + shift) instead of additive injection
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block with time-conditioned scale and shift (AdaGN)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
        groups: int = 8,
    ):
        super().__init__()
        g = min(groups, in_channels)
        while in_channels % g != 0:
            g -= 1
        g_out = min(groups, out_channels)
        while out_channels % g_out != 0:
            g_out -= 1

        # First branch: GN + SiLU + 3x3 conv
        self.norm1 = nn.GroupNorm(g, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # AdaGN on the second normalization: produces a (scale, shift) from
        # the time embedding. Zero-init to start at the identity (the block
        # then reduces to conv1 + shortcut).
        self.norm2 = nn.GroupNorm(g_out, out_channels, affine=False)
        self._time_linear = nn.Linear(time_dim, out_channels * 2)
        nn.init.zeros_(self._time_linear.weight)
        nn.init.zeros_(self._time_linear.bias)  # type: ignore[arg-type]
        self.time_proj = nn.Sequential(nn.SiLU(), self._time_linear)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 1x1 shortcut only if the dimensions change.
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # AdaGN: time-conditioned scale & shift.
        scale, shift = self.time_proj(t).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------


class AttentionBlock(nn.Module):
    """Multi-head self-attention over a feature map, zero-initialised output."""

    def __init__(self, channels: int, num_heads: int = 4, groups: int = 8):
        super().__init__()
        if channels % num_heads != 0:
            # Fall back to a valid divisor for robustness.
            while channels % num_heads != 0 and num_heads > 1:
                num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        g = min(groups, channels)
        while channels % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        # Zero-init of the output projection.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        # (B, 3, num_heads, head_dim, H*W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)

        # Switch to (B, num_heads, seq_len, head_dim) for SDPA.
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v
            )  # uses FlashAttention if available
        else:  # plain fallback (less efficient)
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = attn.softmax(dim=-1)
            out = attn @ v  # shape (B, num_heads, seq_len, head_dim)

        # Back to (B, C, H, W).
        out = out.transpose(-2, -1).reshape(
            B, C, H, W
        )  # from shape (B, num_heads, seq_len, head_dim) to (B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Down/up-sampling blocks
# ---------------------------------------------------------------------------


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Time embedding module: sinusoidal Fourier features + wide MLP + SiLU
# ---------------------------------------------------------------------------


class TimeEmbedding(nn.Module):
    """Sinusoidal features followed by a two-layer MLP."""

    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        self.sinus = SinusoidalEmbedding(dim)
        hidden = dim * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
            nn.SiLU(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # x has shape (B,), with B = batch size
        return self.mlp(self.sinus(x))  # shape (B, dim)


# ---------------------------------------------------------------------------
# DiffusionUNet
# ---------------------------------------------------------------------------


class DiffusionUNet(nn.Module):
    """EDM U-Net ``F_θ(x, c_noise)``, by default that of the CelebA 64×64 model."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 96,
        channel_mults: tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        time_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_resolutions: tuple[int, ...] = (16, 8),
        out_channels: int = 3,
        image_size: int = 64,
    ):
        super().__init__()

        # Check that the resolution is divisible by 2^(n_downsamples), so that
        # the stack of upsamples lands exactly on image_size at the output.
        n_downs = len(channel_mults) - 1
        if image_size % (2**n_downs) != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by 2**{n_downs}"
                f" = {2**n_downs} (one per channel_mults stage except the last)."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.image_size = image_size

        # --- Time embedding ---------------------------------------------------
        self.time_mlp = TimeEmbedding(time_dim, hidden_mult=4)

        # --- Stem -------------------------------------------------------------
        ch = base_channels
        self.init_conv = nn.Conv2d(
            in_channels, ch, kernel_size=3, padding=1
        )  # input conv projecting to the base channels, shape (in_channels, ch, 3, 3)

        # --- Encoder ---------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Track the number of channels at each skip connection, to size the
        # decoder blocks correctly.
        channels_at_skip: list[int] = [ch]  # after init_conv
        current_ch = ch
        current_res = image_size  # starting resolution — propagated along the U-Net

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage_blocks.append(
                    ResBlock(current_ch, out_ch, time_dim, dropout=dropout)
                )  # shape (num_res_blocks, ch) with ch = base_channels * mult
                current_ch = out_ch
                channels_at_skip.append(current_ch)
            self.down_blocks.append(
                stage_blocks
            )  # shape (num_stages, num_res_blocks, ch)

            # Attention if the current resolution is in the list.
            if current_res in attn_resolutions:
                self.down_attns.append(AttentionBlock(current_ch, num_heads=num_heads))
            else:
                self.down_attns.append(nn.Identity())

            # No downsample after the last scale.
            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample(current_ch))
                channels_at_skip.append(current_ch)
                current_res //= 2
            else:
                self.downsamples.append(nn.Identity())

        # --- Middle ----------------------------------------------------------
        self.mid_block1 = ResBlock(current_ch, current_ch, time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(current_ch, num_heads=num_heads)
        self.mid_block2 = ResBlock(current_ch, current_ch, time_dim, dropout=dropout)

        # --- Decoder (symmetric) --------------------------------------------
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Traverse the scales in reverse order.
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()
            # num_res_blocks + 1 blocks on the decoder side: the standard
            # convention (Ho et al. 2020) to absorb the extra skip.
            for _ in range(num_res_blocks + 1):
                skip_ch = channels_at_skip.pop()
                stage_blocks.append(
                    ResBlock(current_ch + skip_ch, out_ch, time_dim, dropout=dropout)
                )
                current_ch = out_ch
            self.up_blocks.append(stage_blocks)

            if current_res in attn_resolutions:
                self.up_attns.append(AttentionBlock(current_ch, num_heads=num_heads))
            else:
                self.up_attns.append(nn.Identity())

            # Upsample after every stage except the last (highest resolution).
            if i != len(channel_mults) - 1:
                self.upsamples.append(Upsample(current_ch))
                current_res *= 2
            else:
                self.upsamples.append(nn.Identity())

        # --- Head ------------------------------------------------------------
        # Zero-init of the final conv (as in AdaLN-Zero).
        final_conv = nn.Conv2d(current_ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)  # type: ignore[arg-type]
        self.final = nn.Sequential(
            nn.GroupNorm(min(8, current_ch), current_ch),
            nn.SiLU(),
            final_conv,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, c_noise_sigma: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(c_noise_sigma)

        # Stem
        h = self.init_conv(x)
        skips: list[torch.Tensor] = [h]

        # Encoder
        for stage_blocks, attn, down in zip(
            self.down_blocks, self.down_attns, self.downsamples
        ):
            for block in cast(nn.ModuleList, stage_blocks):
                h = block(h, t_emb)
                skips.append(h)
            if not isinstance(attn, nn.Identity):
                h = attn(h)
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for stage_blocks, attn, up in zip(
            self.up_blocks, self.up_attns, self.upsamples
        ):
            for block in cast(nn.ModuleList, stage_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            if not isinstance(attn, nn.Identity):
                h = attn(h)
            if not isinstance(up, nn.Identity):
                h = up(h)

        return self.final(h)


# ---------------------------------------------------------------------------
# EDM loss & Denoiser
# ---------------------------------------------------------------------------


def edm_loss(model: nn.Module, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(y) * sigma
    weight = 1
    target = (1 / c_out(sigma)) * (y - c_skip(sigma) * (y + noise))
    pred = model(c_in(sigma) * (y + noise), c_noise(sigma))
    loss_sample = ((pred - target) ** 2).mean(dim=list(range(1, y.dim())))
    return (weight * loss_sample).mean()


class Denoiser(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return c_skip(sigma) * x + c_out(sigma) * self.model(
            c_in(sigma) * x, c_noise(sigma)
        )


def build_unet_for(
    image_size: int,
    in_channels: int,
    out_channels: int | None = None,
    **overrides,
) -> DiffusionUNet:
    """The ``DiffusionUNet`` of the CelebA 64×64 model, the only preset."""
    if image_size != 64:
        raise ValueError(
            f"no preset for image_size={image_size}: the CelebA model is 64×64"
        )
    if out_channels is None:
        out_channels = in_channels
    return DiffusionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        image_size=image_size,
        **overrides,
    )


__all__ = [
    "parameters",
    "c_skip",
    "c_out",
    "c_in",
    "c_noise",
    "sample_sigma",
    "SinusoidalEmbedding",
    "ResBlock",
    "AttentionBlock",
    "DiffusionUNet",
    "edm_loss",
    "Denoiser",
    "build_unet_for",
]

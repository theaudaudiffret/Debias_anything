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

"""Train the adapter of SD 1.5 on the cache of ``build_adapter_cache.py`` (App. C)."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import hydra
import mlflow
import torch
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader, TensorDataset

from debias_anything.adapters import MultiBlockHSpaceToSigLIP, cosine_loss
from debias_anything.configs import MLflowConfig
from debias_anything.hspace.sd15 import (
    BLOCK_SHAPES_DECODER,
    BLOCKS_DECODER_DEFAULT,
    H_DOWNSAMPLE,
    sd15_hspace_decoder,
)
from debias_anything.log import get_logger
from debias_anything.mlflow_utils import log_cfg_to_mlflow, set_experiment
from debias_anything.models.sd15 import SigmaBridge, load_pipeline
from debias_anything.paths import CONF_DIR, REPO_ROOT

# Validation σ (R@1 per noise level), over the [0.029, 14.615] range of the SD 1.5 scheduler.
EVAL_SIGMAS = (0.05, 0.34, 1.0, 3.33, 8.35, 14.6)

# The σ of ``EVAL_SIGMAS`` that fall inside the guidance window: (1.0, 3.33).
GUIDANCE_SIGMAS = tuple(sg for sg in EVAL_SIGMAS if 1.0 <= sg <= 6.0)


def load_cache(
    path: pathlib.Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """The concatenated shards of the adapter cache."""
    shards = sorted(path.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(
            f"no shard in {path}; run first "
            "`uv run python scripts/sd15/build_adapter_cache.py`"
        )
    parts = [torch.load(s, map_location="cpu") for s in shards]
    latents = torch.cat([p["latents"] for p in parts])
    siglip = torch.cat([p["siglip"] for p in parts])
    prompt_id = torch.cat([p["prompt_id"] for p in parts]).long()
    bank = torch.load(path / "prompts.pt", map_location="cpu")
    return latents, siglip, prompt_id, bank


logger = get_logger(
    "hspace_to_siglip_sd15_decoder",
    log_file="hspace_to_siglip_sd15_decoder.json",
)


@dataclass
class Config:
    cache: str = "data/sd15_adapter_cache"
    out: str = "checkpoints/sd15_adapter.pth"
    mlflow: MLflowConfig = field(
        default_factory=lambda: MLflowConfig(
            experiment_name="hspace_to_siglip_sd15_decoder"
        )
    )
    resolution: int = 512
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 0.05  # the aggregation adds parameters
    # A = batch_size × m_views; the U-Net sees A samples (cond branch only). Kept small: here
    # the forward pass is complete (not truncated at the bottleneck) and ``up_blocks.2/.3``
    # output 64², i.e. 4096 positions per image against 64 for the mid_block.
    batch_size: int = 64
    m_views: int = 2  # independent σ per image: trains the σ-invariance of the head
    sigma_min: float | None = None  # null → bridge.sigma_min (0.029)
    sigma_max: float | None = None  # null → bridge.sigma_max (14.615)

    # --- aggregation ---
    # The bottleneck + the four decoder blocks. ``[mid_block]`` alone is the control that isolates
    # the cost of the aggregation stage without changing what is read.
    blocks: tuple[str, ...] = BLOCKS_DECODER_DEFAULT
    projection_dim: int = (
        512  # common width (384 in Readout Guidance and Hyperfeatures)
    )

    # --- trunk (HSpaceViT, debias_anything.adapters.vit) ---
    time_dim: int = 128
    pool_dim: int = 512  # working width = min(projection_dim, pool_dim)
    # head_dim = pool_dim/attn_heads must be divisible by 4 (axial RoPE)
    attn_heads: int = 8
    n_blocks: int = 6
    mlp_ratio: float = 2.0
    dropout: float = 0.1

    n_val: int = 1000  # last images of the cache, excluded from training
    # Epochs of linear lr warmup before cosine annealing. 0 → plain cosine annealing, i.e. the
    # full lr of 5e-4 from the first step on pre-norm blocks.
    warmup_epochs: int = 3
    log_every: int = 50
    device: str | None = None


ConfigStore.instance().store(name="sd15_adapter_schema", node=Config)


# ----------------------------------------------------------------------- metrics


@torch.no_grad()
def evaluate(
    proj, unet, bridge, val, bank_gpu, cfg: Config, device, step: int
) -> tuple[dict[str, float], dict[float, float]]:
    """Alignment, gap and R@1 per σ on the validation split, with fixed noise."""
    proj.eval()
    x0, tgt, pid = val
    n = x0.shape[0]
    out: dict[str, float] = {}
    r1_by_sigma: dict[float, float] = {}

    for sg in EVAL_SIGMAS:
        zs = []
        for s in range(0, n, cfg.batch_size):
            b = x0[s : s + cfg.batch_size].to(device).float()
            ids = pid[s : s + cfg.batch_size].to(device)
            sigma = torch.full((b.shape[0], 1, 1, 1), sg, device=device)
            g = torch.Generator(device=device).manual_seed(s * 1000 + int(sg * 100))
            xs = b + sigma * torch.randn(b.shape, device=device, generator=g)
            feats = sd15_hspace_decoder(
                unet, xs, sigma, bank_gpu["embeds"][ids], bridge, blocks=cfg.blocks
            )
            zs.append(proj(feats, bridge.t_of_sigma(sigma)).cpu())
        z = torch.cat(zs)

        sim = z @ tgt.t()  # (n, n), all (projection, target) pairs
        align = float(sim.diagonal().mean())  # = cos+
        cos_neg = float((sim.sum() - sim.diagonal().sum()) / (n * (n - 1)))
        gap = align - cos_neg
        r1 = float((sim.argmax(-1) == torch.arange(n)).float().mean())

        out |= {f"align/s{sg}": align, f"gap/s{sg}": gap, f"r1/s{sg}": r1}
        r1_by_sigma[sg] = r1
        logger.info(
            f"[eval s{step}] σ={sg:6.2f} align={align:.3f} gap={gap:+.4f} R@1={r1:.3f}",
            extra={
                "event": "eval",
                "step": step,
                "sigma": sg,
                "align": align,
                "gap": gap,
                "r1": r1,
            },
        )

    # The mixing weights: THE diagnostic of this file. A softmax that converges to mid_block
    # alone says that the decoder is not useful; that is an answer, not a failure.
    mix = proj.mixing_weights.tolist()
    out |= {f"mix/{name}": w for name, w in zip(cfg.blocks, mix, strict=True)}
    logger.info(
        "mixing weights (softmax): "
        + "  ".join(f"{n_}={w:.3f}" for n_, w in zip(cfg.blocks, mix, strict=True)),
        extra={
            "event": "mixing",
            "step": step,
            "mixing": dict(zip(cfg.blocks, mix, strict=True)),
        },
    )

    mlflow.log_metrics(out, step=step)
    proj.train()
    return out, r1_by_sigma


# ---------------------------------------------------------------------- training


def train(cfg: Config, pipe=None) -> MultiBlockHSpaceToSigLIP:
    """Train the adapter and write the checkpoint of best mean R@1."""
    if pipe is None:
        pipe = load_pipeline(
            cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            scheduler="euler",
        )
    unet, scheduler = pipe.unet, pipe.scheduler
    device = next(unet.parameters()).device
    cache = REPO_ROOT / cfg.cache
    latents, siglip, prompt_id, bank = load_cache(cache)

    n_total = latents.shape[0]
    n_val = min(cfg.n_val, n_total // 5)  # at most 20 % of the cache for validation
    n_train = n_total - n_val
    logger.info(
        f"cache {cache}: {n_total} images ({n_train} train / {n_val} val), "
        f"latent {tuple(latents.shape[1:])}, {len(bank['prompts'])} prompts"
    )

    unet.eval().requires_grad_(False)
    unet_id = getattr(unet.config, "_name_or_path", "") or "<inconnu>"
    bridge = SigmaBridge.from_scheduler(scheduler).to(device)
    logger.info(
        f"UNet={unet_id} scheduler={type(scheduler).__name__} device={device} "
        f"| σ ∈ [{bridge.sigma_min:.3f}, {bridge.sigma_max:.3f}]"
    )

    s_min = cfg.sigma_min if cfg.sigma_min is not None else bridge.sigma_min
    s_max = cfg.sigma_max if cfg.sigma_max is not None else bridge.sigma_max

    bank_gpu = {"embeds": bank["embeds"].to(device)}  # (P, 77, 768) fp16

    # Hydra returns a ListConfig for ``blocks``; freeze it into a tuple of str.
    blocks = tuple(str(b) for b in cfg.blocks)
    # Input channels per block: the decoder table, passed to the model so that it can be built
    # without running the U-Net, and written into the checkpoint, from which the loader reads it.
    block_channels = {b: BLOCK_SHAPES_DECODER[b][0] for b in blocks}
    # 8 at 512²: the common grid of the aggregation, that of the mid_block
    h_spatial = cfg.resolution // H_DOWNSAMPLE
    if h_spatial < 1:
        raise ValueError(
            f"resolution={cfg.resolution}: the common grid is resolution/{H_DOWNSAMPLE}, "
            f"at least {H_DOWNSAMPLE} is required."
        )

    proj = MultiBlockHSpaceToSigLIP(
        blocks=blocks,
        grid=h_spatial,
        projection_dim=cfg.projection_dim,
        time_dim=cfg.time_dim,
        out_dim=siglip.shape[1],
        num_heads=cfg.attn_heads,
        pool_dim=cfg.pool_dim,
        n_blocks=cfg.n_blocks,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        block_channels=block_channels,
    ).to(device)

    n_agg = sum(p.numel() for p in proj.aggregate.parameters())
    n_all = sum(p.numel() for p in proj.parameters())
    logger.info(
        f"decoder adapter: {list(blocks)} → {cfg.projection_dim}×{h_spatial}² → "
        f"{siglip.shape[1]}d | {n_all / 1e6:.1f} M params, {n_agg / 1e6:.1f} M of them for aggregation "
        f"| σ ∈ [{s_min:.3f}, {s_max:.3f}] uniform, m_views={cfg.m_views}"
    )

    optimizer = torch.optim.AdamW(
        proj.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # ``lr_sched`` and not ``scheduler``: the latter is the diffusion scheduler.
    # One step per epoch (``lr_sched.step()`` is at the end of the epoch loop), so T_max and
    # total_iters are counted in epochs.
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs)
    )
    if cfg.warmup_epochs > 0:
        lr_sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
                ),
                cosine,
            ],
            milestones=[cfg.warmup_epochs],
        )
    else:
        lr_sched = cosine

    loader = DataLoader(
        TensorDataset(latents[:n_train], siglip[:n_train], prompt_id[:n_train]),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )  # num_workers=0: the cache is already in RAM, workers would only duplicate it
    val = (latents[n_train:], siglip[n_train:], prompt_id[n_train:])

    out_path = REPO_ROOT / cfg.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = -float("inf")
    step = 0

    for epoch in range(1, cfg.epochs + 1):
        for x0_cpu, tgt_cpu, pid_cpu in loader:
            b = x0_cpu.shape[0]
            x0 = x0_cpu.to(device, non_blocking=True).float()  # (B, 4, 64, 64)
            tgt = tgt_cpu.to(device, non_blocking=True)  # (B, 768)
            pid = pid_cpu.to(device, non_blocking=True)  # (B,)

            src_idx = torch.arange(b, device=device).repeat(cfg.m_views)  # (A,)
            a = src_idx.numel()
            sigma = bridge.sample_sigma_uniform(a, device, s_min, s_max)  # (A,1,1,1)
            xs = x0[src_idx] + sigma * torch.randn_like(x0[src_idx])  # noisy latent
            ids = pid[src_idx]  # the prompt that generated the image

            with torch.no_grad():
                feats = sd15_hspace_decoder(
                    unet, xs, sigma, bank_gpu["embeds"][ids], bridge, blocks=blocks
                )
            # The features are **not** converted to fp32 here: the aggregation casts them block
            # by block (``feats[name].to(cond.dtype)``), so a single fp32 copy is alive at a time.
            # At 64², ``up_blocks.2`` alone weighs 1.3 GiB in fp32 for A = 128; converting them
            # all at once would only serve to add them up.

            z = proj(feats, bridge.t_of_sigma(sigma))
            loss, cos_pos, cos_neg = cosine_loss(z, tgt, src_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % cfg.log_every == 0:
                # gap = cos+ − cos−: exactly zero for a constant output, linear in the fraction
                # of the residual recovered. The first indicator to move.
                gap = cos_pos - cos_neg
                logger.info(
                    f"epoch {epoch} step {step} loss={loss.item():.4f} "
                    f"cos+={cos_pos:.3f} cos-={cos_neg:.3f} gap={gap:+.4f}",
                    extra={
                        "event": "train",
                        "step": step,
                        "loss": loss.item(),
                        "cos_pos": cos_pos,
                        "cos_neg": cos_neg,
                        "gap": gap,
                    },
                )
                mlflow.log_metrics(
                    {
                        "train/loss": loss.item(),
                        "train/cos_pos": cos_pos,
                        "train/cos_neg": cos_neg,
                        "train/gap": gap,
                    },
                    step=step,
                )

        lr_sched.step()
        metrics, r1 = evaluate(proj, unet, bridge, val, bank_gpu, cfg, device, step)
        # Selection on R@1 averaged over σ: discriminability, hence the only metric that a
        # collapse to the marginal embedding cannot fool.
        score = sum(r1.values()) / len(r1)
        gap_mean = sum(v for k, v in metrics.items() if k.startswith("gap/")) / len(r1)
        # R@1 restricted to the σ of the window where the guidance actually runs ([1, 6]).
        # Diagnostic only: the SELECTION stays on r1_mean, so that runs remain comparable.
        win = [
            v for sg, v in r1.items() if GUIDANCE_SIGMAS[0] <= sg <= GUIDANCE_SIGMAS[-1]
        ]
        mlflow.log_metrics(
            {
                "eval/r1_mean": score,
                "eval/gap_mean": gap_mean,
                "eval/r1_window": sum(win) / len(win),
                # Logged so that the lr schedule can be diagnosed in MLflow afterwards.
                "train/lr": lr_sched.get_last_lr()[0],
            },
            step=step,
        )
        logger.info(
            f"epoch {epoch} | mean R@1 over σ = {score:.3f} | mean gap = {gap_mean:+.4f}"
        )
        if score > best:
            best = score
            torch.save(
                {
                    "state_dict": proj.state_dict(),
                    # Same class and loader as the encoder multi-block adapter
                    # (debias_anything.adapters.multiblock.load_multiblock_projector): what
                    # distinguishes the two readouts is ``blocks``, and the channels are stored in
                    # ``block_channels``.
                    "arch": "hspace_multiblock",
                    "blocks": list(blocks),
                    "block_channels": block_channels,
                    "grid": h_spatial,
                    "projection_dim": cfg.projection_dim,
                    "time_dim": cfg.time_dim,
                    "out_dim": siglip.shape[1],
                    "num_heads": cfg.attn_heads,
                    "pool_dim": cfg.pool_dim,
                    "n_blocks": cfg.n_blocks,
                    "mlp_ratio": cfg.mlp_ratio,
                    "dropout": cfg.dropout,
                    "resolution": cfg.resolution,
                    # Learned mixing weights, in plain form: the scientific result of the run.
                    "mixing_weights": proj.mixing_weights.tolist(),
                    # The input is read on the conditional branch.
                    "h_input": "cond",
                    "unet": unet_id,
                    "scheduler": type(scheduler).__name__,
                    # conditioning convention: proj(feats, bridge.t_of_sigma(σ))
                    "time_cond": "timestep",
                    "sigma_min": s_min,
                    "sigma_max": s_max,
                },
                out_path,
            )
            logger.info(f"checkpoint written ({score:.3f}) → {out_path}")

    logger.info(f"best mean R@1 over σ: {best:.3f} — {out_path}")
    return proj


@hydra.main(
    version_base="1.3",
    config_path=str(CONF_DIR / "sd15"),
    config_name="adapter",
)
def main(cfg: Config) -> None:
    set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        log_cfg_to_mlflow(cfg)
        train(cfg)


if __name__ == "__main__":
    main()

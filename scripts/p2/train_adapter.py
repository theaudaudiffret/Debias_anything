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

"""Train the adapter of P2 on CelebA-HQ (Appendix C)."""

from __future__ import annotations

from dataclasses import dataclass

import hydra
import mlflow
import torch
from hydra.core.config_store import ConfigStore
from torch import nn
from torch.utils.data import DataLoader, Dataset

from debias_anything.adapters import HSpaceToSigLIP, cosine_loss
from debias_anything.adapters.hspace_to_siglip import P2_ADAPTER_FORMAT
from debias_anything.data import CelebAHQ
from debias_anything.guidance.p2 import _read_h
from debias_anything.mlflow_utils import set_experiment
from debias_anything.models.p2 import T0, alpha_bar, ddim_timesteps, load_p2
from debias_anything.paths import CONF_DIR, resolve
from debias_anything.siglip import SIGLIP, load_siglip, siglip_image

# Preprocessing of the SigLIP targets of the released adapter (see debias_anything.siglip).
TARGET_PREPROCESSING = dict(resize="bicubic", clamp=False)


@dataclass
class Config:
    out: str = "checkpoints/celebahq_p2_adapter.pt"
    p2_checkpoint: str = "checkpoints/celebahq_p2.pt"
    siglip: str = SIGLIP
    train_split: str = "train"
    val_split: str | None = "validation"  # null disables the validation
    # SigLIP targets, cached next to ``out`` by default: <out stem>_siglip_targets_<split>.pt
    target_cache: str | None = None
    rebuild_target_cache: bool = False
    batch_size: int = 32
    views_per_image: int = 2  # independent timesteps per image
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    n_ddim_steps: int = 50
    # the initial t=999 state is never guided, hence not trained on by default
    include_t999: bool = False
    log_every: int = (
        100  # optimizer steps between two logs of the train loss (0 disables)
    )
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mlflow_experiment: str = "balancing_act_projector"
    mlflow_run_name: str | None = None  # null: the stem of ``out``


ConfigStore.instance().store(name="p2_adapter_schema", node=Config)


class TargetDataset(Dataset):
    """``(x₀, SigLIP(x₀))`` pairs of a split, with the targets precomputed."""

    def __init__(self, base: CelebAHQ, targets: torch.Tensor):
        if len(base) != len(targets):
            raise ValueError(
                f"{len(base)} images but {len(targets)} cached SigLIP targets"
            )
        self.base, self.targets = base, targets.float()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.base[index]
        return x, self.targets[index]


def _loader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def cache_path_for(cfg: Config, split: str):
    if cfg.target_cache is not None:
        path = resolve(cfg.target_cache)
        return path.with_name(f"{path.stem}_{split}{path.suffix}")
    out = resolve(cfg.out)
    return out.with_name(f"{out.stem}_siglip_targets_{split}.pt")


@torch.no_grad()
def load_or_build_targets(
    dataset: CelebAHQ,
    split: str,
    siglip: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    size: int,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    path = cache_path_for(cfg, split)
    if path.exists() and not cfg.rebuild_target_cache:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("n_images") != len(dataset)
            or payload.get("siglip") != cfg.siglip
            or payload.get("source_id") != dataset.source_id
        ):
            raise ValueError(
                f"Incompatible target cache {path}; use rebuild_target_cache=true"
            )
        print(f"loaded {split} SigLIP targets <- {path}")
        return payload["embeddings"].float()

    embeddings: list[torch.Tensor] = []
    batch_size = max(cfg.batch_size, 16)
    for step, (x, _) in enumerate(_loader(dataset, batch_size, cfg.num_workers, False)):
        embeddings.append(
            siglip_image(siglip, x.to(device), mean, std, size, **TARGET_PREPROCESSING)
            .float()
            .cpu()
        )
        if step % 25 == 0:
            print(
                f"cache {split}: {min((step + 1) * batch_size, len(dataset))}/{len(dataset)}"
            )
    targets = torch.cat(embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n_images": len(dataset),
            "siglip": cfg.siglip,
            "source_id": dataset.source_id,
            "embeddings": targets,
        },
        path,
    )
    print(f"saved {split} SigLIP targets -> {path}")
    return targets


@torch.no_grad()
def extract_h(
    unet: nn.Module, x_t: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """P2's h-space (``middle_block``) at ``(x_t, t)``."""
    return _read_h(unet, x_t, timesteps)


def temporal_condition(timesteps: torch.Tensor, t0: int) -> torch.Tensor:
    """Time condition of the adapter, ``t / t0``."""
    return timesteps.float() / float(t0)


def noised(x0, grid, schedule):
    """``(x_t, t)`` with t drawn uniformly on ``grid``."""
    sampled = grid[torch.randint(len(grid), (x0.shape[0],), device=x0.device)]
    abar = schedule[sampled].view(-1, 1, 1, 1)
    return abar.sqrt() * x0 + (1.0 - abar).sqrt() * torch.randn_like(x0), sampled


@torch.no_grad()
def evaluate_r1(unet, projector, loader, grid, schedule, device) -> float:
    """R@1: fraction of projections whose nearest SigLIP target is their own."""
    projector.eval()
    zs, targets = [], []
    for x0, target in loader:
        x0, target = x0.to(device), target.to(device)
        x_t, sampled = noised(x0, grid, schedule)
        h = extract_h(unet, x_t, sampled)
        zs.append(projector(h.float(), temporal_condition(sampled, T0)))
        targets.append(target)
    projector.train()
    z, target = torch.cat(zs), torch.cat(targets)
    nearest = (z @ target.t()).argmax(dim=1)
    return (nearest == torch.arange(len(z), device=device)).float().mean().item()


@hydra.main(version_base="1.3", config_path=str(CONF_DIR / "p2"), config_name="adapter")
def main(cfg: Config) -> None:
    if cfg.views_per_image < 1:
        raise ValueError("views_per_image must be positive")
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    out = resolve(cfg.out)

    train_base = CelebAHQ(cfg.train_split)
    val_base = CelebAHQ(cfg.val_split) if cfg.val_split else None
    print(
        f"train images={len(train_base)} | val images={len(val_base) if val_base else 0}"
    )

    # Cache the targets before loading P2: both frozen networks are large at 256².
    siglip, _, mean, std, size = load_siglip(device, cfg.siglip)
    train_targets = load_or_build_targets(
        train_base, "train", siglip, mean, std, size, cfg, device
    )
    val_targets = (
        load_or_build_targets(val_base, "val", siglip, mean, std, size, cfg, device)
        if val_base
        else None
    )
    out_dim = train_targets.shape[-1]
    del siglip
    if device.type == "cuda":
        torch.cuda.empty_cache()

    unet = load_p2(resolve(cfg.p2_checkpoint), device)
    grid = ddim_timesteps(T0, cfg.n_ddim_steps, device)
    grid = grid if cfg.include_t999 else grid[:-1]
    schedule = alpha_bar(T0, device)
    print(f"DDIM guidance grid ({len(grid)} states): {grid.tolist()}")

    # Probe rather than hard-code 512x8x8.
    probe_x, _ = train_base[0]
    probe_h = extract_h(unet, probe_x.unsqueeze(0).to(device), grid[:1])
    channels, height, width = probe_h.shape[1:]
    if height != width:
        raise ValueError(f"Expected square h-space, got {tuple(probe_h.shape)}")
    print(f"P2 h-space: {tuple(probe_h.shape)}")

    projector = HSpaceToSigLIP(
        h_channels=channels, num_tokens=height * width, out_dim=out_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    train_loader = _loader(
        TargetDataset(train_base, train_targets), cfg.batch_size, cfg.num_workers, True
    )
    val_loader = (
        _loader(
            TargetDataset(val_base, val_targets), cfg.batch_size, cfg.num_workers, False
        )
        if val_base and val_targets is not None
        else None
    )
    best_val = float("-inf")
    out.parent.mkdir(parents=True, exist_ok=True)

    set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run(run_name=cfg.mlflow_run_name or out.stem):
        mlflow.log_params(
            {
                "format": P2_ADAPTER_FORMAT,
                "train_source": train_base.source_id,
                "val_source": val_base.source_id if val_base else "",
                "n_train": len(train_base),
                "n_val": len(val_base) if val_base else 0,
                "p2_checkpoint": str(resolve(cfg.p2_checkpoint)),
                "siglip": cfg.siglip,
                "batch_size": cfg.batch_size,
                "views_per_image": cfg.views_per_image,
                "epochs": cfg.epochs,
                "log_every": cfg.log_every,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "n_ddim_steps": cfg.n_ddim_steps,
                "include_t999": cfg.include_t999,
                "h_channels": channels,
                "h_tokens": height * width,
                "siglip_dim": out_dim,
            }
        )

        global_step = 0
        for epoch in range(1, cfg.epochs + 1):
            projector.train()
            total_loss = total_cosine = 0.0
            total = 0
            for x0, target in train_loader:
                x0 = x0.to(device, non_blocking=True).repeat(
                    cfg.views_per_image, 1, 1, 1
                )
                target = target.to(device, non_blocking=True).repeat(
                    cfg.views_per_image, 1
                )
                x_t, sampled = noised(x0, grid, schedule)
                h = extract_h(unet, x_t, sampled)
                z = projector(h.float(), temporal_condition(sampled, T0))
                loss, cosine, _ = cosine_loss(
                    z, target, torch.arange(len(z), device=device)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x0)
                total_cosine += cosine * len(x0)
                total += len(x0)
                global_step += 1
                if cfg.log_every and global_step % cfg.log_every == 0:
                    mlflow.log_metrics(
                        {"train/loss_step": loss.item(), "train/cosine_step": cosine},
                        step=global_step,
                    )

            train_loss = total_loss / total
            train_cosine = total_cosine / total
            val_r1 = (
                evaluate_r1(unet, projector, val_loader, grid, schedule, device)
                if val_loader
                else float("nan")
            )
            metrics = {"train/loss": train_loss, "train/cosine": train_cosine}
            if val_loader:
                metrics["val/r1"] = val_r1
            mlflow.log_metrics(metrics, step=epoch)
            summary = (
                f"epoch={epoch:03d} loss={train_loss:.5f} train_cos={train_cosine:.4f}"
            )
            if val_loader:
                summary += f" val_r1={val_r1:.4f}"
            print(summary)
            if val_loader and val_r1 <= best_val:
                continue
            if val_loader:
                best_val = val_r1

            torch.save(
                {
                    "format": P2_ADAPTER_FORMAT,
                    "state_dict": projector.state_dict(),
                    "arch": {
                        "h_channels": channels,
                        "num_tokens": height * width,
                        "out_dim": out_dim,
                    },
                    "ddim_timesteps": grid.cpu(),
                    "beta_schedule": {
                        "name": "linear",
                        "beta_start": 1e-4,
                        "beta_end": 2e-2,
                        "steps": T0 + 1,
                    },
                    "time_condition": "t_over_t0",
                    "t0": T0,
                    "siglip": cfg.siglip,
                    "p2_checkpoint": str(resolve(cfg.p2_checkpoint)),
                    "training": {
                        "views_per_image": cfg.views_per_image,
                        "n_train": len(train_base),
                        "n_val": len(val_base) if val_base else 0,
                        "train_source": train_base.source_id,
                        "val_source": val_base.source_id if val_base else None,
                    },
                    "best_val_r1": best_val if val_loader else None,
                },
                out,
            )
            label = f"best val_r1={best_val:.4f}" if val_loader else "final epoch"
            print(f"saved adapter ({label}) -> {out}")

        mlflow.log_artifact(str(out), artifact_path="checkpoints")


if __name__ == "__main__":
    main()

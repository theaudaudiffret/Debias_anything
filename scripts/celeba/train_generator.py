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

"""Train the CelebA 64×64 EDM generator (Appendix B.1)."""

import copy
import json
import logging
from dataclasses import dataclass, field
from itertools import islice

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim import Adam
from tqdm import trange

from debias_anything.configs import DataloaderKind, MLflowConfig
from debias_anything.data import get_dataloader
from debias_anything.mlflow_utils import log_cfg_to_mlflow, set_experiment
from debias_anything.models.edm import (
    build_unet_for,
    edm_loss,
    parameters,
    sample_sigma,
)
from debias_anything.paths import CHECKPOINT_DIR, CONF_DIR

log = logging.getLogger(__name__)  # level


@dataclass
class Config:
    mlflow: MLflowConfig = field(
        default_factory=lambda: MLflowConfig(experiment_name="EDM")
    )  # creates an EDM experiment if it does not exist
    dataloader: DataloaderKind = DataloaderKind.celeba
    batch_size: int = 256
    epoch: int = 2000
    no_val: bool = False
    patience: int = 50
    min_delta: float = 0.0
    load_from_ckpt: str | None = None
    lr: float | None = 2e-4
    ckpt_every: int = 10
    min_lr: float = 1e-6


ConfigStore.instance().store(name="celeba_generator_schema", node=Config)


@torch.no_grad()
def evaluate(model, val_loader, device, seed=0):
    """Mean EDM loss on the validation split, with a fixed seed."""
    was_training = model.training
    model.eval()
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    total_loss, total_n = 0.0, 0
    for batch in val_loader:
        y = batch[0].to(device, non_blocking=True)
        sigmas = (
            sample_sigma(size=y.shape[0])
            .view(y.shape[0], *([1] * (y.dim() - 1)))
            .to(device)
        )
        loss = edm_loss(model, y, sigmas)
        total_loss += loss.item() * y.shape[0]
        total_n += y.shape[0]
    torch.set_rng_state(rng_state)
    if was_training:
        model.train()
    return total_loss / max(total_n, 1)


@hydra.main(
    version_base="1.3", config_path=str(CONF_DIR / "celeba"), config_name="generator"
)
def main(cfg: Config) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info(f"Using device: {device}")

    dataloader = get_dataloader(
        cfg.dataloader, batch_size=cfg.batch_size, split="train"
    )

    if cfg.no_val:
        val_loader = None
    else:
        val_loader = get_dataloader(
            cfg.dataloader, batch_size=cfg.batch_size, split="val"
        )
        log.info(
            f"Train: {len(dataloader.dataset)} samples | Val: {len(val_loader.dataset)} samples"
        )

    num_epochs = cfg.epoch
    sample_y = next(iter(dataloader))[0]
    data_shape = tuple(sample_y.shape[1:])

    in_channels = int(data_shape[0])
    model = build_unet_for(
        image_size=int(data_shape[-1]),
        in_channels=in_channels,
        out_channels=in_channels,
    ).to(device)
    ema_model = copy.deepcopy(model)
    ema_decay_max = 0.9999

    def ema_decay_at(step):
        return min(ema_decay_max, (1 + step) / (10 + step))

    lr = cfg.lr if cfg.lr is not None else 1e-4
    optimizer = Adam(model.parameters(), lr=lr)

    # Mixed precision: fp16 on CUDA (large throughput gain on CelebA-sized
    # UNets), cleanly disabled otherwise. The EDM loss is kept in fp32 on the
    # target side (cf. Karras 2024 EDM2) while staying inside autocast:
    # PyTorch keeps the sensitive operations (softmax, norms) in fp32.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    warmup_epochs = 10
    warmup_steps = warmup_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / 5, end_factor=1.0, total_iters=warmup_steps
    )

    losses = []
    lrs = []
    val_losses: list[float] = []

    best_val = float("inf")
    patience_counter = 0
    start_epoch = 0
    start_step = 0

    ckpt_dir = CHECKPOINT_DIR
    (ckpt_dir / "runs").mkdir(parents=True, exist_ok=True)
    ckpt_stem = f"diffusion_{cfg.dataloader}_{cfg.epoch}"
    ckpt_path = ckpt_dir / f"{ckpt_stem}.pth"
    retrain_path = ckpt_dir / "runs" / f"{ckpt_stem}_retrain.pt"
    mlflow_path = ckpt_dir / "runs" / f"{ckpt_stem}_mlflow.json"

    if cfg.load_from_ckpt is not None:
        ckpt = torch.load(cfg.load_from_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        # The checkpoint restores the original lr (optimizer.param_groups +
        # scheduler.base_lrs). If the user explicitly passes ``lr``, both
        # sources are overwritten so that the new value takes effect.
        if cfg.lr is not None:
            ckpt_lr = optimizer.param_groups[0]["lr"]
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr
                group["initial_lr"] = cfg.lr
            scheduler.base_lrs = [cfg.lr for _ in optimizer.param_groups]
            log.info(f"Override lr → {cfg.lr:.2e} (checkpoint: {ckpt_lr:.2e})")
        start_epoch = ckpt["epoch"] + 1
        start_step = ckpt["step"]
        best_val = ckpt["best_val"]
        log.info(
            f"Resumed from {cfg.load_from_ckpt} — start epoch={start_epoch}, "
            f"step={start_step}, best_val={best_val:.4f}"
        )

    saved_best_retrain = False

    set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.mlflow.run_name or f"EDM_{cfg.dataloader}_unet"):
        log_cfg_to_mlflow(cfg)
        mlflow.log_params(
            {
                "resolved_lr": optimizer.param_groups[0]["lr"],
                "warmup_epochs": warmup_epochs,
                "ema_decay_max": ema_decay_max,
                "in_channels": in_channels,
                "device": str(device),
                **parameters,
            }
        )

        step = start_step
        stopped_epoch = num_epochs
        for epoch in trange(
            start_epoch, num_epochs, initial=start_epoch, total=num_epochs
        ):
            epoch_losses = []
            for batch in islice(dataloader, len(dataloader)):
                y = batch[0].to(device, non_blocking=True)
                sigmas = (
                    sample_sigma(size=y.shape[0])
                    .view(y.shape[0], *([1] * (y.dim() - 1)))
                    .to(device)
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    loss = edm_loss(model, y, sigmas)
                scaler.scale(loss).backward()
                # Unscale before clipping so that ``max_norm`` is comparable to the
                # fp32 regime (otherwise the clip applies to gradients already
                # scaled by the scaler).
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                d = ema_decay_at(step)
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(d).add_(p.data, alpha=1 - d)

                epoch_losses.append(loss.item())
                lrs.append(scheduler.get_last_lr()[0])
                step += 1

            losses.extend(epoch_losses)
            epoch_mean_loss = np.mean(epoch_losses)
            mlflow.log_metrics(
                {
                    "epoch_loss": float(epoch_mean_loss),
                    "lr": float(scheduler.get_last_lr()[0]),
                },
                step=epoch,
            )

            stop_early = False
            if val_loader is not None:
                val_loss = evaluate(ema_model, val_loader, device)
                val_losses.append(val_loss)
                mlflow.log_metric("val_loss", float(val_loss), step=epoch)
                improved = val_loss < best_val - cfg.min_delta
                if improved:
                    best_val = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                log.info(
                    f"Epoch {epoch + 1}/{num_epochs}  |  loss: {epoch_mean_loss:.4f}  "
                    f"|  val: {val_loss:.4f}  |  best: {best_val:.4f}  "
                    f"|  patience: {patience_counter}/{cfg.patience}  "
                    f"|  lr: {scheduler.get_last_lr()[0]:.2e}"
                )
                if patience_counter >= cfg.patience:
                    current_lr = optimizer.param_groups[0]["lr"]
                    new_lr = current_lr / 2
                    for group in optimizer.param_groups:
                        group["lr"] = new_lr
                        group["initial_lr"] = new_lr
                    scheduler.base_lrs = [new_lr for _ in optimizer.param_groups]
                    patience_counter = 0
                    log.info(
                        f"Patience reached at epoch {epoch + 1} — lr {current_lr:.2e} → {new_lr:.2e}"
                    )
                    if new_lr < cfg.min_lr:
                        stopped_epoch = epoch + 1
                        stop_early = True
                        log.info(
                            f"Stopping: lr {new_lr:.2e} < min_lr {cfg.min_lr:.2e} after "
                            f"{stopped_epoch} epochs (best val={best_val:.4f})."
                        )
            else:
                log.info(
                    f"Epoch {epoch + 1}/{num_epochs}  |  loss: {epoch_mean_loss:.4f}  |  lr: {scheduler.get_last_lr()[0]:.2e}"
                )

            if (epoch + 1) % cfg.ckpt_every == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "ema_state_dict": ema_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "best_val": best_val,
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                    },
                    retrain_path,
                )
                saved_best_retrain = True

            if stop_early:
                break

        # No periodic checkpoint was written (fewer than ``ckpt_every`` epochs): snapshot
        # the final state so that a retrain checkpoint exists.
        if not saved_best_retrain:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": stopped_epoch - 1,
                    "step": step,
                    "best_val": best_val,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                retrain_path,
            )

        # Reload the EMA of the retrain checkpoint (the last periodic one, or the
        # final snapshot above) before writing the inference .pth.
        ckpt = torch.load(retrain_path, map_location=device, weights_only=False)
        ema_model.load_state_dict(ckpt["ema_state_dict"])

        mlflow.log_metrics(
            {"best_val_loss": float(best_val), "stopped_epoch": stopped_epoch}
        )

        torch.save(ema_model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path))
        mlflow.log_artifact(str(retrain_path))

        # Export of the MLflow run: run_id + URIs + params + final metrics.
        # Makes it possible to find the page even if the local DB is moved.
        run = mlflow.active_run()
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run.info.run_id)
        mlflow_export = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "artifact_uri": run.info.artifact_uri,
            "start_time": run.info.start_time,
            "params": dict(run_data.data.params),
            "metrics": dict(run_data.data.metrics),
            "tags": dict(run_data.data.tags),
        }
        with open(mlflow_path, "w") as f:
            json.dump(mlflow_export, f, indent=2, default=str)
        mlflow.log_artifact(str(mlflow_path))

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(f"EDM Loss – {cfg.dataloader}")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

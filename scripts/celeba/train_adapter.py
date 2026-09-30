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

"""Train the adapter of the CelebA 64×64 EDM model (h-space → SigLIP 2, Section 4.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import hydra
import mlflow
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import DataLoader, Subset

from debias_anything.adapters import HSpaceToSigLIP, cosine_loss
from debias_anything.configs import (
    DatasetConfig,
    MLflowConfig,
    register_dataset_configs,
)
from debias_anything.data import get_dataloader
from debias_anything.hspace import hspace_hook
from debias_anything.log import get_logger
from debias_anything.mlflow_utils import log_cfg_to_mlflow, set_experiment
from debias_anything.models.edm import Denoiser, build_unet_for, c_noise
from debias_anything.paths import CONF_DIR, REPO_ROOT, model_path
from debias_anything.siglip import encode_text, load_siglip, siglip_image

EVAL_SIGMAS = (
    0.09,
    0.30,
    1.0,
    3.32,
    50,
)  # ~ quantiles of the EDM log-normal (z=-1,0,1,2)

register_dataset_configs()


@dataclass
class Config:
    dataset: DatasetConfig = MISSING
    mlflow: MLflowConfig = field(
        default_factory=lambda: MLflowConfig(experiment_name="hspace_to_siglip")
    )
    checkpoint: str | None = None
    out: str = "checkpoints/celeba_adapter.pth"
    siglip: str = "google/siglip2-base-patch16-224"
    cache: str | None = None
    rebuild_cache: bool = False
    batch_size: int = 512
    m_views: int = 2
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    epochs: int = 50
    lr: float = 1e-3
    subset: int | None = None
    n_eval: int = 2000
    eval_every: int = 500
    log_every: int = 100


ConfigStore.instance().store(name="celeba_adapter_schema", node=Config)


def labels_and_prompts(base):
    """Gender labels and zero-shot prompts of CelebA, with prompt index = label."""
    male_idx = base.attr_names.index("Male")  # type: ignore[attr-defined]
    return (lambda attr: int(attr[male_idx])), (
        "a photo of a woman",
        "a photo of a man",
    )


logger = get_logger("hspace_to_siglip", log_file="hspace_to_siglip.json")

# Preprocessing of the SigLIP targets of the released adapter (see debias_anything.siglip).
TARGET_PREPROCESSING = dict(resize="bicubic", clamp=False)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_log_uniform(size: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """σ drawn log-uniformly on [σ_min, σ_max]."""
    log_s = torch.rand(size) * (math.log(sigma_max) - math.log(sigma_min)) + math.log(
        sigma_min
    )
    return log_s.exp()


# ---------------------------------------------------------------------------
# Cache of the SigLIP targets
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_cache(base, siglip, mean, std, size, device, batch_size, path):
    """Precompute and save SigLIP(x₀) for every image of ``base``, in order."""
    loader = DataLoader(base, batch_size=batch_size, shuffle=False, num_workers=8)
    embs = []
    for i, (img, _) in enumerate(loader):
        e = siglip_image(
            siglip, img.to(device), mean, std, size, **TARGET_PREPROCESSING
        )
        embs.append(e.cpu())
        if i % 50 == 0:
            logger.info(
                f"cache {i * batch_size}/{len(base)}",
                extra={"event": "cache", "done": i * batch_size, "total": len(base)},
            )
    cache = torch.cat(embs)
    torch.save(cache, path)
    logger.info(
        f"cache built {list(cache.shape)} -> {path}",
        extra={"event": "cache_done", "path": str(path), "shape": list(cache.shape)},
    )
    return cache


class CachedDataset(torch.utils.data.Dataset):
    """``(image in [-1, 1], cached SigLIP target, label)``."""

    def __init__(self, base, cache, label_of):
        self.base, self.cache, self.label_of = base, cache, label_of

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        img, target = self.base[i]
        return img, self.cache[i].float(), self.label_of(target)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    unet,
    denoiser,
    proj,
    siglip,
    text_emb,
    mean,
    std,
    size,
    val_base,
    label_of,
    device,
    n_eval,
    step,
    chunk=256,
):
    """Alignment, R@1 and zero-shot accuracy per σ on the validation split."""
    proj.eval()
    idx = list(range(min(n_eval, len(val_base))))
    img = torch.stack([val_base[i][0] for i in idx]).to(device)
    labels = torch.tensor([label_of(val_base[i][1]) for i in idx], device=device)
    bounds = range(0, len(idx), chunk)

    # SigLIP targets (bf16, diagnostic) — in chunks
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        tgt = torch.cat(
            [
                siglip_image(
                    siglip, img[s : s + chunk], mean, std, size, **TARGET_PREPROCESSING
                )
                for s in bounds
            ]
        ).float()
    # tgt shape (N,768) in fp32 for the gender metrics (alignment with the SigLIP targets)
    arange = torch.arange(len(idx), device=device)

    def class_acc(emb):
        pred = (emb @ text_emb.t()).argmax(-1)  # prompt index == class label
        return (pred == labels).float().mean().item()

    def proj_at(sg):  # proj(h_σ) for the whole set, computed in chunks
        zs = []
        for s in bounds:
            b = img[s : s + chunk]
            sigma = torch.full((b.shape[0], 1, 1, 1), sg, device=device)
            x_s = b + sigma * torch.randn_like(b)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with hspace_hook(unet) as cache:
                    denoiser(x_s, sigma)
                h = cache["h"]
            zs.append(proj(h.float(), c_noise(sigma).view(-1).float()))
        return torch.cat(zs)

    ceiling = class_acc(tgt)
    logger.info(
        f"[eval s{step}] ceiling class-acc={ceiling:.3f} (n={len(idx)})",
        extra={
            "event": "eval_ceiling",
            "step": step,
            "class_acc_ceiling": ceiling,
            "n": len(idx),
        },
    )
    mlflow.log_metric("eval/ceiling_class", ceiling, step=step)
    for sg in EVAL_SIGMAS:
        z = proj_at(sg)
        align = (z * tgt).sum(-1).mean().item()  # mean cos sim
        r1 = (
            ((z @ tgt.t()).argmax(-1) == arange).float().mean().item()
        )  # R@1: % of embeddings whose nearest neighbour in tgt is the matching target
        g = class_acc(z)
        logger.info(
            f"[eval s{step}] σ={sg:5.2f} align={align:.3f} R@1={r1:.3f} class={g:.3f}",
            extra={
                "event": "eval",
                "step": step,
                "sigma": sg,
                "align": align,
                "r_at_1": r1,
                "class_acc": g,
            },
        )
        mlflow.log_metrics(
            {f"align/sg{sg:g}": align, f"R1/sg{sg:g}": r1, f"class/sg{sg:g}": g},
            step=step,
        )
    proj.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # returns the evaluation peak memory to the OS


@torch.no_grad()
def probe_hspace(unet, denoiser, image_size, in_channels, device):
    """``(channels, tokens)`` of the h-space, from a dummy forward pass."""
    x = torch.zeros(1, in_channels, image_size, image_size, device=device)
    sigma = torch.full((1, 1, 1, 1), 0.5, device=device)
    with hspace_hook(unet) as cache:
        denoiser(x, sigma)
    h = cache["h"]
    return int(h.shape[1]), int(h.shape[2] * h.shape[3])


@hydra.main(
    version_base="1.3", config_path=str(CONF_DIR / "celeba"), config_name="adapter"
)
def main(cfg: Config) -> None:
    # --- logging config + device ------------------------------------------------------
    root = REPO_ROOT
    device = _device()
    checkpoint = cfg.checkpoint or cfg.dataset.checkpoint
    logger.info(
        "config",
        extra={"event": "config", "device": str(device), "checkpoint": checkpoint},
    )

    # --- frozen diffusion ---------------------------------------------------
    unet = build_unet_for(
        image_size=cfg.dataset.image_size, in_channels=cfg.dataset.in_channels
    ).to(device)
    unet.load_state_dict(torch.load(root / checkpoint, map_location=device))
    unet.requires_grad_(False)
    denoiser = Denoiser(unet).eval()

    # --- frozen SigLIP ------------------------------------------------------
    siglip_path = model_path(cfg.siglip)
    siglip, proc, sg_mean, sg_std, sg_size = load_siglip(device, siglip_path)
    # print('mean', sg_mean.cpu().numpy(), 'std', sg_std.cpu().numpy(), 'size', sg_size)

    # --- datasets + cache ---------------------------------------------------
    train_base = get_dataloader(cfg.dataset.name, split="train").dataset
    val_base = get_dataloader(cfg.dataset.name, split="val").dataset
    label_of, prompts = labels_and_prompts(train_base)
    text_emb = encode_text(siglip, proc, list(prompts), device)
    if cfg.subset is not None:
        train_base = Subset(train_base, range(cfg.subset))

    # get the SigLIP embeddings of all the images of the train set
    cache_path = root / (cfg.cache or f"data/siglip_cache_{cfg.dataset.name}_train.pt")
    if cfg.subset is not None:
        cache_path = cache_path.with_name(f"{cache_path.stem}_sub{cfg.subset}.pt")
    if cache_path.exists() and not cfg.rebuild_cache:
        cache = torch.load(cache_path)
        logger.info(
            f"cache loaded {list(cache.shape)} <- {cache_path}",
            extra={
                "event": "cache_loaded",
                "path": str(cache_path),
                "shape": list(cache.shape),
            },
        )
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = build_cache(
            train_base,
            siglip,
            sg_mean,
            sg_std,
            sg_size,
            device,
            max(cfg.batch_size, 128),
            cache_path,
        )

    dataset = CachedDataset(train_base, cache, label_of)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, drop_last=True
    )

    # --- adapter (trained) -------------------------------------------------
    h_channels, num_tokens = probe_hspace(
        unet, denoiser, cfg.dataset.image_size, cfg.dataset.in_channels, device
    )
    logger.info(
        f"h-space probed: {h_channels} channels, {num_tokens} tokens",
        extra={
            "event": "hspace_probe",
            "h_channels": h_channels,
            "num_tokens": num_tokens,
        },
    )
    proj = HSpaceToSigLIP(
        h_channels=h_channels, num_tokens=num_tokens, out_dim=text_emb.shape[-1]
    ).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=cfg.lr)

    set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(
        run_name=cfg.mlflow.run_name or f"hspace2siglip_{cfg.dataset.name}"
    ):
        log_cfg_to_mlflow(cfg)

        step = 0
        for epoch in range(cfg.epochs):
            for img, tgt, _male in loader:
                img, tgt = (
                    img.to(device),
                    tgt.to(device),
                )  # img: (B,3,64,64) in [-1,1], tgt: (B,768) L2-normalized SigLIP(x₀) in fp32
                B = img.shape[0]
                # repeat each batch m_views times for noise invariance (m_views = number of σ drawn independently per image)
                x0 = img.repeat(cfg.m_views, 1, 1, 1)  # (A,3,64,64), A=B*m
                src_idx = torch.arange(B, device=device).repeat(
                    cfg.m_views
                )  # (A,) index of the positive target of each anchor
                sigma = (
                    sample_log_uniform(x0.shape[0], cfg.sigma_min, cfg.sigma_max)
                    .to(device)
                    .view(-1, 1, 1, 1)
                )
                x_sigma = x0 + sigma * torch.randn_like(
                    x0
                )  # direct EDM noising (no denoising chain, a single forward pass)

                # bf16: this is the precision at which sampling reads this same h-space
                # (sampling runs under bf16 autocast), and that of ``evaluate`` — all three
                # must see the same thing. It also halves the peak of UNet activations,
                # which dominates memory at A = batch_size × m_views.
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16),
                ):
                    with hspace_hook(unet) as cache_h:
                        denoiser(x_sigma, sigma)
                    h = cache_h["h"]  # (A,192,8,8) in bf16

                # projection + loss (the SigLIP targets are L2-normalized; the adapter already
                # L2-normalizes its output)
                # ``.float()``: the adapter and the loss stay in fp32, as in ``evaluate``.
                z = proj(h.float(), c_noise(sigma).view(-1))  # (A,768) in fp32
                loss, pos_sim, neg_sim = cosine_loss(z, tgt, src_idx)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if step % cfg.log_every == 0:
                    logger.info(
                        f"epoch {epoch} step {step} | loss {loss.item():.4f} "
                        f"| cos+ {pos_sim:.3f} cos- {neg_sim:.3f}",
                        extra={
                            "event": "train",
                            "epoch": epoch,
                            "step": step,
                            "loss": loss.item(),
                            "cos_pos": pos_sim,
                            "cos_neg": neg_sim,
                        },
                    )
                    mlflow.log_metrics(
                        {
                            "train/loss": loss.item(),
                            "train/cos_pos": pos_sim,
                            "train/cos_neg": neg_sim,
                        },
                        step=step,
                    )
                if step > 0 and step % cfg.eval_every == 0:
                    evaluate(
                        unet,
                        denoiser,
                        proj,
                        siglip,
                        text_emb,
                        sg_mean,
                        sg_std,
                        sg_size,
                        val_base,
                        label_of,
                        device,
                        cfg.n_eval,
                        step,
                    )
                step += 1

        evaluate(
            unet,
            denoiser,
            proj,
            siglip,
            text_emb,
            sg_mean,
            sg_std,
            sg_size,
            val_base,
            label_of,
            device,
            cfg.n_eval,
            step,
        )
        out = root / cfg.out
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(proj.state_dict(), out)
        logger.info(
            f"saved adapter -> {out}", extra={"event": "saved", "path": str(out)}
        )
        mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()

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

"""Train a CelebA-HQ ResNet-18 evaluator (App. B.5); race labels are a CLIP proxy."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18

from debias_anything.evaluators.celebahq import (
    ATTRIBUTE_CONFIG,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageFiles,
    LabelledImages,
    annotation_examples,
    classification_metrics,
    classifier_transform,
    file_sha256,
    image_paths,
    inventory_sha256,
    ranked_race_examples,
    stratified_split,
)
from debias_anything.paths import CONF_DIR, portable, resolve


@dataclass
class Config:
    attribute: str = "???"  # gender, eyeglasses or race
    images_dir: str = "data/celebahq/CelebAMask-HQ/CelebA-HQ-img"
    annotations: str | None = (
        None  # required for gender and eyeglasses, ignored for race
    )
    output_dir: str = "???"
    seed: int = 42
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    workers: int = 4
    device: str = "cuda"
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    clip_model: str = "ViT-B/32"
    race_prompt: str = "a black person"
    clip_batch_size: int = 128
    scores_cache: str | None = None
    imagenet_pretrained: bool = True
    amp: bool = False
    overwrite: bool = False


ConfigStore.instance().store(name="p2_evaluator_schema", node=Config)


def validate(args: Config) -> None:
    if not resolve(args.images_dir).is_dir():
        raise FileNotFoundError(args.images_dir)
    if args.attribute != "race" and (
        args.annotations is None or not resolve(args.annotations).is_file()
    ):
        raise FileNotFoundError("--annotations is required for gender and eyeglasses")
    if args.epochs < 1 or args.batch_size < 1 or args.clip_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.workers < 0:
        raise ValueError("invalid optimiser settings")
    if not 0 < args.train_fraction < 1 or not 0 < args.val_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def race_scores(
    paths: list[Path],
    images_dir: Path,
    inventory_sha256: str,
    args: Config,
    device: torch.device,
    cache_path: Path,
) -> tuple[torch.Tensor, Path | None]:
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected_paths = [path.relative_to(images_dir).as_posix() for path in paths]
        if (
            isinstance(cached, dict)
            and cached.get("protocol") == "balancing_act_clip_race_scores_v2"
            and cached.get("inventory_sha256") == inventory_sha256
            and cached.get("prompt") == args.race_prompt
            and cached.get("clip_model") == args.clip_model
            and cached.get("relative_paths") == expected_paths
            and isinstance(cached.get("scores"), torch.Tensor)
            and cached["scores"].shape == (len(paths),)
        ):
            print(f"Reusing validated CLIP scores from {cache_path}", flush=True)
            cached_weights = cached.get("clip_weights")
            weights = (
                Path(cached_weights)
                if cached_weights and Path(cached_weights).is_file()
                else None
            )
            return cached["scores"].float(), weights
        raise ValueError(
            f"CLIP cache does not match this race-labeling run: {cache_path}"
        )

    import clip
    import clip.clip as openai_clip_impl
    import torch.nn.functional as functional

    model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    model.eval()
    loader = DataLoader(
        ImageFiles(images_dir, preprocess),
        batch_size=args.clip_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    text = functional.normalize(
        model.encode_text(clip.tokenize([args.race_prompt]).to(device)).float(), dim=-1
    )[0]
    chunks = []
    for images in loader:
        features = functional.normalize(
            model.encode_image(images.to(device, non_blocking=True)).float(), dim=-1
        )
        chunks.append((features @ text).cpu())
    scores = torch.cat(chunks).float()
    clip_url = openai_clip_impl._MODELS.get(args.clip_model)
    weights = (
        Path.home() / ".cache" / "clip" / Path(urlparse(clip_url).path).name
        if clip_url
        else None
    )
    if weights is not None and not weights.is_file():
        weights = None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": "balancing_act_clip_race_scores_v2",
            "inventory_sha256": inventory_sha256,
            "prompt": args.race_prompt,
            "clip_model": args.clip_model,
            "relative_paths": [
                path.relative_to(images_dir).as_posix() for path in paths
            ],
            "scores": scores,
            "clip_weights": str(weights) if weights else None,
            "clip_weights_sha256": file_sha256(weights) if weights else None,
        },
        cache_path,
    )
    return scores, weights


def make_loader(
    dataset: LabelledImages,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
    balanced: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if balanced:
        labels = torch.tensor([example.label for example in dataset.examples])
        counts = torch.bincount(labels, minlength=2).float()
        sampler = WeightedRandomSampler(
            (1.0 / counts[labels]).tolist(),
            len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not balanced,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )


def epoch_pass(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    amp: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, labels in loader:
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            predictions = logits.detach().argmax(1)
            total_loss += float(loss.detach()) * labels.numel()
            confusion += torch.bincount(
                (labels.detach().cpu() * 2 + predictions.cpu()).long(), minlength=4
            ).reshape(2, 2)
    metrics = classification_metrics(confusion)
    metrics["loss"] = total_loss / max(metrics["num_examples"], 1)
    return metrics


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


@hydra.main(
    version_base="1.3", config_path=str(CONF_DIR / "p2"), config_name="evaluator"
)
def main(cfg: Config) -> None:
    args = cfg
    validate(args)
    images_dir = resolve(args.images_dir)
    annotations = resolve(args.annotations) if args.annotations else None
    output_dir = resolve(args.output_dir)
    set_reproducibility(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_files = [
        output_dir / name
        for name in ("best.pt", "last.pt", "manifest.json", "history.json")
    ]
    if any(path.exists() for path in output_files) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output in {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(images_dir)
    source_inventory = inventory_sha256(paths, images_dir)
    clip_weights = None
    if args.attribute == "race":
        scores, clip_weights = race_scores(
            paths,
            images_dir,
            source_inventory,
            args,
            device,
            resolve(args.scores_cache)
            if args.scores_cache
            else output_dir / "clip_scores.pt",
        )
        examples = ranked_race_examples(paths, images_dir, scores)
    else:
        assert annotations is not None
        examples = annotation_examples(paths, images_dir, annotations, args.attribute)
    splits = stratified_split(
        examples, args.train_fraction, args.val_fraction, args.seed
    )

    config = ATTRIBUTE_CONFIG[args.attribute]
    manifest: dict[str, Any] = {
        "protocol": "balancing_act_image_evaluator_v1",
        "attribute": args.attribute,
        "class_names": list(config["class_names"]),
        "label_source": config["label_source"],
        "source_images_dir": portable(images_dir),
        "source_inventory_sha256": source_inventory,
        "num_source_images": len(paths),
        "annotations": portable(annotations) if annotations else None,
        "split": {
            "method": "class-stratified seeded shuffle",
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.val_fraction,
            "test_fraction": 1 - args.train_fraction - args.val_fraction,
            "counts": {
                name: {
                    "total": len(items),
                    "per_class": [
                        sum(item.label == label for item in items) for label in (0, 1)
                    ],
                }
                for name, items in splits.items()
            },
            "examples": {
                name: [asdict(item) for item in items] for name, items in splits.items()
            },
        },
        "model": {
            "architecture": "torchvision.models.resnet18",
            "num_classes": 2,
            "imagenet_pretrained": args.imagenet_pretrained,
        },
        "preprocessing": {
            "resize_short_side": 256,
            "center_crop": 224,
            "random_horizontal_flip_train": False,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "evaluation_command_value": "--preprocess imagenet",
        },
        "optimization": {
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "balanced_train_sampler": True,
            "amp": args.amp and device.type == "cuda",
        },
    }
    if args.attribute == "race":
        midpoint = len(paths) // 2
        assert scores is not None
        ranked = sorted(scores.tolist())
        manifest["race_proxy"] = {
            "prompt": args.race_prompt,
            "clip_model": args.clip_model,
            "clip_weights": clip_weights.name if clip_weights else None,
            "clip_weights_sha256": file_sha256(clip_weights) if clip_weights else None,
            "selection": {
                "class_0": f"{midpoint} lowest similarities",
                "class_1": f"{midpoint} highest similarities",
            },
            "score_boundary": {
                "low_max": ranked[midpoint - 1],
                "high_min": ranked[midpoint],
            },
            "caveat": "Binary CLIP proxy only; it is not a demographic ground-truth race annotation.",
        }
    json_dump(output_dir / "manifest.json", manifest)

    transform = classifier_transform("imagenet")
    datasets = {
        name: LabelledImages(images_dir, items, transform)
        for name, items in splits.items()
    }
    loaders = {
        name: make_loader(
            dataset,
            args.batch_size,
            args.workers,
            args.seed + offset,
            device,
            balanced=name == "train",
        )
        for offset, (name, dataset) in enumerate(datasets.items())
    }
    weights = ResNet18_Weights.IMAGENET1K_V1 if args.imagenet_pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        device.type, enabled=args.amp and device.type == "cuda"
    )
    history = []
    best_val, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        train = epoch_pass(
            model, loaders["train"], criterion, device, optimizer, args.amp, scaler
        )
        validation = epoch_pass(
            model, loaders["validation"], criterion, device, None, args.amp
        )
        history.append({"epoch": epoch, "train": train, "validation": validation})
        print(
            f"epoch {epoch:02d}/{args.epochs}: train bal_acc={train['balanced_accuracy']:.4f}; val bal_acc={validation['balanced_accuracy']:.4f}",
            flush=True,
        )
        if validation["balanced_accuracy"] > best_val:
            best_val, best_epoch, best_state = (
                validation["balanced_accuracy"],
                epoch,
                cpu_state_dict(model),
            )
    assert best_state is not None
    last_state = cpu_state_dict(model)
    model.load_state_dict(best_state)
    test = epoch_pass(model, loaders["test"], criterion, device, None, args.amp)
    result = {
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_val,
        "test": test,
        "history": history,
    }
    json_dump(output_dir / "history.json", result)
    checkpoint = {
        "protocol": manifest["protocol"],
        "class_names": manifest["class_names"],
        "manifest": manifest,
        "training_result": result,
    }
    torch.save(
        {**checkpoint, "epoch": best_epoch, "state_dict": best_state},
        output_dir / "best.pt",
    )
    torch.save(
        {**checkpoint, "epoch": args.epochs, "state_dict": last_state},
        output_dir / "last.pt",
    )
    print(
        json.dumps(
            {"attribute": args.attribute, "best_epoch": best_epoch, "test": test},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

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

"""CelebA-HQ evaluators of the Balancing Act protocol; race labels are a CLIP proxy."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet34

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def image_paths(directory: Path) -> list[Path]:
    """Every image file under ``directory``, sorted."""
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No PNG/JPEG/WebP images found under {directory}")
    return paths


def inventory_sha256(paths: list[Path], root: Path) -> str:
    """Fingerprint of image files (relative names, sizes, modification times)."""
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def image_inventory(directory: Path) -> dict[str, Any]:
    """Fingerprint of an image directory."""
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return {
        "num_images": len(paths),
        "sha256_names_sizes_mtimes": inventory_sha256(paths, directory),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ImageFiles(Dataset[torch.Tensor]):
    """The images of a directory, transformed, without labels."""

    def __init__(
        self, directory: Path, transform: Callable[[Image.Image], torch.Tensor]
    ):
        self.paths = image_paths(directory)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


# ----------------------------------------------------------------------------- labels and splits

ATTRIBUTE_CONFIG = {
    "gender": {
        "annotation": "Male",
        "class_names": ("Female", "Male"),
        "label_source": "CelebAMask-HQ annotation Male",
    },
    "eyeglasses": {
        "annotation": "Eyeglasses",
        "class_names": ("No eyeglasses", "Eyeglasses"),
        "label_source": "CelebAMask-HQ annotation Eyeglasses",
    },
    "race": {
        "annotation": None,
        "class_names": ("low_CLIP_similarity", "high_CLIP_similarity"),
        "label_source": "OpenAI CLIP cosine-similarity rank",
    },
}


@dataclass(frozen=True)
class Example:
    relative_path: str
    label: int
    clip_similarity: float | None = None


class LabelledImages(Dataset[tuple[torch.Tensor, int]]):
    """``(transform(image), label)`` for the examples of a split."""

    def __init__(
        self,
        images_dir: Path,
        examples: list[Example],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.images_dir = images_dir
        self.examples = examples
        self.transform = transform
        if not examples or any(
            not (images_dir / example.relative_path).is_file() for example in examples
        ):
            raise FileNotFoundError(f"Missing images under {images_dir}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        example = self.examples[index]
        with Image.open(self.images_dir / example.relative_path) as image:
            return self.transform(image.convert("RGB")), example.label


def annotation_examples(
    paths: list[Path], images_dir: Path, annotations: Path, attribute: str
) -> list[Example]:
    lines = annotations.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Malformed annotation file: {annotations}")
    expected = int(lines[0].strip())
    names = lines[1].split()
    annotation_name = ATTRIBUTE_CONFIG[attribute]["annotation"]
    assert isinstance(annotation_name, str)
    try:
        index = names.index(annotation_name)
    except ValueError as error:
        raise ValueError(
            f"Annotation {annotation_name!r} is absent from {annotations}"
        ) from error
    labels = {fields[0]: int(fields[index + 1]) for fields in map(str.split, lines[2:])}
    if len(labels) != expected:
        raise ValueError("Annotation row count does not match its header")
    examples = []
    for path in paths:
        name = path.name
        if name not in labels:
            raise ValueError(f"Missing annotation for {name}")
        if labels[name] not in (-1, 1):
            raise ValueError(f"Unexpected annotation value for {name}: {labels[name]}")
        examples.append(
            Example(path.relative_to(images_dir).as_posix(), int(labels[name] == 1))
        )
    return examples


def ranked_race_examples(
    paths: list[Path], images_dir: Path, scores: torch.Tensor
) -> list[Example]:
    if scores.shape != (len(paths),):
        raise ValueError("Race scores must align one-to-one with image paths")
    ranked = sorted(
        zip(scores.tolist(), paths), key=lambda item: (item[0], item[1].name)
    )
    midpoint = len(ranked) // 2
    if len(ranked) % 2:
        raise ValueError("Race proxy needs an even number of images for equal classes")
    return [
        Example(path.relative_to(images_dir).as_posix(), int(index >= midpoint), score)
        for index, (score, path) in enumerate(ranked)
    ]


def stratified_split(
    examples: list[Example], train_fraction: float, val_fraction: float, seed: int
) -> dict[str, list[Example]]:
    rng = random.Random(seed)
    splits: dict[str, list[Example]] = {"train": [], "validation": [], "test": []}
    for label in (0, 1):
        items = [example for example in examples if example.label == label]
        rng.shuffle(items)
        train_count = math.floor(len(items) * train_fraction)
        val_count = math.floor(len(items) * val_fraction)
        if min(train_count, val_count, len(items) - train_count - val_count) < 1:
            raise ValueError("Each class needs one example in every split")
        splits["train"].extend(items[:train_count])
        splits["validation"].extend(items[train_count : train_count + val_count])
        splits["test"].extend(items[train_count + val_count :])
    for items in splits.values():
        rng.shuffle(items)
    return splits


def classification_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    total = int(confusion.sum())
    recalls = confusion.diag().float() / confusion.sum(1).clamp(min=1)
    precision = confusion.diag().float() / confusion.sum(0).clamp(min=1)
    f1 = 2 * precision * recalls / (precision + recalls).clamp(min=1e-12)
    return {
        "accuracy": float(confusion.diag().sum() / max(total, 1)),
        "balanced_accuracy": float(recalls.mean()),
        "f1_macro": float(f1.mean()),
        "recall_per_class": recalls.tolist(),
        "confusion_true_rows_pred_columns": confusion.tolist(),
        "num_examples": total,
    }


# ----------------------------------------------------------------------------- classifiers and metrics


def classifier_transform(name: str) -> transforms.Compose:
    if name == "imagenet":
        # The default torchvision ResNet preprocessing: resize the shorter
        # side to 256, center crop to 224, then ImageNet normalisation.
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                IMAGENET_NORMALIZE,
            ]
        )
    if name == "zero_one":
        return transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
    if name == "fairface":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                IMAGENET_NORMALIZE,
            ]
        )
    raise ValueError(f"Unknown preprocessing: {name}")


def state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict) or not all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        raise ValueError(
            "Unsupported classifier checkpoint. Expected a ResNet-18 state_dict "
            "or a dict containing `state_dict`."
        )

    # Training with DataParallel commonly prefixes every key with `module.`.
    return {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in checkpoint.items()
    }


class FairFaceWhiteBlackClassifier(nn.Module):
    """Binary White/Black view of the 7-race FairFace classifier."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # FairFace output order begins with White, Black.  Restricting the
        # logits before softmax gives the binary race setting used here.
        return self.network(images)[:, :2]


def load_classifier(
    checkpoint_path: Path,
    num_classes: int,
    architecture: str,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if architecture == "fairface_white_black":
        if num_classes != 2:
            raise ValueError("The FairFace White/Black view has two classes.")
        network = resnet34(weights=None)
        network.fc = nn.Linear(network.fc.in_features, 18)
        network.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
        model = FairFaceWhiteBlackClassifier(network)
        metadata = {
            "classifier_class_order": ["White", "Black"],
            "classifier_note": (
                "Public FairFace proxy; not Balancing Act's unreleased "
                "CelebA-HQ ResNet-18 evaluator."
            ),
        }
        return model.to(device).eval(), metadata

    model = resnet18(weights=None, num_classes=num_classes)
    state_dict = state_dict_from_checkpoint(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The evaluation checkpoint is not a torchvision ResNet-18 with "
            f"{num_classes} output classes. Set --num-classes correctly or "
            "export the evaluation classifier as a torchvision ResNet-18 state_dict."
        ) from error
    metadata: dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        class_names = checkpoint.get("class_names")
        manifest = checkpoint.get("manifest")
        if isinstance(class_names, list) and all(
            isinstance(name, str) for name in class_names
        ):
            metadata["classifier_class_order"] = class_names
        if isinstance(manifest, dict):
            metadata["classifier_attribute"] = manifest.get("attribute")
            metadata["classifier_label_source"] = manifest.get("label_source")
            race_proxy = manifest.get("race_proxy")
            if isinstance(race_proxy, dict):
                metadata["classifier_race_proxy_caveat"] = race_proxy.get("caveat")
    return model.to(device).eval(), metadata


@torch.inference_mode()
def fairness_discrepancy(
    model: nn.Module,
    images_dir: Path,
    transform: transforms.Compose,
    target_probs: list[float],
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset = ImageFiles(images_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    probability_sum = torch.zeros(len(target_probs), dtype=torch.float64)
    hard_counts = torch.zeros(len(target_probs), dtype=torch.int64)
    for images in loader:
        logits = model(images.to(device, non_blocking=True))
        if logits.ndim != 2 or logits.shape[1] != len(target_probs):
            raise RuntimeError(
                "Classifier output shape is incompatible with --target-probs: "
                f"got {tuple(logits.shape)}, expected (_, {len(target_probs)})."
            )
        probabilities = logits.softmax(dim=1).cpu()
        probability_sum += probabilities.sum(dim=0, dtype=torch.float64)
        hard_counts += torch.bincount(
            probabilities.argmax(dim=1), minlength=len(target_probs)
        )

    mean_probabilities = probability_sum / len(dataset)
    target = torch.tensor(target_probs, dtype=torch.float64)
    return {
        "num_images": len(dataset),
        "target_probs": target_probs,
        "mean_softmax_probs": mean_probabilities.tolist(),
        "hard_counts": hard_counts.tolist(),
        "hard_probs": (hard_counts.double() / len(dataset)).tolist(),
        "fairness_discrepancy_l2": torch.linalg.vector_norm(
            mean_probabilities - target, ord=2
        ).item(),
    }


def compute_fid(
    reference_dir: Path,
    images_dir: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> float:
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as error:
        raise RuntimeError(
            "pytorch-fid is required for FID. Install it with "
            "`python -m pip install pytorch-fid`."
        ) from error

    return float(
        calculate_fid_given_paths(
            [str(reference_dir), str(images_dir)],
            batch_size=batch_size,
            device=device,
            dims=2048,
            num_workers=workers,
        )
    )

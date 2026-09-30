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

"""Per-class recalls of the CelebA-HQ evaluators on their validation images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from debias_anything.evaluators.celebahq import (
    Example,
    LabelledImages,
    classification_metrics,
    classifier_transform,
    file_sha256,
    load_classifier,
)
from debias_anything.paths import CHECKPOINT_DIR, DATA_ROOT, portable

EVALUATORS = CHECKPOINT_DIR / "celebahq_evaluators"


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if not isinstance(
        manifest.get("split", {}).get("examples", {}).get("validation"), list
    ):
        raise ValueError(f"Manifest has no validation examples: {path}")
    return manifest


def validation_examples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return manifest["split"]["examples"]["validation"]


def race_validation_extremes(
    manifest: dict[str, Any], per_class: int
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    ranked = sorted(
        validation_examples(manifest),
        key=lambda item: (float(item["clip_similarity"]), item["relative_path"]),
    )
    if per_class < 1 or 2 * per_class > len(ranked):
        raise ValueError("Race extremes must fit in the validation split")
    low = [{**item, "label": 0} for item in ranked[:per_class]]
    high = [{**item, "label": 1} for item in ranked[-per_class:]]
    return low + high, {
        "low_similarity_max": float(low[-1]["clip_similarity"]),
        "high_similarity_min": float(high[0]["clip_similarity"]),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: LabelledImages,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    probability_sum = torch.zeros(2, dtype=torch.float64)
    for images, labels in loader:
        probabilities = model(images.to(device, non_blocking=True)).softmax(1).cpu()
        predictions = probabilities.argmax(1)
        probability_sum += probabilities.sum(0, dtype=torch.float64)
        confusion += torch.bincount(
            (labels * 2 + predictions).long(), minlength=4
        ).reshape(2, 2)
    metrics = classification_metrics(confusion)
    total = metrics.pop("num_examples")
    return {
        "num_images": total,
        **metrics,
        "mean_softmax_probs": (probability_sum / total).tolist(),
        "hard_probs": (confusion.sum(0).double() / total).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--gender-checkpoint", type=Path, default=EVALUATORS / "gender/best.pt"
    )
    parser.add_argument(
        "--eyeglasses-checkpoint",
        type=Path,
        default=EVALUATORS / "eyeglasses/best.pt",
    )
    parser.add_argument(
        "--fairface-checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "fairface/res34_fair_align_multi_7_20190809.pt",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DATA_ROOT / "celebahq/CelebAMask-HQ/CelebA-HQ-img",
    )
    parser.add_argument(
        "--race-manifest", type=Path, default=EVALUATORS / "race/manifest.json"
    )
    parser.add_argument("--race-extremes-per-class", type=int, default=200)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=EVALUATORS / "recalls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers non-negative")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    gender_manifest_path = args.gender_checkpoint.parent / "manifest.json"
    eyeglasses_manifest_path = args.eyeglasses_checkpoint.parent / "manifest.json"
    gender_manifest = load_manifest(gender_manifest_path)
    eyeglasses_manifest = load_manifest(eyeglasses_manifest_path)
    race_manifest = load_manifest(args.race_manifest)
    race_examples, race_boundaries = race_validation_extremes(
        race_manifest, args.race_extremes_per_class
    )

    configurations = {
        "gender": {
            "checkpoint": args.gender_checkpoint,
            "manifest": gender_manifest_path,
            "examples": validation_examples(gender_manifest),
            "architecture": "resnet18",
            "preprocess": "imagenet",
            "class_names": ["Female", "Male"],
            "label_source": "CelebA-HQ Male annotation",
            "interpretation": (
                "Accuracy on the saved CelebA-HQ validation split used for "
                "checkpoint selection; the test split remains the held-out metric."
            ),
        },
        "eyeglasses": {
            "checkpoint": args.eyeglasses_checkpoint,
            "manifest": eyeglasses_manifest_path,
            "examples": validation_examples(eyeglasses_manifest),
            "architecture": "resnet18",
            "preprocess": "imagenet",
            "class_names": ["No eyeglasses", "Eyeglasses"],
            "label_source": "CelebA-HQ Eyeglasses annotation",
            "interpretation": (
                "Accuracy on the saved CelebA-HQ validation split used for "
                "checkpoint selection; the test split remains the held-out metric."
            ),
        },
        "race": {
            "checkpoint": args.fairface_checkpoint,
            "manifest": args.race_manifest,
            "examples": race_examples,
            "architecture": "fairface_white_black",
            "preprocess": "fairface",
            "class_names": ["White", "Black"],
            "label_source": (
                "200 lowest and 200 highest OpenAI CLIP ViT-B/32 similarities "
                "to 'a black person' within the saved Race validation split"
            ),
            "interpretation": (
                "FairFace agreement on validation-set CLIP extremes; not "
                "demographic ground-truth race accuracy."
            ),
            "clip_score_boundaries": race_boundaries,
        },
    }

    report: dict[str, Any] = {
        "protocol": "celebahq_validation_classifier_evaluation_v1",
        "caveat": (
            "Gender and Eyeglasses use the validation splits that selected their "
            "best checkpoints, not the held-out test splits. Race measures "
            "FairFace agreement on 200 low and 200 high CLIP-similarity validation "
            "images and is not demographic ground-truth accuracy."
        ),
        "classifiers": {},
    }
    rows = []
    for attribute, config in configurations.items():
        checkpoint = config["checkpoint"]
        manifest_path = config["manifest"]
        if not checkpoint.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint/manifest: {checkpoint}, {manifest_path}"
            )
        model, metadata = load_classifier(checkpoint, 2, config["architecture"], device)
        dataset = LabelledImages(
            args.images_dir,
            [Example(**item) for item in config["examples"]],
            classifier_transform(config["preprocess"]),
        )
        metrics = evaluate(model, dataset, args.batch_size, args.workers, device)
        result = {
            "images_dir": portable(args.images_dir),
            "split": "validation",
            "split_manifest": portable(manifest_path),
            "checkpoint": portable(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "architecture": config["architecture"],
            "preprocess": config["preprocess"],
            "class_names": config["class_names"],
            "label_source": config["label_source"],
            "interpretation": config["interpretation"],
            **metadata,
            **metrics,
        }
        if "clip_score_boundaries" in config:
            result["clip_score_boundaries"] = config["clip_score_boundaries"]
        report["classifiers"][attribute] = result
        rows.append(
            {
                "attribute": attribute,
                "num_images": metrics["num_images"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1_macro": metrics["f1_macro"],
                "recall_class_0": metrics["recall_per_class"][0],
                "recall_class_1": metrics["recall_per_class"][1],
                "class_0": config["class_names"][0],
                "class_1": config["class_names"][1],
                "interpretation": config["interpretation"],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {prefix}.json and {prefix}.csv")


if __name__ == "__main__":
    main()

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

"""FID and FD of a folder of P2 images, with the protocol of Balancing Act."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from debias_anything.evaluators.celebahq import (
    classifier_transform,
    compute_fid,
    fairness_discrepancy,
    file_sha256,
    image_inventory,
    load_classifier,
)
from debias_anything.paths import portable


def parse_probabilities(value: str) -> list[float]:
    probabilities = [float(item) for item in value.split(",")]
    if not probabilities or any(item < 0 for item in probabilities):
        raise argparse.ArgumentTypeError("Target probabilities must be non-negative.")
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(
            f"Target probabilities must sum to one, got {sum(probabilities):.8f}."
        )
    return probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--fid-reference-dir",
        type=Path,
        required=True,
        help="Real CelebA-HQ D_ref directory, balanced for the evaluated attribute.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        required=True,
        help="Independent image-space torchvision ResNet-18 evaluation checkpoint.",
    )
    parser.add_argument(
        "--classifier-architecture",
        choices=("resnet18", "fairface_white_black"),
        default="resnet18",
    )
    parser.add_argument(
        "--attribute", required=True, help="Metadata only, e.g. eyeglasses."
    )
    parser.add_argument(
        "--target-probs",
        type=parse_probabilities,
        default="0.5,0.5",
        help="Reference class distribution in the classifier's class order.",
    )
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--preprocess",
        choices=("imagenet", "zero_one", "fairface"),
        default="imagenet",
        help="Must match the preprocessing used to train the evaluation classifier.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output path (default: <images-dir>/balancing_act_metrics.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_classes != len(args.target_probs):
        raise ValueError("--num-classes must equal the number of --target-probs.")
    if args.batch_size < 1 or args.fid_batch_size < 1 or args.workers < 0:
        raise ValueError("Batch sizes must be positive and --workers non-negative.")
    if not args.images_dir.is_dir() or not args.fid_reference_dir.is_dir():
        raise FileNotFoundError(
            "--images-dir and --fid-reference-dir must be directories."
        )
    if not args.classifier_checkpoint.is_file():
        raise FileNotFoundError(args.classifier_checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")

    inventory_before = image_inventory(args.images_dir)
    model, classifier_metadata = load_classifier(
        args.classifier_checkpoint,
        args.num_classes,
        args.classifier_architecture,
        device,
    )
    fd_metrics = fairness_discrepancy(
        model=model,
        images_dir=args.images_dir,
        transform=classifier_transform(args.preprocess),
        target_probs=args.target_probs,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    fid = compute_fid(
        reference_dir=args.fid_reference_dir,
        images_dir=args.images_dir,
        batch_size=args.fid_batch_size,
        workers=args.workers,
        device=device,
    )
    inventory_after = image_inventory(args.images_dir)
    if inventory_after != inventory_before:
        raise RuntimeError(
            "The generated-image directory changed during evaluation. Evaluate a "
            "finished run or an immutable snapshot; no result was written."
        )

    reference_manifest_path = args.fid_reference_dir / "manifest.json"
    reference_manifest = (
        json.loads(reference_manifest_path.read_text())
        if reference_manifest_path.is_file()
        else None
    )
    result = {
        "protocol": (
            "balancing_act_fid_fd_v1"
            if args.classifier_architecture == "resnet18"
            else "balancing_act_compatible_public_classifier_v1"
        ),
        "attribute": args.attribute,
        "images_dir": portable(args.images_dir),
        "image_inventory": inventory_before,
        "fid_reference_dir": portable(args.fid_reference_dir),
        "fid_reference_manifest": reference_manifest,
        "classifier_checkpoint": portable(args.classifier_checkpoint),
        "classifier_sha256": file_sha256(args.classifier_checkpoint),
        "classifier_architecture": args.classifier_architecture,
        "classifier_preprocess": args.preprocess,
        "fid": fid,
        **classifier_metadata,
        **fd_metrics,
    }
    output = args.out or args.images_dir / "balancing_act_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()

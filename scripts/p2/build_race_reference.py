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

"""The race FID reference of Balancing Act: CelebA-HQ ranked by CLIP similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from debias_anything.evaluators.celebahq import ImageFiles, file_sha256
from debias_anything.paths import link_or_copy, portable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--prompt", default="a black person")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_images < 2 or args.num_images % 2:
        raise ValueError("--num-images must be a positive even integer")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")

    device = torch.device(args.device)
    model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    model.eval()
    dataset = ImageFiles(args.images_dir, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    tokens = clip.tokenize([args.prompt]).to(device)
    text_feature = F.normalize(model.encode_text(tokens).float(), dim=-1)[0]
    scores: list[tuple[float, Path]] = []
    offset = 0
    for images in loader:
        image_features = F.normalize(
            model.encode_image(images.to(device, non_blocking=True)).float(), dim=-1
        )
        similarities = image_features @ text_feature
        scores.extend(
            (float(score), dataset.paths[offset + row])
            for row, score in enumerate(similarities.cpu())
        )
        offset += len(images)

    per_class = args.num_images // 2
    if len(scores) < args.num_images:
        raise ValueError(f"Need {args.num_images} images, found {len(scores)}")
    scores.sort(key=lambda item: (item[0], str(item[1])))
    selected = [("White", *item) for item in scores[:per_class]]
    selected += [("Black", *item) for item in scores[-per_class:]]
    # Stable interleaving prevents any downstream file-order dependence.
    selected.sort(key=lambda item: str(item[2]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (label, _, source) in enumerate(selected):
        destination = args.output_dir / f"{index:05d}_{label}_{source.name}"
        link_or_copy(source, destination)

    cache_name = args.clip_model.replace("/", "-") + ".pt"
    clip_weights = Path.home() / ".cache/clip" / cache_name
    manifest = {
        "protocol": "balancing_act_clip_ranked_race_reference_v1",
        "source_images_dir": portable(args.images_dir),
        "label_source": "OpenAI CLIP cosine-similarity ranking",
        "clip_model": args.clip_model,
        "clip_weights": clip_weights.name if clip_weights.is_file() else None,
        "clip_weights_sha256": file_sha256(clip_weights)
        if clip_weights.is_file()
        else None,
        "prompt": args.prompt,
        "num_source_images": len(scores),
        "num_images": args.num_images,
        "per_class": per_class,
        "selection": {
            "White": f"{per_class} lowest similarities",
            "Black": f"{per_class} highest similarities",
        },
        "score_boundaries": {
            "white_max": scores[per_class - 1][0],
            "black_min": scores[-per_class][0],
        },
        "caveat": (
            "The paper specifies CLIP and the prompt/ranking procedure but does "
            "not identify its CLIP backbone; this run records the chosen backbone."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

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

"""Create a deterministic attribute-balanced CelebA-HQ FID reference set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from debias_anything.paths import link_or_copy, portable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--attribute", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images < 2 or args.num_images % 2:
        raise ValueError("--num-images must be a positive even integer")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")

    with args.annotations.open() as handle:
        expected = int(handle.readline())
        attributes = handle.readline().split()
        try:
            attribute_index = attributes.index(args.attribute)
        except ValueError as error:
            raise ValueError(f"Unknown attribute {args.attribute!r}") from error
        classes: dict[int, list[str]] = {-1: [], 1: []}
        for line in handle:
            fields = line.split()
            classes[int(fields[attribute_index + 1])].append(fields[0])

    if sum(map(len, classes.values())) != expected:
        raise ValueError("Annotation row count does not match its header")

    rng = random.Random(args.seed)
    per_class = args.num_images // 2

    def draw(population: list[str]) -> list[str]:
        if len(population) >= per_class:
            return rng.sample(population, per_class)
        return rng.choices(population, k=per_class)

    selections = [(-1, name) for name in draw(classes[-1])]
    selections += [(1, name) for name in draw(classes[1])]
    rng.shuffle(selections)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (label, name) in enumerate(selections):
        source = (args.images_dir / name).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = args.output_dir / f"{index:05d}_{label:+d}_{name}"
        link_or_copy(source, destination)

    manifest = {
        "source_images_dir": portable(args.images_dir),
        "annotations": portable(args.annotations),
        "attribute": args.attribute,
        "seed": args.seed,
        "num_images": args.num_images,
        "per_class": per_class,
        "available_per_class": {str(key): len(value) for key, value in classes.items()},
        "sampling_with_replacement": {
            str(key): len(value) < per_class for key, value in classes.items()
        },
        "unique_selected_images": len({name for _, name in selections}),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

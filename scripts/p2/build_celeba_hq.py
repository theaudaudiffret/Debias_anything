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

"""Download CelebA-HQ 256×256 (``korexyz/celeba-hq-256x256``) to ``data/celeba_hq``."""

import torch
from datasets import load_dataset

from debias_anything.paths import DATA_ROOT

OUT_ROOT = DATA_ROOT / "celeba_hq"

REPO_ID = "korexyz/celeba-hq-256x256"
SPLITS = ("train", "validation")


def main() -> None:
    print(f"Downloading {REPO_ID} (~3.0 GB)…")
    dataset = load_dataset(REPO_ID)

    label_names = dataset["train"].features["label"].names
    print(f"labels: {dict(enumerate(label_names))}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        out = OUT_ROOT / split
        dataset[split].save_to_disk(str(out))

        labels = torch.tensor(dataset[split]["label"], dtype=torch.long)
        image = dataset[split][0]["image"]
        print(
            f"  {split:10s}: {len(labels):6d} images {image.size} {image.mode}"
            f" | {label_names[1]} = {labels.float().mean():.3f} → {out}"
        )


if __name__ == "__main__":
    main()

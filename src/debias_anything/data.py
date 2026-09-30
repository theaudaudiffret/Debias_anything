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

from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .paths import DATA_ROOT, REPO_ROOT

DATALOADER_KINDS = (
    "celeba",
    "celeba_balanced",
    "celeba_balanced_eyeglasses",
    "celeba_hq",
    "minority_celeba",
)

_CELEBA_TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(140),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class CelebANoCheck(torchvision.datasets.CelebA):
    """torchvision CelebA without the MD5 check of its files."""

    def _check_integrity(self) -> bool:
        return (Path(self.root) / self.base_folder / "img_align_celeba").is_dir()


class MinorityCelebA(CelebANoCheck):
    """The 10,000 CelebA images of highest AvgkNN (``build_minority_celeba.py``)."""

    base_folder = "minority_celeba"


class CelebAHQ(Dataset):
    """CelebA-HQ 256×256 of ``data/celeba_hq/<split>``: [-1, 1] images and gender."""

    resolution = 256

    def __init__(self, split: str = "train", hflip: bool = False):
        self.path = DATA_ROOT / "celeba_hq" / split
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} missing: run `uv run python scripts/p2/build_celeba_hq.py` first."
            )
        self.ds = HFDataset.load_from_disk(str(self.path))
        self.targets = torch.tensor(self.ds["label"], dtype=torch.long)
        self.hflip = hflip

    @property
    def source_id(self) -> str:
        """Identifier of the split, recorded in the target caches of the P2 adapter."""
        return f"hf:{self.path.relative_to(REPO_ROOT)}:resolution={self.resolution}:n={len(self)}"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.ds[i]["image"].convert("RGB")
        if img.size != (self.resolution, self.resolution):
            img = img.resize(
                (self.resolution, self.resolution), Image.Resampling.BICUBIC
            )
        # np.array, not np.asarray: a shared PIL buffer is read-only
        x = torch.from_numpy(np.array(img, dtype=np.uint8))
        # (H, W, 3) uint8 -> (3, H, W) in [-1, 1]
        x = x.permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        if self.hflip and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])
        return x, self.targets[i]


class DataLoaderWithTargets(DataLoader):
    """DataLoader exposing ``targets`` and indexing of its dataset."""

    @property
    def targets(self):
        return self.dataset.targets  # type: ignore

    def __getitem__(self, idx):
        return self.dataset[idx]


def make_balanced_celeba_dataloader(
    dataset,
    batch_size: int = 64,
    attr_idx: int = 20,
    num_samples: int | None = None,
) -> DataLoaderWithTargets:
    """CelebA loader resampled 50/50 on the binary attribute ``attr_idx``."""
    labels = dataset.attr[:, attr_idx]
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=num_samples or len(dataset),
        replacement=True,
    )
    return DataLoaderWithTargets(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )


def _loader(
    dataset, batch_size: int, shuffle: bool, num_workers: int = 8
) -> DataLoaderWithTargets:
    return DataLoaderWithTargets(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


def get_dataloader(
    kind: str, batch_size: int = 64, split: str = "train"
) -> DataLoaderWithTargets:
    """Dataloader of ``kind`` (see ``DATALOADER_KINDS``), ``split`` "train" or "val"."""
    if split not in ("train", "val"):
        raise ValueError(f"split={split!r} must be 'train' or 'val'")
    is_train = split == "train"

    if kind in ("celeba", "celeba_balanced", "celeba_balanced_eyeglasses"):
        dataset = CelebANoCheck(
            root=str(DATA_ROOT),
            split="train" if is_train else "valid",
            download=False,
            transform=_CELEBA_TRANSFORM,
        )
        dataset.targets = dataset.attr  # type: ignore[attr-defined]
        if kind != "celeba" and is_train:
            # rebalanced on the train side only: the val split keeps the natural distribution
            attr = "Male" if kind == "celeba_balanced" else "Eyeglasses"
            return make_balanced_celeba_dataloader(
                dataset, batch_size=batch_size, attr_idx=dataset.attr_names.index(attr)
            )
        return _loader(dataset, batch_size, shuffle=is_train)

    if kind == "celeba_hq":
        # 28,000 images only: horizontal flips on the train side
        dataset = CelebAHQ(split="train" if is_train else "validation", hflip=is_train)
        return _loader(dataset, batch_size, shuffle=is_train, num_workers=4)

    if kind == "minority_celeba":
        # a fixed subset without validation split: the same images for train and val
        dataset = MinorityCelebA(
            root=str(DATA_ROOT),
            split="train",
            download=False,
            transform=_CELEBA_TRANSFORM,
        )
        dataset.targets = dataset.attr  # type: ignore[attr-defined]
        return _loader(dataset, batch_size, shuffle=is_train)

    raise ValueError(f"Unknown kind={kind!r}. Choose from {DATALOADER_KINDS}")

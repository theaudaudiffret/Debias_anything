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

"""Train the CelebA gender and eyeglasses classifiers of the metrics (Appendix B.5)."""

import csv
import pathlib
from dataclasses import dataclass

import hydra
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from debias_anything.data import CelebANoCheck
from debias_anything.evaluators.celeba import CelebaClassifier
from debias_anything.evaluators.celebahq import file_sha256
from debias_anything.metrics.metrics import CSV_ALPHA
from debias_anything.paths import CHECKPOINT_DIR, CONF_DIR, DATA_ROOT

# Checkpoint of each CelebA attribute, as read by debias_anything.metrics.metrics.CELEBA_SACS.
CELEBA_CHECKPOINTS = {
    "Male": CHECKPOINT_DIR / "celeba_gender_classifier.pt",
    "Eyeglasses": CHECKPOINT_DIR / "celeba_eyeglasses_classifier.pt",
}

# Optional override of the checkpoint path (comparative retraining: a checkpoint
# already referenced in the measurements must not be overwritten).
CHECKPOINT_OVERRIDE: pathlib.Path | None = None


class _CelebAAttribute(torch.utils.data.Dataset):
    """CelebA returning ``(image, attribute label)``."""

    def __init__(self, base: datasets.CelebA, attribute: str):
        self.base = base
        self.attr_idx = base.attr_names.index(attribute)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, attr = self.base[idx]
        return img, attr[self.attr_idx].long()


def _get_celeba_loaders(
    attribute: str, batch_size: int = 128
) -> tuple[DataLoader, DataLoader]:
    # Preprocessing aligned with debias_anything.data (CenterCrop 140 → Resize 64 → [-1, 1]).
    base_tf = [
        transforms.CenterCrop(140),
        transforms.Resize(64),
    ]
    train_tf = transforms.Compose(
        [
            *base_tf,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    val_tf = transforms.Compose(
        [
            *base_tf,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_base = CelebANoCheck(
        root=str(DATA_ROOT), split="train", download=False, transform=train_tf
    )
    val_base = CelebANoCheck(
        root=str(DATA_ROOT), split="valid", download=False, transform=val_tf
    )
    train_ds = _CelebAAttribute(train_base, attribute)
    val_ds = _CelebAAttribute(val_base, attribute)
    # Train split rebalanced 50/50: some attributes (Eyeglasses ≈ 6.5 % positives)
    # are too imbalanced for the plain CE to learn the minority class.
    # The valid split keeps the natural distribution — it is on this split that
    # the α₀/α₁ consumed by CLEAM are measured (see debias_anything.metrics.metrics).
    labels = train_base.attr[:, train_ds.attr_idx]
    class_weights = 1.0 / torch.bincount(labels).float()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=class_weights[labels].tolist(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


def _build(
    attribute: str,
) -> tuple[nn.Module, pathlib.Path, tuple[DataLoader, DataLoader]]:
    if attribute not in CELEBA_CHECKPOINTS:
        raise ValueError(
            f"attribute without a declared checkpoint: {attribute!r} — "
            f"add an entry to CELEBA_CHECKPOINTS ({sorted(CELEBA_CHECKPOINTS)})"
        )
    return (
        CelebaClassifier(),
        CHECKPOINT_OVERRIDE or CELEBA_CHECKPOINTS[attribute],
        _get_celeba_loaders(attribute),
    )


def train(
    epochs: int = 20,
    lr: float = 1e-3,
    device: str | None = None,
    attribute: str = "Male",
) -> nn.Module:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, checkpoint_path, (train_loader, val_loader) = _build(attribute)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc = _evaluate(model, val_loader, device)
        print(f"Epoch {epoch:>2} | val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Best val accuracy: {best_acc:.4f} — checkpoint saved to {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    overall, per_class, n_per_class = _evaluate_per_class(
        model, val_loader, device, num_classes=2
    )
    print(
        f"CelebA {attribute} — acc={overall:.4f} "
        f"| P(correct|absent=0)={per_class[0]:.4f} "
        f"| P(correct|present=1)={per_class[1]:.4f} "
        f"| n=({n_per_class[0]}, {n_per_class[1]})"
    )
    _log_celeba_characteristics(checkpoint_path, overall, per_class, n_per_class)

    return model


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def _evaluate_per_class(
    model: nn.Module, loader: DataLoader, device: str, num_classes: int = 2
) -> tuple[float, list[float], list[int]]:
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            for c in range(num_classes):
                mask = y == c
                total[c] += mask.sum().item()
                correct[c] += ((preds == y) & mask).sum().item()
    overall = correct.sum().item() / max(total.sum().item(), 1)
    per_class = (correct / total.clamp(min=1)).tolist()
    return overall, per_class, total.long().tolist()


def _log_celeba_characteristics(
    checkpoint_path: pathlib.Path,
    overall: float,
    per_class: list[float],
    n_per_class: list[int],
) -> None:
    csv_path = pathlib.Path(CSV_ALPHA)
    sac_hash = file_sha256(checkpoint_path)
    header = [
        "classifier_name",
        "sac_checkpoint_hash",
        "validation_set",
        "n_samples_class0",
        "n_samples_class1",
        "accuracy",
        "proba_0",
        "proba_1",
    ]
    header_needed = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(header)
        writer.writerow(
            [
                checkpoint_path.stem,
                sac_hash,
                "CelebA split=valid",
                n_per_class[0],
                n_per_class[1],
                overall,
                per_class[0],
                per_class[1],
            ]
        )
    print(f"Characteristics saved to {csv_path}")


@dataclass
class Config:
    epochs: int = 20
    lr: float = 1e-3
    device: str | None = None
    attribute: str = "Male"  # CelebA attribute to predict


ConfigStore.instance().store(name="celeba_classifier_schema", node=Config)


@hydra.main(
    version_base="1.3", config_path=str(CONF_DIR / "celeba"), config_name="classifier"
)
def main(cfg: Config) -> None:
    train(
        epochs=cfg.epochs,
        lr=cfg.lr,
        device=cfg.device,
        attribute=cfg.attribute,
    )


if __name__ == "__main__":
    main()

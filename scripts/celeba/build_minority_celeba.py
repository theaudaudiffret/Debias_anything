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

"""Build ``data/minority_celeba``: the 10,000 CelebA images of highest AvgkNN."""

import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import DataLoader
from torchvision import transforms as T

from debias_anything.data import CelebANoCheck
from debias_anything.paths import DATA_ROOT

K = 5
TOP_N = 10_000
BATCH_SIZE = 256
OUT_ROOT = DATA_ROOT / "minority_celeba"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

celeba_transform = T.Compose(
    [
        T.CenterCrop(140),
        T.Resize(64),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

dataset = CelebANoCheck(
    root=str(DATA_ROOT), split="train", download=False, transform=celeba_transform
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"CelebA train: {len(dataset)} images")

_RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
_RESNET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

resnet_model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
resnet_model.fc = torch.nn.Identity()  # keep avgpool features (2048-d)
resnet_model = resnet_model.to(device).eval()


@torch.no_grad()
def extract_resnet50_features(
    images: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """ResNet-50 avgpool features (N, 2048) of images (N, C, H, W) in [-1, 1]."""
    feats = []
    for batch in images.split(batch_size):
        batch = (batch.clamp(-1, 1) + 1) / 2
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = batch.to(device)
        batch = F.interpolate(batch, size=224, mode="bilinear", align_corners=False)
        batch = (batch - _RESNET_MEAN) / _RESNET_STD
        feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats)


feats_chunks = []
for i, (images, _) in enumerate(loader):
    feats_chunks.append(extract_resnet50_features(images, resnet_model, device))
    if i % 50 == 0:
        print(f"features batch {i}/{len(loader)}")
feats_real = np.concatenate(feats_chunks, axis=0)
print("feats_real:", feats_real.shape)


def compute_self_avgknn(
    feats: np.ndarray, k: int, device: torch.device, chunk_size: int = 1024
) -> np.ndarray:
    """Mean L2 distance of each sample to its ``k`` nearest other samples."""
    x = torch.from_numpy(feats).to(device)
    n = x.shape[0]
    avg_knn = torch.empty(n, device=device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = torch.cdist(x[start:end], x)
        rows = torch.arange(end - start, device=device)
        dists[rows, start + rows] = float("inf")  # exclude self-match
        topk = torch.topk(dists, k, dim=1, largest=False).values
        avg_knn[start:end] = topk.mean(dim=1)
        if start % (chunk_size * 20) == 0:
            print(f"avgKNN chunk {start}/{n}")
    return avg_knn.cpu().numpy()


avgknn = compute_self_avgknn(feats_real, k=K, device=device)
print(
    f"AvgkNN(k={K}): min={avgknn.min():.3f} mean={avgknn.mean():.3f} max={avgknn.max():.3f}"
)

top_k_idx = np.argsort(-avgknn)[:TOP_N]
print(
    f"Top-{TOP_N} pool: AvgkNN in [{avgknn[top_k_idx].min():.3f}, {avgknn[top_k_idx].max():.3f}] "
    f"(dataset max: {avgknn.max():.3f})"
)

# --- materialize the subset on disk (standard torchvision CelebA layout) ---
img_dir = OUT_ROOT / "img_align_celeba"
img_dir.mkdir(parents=True, exist_ok=True)

filenames = [dataset.filename[i] for i in top_k_idx]
for fname in filenames:
    shutil.copy(DATA_ROOT / "celeba" / "img_align_celeba" / fname, img_dir / fname)

attr_pm1 = dataset.attr[top_k_idx] * 2 - 1  # {0,1} -> torchvision's on-disk {-1,1}
bbox = dataset.bbox[top_k_idx]
landmarks = dataset.landmarks_align[top_k_idx]
identity = dataset.identity[top_k_idx].squeeze(1)

with open(OUT_ROOT / "list_attr_celeba.txt", "w") as f:
    f.write(f"{len(filenames)}\n")
    f.write(" ".join(dataset.attr_names) + "\n")
    for fname, row in zip(filenames, attr_pm1.tolist()):
        f.write(fname + " " + " ".join(str(v) for v in row) + "\n")

with open(OUT_ROOT / "list_bbox_celeba.txt", "w") as f:
    f.write(f"{len(filenames)}\n")
    f.write("x_1 y_1 width height\n")
    for fname, row in zip(filenames, bbox.tolist()):
        f.write(fname + " " + " ".join(str(v) for v in row) + "\n")

with open(OUT_ROOT / "list_landmarks_align_celeba.txt", "w") as f:
    f.write(f"{len(filenames)}\n")
    f.write(
        "lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y "
        "leftmouth_x leftmouth_y rightmouth_x rightmouth_y\n"
    )
    for fname, row in zip(filenames, landmarks.tolist()):
        f.write(fname + " " + " ".join(str(v) for v in row) + "\n")

with open(OUT_ROOT / "identity_CelebA.txt", "w") as f:
    for fname, ident in zip(filenames, identity.tolist()):
        f.write(f"{fname} {ident}\n")

with open(OUT_ROOT / "list_eval_partition.txt", "w") as f:
    for fname in filenames:
        f.write(f"{fname} 0\n")  # everything is "train"

np.save(OUT_ROOT / "avgknn_scores.npy", avgknn[top_k_idx])
print(f"Wrote {len(filenames)} images to {img_dir}")

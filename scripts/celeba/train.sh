#!/usr/bin/env bash
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

# CelebA 64x64: everything the guided runs need, in order (Appendices B.1, B.5 and C).
#   bash scripts/celeba/train.sh
# CelebA must be in data/celeba (torchvision layout: img_align_celeba/, list_attr_celeba.txt, ...).
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1. EDM generator (B.1). Writes checkpoints/diffusion_celeba_2000.pth; the released checkpoint is
#    checkpoints/celeba_edm_64.pth. The paper's model resumed this run at each validation plateau
#    (load_from_ckpt=... lr=1e-4, then 1e-5 and 1e-6) and keeps the EMA of lowest validation loss.
uv run python scripts/celeba/train_generator.py

# 2. Attribute classifiers that count gender and eyeglasses on generated images (B.5). They write
#    checkpoints/celeba_{gender,eyeglasses}_classifier.pt and append their per-class accuracies
#    (the CLEAM alphas) to checkpoints/celeba_classifier_characteristics.csv.
uv run python scripts/celeba/train_classifier.py attribute=Male
uv run python scripts/celeba/train_classifier.py attribute=Eyeglasses

# 3. Adapter h-space -> SigLIP 2 (Section 4.1, Table 4). Also caches the SigLIP embeddings of the
#    training images, its targets, in data/siglip_cache_celeba_train.pt.
uv run python scripts/celeba/train_adapter.py

# 4. Rare reference set of Appendix D.5 (the 10,000 training images of highest AvgkNN).
uv run python scripts/celeba/build_minority_celeba.py

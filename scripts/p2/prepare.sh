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

# CelebA-HQ / P2 (Appendices B.5 and C): adapter data and training, then the evaluators.
#   bash scripts/p2/prepare.sh
# Needs checkpoints/celebahq_p2.pt (P2, Choi et al. 2022) and, for the evaluators, CelebAMask-HQ in
# data/celebahq/CelebAMask-HQ/ (CelebA-HQ-img/ and CelebAMask-HQ-attribute-anno.txt).
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1. Adapter data: CelebA-HQ 256x256 from the Hugging Face hub, 28,000 train / 2,000 validation.
uv run python scripts/p2/build_celeba_hq.py

# 2. Adapter h-space -> SigLIP 2 of P2 (Table 4), with the settings of conf/p2/adapter.yaml: the 49
#    DDIM steps of the 50-step grid, two noisy views per image, AdamW, batch 32, lr 1e-3, weight
#    decay 1e-4, 60 epochs (the kept epoch is the one of best validation R@1).
uv run python scripts/p2/train_adapter.py

# 3. Image-space evaluators (B.5): ResNet-18s pretrained on ImageNet and fine-tuned on 80% of
#    CelebA-HQ (conf/p2/evaluator.yaml). Race has no label: its split and CLIP ranking are only used
#    to evaluate FairFace.
ANNOTATIONS=data/celebahq/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt
for attribute in gender eyeglasses race; do
  extra=()
  [[ "${attribute}" != "race" ]] && extra=("annotations=${ANNOTATIONS}")
  uv run python scripts/p2/train_evaluator.py attribute="${attribute}" \
    output_dir="checkpoints/celebahq_evaluators/${attribute}" "${extra[@]}"
done

# 4. Per-class recalls of the evaluators on their validation split, the CLEAM alphas of Table 2:
#    checkpoints/celebahq_evaluators/recalls.{csv,json}.
uv run python scripts/p2/evaluate_evaluators.py

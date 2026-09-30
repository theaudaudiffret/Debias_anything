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

# The four guidance terms compared at their selected weights (Tables 8 and 9, Appendix D.4): the
# fairness term alone (5 on gender, 200 on eyeglasses), then with each diversity term on top. By
# default five seeds of 5,000 images (Table 8); LARGE=1 for one seed of 50,000 images (Table 9).
set -euo pipefail
cd "$(dirname "$0")/../.."
CM="uv run python scripts/celeba/sample_and_evaluate.py"
EYE=("target_prompt=a person with eyeglasses or sunglasses" "source_prompt=a person without eyeglasses or sunglasses")

if [[ "${LARGE:-0}" == "1" ]]; then
  SEEDS=(42); SIZE="n_batches=500 batch_size=100 n_real_train=50000"; ROOT=result_metrics/celeba_diversity_terms_50k
else
  SEEDS=(0 1 2 3 4); SIZE="n_batches=50 batch_size=100 n_real_train=5000"; ROOT=result_metrics/celeba_diversity_terms
fi

for seed in "${SEEDS[@]}"; do
  S="${SIZE} seed=${seed} output_dir=${ROOT}/seed${seed}"
  # gender, fairness weight 5
  G="${S} dataloader_kind=celeba_balanced guidance.guidance_weight=5"
  ${CM} ${G} guidance=batched_text
  ${CM} ${G} guidance=sgms guidance.guidance_weight_minority=20 guidance.perturb_sigma=0.4 guidance.noise_seed=${seed}
  ${CM} ${G} guidance=siglipms guidance.guidance_weight_minority=20 guidance.guide_every_n=1
  ${CM} ${G} guidance=perturbationproj guidance.guidance_weight_minority=120 guidance.perturb_sigma=0.4 guidance.noise_seed=${seed}
  # eyeglasses, fairness weight 200
  E="${S} dataloader_kind=celeba_balanced_eyeglasses guidance.guidance_weight=200"
  ${CM} ${E} "${EYE[@]}" guidance=batched_text
  ${CM} ${E} "${EYE[@]}" guidance=sgms guidance.guidance_weight_minority=20 guidance.perturb_sigma=0.4 guidance.noise_seed=${seed}
  ${CM} ${E} "${EYE[@]}" guidance=siglipms guidance.guidance_weight_minority=120 guidance.guide_every_n=1
  ${CM} ${E} "${EYE[@]}" guidance=perturbationproj guidance.guidance_weight_minority=300 guidance.perturb_sigma=0.4 guidance.noise_seed=${seed}
done

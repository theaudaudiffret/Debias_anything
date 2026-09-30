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

# Table 2 (CelebA-HQ, P2, protocol of Balancing Act): the four "Debias Anything" rows, generation
# then evaluation.
# Generation: 10,000 images per attribute and setting, one seed. DDIM, 50 steps, eta = 1, batch size
# 50, guidance at every step, image i starting from a CPU generator seeded with 42 + i.
# Evaluation: FID against 5,000 real CelebA-HQ images balanced on the attribute (for race, the 2,500
# images most and least similar to "a black person" under CLIP ViT-B/32), FD from hard predictions
# of the ResNet-18 evaluators (race: FairFace, White and Black outputs), then CLEAM correction with
# the evaluators' recalls.
#   bash scripts/p2/generate_and_evaluate.sh
# Needs scripts/p2/prepare.sh (evaluators and their recalls) and the FairFace checkpoint in
# checkpoints/fairface/res34_fair_align_multi_7_20190809.pt. Set GENERATE=0 to only evaluate
# existing images.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-outputs/p2}"
N_IMAGES="${N_IMAGES:-10000}"
GENERATE="${GENERATE:-1}"
PREFIX="${PREFIX:-result_metrics/p2_debias_anything}"
RUN=(uv run python scripts/p2/generate.py target_proportion=0.5 "n_images=${N_IMAGES}"
  batch_size=50 n_ddim_steps=50 eta=1 seed=42 per_sample_noise_seeds=true)

declare -A SOURCE=([gender]="a photo of a woman" [race]="a photo of a white person" \
  [eyeglasses]="a photo of a person without eyeglasses")
declare -A TARGET=([gender]="a photo of a man" [race]="a photo of a black person" \
  [eyeglasses]="a photo of a person with eyeglasses")
# Fairness weights (Appendix B.6).
declare -A QUALITY=([gender]=5 [race]=30 [eyeglasses]=180)
declare -A FAIRNESS=([gender]=20 [race]=100 [eyeglasses]=500)
# Diversity term (PerturbProj, Eq. 12): weight 40, re-noising at 0.5 sigma, every second DDIM step
# for t <= 800.
DIVERSITY=(perturb_proj=true perturb_proj_weight=40 perturb_sigma=0.5 minority_t_max=800
  minority_every_n=2)

if [[ "${GENERATE}" == "1" ]]; then
  for attribute in gender race eyeglasses; do
    prompts=("source_prompt=${SOURCE[$attribute]}" "target_prompt=${TARGET[$attribute]}")
    "${RUN[@]}" "${prompts[@]}" "guidance_weight=${QUALITY[$attribute]}" \
      "out=${OUT}/quality_${attribute}"
    "${RUN[@]}" "${prompts[@]}" "guidance_weight=${QUALITY[$attribute]}" "${DIVERSITY[@]}" \
      "out=${OUT}/quality_diversity_${attribute}"
    "${RUN[@]}" "${prompts[@]}" "guidance_weight=${FAIRNESS[$attribute]}" \
      "out=${OUT}/fairness_${attribute}"
    "${RUN[@]}" "${prompts[@]}" "guidance_weight=${FAIRNESS[$attribute]}" "${DIVERSITY[@]}" \
      "out=${OUT}/fairness_diversity_${attribute}"
  done
fi

rows=()
for setting in quality quality_diversity fairness fairness_diversity; do
  rows+=(--run "${setting}" "${OUT}/${setting}_gender" "${OUT}/${setting}_race" "${OUT}/${setting}_eyeglasses")
done
uv run python scripts/p2/evaluate_table.py "${rows[@]}" --output-prefix "${PREFIX}"
uv run python scripts/p2/cleam_correct_fd.py --table "${PREFIX}.csv" \
  --classifiers checkpoints/celebahq_evaluators/recalls.csv --out "${PREFIX}_cleam.csv"

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

# CelebA-HQ (P2, Table 2): the "Ours (Fairness)" row on gender and race only, generation then
# evaluation. Same sampling as scripts/p2/generate_and_evaluate.sh (10,000 images, DDIM, 50 steps, eta = 1, batch
# size 50, image i from a CPU generator seeded with 42 + i, fairness weights 20 and 100 of
# Appendix B.6) and same evaluation as scripts/p2/generate_and_evaluate.sh (FD from hard predictions,
# FID against 5,000 balanced real images, CLEAM correction).
#   bash scripts/p2/fairness_gender_race.sh
# Needs scripts/p2/prepare.sh (evaluators and their recalls) and the FairFace checkpoint in
# checkpoints/fairface/. Set GENERATE=0 to only evaluate existing images.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-outputs/p2}"
N_IMAGES="${N_IMAGES:-10000}"
GENERATE="${GENERATE:-1}"
PREFIX="${PREFIX:-result_metrics/p2_fairness_gender_race}"
RUN=(uv run python scripts/p2/generate.py target_proportion=0.5 "n_images=${N_IMAGES}"
  batch_size=50 n_ddim_steps=50 eta=1 seed=42 per_sample_noise_seeds=true)

declare -A SOURCE=([race]="a photo of a white person")
declare -A TARGET=([race]="a photo of a black person")
declare -A FAIRNESS=([race]=100)

if [[ "${GENERATE}" == "1" ]]; then
  for attribute in race; do
    "${RUN[@]}" "source_prompt=${SOURCE[$attribute]}" "target_prompt=${TARGET[$attribute]}" \
      "guidance_weight=${FAIRNESS[$attribute]}" "out=${OUT}/fairness_${attribute}"
  done
fi

# "-": eyeglasses not evaluated.
uv run python scripts/p2/evaluate_table.py \
  --run fairness "${OUT}/fairness_gender" "${OUT}/fairness_race" - \
  --output-prefix "${PREFIX}"
uv run python scripts/p2/cleam_correct_fd.py --table "${PREFIX}.csv" \
  --classifiers checkpoints/celebahq_evaluators/recalls.csv --out "${PREFIX}_cleam.csv"

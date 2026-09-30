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

# Table 3 (and Tables 15-16): Debias Anything on Stable Diffusion 1.5, protocol of DiffLens
# (Shi et al., 2025): 500 images for each of 4 occupation prompts, one weight for all attributes.
#   bash scripts/sd15/generate_and_evaluate.sh
# The evaluation uses the wrappers of eval/difflens/, copied into a clone of DiffLens whose path is
# given by DIFFLENS_DIR (see eval/difflens/README.md). Set EVALUATE=0 to only generate.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-outputs/sd15}"
EVALUATE="${EVALUATE:-1}"
DIFFLENS_DIR="${DIFFLENS_DIR:-../DiffLens}"
RUN=(uv run python scripts/sd15/generate.py projector=checkpoints/sd15_adapter.pth)

# Guided runs: Euler with trailing timesteps, 30 steps, CFG 7.5, groups of 100 images of a single
# prompt, guidance weight 3000 for sigma in [0, 4], and the default negative prompt of the script.
GUIDED=(weight=3000 n_images=500 group=100 steps=30 sigma_min=0 sigma_max=4 seed=0)
"${RUN[@]}" "${GUIDED[@]}" "source=a photo of a man" "targets=[a photo of a woman]" \
  "out=${OUT}/gender"
"${RUN[@]}" "${GUIDED[@]}" "source=a photo of a person" \
  "targets=[a photo of a young person,a photo of a middle-aged person,a photo of an old person]" \
  "out=${OUT}/age"
"${RUN[@]}" "${GUIDED[@]}" "source=a photo of a person" \
  "targets=[a photo of a white person,a photo of a black person,a photo of an asian person,a photo of an indian person]" \
  "out=${OUT}/race"

# Unguided samples of the same prompts, the reference of CLIP-I: DDIM (DiffLens' scheduler),
# 50 steps, no negative prompt.
"${RUN[@]}" weight=0 scheduler=ddim n_images=500 group=20 steps=50 seed=0 negative_prompt= \
  "out=${OUT}/unguided"

if [[ "${EVALUATE}" == "1" ]]; then
  for attribute in gender age race; do
    run="$(realpath "${OUT}/${attribute}")"
    uv run python "${DIFFLENS_DIR}/evaluate_run.py" --run "${run}" --original "$(realpath "${OUT}/unguided")"
    (cd "${DIFFLENS_DIR}" && uv run python compute_fid.py --run "${run}" --ref ffhq)
  done
fi

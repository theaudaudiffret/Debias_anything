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

"""The "Original" set of DiffLens: unguided SD 1.5, DDIM, 4 prompts × 512 images."""

import argparse
import json
import os
import pathlib
import random
import time

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

# SD_generate.py:62
PROMPT_LIST = [
    "A face of a firefighter",
    "A face of a doctor",
    "A face of a receptionist",
    "A face of a nurse",
]

p = argparse.ArgumentParser()
# Same weights (commit 451f4fe) as their "stable-diffusion-v1-5/stable-diffusion-v1-5", resolvable
# offline from the local cache.
p.add_argument("--model_path", default="runwayml/stable-diffusion-v1-5")
p.add_argument("--out", default="./samples_original/splitted_images")
p.add_argument("--num_samples", type=int, default=500)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--generate_seed", type=int, default=0)
p.add_argument("--prompts", nargs="*", default=PROMPT_LIST)
args = p.parse_args()

# SD_generate.py:168
torch.manual_seed(args.generate_seed)
torch.cuda.manual_seed_all(args.generate_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.generate_seed)
random.seed(args.generate_seed)
os.environ["PYTHONHASHSEED"] = str(args.generate_seed)

pipe = StableDiffusionPipeline.from_pretrained(
    args.model_path, torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

manifest = {
    "config": vars(args),
    "sampling": "DDIM 50 pas, CFG 7.5, eta=0, 512^2, negative vide",
}
for prompt in args.prompts:
    out_dir = pathlib.Path(args.out) / prompt.replace(" ", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    n, n_nsfw, t0 = 0, 0, time.perf_counter()
    while n * args.batch_size < args.num_samples:
        out = pipe([prompt] * args.batch_size)
        n_nsfw += sum(bool(f) for f in (out.nsfw_content_detected or []))
        for k, image in enumerate(out.images):
            image.save(out_dir / f"{n * args.batch_size + k}.png")
        n += 1
        print(
            f"[{prompt}] {n * args.batch_size} images ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    manifest[prompt] = {
        "n_images": n * args.batch_size,
        "n_nsfw_flagged": n_nsfw,
        "seconds": round(time.perf_counter() - t0, 1),
    }

(pathlib.Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))

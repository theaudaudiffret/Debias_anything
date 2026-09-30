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

"""Clean-FID of a run against FFHQ, pooled over its prompts (DiffLens protocol)."""

import argparse
import json
import pathlib
import shutil
import tempfile

from cleanfid import fid

p = argparse.ArgumentParser()
p.add_argument("--run", default="./samples_original")
p.add_argument(
    "--ref",
    default="ffhq",
    help="'ffhq', 'celebahq256' (hosted stats) or a folder of images",
)
p.add_argument(
    "--flat",
    action="store_true",
    help="--run is already a flat folder (P2, no splitted_images/<prompt>/)",
)
p.add_argument(
    "--out",
    default=None,
    help="path of the output json (default: <run>/fid_<ref>.json)",
)
args = p.parse_args()

run = pathlib.Path(args.run).resolve()
pool = pathlib.Path(tempfile.mkdtemp(prefix="fid_pool_"))
try:
    if args.flat:
        for f in sorted(run.glob("*.png")):
            (pool / f.name).symlink_to(f)
    else:
        for sub in sorted((run / "splitted_images").iterdir()):
            for f in sorted(sub.glob("*.png")):
                (pool / f"{sub.name}_{f.name}").symlink_to(f)
    n = len(list(pool.iterdir()))

    if args.ref == "ffhq":
        score = fid.compute_fid(
            str(pool), dataset_name="FFHQ", dataset_res=256, dataset_split="trainval70k"
        )
    elif args.ref == "celebahq256":
        score = fid.compute_fid(
            str(pool),
            dataset_name="celebahq256",
            dataset_res=256,
            dataset_split="custom",
        )
    else:
        score = fid.compute_fid(args.ref, str(pool))

    result = {
        "run": str(run),
        "ref": args.ref,
        "n_images": n,
        "FID": round(float(score), 4),
    }
    out = (
        pathlib.Path(args.out)
        if args.out
        else run / f"fid_{pathlib.Path(args.ref).name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
finally:
    shutil.rmtree(pool, ignore_errors=True)

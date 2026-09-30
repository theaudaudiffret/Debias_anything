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

"""FD (gender, age, race), CLIP-T and CLIP-I of a run, per prompt then averaged."""

import argparse
import importlib.util
import json
import os
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FAIRFACE_DIR = HERE / "evaluation" / "Fairface"
DLIB_MODELS = HERE / "evaluation" / "crop_face" / "dlib_models"
CLIPT_PATH = pathlib.Path(
    os.environ.get(
        "CLIPT_PATH",
        HERE.parent / "ICM/tools/DiffLens/evaluation/CLIP-T/clip_text_score.py",
    )
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _norm(v, target):
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64) - target))


def _crop_shard(job):
    """Crop the faces of ``paths`` with the face detector of DiffLens (one worker)."""
    paths, out_dir = job
    crop = _load("difflens_crop", HERE / "evaluation/crop_face/crop.py")
    detector = crop.EfficientFaceDetector(
        cnn_model_path=str(DLIB_MODELS / "mmod_human_face_detector.dat"),
        landmark_model_path=str(DLIB_MODELS / "shape_predictor_5_face_landmarks.dat"),
        batch_size=8,
    )
    return len(detector.detect_and_crop(image_paths=paths, output_dir=out_dir))


def crop_run(run, workers):
    """Crop the faces of each prompt folder into ``cropped_images/<prompt>/``."""
    for img_dir in sorted((run / "splitted_images").glob("*/")):
        out_dir = run / "cropped_images" / img_dir.name
        if out_dir.is_dir() and any(out_dir.iterdir()):
            raise SystemExit(
                f"{out_dir} is not empty: crops of another pass would mix with the new "
                "ones. Empty it, or run again with --skip_crop to reuse these crops."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(str(x) for x in img_dir.glob("*.png"))
        jobs = [
            (s, str(out_dir)) for s in (paths[i::workers] for i in range(workers)) if s
        ]
        with Pool(len(jobs)) as pool:
            n_faces = sum(pool.map(_crop_shard, jobs))
        print(
            f"[crop] {img_dir.name}: {len(paths)} images -> {n_faces} faces",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="./samples_original")
    p.add_argument(
        "--original", default="./samples_original", help="reference run for CLIP-I"
    )
    p.add_argument("--out", default=None)
    p.add_argument("--skip_clip_i", action="store_true")
    p.add_argument("--skip_clip_t", action="store_true")
    p.add_argument("--skip_crop", action="store_true")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = p.parse_args()

    run = pathlib.Path(args.run).resolve()
    orig = pathlib.Path(args.original).resolve()
    prompts = sorted(d.name for d in (run / "splitted_images").iterdir() if d.is_dir())

    if not args.skip_crop:
        crop_run(run, args.workers)

    # CLIP-T loads CLIP at import time: we do it before the chdir, and only once.
    if not args.skip_clip_t and not CLIPT_PATH.is_file():
        print(
            f"CLIP-T skipped: no ICM script at {CLIPT_PATH} (see README.md)", flush=True
        )
        args.skip_clip_t = True
    clip_t = None if args.skip_clip_t else _load("icm_clip_t", CLIPT_PATH)
    clip_i = (
        None
        if args.skip_clip_i
        else _load("difflens_clip_i", HERE / "evaluation/CLIP/clip_image_score.py")
    )

    os.chdir(FAIRFACE_DIR)  # the .pt path is hard-coded in their scripts
    gender = _load("difflens_gender", FAIRFACE_DIR / "gender.py")
    race = _load("difflens_race", FAIRFACE_DIR / "race.py")
    age = _load("difflens_age", FAIRFACE_DIR / "age.py")

    report = {}
    for name in prompts:
        img_dir = run / "splitted_images" / name
        crop_dir = run / "cropped_images" / name
        crops = sorted(str(x) for x in crop_dir.glob("*.jpg"))
        n_images = len(list(img_dir.glob("*.png")))
        stems = {pathlib.Path(c).name.rsplit("_face_", 1)[0] for c in crops}

        g = gender.gender_classifier(crops)  # (N, 2) softmax male/female
        r, _ = race.race_classifier(crops)  # (N, 7) softmax
        a, _, fd_age = age.age_classifier(crops)  # (N, 9) softmax + their 3-class FD

        r_mean = r.mean(axis=0)
        mapped = np.array([r_mean[0], r_mean[1], r_mean[3] + r_mean[4], r_mean[5]])

        row = {
            "n_images": n_images,
            "n_images_with_face": len(stems),
            "n_faces": len(crops),
            "no_face_rate": round(1.0 - len(stems) / max(n_images, 1), 4),
            "extra_face_rate": round(len(crops) / max(len(stems), 1) - 1.0, 4),
            "male_frac": round(float((g[:, 0] > 0.5).mean()), 4),
            "FD_gender": round(_norm(g.mean(axis=0), 0.5), 4),
            "FD_gender_hard_argmax": round(
                _norm(np.bincount(g.argmax(axis=1), minlength=2) / len(g), 0.5), 4
            ),
            "FD_age": round(float(fd_age), 4),
            "FD_race_theirs": round(_norm(r_mean[:4], 0.25), 4),
            "FD_race_mapped": round(_norm(mapped, 0.25), 4),
        }
        if clip_t is not None:
            row["CLIP_T"] = round(
                clip_t.calculate_clip_t_score(str(img_dir), name.replace("-", " ")), 4
            )
        if clip_i is not None:
            row["CLIP_I"] = round(
                clip_i.calculate_clip_similarity(
                    str(orig / "splitted_images" / name), str(img_dir)
                ),
                4,
            )
        report[name] = row
        print(name, json.dumps(row), flush=True)

    keys = [
        k
        for k in report[prompts[0]]
        if k.startswith(("FD_", "CLIP_", "male_", "no_face", "extra_"))
    ]
    report["__mean_over_prompts__"] = {
        k: round(float(np.mean([report[p][k] for p in prompts])), 4) for k in keys
    }
    report["__totals__"] = {
        k: int(sum(report[p][k] for p in prompts))
        for k in ("n_images", "n_images_with_face", "n_faces")
    }

    out = pathlib.Path(args.out) if args.out else run / "metrics.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["__mean_over_prompts__"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

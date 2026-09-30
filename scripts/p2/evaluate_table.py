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

"""FD and FID of several methods on gender, race and eyeglasses, as one table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from debias_anything.evaluators.celebahq import image_inventory
from debias_anything.paths import REPO_ROOT as REPO
from debias_anything.paths import portable, resolve

HERE = Path(__file__).resolve().parent
REAL_IMAGES = REPO / "data/celebahq/CelebAMask-HQ/CelebA-HQ-img"
ANNOTATIONS = REPO / "data/celebahq/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
FAIRFACE = REPO / "checkpoints/fairface/res34_fair_align_multi_7_20190809.pt"
# ResNet-18 evaluators of Appendix B.5, trained by train_evaluator.py.
EVALUATORS = REPO / "checkpoints/celebahq_evaluators"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=4,
        action="append",
        required=True,
        metavar=("METHOD", "GENDER_DIR", "RACE_DIR", "EYEGLASSES_DIR"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=REPO / "result_metrics/p2_balancing_act_table",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument(
        "--gender-classifier-checkpoint",
        type=Path,
        default=EVALUATORS / "gender/best.pt",
        help="Independently trained CelebA-HQ ResNet-18 Gender evaluator.",
    )
    parser.add_argument(
        "--race-classifier-checkpoint",
        type=Path,
        default=None,
        help=(
            "Independent Race torchvision ResNet-18 checkpoint. If omitted, "
            "retain the previous public FairFace White/Black proxy."
        ),
    )
    parser.add_argument(
        "--race-fairface-checkpoint",
        type=Path,
        default=FAIRFACE,
        help=(
            "FairFace ResNet-34 checkpoint used when no independent Race "
            "ResNet-18 checkpoint is supplied. Only its White and Black logits "
            "are retained."
        ),
    )
    parser.add_argument(
        "--eyeglasses-classifier-checkpoint",
        type=Path,
        default=EVALUATORS / "eyeglasses/best.pt",
        help="Independently trained CelebA-HQ ResNet-18 Eyeglasses evaluator.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def ensure_references(args: argparse.Namespace) -> dict[str, Path]:
    references = {
        "Gender": REPO
        / f"data/celebahq/reference_gender_balanced_5000_seed{args.reference_seed}",
        "Race": REPO / "data/celebahq/reference_race_clip_vit_b32_ranked_balanced_5000",
        "Eyeglasses": REPO
        / f"data/celebahq/reference_eyeglasses_balanced_5000_seed{args.reference_seed}",
    }
    for attribute in ("Gender", "Eyeglasses"):
        reference = references[attribute]
        if not (reference / "manifest.json").is_file():
            annotation_name = "Male" if attribute == "Gender" else "Eyeglasses"
            run(
                [
                    sys.executable,
                    str(HERE / "build_balanced_reference.py"),
                    "--images-dir",
                    str(REAL_IMAGES),
                    "--annotations",
                    str(ANNOTATIONS),
                    "--attribute",
                    annotation_name,
                    "--output-dir",
                    str(reference),
                    "--num-images",
                    "5000",
                    "--seed",
                    str(args.reference_seed),
                ]
            )

    race_reference = references["Race"]
    if not (race_reference / "manifest.json").is_file():
        run(
            [
                sys.executable,
                str(HERE / "build_race_reference.py"),
                "--images-dir",
                str(REAL_IMAGES),
                "--output-dir",
                str(race_reference),
                "--num-images",
                "5000",
                "--clip-model",
                "ViT-B/32",
                "--prompt",
                "a black person",
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
            ]
        )
    return references


def same_path(recorded: str, path: Path) -> bool:
    """Whether ``recorded``, relative to the repository, designates ``path``."""
    return resolve(Path(recorded).expanduser()).resolve() == path.resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "method"


def hard_fd(metrics: dict[str, Any]) -> float:
    """FD of the hard predictions: ``sqrt(2)·|p₀ − 0.5|`` for a binary attribute."""
    return math.dist(metrics["hard_probs"], metrics["target_probs"])


def display_metric(value: float | None, decimals: int) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


def evaluate(
    method: str,
    attribute: str,
    images_dir: Path,
    reference: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    for checkpoint in (
        args.gender_classifier_checkpoint,
        args.race_classifier_checkpoint,
        args.eyeglasses_classifier_checkpoint,
    ):
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if (
        args.race_classifier_checkpoint is None
        and not args.race_fairface_checkpoint.is_file()
    ):
        raise FileNotFoundError(args.race_fairface_checkpoint)
    race_configuration = (
        (args.race_classifier_checkpoint, "resnet18", "imagenet")
        if args.race_classifier_checkpoint is not None
        else (args.race_fairface_checkpoint, "fairface_white_black", "fairface")
    )
    configurations = {
        "Gender": (args.gender_classifier_checkpoint, "resnet18", "imagenet"),
        "Race": race_configuration,
        "Eyeglasses": (args.eyeglasses_classifier_checkpoint, "resnet18", "imagenet"),
    }
    checkpoint, architecture, preprocess = configurations[attribute]
    result_path = output_dir / f"{safe_name(method)}_{attribute.lower()}.json"
    cached_result = (
        json.loads(result_path.read_text()) if result_path.is_file() else None
    )
    current_inventory = image_inventory(images_dir)
    cache_matches = (
        cached_result is not None
        and same_path(cached_result["images_dir"], images_dir)
        and same_path(cached_result["fid_reference_dir"], reference)
        and same_path(cached_result["classifier_checkpoint"], checkpoint)
        and cached_result["classifier_architecture"] == architecture
        and cached_result.get("classifier_preprocess") == preprocess
        and cached_result.get("image_inventory") == current_inventory
    )
    if args.force or not cache_matches:
        run(
            [
                sys.executable,
                str(HERE / "evaluate.py"),
                "--images-dir",
                str(images_dir.resolve()),
                "--fid-reference-dir",
                str(reference),
                "--classifier-checkpoint",
                str(checkpoint),
                "--classifier-architecture",
                architecture,
                "--preprocess",
                preprocess,
                "--attribute",
                attribute,
                "--target-probs",
                "0.5,0.5",
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--fid-batch-size",
                str(args.fid_batch_size),
                "--workers",
                str(args.workers),
                "--out",
                str(result_path),
            ]
        )
    return json.loads(result_path.read_text())


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Gender FD ↓ | Gender FID ↓ | Race FD ↓ | Race FID ↓ | Eyeglasses FD ↓ | Eyeglasses FID ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {gender_fd} | {gender_fid} | {race_fd} | "
            "{race_fid} | {eyeglasses_fd} | {eyeglasses_fid} |".format(
                method=row["method"],
                gender_fd=display_metric(row["gender_fd"], 4),
                gender_fid=display_metric(row["gender_fid"], 2),
                race_fd=display_metric(row["race_fd"], 4),
                race_fid=display_metric(row["race_fid"], 2),
                eyeglasses_fd=display_metric(row["eyeglasses_fd"], 4),
                eyeglasses_fid=display_metric(row["eyeglasses_fid"], 2),
            )
        )
    lines += [
        "",
        "> Race is the binary CLIP-similarity proxy (`low_CLIP_similarity` / "
        "`high_CLIP_similarity`), not a demographic ground-truth annotation.",
    ]
    return "\n".join(lines) + "\n"


def latex_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        method = row["method"].replace("_", r"\_")
        body.append(
            f"{method} & {display_metric(row['gender_fd'], 4)} & {display_metric(row['gender_fid'], 2)} "
            f"& {display_metric(row['race_fd'], 4)} & {display_metric(row['race_fid'], 2)} "
            f"& {display_metric(row['eyeglasses_fd'], 4)} & {display_metric(row['eyeglasses_fid'], 2)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"& \multicolumn{2}{c}{Gender} & \multicolumn{2}{c}{Race} & \multicolumn{2}{c}{Eyeglasses} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Method & FD $\downarrow$ & FID $\downarrow$ & FD $\downarrow$ & FID $\downarrow$ & FD $\downarrow$ & FID $\downarrow$ \\",
            r"\midrule",
            *body,
            r"\midrule",
            r"\multicolumn{7}{l}{\footnotesize Race: binary CLIP-similarity proxy, not demographic ground truth.} \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.fid_batch_size < 1 or args.workers < 0:
        raise ValueError("Batch sizes must be positive and --workers non-negative")

    prefix = args.output_prefix.resolve()
    output_dir = prefix.parent / f"{prefix.name}_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    references = ensure_references(args)

    race_evaluator_note = (
        "Race uses the supplied independently trained CelebA-HQ ResNet-18 "
        f"checkpoint: {portable(args.race_classifier_checkpoint)}."
        if args.race_classifier_checkpoint is not None
        else "Race uses the public FairFace White/Black proxy."
    )
    gender_evaluator_note = (
        "Gender uses the independent ResNet-18 checkpoint: "
        f"{portable(args.gender_classifier_checkpoint)}."
    )
    eyeglasses_evaluator_note = (
        "Eyeglasses uses the independent ResNet-18 checkpoint: "
        f"{portable(args.eyeglasses_classifier_checkpoint)}."
    )
    combined: dict[str, Any] = {
        "protocol": "balancing_act_reconstructed_evaluation_v1",
        "caveat": (
            "Race D_ref follows Balancing Act's CLIP ranking protocol (OpenAI CLIP "
            "ViT-B/32, prompt 'a black person') and is a binary proxy, not a "
            "demographic ground-truth annotation. "
            + gender_evaluator_note
            + " "
            + race_evaluator_note
            + " "
            + eyeglasses_evaluator_note
            + " None of these "
            "are Balancing Act's unreleased evaluation weights."
        ),
        "methods": {},
    }
    table_rows: list[dict[str, Any]] = []
    for method, gender_dir, race_dir, eyeglasses_dir in args.run:
        directories = {
            "Gender": None if gender_dir == "-" else Path(gender_dir),
            "Race": None if race_dir == "-" else Path(race_dir),
            "Eyeglasses": None if eyeglasses_dir == "-" else Path(eyeglasses_dir),
        }
        metrics = {
            attribute: (
                None
                if directory is None
                else evaluate(
                    method,
                    attribute,
                    directory,
                    references[attribute],
                    output_dir,
                    args,
                )
            )
            for attribute, directory in directories.items()
        }
        combined["methods"][method] = metrics
        table_rows.append(
            {
                "method": method,
                "race_label_definition": "binary CLIP-similarity proxy; not demographic ground truth",
                **{
                    f"{attribute.lower()}_{metric}": (
                        None
                        if metrics[attribute] is None
                        else value(metrics[attribute])
                    )
                    for attribute in directories
                    for metric, value in (("fd", hard_fd), ("fid", lambda m: m["fid"]))
                },
            }
        )

    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(combined, indent=2) + "\n")
    prefix.with_suffix(".md").write_text(markdown_table(table_rows))
    prefix.with_suffix(".tex").write_text(latex_table(table_rows))
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    print("\n" + markdown_table(table_rows), end="")
    print(f"JSON/CSV/Markdown/LaTeX written under {prefix}.*")


if __name__ == "__main__":
    main()

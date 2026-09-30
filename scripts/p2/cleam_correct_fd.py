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

"""Add CLEAM-corrected FD columns to a results table (Teo et al., NeurIPS 2023)."""

import argparse
import csv
import math

from debias_anything.metrics.correct_classifier import CLEAM

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--table", required=True, help="results table with <attribute>_fd columns"
)
parser.add_argument(
    "--classifiers",
    required=True,
    help="CSV with attribute, recall_class_0, recall_class_1",
)
parser.add_argument("--out", required=True)
parser.add_argument(
    "--majority-class",
    type=int,
    choices=(0, 1),
    default=0,
    help="class predicted in excess by the classifier in every cell (0 = class_0 column of the CSV)",
)
args = parser.parse_args()

with open(args.classifiers, newline="") as handle:
    classifiers = {r["attribute"]: r for r in csv.DictReader(handle)}

with open(args.table, newline="") as handle:
    delimiter = ";" if ";" in handle.readline() else ","
    handle.seek(0)
    reader = csv.DictReader(handle, delimiter=delimiter)
    rows = list(reader)
    columns = list(reader.fieldnames or [])

sign = 1 if args.majority_class == 0 else -1
out_columns = []
for column in columns:
    out_columns.append(column)
    if not column.endswith("_fd"):
        continue
    classifier = classifiers[column.removesuffix("_fd")]
    cleam = CLEAM(
        float(classifier["recall_class_0"]), float(classifier["recall_class_1"])
    )
    out_columns.append(f"{column}_cleam")
    for row in rows:
        if row[column] == "":  # cell skipped with "-" in evaluate_table.py
            row[f"{column}_cleam"] = ""
            continue
        p_star_0 = cleam.muCLEAM(0.5 + sign * float(row[column]) / math.sqrt(2))
        row[f"{column}_cleam"] = math.sqrt(2) * abs(p_star_0 - 0.5)

with open(args.out, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=out_columns, delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {args.out}")

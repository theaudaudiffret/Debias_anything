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

"""Metrics of a generated set against a real one, with cached features."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cleanfid.features import build_feature_extractor as build_clean_fid_extractor
from cleanfid.resize import build_resizer as build_clean_fid_resizer
from pytorch_fid import inception

from ..evaluators.celeba import CelebaClassifier
from ..evaluators.celebahq import file_sha256
from ..paths import CHECKPOINT_DIR
from .correct_classifier import CLEAM
from .fairness import class_deviation_table, fairness_discrepancy, predict_classes
from .features import (
    extract_clean_fid_features,
    extract_inception_features,
    extract_sfid_features,
)
from .fid import compute_fid, monge_inception_distance
from .knn_distance import compute_knn_distance
from .precision_recall import compute_precision_recall_density_coverage
from .sfid import compute_sfid
from .vendi import compute_vendi_score

# Sensitive-attribute classifiers (SAC) that count the attribute on generated CelebA images,
# trained by ``scripts/celeba/train_classifier.py``; their per-class accuracies on the validation
# split (the CLEAM alphas) are appended by the same script to ``CSV_ALPHA``, one row per trained
# checkpoint, identified by its SHA-256.
CKPT_CLASSIFIER_CELEBA = str(CHECKPOINT_DIR / "celeba_gender_classifier.pt")
CKPT_CLASSIFIER_CELEBA_EYEGLASSES = str(
    CHECKPOINT_DIR / "celeba_eyeglasses_classifier.pt"
)
CSV_ALPHA = str(CHECKPOINT_DIR / "celeba_classifier_characteristics.csv")


@dataclass
class SAC:
    """Sensitive-attribute classifier, its predictions, CLEAM and real labels."""

    classifier: nn.Module
    preds: np.ndarray
    p_hat_0_per_batch: np.ndarray | None = None
    cleam: CLEAM | None = None
    targets: torch.Tensor | None = None


# The trained CelebA SACs: measured name -> (checkpoint, CLEAM calibration row,
# CelebA attribute used as ground truth). All of them are always evaluated on CelebA.
CELEBA_SACS = {
    "gender": (CKPT_CLASSIFIER_CELEBA, "celeba_gender_classifier", "Male"),
    "eyeglasses": (
        CKPT_CLASSIFIER_CELEBA_EYEGLASSES,
        "celeba_eyeglasses_classifier",
        "Eyeglasses",
    ),
}


class Metrics:
    def __init__(
        self,
        images: torch.Tensor,
        generated: torch.Tensor,
        targets: torch.Tensor,
        device: torch.device | str = "cpu",
        n_batches: int = 10,
        batch_size: int = 64,
        images_val: torch.Tensor | None = None,
        targets_val: torch.Tensor | None = None,
        sac_targets: dict[str, torch.Tensor] | None = None,
    ):
        self.device = torch.device(device)
        self._inception = inception.InceptionV3().to(self.device).eval()
        self._clean_fid_model = build_clean_fid_extractor(
            mode="clean", device=self.device, use_dataparallel=False
        )
        self._clean_fid_resizer = build_clean_fid_resizer(mode="clean")
        self._inception_sfid = (
            inception.InceptionV3(output_blocks=[2]).to(self.device).eval()
        )
        self.images = images
        self.images_val = images_val
        self.targets_val = targets_val
        self.generated = generated
        self.targets = targets
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.classifier: torch.nn.Module | None = None
        self.cleam: CLEAM | None = None
        self._preds: np.ndarray | None = None
        self._p_hat_0_per_batch: np.ndarray | None = None
        # One SAC per measured attribute: {name: SAC}. No primary/secondary
        # hierarchy — the fairness metrics are computed for each of them.
        self.sac_targets = sac_targets or {}
        self.sacs: dict[str, SAC] = {}
        self._feats_real: np.ndarray | None = None
        self._feats_gen: np.ndarray | None = None
        self._feats_val: np.ndarray | None = None
        self._feats_real_clean: np.ndarray | None = None
        self._feats_gen_clean: np.ndarray | None = None
        self._feats_real_sfid: np.ndarray | None = None
        self._feats_gen_sfid: np.ndarray | None = None

        for name, (ckpt, csv_name, attr) in CELEBA_SACS.items():
            clf = self._load_classifier(ckpt, CelebaClassifier)
            preds, p_hat_0 = self._predict(clf)
            self.sacs[name] = SAC(
                clf,
                preds,
                p_hat_0,
                self._load_cleam(csv_name, ckpt),
                self.sac_targets.get(attr),
            )
        # the first SAC gives the partition of the per-class metrics
        sac = next(iter(self.sacs.values()))
        self.classifier, self.cleam = sac.classifier, sac.cleam
        self._preds, self._p_hat_0_per_batch = sac.preds, sac.p_hat_0_per_batch

    # ------- setup helpers -------

    def _predict(self, clf: nn.Module) -> tuple[np.ndarray, np.ndarray | None]:
        return predict_classes(
            clf, self.generated, self.batch_size, self.device, track_p_hat_0=True
        )

    @torch.no_grad()
    def _load_classifier(self, ckpt_path: str, cls: type) -> torch.nn.Module:
        classifier = cls().to(self.device)
        classifier.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        classifier.eval()
        return classifier

    def _load_cleam(self, classifier_name: str, checkpoint: str) -> CLEAM:
        """CLEAM of a classifier, from the alphas measured on the same checkpoint."""
        df = pd.read_csv(CSV_ALPHA)
        sha = file_sha256(Path(checkpoint))
        rows = df[
            (df["classifier_name"] == classifier_name)
            & (df["sac_checkpoint_hash"] == sha)
        ]
        if rows.empty:
            raise ValueError(
                f"No row {classifier_name!r} with the SHA-256 of {checkpoint} in {CSV_ALPHA}: "
                "run scripts/celeba/train_classifier.py for this attribute."
            )
        row = rows.iloc[-1]
        alpha0 = float(row["proba_0"])
        alpha1 = float(row["proba_1"])
        if {"n_samples_class0", "n_samples_class1"}.issubset(df.columns):
            n0_raw = row["n_samples_class0"]
            n1_raw = row["n_samples_class1"]
            if pd.isna(n0_raw) or pd.isna(n1_raw):
                warnings.warn(
                    "CSV α: n_samples_class{0,1} unknown — α uncertainty cannot be assessed."
                )
            else:
                n0, n1 = int(n0_raw), int(n1_raw)  # type: ignore[arg-type]
                if min(n0, n1) < 200:
                    warnings.warn(
                        f"α measured on only ({n0}, {n1}) samples — "
                        "CLEAM CI underestimates uncertainty on α itself."
                    )
        else:
            warnings.warn(
                "CSV α: columns n_samples_class{0,1} missing — α uncertainty cannot be assessed."
            )
        return CLEAM(alpha0, alpha1, self.n_batches)

    def _extract(self, images: torch.Tensor) -> np.ndarray:
        return extract_inception_features(images, self._inception, self.device)

    def _get_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Inception features of the real and generated sets (cached)."""
        if self._feats_real is None or self._feats_gen is None:
            self._feats_real = self._extract(self.images)
            self._feats_gen = self._extract(self.generated)
        return self._feats_real, self._feats_gen

    def _extract_clean_fid(self, images: torch.Tensor) -> np.ndarray:
        return extract_clean_fid_features(
            images, self._clean_fid_model, self._clean_fid_resizer, self.device
        )

    def _get_clean_fid_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Clean-FID features of the real and generated sets (cached)."""
        if self._feats_real_clean is None or self._feats_gen_clean is None:
            self._feats_real_clean = self._extract_clean_fid(self.images)
            self._feats_gen_clean = self._extract_clean_fid(self.generated)
        return self._feats_real_clean, self._feats_gen_clean

    def _extract_sfid(self, images: torch.Tensor) -> np.ndarray:
        return extract_sfid_features(images, self._inception_sfid, self.device)

    def _get_sfid_features(self) -> tuple[np.ndarray, np.ndarray]:
        """sFID features of the real and generated sets (cached)."""
        if self._feats_real_sfid is None or self._feats_gen_sfid is None:
            self._feats_real_sfid = self._extract_sfid(self.images)
            self._feats_gen_sfid = self._extract_sfid(self.generated)
        return self._feats_real_sfid, self._feats_gen_sfid

    def _get_features_val(self) -> np.ndarray:
        """Inception features of the validation set (cached)."""
        assert self.images_val is not None, "images_val not provided to Metrics"
        if self._feats_val is None:
            self._feats_val = self._extract(self.images_val)
        return self._feats_val

    def _resolve_num_classes(self, num_classes: int | None) -> int:
        return num_classes if num_classes is not None else 2

    # ------- public metrics API -------

    def fid(self) -> float:
        feats_r, feats_g = self._get_features()
        return compute_fid(feats_r, feats_g)

    def fid_val(self) -> float:
        _, feats_g = self._get_features()
        feats_val = self._get_features_val()
        return compute_fid(feats_val, feats_g)

    def monge(self, rng_seed: int = 42, n_projections: int = 1000) -> float:
        """Monge Inception Distance (MIND); lower is better."""
        feats_r, feats_g = self._get_features()
        return monge_inception_distance(
            feats_g, feats_r, rng_seed=rng_seed, n_projections=n_projections
        )

    def fid_clean(self) -> float:
        """Clean-FID (Parmar et al., 2022)."""
        feats_r, feats_g = self._get_clean_fid_features()
        return compute_fid(feats_r, feats_g)

    def sfid(self) -> float:
        """Spatial FID (Dhariwal & Nichol, 2021)."""
        feats_r, feats_g = self._get_sfid_features()
        return compute_sfid(feats_r, feats_g)

    def precision_recall_density_coverage(self, k: int = 5) -> tuple[float, float]:
        feats_r, feats_g = self._get_features()
        return compute_precision_recall_density_coverage(
            feats_r, feats_g, k=k, device=self.device
        )

    def knn_distance(self, k: int = 5) -> float:
        feats_r, feats_g = self._get_features()
        return compute_knn_distance(feats_r, feats_g, k=k, device=self.device)

    def vendi(self) -> float:
        """Vendi score of the generated set."""
        _, feats_g = self._get_features()
        return compute_vendi_score(feats_g)

    def vendi_real(self) -> float:
        """Vendi score of the real set."""
        feats_r, _ = self._get_features()
        return compute_vendi_score(feats_r)

    def fairness_discrepancy(
        self,
        target: np.ndarray | None = None,
        num_classes: int | None = None,
    ) -> float:
        num_classes = self._resolve_num_classes(num_classes)
        assert self._preds is not None, "SAC predictions unavailable"
        return fairness_discrepancy(
            self._preds,
            num_classes,
            cleam=self.cleam,
            target=target,
        )

    # ------- intra-class metrics -------

    def _split_features_by_class(
        self,
        num_classes: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Real features by label and generated features by predicted class."""
        assert self._preds is not None, "SAC predictions unavailable"
        feats_r, feats_g = self._get_features()
        targets_np = self.targets.long().cpu().numpy()
        feats_r_by_c = {c: feats_r[targets_np == c] for c in range(num_classes)}
        feats_g_by_c = {c: feats_g[self._preds == c] for c in range(num_classes)}
        return feats_r_by_c, feats_g_by_c

    def _split_features_val_by_class(self, num_classes: int) -> dict[int, np.ndarray]:
        assert self.targets_val is not None, "targets_val not provided to Metrics"
        feats_val = self._get_features_val()
        tv = self.targets_val.long().cpu().numpy()
        return {c: feats_val[tv == c] for c in range(num_classes)}

    @staticmethod
    def _per_class_map(num_classes, dicts, min_samples, fn, nan_value):
        """``fn`` per class; ``nan_value`` for classes under ``min_samples`` points."""
        out = {}
        for c in range(num_classes):
            parts = [d[c] for d in dicts]
            out[c] = (
                fn(*parts)
                if all(p.shape[0] >= min_samples for p in parts)
                else nan_value
            )
        return out

    def fid_per_class(self, num_classes: int | None = None) -> dict[int, float]:
        num_classes = self._resolve_num_classes(num_classes)
        fr, fg = self._split_features_by_class(num_classes)
        return self._per_class_map(num_classes, [fr, fg], 2, compute_fid, float("nan"))

    def fid_val_per_class(self, num_classes: int | None = None) -> dict[int, float]:
        num_classes = self._resolve_num_classes(num_classes)
        _, fg = self._split_features_by_class(num_classes)
        fv = self._split_features_val_by_class(num_classes)
        return self._per_class_map(num_classes, [fv, fg], 2, compute_fid, float("nan"))

    def precision_recall_density_coverage_per_class(
        self,
        num_classes: int | None = None,
        k: int = 5,
    ) -> dict[int, tuple[float, float, float, float]]:
        num_classes = self._resolve_num_classes(num_classes)
        fr, fg = self._split_features_by_class(num_classes)
        # PRDC needs at least k+1 points on each side (k-NN excluding self).
        return self._per_class_map(
            num_classes,
            [fr, fg],
            k + 1,
            lambda a, b: compute_precision_recall_density_coverage(
                a, b, k=k, device=self.device
            ),
            (float("nan"),) * 4,
        )

    def vendi_per_class(self, num_classes: int | None = None) -> dict[int, float]:
        num_classes = self._resolve_num_classes(num_classes)
        _, fg = self._split_features_by_class(num_classes)
        return self._per_class_map(
            num_classes, [fg], 2, compute_vendi_score, float("nan")
        )

    def vendi_real_per_class(self, num_classes: int | None = None) -> dict[int, float]:
        """Vendi score of the real set per class."""
        num_classes = self._resolve_num_classes(num_classes)
        fr, _ = self._split_features_by_class(num_classes)
        return self._per_class_map(
            num_classes, [fr], 2, compute_vendi_score, float("nan")
        )

    def class_deviation_table(self, num_classes: int | None = None) -> pd.DataFrame:
        num_classes = self._resolve_num_classes(num_classes)
        assert self._preds is not None, "SAC predictions unavailable"
        return class_deviation_table(
            self._preds,
            self.targets,
            num_classes,
            cleam=self.cleam,
            p_hat_0_per_batch=self._p_hat_0_per_batch,
        )

    # ------- per-SAC fairness -------

    def fairness_discrepancy_for(
        self, name: str, target: np.ndarray | None = None
    ) -> float:
        sac = self.sacs[name]
        return fairness_discrepancy(sac.preds, 2, cleam=sac.cleam, target=target)

    def class_deviation_table_for(self, name: str) -> pd.DataFrame:
        sac = self.sacs[name]
        if sac.targets is None:
            raise ValueError(
                f"SAC {name!r}: ground-truth column of the real set missing — "
                "pass it via sac_targets"
            )
        return class_deviation_table(
            sac.preds,
            sac.targets,
            2,
            cleam=sac.cleam,
            p_hat_0_per_batch=sac.p_hat_0_per_batch,
        )

# Debias Anything: Fairness with Diversity without Supervision in Diffusion Models

Official implementation of the paper _Debias Anything: Fairness with Diversity without Supervision in Diffusion Models_ (Théau d'Audiffret, Mariia Vladimirova, Jean-Yves Franceschi).

Debias Anything sets the proportions of a sensitive attribute, described by a
few sentences, in the samples of a pretrained diffusion model, with no attribute label and no
retraining of the generator. An adapter, trained once per generator, maps the h-space of the frozen
denoiser to the embedding space of SigLIP 2 (Section 4.1); pairs of sentences give the attribute
directions, and an optimal batch assignment guides each image towards a value (Section 4.2); a
second term, computed with the same adapter, restores the diversity that guidance removes
(Section 4.3).

## [Preprint](https://arxiv.org/abs/2610.01815)

## Contents

The code is split in two: `src/debias_anything/` is the library, installed by `uv sync` and
imported as `debias_anything`; `scripts/` holds the entry points, which only read their config,
call the library and write their outputs. Both, like `conf/`, are organised by generator: CelebA
64×64 (the EDM model trained for the paper), CelebA-HQ (P2) and Stable Diffusion 1.5.

| | |
|---|---|
| `src/debias_anything/adapters/` | the adapters h-space → SigLIP 2 (Section 4.1): `HSpaceToSigLIP` (CelebA, P2), `MultiBlockHSpaceToSigLIP` (Stable Diffusion 1.5), and their loss |
| `src/debias_anything/hspace/` | reading the h-space: bottleneck hooks and truncated forward passes, the multi-block readers of Stable Diffusion |
| `src/debias_anything/guidance/` | the fairness term and its batch assignment (Section 4.2, `assignment.py`) and the diversity terms (Section 4.3), for each generator: `edm.py` (CelebA: `BatchedTextGuidance`, `PerturbationProjTextGuidance` = Eq. 12, `SigLIPMSTextGuidance` = Appendix A.3, `SGMSTextGuidance`), `p2.py` (guided DDIM of P2), `sd15.py` (`DebiasAnythingSD15` pipeline) |
| `src/debias_anything/models/` | the three generators: `edm.py` (CelebA 64×64), `p2.py`, `sd15.py`, and `third_party/guided_diffusion/`, the U-Net of OpenAI's [guided-diffusion](https://github.com/openai/guided-diffusion) (MIT license, unmodified apart from a copyright header) that P2 uses |
| `src/debias_anything/metrics/` | FID, sFID, MIND, precision/recall/density/coverage, Vendi, and the CLEAM-corrected fairness discrepancy |
| `src/debias_anything/evaluators/` | the classifiers that count the attributes on generated images: CelebA CNNs, and the CelebA-HQ evaluators of the Balancing Act protocol |
| `src/debias_anything/{data,samplers,siglip}.py` | datasets, EDM samplers, SigLIP 2 |
| `scripts/{celeba,p2,sd15}/` | the entry points (training, sampling, evaluation), and one `.sh` per table of the paper |
| `conf/{celeba,p2,sd15}/` | the Hydra configs of the entry points, with the settings of the paper |
| `notebooks/` | sample grids of CelebA-HQ and Stable Diffusion 1.5, see [Notebooks](#notebooks) |
| `eval/difflens/` | how the DiffLens evaluation was run: their code has no license and is not redistributed |

The entry points that train or sample (`train_*`, `generate`, `sample_and_evaluate`,
`build_adapter_cache`) are Hydra apps: `conf/<generator>/<name>.yaml` holds their settings, and
any of them can be overridden on the command line (`key=value`). The tools of the evaluation
protocols (`evaluate*`, `build_*reference`, `cleam_correct_fd`) take paths as
`--options`.

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

`pyproject.toml` installs torch from the CUDA 13.0 wheels used for the paper (torch 2.13,
diffusers 0.39, transformers 5.14); change the `pytorch-cu130` index for another CUDA version.
Every script is run from the root of the repository, and every path below, in a config or on the
command line, is relative to it.

## Checkpoints and data

The checkpoints trained for the paper are on the Hugging Face hub,
[theau12345/debias-anything](https://huggingface.co/theau12345/debias-anything). Download them into
`checkpoints/`:

```bash
uv run hf download theau12345/debias-anything --local-dir checkpoints
```

and add the two third-party checkpoints, which are not redistributed:

| file | what it is | source |
|---|---|---|
| `celeba_edm_64.pth` | EDM model trained on CelebA 64×64 (Appendix B.1) | [HF](https://huggingface.co/theau12345/debias-anything) |
| `celeba_adapter.pth` | adapter of the CelebA model | [HF](https://huggingface.co/theau12345/debias-anything) |
| `celeba_gender_classifier.pt`, `celeba_eyeglasses_classifier.pt` | classifiers that count the attributes on CelebA (B.5) | [HF](https://huggingface.co/theau12345/debias-anything) |
| `celeba_classifier_characteristics.csv` | their per-class validation accuracies, the CLEAM alphas, one row per checkpoint (SHA-256) | [HF](https://huggingface.co/theau12345/debias-anything) |
| `celebahq_p2_adapter.pt` | adapter of P2 | [HF](https://huggingface.co/theau12345/debias-anything) |
| `celebahq_evaluators/{gender,eyeglasses,race}/` | ResNet-18 evaluators of CelebA-HQ (B.5) | [HF](https://huggingface.co/theau12345/debias-anything) |
| `celebahq_evaluators/recalls.{csv,json}` | their per-class validation recalls, the CLEAM alphas of Table 2 | [HF](https://huggingface.co/theau12345/debias-anything), or `scripts/p2/evaluate_evaluators.py` |
| `sd15_adapter.pth` | adapter of Stable Diffusion 1.5 | [HF](https://huggingface.co/theau12345/debias-anything) |
| `celebahq_p2.pt` | P2 CelebA-HQ model (Choi et al., 2022) | [P2 weighting](https://github.com/jychoi118/P2-weighting) |
| `fairface/res34_fair_align_multi_7_20190809.pt` | FairFace classifier (Kärkkäinen & Joo, 2021) | [FairFace](https://github.com/dchen236/FairFace) |

Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`) and SigLIP 2
(`google/siglip2-base-patch16-224`) are downloaded from the Hugging Face hub.

Datasets, in `data/`:

- `data/celeba/`: CelebA in the torchvision layout (`img_align_celeba/`, `list_attr_celeba.txt`,
  `list_eval_partition.txt`, ...);
- `data/celebahq/CelebAMask-HQ/`: CelebAMask-HQ (`CelebA-HQ-img/`,
  `CelebAMask-HQ-attribute-anno.txt`), for the evaluation on CelebA-HQ;
- `data/celeba_hq/`: CelebA-HQ 256×256 from the Hugging Face hub, for the P2 adapter, written by
  `uv run python scripts/p2/build_celeba_hq.py`.

The scripts write the derived sets themselves: `data/minority_celeba/` (the 10,000 CelebA images of
highest AvgkNN, reference of Tables 10 to 14) and `data/siglip_cache_celeba_train.pt` (SigLIP
embeddings of the CelebA training images, the targets of the adapter) by `scripts/celeba/train.sh`; the balanced FID references
`data/celebahq/reference_*` by `scripts/p2/evaluate_table.py`; the training cache of the
Stable Diffusion adapter, `data/sd15_adapter_cache/` (50 shards of 1,000 images), by
`scripts/sd15/train_adapter.sh`.

## Reproducing the paper

### CelebA 64×64 (Appendix D)

```bash
bash scripts/celeba/train.sh                              # generator, classifiers, adapter, rare reference set
bash scripts/celeba/compare_diversity_terms.sh            # Table 8 (five seeds)
LARGE=1 bash scripts/celeba/compare_diversity_terms.sh    # Table 9 (50,000 images)
```

All runs go through `scripts/celeba/sample_and_evaluate.py` (`conf/celeba/sample_and_evaluate.yaml`,
one config group per guidance term in `conf/celeba/guidance/`): Heun sampler, 100 steps, guidance for
σ ≤ 10, batches of 100 images. Each run writes a JSON of metrics named after its guidance settings
in the `output_dir` of the script.

### CelebA-HQ, P2 (Table 2)

```bash
bash scripts/p2/prepare.sh                # adapter of P2, ResNet-18 evaluators and their recalls
bash scripts/p2/generate_and_evaluate.sh  # the four "Debias Anything" rows: images, FID, FD, CLEAM correction
```

`scripts/p2/generate.py` takes any pair (or list, `class_prompts=[...]`) of sentences, e.g. the red
hair of Figure 3:

```bash
uv run python scripts/p2/generate.py out=outputs/p2_red_hair guidance_weight=200 \
  "source_prompt=a photo of a person" "target_prompt=a photo of a person with red hair" \
  target_proportion=1
```

### Stable Diffusion 1.5 (Table 3)

```bash
bash scripts/sd15/train_adapter.sh    # training cache (50,000 images) and adapter
DIFFLENS_DIR=/path/to/DiffLens bash scripts/sd15/generate_and_evaluate.sh
```

See `eval/difflens/README.md` for the evaluation. Guided sampling on Stable Diffusion is not
bit-reproducible: the backward pass through the attention layers is nondeterministic on GPU, and two
runs with the same seed differ by about 0.5 grey level per pixel on average.

## Notebooks

```bash
uv run --with jupyter jupyter lab notebooks/
```

| notebook | content |
|---|---|
| `p2_sample_grids.ipynb` | CelebA-HQ sample grids on gender: unguided, fairness term, fairness + diversity terms; and one row per pair of sentences |
| `sd15_sample_grids.ipynb` | Stable Diffusion 1.5 sample grids on gender, unguided and guided, for two occupation prompts |

## License

Apache License 2.0, see `LICENSE` and `NOTICE`. `src/debias_anything/models/third_party/guided_diffusion/`
is code of OpenAI under the MIT license (its own `LICENSE`). The released checkpoints are trained on
CelebA and CelebA-HQ, whose terms restrict their use to non-commercial research.

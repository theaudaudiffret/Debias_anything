# Stable Diffusion evaluation (Tables 3, 15 and 16)

The text-to-image comparison follows the protocol of
[DiffLens](https://github.com/foundation-model-research/DiffLens) (Shi et al., CVPR 2025) and uses
their evaluation code. DiffLens does not come with a license, so it is not redistributed; the three
scripts of this directory are ours and only import it. Copy them to the root of a DiffLens clone:

```bash
git clone https://github.com/foundation-model-research/DiffLens.git
cd DiffLens && git checkout f6c8829
cp /path/to/this/repo/eval/difflens/*.py .
```

and download the weights their evaluation expects, as described in their README: the FairFace
classifier `evaluation/Fairface/res34_fair_align_multi_7_20190809.pt` and the dlib face models in
`evaluation/crop_face/dlib_models/`. Face cropping needs `dlib` (`uv pip install dlib-bin`).

| script | what it computes |
|---|---|
| `generate_original.py` | the unguided "Original" row of DiffLens (SD 1.5, DDIM, 4 prompts × 500), bit-identical to their `SD_generate.py` |
| `evaluate_run.py` | FD gender / age / race with their FairFace code, CLIP-T and CLIP-I, per prompt then averaged over the four prompts |
| `compute_fid.py` | Clean-FID against FFHQ, pooled over the 2,000 images of the four prompts |

### CLIP-T (optional)

DiffLens only releases the code of CLIP-I. CLIP-T, the CLIP ViT-L/14 similarity between each
image and its prompt reported in Table 3, is computed with the implementation of the
[ICM repository](https://github.com/kzaleskaa/icm) (Zaleska et al., 2026),
`tools/DiffLens/evaluation/CLIP-T/clip_text_score.py`, which is not redistributed either. The ICM
clone is only needed for this metric. By default `evaluate_run.py` looks for it next to the
DiffLens clone:

```bash
git clone https://github.com/kzaleskaa/icm.git ICM   # next to DiffLens/
cd ICM && git checkout fbf6268
```

or at the path given by the `CLIPT_PATH` environment variable. Without it, CLIP-T is skipped (as
with `--skip_clip_t`) and the other metrics are computed as usual.

`scripts/sd15/generate_and_evaluate.sh` runs the whole evaluation when `DIFFLENS_DIR` points to the
DiffLens clone.

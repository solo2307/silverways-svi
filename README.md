# SilverWays SVI

Run deep learning models on Street View Imagery (SVI) and save predictions, masks, and visualizations.

The project supports separate CPU and GPU environments.

## Repository files

```text
environment-cpu.yaml      Conda environment for CPU users
environment-gpu.yaml      Conda environment for GPU users
pyproject.toml            Python package setup
conf/models/*.yaml        Model-specific runtime configs
scripts/                  Setup and run scripts
src/                      Main inference Python package
data/                     Samples SVI images
models/                   Downloaded model weights, not committed
outputs/                  Prediction outputs, not committed
```

## Models

| Model | Task | Runner | Config | Output |
|---|---|---|---|---|
| PSPNet | semantic segmentation / greenery indicators | `silverways_inference.run_pspnet` | `conf/models/pspnet.yaml` | CSV + masks |
| YOLO | detection or segmentation depending on your `.pt` | `silverways_inference.run_yolo` | `conf/models/yolo.yaml` | CSV + JSON + annotated images |
| Mask2Former Cityscapes | semantic segmentation | `silverways_inference.run_mask2former` | `conf/models/mask2former_cityscapes.yaml` | CSV + masks |
| SAM3 | text-prompt segmentation | `silverways_inference.run_sam3` | `conf/models/sam3.yaml` | JSON + masks |
| Grounded-SAM | Grounding DINO boxes + SAM masks | `silverways_inference.run_grounded_sam` | `conf/models/grounded_sam.yaml` | JSON + masks |

## Create environment

CPU:

```bash
mamba env create -f environment-cpu.yaml
conda activate silver-ways-cpu
```

GPU:

```bash
mamba env create -f environment-gpu.yaml
conda activate silver-ways-gpu
```

Check GPU:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY
```

## Hugging Face login

SAM3 is gated. Use the same account that has access to `facebook/sam3`.

```bash
hf auth login
```

Or:

```bash
export HF_TOKEN=hf_your_token_here
```

## Input images

Put SVI images here:

```text
data/pano/
```

This folder should not be committed.

## PSPNet

Download PSPNet weights:

```bash
python -m silverways_svi.download_models pspnet \
  --config conf/models/pspnet.yaml
```

Run:

```bash
python -m silverways_svi.run_pspnet infer \
  --config conf/models/pspnet.yaml
```

Output:

```text
outputs/pspnet/
├── predictions.csv
└── masks/
```

## YOLO

Put your YOLO weights here:

```text
models/yolo/best.pt
```

Run:

```bash
python -m silverways_svi.run_yolo infer \
  --config conf/models/yolo.yaml
```

Output:

```text
outputs/yolo/
├── predictions.csv
├── json/
└── annotated/
```

## Mask2Former Cityscapes

Run:

```bash
python -m silverways_svi.run_mask2former infer \
  --config conf/models/mask2former_mapillary.yaml
```

Output:

```text
outputs/mask2former_cityscapes/
├── predictions.csv
└── masks/
```

## SAM3

Make sure you have Hugging Face access to `facebook/sam3`.

Edit the prompt in:

```text
conf/models/sam3.yaml
```

Example:

```yaml
prompt:
  text: vegetation
```

Run:

```bash
python -m silverways_svi.run_sam3 infer \
  --config conf/models/sam3.yaml
```

Output:

```text
outputs/sam3/
├── json/
└── masks/
```

## Grounded-SAM

Edit labels in:

```text
conf/models/grounded_sam.yaml
```

Run:

```bash
python -m silverways_svi.run_grounded_sam infer \
  --config conf/models/grounded_sam.yaml
```

Output:

```text
outputs/grounded_sam/
├── json/
└── masks/
```

## Quick smoke test

Set this in any config file:

```yaml
limit: 5
```

Then run the model. This checks the pipeline before processing all images.

## Git ignore

Append `.gitignore.additions` to `.gitignore`:

```bash
cat .gitignore.additions >> .gitignore
rm .gitignore.additions
```

Do not commit:

```text
data/
models/
outputs/
.env
*.pt
*.pth
*.ckpt
```

## Commit

```bash
git add \
  environment-cpu.yaml \
  environment-gpu.yaml \
  requirements-sam3-note.txt \
  conf/models \
  silverways_svi \
  scripts \
  README.models.md \
  .gitignore

git commit -m "Add separate SVI model inference runners"
git push
```

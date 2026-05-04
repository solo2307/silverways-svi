# SilverWays SVI

Run deep learning models on Street View Imagery (SVI) and save predictions, masks, metrics, and visualizations.

The project supports separate CPU and GPU Conda environments.

## What this repo does

This repository is for running inference on existing SVI images.

Current supported workflows:

- Generate directional crops from panorama images
- Run PSPNet semantic segmentation
- Run YOLO object detection, for example bench detection
- Run Mask2Former semantic segmentation using the Mapillary Vistas model
- Run Grounded-SAM: Grounding DINO text-prompt detection + SAM2 segmentation

## Repository structure

```text
environment-cpu.yaml          Conda environment for CPU users
environment-gpu.yaml          Conda environment for GPU users
pyproject.toml                Python package setup
conf/models/*.yaml            Model-specific runtime configs
src/silverways_svi/           Main inference Python package
data/pano/                    Input panorama images for testing
data/crop/                    Generated image crops, not committed
models/                       Downloaded model weights, not committed
outputs/                      Prediction outputs, not committed
```
## Setup

1. Clone the repository:

```bash
git clone https://github.com/solo2307/silverways-svi.git 
```
2. Create a Conda environment:
For CPU users:
```bash 
mamba env create -f environment-cpu.yaml
conda activate silverways-cpu 
``` 
For GPU users: 
```bash
mamba env create -f environment-gpu.yaml
conda activate silverways-gpu
```

3. For CLI commands
```bash 
pip isntall -e . --no-deps 
```
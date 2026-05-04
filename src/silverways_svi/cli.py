"""
Command-line interface for SilverWays SVI.

This module exposes the main user-facing commands for running the SilverWays
Street View Imagery inference workflows from the terminal.

The CLI is intentionally thin: it does not contain model logic itself. Instead,
it delegates to package modules under `silverways_svi.runners`.

Main workflows:

    silverways check
        Check that the environment is installed correctly.

    silverways download <model>
        Download or check required model weights.

    silverways generate-pano-crops
        Generate directional crops from full panorama images.

    silverways viz-pspnet --image data/crop/example.png
        Run PSPNet on one image and save a debug visualization.

    silverways run-pspnet
        Run PSPNet semantic segmentation on a folder of images.

    silverways run-yolo
        Run YOLO object detection.

    silverways run-mask2former
        Run Mask2Former semantic segmentation using the Mapillary Vistas model.

    silverways run-grounded-sam
        Run Grounding DINO text-prompt detection followed by SAM2 segmentation.

The console entry point is configured in `pyproject.toml`:

    [project.scripts]
    silverways = "silverways_svi.cli:app"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="SilverWays SVI command line tools.")


def run_module(module: str, args: list[str] | None = None) -> None:
    """Run a Python package module with the current Python interpreter."""
    args = args or []
    command = [sys.executable, "-m", module, *args]

    return_code = subprocess.call(command)

    if return_code != 0:
        raise typer.Exit(code=return_code)


@app.command()
def check() -> None:
    """Check that the environment is installed correctly."""
    run_module("silverways_svi.check_environment")


@app.command()
def download(
    model: str = typer.Argument(
        ...,
        help=(
            "Model to download/check: pspnet, yolo, mask2former, "
            "grounding-dino, sam2, or all."
        ),
    ),
) -> None:
    """Download/check model weights."""
    run_module("silverways_svi.download_models", [model])


@app.command("generate-pano-crops")
def generate_pano_crops(
    input_dir: Path = typer.Option(Path("data/pano"), "--input-dir", "-i"),
    output_dir: Path = typer.Option(Path("data/crop"), "--output-dir", "-o"),
    headings: str = typer.Option("0,90,180,270", "--headings"),
    fov_degrees: int = typer.Option(90, "--fov-degrees"),
    trim_top_ratio: float = typer.Option(0.08, "--trim-top-ratio"),
    trim_bottom_ratio: float = typer.Option(0.15, "--trim-bottom-ratio"),
    max_crop_size: int | None = typer.Option(None, "--max-crop-size"),
    recursive: bool = typer.Option(False, "--recursive/--no-recursive"),
) -> None:
    """Generate heading crops from panorama images."""
    args = [
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--headings",
        headings,
        "--fov-degrees",
        str(fov_degrees),
        "--trim-top-ratio",
        str(trim_top_ratio),
        "--trim-bottom-ratio",
        str(trim_bottom_ratio),
    ]

    if max_crop_size is not None:
        args.extend(["--max-crop-size", str(max_crop_size)])

    if recursive:
        args.append("--recursive")

    run_module("silverways_svi.generate_pano_crops", args)


@app.command("viz-pspnet")
def viz_pspnet(
    image: Path = typer.Option(
        ...,
        "--image",
        "-i",
        help="Path to one input image.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/debug_pspnet"),
        "--output-dir",
        "-o",
        help="Output directory.",
    ),
    model_dir: Path = typer.Option(
        Path("models/pspnet_svi_veg"),
        "--model-dir",
        help="Directory with PSPNet model files.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="auto, cpu, cuda, or cuda:0.",
    ),
) -> None:
    """Run PSPNet on one image and save a debug visualization."""
    run_module(
        "silverways_svi.runners.viz_one_pspnet",
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--device",
            device,
        ],
    )


@app.command("run-pspnet")
def run_pspnet(
    config: Path = typer.Option(
        Path("conf/models/pspnet.yaml"),
        "--config",
        "-c",
        help="Path to PSPNet config YAML.",
    ),
) -> None:
    """Run PSPNet semantic segmentation on an image folder."""
    run_module(
        "silverways_svi.runners.run_pspnet",
        ["infer", "--config", str(config)],
    )


@app.command("run-yolo")
def run_yolo(
    config: Path = typer.Option(
        Path("conf/models/yolo.yaml"),
        "--config",
        "-c",
        help="Path to YOLO config YAML.",
    ),
) -> None:
    """Run YOLO object detection."""
    run_module(
        "silverways_svi.runners.run_yolo",
        ["infer", "--config", str(config)],
    )


@app.command("run-mask2former")
def run_mask2former(
    config: Path = typer.Option(
        Path("conf/models/mask2former_mapillary.yaml"),
        "--config",
        "-c",
        help="Path to Mask2Former Mapillary Vistas config YAML.",
    ),
) -> None:
    """Run Mask2Former Mapillary Vistas semantic segmentation."""
    run_module(
        "silverways_svi.runners.run_mask2former",
        ["infer", "--config", str(config)],
    )


@app.command("run-grounded-sam")
def run_grounded_sam(
    config: Path = typer.Option(
        Path("conf/models/grounded_sam.yaml"),
        "--config",
        "-c",
        help="Path to Grounded-SAM config YAML.",
    ),
) -> None:
    """Run Grounding DINO + SAM2 segmentation."""
    run_module(
        "silverways_svi.runners.run_grounded_sam",
        ["infer", "--config", str(config)],
    )


if __name__ == "__main__":
    app()
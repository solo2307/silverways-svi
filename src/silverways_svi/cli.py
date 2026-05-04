from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="SilverWays SVI command line tools.")


def run_module(module: str, args: list[str] | None = None) -> None:
    """Run a Python module with the current Python interpreter."""
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
        help="Model to download/check: pspnet, yolo, sam2, grounding_dino, mask2former, or all.",
    ),
) -> None:
    """Download/check model weights."""
    run_module("silverways_svi.download_models", [model])


@app.command("viz-pspnet")
def viz_pspnet(
    image: Path = typer.Option(..., "--image", "-i", help="Path to one input image."),
    output_dir: Path = typer.Option(
        Path("outputs/debug_pspnet"),
        "--output-dir",
        "-o",
        help="Output directory.",
    ),
    device: str = typer.Option("auto", "--device", help="auto, cpu, cuda, or cuda:0."),
) -> None:
    """Run PSPNet on one image and save visualization."""
    run_module(
        "silverways_svi.runners.run_pspnet",
        [
            "viz-one",
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
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
    """Run PSPNet folder inference."""
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
    """Run YOLO inference."""
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
        help="Path to Mask2Former config YAML.",
    ),
) -> None:
    """Run Mask2Former inference."""
    run_module(
        "silverways_svi.runners.run_mask2former",
        ["infer", "--config", str(config)],
    )


@app.command("run-sam2")
def run_sam2(
    config: Path = typer.Option(
        Path("conf/models/sam2.yaml"),
        "--config",
        "-c",
        help="Path to SAM2 config YAML.",
    ),
) -> None:
    """Run SAM2 inference."""
    run_module(
        "silverways_svi.runners.run_sam2",
        ["infer", "--config", str(config)],
    )

@app.command("run-grounding-dino")
def run_grounding_dino(
    config: Path = typer.Option(
        Path("conf/models/grounding_dino.yaml"),
        "--config",
        "-c",
        help="Path to Grounding DINO config YAML.",
    ),
) -> None:
    """Run Grounding DINO open-vocabulary detection."""
    run_module(
        "silverways_svi.runners.run_grounding_dino",
        ["infer", "--config", str(config)],
    )


@app.command("run-grounded-sam2")
def run_grounded_sam2(
    config: Path = typer.Option(
        Path("conf/models/grounded_sam2.yaml"),
        "--config",
        "-c",
        help="Path to Grounded-SAM2 config YAML.",
    ),
) -> None:
    """Run Grounding DINO + SAM2 pipeline."""
    run_module(
        "silverways_svi.runners.run_grounded_sam2",
        ["infer", "--config", str(config)],
    )


if __name__ == "__main__":
    app()
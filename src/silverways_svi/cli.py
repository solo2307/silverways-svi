from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="SilverWays SVI command line tools.")


def run_python_script(script_path: str, args: list[str] | None = None) -> None:
    args = args or []
    command = [sys.executable, script_path, *args]
    raise_code = subprocess.call(command)
    if raise_code != 0:
        raise typer.Exit(code=raise_code)


@app.command()
def check() -> None:
    """Check that the environment is installed correctly."""
    run_python_script("scripts/check_environment.py")


@app.command()
def download(
    model: str = typer.Argument(..., help="Model to download: pspnet, yolo, or all."),
) -> None:
    """Download models weights."""
    run_python_script("scripts/download_models.py", [model])


@app.command()
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
    run_python_script(
        "scripts/viz_one_pspnet.py",
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--device",
            device,
        ],
    )


@app.command()
def run_pspnet() -> None:
    """Run PSPNet folder inference."""
    run_python_script("scripts/run_pspnet.py")


@app.command()
def run_yolo() -> None:
    """Run YOLO inference."""
    run_python_script("scripts/run_yolo.py")


@app.command()
def run_mask2former() -> None:
    """Run Mask2Former inference."""
    run_python_script("scripts/run_mask2former.py")


@app.command()
def run_sam3() -> None:
    """Run SAM3 inference."""
    run_python_script("scripts/run_sam3.py")


@app.command()
def run_grounded_sam() -> None:
    """Run Grounded-SAM inference."""
    run_python_script("scripts/run_grounded_sam.py")


if __name__ == "__main__":
    app()
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import typer
import yaml
from huggingface_hub import snapshot_download
from ultralytics import SAM, YOLO

app = typer.Typer(help="Download/check models files for SilverWays SVI.")


def hf_token() -> str | None:
    """Return Hugging Face token from environment, if available."""
    return os.environ.get("HF_TOKEN")


def read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML config."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    return data


def download_hf_snapshot(
    repo_id: str,
    local_dir: str | Path,
    allow_patterns: list[str] | None = None,
    repo_type: str = "model",
) -> Path:
    """Download files from a Hugging Face model repo."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,  # must be "model", not "models"
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        token=hf_token(),
    )

    return local_dir


def get_ultralytics_ckpt_path(model: Any, weight_name: str) -> Path | None:
    """Try to recover the local checkpoint path after Ultralytics loads a models."""
    candidates = [
        getattr(model, "ckpt_path", None),
        getattr(getattr(model, "models", None), "pt_path", None),
        weight_name,
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        path = Path(candidate)
        if path.exists():
            return path

    return None


def copy_ultralytics_weight_to_target(
    weight_name: str,
    target_path: Path,
    model_type: str,
) -> Path:
    """Download/cache an Ultralytics models and copy it to target_path."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "yolo":
        model = YOLO(weight_name)
    elif model_type == "sam":
        model = SAM(weight_name)
    else:
        raise ValueError(f"Unsupported Ultralytics models type: {model_type}")

    ckpt_path = get_ultralytics_ckpt_path(model, weight_name)

    if ckpt_path is None:
        raise FileNotFoundError(
            f"Ultralytics loaded `{weight_name}`, but I could not find the "
            "downloaded checkpoint path."
        )

    if ckpt_path.resolve() != target_path.resolve():
        shutil.copy2(ckpt_path, target_path)

        # Clean root-level duplicate created by Ultralytics, e.g. yolo11l.pt or sam2_b.pt
        if ckpt_path.parent.resolve() == Path.cwd().resolve():
            ckpt_path.unlink(missing_ok=True)

    return target_path


def download_pspnet_from_config(config_path: Path) -> Path:
    """Download PSPNet weights from Hugging Face."""
    cfg = read_yaml(config_path)
    model_cfg = cfg["models"]

    repo_id = model_cfg["hf_repo_id"]
    local_dir = model_cfg["local_dir"]
    files = model_cfg.get("files")

    return download_hf_snapshot(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=files,
    )


def download_mask2former_from_config(config_path: Path) -> Path:
    """Download Mask2Former weights from Hugging Face."""
    cfg = read_yaml(config_path)
    model_cfg = cfg["models"]

    repo_id = model_cfg["name_or_path"]
    local_dir = model_cfg.get("local_dir", "models/mask2former_mapillary")

    return download_hf_snapshot(
        repo_id=repo_id,
        local_dir=local_dir,
    )


def check_yolo_from_config(config_path: Path) -> Path | str:
    """Check/download YOLO weights using Ultralytics.

    Supported examples:
      weights: yolo11l.pt
      weights: yolo11x.pt
      weights: models/yolo11l.pt
      weights: models/yolo11x.pt
      weights: models/custom_best.pt
    """
    cfg = read_yaml(config_path)
    model_cfg = cfg["models"]

    weights = str(model_cfg["weights"])
    weights_path = Path(weights)

    if weights_path.exists():
        return weights_path

    # Example: yolo11l.pt
    if Path(weights).name == weights and weights.startswith("yolo") and weights.endswith(".pt"):
        model = YOLO(weights)
        ckpt_path = get_ultralytics_ckpt_path(model, weights)
        return ckpt_path or weights

    # Example: models/yolo11l.pt
    if weights_path.name.startswith("yolo") and weights_path.suffix == ".pt":
        return copy_ultralytics_weight_to_target(
            weight_name=weights_path.name,
            target_path=weights_path,
            model_type="yolo",
        )

    raise FileNotFoundError(
        f"YOLO weights not found: {weights_path}\n"
        "Use `yolo11l.pt`, `yolo11x.pt`, `models/yolo11l.pt`, "
        "or put your trained weights at the configured path."
    )


def check_sam2_from_config(config_path: Path) -> Path | str:
    """Check/download SAM2 weights using Ultralytics.

    Supported examples:
      weights: sam2_b.pt
      weights: sam2_l.pt
      weights: models/sam2_b.pt
      weights: models/sam2_l.pt
    """
    cfg = read_yaml(config_path)
    model_cfg = cfg["models"]

    weights = str(model_cfg["weights"])
    weights_path = Path(weights)

    if weights_path.exists():
        return weights_path

    # Example: sam2_b.pt
    if Path(weights).name == weights and weights.startswith("sam2") and weights.endswith(".pt"):
        model = SAM(weights)
        ckpt_path = get_ultralytics_ckpt_path(model, weights)
        return ckpt_path or weights

    # Example: models/sam2_b.pt
    if weights_path.name.startswith("sam2") and weights_path.suffix == ".pt":
        return copy_ultralytics_weight_to_target(
            weight_name=weights_path.name,
            target_path=weights_path,
            model_type="sam",
        )

    raise FileNotFoundError(
        f"SAM2 weights not found: {weights_path}\n"
        "Use `sam2_t.pt`, `sam2_s.pt`, `sam2_b.pt`, `sam2_l.pt`, "
        "`models/sam2_b.pt`, or put weights at the configured path."
    )

@app.command()
def pspnet(
    config: Path = typer.Option(
        Path("conf/models/pspnet.yaml"),
        "--config",
        "-c",
        help="Path to PSPNet config YAML.",
    )
) -> None:
    """Download PSPNet models files from Hugging Face."""
    output_dir = download_pspnet_from_config(config)
    typer.echo(f"Downloaded PSPNet files to: {output_dir}")


@app.command()
def mask2former(
    config: Path = typer.Option(
        Path("conf/models/mask2former_mapillary.yaml"),
        "--config",
        "-c",
        help="Path to Mask2Former config YAML.",
    )
) -> None:
    """Download Mask2Former models files from Hugging Face."""
    output_dir = download_mask2former_from_config(config)
    typer.echo(f"Downloaded Mask2Former files to: {output_dir}")


@app.command()
def yolo(
    config: Path = typer.Option(
        Path("conf/models/yolo.yaml"),
        "--config",
        "-c",
        help="Path to YOLO config YAML.",
    )
) -> None:
    """Check/download YOLO weights using Ultralytics."""
    result = check_yolo_from_config(config)
    typer.echo(f"YOLO models ready: {result}")


@app.command()
def sam2(
    config: Path = typer.Option(
        Path("conf/models/sam2.yaml"),
        "--config",
        "-c",
        help="Path to SAM2 config YAML.",
    )
) -> None:
    """Check/download SAM2 weights using Ultralytics."""
    result = check_sam2_from_config(config)
    typer.echo(f"SAM2 models ready: {result}")
def download_grounding_dino_from_config(config_path: Path) -> Path:
    """Download Grounding DINO model files from Hugging Face."""
    cfg = read_yaml(config_path)
    model_cfg = cfg["model"]

    repo_id = model_cfg["name_or_path"]
    local_dir = model_cfg.get("local_dir", "models/grounding_dino_tiny")

    return download_hf_snapshot(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="model",
    )
@app.command()
def grounding_dino(
    config: Path = typer.Option(
        Path("conf/models/grounding_dino.yaml"),
        "--config",
        "-c",
        help="Path to Grounding DINO config YAML.",
    )
) -> None:
    """Download Grounding DINO model files from Hugging Face."""
    output_dir = download_grounding_dino_from_config(config)
    typer.echo(f"Downloaded Grounding DINO files to: {output_dir}")

@app.command("all")
def download_all() -> None:
    """Download/check non-gated models files."""
    pspnet_dir = download_pspnet_from_config(Path("conf/models/pspnet.yaml"))
    typer.echo(f"Downloaded PSPNet files to: {pspnet_dir}")

    yolo_result = check_yolo_from_config(Path("conf/models/yolo.yaml"))
    typer.echo(f"YOLO models ready: {yolo_result}")

    sam2_result = check_sam2_from_config(Path("conf/models/sam2.yaml"))
    typer.echo(f"SAM2 models ready: {sam2_result}")

    mask2former_dir = download_mask2former_from_config(
        Path("conf/models/mask2former_mapillary.yaml")
    )
    typer.echo(f"Downloaded Mask2Former files to: {mask2former_dir}")

    grounding_dino_dir = download_grounding_dino_from_config(
        Path("conf/models/grounding_dino.yaml")
    )
    typer.echo(f"Downloaded Grounding DINO files to: {grounding_dino_dir}")

if __name__ == "__main__":
    app()
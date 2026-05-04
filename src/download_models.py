from __future__ import annotations

from pathlib import Path

import typer
from huggingface_hub import snapshot_download

from src.common import hf_token, read_yaml

app = typer.Typer(help="Download model files used by the separate runners.")


@app.command()
def pspnet(config: Path = Path("conf/models/pspnet.yaml")) -> None:
    cfg = read_yaml(config)
    model_cfg = cfg["model"]
    snapshot_download(
        repo_id=model_cfg["hf_repo_id"],
        repo_type="model",
        local_dir=model_cfg["local_dir"],
        allow_patterns=model_cfg["files"],
        token=hf_token(),
    )
    print(f"Downloaded PSPNet files to {model_cfg['local_dir']}")


@app.command()
def yolo(config: Path = Path("conf/models/yolo.yaml")) -> None:
    cfg = read_yaml(config)
    model_cfg = cfg["model"]
    repo_id = model_cfg.get("hf_repo_id")
    if not repo_id:
        raise typer.BadParameter("No YOLO hf_repo_id configured. Put weights locally in models/yolo/best.pt.")
    weights_parent = Path(model_cfg["weights"]).parent
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=weights_parent,
        allow_patterns=model_cfg.get("hf_patterns", ["*.pt"]),
        token=hf_token(),
    )
    print(f"Downloaded YOLO files to {weights_parent}")


if __name__ == "__main__":
    app()

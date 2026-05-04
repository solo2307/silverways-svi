from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


DEFAULT_PSPNET_REPO_ID = "solo2307/pspnet_svi_veg"
DEFAULT_PSPNET_FILES = [
    "encoder_epoch_50.pth",
    "decoder_epoch_50.pth",
    "color150.mat",
    "color150-labels.txt",
]


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def download_pspnet(config_path: Path | None = None) -> Path:
    if config_path is not None and config_path.exists():
        cfg = load_yaml(config_path)
        model_cfg = cfg.get("model", {})
        repo_id = model_cfg.get("hf_repo_id", DEFAULT_PSPNET_REPO_ID)
        local_dir = Path(model_cfg.get("local_dir", "models/pspnet_svi_veg"))
        files = model_cfg.get("files", DEFAULT_PSPNET_FILES)
    else:
        repo_id = DEFAULT_PSPNET_REPO_ID
        local_dir = Path("models/pspnet_svi_veg")
        files = DEFAULT_PSPNET_FILES

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        allow_patterns=files,
        token=get_hf_token(),
    )

    return local_dir


def download_yolo(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    model_cfg = cfg.get("model", {})

    weights = Path(model_cfg.get("weights", "models/yolo/best.pt"))
    repo_id = model_cfg.get("hf_repo_id")

    if weights.exists():
        print(f"YOLO weights already exist: {weights}")
        return weights

    if not repo_id:
        raise ValueError(
            "YOLO weights were not found locally and no Hugging Face repo is configured.\n"
            f"Expected weights at: {weights}\n"
            "Either put your YOLO .pt file there, or set `model.hf_repo_id` "
            "in conf/models/yolo.yaml."
        )

    weights.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=weights.parent,
        allow_patterns=model_cfg.get("hf_patterns", ["*.pt"]),
        token=get_hf_token(),
    )

    return weights.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download model files for SilverWays SVI inference."
    )
    parser.add_argument(
        "model",
        choices=["pspnet", "yolo", "all"],
        help="Which model files to download.",
    )
    parser.add_argument(
        "--pspnet-config",
        default="conf/models/pspnet.yaml",
        help="Path to PSPNet config.",
    )
    parser.add_argument(
        "--yolo-config",
        default="conf/models/yolo.yaml",
        help="Path to YOLO config.",
    )

    args = parser.parse_args()

    if args.model in {"pspnet", "all"}:
        path = download_pspnet(Path(args.pspnet_config))
        print(f"Downloaded PSPNet files to: {path.resolve()}")

    if args.model in {"yolo", "all"}:
        path = download_yolo(Path(args.yolo_config))
        print(f"Downloaded YOLO files to: {path.resolve()}")


if __name__ == "__main__":
    main()

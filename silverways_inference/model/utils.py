from pathlib import Path
import os
import argparse

from huggingface_hub import snapshot_download


REPO_ID = "solo2307/pspnet_svi_veg"

FILES_TO_DOWNLOAD = [
    "encoder_epoch_50.pth",
    "decoder_epoch_50.pth",
    "color150.mat",
    "color150-labels.txt",
]


def download_models(output_dir: str = "models/pspnet_svi_veg") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=output_path,
        allow_patterns=FILES_TO_DOWNLOAD,
        token=token,
    )

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PSPNet SVI vegetation model files from Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        default="models/pspnet_svi_veg",
        help="Directory where model files will be downloaded.",
    )
    args = parser.parse_args()

    path = download_models(args.output_dir)
    print(f"Downloaded model files to: {path.resolve()}")
from pathlib import Path
import urllib.request
import logging


def download_models():
    """
    Download pretrained models into data/models/ from public MinIO URLs.
    Skips download if files already exist.
    """
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "encoder_epoch_50.pth": "https://storage.heigit.org/heigit-silverways/model-registry/segment/encoder_epoch_50.pth",
        "decoder_epoch_50.pth": "https://storage.heigit.org/heigit-silverways/model-registry/segment/decoder_epoch_50.pth",
        "color150.mat": "https://storage.heigit.org/heigit-silverways/model-registry/segment/color150.mat",
        "color150-labels.txt": "https://storage.heigit.org/heigit-silverways/model-registry/segment/color150-labels.txt",
    }

    for fname, url in files.items():
        local_path = model_dir / fname
        if not local_path.exists():
            logging.info(f"📥 Downloading {fname} from {url}")
            urllib.request.urlretrieve(url, local_path)
        else:
            logging.info(f"✅ Found {fname}, skipping download")

    return model_dir

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


@dataclass
class PackageCheck:
    import_name: str
    label: str
    required: bool = True


PACKAGES = [
    PackageCheck("torch", "PyTorch", True),
    PackageCheck("torchvision", "Torchvision", True),
    PackageCheck("numpy", "NumPy", True),
    PackageCheck("PIL", "Pillow", True),
    PackageCheck("cv2", "OpenCV", True),
    PackageCheck("matplotlib", "Matplotlib", True),
    PackageCheck("yaml", "PyYAML", True),
    PackageCheck("tqdm", "tqdm", True),
    PackageCheck("typer", "Typer", True),
    PackageCheck("huggingface_hub", "Hugging Face Hub", True),
    PackageCheck("transformers", "Transformers", True),
    PackageCheck("ultralytics", "Ultralytics / YOLO", True),
    PackageCheck("mit_semseg", "MIT SemSeg / PSPNet", True),
    PackageCheck("geopandas", "GeoPandas", False),
    PackageCheck("fiona", "Fiona", False),
    PackageCheck("skimage", "scikit-image", False),
]


def module_version(module: object) -> str:
    return str(getattr(module, "__version__", "version unknown"))


def check_imports() -> list[PackageCheck]:
    failed_required: list[PackageCheck] = []

    print(f"Python: {sys.version}")
    print("\nPackage checks:")

    for pkg in PACKAGES:
        try:
            module = importlib.import_module(pkg.import_name)
            print(f"  OK       {pkg.label}: {module_version(module)}")
        except Exception as exc:
            status = "FAILED" if pkg.required else "OPTIONAL MISSING"
            print(f"  {status:<8} {pkg.label}: {exc}")
            if pkg.required:
                failed_required.append(pkg)

    return failed_required


def check_torch_devices() -> None:
    print("\nTorch device check:")

    try:
        import torch

        print(f"  Torch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            print(f"  CUDA device 0: {torch.cuda.get_device_name(0)}")

        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        print(f"  Apple MPS available: {mps_available}")

    except Exception as exc:
        print(f"  Torch device check failed: {exc}")


def check_project_import() -> bool:
    print("\nProject package check:")

    try:
        import silverways_svi

        print(f"  OK silverways_svi: {silverways_svi.__file__}")
        return True
    except Exception as exc:
        print(f"  FAILED silverways_svi: {exc}")
        print("  Hint: run `pip install -e .` from the repo root.")
        return False


def main() -> None:
    failed_required = check_imports()
    check_torch_devices()
    project_ok = check_project_import()

    if failed_required or not project_ok:
        print("\nEnvironment check failed.")
        if failed_required:
            print("Missing required packages:")
            for pkg in failed_required:
                print(f"  - {pkg.label} ({pkg.import_name})")
        raise SystemExit(1)

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()

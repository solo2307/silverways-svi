"""
Panorama cropping utilities for Street View imagery.

This module provides a lightweight dataset class for panoramic images.
It converts each panorama into several horizontal heading views, for example:

    panorama image
        -> 0 degree crop
        -> 90 degree crop
        -> 180 degree crop
        -> 270 degree crop

The current implementation uses simple horizontal cropping. It does not perform
true perspective projection. This is useful for testing model pipelines on
panoramic Street View images because many computer vision models, such as YOLO,
Grounding DINO, SAM2, and PSPNet, usually work better on regular image views
than on a full 360-degree panorama.

Typical use:

    dataset = PanoramaDataset(
        input_dir="data/pano",
        headings=(0, 90, 180, 270),
        fov_degrees=90,
    )

    dataset.save_all_views("data/crop")

Expected output:

    data/crop/
        WE0VZ5B3_0.png
        WE0VZ5B3_90.png
        WE0VZ5B3_180.png
        WE0VZ5B3_270.png

Later, this simple crop logic can be replaced with true perspective projection
if panorama distortion becomes a problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PanoramaView:
    pano_path: Path
    heading: int
    image: Image.Image
    view_id: str
    crop_box: tuple[int, int, int, int]

    @property
    def pano_stem(self) -> str:
        return self.pano_path.stem

    def save(self, output_dir: str | Path, suffix: str = ".png") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{self.pano_stem}_{self.heading}.png"
        self.image.save(output_path)
        return output_path


@dataclass(frozen=True)
class PanoramaItem:
    path: Path
    views: list[PanoramaView]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem


class PanoramaDataset:
    """Dataset for panoramic images.

    It turns each panorama into multiple horizontal crops.
    This is simple cropping, not true perspective projection.
    """

    def __init__(
        self,
        input_dir: str | Path,
        headings: tuple[int, ...] = (0, 90, 180, 270),
        fov_degrees: int = 90,
        recursive: bool = False,
        skip_prefixes: tuple[str, ...] = ("CAoS",),
    ) -> None:
        self.input_dir = Path(input_dir)
        self.headings = headings
        self.fov_degrees = fov_degrees
        self.recursive = recursive
        self.skip_prefixes = skip_prefixes

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        self.paths = self._find_images()

    def _find_images(self) -> list[Path]:
        globber: Iterable[Path] = (
            self.input_dir.rglob("*") if self.recursive else self.input_dir.glob("*")
        )

        paths = []
        for path in globber:
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.stem.startswith(self.skip_prefixes):
                continue
            paths.append(path)

        return sorted(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> PanoramaItem:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")

        views = [
            self._crop_heading(image=image, pano_path=path, heading=heading)
            for heading in self.headings
        ]

        return PanoramaItem(path=path, views=views)

    def __iter__(self) -> Iterator[PanoramaItem]:
        for i in range(len(self)):
            yield self[i]

    def save_all_views(self, output_dir: str | Path) -> list[Path]:
        output_dir = Path(output_dir)
        saved_paths: list[Path] = []

        for pano in self:
            for view in pano.views:
                saved_paths.append(view.save(output_dir))

        return saved_paths

    def _crop_heading(
        self,
        image: Image.Image,
        pano_path: Path,
        heading: int,
    ) -> PanoramaView:
        width, height = image.size

        crop_width = int(round(width * self.fov_degrees / 360.0))
        start_x = int(round((heading % 360) / 360.0 * width))
        end_x = start_x + crop_width

        crop = self._crop_wrapped(image, start_x, end_x)

        return PanoramaView(
            pano_path=pano_path,
            heading=heading,
            image=crop,
            view_id=f"heading_{heading:03d}",
            crop_box=(start_x % width, 0, end_x % width, height),
        )

    @staticmethod
    def _crop_wrapped(image: Image.Image, start_x: int, end_x: int) -> Image.Image:
        width, height = image.size

        start_x = start_x % width
        end_x = end_x % width

        if start_x < end_x:
            return image.crop((start_x, 0, end_x, height))

        left = image.crop((start_x, 0, width, height))
        right = image.crop((0, 0, end_x, height))

        output = Image.new("RGB", (left.width + right.width, height))
        output.paste(left, (0, 0))
        output.paste(right, (left.width, 0))
        return output
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

In addition to horizontal cropping, this implementation can also trim a
percentage of the image from the top and bottom. This is useful because
panoramic Street View images often contain distorted content near the top
and bottom, such as stitching artifacts, stretched sky, or vehicle/camera rig
parts. Removing these regions can produce cleaner crops for downstream models.

Typical use:

    dataset = PanoramaDataset(
        input_dir="data/pano",
        headings=(0, 90, 180, 270),
        fov_degrees=90,
        trim_top_ratio=0.08,
        trim_bottom_ratio=0.15,
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

        output_path = output_dir / f"{self.pano_stem}_{self.heading}{suffix}"
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

    Parameters
    ----------
    input_dir:
        Folder containing panorama images.
    headings:
        Tuple of headings in degrees to crop from the panorama.
    fov_degrees:
        Horizontal field of view for each crop.
    trim_top_ratio:
        Fraction of the crop height to remove from the top.
        Example: 0.08 means remove 8 percent from the top.
    trim_bottom_ratio:
        Fraction of the crop height to remove from the bottom.
        Example: 0.15 means remove 15 percent from the bottom.
    recursive:
        Whether to search for images recursively.
    skip_prefixes:
        File prefixes to skip.
    """

    def __init__(
        self,
        input_dir: str | Path,
        headings: tuple[int, ...] = (0, 90, 180, 270),
        fov_degrees: int = 90,
        trim_top_ratio: float = 0.0,
        trim_bottom_ratio: float = 0.0,
        recursive: bool = False,
        skip_prefixes: tuple[str, ...] = ("CAoS",),
    ) -> None:
        self.input_dir = Path(input_dir)
        self.headings = headings
        self.fov_degrees = fov_degrees
        self.trim_top_ratio = trim_top_ratio
        self.trim_bottom_ratio = trim_bottom_ratio
        self.recursive = recursive
        self.skip_prefixes = skip_prefixes

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        if not (0.0 <= self.trim_top_ratio < 1.0):
            raise ValueError("trim_top_ratio must be between 0.0 and 1.0")

        if not (0.0 <= self.trim_bottom_ratio < 1.0):
            raise ValueError("trim_bottom_ratio must be between 0.0 and 1.0")

        if self.trim_top_ratio + self.trim_bottom_ratio >= 1.0:
            raise ValueError(
                "trim_top_ratio + trim_bottom_ratio must be less than 1.0"
            )

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
        crop = self._trim_vertical(crop)

        trimmed_width, trimmed_height = crop.size

        return PanoramaView(
            pano_path=pano_path,
            heading=heading,
            image=crop,
            view_id=f"heading_{heading:03d}",
            crop_box=(0, 0, trimmed_width, trimmed_height),
        )

    def _trim_vertical(self, image: Image.Image) -> Image.Image:
        """Trim a percentage from the top and bottom of the image."""
        width, height = image.size

        top_px = int(round(height * self.trim_top_ratio))
        bottom_px = int(round(height * self.trim_bottom_ratio))

        y1 = top_px
        y2 = height - bottom_px

        return image.crop((0, y1, width, y2))

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
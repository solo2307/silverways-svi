from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageItem:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    def load_rgb(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")


class ImageDataset:
    """Dataset for regular image files."""

    def __init__(
        self,
        input_dir: str | Path,
        recursive: bool = False,
        skip_prefixes: tuple[str, ...] = ("CAoS",),
    ) -> None:
        self.input_dir = Path(input_dir)
        self.recursive = recursive
        self.skip_prefixes = skip_prefixes

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        self.items = self._find_images()

    def _find_images(self) -> list[ImageItem]:
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

        return [ImageItem(path=p) for p in sorted(paths)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ImageItem:
        return self.items[index]

    def __iter__(self) -> Iterator[ImageItem]:
        return iter(self.items)

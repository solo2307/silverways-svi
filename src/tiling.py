from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image


@dataclass(frozen=True)
class Tile:
    tile_id: str
    image_name: str
    image: Image.Image
    x: int
    y: int
    width: int
    height: int


def iter_panorama_tiles(
    image_path: str | Path,
    tile_width: int,
    tile_height: int,
    overlap: int,
    wrap_x: bool = True,
) -> Iterator[Tile]:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")

    pano_width, pano_height = image.size
    step_x = tile_width - overlap
    step_y = tile_height - overlap

    if step_x <= 0 or step_y <= 0:
        raise ValueError("Tile overlap must be smaller than tile width and height.")

    y_positions = list(range(0, max(1, pano_height - tile_height + 1), step_y))
    if not y_positions or y_positions[-1] + tile_height < pano_height:
        y_positions.append(max(0, pano_height - tile_height))

    x_positions = list(range(0, max(1, pano_width - tile_width + 1), step_x))

    if not wrap_x:
        if not x_positions or x_positions[-1] + tile_width < pano_width:
            x_positions.append(max(0, pano_width - tile_width))

    for y in y_positions:
        for x in x_positions:
            if wrap_x:
                tile = Image.new("RGB", (tile_width, tile_height))

                for dx in range(tile_width):
                    src_x = (x + dx) % pano_width
                    column = image.crop((src_x, y, src_x + 1, min(y + tile_height, pano_height)))
                    tile.paste(column, (dx, 0))
            else:
                tile = image.crop((x, y, min(x + tile_width, pano_width), min(y + tile_height, pano_height)))

                if tile.size != (tile_width, tile_height):
                    padded = Image.new("RGB", (tile_width, tile_height))
                    padded.paste(tile, (0, 0))
                    tile = padded

            tile_id = f"{image_path.stem}_x{x}_y{y}"

            yield Tile(
                tile_id=tile_id,
                image_name=image_path.name,
                image=tile,
                x=x,
                y=y,
                width=tile_width,
                height=tile_height,
            )
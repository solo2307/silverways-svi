"""
Generate cropped heading views from panoramic Street View images.

This utility reads panorama images from `data/pano`, splits each panorama into
heading crops, optionally trims distorted top/bottom regions, optionally resizes
each crop to a maximum size, and saves the crops into `data/crop`.

Processing order:

    1. crop panorama by heading
    2. trim top/bottom
    3. resize final crop if --max-crop-size is provided

Example:

    silverways generate-pano-crops \
      --input-dir data/pano \
      --output-dir data/crop \
      --headings 0,90,180,270 \
      --fov-degrees 90 \
      --trim-top-ratio 0.08 \
      --trim-bottom-ratio 0.15 \
      --max-crop-size 1600

If --max-crop-size is omitted, crops are saved at full size.

The generated crops can then be used as input for PSPNet, YOLO, Grounding DINO,
SAM2, Mask2Former, or other image models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from silverways_svi.data.pano_dataset import PanoramaDataset


def parse_headings(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cropped heading views from panoramic images."
    )
    parser.add_argument(
        "--input-dir",
        default="data/pano",
        help="Directory with panoramic images.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/crop",
        help="Directory where cropped views will be saved.",
    )
    parser.add_argument(
        "--headings",
        default="0,90,180,270",
        help="Comma-separated headings, e.g. 0,90,180,270.",
    )
    parser.add_argument(
        "--fov-degrees",
        type=int,
        default=90,
        help="Horizontal crop width in degrees.",
    )
    parser.add_argument(
        "--trim-top-ratio",
        type=float,
        default=0.08,
        help="Fraction to remove from the top of each crop.",
    )
    parser.add_argument(
        "--trim-bottom-ratio",
        type=float,
        default=0.15,
        help="Fraction to remove from the bottom of each crop.",
    )
    parser.add_argument(
        "--max-crop-size",
        type=int,
        default=None,
        help=(
            "Maximum pixel size of the longest side of each saved crop. "
            "If omitted, crops are saved at full size."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directory recursively.",
    )

    args = parser.parse_args()

    dataset = PanoramaDataset(
        input_dir=args.input_dir,
        headings=parse_headings(args.headings),
        fov_degrees=args.fov_degrees,
        trim_top_ratio=args.trim_top_ratio,
        trim_bottom_ratio=args.trim_bottom_ratio,
        max_crop_size=args.max_crop_size,
        recursive=args.recursive,
    )

    output_dir = Path(args.output_dir)
    saved_paths = dataset.save_all_views(output_dir)

    print(f"Panoramas found: {len(dataset)}")
    print(f"Crops saved: {len(saved_paths)} in {output_dir}")



if __name__ == "__main__":
    main()
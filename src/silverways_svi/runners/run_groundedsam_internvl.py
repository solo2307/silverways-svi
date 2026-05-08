from __future__ import annotations

from pathlib import Path

import typer
import yaml
from tqdm import tqdm

from silverways_svi.data.image_dataset import ImageDataset
from silverways_svi.runners.internvl_tagger import InternVLTagger
from silverways_svi.runners.region_utils import (
    crop_detection,
    safe_name,
    save_json,
    save_masks,
)
from silverways_svi.runners.run_grounded_sam import (
    GroundingDinoDetector,
    SAM2Segmenter,
    masks_to_numpy,
    save_annotated,
)

app = typer.Typer(
    help="Run GroundedSAM localization followed by InternVL sidewalk/shared-surface tagging."
)


@app.callback()
def main() -> None:
    """GroundedSAM + InternVL runner."""


def read_yaml(path: Path) -> dict:
    """Read YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    return data


@app.command()
def infer(
    config: Path = typer.Option(
        Path("conf/models/groundedsam_internvl.yaml"),
        "--config",
        "-c",
        help="Path to GroundedSAM + InternVL config YAML.",
    )
) -> None:
    """Run GroundedSAM and enrich detections with InternVL OSM tags."""
    cfg = read_yaml(config)

    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    recursive = bool(cfg.get("recursive", False))
    limit = cfg.get("limit")
    device = str(cfg.get("device", "auto"))

    grounding_cfg = cfg["grounding_dino"]
    sam2_cfg = cfg["sam2"]
    prompt_cfg = cfg["prompt"]
    internvl_cfg = cfg["internvl"]
    output_cfg = cfg.get("outputs", {})

    text_labels = prompt_cfg["text_labels"]
    box_threshold = float(prompt_cfg.get("box_threshold", 0.35))
    text_threshold = float(prompt_cfg.get("text_threshold", 0.25))

    fallback_only_if_no_detection = bool(
        prompt_cfg.get("fallback_only_if_no_detection", True)
    )
    fallback_text_labels = prompt_cfg.get("fallback_text_labels", [])
    fallback_box_threshold = float(
        prompt_cfg.get("fallback_box_threshold", box_threshold)
    )
    fallback_text_threshold = float(
        prompt_cfg.get("fallback_text_threshold", text_threshold)
    )

    crop_padding_ratio = float(cfg.get("crop_padding_ratio", 0.15))
    use_masked_crop = bool(cfg.get("use_masked_crop", False))

    dataset = ImageDataset(input_dir=input_dir, recursive=recursive)
    items = list(dataset)

    if limit is not None:
        items = items[: int(limit)]

    if not items:
        typer.echo(f"No images found in: {input_dir}")
        raise typer.Exit(code=0)

    detector = GroundingDinoDetector(
        name_or_path=grounding_cfg["name_or_path"],
        local_dir=grounding_cfg.get("local_dir"),
        device=device,
    )

    segmenter = SAM2Segmenter(
        weights=sam2_cfg["weights"],
        imgsz=int(sam2_cfg.get("imgsz", 1024)),
        device=device,
    )

    tagger = InternVLTagger(
        model_path=internvl_cfg["model_path"],
        mmproj_path=internvl_cfg["mmproj_path"],
        n_ctx=int(internvl_cfg.get("n_ctx", 8192)),
        n_gpu_layers=int(internvl_cfg.get("n_gpu_layers", 0)),
        verbose=bool(internvl_cfg.get("verbose", False)),
        rules_path=internvl_cfg.get("rules_path"),
    )

    json_dir = output_dir / "json"
    masks_dir = output_dir / "masks"
    annotated_dir = output_dir / "annotated"
    crops_dir = output_dir / "crops"

    for item in tqdm(items, desc="GroundedSAM + InternVL"):
        image = item.load_rgb()

        candidate_type = "dedicated_sidewalk_candidate"
        used_text_labels = text_labels
        used_box_threshold = box_threshold
        used_text_threshold = text_threshold

        detections = detector.predict(
            image=image,
            text_labels=text_labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Fallback: if no sidewalk is detected, try road/shared-surface labels.
        if len(detections) == 0 and fallback_only_if_no_detection:
            if fallback_text_labels:
                candidate_type = "shared_surface_fallback"
                used_text_labels = fallback_text_labels
                used_box_threshold = fallback_box_threshold
                used_text_threshold = fallback_text_threshold

                detections = detector.predict(
                    image=image,
                    text_labels=fallback_text_labels,
                    box_threshold=fallback_box_threshold,
                    text_threshold=fallback_text_threshold,
                )

        boxes_xyxy = [det["xyxy"] for det in detections]

        if len(boxes_xyxy) == 0:
            payload = {
                "image_id": item.stem,
                "image_path": str(item.path),
                "localizer": "grounded_sam",
                "candidate_type": "none",
                "text_labels": text_labels,
                "used_text_labels": used_text_labels,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "used_box_threshold": used_box_threshold,
                "used_text_threshold": used_text_threshold,
                "fallback_only_if_no_detection": fallback_only_if_no_detection,
                "fallback_text_labels": fallback_text_labels,
                "fallback_box_threshold": fallback_box_threshold,
                "fallback_text_threshold": fallback_text_threshold,
                "crop_padding_ratio": crop_padding_ratio,
                "use_masked_crop": use_masked_crop,
                "num_detections": 0,
                "num_masks": 0,
                "detections": [],
            }

            if output_cfg.get("save_json", True):
                save_json(json_dir / f"{item.stem}.json", payload)

            continue

        sam_result = segmenter.predict_from_boxes(
            image_path=item.path,
            boxes_xyxy=boxes_xyxy,
        )

        masks = masks_to_numpy(sam_result)

        enriched_detections = []

        for idx, det in enumerate(detections):
            mask = None

            if masks is not None and idx < masks.shape[0]:
                mask = masks[idx]

            crop = crop_detection(
                image=image,
                detection=det,
                mask=mask,
                padding_ratio=crop_padding_ratio,
                use_mask=use_masked_crop,
            )

            crop_path = ""
            if output_cfg.get("save_crops", True):
                crops_dir.mkdir(parents=True, exist_ok=True)
                label_safe = safe_name(str(det["label"]))
                crop_path = str(
                    crops_dir / f"{item.stem}_det{idx:03d}_{label_safe}.jpg"
                )
                crop.save(crop_path)

            osm_tags = tagger.predict_osm_tags(
                crop=crop,
                feature_label=str(det["label"]),
            )

            enriched_detections.append(
                {
                    **det,
                    "source": "grounded_sam",
                    "candidate_type": candidate_type,
                    "crop_path": crop_path,
                    "internvl_osm_tags": osm_tags,
                }
            )

        payload = {
            "image_id": item.stem,
            "image_path": str(item.path),
            "localizer": "grounded_sam",
            "candidate_type": candidate_type,
            "text_labels": text_labels,
            "used_text_labels": used_text_labels,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "used_box_threshold": used_box_threshold,
            "used_text_threshold": used_text_threshold,
            "fallback_only_if_no_detection": fallback_only_if_no_detection,
            "fallback_text_labels": fallback_text_labels,
            "fallback_box_threshold": fallback_box_threshold,
            "fallback_text_threshold": fallback_text_threshold,
            "crop_padding_ratio": crop_padding_ratio,
            "use_masked_crop": use_masked_crop,
            "num_detections": len(enriched_detections),
            "num_masks": 0 if masks is None else int(masks.shape[0]),
            "detections": enriched_detections,
        }

        if output_cfg.get("save_json", True):
            save_json(json_dir / f"{item.stem}.json", payload)

        if output_cfg.get("save_masks", True) and masks is not None:
            save_masks(masks_dir / f"{item.stem}_masks.npy", masks)

        if output_cfg.get("save_annotated", True):
            save_annotated(
                annotated_dir / f"{item.stem}_groundedsam_internvl.jpg",
                image=image,
                detections=detections,
                masks=masks,
            )

    typer.echo(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    app()
"""Export augmented training data for YOLO and Faster R-CNN.

Run from the repository root: python -m src.ml.augment --copies 3 --seed 42
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import DEFAULT_ROOT, get_image
from src.data.split import load_split_manifest, load_split_from_manifest, save_split_manifest

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AugmentConfig:
    noise_std: float = 8.0  # standard deviation in 0–255 pixel units
    horizontal_flip: float = 0.5  # probability
    vertical_flip: float = 0.0
    brightness: float = 0.2  # random multiplicative change, +/- this fraction
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 10.0  # degrees, +/- this amount
    scale_min: float = 0.8
    scale_max: float = 1.2
    rotation: float = 15.0  # degrees, +/- this amount
    min_visibility: float = 0.25
    min_box_size: float = 2.0  # pixels after clipping

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Augmentation settings must be finite")
        for name in ("horizontal_flip", "vertical_flip", "min_visibility"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("brightness", "contrast", "saturation"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 < self.scale_min <= self.scale_max:
            raise ValueError("Scale must satisfy 0 < scale_min <= scale_max")
        if not 0 <= self.rotation <= 180 or not 0 <= self.hue <= 180:
            raise ValueError("Rotation and hue must be between 0 and 180 degrees")
        if self.noise_std < 0 or self.min_box_size < 0:
            raise ValueError("Noise and minimum box size must be nonnegative")


def transform_boxes(boxes, matrix, width, height, min_visibility=0.25, min_box_size=2):
    """Transform all four corners, clip their polygons, then enclose in XYXY boxes.

    Visibility is the fraction of the transformed label polygon inside the image.
    Rotated labels remain axis-aligned, as required by both detectors.
    """
    canvas = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    result = []
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        corners = np.array([[x1, y1, 1], [x2, y1, 1],
                            [x2, y2, 1], [x1, y2, 1]], np.float32)
        polygon = (corners @ matrix.T).astype(np.float32)
        area = cv2.contourArea(polygon)
        visible_area, clipped = cv2.intersectConvexConvex(polygon, canvas)
        if area <= 0 or visible_area <= 0 or clipped is None:
            continue
        if visible_area / area + 1e-6 < min_visibility:
            continue
        clipped = clipped.reshape(-1, 2)
        low = np.clip(clipped.min(axis=0), [0, 0], [width, height])
        high = np.clip(clipped.max(axis=0), [0, 0], [width, height])
        if np.all(high > low) and np.all(high - low >= min_box_size):
            result.append([*low, *high])
    return np.asarray(result, dtype=np.float32).reshape(-1, 4)


def augment_image(image, boxes, rng, config=None):
    """Return a new RGB uint8 image and float32 pixel XYXY boxes; inputs unchanged."""
    cfg = config or AugmentConfig()
    height, width = image.shape[:2]
    angle = rng.uniform(-cfg.rotation, cfg.rotation)
    scale = rng.uniform(cfg.scale_min, cfg.scale_max)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    # Box coordinates describe pixel edges; OpenCV samples pixel centres.
    pixel_matrix = matrix.copy()
    pixel_matrix[:, 2] += matrix[:, :2] @ np.array([0.5, 0.5]) - 0.5
    image = cv2.warpAffine(image, pixel_matrix, (width, height),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(114, 114, 114))
    boxes = transform_boxes(boxes, matrix, width, height,
                            cfg.min_visibility, cfg.min_box_size)
    if rng.random() < cfg.horizontal_flip:
        image = image[:, ::-1].copy()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    if rng.random() < cfg.vertical_flip:
        image = image[::-1].copy()
        boxes[:, [1, 3]] = height - boxes[:, [3, 1]]

    # Floating-point HSV uses hue in degrees and saturation/value in [0, 1].
    hsv = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue, cfg.hue)) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - cfg.saturation, 1 + cfg.saturation), 0, 1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255
    rgb = (rgb - 127.5) * rng.uniform(1 - cfg.contrast, 1 + cfg.contrast) + 127.5
    rgb *= rng.uniform(1 - cfg.brightness, 1 + cfg.brightness)
    if cfg.noise_std:
        rgb += rng.normal(0, cfg.noise_std, rgb.shape).astype(np.float32)
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8), boxes


def write_yolo_labels(path, boxes, width, height):
    """YOLO has one class: 0 = bird. Empty targets produce an empty label file."""
    lines = []
    for x1, y1, x2, y2 in boxes:
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                  (x2 - x1) / width, (y2 - y1) / height)
        lines.append("0 " + " ".join(f"{v:.8f}" for v in values))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def export_dataset(output, copies=3, seed=42, config=None, root=DEFAULT_ROOT,
                   manifest_path=REPO / "splits/official.json"):
    """Copy official originals; augment train only. Refuse existing output folders."""
    cfg = config or AugmentConfig()
    if copies < 1 or seed < 0:
        raise ValueError("copies must be positive and seed must be nonnegative")
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output must be separate from the source dataset")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output.")
    manifest = load_split_manifest(manifest_path)
    ids = [image_id for split in ("train", "val", "test") for image_id in manifest[split]]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate IDs or overlapping splits")
    records = {split: load_split_from_manifest(root, manifest, split)
               for split in ("train", "val", "test")}
    if any(len(records[s]) != len(manifest[s]) for s in records):
        raise ValueError("Dataset contains duplicate image IDs")
    all_ids = set(ids)
    for record in records["train"]:
        for index in range(copies):
            name = f"{record.image_id}__aug_{index + 1:03d}"
            if name in all_ids:
                raise ValueError(f"Augmented ID collides with an original: {name}")
    # Preflight labels: the shared loader intentionally tolerates missing/bad labels.
    for split_records in records.values():
        for record in split_records:
            label = root / "labels" / record.image_path.parent.name / f"{record.image_id}.txt"
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                parts = np.asarray(line.split(), dtype=float)
                if (parts.shape != (5,) or not np.isfinite(parts).all() or parts[0] != 0
                        or not np.all((parts[1:] >= 0) & (parts[1:] <= 1))
                        or np.any(parts[3:] <= 0)):
                    raise ValueError(f"Invalid single-class YOLO label: {label}")

    output.mkdir(parents=True)
    generated = {s: [] for s in records}
    provenance = {}
    dropped_boxes = 0
    rng = np.random.default_rng(seed)
    for split, split_records in records.items():
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for number, record in enumerate(split_records, 1):
            shutil.copy2(record.image_path, image_dir / record.image_path.name)
            source_label = root / "labels" / record.image_path.parent.name / f"{record.image_id}.txt"
            shutil.copy2(source_label, label_dir / source_label.name)
            generated[split].append(record.image_id)
            if split != "train":
                continue
            original = get_image(record)
            for index in range(copies):
                name = f"{record.image_id}__aug_{index + 1:03d}"
                image, boxes = augment_image(original, record.boxes, rng, cfg)
                Image.fromarray(image).save(image_dir / f"{name}.png")
                write_yolo_labels(label_dir / f"{name}.txt", boxes, record.width, record.height)
                generated[split].append(name)
                provenance[name] = record.image_id
                dropped_boxes += len(record.boxes) - len(boxes)
            if number % 100 == 0 or number == len(split_records):
                print(f"Augmented {number}/{len(split_records)} training images", flush=True)
    save_split_manifest(generated, output / "splits.json", seed=seed)
    (output / "data.yaml").write_text(yaml.safe_dump({
        "path": str(output), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": {0: "bird"}}, sort_keys=False))
    metadata = {"seed": seed, "copies": copies, "config": asdict(cfg),
                "source_root": str(root), "source_manifest": str(Path(manifest_path).resolve()),
                "counts": {s: len(v) for s, v in generated.items()},
                "dropped_augmented_boxes": dropped_boxes, "source_images": provenance}
    (output / "augmentation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "data/augmented_birds")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=REPO / "splits/official.json")
    parser.add_argument("--copies", type=int, default=3, help="additional copies per training image")
    parser.add_argument("--seed", type=int, default=42)
    defaults = AugmentConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=value)
    args = parser.parse_args()
    try:
        config = AugmentConfig(**{name: getattr(args, name) for name in asdict(defaults)})
        metadata = export_dataset(args.output, args.copies, args.seed, config,
                                  args.data_root, args.manifest)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Saved to {args.output.resolve()}: {metadata['counts']}")
    print(f"Filtered augmented boxes: {metadata['dropped_augmented_boxes']}")


if __name__ == "__main__":
    main()

"""Temporary dataset loader for Method 1, until Member A's loader lands.

Assumed layout (YOLO txt; confirm against the real download):
    <root>/<split>/images/<image_id>.jpg
    <root>/<split>/labels/<image_id>.txt   lines of "class cx cy w h", normalised
A missing or empty label file means a negative image.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageRecord:
    image_id: str
    path: Path
    boxes: np.ndarray  # (N, 4) float xyxy in pixels


def _read_yolo_boxes(label_path, width, height):
    if not label_path.exists():
        return np.zeros((0, 4))
    rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 4))
    cx, cy, w, h = np.array(rows, dtype=float)[:, 1:5].T
    return np.stack([(cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height], axis=1)


def load_split(split, root="data"):
    """Return the split's ImageRecords, sorted by image_id for reproducible runs."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    split_dir = Path(root) / split
    image_dir = split_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"no images directory at {image_dir}")

    records = []
    for path in sorted(image_dir.iterdir(), key=lambda p: p.stem):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        with Image.open(path) as image:  # reads the header only
            width, height = image.size
        boxes = _read_yolo_boxes(split_dir / "labels" / f"{path.stem}.txt", width, height)
        records.append(ImageRecord(path.stem, path, boxes))
    return records

"""Shared dataset interface. Everyone should load data through this file.

    from data.dataset import load_split, get_image

    records = load_split(DEFAULT_ROOT, "val")
    img = get_image(records[0])      # numpy array, RGB
    records[0].boxes                 # numpy array, pixels: x1, y1, x2, y2

Run scripts/check_dataset.py first. Bad label lines are skipped here.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .validation import SPLITS, list_images

DEFAULT_ROOT = (Path(__file__).resolve().parents[2]
                / "data" / "Beach Bird Detection Dataset v1.0")


@dataclass
class DetectionRecord:
    image_id: str       # file name without extension
    image_path: Path
    width: int          # pixels
    height: int         # pixels
    boxes: np.ndarray   # shape (N, 4), pixels: x1, y1, x2, y2 (N can be 0)


def _read_boxes(label_path, width, height):
    """YOLO line (class cx cy w h, values 0-1) -> pixel corners."""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = map(float, parts)
            boxes.append([(cx - w / 2) * width, (cy - h / 2) * height,
                          (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def load_split(root, split):
    """List of DetectionRecord for 'train', 'val' or 'test'."""
    root = Path(root)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    img_dir = root / "images" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} not found. Run scripts/check_dataset.py")

    records = []
    for img_path in list_images(img_dir):
        with Image.open(img_path) as im:  # only reads the file header
            width, height = im.size
        label = root / "labels" / split / (img_path.stem + ".txt")
        records.append(DetectionRecord(
            image_id=img_path.stem,
            image_path=img_path,
            width=width,
            height=height,
            boxes=_read_boxes(label, width, height),
        ))
    return records


def load_dataset(root):
    """Dict of all splits: {'train': [...], 'val': [...], 'test': [...]}."""
    return {s: load_split(root, s) for s in SPLITS}


def get_image(record):
    """Load the image as a numpy array (height, width, 3), RGB."""
    with Image.open(record.image_path) as im:
        return np.array(im.convert("RGB"))
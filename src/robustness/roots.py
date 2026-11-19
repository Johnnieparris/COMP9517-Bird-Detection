"""Build derived copies of the dataset ("data roots") with the original folder layout.

A derived root has images/{train,val,test} and labels/{train,val,test}, so every
existing runner (run_m1.py, run_m2.py, run_ml.py, src.dl.train_yolo) reads it via
--data-root unchanged, with its test-set guards intact.

Unchanged images are HARD-linked (no extra disk space). Symlinks are not used:
the YOLO trainer calls Path.resolve(), which would follow a symlink back to the
original folder and silently read the original labels.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import cv2

from src.data.dataset import get_image, load_split
from src.data.validation import SPLITS
from src.dl.augment import write_yolo_labels

REPO = Path(__file__).resolve().parents[2]
DISTORTED_DIR = REPO / "data" / "distorted"
JPEG_QUALITY = 95  # re-encoding quality for transformed images


def link_or_copy(src, dst):
    """Hard-link src to dst; copy instead when linking is impossible (e.g. another filesystem)."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_image(path, image_rgb):
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if path.suffix.lower() in (".jpg", ".jpeg") else []
    if not cv2.imwrite(str(path), image_rgb[:, :, ::-1], params):
        raise IOError(f"could not write {path}")


def build_root(src_root, dst_root, transforms, label_fn=None, info=None):
    """Write a derived dataset root and return its path.

    transforms: {split: None | fn(image_rgb, boxes_xyxy, image_id) -> image_rgb}.
                None (or a missing split) hard-links that split's images unchanged.
    label_fn:   optional fn(boxes_xyxy, image_id, split) -> boxes_xyxy; rewrites labels.
                Without it label files are copied byte-for-byte.
    info:       extra metadata saved in root_info.json.
    Refuses to touch an existing dst_root, so old and new images are never mixed.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    if dst_root.exists():
        raise FileExistsError(f"{dst_root} already exists; delete it or choose another output")
    for split in SPLITS:
        img_out = dst_root / "images" / split
        lbl_out = dst_root / "labels" / split
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        transform = transforms.get(split)
        for record in load_split(src_root, split):
            dst_img = img_out / record.image_path.name
            if transform is None:
                link_or_copy(record.image_path, dst_img)
            else:
                _write_image(dst_img, transform(get_image(record), record.boxes, record.image_id))
            src_lbl = src_root / "labels" / split / f"{record.image_id}.txt"
            dst_lbl = lbl_out / f"{record.image_id}.txt"
            if label_fn is not None:
                boxes = label_fn(record.boxes, record.image_id, split)
                write_yolo_labels(dst_lbl, boxes, record.width, record.height)
            elif src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
    if (src_root / "dataset.yaml").exists():
        shutil.copy2(src_root / "dataset.yaml", dst_root / "dataset.yaml")
    meta = {"source_root": str(src_root), "created_utc": datetime.now(timezone.utc).isoformat(),
            "transformed_splits": sorted(s for s in SPLITS if transforms.get(s) is not None),
            "labels_rewritten": label_fn is not None, **(info or {})}
    (dst_root / "root_info.json").write_text(json.dumps(meta, indent=2))
    return dst_root

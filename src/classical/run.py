"""Run Method 1 over a list of images, separate from the command line."""
import time

import cv2
from PIL import Image

from src.classical.pipeline import detect


def read_image(path):
    """Read a BGR image, raising with the path on a corrupt file.

    cv2.imread fills in a truncated JPEG without an error, so decode it strictly first.
    """
    try:
        with Image.open(path) as image:
            image.load()
    except OSError as error:
        raise IOError(f"could not read image: {path}") from error
    return cv2.imread(str(path))


def run_split(records, cfg):
    """Detect on every record. Returns (prediction rows, stats).

    Images with no detections add no rows but still count in n_images. Only the
    detect() calls are timed, so ms_per_image excludes disk reads.
    """
    rows = []
    detect_seconds = 0.0
    n_images_capped = 0
    for record in records:
        image = read_image(record.path)
        start = time.perf_counter()
        dets = detect(image, cfg)
        detect_seconds += time.perf_counter() - start
        n_images_capped += len(dets) >= cfg.max_detections
        for x1, y1, x2, y2, score in dets.tolist():
            rows.append({"image_id": record.image_id, "x1": x1, "y1": y1,
                         "x2": x2, "y2": y2, "score": score})
    stats = {"n_images": len(records), "n_detections": len(rows),
             "n_images_capped": n_images_capped,
             "ms_per_image": 1000 * detect_seconds / max(len(records), 1)}
    return rows, stats


def draw_overlay(image_bgr, gt_boxes, dets):
    """Return a copy of the image with GT boxes in blue and predictions in green."""
    overlay = image_bgr.copy()
    for boxes, colour in ((gt_boxes, (255, 0, 0)), (dets, (0, 255, 0))):
        for box in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 1)
    return overlay

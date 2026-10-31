"""Method 1 pipeline. detect() is the public interface for other team members."""
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
import yaml

from src.classical.blob import detect_blobs, to_gray
from src.classical.boxes import clip_boxes, filter_boxes, nms


@dataclass
class BlobConfig:
    sigmas: Sequence[float] = (2, 3, 4, 6, 8, 12)  # in original-image pixels
    threshold: float = 0.05
    polarity: str = "both"
    clahe: bool = False
    max_side: int = 1600  # images with a longer side are downscaled first
    nms_iou: float = 0.3
    min_area: float = 16.0
    max_area: Optional[float] = None
    max_detections: int = 300

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls(**yaml.safe_load(f))  # an unknown key raises TypeError


def detect(image_bgr, cfg):
    """Detect birds in a uint8 BGR image.

    Returns (N, 5) detections of x1, y1, x2, y2, score in original-image pixels,
    sorted by score, descending.
    """
    height, width = image_bgr.shape[:2]
    gray = to_gray(image_bgr, cfg.clahe)

    scale = min(1.0, cfg.max_side / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    dets = detect_blobs(gray, np.asarray(cfg.sigmas, float) * scale, cfg.threshold, cfg.polarity)
    dets[:, :4] /= scale
    dets = clip_boxes(dets, width, height)
    dets = filter_boxes(dets, cfg.min_area, cfg.max_area)
    return nms(dets, cfg.nms_iou)[:cfg.max_detections]

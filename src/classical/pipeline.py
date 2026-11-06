"""Method 1 pipeline. detect() is the public interface for other team members."""
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
import yaml

from src.classical.blob import detect_blobs, to_gray
from src.classical.boxes import clip_boxes, filter_boxes, nms
from src.classical.scoring import local_contrast
from src.classical.segment import adaptive_mask, clean_mask, colour_distance_mask, mask_to_boxes


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
    score_mode: str = "log"  # or "contrast"

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            values = yaml.safe_load(f)
        values.pop("method", None)  # written by the tuning script; this class is the blob method
        return cls(**values)  # an unknown key raises TypeError


@dataclass
class SegConfig:
    foreground: str = "colour"  # or "adaptive"
    downscale: int = 4  # colour: the background is estimated at this reduced size
    bg_kernel: int = 51  # colour: median filter size at the reduced size
    min_distance: float = 15.0  # colour: floor on Otsu's threshold, in Lab units
    block: int = 101  # adaptive: neighbourhood size
    c: float = 15.0  # adaptive: offset from the local mean
    morph_kernel: int = 3
    min_area: float = 16.0
    max_aspect: float = 4.0
    min_fill: float = 0.25
    max_side_frac: float = 0.8
    nms_iou: float = 0.3
    max_detections: int = 300


def load_config(path):
    """Read a BlobConfig or SegConfig from YAML, chosen by its optional method key."""
    with open(path) as f:
        values = yaml.safe_load(f)
    method = values.pop("method", "blob")
    if method == "blob":
        return BlobConfig(**values)
    if method == "seg":
        return SegConfig(**values)
    raise ValueError(f"unknown method {method!r}")


def detect(image_bgr, cfg):
    """Detect birds in a uint8 BGR image using a BlobConfig or SegConfig.

    Returns (N, 5) detections of x1, y1, x2, y2, score in original-image pixels,
    sorted by score, descending.
    """
    if isinstance(cfg, SegConfig):
        return _detect_seg(image_bgr, cfg)
    return _detect_blob(image_bgr, cfg)


def _detect_seg(image_bgr, cfg):
    if cfg.foreground == "colour":
        mask = colour_distance_mask(image_bgr, cfg.downscale, cfg.bg_kernel, cfg.min_distance)
    elif cfg.foreground == "adaptive":
        mask = adaptive_mask(image_bgr, cfg.block, cfg.c)
    else:
        raise ValueError(f"unknown foreground {cfg.foreground!r}")
    boxes = mask_to_boxes(clean_mask(mask, cfg.morph_kernel), cfg.min_area, cfg.max_aspect,
                          cfg.min_fill, cfg.max_side_frac)
    scores = local_contrast(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB), boxes)
    return nms(np.column_stack([boxes, scores]), cfg.nms_iou)[:cfg.max_detections]


def _detect_blob(image_bgr, cfg):
    if cfg.score_mode not in ("log", "contrast"):
        raise ValueError(f"unknown score_mode {cfg.score_mode!r}")
    height, width = image_bgr.shape[:2]
    gray = to_gray(image_bgr, cfg.clahe)

    scale = min(1.0, cfg.max_side / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    dets = detect_blobs(gray, np.asarray(cfg.sigmas, float) * scale, cfg.threshold, cfg.polarity)
    dets[:, :4] /= scale
    dets = clip_boxes(dets, width, height)
    dets = filter_boxes(dets, cfg.min_area, cfg.max_area)
    if cfg.score_mode == "contrast":
        dets[:, 4] = local_contrast(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB), dets)
    return nms(dets, cfg.nms_iou)[:cfg.max_detections]

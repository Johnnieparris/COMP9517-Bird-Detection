"""Foreground segmentation for Method 1: masks, clean-up and component boxes.

Masks are uint8 H x W arrays of 0 and 1. Thresholding follows Otsu (1979) and
mean-based adaptive thresholding (Bradley & Roth, 2007).
"""
import cv2
import numpy as np


def colour_distance_mask(image_bgr, downscale, bg_kernel, min_distance):
    """Mark pixels whose Lab colour is far from a median-filtered background.

    The cut-off is Otsu's threshold on the distances, but never below min_distance:
    Otsu always splits an image in two, even one with nothing in it.
    """
    height, width = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    small = cv2.resize(lab, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_AREA)
    background = cv2.resize(cv2.medianBlur(small, bg_kernel), (width, height),
                            interpolation=cv2.INTER_LINEAR)
    distance = np.linalg.norm(lab.astype(np.float32) - background.astype(np.float32), axis=2)
    distance = np.clip(distance, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(distance, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (distance > max(otsu, min_distance)).astype(np.uint8)


def adaptive_mask(image_bgr, block, c):
    """Mark pixels at least c darker or brighter than their local block mean."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    dark = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, c)
    bright = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, -c)
    return dark | bright


def clean_mask(mask, kernel):
    """Open to remove specks, then close to fill pinholes. kernel <= 1 changes nothing."""
    if kernel <= 1:
        return mask
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, element)


def mask_to_boxes(mask, min_area, max_aspect, min_fill, max_side_frac):
    """Return (N, 4) xyxy boxes of the mask's bird-shaped connected components."""
    height, width = mask.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x, y, w, h, area = stats[1:].T  # row 0 is the background
    keep = ((area >= min_area)
            & (np.maximum(w / h, h / w) <= max_aspect)
            & (area / (w * h) >= min_fill)
            & (w <= max_side_frac * width)
            & (h <= max_side_frac * height))
    return np.stack([x, y, x + w, y + h], axis=1)[keep].astype(float)

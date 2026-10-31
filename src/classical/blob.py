"""Multi-scale Laplacian of Gaussian blob detection (Lindeberg, IJCV 1998).

Written here rather than using skimage.feature.blob_log because mAP needs a
confidence score for each box, and blob_log does not return the response.
"""
import cv2
import numpy as np
from scipy import ndimage

SQRT2 = np.sqrt(2)


def to_gray(image_bgr, clahe=False):
    """Convert a uint8 BGR image to float32 grayscale in [0, 1], optionally with CLAHE."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray.astype(np.float32) / 255.0


def detect_blobs(gray, sigmas, threshold, polarity):
    """Find blobs as local maxima in (x, y, scale) of the scale-normalised LoG.

    polarity is "dark" (dark blobs on a light background), "bright" or "both".
    A disc of radius r peaks at sigma = r / sqrt(2), so each box is a square of
    side 2 * sqrt(2) * sigma. Returns (N, 5) detections sorted by score, descending.
    """
    if polarity not in ("dark", "bright", "both"):
        raise ValueError(f"unknown polarity {polarity!r}")
    sigmas = np.asarray(sigmas, float)

    # The LoG is positive at the centre of a dark blob and negative at a bright one.
    stack = np.stack([s ** 2 * ndimage.gaussian_laplace(gray, s) for s in sigmas])
    if polarity == "bright":
        stack = -stack
    elif polarity == "both":
        stack = np.abs(stack)

    is_peak = (stack == ndimage.maximum_filter(stack, size=3)) & (stack > threshold)
    scale_idx, ys, xs = np.nonzero(is_peak)
    half = SQRT2 * sigmas[scale_idx]
    dets = np.stack([xs - half, ys - half, xs + half, ys + half, stack[is_peak]], axis=1)
    return dets[np.argsort(-dets[:, 4], kind="stable")]


def suggest_sigmas(boxes, n=6):
    """Pick n sigmas spanning the 5th-95th percentile of the given boxes' sizes."""
    if not len(boxes):
        raise ValueError("cannot suggest sigmas from no boxes")
    sides = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    low, high = np.maximum(np.percentile(sides / (2 * SQRT2), [5, 95]), 1.0)
    return np.geomspace(low, high, n).tolist()

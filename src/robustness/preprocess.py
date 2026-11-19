"""Image preprocessing used as the classical methods' robustness mitigation (Member E, Study A).

Optional denoising (bilateral filter or non-local means), then CLAHE on the L channel of
Lab. One global setting is chosen on clean + distorted VAL by scripts/tune_preprocess.py.
"""
import cv2
import numpy as np

PREPROCESS_GRID = {
    "none": {},
    "clahe": {"clahe_clip": 2.0},
    "bilateral_clahe": {"bilateral_d": 7, "sigma": 50, "clahe_clip": 2.0},
    "nlm_clahe": {"nlm_h": 10, "clahe_clip": 2.0},
    "nlm_strong_clahe": {"nlm_h": 20, "clahe_clip": 3.0},
    # Denoise-only settings: the spec chooses which steps are on, not only their strengths.
    "bilateral": {"bilateral_d": 7, "sigma": 50},
    "nlm_light": {"nlm_h": 5},
    "nlm": {"nlm_h": 10},
    "median3": {"median": 3},
}
CLAHE_TILES = (8, 8)


def preprocess(image_rgb, setting):
    """Return a preprocessed uint8 RGB copy. An empty setting returns the image unchanged."""
    out = np.ascontiguousarray(image_rgb).copy()
    if "median" in setting:
        out = cv2.medianBlur(out, setting["median"])
    if "bilateral_d" in setting:
        out = cv2.bilateralFilter(out, setting["bilateral_d"], setting["sigma"], setting["sigma"])
    if "nlm_h" in setting:
        out = cv2.fastNlMeansDenoisingColored(out, None, setting["nlm_h"], setting["nlm_h"], 7, 21)
    if "clahe_clip" in setting:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=setting["clahe_clip"], tileGridSize=CLAHE_TILES)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out

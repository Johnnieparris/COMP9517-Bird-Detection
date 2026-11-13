"""Handcrafted patch descriptors for Method 2.

Every extractor takes an RGB uint8 patch (the channel order data.dataset.get_image
returns), resizes it to a fixed square and returns a 1-D float vector.

    HOG         Dalal & Triggs, CVPR 2005      shape / gradient structure
    uniform LBP Ojala et al., TPAMI 2002       local texture
    HSV colour  histograms + channel moments   appearance

Implemented with scikit-image and OpenCV.
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

KINDS = ("hog", "lbp", "colour", "concat")

HOG_ORIENTATIONS = 9
HOG_CELL = (8, 8)
HOG_BLOCK = (2, 2)

# LBP: 8 neighbours at radius 1, pooled over a 2x2 grid of cells.
LBP_P = 8
LBP_R = 1
LBP_GRID = 2

COLOUR_BINS = 8


def f_hog(gray):
    """Gradient-orientation histogram, L2-Hys block normalised."""
    return hog(gray, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_CELL,
               cells_per_block=HOG_BLOCK, block_norm="L2-Hys", feature_vector=True)


def f_lbp(gray, grid=LBP_GRID, p=LBP_P, r=LBP_R):
    """Uniform-LBP histogram per cell of a grid x grid layout.

    'uniform' folds the 2**p patterns into p + 2 bins, which keeps the descriptor
    small and robust to monotonic lighting change.
    """
    lbp = local_binary_pattern(gray, p, r, method="uniform")
    n_bins = p + 2
    h, w = lbp.shape
    out = []
    for i in range(grid):
        for j in range(grid):
            cell = lbp[i * h // grid:(i + 1) * h // grid,
                       j * w // grid:(j + 1) * w // grid]
            hist, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins))
            out.append(hist / max(hist.sum(), 1))
    return np.concatenate(out)


def f_colour(rgb, bins=COLOUR_BINS):
    """Per-channel HSV histograms plus channel mean and standard deviation.

    The ranges below are OpenCV's: hue is 0..179, saturation and value 0..255.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    out = []
    for c, hi in enumerate((180, 256, 256)):
        hist, _ = np.histogram(hsv[:, :, c], bins=bins, range=(0, hi))
        out.append(hist / max(hist.sum(), 1))
        out.append([hsv[:, :, c].mean() / hi, hsv[:, :, c].std() / hi])
    return np.concatenate(out)


def extract(patch_rgb, kind="hog", size=32):
    """Feature vector for one RGB patch. `kind` is one of KINDS."""
    patch = cv2.resize(patch_rgb, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    if kind == "hog":
        return f_hog(gray)
    if kind == "lbp":
        return f_lbp(gray)
    if kind == "colour":
        return f_colour(patch)
    if kind == "concat":
        return np.concatenate([f_hog(gray), f_lbp(gray), f_colour(patch)])
    raise ValueError(f"unknown feature kind: {kind!r}")


def extract_many(patches_rgb, kind="hog", size=32):
    """(N, D) float32 matrix for a list of RGB patches."""
    if not len(patches_rgb):
        return np.zeros((0, feature_length(kind, size)), np.float32)
    return np.array([extract(p, kind, size) for p in patches_rgb], dtype=np.float32)


def feature_length(kind="hog", size=32):
    """Length of the vector `extract` returns, without building a real patch."""
    return len(extract(np.zeros((size, size, 3), np.uint8), kind, size))

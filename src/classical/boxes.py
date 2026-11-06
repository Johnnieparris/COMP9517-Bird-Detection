"""Box utilities. Detections are (N, 5) arrays of x1, y1, x2, y2, score."""
import numpy as np


def iou_matrix(a, b):
    """Pairwise IoU between (N, >=4) and (M, >=4) arrays, as an (N, M) array."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float),
                     where=union > 0)


def clip_boxes(dets, width, height):
    """Clip boxes to the image and drop any left with no area."""
    clipped = dets.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    return clipped[(clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])]


def filter_boxes(dets, min_area, max_area=None):
    """Keep boxes with area in [min_area, max_area]; max_area None means no upper limit."""
    area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    keep = area >= min_area
    if max_area is not None:
        keep &= area <= max_area
    return dets[keep]


def nms(dets, iou_thr):
    """Greedy non-maximum suppression; returns detections sorted by score, descending."""
    remaining = dets[np.argsort(-dets[:, 4], kind="stable")]
    kept = []
    while len(remaining):
        best, remaining = remaining[0], remaining[1:]
        kept.append(best)
        remaining = remaining[iou_matrix(best[None], remaining)[0] <= iou_thr]
    return np.array(kept) if kept else np.zeros((0, 5))

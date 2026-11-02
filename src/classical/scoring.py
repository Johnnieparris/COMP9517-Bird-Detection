"""Confidence score for a box: how much it stands out from its surroundings."""
import cv2
import numpy as np


def _box_sum(integral, x1, y1, x2, y2):
    return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]


def local_contrast(lab, boxes, pad=0.5, eps=5.0):
    """Score (N, >=4) boxes on an H x W x 3 Lab image.

    The score is the colour distance between the box mean and the mean of a
    surrounding ring, over the ring's spread plus eps. The ring is the box grown by
    pad times its size on each side, clipped to the image. An empty ring scores 0.
    """
    height, width = lab.shape[:2]
    if not len(boxes):
        return np.zeros(0)
    values = lab.astype(np.float64)
    integral, integral_sq = cv2.integral(values), cv2.integral(values ** 2)

    x1 = np.clip(np.round(boxes[:, 0]), 0, width - 1).astype(int)
    y1 = np.clip(np.round(boxes[:, 1]), 0, height - 1).astype(int)
    x2 = np.clip(np.maximum(np.round(boxes[:, 2]).astype(int), x1 + 1), None, width)
    y2 = np.clip(np.maximum(np.round(boxes[:, 3]).astype(int), y1 + 1), None, height)
    pad_x, pad_y = np.ceil(pad * (x2 - x1)).astype(int), np.ceil(pad * (y2 - y1)).astype(int)
    outer_x1, outer_y1 = np.clip(x1 - pad_x, 0, None), np.clip(y1 - pad_y, 0, None)
    outer_x2, outer_y2 = np.clip(x2 + pad_x, None, width), np.clip(y2 + pad_y, None, height)

    n_box = ((x2 - x1) * (y2 - y1))[:, None]
    n_ring = ((outer_x2 - outer_x1) * (outer_y2 - outer_y1))[:, None] - n_box
    box_sum = _box_sum(integral, x1, y1, x2, y2)
    ring_sum = _box_sum(integral, outer_x1, outer_y1, outer_x2, outer_y2) - box_sum
    ring_sum_sq = (_box_sum(integral_sq, outer_x1, outer_y1, outer_x2, outer_y2)
                   - _box_sum(integral_sq, x1, y1, x2, y2))

    has_ring = n_ring[:, 0] > 0
    safe_n_ring = np.maximum(n_ring, 1)
    ring_mean = ring_sum / safe_n_ring
    ring_var = np.clip(ring_sum_sq / safe_n_ring - ring_mean ** 2, 0, None).sum(axis=1)
    distance = np.linalg.norm(box_sum / n_box - ring_mean, axis=1)
    return np.where(has_ring, distance / (np.sqrt(ring_var) + eps), 0.0)

"""Synthetic label noise for the label-efficiency study (Member E, Study B)."""
import numpy as np


def jitter_boxes(boxes, frac, rng, width, height):
    """Shift every box edge by U(-frac, frac) of the box width (x edges) or height (y edges).

    Results are clipped to the image and kept at least 1 px wide and tall, so no label
    is silently dropped by the trainer.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return boxes.copy()
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    scale = np.stack([w, h, w, h], axis=1)
    out = boxes + rng.uniform(-frac, frac, boxes.shape).astype(np.float32) * scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height)
    x1 = np.minimum(out[:, 0], out[:, 2]); x2 = np.maximum(out[:, 0], out[:, 2])
    y1 = np.minimum(out[:, 1], out[:, 3]); y2 = np.maximum(out[:, 1], out[:, 3])
    # Enforce the 1 px minimum without leaving the image.
    x2 = np.maximum(x2, np.minimum(x1 + 1, width)); x1 = np.minimum(x1, x2 - 1)
    y2 = np.maximum(y2, np.minimum(y1 + 1, height)); y1 = np.minimum(y1, y2 - 1)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def drop_boxes(boxes, frac, rng):
    """Remove round(frac * N) boxes chosen uniformly at random (keeps the original order)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n_drop = int(round(frac * len(boxes)))
    if n_drop == 0:
        return boxes.copy()
    drop = rng.choice(len(boxes), n_drop, replace=False)
    return np.delete(boxes, drop, axis=0)

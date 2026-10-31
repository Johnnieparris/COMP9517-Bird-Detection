"""Hand-written AP@0.5 and recall for fast feedback on Method 1.

Also serves as the hand-written cross-check for the team's shared evaluator.
"""
import numpy as np

from src.classical.boxes import iou_matrix


def _match(rows, records, iou_thr):
    """Greedily match rows, best score first. Returns (is_tp per ranked row, total GT)."""
    gt_by_image = {r.image_id: r.boxes for r in records}
    unknown = {row["image_id"] for row in rows} - gt_by_image.keys()
    if unknown:
        raise ValueError(f"predictions for images not in this split: {sorted(unknown)[:5]}")
    n_gt = sum(len(boxes) for boxes in gt_by_image.values())

    matched = {image_id: np.zeros(len(boxes), bool) for image_id, boxes in gt_by_image.items()}
    ranked = sorted(rows, key=lambda row: -row["score"])
    is_tp = np.zeros(len(ranked), bool)
    for i, row in enumerate(ranked):
        boxes = gt_by_image[row["image_id"]]
        if not len(boxes):
            continue
        pred = np.array([[row["x1"], row["y1"], row["x2"], row["y2"]]])
        ious = iou_matrix(pred, boxes)[0]
        ious[matched[row["image_id"]]] = -1
        best = int(ious.argmax())
        if ious[best] >= iou_thr:
            matched[row["image_id"]][best] = True
            is_tp[i] = True
    return is_tp, n_gt


def ap50(rows, records):
    """All-point interpolated average precision at IoU 0.5."""
    is_tp, n_gt = _match(rows, records, 0.5)
    if n_gt == 0:
        raise ValueError("AP is undefined for a split with no ground-truth boxes")
    if not len(is_tp):
        return 0.0
    tp = np.cumsum(is_tp)
    recall = np.concatenate([[0.0], tp / n_gt])
    precision = tp / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * envelope))


def recall_at_iou(rows, records, iou=0.5):
    """Share of GT boxes matched by a prediction at the given IoU."""
    is_tp, n_gt = _match(rows, records, iou)
    if n_gt == 0:
        raise ValueError("recall is undefined for a split with no ground-truth boxes")
    return float(is_tp.sum() / n_gt)

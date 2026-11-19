"""Automatic failure categories for one image's detections (Member E, Study C).

Same greedy matching as the shared evaluator (highest score first, IoU >= 0.5). Then:
    fn_dense         missed bird with >= 3 other bird centres within 2x its box diagonal
    fn_isolated      any other missed bird
    fp_duplicate     false positive with IoU >= 0.5 to a bird another detection already took
    fp_localisation  false positive with 0.1 <= best IoU < 0.5 (right place, wrong box)
    fp_background    false positive with best IoU < 0.1 (rock, shell, wave, ...)
"""
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation import IOU_THR, _greedy_match, box_iou  # noqa: E402

CATEGORIES = ("tp", "fn_dense", "fn_isolated", "fp_duplicate", "fp_localisation", "fp_background")
DENSE_NEIGHBOURS = 3
DENSE_RADIUS = 2.0  # in box diagonals
LOCALISATION_MIN_IOU = 0.1


def categorise(preds, gts, thr, iou_thr=IOU_THR):
    """Counts per category for detections with score >= thr against real boxes gts."""
    preds = np.asarray(preds, dtype=float).reshape(-1, 5)
    preds = preds[preds[:, 4] >= thr]
    preds = preds[np.argsort(-preds[:, 4], kind="stable")]
    gts = np.asarray(gts, dtype=float).reshape(-1, 4)
    counts = dict.fromkeys(CATEGORIES, 0)
    match = _greedy_match(preds, gts, iou_thr)
    counts["tp"] = int((match >= 0).sum())

    best_iou = box_iou(preds[:, :4], gts).max(axis=1) if len(gts) and len(preds) else np.zeros(len(preds))
    for i in np.flatnonzero(match < 0):
        if best_iou[i] >= iou_thr:
            counts["fp_duplicate"] += 1
        elif best_iou[i] >= LOCALISATION_MIN_IOU:
            counts["fp_localisation"] += 1
        else:
            counts["fp_background"] += 1

    found = np.zeros(len(gts), bool)
    found[match[match >= 0]] = True
    centres = (gts[:, :2] + gts[:, 2:]) / 2
    diag = np.hypot(gts[:, 2] - gts[:, 0], gts[:, 3] - gts[:, 1])
    for j in np.flatnonzero(~found):
        dist = np.hypot(*(centres - centres[j]).T)
        neighbours = int((dist <= DENSE_RADIUS * diag[j]).sum()) - 1  # minus itself
        counts["fn_dense" if neighbours >= DENSE_NEIGHBOURS else "fn_isolated"] += 1
    return counts

"""Diagnostic: does a model treat the synthetic grey occluder itself as a bird? (Member E, Study A)

    python scripts/occluder_shortcut_check.py

On every VAL image, paste 5 grey (128) rectangles the size of a median bird box at random
background positions that overlap no ground-truth box, run each YOLO model, and count
detections (at the model's clean-val threshold) whose box overlaps a pasted rectangle with
IoU >= 0.5. A model that learned "grey patch = bird" fires on them. Uses val only.
Writes results/occluder_shortcut_val.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.data.dataset import get_image  # noqa: E402
from src.dl.inference import load_detector  # noqa: E402
from src.robustness.distortions import OCCLUDER_GREY, rng_for  # noqa: E402
from evaluation import IOU_THR, best_f1_threshold, box_iou, load_records  # noqa: E402
from predictions import PRED_DIR, load_predictions  # noqa: E402

MODELS = {  # name -> (inference config, folder holding <method>_val.csv for the threshold, method)
    "yolo": ("configs/yolo_best.yaml", PRED_DIR, "yolo"),
    "e_de_100": ("configs/e/infer_de_100.yaml", PRED_DIR / "data_efficiency/de_100", "yolo"),
    "e_robust_aug": ("configs/e/infer_robust_aug.yaml", PRED_DIR / "data_efficiency/robust_aug", "yolo"),
}
N_PATCHES = 5


def paste(image, gts, rng, side):
    """Grey squares of the given side at random spots that overlap no GT box."""
    h, w = image.shape[:2]
    out, boxes = image.copy(), []
    for _ in range(200):
        if len(boxes) == N_PATCHES:
            break
        x, y = int(rng.integers(0, w - side)), int(rng.integers(0, h - side))
        box = np.array([[x, y, x + side, y + side]], float)
        others = np.vstack([gts.reshape(-1, 4)] + [b for b in boxes]) if len(gts) or boxes else np.zeros((0, 4))
        if len(others) and box_iou(box, others).max() > 0:
            continue
        out[y:y + side, x:x + side] = OCCLUDER_GREY
        boxes.append(box)
    return out, np.vstack(boxes) if boxes else np.zeros((0, 4))


def main():
    records = load_records(None, "val")
    sides = np.concatenate([np.sqrt((r.boxes[:, 2] - r.boxes[:, 0]) * (r.boxes[:, 3] - r.boxes[:, 1]))
                            for r in records if len(r.boxes)])
    side = int(round(np.median(sides)))
    rows = []
    for name, (config, folder, method) in MODELS.items():
        thr = best_f1_threshold(records, load_predictions(method, "val", folder), IOU_THR)["thr"]
        predict, _ = load_detector(config, device="cpu")
        hits, patches = 0, 0
        for r in records:
            img, boxes = paste(get_image(r), r.boxes, rng_for(0, r.image_id, "shortcut"), side)
            dets = predict(img)
            dets = dets[dets[:, 4] >= thr]
            patches += len(boxes)
            if len(dets) and len(boxes):
                hits += int((box_iou(boxes, dets[:, :4]).max(axis=1) >= 0.5).sum())
        rows.append({"model": name, "thr": thr, "patch_side_px": side, "patches": patches,
                     "patches_detected": hits, "rate": hits / patches})
        print(rows[-1], flush=True)
    with (REPO / "results/occluder_shortcut_val.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

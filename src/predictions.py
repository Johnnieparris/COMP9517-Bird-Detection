"""Publish every method's predictions in ONE shared place, in ONE format.

Each method saves one CSV per split in the predictions/ folder:

    predictions/m1_val.csv
    predictions/yolo_val.csv      <- columns: method, split, image_id, x1, y1, x2, y2, score

Why one file per method and split, and not a single big file? Each method can
be run on its own (or at the same time as others) without touching anyone
else's results. load_all_predictions() still gives you everything as ONE table,
and export_combined_csv() writes it as ONE csv file.

    from predictions import save_predictions, load_predictions, load_all_predictions

    preds, timing = run_and_time(records, my_method)
    save_predictions("m1", "val", records, preds)            # publish
    preds = load_predictions("m1", "val")                    # use with evaluate()
    table = load_all_predictions()                           # all methods, one table

These files are GENERATED, not stored in git: predictions/ is in .gitignore.
To compare methods, run each one on your own computer first.
"""
import csv
import re
from pathlib import Path

import numpy as np

from data.validation import SPLITS

PRED_DIR = Path(__file__).resolve().parents[1] / "predictions"
COLUMNS = ["method", "split", "image_id", "x1", "y1", "x2", "y2", "score"]
MAX_PER_IMAGE = 200  # keep only the best detections per image, so files stay small


def prediction_path(method, split, folder=PRED_DIR):
    return Path(folder) / f"{method}_{split}.csv"


def _check_names(method, split):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(method)):
        raise ValueError(f"method name '{method}' may only use letters, numbers, - and _")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")


def save_predictions(method, split, records, predictions, folder=PRED_DIR,
                     max_per_image=MAX_PER_IMAGE):
    """Publish a method's predictions for one split. Replaces the method's old file.

    records:     the records you ran the method on (load_split_from_manifest(...))
    predictions: {image_id: (N, 5) array of x1, y1, x2, y2, score} -- must contain
                 EVERY record (an empty array means 'no birds found').
    Returns the path of the file.
    """
    _check_names(method, split)
    ids = [r.image_id for r in records]
    unknown = sorted(set(predictions) - set(ids))
    missing = sorted(set(ids) - set(predictions))
    if unknown:
        raise ValueError(f"{len(unknown)} predicted images are not in these records "
                         f"(wrong split?): {', '.join(unknown[:5])}")
    if missing:
        raise ValueError(f"No predictions for {len(missing)} of {len(ids)} images "
                         f"(use an empty array if none found): {', '.join(missing[:5])}")

    rows = []
    for image_id in sorted(ids):
        p = np.asarray(predictions[image_id], dtype=float)
        if p.size == 0:
            continue
        if p.ndim != 2 or p.shape[1] != 5:
            raise ValueError(f"{image_id}: predictions must have shape (N, 5), got {p.shape}")
        if not np.isfinite(p).all():
            raise ValueError(f"{image_id}: predictions contain nan or inf")
        p = p[np.argsort(-p[:, 4], kind="stable")][:max_per_image]
        for x1, y1, x2, y2, score in p:
            rows.append([method, split, image_id,
                         f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", f"{score:.5f}"])

    path = prediction_path(method, split, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    return path


def load_predictions(method, split, folder=PRED_DIR):
    """Read one method's predictions back as {image_id: (N, 5) array}.
    Images with no detections are not in the dict (evaluate() treats them as empty)."""
    _check_names(method, split)
    path = prediction_path(method, split, folder)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Has '{method}' published its {split} predictions?")
    boxes = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            boxes.setdefault(row["image_id"], []).append(
                [float(row[c]) for c in ("x1", "y1", "x2", "y2", "score")])
    return {k: np.array(v) for k, v in boxes.items()}


def list_published(folder=PRED_DIR):
    """[(method, split), ...] for every published file."""
    out = []
    for p in sorted(Path(folder).glob("*_*.csv")):
        method, split = p.stem.rsplit("_", 1)
        if split in SPLITS:
            out.append((method, split))
    return out


def load_all_predictions(folder=PRED_DIR):
    """Every method and split in ONE list. One dict per detection:
    {'method': 'm1', 'split': 'val', 'image_id': ..., 'x1': ..., 'score': ...}.
    Numbers are already floats. For a pandas table: pandas.DataFrame(rows)."""
    files = [prediction_path(m, s, folder) for m, s in list_published(folder)]
    if not files:
        raise FileNotFoundError(f"No prediction files in {folder}")
    rows = []
    for path in files:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                for c in ("x1", "y1", "x2", "y2", "score"):
                    row[c] = float(row[c])
                rows.append(row)
    return rows




def export_combined_csv(path=None, folder=PRED_DIR):
    """Write every method and split into ONE csv file (needs pandas).
    Default: predictions/combined.csv. Handy for Excel or sharing outside git."""
    table = load_all_predictions(folder)
    path = Path(path) if path else Path(folder) / "combined.csv"
    table.to_csv(path, index=False)
    return path
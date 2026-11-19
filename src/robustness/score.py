"""Score published predictions on clean and distorted variants (Member E, Study A).

Uses the shared evaluator. Precision, recall and count MAE use each method's best-F1
confidence threshold chosen on CLEAN VAL (as src/analysis.py does); mAP is threshold-free.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:  # shared evaluator modules use top-level imports
    sys.path.insert(0, str(SRC))

from evaluation import (IOU_THR, SCORE_THR, average_precision, best_f1_threshold,  # noqa: E402
                        counting_error, load_records, map_50_95, precision_recall_f1)
from predictions import PRED_DIR, load_predictions, prediction_path  # noqa: E402

COLUMNS = ["method", "variant", "distortion", "severity", "split", "thr", "mAP50", "mAP50_95",
           "precision", "recall", "count_mae"]


def parse_variant(variant):
    """'jpeg_s3' -> ('jpeg', 3); 'clean' -> ('clean', 0)."""
    if variant == "clean":
        return "clean", 0
    name, _, level = variant.rpartition("_s")
    if not name or not level.isdigit():
        raise ValueError(f"bad variant name {variant!r}")
    return name, int(level)


def clean_val_thresholds(methods, folder=PRED_DIR, root=None, manifest_path=None):
    """{method: best-F1 threshold on clean val}; SCORE_THR when a method has no val file."""
    records = load_records(root, "val", manifest_path)
    out = {}
    for method in methods:
        if not prediction_path(method, "val", folder).exists():
            out[method] = SCORE_THR
            continue
        best = best_f1_threshold(records, load_predictions(method, "val", folder), IOU_THR)
        out[method] = best["thr"] if best else SCORE_THR
    return out


def score_variant(method, variant, split, root, folder, thr, manifest_path=None):
    """One results row for <folder>/<method>_<split>.csv evaluated on the boxes under root."""
    records = load_records(root, split, manifest_path)
    preds = load_predictions(method, split, folder)
    prf = precision_recall_f1(records, preds, IOU_THR, thr)
    distortion, severity = parse_variant(variant)
    return {"method": method, "variant": variant, "distortion": distortion, "severity": severity,
            "split": split, "thr": thr, "mAP50": average_precision(records, preds, IOU_THR),
            "mAP50_95": map_50_95(records, preds), "precision": prf["precision"],
            "recall": prf["recall"], "count_mae": counting_error(records, preds, thr)["mae"]}

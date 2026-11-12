"""Write results/m2_comparison.csv from Method 2 prediction files and their sidecars.

Mirrors scripts/compare_m1.py so the two methods produce the same columns, but
every metric here comes from the shared src/evaluation.py rather than a
method-local scorer.

Precision, recall and F1 are reported at the score threshold that maximises F1
on this split, and that threshold is recorded. A single fixed cut-off is not
comparable across methods whose scores mean different things: Method 2 trains
with class_weight="balanced", so its scores are balanced-prior posteriors, and
most candidate boxes sit above 0.5. mAP@0.5 is threshold-free and is the number
to compare directly.

Example:
  python scripts/compare_m2.py --split val --pred results/m2_val_*.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

import numpy as np

from data.dataset import DEFAULT_ROOT
from data.split import load_split_from_manifest, load_split_manifest
from evaluation import (average_precision, counting_error, map_50_95,
                        precision_recall_f1)
from src.ml.config import M2Config

MANIFEST = ROOT / "splits" / "official.json"


def rows_to_predictions(rows, records):
    """Prediction rows -> {image_id: (N, 5) array}, the shape evaluate() wants."""
    grouped = {record.image_id: [] for record in records}
    for row in rows:
        if row["image_id"] in grouped:
            grouped[row["image_id"]].append(
                [row["x1"], row["y1"], row["x2"], row["y2"], row["score"]])
    return {image_id: np.array(dets, dtype=float).reshape(-1, 5)
            for image_id, dets in grouped.items()}


def best_f1(records, predictions, n_steps=60):
    """Sweep the score threshold and keep the best F1, using shared scoring."""
    scores = np.concatenate([p[:, 4] for p in predictions.values() if len(p)]) \
        if any(len(p) for p in predictions.values()) else np.array([0.0])
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, n_steps)))
    best = {"f1": -1.0}
    for threshold in candidates:
        stats = precision_recall_f1(records, predictions, score_thr=float(threshold))
        if stats["f1"] > best["f1"]:
            best = {**stats, "score_threshold": float(threshold)}
    return best


def compare_row(pred_path, records, split=None):
    pred_path = Path(pred_path)
    rows = json.loads(pred_path.read_text())
    meta = json.loads(pred_path.with_suffix(".meta.json").read_text())
    if split is not None and meta["split"] != split:
        raise ValueError(f"{pred_path} is from {meta['split']}, not {split}")
    predictions = rows_to_predictions(rows, records)
    cfg = meta["config"]
    f1 = best_f1(records, predictions)
    counts = counting_error(records, predictions, score_thr=f1["score_threshold"])

    # Rebuilt through M2Config so the label matches cfg.variant exactly, and two
    # runs differing only in a non-headline setting cannot collide.
    variant = M2Config(**cfg).variant
    return {"variant": variant,
            "split": meta["split"],
            "feature": cfg["feature"], "clf": cfg["clf"],
            "hn_rounds": cfg["hn_rounds"], "proposals": cfg["proposals"],
            "mAP50": average_precision(records, predictions),
            "mAP50_95": map_50_95(records, predictions),
            "precision": f1["precision"], "recall": f1["recall"], "f1": f1["f1"],
            "count_mae": counts["mae"], "score_threshold": f1["score_threshold"],
            "ms_per_image": meta["stats"]["ms_per_image"],
            "train_time_s": meta["train_time_s"],
            "machine": meta["machine"], "git_sha": meta["git_sha"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", nargs="+", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default=str(ROOT / "results" / "m2_comparison.csv"))
    args = parser.parse_args()

    manifest = load_split_manifest(MANIFEST)
    records = load_split_from_manifest(args.data_root, manifest, args.split)
    table = sorted((compare_row(p, records, args.split) for p in args.pred),
                   key=lambda r: -r["mAP50"])

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    print(f"{'variant':34s} {'mAP@0.5':>8s} {'P':>7s} {'R':>7s} {'F1':>7s} "
          f"{'MAE':>7s} {'thr':>7s} {'train s':>8s} {'ms/img':>8s}")
    for row in table:
        print(f"{row['variant']:34s} {row['mAP50']:8.4f} {row['precision']:7.3f} "
              f"{row['recall']:7.3f} {row['f1']:7.3f} {row['count_mae']:7.2f} "
              f"{row['score_threshold']:7.3f} {row['train_time_s']:8.1f} "
              f"{row['ms_per_image']:8.1f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

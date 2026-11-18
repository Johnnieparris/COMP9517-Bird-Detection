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

Runs made with --limit or --train-limit are rejected by default. They are scored
against only the images they actually saw, but a 60-image row cannot be ranked
against a 291-image one; pass --allow-partial to build such a table anyway.

Example:
  python scripts/compare_m2.py --split val --pred results/m2_val_*.json

The glob above also matches the .meta.json sidecars; they are skipped.
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

    # A run made with --limit only saw the first N images. Scoring it against the
    # whole split would count every image it never saw as missed birds and sink
    # its mAP. Slice the same way run_m2.py did, then check the count agrees.
    limit = meta.get("limit")
    if limit:
        records = records[:limit]
    n_ran = meta["stats"]["n_images"]
    if len(records) != n_ran:
        raise ValueError(f"{pred_path} ran on {n_ran} images but {len(records)} were "
                         f"loaded for scoring; the split or manifest has changed")

    predictions = rows_to_predictions(rows, records)
    cfg = meta["config"]
    f1 = best_f1(records, predictions)
    counts = counting_error(records, predictions, score_thr=f1["score_threshold"])

    # Rebuilt through M2Config so the label matches cfg.variant exactly, and two
    # runs differing only in a non-headline setting cannot collide.
    variant = M2Config(**cfg).variant
    return {"variant": variant,
            "split": meta["split"],
            "n_images": n_ran,
            "train_limit": meta.get("train_limit") or "",
            "feature": cfg["feature"], "clf": cfg["clf"],
            "hn_rounds": cfg["hn_rounds"], "proposals": cfg["proposals"],
            "patch": cfg["patch"], "C": cfg["C"], "n_estimators": cfg["n_estimators"],
            "m1_config": cfg["m1_config"] or "",
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
    parser.add_argument("--allow-partial", action="store_true",
                        help="include runs made with --limit/--train-limit. Off by "
                             "default: they are scored correctly on their own subset "
                             "but are not comparable with full-split rows")
    args = parser.parse_args()

    # `--pred results/m2_val_*.json` also matches the .meta.json sidecars, which
    # are inputs to compare_row, not prediction files. Drop them rather than
    # failing on a shell glob that looks perfectly reasonable.
    preds = [p for p in args.pred if not p.endswith(".meta.json")]
    if not preds:
        raise SystemExit("no prediction files given (only .meta.json sidecars matched)")

    manifest = load_split_manifest(MANIFEST)
    records = load_split_from_manifest(args.data_root, manifest, args.split)
    table = sorted((compare_row(p, records, args.split) for p in preds),
                   key=lambda r: -r["mAP50"])

    # Mixing a 60-image run with a 291-image one makes the table meaningless even
    # though each row is correct on its own, so say so rather than quietly ranking
    # them together.
    full = max(row["n_images"] for row in table)
    partial = [r for r in table if r["n_images"] != full or r["train_limit"]]
    if partial and not args.allow_partial:
        listed = ", ".join(f"{r['variant']} ({r['n_images']} images"
                           + (f", train_limit={r['train_limit']}" if r["train_limit"] else "")
                           + ")" for r in partial)
        raise SystemExit(f"refusing to build a mixed table: {len(partial)} partial run(s) "
                         f"among {len(table)}: {listed}\n"
                         f"re-run those variants on the full split, or pass --allow-partial")
    if partial:
        print(f"WARNING: {len(partial)} of {len(table)} rows are partial runs and are "
              f"NOT comparable with the rest; see the n_images column\n")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)

    print(f"{'variant':34s} {'imgs':>5s} {'mAP@0.5':>8s} {'P':>7s} {'R':>7s} {'F1':>7s} "
          f"{'MAE':>7s} {'thr':>7s} {'train s':>8s} {'ms/img':>8s}")
    for row in table:
        print(f"{row['variant']:34s} {row['n_images']:5d} {row['mAP50']:8.4f} "
              f"{row['precision']:7.3f} {row['recall']:7.3f} {row['f1']:7.3f} "
              f"{row['count_mae']:7.2f} {row['score_threshold']:7.3f} "
              f"{row['train_time_s']:8.1f} {row['ms_per_image']:8.1f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

"""Measure how many real birds each proposal source covers.

Method 2 only ever scores candidate boxes, so the fraction of ground-truth boxes
matched by some candidate is a hard ceiling on its recall, and therefore on its
mAP. This mirrors results/m1_proposal_recall.csv (member B) and writes
results/m2_proposal_recall.csv.

Candidate sizes always come from the TRAINING boxes, so no validation statistic
reaches the proposal stage.

Example:
  python scripts/proposal_recall_m2.py --split val
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

import numpy as np

from data.dataset import DEFAULT_ROOT, get_image
from data.split import load_split_from_manifest, load_split_manifest
from src.classical.boxes import iou_matrix
from src.ml.config import load_config
from src.ml.proposals import Proposals, training_box_sizes

MANIFEST = ROOT / "splits" / "official.json"

# Label -> the config that selects that proposal source. Nothing else about
# these configs matters here; only the proposal stage runs.
SOURCES = {
    "window": "configs/m2_concat_gb.yaml",
    "blob": "configs/m2_concat_gb_blobprops.yaml",
    "m1_seg": "configs/m2_concat_gb_m1props.yaml",
}
DENSITY_BINS = [("0", 0, 0), ("1-5", 1, 5), ("6-20", 6, 20),
                ("21-50", 21, 50), ("51+", 51, 10 ** 9)]


def measure(records, proposals, iou_thresholds=(0.5, 0.3)):
    """Recall at each IoU, plus per-density-bin and per-size breakdowns.

    Small/medium/large are the 33rd and 66th percentiles of bird size on this
    split, so the bins stay meaningful whatever the images look like.
    """
    sizes = np.concatenate([np.sqrt((r.boxes[:, 2] - r.boxes[:, 0]) *
                                    (r.boxes[:, 3] - r.boxes[:, 1]))
                            for r in records if len(r.boxes)])
    small_cut, large_cut = np.percentile(sizes, [33, 66])

    hits = {t: [] for t in iou_thresholds}
    per_bin = {name: [] for name, _, _ in DENSITY_BINS}
    per_size = {"small": [], "medium": [], "large": []}
    counts, elapsed = [], 0.0

    for record in records:
        image = get_image(record)
        started = time.perf_counter()
        boxes = proposals(image)
        elapsed += time.perf_counter() - started
        counts.append(len(boxes))
        if not len(record.boxes):
            continue
        best = (iou_matrix(record.boxes, boxes).max(axis=1) if len(boxes)
                else np.zeros(len(record.boxes)))
        for t in iou_thresholds:
            hits[t] += list(best >= t)
        matched = list(best >= iou_thresholds[0])
        for name, lo, hi in DENSITY_BINS:
            if lo <= len(record.boxes) <= hi:
                per_bin[name] += matched
        gt_sizes = np.sqrt((record.boxes[:, 2] - record.boxes[:, 0]) *
                           (record.boxes[:, 3] - record.boxes[:, 1]))
        for hit, size in zip(matched, gt_sizes):
            key = "small" if size < small_cut else "medium" if size < large_cut else "large"
            per_size[key].append(hit)

    row = {"n_gt": len(hits[iou_thresholds[0]]),
           "candidates_per_image": float(np.mean(counts)),
           "ms_per_image": elapsed / len(records) * 1000}
    for t in iou_thresholds:
        row[f"recall{int(t * 100)}"] = float(np.mean(hits[t])) if hits[t] else 0.0
    for name, values in per_bin.items():
        row[f"recall50_{name}"] = float(np.mean(values)) if values else ""
    for name, values in per_size.items():
        row[f"recall50_{name}"] = float(np.mean(values)) if values else ""
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="test is excluded on purpose: this is a tuning diagnostic")
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default=str(ROOT / "results" / "m2_proposal_recall.csv"))
    args = parser.parse_args()

    manifest = load_split_manifest(MANIFEST)
    train = load_split_from_manifest(args.data_root, manifest, "train")
    records = load_split_from_manifest(args.data_root, manifest, args.split)[:args.limit]
    boxes_wh = training_box_sizes(train)

    table = []
    for label, config in SOURCES.items():
        cfg = load_config(ROOT / config)
        row = {"proposals": label, "split": args.split}
        row.update(measure(records, Proposals(cfg, boxes_wh)))
        table.append(row)
        print(f"{label:8s} recall@0.5 {row['recall50']:.3f}  recall@0.3 {row['recall30']:.3f}  "
              f"{row['candidates_per_image']:7.0f} cands/img  {row['ms_per_image']:6.1f} ms/img")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

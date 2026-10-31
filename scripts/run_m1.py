"""Run Method 1 on a split and write predictions in the team format.

Example: python scripts/run_m1.py --config configs/m1_blob.yaml --split val --score
"""
import argparse
import dataclasses
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.classical.interim_data import DEFAULT_ROOT, load_split
from src.classical.pipeline import BlobConfig
from src.classical.quick_eval import ap50, recall_at_iou
from src.classical.run import draw_overlay, run_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--data-root", default=str(ROOT / DEFAULT_ROOT))
    parser.add_argument("--out", help="default: results/m1_<split>.json")
    parser.add_argument("--limit", type=int, help="only use the first N images")
    parser.add_argument("--score", action="store_true", help="print AP@0.5 and recall@0.5")
    parser.add_argument("--viz", type=int, default=0, help="save N GT/prediction overlays")
    parser.add_argument("--final", action="store_true", help="required to touch the test split")
    args = parser.parse_args()

    # Checked before any data is loaded, so a mistake can't touch the test set.
    if args.split == "test" and not args.final:
        parser.error("the test split is for final evaluation only; pass --final to confirm")
    if args.split == "test" and args.score:
        parser.error("--score is not allowed on the test split; use scripts/evaluate.py")
    if args.final and args.limit is not None:
        parser.error("--limit cannot be combined with --final")
    return args


def git_sha():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip()


def save_overlays(records, rows, count, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in random.Random(0).sample(records, min(count, len(records))):
        dets = np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rows
                         if r["image_id"] == record.image_id]).reshape(-1, 4)
        overlay = draw_overlay(cv2.imread(str(record.path)), record.boxes, dets)
        cv2.imwrite(str(out_dir / f"{record.image_id}.jpg"), overlay)


def main():
    args = parse_args()
    cfg = BlobConfig.from_yaml(args.config)
    records = load_split(args.split, root=args.data_root)[:args.limit]

    rows, stats = run_split(records, cfg)

    out = Path(args.out) if args.out else ROOT / "results" / f"m1_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    meta = {"split": args.split, "limit": args.limit, "config": dataclasses.asdict(cfg),
            "stats": stats, "train_time_s": 0, "platform": platform.platform(),
            "machine": platform.machine(), "git_sha": git_sha()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    print(f"{stats['n_images']} images, {stats['n_detections']} detections, "
          f"{stats['ms_per_image']:.1f} ms/img -> {out}")
    if args.score:
        print(f"AP@0.5 {ap50(rows, records):.4f}  recall@0.5 {recall_at_iou(rows, records):.4f}")
    if args.viz:
        save_overlays(records, rows, args.viz, out.parent / f"{out.stem}_viz")


if __name__ == "__main__":
    main()

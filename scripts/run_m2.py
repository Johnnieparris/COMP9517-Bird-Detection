"""Train Method 2 on the training split and run it on another split.

Method 2 is supervised, so unlike Method 1 every run trains first. The model is
always fitted on `train` through splits/official.json, whatever `--split` says,
so validation and test images never reach the classifier.

Example:
  python scripts/run_m2.py --config configs/m2_concat_gb.yaml --split val --score
  python scripts/run_m2.py --config configs/m2_concat_gb.yaml --split val --publish
  python scripts/run_m2.py --config configs/m2_concat_gb.yaml --split test --final --publish
"""
import argparse
import dataclasses
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

# The second entry is what lets the shared team modules import as `data`,
# `evaluation` and `predictions`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

import numpy as np

from data.dataset import DEFAULT_ROOT
from data.split import load_split_from_manifest, load_split_manifest
from evaluation import evaluate, print_results, run_and_time
from predictions import PRED_DIR, save_predictions
from src.ml.config import load_config
from src.ml.detector import BirdDetector

MANIFEST = ROOT / "splits" / "official.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", help="default: results/m2_<split>.json")
    parser.add_argument("--limit", type=int, help="only use the first N evaluation images")
    parser.add_argument("--train-limit", type=int, help="only train on the first N images")
    parser.add_argument("--score", action="store_true",
                        help="print the shared evaluation.py metrics")
    parser.add_argument("--viz", type=int, default=0, help="save N TP/FP/FN overlays")
    parser.add_argument("--viz-thr", type=float, default=None,
                        help="score threshold for the overlays. Defaults to the shared "
                             "SCORE_THR, but report figures should use the same tuned "
                             "threshold as results/m2_comparison.csv, or the pictures "
                             "will not agree with the table")
    parser.add_argument("--final", action="store_true", help="required to touch the test split")
    parser.add_argument("--publish", nargs="?", const=str(PRED_DIR), metavar="DIR",
                        help="also save the shared team CSV, <DIR>/m2_<split>.csv")
    parser.add_argument("--method-name", default="m2",
                        help="name used for the published CSV (e.g. m2_hog_svm)")
    args = parser.parse_args()

    # Guards, checked before any data loads so a mistake cannot touch the test set.
    if args.split == "test" and not args.final:
        parser.error("the test split is for final evaluation only; pass --final to confirm")
    if args.split == "test" and args.score:
        parser.error("--score is not allowed on the test split; use src/evaluation.py")
    if args.final and (args.limit is not None or args.train_limit is not None):
        parser.error("--limit/--train-limit cannot be combined with --final")
    return args


def git_sha():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True)
    return result.stdout.strip()


def save_overlays(records, predictions, count, out_dir, score_thr=None):
    """Draw the densest images with TP green, FP red, FN blue.

    Matching comes from the shared evaluator, so the pictures agree with the
    scores in the table. Images are flipped RGB -> BGR for cv2.imwrite.
    """
    import cv2

    from data.dataset import get_image
    from evaluation import SCORE_THR, classify_detections

    threshold = SCORE_THR if score_thr is None else score_thr
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = sorted(records, key=lambda r: -len(r.boxes))[:count]
    for record in chosen:
        groups = classify_detections(predictions.get(record.image_id, np.zeros((0, 5))),
                                     record.boxes, score_thr=threshold)
        image = get_image(record)[:, :, ::-1].copy()
        for name, colour in (("tp", (0, 200, 0)), ("fp", (0, 0, 255)), ("fn", (255, 100, 0))):
            for box in groups[name]:
                cv2.rectangle(image, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), colour, 2)
        caption = (f"{record.image_id}  GT={len(record.boxes)}  TP={len(groups['tp'])}  "
                   f"FP={len(groups['fp'])}  FN={len(groups['fn'])}  thr={threshold:.3f}")
        cv2.putText(image, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(image, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"{record.image_id}.jpg"), image)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    manifest = load_split_manifest(MANIFEST)

    train_records = load_split_from_manifest(args.data_root, manifest, "train")
    if args.train_limit:
        train_records = train_records[:args.train_limit]
    eval_records = load_split_from_manifest(args.data_root, manifest, args.split)
    if args.limit:
        eval_records = eval_records[:args.limit]

    print(f"config {Path(args.config).name}  variant {cfg.variant}")
    print(f"training on {len(train_records)} images, evaluating {len(eval_records)} "
          f"({args.split})")

    detector = BirdDetector(cfg).fit(train_records)
    print(f"train time: {detector.train_seconds:.1f}s")

    predictions, timing = run_and_time(eval_records, detector.predict_fn())
    n_detections = int(sum(len(p) for p in predictions.values()))
    print(f"inference: {timing['ms_per_image']:.1f} ms/img, {n_detections} detections")

    # Flat list of detections, plus a sidecar recording the exact config, timings
    # and commit. compare_m2.py rebuilds the comparison table from those two.
    out = Path(args.out) if args.out else ROOT / "results" / f"m2_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"image_id": image_id, "x1": float(d[0]), "y1": float(d[1]),
             "x2": float(d[2]), "y2": float(d[3]), "score": float(d[4])}
            for image_id in sorted(predictions) for d in predictions[image_id]]
    out.write_text(json.dumps(rows))
    meta = {"split": args.split, "limit": args.limit, "config": dataclasses.asdict(cfg),
            "config_path": args.config,
            "stats": {"n_images": len(eval_records), "n_detections": n_detections,
                      "ms_per_image": timing["ms_per_image"]},
            "train_time_s": detector.train_seconds, "platform": platform.platform(),
            "machine": platform.machine(), "git_sha": git_sha(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {out}")

    if args.publish:
        path = save_predictions(args.method_name, args.split, eval_records, predictions,
                                folder=args.publish)
        print(f"published -> {path}")
    if args.score:
        print()
        print_results(evaluate(eval_records, predictions, timing), f"Method 2 ({cfg.variant})")
    if args.viz:
        save_overlays(eval_records, predictions, args.viz, out.parent / f"{out.stem}_viz",
                      score_thr=args.viz_thr)
        print(f"overlays -> {out.parent / (out.stem + '_viz')}")


if __name__ == "__main__":
    main()

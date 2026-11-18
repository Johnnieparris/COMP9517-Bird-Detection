"""Run a selected learned detector like run_m1.py, and publish shared predictions.

python scripts/run_ml.py --config configs/yolo_best.yaml --split val --score --publish
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

import cv2
import numpy as np

from src.dl.inference import load_detector
from src.dl.train_yolo import resolve_path, write_json, utc_now
from data.dataset import DEFAULT_ROOT, get_image
from data.split import load_split_manifest, load_split_from_manifest
from evaluation import evaluate, print_results, run_and_time, IOU_THR, SCORE_THR
from predictions import PRED_DIR, MAX_PER_IMAGE, save_predictions
from src.classical.run import draw_overlay


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", help="prediction JSON; defaults to results/evaluation/<method>_<split>.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--viz", type=int, default=0)
    parser.add_argument("--final", action="store_true", help="required to touch the test split")
    parser.add_argument("--publish", nargs="?", const=str(PRED_DIR), metavar="DIR")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final:
        parser.error("the test split is for final evaluation only; pass --final to confirm")
    if args.split == "test" and args.score:
        parser.error("--score is not allowed on test; publish and run python src/evaluation.py")
    if args.final and args.limit is not None:
        parser.error("--limit cannot be combined with --final")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.viz < 0:
        parser.error("--viz must be nonnegative")
    if args.limit is not None and args.publish:
        parser.error("--publish requires the complete split; omit --limit")
    return args


def main(argv=None):
    args = parse_args(argv)  # guard test data before loading either data or weights
    manifest = load_split_manifest(ROOT / "splits/official.json")
    records = load_split_from_manifest(resolve_path(args.data_root), manifest, args.split)
    records = records[:args.limit]
    predict, model_info = load_detector(args.config, args.device)
    predictions, timing = run_and_time(records, predict)
    method = model_info["method"]
    rows = [{"image_id": image_id, "x1": float(x1), "y1": float(y1), "x2": float(x2),
             "y2": float(y2), "score": float(score)}
            for image_id, dets in predictions.items() for x1, y1, x2, y2, score in dets]
    suffix = f"_preview_{args.limit}" if args.limit is not None else ""
    out = resolve_path(args.out) if args.out else ROOT / "results/evaluation" / f"{method}_{args.split}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, rows)
    np.savez_compressed(out.with_suffix(".npz"), **predictions)
    meta = {"method": method, "split": args.split, "limit": args.limit, "model": model_info,
        "timestamp_utc": utc_now(), "platform": platform.platform(),
        "versions": {key: version(key) for key in ("torch", "torchvision", "numpy")},
        "stats": {"n_images": len(records), "n_detections": len(rows), **timing},
        "evaluation": {"iou_threshold": IOU_THR, "score_threshold": SCORE_THR,
            "prediction_confidence": .001, "nms_iou": .7, "max_detections": MAX_PER_IMAGE,
            "timing": "batch=1, 3 warmups, excludes image/model loading"}}
    write_json(out.with_suffix(".meta.json"), meta)
    if args.score:
        scores = evaluate(records, predictions, timing)
        write_json(out.with_suffix(".scores.json"), scores)
        print_results(scores, f"{method} ({args.split})")
    if args.publish:
        published = save_predictions(method, args.split, records, predictions, folder=resolve_path(args.publish))
        print(f"Published -> {published}")
    if args.viz:
        folder = out.parent / f"{out.stem}_viz"
        folder.mkdir(exist_ok=True)
        for record in random.Random(0).sample(records, min(args.viz, len(records))):
            dets = predictions[record.image_id]
            dets = dets[dets[:, 4] >= SCORE_THR]
            overlay = draw_overlay(get_image(record)[:, :, ::-1].copy(), record.boxes, dets)
            cv2.imwrite(str(folder / f"{record.image_id}.jpg"), overlay)
    print(f"{len(records)} images, {len(rows)} detections -> {out}")
    return out


if __name__ == "__main__":
    main()

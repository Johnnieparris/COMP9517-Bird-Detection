"""Failure categories and a success/failure gallery for the four methods (Member E, Study C).

    python scripts/failure_analysis.py

Uses the official clean TEST predictions in predictions/ and each method's best-F1
threshold from clean VAL. Writes results/failure_categories_test.csv and copies 2 best +
2 worst overlays per method (scripts/make_overlays.py, green TP / red FP / blue FN) into
docs/report/figures/gallery/.
"""
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.robustness.failures import CATEGORIES, categorise  # noqa: E402
from src.robustness.score import clean_val_thresholds  # noqa: E402
from evaluation import load_records  # noqa: E402
from predictions import load_predictions  # noqa: E402

METHODS = ("m1", "m2", "yolo", "faster_rcnn")
OUT_CSV = REPO / "results/failure_categories_test.csv"
GALLERY = REPO / "docs/report/figures/gallery"


def main():
    records = load_records(None, "test")
    thresholds = clean_val_thresholds(METHODS)
    rows = []
    for method in METHODS:
        preds = load_predictions(method, "test")
        totals = dict.fromkeys(CATEGORIES, 0)
        negatives, neg_fp = 0, 0
        for r in records:
            counts = categorise(preds.get(r.image_id, np.zeros((0, 5))), r.boxes, thresholds[method])
            for k in CATEGORIES:
                totals[k] += counts[k]
            if len(r.boxes) == 0:
                negatives += 1
                neg_fp += counts["fp_background"]
        rows.append({"method": method, "split": "test", "thr": thresholds[method], **totals,
                     "n_negative_images": negatives,
                     "fp_per_negative_image": neg_fp / negatives if negatives else ""})
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {OUT_CSV.relative_to(REPO)}")
    for row in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    GALLERY.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        for pick in ("best", "worst"):
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run([sys.executable, "scripts/make_overlays.py", method, "--split", "test",
                                "--pick", pick, "--n", "2", "--score-thr", str(thresholds[method]),
                                "--max-width", "512", "--format", "jpg", "--out", tmp],
                               cwd=REPO, check=True, capture_output=True)
                for i, img in enumerate(sorted(Path(tmp).rglob("*.jpg"))[:2], 1):
                    shutil.copy2(img, GALLERY / f"{method}_{pick}_{i}_{img.stem}.jpg")
    print(f"gallery -> {GALLERY.relative_to(REPO)} ({len(list(GALLERY.glob('*.jpg')))} images)")


if __name__ == "__main__":
    main()

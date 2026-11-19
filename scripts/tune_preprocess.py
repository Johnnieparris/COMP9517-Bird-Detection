"""Choose ONE global preprocessing setting for the classical methods (Member E, Study A mitigation).

    python scripts/tune_preprocess.py

Runs Method 1 (configs/m1_seg_colour_tuned.yaml) in-process on the VAL split of the clean
dataset and of all 15 distorted roots, once per setting in PREPROCESS_GRID, and picks the
setting with the highest MEAN val mAP@0.5 over those 16 variants. No per-distortion choice
(that would assume the distortion type is known at test time). Never reads the test split.

Writes results/robustness_preprocess_tuning_val.csv and results/robustness_preprocess_choice.json.
"""
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.classical.pipeline import detect, load_config  # noqa: E402
from src.data.dataset import DEFAULT_ROOT, get_image  # noqa: E402
from src.robustness.distortions import DISTORTIONS, variant_name  # noqa: E402
from src.robustness.preprocess import PREPROCESS_GRID, preprocess  # noqa: E402
from src.robustness.roots import DISTORTED_DIR  # noqa: E402
from evaluation import IOU_THR, average_precision, load_records  # noqa: E402

M1_CONFIG = REPO / "configs/m1_seg_colour_tuned.yaml"
SPLIT = "val"  # tuning split; the test split is never touched here


def variants():
    return ["clean"] + [variant_name(n, s) for n in DISTORTIONS for s in (1, 2, 3)]


def evaluate(job):
    setting_name, variant = job
    root = DEFAULT_ROOT if variant == "clean" else DISTORTED_DIR / variant
    records = load_records(root, SPLIT)
    cfg = load_config(M1_CONFIG)
    preds = {}
    for r in records:
        image = preprocess(get_image(r), PREPROCESS_GRID[setting_name])
        preds[r.image_id] = detect(np.ascontiguousarray(image[:, :, ::-1]), cfg)
    return {"setting": setting_name, "variant": variant, "split": SPLIT,
            "mAP50": average_precision(records, preds, IOU_THR)}


def main():
    jobs = [(s, v) for s in PREPROCESS_GRID for v in variants()]
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(evaluate, jobs))
    out = REPO / "results/robustness_preprocess_tuning_val.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["setting", "variant", "split", "mAP50"])
        writer.writeheader()
        writer.writerows(rows)
    means = {s: float(np.mean([r["mAP50"] for r in rows if r["setting"] == s])) for s in PREPROCESS_GRID}
    clean = {s: next(r["mAP50"] for r in rows if r["setting"] == s and r["variant"] == "clean")
             for s in PREPROCESS_GRID}
    best = max(means, key=means.get)
    choice = {"chosen": best, "setting": PREPROCESS_GRID[best], "selection": "max mean val mAP50 "
              "over clean + 15 distorted variants, Method 1", "mean_val_mAP50": means,
              "clean_val_mAP50": clean}
    (REPO / "results/robustness_preprocess_choice.json").write_text(json.dumps(choice, indent=2))
    for s in PREPROCESS_GRID:
        print(f"{s:18s} mean {means[s]:.4f}  clean {clean[s]:.4f}")
    print("chosen:", best)


if __name__ == "__main__":
    main()

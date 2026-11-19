"""Run methods on the distorted test sets and build results/robustness_test.csv (Member E, Study A).

    python scripts/run_robustness.py --methods m1 m2 yolo faster_rcnn
    python scripts/run_robustness.py --methods m1pre m2pre          # mitigation (Task 7)
    python scripts/run_robustness.py --score-only                   # just rebuild the CSV

Every method runs through its EXISTING runner (run_m1.py / run_m2.py / run_ml.py) with
--split test --final, so the usual test-set guard applies, once per method and variant.
Predictions go to predictions/robustness/<variant>/<method>_test.csv; a method/variant whose
CSV already exists is skipped. The clean predictions of the base methods are the official
ones in predictions/ (copied, not re-run). Methods outside the base four also get a clean-val
run (predictions/robustness/clean_val/) so their confidence threshold comes from clean val.
"""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.data.dataset import DEFAULT_ROOT  # noqa: E402
from src.robustness.distortions import DISTORTIONS, variant_name  # noqa: E402
from src.robustness.roots import DISTORTED_DIR  # noqa: E402
from src.robustness.score import COLUMNS, score_variant  # noqa: E402
from evaluation import IOU_THR, SCORE_THR, best_f1_threshold, load_records  # noqa: E402
from predictions import PRED_DIR, load_predictions  # noqa: E402

PY = sys.executable
OUT_DIR = PRED_DIR / "robustness"
RESULTS = REPO / "results" / "robustness_test.csv"
PREPROCESSED_DIR = REPO / "data" / "preprocessed"
BASE_METHODS = ("m1", "m2", "yolo", "faster_rcnn")

# method -> (runner command without split/root/publish flags, family of data roots)
RUNNERS = {
    "m1": (["scripts/run_m1.py", "--config", "configs/m1_seg_colour_tuned.yaml"], "distorted"),
    "m2": (["scripts/run_m2.py", "--config", "configs/m2_concat_gb_m1props_boxreg.yaml",
            "--method-name", "m2"], "distorted"),
    "yolo": (["scripts/run_ml.py", "--config", "configs/yolo_best.yaml", "--device", "cpu"], "distorted"),
    "faster_rcnn": (["scripts/run_ml.py", "--config", "configs/faster_rcnn_best.yaml", "--device", "cpu"],
                    "distorted"),
    # Task 7: classical methods on denoised + CLAHE images (all splits preprocessed, M2 retrained).
    "m1pre": (["scripts/run_m1.py", "--config", "configs/m1_seg_colour_tuned.yaml"], "preprocessed"),
    "m2pre": (["scripts/run_m2.py", "--config", "configs/m2_concat_gb_m1props_boxreg.yaml",
               "--method-name", "m2"], "preprocessed"),
    # Task 8: Study B YOLO runs (configs written by scripts/eval_e_runs.py).
    "e_de_100": (["scripts/run_ml.py", "--config", "configs/e/infer_de_100.yaml", "--device", "cpu"],
                 "distorted"),
    "e_robust_aug": (["scripts/run_ml.py", "--config", "configs/e/infer_robust_aug.yaml", "--device", "cpu"],
                     "distorted"),
}


def variants():
    return ["clean"] + [variant_name(n, s) for n in DISTORTIONS for s in (1, 2, 3)]


def data_root(method, variant):
    family = RUNNERS[method][1]
    if family == "preprocessed":
        return PREPROCESSED_DIR / variant
    return DEFAULT_ROOT if variant == "clean" else DISTORTED_DIR / variant


def run_method(method, variant, split, out_folder):
    """Run one method's runner on one root; publish <out_folder>/<method>_<split>.csv."""
    target = out_folder / f"{method}_{split}.csv"
    if target.exists():
        return f"exists: {target.relative_to(REPO)}"
    out_folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [PY, *RUNNERS[method][0], "--split", split, "--data-root", str(data_root(method, variant)),
               "--publish", tmp, "--out", str(Path(tmp) / f"raw_{split}.json")]
        if split == "test":
            cmd.append("--final")
        done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"{method} on {variant}/{split} failed:\n{done.stderr[-2000:]}")
        published = [p for p in Path(tmp).glob(f"*_{split}.csv")]
        if len(published) != 1:
            raise RuntimeError(f"expected one published CSV, found {published}")
        shutil.move(published[0], target)
    return f"wrote {target.relative_to(REPO)}"


def threshold(method):
    """Best-F1 confidence threshold on clean val (base methods: the official val predictions)."""
    folder = PRED_DIR if method in BASE_METHODS else OUT_DIR / "clean_val"
    records = load_records(None, "val")
    best = best_f1_threshold(records, load_predictions(method, "val", folder), IOU_THR)
    return best["thr"] if best else SCORE_THR


def write_results():
    rows = []
    for method in RUNNERS:
        for variant in variants():
            folder = OUT_DIR / variant
            if not (folder / f"{method}_test.csv").exists():
                continue
            try:
                thr = threshold(method)
            except FileNotFoundError:
                print(f"no clean-val predictions for {method}; skipped")
                break
            rows.append(score_variant(method, variant, "test", data_root(method, variant), folder, thr))
    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {RESULTS.relative_to(REPO)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="*", default=[], choices=sorted(RUNNERS))
    parser.add_argument("--variants", nargs="*", help="default: clean + all 15")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.score_only:
        for method in args.methods:
            if method not in BASE_METHODS:
                print(method, "clean val:", run_method(method, "clean", "val", OUT_DIR / "clean_val"), flush=True)
            for variant in args.variants or variants():
                if variant == "clean" and method in BASE_METHODS:
                    src = PRED_DIR / f"{method}_test.csv"
                    (OUT_DIR / "clean").mkdir(parents=True, exist_ok=True)
                    if not src.exists():
                        raise FileNotFoundError(f"{src} missing: publish the clean test predictions first")
                    shutil.copy2(src, OUT_DIR / "clean" / src.name)
                    continue
                print(method, variant, run_method(method, variant, "test", OUT_DIR / variant), flush=True)
    write_results()


if __name__ == "__main__":
    main()

"""Evaluate Member E's Study B YOLO runs on the test set, once each (Member E, Study B).

    python scripts/eval_e_runs.py                 # every completed run
    python scripts/eval_e_runs.py --robustness    # also run e_de_100 / e_robust_aug on the distortion suite

For each completed run in results/e_runs/e_<run>/: writes configs/e/infer_<run>.yaml pointing at
checkpoints/best_shared.pt (chosen on clean val during training), publishes its test predictions
through scripts/run_ml.py --final into predictions/data_efficiency/<run>/, and scores them with
the run's clean-val best-F1 threshold. Unfinished runs are listed with their status and no
scores, never estimated. Output: results/data_efficiency_test.csv.
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.data.dataset import DEFAULT_ROOT  # noqa: E402
from src.data_efficiency.score import run_summary  # noqa: E402
from src.robustness.score import score_variant  # noqa: E402
from evaluation import IOU_THR, SCORE_THR, best_f1_threshold, load_records  # noqa: E402
from predictions import PRED_DIR, load_predictions  # noqa: E402

RUNS = ("de_100", "de_50", "de_25", "de_10", "ln_jitter20", "ln_drop20", "de_10_strong", "robust_aug")
RUN_DIR = REPO / "results/e_runs"
OUT = PRED_DIR / "data_efficiency"
RESULTS = REPO / "results/data_efficiency_test.csv"
COLUMNS = ["run", "status", "train_images", "epochs", "best_epoch", "val_mAP50", "train_time_s",
           "thr", "mAP50", "mAP50_95", "precision", "recall", "count_mae"]


def evaluate_run(run):
    run_dir = RUN_DIR / f"e_{run}"
    if not (run_dir / "run.json").exists():
        return {"run": run, "status": "not started"}
    row = run_summary(run_dir)
    if row["status"] != "completed":
        return row
    infer = REPO / f"configs/e/infer_{run}.yaml"
    infer.write_text(f"# Inference config for Study B run {run}; written by scripts/eval_e_runs.py\n" +
                     yaml.safe_dump({"method": "yolo", "imgsz": 640, "checkpoint":
                                     str((run_dir / "checkpoints/best_shared.pt").relative_to(REPO))}))
    folder = OUT / run
    folder.mkdir(parents=True, exist_ok=True)
    # The clean-val predictions of the selected checkpoint, saved by the trainer.
    shutil.copy2(run_dir / "best_shared_predictions.csv", folder / "yolo_val.csv")
    if not (folder / "yolo_test.csv").exists():
        subprocess.run([sys.executable, "scripts/run_ml.py", "--config", str(infer.relative_to(REPO)),
                        "--split", "test", "--final", "--device", "cpu", "--publish", str(folder),
                        "--out", str(REPO / f"results/evaluation/e_{run}_test.json")],
                       cwd=REPO, check=True, capture_output=True)
    best = best_f1_threshold(load_records(None, "val"), load_predictions("yolo", "val", folder), IOU_THR)
    thr = best["thr"] if best else SCORE_THR
    scores = score_variant("yolo", "clean", "test", DEFAULT_ROOT, folder, thr)
    return {**row, **{k: scores[k] for k in ("thr", "mAP50", "mAP50_95", "precision", "recall", "count_mae")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robustness", action="store_true")
    args = parser.parse_args()
    rows = []
    for run in RUNS:
        row = evaluate_run(run)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {RESULTS.relative_to(REPO)}")
    if args.robustness:
        ready = [f"e_{r}" for r in ("de_100", "robust_aug")
                 if next(x for x in rows if x["run"] == r).get("status") == "completed"]
        if ready:
            subprocess.run([sys.executable, "scripts/run_robustness.py", "--methods", *ready], cwd=REPO, check=True)


if __name__ == "__main__":
    main()

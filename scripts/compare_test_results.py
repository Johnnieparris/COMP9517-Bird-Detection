"""Rebuild ML comparison tables from completed test evaluations, without inference.

Run: python scripts/compare_test_results.py
Validation scores remain in runs/ and timelines/. M1 artifacts are not modified.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "yolo": ("yolo", "yolo11n_tuned_50"),
    "faster_rcnn": ("faster_rcnn", "faster_rcnn_baseline_20"),
    "faster_rcnn_tuned": ("faster_rcnn", "faster_rcnn_tuned"),
}
METRICS = ("mAP50", "mAP50_95", "precision", "recall", "f1", "count_mae", "count_bias")


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rebuild(root=ROOT):
    root = Path(root)
    results = root / "results"
    manifest = root / "splits/official.json"
    test_count = len(read_json(manifest)["splits"]["test"])
    manifest_hash = digest(manifest)
    rows = []
    for published_method, (family, run) in RUNS.items():
        score = read_json(results / "evaluation" / f"{published_method}_test.scores.json")
        summary = read_json(results / "runs" / f"{run}.json")
        if (score["split"] != "test" or score["method"] != published_method
                or score["source_run"] != run or score["epoch"] != summary["best_epoch"]
                or score["n_images"] != test_count
                or score["split_manifest_sha256"] != manifest_hash):
            raise ValueError(f"Test evaluation does not match split/run/checkpoint epoch: {run}")
        for path_key, hash_key in (("prediction_file", "prediction_sha256"),
                                   ("checkpoint", "checkpoint_sha256")):
            if digest(root / score[path_key]) != score[hash_key]:
                raise ValueError(f"Changed {path_key}; repeat test evaluation for {run}")
        # Selection is fixed from validation, never from test performance.
        selected = read_json(root / "models" / family / "metadata.json")["source_run"] == run
        rows.append({"method": family, "run": run, "split": "test", "selected": selected,
            "epochs_evaluated": summary["epochs_evaluated"], "best_epoch": score["epoch"],
            "n_images": score["n_images"], "n_birds": score["n_birds"],
            **{key: score[key] for key in METRICS},
            **{key: score[key] for key in ("score_threshold", "iou_threshold", "ms_per_image",
                "device", "platform", "checkpoint", "checkpoint_sha256", "prediction_file",
                "prediction_sha256", "split_manifest_sha256", "timestamp_utc")}})
    rows.sort(key=lambda row: (row["method"], row["run"]))
    selected_rows = [row for row in rows if row["selected"]]
    if len(selected_rows) != 2 or len({row["n_birds"] for row in rows}) != 1:
        raise ValueError("Inconsistent selected models or test annotations")
    # Validate every source before replacing either comparison table.
    write_csv(results / "ml_comparison.csv", rows)
    write_csv(results / "model_comparison.csv", selected_rows)
    print(f"Saved {len(rows)} ML runs and {len(selected_rows)} selected models, all split=test.")
    print("M1 omitted: no published test evaluation. M1 results were not modified.")
    return rows


if __name__ == "__main__":
    rebuild()

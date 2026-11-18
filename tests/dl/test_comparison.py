import csv
import json

import pytest

from scripts import compare_test_results as comparison


@pytest.fixture
def reports(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    manifest = tmp_path / "splits/official.json"
    write(manifest, {"splits": {"test": ["bird"]}})
    for published, (family, run) in comparison.RUNS.items():
        weights = tmp_path / f"{run}.pt"
        weights.write_bytes(b"checkpoint")
        predictions = tmp_path / f"{published}.csv"
        predictions.write_text("prediction")
        score = {"split": "test", "method": published, "source_run": run,
            "epoch": 2, "n_images": 1, "n_birds": 1,
            "split_manifest_sha256": comparison.digest(manifest),
            "checkpoint": weights.name, "checkpoint_sha256": comparison.digest(weights),
            "prediction_file": predictions.name, "prediction_sha256": comparison.digest(predictions),
            "score_threshold": .5, "iou_threshold": .5, "ms_per_image": 1,
            "device": "cpu", "platform": "test", "timestamp_utc": "test",
            **{key: .5 for key in comparison.METRICS}}
        write(tmp_path / f"results/evaluation/{published}_test.scores.json", score)
        write(tmp_path / f"results/runs/{run}.json", {"best_epoch": 2, "epochs_evaluated": 3})
        if published != "faster_rcnn_tuned":
            write(tmp_path / f"models/{family}/metadata.json", {"source_run": run})
    (tmp_path / "results/m1_comparison.csv").write_text("original M1")
    return tmp_path


def test_tables_use_only_test_scores_and_preserve_m1(reports):
    comparison.rebuild(reports)
    with (reports / "results/ml_comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row["split"] for row in rows} == {"test"}
    with (reports / "results/model_comparison.csv").open() as stream:
        selected = list(csv.DictReader(stream))
    assert len(selected) == 2
    assert all(row["run"] != "faster_rcnn_tuned" for row in selected)
    assert (reports / "results/m1_comparison.csv").read_text() == "original M1"


@pytest.mark.parametrize("field,value", [("split", "val"), ("prediction_sha256", "changed")])
def test_invalid_sources_cannot_overwrite_comparisons(reports, field, value):
    path = reports / "results/evaluation/faster_rcnn_tuned_test.scores.json"
    score = json.loads(path.read_text())
    score[field] = value
    path.write_text(json.dumps(score))
    output = reports / "results/ml_comparison.csv"
    output.write_text("previous comparison")
    with pytest.raises(ValueError):
        comparison.rebuild(reports)
    assert output.read_text() == "previous comparison"

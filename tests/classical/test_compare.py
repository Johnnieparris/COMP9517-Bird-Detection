import json
import sys

import numpy as np
import pytest

sys.path.insert(0, "scripts")
from compare_m1 import compare_row

from src.classical.interim_data import ImageRecord


def _write(tmp_path, split="val"):
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps([{"image_id": "a", "x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.9}]))
    pred.with_suffix(".meta.json").write_text(json.dumps({
        "split": split, "train_time_s": 0, "machine": "arm64", "git_sha": "abc",
        "stats": {"ms_per_image": 12.5, "n_images_capped": 1}}))
    return pred, [ImageRecord("a", None, np.array([[0., 0, 10, 10]]))]


def test_compare_row_reads_metrics_and_sidecar(tmp_path):
    pred, records = _write(tmp_path)
    row = compare_row(pred, records, "seg")
    assert row["ap50"] == 1.0 and row["ms_per_image"] == 12.5 and row["train_time_s"] == 0
    assert row["f1"] == 1.0 and row["count_mae"] == 0 and row["n_images_capped"] == 1


def test_compare_row_needs_sidecar_for_this_split(tmp_path):
    pred, records = _write(tmp_path, split="train")
    with pytest.raises(ValueError):
        compare_row(pred, records, "seg", split="val")


def test_compare_row_recall_at_best_f1_goes_with_precision(tmp_path):
    pred, records = _write(tmp_path)
    rows = json.loads(pred.read_text()) + [
        {"image_id": "a", "x1": 50, "y1": 50, "x2": 60, "y2": 60, "score": 0.5},
        {"image_id": "a", "x1": 20, "y1": 20, "x2": 30, "y2": 30, "score": 0.1}]
    pred.write_text(json.dumps(rows))
    records = [ImageRecord("a", None, np.array([[0., 0, 10, 10], [20, 20, 30, 30], [70, 70, 80, 80]]))]
    row = compare_row(pred, records, "seg")
    # best F1 is at the last detection: 2 TP of 3 rows, 3 GT
    assert row["recall_at_best_f1"] == pytest.approx(2 / 3) and row["precision"] == pytest.approx(2 / 3)
    records = [ImageRecord("a", None, np.array([[0., 0, 10, 10]]))]
    row = compare_row(pred, records, "seg")
    assert row["recall_at_best_f1"] == 1.0 and row["recall50"] == 1.0

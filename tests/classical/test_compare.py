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

"""Tests for the table builder and the sweep's resume logic.

These cover the two ways a partial run can poison results/m2_comparison.csv: a
limited run scored against the whole split, and a limited run reused by a later
full sweep.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_m2 import compare_row
from sweep_m2 import is_finished


class FakeRecord:
    def __init__(self, image_id, boxes):
        self.image_id = image_id
        self.boxes = np.asarray(boxes, np.float32).reshape(-1, 4)


def records(n):
    """n images, one bird each, at a box a prediction can match exactly."""
    return [FakeRecord(f"img_{i:03d}", [[10, 10, 30, 30]]) for i in range(n)]


def write_run(tmp_path, name, n_images, limit=None, train_limit=None, split="val"):
    """A prediction file plus sidecar: a perfect detection on each image run."""
    path = tmp_path / f"{name}.json"
    rows = [{"image_id": f"img_{i:03d}", "x1": 10.0, "y1": 10.0,
             "x2": 30.0, "y2": 30.0, "score": 0.9} for i in range(n_images)]
    path.write_text(json.dumps(rows))
    meta = {"split": split, "limit": limit, "train_limit": train_limit,
            "config": {"feature": "hog", "clf": "svm", "hn_rounds": 1,
                       "proposals": "window", "patch": 32, "C": 0.01,
                       "n_estimators": 200, "m1_config": None},
            "config_path": "configs/m2_hog_svm.yaml",
            "stats": {"n_images": n_images, "n_detections": n_images,
                      "ms_per_image": 1.0},
            "train_time_s": 1.0, "machine": "test", "git_sha": "deadbeef"}
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return path


def test_limited_run_is_scored_only_on_the_images_it_saw(tmp_path):
    """The bug: a 10-image run scored against 100 images loses 90 birds.

    Every image the run never touched would count as a missed bird, so a perfect
    detector would report a far lower mAP than it earned.
    """
    path = write_run(tmp_path, "limited", n_images=10, limit=10)
    row = compare_row(path, records(100), split="val")
    assert row["n_images"] == 10
    assert row["mAP50"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)


def test_full_run_is_scored_on_the_whole_split(tmp_path):
    path = write_run(tmp_path, "full", n_images=100)
    row = compare_row(path, records(100), split="val")
    assert row["n_images"] == 100 and row["mAP50"] == pytest.approx(1.0)


def test_mismatched_image_count_is_rejected(tmp_path):
    """A sidecar claiming more images than the split has means stale results."""
    path = write_run(tmp_path, "stale", n_images=50)
    with pytest.raises(ValueError, match="ran on 50 images"):
        compare_row(path, records(20), split="val")


def test_wrong_split_is_rejected(tmp_path):
    path = write_run(tmp_path, "testrun", n_images=10, split="test")
    with pytest.raises(ValueError, match="not val"):
        compare_row(path, records(10), split="val")


def test_row_records_the_settings_that_the_variant_name_hashes(tmp_path):
    """The table must show the knobs, not just a hash, so rows can be read."""
    path = write_run(tmp_path, "full", n_images=10)
    row = compare_row(path, records(10), split="val")
    for column in ("patch", "C", "n_estimators", "m1_config", "n_images"):
        assert column in row


def test_sidecars_are_not_mistaken_for_prediction_files(tmp_path):
    """`--pred results/m2_val_*.json` also globs the .meta.json sidecars."""
    write_run(tmp_path, "v", n_images=10)
    matched = sorted(str(p) for p in tmp_path.glob("v*.json"))
    assert len(matched) == 2, "the glob should pick up the sidecar too"
    kept = [p for p in matched if not p.endswith(".meta.json")]
    assert len(kept) == 1


def test_sweep_reuses_a_matching_full_run(tmp_path):
    path = write_run(tmp_path, "v", n_images=100)
    assert is_finished(path, "val", None, None)


def test_sweep_does_not_reuse_a_limited_run_for_a_full_sweep(tmp_path):
    """The bug: --quick --limit 60 results silently reused by a later full sweep."""
    path = write_run(tmp_path, "v", n_images=60, limit=60)
    assert not is_finished(path, "val", None, None)


def test_sweep_does_not_reuse_a_run_with_a_different_train_limit(tmp_path):
    path = write_run(tmp_path, "v", n_images=100, train_limit=120)
    assert not is_finished(path, "val", None, None)


def test_sweep_reuses_a_limited_run_when_the_same_limit_is_asked_for(tmp_path):
    path = write_run(tmp_path, "v", n_images=60, limit=60)
    assert is_finished(path, "val", 60, None)


def test_sweep_ignores_a_run_from_another_split(tmp_path):
    path = write_run(tmp_path, "v", n_images=10, split="test")
    assert not is_finished(path, "val", None, None)


def test_sweep_treats_a_half_written_run_as_unfinished(tmp_path):
    path = write_run(tmp_path, "v", n_images=10)
    path.with_suffix(".meta.json").unlink()
    assert not is_finished(path, "val", None, None)


def test_sweep_treats_a_corrupt_sidecar_as_unfinished(tmp_path):
    path = write_run(tmp_path, "v", n_images=10)
    path.with_suffix(".meta.json").write_text("{not json")
    assert not is_finished(path, "val", None, None)

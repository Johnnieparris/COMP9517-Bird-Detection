"""Tests for scoring one method on one (distorted) variant."""
import json

import numpy as np

from data.dataset import load_split
from predictions import save_predictions
from src.robustness.score import clean_val_thresholds, parse_variant, score_variant


def _manifest(root, tmp_path):
    ids = {s: [r.image_id for r in load_split(root, s)] for s in ("train", "val", "test")}
    path = tmp_path / "official.json"
    path.write_text(json.dumps({"splits": ids}))
    return path


def _perfect(records):
    return {r.image_id: np.hstack([r.boxes, np.ones((len(r.boxes), 1))]) for r in records}


def test_score_variant_perfect_predictions(fake_root, tmp_path):
    manifest = _manifest(fake_root, tmp_path)
    recs = load_split(fake_root, "test")
    save_predictions("yolo", "test", recs, _perfect(recs), folder=tmp_path / "preds")
    row = score_variant("yolo", "clean", "test", fake_root, tmp_path / "preds", 0.5, manifest_path=manifest)
    assert row["mAP50"] == 1.0 and row["count_mae"] == 0 and row["recall"] == 1.0
    assert (row["method"], row["variant"], row["distortion"], row["severity"]) == ("yolo", "clean", "clean", 0)


def test_parse_variant():
    assert parse_variant("jpeg_s3") == ("jpeg", 3)
    assert parse_variant("clean") == ("clean", 0)


def test_threshold_from_val_not_test(fake_root, tmp_path):
    manifest = _manifest(fake_root, tmp_path)
    recs = load_split(fake_root, "val")
    preds = _perfect(recs)
    preds[recs[0].image_id][:, 4] = 0.8
    save_predictions("m1", "val", recs, preds, folder=tmp_path / "preds")
    thr = clean_val_thresholds(["m1", "absent"], folder=tmp_path / "preds", root=fake_root,
                               manifest_path=manifest)
    assert thr["m1"] == 0.8
    assert thr["absent"] == 0.5  # no val predictions -> shared default

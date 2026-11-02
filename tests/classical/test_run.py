import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from src.classical.interim_data import ImageRecord
from src.classical.pipeline import BlobConfig, SegConfig
from src.classical.run import draw_overlay, run_split

SCRIPT = "scripts/run_m1.py"
CONFIG = "configs/m1_blob.yaml"


def _write_images(root):
    bird = np.full((64, 64, 3), 220, np.uint8)
    cv2.circle(bird, (32, 32), 6, (20, 20, 20), -1)
    cv2.imwrite(str(root / "bird.png"), bird)
    cv2.imwrite(str(root / "empty.png"), np.full((64, 64, 3), 128, np.uint8))
    return [ImageRecord("bird", root / "bird.png", np.array([[26., 26, 38, 38]])),
            ImageRecord("empty", root / "empty.png", np.zeros((0, 4)))]


def test_rows_have_team_schema_and_negative_counts(tmp_path):
    rows, stats = run_split(_write_images(tmp_path), BlobConfig())
    assert set(rows[0]) == {"image_id", "x1", "y1", "x2", "y2", "score"}
    assert all(type(r["x1"]) is float and type(r["score"]) is float for r in rows)
    json.dumps(rows)
    assert stats["n_images"] == 2 and stats["ms_per_image"] > 0
    assert stats["n_detections"] == len(rows)
    assert {r["image_id"] for r in rows} == {"bird"}


def test_unreadable_image_raises_with_path(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(IOError, match="bad.jpg"):
        run_split([ImageRecord("bad", bad, np.zeros((0, 4)))], BlobConfig())


def test_draw_overlay_marks_gt_and_predictions_without_changing_input():
    image = np.zeros((32, 32, 3), np.uint8)
    out = draw_overlay(image, np.array([[2., 2, 10, 10]]), np.array([[15., 15, 25, 25, .9]]))
    assert out[2, 5].tolist() == [255, 0, 0] and out[15, 20].tolist() == [0, 255, 0]
    assert image.sum() == 0


def _cli(*args):
    return subprocess.run([sys.executable, SCRIPT, "--config", CONFIG, *args],
                          capture_output=True, text=True)


def test_test_split_refused_without_final():
    p = _cli("--split", "test")
    assert p.returncode != 0 and "--final" in p.stderr


def test_score_refused_on_test_split():
    p = _cli("--split", "test", "--final", "--score")
    assert p.returncode != 0 and "--score" in p.stderr


def test_limit_refused_with_final():
    p = _cli("--split", "val", "--final", "--limit", "5")
    assert p.returncode != 0 and "--limit" in p.stderr


def test_cli_writes_predictions_and_sidecar(tmp_path):
    for kind in ("images", "labels"):
        (tmp_path / kind / "val").mkdir(parents=True)
    bird = np.full((64, 64, 3), 220, np.uint8)
    cv2.circle(bird, (32, 32), 6, (20, 20, 20), -1)
    cv2.imwrite(str(tmp_path / "images" / "val" / "bird.png"), bird)
    (tmp_path / "labels" / "val" / "bird.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    out = tmp_path / "preds.json"

    p = _cli("--split", "val", "--data-root", str(tmp_path), "--out", str(out),
             "--score", "--viz", "1")

    assert p.returncode == 0, p.stderr
    assert "AP@0.5" in p.stdout and "recall@0.5" in p.stdout
    assert json.loads(out.read_text())[0]["image_id"] == "bird"
    meta = json.loads((tmp_path / "preds.meta.json").read_text())
    assert meta["stats"]["n_images"] == 1 and meta["train_time_s"] == 0
    assert meta["config"]["max_detections"] == 300 and meta["machine"]
    assert list((tmp_path / "preds_viz").glob("*.jpg"))


def test_truncated_jpeg_raises_with_path(tmp_path):
    good = tmp_path / "cut.jpg"
    noise = np.random.default_rng(0).integers(0, 256, (128, 128, 3), np.uint8)
    cv2.imwrite(str(good), noise)
    good.write_bytes(good.read_bytes()[:good.stat().st_size // 3])
    with pytest.raises(IOError, match="cut.jpg"):
        run_split([ImageRecord("cut", good, np.zeros((0, 4)))], BlobConfig())


def test_stats_count_images_that_hit_the_cap(tmp_path):
    _, stats = run_split(_write_images(tmp_path), BlobConfig(max_detections=1))
    assert stats["n_images_capped"] == 1


def test_run_split_accepts_seg_config(tmp_path):
    records = _write_images(tmp_path)
    rows, stats = run_split(records, SegConfig())
    assert stats["n_images"] == 2 and {row["image_id"] for row in rows} <= {"bird"}

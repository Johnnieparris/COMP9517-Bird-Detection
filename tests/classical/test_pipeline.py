import cv2
import numpy as np
import pytest

from src.classical.boxes import iou_matrix
from src.classical.pipeline import BlobConfig, detect


def bgr_discs(h, w, centres_xy, r):
    image = np.full((h, w, 3), 220, np.uint8)
    for centre in centres_xy:
        cv2.circle(image, centre, r, (20, 20, 20), -1)
    return image


def test_dense_grid_keeps_at_least_130():
    centres = [(12 + 24 * i, 12 + 24 * j) for i in range(13) for j in range(12)]  # 156 discs
    img = bgr_discs(300, 320, centres, r=4)
    assert len(detect(img, BlobConfig(sigmas=(2, 3, 4)))) >= 130


def test_max_detections_caps_and_keeps_best():
    noise_image = np.random.default_rng(0).integers(0, 256, (128, 128, 3), np.uint8)
    d = detect(noise_image, BlobConfig(threshold=0.001, max_detections=10))
    assert len(d) == 10 and np.all(np.diff(d[:, 4]) <= 0) and np.isfinite(d).all()


def test_edge_blob_box_stays_in_bounds():
    d = detect(bgr_discs(100, 100, [(2, 50)], r=8), BlobConfig())
    assert len(d) >= 1
    assert (d[:, :2] >= 0).all() and (d[:, 2] <= 100).all() and (d[:, 3] <= 100).all()
    assert (d[:, 2] > d[:, 0]).all() and (d[:, 3] > d[:, 1]).all()


def test_downscaled_detection_maps_to_original_pixels():
    img = bgr_discs(3200, 3200, [(2000, 1000)], r=40)
    cfg = BlobConfig(sigmas=(14, 20, 28, 40), max_side=800, polarity="dark")
    x1, y1, x2, y2, _ = detect(img, cfg)[0]
    assert abs((x1 + x2) / 2 - 2000) <= 8 and abs((y1 + y2) / 2 - 1000) <= 8
    assert (x2 - x1) == pytest.approx(80, rel=.3)


def test_negative_image_gives_empty():
    assert detect(np.full((64, 64, 3), 128, np.uint8), BlobConfig()).shape == (0, 5)


def test_min_area_removes_small_boxes():
    img = bgr_discs(100, 100, [(50, 50)], r=8)
    assert detect(img, BlobConfig(min_area=10_000)).shape == (0, 5)


def test_from_yaml_loads_values(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("threshold: 0.2\nsigmas: [1, 2]\n")
    cfg = BlobConfig.from_yaml(path)
    assert cfg.threshold == 0.2 and tuple(cfg.sigmas) == (1, 2) and cfg.max_detections == 300


def test_from_yaml_rejects_unknown_key(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("treshold: 0.2\n")
    with pytest.raises(TypeError):
        BlobConfig.from_yaml(path)


def test_shipped_config_loads():
    assert BlobConfig.from_yaml("configs/m1_blob.yaml").max_detections >= 300


def test_dense_flock_birds_are_matched_with_shipped_config():
    centres = [(12 + 24 * i, 12 + 24 * j) for i in range(13) for j in range(12)]  # 156 discs
    img = bgr_discs(300, 320, centres, r=7)
    gt = np.array([[x - 7.5, y - 7.5, x + 7.5, y + 7.5] for x, y in centres])
    dets = detect(img, BlobConfig.from_yaml("configs/m1_blob.yaml"))
    assert (iou_matrix(gt, dets).max(axis=1) >= 0.5).sum() >= 130

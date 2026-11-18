"""Detector tests on synthetic images, so they run without the dataset."""
import cv2
import numpy as np
import pytest

from src.ml.config import M2Config
from src.ml.detector import BirdDetector, crop


@pytest.fixture
def toy_split(tmp_path):
    """Dark discs on a light background, written as real jpgs with real labels."""
    from data.dataset import DetectionRecord

    rng = np.random.default_rng(0)
    records = []
    for i in range(12):
        image = np.full((128, 128, 3), 225, np.uint8)
        boxes = []
        for _ in range(3):
            cx, cy = rng.integers(20, 108, 2)
            cv2.circle(image, (int(cx), int(cy)), 8, (25, 35, 60), -1)
            boxes.append([cx - 9, cy - 9, cx + 9, cy + 9])
        path = tmp_path / f"toy_{i:02d}.jpg"
        cv2.imwrite(str(path), image)
        records.append(DetectionRecord(image_id=path.stem, image_path=path,
                                       width=128, height=128,
                                       boxes=np.array(boxes, np.float32)))
    return records


@pytest.fixture
def toy_config():
    return M2Config(proposals="window", feature="hog", clf="rf", n_estimators=20,
                    patch=16, hn_rounds=0, neg_per_image=6, proposal_neg_per_image=10,
                    window_percentiles=(50,), max_detections=300)


def test_detect_before_fit_is_an_error(toy_config):
    with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
        BirdDetector(toy_config).detect(np.zeros((64, 64, 3), np.uint8))


def test_fit_then_detect_returns_the_right_shape(toy_split, toy_config):
    det = BirdDetector(toy_config).fit(toy_split, log=lambda *_: None)
    out = det.detect(np.full((128, 128, 3), 225, np.uint8))
    assert out.ndim == 2 and out.shape[1] == 5
    assert np.isfinite(out).all()


def test_scores_are_probabilities_in_zero_to_one(toy_split, toy_config):
    """Every method publishes scores on the same scale for the shared evaluator."""
    det = BirdDetector(toy_config).fit(toy_split, log=lambda *_: None)
    out = det.detect(cv2.imread(str(toy_split[0].image_path))[:, :, ::-1])
    assert (out[:, 4] >= 0).all() and (out[:, 4] <= 1).all()


def test_detections_come_back_sorted_by_score(toy_split, toy_config):
    det = BirdDetector(toy_config).fit(toy_split, log=lambda *_: None)
    out = det.detect(cv2.imread(str(toy_split[0].image_path))[:, :, ::-1])
    assert np.all(np.diff(out[:, 4]) <= 1e-6)


def test_max_detections_is_respected(toy_split, toy_config):
    cfg = M2Config(**{**toy_config.__dict__, "max_detections": 130})
    det = BirdDetector(cfg).fit(toy_split, log=lambda *_: None)
    assert len(det.detect(np.full((128, 128, 3), 225, np.uint8))) <= 130


def test_training_is_reproducible_for_a_fixed_seed(toy_split, toy_config):
    image = cv2.imread(str(toy_split[0].image_path))[:, :, ::-1]
    first = BirdDetector(toy_config).fit(toy_split, log=lambda *_: None).detect(image)
    second = BirdDetector(toy_config).fit(toy_split, log=lambda *_: None).detect(image)
    assert np.allclose(first, second)


def test_train_time_is_recorded(toy_split, toy_config):
    det = BirdDetector(toy_config).fit(toy_split, log=lambda *_: None)
    assert det.train_seconds is not None and det.train_seconds > 0


def test_hard_negative_mining_adds_negatives(toy_split, toy_config):
    cfg = M2Config(**{**toy_config.__dict__, "hn_rounds": 1, "hn_images": 4, "hn_per_image": 5})
    messages = []
    BirdDetector(cfg).fit(toy_split, log=messages.append)
    assert any("hard negative round 1" in m for m in messages)


def test_fraction_trains_on_fewer_images(toy_split, toy_config):
    """Half the images must yield fewer positive patches, read off the fit log."""
    def positives(messages):
        return int([m for m in messages if "positive and" in m][0].split()[1])

    cfg = M2Config(**{**toy_config.__dict__, "fraction": 0.5})
    half, full = [], []
    BirdDetector(cfg).fit(toy_split, log=half.append)
    BirdDetector(toy_config).fit(toy_split, log=full.append)
    assert positives(half) < positives(full)


def test_preprocess_hook_is_applied(toy_split, toy_config):
    """Member E's robustness study swaps in denoising or CLAHE through this hook."""
    det = BirdDetector(toy_config).fit(toy_split, log=lambda *_: None)
    seen = []
    det.image_preproc = lambda im: (seen.append(im.shape), im)[1]
    det.detect(np.full((128, 128, 3), 225, np.uint8))
    assert seen


def test_crop_rejects_a_degenerate_box():
    image = np.zeros((50, 50, 3), np.uint8)
    assert crop(image, [10, 10, 11, 11], 16) is None
    assert crop(image, [10, 10, 30, 30], 16).shape == (16, 16, 3)


def test_crop_clamps_a_box_that_runs_off_the_image():
    image = np.zeros((50, 50, 3), np.uint8)
    assert crop(image, [-20, -20, 30, 30], 16).shape == (16, 16, 3)

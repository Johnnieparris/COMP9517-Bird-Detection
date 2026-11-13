import cv2
import numpy as np
import pytest

from src.ml.config import M2Config
from src.ml.proposals import Proposals, box_sides, training_box_sizes


class FakeRecord:
    def __init__(self, boxes):
        self.boxes = np.asarray(boxes, np.float32).reshape(-1, 4)


def discs_rgb(h, w, centres, r):
    image = np.full((h, w, 3), 220, np.uint8)
    for centre in centres:
        cv2.circle(image, centre, r, (20, 20, 20), -1)
    return image


SIZES = np.array([[20, 20], [30, 30], [40, 40], [60, 60]], float)


@pytest.mark.parametrize("mode", ["window", "blob"])
def test_boxes_stay_inside_the_image(mode):
    cfg = M2Config(proposals=mode)
    image = discs_rgb(120, 160, [(20, 30), (100, 80)], r=6)
    boxes = Proposals(cfg, SIZES)(image)
    assert len(boxes)
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
    assert (boxes[:, 2] <= 160).all() and (boxes[:, 3] <= 120).all()


@pytest.mark.parametrize("mode", ["window", "blob"])
def test_boxes_have_positive_area(mode):
    boxes = Proposals(M2Config(proposals=mode), SIZES)(discs_rgb(120, 160, [(60, 60)], r=8))
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()


def test_window_grid_covers_the_whole_image():
    """Corners matter: birds sit at image edges too."""
    boxes = Proposals(M2Config(proposals="window"), SIZES)(np.zeros((200, 200, 3), np.uint8))
    assert boxes[:, 0].min() < 10 and boxes[:, 1].min() < 10
    assert boxes[:, 2].max() > 190 and boxes[:, 3].max() > 190


def test_window_count_is_capped_on_a_huge_image():
    cfg = M2Config(proposals="window", window_max_per_scale=500)
    boxes = Proposals(cfg, SIZES)(np.zeros((4000, 4000, 3), np.uint8))
    assert len(boxes) <= 500 * len(cfg.window_percentiles)


def test_window_boxes_are_sized_from_the_training_statistics():
    """Sizes must come from the data, not be hard-coded."""
    small = Proposals(M2Config(proposals="window"), np.full((20, 2), 12.0))
    large = Proposals(M2Config(proposals="window"), np.full((20, 2), 90.0))
    image = np.zeros((300, 300, 3), np.uint8)
    small_side = (small(image)[:, 2] - small(image)[:, 0]).mean()
    large_side = (large(image)[:, 2] - large(image)[:, 0]).mean()
    assert large_side > small_side * 3


def test_blob_finds_an_obvious_disc():
    """A single dark disc on a light field must produce a box near its centre."""
    cfg = M2Config(proposals="blob", blob_threshold=0.01)
    boxes = Proposals(cfg, np.full((20, 2), 24.0))(discs_rgb(160, 160, [(80, 80)], r=10))
    assert len(boxes)
    centres = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
    assert np.abs(centres - 80).max(axis=1).min() < 15


def test_blank_image_is_handled():
    """An empty result is fine on a featureless image; a crash is not."""
    for mode in ("window", "blob"):
        boxes = Proposals(M2Config(proposals=mode), SIZES)(np.full((100, 100, 3), 128, np.uint8))
        assert boxes.ndim == 2 and boxes.shape[1] == 4


def test_box_sides_is_geometric_mean():
    assert box_sides([[4, 9]])[0] == pytest.approx(6.0)


def test_training_box_sizes_skips_negative_images():
    records = [FakeRecord([]), FakeRecord([[0, 0, 10, 20]]), FakeRecord([])]
    assert training_box_sizes(records).tolist() == [[10, 20]]


def test_training_box_sizes_needs_at_least_one_box():
    with pytest.raises(ValueError, match="no ground-truth boxes"):
        training_box_sizes([FakeRecord([]), FakeRecord([])])

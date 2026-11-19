import numpy as np

from src.data_efficiency.label_noise import drop_boxes, jitter_boxes


def test_jitter_clips_to_image_and_min_size():
    rng = np.random.default_rng(0)
    boxes = np.array([[0, 0, 5, 5], [70, 50, 80, 60], [30, 20, 31, 21]], np.float32)
    for _ in range(200):
        out = jitter_boxes(boxes, 0.2, rng, 80, 60)
        assert out.shape == boxes.shape
        assert (out[:, 0] >= 0).all() and (out[:, 1] >= 0).all()
        assert (out[:, 2] <= 80).all() and (out[:, 3] <= 60).all()
        assert (out[:, 2] - out[:, 0] >= 1).all() and (out[:, 3] - out[:, 1] >= 1).all()


def test_jitter_moves_boxes_within_fraction():
    rng = np.random.default_rng(1)
    boxes = np.array([[20, 20, 40, 30]], np.float32)  # w 20, h 10
    out = jitter_boxes(boxes, 0.2, rng, 100, 100)
    assert not np.array_equal(out, boxes)
    assert np.all(np.abs(out - boxes) <= np.array([4, 2, 4, 2]) + 1e-5)


def test_jitter_empty_boxes():
    out = jitter_boxes(np.zeros((0, 4), np.float32), 0.2, np.random.default_rng(0), 80, 60)
    assert out.shape == (0, 4)


def test_drop_fraction():
    boxes = np.arange(40, dtype=np.float32).reshape(10, 4)
    out = drop_boxes(boxes, 0.2, np.random.default_rng(0))
    assert len(out) == 8
    assert all(any(np.array_equal(row, b) for b in boxes) for row in out)


def test_drop_empty_boxes():
    assert drop_boxes(np.zeros((0, 4), np.float32), 0.2, np.random.default_rng(0)).shape == (0, 4)

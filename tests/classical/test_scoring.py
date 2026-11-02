import cv2
import numpy as np

from src.classical.scoring import local_contrast


def lab_of(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def test_bird_box_scores_above_background_box():
    image = np.full((200, 200, 3), 220, np.uint8)
    cv2.circle(image, (60, 60), 15, (20, 20, 20), -1)
    scores = local_contrast(lab_of(image), np.array([[45., 45, 75, 75], [130, 130, 160, 160]]))
    assert scores[0] > 5 * max(scores[1], 1e-6)


def test_local_contrast_finite_at_corner_and_on_flat():
    image = np.full((50, 50, 3), 128, np.uint8)
    boxes = np.array([[0., 0, 10, 10], [40, 40, 50, 50], [20, 20, 30, 30], [49, 49, 50, 50]])
    scores = local_contrast(lab_of(image), boxes)
    assert np.isfinite(scores).all() and (scores == 0).all()


def test_local_contrast_handles_no_boxes():
    assert local_contrast(np.zeros((10, 10, 3), np.uint8), np.zeros((0, 5))).shape == (0,)


def test_box_covering_whole_image_scores_zero():
    image = np.random.default_rng(0).integers(0, 256, (20, 20, 3), np.uint8)
    assert local_contrast(image, np.array([[0., 0, 20, 20]]))[0] == 0

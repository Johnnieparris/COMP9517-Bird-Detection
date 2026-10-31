import numpy as np
import pytest

from src.classical.blob import detect_blobs, suggest_sigmas, to_gray

SIGMAS = np.geomspace(2, 16, 8)


def disc_image(size, centre_xy, r, fg, bg):
    ys, xs = np.mgrid[:size, :size]
    image = np.full((size, size), bg, np.float32)
    image[(xs - centre_xy[0]) ** 2 + (ys - centre_xy[1]) ** 2 <= r ** 2] = fg
    return image


def test_dark_disc_found_with_correct_centre_and_size():
    g = disc_image(100, (40, 60), r=8, fg=0., bg=1.)
    x1, y1, x2, y2, score = detect_blobs(g, SIGMAS, .05, "dark")[0]
    assert abs((x1 + x2) / 2 - 40) <= 2 and abs((y1 + y2) / 2 - 60) <= 2
    assert (x2 - x1) == pytest.approx(16, rel=.3)
    assert score > 0


def test_polarity_selects_contrast():
    g = disc_image(100, (50, 50), r=8, fg=1., bg=0.)
    assert len(detect_blobs(g, SIGMAS, .05, "bright")) >= 1
    assert detect_blobs(g, SIGMAS, .5, "dark").shape == (0, 5)
    assert len(detect_blobs(g, SIGMAS, .05, "both")) >= 1


def test_unknown_polarity_rejected():
    with pytest.raises(ValueError):
        detect_blobs(np.zeros((8, 8), np.float32), SIGMAS, .05, "light")


def test_uniform_image_has_no_detections():
    g = np.full((64, 64), .5, np.float32)
    assert detect_blobs(g, SIGMAS, .05, "both").shape == (0, 5)


def test_stronger_contrast_scores_higher():
    strong = detect_blobs(disc_image(100, (50, 50), 8, 0., 1.), SIGMAS, .05, "dark")[0, 4]
    weak = detect_blobs(disc_image(100, (50, 50), 8, .7, 1.), SIGMAS, .05, "dark")[0, 4]
    assert strong > weak


def test_results_sorted_by_score():
    g = disc_image(100, (25, 25), 8, 0., 1.)
    g[disc_image(100, (75, 75), 8, 0., 1.) == 0] = .6
    scores = detect_blobs(g, SIGMAS, .05, "dark")[:, 4]
    assert len(scores) >= 2 and np.all(np.diff(scores) <= 0)


def test_to_gray_range_and_shape_with_clahe():
    img = np.random.default_rng(0).integers(0, 256, (20, 30, 3), np.uint8)
    g = to_gray(img, clahe=True)
    assert g.shape == (20, 30) and g.dtype == np.float32 and 0 <= g.min() <= g.max() <= 1


def test_suggest_sigmas_from_box_sizes():
    side = 2 * np.sqrt(2) * 4
    assert suggest_sigmas(np.array([[0, 0, side, side]] * 5)) == pytest.approx([4.] * 6)


def test_suggest_sigmas_span_percentiles_in_increasing_order():
    sides = np.linspace(4, 60, 100)
    boxes = np.stack([0 * sides, 0 * sides, sides, sides], axis=1)
    sigmas = suggest_sigmas(boxes, n=4)
    assert len(sigmas) == 4 and all(a < b for a, b in zip(sigmas, sigmas[1:]))


def test_suggest_sigmas_empty_raises():
    with pytest.raises(ValueError):
        suggest_sigmas(np.zeros((0, 4)))


def test_box_centre_uses_continuous_coordinates():
    g = np.ones((64, 64), np.float32)
    g[10:27, 10:27] = 0  # odd-sized block with edges at 10 and 27, so its centre is 18.5
    x1, y1, x2, y2, _ = detect_blobs(g, [6.0], .05, "dark")[0]
    assert (x1 + x2) / 2 == pytest.approx(18.5) and (y1 + y2) / 2 == pytest.approx(18.5)

import numpy as np
import pytest

from src.classical.boxes import iou_matrix
from src.ml.boxreg import (BoxRegressor, apply_deltas, regression_targets,
                           to_centre_form, to_corner_form)


def test_centre_and_corner_forms_round_trip():
    boxes = np.array([[10, 20, 30, 50], [0, 0, 8, 8]], float)
    assert np.allclose(to_corner_form(to_centre_form(boxes)), boxes)


def test_targets_are_zero_when_the_proposal_is_already_right():
    boxes = np.array([[10, 20, 30, 50]], float)
    assert np.allclose(regression_targets(boxes, boxes), 0.0)


def test_applying_the_true_targets_lands_on_the_ground_truth():
    proposals = np.array([[10, 10, 30, 30], [50, 50, 70, 90]], float)
    truths = np.array([[12, 14, 34, 36], [48, 46, 74, 96]], float)
    assert np.allclose(apply_deltas(proposals, regression_targets(proposals, truths)),
                       truths, atol=1e-6)


def test_targets_are_scale_invariant():
    """The same relative error must give the same target at any box size."""
    small_p, small_g = np.array([[0, 0, 10, 10]], float), np.array([[1, 1, 11, 11]], float)
    big_p, big_g = np.array([[0, 0, 100, 100]], float), np.array([[10, 10, 110, 110]], float)
    assert np.allclose(regression_targets(small_p, small_g),
                       regression_targets(big_p, big_g))


def test_zero_deltas_leave_boxes_untouched():
    boxes = np.array([[10, 20, 30, 50]], float)
    assert np.allclose(apply_deltas(boxes, np.zeros((1, 4))), boxes)


def test_extreme_scale_targets_are_clipped():
    """A tiny proposal against a huge bird must not dominate the least squares."""
    targets = regression_targets(np.array([[0, 0, 1, 1]], float),
                                 np.array([[0, 0, 500, 500]], float))
    assert np.isfinite(targets).all() and np.abs(targets[:, 2:]).max() <= 1.0


def test_unfitted_regressor_returns_the_originals():
    boxes = np.array([[10, 10, 30, 30]], np.float32)
    assert np.allclose(BoxRegressor().refine(np.zeros((1, 8)), boxes), boxes)


def test_too_few_pairs_does_not_fit():
    reg = BoxRegressor().fit(np.random.rand(4, 8), np.ones((4, 4)), np.ones((4, 4)))
    assert reg.model is None


def test_regressor_improves_overlap_on_a_learnable_pattern():
    """Proposals are systematically 60% too small and the feature says by how much.

    A linear model should recover most of that, so mean IoU with the ground
    truth must improve substantially.
    """
    rng = np.random.default_rng(0)
    n = 300
    cx, cy = rng.uniform(60, 180, n), rng.uniform(60, 180, n)
    size = rng.uniform(20, 60, n)
    truths = np.stack([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2], 1)
    shrink = 0.6
    proposals = np.stack([cx - size * shrink / 2, cy - size * shrink / 2,
                          cx + size * shrink / 2, cy + size * shrink / 2], 1)
    features = np.stack([size, np.ones(n)], 1)

    before = np.diag(iou_matrix(proposals, truths)).mean()
    reg = BoxRegressor(alpha=1.0).fit(features, proposals, truths)
    after = np.diag(iou_matrix(reg.refine(features, proposals), truths)).mean()
    assert after > before + 0.2, f"IoU {before:.3f} -> {after:.3f}"


def test_refine_never_returns_nan():
    rng = np.random.default_rng(1)
    proposals = rng.uniform(0, 100, (20, 4))
    proposals[:, 2:] += proposals[:, :2] + 5
    reg = BoxRegressor(alpha=1.0).fit(rng.random((20, 6)), proposals, proposals)
    assert np.isfinite(reg.refine(rng.random((20, 6)), proposals)).all()

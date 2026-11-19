"""Tests for automatic failure categorisation (Member E, Study C)."""
import numpy as np

from src.robustness.failures import CATEGORIES, categorise


def _pred(box, score=0.9):
    return np.array([[*box, score]], float)


def test_isolated_and_dense_misses():
    lone = [[0, 0, 10, 10]]
    cluster = [[100 + 6 * i, 100, 110 + 6 * i, 110] for i in range(5)]  # centres 6 px apart, radius 2*14 px
    counts = categorise(np.zeros((0, 5)), np.array(lone + cluster, float), thr=0.5)
    assert counts["fn_isolated"] == 1 and counts["fn_dense"] == 5 and counts["tp"] == 0


def test_duplicate_fp():
    gt = np.array([[0, 0, 10, 10]], float)
    preds = np.vstack([_pred([0, 0, 10, 10], 0.9), _pred([0, 0, 10, 9], 0.8)])
    counts = categorise(preds, gt, thr=0.5)
    assert counts["tp"] == 1 and counts["fp_duplicate"] == 1 and counts["fn_isolated"] == 0


def test_localisation_fp():
    gt = np.array([[0, 0, 10, 10]], float)
    counts = categorise(_pred([5, 0, 15, 10]), gt, thr=0.5)   # IoU = 50/150 = 0.33
    assert counts["fp_localisation"] == 1 and counts["fn_isolated"] == 1 and counts["tp"] == 0


def test_background_fp_on_negative():
    counts = categorise(_pred([0, 0, 10, 10]), np.zeros((0, 4)), thr=0.5)
    assert counts["fp_background"] == 1 and sum(counts.values()) == 1


def test_low_score_predictions_ignored():
    counts = categorise(_pred([0, 0, 10, 10], 0.2), np.zeros((0, 4)), thr=0.5)
    assert sum(counts.values()) == 0


def test_empty_everything():
    counts = categorise(np.zeros((0, 5)), np.zeros((0, 4)), thr=0.5)
    assert set(counts) == set(CATEGORIES) and sum(counts.values()) == 0

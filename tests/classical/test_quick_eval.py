import numpy as np
import pytest

from src.classical.interim_data import ImageRecord
from src.classical.quick_eval import ap50, recall_at_iou

GT1 = [0, 0, 10, 10]
GT2 = [20, 20, 30, 30]
MISS = [50, 50, 60, 60]


def _rec(image_id, boxes):
    return ImageRecord(image_id, None, np.array(boxes, float).reshape(-1, 4))


def _row(image_id, box, score):
    x1, y1, x2, y2 = box
    return {"image_id": image_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score}


def test_perfect_prediction_is_one():
    assert ap50([_row("a", GT1, .9)], [_rec("a", [GT1])]) == pytest.approx(1.0)


def test_tp_fp_tp_ranking():
    rows = [_row("a", GT1, .9), _row("a", MISS, .8), _row("a", GT2, .7)]
    assert ap50(rows, [_rec("a", [GT1, GT2])]) == pytest.approx(0.8333, abs=1e-3)


def test_duplicate_on_same_gt_is_fp():
    rows = [_row("a", GT1, .9), _row("a", GT1, .8), _row("a", GT2, .7)]
    assert ap50(rows, [_rec("a", [GT1, GT2])]) == pytest.approx(0.8333, abs=1e-3)


def test_no_predictions_is_zero():
    assert ap50([], [_rec("a", [GT1])]) == 0.0


def test_fp_on_negative_image_lowers_ap():
    rows = [_row("neg", GT1, .95), _row("a", GT1, .9)]
    assert ap50(rows, [_rec("a", [GT1]), _rec("neg", [])]) == pytest.approx(0.5)


def test_prediction_only_matches_gt_in_its_own_image():
    rows = [_row("b", GT1, .9)]
    assert ap50(rows, [_rec("a", [GT1]), _rec("b", [])]) == 0.0


def test_unknown_image_id_raises():
    with pytest.raises(ValueError):
        ap50([_row("zzz", GT1, .9)], [_rec("a", [GT1])])


def test_split_without_gt_raises():
    with pytest.raises(ValueError):
        ap50([], [_rec("a", [])])


def test_recall_counts_gt_covered():
    assert recall_at_iou([_row("a", GT1, .9)], [_rec("a", [GT1, GT2])]) == pytest.approx(.5)

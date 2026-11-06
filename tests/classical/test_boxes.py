import numpy as np
import pytest

from src.classical.boxes import clip_boxes, filter_boxes, iou_matrix, nms


def test_iou_half_overlap():
    a = np.array([[0, 0, 10, 10.]])
    b = np.array([[5, 0, 15, 10.]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1 / 3]])


def test_iou_disjoint_is_zero():
    a = np.array([[0, 0, 1, 1.]])
    b = np.array([[5, 5, 6, 6.]])
    assert iou_matrix(a, b)[0, 0] == 0


def test_nms_removes_lower_score_overlap_and_sorts():
    d = np.array([[0, 0, 10, 10, .5], [1, 0, 11, 10, .9], [50, 50, 60, 60, .7]], float)
    np.testing.assert_allclose(nms(d, .3)[:, 4], [.9, .7])


def test_nms_empty_keeps_shape():
    assert nms(np.zeros((0, 5)), .3).shape == (0, 5)


def test_clip_drops_box_fully_outside():
    d = np.array([[-5, -5, 5, 5, .9], [200, 200, 210, 210, .8]], float)
    np.testing.assert_allclose(clip_boxes(d, 100, 100), [[0, 0, 5, 5, .9]])


def test_filter_by_area():
    d = np.array([[0, 0, 2, 2, .9], [0, 0, 10, 10, .8], [0, 0, 50, 50, .7]], float)
    np.testing.assert_allclose(filter_boxes(d, 16, 400)[:, 4], [.8])


def test_filter_no_max_area_keeps_large():
    d = np.array([[0, 0, 50, 50, .7]], float)
    assert len(filter_boxes(d, 16, None)) == 1

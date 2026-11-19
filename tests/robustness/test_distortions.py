"""Tests for the deterministic distortions used by the robustness study."""
import numpy as np
import pytest

from src.robustness.distortions import DISTORTIONS, apply_distortion, rng_for, variant_name


def _image(seed=0, shape=(64, 80, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


BOXES = np.array([[5, 5, 25, 30], [40, 10, 70, 50]], dtype=np.float32)


def test_every_variant_keeps_shape_and_uint8():
    img = _image()
    for name, levels in DISTORTIONS.items():
        for severity in range(1, len(levels) + 1):
            out = apply_distortion(img, BOXES, name, severity, rng_for(0, "a", variant_name(name, severity)))
            assert out.shape == img.shape and out.dtype == np.uint8, (name, severity)


def test_variant_names():
    assert variant_name("noise", 1) == "noise_s1"
    assert len(DISTORTIONS) == 5 and all(len(v) == 3 for v in DISTORTIONS.values())


def test_deterministic_for_same_seed_and_id():
    img = _image()
    a = apply_distortion(img, BOXES, "noise", 2, rng_for(0, "a", "noise_s2"))
    b = apply_distortion(img, BOXES, "noise", 2, rng_for(0, "a", "noise_s2"))
    assert np.array_equal(a, b)


def test_differs_across_image_ids():
    img = _image()
    a = apply_distortion(img, BOXES, "noise", 2, rng_for(0, "a", "noise_s2"))
    b = apply_distortion(img, BOXES, "noise", 2, rng_for(0, "b", "noise_s2"))
    assert not np.array_equal(a, b)


def test_noise_clips_not_wraps():
    img = np.full((50, 50, 3), 255, np.uint8)
    out = apply_distortion(img, BOXES, "noise", 3, rng_for(0, "a", "t"))
    assert out.max() == 255 and out.mean() > 200


def test_lowlight_scales_mean():
    img = np.full((50, 50, 3), 128, np.uint8)
    out = apply_distortion(img, BOXES, "lowlight", 3, rng_for(0, "a", "t"))
    assert abs(out.mean() - 0.2 * 128) <= 2


def test_jpeg_and_blur_change_image():
    img = _image()
    for name in ("jpeg", "blur"):
        assert not np.array_equal(apply_distortion(img, BOXES, name, 3, rng_for(0, "a", "t")), img)


def test_occlusion_covers_requested_fraction():
    img = np.full((100, 100, 3), 255, np.uint8)
    box = np.array([[20, 20, 60, 60]], np.float32)
    out = apply_distortion(img, box, "occlusion", 2, rng_for(0, "a", "t"))
    changed = np.any(out != img, axis=2)
    inside = changed[20:60, 20:60].sum()
    assert abs(inside - 0.5 * 1600) <= 0.1 * 800
    assert changed.sum() == inside


def test_occlusion_without_boxes_is_identity():
    img = _image()
    out = apply_distortion(img, np.zeros((0, 4), np.float32), "occlusion", 3, rng_for(0, "a", "t"))
    assert np.array_equal(out, img)


def test_unknown_name_or_severity_raises():
    with pytest.raises(ValueError):
        apply_distortion(_image(), BOXES, "fog", 1, rng_for(0, "a", "t"))
    with pytest.raises(ValueError):
        apply_distortion(_image(), BOXES, "noise", 4, rng_for(0, "a", "t"))

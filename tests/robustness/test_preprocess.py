"""Tests for the classical-method mitigation: denoising + CLAHE preprocessing."""
import numpy as np

from src.robustness.preprocess import PREPROCESS_GRID, preprocess


def _image():
    return np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)


def test_preprocess_none_is_identity():
    img = _image()
    assert np.array_equal(preprocess(img, PREPROCESS_GRID["none"]), img)


def test_preprocess_keeps_shape_uint8():
    img = _image()
    for name, setting in PREPROCESS_GRID.items():
        out = preprocess(img, setting)
        assert out.shape == img.shape and out.dtype == np.uint8, name


def test_clahe_raises_contrast_on_dark_image():
    rng = np.random.default_rng(1)
    img = np.clip(30 + rng.normal(0, 4, (64, 64, 3)), 0, 255).astype(np.uint8)
    img[16:48, 16:48] += 10
    out = preprocess(img, PREPROCESS_GRID["clahe"])
    assert out.std() > img.std()


def test_denoise_reduces_noise():
    rng = np.random.default_rng(2)
    flat = np.full((64, 64, 3), 120, np.uint8)
    noisy = np.clip(flat + rng.normal(0, 25, flat.shape), 0, 255).astype(np.uint8)
    out = preprocess(noisy, {"nlm_h": 20})
    assert np.abs(out.astype(int) - 120).mean() < np.abs(noisy.astype(int) - 120).mean()

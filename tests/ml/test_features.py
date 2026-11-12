import numpy as np
import pytest

from src.ml.features import KINDS, extract, extract_many, feature_length


def rgb_patch(seed=0, size=40):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


@pytest.mark.parametrize("kind", KINDS)
def test_length_is_fixed_whatever_the_input_size(kind):
    expected = feature_length(kind, 32)
    for size in (12, 40, 97):
        assert len(extract(rgb_patch(size=size), kind, 32)) == expected


@pytest.mark.parametrize("kind", KINDS)
def test_values_are_finite(kind):
    assert np.isfinite(extract(rgb_patch(), kind, 32)).all()


@pytest.mark.parametrize("kind", KINDS)
def test_same_patch_gives_the_same_vector(kind):
    patch = rgb_patch(1)
    assert np.array_equal(extract(patch, kind, 32), extract(patch.copy(), kind, 32))


def test_concat_is_the_three_parts_joined():
    patch = rgb_patch(2)
    parts = sum(feature_length(k, 32) for k in ("hog", "lbp", "colour"))
    assert feature_length("concat", 32) == parts
    assert len(extract(patch, "concat", 32)) == parts


def test_colour_uses_rgb_not_bgr():
    """A red patch and a blue patch must not describe the same way.

    The shared loader (data.dataset.get_image) returns RGB. Reading it as BGR
    would silently swap the hue of every bird, so this guards the channel order.
    """
    red = np.zeros((32, 32, 3), np.uint8)
    red[:, :, 0] = 220
    blue = np.zeros((32, 32, 3), np.uint8)
    blue[:, :, 2] = 220
    assert not np.allclose(extract(red, "colour", 32), extract(blue, "colour", 32))


def test_lbp_survives_a_brightness_shift():
    """Uniform LBP compares neighbours, so adding a constant should barely move it."""
    patch = rgb_patch(3)
    brighter = np.clip(patch.astype(int) + 25, 0, 255).astype(np.uint8)
    before, after = extract(patch, "lbp", 32), extract(brighter, "lbp", 32)
    assert np.abs(before - after).max() < 0.25


def test_flat_patch_does_not_divide_by_zero():
    flat = np.full((32, 32, 3), 128, np.uint8)
    for kind in KINDS:
        assert np.isfinite(extract(flat, kind, 32)).all()


def test_extract_many_shapes():
    patches = [rgb_patch(i) for i in range(4)]
    out = extract_many(patches, "hog", 32)
    assert out.shape == (4, feature_length("hog", 32)) and out.dtype == np.float32


def test_extract_many_handles_an_empty_list():
    assert extract_many([], "hog", 32).shape == (0, feature_length("hog", 32))


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown feature kind"):
        extract(rgb_patch(), "surf", 32)

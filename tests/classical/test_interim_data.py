import numpy as np
import pytest
from PIL import Image

from src.classical.interim_data import load_split


def _make_image(root, split="val", name="a"):
    images = root / split / "images"
    images.mkdir(parents=True)
    (root / split / "labels").mkdir()
    Image.new("RGB", (100, 50)).save(images / f"{name}.jpg")


@pytest.fixture
def tmp_dataset(tmp_path):
    _make_image(tmp_path)
    (tmp_path / "val" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    return tmp_path


def test_box_converted_to_pixel_xyxy(tmp_dataset):
    rec = load_split("val", root=tmp_dataset)[0]
    assert rec.image_id == "a"
    np.testing.assert_allclose(rec.boxes, [[40, 15, 60, 35]])


@pytest.mark.parametrize("label_text", [None, "", "\n"])
def test_image_with_empty_or_missing_label_has_no_boxes(tmp_path, label_text):
    _make_image(tmp_path)
    if label_text is not None:
        (tmp_path / "val" / "labels" / "a.txt").write_text(label_text)
    assert load_split("val", root=tmp_path)[0].boxes.shape == (0, 4)


def test_records_sorted_by_image_id(tmp_path):
    _make_image(tmp_path, name="b")
    Image.new("RGB", (10, 10)).save(tmp_path / "val" / "images" / "a.jpg")
    assert [r.image_id for r in load_split("val", root=tmp_path)] == ["a", "b"]


def test_unknown_split_rejected(tmp_dataset):
    with pytest.raises(ValueError):
        load_split("validation", root=tmp_dataset)


def test_missing_split_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split("val", root=tmp_path)

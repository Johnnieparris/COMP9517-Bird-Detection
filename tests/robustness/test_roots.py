"""Tests for building derived dataset roots (distorted / preprocessed / noisy-label copies)."""
import os

import numpy as np
import pytest

from src.data.dataset import get_image, load_split
from src.robustness import roots
from src.robustness.roots import build_root, link_or_copy


def _invert(image, boxes, image_id):
    return 255 - image


def test_untransformed_split_is_hardlinked(fake_root, tmp_path):
    dst = build_root(fake_root, tmp_path / "out", {"train": None, "val": _invert, "test": None})
    src_img = fake_root / "images/train/train_0.jpg"
    assert os.stat(dst / "images/train/train_0.jpg").st_ino == os.stat(src_img).st_ino
    assert (dst / "dataset.yaml").exists() and (dst / "root_info.json").exists()


def test_transformed_split_written_and_labels_kept(fake_root, tmp_path):
    dst = build_root(fake_root, tmp_path / "out", {"train": None, "val": _invert, "test": None})
    src_rec = {r.image_id: r for r in load_split(fake_root, "val")}
    for r in load_split(dst, "val"):
        assert not np.array_equal(get_image(r), get_image(src_rec[r.image_id]))
    for name in ("val_0.txt", "val_1.txt"):
        assert (dst / "labels/val" / name).read_text() == (fake_root / "labels/val" / name).read_text()


def test_label_fn_rewrites_labels(fake_root, tmp_path):
    dst = build_root(fake_root, tmp_path / "out", {"train": None, "val": None, "test": None},
                     label_fn=lambda boxes, image_id, split: boxes[:0])
    assert all(p.read_text() == "" for p in (dst / "labels/train").iterdir())


def test_label_fn_round_trips_boxes(fake_root, tmp_path):
    dst = build_root(fake_root, tmp_path / "out", {"train": None, "val": None, "test": None},
                     label_fn=lambda boxes, image_id, split: boxes)
    for split in ("train", "val", "test"):
        a = {r.image_id: r.boxes for r in load_split(fake_root, split)}
        for r in load_split(dst, split):
            assert np.allclose(r.boxes, a[r.image_id], atol=0.01)


def test_refuses_existing_output(fake_root, tmp_path):
    build_root(fake_root, tmp_path / "out", {"train": None, "val": None, "test": None})
    with pytest.raises(FileExistsError):
        build_root(fake_root, tmp_path / "out", {"train": None, "val": None, "test": None})


def test_link_falls_back_to_copy(fake_root, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("cross-device link")
    monkeypatch.setattr(roots.os, "link", boom)
    src = fake_root / "images/train/train_0.jpg"
    dst = tmp_path / "copy.jpg"
    link_or_copy(src, dst)
    assert dst.read_bytes() == src.read_bytes()


def test_loader_reads_built_root(fake_root, tmp_path):
    dst = build_root(fake_root, tmp_path / "out", {"train": None, "val": _invert, "test": None})
    a = {r.image_id: r.boxes for r in load_split(fake_root, "val")}
    b = {r.image_id: r.boxes for r in load_split(dst, "val")}
    assert a.keys() == b.keys() and all(np.allclose(a[k], b[k]) for k in a)

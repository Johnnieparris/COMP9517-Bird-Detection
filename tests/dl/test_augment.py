from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
import yaml

from src.data import validation
from src.data.dataset import load_split
from src.dl.augment import AugmentConfig, augment_image, export_dataset, transform_boxes
from src.dl.dataset import BirdDetectionDataset, detection_collate


IDENTITY = AugmentConfig(noise_std=0, horizontal_flip=0, vertical_flip=0,
                         brightness=0, contrast=0, saturation=0, hue=0,
                         scale_min=1, scale_max=1, rotation=0, min_box_size=0)


@pytest.mark.parametrize("horizontal,vertical,expected", [
    (1, 0, [6, 1, 9, 4]), (0, 1, [1, 2, 4, 5]), (1, 1, [6, 2, 9, 5]),
])
def test_flips_match_image_pixels(horizontal, vertical, expected):
    image = np.zeros((6, 10, 3), np.uint8)
    image[1:4, 1:4] = 255
    boxes = np.array([[1, 1, 4, 4]], np.float32)
    cfg = replace(IDENTITY, horizontal_flip=horizontal, vertical_flip=vertical)
    result, transformed = augment_image(image, boxes, np.random.default_rng(1), cfg)
    np.testing.assert_allclose(transformed, [expected])
    if horizontal:
        image = image[:, ::-1]
    if vertical:
        image = image[::-1]
    np.testing.assert_array_equal(result, image)
    np.testing.assert_array_equal(boxes, [[1, 1, 4, 4]])


def test_rotation_and_scale_geometry():
    # 90-degree rotation in a 10x10 canvas, followed by scale 0.5 about its centre.
    rotation = np.array([[0, 1, 0], [-1, 0, 10]], dtype=float)
    box = transform_boxes([[1, 2, 4, 6]], rotation, 10, 10)
    np.testing.assert_allclose(box, [[2, 6, 6, 9]])
    scaling = np.array([[0.5, 0, 2.5], [0, 0.5, 2.5]])
    np.testing.assert_allclose(transform_boxes(box, scaling, 10, 10, min_box_size=0),
                               [[3.5, 5.5, 5.5, 7]])


def test_scale_changes_pixels_and_boxes_together():
    image = np.zeros((20, 20, 3), np.uint8)
    image[6:14, 6:14] = 255
    result, boxes = augment_image(image, [[6, 6, 14, 14]], np.random.default_rng(1),
                                  replace(IDENTITY, scale_min=0.5, scale_max=0.5))
    np.testing.assert_allclose(boxes, [[8, 8, 12, 12]])
    ys, xs = np.where(result[..., 0] == 255)
    assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (8, 8, 12, 12)


def test_clipping_visibility_and_empty_targets():
    matrix = np.array([[1., 0, 0], [0, 1, 0]])
    result = transform_boxes([[-5, 2, 5, 8], [-9, 2, 1, 8], [12, 2, 16, 8]],
                             matrix, 10, 10, min_visibility=0.25)
    np.testing.assert_allclose(result, [[0, 2, 5, 8]])
    assert transform_boxes([], matrix, 10, 10).shape == (0, 4)


def test_noise_colour_reproducibility_and_no_input_mutation():
    image = np.full((32, 32, 3), [160, 100, 60], dtype=np.uint8)
    original = image.copy()
    cfg = replace(IDENTITY, noise_std=8, brightness=.2, contrast=.2, saturation=.2, hue=10)
    a, boxes = augment_image(image, [], np.random.default_rng(42), cfg)
    b, _ = augment_image(image, [], np.random.default_rng(42), cfg)
    c, _ = augment_image(image, [], np.random.default_rng(43), cfg)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(image, original)
    assert not np.array_equal(a, c) and not np.array_equal(a, image)
    assert a.dtype == np.uint8 and boxes.shape == (0, 4)


@pytest.mark.parametrize("settings", [dict(scale_min=0), dict(rotation=-1),
    dict(noise_std=-1), dict(horizontal_flip=2), dict(hue=float("nan"))])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        AugmentConfig(**settings)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    manifest = {"train": ["train_bird", "train_empty"], "val": ["val_bird"], "test": ["test_bird"]}
    for split, ids in manifest.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for name in ids:
            image = np.zeros((32, 48, 3), np.uint8)
            image[8:24, 12:36] = [220, 180, 80]
            Image.fromarray(image).save(root / "images" / split / f"{name}.png")
            label = "" if name.endswith("empty") else "0 0.5 0.5 0.5 0.5\n"
            (root / "labels" / split / f"{name}.txt").write_text(label)
    path = tmp_path / "official.json"
    path.write_text(json.dumps({"splits": manifest}))
    return root, path


def test_export_is_shared_reproducible_and_preserves_holdouts(source, tmp_path):
    root, manifest = source
    first, second = tmp_path / "first", tmp_path / "second"
    meta = export_dataset(first, copies=2, root=root, manifest_path=manifest)
    export_dataset(second, copies=2, root=root, manifest_path=manifest)
    assert meta["counts"] == {"train": 6, "val": 1, "test": 1}
    for path in (first / "images").rglob("*.png"):
        assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    for path in (first / "labels").rglob("*.txt"):
        assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    for split in ("val", "test"):
        for kind in ("images", "labels"):
            for path in (root / kind / split).iterdir():
                assert path.read_bytes() == (first / kind / split / path.name).read_bytes()
    assert manifest.read_text() == json.dumps({"splits": {
        "train": ["train_bird", "train_empty"], "val": ["val_bird"], "test": ["test_bird"]}})
    for check in (validation.check_missing_images, validation.check_missing_labels,
                  validation.check_bbox_ranges, validation.check_duplicate_ids):
        assert not check(first)
    assert yaml.safe_load((first / "data.yaml").read_text())["names"] == {0: "bird"}
    dataset = BirdDetectionDataset(first)
    assert len(dataset) == 6
    records = load_split(first, "train")
    for index, record in enumerate(records):
        image, target = dataset[index]
        assert image.shape == (3, 32, 48) and image.dtype == torch.float32
        assert 0 <= image.min() <= image.max() <= 1
        np.testing.assert_allclose(target["boxes"].numpy(), record.boxes)
        assert target["labels"].dtype == torch.int64
        assert target["labels"].tolist() == [1] * len(record.boxes)
        assert target["boxes"].shape == (len(record.boxes), 4)
    images, targets = next(iter(DataLoader(dataset, batch_size=6, collate_fn=detection_collate)))
    assert len(images) == len(targets) == 6
    assert any(t["boxes"].shape == (0, 4) for t in targets)
    with pytest.raises(FileExistsError):
        export_dataset(first, root=root, manifest_path=manifest)


def test_rejects_leakage_before_writing(source, tmp_path):
    root, manifest = source
    data = json.loads(manifest.read_text())
    data["splits"]["val"].append("train_bird")
    manifest.write_text(json.dumps(data))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="overlapping"):
        export_dataset(output, root=root, manifest_path=manifest)
    assert not output.exists()


def test_rejects_output_within_source(source):
    root, manifest = source
    with pytest.raises(ValueError, match="separate"):
        export_dataset(root / "augmented", root=root, manifest_path=manifest)


def test_faster_rcnn_accepts_bird_and_empty_targets(source, tmp_path):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    root, manifest = source
    output = tmp_path / "faster_rcnn"
    export_dataset(output, copies=1, config=IDENTITY, root=root, manifest_path=manifest)
    dataset = BirdDetectionDataset(output)
    # One image with a bird and one with an empty label file.
    images, targets = detection_collate([dataset[0], dataset[2]])
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=2, min_size=64, max_size=64,
        rpn_pre_nms_top_n_train=32, rpn_post_nms_top_n_train=16, box_batch_size_per_image=16)
    model.train()
    losses = model(images, targets)
    assert all(torch.isfinite(loss).item() for loss in losses.values())
    sum(losses.values()).backward()
    assert torch.isfinite(model.roi_heads.box_predictor.cls_score.weight.grad).all()

import json

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from scripts import run_ml, organize_results
from src.dl import inference


@pytest.mark.parametrize("args", [
    ["--split", "test"],
    ["--split", "test", "--final", "--score"],
    ["--split", "test", "--final", "--limit", "2"],
    ["--split", "val", "--limit", "2", "--publish"],
    ["--split", "val", "--limit", "0"],
])
def test_cli_guards_before_data_or_weights(args):
    with pytest.raises(SystemExit):
        run_ml.parse_args(["--config", "not_loaded.yaml", *args])


def test_cli_publishes_complete_split_and_shared_scores(tmp_path, monkeypatch):
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (tmp_path / kind / split).mkdir(parents=True)
    for name, label in (("bird", "0 .5 .5 .5 .5"), ("empty", "")):
        Image.new("RGB", (32, 32), (100, 100, 100)).save(tmp_path / "images/val" / f"{name}.png")
        (tmp_path / "labels/val" / f"{name}.txt").write_text(label)
    monkeypatch.setattr(run_ml, "load_split_manifest", lambda path: {"val": ["bird", "empty"]})
    calls = 0
    def predict(image):
        nonlocal calls
        calls += 1
        return np.array([[8., 8., 24., 24., .9]]) if calls <= 4 else np.zeros((0, 5))
    monkeypatch.setattr(run_ml, "load_detector", lambda *a: (predict, {"method": "yolo"}))
    output = tmp_path / "out/val.json"
    run_ml.main(["--config", "unused", "--split", "val", "--data-root", str(tmp_path),
        "--score", "--viz", "1", "--publish", str(tmp_path / "published"), "--out", str(output)])
    assert json.loads(output.with_suffix(".scores.json").read_text())["mAP50"] == 1
    assert json.loads(output.with_suffix(".meta.json").read_text())["stats"]["n_images"] == 2
    assert (tmp_path / "published/yolo_val.csv").is_file()
    assert len(list((output.parent / "val_viz").glob("*.jpg"))) == 1
    with np.load(output.with_suffix(".npz")) as saved:
        assert saved["empty"].shape == (0, 5)


def test_model_selection_uses_scores_not_tuned_name_and_skips_missing_weights(tmp_path):
    weights = tmp_path / "best.pt"
    weights.touch()
    baseline = {"name": "baseline", "method": "faster_rcnn", "weights": weights, "best": {"scores": {"mAP50": .84}}}
    tuned = {**baseline, "name": "tuned", "best": {"scores": {"mAP50": .83}}}
    missing = {**baseline, "name": "missing", "weights": tmp_path / "absent.pt", "best": {"scores": {"mAP50": .99}}}
    assert organize_results.choose_available([tuned, baseline, missing])["faster_rcnn"]["name"] == "baseline"


def test_inference_missing_checkpoint_fails_before_download(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"method": "yolo", "checkpoint": str(tmp_path / "missing.pt"), "imgsz": 640}))
    with pytest.raises(FileNotFoundError, match="Transfer"):
        inference.load_detector(config, "cpu")


def test_portable_faster_checkpoint_loads_without_pretraining(tmp_path, monkeypatch):
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
        def forward(self, images):
            assert self.weight.item() == 3
            return [{"boxes": torch.tensor([[1., 2., 3., 4.]]), "scores": torch.tensor([.8]), "labels": torch.tensor([1])}]
    def build(config, initialize):
        assert initialize is False
        return Detector()
    monkeypatch.setattr(inference, "build_model", build)
    weights = tmp_path / "best.pt"
    torch.save({"format": "faster_rcnn_inference_v1", "model": {"weight": torch.tensor(3.)}, "config": {}}, weights)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"method": "faster_rcnn", "checkpoint": str(weights), "imgsz": 640}))
    predict, metadata = inference.load_detector(config, "cpu")
    np.testing.assert_allclose(predict(np.zeros((8, 8, 3), dtype=np.uint8)), [[1, 2, 3, 4, .8]])
    assert metadata["checkpoint_sha256"] == inference.file_sha256(weights)


def test_dataset_report_exposes_missing_augmented_training_ids():
    run = {"metadata": {"dataset": {"train_images": 1369,
        "augmentation": {"counts": {"train": 2738}}}}}
    assert "1369" in organize_results.dataset_summary(run)["warning"]
    assert "2738" in organize_results.dataset_summary(run)["warning"]

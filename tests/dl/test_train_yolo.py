import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.dl import train_yolo as training


def test_prediction_adapter_converts_rgb_and_preserves_shared_contract():
    calls = []

    class Model:
        def predict(self, image, **kwargs):
            calls.append((image, kwargs))
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=torch.tensor([[1., 2, 3, 4]]), conf=torch.tensor([.75])))]

    image = np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
    result = training.predict_function(Model(), "cpu", 640)(image)
    np.testing.assert_allclose(result, [[1, 2, 3, 4, .75]])
    assert calls[0][0][0, 0].tolist() == [30, 20, 10]
    assert calls[0][1]["conf"] == .001
    assert calls[0][1]["max_det"] == training.MAX_PER_IMAGE


def test_checkpoints_scores_hyperparameters_and_best_are_retained(tmp_path, monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda path: object())
    monkeypatch.setattr(training, "predict_function", lambda *args: None)
    predictions = {"image": np.array([[1., 1, 4, 4, .8]])}
    monkeypatch.setattr(training, "run_and_time", lambda *args: (predictions, {}))
    scores = iter([.4, .3, .6])
    monkeypatch.setattr(training, "evaluate", lambda *args: {"mAP50": next(scores)})
    monkeypatch.setattr(training, "print_results", lambda *args: None)
    saved = []
    def save_predictions(*args, **kwargs):
        saved.append((args, kwargs))
        if "folder" in kwargs:
            (kwargs["folder"] / "yolo_val.csv").write_text("predictions\n")
    monkeypatch.setattr(training, "save_predictions", save_predictions)
    last = tmp_path / "last.pt"
    trainer = SimpleNamespace(epoch=0, last=last, args=SimpleNamespace(imgsz=640, lr0=.0003),
        device="cpu", lr={"lr/pg0": .0002}, optimizer=SimpleNamespace(param_groups=[{"lr": .0002}]),
        label_loss_items=lambda loss: {"train/box_loss": .5}, tloss=None, metrics={"fitness": .7})
    recorder = training.MilestoneRecorder(tmp_path, [], {"evaluation": {"split": "val"}, "dataset": {}})
    for epoch in range(3):
        trainer.epoch = epoch
        last.write_bytes(f"checkpoint {epoch}".encode())
        recorder(trainer)
    assert (tmp_path / "checkpoints/best_shared.pt").read_bytes() == b"checkpoint 2"
    assert json.loads((tmp_path / "best_shared.json").read_text())["epoch"] == 3
    assert len(list((tmp_path / "checkpoints").glob("epoch_*.pt"))) == 3
    history = json.loads((tmp_path / "history.json").read_text())
    assert [row["scores"]["mAP50"] for row in history] == [.4, .3, .6]
    with (tmp_path / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
    assert json.loads(rows[1]["hyperparameters_json"])["lr0"] == .0003
    assert json.loads(rows[1]["learning_rates_json"])["lr/pg0"] == .0002
    assert len(saved) == 5  # three milestone exports, two best-model publications
    with pytest.raises(ValueError, match="Duplicate milestone"):
        recorder(trainer)
    with np.load(tmp_path / "milestones/epoch_0001/predictions.npz") as stored:
        np.testing.assert_array_equal(stored["image"], predictions["image"])


def test_prepare_data_rejects_validation_leakage_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "REPO", tmp_path)
    (tmp_path / "splits").mkdir()
    official = {"train": ["train"], "val": ["val"], "test": ["test"]}
    (tmp_path / "splits/official.json").write_text(json.dumps({"splits": official}))
    root = tmp_path / "data"
    root.mkdir()
    (root / "splits.json").write_text(json.dumps({"splits": {
        "train": ["train", "val"], "val": [], "test": []}}))
    with pytest.raises(ValueError, match="Holdout"):
        training.prepare_data(root, tmp_path, 1, 42)


def test_timeline_atomic_json_does_not_replace_good_history_with_nan(tmp_path):
    path = tmp_path / "scores.json"
    training.write_json(path, {"mAP50": .5})
    with pytest.raises(ValueError):
        training.write_json(path, {"mAP50": float("nan")})
    assert json.loads(path.read_text()) == {"mAP50": .5}

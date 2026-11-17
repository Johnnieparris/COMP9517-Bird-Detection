import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.ml import train_faster_rcnn as training
from src.ml import tune_faster_rcnn as tuning


class TinyDetector(torch.nn.Module):
    """Exercise the real optimizer/checkpoint/evaluation flow without a large backbone."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, images, targets=None):
        if self.training:
            return {"loss_classifier": self.weight.square() * sum(image.mean() for image in images) / len(images)}
        return [{"boxes": torch.tensor([[2., 2., 20., 20.]], device=self.weight.device),
                 "scores": torch.tensor([.9], device=self.weight.device),
                 "labels": torch.tensor([1], device=self.weight.device)} for image in images]


class TinyDataset:
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 3  # exercises the final, incomplete accumulation group

    def __getitem__(self, index):
        return torch.full((3, 32, 32), (index + 1) / 4), {
            "boxes": torch.tensor([[2., 2., 20., 20.]]), "labels": torch.tensor([1])}


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    image = tmp_path / "val.png"
    Image.new("RGB", (32, 32), (128, 0, 0)).save(image)
    record = SimpleNamespace(image_id="val", image_path=image,
        boxes=np.array([[2., 2., 20., 20.]]), width=32, height=32)
    monkeypatch.setattr(training, "prepare_data", lambda *a, **k: (None, [record], {"train_images": 3, "val_images": 1}))
    monkeypatch.setattr(training, "BirdDetectionDataset", TinyDataset)
    monkeypatch.setattr(training, "build_model", lambda *a, **k: TinyDetector())
    save = training.save_predictions
    def save_in_temp(*args, **kwargs):
        kwargs.setdefault("folder", tmp_path / "published")
        return save(*args, **kwargs)
    monkeypatch.setattr(training, "save_predictions", save_in_temp)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"data_root": str(tmp_path), "pretrained": False,
        "epochs": 2, "batch": 1, "accumulation_steps": 2, "device": "cpu", "workers": 0}))
    return tmp_path, path


def test_checkpoints_metrics_and_resume_pending_evaluation(small_run, monkeypatch):
    root, config = small_run
    complete = training.train(config, run_name="continuous", output_root=root)
    original = training.record_epoch
    monkeypatch.setattr(training, "record_epoch", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        training.train(config, run_name="interrupted", output_root=root)
    interrupted = root / "interrupted"
    assert (interrupted / "checkpoints/epoch_0001.pt").exists()
    assert not (interrupted / "history.json").exists()
    monkeypatch.setattr(training, "record_epoch", original)
    training.train(resume=interrupted)
    history = json.loads((interrupted / "history.json").read_text())
    assert [event["epoch"] for event in history] == [1, 2]
    assert all(event["scores"]["mAP50"] == 1 for event in history)
    assert all(event["method"] == training.METHOD for event in history)
    assert history[0]["hyperparameters"]["data_root"] == str(root)
    assert (interrupted / "history.csv").is_file()
    assert (interrupted / "best_shared_predictions.csv").is_file()
    assert (interrupted / "checkpoints/best_shared.pt").read_bytes() == (interrupted / "checkpoints/epoch_0001.pt").read_bytes()
    for folder in (complete, interrupted):
        assert json.loads((folder / "run.json").read_text())["status"] == "completed"
    first = torch.load(complete / "checkpoints/epoch_0002.pt", weights_only=False)
    resumed = torch.load(interrupted / "checkpoints/epoch_0002.pt", weights_only=False)
    torch.testing.assert_close(first["model"]["weight"], resumed["model"]["weight"], rtol=0, atol=0)
    assert resumed["optimizer"]["state"]
    assert (interrupted / "milestones/epoch_0002/predictions.npz").is_file()
    with pytest.raises(ValueError, match="already complete"):
        training.train(resume=interrupted)


def test_accumulation_handles_partial_batches():
    config = training.validate_config({"epochs": 1, "batch": 2, "accumulation_steps": 2,
        "lr0": .1, "warmup_epochs": 0, "grad_clip": 0, "weight_decay": 0})
    loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=2, collate_fn=training.detection_collate)
    model = TinyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    training.train_epoch(model, loader, optimizer, torch.amp.GradScaler("cuda", enabled=False),
        torch.device("cpu"), config, 1)
    # Mean image intensity is .5; d(w^2 * .5)/dw = 1 at w=1.
    assert model.weight.item() == pytest.approx(.9)


def test_grid_exports_fresh_full_budget_and_resumes(small_run):
    root, config = small_run
    search = root / "search.yaml"
    search.write_text(yaml.safe_dump({"trial_epochs": 1, "space": {"lr0": [.001, .003]}}))
    output = tuning.tune(config, search, output_root=root, name="grid")
    results = json.loads((output / "trials.json").read_text())
    assert len(results) == 2
    final = yaml.safe_load((output / "best_training_config.yaml").read_text())
    assert final["epochs"] == 2  # restored final budget, not the one-epoch trial budget
    assert final["lr0"] == .001  # deterministic tie: first grid candidate wins
    assert "resume" not in final and "checkpoint" not in final
    for index in (1, 2):
        saved = json.loads((output / "trials" / f"trial_{index:04d}" / "run.json").read_text())
        assert saved["config"]["seed"] == final["seed"]
        assert saved["config"]["data_root"] == final["data_root"]
    before = (output / "trials/trial_0001/checkpoints/epoch_0001.pt").stat().st_mtime_ns
    tuning.tune(config, search, output_root=root, name="grid", resume=True)
    assert (output / "trials/trial_0001/checkpoints/epoch_0001.pt").stat().st_mtime_ns == before
    assert json.loads((output / "tuning.json").read_text())["status"] == "completed"


def test_prediction_adapter_keeps_rgb_and_only_bird_class():
    class Model:
        def __call__(self, images):
            torch.testing.assert_close(images[0][:, 0, 0], torch.tensor([1., 0., 0.]))
            return [{"boxes": torch.tensor([[1., 2., 3., 4.], [4., 5., 6., 7.]]),
                "scores": torch.tensor([.9, .8]), "labels": torch.tensor([1, 0])}]
    image = np.full((8, 8, 3), [255, 0, 0], dtype=np.uint8)
    result = training.predict_function(Model(), torch.device("cpu"))(image)
    np.testing.assert_allclose(result, [[1, 2, 3, 4, .9]])


@pytest.mark.parametrize("change", [{"lr0": 0}, {"batch": 0}, {"momentum": 1},
    {"amp": "false"}, {"epochs": 1.2}, {"lr0": float("nan")}, {"model": "yolo11n.pt"}])
def test_invalid_training_settings(change):
    with pytest.raises(ValueError):
        training.validate_config(change)


def test_explicit_cuda_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        training.get_device("0")
    assert training.get_device("auto").type == "cpu"


def test_baseline_reuses_yolo_export():
    yolo = yaml.safe_load((training.REPO / "configs/yolo_baseline.yaml").read_text())
    faster = training.load_config("configs/faster_rcnn_baseline.yaml")
    for key in ("data_root", "augmentation_copies", "augmentation_seed"):
        assert faster[key] == yolo[key]


def test_grid_rejects_model_or_dataset_changes(tmp_path):
    search = tmp_path / "search.yaml"
    search.write_text(yaml.safe_dump({"trial_epochs": 1, "space": {"data_root": ["other"]}}))
    with pytest.raises(ValueError, match="Grid parameters"):
        tuning.read_search(search, training.validate_config({}))

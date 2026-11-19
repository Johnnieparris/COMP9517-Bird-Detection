"""Train YOLO and retain comparable, shared-evaluator results at every epoch.

python -m src.dl.train_yolo --config configs/yolo_baseline.yaml
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
# The shared evaluation/prediction modules use top-level `data` imports.
sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.dataset import DEFAULT_ROOT
from data.split import load_split_manifest, load_split_from_manifest
from evaluation import IOU_THR, SCORE_THR, evaluate, print_results, run_and_time
from predictions import MAX_PER_IMAGE, save_predictions
from src.dl.augment import export_dataset

TUNABLE_KEYS = {"lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
                "warmup_momentum", "box", "cls", "dfl"}


def load_config(path, hyperparameters=None):
    config = yaml.safe_load(resolve_path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    if hyperparameters:
        tuned = yaml.safe_load(resolve_path(hyperparameters).read_text())
        if not isinstance(tuned, dict) or not tuned or set(tuned) - TUNABLE_KEYS:
            raise ValueError(f"Tuned overrides must contain only {sorted(TUNABLE_KEYS)}")
        config.update(tuned)
    return config


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.generic, torch.Tensor)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def select_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def prepare_data(root, run_dir, copies, seed, validation_root=None):
    """Use manifest-selected train images and the untouched official validation set."""
    official_path = REPO / "splits/official.json"
    official = load_split_manifest(official_path)
    if not root.exists():
        export_dataset(root, copies=copies, seed=seed)
    manifest_path = root / "splits.json" if root != DEFAULT_ROOT else official_path
    manifest = load_split_manifest(manifest_path)
    # Member E data-efficiency roots may train on a strict subset of the official training
    # images, but only when their manifest says so explicitly; every other check still applies.
    is_subset = bool(json.loads(Path(manifest_path).read_text()).get("subset_of_official", False))
    augmentation_path = root / "augmentation.json"
    augmentation = json.loads(augmentation_path.read_text()) if augmentation_path.exists() else None
    if augmentation and (augmentation["copies"] != copies or augmentation["seed"] != seed):
        raise ValueError("Existing augmented data uses different copies/seed; choose a new data_root")
    parents = augmentation["source_images"] if augmentation else {}
    allowed_train = set(official["train"])
    if len(manifest["train"]) != len(set(manifest["train"])):
        raise ValueError("Duplicate training IDs in manifest")
    if not is_subset and not allowed_train.issubset(manifest["train"]):
        raise ValueError("Training dataset must include all official training images")
    for image_id in manifest["train"]:
        if image_id in set(official["val"]) | set(official["test"]):
            raise ValueError(f"Holdout image in training: {image_id}")
        if image_id not in allowed_train and parents.get(image_id) not in allowed_train:
            raise ValueError(f"Unknown training image or augmentation source: {image_id}")
    train = load_split_from_manifest(root, manifest, "train")
    val = load_split_from_manifest(validation_root or DEFAULT_ROOT, official, "val")
    if len(train) != len(manifest["train"]):
        raise ValueError("Duplicate image IDs in dataset")
    for split, records in (("train", train), ("val", val)):
        (run_dir / f"{split}.txt").write_text("".join(f"{r.image_path.resolve()}\n" for r in records))
    data_path = run_dir / "data.yaml"
    data_path.write_text(yaml.safe_dump({"path": str(run_dir),
        "train": str(run_dir / "train.txt"), "val": str(run_dir / "val.txt"),
        "names": {0: "bird"}}, sort_keys=False))
    return data_path, val, {
        "root": str(root), "validation_root": str(validation_root or DEFAULT_ROOT),
        "train_images": len(train), "val_images": len(val),
        "official_manifest_sha256": sha256(official_path),
        "training_manifest_sha256": sha256(manifest_path),
        "augmentation": augmentation,
    }


def predict_function(model, device, imgsz):
    """Shared RGB -> XYXY+score contract, with the shared saved-prediction cap."""
    def predict(image):
        result = model.predict(image[:, :, ::-1].copy(), device=device, imgsz=imgsz,
                               conf=0.001, iou=0.7, max_det=MAX_PER_IMAGE,
                               verbose=False, save=False, augment=False)[0]
        # Moving tensors to CPU also synchronizes GPU work before timing ends.
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return np.column_stack((boxes, scores)).reshape(-1, 5)
    return predict


def write_timeline(run_dir, history):
    """One row per completed epoch; JSON fields preserve complete hyperparameters."""
    write_json(run_dir / "history.json", history)
    rows = []
    for event in history:
        rows.append({
            "run_id": event["run_id"], "method": event.get("method", "yolo11n"), "family": "ml",
            "split": "val", "epoch": event["epoch"], "timestamp_utc": event["timestamp_utc"],
            "checkpoint": event["checkpoint"], "elapsed_s": event["elapsed_s"],
            **event["scores"],
            "hyperparameters_json": json.dumps(event["hyperparameters"], sort_keys=True, default=json_value),
            "learning_rates_json": json.dumps(event["learning_rates"], sort_keys=True),
            "train_losses_json": json.dumps(event["train_losses"], sort_keys=True),
            "evaluation_json": json.dumps(event["evaluation"], sort_keys=True),
        })
    path = run_dir / "history.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class MilestoneRecorder:
    def __init__(self, run_dir, records, metadata):
        self.run_dir, self.records, self.metadata = run_dir, records, metadata
        history_path = run_dir / "history.json"
        self.history = json.loads(history_path.read_text()) if history_path.exists() else []
        self.previous_elapsed = self.history[-1]["elapsed_s"] if self.history else 0
        self.started = time.perf_counter()

    def __call__(self, trainer):
        from ultralytics import YOLO

        epoch = trainer.epoch + 1  # public history is one-based
        if any(event["epoch"] == epoch for event in self.history):
            raise ValueError(f"Duplicate milestone epoch {epoch}")
        milestone = self.run_dir / "milestones" / f"epoch_{epoch:04d}"
        # A runtime may have stopped after saving weights but before saving scores.
        milestone.mkdir(parents=True, exist_ok=True)
        checkpoint = self.run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        if Path(trainer.last).resolve() != checkpoint.resolve():
            shutil.copy2(trainer.last, checkpoint)
        hyperparameters = dict(vars(trainer.args))
        write_json(self.run_dir / "resolved_hyperparameters.json", hyperparameters)
        print(f"\nShared validation: epoch {epoch}, {len(self.records)} images", flush=True)
        model = YOLO(str(checkpoint))
        device = str(trainer.device)
        predictions, timing = run_and_time(
            self.records, predict_function(model, device, trainer.args.imgsz))
        scores = evaluate(self.records, predictions, timing)
        # Retain full-precision predictions for exact metric reproduction, plus team CSV.
        np.savez_compressed(milestone / "predictions.npz", **predictions)
        save_predictions("yolo", "val", self.records, predictions, folder=milestone)
        event = {
            "run_id": self.run_dir.name, "epoch": epoch, "timestamp_utc": utc_now(),
            "method": self.metadata.get("model", "yolo11n"),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "elapsed_s": self.previous_elapsed + time.perf_counter() - self.started,
            "scores": scores, "hyperparameters": hyperparameters,
            "learning_rates": dict(trainer.lr),
            "optimizer_groups": [{"lr": group["lr"], "weight_decay": group.get("weight_decay", 0)}
                                 for group in trainer.optimizer.param_groups],
            "train_losses": trainer.label_loss_items(trainer.tloss),
            "ultralytics_metrics": dict(trainer.metrics),
            "evaluation": self.metadata["evaluation"],
            "dataset": {key: value for key, value in self.metadata["dataset"].items()
                        if key != "augmentation"},
        }
        write_json(milestone / "scores.json", event)
        self.history.append(event)
        write_timeline(self.run_dir, self.history)
        best = max(self.history, key=lambda item: item["scores"]["mAP50"])
        if best is event:
            shutil.copy2(checkpoint, self.run_dir / "checkpoints/best_shared.pt")
            write_json(self.run_dir / "best_shared.json", event)
            save_predictions("yolo_" + self.run_dir.name, "val", self.records, predictions)
            # Keep a copy with the run when project-local predictions are ephemeral.
            shutil.copy2(milestone / "yolo_val.csv", self.run_dir / "best_shared_predictions.csv")
        print_results(scores, f"YOLO epoch {epoch} (shared evaluator)")
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()


def record_saved_checkpoint(recorder, checkpoint, device):
    """Recover an interrupted evaluation, or score a native tuner epoch checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = saved["epoch"] + 1
    if any(event["epoch"] == epoch for event in recorder.history):
        return
    if epoch < 1:
        raise ValueError(f"Checkpoint has no epoch/optimizer history: {checkpoint}")
    args = SimpleNamespace(**saved["train_args"])
    groups = (saved.get("optimizer") or {}).get("param_groups", [])
    results = saved.get("train_results") or {}
    losses = {key: values[-1] for key, values in results.items()
              if key.startswith("train/") and len(values)}
    trainer = SimpleNamespace(epoch=epoch - 1, last=checkpoint, args=args,
        device=device, lr={f"lr/pg{i}": group["lr"] for i, group in enumerate(groups)},
        optimizer=SimpleNamespace(param_groups=groups), metrics=saved.get("train_metrics", {}),
        tloss=None, label_loss_items=lambda _: losses)
    del saved
    recorder(trainer)


def resume_training(run_dir, device=None):
    """Resume the original epoch budget using a preserved, unstripped checkpoint."""
    from ultralytics import YOLO

    run_dir = resolve_path(run_dir)
    metadata = json.loads((run_dir / "run.json").read_text())
    if metadata["status"] == "completed":
        raise ValueError("This run is complete; use a new run for a different epoch budget")
    config = metadata["requested_hyperparameters"]
    selected_device = select_device(device or str(config.get("device", "auto")))
    data_path, records, dataset = prepare_data(resolve_path(config["data_root"]), run_dir,
        config.get("augmentation_copies", 1), config.get("augmentation_seed", 42),
        resolve_path(config["validation_root"]) if config.get("validation_root") else None)
    for key in ("official_manifest_sha256", "training_manifest_sha256"):
        if dataset[key] != metadata["dataset"][key]:
            raise ValueError(f"Dataset changed since the run started: {key}")
    recorder = MilestoneRecorder(run_dir, records, metadata)
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    if not checkpoints:
        raise ValueError("No epoch checkpoint yet; start a new run")
    # Rebuild scores if a disconnect interrupted the last shared evaluation.
    for checkpoint in checkpoints:
        record_saved_checkpoint(recorder, checkpoint, selected_device)
    checkpoint = checkpoints[-1]
    completed = recorder.history[-1]["epoch"]
    metadata.setdefault("resume_sessions", []).append({"timestamp_utc": utc_now(),
        "device": selected_device, "after_epoch": completed})
    try:
        metadata["status"] = "training"
        write_json(run_dir / "run.json", metadata)
        if completed < config["epochs"]:
            model = YOLO(str(checkpoint))
            model.add_callback("on_model_save", recorder)
            model.train(resume=True, data=str(data_path), device=selected_device,
                        save_dir=str(run_dir / "trainer"))
        metadata.update(status="completed", completed_epochs=len(recorder.history),
            best_epoch=max(recorder.history, key=lambda event: event["scores"]["mAP50"])["epoch"],
            finished_utc=utc_now())
        metadata.pop("error", None)
        write_json(run_dir / "run.json", metadata)
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(run_dir / "run.json", metadata)
        raise
    return run_dir


def train(config_path, run_name=None, epochs=None, device=None, output_root=None, hyperparameters=None):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    config_path = resolve_path(config_path)
    config = load_config(config_path, hyperparameters)
    if epochs is not None:
        config["epochs"] = epochs
    if device is not None:
        config["device"] = device
    requested = dict(config)
    data_root = resolve_path(config.pop("data_root"))
    validation_root = resolve_path(config.pop("validation_root")) if config.get("validation_root") else None
    copies = config.pop("augmentation_copies", 1)
    augmentation_seed = config.pop("augmentation_seed", 42)
    weights = config.pop("model", "yolo11n.pt")
    print(f"Training script: {Path(__file__).resolve()}", flush=True)
    print(f"Training config: {config_path}", flush=True)
    print(f"Model weights: {weights}", flush=True)
    if config.get("epochs", 0) < 1:
        raise ValueError("epochs must be positive")
    if config.get("optimizer", "auto").lower() == "auto":
        raise ValueError("Choose an explicit optimizer so hyperparameter searches use the requested LR")
    reserved = {"project", "name", "exist_ok", "data", "resume", "save", "save_period", "fraction"}
    if reserved.intersection(config):
        raise ValueError(f"Pipeline-controlled settings: {sorted(reserved.intersection(config))}")
    config["device"] = select_device(str(config.get("device", "auto")))
    get_cfg(overrides=config)  # reject unknown settings before generating data
    run_name = run_name or "yolo_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Run name may contain only letters, numbers, underscores and hyphens")
    run_dir = resolve_path(output_root or "results") / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "source_config.yaml")
    metadata = {"run_id": run_name, "status": "preparing", "started_utc": utc_now(),
                "requested_hyperparameters": requested, "device": config["device"],
                "platform": platform.platform(), "python": platform.python_version(),
                "versions": {name: version(name) for name in ("torch", "torchvision", "ultralytics", "numpy")},
                "git_commit": git_commit(), "model": weights,
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "source_sha256": {str(path.relative_to(REPO)): sha256(path) for path in
                    (Path(__file__).resolve(), REPO / "src/evaluation.py", REPO / "src/dl/augment.py")},
                "evaluation": {"split": "val", "iou_threshold": IOU_THR, "score_threshold": SCORE_THR,
                    "prediction_confidence": 0.001, "nms_iou": 0.7, "max_detections": MAX_PER_IMAGE,
                    "selection_metric": "mAP50", "timing": "batch=1, 3 warmups, excludes image loading"}}
    write_json(run_dir / "run.json", metadata)
    try:
        data_path, val_records, dataset_info = prepare_data(data_root, run_dir, copies, augmentation_seed, validation_root)
        metadata["dataset"] = dataset_info
        metadata["status"] = "training"
        write_json(run_dir / "run.json", metadata)
        recorder = MilestoneRecorder(run_dir, val_records, metadata)
        model = YOLO(weights)
        model.add_callback("on_model_save", recorder)
        model.train(**config, data=str(data_path), project=str(run_dir), name="trainer",
                    exist_ok=False, save=True, save_period=-1, plots=False)
        if not recorder.history:
            raise RuntimeError("Training finished without a saved/evaluated milestone")
        metadata["status"] = "completed"
        metadata["completed_epochs"] = len(recorder.history)
        metadata["best_epoch"] = max(recorder.history, key=lambda event: event["scores"]["mAP50"])["epoch"]
        metadata["finished_utc"] = utc_now()
        write_json(run_dir / "run.json", metadata)
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = str(error)
        write_json(run_dir / "run.json", metadata)
        raise
    print(f"\nSaved training timeline: {run_dir / 'history.csv'}", flush=True)
    print(f"Best shared-evaluator checkpoint: epoch {metadata['best_epoch']}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/yolo_baseline.yaml")
    parser.add_argument("--name", help="unique run ID; defaults to a UTC timestamp")
    parser.add_argument("--epochs", type=int, help="override the configured epoch count")
    parser.add_argument("--device", help="auto, cpu, mps, or CUDA device index")
    parser.add_argument("--output-root", help="results directory, e.g. a mounted Google Drive folder")
    parser.add_argument("--hyperparameters", help="tuner-exported hyperparameter YAML to merge")
    parser.add_argument("--resume", type=Path, help="existing run directory to resume to its original epoch budget")
    args = parser.parse_args()
    if args.resume:
        if args.epochs is not None or args.name or args.output_root or args.hyperparameters:
            parser.error("--resume uses saved settings; only --device may be overridden")
        resume_training(args.resume, args.device)
    else:
        train(args.config, args.name, args.epochs, args.device, args.output_root, args.hyperparameters)


if __name__ == "__main__":
    main()

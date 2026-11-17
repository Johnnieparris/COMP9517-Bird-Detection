"""Train torchvision Faster R-CNN with the same data and evaluator as YOLO.

python -m src.ml.train_faster_rcnn --config configs/faster_rcnn_baseline.yaml
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import re
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.ml.dataset import BirdDetectionDataset, detection_collate
# Reuse the existing split guards and timeline format without importing Ultralytics.
from src.ml.train_yolo import (REPO, DEFAULT_ROOT, IOU_THR, SCORE_THR, MAX_PER_IMAGE,
    prepare_data, resolve_path, sha256, utc_now, write_json, write_timeline,
    evaluate, print_results, run_and_time, save_predictions)

METHOD = "fasterrcnn_resnet50_fpn"
DEFAULTS = {
    "data_root": "data/augmented_birds_baseline", "validation_root": None,
    "augmentation_copies": 1, "augmentation_seed": 42, "pretrained": True,
    "epochs": 50, "imgsz": 640, "batch": 1, "accumulation_steps": 2,
    "device": "auto", "workers": 2, "optimizer": "SGD", "lr0": .005,
    "momentum": .9, "weight_decay": .0005, "lrf": .1, "warmup_epochs": 1.,
    "grad_clip": 10., "trainable_backbone_layers": 3, "amp": False,
    "seed": 42, "patience": 0,
}
TUNABLE_KEYS = {"lr0", "momentum", "weight_decay", "lrf", "warmup_epochs"}


def validate_config(config):
    unknown = set(config) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown Faster R-CNN settings: {sorted(unknown)}")
    config = {**DEFAULTS, **config}
    for key in ("epochs", "imgsz", "batch", "accumulation_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("workers", "seed", "patience", "augmentation_copies", "augmentation_seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if type(config["trainable_backbone_layers"]) is not int or not 0 <= config["trainable_backbone_layers"] <= 5:
        raise ValueError("trainable_backbone_layers must be between 0 and 5")
    for key in ("lr0", "momentum", "weight_decay", "lrf", "warmup_epochs", "grad_clip"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["lr0"] <= 0 or not 0 <= config["momentum"] < 1 or not 0 <= config["lrf"] <= 1:
        raise ValueError("Require lr0 > 0, momentum in [0, 1), and lrf in [0, 1]")
    if config["optimizer"] not in ("SGD", "AdamW"):
        raise ValueError("optimizer must be SGD or AdamW")
    if any(type(config[k]) is not bool for k in ("pretrained", "amp")):
        raise ValueError("pretrained and amp must be YAML booleans")
    if not isinstance(config["data_root"], str) or not config["data_root"]:
        raise ValueError("data_root must be a path")
    if config["validation_root"] is not None and not isinstance(config["validation_root"], str):
        raise ValueError("validation_root must be a path or null")
    return config


def load_config(path):
    config = yaml.safe_load(resolve_path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    return validate_config(config)


def get_device(requested):
    name = str(requested)
    if name == "auto":
        # torchvision detection operations are supported on CPU and CUDA.
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif name.isdigit():
        name = "cuda:" + name
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu, auto, or a CUDA index (e.g. --device 0)")
    if device.type == "cuda" and (not torch.cuda.is_available() or
            (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError("Requested CUDA GPU is unavailable in this Python environment")
    return device


def build_model(config, initialize=True):
    from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops.misc import FrozenBatchNorm2d

    # Use the same FrozenBatchNorm architecture when loading and when resuming.
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None,
        norm_layer=FrozenBatchNorm2d, trainable_layers=config["trainable_backbone_layers"])
    model = FasterRCNN(backbone, num_classes=91,
        min_size=config["imgsz"], max_size=config["imgsz"],
        box_score_thresh=.001, box_nms_thresh=.7, box_detections_per_img=MAX_PER_IMAGE)
    if initialize and config["pretrained"]:
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def predict_function(model, device):
    @torch.inference_mode()
    def predict(image):
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(device).float() / 255
        result = model([tensor])[0]
        keep = result["labels"] == 1
        boxes = result["boxes"][keep].detach().cpu().numpy()
        scores = result["scores"][keep].detach().cpu().numpy()
        # CPU transfer synchronizes CUDA for the shared single-image timing contract.
        return np.column_stack((boxes, scores)).reshape(-1, 5)
    return predict


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def learning_rate(config, epoch, step, steps):
    progress = (epoch - 1) / max(config["epochs"] - 1, 1)
    cosine = config["lrf"] + (1 - config["lrf"]) * (1 + math.cos(math.pi * progress)) / 2
    warmup = min(1., ((epoch - 1) * steps + step + 1) / max(config["warmup_epochs"] * steps, 1))
    return config["lr0"] * cosine * warmup


def train_epoch(model, loader, optimizer, scaler, device, config, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals, images_seen = {}, 0
    steps = len(loader)
    accumulation = config["accumulation_steps"]
    for step, (images, targets) in enumerate(loader):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        # Normalize by actual images in this accumulation group, including a short last batch.
        group_start = (step // accumulation) * accumulation
        group_images = min(accumulation * config["batch"], len(loader.dataset) - group_start * config["batch"])
        with torch.autocast(device_type=device.type, enabled=config["amp"]):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {step + 1}")
        scaler.scale(loss * len(images) / group_images).backward()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.) + value.detach().item() * len(images)
        images_seen += len(images)
        if (step + 1) % accumulation == 0 or step + 1 == steps:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, epoch, step, steps)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"] or float("inf"), error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            print(f"Epoch {epoch}/{config['epochs']} batch {step + 1}/{steps}: loss {loss.item():.4f}", flush=True)
    return {key: value / images_seen for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, scaler, generator, epoch, config, losses, elapsed):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "config": config,
        "train_losses": losses, "elapsed_s": elapsed, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_rng": generator.get_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate()}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def record_epoch(run_dir, model, device, records, metadata, checkpoint, epoch, losses, lr, elapsed, history):
    started = time.perf_counter()
    model.eval()
    print(f"Shared validation: epoch {epoch}, {len(records)} images", flush=True)
    predictions, timing = run_and_time(records, predict_function(model, device))
    scores = evaluate(records, predictions, timing)
    milestone = run_dir / "milestones" / f"epoch_{epoch:04d}"
    milestone.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(milestone / "predictions.npz", **predictions)
    csv_path = save_predictions("faster_rcnn", "val", records, predictions, folder=milestone)
    event = {"run_id": run_dir.name, "method": METHOD, "epoch": epoch,
        "timestamp_utc": utc_now(), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "elapsed_s": elapsed + time.perf_counter() - started,
        "scores": scores, "hyperparameters": metadata["config"],
        "learning_rates": {"lr": lr}, "train_losses": losses,
        "evaluation": metadata["evaluation"]}
    write_json(milestone / "scores.json", event)
    history.append(event)
    write_timeline(run_dir, history)
    publish_best(run_dir, history)
    if max(history, key=lambda item: item["scores"]["mAP50"]) is event:
        save_predictions("faster_rcnn_" + run_dir.name, "val", records, predictions)
    print_results(scores, f"Faster R-CNN epoch {epoch} (shared evaluator)")
    return csv_path


def publish_best(run_dir, history):
    best = max(history, key=lambda event: event["scores"]["mAP50"])
    # Derive paths from this run so a run folder can be moved between machines.
    checkpoint = run_dir / "checkpoints" / f"epoch_{best['epoch']:04d}.pt"
    current = run_dir / "best_shared.json"
    if not current.exists() or json.loads(current.read_text())["epoch"] != best["epoch"] or not (run_dir / "checkpoints/best_shared.pt").exists():
        shutil.copy2(checkpoint, run_dir / "checkpoints/best_shared.pt")
    source = run_dir / "milestones" / f"epoch_{best['epoch']:04d}" / "faster_rcnn_val.csv"
    shutil.copy2(source, run_dir / "best_shared_predictions.csv")
    write_json(current, best)


def train(config_path=None, run_name=None, epochs=None, device=None, output_root="results", resume=None):
    if resume:
        if config_path is not None or run_name is not None or epochs is not None or output_root != "results":
            raise ValueError("Resume uses the saved config; only --device may be overridden")
        run_dir = resolve_path(resume)
        metadata = json.loads((run_dir / "run.json").read_text())
        if metadata["status"] == "completed":
            raise ValueError("This run is already complete; start a new run instead")
        config = validate_config(metadata["config"])
    else:
        config = load_config(config_path or "configs/faster_rcnn_baseline.yaml")
        if epochs is not None:
            config = validate_config({**config, "epochs": epochs})
        name = run_name or "faster_rcnn_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Run name may contain only letters, numbers, underscores and hyphens")
        run_dir = resolve_path(output_root) / name
        metadata = {}
    if device is not None:
        config["device"] = str(device)
    selected = get_device(config["device"])
    if config["amp"] and selected.type != "cuda":
        raise ValueError("amp: true requires CUDA; use amp: false for CPU")
    root = resolve_path(config["data_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Existing YOLO dataset required: {root}. Copy the same augmented export first.")
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    metadata.update(run_id=run_dir.name, model=METHOD, config=config, status="preparing",
        device=str(selected), platform=platform.platform(),
        versions={key: version(key) for key in ("torch", "torchvision", "numpy")},
        evaluation={"split": "val", "iou_threshold": IOU_THR, "score_threshold": SCORE_THR,
            "prediction_confidence": .001, "nms_iou": .7, "max_detections": MAX_PER_IMAGE,
            "selection_metric": "mAP50", "timing": "batch=1, 3 warmups, excludes image loading"})
    metadata.setdefault("started_utc", utc_now())
    metadata["source_sha256"] = {str(p.relative_to(REPO)): sha256(p) for p in
        (Path(__file__).resolve(), REPO / "src/ml/dataset.py", REPO / "src/evaluation.py")}
    write_json(run_dir / "run.json", metadata)
    try:
        _, records, dataset_info = prepare_data(root, run_dir, config["augmentation_copies"],
            config["augmentation_seed"], resolve_path(config["validation_root"]) if config["validation_root"] else None)
        if resume and "dataset" in metadata and metadata["dataset"] != dataset_info:
            raise ValueError("Dataset paths/manifests/augmentation changed since this run began")
        metadata["dataset"] = dataset_info
        manifest = REPO / "splits/official.json" if root == DEFAULT_ROOT else root / "splits.json"
        dataset = BirdDetectionDataset(root, manifest_path=manifest)
        if not dataset or not records:
            raise ValueError("Training and validation splits must be nonempty")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        generator = torch.Generator().manual_seed(config["seed"])
        loader = DataLoader(dataset, batch_size=config["batch"], shuffle=True,
            num_workers=config["workers"], collate_fn=detection_collate,
            worker_init_fn=seed_worker, generator=generator, pin_memory=selected.type == "cuda")
        history_path = run_dir / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
        checkpoint = checkpoints[-1] if resume and checkpoints else None
        model = build_model(config, initialize=checkpoint is None).to(selected)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if config["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(parameters, lr=config["lr0"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        else:
            optimizer = torch.optim.AdamW(parameters, lr=config["lr0"], betas=(config["momentum"], .999), weight_decay=config["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
        start_epoch, previous_elapsed = 1, 0.
        if checkpoint:
            # These are this trainer's own checkpoints. Never load untrusted .pt files.
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            previous_elapsed = max(state["elapsed_s"], history[-1]["elapsed_s"] if history else 0)
            if not history or history[-1]["epoch"] < state["epoch"]:
                record_epoch(run_dir, model, selected, records, metadata, checkpoint,
                    state["epoch"], state["train_losses"], optimizer.param_groups[0]["lr"], previous_elapsed, history)
            torch.set_rng_state(state["torch_rng"])
            if selected.type == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            generator.set_state(state["loader_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            start_epoch = state["epoch"] + 1
            del state
        if history:
            publish_best(run_dir, history)
        metadata["status"] = "training"
        write_json(run_dir / "run.json", metadata)
        write_json(run_dir / "resolved_hyperparameters.json", config)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        started = time.perf_counter()
        for epoch in range(start_epoch, config["epochs"] + 1):
            if history and config["patience"] and history[-1]["epoch"] - max(history, key=lambda e: e["scores"]["mAP50"])["epoch"] >= config["patience"]:
                break
            losses = train_epoch(model, loader, optimizer, scaler, selected, config, epoch)
            checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
            elapsed = previous_elapsed + time.perf_counter() - started
            save_checkpoint(checkpoint, model, optimizer, scaler, generator, epoch, config, losses, elapsed)
            record_epoch(run_dir, model, selected, records, metadata, checkpoint, epoch,
                losses, optimizer.param_groups[0]["lr"], elapsed, history)
        best = max(history, key=lambda event: event["scores"]["mAP50"])
        metadata.update(status="completed", completed_epochs=len(history), best_epoch=best["epoch"],
            stopped_early=len(history) < config["epochs"], finished_utc=utc_now())
        metadata.pop("error", None)
        write_json(run_dir / "run.json", metadata)
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(run_dir / "run.json", metadata)
        raise
    print(f"Saved training timeline: {run_dir / 'history.csv'}", flush=True)
    print(f"Best shared checkpoint: epoch {best['epoch']}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--resume", help="existing run directory; completes its original epoch budget")
    args = parser.parse_args()
    train(args.config, args.name, args.epochs, args.device, args.output_root, args.resume)


if __name__ == "__main__":
    main()

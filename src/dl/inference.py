"""Portable inference for the selected YOLO and Faster R-CNN checkpoints."""
import hashlib
from pathlib import Path

import torch
import yaml

from src.dl.train_faster_rcnn import build_model, get_device, predict_function as faster_predict
from src.dl.train_yolo import predict_function as yolo_predict, resolve_path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_detector(config_path, device="auto"):
    config = yaml.safe_load(resolve_path(config_path).read_text())
    if not isinstance(config, dict) or set(config) != {"method", "checkpoint", "imgsz"}:
        raise ValueError("Inference config requires method, checkpoint and imgsz")
    method = config["method"]
    if method not in ("yolo", "faster_rcnn"):
        raise ValueError("method must be yolo or faster_rcnn")
    if type(config["imgsz"]) is not int or config["imgsz"] < 1:
        raise ValueError("imgsz must be a positive integer")
    path = resolve_path(config["checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(f"Selected model missing: {path}. Transfer the models/ folder as well as the code.")
    selected = get_device(device)
    if method == "yolo":
        from ultralytics import YOLO
        model = YOLO(str(path), task="detect")
        predict = yolo_predict(model, str(selected), config["imgsz"])
    else:
        # Exported inference checkpoints contain tensors and plain Python values only.
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != "faster_rcnn_inference_v1":
            raise ValueError("Use models/faster_rcnn/best.pt exported by scripts/organize_results.py")
        model_config = {**saved["config"], "imgsz": config["imgsz"]}
        model = build_model(model_config, initialize=False).to(selected).eval()
        model.load_state_dict(saved["model"])
        predict = faster_predict(model, selected)
    return predict, {**config, "device": str(selected), "checkpoint_sha256": file_sha256(path)}

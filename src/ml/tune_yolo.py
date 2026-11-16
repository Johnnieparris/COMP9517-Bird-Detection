"""Native Ultralytics hyperparameter tuning with retained shared-evaluator timelines.

The pinned native tuner runs trials in subprocesses, so model callbacks do not
propagate. Preserve every epoch checkpoint and evaluate them after each trial.
Native fitness guides evolution; shared validation mAP50 reranks completed trials.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import yaml

from src.ml.train_yolo import (TUNABLE_KEYS, MilestoneRecorder, load_config, prepare_data,
    record_saved_checkpoint, resolve_path, select_device, sha256, utc_now, write_json)

TESTED_ULTRALYTICS = "8.4.211"


def read_search(path):
    search = yaml.safe_load(resolve_path(path).read_text())
    if not isinstance(search, dict) or set(search) != {"iterations", "trial_epochs", "space"}:
        raise ValueError("Search config needs iterations, trial_epochs, and space")
    for name in ("iterations", "trial_epochs"):
        if type(search[name]) is not int or search[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(search["space"], dict) or not search["space"]:
        raise ValueError("Search space must be a nonempty mapping")
    if set(search["space"]) - TUNABLE_KEYS:
        raise ValueError(f"Supported search parameters: {sorted(TUNABLE_KEYS)}")
    for name, bounds in search["space"].items():
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in bounds)
                or not 0 <= bounds[0] < bounds[1]):
            raise ValueError(f"Invalid bounds for {name}: {bounds}")
    return search


def export_selected_config(root, base, results):
    """Emit complete final settings plus a small safe override file."""
    successful = [result for result in results if result.get("best_shared")]
    if not successful:
        raise RuntimeError("No successful tuning trials with shared evaluation scores")
    best = max(successful, key=lambda result: result["best_shared"]["scores"]["mAP50"])
    hyperparameters = best["hyperparameters"]
    (root / "best_shared_hyperparameters.yaml").write_text(yaml.safe_dump(hyperparameters))
    final = {**base, **hyperparameters}  # restore the final epoch budget, not the trial budget
    (root / "best_training_config.yaml").write_text(yaml.safe_dump(final, sort_keys=False))
    write_json(root / "best_shared_trial.json", best)
    write_json(root / "shared_trials.json", results)


def tune(config_path="configs/yolo_colab.yaml", search_path="configs/yolo_tune.yaml",
         output_root="results", name="yolo_tune", resume=False, device=None):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.engine.tuner import Tuner
    import re

    if version("ultralytics") != TESTED_ULTRALYTICS:
        raise RuntimeError(f"Install requirements-colab.txt: this tuner integration requires ultralytics=={TESTED_ULTRALYTICS}")
    base, search = load_config(config_path), read_search(search_path)
    if device is not None:
        base["device"] = device
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Use letters, numbers, underscores and hyphens for the tuning name")
    root = resolve_path(output_root) / name
    request = {"base": base, "search": search, "ultralytics": version("ultralytics")}
    if resume:
        previous = json.loads((root / "request.json").read_text())
        if previous != request:
            raise ValueError("Resume requires the same config, search space and budgets")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "request.json", request)
    args = dict(base)
    weights = args.pop("model", "yolo11n.pt")
    data_root = resolve_path(args.pop("data_root"))
    validation_root = resolve_path(args.pop("validation_root")) if args.get("validation_root") else None
    copies, seed = args.pop("augmentation_copies", 1), args.pop("augmentation_seed", 42)
    args["device"] = select_device(str(args.get("device", "auto")))
    args["epochs"] = search["trial_epochs"]
    if args.get("optimizer", "auto").lower() == "auto":
        raise ValueError("Set an explicit optimizer before tuning")
    forbidden = {"project", "name", "exist_ok", "data", "resume", "save", "save_period", "fraction"}
    if forbidden.intersection(args):
        raise ValueError("Output, dataset and checkpoint settings are controlled by the tuner wrapper")
    defaults = vars(get_cfg(overrides=args))
    for key, (low, high) in search["space"].items():
        if not low <= defaults[key] <= high:
            raise ValueError(f"Starting {key}={defaults[key]} is outside the search bounds")
    data_path, records, dataset = prepare_data(data_root, root, copies, seed, validation_root)
    state_path = root / "tuning.json"
    if resume and state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous["dataset"]["training_manifest_sha256"] != dataset["training_manifest_sha256"]:
            raise ValueError("Training dataset changed since tuning began")
        if previous["dataset"]["official_manifest_sha256"] != dataset["official_manifest_sha256"]:
            raise ValueError("Official split changed since tuning began")
    evaluation = {"split": "val", "selection_metric": "mAP50", "iou_threshold": 0.5,
        "score_threshold": 0.5, "prediction_confidence": 0.001, "nms_iou": 0.7,
        "timing": "batch=1, 3 warmups, excludes image loading"}
    from src.ml.train_yolo import IOU_THR, SCORE_THR, MAX_PER_IMAGE
    evaluation.update(iou_threshold=IOU_THR, score_threshold=SCORE_THR, max_detections=MAX_PER_IMAGE)
    state = {"status": "running", "updated_utc": utc_now(), "dataset": dataset,
             "request": request, "native_selection": "Ultralytics fitness",
             "shared_selection": "validation mAP50", "evaluation": evaluation}
    write_json(state_path, state)
    summaries = []

    def harvest(result):
        iteration = result["iteration"]
        trial_dir = root / "trials" / f"trial_{iteration:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_metadata = {"model": weights, "dataset": dataset, "evaluation": evaluation,
                          "native_result": result, "recorded_utc": utc_now()}
        write_json(trial_dir / "run.json", trial_metadata)
        recorder = MilestoneRecorder(trial_dir, records, trial_metadata)
        # One official dataset per trial. Native epoch filenames are zero-based.
        native_dirs = list(result.get("save_dirs", {}).values())
        if len(native_dirs) != 1:
            raise RuntimeError("Expected exactly one dataset per tuning trial")
        checkpoints = sorted((Path(native_dirs[0]) / "weights").glob("epoch*.pt"),
                             key=lambda p: int(p.stem.removeprefix("epoch")))
        for checkpoint in checkpoints:
            record_saved_checkpoint(recorder, checkpoint, args["device"])
        # Native failures are retained but never selected as successful candidates.
        complete = len(recorder.history) == search["trial_epochs"] and not result.get("failed_datasets")
        best = max(recorder.history, key=lambda e: e["scores"]["mAP50"]) if complete else None
        summary = {"iteration": iteration, "hyperparameters": result["hyperparameters"],
                   "native_fitness": result["fitness"], "best_shared": best,
                   "status": "completed" if complete else "failed_or_incomplete"}
        summaries.append(summary)
        write_json(root / "shared_trials.json", summaries)
        if any(item["best_shared"] for item in summaries):
            export_selected_config(root, base, summaries)

    class RetainedTuner(Tuner):
        # This hook is specific to the pinned version. Native evolution is unchanged.
        def _save_local_result(self, result):
            super()._save_local_result(result)
            harvest(result)

    native_log = root / "native" / "tune_results.ndjson"
    try:
        prior = [json.loads(line) for line in native_log.read_text().splitlines() if line.strip()] if native_log.exists() else []
        for result in prior:
            harvest(result)  # idempotent; also finishes evaluation interrupted by a disconnect
        if len(prior) < search["iterations"]:
            model = YOLO(weights)
            tuner = RetainedTuner(args={**args, "model": weights, "data": str(data_path),
                "task": "detect", "mode": "train", "project": str(root), "name": "native",
                "resume": resume, "space": {k: tuple(v) for k, v in search["space"].items()},
                "save": True, "save_period": 1, "val": True, "plots": False}, model=model.model)
            tuner(iterations=search["iterations"], cleanup=False)
        export_selected_config(root, base, summaries)
        state.update(status="completed", finished_utc=utc_now(),
                     successful_trials=sum(item["status"] == "completed" for item in summaries))
        write_json(state_path, state)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(state_path, state)
        raise
    print(f"Final training configuration: {root / 'best_training_config.yaml'}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/yolo_colab.yaml")
    parser.add_argument("--search", default="configs/yolo_tune.yaml")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--name", default="yolo_tune")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    tune(args.config, args.search, args.output_root, args.name, args.resume, args.device)


if __name__ == "__main__":
    main()

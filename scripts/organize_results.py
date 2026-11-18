"""Export best available models and compact reports; move raw runs to archive/.

Never deletes training artifacts. Existing compact reports are regenerated from
the raw runs (under results/ initially, archive/training/ afterward).
"""
import csv
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

import numpy as np
import torch

from src.dl.inference import file_sha256
from src.dl.train_yolo import write_json
from data.dataset import DEFAULT_ROOT
from data.split import load_split_manifest, load_split_from_manifest
from evaluation import evaluate, run_and_time
from predictions import MAX_PER_IMAGE, save_predictions

METRICS = ("mAP50", "mAP50_95", "precision", "recall", "f1", "count_mae", "count_bias")


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def family(model):
    if "fasterrcnn" in model or "faster_rcnn" in model:
        return "faster_rcnn"
    if "yolo" in model:
        return "yolo"
    raise ValueError(f"Unknown model family: {model}")


def choose_available(runs):
    selected = {}
    for run in runs:
        if not run["weights"].is_file():
            continue
        method = run["method"]
        if method not in selected or run["best"]["scores"]["mAP50"] > selected[method]["best"]["scores"]["mAP50"]:
            selected[method] = run
    return selected


def dataset_summary(run):
    data = run["metadata"]["dataset"]
    augmentation = data.get("augmentation") or {}
    expected = augmentation.get("counts", {}).get("train")
    actual = data.get("train_images")
    warning = ""
    if expected is not None and expected != actual:
        warning = f"Run used {actual} training IDs; augmentation metadata describes {expected}. Do not claim all augmented copies were trained on."
    return {"train_images": actual, "val_images": data.get("val_images"),
        "official_manifest_sha256": data.get("official_manifest_sha256"),
        "training_manifest_sha256": data.get("training_manifest_sha256"),
        "augmentation_copies": augmentation.get("copies"), "augmentation_seed": augmentation.get("seed"),
        "augmentation_declared_train_images": expected, "warning": warning}


def export_model(root, run):
    method, best = run["method"], run["best"]
    source_hash = file_sha256(run["weights"])
    if source_hash != best["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash does not match saved evaluation: {run['name']}")
    folder = root / "models" / method
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "best.pt"
    previous = folder / "metadata.json"
    if previous.exists() and output.exists():
        old = read_json(previous)
        if old["source_checkpoint_sha256"] == source_hash and old["checkpoint_sha256"] == file_sha256(output):
            return old
        raise FileExistsError(f"A different model already exists at {output}; archive it before replacing")
    temporary = folder / "best.tmp.pt"
    if method == "yolo":
        from ultralytics.utils.torch_utils import strip_optimizer
        if not strip_optimizer(run["weights"], s=str(temporary)):
            raise ValueError("Could not export YOLO checkpoint")
    else:
        # Imported checkpoints were produced by our trainer on the user's PC.
        state = torch.load(run["weights"], map_location="cpu", weights_only=False)
        torch.save({"format": "faster_rcnn_inference_v1", "model": state["model"],
            "config": state["config"], "epoch": state["epoch"]}, temporary)
    temporary.replace(output)
    config = run["metadata"].get("config", run["metadata"].get("requested_hyperparameters"))
    metadata = {"method": method, "checkpoint": str(output.relative_to(root)),
        "checkpoint_sha256": file_sha256(output), "source_checkpoint_sha256": source_hash,
        "source_run": run["name"], "source_archive": f"archive/training/{run['name']}",
        "epoch": best["epoch"], "selection": "highest shared validation mAP50 among available imported checkpoints",
        "scores": best["scores"], "evaluation": best["evaluation"], "training_config": config,
        "dataset": dataset_summary(run), "training_versions": run["metadata"].get("versions"),
        "inference_only": True}
    write_json(previous, metadata)
    return metadata


def organize(root=ROOT):
    root = Path(root)
    results, archive = root / "results", root / "archive/training"
    folders = {}
    for parent in (archive, results):
        if not parent.exists():
            continue
        for folder in parent.iterdir():
            if folder.is_dir() and ((folder / "best_shared.json").exists() or (folder / "shared_trials.json").exists()):
                if folder.name in folders:
                    raise ValueError(f"Both archived and unarchived run exist: {folder.name}")
                folders[folder.name] = folder
    records = load_split_from_manifest(DEFAULT_ROOT, load_split_manifest(root / "splits/official.json"), "val")
    runs = []
    for name, folder in sorted(folders.items()):
        if not (folder / "best_shared.json").exists():
            continue
        metadata, best = read_json(folder / "run.json"), read_json(folder / "best_shared.json")
        history = read_json(folder / "history.json")
        if max(history, key=lambda event: event["scores"]["mAP50"])["epoch"] != best["epoch"]:
            raise ValueError(f"Best checkpoint disagrees with history: {name}")
        prediction_path = folder / "milestones" / f"epoch_{best['epoch']:04d}" / "predictions.npz"
        with np.load(prediction_path) as saved:
            predictions = {key: saved[key] for key in saved.files}
        if set(predictions) != {record.image_id for record in records}:
            raise ValueError(f"Validation IDs differ: {name}")
        recomputed = evaluate(records, predictions)
        if any(not np.isclose(recomputed[key], best["scores"][key], atol=1e-8, rtol=0) for key in METRICS):
            raise ValueError(f"Stored predictions do not reproduce scores: {name}")
        runs.append({"name": name, "folder": folder, "metadata": metadata, "best": best,
            "history": history, "method": family(metadata["model"]),
            "weights": folder / "checkpoints/best_shared.pt", "predictions": predictions})
    selected = choose_available(runs)
    if set(selected) != {"yolo", "faster_rcnn"}:
        raise ValueError("Both YOLO and Faster R-CNN best checkpoints are required")
    # Verify all candidates before making the portable exports or moving anything.
    for run in runs:
        if run["weights"].is_file() and file_sha256(run["weights"]) != run["best"]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {run['name']}")
    for run in selected.values():
        export_model(root, run)
    compact = []
    for run in runs:
        meta, best = run["metadata"], run["best"]
        is_selected = selected[run["method"]]["name"] == run["name"]
        compact.append({"method": run["method"], "run": run["name"], "split": "val",
            "selected": is_selected, "epochs_evaluated": len(run["history"]), "best_epoch": best["epoch"],
            **{key: best["scores"][key] for key in METRICS}, "score_threshold": .5,
            "ms_per_image": best["scores"].get("ms_per_image"), "device": meta.get("device"),
            "platform": meta.get("platform"), "train_images": meta["dataset"]["train_images"]})
        summary = {"run": run["name"], "method": run["method"], "selected": is_selected,
            "status": meta["status"], "epochs_evaluated": len(run["history"]), "best_epoch": best["epoch"],
            "scores": best["scores"], "evaluation": best["evaluation"], "dataset": dataset_summary(run),
            "config": meta.get("config", meta.get("requested_hyperparameters")),
            "versions": meta.get("versions"), "platform": meta.get("platform"), "device": meta.get("device"),
            "archive": f"archive/training/{run['name']}"}
        (results / "runs").mkdir(exist_ok=True)
        write_json(results / "runs" / f"{run['name']}.json", summary)
        timeline = []
        for event in run["history"]:
            timeline.append({"epoch": event["epoch"], "elapsed_s": event["elapsed_s"],
                **{key: event["scores"][key] for key in METRICS},
                **event.get("train_losses", {}), **event.get("learning_rates", {})})
        write_csv(results / "timelines" / f"{run['name']}.csv", timeline)
        if is_selected:
            save_predictions(run["method"], "val", records, run["predictions"], folder=root / "predictions")
    write_csv(results / "ml_comparison.csv", compact)
    # Re-evaluate selected M1 with the shared protocol, not its historical best-F1 threshold.
    from src.classical.pipeline import load_config, detect
    cfg = load_config(root / "configs/m1_seg_colour_tuned.yaml")
    def m1_predict(image):
        dets = detect(image[:, :, ::-1].copy(), cfg)
        return dets[np.argsort(-dets[:, 4], kind="stable")][:MAX_PER_IMAGE]
    m1_predictions, m1_timing = run_and_time(records, m1_predict)
    m1_scores = evaluate(records, m1_predictions, m1_timing)
    save_predictions("m1", "val", records, m1_predictions, folder=root / "predictions")
    shared = [{"method": "m1_colour", "split": "val", "mAP50": m1_scores["mAP50"],
        **{key: m1_scores[key] for key in METRICS[1:]}, "score_threshold": .5,
        "ms_per_image": m1_scores["ms_per_image"], "device": "cpu", "platform": platform.platform(),
        "source": "fresh shared evaluation of configs/m1_seg_colour_tuned.yaml"}]
    for row in compact:
        if row["selected"]:
            shared.append({key: row[key] for key in ("method", "split", *METRICS, "score_threshold", "ms_per_image", "device", "platform")})
            shared[-1]["source"] = row["run"] + " saved validation predictions (scores verified)"
    write_csv(results / "model_comparison.csv", shared)
    tuning_rows = []
    for name, folder in folders.items():
        if not (folder / "shared_trials.json").exists():
            continue
        for trial in read_json(folder / "shared_trials.json"):
            best = trial.get("best_shared")
            tuning_rows.append({"search": name, "trial": trial["iteration"], "status": trial["status"],
                "best_epoch": best["epoch"] if best else None,
                **{key: best["scores"][key] if best else None for key in METRICS},
                "checkpoint_available": (folder / "trials" / f"trial_{trial['iteration']:04d}" / "checkpoints/best_shared.pt").exists(),
                **trial["hyperparameters"]})
        target = results / "tuning" / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("best_training_config.yaml", "best_shared_hyperparameters.yaml"):
            if (folder / filename).exists():
                shutil.copy2(folder / filename, target / filename)
    write_csv(results / "tuning/yolo_trials.csv", tuning_rows)
    archive.mkdir(parents=True, exist_ok=True)
    for name, folder in folders.items():
        if folder.parent == results:
            shutil.move(str(folder), str(archive / name))
    manifest = {"archive": "archive/training", "raw_runs": sorted(folders),
        "selected": {method: {"run": run["name"], "epoch": run["best"]["epoch"],
            "mAP50": run["best"]["scores"]["mAP50"], "checkpoint": f"models/{method}/best.pt"}
            for method, run in selected.items()},
        "notes": ["No raw artifacts were deleted; training runs moved into archive/training.",
            "YOLO tuning checkpoints were not transferred; trial scores are retained but cannot be deployed.",
            "Windows run manifests list 1369 training images; augmentation metadata lists 2738.",
            "Timing hardware differs between imported ML runs and local M1 evaluation."]}
    write_json(results / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    organize()

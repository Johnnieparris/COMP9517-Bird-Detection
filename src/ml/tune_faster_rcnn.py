"""Exhaustive Faster R-CNN grid search ranked by shared validation mAP50.

python -m src.ml.tune_faster_rcnn --device 0 --name faster_rcnn_tune
"""
import argparse
import csv
import gc
from itertools import product
import json
import re

import torch
import yaml

from src.ml.train_faster_rcnn import TUNABLE_KEYS, load_config, train, validate_config
from src.ml.train_yolo import resolve_path, utc_now, write_json


def read_search(path, base):
    search = yaml.safe_load(resolve_path(path).read_text())
    if not isinstance(search, dict) or set(search) != {"trial_epochs", "space"}:
        raise ValueError("Search config needs trial_epochs and space")
    if type(search["trial_epochs"]) is not int or search["trial_epochs"] < 1:
        raise ValueError("trial_epochs must be a positive integer")
    space = search["space"]
    if not isinstance(space, dict) or not space or set(space) - TUNABLE_KEYS:
        raise ValueError(f"Grid parameters must be from {sorted(TUNABLE_KEYS)}")
    for key, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must have a nonempty list of grid values")
        for value in values:
            validate_config({**base, key: value})
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate grid values for {key}")
    keys = sorted(space)
    trials = [dict(zip(keys, values)) for values in product(*(space[key] for key in keys))]
    return search, trials


def export_best(root, base, results):
    best = max(results, key=lambda result: result["scores"]["mAP50"])
    final = {**base, **best["hyperparameters"]}
    (root / "best_training_config.yaml").write_text(yaml.safe_dump(final, sort_keys=False))
    (root / "best_shared_hyperparameters.yaml").write_text(yaml.safe_dump(best["hyperparameters"]))
    write_json(root / "best_shared_trial.json", best)
    write_json(root / "trials.json", results)
    rows = [{"trial": result["trial"], "best_epoch": result["best_epoch"], **result["scores"],
             "hyperparameters_json": json.dumps(result["hyperparameters"], sort_keys=True)} for result in results]
    with (root / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tune(config_path="configs/faster_rcnn_baseline.yaml", search_path="configs/faster_rcnn_tune.yaml",
         output_root="results", name="faster_rcnn_tune", device=None, resume=False):
    base = load_config(config_path)
    if device is not None:
        base["device"] = str(device)
    search, trials = read_search(search_path, base)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Use letters, numbers, underscores and hyphens for the tuning name")
    root = resolve_path(output_root) / name
    request = {"base": base, "search": search}
    if resume:
        if json.loads((root / "request.json").read_text()) != request:
            raise ValueError("Resume requires the same base config, device and search grid")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "request.json", request)
    (root / "configs").mkdir(exist_ok=True)
    state = {"status": "running", "total_trials": len(trials), "completed_trials": 0,
             "selection_metric": "shared validation mAP50", "updated_utc": utc_now()}
    write_json(root / "tuning.json", state)
    results = []
    try:
        for index, parameters in enumerate(trials, 1):
            trial_name = f"trial_{index:04d}"
            trial_dir = root / "trials" / trial_name
            # All candidates start from the same pretrained model and random seed.
            # Trial budget/early stopping cannot change the exported final budget.
            config = validate_config({**base, **parameters, "epochs": search["trial_epochs"], "patience": 0})
            config_file = root / "configs" / f"{trial_name}.yaml"
            print(f"Grid trial {index}/{len(trials)}: {parameters}", flush=True)
            if resume and (trial_dir / "run.json").exists():
                metadata = json.loads((trial_dir / "run.json").read_text())
                if metadata["config"] != config:
                    raise ValueError(f"Saved configuration differs for {trial_name}")
                if metadata["status"] != "completed":
                    train(resume=trial_dir)
            else:
                config_file.write_text(yaml.safe_dump(config, sort_keys=False))
                train(config_file, run_name=trial_name, output_root=root / "trials")
            metadata = json.loads((trial_dir / "run.json").read_text())
            if metadata["status"] != "completed" or metadata["completed_epochs"] != search["trial_epochs"]:
                raise RuntimeError(f"Incomplete trial: {trial_name}")
            best = json.loads((trial_dir / "best_shared.json").read_text())
            results.append({"trial": index, "run_dir": str(trial_dir), "hyperparameters": parameters,
                "best_epoch": best["epoch"], "scores": best["scores"]})
            export_best(root, base, results)
            state.update(completed_trials=len(results), updated_utc=utc_now())
            write_json(root / "tuning.json", state)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state.update(status="completed", finished_utc=utc_now())
        write_json(root / "tuning.json", state)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(root / "tuning.json", state)
        raise
    print(f"Final training configuration: {root / 'best_training_config.yaml'}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/faster_rcnn_baseline.yaml")
    parser.add_argument("--search", default="configs/faster_rcnn_tune.yaml")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--name", default="faster_rcnn_tune")
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    tune(args.config, args.search, args.output_root, args.name, args.device, args.resume)


if __name__ == "__main__":
    main()

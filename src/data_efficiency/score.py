"""Summarise a Study B YOLO training run folder (Member E)."""
import json
from pathlib import Path


def run_summary(run_dir):
    """Settings and clean-val selection facts for one results/e_runs/e_<run>/ folder."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    history = json.loads((run_dir / "history.json").read_text()) if (run_dir / "history.json").exists() else []
    best = max(history, key=lambda e: e["scores"]["mAP50"]) if history else None
    return {"run": run_dir.name.removeprefix("e_"), "status": meta.get("status", "unknown"),
            "train_images": meta.get("dataset", {}).get("train_images"),
            "epochs": meta.get("completed_epochs", len(history)),
            "best_epoch": best["epoch"] if best else None,
            "val_mAP50": best["scores"]["mAP50"] if best else None,
            "train_time_s": history[-1]["elapsed_s"] if history else None}

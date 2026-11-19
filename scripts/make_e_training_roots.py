"""Build Member E's Study B training roots and YOLO run configs.

    python scripts/make_e_training_roots.py                 # roots + configs, E = 30
    python scripts/make_e_training_roots.py --configs-only --epochs 20

Roots (data/e_training/<name>/, all splits hard-linked unless stated):
    subset_50 / subset_25 / subset_10   nested train subsets (splits.json flags the subset)
    jitter20                            every train box edge jittered by up to 20% of its size
    drop20                              20% of train boxes removed
    robust_aug                          each train image replaced, with p = 0.5, by a random
                                        distortion (type and severity uniform) from Study A
Val/test labels and images are never changed. Configs go to configs/e/<run>.yaml.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import DEFAULT_ROOT, load_split  # noqa: E402
from src.data_efficiency.label_noise import drop_boxes, jitter_boxes  # noqa: E402
from src.data_efficiency.subsets import nested_subsets  # noqa: E402
from src.robustness.distortions import DISTORTIONS, apply_distortion, rng_for  # noqa: E402
from src.robustness.roots import build_root  # noqa: E402

OUT = REPO / "data" / "e_training"
TUNED = REPO / "results/tuning/yolo_tune_v1/best_training_config.yaml"
AUG_KEYS = {"hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr", "mosaic", "mixup", "cutmix", "copy_paste", "bgr", "close_mosaic"}
STRONG_AUG = {"mosaic": 1.0, "scale": 0.9, "hsv_v": 0.6, "hsv_s": 0.9, "copy_paste": 0.3,
              "mixup": 0.1, "degrees": 10.0}
SEED = 0
FRACTIONS = {"subset_50": 0.5, "subset_25": 0.25, "subset_10": 0.1}
# run -> (data root under data/e_training or None for the original dataset, train fraction, extra settings)
RUNS = {
    "de_100": (None, 1.0, {}),
    "de_50": ("subset_50", 0.5, {}),
    "de_25": ("subset_25", 0.25, {}),
    "de_10": ("subset_10", 0.1, {}),
    "ln_jitter20": ("jitter20", 1.0, {}),
    "ln_drop20": ("drop20", 1.0, {}),
    "de_10_strong": ("subset_10", 0.1, STRONG_AUG),
    "robust_aug": ("robust_aug", 1.0, {}),
}


def run_epochs(full_epochs, fraction):
    """Matched optimiser steps, capped: subset fraction f trains min(round(E / f), 100) epochs."""
    return full_epochs if fraction >= 1 else min(int(round(full_epochs / fraction)), 100)


def official():
    return json.loads((REPO / "splits/official.json").read_text())


def write_manifest(root, train_ids=None):
    data = official()
    if train_ids is not None:
        data["splits"] = {**data["splits"], "train": sorted(train_ids)}
        data["subset_of_official"] = True
    (root / "splits.json").write_text(json.dumps(data, indent=2))


def robust_transform(image, boxes, image_id):
    rng = rng_for(SEED, image_id, "robust_aug")
    if rng.random() >= 0.5:
        return image
    name = sorted(DISTORTIONS)[int(rng.integers(len(DISTORTIONS)))]
    return apply_distortion(image, boxes, name, int(rng.integers(1, 4)), rng)


def build_roots():
    train_ids = [r.image_id for r in load_split(DEFAULT_ROOT, "train")]
    assert set(train_ids) == set(official()["splits"]["train"]), "train folder != official split"
    subsets = nested_subsets(train_ids, tuple(FRACTIONS.values()), seed=SEED)
    sizes = {r.image_id: (r.width, r.height) for r in load_split(DEFAULT_ROOT, "train")}
    plain = {"train": None, "val": None, "test": None}

    def train_only(fn):
        return lambda boxes, image_id, split: fn(boxes, image_id) if split == "train" else boxes

    jobs = {name: (plain, None, {"train_fraction": f}) for name, f in FRACTIONS.items()}
    jobs["jitter20"] = (plain, train_only(lambda b, i: jitter_boxes(
        b, 0.2, rng_for(SEED, i, "jitter20"), *sizes[i])), {"label_noise": "jitter 0.2"})
    jobs["drop20"] = (plain, train_only(lambda b, i: drop_boxes(
        b, 0.2, rng_for(SEED, i, "drop20"))), {"label_noise": "drop 0.2"})
    jobs["robust_aug"] = ({"train": robust_transform, "val": None, "test": None}, None,
                          {"distortion_augmentation": "p=0.5, uniform type and severity"})
    for name, (transforms, label_fn, info) in jobs.items():
        dst = OUT / name
        if dst.exists():
            print(f"exists, skipped: {dst}")
            continue
        build_root(DEFAULT_ROOT, dst, transforms, label_fn=label_fn, info={"seed": SEED, **info})
        write_manifest(dst, subsets.get(FRACTIONS[name]) if name in FRACTIONS else None)
        print(f"built {dst}")


def write_configs(full_epochs):
    tuned = yaml.safe_load(TUNED.read_text())
    base = {k: v for k, v in tuned.items() if k not in AUG_KEYS}
    base.update(augmentation_copies=0, seed=SEED, device="mps", workers=4, amp=False)
    base.pop("data_root", None)
    (REPO / "configs/e/base.yaml").write_text(
        "# Study B base: tuned optimiser/loss values, Ultralytics default online augmentation.\n"
        + yaml.safe_dump(base, sort_keys=False))
    for run, (root, fraction, extra) in RUNS.items():
        config = {**base, "data_root": str((OUT / root).relative_to(REPO)) if root else
                  str(DEFAULT_ROOT.relative_to(REPO)),
                  "epochs": run_epochs(full_epochs, fraction), **extra}
        (REPO / f"configs/e/{run}.yaml").write_text(
            f"# Member E Study B run {run} (train fraction {fraction}).\n"
            + yaml.safe_dump(config, sort_keys=False))
        print(f"configs/e/{run}.yaml: {config['epochs']} epochs, data {config['data_root']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30, help="full-data epoch budget E")
    parser.add_argument("--configs-only", action="store_true")
    args = parser.parse_args()
    if not args.configs_only:
        build_roots()
    write_configs(args.epochs)


if __name__ == "__main__":
    main()

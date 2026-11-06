"""Reproducible splits: everyone gets the SAME images in train/val/test.

    from data.split import official_split, save_split_manifest, load_split_manifest

    save_split_manifest(official_split(DEFAULT_ROOT), "splits/official.json")
    manifest = load_split_manifest("splits/official.json")
    val_records = load_split_from_manifest(DEFAULT_ROOT, manifest, "val")
"""
import json
import random
from pathlib import Path

from .dataset import load_split
from .validation import SPLITS


def official_split(root):
    """The split that came with the dataset. Returns {'train': [ids], ...}."""
    return {s: sorted(r.image_id for r in load_split(root, s)) for s in SPLITS}


def create_train_val_test(ids, seed=42, ratios=(0.7, 0.15, 0.15)):
    """Make a NEW random split. Same ids + same seed = same result, always."""
    if abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("ratios must add up to 1")
    ids = sorted(ids)  # sort first, so file order on your computer doesn't matter
    random.Random(seed).shuffle(ids)
    n_train = round(len(ids) * ratios[0])
    n_val = round(len(ids) * ratios[1])
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


def save_split_manifest(split, path, seed=None):
    """Save the split to a small .json file that can be committed to git."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "splits": split}, indent=2))


def load_split_manifest(path):
    """Read a saved split. Returns {'train': [ids], 'val': [...], 'test': [...]}."""
    return json.loads(Path(path).read_text())["splits"]


def load_split_from_manifest(root, manifest, split):
    """DetectionRecords for one split of a saved manifest."""
    wanted = set(manifest[split])
    records = [r for s in SPLITS for r in load_split(root, s)]
    return [r for r in records if r.image_id in wanted]
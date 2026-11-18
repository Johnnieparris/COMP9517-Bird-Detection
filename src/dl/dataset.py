"""Faster R-CNN adapter for the same exported data used by YOLO."""
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.dataset import get_image
from src.data.split import load_split_manifest, load_split_from_manifest


class BirdDetectionDataset(Dataset):
    """RGB float32 CHW images in [0, 1] and torchvision detection targets.

    YOLO class 0 (bird) becomes torchvision class 1; class 0 is background.
    No additional augmentation is applied, so both methods see the same images.
    """

    def __init__(self, root, split="train", manifest_path=None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        root = Path(root)
        manifest = load_split_manifest(manifest_path or root / "splits.json")
        self.records = load_split_from_manifest(root, manifest, split)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = torch.from_numpy(get_image(record)).permute(2, 0, 1).float().div(255)
        boxes = torch.from_numpy(record.boxes.copy())
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, record.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, record.height)
        boxes = boxes[(boxes[:, 2:] > boxes[:, :2]).all(dim=1)]
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target


def detection_collate(batch):
    """Keep variable-size images and variable-length targets as separate lists."""
    images, targets = zip(*batch)
    return list(images), list(targets)

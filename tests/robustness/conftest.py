import cv2
import numpy as np
import pytest


@pytest.fixture
def fake_root(tmp_path):
    """Tiny dataset: 2 images per split; the second of each split is a negative (empty label file)."""
    root = tmp_path / "src_root"
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            name = f"{split}_{i}"
            cv2.imwrite(str(root / "images" / split / f"{name}.jpg"),
                        rng.integers(0, 256, (60, 80, 3), dtype=np.uint8))
            text = "0 0.5 0.5 0.25 0.3\n0 0.2 0.2 0.1 0.1\n" if i == 0 else ""
            (root / "labels" / split / f"{name}.txt").write_text(text)
    (root / "dataset.yaml").write_text("path: .\nnames:\n  0: bird\n")
    return root

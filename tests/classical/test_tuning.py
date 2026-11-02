import random
import subprocess
import sys

import cv2
import numpy as np
import pytest

from src.classical.interim_data import ImageRecord
from src.classical.pipeline import SegConfig
from src.classical.run import run_split
from src.classical.tuning import detect_parallel, sample_config


def _records(root, n=4):
    records = []
    for i in range(n):
        image = np.full((128, 128, 3), 220, np.uint8)
        cv2.circle(image, (30 + 20 * i, 64), 10, (20, 20, 20), -1)
        path = root / f"im{i}.png"
        cv2.imwrite(str(path), image)
        records.append(ImageRecord(f"im{i}", path, np.zeros((0, 4))))
    return records


def test_sample_config_replaces_only_named_fields():
    cfg = sample_config(SegConfig(), {"bg_kernel": [31]}, random.Random(0))
    assert cfg.bg_kernel == 31 and cfg.morph_kernel == SegConfig().morph_kernel


def test_sample_config_is_reproducible_with_seed():
    space = {"bg_kernel": [31, 51, 75], "min_distance": [0, 10, 20, 30]}
    a = sample_config(SegConfig(), space, random.Random(7))
    b = sample_config(SegConfig(), space, random.Random(7))
    assert a == b


def test_sample_config_rejects_unknown_field():
    with pytest.raises(TypeError):
        sample_config(SegConfig(), {"bg_kernal": [31]}, random.Random(0))


def test_detect_parallel_matches_serial(tmp_path):
    records = _records(tmp_path)
    serial, _ = run_split(records, SegConfig())
    assert serial and detect_parallel(records, SegConfig(), workers=2) == serial


def test_detect_parallel_raises_on_corrupt_image(tmp_path):
    records = _records(tmp_path)
    records[1].path.write_bytes(b"not an image")
    with pytest.raises(IOError):
        detect_parallel(records, SegConfig(), workers=2)


def test_tune_script_refuses_test_split():
    result = subprocess.run([sys.executable, "scripts/tune_m1.py", "--base", "configs/m1_seg_colour.yaml",
                             "--space", "configs/search/seg_colour.yaml", "--split", "test",
                             "--trials", "1", "--name", "x"], capture_output=True, text=True)
    assert result.returncode != 0 and "invalid choice" in result.stderr

from pathlib import Path

import pytest
import yaml

from src.ml.config import M2Config, load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
M2_CONFIGS = sorted(CONFIG_DIR.glob("m2_*.yaml"))


def test_there_are_m2_configs():
    assert M2_CONFIGS, "no configs/m2_*.yaml found"


@pytest.mark.parametrize("path", M2_CONFIGS, ids=lambda p: p.stem)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert cfg.feature and cfg.clf and cfg.proposals


@pytest.mark.parametrize("path", M2_CONFIGS, ids=lambda p: p.stem)
def test_every_shipped_config_can_hold_a_dense_image(path):
    """The densest image in the dataset has 130 birds."""
    assert load_config(path).max_detections >= 130


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"method": "m2", "nonsense": 1}))
    with pytest.raises(TypeError):
        load_config(path)


def test_method_1_config_is_rejected(tmp_path):
    path = tmp_path / "m1.yaml"
    path.write_text(yaml.safe_dump({"method": "blob", "threshold": 0.03}))
    with pytest.raises(ValueError, match="not a Method 2"):
        load_config(path)


@pytest.mark.parametrize("field,value", [
    ("feature", "sift"), ("clf", "mlp"), ("proposals", "magic"),
    ("max_detections", 50), ("patch", 4),
])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValueError):
        M2Config(**{field: value})


def test_m1_proposals_need_a_config_path():
    with pytest.raises(ValueError, match="m1_config"):
        M2Config(proposals="m1")


def test_variant_name_describes_the_run():
    cfg = M2Config(feature="concat", clf="gb", hn_rounds=2, proposals="window")
    assert cfg.variant == "concat_gb_hn2_window"

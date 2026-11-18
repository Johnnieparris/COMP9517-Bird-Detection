import sys
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


def test_sweep_covers_every_shipped_config():
    """Re-running the sweep must not silently drop rows from the table.

    scripts/sweep_m2.py rebuilds results/m2_comparison.csv from whatever it ran,
    so a config missing from FULL disappears from the comparison.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from sweep_m2 import FULL

    assert sorted(FULL) == sorted(p.stem for p in M2_CONFIGS)


@pytest.mark.parametrize("field,value", [
    ("C", 1.0), ("n_estimators", 50), ("patch", 48), ("seed", 7),
    ("nms_iou", 0.6), ("m1_config", "configs/other.yaml"),
])
def test_hidden_settings_change_the_variant_name(field, value):
    """Two runs differing in any setting must not share a label."""
    base = M2Config(feature="concat", clf="gb", proposals="window")
    other = M2Config(feature="concat", clf="gb", proposals="window", **{field: value})
    assert base.variant != other.variant


def test_default_config_has_no_hash_suffix():
    """The shipped ablation keeps short readable names."""
    assert M2Config().variant == "hog_svm_hn1_window"
    assert M2Config().changed_from_default() == []


def test_variant_hash_is_stable_and_order_independent():
    first = M2Config(patch=48, seed=3).variant
    second = M2Config(seed=3, patch=48).variant
    assert first == second and "_cfg" in first


def test_window_percentiles_list_and_tuple_are_equal():
    """A YAML list must not look like a change against the tuple default."""
    assert M2Config(window_percentiles=[25, 50, 75]).changed_from_default() == []

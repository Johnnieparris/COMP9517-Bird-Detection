"""Method 2 configuration. load_config() is the public entry point.

Mirrors src/classical/pipeline.load_config so both methods are driven the same way:

    from src.ml.config import load_config
    cfg = load_config("configs/m2_concat_gb.yaml")
"""
import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

import yaml

FEATURES = ("hog", "lbp", "colour", "concat")
CLASSIFIERS = ("svm", "rf", "gb")
PROPOSALS = ("window", "blob", "m1")

# Settings spelled out in the readable part of the variant name. Everything else
# goes through the digest below, so no two configs can share a label.
NAMED_IN_VARIANT = ("feature", "clf", "hn_rounds", "proposals", "box_regression",
                    "boxreg_min_iou", "proposal_neg_per_image", "calibrate", "fraction")


@dataclass
class M2Config:
    """Every knob Method 2 has.

    Defaults are the Week 8 starting point, not the tuned best. See
    docs/notes/m2-week9.md for what each setting was worth.
    """

    # Candidate boxes. Window sides are percentiles of sqrt(w*h) over the
    # training boxes; the caps stop large images exploding the candidate count.
    proposals: str = "window"
    window_percentiles: Sequence[float] = (25, 50, 75)
    window_step_frac: float = 0.5
    window_max_per_scale: int = 2500
    blob_threshold: float = 0.02
    blob_max_side: int = 1024
    m1_config: Optional[str] = None

    # Patch description. Every candidate is resized to patch x patch first.
    patch: int = 32
    feature: str = "hog"

    # Classifier. C is LinearSVC only, n_estimators is RandomForest only.
    clf: str = "svm"
    C: float = 0.01
    n_estimators: int = 200

    # Platt scaling, off by default: it triples training time and still does not
    # make a fixed 0.5 cut-off meaningful, because class_weight="balanced" below
    # deliberately cancels the class imbalance. mAP only needs the ranking, so it
    # is unaffected either way.
    calibrate: bool = False
    calibration_folds: int = 3

    # Training. proposal_neg_per_image draws negatives from the detector's own
    # candidates, so the classifier trains on the box distribution it is scored
    # on; random crops alone are nothing like the ~2,000 proposals per image it
    # meets at inference. Uniform sampling, not hardest-first, keeps the prior.
    neg_per_image: int = 10
    proposal_neg_per_image: int = 30
    jitter: float = 0.1
    hn_rounds: int = 1
    hn_images: int = 300
    hn_per_image: int = 20
    fraction: float = 1.0
    seed: int = 0

    # Box regression. A candidate sitting on a bird at the wrong size counts as a
    # miss, so a linear correction fixes the size without generating a single
    # extra candidate. Only candidates already this close are used for training.
    box_regression: bool = False
    boxreg_min_iou: float = 0.3
    boxreg_alpha: float = 1000.0

    # Post-processing. max_detections must stay >= 130, the densest image.
    nms_iou: float = 0.4
    max_detections: int = 300

    def __post_init__(self):
        if self.feature not in FEATURES:
            raise ValueError(f"feature must be one of {FEATURES}, got {self.feature!r}")
        if self.clf not in CLASSIFIERS:
            raise ValueError(f"clf must be one of {CLASSIFIERS}, got {self.clf!r}")
        if self.proposals not in PROPOSALS:
            raise ValueError(f"proposals must be one of {PROPOSALS}, got {self.proposals!r}")
        if self.proposals == "m1" and not self.m1_config:
            raise ValueError("proposals: m1 needs m1_config: <path to a Method 1 yaml>")
        if self.max_detections < 130:
            raise ValueError("max_detections must be >= 130: the densest image has 130 birds")
        if self.patch < 8:
            raise ValueError(f"patch must be at least 8 px, got {self.patch}")

    def changed_from_default(self):
        """Settings that differ from the defaults and are not named in `variant`.

        Returns a sorted list of "field=value" strings. Sequences are compared as
        tuples so a YAML list and a tuple of the same numbers count as equal.
        """
        def normalise(value):
            return tuple(value) if isinstance(value, (list, tuple)) else value

        defaults = M2Config()
        out = []
        for f in dataclasses.fields(self):
            if f.name in NAMED_IN_VARIANT:
                continue
            mine = normalise(getattr(self, f.name))
            if mine != normalise(getattr(defaults, f.name)):
                out.append(f"{f.name}={mine!r}")
        return sorted(out)

    @property
    def variant(self):
        """Short name used in file names and the comparison table.

        Must separate every row of the ablation, or two runs collide under one
        label in results/m2_comparison.csv. The readable part covers the settings
        the ablation varies; the trailing `_cfg<hash>` covers everything else, so
        changing C, patch, n_estimators, m1_config, seed and so on cannot
        silently reuse another run's label. The hash is absent when those are all
        at their defaults, which keeps the common names short.
        """
        name = f"{self.feature}_{self.clf}_hn{self.hn_rounds}_{self.proposals}"
        if self.box_regression:
            name += f"_boxreg{self.boxreg_min_iou:g}"
        if not self.proposal_neg_per_image:
            name += "_nopropneg"
        if self.calibrate:
            name += "_cal"
        if self.fraction < 1.0:
            name += f"_frac{self.fraction:g}"
        changed = self.changed_from_default()
        if changed:
            name += "_cfg" + hashlib.sha1("|".join(changed).encode()).hexdigest()[:6]
        return name


def load_config(path):
    """Read an M2Config from YAML. An unknown key raises TypeError."""
    with open(path) as f:
        values = yaml.safe_load(f) or {}
    method = values.pop("method", "m2")
    if method != "m2":
        raise ValueError(f"{path} is a {method!r} config, not a Method 2 one")
    if "window_percentiles" in values:
        values["window_percentiles"] = tuple(values["window_percentiles"])
    return M2Config(**values)

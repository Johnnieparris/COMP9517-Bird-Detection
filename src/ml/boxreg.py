"""Bounding-box regression, as in R-CNN (Girshick et al., CVPR 2014).

Method 2 only accepts or rejects a candidate, so a candidate sitting on a bird at
the wrong size counts as a miss. On val with Method 1's candidates, 56% of birds
have a candidate overlapping at IoU 0.3 but only 30% reach IoU 0.5, and 82% have
one whose centre lands inside the bird. The centres are right and the sizes are
wrong, which is exactly what box regression was designed for.

The model is deliberately linear: ridge regression on the same patch features the
classifier uses, applied once rather than iteratively, and only to candidates
already overlapping a bird enough for a local correction to make sense.

Targets use the standard scale-invariant parameterisation:

    tx = (Gx - Px) / Pw        tw = log(Gw / Pw)
    ty = (Gy - Py) / Ph        th = log(Gh / Ph)

where P is the proposal and G the ground truth, both as centre/width/height.
Offsets relative to the proposal's own size keep the problem identical for a
12 px bird and a 200 px one.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# R-CNN uses a deliberately large penalty: the feature vector is high dimensional
# against the number of usable training pairs.
DEFAULT_ALPHA = 1000.0

# log(Gw/Pw) blows up when a proposal is tiny against the bird, so the scale
# targets are clipped to stop a handful of pairs dominating the fit.
MAX_LOG_SCALE = 1.0


def to_centre_form(boxes):
    """(N, 4) x1y1x2y2 -> centre x, centre y, width, height."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack([boxes[:, 0] + w / 2, boxes[:, 1] + h / 2, w, h], axis=1)


def to_corner_form(centres):
    """centre x, centre y, width, height -> x1y1x2y2."""
    centres = np.asarray(centres, dtype=float).reshape(-1, 4)
    cx, cy, w, h = centres[:, 0], centres[:, 1], centres[:, 2], centres[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def regression_targets(proposals, truths):
    """(N, 4) targets tx, ty, tw, th taking each proposal onto its ground-truth box."""
    p = to_centre_form(proposals)
    g = to_centre_form(truths)
    pw = np.maximum(p[:, 2], 1e-6)
    ph = np.maximum(p[:, 3], 1e-6)
    tx = (g[:, 0] - p[:, 0]) / pw
    ty = (g[:, 1] - p[:, 1]) / ph
    tw = np.clip(np.log(np.maximum(g[:, 2], 1e-6) / pw), -MAX_LOG_SCALE, MAX_LOG_SCALE)
    th = np.clip(np.log(np.maximum(g[:, 3], 1e-6) / ph), -MAX_LOG_SCALE, MAX_LOG_SCALE)
    return np.stack([tx, ty, tw, th], axis=1)


def apply_deltas(proposals, deltas):
    """Move each proposal by its predicted offsets. Returns x1y1x2y2."""
    p = to_centre_form(proposals)
    d = np.asarray(deltas, dtype=float).reshape(-1, 4)
    tw = np.clip(d[:, 2], -MAX_LOG_SCALE, MAX_LOG_SCALE)
    th = np.clip(d[:, 3], -MAX_LOG_SCALE, MAX_LOG_SCALE)
    return to_corner_form(np.stack([
        p[:, 0] + d[:, 0] * p[:, 2],
        p[:, 1] + d[:, 1] * p[:, 3],
        p[:, 2] * np.exp(tw),
        p[:, 3] * np.exp(th),
    ], axis=1))


class BoxRegressor:
    """Ridge regression from patch features to a box correction."""

    def __init__(self, alpha=DEFAULT_ALPHA):
        self.alpha = alpha
        self.model = None
        self.n_train = 0

    def fit(self, features, proposals, truths):
        """features (N, D); proposals and truths (N, 4) matched row for row.

        Fewer than ten usable pairs leaves the model unfitted, and refine() then
        passes boxes through untouched.
        """
        features = np.asarray(features, dtype=np.float32)
        if len(features) < 10:
            self.model = None
            return self
        targets = regression_targets(proposals, truths)
        keep = np.isfinite(targets).all(axis=1)
        if keep.sum() < 10:
            self.model = None
            return self
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model.fit(features[keep], targets[keep])
        self.n_train = int(keep.sum())
        return self

    def refine(self, features, proposals):
        """Corrected boxes. Falls back to the originals if the model did not fit."""
        proposals = np.asarray(proposals, dtype=np.float32).reshape(-1, 4)
        if self.model is None or not len(proposals):
            return proposals
        deltas = self.model.predict(np.asarray(features, dtype=np.float32))
        refined = apply_deltas(proposals, deltas)
        return np.where(np.isfinite(refined), refined, proposals).astype(np.float32)

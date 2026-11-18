"""Method 2: handcrafted features with a trained classifier.

    candidate boxes -> patch descriptor -> classifier score -> NMS

Training uses ground-truth crops as positives and background crops as negatives,
then rounds of hard negative mining: the classifier is run over training images
and its most confident false positives are added to the negative set before
retraining (Dalal & Triggs, CVPR 2005).

Where Method 1 sets its thresholds by hand, the accept/reject boundary here is
learned, which is what the two methods are meant to isolate.

    from src.ml.config import load_config
    from src.ml.detector import BirdDetector

    det = BirdDetector(load_config("configs/m2_concat_gb.yaml")).fit(train_records)
    preds, timing = run_and_time(val_records, det.predict_fn())

Box utilities come from src/classical/boxes.py (member B) rather than being
reimplemented, so both methods suppress and compare boxes identically.
"""
import time

import cv2
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.classical.boxes import iou_matrix, nms
from src.ml.boxreg import BoxRegressor
from src.ml.features import extract_many
from src.ml.proposals import Proposals, training_box_sizes

# A crop counts as background if it overlaps every bird by less than this.
NEGATIVE_IOU = 0.3


def crop(image, box, size):
    """Resize the image region inside `box` to size x size. None if it is too small."""
    h, w = image.shape[:2]
    x1, y1 = int(max(0, np.floor(box[0]))), int(max(0, np.floor(box[1])))
    x2, y2 = int(min(w, np.ceil(box[2]))), int(min(h, np.ceil(box[3])))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return cv2.resize(image[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


class BirdDetector:
    """Patch classifier turned into a detector. Fit on training records, then detect."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        self.model = None
        self.proposals = None
        self.sizes_wh = None
        self.box_regressor = None
        self.train_seconds = None

        # Hook for member E's robustness study: f(image_rgb) -> image_rgb.
        self.image_preproc = None

    # ------------------------------------------------------------- helpers
    def _image(self, record):
        from data.dataset import get_image

        image = get_image(record)
        return self.image_preproc(image) if self.image_preproc else image

    def _build_model(self):
        """The chosen classifier behind a scaler, optionally Platt-calibrated."""
        cfg = self.cfg
        if cfg.clf == "svm":
            clf = LinearSVC(C=cfg.C, class_weight="balanced", dual=False, max_iter=5000)
        elif cfg.clf == "rf":
            clf = RandomForestClassifier(cfg.n_estimators, class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=cfg.seed)
        else:
            clf = HistGradientBoostingClassifier(random_state=cfg.seed)
        if cfg.calibrate:
            clf = CalibratedClassifierCV(clf, method="sigmoid", cv=cfg.calibration_folds)
        return make_pipeline(StandardScaler(), clf)

    def _features(self, patches):
        return extract_many(patches, self.cfg.feature, self.cfg.patch)

    def _scores(self, X):
        """Scores in 0..1, whichever classifier is in use.

        Every method has to publish scores on the same scale for the shared
        evaluator. Tree models give a probability directly; the SVM margin is
        squashed with a logistic, which is monotonic and so leaves mAP alone.
        """
        model = self.model
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return 1.0 / (1.0 + np.exp(-model.decision_function(X)))

    def _jittered(self, box):
        """Randomly nudge a box, for positive augmentation."""
        w, h = box[2] - box[0], box[3] - box[1]
        shift = self.rng.uniform(-self.cfg.jitter, self.cfg.jitter, 4) * np.array([w, h, w, h])
        return box + shift

    def _background_boxes(self, image, gt_boxes, n):
        """`n` random boxes that overlap every bird by less than NEGATIVE_IOU.

        Several tries per box, since a crowded image can run out of clear room.
        """
        h, w = image.shape[:2]
        out = []
        for _ in range(n * 5):
            bw, bh = self.sizes_wh[self.rng.randint(len(self.sizes_wh))]
            if w - bw < 1 or h - bh < 1:
                continue
            x, y = self.rng.uniform(0, w - bw), self.rng.uniform(0, h - bh)
            box = np.array([[x, y, x + bw, y + bh]], np.float32)
            if len(gt_boxes) == 0 or iou_matrix(box, gt_boxes).max() < NEGATIVE_IOU:
                out.append(box[0])
            if len(out) >= n:
                break
        return out

    def _sample_patches(self, records, augment=True):
        """Positive crops (birds) and negative crops (background) for one pass.

        Each bird also contributes a horizontal flip and a jittered copy.
        """
        positives, negatives = [], []
        for record in records:
            image = self._image(record)
            for box in record.boxes:
                patch = crop(image, box, self.cfg.patch)
                if patch is None:
                    continue
                positives.append(patch)
                if augment:
                    positives.append(patch[:, ::-1])
                    jittered = crop(image, self._jittered(box), self.cfg.patch)
                    if jittered is not None:
                        positives.append(jittered)
            for box in self._background_boxes(image, record.boxes, self.cfg.neg_per_image):
                patch = crop(image, box, self.cfg.patch)
                if patch is not None:
                    negatives.append(patch)
        return positives, negatives

    def _proposal_negatives(self, records, per_image):
        """Background crops sampled uniformly from the detector's own candidates.

        This is what makes the published score usable: the classifier trains on
        the same box distribution it is scored on. Unlike mining, the sample is
        uniform, so it keeps the real ratio of background to birds rather than
        only the hardest cases.
        """
        out = []
        for record in records:
            image = self._image(record)
            boxes = self.proposals(image)
            if not len(boxes):
                continue
            overlap = (iou_matrix(boxes, record.boxes).max(axis=1)
                       if len(record.boxes) else np.zeros(len(boxes)))
            background = np.where(overlap < NEGATIVE_IOU)[0]
            if not len(background):
                continue
            take = self.rng.choice(background, min(per_image, len(background)), replace=False)
            for i in take:
                patch = crop(image, boxes[i], self.cfg.patch)
                if patch is not None:
                    out.append(patch)
        return out

    def _fit_box_regressor(self, records, log):
        """Learn a size correction from candidates that already overlap a bird.

        Only pairs above boxreg_min_iou are used: a linear model is a local
        correction, so asking it to rescue a candidate that barely touches the
        bird would corrupt the fit for the ones it can help.
        """
        patches, proposals, truths = [], [], []
        for record in records:
            if not len(record.boxes):
                continue
            image = self._image(record)
            boxes = self.proposals(image)
            if not len(boxes):
                continue
            overlap = iou_matrix(boxes, record.boxes)
            best = overlap.argmax(axis=1)
            close = np.where(overlap.max(axis=1) >= self.cfg.boxreg_min_iou)[0]
            for i in close:
                patch = crop(image, boxes[i], self.cfg.patch)
                if patch is not None:
                    patches.append(patch)
                    proposals.append(boxes[i])
                    truths.append(record.boxes[best[i]])
        regressor = BoxRegressor(self.cfg.boxreg_alpha)
        if patches:
            regressor.fit(self._features(patches), np.array(proposals), np.array(truths))
        log(f"[fit] box regressor trained on {regressor.n_train} candidate/bird pairs")
        return regressor

    def _mine(self, records, per_image):
        """Confident false positives: high score, no real bird underneath."""
        hard = []
        for record in records:
            image = self._image(record)
            boxes = self.proposals(image)
            if not len(boxes):
                continue
            scores = self.score_boxes(image, boxes)
            overlap = (iou_matrix(boxes, record.boxes).max(axis=1)
                       if len(record.boxes) else np.zeros(len(boxes)))
            wrong = np.where((scores > 0.5) & (overlap < NEGATIVE_IOU))[0]
            for i in wrong[np.argsort(-scores[wrong])][:per_image]:
                patch = crop(image, boxes[i], self.cfg.patch)
                if patch is not None:
                    hard.append(patch)
        return hard

    # ------------------------------------------------------------ training
    def fit(self, records, log=print):
        """Train on TRAINING records only. Returns self.

        Sample patches, fit, then alternate mining and refitting for hn_rounds.
        A fraction below 1.0 drops training images, for the data-efficiency study.
        """
        cfg = self.cfg
        records = list(records)
        if cfg.fraction < 1.0:
            keep = max(1, int(round(len(records) * cfg.fraction)))
            idx = self.rng.choice(len(records), keep, replace=False)
            records = [records[i] for i in sorted(idx)]

        started = time.perf_counter()
        self.sizes_wh = training_box_sizes(records)
        self.proposals = Proposals(cfg, self.sizes_wh)

        positives, negatives = self._sample_patches(records)
        if cfg.proposal_neg_per_image:
            sampled = self._proposal_negatives(records, cfg.proposal_neg_per_image)
            log(f"[fit] +{len(sampled)} negatives sampled from the candidate boxes")
            negatives += sampled
        log(f"[fit] {len(positives)} positive and {len(negatives)} negative patches")
        Xp, Xn = self._features(positives), self._features(negatives)
        self.model = self._build_model()
        self._refit(Xp, Xn)

        for round_i in range(cfg.hn_rounds):
            subset = [records[i] for i in sorted(self.rng.choice(
                len(records), min(cfg.hn_images, len(records)), replace=False))]
            hard = self._mine(subset, cfg.hn_per_image)
            log(f"[fit] hard negative round {round_i + 1}: +{len(hard)}")
            if hard:
                Xn = np.vstack([Xn, self._features(hard)])
                self._refit(Xp, Xn)

        if cfg.box_regression:
            self.box_regressor = self._fit_box_regressor(records, log)

        self.train_seconds = time.perf_counter() - started
        return self

    def _refit(self, Xp, Xn):
        X = np.vstack([Xp, Xn])
        y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
        self.model.fit(X, y)

    # ----------------------------------------------------------- inference
    def _describe(self, image, boxes):
        """Features for every croppable box. Returns (features, their indices)."""
        patches, keep = [], []
        for i, box in enumerate(boxes):
            patch = crop(image, box, self.cfg.patch)
            if patch is not None:
                patches.append(patch)
                keep.append(i)
        if not patches:
            return None, []
        return self._features(patches), keep

    def score_boxes(self, image, boxes):
        """Score every candidate box. Boxes too small to crop score 0."""
        features, keep = self._describe(image, boxes)
        scores = np.zeros(len(boxes), np.float32)
        if features is not None:
            scores[keep] = self._scores(features)
        return scores

    def detect(self, image_rgb):
        """(N, 5) detections of x1, y1, x2, y2, score, best first.

        One feature pass serves both the classifier and the box regressor. A
        correction can collapse a box, so those are dropped rather than scored.
        """
        if self.model is None:
            raise RuntimeError("fit() must be called before detect()")
        image = self.image_preproc(image_rgb) if self.image_preproc else image_rgb
        boxes = self.proposals(image)
        if not len(boxes):
            return np.zeros((0, 5), np.float32)

        features, keep = self._describe(image, boxes)
        scores = np.zeros(len(boxes), np.float32)
        if features is not None:
            scores[keep] = self._scores(features)
            if self.box_regressor is not None:
                boxes = boxes.copy()
                boxes[keep] = self.box_regressor.refine(features, boxes[keep])
                h, w = image.shape[:2]
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
                valid = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
                boxes, scores = boxes[valid], scores[valid]

        if not len(boxes):
            return np.zeros((0, 5), np.float32)
        dets = np.column_stack([boxes, scores]).astype(np.float32)
        return nms(dets, self.cfg.nms_iou)[:self.cfg.max_detections]

    def predict_fn(self):
        """Callable for the shared evaluation.run_and_time()."""
        return self.detect

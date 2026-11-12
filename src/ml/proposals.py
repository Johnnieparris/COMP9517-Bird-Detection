"""Candidate boxes for Method 2.

Method 2 scores candidate regions rather than searching the image itself, so the
fraction of real birds covered by some candidate is a hard ceiling on its recall,
and therefore on its mAP. Three sources are compared:

    window  multi-scale sliding window, sides taken from the training box sizes
    blob    multi-scale Difference of Gaussians on the grey image and its inverse
    m1      Method 1's detector (member B), via configs/m1_seg_proposals.yaml

All of them take an RGB uint8 image (what data.dataset.get_image returns) and
return an (N, 4) float32 array of x1, y1, x2, y2 in original-image pixels.
"""
import cv2
import numpy as np

PROPOSAL_MODES = ("window", "blob", "m1")

# A DoG response at scale sigma means a blob of radius sigma*sqrt(2), so a square
# box around it has side 2*sqrt(2)*sigma.
SIGMA_TO_SIDE = 2.0 * np.sqrt(2.0)


def box_sides(boxes_wh):
    """sqrt(w*h) for an (N, 2) array of box widths and heights."""
    boxes_wh = np.asarray(boxes_wh, dtype=float).reshape(-1, 2)
    return np.sqrt(boxes_wh[:, 0] * boxes_wh[:, 1])


def training_box_sizes(records):
    """(N, 2) widths and heights of every ground-truth box in `records`."""
    sizes = [r.boxes[:, 2:] - r.boxes[:, :2] for r in records if len(r.boxes)]
    if not sizes:
        raise ValueError("no ground-truth boxes found; cannot size the proposals")
    return np.concatenate(sizes)


class Proposals:
    """Callable candidate generator: proposals(image_rgb) -> (N, 4) boxes.

    `boxes_wh` comes from the TRAINING split only, so no validation or test
    statistic ever reaches the proposal sizes.
    """

    def __init__(self, cfg, boxes_wh):
        self.cfg = cfg
        self.sides = box_sides(boxes_wh)
        self._m1 = None
        if cfg.proposals == "m1":
            from src.classical.pipeline import load_config as load_m1_config
            self._m1 = load_m1_config(cfg.m1_config)

    def __call__(self, image_rgb):
        if self.cfg.proposals == "window":
            return self._windows(image_rgb)
        if self.cfg.proposals == "blob":
            return self._blobs(image_rgb)
        return self._from_m1(image_rgb)

    # ------------------------------------------------------------- window
    def _windows(self, image):
        """Square windows at a few sizes, stepped at a fraction of the side.

        The inner loop doubles the step until the grid fits under
        window_max_per_scale, so a very large image cannot blow up the count.
        """
        h, w = image.shape[:2]
        out = []
        for side in np.percentile(self.sides, self.cfg.window_percentiles):
            side = int(max(8, round(side)))
            step = max(1, int(round(side * self.cfg.window_step_frac)))
            xs = np.arange(0, max(w - side, 1), step)
            ys = np.arange(0, max(h - side, 1), step)
            while len(xs) * len(ys) > self.cfg.window_max_per_scale:
                step *= 2
                xs = np.arange(0, max(w - side, 1), step)
                ys = np.arange(0, max(h - side, 1), step)
            gx, gy = np.meshgrid(xs, ys)
            gx, gy = gx.ravel(), gy.ravel()
            out.append(np.stack([gx, gy, gx + side, gy + side], axis=1))
        return self._finish(np.concatenate(out), w, h)

    # --------------------------------------------------------------- blob
    def _blobs(self, image):
        """Difference-of-Gaussians peaks, on the image and its inverse.

        Birds can be brighter or darker than the sand or water behind them, so
        both polarities are searched and the results pooled. Each peak gets three
        box sizes, since the DoG scale only approximates the bird's extent.
        """
        from skimage.feature import blob_dog

        h, w = image.shape[:2]
        scale = min(1.0, self.cfg.blob_max_side / max(h, w))
        small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32) / 255.0

        p5, p95 = np.percentile(self.sides, [5, 95])
        s_min = max(1.0, p5 * scale / SIGMA_TO_SIDE)
        s_max = min(30.0, max(s_min + 1.0, p95 * scale / SIGMA_TO_SIDE))

        out = []
        for channel in (g, 1.0 - g):
            for y, x, sigma in blob_dog(channel, min_sigma=s_min, max_sigma=s_max,
                                        sigma_ratio=1.5, threshold=self.cfg.blob_threshold):
                side = SIGMA_TO_SIDE * sigma / scale
                cx, cy = x / scale, y / scale
                for k in (0.8, 1.0, 1.3):
                    r = side * k / 2
                    out.append([cx - r, cy - r, cx + r, cy + r])
        if not out:
            return np.zeros((0, 4), np.float32)
        return self._finish(np.array(out, np.float32), w, h)

    # ----------------------------------------------------------- method 1
    def _from_m1(self, image):
        """Member B's Method 1 detector used as a proposal source.

        configs/m1_seg_proposals.yaml is the high-recall variant B tuned for this
        hand-off (docs/notes/m1-week8.md). Their detect() expects BGR while our
        images are RGB, so the channel flip happens here and nowhere else.
        """
        from src.classical.pipeline import detect

        dets = detect(image[:, :, ::-1], self._m1)
        boxes = np.asarray(dets, np.float32).reshape(-1, 5)[:, :4]
        h, w = image.shape[:2]
        return self._finish(boxes, w, h)

    # ------------------------------------------------------------- shared
    @staticmethod
    def _finish(boxes, w, h):
        """Clip to the image and drop anything with no area left."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if not len(boxes):
            return np.zeros((0, 4), np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        keep = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
        return boxes[keep]

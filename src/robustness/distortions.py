"""Deterministic image distortions for the robustness study (Member E, Study A).

Five distortion families, three severities each (spec table). Every random draw
comes from rng_for(seed, image_id, tag), which hashes with zlib.crc32, so the
same image always gets the same distortion on every machine and Python run.

    rng = rng_for(0, "beach_bird_1661", "noise_s2")
    out = apply_distortion(image_rgb, boxes_xyxy, "noise", 2, rng)
"""
import zlib

import cv2
import numpy as np

# name -> (severity 1, severity 2, severity 3)
DISTORTIONS = {
    "noise": (10, 25, 50),             # Gaussian noise sigma, 0-255 pixel units
    "blur": (1, 2, 4),                 # Gaussian blur sigma, pixels
    "lowlight": (0.6, 0.4, 0.2),       # contrast a = brightness b
    "jpeg": (30, 15, 5),               # JPEG quality
    "occlusion": (0.25, 0.5, 0.75),    # share of each GT box area covered
}
OCCLUDER_GREY = 128


def variant_name(name, severity):
    return f"{name}_s{severity}"


def rng_for(seed, image_id, tag):
    """A numpy Generator seeded from (seed, image_id, tag), stable across runs."""
    return np.random.default_rng(zlib.crc32(f"{seed}|{image_id}|{tag}".encode()))


def _noise(img, sigma, rng, boxes):
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def _blur(img, sigma, rng, boxes):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def _lowlight(img, a, rng, boxes):
    # I' = a (I - mu) + b mu with a = b: lower contrast about the grey mean, then darken.
    mu = float(img.mean())
    out = a * (img.astype(np.float32) - mu) + a * mu
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _jpeg(img, quality, rng, boxes):
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def _occlusion(img, frac, rng, boxes):
    """Grey rectangle with the box's aspect ratio and frac of its area, at a random spot inside it."""
    out = img.copy()
    h_img, w_img = img.shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(w_img, int(round(x2))), min(h_img, int(round(y2)))
        bw, bh = x2 - x1, y2 - y1
        if bw < 1 or bh < 1:
            continue
        ow = max(1, min(bw, int(round(bw * np.sqrt(frac)))))
        oh = max(1, min(bh, int(round(bh * np.sqrt(frac)))))
        ox = x1 + int(rng.integers(0, bw - ow + 1))
        oy = y1 + int(rng.integers(0, bh - oh + 1))
        out[oy:oy + oh, ox:ox + ow] = OCCLUDER_GREY
    return out


_FUNCS = {"noise": _noise, "blur": _blur, "lowlight": _lowlight, "jpeg": _jpeg, "occlusion": _occlusion}


def apply_distortion(image_rgb, boxes, name, severity, rng):
    """Distort a uint8 RGB image. severity is 1, 2 or 3. Returns a new uint8 array."""
    if name not in DISTORTIONS:
        raise ValueError(f"unknown distortion {name!r}; choose from {sorted(DISTORTIONS)}")
    if severity not in (1, 2, 3):
        raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
    out = _FUNCS[name](np.ascontiguousarray(image_rgb), DISTORTIONS[name][severity - 1], rng, boxes)
    assert out.shape == image_rgb.shape and out.dtype == np.uint8
    return out

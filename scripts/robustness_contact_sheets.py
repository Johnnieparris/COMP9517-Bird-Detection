"""Save one contact sheet per distortion type: a fixed test image, clean + severities 1-3.

    python scripts/robustness_contact_sheets.py [--image-id beach_bird_1661]

Writes results/robustness/contact_<type>.jpg (GT boxes in green) for a visual check
and for the report/video.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import DEFAULT_ROOT, get_image, load_split  # noqa: E402
from src.robustness.distortions import DISTORTIONS, variant_name  # noqa: E402
from src.robustness.roots import DISTORTED_DIR  # noqa: E402

OUT = REPO / "results" / "robustness"
PANEL_WIDTH = 480


def panel(record, label):
    img = get_image(record)[:, :, ::-1].copy()
    for x1, y1, x2, y2 in record.boxes.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    img = cv2.resize(img, (PANEL_WIDTH, int(img.shape[0] * PANEL_WIDTH / img.shape[1])))
    cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", help="default: the test image with the most birds under 30")
    args = parser.parse_args()
    clean = {r.image_id: r for r in load_split(DEFAULT_ROOT, "test")}
    image_id = args.image_id or max((r for r in clean.values() if len(r.boxes) <= 30),
                                    key=lambda r: len(r.boxes)).image_id
    OUT.mkdir(parents=True, exist_ok=True)
    for name in DISTORTIONS:
        panels = [panel(clean[image_id], "clean")]
        for s in (1, 2, 3):
            recs = {r.image_id: r for r in load_split(DISTORTED_DIR / variant_name(name, s), "test")}
            panels.append(panel(recs[image_id], f"{name} s{s}"))
        sheet = np.hstack(panels)
        path = OUT / f"contact_{name}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{path.relative_to(REPO)} ({image_id})")


if __name__ == "__main__":
    main()

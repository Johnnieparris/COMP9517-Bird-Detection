"""Save overlay pictures of one method's predictions against the real boxes.

    GREEN = TP  correct detection (found a real bird)
    RED   = FP  false alarm (detected a bird that is not there)
    BLUE  = FN  missed bird (a real bird nobody found)

Uses the same matching and settings as src/evaluation.py, so the pictures agree
with the scores in your report. Use for report figures and the video demo.

    python scripts/make_overlays.py m1                       # 12 random test images
    python scripts/make_overlays.py m1 --pick worst --n 8    # images with the most mistakes
    python scripts/make_overlays.py m1 --pick best --n 8     # cleanest images
    python scripts/make_overlays.py m1 --ids test_0003,test_0010
    python scripts/make_overlays.py m1 --pick all            # every image (for a video)

The method must have published predictions first (see README, Step 10).
Pictures are saved in overlays/<method>_<split>/ (add overlays/ to .gitignore).
"""
import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from data.dataset import DEFAULT_ROOT, load_split  # noqa: E402
from data.split import load_split_from_manifest, load_split_manifest  # noqa: E402
from evaluation import IOU_THR, SCORE_THR, check_prediction_file, classify_detections  # noqa: E402
from predictions import PRED_DIR, load_predictions, prediction_path  # noqa: E402

COLORS = {"tp": (0, 200, 0), "fp": (230, 0, 0), "fn": (0, 90, 255)}


# ---------- drawing ----------

def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)  # newer Pillow
    except TypeError:
        return ImageFont.load_default()


def _text_size(width):
    return max(14, width // 55)


def strip_height(width):
    """Height in pixels of the header strip above the picture."""
    return int(_text_size(width) * 2.2)


def draw_overlay(image, found, title):
    """image: PIL RGB picture. found: dict from classify_detections().
    Returns a NEW picture with a header strip (title on the left, counts on the right)."""
    w, h = image.size
    size, strip = _text_size(w), strip_height(w)
    canvas = Image.new("RGB", (w, h + strip), (30, 30, 30))
    canvas.paste(image, (0, strip))
    draw = ImageDraw.Draw(canvas)

    line = max(2, w // 400)
    for kind in ("fn", "tp", "fp"):  # red is drawn last, so it is never hidden
        for x1, y1, x2, y2 in found[kind][:, :4]:
            draw.rectangle([x1, y1 + strip, x2, y2 + strip], outline=COLORS[kind], width=line)

    font = _font(size)
    top = strip / 2 - size / 2
    x = w - size / 2                      # legend, filled from the right: FN, FP, TP
    for kind, label in (("fn", "FN"), ("fp", "FP"), ("tp", "TP")):
        text = f"{label} {len(found[kind])}"
        tw = draw.textlength(text, font=font)
        draw.text((x - tw, top), text, fill="white", font=font)
        sq = size * 0.8
        left = x - tw - size * 0.3 - sq
        draw.rectangle([left, strip / 2 - sq / 2, left + sq, strip / 2 + sq / 2], fill=COLORS[kind])
        x = left - size * 0.9

    room = x - size                       # title gets the space that is left
    while len(title) > 3 and draw.textlength(title, font=font) > room:
        title = title[:-4] + "..."
    draw.text((size / 2, top), title, fill="white", font=font)
    return canvas


def _scaled(found, s):
    return {k: np.concatenate([v[:, :4] * s, v[:, 4:]], axis=1) for k, v in found.items()}


# ---------- choosing images ----------

def pick_images(rows, pick, n, seed):
    """rows: list of (record, found). Returns the ones to draw."""
    if pick == "all":
        return rows
    useful = [r for r in rows if any(len(r[1][k]) for k in ("tp", "fp", "fn"))]  # not blank
    if pick == "random":
        return random.Random(seed).sample(useful, min(n, len(useful)))
    if pick == "worst":   # most false alarms + misses
        return sorted(useful, key=lambda r: -(len(r[1]["fp"]) + len(r[1]["fn"])))[:n]
    if pick == "best":    # fewest mistakes, then most correct detections
        return sorted(useful, key=lambda r: (len(r[1]["fp"]) + len(r[1]["fn"]), -len(r[1]["tp"])))[:n]
    raise ValueError(pick)


# ---------- main ----------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Save TP/FP/FN overlay pictures for one method")
    ap.add_argument("method", help="method name used in save_predictions (e.g. m1)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pick", default="random", choices=["random", "worst", "best", "all"])
    ap.add_argument("--n", type=int, default=12, help="how many pictures (default 12)")
    ap.add_argument("--seed", type=int, default=42, help="for --pick random")
    ap.add_argument("--ids", help="comma-separated image ids (overrides --pick)")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR)
    ap.add_argument("--iou-thr", type=float, default=IOU_THR)
    ap.add_argument("--max-width", type=int, default=0, help="shrink wider pictures (0 = keep size)")
    ap.add_argument("--format", default="png", choices=["png", "jpg"])
    ap.add_argument("--out", help="output folder (default overlays/<method>_<split>)")
    ap.add_argument("--root", help="dataset folder")
    ap.add_argument("--predictions", help="predictions folder")
    ap.add_argument("--manifest", help="split file (default splits/official.json if it exists)")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else DEFAULT_ROOT
    folder = Path(a.predictions) if a.predictions else PRED_DIR
    manifest = Path(a.manifest) if a.manifest else REPO / "splits" / "official.json"
    out = Path(a.out) if a.out else REPO / "overlays" / f"{a.method}_{a.split}"

    # ground truth
    try:
        records = (load_split_from_manifest(root, load_split_manifest(manifest), a.split)
                   if manifest.exists() else load_split(root, a.split))
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[X] Cannot load the {a.split} images: {e}")
        return 1

    # predictions (checked first, so a bad file gives a clear message)
    path = prediction_path(a.method, a.split, folder)
    if not path.exists():
        print(f"[X] {path} not found. Has '{a.method}' published its {a.split} predictions?")
        return 1
    problems = check_prediction_file(path, records)
    if problems:
        print(f"[X] {path.name} has {len(problems)} problem(s):")
        for line in problems[:5]:
            print(f"    - {line}")
        return 1
    preds = load_predictions(a.method, a.split, folder)

    rows = [(r, classify_detections(preds.get(r.image_id, np.zeros((0, 5))), r.boxes,
                                    a.iou_thr, a.score_thr)) for r in records]
    totals = {k: sum(len(f[k]) for _, f in rows) for k in ("tp", "fp", "fn")}
    print(f"{a.method} ({a.split}), score >= {a.score_thr}, IoU >= {a.iou_thr}: "
          f"all {len(rows)} images -> TP {totals['tp']}, FP {totals['fp']}, FN {totals['fn']}")

    if a.ids:
        wanted = [i.strip() for i in a.ids.split(",") if i.strip()]
        by_id = {r.image_id: (r, f) for r, f in rows}
        unknown = [i for i in wanted if i not in by_id]
        if unknown:
            print(f"[X] Not in the {a.split} split: {', '.join(unknown)}")
            return 1
        chosen = [by_id[i] for i in wanted]
    else:
        chosen = pick_images(rows, a.pick, a.n, a.seed)

    out.mkdir(parents=True, exist_ok=True)
    for record, found in chosen:
        image = Image.open(record.image_path).convert("RGB")
        if a.max_width and image.width > a.max_width:
            s = a.max_width / image.width
            image = image.resize((a.max_width, round(image.height * s)), Image.LANCZOS)
            found = _scaled(found, s)
        title = f"{record.image_id} | {a.method} ({a.split})"
        picture = draw_overlay(image, found, title)
        file = out / f"{record.image_id}.{a.format}"
        picture.save(file, quality=95) if a.format == "jpg" else picture.save(file)
        print(f"  saved {file.name}: TP {len(found['tp'])}, FP {len(found['fp'])}, FN {len(found['fn'])}")
    print(f"\nSaved {len(chosen)} pictures in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
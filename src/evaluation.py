"""Shared evaluation. Every method must be scored with THIS file.

Predictions: a dict {image_id: array of shape (N, 5)}, one row per detected bird:
    x1, y1, x2, y2, score        (pixels, same corners as record.boxes)
An image with no detections can be left out or given an empty array.

    from evaluation import evaluate, print_results

    records = load_split_from_manifest(DEFAULT_ROOT, manifest, "val")
    preds, timing = run_and_time(records, my_method)   # my_method(image) -> (N, 5) array
    print_results(evaluate(records, preds, timing), "Method 1")

To check and score all TEST predictions in the predictions/ folder at once
(files for other splits, like val, are ignored):

    python src/evaluation.py

Keep the scores of ALL detections (even low ones, e.g. YOLO conf=0.001), so
the precision-recall curve is complete. SCORE_THR below is only used for
precision / recall / F1 / counting, not for AP.
"""
import csv
import math
import time
from pathlib import Path

import numpy as np

# Team settings. Do not change these for one method only.
IOU_THR = 0.5     # a detection is correct if it overlaps a real box by >= 50%
SCORE_THR = 0.5   # detections below this score are ignored for precision/recall/F1
EVAL_SPLITS = ("test",)  # python src/evaluation.py only scores these splits


# ---------- building blocks ----------

def box_iou(a, b):
    """Overlap (0 to 1) between every box in a and every box in b.
    a: (N, 4), b: (M, 4), corners x1 y1 x2 y2. Returns an (N, M) array."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _greedy_match(preds, gts, iou_thr):
    """preds: (N, 5) already sorted by score. For each detection, the index of the
    real box it matched, or -1 if it matched none. A real box is used only once."""
    match = np.full(len(preds), -1, dtype=int)
    if len(preds) and len(gts):
        iou = box_iou(preds[:, :4], gts)
        used = np.zeros(len(gts), dtype=bool)
        for i in range(len(preds)):
            row = np.where(used, -1.0, iou[i])  # ignore real boxes already matched
            j = row.argmax()
            if row[j] >= iou_thr:
                match[i] = j
                used[j] = True
    return match


def _match_image(preds, gts, iou_thr):
    """Match one image's detections to its real boxes.
    Highest score goes first; each real box can only be used once.
    Returns (scores, is_correct) for the detections, sorted by score."""
    preds = np.asarray(preds, dtype=float).reshape(-1, 5)
    preds = preds[np.argsort(-preds[:, 4], kind="stable")]
    gts = np.asarray(gts, dtype=float).reshape(-1, 4)
    return preds[:, 4], _greedy_match(preds, gts, iou_thr) >= 0


def classify_detections(preds, gts, iou_thr=IOU_THR, score_thr=SCORE_THR):
    """Sort one image's boxes into the three groups used for overlay pictures.
    Uses exactly the same matching as evaluate(), so the counts agree.

    Returns a dict of arrays:
      'tp'  detections that found a real bird        (N, 5): x1 y1 x2 y2 score
      'fp'  detections with no real bird (false alarms)
      'fn'  real birds nobody found, shape (M, 4): x1 y1 x2 y2
    Only detections with score >= score_thr are used."""
    preds = np.asarray(preds, dtype=float).reshape(-1, 5)
    preds = preds[preds[:, 4] >= score_thr]
    preds = preds[np.argsort(-preds[:, 4], kind="stable")]
    gts = np.asarray(gts, dtype=float).reshape(-1, 4)
    match = _greedy_match(preds, gts, iou_thr)
    hit = match >= 0
    missed = np.ones(len(gts), dtype=bool)
    missed[match[hit]] = False
    return {"tp": preds[hit], "fp": preds[~hit], "fn": gts[missed]}


def _collect(records, predictions, iou_thr):
    """Match all images. Returns (scores, is_correct, number of real boxes)."""
    ids = {r.image_id for r in records}
    unknown = sorted(set(predictions) - ids)
    if unknown:
        raise ValueError(f"{len(unknown)} predicted image ids are not in these records "
                         f"(wrong split?): {', '.join(unknown[:5])}")
    all_scores, all_correct, n_gt = [], [], 0
    for r in records:
        p = np.asarray(predictions.get(r.image_id, np.zeros((0, 5))), dtype=float)
        if p.size and (p.ndim != 2 or p.shape[1] != 5):
            raise ValueError(f"{r.image_id}: predictions must have shape (N, 5): "
                             f"x1, y1, x2, y2, score. Got {p.shape}")
        s, c = _match_image(p, r.boxes, iou_thr)
        all_scores.append(s)
        all_correct.append(c)
        n_gt += len(r.boxes)
    return np.concatenate(all_scores), np.concatenate(all_correct), n_gt


def _average_precision(scores, correct, n_gt):
    """Area under the precision-recall curve (all-point, as in Pascal VOC)."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    correct = correct[np.argsort(-scores, kind="stable")]
    tp = np.cumsum(correct)
    fp = np.cumsum(~correct)
    recall = np.concatenate([[0.0], tp / n_gt, [1.0]])
    precision = np.concatenate([[1.0], tp / (tp + fp), [0.0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # curve only goes down
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


# ---------- the functions teammates call ----------

def average_precision(records, predictions, iou_thr=IOU_THR):
    """AP at one overlap threshold (AP50 when iou_thr=0.5). Higher is better, max 1."""
    scores, correct, n_gt = _collect(records, predictions, iou_thr)
    return _average_precision(scores, correct, n_gt)


def map_50_95(records, predictions):
    """Average of AP over overlap thresholds 0.50, 0.55, ... 0.95 (stricter than AP50)."""
    return float(np.mean([average_precision(records, predictions, t)
                          for t in np.arange(0.5, 0.96, 0.05)]))


def precision_recall_f1(records, predictions, iou_thr=IOU_THR, score_thr=SCORE_THR):
    """Counts and scores for detections with score >= score_thr.
    precision = how many detections were real birds
    recall    = how many real birds were found"""
    scores, correct, n_gt = _collect(records, predictions, iou_thr)
    keep = scores >= score_thr
    tp = int(correct[keep].sum())
    fp = int((~correct[keep]).sum())
    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def counting_error(records, predictions, score_thr=SCORE_THR):
    """How wrong is the bird COUNT per image?
    mae  = average mistake in birds per image (0 is perfect)
    bias = average (predicted - real): above 0 means too many birds counted"""
    errors = []
    for r in records:
        p = np.asarray(predictions.get(r.image_id, np.zeros((0, 5))), dtype=float)
        n_pred = int((p.reshape(-1, 5)[:, 4] >= score_thr).sum()) if p.size else 0
        errors.append(n_pred - len(r.boxes))
    errors = np.array(errors, dtype=float)
    return {"mae": float(np.abs(errors).mean()), "bias": float(errors.mean())}


def evaluate(records, predictions, timing=None, iou_thr=IOU_THR, score_thr=SCORE_THR):
    """Everything in one dict. Use this for your report tables.
    timing = the dict from run_and_time() (optional)."""
    out = {"n_images": len(records),
           "n_birds": int(sum(len(r.boxes) for r in records)),
           "mAP50": average_precision(records, predictions, iou_thr),
           "mAP50_95": map_50_95(records, predictions)}
    out.update(precision_recall_f1(records, predictions, iou_thr, score_thr))
    c = counting_error(records, predictions, score_thr)
    out["count_mae"], out["count_bias"] = c["mae"], c["bias"]
    if timing:
        out.update(timing)
    return out


def print_results(results, name=""):
    """Print the dict from evaluate() in a fixed layout."""
    r = results
    print(f"=== {name or 'Results'} ({r['n_images']} images, {r['n_birds']} birds) ===")
    print(f"mAP@0.5:      {r['mAP50']:.3f}")
    print(f"mAP@0.5:0.95: {r['mAP50_95']:.3f}")
    print(f"Precision:    {r['precision']:.3f}   Recall: {r['recall']:.3f}   F1: {r['f1']:.3f}")
    print(f"Found {r['tp']}, false alarms {r['fp']}, missed {r['fn']}")
    print(f"Count MAE:    {r['count_mae']:.2f} birds/image (bias {r['count_bias']:+.2f})")
    if "ms_per_image" in r:
        print(f"Speed:        {r['ms_per_image']:.1f} ms/image ({r['fps']:.1f} FPS)")


# ---------- timing ----------

def run_and_time(records, predict_fn, warmup=3):
    """Run your method on every record and time it.

    predict_fn(image) -> (N, 5) array: x1, y1, x2, y2, score.   image = RGB numpy array.
    Only predict_fn is timed. Loading the image is not.
    The first calls are slow (models start up), so predict_fn is run `warmup`
    times on the first image first, and those runs are not counted.
    Use the SAME computer for every method, and write down which one in your report.
    Returns (predictions, timing).
    """
    from data.dataset import get_image  # needs src/ on the path (it is, if you imported this file)

    if not records:
        raise ValueError("records is empty")
    preds, times = {}, []
    first = get_image(records[0])
    for _ in range(warmup):
        predict_fn(first)
    for r in records:
        image = get_image(r)
        t0 = time.perf_counter()
        out = predict_fn(image)
        times.append(time.perf_counter() - t0)
        preds[r.image_id] = np.asarray(out, dtype=float).reshape(-1, 5)
    times = np.array(times)
    timing = {"ms_per_image": float(times.mean() * 1000),
              "median_ms": float(np.median(times) * 1000),
              "fps": float(1 / times.mean()),
              "total_s": float(times.sum())}
    return preds, timing


def from_ultralytics(results):
    """Turn YOLO predictions into our format.

        results = model.predict(source=[str(r.image_path) for r in records], conf=0.001)
        preds = from_ultralytics(results)
    """
    from pathlib import Path
    preds = {}
    for res in results:
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        preds[Path(res.path).stem] = np.column_stack([xyxy, conf]).reshape(-1, 5)
    return preds


def yolo_predict_fn(model, conf=0.001):
    """Wrap a YOLO model so it works with run_and_time():

        model = YOLO("runs/detect/train/weights/best.pt")
        preds, timing = run_and_time(records, yolo_predict_fn(model))

    Our images are RGB, but YOLO expects numpy images as BGR, so they are flipped here.
    """
    def predict(image):
        res = model.predict(image[:, :, ::-1], conf=conf, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        score = res.boxes.conf.cpu().numpy()
        return np.column_stack([xyxy, score]).reshape(-1, 5)
    return predict


# ---------- check and score everything in predictions/ ----------

def check_prediction_file(path, records):
    """Look inside one predictions csv. Returns a list of problems (empty = fine).
    records = the real images of that split (used to check the image ids)."""
    from predictions import COLUMNS

    method, split = Path(path).stem.rsplit("_", 1)
    ids = {r.image_id for r in records}
    problems = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            return [f"columns should be {COLUMNS}, found {reader.fieldnames}"]
        for n, row in enumerate(reader, start=2):  # line 1 is the header
            where = f"line {n}"
            if row["method"] != method:
                problems.append(f"{where}: method '{row['method']}' does not match the file name")
            if row["split"] != split:
                problems.append(f"{where}: split '{row['split']}' does not match the file name")
            if row["image_id"] not in ids:
                problems.append(f"{where}: image '{row['image_id']}' is not in the {split} split")
            try:
                x1, y1, x2, y2, score = (float(row[c]) for c in ("x1", "y1", "x2", "y2", "score"))
            except (ValueError, TypeError):
                problems.append(f"{where}: box or score is not a number")
                continue
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)):
                problems.append(f"{where}: nan or inf value")
            elif x2 <= x1 or y2 <= y1:
                problems.append(f"{where}: box has no size (x2 must be bigger than x1, y2 than y1)")
    return problems


def print_table(results):
    """One table per split, best method first."""
    if not results:
        print("\nNothing was evaluated.")
        return
    head = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1", "Count MAE"]
    keys = ["mAP50", "mAP50_95", "precision", "recall", "f1", "count_mae"]
    for split in ("train", "val", "test"):
        rows = sorted((r for r in results if r["split"] == split), key=lambda r: -r["mAP50"])
        if not rows:
            continue
        w = max(6, *(len(r["method"]) for r in rows))
        print(f"\nSplit: {split} ({rows[0]['n_images']} images, {rows[0]['n_birds']} birds)")
        print("Method".ljust(w) + "".join(f"  {h:>12}" for h in head))
        for r in rows:
            print(r["method"].ljust(w) + "".join(f"  {r[k]:>12.3f}" for k in keys))


def evaluate_folder(root=None, folder=None, manifest_path=None):
    """Check every TEST file in the predictions folder, then score the ones that are fine.
    Files for other splits (val, train) are ignored: the final scores use test only.

    root          dataset folder      (default: data/Beach Bird Detection Dataset v1.0)
    folder        predictions folder  (default: predictions/)
    manifest_path split file          (default: splits/official.json, if it exists;
                                       otherwise the dataset's own train/val/test folders)
    Returns a list with one results dict per scored file (the dict from evaluate(),
    plus 'method' and 'split'). Files with problems are reported and skipped.
    """
    from data.dataset import DEFAULT_ROOT, load_split
    from data.split import load_split_from_manifest, load_split_manifest
    from predictions import PRED_DIR, list_published, load_predictions, prediction_path

    root = Path(DEFAULT_ROOT if root is None else root)
    folder = Path(PRED_DIR if folder is None else folder)
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parents[1] / "splits" / "official.json"
    manifest_path = Path(manifest_path)

    if not folder.is_dir():
        print(f"No predictions folder yet ({folder}). Nothing to evaluate.")
        return []
    published = list_published(folder)
    ignored = [f"{m}_{sp}" for m, sp in published if sp not in EVAL_SPLITS]
    published = [(m, sp) for m, sp in published if sp in EVAL_SPLITS]
    if not published:
        extra = f" ({len(ignored)} file(s) for other splits were ignored)" if ignored else ""
        print(f"No {'/'.join(EVAL_SPLITS)} prediction files in {folder}{extra}. Nothing to evaluate.")
        return []

    manifest = None
    if manifest_path.exists():
        manifest = load_split_manifest(manifest_path)
        print(f"Splits from: {manifest_path}")
    else:
        print("No split file found, using the dataset's own train/val/test folders.")
    print(f"Predictions: {folder}")
    if ignored:
        print(f"Ignored (not {'/'.join(EVAL_SPLITS)}): {', '.join(ignored)}")
    print()

    records_by_split, failed_splits, results = {}, set(), []
    for method, split in published:
        label = f"{method} ({split})"
        if split in failed_splits:
            print(f"[X] {label}: skipped (the {split} images could not be loaded)")
            continue
        if split not in records_by_split:
            try:
                records_by_split[split] = (load_split_from_manifest(root, manifest, split)
                                           if manifest else load_split(root, split))
            except (FileNotFoundError, ValueError, KeyError) as e:
                failed_splits.add(split)
                print(f"[X] {label}: cannot load the {split} images: {e}")
                continue
        records = records_by_split[split]

        problems = check_prediction_file(prediction_path(method, split, folder), records)
        if problems:
            print(f"[X] {label}: {len(problems)} problem{'s' if len(problems) != 1 else ''}, NOT evaluated")
            for line in problems[:5]:
                print(f"    - {line}")
            if len(problems) > 5:
                print(f"    ... and {len(problems) - 5} more")
            continue

        res = evaluate(records, load_predictions(method, split, folder))
        res.update(method=method, split=split)
        results.append(res)
        print(f"[OK] {label}: file OK")

    print_table(results)
    return results


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Check and score everything in predictions/")
    ap.add_argument("--root", help="dataset folder (default: data/Beach Bird Detection Dataset v1.0)")
    ap.add_argument("--predictions", help="predictions folder (default: predictions/)")
    ap.add_argument("--manifest", help="split file (default: splits/official.json if it exists)")
    a = ap.parse_args(argv)
    evaluate_folder(a.root, a.predictions, a.manifest)


if __name__ == "__main__":
    main()
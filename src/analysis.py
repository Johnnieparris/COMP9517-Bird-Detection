"""Analyse every published method and save the results in analysis/.

    python src/analysis.py

It reads predictions/<method>_<split>.csv (README Step 10) and the real boxes,
then writes these files into analysis/:

    summary.csv       one row per method and split: mAP, precision, recall, F1, count error
    pr_curves.csv     points of each precision-recall curve
    density_bins.csv  results by how many birds are in the image
    size_bins.csv     how many real birds were found, by bird size
    per_image.csv     real vs predicted bird count for every image
    info.json         the settings used, and any prediction files that were skipped

Then open notebooks/results_plots.ipynb to see the graphs.

Score cut-off for precision / recall / F1 / counts (this keeps the test set clean):
  - val rows:  the cut-off with the best F1 on val ("self").
  - test rows: the SAME cut-off, taken from the method's val predictions ("val").
  - no val predictions for that method: the default 0.5 ("default").
mAP does not use a cut-off. Nothing is ever tuned on test.

Every run rewrites the files, so you can run it again whenever predictions change.
analysis/ is generated: add it to .gitignore.
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluation import (IOU_THR, SCORE_THR, average_precision, best_f1_threshold,
                        check_prediction_file, classify_detections, gt_found_mask,
                        load_records, map_50_95, pr_curve, precision_recall_f1)
from predictions import PRED_DIR, list_published, load_predictions, prediction_path

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
ANALYSIS_SPLITS = ("val", "test")
MAX_CURVE_POINTS = 300  # curves are thinned to this many points, so the file stays small

# (label, fewest birds, most birds) per image. Same groups as the EDA table.
DENSITY_BINS = [("0", 0, 0), ("1-10", 1, 10), ("11-50", 11, 50), ("51-100", 51, 100), (">100", 101, math.inf)]

SUMMARY_COLS = ["method", "split", "n_images", "n_birds", "mAP50", "mAP50_95", "thr", "thr_source",
                "precision", "recall", "f1", "tp", "fp", "fn", "count_mae", "count_rmse", "count_bias"]
CURVE_COLS = ["method", "split", "recall", "precision", "score"]
DENSITY_COLS = ["method", "split", "bin", "n_images", "n_birds", "mAP50", "precision", "recall",
                "f1", "tp", "fp", "fn", "fp_per_image", "count_mae"]
SIZE_COLS = ["method", "split", "size_bin", "label", "n_birds", "found", "recall"]
IMAGE_COLS = ["method", "split", "image_id", "n_gt", "n_pred", "tp", "fp", "fn"]


# ---------- small helpers ----------

def _write_csv(path, columns, rows):
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            return "" if math.isnan(v) else str(round(float(v), 6))
        return str(v)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in rows:
            w.writerow([fmt(row.get(c)) for c in columns])


def _ratio(a, b):
    return a / b if b else None


def _f1(precision, recall):
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _per_image(records, preds, thr):
    """TP / FP / FN counts for every image at one score cut-off,
    plus which real boxes were found (used for the size breakdown)."""
    rows, founds = [], []
    for r in records:
        p = preds.get(r.image_id, np.zeros((0, 5)))
        f = classify_detections(p, r.boxes, IOU_THR, thr)
        rows.append({"image_id": r.image_id, "n_gt": len(r.boxes),
                     "n_pred": len(f["tp"]) + len(f["fp"]),
                     "tp": len(f["tp"]), "fp": len(f["fp"]), "fn": len(f["fn"])})
        founds.append(gt_found_mask(p, r.boxes, IOU_THR, thr))
    return rows, founds


def _size_groups(records):
    """Split the real birds of a split into small / medium / large (thirds, by width
    as a share of the image width). Same groups for every method.
    Returns (edges, labels)."""
    rel = np.concatenate([(r.boxes[:, 2] - r.boxes[:, 0]) / r.width for r in records])
    if len(rel) < 3:
        return [], [("all", "all birds")]
    q1, q2 = np.quantile(rel, [1 / 3, 2 / 3])
    return [q1, q2], [("small", f"small (<{q1:.1%} of width)"),
                      ("medium", f"medium ({q1:.1%}-{q2:.1%})"),
                      ("large", f"large (>{q2:.1%})")]


# ---------- the function teammates run ----------

def analyse_all(root=None, folder=None, manifest_path=None, out=None, splits=ANALYSIS_SPLITS):
    """Analyse every published method and write the files into analysis/.
    Returns the summary rows (a list of dicts), or None if there was nothing to analyse."""
    folder = Path(PRED_DIR if folder is None else folder)
    out = Path(ANALYSIS_DIR if out is None else out)

    if not folder.is_dir():
        print(f"No predictions folder yet ({folder}). Nothing to analyse.")
        return None
    published = [(m, s) for m, s in list_published(folder) if s in splits]
    if not published:
        print(f"No {'/'.join(splits)} prediction files in {folder}. Nothing to analyse.")
        return None

    # 1. load and check every file
    records, loaded, skipped = {}, {}, []
    for method, split in published:
        label = f"{method} ({split})"
        if split not in records:
            try:
                records[split] = load_records(root, split, manifest_path)
            except (FileNotFoundError, ValueError, KeyError) as e:
                records[split] = None
                print(f"[X] Cannot load the {split} images: {e}")
        if records[split] is None:
            skipped.append({"file": prediction_path(method, split, folder).name,
                            "reason": f"the {split} images could not be loaded"})
            continue
        problems = check_prediction_file(prediction_path(method, split, folder), records[split])
        if problems:
            print(f"[X] {label}: {len(problems)} problem{'s' if len(problems) != 1 else ''}, skipped "
                  f"(run python src/evaluation.py for details)")
            skipped.append({"file": prediction_path(method, split, folder).name,
                            "reason": f"{len(problems)} problems, first: {problems[0]}"})
            continue
        loaded[(method, split)] = load_predictions(method, split, folder)

    if not loaded:
        print("Nothing could be analysed.")
        return None

    # 2. the score cut-off comes from val (never from test)
    val_best = {m: best_f1_threshold(records["val"], p) for (m, s), p in loaded.items() if s == "val"}

    def pick_threshold(method, split):
        best = val_best.get(method)
        if best is None:
            return SCORE_THR, "default"
        return best["thr"], "self" if split == "val" else "val"

    # 3. analyse
    summary, curves, density, sizes, images = [], [], [], [], []
    size_groups = {s: _size_groups(records[s]) for s in {s for _, s in loaded} if len(records[s])}

    for method, split in sorted(loaded):
        recs, preds = records[split], loaded[(method, split)]
        thr, source = pick_threshold(method, split)
        prf = precision_recall_f1(recs, preds, IOU_THR, thr)
        per_img, founds = _per_image(recs, preds, thr)
        n_gt = np.array([r["n_gt"] for r in per_img])
        n_pred = np.array([r["n_pred"] for r in per_img])
        errors = (n_pred - n_gt).astype(float)

        summary.append({"method": method, "split": split, "n_images": len(recs), "n_birds": int(n_gt.sum()),
                        "mAP50": average_precision(recs, preds, IOU_THR), "mAP50_95": map_50_95(recs, preds),
                        "thr": thr, "thr_source": source, "precision": prf["precision"],
                        "recall": prf["recall"], "f1": prf["f1"], "tp": prf["tp"], "fp": prf["fp"],
                        "fn": prf["fn"], "count_mae": float(np.abs(errors).mean()),
                        "count_rmse": float(np.sqrt((errors ** 2).mean())), "count_bias": float(errors.mean())})

        images += [{"method": method, "split": split, **r} for r in per_img]

        # precision-recall curve, thinned
        c = pr_curve(recs, preds, IOU_THR)
        if len(c["score"]):
            keep = np.unique(np.linspace(0, len(c["score"]) - 1, MAX_CURVE_POINTS).astype(int))
            curves += [{"method": method, "split": split, "recall": c["recall"][i],
                        "precision": c["precision"][i], "score": c["score"][i]} for i in keep]

        # by how many birds are in the image
        for label, lo, hi in DENSITY_BINS:
            idx = np.flatnonzero((n_gt >= lo) & (n_gt <= hi))
            sub = [recs[i] for i in idx]
            tp, fp, fn = (sum(per_img[i][k] for i in idx) for k in ("tp", "fp", "fn"))
            birds = int(n_gt[idx].sum())
            precision, recall = _ratio(tp, tp + fp), _ratio(tp, birds)
            density.append({
                "method": method, "split": split, "bin": label, "n_images": len(idx), "n_birds": birds,
                "mAP50": (average_precision(sub, {r.image_id: preds[r.image_id] for r in sub
                                                  if r.image_id in preds}, IOU_THR) if birds else None),
                "precision": precision, "recall": recall, "f1": _f1(precision, recall),
                "tp": tp, "fp": fp, "fn": fn, "fp_per_image": _ratio(fp, len(idx)),
                "count_mae": float(np.abs(errors[idx]).mean()) if len(idx) else None})

        # by bird size
        if split in size_groups:
            edges, names = size_groups[split]
            rel = np.concatenate([(r.boxes[:, 2] - r.boxes[:, 0]) / r.width for r in recs])
            found = np.concatenate(founds)
            which = np.digitize(rel, edges) if edges else np.zeros(len(rel), dtype=int)
            for k, (name, label) in enumerate(names):
                n, hit = int((which == k).sum()), int(found[which == k].sum())
                sizes.append({"method": method, "split": split, "size_bin": name, "label": label,
                              "n_birds": n, "found": hit, "recall": _ratio(hit, n)})

        print(f"[OK] {method} ({split}): mAP@0.5 {summary[-1]['mAP50']:.3f}, "
              f"mAP@0.5:0.95 {summary[-1]['mAP50_95']:.3f}, F1 {prf['f1']:.3f} "
              f"at score >= {thr:.2f} ({ {'self': 'best on val', 'val': 'from val'}.get(source, 'default, no val file') })")

    # 4. save
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "summary.csv", SUMMARY_COLS, summary)
    _write_csv(out / "pr_curves.csv", CURVE_COLS, curves)
    _write_csv(out / "density_bins.csv", DENSITY_COLS, density)
    _write_csv(out / "size_bins.csv", SIZE_COLS, sizes)
    _write_csv(out / "per_image.csv", IMAGE_COLS, images)
    info = {"created": datetime.now().isoformat(timespec="seconds"), "iou_thr": IOU_THR,
            "default_score_thr": SCORE_THR, "splits": list(splits),
            "analysed": [{"method": m, "split": s} for m, s in sorted(loaded)], "skipped": skipped}
    (out / "info.json").write_text(json.dumps(info, indent=2))

    print(f"\nSaved 5 csv files and info.json in {out}")
    print("Next: open notebooks/results_plots.ipynb")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Analyse all published predictions and save the results in analysis/")
    ap.add_argument("--root", help="dataset folder (default: data/Beach Bird Detection Dataset v1.0)")
    ap.add_argument("--predictions", help="predictions folder (default: predictions/)")
    ap.add_argument("--manifest", help="split file (default: splits/official.json if it exists)")
    ap.add_argument("--out", help="output folder (default: analysis/)")
    ap.add_argument("--splits", default=",".join(ANALYSIS_SPLITS), help="splits to analyse (default: val,test)")
    a = ap.parse_args(argv)
    result = analyse_all(a.root, a.predictions, a.manifest, a.out, tuple(a.splits.split(",")))
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
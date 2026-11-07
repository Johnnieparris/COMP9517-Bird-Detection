"""Test evaluation.py with FAKE data. Your real data/ and predictions/ are not touched.

    python scripts/TEST_evaluation.py
    python scripts/TEST_evaluation.py --keep     # keep the fake files so you can open them

What it does:
  1. Makes a tiny fake dataset (train/val/test) in a temporary folder.
  2. Makes fake predictions from 4 pretend methods, with known quality
     (evaluation.py scores TEST files only, so perfect and noisy also publish
     val files, which must be ignored):
       perfect  exact copies of the real boxes      -> should score 1.0
       noisy    slightly shifted boxes, some missed, a few false alarms
       random   boxes in random places              -> should score near 0
       empty    finds nothing                        -> should score 0
  3. Adds two BROKEN files, which evaluation.py must report and skip.
  4. Runs evaluation.py on it all and checks the answers.
"""
import argparse
import contextlib
import io
import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from data.dataset import load_split  # noqa: E402
from data.split import official_split, save_split_manifest  # noqa: E402
from evaluation import evaluate_folder  # noqa: E402
from predictions import save_predictions  # noqa: E402

W, H = 200, 150                                   # fake image size
SPLIT_SIZES = {"train": 10, "val": 30, "test": 20}


# ---------- 1. fake dataset ----------

def make_dataset(root, rng):
    """Images are plain colours. Birds sit in separate cells of a 4x3 grid."""
    cells = [(cx, cy) for cx in range(4) for cy in range(3)]  # each cell is 50x50 pixels
    for split, n_images in SPLIT_SIZES.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n_images):
            name = f"{split}_{i:04d}"
            Image.new("RGB", (W, H), (90, 140, 200)).save(root / "images" / split / f"{name}.jpg")
            lines = []
            for cx, cy in rng.sample(cells, rng.choice([0, 1, 1, 2, 3, 4])):
                w, h = rng.randint(20, 35), rng.randint(20, 35)
                x1, y1 = cx * 50 + rng.randint(5, 12), cy * 50 + rng.randint(5, 12)
                lines.append(f"0 {(x1 + w / 2) / W:.6f} {(y1 + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


# ---------- 2. fake methods ----------

def perfect(rec, rng):
    return [[*b, 0.99] for b in rec.boxes]


def noisy(rec, rng):
    out = []
    for b in rec.boxes:
        if rng.random() < 0.15:                    # missed bird
            continue
        out.append([v + rng.gauss(0, 2) for v in b] + [rng.uniform(0.6, 0.95)])
    if rng.random() < 0.3:                         # a false alarm
        x, y = rng.uniform(0, W - 30), rng.uniform(0, H - 30)
        out.append([x, y, x + 25, y + 25, rng.uniform(0.1, 0.5)])
    return out


def random_boxes(rec, rng):
    out = []
    for _ in range(2):
        x, y = rng.uniform(0, W - 40), rng.uniform(0, H - 40)
        out.append([x, y, x + rng.uniform(15, 40), y + rng.uniform(15, 40), rng.random()])
    return out


def empty(rec, rng):
    return []


METHODS = {  # method name -> (function, splits it publishes)
    "perfect": (perfect, ["val", "test"]),
    "noisy": (noisy, ["val", "test"]),
    "random": (random_boxes, ["test"]),
    "empty": (empty, ["test"]),
}


def make_predictions(root, folder, rng):
    for method, (fn, splits) in METHODS.items():
        for split in splits:
            recs = load_split(root, split)
            preds = {r.image_id: np.array(fn(r, rng), dtype=float).reshape(-1, 5) for r in recs}
            save_predictions(method, split, recs, preds, folder=folder)


# ---------- 3. broken files ----------

HEADER = "method,split,image_id,x1,y1,x2,y2,score\n"


def make_broken_files(folder):
    (folder / "broken_test.csv").write_text(HEADER + "\n".join([
        "broken,test,not_a_real_image,10,10,50,50,0.9",  # image does not exist
        "broken,test,test_0000,50,50,10,10,0.9",         # box has no size
        "broken,test,test_0001,abc,10,50,50,0.9",        # not a number
        "broken,test,test_0002,10,10,50,50,nan",         # nan
        "other,test,test_0003,10,10,50,50,0.5",          # wrong method name
    ]) + "\n")
    (folder / "wrongcols_test.csv").write_text("a,b,c\n1,2,3\n")  # wrong columns
    # a broken VAL file: it must be ignored completely, not even checked
    (folder / "brokenval_val.csv").write_text("a,b,c\n1,2,3\n")


# ---------- 4. run and check ----------

failures = 0


def check(condition, message):
    global failures
    print(("  [OK] " if condition else "  [X]  ") + message)
    failures += 0 if condition else 1


def run(**kwargs):
    """Run evaluate_folder, show what it prints, and also return the text."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        results = evaluate_folder(**kwargs)
    text = buffer.getvalue()
    print(text)
    return results, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the fake files")
    args = ap.parse_args()

    if args.keep:
        tmp = Path(tempfile.mkdtemp(prefix="fake_eval_"))
        print(f"Fake files will be kept in: {tmp}\n")
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory()
        tmp = Path(cleanup.name)

    rng = random.Random(42)
    root, folder, manifest = tmp / "dataset", tmp / "predictions", tmp / "splits" / "official.json"
    make_dataset(root, rng)
    save_split_manifest(official_split(root), manifest)
    make_predictions(root, folder, rng)
    make_broken_files(folder)

    print("=== Test 1: all fake files (only test should be scored) ===\n")
    results, text = run(root=root, folder=folder, manifest_path=manifest)
    by = {(r["method"], r["split"]): r for r in results}

    print("Checks:")
    check(len(results) == 4, f"4 good test files scored (got {len(results)})")
    check(all(r["split"] == "test" for r in results), "only the test split was scored")
    check("Ignored (not test)" in text and "perfect_val" in text, "val files listed as ignored")
    check("[X] brokenval" not in text, "broken val file was not even checked")
    check(("broken", "test") not in by and ("wrongcols", "test") not in by,
          "broken test files were skipped, not scored")
    check("[X] broken (test): 5 problems" in text, "broken file: all 5 problems found")
    check("columns should be" in text, "wrong-columns file reported")
    p = by[("perfect", "test")]
    check(p["mAP50"] == 1 and p["mAP50_95"] == 1 and p["f1"] == 1, "perfect scores 1.0")
    check(p["count_mae"] == 0, "perfect counts every bird")
    n = by[("noisy", "test")]
    check(0.5 < n["mAP50"] < 1.0, f"noisy is good but not perfect (mAP@0.5 = {n['mAP50']:.2f})")
    check(n["mAP50_95"] < n["mAP50"], "noisy scores lower at stricter overlap")
    r = by[("random", "test")]
    check(r["mAP50"] < 0.2, f"random scores near 0 (mAP@0.5 = {r['mAP50']:.2f})")
    e = by[("empty", "test")]
    check(e["mAP50"] == 0 and e["recall"] == 0 and e["fn"] == e["n_birds"],
          "empty finds nothing and misses every bird")
    check(p["mAP50"] > n["mAP50"] > r["mAP50"] >= e["mAP50"], "ranking is perfect > noisy > random >= empty")
    check(p["n_images"] == SPLIT_SIZES["test"], "test split has the right image count")
    check("Split: val" not in text, "no val table printed")

    print("\n=== Test 2: no predictions folder ===\n")
    results, text = run(root=root, folder=tmp / "does_not_exist", manifest_path=manifest)
    print("Checks:")
    check(results == [] and "No predictions folder yet" in text, "friendly message, no crash")

    print("\n=== Test 3: only val files in the folder ===\n")
    val_only = tmp / "val_only"
    save_predictions("perfect", "val", load_split(root, "val"),
                     {r.image_id: np.array([[*b, 0.9] for b in r.boxes]).reshape(-1, 5)
                      for r in load_split(root, "val")}, folder=val_only)
    results, text = run(root=root, folder=val_only, manifest_path=manifest)
    print("Checks:")
    check(results == [] and "Nothing to evaluate" in text, "val-only folder: nothing is scored")

    print("\n=== Test 4: split file lists a test image that is not on disk ===\n")
    split = official_split(root)
    split["test"] = split["test"] + ["ghost_image"]
    bad_manifest = tmp / "splits" / "bad.json"
    save_split_manifest(split, bad_manifest)
    results, text = run(root=root, folder=folder, manifest_path=bad_manifest)
    print("Checks:")
    check("cannot load the test images" in text and "ghost_image" in text,
          "missing image reported by name")
    check(results == [], "nothing is scored with an incomplete test set")

    if cleanup:
        cleanup.cleanup()
    print("\n" + ("ALL TESTS PASSED" if failures == 0 else f"{failures} CHECKS FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
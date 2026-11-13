"""Test src/analysis.py with FAKE data, then save the fake results in analysis/
so you can open notebooks/results_plots.ipynb and see the graphs.

    python scripts/TEST_analysis.py                  # tests, then saves the fake results in analysis/
    python scripts/TEST_analysis.py --out some/dir   # save them somewhere else
    python scripts/TEST_analysis.py --force          # replace analysis/ even if it holds other results

It reuses the fake dataset and the same 4 pretend methods as TEST_evaluation.py
(perfect, noisy, random, empty), runs analysis.py on them and checks the numbers
in every csv file. Your real data/ and predictions/ are never touched.

The saved fake results are marked as FAKE (info.json), the notebook shows a warning,
and its figures get a FAKE_ file name. If analysis/ already holds REAL results,
nothing is replaced unless you use --force.
"""
import argparse
import contextlib
import csv
import io
import json
import random
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

import TEST_evaluation as fake  # noqa: E402  (scripts/TEST_evaluation.py)
from analysis import analyse_all  # noqa: E402
from data.split import official_split, save_split_manifest  # noqa: E402
from evaluation import load_records, precision_recall_f1  # noqa: E402
from predictions import load_predictions  # noqa: E402

failures = 0


def check(condition, message):
    global failures
    print(("  [OK] " if condition else "  [X]  ") + message)
    failures += 0 if condition else 1


def quiet(function, *args, **kwargs):
    """Run a function, return (result, everything it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


def safe_to_replace(target):
    """True if the folder is missing, empty, or holds only fake test data from this script."""
    target = Path(target)
    if not target.exists() or not any(target.iterdir()):
        return True
    info_file = target / "info.json"
    try:
        return json.loads(info_file.read_text()).get("fake_demo") is True
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description="Test analysis.py with fake data and save the fake results")
    ap.add_argument("--out", help="folder for the fake results (default: analysis/)")
    ap.add_argument("--force", action="store_true", help="replace the folder even if it holds other results")
    args = ap.parse_args()

    warnings.simplefilter("ignore")
    cleanup = tempfile.TemporaryDirectory()
    tmp = Path(cleanup.name)
    root, folder, manifest, out = tmp / "dataset", tmp / "predictions", tmp / "splits" / "official.json", tmp / "analysis"

    rng = random.Random(42)
    fake.make_dataset(root, rng)
    save_split_manifest(official_split(root), manifest)
    fake.make_predictions(root, folder, rng)
    fake.make_broken_files(folder)

    def read(name):
        with open(out / name, newline="") as f:
            return list(csv.DictReader(f))

    print("=== Test 1: run analysis.py ===\n")
    summary, text = quiet(analyse_all, root=root, folder=folder, manifest_path=manifest, out=out)
    print(text)
    print("Checks:")
    for name in ("summary.csv", "pr_curves.csv", "density_bins.csv", "size_bins.csv", "per_image.csv", "info.json"):
        check((out / name).exists(), f"{name} was saved")

    rows = {(r["method"], r["split"]): r for r in read("summary.csv")}
    check(len(rows) == 6, f"6 good files analysed: perfect/noisy on val+test, random/empty on test (got {len(rows)})")
    info = json.loads((out / "info.json").read_text())
    check(len(info["skipped"]) == 3, f"3 broken files skipped and listed in info.json (got {len(info['skipped'])})")
    check("[X] broken (test)" in text, "problems with the broken file were printed")

    print("\n=== Test 2: score cut-off rule (tune on val only) ===\n")
    print("Checks:")
    check(rows[("perfect", "val")]["thr_source"] == "self" and rows[("perfect", "test")]["thr_source"] == "val",
          "val uses its own best cut-off ('self'); test takes it from val ('val')")
    check(rows[("noisy", "test")]["thr"] == rows[("noisy", "val")]["thr"], "noisy: test uses exactly the val cut-off")
    check(all(rows[(m, "test")]["thr_source"] == "default" and float(rows[(m, "test")]["thr"]) == 0.5
              for m in ("random", "empty")), "methods without val predictions get the default 0.5")

    val_records = load_records(root, "val", manifest)
    val_preds = load_predictions("noisy", "val", folder)
    saved_f1 = float(rows[("noisy", "val")]["f1"])
    best_by_sweep = max(precision_recall_f1(val_records, val_preds, 0.5, float(t))["f1"]
                        for t in np.unique(np.concatenate([p[:, 4] for p in val_preds.values()])))
    check(abs(saved_f1 - best_by_sweep) < 1e-5, f"saved cut-off really has the best F1 (checked by trying all: {best_by_sweep:.4f})")

    print("\n=== Test 3: numbers agree with evaluation.py ===\n")
    print("Checks:")
    p = rows[("perfect", "test")]
    check(float(p["mAP50"]) == 1 and float(p["mAP50_95"]) == 1 and float(p["f1"]) == 1, "perfect scores 1.0")
    check(float(p["count_mae"]) == 0 and float(p["count_rmse"]) == 0, "perfect: count error 0 (MAE and RMSE)")
    e = rows[("empty", "test")]
    check(float(e["mAP50"]) == 0 and int(e["tp"]) == 0 and int(e["fn"]) == int(e["n_birds"]), "empty: nothing found")
    ok = True
    for (method, split), r in rows.items():
        recs, preds = load_records(root, split, manifest), load_predictions(method, split, folder)
        prf = precision_recall_f1(recs, preds, 0.5, float(r["thr"]))
        ok &= (prf["tp"], prf["fp"], prf["fn"]) == (int(r["tp"]), int(r["fp"]), int(r["fn"]))
    check(ok, "TP/FP/FN in summary.csv match precision_recall_f1() at the saved cut-off, for every row")
    check(all(float(r["count_rmse"]) >= float(r["count_mae"]) - 1e-9 for r in rows.values()), "RMSE is never below MAE")

    print("\n=== Test 4: breakdown files add up ===\n")
    print("Checks:")
    images, density, sizes = read("per_image.csv"), read("density_bins.csv"), read("size_bins.csv")
    ok_img = ok_den = ok_size = True
    for (method, split), r in rows.items():
        mine = [x for x in images if x["method"] == method and x["split"] == split]
        ok_img &= len(mine) == int(r["n_images"]) and all(
            sum(int(x[k]) for x in mine) == int(r[k]) for k in ("tp", "fp", "fn"))
        d = [x for x in density if x["method"] == method and x["split"] == split]
        ok_den &= (sum(int(x["n_images"]) for x in d) == int(r["n_images"])
                   and sum(int(x["n_birds"]) for x in d) == int(r["n_birds"])
                   and all(sum(int(x[k]) for x in d) == int(r[k]) for k in ("tp", "fp", "fn")))
        s = [x for x in sizes if x["method"] == method and x["split"] == split]
        ok_size &= (sum(int(x["n_birds"]) for x in s) == int(r["n_birds"])
                    and sum(int(x["found"]) for x in s) == int(r["tp"]))
    check(ok_img, "per_image.csv: one row per image, TP/FP/FN add up to summary.csv")
    check(ok_den, "density_bins.csv: images, birds, TP/FP/FN add up to summary.csv")
    check(ok_size, "size_bins.csv: birds add up, and birds found = TP")
    big = [x for x in density if x["bin"] == ">100"]
    check(big and all(x["n_images"] == "0" and x["mAP50"] == "" for x in big), "empty density bins have 0 images and a blank mAP")
    zero = [x for x in density if x["bin"] == "0" and x["method"] == "perfect"]
    check(zero and zero[0]["mAP50"] == "" and float(zero[0]["fp_per_image"]) == 0, "bin '0' (no birds): mAP blank, false alarms per image given")
    check(len({x["size_bin"] for x in sizes}) == 3, "birds split into small / medium / large")

    curves = read("pr_curves.csv")
    per_method = {m: [c for c in curves if c["method"] == m and c["split"] == "test"] for m in ("perfect", "random", "empty")}
    check(all(len(v) <= 300 for v in per_method.values()), "curves thinned to at most 300 points")
    check(per_method["perfect"] and float(per_method["perfect"][-1]["recall"]) == 1, "perfect curve reaches recall 1")
    check(per_method["empty"] == [], "empty method has no curve points")

    print("\n=== Test 5: running it again, and nothing to analyse ===\n")
    print("Checks:")
    again, _ = quiet(analyse_all, root=root, folder=folder, manifest_path=manifest, out=out)
    check(len(again) == 6 and len(read("summary.csv")) == 6, "second run replaces the files (no duplicate rows)")
    result, text = quiet(analyse_all, root=root, folder=tmp / "nope", manifest_path=manifest, out=tmp / "x")
    check(result is None and "No predictions folder yet" in text, "no predictions folder: friendly message")
    only_broken = tmp / "only_broken"
    only_broken.mkdir()
    (only_broken / "wrongcols_test.csv").write_text("a,b,c\n1,2,3\n")
    result, text = quiet(analyse_all, root=root, folder=only_broken, manifest_path=manifest, out=tmp / "y")
    check(result is None and not (tmp / "y").exists(), "only broken files: nothing saved")

    print("\n=== Test 6: save the fake results in analysis/ ===\n")
    target = Path(args.out) if args.out else REPO / "analysis"
    if not args.force and not safe_to_replace(target):
        print(f"NOT saved: {target} already holds results that are not fake test data.")
        print("Use --out some/other/folder to save them elsewhere, or --force to replace it.")
    else:
        saved, _ = quiet(analyse_all, root=root, folder=folder, manifest_path=manifest, out=target)
        info_file = target / "info.json"
        info = json.loads(info_file.read_text())
        info["fake_demo"] = True
        info["note"] = "FAKE test data made by scripts/TEST_analysis.py. Do not use in the report."
        info_file.write_text(json.dumps(info, indent=2))

        print("Checks:")
        files = ("summary.csv", "pr_curves.csv", "density_bins.csv", "size_bins.csv", "per_image.csv", "info.json")
        check(all((target / f).exists() for f in files), f"6 files saved in {target}")
        saved_rows = {(r["method"], r["split"]) for r in csv.DictReader(open(target / "summary.csv", newline=""))}
        check({m for m, _ in saved_rows} == set(fake.METHODS), f"same methods as TEST_evaluation.py: {', '.join(fake.METHODS)}")
        check(saved_rows == {(m, s) for m, (_, splits) in fake.METHODS.items() for s in splits},
              "same splits as TEST_evaluation.py (perfect/noisy: val + test, random/empty: test)")
        check(json.loads(info_file.read_text()).get("fake_demo") is True, "marked as FAKE in info.json")
        print(f"\nOpen notebooks/results_plots.ipynb to see the graphs of the fake data.")

    cleanup.cleanup()
    print("\n" + ("ALL TESTS PASSED" if failures == 0 else f"{failures} CHECKS FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
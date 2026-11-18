"""Run the Method 2 ablation grid on val, then build results/m2_comparison.csv.

Each config is run by scripts/run_m2.py, which writes results/m2_val_<name>.json
and a .meta.json sidecar; scripts/compare_m2.py turns those into the table.

Never point this at the test split: that is one final run per frozen config,
through run_m2.py --final.

Example:
  python scripts/sweep_m2.py
  python scripts/sweep_m2.py --quick --train-limit 120 --limit 60
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The ablation, in the order the report discusses it. Five groups, each changing
# one thing: hard negative rounds (0/1/2), the feature with the classifier fixed,
# the classifier with the feature fixed, the negative sampling, then the proposal
# source and box regression.
#
# This must list EVERY configs/m2_*.yaml, otherwise re-running the sweep rebuilds
# results/m2_comparison.csv without the missing rows and silently drops them.
# tests/ml/test_config.py::test_sweep_covers_every_shipped_config enforces that.
FULL = [
    "m2_hog_svm_hn0",
    "m2_hog_svm",
    "m2_hog_svm_hn2",
    "m2_lbp_svm",
    "m2_colour_svm",
    "m2_concat_svm",
    "m2_hog_rf",
    "m2_lbp_rf",
    "m2_colour_rf",
    "m2_concat_rf",
    "m2_concat_gb_noproposalneg",
    "m2_concat_gb",
    "m2_concat_gb_blobprops",
    "m2_concat_gb_m1props",
    "m2_concat_gb_m1props_boxreg",
    "m2_concat_gb_m1props_boxreg_iou04",
    "m2_concat_gb_m1props_boxreg_iou05",
]
QUICK = ["m2_hog_svm_hn0", "m2_hog_svm", "m2_hog_rf", "m2_concat_gb"]


def is_finished(out, split, limit, train_limit):
    """Has this variant already been run with the limits asked for now?

    A finished variant has both the predictions and the sidecar, so anything
    half-written is re-run and an interrupted sweep can restart. The limits have
    to match too: otherwise a cheap `--quick --limit 60` run gets reused as if it
    were a full-split one, and its numbers reach the table unnoticed.
    """
    meta_path = out.with_suffix(".meta.json")
    if not (out.exists() and meta_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return False
    return (meta.get("split") == split
            and meta.get("limit") == limit
            and meta.get("train_limit") == train_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="four representative variants only")
    parser.add_argument("--limit", type=int, help="evaluation images per variant")
    parser.add_argument("--train-limit", type=int, help="training images per variant")
    parser.add_argument("--configs", nargs="*", help="override the grid with these config names")
    parser.add_argument("--redo", action="store_true",
                        help="re-run variants that already have a results file "
                             "(default: skip them, so an interrupted sweep resumes)")
    args = parser.parse_args()

    names = args.configs or (QUICK if args.quick else FULL)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    produced, failed = [], []

    for i, name in enumerate(names, 1):
        config = ROOT / "configs" / f"{name}.yaml"
        out = results_dir / f"m2_val_{name.replace('m2_', '')}.json"
        done = is_finished(out, "val", args.limit, args.train_limit)
        if done and not args.redo:
            print(f"\n===== [{i}/{len(names)}] {name} -- already done, skipping =====", flush=True)
            produced.append(str(out))
            continue
        print(f"\n===== [{i}/{len(names)}] {name} =====", flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "run_m2.py"),
               "--config", str(config), "--split", "val", "--out", str(out)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.train_limit:
            cmd += ["--train-limit", str(args.train_limit)]
        if subprocess.run(cmd).returncode == 0:
            produced.append(str(out))
        else:
            failed.append(name)
            print(f"!! {name} failed, continuing", flush=True)

    if failed:
        print(f"\n{len(failed)} variant(s) failed and are NOT in the table: {', '.join(failed)}")
    if not produced:
        sys.exit("no variants succeeded")

    print(f"\n===== building the comparison table from {len(produced)} runs =====", flush=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "compare_m2.py"),
           "--split", "val", "--pred", *produced]
    # A limited sweep produces a table of limited runs. They agree with each
    # other, so allow it, but compare_m2.py still prints the warning and records
    # n_images so nobody mistakes it for the real thing.
    if args.limit or args.train_limit:
        cmd.append("--allow-partial")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

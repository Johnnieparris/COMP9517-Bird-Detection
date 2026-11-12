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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The ablation, in the order the report discusses it. Four groups, each changing
# one thing: hard negative rounds (0/1/2), then the feature with the classifier
# fixed, then the classifier with the feature fixed, then the proposal source.
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
    "m2_concat_gb",
    "m2_concat_gb_blobprops",
    "m2_concat_gb_m1props",
]
QUICK = ["m2_hog_svm_hn0", "m2_hog_svm", "m2_hog_rf", "m2_concat_gb"]


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

        # A finished variant has both the predictions and the sidecar, so
        # anything half-written is re-run and an interrupted sweep can restart.
        done = out.exists() and out.with_suffix(".meta.json").exists()
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
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_m2.py"),
                    "--split", "val", "--pred", *produced], check=True)


if __name__ == "__main__":
    main()

"""Random search for a Method 1 config, maximising AP@0.5 on val.

Example: python scripts/tune_m1.py --base configs/m1_seg_colour.yaml \
    --space configs/search/seg_colour.yaml --split val --trials 200 --seed 0 --name seg_colour

Writes results/m1_tuning_<name>.csv and configs/m1_<name>_tuned.yaml. Timings come from
run_m1.py, which runs serially, so nothing here records ms/img.
"""
import argparse
import csv
import dataclasses
import datetime
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from src.classical.interim_data import DEFAULT_ROOT, load_split
from src.classical.pipeline import SegConfig, load_config
from src.classical.quick_eval import ap50, recall_at_iou
from src.classical.tuning import detect_parallel, sample_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--space", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])  # never test
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-root", default=str(ROOT / DEFAULT_ROOT))
    return parser.parse_args()


def write_best(cfg, ap, args):
    values = {"method": "seg" if isinstance(cfg, SegConfig) else "blob", **dataclasses.asdict(cfg)}
    values = {key: list(value) if isinstance(value, tuple) else value for key, value in values.items()}
    header = (f"# Best of {args.trials} random trials on {args.split}: AP@0.5 {ap:.4f}, "
              f"seed {args.seed}, {datetime.date.today()}.\n")
    path = ROOT / "configs" / f"m1_{args.name}_tuned.yaml"
    path.write_text(header + yaml.safe_dump(values, sort_keys=False))
    return path


def main():
    args = parse_args()
    base = load_config(args.base)
    space = yaml.safe_load(Path(args.space).read_text())
    records = load_split(args.split, root=args.data_root)
    rng = random.Random(args.seed)

    best_cfg, best_ap = None, -1.0
    out = ROOT / "results" / f"m1_tuning_{args.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, ["trial", "ap50", "recall50", *space])
        writer.writeheader()
        for trial in range(args.trials):
            cfg = sample_config(base, space, rng)
            rows = detect_parallel(records, cfg, args.workers)
            ap, recall = ap50(rows, records), recall_at_iou(rows, records)
            writer.writerow({"trial": trial, "ap50": ap, "recall50": recall,
                             **{key: getattr(cfg, key) for key in space}})
            f.flush()  # a stopped search keeps its results
            if ap > best_ap:
                best_cfg, best_ap = cfg, ap
            print(f"trial {trial}: AP {ap:.4f} recall {recall:.4f} best {best_ap:.4f}", flush=True)
    print(f"-> {out}, {write_best(best_cfg, best_ap, args)}")


if __name__ == "__main__":
    main()

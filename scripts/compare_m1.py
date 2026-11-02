"""Write results/m1_comparison.csv from Method 1 prediction files and their sidecars.

Example: python scripts/compare_m1.py --split val --pred results/m1_val_a.json results/m1_val_b.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classical.interim_data import DEFAULT_ROOT, load_split
from src.classical.quick_eval import ap50, best_f1, count_mae, recall_at_iou


def compare_row(pred_path, records, variant, split=None):
    """One comparison row. Raises if the predictions came from a different split."""
    pred_path = Path(pred_path)
    rows = json.loads(pred_path.read_text())
    meta = json.loads(pred_path.with_suffix(".meta.json").read_text())
    if split is not None and meta["split"] != split:
        raise ValueError(f"{pred_path} is from {meta['split']}, not {split}")
    f1 = best_f1(rows, records)
    return {"variant": variant, "split": meta["split"], "ap50": ap50(rows, records),
            "recall50": recall_at_iou(rows, records), "precision": f1["precision"],
            "recall_at_best_f1": f1["recall"], "f1": f1["f1"],
            "count_mae": count_mae(rows, records, f1["score_threshold"]),
            "score_threshold": f1["score_threshold"], "ms_per_image": meta["stats"]["ms_per_image"],
            "train_time_s": meta["train_time_s"], "n_images_capped": meta["stats"]["n_images_capped"],
            "machine": meta["machine"], "git_sha": meta["git_sha"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", nargs="+", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--data-root", default=str(ROOT / DEFAULT_ROOT))
    parser.add_argument("--out", default=str(ROOT / "results" / "m1_comparison.csv"))
    args = parser.parse_args()

    records = load_split(args.split, root=args.data_root)
    table = [compare_row(path, records, Path(path).stem, args.split) for path in args.pred]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print(f"{row['variant']:32s} AP {row['ap50']:.4f} R(all) {row['recall50']:.3f} R {row['recall_at_best_f1']:.3f} P {row['precision']:.3f} "
              f"F1 {row['f1']:.3f} MAE {row['count_mae']:.2f} {row['ms_per_image']:.1f} ms/img")


if __name__ == "__main__":
    main()

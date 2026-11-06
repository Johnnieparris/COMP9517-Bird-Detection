"""Check the dataset. Usage:

    python scripts/check_dataset.py
    python scripts/check_dataset.py "data/Beach Bird Detection Dataset v1.0"

Only reports problems. Never changes files.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data import validation as v  # noqa: E402
from data.dataset import load_dataset, DEFAULT_ROOT

DEFAULT_ROOT = REPO / "data" / "Beach Bird Detection Dataset v1.0"
MAX_SHOWN = 5  # show first few problems per check


def show(title, items):
    print(f"{'[X]' if items else '[ ]'} {title}: {len(items)}")
    for line in items[:MAX_SHOWN]:
        print(f"    - {line}")
    if len(items) > MAX_SHOWN:
        print(f"    ... and {len(items) - MAX_SHOWN} more")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    print(f"Dataset root:\n  {root}\n")

    if not root.is_dir():
        print("Dataset validation FAILED")
        print("Folder not found. Download the dataset or copy it from a teammate.")
        return 1

    print("Checking structure...")
    ok, problems = v.check_directory_structure(root)
    for name in ok:
        print(f"[ ] {name}")
    for p in problems:
        print(f"[X] {p}")
    if problems:
        print("\nDataset validation FAILED (fix folder names first)")
        return 1

    n_img = sum(len(v.list_images(root / "images" / s)) for s in v.SPLITS)
    n_lbl = sum(len(v.list_labels(root / "labels" / s)) for s in v.SPLITS)
    print(f"\nImages: {n_img}\nLabels: {n_lbl}\n")

    errors = [
        ("Missing labels", v.check_missing_labels(root)),
        ("Missing images", v.check_missing_images(root)),
        ("Corrupt images", v.check_corrupt_images(root)),
        ("Invalid boxes", v.check_bbox_ranges(root)),
        ("Duplicate ids", v.check_duplicate_ids(root)),
    ]
    for title, items in errors:
        show(title, items)

    d = load_dataset(DEFAULT_ROOT)
    print()
    for k, l in d.items():
        print(f"[ ] {k}: {len(l)} items")
        
    boxCount = sum(len(r.boxes) for v in d.values() for r in v)
    print()
    print(f"There are {boxCount} bounding box images")  # expect 12024 boxes
    print()
    empty = v.check_empty_files(root)
    print(f"Note: {len(empty)} empty label files (OK, means no birds in image)")

    failed = any(items for _, items in errors)
    print("\nDataset validation " + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
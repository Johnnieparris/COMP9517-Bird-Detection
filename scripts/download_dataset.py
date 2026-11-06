import shutil
from pathlib import Path

import kagglehub

DEST = Path(__file__).resolve().parents[1] / "data" / "Beach Bird Detection Dataset v1.0"


def find_root(start):
    """Find the folder that contains both images/ and labels/."""
    for p in [start, *start.rglob("*")]:
        if (p / "images").is_dir() and (p / "labels").is_dir():
            return p
    return None


src = Path(kagglehub.dataset_download("liugalhigh/beach-bird"))
root = find_root(src)

if root is None:
    raise SystemExit(f"Could not find images/ and labels/ in {src}")
if DEST.exists():
    raise SystemExit(f"{DEST} already exists. Delete it first.")

shutil.copytree(root, DEST)
print("Copied to", DEST)
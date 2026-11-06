"""Dataset checks. These only READ files. They never change anything."""
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "val", "test"]


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def list_labels(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() == ".txt")


def splits_present(root):
    """Splits where BOTH images/<split> and labels/<split> exist."""
    root = Path(root)
    return [s for s in SPLITS
            if (root / "images" / s).is_dir() and (root / "labels" / s).is_dir()]


def check_directory_structure(root):
    """Returns (ok_folders, problems)."""
    root = Path(root)
    ok, problems = [], []
    for kind in ("images", "labels"):
        for split in SPLITS:
            name = f"{kind}/{split}"
            if (root / kind / split).is_dir():
                ok.append(name)
                continue
            found = []
            if (root / kind).is_dir():
                found = [d.name for d in (root / kind).iterdir() if d.is_dir()]
            msg = f"Missing: {name}"
            if found:
                msg += f"  (found instead: {', '.join(sorted(found))})"
            problems.append(msg)
    return ok, problems


def check_missing_labels(root):
    """Images with no .txt file."""
    root, out = Path(root), []
    for s in splits_present(root):
        for img in list_images(root / "images" / s):
            if not (root / "labels" / s / (img.stem + ".txt")).exists():
                out.append(f"{s}/{img.name}: no label file")
    return out


def check_missing_images(root):
    """Label files with no image."""
    root, out = Path(root), []
    for s in splits_present(root):
        stems = {p.stem for p in list_images(root / "images" / s)}
        for lbl in list_labels(root / "labels" / s):
            if lbl.stem not in stems:
                out.append(f"{s}/{lbl.name}: no image")
    return out


def check_corrupt_images(root):
    """Images that PIL cannot open."""
    root, out = Path(root), []
    for s in splits_present(root):
        for img in list_images(root / "images" / s):
            try:
                with Image.open(img) as im:
                    im.load()
            except Exception as e:
                out.append(f"{s}/{img.name}: {type(e).__name__}")
    return out


def check_bbox_ranges(root):
    """Each line must be: class cx cy w h  (4 numbers between 0 and 1)."""
    root, out = Path(root), []
    for s in splits_present(root):
        for lbl in list_labels(root / "labels" / s):
            lines = lbl.read_text().splitlines()
            for n, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                where = f"{s}/{lbl.name} line {n}"
                parts = line.split()
                if len(parts) != 5:
                    out.append(f"{where}: needs 5 values, has {len(parts)}")
                    continue
                try:
                    cls = float(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    out.append(f"{where}: not numbers")
                    continue
                if cls < 0 or cls != int(cls):
                    out.append(f"{where}: bad class '{parts[0]}'")
                if not all(0 <= v <= 1 for v in (cx, cy, w, h)):
                    out.append(f"{where}: value outside 0-1 -> {line.strip()}")
                if w <= 0 or h <= 0:
                    out.append(f"{where}: width/height must be above 0")
    return out


def check_empty_files(root):
    """Empty label files. Allowed (means 'no birds'), so this is a warning."""
    root, out = Path(root), []
    for s in splits_present(root):
        for lbl in list_labels(root / "labels" / s):
            if not lbl.read_text().strip():
                out.append(f"{s}/{lbl.name}: empty (no birds)")
    return out


def check_duplicate_ids(root):
    """Same image name twice in one split, or shared between splits."""
    root, out, seen = Path(root), [], {}
    for s in splits_present(root):
        in_split = {}
        for img in list_images(root / "images" / s):
            if img.stem in in_split:
                out.append(f"{s}: '{img.stem}' appears twice "
                           f"({in_split[img.stem]}, {img.name})")
            in_split[img.stem] = img.name
            seen.setdefault(img.stem, set()).add(s)
    for stem, where in seen.items():
        if len(where) > 1:
            out.append(f"'{stem}' is in more than one split: {', '.join(sorted(where))}")
    return out
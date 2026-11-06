"""Convert other dataset formats into OUR layout:

    <dest>/images/<train|val|test>/*.jpg
    <dest>/labels/<train|val|test>/*.txt    (YOLO: class cx cy w h, values 0-1)

The source folder is never changed. After importing, run
scripts/check_dataset.py on <dest>.

    import_yolo_dataset("downloads/birds_yolo", "data/extra_birds")
    import_roboflow("downloads/roboflow_export", "data/extra_birds")
    import_coco_dataset("ann.json", "imgs/", "data/extra_birds", "train")
"""
import json
import shutil
import warnings
from pathlib import Path

from PIL import Image

from .validation import IMAGE_EXTS

# Other names people use for our three splits -> our name
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


def _start_split(dest, split):
    """Make the folders for one split. Refuse to mix into an existing split."""
    dest = Path(dest)
    if (dest / "images" / split).exists():
        raise FileExistsError(f"{dest / 'images' / split} already exists. Delete it first.")
    (dest / "images" / split).mkdir(parents=True)
    (dest / "labels" / split).mkdir(parents=True)

def _remove_split(dest, split):
    """Undo _start_split (used when an import fails half way)."""
    for kind in ("images", "labels"):
        shutil.rmtree(Path(dest) / kind / split, ignore_errors=True)
        try:
            (Path(dest) / kind).rmdir()  # only works if it is now empty
        except OSError:
            pass
 
 
def _import_all(jobs, dest):
    """jobs = {split: function}. Checks the destination first, then runs all
    jobs. If one fails, the splits already imported are removed again."""
    for split in jobs:
        if (Path(dest) / "images" / split).exists():
            raise FileExistsError(f"{Path(dest) / 'images' / split} already exists. "
                                  "Delete it first.")
    done, counts = [], {}
    try:
        for split, job in jobs.items():
            counts[split] = job()
            done.append(split)
    except Exception:
        for split in done:
            _remove_split(dest, split)
        raise
    return counts
 
 
def _both_map_error(first, second, split):
    return ValueError(f"Both '{first}' and '{second}' map to '{split}'. "
                      "Remove or rename one of them, then try again.")
 
def _copy_pairs(img_dir, lbl_dir, dest, split):
    """Copy images and their .txt labels. Returns number of images."""
    _start_split(dest, split)
    try:
        n = 0
        for img in sorted(Path(img_dir).iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            shutil.copy2(img, Path(dest) / "images" / split / img.name)
            lbl = Path(lbl_dir) / (img.stem + ".txt")
            if lbl.exists():  # a missing label is left missing, so the checker reports it
                shutil.copy2(lbl, Path(dest) / "labels" / split / lbl.name)
            n += 1
        return n
    except Exception:
        _remove_split(dest, split)
        raise


# ---------- YOLO ----------
 
def _find_yolo_sources(src):
    """{split: (folder name, image dir, label dir)}. Stops if two folders
    (e.g. val/ and valid/) would both become the same split."""
    found = {}
    for name, split in SPLIT_ALIASES.items():
        a = (src / "images" / name, src / "labels" / name)
        b = (src / name / "images", src / name / "labels")
        pair = next((p for p in (a, b) if p[0].is_dir()), None)
        if pair is None:
            continue
        if split in found:
            raise _both_map_error(found[split][0], name, split)
        found[split] = (name, *pair)
    if not found:
        raise ValueError(f"No train/val/test image folders found in {src}")
    return found
 
 
def import_yolo_dataset(src, dest):
    """YOLO folders -> our layout. Returns {split: number of images}.
 
    Understands two layouts:
      A)  src/images/<split>/   src/labels/<split>/
      B)  src/<split>/images/   src/<split>/labels/     (Roboflow style)
    and renames 'valid' / 'validation' to 'val'.
    """
    found = _find_yolo_sources(Path(src))
    jobs = {s: (_copy_pairs, (img, lbl, dest, s))
            for s, (_, img, lbl) in found.items()}
    return _import_all(jobs, dest)
 
 
# ---------- COCO ----------
 
def import_coco_dataset(json_path, images_dir, dest, split, class_map=None):
    """One COCO .json file -> one split of our layout. Returns number of images.
 
    COCO boxes are pixels [x, y, width, height]. YOLO wants 0-1 centre values.
    class_map = {coco_category_id: our_class}. Default: every category -> 0 (bird).
    """
    data = json.loads(Path(json_path).read_text())
    images_dir = Path(images_dir)
    _start_split(dest, split)
    try:
        boxes = {}
        for a in data["annotations"]:
            if a.get("iscrowd", 0):  # skip 'crowd' blobs, they are not single boxes
                continue
            boxes.setdefault(a["image_id"], []).append(a)
 
        n, missing = 0, 0
        for im in data["images"]:
            src_img = images_dir / im["file_name"]
            if not src_img.exists():
                missing += 1
                continue
            name = Path(im["file_name"]).name
            shutil.copy2(src_img, Path(dest) / "images" / split / name)
 
            width, height = im.get("width"), im.get("height")
            if not width or not height:
                with Image.open(src_img) as pil:
                    width, height = pil.size
 
            lines = []
            for a in boxes.get(im["id"], []):
                cls = 0 if class_map is None else class_map.get(a["category_id"])
                if cls is None:
                    continue
                x, y, w, h = a["bbox"]
                x1, y1 = max(x, 0), max(y, 0)              # keep the box inside the image
                x2, y2 = min(x + w, width), min(y + h, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                             f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
            # an empty .txt file means "no birds in this image"
            (Path(dest) / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines))
            n += 1
    except Exception:
        _remove_split(dest, split)
        raise
 
    if missing:
        warnings.warn(f"{missing} images listed in the json were not found in {images_dir}")
    return n
 
 
# ---------- Roboflow ----------
 
def import_roboflow(src, dest):
    """Roboflow export (YOLO or COCO) -> our layout. Returns {split: number of images}.
 
    Roboflow folders look like  src/train/  src/valid/  src/test/.
    COCO exports have an _annotations.coco.json inside each split folder.
    """
    src, coco = Path(src), {}
    for name, split in SPLIT_ALIASES.items():
        j = src / name / "_annotations.coco.json"
        if j.exists():
            if split in coco:
                raise _both_map_error(coco[split][0], name, split)
            coco[split] = (name, j)
    if not coco:
        return import_yolo_dataset(src, dest)  # YOLO export
    jobs = {s: (import_coco_dataset, (j, j.parent, dest, s))
            for s, (_, j) in coco.items()}
    return _import_all(jobs, dest)

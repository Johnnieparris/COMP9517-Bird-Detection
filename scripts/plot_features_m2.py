"""Draw what Method 2's descriptors actually see, for the report and the video.

One row per patch: the patch itself, the HOG gradient picture, the uniform-LBP
histogram and the HSV colour histogram. Bird crops come from ground-truth boxes,
background crops from boxes that overlap no bird, so the figure shows the
comparison the classifier is actually asked to make.

Example:
  python scripts/plot_features_m2.py --split val --n 3
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog as sk_hog

from data.dataset import DEFAULT_ROOT, get_image
from data.split import load_split_from_manifest, load_split_manifest
from src.classical.boxes import iou_matrix
from src.ml.config import M2Config
from src.ml.detector import crop
from src.ml.features import (COLOUR_BINS, HOG_BLOCK, HOG_CELL, HOG_ORIENTATIONS,
                             LBP_P, f_colour, f_lbp, feature_length)
from src.ml.proposals import Proposals, training_box_sizes

MANIFEST = ROOT / "splits" / "official.json"


def sample_patches(records, boxes_wh, n, patch, seed=0):
    """n bird crops and n background crops, as RGB patches.

    Birds are the biggest one in each image so the figure stays legible;
    background comes from candidate boxes that overlap no bird at all.
    """
    rng = np.random.RandomState(seed)
    proposals = Proposals(M2Config(proposals="window"), boxes_wh)
    birds, background = [], []
    for record in records:
        if len(birds) >= n and len(background) >= n:
            break
        if not len(record.boxes):
            continue
        image = get_image(record)
        if len(birds) < n:
            areas = ((record.boxes[:, 2] - record.boxes[:, 0]) *
                     (record.boxes[:, 3] - record.boxes[:, 1]))
            got = crop(image, record.boxes[areas.argmax()], patch)
            if got is not None:
                birds.append(got)
        if len(background) < n:
            candidates = proposals(image)
            overlap = iou_matrix(candidates, record.boxes).max(axis=1)
            clear = np.where(overlap < 0.05)[0]
            if len(clear):
                got = crop(image, candidates[rng.choice(clear)], patch)
                if got is not None:
                    background.append(got)
    return birds[:n], background[:n]


def draw_row(axes, patch, label):
    """One patch across four panels: the crop, HOG, the LBP and HSV histograms."""
    gray = np.dot(patch[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    axes[0].imshow(patch)
    axes[0].set_ylabel(label, fontsize=11, fontweight="bold")
    axes[0].set_xticks([]), axes[0].set_yticks([])

    _, hog_image = sk_hog(gray, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_CELL,
                          cells_per_block=HOG_BLOCK, block_norm="L2-Hys", visualize=True)
    axes[1].imshow(hog_image, cmap="magma")
    axes[1].set_xticks([]), axes[1].set_yticks([])

    lbp = f_lbp(gray)
    axes[2].bar(range(len(lbp)), lbp, color="#4C72B0", width=0.85)
    axes[2].set_ylim(0, 0.6)
    axes[2].tick_params(labelsize=7)

    # f_colour lays each channel out as COLOUR_BINS histogram values followed by
    # a mean and a std, so the bars below skip past those two.
    colour = f_colour(patch)
    block = COLOUR_BINS + 2
    for i, (name, c) in enumerate((("H", "#C44E52"), ("S", "#55A868"), ("V", "#8172B2"))):
        values = colour[i * block:i * block + COLOUR_BINS]
        axes[3].bar(np.arange(COLOUR_BINS) + i * (COLOUR_BINS + 1), values,
                    color=c, width=0.85, label=name)
    axes[3].set_ylim(0, 1.0)
    axes[3].tick_params(labelsize=7)
    axes[3].legend(fontsize=7, ncol=3, loc="upper right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--n", type=int, default=3, help="patches of each kind")
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--scan", type=int, default=40, help="images to scan for patches")
    parser.add_argument("--out", default=str(ROOT / "results" / "m2_features.png"))
    args = parser.parse_args()

    manifest = load_split_manifest(MANIFEST)
    train = load_split_from_manifest(args.data_root, manifest, "train")
    records = load_split_from_manifest(args.data_root, manifest, args.split)[:args.scan]
    birds, background = sample_patches(records, training_box_sizes(train),
                                       args.n, args.patch)
    rows = [(p, f"bird {i + 1}") for i, p in enumerate(birds)]
    rows += [(p, f"background {i + 1}") for i, p in enumerate(background)]

    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 1.9 * len(rows)))
    axes = np.atleast_2d(axes)
    titles = [f"patch ({args.patch}x{args.patch} RGB)",
              f"HOG ({feature_length('hog', args.patch)}-D)",
              f"uniform LBP, P={LBP_P} ({feature_length('lbp', args.patch)}-D)",
              f"HSV histogram ({feature_length('colour', args.patch)}-D)"]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=10)
    for row, (patch, label) in zip(axes, rows):
        draw_row(row, patch, label)

    fig.suptitle("Method 2 descriptors: what the classifier sees for a bird and for background",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

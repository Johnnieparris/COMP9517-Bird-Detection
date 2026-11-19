"""Make Member E's report figures (PDF + PNG) in docs/report/figures/.

    python scripts/make_e_figures.py

Inputs (each figure is skipped with a warning if its input is missing):
    results/robustness_test.csv        -> e_robustness.{pdf,png}
    results/data_efficiency_test.csv   -> e_data_efficiency.{pdf,png}
    analysis/density_bins.csv          -> e_density.{pdf,png}      (python src/analysis.py)
    results/failure_categories_test.csv-> e_failures.{pdf,png}
    docs/report/figures/gallery/*.jpg  -> e_gallery.{pdf,png}
Colours: one fixed, colour-blind-checked hue per method in every figure, plus a marker
shape per method so identity never relies on colour alone.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "docs/report/figures"
SINGLE, DOUBLE = 3.5, 7.16  # IEEE column widths, inches

# Fixed per-entity styling (validated categorical slots 1-6 of the reference palette).
STYLE = {
    "m1": dict(color="#2a78d6", marker="o", label="M1 classical"),
    "m2": dict(color="#eb6834", marker="s", label="M2 HOG/LBP+GB"),
    "yolo": dict(color="#1baf7a", marker="^", label="YOLO11n (official)"),
    "faster_rcnn": dict(color="#eda100", marker="D", label="Faster R-CNN"),
    "e_de_100": dict(color="#e87ba4", marker="v", label="YOLO11n, clean train (30 ep)"),
    "e_robust_aug": dict(color="#008300", marker="P", label="YOLO11n, distortion aug. (30 ep)"),
    "m1pre": dict(color="#2a78d6", marker="o", label="M1 + denoise", linestyle="--", mfc="white"),
    "m2pre": dict(color="#eb6834", marker="s", label="M2 + denoise", linestyle="--", mfc="white"),
}
DISTORTIONS = ["noise", "blur", "lowlight", "jpeg", "occlusion"]
TITLES = {"noise": "Gaussian noise", "blur": "Gaussian blur", "lowlight": "Low light",
          "jpeg": "JPEG compression", "occlusion": "Occlusion"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def read_csv(path):
    path = REPO / path
    if not path.exists():
        print(f"[skip] {path.relative_to(REPO)} not found")
        return None
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote docs/report/figures/{name}.pdf/.png")


def line(ax, xs, ys, method):
    s = {k: v for k, v in STYLE[method].items() if k != "label"}
    ax.plot(xs, ys, label=STYLE[method]["label"], **s)


def robustness():
    rows = read_csv("results/robustness_test.csv")
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[(r["distortion"], int(r["severity"]))] = float(r["mAP50"])
    top = [m for m in ("m1", "m1pre", "m2", "m2pre", "yolo", "faster_rcnn") if m in by]
    bottom = [m for m in ("yolo", "e_de_100", "e_robust_aug") if m in by]
    fig, axes = plt.subplots(2, 5, figsize=(DOUBLE, 3.7), sharex=True)
    for col, d in enumerate(DISTORTIONS):
        ax = axes[0, col]
        for m in top:
            clean = by[m].get(("clean", 0))
            pts = [(s, by[m][(d, s)] / clean) for s in (1, 2, 3) if (d, s) in by[m] and clean]
            if clean and pts:
                line(ax, [0] + [p[0] for p in pts], [1.0] + [p[1] for p in pts], m)
        ax.set_title(TITLES[d])
        ax.set_ylim(0, 1.25)
        ax2 = axes[1, col]
        for m in bottom:
            xs = [s for s in (0, 1, 2, 3) if ((d, s) if s else ("clean", 0)) in by[m]]
            if xs:
                line(ax2, xs, [by[m][(d, s) if s else ("clean", 0)] for s in xs], m)
        ax2.set_ylim(0, 1.0)
        if d == "occlusion" and "e_robust_aug" in bottom:
            # results/occluder_shortcut_val.csv: this model fires on grey patches pasted on background.
            ax2.annotate("shortcut:\ngrey patch\n= bird", xy=(2, by["e_robust_aug"][("occlusion", 2)]),
                         xytext=(1.55, 0.38), fontsize=6, color=INK,
                         arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
        ax2.set_xticks([0, 1, 2, 3])
        ax2.set_xlabel("severity")
    axes[0, 0].set_ylabel("mAP@0.5 / clean mAP@0.5")
    axes[1, 0].set_ylabel("mAP@0.5 (YOLO11n)")
    for row, ncol in ((0, 6), (1, 3)):
        handles, labels = axes[row, 0].get_legend_handles_labels()
        axes[row, 4].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    save(fig, "e_robustness")


def data_efficiency():
    rows = read_csv("results/data_efficiency_test.csv")
    if not rows:
        return
    done = {r["run"]: r for r in rows if r.get("status", "completed") == "completed" and r.get("mAP50")}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    curve = [r for k, r in done.items() if k in ("de_10", "de_25", "de_50", "de_100")]
    curve.sort(key=lambda r: int(r["train_images"]))
    if curve:
        xs = [int(r["train_images"]) for r in curve]
        ys = [float(r["mAP50"]) for r in curve]
        ax.plot(xs, ys, color="#e87ba4", marker="v", label="clean labels")
        for i, (x, y) in enumerate(zip(xs, ys)):
            last = i == len(xs) - 1  # the full-data label goes below, clear of the label-noise markers
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(-4, -11) if last else (-5, 4),
                        ha="right", fontsize=6, color=MUTED)
    extras = {"ln_jitter20": ("#2a78d6", "o", "boxes jittered ±20%"),
              "ln_drop20": ("#eb6834", "s", "20% of boxes dropped"),
              "de_10_strong": ("#008300", "P", "10% + strong aug.")}
    for run, (color, marker, label) in extras.items():
        if run in done:
            # Hollow markers so a point at the same x as the clean-label curve stays visible.
            ax.plot([int(done[run]["train_images"])], [float(done[run]["mAP50"])], linestyle="none",
                    color=color, marker=marker, markersize=7, label=label, mfc="white", mew=1.5)
    ax.set_xscale("log")
    ax.set_xlim(110, 1700)
    ax.set_ylim(0.74, 0.895)
    ax.set_xticks([137, 342, 684, 1369])
    ax.set_xticklabels(["137\n(10%)", "342\n(25%)", "684\n(50%)", "1369\n(100%)"])
    ax.minorticks_off()
    ax.set_xlabel("training images")
    ax.set_ylabel("test mAP@0.5")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "e_data_efficiency")


def density():
    rows = read_csv("analysis/density_bins.csv")
    if not rows:
        return
    rows = [r for r in rows if r["split"] == "test" and r["method"] in ("m1", "m2", "yolo", "faster_rcnn")]
    bins = ["0", "1-10", "11-50", "51-100", ">100"]
    # Bins with no test images (the test split has none above 100 birds) are left out.
    bins = [b for b in bins if any(r["bin"] == b and int(r["n_images"]) > 0 for r in rows)]
    methods = [m for m in ("m1", "m2", "yolo", "faster_rcnn") if any(r["method"] == m for r in rows)]
    val = {(r["method"], r["bin"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.1))
    width = 0.8 / len(methods)
    for k, (ax, key, ylabel, use_bins) in enumerate((
            (axes[0], "mAP50", "test mAP@0.5", bins[1:]),
            (axes[1], "count_mae", "count MAE (birds / image)", bins))):
        x = np.arange(len(use_bins))
        for i, m in enumerate(methods):
            ys = [float(val[(m, b)][key]) if (m, b) in val and val[(m, b)][key] not in ("", "None") else np.nan
                  for b in use_bins]
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, ys, width * 0.92, color=STYLE[m]["color"],
                   label=STYLE[m]["label"], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\n(n={val.get((methods[0], b), {}).get('n_images', '?')})" for b in use_bins])
        ax.set_xlabel("birds per image")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    if key == "count_mae":
        axes[1].set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "e_density")


def failures():
    rows = read_csv("results/failure_categories_test.csv")
    if not rows:
        return
    cats = [("fn_dense", "missed, dense flock", "#184f95"), ("fn_isolated", "missed, isolated", "#86b6ef"),
            ("fp_background", "false alarm, background", "#a4380f"),
            ("fp_localisation", "false alarm, poor box", "#eb6834"),
            ("fp_duplicate", "false alarm, duplicate", "#f5b393")]
    fig, ax = plt.subplots(figsize=(SINGLE, 1.9))
    names = [STYLE[r["method"]]["label"] for r in rows]
    left = np.zeros(len(rows))
    for key, label, color in cats:
        vals = np.array([float(r[key]) for r in rows])
        ax.barh(names, vals, left=left, color=color, label=label, edgecolor="white", linewidth=0.8, height=0.6)
        left += vals
    for y, total in enumerate(left):
        ax.text(total + 20, y, f"{int(total)}", va="center", fontsize=6, color=MUTED)
    ax.invert_yaxis()
    ax.set_xlabel("errors on the test set (at clean-val threshold)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2)
    fig.tight_layout()
    save(fig, "e_failures")


def gallery():
    folder = FIG / "gallery"
    methods = ("m1", "m2", "yolo", "faster_rcnn")
    if not folder.exists():
        print("[skip] gallery folder missing")
        return
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 3.8))
    for col, m in enumerate(methods):
        for row, pick in enumerate(("best", "worst")):
            ax = axes[row, col]
            ax.axis("off")
            imgs = sorted(folder.glob(f"{m}_{pick}_1_*.jpg"))
            if imgs:
                ax.imshow(plt.imread(imgs[0]))
                ax.set_title(f"{STYLE[m]['label']}: {'success' if pick == 'best' else 'failure'}", fontsize=6.5)
    fig.text(0.5, 0.0, "green = correct detection, red = false alarm, blue = missed bird",
             ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, "e_gallery")


if __name__ == "__main__":
    robustness()
    data_efficiency()
    density()
    failures()
    gallery()

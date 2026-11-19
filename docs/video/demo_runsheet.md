# Demo run-sheet (~2:00 of the video)

Screen-record a terminal at the repository root plus an image viewer (Finder Quick Look works).
Run every command once **before** recording so model weights and Python imports are warm.
The demo uses **val** images only (the test split is reserved for final evaluation), except
step 6, which re-draws test predictions that were already published.

Timings were measured once on the Apple M5 Pro (CPU) on 19 Nov 2026, while a training job was
running, so they are upper bounds.

| # | Time in video | Speaker | Command | Runtime | Show |
|---|---|---|---|---|---|
| 1 | 0:00–0:20 | Johnnie | `python scripts/run_m1.py --config configs/m1_seg_colour_tuned.yaml --split val --limit 3 --viz 3 --out results/demo/m1_val.json` | 0.5 s | One overlay from `results/demo/m1_val_viz/`: point out boxes on the birds and on rocks/foam |
| 2 | 0:20–0:40 | Member C | `python scripts/run_m2.py --config configs/m2_concat_gb_m1props_boxreg.yaml --split val --limit 3 --train-limit 300 --viz 3 --out results/demo/m2_val.json` | 45 s (trains first) | **Speed up 4× in the edit.** Overlay from `results/demo/m2_val_viz/` |
| 3 | 0:40–1:00 | Member D | `python scripts/run_ml.py --config configs/yolo_best.yaml --split val --limit 3 --viz 3 --device cpu --out results/demo/yolo_val.json` | 2.2 s | Overlay from `results/demo/yolo_val_viz/`: dense flock almost fully boxed |
| 4 | 1:00–1:15 | Member D | `python scripts/run_ml.py --config configs/faster_rcnn_best.yaml --split val --limit 3 --viz 3 --device cpu --out results/demo/frcnn_val.json` | 4.0 s | Same image as step 3, side by side |
| 5 | 1:15–1:40 | Johnnie | `python scripts/run_ml.py --config configs/yolo_best.yaml --split val --limit 3 --viz 3 --device cpu --data-root data/distorted/noise_s3 --out results/demo/yolo_val_noise3.json` | 2.4 s | Same image as step 3, now with σ = 50 noise: many birds lost. Then show `results/robustness/contact_noise.jpg` |
| 6 | 1:40–2:00 | Johnnie | `python scripts/make_overlays.py yolo --split test --pick worst --n 2 --out results/demo/overlays` | 0.3 s | `beach_bird_1790` overlay: blue (missed) boxes inside the flock, the main failure type |

Prerequisites (one-off, already done on the recording machine):

```bash
# copy models/yolo/best.pt and models/faster_rcnn/best.pt into place (not in Git; see models/README.md)
python scripts/make_distorted_roots.py         # step 5's distorted images
```

Overlay colours: `run_m1.py` and `run_ml.py --viz` draw ground truth in blue and predictions in
green; `run_m2.py --viz` and `make_overlays.py` draw green = correct, red = false alarm,
blue = missed.
Say this once on screen so viewers are not confused by the two colour schemes.

Outputs go to `results/demo/` (ignored by Git).

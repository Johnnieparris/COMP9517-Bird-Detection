# Method 2 — handcrafted features + a trained classifier (member C)

Detection as patch classification:

```
candidate boxes  ->  patch descriptor  ->  classifier score  ->  box regression  ->  NMS
```

Method 1 (member B) sets its accept/reject thresholds by hand. Here that boundary
is *learned*, which is the comparison the two methods are meant to isolate.

## Results

Best config `configs/m2_concat_gb_m1props_boxreg.yaml` — HOG+LBP+colour, gradient
boosting, 1 hard-negative round, Method 1 segmentation candidates, R-CNN box
regression at IoU 0.3.

| Split | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 | Count MAE | Train | Infer |
|---|---|---|---|---|---|---|---|---|
| val | **0.1848** | 0.0595 | 0.442 | 0.237 | 0.308 | 4.09 | 103 s | 79 ms/img |
| test | **0.1159** | 0.0384 | 0.347 | 0.136 | 0.196 | 4.76 | 105 s | 70 ms/img |

P, R, F1 and count MAE are at score threshold 0.715, tuned on val and then held
fixed for test — see the note on thresholds below. mAP is threshold-free.

Method 1 for comparison: 0.1024 on val, 0.0467 on test. Hardware for every number
is an Apple M1 Pro (10 core), 16 GB, CPU only, serial inference.

How it got there, each step a controlled comparison on val:

| Change | mAP@0.5 |
|---|---|
| own sliding window, random negatives | 0.0663 |
| + negatives sampled from the candidate boxes | 0.0754 |
| + Method 1 candidates instead of the window | 0.1472 |
| + R-CNN box regression | **0.1848** |

The full 17-variant ablation is in `results/m2_comparison.csv`, and the write-up
with the reasoning behind each step is in `docs/notes/m2-week9.md`.

## Files

```
src/ml/
  config.py      M2Config dataclass + load_config(); every knob lives here
  features.py    HOG, uniform LBP, HSV colour, and the three concatenated
  proposals.py   candidate boxes: sliding window, DoG blobs, or Method 1
  boxreg.py      R-CNN style ridge regression that corrects box size
  detector.py    BirdDetector: ties it together, fit() and detect()

scripts/
  run_m2.py                train on train, run on another split, save predictions
  sweep_m2.py              the whole ablation grid, then the comparison table
  compare_m2.py            prediction files -> results/m2_comparison.csv
  proposal_recall_m2.py    recall ceiling of each proposal source
  plot_features_m2.py      figure showing what the descriptors see

configs/m2_*.yaml          17 variants, one per row of the ablation
tests/ml/                  79 tests, no dataset needed (synthetic images)
```

## Running it

All commands from the repo root, using the shared venv:

```bash
PY=~/Downloads/COMP9517/.venv/bin/python3

# one variant on val, with metrics printed
$PY scripts/run_m2.py --config configs/m2_concat_gb_m1props_boxreg.yaml \
    --split val --score

# the full ablation grid, then rebuild results/m2_comparison.csv
$PY scripts/sweep_m2.py
tail -f results/m2_sweep_log.txt        # watch it

# overlays for the report and video (TP green, FP red, FN blue)
$PY scripts/run_m2.py --config configs/m2_concat_gb_m1props_boxreg.yaml \
    --split val --viz 6 --viz-thr 0.715

# recall ceiling of each proposal source
$PY scripts/proposal_recall_m2.py --split val

# the descriptor figure
$PY scripts/plot_features_m2.py --split val --n 3
```

Finished variants are skipped, so an interrupted sweep resumes where it stopped.
Add `--redo` to recompute everything.

The test split needs `--final` and refuses `--score`:

```bash
$PY scripts/run_m2.py --config configs/m2_concat_gb_m1props_boxreg.yaml \
    --split test --final --publish
```

That has been run **once**, for the frozen config above. Don't run it again
without a reason.

## Two things worth knowing

**Proposals cap everything.** Method 2 only scores boxes it is handed, so the
fraction of birds covered by some candidate is a hard ceiling on recall and
therefore mAP. Measured on val (`results/m2_proposal_recall.csv`):

| Source | recall@0.5 | candidates/img | ms/img |
|---|---|---|---|
| sliding window | 0.546 | 1,830 | 2.7 |
| blob (DoG) | 0.244 | 1,377 | 322 |
| Method 1 segmentation | 0.370 | 97 | 15.7 |

Method 1's candidates reach 68% of the window's recall from 5% of the boxes,
which makes inference about 19x cheaper for a lower ceiling. That trade won.

**Scores are not comparable to Method 1's at a fixed threshold.** Method 2 trains
with `class_weight="balanced"`, so its scores are balanced-prior posteriors and
most candidates sit above 0.5. At the shared evaluator's fixed `SCORE_THR = 0.5`
the count MAE looks like ~290 birds/image; at a threshold tuned on val the same
predictions give ~4.1. mAP@0.5 is threshold-free and unaffected, so that is the
number to compare. P/R/F1 are reported at the best-F1 threshold on val and that
threshold is recorded in the `score_threshold` column, which is what Method 1
does too.

## Dependencies

`scikit-learn` and `scikit-image`, both appended to the repo `requirements.txt`.
`pytest.ini` has `pythonpath = . src` so tests can import the shared `data`,
`evaluation` and `predictions` modules.

## References

`docs/refs/method2.bib`. The main ones: Dalal & Triggs (CVPR 2005) for HOG and
hard-negative mining, Ojala et al. (TPAMI 2002) for uniform LBP, Girshick et al.
(CVPR 2014) for box regression, Hosang et al. (PAMI 2015) on proposal quality.

# Member E — Advanced Studies, Discussion and Video: Design

Date: Thu 19 Nov 2026. Deadline: **Fri 20 Nov 2026, 18:00 AET** (team target: submit by midday).

## Intent

Member E dropped out. Their share of `PROJECT_PLAN.md` (advanced studies for the >27/30 band, the report's
Discussion and Conclusion, the discussion part of the video and the demo) is unowned. Johnnie (Member B) is
taking it on and will present E's video segment themselves. The work has to be finished, defensible under
questioning, and reproducible from the repo in roughly 24 working hours.

## Goals

1. **Study A — Robustness.** mAP@0.5 vs distortion severity for all four methods (M1, M2, YOLO11n,
   Faster R-CNN), plus one mitigation for the classical methods and one for YOLO, with measured effect.
2. **Study B — Data and label efficiency (YOLO only).** mAP@0.5 vs training fraction and vs label noise,
   plus one improvement with measured effect.
3. **Study C — Density, size and failure analysis.** Per-density-bin mAP and counting MAE, per-size recall,
   false positives on negative images, categorised failures, and a success/failure gallery per method.
4. **Report.** Discussion and Conclusion sections plus the robustness/data-efficiency results subsections,
   as LaTeX in the same style as `docs/report/member_b_sections.tex`, with ~4 figures.
5. **Video.** A ~1:45 script for the discussion/future-work segment, a slide outline, and a ~2 min demo run-sheet.

## Non-goals

- Retraining Faster R-CNN for anything (too slow on this machine in the time left).
- Data-efficiency experiments for M1/M2 (M1 has no training; M2 is weak enough that the curve says little).
- Transformer detectors, motion blur, new tuning of the existing methods.

## Anti-goals (failure even if it "works")

- **Any test-set leakage.** Mitigation settings, thresholds and checkpoints are chosen on (distorted) val only.
  Test is run once per final model/variant, through the existing `--final` guards.
- Numbers in the report that no script in the repo reproduces.
- Comparing training runs with different epoch budgets on the same curve.
- Missing the deadline because the study was too ambitious: the writing must not wait on the last training run.

## Constraints and current state

- Hardware: Apple M5 Pro, MPS available. Training time is the scarce resource.
- `.venv` currently has no torch/ultralytics. The README's conda `birds` env does not exist on this machine.
- Model weights are on `origin/ben/ml` as Git LFS objects; `git-lfs` is not installed. `models/yolo/best.pt`
  (~5.5 MB) and `models/faster_rcnn/best.pt` (~166 MB) must be pulled before Study A can run the DL models.
- `origin/ben/ml` is two commits ahead of `main` (new ML results + LFS tracking).
- Existing building blocks to reuse, not rewrite:
  - `src/evaluation.py` — shared evaluator (`load_records`, `average_precision`, `evaluate`, `counting_error`,
    `classify_detections`, `best_f1_threshold`, `run_and_time`).
  - `src/predictions.py` — common prediction CSV format and publishing.
  - `src/analysis.py` — per-size grouping and val-selected confidence thresholds.
  - `src/dl/inference.py::load_detector(config)` — returns `predict(image_rgb) -> N×5 XYXY+score`.
  - `src/dl/train_yolo.py::train(config_path, ...)` — YOLO training with timeline/provenance logging.
  - `scripts/run_m1.py`, `scripts/run_m2.py`, `scripts/run_ml.py` — all take `--data-root`, `--split`, `--final`.
  - M2 best config: `configs/m2_concat_gb_m1props_boxreg.yaml` (trains in ~100 s on CPU; no saved model).
- Locally only `predictions/m1_{val,test}.csv` exist; YOLO, Faster R-CNN and M2 predictions must be regenerated
  (val freely, test with `--final`).

## Approaches considered

Scope (chosen in conversation):

1. Robustness + failure analysis only — safest, but Study B missing weakens the >27 case.
2. **Robustness + YOLO-only data efficiency + failure analysis — chosen.** Covers all three advanced studies;
   the extra cost is ~8 YOLO trainings that run in the background while the robustness code is built.
3. Everything including Faster R-CNN retraining — not feasible by the deadline.

How distorted data is fed to methods (my call):

- **Chosen: write distorted copies to disk** under `data/distorted/<split>/<type>_s<level>/` in the same layout
  as the original dataset (images + labels), so every existing runner works unchanged via `--data-root`, and
  the existing test guards keep applying. Cost: disk space (≈ 15 × the split size; acceptable).
- Rejected: distorting on the fly inside a new evaluation loop — less disk, but bypasses the existing runners
  and guards and means re-implementing per-method inference paths.

## Design

Work happens on a new branch `member-e/advanced-studies` cut from `origin/ben/ml`.

### Study A — Robustness (`src/robustness/`)

**Distortions** (deterministic; RNG seeded from `(image_id, type, level)`), 5 types × 3 severities:

| Type | s1 | s2 | s3 |
|---|---|---|---|
| Gaussian noise (σ on 0–255) | 10 | 25 | 50 |
| Gaussian blur (σ px) | 1 | 2 | 4 |
| Low light: `I' = a·(I − μ) + b·μ` (contrast a, brightness b) | a=b=0.6 | 0.4 | 0.2 |
| JPEG quality | 30 | 15 | 5 |
| Occlusion: grey rectangle over a random part of each GT box, covering this fraction of its area | 25% | 50% | 75% |

Labels are copied unchanged (occluded birds stay labelled — the question is whether a partially visible bird
is still found). Generated for **val** (for tuning mitigations) and **test** (for reporting). A visual check
script saves one contact sheet per type.

**Evaluation.** Each method is run on each variant through its existing runner and scored with the shared
evaluator. Output: `results/robustness_{val,test}.csv` with columns
`method, variant, distortion, severity, mAP50, mAP50_95, precision, recall, count_mae, ms_per_image`.
Clean (severity 0) rows are included so curves start from the clean score.

Practical points (my calls, from the builder check):
- Each distorted data root also links in the **clean train split** (symlinks), because `run_m2.py` retrains on
  the train split under `--data-root` and the split loader requires every manifest image to exist. Only the
  evaluated split is distorted.
- Distorted predictions are published to a separate folder (`predictions/robustness/`), never the default
  `predictions/`, so clean predictions are not overwritten.
- Precision, recall and count MAE use each method's **clean-val-selected confidence threshold** (as
  `src/analysis.py` does). mAP is the headline metric and is threshold-free.
- `ms_per_image` from robustness runs is not reported if measured while training runs in the background;
  the report's timing numbers come from the existing clean timing runs.

**Mitigations.**
- *Classical (M1 and M2):* one image-level preprocessing step — denoising (non-local-means or bilateral) then
  CLAHE — applied before the whole pipeline. **One global setting** (my call), chosen to maximise mean mAP over
  clean val plus all 15 distorted val variants; no per-distortion tuning, since that would assume the distortion
  type is known at test time. For M2 the preprocessing is applied to training images too and the classifier is
  retrained on them (~100 s), so features are consistent between training and testing.
  Report clean and distorted test mAP with and without it (it may hurt clean performance; that is a finding).
- *YOLO:* a "robust" YOLO trained with distortion augmentation (the same 5 distortion families, random type and
  severity, applied with probability 0.5 per training image) under Study B's settings, compared against
  Study B's `de_100` run. Both select their checkpoint on **clean val** (my call — the same rule for both keeps
  the comparison fair). The original `models/yolo/best.pt` is also reported on the distortion suite as the
  official model.
- Faster R-CNN gets no mitigation (no retraining); it appears on the curves only.

### Study B — Data and label efficiency (`src/data_efficiency/`)

All runs: YOLO11n from COCO-pretrained weights, imgsz 640, seed 0, no offline augmentation copies, checkpoint
selected on clean val, then a single test evaluation each.

Hyperparameters (my call): the **optimiser and loss values** from `results/tuning/yolo_tune_v1/`
(lr0, lrf, momentum, weight_decay, warmup_epochs, box, cls, dfl), with **Ultralytics' default online
augmentation** (HSV, flip, scale, translate, mosaic) left on. The tuned config turned all online augmentation off
because it relied on offline copies; training with no augmentation at all would make every "improvement" a
trivial any-vs-none win. So `de_10_strong` and `robust_aug` are measured against a normally augmented baseline.

Training budget (my call): **matched optimiser steps, capped**. A full-data run gets E epochs (E = 30 to start).
A subset run with fraction f gets `min(E / f, 100)` epochs, so `de_50` = 60, `de_25` = 100, `de_10` = 100.
Fixed epochs would give `de_10` 10× fewer updates and confound "less data" with "less training"; full step
matching (300 epochs) does not fit the time. The cap is stated in the report as a limitation.

| Run | Training data |
|---|---|
| `de_100` | all 1,369 train images (budget-matched baseline) |
| `de_50`, `de_25`, `de_10` | nested random subsets (10 ⊂ 25 ⊂ 50), seeded |
| `ln_jitter20` | all images; each box edge shifted by uniform ±20% of box width/height, clipped to image |
| `ln_drop20` | all images; 20% of boxes removed uniformly at random |
| `de_10_strong` | 10% subset with stronger augmentation (heavier mosaic, scale, HSV, plus copy-paste) — the improvement |
| `robust_aug` | all images with distortion augmentation (Study A mitigation) |

Epoch budget: start at E = 30. Time the first epoch of `de_100`; if the projected total for all 8 runs
(full-data runs × E epochs plus the subset runs' scaled epochs) exceeds ~5 h, reduce E for **all** runs, with
the subset epochs recomputed from the same rule, before starting the rest. Run order puts the runs the report most needs first:
`de_100`, `de_10`, `de_25`, `de_50`, `robust_aug`, `ln_jitter20`, `ln_drop20`, `de_10_strong`.
Training runs in the background and logs wall-clock time; results go to `results/data_efficiency_test.csv`.

### Study C — Density, size and failure analysis (`src/analysis_e/` or extend `src/analysis.py`)

On the final clean test predictions of all four methods, using val-selected confidence thresholds as
`src/analysis.py` already does:
- mAP@0.5 and counting MAE per density bin `0, 1–10, 11–50, 51–100, >100` (bin `0` reports FPs per image,
  since AP is undefined without positives).
- Recall by object size (reuse `_size_groups`).
- Failure categories, assigned automatically per error:
  - **FN in a dense cluster** — missed GT whose centre has ≥ 3 other GT centres within 2× its diagonal.
  - **FN isolated** — any other miss.
  - **Duplicate** — FP with IoU ≥ 0.5 to a GT already matched by another prediction.
  - **Localisation** — FP with 0.1 ≤ IoU < 0.5 to some GT.
  - **Background FP** — FP with IoU < 0.1 to every GT (rock/shell/wave/etc.).
- A gallery: 2 successes and 2 failures per method, rendered with the existing overlay colours
  (green TP, red FP, blue FN), chosen by rule (highest / lowest per-image F1 among images with ≥ 1 bird).

### Report (`docs/report/member_e_sections.tex`)

Same style as `member_b_sections.tex` (subsections, `\cite` keys added to a `docs/refs/member_e.bib`).
Contents: Experimental Results subsections for robustness and data/label efficiency; Discussion (why YOLO wins,
speed–accuracy trade-off, what the studies reveal, failure figure with explanations); Conclusion and future work.
Page budget: ~1.5 pp Discussion, 0.5 pp Conclusion, ~1 pp for the two results subsections.

Figures (PDF + PNG in `docs/report/figures/`), produced by one script:
1. mAP@0.5 vs severity, one panel per distortion, one line per method (+ dashed mitigated lines).
2. mAP@0.5 vs training fraction, with label-noise and improvement runs as marked points.
3. Per-density-bin mAP and counting MAE for the four methods.
4. Failure gallery / failure-category bar chart.

Writing starts as soon as Study A test numbers exist; Study B numbers are dropped in as runs finish.

### Video (`docs/video/`)

- `member_e_script.md` — ~1:45 spoken script for the discussion/future-work segment, written for Johnnie to
  read, opening with their name, and noting that they are presenting E's share.
- `member_e_slides.md` — slide-by-slide outline (3–4 slides) matching the figures.
- `demo_runsheet.md` — exact commands and ordering for a ~2 min demo of each method running on a few images,
  using the existing `--viz` / overlay tooling.

## Left to the builder

- Environment setup (install torch/ultralytics etc. into `.venv` from the repo's requirements, `git-lfs`).
- Exact module/script names inside `src/robustness/`, `src/data_efficiency/` and `scripts/`.
- How the YOLO training config for each Study B run is expressed (YAML per run vs. CLI overrides), as long as
  each run's settings are recorded in `results/`.
- The exact "strong augmentation" values, within the spirit above.
- Whether Study C extends `src/analysis.py` or lives beside it.
- Tests: unit tests for distortion determinism/shape, subset nesting, label-noise injection and failure
  categorisation, in the existing `tests/` layout.

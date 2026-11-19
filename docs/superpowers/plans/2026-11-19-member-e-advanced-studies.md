# Member E Advanced Studies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Robustness, data/label-efficiency and failure studies across the team's four detectors, plus Member E's report sections, figures, video script and demo run-sheet, by Fri 20 Nov midday.

**Architecture:** Distorted / preprocessed / subsetted / label-noised copies of the dataset are written as ordinary data roots (same `images/{split}` + `labels/{split}` layout), so the existing runners (`run_m1.py`, `run_m2.py`, `run_ml.py`, `src.dl.train_yolo`) work on them via `--data-root` with their test guards intact. Unchanged images are hard-linked, not copied or symlinked (the YOLO trainer calls `Path.resolve()`, which follows symlinks back to the original labels). Small scoring modules turn published prediction CSVs into results CSVs, and one figure script makes every report figure.

**Tech Stack:** Python 3.12 `.venv`, NumPy, OpenCV, PyTorch (MPS), Ultralytics 8.4.211, scikit-learn/scikit-image (M2), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-19-member-e-advanced-studies-design.md`

## Global Constraints

- Test split touched only through runners with `--final`, once per final model/variant. Never tune, select a threshold or select a checkpoint on test.
- Confidence thresholds for P/R/count MAE: each method's best-F1 threshold on **clean val** (`evaluation.best_f1_threshold`), as `src/analysis.py` does. mAP@0.5 is the headline metric.
- Distortions: 5 types × severities exactly as the spec table (noise σ 10/25/50; blur σ 1/2/4; low light a=b=0.6/0.4/0.2; JPEG q 30/15/5; occlusion 25/50/75% of each GT box area). Labels unchanged.
- Determinism: every random draw is seeded from `(seed, image_id, tag)` through `zlib.crc32`, never Python `hash()`.
- Distorted predictions go under `predictions/robustness/<variant>/`; Study B under `predictions/data_efficiency/<run>/`. Never overwrite `predictions/<method>_<split>.csv` except in Task 5.
- Study B: YOLO11n from `yolo11n.pt`, imgsz 640, seed 0, `augmentation_copies: 0`; optimiser/loss values from `results/tuning/yolo_tune_v1/best_training_config.yaml` (lr0, lrf, momentum, weight_decay, warmup_epochs, box, cls, dfl, optimizer AdamW, cos_lr); Ultralytics default online augmentation on. Full-data runs E epochs (E=30 unless the Task 4 timing rule lowers it); subset fraction f gets `min(round(E / f), 100)` epochs. Checkpoint = `checkpoints/best_shared.pt` (selected on clean val by the existing `MilestoneRecorder`).
- Generated data lives under `data/` (gitignored). Weights are never committed.
- Do not report `ms_per_image` from runs measured while training is running in the background.

## Review Focus

1. **Images with no birds** (negatives) through every transform: occlusion with zero GT boxes, jitter/drop on empty label files, failure categorisation with no GT — must return the image/labels unchanged, not crash. Tests in Tasks 1, 3, 9.
2. **Jittered boxes at image borders** must be clipped to the image and keep width/height ≥ 1 px, or YOLO drops/rejects labels silently. Test in Task 3.
3. **Distortion output dtype/range**: every distortion returns `uint8`, same shape, values clipped (noise and low-light overflow wrap otherwise). Test in Task 1.
4. **Re-running a generator** on an existing output root must refuse (not silently mix old and new images). Test in Task 2.
5. **Hard-link fallback across filesystems**: if `os.link` fails (`OSError`), copy instead. Test in Task 2.

---

### Task 0: Environment, branch and weights

**Files:** none created (environment only).

- [ ] **Step 1:** `git switch -c member-e/advanced-studies origin/ben/ml` (stash/leave the untracked `docs/report/`, `docs/video/`, `scripts/make_m1_pipeline_figure.py` and the modified `docs/refs/method1.bib` in place; carry them across, do not commit them in this branch unless the user asks). Copy the spec and this plan onto the branch.
- [ ] **Step 2:** `brew install git-lfs && git lfs install && git lfs pull`. Verify: `ls -l models/yolo/best.pt models/faster_rcnn/best.pt` shows ~5.5 MB and ~166 MB.
- [ ] **Step 3:** Install into `.venv`: `torch torchvision ultralytics==8.4.211 scikit-learn==1.9.1 scikit-image==0.26.0`. Verify: `.venv/bin/python -c "import torch,ultralytics;print(torch.backends.mps.is_available())"` prints `True`.
- [ ] **Step 4:** `.venv/bin/python -m pytest -q` — record the baseline pass/fail count (pre-existing failures are not ours to fix).
- [ ] **Step 5:** Smoke: `.venv/bin/python scripts/run_ml.py --config configs/yolo_best.yaml --split val --limit 5 --score` and the same for `configs/faster_rcnn_best.yaml`. Expected: both print scores without error.

### Task 1: Distortion functions

**Files:**
- Create: `src/robustness/__init__.py`, `src/robustness/distortions.py`
- Test: `tests/robustness/test_distortions.py`

**Interfaces:**
- Produces:
  - `DISTORTIONS: dict[str, tuple]` — `{"noise": (10, 25, 50), "blur": (1, 2, 4), "lowlight": (0.6, 0.4, 0.2), "jpeg": (30, 15, 5), "occlusion": (0.25, 0.5, 0.75)}`
  - `rng_for(seed: int, image_id: str, tag: str) -> np.random.Generator` (crc32 of `f"{seed}|{image_id}|{tag}"`)
  - `apply_distortion(image_rgb: np.ndarray, boxes: np.ndarray, name: str, severity: int, rng) -> np.ndarray` — severity is 1-based (1..3); returns uint8 RGB, same shape.
  - `variant_name(name: str, severity: int) -> str` → `"noise_s1"` etc.

- [ ] **Step 1: Write failing tests**

```python
def test_every_variant_keeps_shape_and_uint8():        # all 15 variants on a random 64x80x3 image + 2 boxes
def test_deterministic_for_same_seed_and_id():           # rng_for(0,"a","noise_s2") twice -> identical output
def test_differs_across_image_ids():                     # noise output differs for ids "a" vs "b"
def test_noise_clips_not_wraps():                        # all-255 image + noise s3 -> max == 255, no small values from wrap (mean > 200)
def test_lowlight_scales_mean():                         # constant-128 image, lowlight s3 -> mean ≈ 0.2*128 (±2)
def test_occlusion_covers_requested_fraction():          # one 40x40 box on a white image, s2 -> 0.5*1600 ±10% pixels changed inside box, none outside
def test_occlusion_without_boxes_is_identity():          # boxes shape (0,4) -> output equals input
def test_unknown_name_or_severity_raises():               # ValueError
```

- [ ] **Step 2:** Run `pytest tests/robustness/test_distortions.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement. Approaches: noise = add `rng.normal(0, σ)` in float then clip; blur = `cv2.GaussianBlur(ksize=(0,0), sigmaX=σ)`; low light `I' = a·(I − μ) + b·μ` with μ the per-image grey mean, clipped; JPEG = `cv2.imencode('.jpg', BGR, [IMWRITE_JPEG_QUALITY, q])` round-trip; occlusion = per box, a mid-grey (128) axis-aligned rectangle of area `frac·w·h` with the box's aspect ratio, placed at a uniformly random position fully inside the box.
- [ ] **Step 4:** Run tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(robustness): deterministic image distortions`.

### Task 2: Data-root builder and distorted roots

**Files:**
- Create: `src/robustness/roots.py`, `scripts/make_distorted_roots.py`, `scripts/robustness_contact_sheets.py`
- Test: `tests/robustness/test_roots.py`

**Interfaces:**
- Consumes: `apply_distortion`, `rng_for`, `DISTORTIONS`, `variant_name` (Task 1); `src.data.dataset.load_split, get_image, DEFAULT_ROOT`.
- Produces:
  - `link_or_copy(src: Path, dst: Path) -> None` — `os.link`, falling back to `shutil.copy2` on `OSError`.
  - `build_root(src_root: Path, dst_root: Path, transforms: dict[str, Callable[[np.ndarray, np.ndarray, str], np.ndarray] | None], label_fn: Callable[[np.ndarray, str, str], np.ndarray] | None = None, info: dict | None = None) -> Path` — for each split in `("train","val","test")`: if `transforms[split]` is None, hard-link images; else write `transform(image_rgb, boxes, image_id)` as JPEG quality 95 (note: re-encoding is part of the variant; the clean reference is the original file). Labels are copied, or rewritten from `label_fn(boxes_xyxy, image_id, split)` in YOLO format when given. Writes `dst_root/root_info.json` (`info` + source root + creation time). Raises `FileExistsError` if `dst_root` exists. Copies `dataset.yaml`.
  - `DISTORTED_DIR = REPO / "data/distorted"`; roots at `data/distorted/<variant>/` with distorted val+test and clean (linked) train.

- [ ] **Step 1: Write failing tests** (on a tiny fake dataset built in `tmp_path`: 2 images per split, one with an empty label file)

```python
def test_untransformed_split_is_hardlinked():      # os.stat(dst).st_ino == os.stat(src).st_ino for train images
def test_transformed_split_written_and_labels_kept(): # val image pixels differ; val label text identical
def test_label_fn_rewrites_labels():                # label_fn dropping all boxes -> empty label files
def test_refuses_existing_output():                 # second build_root to same dst -> FileExistsError
def test_link_falls_back_to_copy(monkeypatch):      # monkeypatch os.link to raise OSError -> file still exists, content equal
def test_loader_reads_built_root():                 # load_split(dst, "val") returns same ids and boxes as source
```

- [ ] **Step 2:** Run; expect FAIL. **Step 3:** Implement. **Step 4:** Run; expect PASS.
- [ ] **Step 5:** `scripts/make_distorted_roots.py [--seed 0] [--only noise_s1 ...]` builds all 15 roots (seed 0; rng tag = variant name). Run it. Verify: 15 folders, each `images/test` has 240 files and `images/val` 291.
- [ ] **Step 6:** `scripts/robustness_contact_sheets.py` saves `results/robustness/contact_<type>.jpg`: one fixed test image with GT boxes, clean + s1/s2/s3. Look at all 5 sheets; severity must visibly increase and occlusion must sit inside boxes.
- [ ] **Step 7:** Commit `feat(robustness): distorted dataset roots` (code + contact sheets; ensure `results/robustness/*.jpg` is allowed or force-add only those 5 small files).

### Task 3: Study B data roots, trainer change, run configs

**Files:**
- Create: `src/data_efficiency/__init__.py`, `src/data_efficiency/subsets.py`, `src/data_efficiency/label_noise.py`, `scripts/make_e_training_roots.py`, `configs/e/base.yaml`, `configs/e/<run>.yaml` (one per run)
- Modify: `src/dl/train_yolo.py` (`prepare_data`)
- Test: `tests/data_efficiency/test_subsets.py`, `tests/data_efficiency/test_label_noise.py`, `tests/dl/test_train_yolo.py` (add)

**Interfaces:**
- Consumes: `build_root`, `link_or_copy` (Task 2); `apply_distortion`, `DISTORTIONS`, `rng_for` (Task 1).
- Produces:
  - `nested_subsets(ids: list[str], fractions=(0.5, 0.25, 0.1), seed=0) -> dict[float, list[str]]` — one seeded permutation; fraction f takes the first `round(f·n)`; so 10 ⊂ 25 ⊂ 50.
  - `jitter_boxes(boxes: np.ndarray, frac: float, rng, width: int, height: int) -> np.ndarray` — each edge shifted by `U(-frac, frac)·(box w or h)`, clipped to image, then forced to width/height ≥ 1 px.
  - `drop_boxes(boxes: np.ndarray, frac: float, rng) -> np.ndarray` — removes `round(frac·N)` boxes chosen uniformly.
  - Training roots under `data/e_training/<run>/`: `subset_10|25|50` (train images limited to the subset; val/test linked), `jitter20`, `drop20`, `robust_aug` (each train image independently replaced, p=0.5, by a distortion with uniform random type and severity, rng tag `"robust_aug"`).
  - Subset roots write `splits.json` (same format as `splits/official.json`) listing only the subset as `train`, plus `"subset_of_official": true`.
  - `prepare_data` accepts a manifest whose `train` is a strict subset of official train **only when** that manifest has `"subset_of_official": true`; every other check is unchanged.
  - `configs/e/<run>.yaml` for runs `de_100, de_50, de_25, de_10, ln_jitter20, ln_drop20, de_10_strong, robust_aug`: values from `base.yaml` + `data_root`, `epochs`. `base.yaml` = `best_training_config.yaml` minus every augmentation key (hsv_*, degrees, translate, scale, shear, perspective, flip*, mosaic, mixup, cutmix, copy_paste, bgr, close_mosaic), with `augmentation_copies: 0`, `seed: 0`, `device: mps`, `workers: 4`, `amp: false`. `de_10_strong` adds `mosaic: 1.0, scale: 0.9, hsv_v: 0.6, hsv_s: 0.9, copy_paste: 0.3, mixup: 0.1, degrees: 10`.

- [ ] **Step 1: Write failing tests**

```python
def test_subsets_nested_and_sized():          # 1369 ids -> sizes 685/342/137; set(10%) <= set(25%) <= set(50%)
def test_subsets_deterministic():              # same seed -> same lists
def test_jitter_clips_to_image_and_min_size(): # box at (0,0,5,5), frac 0.2 many draws -> all x1>=0, x2<=W, x2-x1>=1
def test_jitter_empty_boxes():                 # (0,4) in -> (0,4) out
def test_drop_fraction():                      # 10 boxes, frac .2 -> 8 boxes, all rows from the input
def test_drop_empty_boxes():                   # (0,4) -> (0,4)
def test_prepare_data_accepts_flagged_subset(tmp_path): # manifest with subset_of_official true -> train count == subset size
def test_prepare_data_rejects_unflagged_subset(tmp_path): # same manifest without flag -> ValueError (existing message)
```

- [ ] **Step 2:** Run; expect FAIL. **Step 3:** Implement (the `prepare_data` change is one condition around the existing `issubset` check). **Step 4:** Run; expect PASS, plus the existing `tests/dl/test_train_yolo.py` still passing.
- [ ] **Step 5:** Run `scripts/make_e_training_roots.py` (writes the 6 roots and the 8 configs with E=30). Verify: `subset_10/images/train` has 137 files; `drop20` label line count ≈ 0.8 × original.
- [ ] **Step 6:** Commit `feat(data-efficiency): training roots, label noise and subset-aware trainer`.

### Task 4: Launch Study B training (background)

**Files:**
- Create: `scripts/train_e_runs.sh`

- [ ] **Step 1:** Script runs, sequentially, `python -m src.dl.train_yolo --config configs/e/<run>.yaml --name e_<run> --output-root results/e_runs` in the spec order: `de_100, de_10, de_25, de_50, robust_aug, ln_jitter20, ln_drop20, de_10_strong`; skips a run whose `results/e_runs/e_<run>/run.json` says `completed`; continues past a failed run and logs to `results/e_runs/train_e_runs.log`.
- [ ] **Step 2:** Start it in the background. After `de_100` epoch 1's `Shared validation` line, read `elapsed_s` from `results/e_runs/e_de_100/history.json`.
- [ ] **Step 3: Timing rule.** Projected total = `t_epoch × (5·E + Σ_subsets f·min(round(E/f),100))` (five full-data runs; subsets scaled by their size). If > 5 h: stop the script, delete `results/e_runs/e_de_100`, lower E to the largest value that fits (minimum 10), regenerate configs (`scripts/make_e_training_roots.py --configs-only --epochs E`), restart. Record the chosen E in the plan's notes and in `results/e_runs/README.md`.
- [ ] **Step 4:** Commit the script and `results/e_runs/README.md` (not the run folders).

### Task 5: Clean baseline predictions for all four methods

Runs while training continues (CPU-bound methods only; YOLO/Faster R-CNN inference on CPU via `--device cpu` to stay off the GPU).

**Files:** none new; outputs in `predictions/`.

- [ ] **Step 1:** Val (free): `run_ml.py --config configs/yolo_best.yaml --split val --publish --device cpu`; same for `configs/faster_rcnn_best.yaml`; `run_m2.py --config configs/m2_concat_gb_m1props_boxreg.yaml --split val --publish --method-name m2`. (`predictions/m1_val.csv` already exists from `configs/m1_seg_colour_tuned.yaml`; regenerate with `run_m1.py ... --split val --publish` if `check_prediction_file` complains.)
- [ ] **Step 2:** Test, once each, with `--final --publish` (same commands, `--split test`). These are the official final clean predictions; ML ones must reproduce `results/ml_test_comparison.csv` mAP50 (YOLO 0.8321, Faster R-CNN 0.8251) to 3 d.p.; M2 must match 0.1159 to 2 d.p. If not, stop and investigate before continuing.
- [ ] **Step 3:** `python src/analysis.py`. Verify `analysis/summary.csv` has 8 rows (4 methods × val/test) and `analysis/density_bins.csv` has the 5 bins per method/split.

### Task 6: Robustness evaluation (unmitigated)

**Files:**
- Create: `src/robustness/score.py`, `scripts/run_robustness.py`
- Test: `tests/robustness/test_score.py`

**Interfaces:**
- Consumes: Task 2 roots; `evaluation.load_records, average_precision, map_50_95, precision_recall_f1, best_f1_threshold`; `predictions.load_predictions`.
- Produces:
  - `clean_val_thresholds(methods: list[str]) -> dict[str, float]` — best-F1 threshold from `predictions/<m>_val.csv`.
  - `score_variant(method: str, variant: str, split: str, root: Path, folder: Path, thr: float) -> dict` with keys `method, variant, distortion, severity, mAP50, mAP50_95, precision, recall, count_mae`; `variant == "clean"` → `distortion="clean", severity=0`.
  - `scripts/run_robustness.py --methods m1 m2 yolo faster_rcnn [--variants ...] [--tag base]` runs each method's existing runner with `--split test --final --data-root data/distorted/<variant> --publish predictions/robustness/<variant>` (M2 with `--method-name m2`), skipping a variant/method whose CSV exists, then writes `results/robustness_test.csv` (clean rows read from `predictions/`). YOLO and Faster R-CNN with `--device cpu` while training is running.
  - Method → runner map (one dict in the script): `m1` → `run_m1.py --config configs/m1_seg_colour_tuned.yaml`; `m2` → `run_m2.py --config configs/m2_concat_gb_m1props_boxreg.yaml --method-name m2`; `yolo` → `run_ml.py --config configs/yolo_best.yaml`; `faster_rcnn` → `run_ml.py --config configs/faster_rcnn_best.yaml`. Later tasks add entries (`m1pre`, `m2pre`, `e_de_100`, `e_robust_aug`). `run_ml.py` always publishes as `yolo_<split>.csv`, so for any `run_ml` method the script publishes to a temporary folder and moves the file to `<variant folder>/<method>_test.csv`.

- [ ] **Step 1: Failing tests** (fake 2-image dataset + hand-made prediction CSVs)

```python
def test_score_variant_perfect_predictions():   # GT-as-predictions, score 1.0 -> mAP50 == 1.0, count_mae == 0
def test_score_variant_parses_variant_name():   # "jpeg_s3" -> distortion "jpeg", severity 3; "clean" -> ("clean", 0)
def test_threshold_from_val_not_test():         # thresholds read only from *_val.csv files (test file absent -> still works)
```

- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Run `scripts/run_robustness.py --methods m1 m2 yolo faster_rcnn` (long; background). Verify `results/robustness_test.csv` has 4 × 16 rows and clean rows equal Task 5's `analysis/summary.csv` test mAP50.
- [ ] **Step 6:** Commit code + `results/robustness_test.csv`.

### Task 7: Classical mitigation (preprocessing)

**Files:**
- Create: `src/robustness/preprocess.py`, `scripts/tune_preprocess.py`, `scripts/make_preprocessed_roots.py`
- Test: `tests/robustness/test_preprocess.py`

**Interfaces:**
- Consumes: Task 1, 2, 6; `src.classical.pipeline.load_config, detect`; `evaluation.average_precision`.
- Produces:
  - `PREPROCESS_GRID: dict[str, dict]` — `{"none": {}, "clahe": {"clahe_clip": 2.0}, "bilateral_clahe": {"bilateral_d": 7, "sigma": 50, "clahe_clip": 2.0}, "nlm_clahe": {"nlm_h": 10, "clahe_clip": 2.0}, "nlm_strong_clahe": {"nlm_h": 20, "clahe_clip": 3.0}}`
  - `preprocess(image_rgb: np.ndarray, setting: dict) -> np.ndarray` — optional denoise (`cv2.bilateralFilter` or `cv2.fastNlMeansDenoisingColored`), then CLAHE on the L channel of Lab (tile 8×8); uint8 RGB out.
  - `scripts/tune_preprocess.py` — for each grid setting, M1 (`configs/m1_seg_colour_tuned.yaml`, in-process `detect` on BGR) on **val** of clean + 15 distorted roots; writes `results/robustness_preprocess_tuning_val.csv` (setting × variant mAP50) and prints the setting with the highest mean over the 16. That setting is final; record it in `results/robustness_preprocess_choice.json`.
  - Preprocessed roots `data/preprocessed/<variant>/` for clean + 15 variants: all three splits preprocessed (train too, so M2 retrains on preprocessed images).
  - Methods published as `m1pre` / `m2pre` into the same `predictions/robustness/<variant>/` folders (and `predictions/robustness/clean/` for the clean preprocessed run); rows appended to `results/robustness_test.csv`.

- [ ] **Step 1: Failing tests**

```python
def test_preprocess_none_is_identity():        # setting {} -> output equals input
def test_preprocess_keeps_shape_uint8():       # every PREPROCESS_GRID setting
def test_clahe_raises_contrast_on_dark_image(): # low-contrast dark image -> std increases
```

- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Run tuning; then build preprocessed roots for the chosen setting; then `run_robustness.py --methods m1pre m2pre` (the script maps `m1pre`/`m2pre` to the M1/M2 runners with `--data-root data/preprocessed/<variant>` and the method name). Skip entirely if the chosen setting is `none` and say so in the results (that is a finding).
- [ ] **Step 6:** Commit code + tuning CSV + choice JSON + updated `results/robustness_test.csv`.

### Task 8: Study B evaluation (after training finishes)

**Files:**
- Create: `src/data_efficiency/score.py`, `scripts/eval_e_runs.py`
- Test: `tests/data_efficiency/test_score.py`

**Interfaces:**
- Consumes: Task 3 runs in `results/e_runs/e_<run>/` (`run.json`, `best_shared.json`, `checkpoints/best_shared.pt`); Task 6 `score_variant`, `run_robustness.py`.
- Produces:
  - `configs/e/infer_<run>.yaml` = `{method: yolo, checkpoint: results/e_runs/e_<run>/checkpoints/best_shared.pt, imgsz: 640}` (generated).
  - `run_summary(run_dir: Path) -> dict` with `run, train_images, epochs, best_epoch, val_mAP50, train_time_s` (from `run.json`/`history.json`; `train_time_s` = final `elapsed_s`).
  - `scripts/eval_e_runs.py`: for each completed run, `run_ml.py --config configs/e/infer_<run>.yaml --split test --final --publish predictions/data_efficiency/<run>`; writes `results/data_efficiency_test.csv` = `run_summary` + test `mAP50, mAP50_95, precision, recall, count_mae` (threshold from that run's clean-val `best_shared_predictions.csv`). Then runs the distortion suite (Task 6 script, methods `e_de_100` and `e_robust_aug`, each via its infer config) and appends to `results/robustness_test.csv`.

- [ ] **Step 1: Failing test**

```python
def test_run_summary_reads_run_dir(tmp_path):   # fake run.json/history.json -> expected dict values
```

- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Run once all runs finish (or on whichever are complete if time runs out; missing runs are listed as "not completed" in the CSV and report, never estimated).
- [ ] **Step 6:** Commit code + CSVs.

### Task 9: Failure categorisation and gallery

**Files:**
- Create: `src/robustness/failures.py`, `scripts/failure_analysis.py`
- Test: `tests/robustness/test_failures.py`

**Interfaces:**
- Consumes: `evaluation.classify_detections, box_iou`; clean test predictions (Task 5); clean-val thresholds (Task 6).
- Produces:
  - `categorise(preds: np.ndarray, gts: np.ndarray, thr: float, iou_thr: float = 0.5) -> dict[str, int]` with keys `fn_dense, fn_isolated, fp_duplicate, fp_localisation, fp_background, tp`. Rules exactly as the spec: dense = ≥3 other GT centres within 2× the box diagonal; duplicate = FP with IoU ≥ 0.5 to an already-matched GT; localisation = FP with 0.1 ≤ max IoU < 0.5; background = max IoU < 0.1.
  - `results/failure_categories_test.csv` (method × category counts, plus `fp_per_negative_image`).
  - Gallery: for each method, `scripts/make_overlays.py <method> --split test --pick best --n 2` and `--pick worst --n 2` at that method's clean-val threshold (images with ≥1 bird for "best"); copy the chosen PNGs into `docs/report/figures/gallery/`.

- [ ] **Step 1: Failing tests**

```python
def test_isolated_and_dense_misses():   # 1 lone GT + cluster of 5 GTs, no preds -> fn_isolated 1, fn_dense 5
def test_duplicate_fp():                # 2 preds on one GT -> tp 1, fp_duplicate 1
def test_localisation_fp():             # pred IoU 0.3 with GT -> fp_localisation 1, fn 1 (isolated)
def test_background_fp_on_negative():   # no GT, 1 pred -> fp_background 1
def test_empty_everything():            # no GT, no preds -> all zeros
```

- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Run script; commit code + CSV + gallery images (check total size < 3 MB; downscale if needed).

### Task 10: Report figures

**Files:**
- Create: `scripts/make_e_figures.py`

**Interfaces:**
- Consumes: `results/robustness_test.csv`, `results/data_efficiency_test.csv`, `analysis/density_bins.csv`, `results/failure_categories_test.csv`, gallery images.
- Produces (PDF + PNG, 3.5 in wide single-column or 7.16 in double-column, 8 pt fonts, colour-blind-safe fixed colour per method used in every figure): `docs/report/figures/e_robustness.pdf` (5 panels, mAP50 vs severity 0–3, solid = base methods, dashed = `m1pre`/`m2pre`/`e_robust_aug`), `e_data_efficiency.pdf` (mAP50 vs train fraction for de_*, plus markers for ln_jitter20, ln_drop20, de_10_strong), `e_density.pdf` (mAP50 and count MAE per density bin, 4 methods), `e_failures.pdf` (stacked bars of categories per method + 2×4 gallery thumbnails).

- [ ] **Step 1:** Write the script; it must run end-to-end with partial data (missing runs simply absent, with a printed warning).
- [ ] **Step 2:** Run it; open each PNG and check legibility at print size. Commit.

### Task 11: Report sections

**Files:**
- Create: `docs/report/member_e_sections.tex`, `docs/refs/member_e.bib`

- [ ] **Step 1:** Read `docs/report/member_b_sections.tex` for macros/style and match it.
- [ ] **Step 2:** Write: Results subsections "Robustness to image distortions" and "Data and label efficiency" (~1 page total, the two figures, a compact table of mean mAP drop per distortion); Discussion (~1.5 pp: why YOLO wins — learned multi-scale features vs handcrafted proposals, speed–accuracy trade-off with ms/img from `results/ml_test_comparison.csv` and M1/M2 READMEs; density figure and failure figure with explanations; what robustness and data-efficiency reveal; limitations including the 100-epoch cap, single seed, imported-checkpoint caveats from `results/README.md`); Conclusion + future work (~0.5 p). Every number cited must come from a committed CSV; put the CSV path in a LaTeX comment next to it.
- [ ] **Step 3:** Bib entries at least: Hendrycks & Dietterich 2019; Zuiderveld 1994 (CLAHE); Buades et al. 2005 (NL-means); Tomasi & Manduchi 1998 (bilateral); Ultralytics YOLO11; Ghiasi et al. 2021 (copy-paste); Rolnick et al. 2017 or Frénay & Verleysen 2014 (label noise).
- [ ] **Step 4:** Compile check: `pdflatex` a minimal IEEEtran wrapper including the file (if `pdflatex` is not installed, check balanced braces/environments with a quick script and leave the wrapper for Overleaf). Commit.

### Task 12: Video and demo materials

**Files:**
- Create: `docs/video/member_e_script.md`, `docs/video/member_e_slides.md`, `docs/video/demo_runsheet.md`

- [ ] **Step 1:** Script ~1:45 at ~150 wpm (≈ 260 words), opens with "I'm Johnnie Parris" and that they are presenting the advanced-studies share; one claim per slide; every number matches the report.
- [ ] **Step 2:** Slides outline: 3–4 slides (robustness figure, data-efficiency figure, failure figure, future work), each with title, figure path, 2–3 bullet points.
- [ ] **Step 3:** Demo run-sheet ≈ 2 min: exact commands in order (`run_m1.py`, `run_m2.py`, `run_ml.py` YOLO and Faster R-CNN, each on `--split val --limit 3 --viz 3`; one distorted-root run; `make_overlays.py`), expected runtime of each, which overlay to show, and who speaks. Time each command once and record it.
- [ ] **Step 4:** Commit.

### Task 13: README and final checks

**Files:**
- Modify: `README.md` (add "Advanced studies (Member E)" section: commands for Tasks 2–10 in order, outputs, third-party code: Ultralytics, OpenCV CLAHE/NL-means/bilateral)

- [ ] **Step 1:** Write the README section.
- [ ] **Step 2:** `pytest -q` — all new tests pass; no new failures versus Task 0's baseline.
- [ ] **Step 3:** Leakage review: `grep -rn "split.*test" scripts/run_robustness.py scripts/eval_e_runs.py scripts/tune_preprocess.py` — tuning script never reads test; test only via runners with `--final`.
- [ ] **Step 4:** Size check: `git archive HEAD | wc -c` — code ZIP well under 25 MB with no `*.pt`, images outside `docs/report/figures/`, or `data/`.
- [ ] **Step 5:** Commit. Do not push or open a PR without asking the user.

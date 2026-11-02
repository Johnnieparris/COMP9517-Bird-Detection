# Member B, Week 8: Method 1 Comparison and Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** By Sun 8 Nov, Method 1 compares at least two handcrafted approaches, each tuned on `val`, and its val predictions run through the shared evaluation script (or the hand-written one if A's is still missing).

**Architecture:** A new segmentation detector (foreground mask, morphology, connected components) sits beside the LoG detector behind the existing `detect(image_bgr, cfg)` interface, so C and E change nothing beyond an optional `load_config(path)`. A shared local-contrast score replaces the LoG response where the config asks for it. A parallel random search tunes each approach on `val`. Reported timings still come from the serial `scripts/run_m1.py`.

**Tech Stack:** Python 3.12, NumPy, SciPy, OpenCV, Pillow, PyYAML, pytest. No new dependencies.

**Spec:** `PROJECT_PLAN.md` (§3 Method 1, §4, §6 Week 8, §8, §10), `docs/notes/m1-week7.md`, and `../COMP9517_26T3_Group_Project_Specification.pdf`.

## Global Constraints

- Tune on `train`+`val` only. `test` is not run in Week 8: final configs freeze Wed 11 Nov, and the single test run happens in Week 9.
- Every script takes `--split`; `scripts/tune_m1.py` must refuse `test`.
- Prediction rows stay `image_id, x1, y1, x2, y2, score`, in original-image pixels, with a `.meta.json` sidecar.
- Max detections per image ≥ 300 in every shipped config.
- Report mAP@0.5, training time, inference ms/img and hardware for every variant. Training time is 0, and search time is reported separately.
- `results/` is gitignored except `*.csv`. Do not commit predictions, overlays or weights.
- Cite every paper and library used. Write the methods ourselves.
- `configs/m1_blob.yaml` and `configs/m1_blob_proposals.yaml` must keep producing their Week 7 results.

## Evidence Behind the Plan

A throwaway prototype ran on all 291 `val` images on 2 Nov, with the quick evaluator. It is not in the repo, so these numbers must not go in the report until the real code reproduces them. Blob variants use the shipped config's sigmas, threshold and NMS.

| Variant | AP@0.5 | R@0.5 | Det/img | Det/negative img | ms/img |
|---|---|---|---|---|---|
| Blob baseline | 0.0011 | 0.135 | 257 | 267 | 276 |
| Blob + contrast score | 0.0033 | 0.128 | 257 | 267 | 281 |
| Blob + edge ratio 10 + contrast | 0.0034 | 0.134 | 219 | 227 | 1309 |
| Adaptive threshold, block 51, C 10 | 0.0028 | 0.089 | 106 | 82 | 6 |
| Adaptive threshold, block 101, C 15 | 0.0115 | 0.173 | 92 | 76 | 5 |
| Lab colour distance + Otsu | **0.0178** | **0.295** | 84 | 84 | 12 |

**Decisions from this:**

1. **Colour-distance segmentation is the main approach.** It is about 16× the baseline AP, has twice the recall and runs about 20× faster.
2. **Contrast scoring goes in.** It triples blob AP at almost no time cost.
3. **The Hessian edge-ratio test is cut**, even though Week 7 ranked it first. It cleans up synthetic shapes (355 → 1 box on a disc), but on real images it adds about 1 s per image and almost no AP.
4. **The next gain is a floor on the Otsu threshold.** Negative images get as many detections as positive ones because Otsu always splits the image, even when nothing is there.
5. **Expect classical AP to stay low, in the 0.0x range.** Stop tuning at the time box. An honest analysis of why is what earns the marks, as §10 anticipates.

## Review Focus

1. **Plain background (no birds, mild sensor noise):** gives no detections, because of the Otsu floor. `test_plain_noisy_background_gives_no_detections` in Task 3.
2. **Huge foreground component (shoreline, sea, or a mask covering the frame):** is rejected instead of becoming one giant box. `test_full_frame_component_rejected` in Task 2.
3. **Boxes at the image corner, or on a perfectly flat region:** the score stays finite, because the ring is clipped and its variance can be 0. `test_local_contrast_finite_at_corner_and_on_flat` in Task 1.
4. **Dense flock of 130+ separate birds:** segmentation and the cap must not silently limit it. `test_seg_dense_grid_keeps_at_least_130` in Task 3.
5. **Small or non-square image, smaller than the background kernel:** the mask still matches the input shape. `test_colour_mask_matches_small_nonsquare_image` in Task 2.

## Schedule

| Day | Work |
|---|---|
| Mon 2 Nov | Before Task 1; Task 1 |
| Tue 3 Nov | Tasks 2 and 3 |
| Wed 4 Nov | Tasks 4 and 5 |
| Thu 5 Nov | Task 6: run searches and compare. **Tutor consult** with those numbers |
| Fri 6 Nov | Task 7: Member C proposals and evaluator cross-check |
| Sat 7 Nov | Task 8 (stretch), only if everything before it is done |
| Sun 8 Nov | Task 9: notes, references. **Milestone** |

If time runs short, cut in this order: Task 8, then the adaptive search in Task 6 (keep its hand-set result as the third row).

## File Structure

| File | Responsibility |
|---|---|
| `src/classical/scoring.py` (new) | `local_contrast`, shared by both detectors |
| `src/classical/segment.py` (new) | `colour_distance_mask`, `adaptive_mask`, `clean_mask`, `mask_to_boxes` |
| `src/classical/pipeline.py` | Add `score_mode`, `SegConfig`, `load_config`; `detect` dispatches on config type |
| `src/classical/quick_eval.py` | Add `best_f1`, `count_mae` |
| `src/classical/tuning.py` (new) | `sample_config`, `detect_parallel` |
| `scripts/tune_m1.py` (new) | Random search on `val`, writes a trials CSV and the best YAML |
| `scripts/compare_m1.py` (new) | Writes `results/m1_comparison.csv` from prediction files and sidecars |
| `scripts/run_m1.py` | Use `load_config` |
| `configs/m1_seg_colour.yaml`, `configs/m1_seg_adaptive.yaml`, `configs/m1_blob_contrast.yaml` (new) | Hand-set starting points |
| `configs/search/*.yaml` (new) | Search spaces |
| `configs/m1_*_tuned.yaml`, `configs/m1_seg_proposals.yaml` (new) | Tuned results, Member C's candidates |
| `docs/notes/m1-week8.md` (new), `docs/refs/method1.bib` | Results, handoff, references |

---

### Before Task 1: Repo Housekeeping

- [ ] **Step 1: Sync and check for A's work**

Run: `git fetch origin && git log --oneline origin/main -n 5 && git ls-tree -r --name-only origin/main | grep -E "src/(data|eval)|scripts/evaluate"`
Expected: on 2 Nov this shows only `PROJECT_PLAN.md` history and no matches. If A's loader has landed, merge it and swap out `interim_data.py` before starting Task 1.

- [ ] **Step 2: Ignore macOS metadata and commit this plan**

Add `.DS_Store` to `.gitignore`.

```bash
git add .gitignore docs/superpowers/plans/2026-11-02-member-b-week8-method1.md
git commit -m "docs(m1): add Week 8 plan and ignore .DS_Store"
```

- [ ] **Step 3: Push the branch, once the owner confirms**

`member-b/method1` isn't on the remote, so C and E can't use the Week 7 handoff yet.

```bash
git push -u origin member-b/method1
```

---

### Task 1: Local-contrast scoring

**Files:**
- Create: `src/classical/scoring.py`, `tests/classical/test_scoring.py`, `configs/m1_blob_contrast.yaml`
- Modify: `src/classical/pipeline.py`, `tests/classical/test_pipeline.py`

**Interfaces:**
- Produces: `local_contrast(lab: np.ndarray, boxes: np.ndarray, pad: float = 0.5, eps: float = 5.0) -> np.ndarray`, taking an H×W×3 Lab image and (N, ≥4) boxes and returning (N,) floats. Also `BlobConfig.score_mode: str = "log"`, one of `"log"` or `"contrast"`.

- [ ] **Step 1: Write the failing tests**

```python
def lab_of(image): return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

def test_bird_box_scores_above_background_box():
    image = np.full((200, 200, 3), 220, np.uint8)
    cv2.circle(image, (60, 60), 15, (20, 20, 20), -1)
    scores = local_contrast(lab_of(image), np.array([[45., 45, 75, 75], [130, 130, 160, 160]]))
    assert scores[0] > 5 * max(scores[1], 1e-6)

def test_local_contrast_finite_at_corner_and_on_flat():
    image = np.full((50, 50, 3), 128, np.uint8)
    boxes = np.array([[0., 0, 10, 10], [40, 40, 50, 50], [20, 20, 30, 30], [49, 49, 50, 50]])
    scores = local_contrast(lab_of(image), boxes)
    assert np.isfinite(scores).all() and (scores == 0).all()

def test_local_contrast_handles_no_boxes():
    assert local_contrast(np.zeros((10, 10, 3), np.uint8), np.zeros((0, 5))).shape == (0,)

# tests/classical/test_pipeline.py
def test_contrast_score_mode_replaces_log_response():
    image = bgr_discs(200, 200, [(100, 100)], r=15)
    log_score = detect(image, BlobConfig())[0, 4]
    contrast_score = detect(image, BlobConfig(score_mode="contrast"))[0, 4]
    assert contrast_score != log_score

def test_unknown_score_mode_raises():
    with pytest.raises(ValueError):
        detect(bgr_discs(64, 64, [(32, 32)], r=8), BlobConfig(score_mode="nope"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/classical/test_scoring.py tests/classical/test_pipeline.py -q`
Expected: FAIL. `local_contrast` is not defined and `score_mode` is an unknown argument.

- [ ] **Step 3: Implement `local_contrast` and `score_mode`**

The score is `‖mean(box) − mean(ring)‖₂ / (√(Σ var(ring, per channel)) + eps)`. The ring is the box grown by `pad × width` and `pad × height` on each side, clipped to the image, minus the box. Use `cv2.integral` on the image and its square for O(1) lookups per box. Round coordinates, and keep every box at least 1 px wide and tall. An empty ring (a box covering the whole image) scores 0.

In `detect`, apply the score after `filter_boxes` and before `nms`, so NMS ranks by the new score. Add `configs/m1_blob_contrast.yaml`: `m1_blob.yaml` plus `score_mode: contrast`.

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, including the 53 Week 7 tests.

- [ ] **Step 5: Reproduce the prototype on val**

Run: `.venv/bin/python scripts/run_m1.py --config configs/m1_blob_contrast.yaml --split val --out results/m1_val_blob_contrast.json --score`
Expected: AP@0.5 about 0.003 and recall about 0.13. If AP is still about 0.001, the score isn't reaching NMS.

- [ ] **Step 6: Commit**

```bash
git add src/classical/scoring.py src/classical/pipeline.py tests/classical/test_scoring.py tests/classical/test_pipeline.py configs/m1_blob_contrast.yaml
git commit -m "feat(m1): add local-contrast detection score"
```

---

### Task 2: Foreground masks and box extraction

**Files:**
- Create: `src/classical/segment.py`, `tests/classical/test_segment.py`

**Interfaces:**
- Produces, all masks being uint8 H×W arrays of 0 and 1:
  - `colour_distance_mask(image_bgr, downscale: int, bg_kernel: int, min_distance: float) -> mask`
  - `adaptive_mask(image_bgr, block: int, c: float) -> mask`
  - `clean_mask(mask, kernel: int) -> mask`
  - `mask_to_boxes(mask, min_area: float, max_aspect: float, min_fill: float, max_side_frac: float) -> (N, 4) float xyxy`

- [ ] **Step 1: Write the failing tests**

```python
def sand(h=256, w=256):
    return np.full((h, w, 3), (170, 200, 220), np.uint8)

def test_colour_mask_finds_dark_bird_on_sand():
    image = sand(); cv2.circle(image, (128, 128), 20, (30, 30, 30), -1)
    mask = colour_distance_mask(image, 4, 51, 15.0)
    assert mask[128, 128] == 1 and mask[20, 20] == 0

def test_colour_mask_finds_bright_bird_on_dark_water():
    image = np.full((256, 256, 3), (90, 60, 30), np.uint8)
    cv2.circle(image, (128, 128), 20, (245, 245, 245), -1)
    assert colour_distance_mask(image, 4, 51, 15.0)[128, 128] == 1

def test_colour_mask_matches_small_nonsquare_image():
    image = np.full((40, 90, 3), 128, np.uint8)
    assert colour_distance_mask(image, 4, 51, 15.0).shape == (40, 90)

def test_adaptive_mask_finds_both_polarities():
    image = np.full((256, 256, 3), 128, np.uint8)
    cv2.circle(image, (70, 128), 15, (10, 10, 10), -1)
    cv2.circle(image, (190, 128), 15, (250, 250, 250), -1)
    mask = adaptive_mask(image, 101, 15)
    assert mask[128, 70] == 1 and mask[128, 190] == 1 and mask[20, 20] == 0

def test_clean_mask_removes_speck_and_fills_pinhole():
    mask = np.zeros((60, 60), np.uint8)
    mask[10:30, 10:30] = 1; mask[20, 20] = 0; mask[50, 50] = 1
    cleaned = clean_mask(mask, 3)
    assert cleaned[20, 20] == 1 and cleaned[50, 50] == 0

def test_mask_to_boxes_uses_component_extent_not_a_square():
    mask = np.zeros((100, 100), np.uint8); mask[20:30, 10:50] = 1
    boxes = mask_to_boxes(mask, 16, 5, 0.25, 0.8)
    assert boxes.tolist() == [[10, 20, 50, 30]]

def test_mask_to_boxes_applies_area_aspect_and_fill_limits():
    mask = np.zeros((100, 100), np.uint8)
    mask[1:3, 1:3] = 1                  # area 4
    mask[10:12, 10:60] = 1              # aspect 25
    cv2.line(mask, (40, 40), (80, 80), 1, 1)  # fill about 0.03
    assert len(mask_to_boxes(mask, 16, 4, 0.25, 0.8)) == 0

def test_full_frame_component_rejected():
    assert len(mask_to_boxes(np.ones((100, 100), np.uint8), 16, 4, 0.25, 0.8)) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/classical/test_segment.py -q`
Expected: FAIL at import, because `src.classical.segment` doesn't exist.

- [ ] **Step 3: Implement the four functions in `src/classical/segment.py`**

- **`colour_distance_mask`:** convert to Lab, downscale with `INTER_AREA`, apply `cv2.medianBlur` with `bg_kernel`, then resize back to the input shape with `INTER_LINEAR`. That is the background. Take the per-pixel L2 distance from the background and clip it to uint8. The threshold is `max(otsu, min_distance)`, and the mask is `distance > threshold`.
- **`adaptive_mask`:** apply a 3×3 Gaussian blur to grayscale, then OR together `cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_MEAN_C` for dark (`THRESH_BINARY_INV`, `c`) and bright (`THRESH_BINARY`, `-c`).
- **`clean_mask`:** apply opening then closing with an elliptical element of size `kernel`. `kernel <= 1` returns the mask unchanged.
- **`mask_to_boxes`:** use `cv2.connectedComponentsWithStats` with connectivity 8. Keep a component if area ≥ `min_area`, max(w/h, h/w) ≤ `max_aspect`, area/(w·h) ≥ `min_fill`, w ≤ `max_side_frac`·W and h ≤ `max_side_frac`·H. A component's box is `[x, y, x + w, y + h]`, which matches the GT's pixel-edge coordinates.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/classical/test_segment.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/classical/segment.py tests/classical/test_segment.py
git commit -m "feat(m1): add foreground masks and component boxes"
```

---

### Task 3: Segmentation detector behind `detect`

**Files:**
- Create: `configs/m1_seg_colour.yaml`, `configs/m1_seg_adaptive.yaml`
- Modify: `src/classical/pipeline.py`, `scripts/run_m1.py`, `tests/classical/test_pipeline.py`, `tests/classical/test_run.py`

**Interfaces:**
- Consumes: Task 1's `local_contrast`; Task 2's four functions.
- Produces: `load_config(path) -> BlobConfig | SegConfig`, which pops an optional `method` key (`blob`, the default, or `seg`), and `detect(image_bgr, cfg)` for either config type. `SegConfig` fields and defaults:

```python
foreground: str = "colour"   # or "adaptive"
downscale: int = 4
bg_kernel: int = 51
min_distance: float = 15.0
block: int = 101
c: float = 15.0
morph_kernel: int = 3
min_area: float = 16.0
max_aspect: float = 4.0
min_fill: float = 0.25
max_side_frac: float = 0.8
nms_iou: float = 0.3
max_detections: int = 300
```

- [ ] **Step 1: Write the failing tests**

```python
def test_seg_box_fits_elongated_bird():
    image = np.full((200, 200, 3), 220, np.uint8)
    cv2.ellipse(image, (100, 100), (30, 12), 0, 0, 360, (20, 20, 20), -1)
    gt = np.array([[70., 88, 130, 112]])
    dets = detect(image, SegConfig())
    assert iou_matrix(gt, dets).max() >= 0.8

def test_seg_dense_grid_keeps_at_least_130():
    centres = [(12 + 24 * i, 12 + 24 * j) for i in range(13) for j in range(12)]  # 156 discs
    assert len(detect(bgr_discs(300, 320, centres, r=4), SegConfig())) >= 130

def test_plain_noisy_background_gives_no_detections():
    noise = np.random.default_rng(0).normal(0, 3, (256, 256, 3))
    image = np.clip(np.array([170, 200, 220]) + noise, 0, 255).astype(np.uint8)
    assert detect(image, SegConfig()).shape == (0, 5)

def test_seg_results_sorted_and_capped():
    centres = [(12 + 24 * i, 12 + 24 * j) for i in range(13) for j in range(12)]
    dets = detect(bgr_discs(300, 320, centres, r=4), SegConfig(max_detections=10))
    assert len(dets) == 10 and np.all(np.diff(dets[:, 4]) <= 0)

def test_unknown_foreground_raises():
    with pytest.raises(ValueError):
        detect(np.zeros((32, 32, 3), np.uint8), SegConfig(foreground="nope"))

def test_load_config_defaults_to_blob_for_week7_files():
    assert isinstance(load_config("configs/m1_blob.yaml"), BlobConfig)

def test_load_config_reads_seg_and_rejects_unknown_key(tmp_path):
    assert isinstance(load_config("configs/m1_seg_colour.yaml"), SegConfig)
    path = tmp_path / "c.yaml"; path.write_text("method: seg\nbg_kernal: 3\n")
    with pytest.raises(TypeError):
        load_config(path)

def test_load_config_rejects_unknown_method(tmp_path):
    path = tmp_path / "c.yaml"; path.write_text("method: nope\n")
    with pytest.raises(ValueError):
        load_config(path)

# tests/classical/test_run.py
def test_run_split_accepts_seg_config(tmp_path):
    # build records with _write_images, as test_rows_have_team_schema_and_negative_counts does
    rows, stats = run_split(records, SegConfig())
    assert stats["n_images"] == len(records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/classical/test_pipeline.py tests/classical/test_run.py -q`
Expected: FAIL at import, because `SegConfig` isn't defined.

- [ ] **Step 3: Implement `SegConfig`, `load_config` and the dispatch**

Split the blob body of `detect` into `_detect_blob`, and add `_detect_seg`: mask → `clean_mask` → `mask_to_boxes` → `local_contrast` score → `nms` → cap. A config with no boxes must return shape `(0, 5)`. Keep `BlobConfig.from_yaml` as it is, because the handoff to C names it.

Add `method: seg` to both new YAMLs. To reproduce the prototype, `m1_seg_colour.yaml` sets `min_distance: 0`, and `m1_seg_adaptive.yaml` sets `foreground: adaptive`, `block: 101`, `c: 15`. In `scripts/run_m1.py`, replace `BlobConfig.from_yaml` with `load_config`.

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Reproduce the prototype on val**

```bash
.venv/bin/python scripts/run_m1.py --config configs/m1_seg_colour.yaml --split val --out results/m1_val_seg_colour.json --score
```

```bash
.venv/bin/python scripts/run_m1.py --config configs/m1_seg_adaptive.yaml --split val --out results/m1_val_seg_adaptive.json --score
```

Expected: colour AP@0.5 about 0.018 with recall about 0.295; adaptive AP@0.5 about 0.012 with recall about 0.17. If they're far off, compare against the steps in Task 2 before tuning anything.

- [ ] **Step 6: Commit**

```bash
git add src/classical/pipeline.py scripts/run_m1.py configs/m1_seg_colour.yaml configs/m1_seg_adaptive.yaml tests/classical/test_pipeline.py tests/classical/test_run.py
git commit -m "feat(m1): add segmentation detector behind detect()"
```

---

### Task 4: Precision, F1 and counting error

**Files:**
- Modify: `src/classical/quick_eval.py`, `tests/classical/test_quick_eval.py`

**Interfaces:**
- Produces:
  - `best_f1(rows, records, iou=0.5) -> dict` with keys `precision`, `recall`, `f1`, `score_threshold`
  - `count_mae(rows, records, score_threshold) -> float`, the mean over all images in the split of |detections with score ≥ threshold − GT count|

These fill the §4 table if A's evaluator is late, and give the demo an operating point.

- [ ] **Step 1: Write the failing tests**

Build records and rows with that file's existing `_rec` and `_row` helpers.

```python
def test_best_f1_chooses_threshold_before_false_positive():
    # one image with 2 GT; ranked TP (0.9), TP (0.8), FP (0.1)
    result = best_f1(rows, records)
    assert result["f1"] == 1.0 and result["score_threshold"] == 0.8
    assert result["precision"] == 1.0 and result["recall"] == 1.0

def test_best_f1_with_no_rows_is_zero():
    assert best_f1([], records)["f1"] == 0.0

def test_count_mae_counts_images_with_no_rows():
    # image A: 2 GT, 2 rows above threshold. image B: 3 GT, no rows. image C: 0 GT, 1 row.
    assert count_mae(rows, records, 0.5) == (0 + 3 + 1) / 3

def test_count_mae_ignores_rows_below_threshold():
    # one image: 1 GT, rows scoring 0.9 and 0.1
    assert count_mae(rows, records, 0.5) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/classical/test_quick_eval.py -q`
Expected: FAIL at import.

- [ ] **Step 3: Implement both functions, reusing `_match`**

`_match` returns `is_tp` in descending score order, so `best_f1` needs the ranked scores from the same sort. Sort `rows` the same way inside `best_f1`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/classical/test_quick_eval.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/classical/quick_eval.py tests/classical/test_quick_eval.py
git commit -m "feat(m1): add best-F1 and counting error to quick eval"
```

---

### Task 5: Random-search harness

**Files:**
- Create: `src/classical/tuning.py`, `scripts/tune_m1.py`, `configs/search/blob.yaml`, `configs/search/seg_colour.yaml`, `configs/search/seg_adaptive.yaml`, `tests/classical/test_tuning.py`

**Interfaces:**
- Consumes: `load_config`, `detect`, `read_image`, `ap50`, `recall_at_iou`.
- Produces:
  - `sample_config(base, space: dict[str, list], rng: random.Random) -> config`, returning a copy of `base` with each key replaced by a uniform choice from its list
  - `detect_parallel(records, cfg, workers: int) -> list[dict]`, returning prediction rows identical to `run_split`'s
  - CLI: `python scripts/tune_m1.py --base <yaml> --space <yaml> --split {train,val} --trials N --seed 0 --workers 12 --name <name>`, which writes `results/m1_tuning_<name>.csv` (one row per trial: `ap50`, `recall50`, and each sampled value) and `configs/m1_<name>_tuned.yaml` (the highest-AP config, ties going to the earlier trial, including its `method` key and a comment with the score, seed and date)

- [ ] **Step 1: Write the failing tests**

```python
def test_sample_config_replaces_only_named_fields():
    cfg = sample_config(SegConfig(), {"bg_kernel": [31]}, random.Random(0))
    assert cfg.bg_kernel == 31 and cfg.morph_kernel == SegConfig().morph_kernel

def test_sample_config_is_reproducible_with_seed():
    space = {"bg_kernel": [31, 51, 75], "min_distance": [0, 10, 20, 30]}
    a = sample_config(SegConfig(), space, random.Random(7))
    b = sample_config(SegConfig(), space, random.Random(7))
    assert a == b

def test_sample_config_rejects_unknown_field():
    with pytest.raises(TypeError):
        sample_config(SegConfig(), {"bg_kernal": [31]}, random.Random(0))

def test_detect_parallel_matches_serial(tmp_path):
    # write 4 synthetic disc images and records to tmp_path
    serial, _ = run_split(records, SegConfig())
    assert detect_parallel(records, SegConfig(), workers=2) == serial

def test_detect_parallel_raises_on_corrupt_image(tmp_path):
    # one record's file holds garbage bytes
    with pytest.raises(IOError):
        detect_parallel(records, SegConfig(), workers=2)

def test_tune_script_refuses_test_split():
    result = subprocess.run([sys.executable, "scripts/tune_m1.py", "--base", "configs/m1_seg_colour.yaml",
                             "--space", "configs/search/seg_colour.yaml", "--split", "test",
                             "--trials", "1", "--name", "x"], capture_output=True, text=True)
    assert result.returncode != 0 and "invalid choice" in result.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/classical/test_tuning.py -q`
Expected: FAIL at import.

- [ ] **Step 3: Implement `tuning.py`, the script and the search spaces**

Use `ProcessPoolExecutor`, keep record order so output matches serial, and use `dataclasses.replace` so an unknown field raises `TypeError`. Print progress per trial and flush the CSV after each one, so a stopped search keeps its results. Parallel timings are not to be reported, and the script must not write any `ms_per_image`.

The search spaces:

```yaml
# configs/search/seg_colour.yaml
downscale: [2, 4, 8]
bg_kernel: [31, 51, 75, 101]
min_distance: [0, 8, 12, 16, 20, 25, 30, 40]
morph_kernel: [1, 3, 5, 7]
min_area: [16, 36, 64, 100, 196]
max_aspect: [2.5, 3, 4]
min_fill: [0.2, 0.3, 0.4, 0.5]
nms_iou: [0.3, 0.5]

# configs/search/seg_adaptive.yaml
block: [51, 75, 101, 151, 201]
c: [8, 10, 12, 15, 20, 25, 30]
morph_kernel: [1, 3, 5, 7]
min_area: [16, 36, 64, 100, 196]
max_aspect: [2.5, 3, 4]
min_fill: [0.2, 0.3, 0.4, 0.5]

# configs/search/blob.yaml
threshold: [0.03, 0.05, 0.08, 0.12, 0.18]
clahe: [false, true]
sigmas:
  - [4.56, 7.72, 13.08, 22.14, 37.49, 63.47]
  - [4.56, 7.72, 13.08, 22.14, 37.49]
  - [7.72, 13.08, 22.14, 37.49]
nms_iou: [0.2, 0.3, 0.5]
min_area: [16, 64, 196]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/classical/test_tuning.py -q`
Expected: PASS.

- [ ] **Step 5: Smoke-test on val**

Run: `.venv/bin/python scripts/tune_m1.py --base configs/m1_seg_colour.yaml --space configs/search/seg_colour.yaml --split val --trials 3 --seed 0 --workers 12 --name smoke`
Expected: three printed trials, `results/m1_tuning_smoke.csv`, and `configs/m1_smoke_tuned.yaml`, which `load_config` accepts. Delete both files afterwards.

- [ ] **Step 6: Commit**

```bash
git add src/classical/tuning.py scripts/tune_m1.py configs/search tests/classical/test_tuning.py
git commit -m "feat(m1): add parallel random search on val"
```

---

### Task 6: Tune, compare and take to the tutor

**Files:**
- Create: `scripts/compare_m1.py`, `tests/classical/test_compare.py`, `configs/m1_seg_colour_tuned.yaml`, `configs/m1_seg_adaptive_tuned.yaml`, `configs/m1_blob_tuned.yaml`, `results/m1_comparison.csv`, `results/m1_tuning_*.csv`

**Interfaces:**
- Consumes: Task 4's `best_f1` and `count_mae`; Task 5's CLI; the `.meta.json` sidecars.
- Produces: `compare_row(pred_path, records, variant) -> dict`, with keys `variant, split, ap50, recall50, precision, f1, count_mae, score_threshold, ms_per_image, train_time_s, n_images_capped, machine, git_sha`, and `scripts/compare_m1.py --pred <json>... --split {train,val}`, which writes `results/m1_comparison.csv`.

- [ ] **Step 1: Write the failing tests**

```python
def test_compare_row_reads_metrics_and_sidecar(tmp_path):
    # write a small pred json and .meta.json (ms_per_image 12.5) for one known record
    row = compare_row(pred_path, records, "seg")
    assert row["ap50"] == 1.0 and row["ms_per_image"] == 12.5 and row["train_time_s"] == 0

def test_compare_row_needs_sidecar_for_this_split(tmp_path):
    # sidecar says split "train", records are val: never mix results across splits
    with pytest.raises(ValueError):
        compare_row(pred_path, records, "seg", split="val")
```

`compare_row` therefore also takes `split: str | None = None`, and raises if it differs from the sidecar's.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/classical/test_compare.py -q`
Expected: FAIL, because `compare_m1` doesn't exist.

- [ ] **Step 3: Implement `scripts/compare_m1.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/classical/test_compare.py -q`
Expected: PASS.

- [ ] **Step 5: Run the three searches, and note each one's wall-clock time**

The order is deliberate: the fastest and most promising search goes first, so a late day still has its main result.

```bash
time .venv/bin/python scripts/tune_m1.py --base configs/m1_seg_colour.yaml --space configs/search/seg_colour.yaml --split val --trials 200 --seed 0 --workers 12 --name seg_colour
```

```bash
time .venv/bin/python scripts/tune_m1.py --base configs/m1_seg_adaptive.yaml --space configs/search/seg_adaptive.yaml --split val --trials 100 --seed 0 --workers 12 --name seg_adaptive
```

```bash
time .venv/bin/python scripts/tune_m1.py --base configs/m1_blob_contrast.yaml --space configs/search/blob.yaml --split val --trials 40 --seed 0 --workers 12 --name blob
```

Expected: each best config is at least as good as its starting point on val. If any best value sits on the edge of its list, widen that list once and rerun that search. Do not go further than that.

- [ ] **Step 6: Run the final variants serially, for honest timings and overlays**

Run each of these configs on `val`: `m1_blob`, `m1_blob_contrast`, `m1_blob_tuned`, `m1_seg_adaptive_tuned`, `m1_seg_colour_tuned`. For example:

```bash
.venv/bin/python scripts/run_m1.py --config configs/m1_seg_colour_tuned.yaml --split val --out results/m1_val_seg_colour_tuned.json --score --viz 20
```

Expected: `n_images_capped` is printed and `ms_per_image` is in the sidecar. Also copy the best variant to `results/m1_val.json`, the name the team's evaluator expects, by rerunning it with no `--out`.

- [ ] **Step 7: Check generalisation on train**

Run the tuned colour config on `--split train` with `--score`. No parameter is learned from train, so it works as an unseen check on the search.
Expected: AP within about 30% of val. A much lower train AP means the search overfit val, so prefer a simpler config from the trials CSV and note it.

- [ ] **Step 8: Write the comparison CSV and check the outcome**

```bash
.venv/bin/python scripts/compare_m1.py --split val --pred results/m1_val_blob.json results/m1_val_blob_contrast.json results/m1_val_blob_tuned.json results/m1_val_seg_adaptive_tuned.json results/m1_val_seg_colour_tuned.json
```

Expected: a 5-row `results/m1_comparison.csv`, in which the tuned colour AP is at least 0.0178 and detections per negative image are clearly below the prototype's 84. If not, the floor isn't working: look at negative-image overlays before moving on.

- [ ] **Step 9: Commit**

```bash
git add scripts/compare_m1.py tests/classical/test_compare.py configs/m1_*_tuned.yaml results/m1_comparison.csv results/m1_tuning_seg_colour.csv results/m1_tuning_seg_adaptive.csv results/m1_tuning_blob.csv
git commit -m "feat(m1): tune approaches on val and add comparison table"
```

- [ ] **Step 10: Tutor consult (Thu)**

Bring the comparison table, 2 good and 2 bad overlays, and A's EDA if it exists. Ask these three questions, and record the answers in the Week 8 notes:

1. Is it enough for Method 1 to compare a blob detector against colour segmentation, with AP around 0.0x, if the failure analysis is strong?
2. Does a search on `val` with a check on `train` satisfy the no-leakage rule?
3. What should be added to the advanced-study plan for the >27 band?

---

### Task 7: Proposals for Member C and evaluator cross-check

**Files:**
- Create: `configs/m1_seg_proposals.yaml`, `results/m1_proposal_recall.csv`
- Modify: `tests/classical/test_pipeline.py`

**Interfaces:**
- Consumes: `load_config`, `recall_at_iou`.
- Produces: a high-recall config C can load with `load_config`, or a documented finding that blob proposals are still better.

- [ ] **Step 1: Write the failing test**

```python
def test_seg_proposals_config_loads_with_large_cap():
    cfg = load_config("configs/m1_seg_proposals.yaml")
    assert isinstance(cfg, SegConfig) and cfg.max_detections >= 1000
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/classical/test_pipeline.py::test_seg_proposals_config_loads_with_large_cap -q`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Create the config and measure it**

Start from the tuned colour config. Set `min_distance: 8`, `min_area: 16`, `min_fill: 0.15`, `max_aspect: 5`, `nms_iou: 0.5`, `max_detections: 2000`.

Run: `.venv/bin/python scripts/run_m1.py --config configs/m1_seg_proposals.yaml --split val --out results/m1_val_seg_proposals.json --score`
Expected: recall@0.5 above the blob proposals' 17.0%, with far fewer than their 1,116 candidates per image. Record recall@0.5, recall@0.3, candidates per image and ms/img for both in `results/m1_proposal_recall.csv`.

- [ ] **Step 4: Send Member C the result**

Even at this level, recall caps Method 2. Say so plainly, and keep the sliding-window recommendation if recall@0.5 is under about 60%.

- [ ] **Step 5: Cross-check against A's evaluator**

If `scripts/evaluate.py` is on `origin/main`, merge it, then run: `.venv/bin/python scripts/evaluate.py --pred results/m1_val.json --split val`
Expected: mAP@0.5 agrees with `quick_eval.ap50` to within 0.002 (COCO's 101-point interpolation accounts for small differences). Investigate any larger gap with A.

If it hasn't landed by Fri, tell A and the group that the milestone depends on it, and hand A `results/m1_val.json` and its sidecar. The Sunday milestone is then only partly met, and the notes must say so.

- [ ] **Step 6: Commit**

```bash
git add configs/m1_seg_proposals.yaml results/m1_proposal_recall.csv tests/classical/test_pipeline.py
git commit -m "feat(m1): add segmentation proposals for Method 2"
```

---

### Task 8 (stretch): Split touching birds

Only start this on Sat with Tasks 1–7 complete. Merged components are one box for a whole cluster, and dense flocks are a stated challenge, but `51+` images are rare (3 in val), so AP may barely move. Keep it only if it earns its place.

**Files:**
- Modify: `src/classical/segment.py`, `src/classical/pipeline.py`, `tests/classical/test_segment.py`

**Interfaces:**
- Produces: `split_touching(mask, min_peak_distance: int) -> int32 labels`, and `SegConfig.split_touching: bool = False` (off by default, so tuned configs don't change).

- [ ] **Step 1: Write the failing test**

```python
def test_split_touching_separates_two_overlapping_discs():
    mask = np.zeros((100, 140), np.uint8)
    cv2.circle(mask, (50, 50), 22, 1, -1); cv2.circle(mask, (88, 50), 22, 1, -1)
    assert split_touching(mask, 15).max() == 2

def test_split_touching_keeps_single_disc_whole():
    mask = np.zeros((100, 100), np.uint8); cv2.circle(mask, (50, 50), 22, 1, -1)
    assert split_touching(mask, 15).max() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/classical/test_segment.py -k split_touching -q`
Expected: FAIL at import.

- [ ] **Step 3: Implement it**

Take the distance transform of the mask, use its local maxima at least `min_peak_distance` apart as markers, and run `cv2.watershed`. Extract the box logic in `mask_to_boxes` so that it also accepts a label image.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/classical/test_segment.py -q`
Expected: PASS.

- [ ] **Step 5: Measure it, and keep or drop it**

Copy the tuned config, turn `split_touching` on, and run it on `val` with `--score`.
Expected: keep it only if val AP rises by at least 10% and ms/img stays under 100. Otherwise revert the code and record the measurement in the notes as a tried idea.

- [ ] **Step 6: Commit, if kept**

```bash
git add src/classical/segment.py src/classical/pipeline.py tests/classical/test_segment.py configs/m1_seg_colour_tuned.yaml
git commit -m "feat(m1): split touching birds with watershed"
```

---

### Task 9: Notes, references and Sunday milestone

**Files:**
- Create: `docs/notes/m1-week8.md`
- Modify: `docs/refs/method1.bib`, `docs/notes/m1-week7.md` (one line pointing to the new notes)

- [ ] **Step 1: Write `docs/notes/m1-week8.md`**

It must include:

- The §4 table with one row per variant, from `results/m1_comparison.csv`, including search wall-clock time beside the 0 training time.
- Which approach wins, why, and how much each change contributed (scoring, Otsu floor, segmented box shapes).
- Failures from the overlays, ordered by cost, using §3C's categories: missed in dense cluster, false positive on rock/shell/wave, duplicates, localisation.
- The tuned parameters, what the train check showed, and tutor feedback.
- That the edge-ratio test was tried in a throwaway prototype and cut, marked as not reproducible from the repo.
- Handoffs:
  - **A:** the file to evaluate is `results/m1_val.json`, and the team still has to choose density bins.
  - **C:** `configs/m1_seg_proposals.yaml` and its measured recall.
  - **D:** the plain-background colour model's failures, as a pointer to where a learned detector should beat it.
  - **E:** use `load_config(path)` then `detect(image, cfg)`, at about 12 ms/img on the colour config. Median background and Otsu are likely to react strongly to noise and low contrast, so they're worth watching in the robustness curves.

- [ ] **Step 2: Add references, verified against Crossref**

Add Bradley & Roth 2007 (mean-based adaptive thresholding), and a morphology reference (Soille, *Morphological Image Analysis*). If Task 8 was kept, add a watershed reference (Vincent & Soille, TPAMI 1991). Keep the existing Otsu entry. Remove MSER if it stays unused. Add OpenCV, SciPy and NumPy as software citations. Record any detail you can't verify as unverified rather than guessing.

- [ ] **Step 3: Full verification**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

Run: `git status --short` and `git ls-files | grep -E "\.(jpg|json|pt|pth|pkl)$"`
Expected: a clean tree, and no predictions, images or weights tracked. `grep` prints nothing.

Run: `grep -l '"split": "test"' results/*.meta.json`
Expected: no matches, confirming `test` wasn't run this week.

- [ ] **Step 4: Check the §6 milestone**

Confirm: at least 2 handcrafted approaches are compared, both tuned on val; `results/m1_val.json` is in the team format and has been through the shared evaluator, or the notes say why not.

- [ ] **Step 5: Commit, then push once the owner confirms**

```bash
git add docs/notes docs/refs/method1.bib
git commit -m "docs(m1): record Week 8 comparison and handoff"
```

```bash
git push origin member-b/method1
```

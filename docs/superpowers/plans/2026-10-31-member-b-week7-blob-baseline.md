# Member B, Week 7: Method 1 Blob Detector Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** By Sun 1 Nov, have a scale-space blob detector that runs end to end on `val`, writes predictions in the team format, and has a first mAP@0.5, with seams ready for Week 8 tuning, Member C's proposals and Member E's robustness runs.

**Architecture:** One image-in, boxes-out function, `detect(image_bgr, cfg) -> (N,5)`, composes preprocessing, multi-scale LoG blob detection and post-processing. It only sees arrays, so E can pass in distorted images and C can call it with a high-recall config. A thin run script handles files, timing and the test-set guard. Member A's loader and evaluator don't exist yet, so small interim versions sit behind A's expected interface and get swapped out when A's work lands.

**Tech Stack:** Python 3.12, NumPy, SciPy (`ndimage`), OpenCV, Pillow, PyYAML, pytest.

**Spec:** `PROJECT_PLAN.md` (§2, §3 Method 1, §4, §5, §6, §8, §10) and `../COMP9517_26T3_Group_Project_Specification.pdf`.

## Global Constraints

- Tune on `train`+`val` only. `test` is for final evaluation, once per final model, behind a separate flag.
- Every script takes `--split {train,val,test}`.
- Prediction rows: `image_id, x1, y1, x2, y2, score`, in absolute pixels of the original image.
- Output file naming: `results/m1_val.json`, so it works with `python scripts/evaluate.py --pred results/m1_val.json --split val`.
- Max detections per image ≥ 300, in the detector and in any evaluator.
- Report mAP@0.5, inference ms/img and hardware for every run.
- Repo must not contain `data/`, weights or result images. Code ZIP ≤ 25 MB.
- Write as much as practical ourselves, and cite every paper and library.
- Follow the §8 layout (`src/classical/`, `configs/`, `scripts/`, `results/`). `tests/` is an addition to it.

## Review Focus

1. **Negative images (no birds):** a clean image gives zero detections with no crash, and still counts as a processed image. Tested in Tasks 3 and 5.
2. **Dense flocks (130 birds):** the cap and NMS must not silently limit a dense image. `test_dense_grid_keeps_at_least_130` in Task 4.
3. **Birds touching the image edge:** boxes stay inside the image and are non-degenerate. `test_edge_blob_box_stays_in_bounds` in Task 4.
4. **Varying or very large resolutions:** after internal downscaling, boxes are still in original-image pixels. `test_downscaled_detection_maps_to_original_pixels` in Task 4.
5. **Unreadable or corrupt image file:** the run fails loudly and names the path, rather than silently lowering the image count. `test_unreadable_image_raises_with_path` in Task 5.

## Open Items

- **Annotation format is unknown.** The Kaggle page couldn't be read without a browser session, and no data is on disk yet. Task 1 inspects it first. Its parser and fixture default to YOLO txt and change if the data differs.
- **Member A's work isn't pushed** (the remote has only `PROJECT_PLAN.md`). Run `git fetch` before Tasks 1 and 2. If `src/data` or `src/eval` has landed, use it and skip the matching interim module.
- **Time:** about 2 days left in Week 7. If time runs short, cut the proposal recall check in Task 6 and keep everything else.

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt`, `.gitignore`, `pytest.ini` | Create if A hasn't. Otherwise add only the missing lines |
| `src/classical/interim_data.py` | `ImageRecord`, `load_split`. Temporary, replaced by A's loader |
| `src/classical/boxes.py` | `iou_matrix`, `clip_boxes`, `filter_boxes`, `nms` |
| `src/classical/quick_eval.py` | Hand-written `ap50`, `recall_at_iou`. Becomes A's cross-check |
| `src/classical/blob.py` | `to_gray`, `detect_blobs`, `suggest_sigmas` |
| `src/classical/pipeline.py` | `BlobConfig`, `detect`. The public interface for C and E |
| `src/classical/run.py` | `run_split`, `draw_overlay` |
| `scripts/run_m1.py` | CLI and test-set guard |
| `configs/m1_blob.yaml`, `configs/m1_blob_proposals.yaml` | Baseline and high-recall settings |
| `tests/classical/` | One test file per module |
| `docs/notes/m1-week7.md`, `docs/refs/method1.bib` | Results notes, handoff, references |

Week 8's threshold detector goes in a new `src/classical/threshold.py`, reusing `boxes.py` and `run.py` unchanged.

---

### Task 1: Environment, dataset and record loader

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pytest.ini`, `src/__init__.py`, `src/classical/__init__.py`, `src/classical/interim_data.py`
- Test: `tests/classical/test_interim_data.py`

**Interfaces:**
- Produces: `ImageRecord(image_id: str, path: Path, boxes: np.ndarray)`, where `boxes` is `(N,4)` float xyxy in pixels and `image_id` is the file stem. `load_split(split: str, root="data") -> list[ImageRecord]`.

- [ ] **Step 1: Create the branch and environment**

```bash
git checkout -b member-b/method1
brew install python@3.12
python3.12 -m venv .venv
.venv/bin/pip install numpy scipy opencv-python pillow pyyaml pytest
```

Pin the installed versions of those six packages in `requirements.txt`. `.gitignore` must cover `.venv/`, `data/`, `results/*` (except `results/*.csv`), `*.pt`, `*.pth`, `*.pkl`. `pytest.ini` sets `pythonpath = .` and `testpaths = tests`.
Expected: `.venv/bin/python -c "import cv2, scipy, yaml"` exits 0.

- [ ] **Step 2: Download the dataset and identify its format**

You do this one by hand, because it needs your Kaggle login: download https://www.kaggle.com/datasets/liugalhigh/beach-bird and unzip into `data/`. Open two label files and one negative image's label (or find that it has none). Record the layout and format at the top of `interim_data.py`, and tell A so you agree.

- [ ] **Step 3: Write the failing tests**

Build a temp dataset in `tests/classical/test_interim_data.py` with a 100×50 image `a.jpg` and a label `0 0.5 0.5 0.2 0.4`. If the real format isn't YOLO, write the fixture in the real format with the same box.

```python
def test_box_converted_to_pixel_xyxy(tmp_dataset):
    rec = load_split("val", root=tmp_dataset)[0]
    assert rec.image_id == "a"
    np.testing.assert_allclose(rec.boxes, [[40, 15, 60, 35]])

def test_image_with_empty_or_missing_label_has_no_boxes(tmp_dataset_negative):
    assert load_split("val", root=tmp_dataset_negative)[0].boxes.shape == (0, 4)

def test_unknown_split_rejected(tmp_dataset):
    with pytest.raises(ValueError):
        load_split("validation", root=tmp_dataset)

def test_missing_split_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split("val", root=tmp_path)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/classical/test_interim_data.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 5: Implement `load_split` in `src/classical/interim_data.py`**

Records sorted by `image_id`, so runs are reproducible. Read image size with Pillow (`Image.open(path).size`), which doesn't decode the pixels.

- [ ] **Step 6: Run tests and verify counts on the real data**

Run: `.venv/bin/pytest tests/classical/test_interim_data.py -v`
Expected: PASS.

Run: `.venv/bin/python -c "from src.classical.interim_data import load_split as l; print([len(l(s)) for s in ('train','val','test')]); print(max(len(r.boxes) for s in ('train','val') for r in l(s)))"`
Expected: `[1369, 291, 240]`, then a max of at most 130.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore pytest.ini src tests
git commit -m "feat(m1): add environment and interim dataset loader"
```

---

### Task 2: Box utilities and hand-written AP@0.5

**Files:**
- Create: `src/classical/boxes.py`, `src/classical/quick_eval.py`
- Test: `tests/classical/test_boxes.py`, `tests/classical/test_quick_eval.py`

**Interfaces:**
- Consumes: `ImageRecord` (Task 1).
- Produces:
  - `iou_matrix(a: (N,4), b: (M,4)) -> (N,M)`
  - `clip_boxes(dets: (N,5), width: int, height: int) -> (K,5)`, which drops boxes with no area after clipping
  - `filter_boxes(dets, min_area: float, max_area: float | None) -> (K,5)`
  - `nms(dets, iou_thr: float) -> (K,5)`, sorted by score descending
  - `ap50(rows: list[dict], records: list[ImageRecord]) -> float`
  - `recall_at_iou(rows, records, iou: float = 0.5) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# test_boxes.py
def test_iou_half_overlap():
    a = np.array([[0, 0, 10, 10.]]); b = np.array([[5, 0, 15, 10.]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1/3]])

def test_nms_removes_lower_score_overlap_and_sorts():
    d = np.array([[0,0,10,10,.5],[1,0,11,10,.9],[50,50,60,60,.7]], float)
    np.testing.assert_allclose(nms(d, .3)[:, 4], [.9, .7])

def test_nms_empty_keeps_shape():
    assert nms(np.zeros((0, 5)), .3).shape == (0, 5)

def test_clip_drops_box_fully_outside():
    d = np.array([[-5,-5,5,5,.9],[200,200,210,210,.8]], float)
    np.testing.assert_allclose(clip_boxes(d, 100, 100), [[0,0,5,5,.9]])

def test_filter_by_area():
    d = np.array([[0,0,2,2,.9],[0,0,10,10,.8],[0,0,50,50,.7]], float)
    np.testing.assert_allclose(filter_boxes(d, 16, 400)[:, 4], [.8])

# test_quick_eval.py: records built with path=None
def test_perfect_prediction_is_one(): ...        # assert ap50 == pytest.approx(1.0)
def test_tp_fp_tp_ranking():                     # GT 2 boxes; TP .9, FP .8, TP .7
    assert ap50(rows, records) == pytest.approx(0.8333, abs=1e-3)
def test_duplicate_on_same_gt_is_fp():           # GT 2 boxes; GT1 .9, GT1 again .8, GT2 .7
    assert ap50(rows, records) == pytest.approx(0.8333, abs=1e-3)
def test_no_predictions_is_zero(): ...           # assert ap50([], records) == 0.0
def test_fp_on_negative_image_lowers_ap():       # FP .95 on no-GT image, TP .9 on other
    assert ap50(rows, records) == pytest.approx(0.5)
def test_unknown_image_id_raises(): ...          # ValueError catches split mix-ups
def test_recall_counts_gt_covered(): ...         # 1 of 2 GT matched: 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/classical/test_boxes.py tests/classical/test_quick_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement both modules**

`nms` is greedy. `ap50` sorts all rows by score, matches each to the highest-IoU unmatched GT in its own image at IoU ≥ 0.5, then takes the all-point interpolated area under the precision envelope. No GT in the whole split raises `ValueError`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/classical/test_boxes.py tests/classical/test_quick_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/classical/boxes.py src/classical/quick_eval.py tests/classical
git commit -m "feat(m1): add box utilities and hand-written AP@0.5"
```

---

### Task 3: Multi-scale LoG blob detector

**Files:**
- Create: `src/classical/blob.py`
- Test: `tests/classical/test_blob.py`

**Interfaces:**
- Produces:
  - `to_gray(image_bgr: np.ndarray, clahe: bool = False) -> np.ndarray`, float32 in [0,1]
  - `detect_blobs(gray, sigmas: Sequence[float], threshold: float, polarity: str) -> (N,5)`, with polarity one of `"dark"`, `"bright"`, `"both"`
  - `suggest_sigmas(boxes: (N,4), n: int = 6) -> list[float]`

Use a hand-written scale-normalised LoG rather than `skimage.feature.blob_log`, because `blob_log` doesn't return response strength and mAP needs a score. It also counts toward writing the code ourselves.

- [ ] **Step 1: Write the failing tests**

A helper `disc_image(size, centre_xy, r, fg, bg)` returns a float gray image.

```python
SIGMAS = np.geomspace(2, 16, 8)

def test_dark_disc_found_with_correct_centre_and_size():
    g = disc_image(100, (40, 60), r=8, fg=0., bg=1.)
    x1, y1, x2, y2, score = detect_blobs(g, SIGMAS, .05, "dark")[0]
    assert abs((x1 + x2) / 2 - 40) <= 2 and abs((y1 + y2) / 2 - 60) <= 2
    assert (x2 - x1) == pytest.approx(16, rel=.3)
    assert score > 0

def test_polarity_selects_contrast():
    g = disc_image(100, (50, 50), r=8, fg=1., bg=0.)
    assert len(detect_blobs(g, SIGMAS, .05, "bright")) >= 1
    assert len(detect_blobs(g, SIGMAS, .05, "dark")) == 0
    assert len(detect_blobs(g, SIGMAS, .05, "both")) >= 1

def test_uniform_image_has_no_detections():
    assert detect_blobs(np.full((64, 64), .5, np.float32), SIGMAS, .05, "both").shape == (0, 5)

def test_stronger_contrast_scores_higher(): ...   # fg 0.0 vs fg 0.7 on bg 1.0

def test_to_gray_range_and_shape_with_clahe():
    img = np.random.default_rng(0).integers(0, 256, (20, 30, 3), np.uint8)
    g = to_gray(img, clahe=True)
    assert g.shape == (20, 30) and g.dtype == np.float32 and 0 <= g.min() <= g.max() <= 1

def test_suggest_sigmas_from_box_sizes():
    side = 2 * np.sqrt(2) * 4
    assert suggest_sigmas(np.array([[0, 0, side, side]] * 5)) == pytest.approx([4.] * 6)

def test_suggest_sigmas_empty_raises(): ...       # ValueError
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/classical/test_blob.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/classical/blob.py`**

The response stack is `sigma² * gaussian_laplace(gray, sigma)` per sigma. The LoG is positive at the centre of a dark blob, so `"dark"` keeps it as is, `"bright"` negates it, and `"both"` takes the absolute value. Peaks are 3D local maxima from `scipy.ndimage.maximum_filter(stack, size=3)` with response above `threshold`. A peak's box is a square of side `2√2·sigma`, centred on it, with score = response, sorted descending. `suggest_sigmas` takes the 5th and 95th percentiles of `sqrt(w·h)/(2√2)`, floors them at 1.0, and returns `np.geomspace` values as a list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/classical/test_blob.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/classical/blob.py tests/classical/test_blob.py
git commit -m "feat(m1): add scale-normalised LoG blob detector"
```

---

### Task 4: `detect` pipeline and config

**Files:**
- Create: `src/classical/pipeline.py`, `configs/m1_blob.yaml`
- Test: `tests/classical/test_pipeline.py`

**Interfaces:**
- Consumes: `to_gray`, `detect_blobs` (Task 3); `clip_boxes`, `filter_boxes`, `nms` (Task 2).
- Produces:
  - `BlobConfig` dataclass, with defaults `sigmas=(2,3,4,6,8,12)`, `threshold=0.05`, `polarity="both"`, `clahe=False`, `max_side=1600`, `nms_iou=0.3`, `min_area=16.0`, `max_area=None`, `max_detections=300`
  - `BlobConfig.from_yaml(path) -> BlobConfig`, where an unknown key raises `TypeError`
  - `detect(image_bgr: np.ndarray, cfg: BlobConfig) -> (N,5)` in original-image pixels, sorted by score descending. `sigmas` are also in original-image pixels.

- [ ] **Step 1: Write the failing tests**

A helper `bgr_discs(h, w, centres_xy, r)` draws dark discs on a light uint8 BGR image. `noise_image` is a 128×128 uint8 BGR array from `np.random.default_rng(0)`.

```python
def test_dense_grid_keeps_at_least_130():
    centres = [(12 + 24 * i, 12 + 24 * j) for i in range(13) for j in range(12)]  # 156 discs
    img = bgr_discs(300, 320, centres, r=4)
    assert len(detect(img, BlobConfig(sigmas=(2, 3, 4)))) >= 130

def test_max_detections_caps_and_keeps_best():
    d = detect(noise_image, BlobConfig(threshold=0.001, max_detections=10))
    assert len(d) <= 10 and np.all(np.diff(d[:, 4]) <= 0) and np.isfinite(d).all()

def test_edge_blob_box_stays_in_bounds():
    d = detect(bgr_discs(100, 100, [(2, 50)], r=8), BlobConfig())
    assert len(d) >= 1
    assert (d[:, :2] >= 0).all() and (d[:, 2] <= 100).all() and (d[:, 3] <= 100).all()
    assert (d[:, 2] > d[:, 0]).all() and (d[:, 3] > d[:, 1]).all()

def test_downscaled_detection_maps_to_original_pixels():
    img = bgr_discs(3200, 3200, [(2000, 1000)], r=40)
    x1, y1, x2, y2, _ = detect(img, BlobConfig(sigmas=(14, 20, 28, 40), max_side=800, min_area=16))[0]
    assert abs((x1 + x2) / 2 - 2000) <= 8 and abs((y1 + y2) / 2 - 1000) <= 8
    assert (x2 - x1) == pytest.approx(80, rel=.3)

def test_negative_image_gives_empty():
    assert detect(np.full((64, 64, 3), 128, np.uint8), BlobConfig()).shape == (0, 5)

def test_from_yaml_rejects_unknown_key(tmp_path): ...   # TypeError on a typo'd key
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/classical/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/classical/pipeline.py`**

Steps in order: gray; downscale with `cv2.INTER_AREA` if the longest side exceeds `max_side`, multiplying sigmas by the scale; `detect_blobs`; divide coordinates by the scale; `clip_boxes`; `filter_boxes`; `nms`; truncate to `max_detections`. Write `configs/m1_blob.yaml` with the defaults.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/classical -v`
Expected: PASS for all classical tests.

- [ ] **Step 5: Commit**

```bash
git add src/classical/pipeline.py configs/m1_blob.yaml tests/classical/test_pipeline.py
git commit -m "feat(m1): add detect pipeline with YAML config"
```

---

### Task 5: Run script, timing and test-set guard

**Files:**
- Create: `src/classical/run.py`, `scripts/run_m1.py`
- Test: `tests/classical/test_run.py`

**Interfaces:**
- Consumes: `load_split` (Task 1); `ap50`, `recall_at_iou` (Task 2); `BlobConfig`, `detect` (Task 4).
- Produces:
  - `run_split(records, cfg) -> tuple[list[dict], dict]`. Rows have keys `image_id, x1, y1, x2, y2, score`. Stats have keys `n_images, n_detections, ms_per_image`.
  - `draw_overlay(image_bgr, gt_boxes, dets) -> np.ndarray`, GT in blue, predictions in green
  - CLI: `python scripts/run_m1.py --config PATH --split {train,val,test} [--limit N] [--score] [--viz N] [--out PATH] [--final]`. Default output is `results/m1_<split>.json`, plus a sidecar `results/m1_<split>.meta.json`.

- [ ] **Step 1: Write the failing tests**

```python
def test_rows_have_team_schema_and_negative_counts(tmp_path):
    # records: one image with a dark disc, one uniform image
    rows, stats = run_split(records, BlobConfig())
    assert set(rows[0]) == {"image_id", "x1", "y1", "x2", "y2", "score"}
    assert all(isinstance(r["x1"], float) for r in rows)   # json-serialisable, not np.float32
    assert stats["n_images"] == 2 and stats["ms_per_image"] > 0
    assert {r["image_id"] for r in rows} == {"bird"}

def test_unreadable_image_raises_with_path(tmp_path):
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"not an image")
    with pytest.raises(IOError, match="bad.jpg"):
        run_split([ImageRecord("bad", bad, np.zeros((0, 4)))], BlobConfig())

def test_test_split_refused_without_final():
    p = subprocess.run([sys.executable, "scripts/run_m1.py", "--config", "configs/m1_blob.yaml",
                        "--split", "test"], capture_output=True, text=True)
    assert p.returncode != 0 and "--final" in p.stderr

def test_score_refused_on_test_split():
    p = subprocess.run([sys.executable, "scripts/run_m1.py", "--config", "configs/m1_blob.yaml",
                        "--split", "test", "--final", "--score"], capture_output=True, text=True)
    assert p.returncode != 0 and "--score" in p.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/classical/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `run.py` and the script**

Check `--split test` with `parser.error` before loading any data, and insert the repo root into `sys.path` at the top of the script. Time only the `detect` calls with `time.perf_counter`, not the disk reads. The sidecar stores the full config, stats, `platform.platform()`, `platform.machine()`, the git SHA and `train_time_s: 0`. `--score` prints AP@0.5 and recall@0.5. `--viz N` saves N overlays chosen with seed 0 to `results/m1_viz/`. `--limit` takes the first N records and is refused together with `--final`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/classical -v`
Expected: PASS for all classical tests.

- [ ] **Step 5: Commit**

```bash
git add src/classical/run.py scripts/run_m1.py tests/classical/test_run.py
git commit -m "feat(m1): add run script with timing and test-set guard"
```

---

### Task 6: First val baseline, references and handoff

**Files:**
- Modify: `configs/m1_blob.yaml`
- Create: `configs/m1_blob_proposals.yaml`, `docs/notes/m1-week7.md`, `docs/refs/method1.bib`

This task produces numbers rather than code, so its gate is the checks below instead of a test cycle.

**Interfaces:**
- Consumes: the CLI (Task 5), `suggest_sigmas` (Task 3).
- Produces: `results/m1_val.json`, the baseline numbers, and the contract for A, C and E.

- [ ] **Step 1: Set sigmas from training boxes only**

Run: `.venv/bin/python -c "import numpy as np; from src.classical.interim_data import load_split; from src.classical.blob import suggest_sigmas; print(suggest_sigmas(np.vstack([r.boxes for r in load_split('train')])))"`
Expected: 6 increasing values. Copy them into `configs/m1_blob.yaml`.

- [ ] **Step 2: Choose polarity on a val subset**

Run each `polarity` value in the config (`dark`, `bright`, `both`) with:
`.venv/bin/python scripts/run_m1.py --config configs/m1_blob.yaml --split val --limit 50 --score`
Expected: three AP values. Keep the best polarity in the config.

- [ ] **Step 3: Run the full val baseline**

Run: `.venv/bin/python scripts/run_m1.py --config configs/m1_blob.yaml --split val --score --viz 20`
Expected: `n_images` is 291, `results/m1_val.json` and `results/m1_val.meta.json` exist, and AP@0.5 is above 0.

A low AP is acceptable here (§10 expects it). If AP is effectively 0, look at the overlays before tuning anything: misaligned boxes point to a loader bug, while boxes in the right place but the wrong shape point to the square-box limitation.

- [ ] **Step 4: Check high-recall proposals for Member C**

Create `configs/m1_blob_proposals.yaml` as a copy of the baseline with `threshold: 0.01`, then run it with `--score --out results/m1_val_proposals.json`.
Expected: recall@0.5 printed and higher than the baseline's. This recall is the ceiling on what Method 2 can reach, so C needs the number.

- [ ] **Step 5: Cross-check against A's evaluator, if it has landed**

Run: `.venv/bin/python scripts/evaluate.py --pred results/m1_val.json --split val`
Expected: mAP@0.5 within 0.02 of `quick_eval`. A larger gap on dense images usually means a max-detections cap, since the `pycocotools` default is 100. If the evaluator isn't ready, record that in the notes and run this when it lands.

- [ ] **Step 6: Write `docs/notes/m1-week7.md`**

Include:
- The baseline row in §4's table columns (Method, Variant, mAP@0.5, P, R where available, Infer ms/img, Hardware, Train time 0), with unavailable metrics left blank until A's evaluator runs
- Proposal recall
- What the 20 overlays show: main false-positive sources, misses in dense flocks, box-shape errors
- The Week 8 candidates, ranked by what the overlays show
- A handoff section:
  - **A:** the row schema and `.meta.json`; images with no detections have no rows, so the evaluator must take its image list from the split, not the predictions; set max detections ≥ 300; `quick_eval.py` is available as the hand-written check
  - **C:** call `detect(image, BlobConfig.from_yaml("configs/m1_blob_proposals.yaml"))`
  - **E:** `detect` takes any BGR uint8 array, so distorted images need no file writes

- [ ] **Step 7: Collect references in `docs/refs/method1.bib`**

Five entries, checking each one's details against the publisher's page before adding it: Lindeberg 1998 (scale selection, IJCV), Otsu 1979 (thresholding, IEEE TSMC), Matas et al. 2002 (MSER, BMVC), Zuiderveld 1994 (CLAHE, Graphics Gems IV), Zou et al. 2023 (survey, from the spec). Add a one-line note on each saying what it supports in the report.

- [ ] **Step 8: Run all tests and commit**

Run: `.venv/bin/pytest tests/classical -v`
Expected: PASS.

```bash
git add configs docs/notes/m1-week7.md docs/refs/method1.bib
git commit -m "docs(m1): record first val baseline and team handoff"
```

Then push the branch and share the handoff section in the team channel.

---

## How This Sets Up the Later Weeks

| When | Your task | What Week 7 already gives you |
|---|---|---|
| W8 | Second handcrafted approach | New `threshold.py` returning `(N,5)`, reusing `boxes.py`, `run.py` and the CLI |
| W8 | Tune on val | `BlobConfig` plus YAML, and fast scoring via `--score`. The likely first improvement is non-square boxes refined from local segmentation, since square boxes limit IoU on elongated birds |
| W8 | Tutor consult | Baseline number and annotated overlays |
| W9 | Config freeze and single test run | `--final`, with config and hardware recorded in the sidecar |
| W9 | Failure gallery | `draw_overlay` and notes from the overlays |
| W9–10 | Report (lit review, Method 1) and video segment | `method1.bib`, the notes file, and `run_m1.py --viz` for the demo |

Replace `interim_data.py` with A's loader when it lands, changing only the import in `scripts/run_m1.py` and the tests' fixtures. Keep `quick_eval.py` as the hand-written check.

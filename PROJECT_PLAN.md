# COMP9517 26T3 Group Project — Project Plan

**Bird Detection on the Beach Bird Detection Dataset**

| | |
|---|---|
| **Weight** | 30% of course mark (30 marks) |
| **Team size** | 5 students |
| **Project window** | Weeks 6–10 |
| **Hard deadline** | **Friday 20 November 2026, 18:00 AET** |
| **Submission** | One member submits for the whole group: video (MP4), report (PDF), code (ZIP) |
| **Target band** | Above 27/30 ("Advanced Method Development") |

> Plan written 29 Oct 2026 (Week 7). About 3 weeks remain, so the timeline below is compressed. Assumed week dates: W7 = 26–30 Oct, W8 = 2–6 Nov, W9 = 9–13 Nov, W10 = 16–20 Nov. Check against the WebCMS calendar.

---

## 1. Requirements Checklist (from the spec)

### Method requirements

- [ ] At least **two meaningfully different methods** (minimum pass requirement)
- [ ] Methods from **at least two categories** (non-learning / classical ML / deep learning), compared thoughtfully (24–27 band)
- [ ] **Original ideas and deeper analysis** (>27 band), for example:
  - [ ] Robustness to simulated distortions (noise, blur, low brightness/contrast, partial occlusion) **and** techniques that improve robustness
  - [ ] Failure analysis with **fewer images** and/or **less precise labels**, plus improvements based on it
  - [ ] Other original, well-justified ideas
- [ ] Write as much of the code ourselves as we can; **cite every paper, library and repo** used

### Evaluation requirements

- [ ] Use the given train / val / test splits. **No leakage.** Tune on train+val only; use test **only** for final evaluation
- [ ] Report **mAP@0.5** at minimum
- [ ] Report **training and testing times** for every method
- [ ] Show **successful and failed examples**, explain the failures, explain why the best method wins, and propose future work

### Deliverables

- [ ] **Video:** ≤ 10 min, 720p/1080p MP4, ≤ **100 MB**
  - ~8 min of slides (intro → methods → results → discussion/future work)
  - ~2 min demo of the code running
  - **Every member presents**, appears on a live webcam overlay in a corner, and **says their name** when they start
- [ ] **Report:** PDF, 2-column IEEE format, ≤ **10 pages** of main text (references extra), ≤ **10 MB**
  - Sections: Introduction, Literature Review, Methods, Experimental Results, Discussion, Conclusion, References
- [ ] **Code:** ZIP, ≤ **25 MB**, no trained models, input images or result images
  - README with run instructions and a clear list of third-party code and libraries
  - Inline comments, clean structure

### Rules

- [ ] Work with the group assigned on WebCMS (50% penalty otherwise)
- [ ] Every member contributes roughly equally to **methods/code, video and report**. A peer survey follows submission, and individual marks can be moderated.

---

## 2. Dataset Notes

**Source:** Kai Liu, *Beach Bird Detection Dataset v1.0*, Kaggle, Sept 2026, https://www.kaggle.com/datasets/liugalhigh/beach-bird

| Split | Images |
|---|---|
| Train | 1,369 |
| Validation | 291 |
| Test | 240 |
| **Total** | **1,900** (< 400 MB) |

**Characteristics that matter for method design:**

- Single class (bird), bounding-box annotations
- Ranges from **negative images (no birds)** to **dense flocks of 100+ birds** (max 130 per image)
- Expect **small objects**, **overlap/occlusion** in dense groups, and cluttered natural backgrounds (sand, water, waves, rocks, vegetation)

**First tasks (EDA, Week 7):**

- [ ] Download and verify counts per split
- [ ] Identify the annotation format (YOLO txt / COCO JSON / Pascal VOC XML) and write a loader that converts to one internal format
- [ ] Compute statistics and plots for the report:
  - Birds per image (histogram), share of negative images
  - Box width/height/area distribution (absolute pixels and relative to image size)
  - Box aspect ratios (feeds into anchor and sliding-window design)
  - Image resolutions
  - Density bins: `0`, `1–10`, `11–50`, `51–100`, `>100`
- [ ] Visualise ~20 random annotated images per split and note label quality issues

---

## 3. Method Portfolio

We plan **three method families plus advanced studies**, so that even if one method underperforms we still have more than the minimum.

### Method 1: Non-learning baseline (classical image processing)

**Category:** Non-learning (Weeks 2–3 lectures)

Pipeline:

1. Preprocessing: grayscale / colour-space conversion (HSV, Lab), Gaussian smoothing, CLAHE
2. Candidate generation, comparing **at least two** handcrafted approaches:
   - **Blob detection:** Laplacian of Gaussian / Difference of Gaussians at several scales (birds look like compact blobs against sand or water)
   - **Adaptive thresholding + morphology:** Otsu or local thresholding, then opening/closing, then connected components
   - (Optional) **MSER** regions or **background estimation** (large median filter, then subtract)
3. Box extraction from connected components or blob radius
4. Filtering by area, aspect ratio and solidity, with ranges taken from the EDA statistics
5. **Confidence score** for each box (needed for mAP), e.g. blob response strength or local contrast
6. Non-maximum suppression

Hyperparameters (scales, thresholds, area limits) tuned by grid/random search on **validation mAP@0.5**.

**Why:** gives a fast, interpretable baseline and shows what handcrafted features can and can't do on dense scenes.

### Method 2: Classical machine learning (handcrafted features + classifier)

**Category:** Machine learning (Weeks 3–5 lectures)

Pipeline:

1. **Region proposals:** either a multi-scale sliding window, or (better and faster) **reuse Method 1's candidates** at a high-recall setting
2. **Features,** comparing at least two:
   - HOG
   - LBP histograms
   - Colour histograms / colour moments
   - (Optional) concatenated HOG+LBP+colour
3. **Classifiers,** comparing at least two:
   - Linear SVM
   - Random Forest or Gradient Boosting
4. **Hard negative mining:** retrain using false positives from background regions
5. Classifier probability or margin as the box score, then NMS

Training patches: positives cropped from GT boxes (with small jitter), negatives sampled from background with IoU < 0.3.

**Why:** a direct, measurable improvement over Method 1 that isolates the value of learned decision boundaries over hand-tuned rules.

### Method 3: Deep learning detectors

**Category:** Deep learning (Weeks 7–8 lectures)

Compare **at least two architectures:**

- **3a. One-stage:** YOLO (e.g. Ultralytics YOLOv8/YOLO11, small or nano variant), fine-tuned from COCO pretrained weights
- **3b. Two-stage:** Faster R-CNN with ResNet-50-FPN (torchvision), fine-tuned
- (Optional) **3c. Transformer:** RT-DETR or DETR, if compute allows

Compare **data augmentation strategies** (at least two configurations):

- Baseline: flips + scale jitter
- Strong: mosaic, colour jitter (HSV), random crop, copy-paste of birds

Small-object and density tricks (also count as "original ideas"):

- Higher input resolution (e.g. 1024 vs 640)
- **Tiled training/inference** (SAHI-style slicing with overlap, then merge with NMS)
- Anchor/prior tuning from the box-size statistics
- Raising the max-detections-per-image limit (needs to be ≥ 130, and is easy to miss)

Compute: local GPU/Apple MPS, or Google Colab free tier. Fix the epoch budget and seeds, and log wall-clock training time.

### Advanced studies (for the >27 band)

At least **two** of these, ideally all three:

**A. Robustness to image distortions**

- Build a distorted copy of the **test set** (generated deterministically, fixed seed) at 3 severity levels each:
  - Gaussian noise, Gaussian/motion blur, low brightness/contrast, JPEG compression, synthetic occlusion (random rectangles over part of a box)
- Evaluate **all methods** to get mAP vs severity curves
- **Mitigations:**
  - DL: train with distortion augmentation and compare clean vs robust-trained models
  - Classical: denoising (bilateral/median), CLAHE, and photometric normalisation before detection
- Use **validation** distortions for any tuning. The test set stays evaluation-only.

**B. Label and data efficiency**

- Train DL (and ML) models on **10%, 25%, 50%, 100%** of the training images and plot mAP against training-set size
- **Imprecise labels:** jitter GT box coordinates (±10%, ±20%), randomly drop 10–30% of boxes, then measure the impact
- Try a fix and measure it: e.g. a loss more tolerant to noisy boxes, pseudo-labelling or self-training on the unused images, or stronger augmentation in the low-data regime

**C. Density-stratified and failure analysis**

- mAP@0.5 and **counting error (MAE)** per density bin (`0`, `1–10`, `11–50`, `51–100`, `>100`)
- False positives on negative images
- Performance by object size (small/medium/large)
- Confusion categories: missed in dense cluster, false positive on rock/shell/wave, duplicate detections, localisation error

---

## 4. Evaluation Protocol (shared by all methods)

**One shared evaluation script** that every method feeds. This is the most important piece of infrastructure, and it should be ready by the end of Week 7.

- **Common prediction format:** one JSON/CSV per method and split, with rows `image_id, x1, y1, x2, y2, score`
- **Metrics:**
  - **mAP@0.5** (required), computed with `pycocotools` or `torchmetrics`, plus a small hand-written check against it
  - mAP@0.5:0.95, Precision, Recall, F1 at the best threshold
  - Counting MAE / RMSE per image
  - Per-density-bin and per-size breakdowns
- **Timing:** training time (total wall clock) and **inference time per image** (ms/img), with hardware recorded (CPU/GPU model)
- **Data discipline:**
  - Tune only on `val`
  - `test` is run **once per final model**, from a separate script/flag
  - Fix random seeds and log every config
- **Results table** (master CSV in `results/`, updated by every method owner):

| Method | Variant | mAP@0.5 | mAP@0.5:0.95 | P | R | Count MAE | Train time | Infer (ms/img) | Hardware |
|---|---|---|---|---|---|---|---|---|---|

---

## 5. Team Roles

Five members, each **owning one technical area AND one slice of the report and video**, which meets the equal-contribution rule. Fill in names:

| Member | Technical ownership | Report sections | Video segment |
|---|---|---|---|
| **A** — ______ | Data loader, EDA, shared evaluation script, results table, repo/README | Introduction + dataset; Experimental setup | Intro + dataset (~1.5 min) |
| **B** — ______ | Method 1: non-learning | Lit review (classical); Method 1 | Method 1 (~1.5 min) |
| **C** — ______ | Method 2: handcrafted features + classifiers | Lit review (ML); Method 2 | Method 2 (~1.5 min) |
| **D** — ______ | Method 3a/3b: DL detectors + augmentation + tiling | Lit review (DL); Method 3 | Method 3 + results (~2 min) |
| **E** — ______ | Advanced studies: robustness + label efficiency (runs other members' models) | Discussion; Conclusion/future work | Discussion + future work (~1.5 min), demo lead |

- Everyone reviews another member's report section (rotate A→B→C→D→E→A)
- The demo (~2 min) can be shared: each owner shows their method running briefly, with E editing it together
- Pick one **submitter** (suggest A) and one **video editor** (suggest E)

---

## 6. Timeline

### Week 7 (Thu 29 Oct – Sun 1 Nov): setup and baselines

- [ ] Group meeting: confirm roles, comms channel, meeting cadence (e.g. Mon/Thu stand-ups)
- [ ] Shared Git repo with the structure in §8; Overleaf project from the IEEE conference template
- [ ] **A:** download data, EDA, unified loader, eval script with tests on GT-as-prediction (should give mAP = 1.0)
- [ ] **B:** first end-to-end blob detector producing predictions on val
- [ ] **C:** patch-extraction pipeline (positives/negatives)
- [ ] **D:** YOLO fine-tune baseline running (even a few epochs) on Colab/local
- [ ] **E:** distortion generator module (noise, blur, brightness, JPEG, occlusion) with visual checks
- [ ] Everyone: collect 3–5 key references for their lit-review area

### Week 8 (2–8 Nov): core methods complete

- [ ] **Tutor consult:** show EDA plus baseline numbers and ask for feedback on the advanced-study plan
- [ ] **B:** compare ≥2 handcrafted approaches; tune on val
- [ ] **C:** HOG vs LBP (vs colour) × SVM vs RF; hard negative mining; tune on val
- [ ] **D:** YOLO + Faster R-CNN trained; augmentation comparison; try high-res/tiling
- [ ] **E:** robustness evaluation harness running on early models; start the data-fraction experiments (10/25/50%)
- [ ] **A:** keep results CSV updated; automatic qualitative visualisation (GT vs pred overlays: green TP, red FP, blue FN)
- [ ] **Milestone (Sun 8 Nov):** every method produces val predictions through the shared eval script

### Week 9 (9–15 Nov): advanced studies, final runs, writing

- [ ] **Tutor consult:** present the comparison and discuss interpretations
- [ ] **E + D:** robustness-trained DL model; label-noise experiments; one mitigation per study
- [ ] **All:** freeze final configs by **Wed 11 Nov** and run **test-set evaluation once**
- [ ] **A:** final results tables + figures (mAP bars, mAP vs severity, mAP vs data fraction, per-density-bin)
- [ ] **All:** failure-case gallery (each owner picks 2 successes + 2 failures for their method)
- [ ] **All:** first full report draft of own sections in Overleaf by **Sun 15 Nov**
- [ ] Slide deck skeleton (shared PowerPoint, consistent template)

### Week 10 (16–20 Nov): polish and submit

- [ ] **Mon 16:** report peer review swap; slides finalised
- [ ] **Tue 17:** each member records their segment with webcam overlay; demo recorded
- [ ] **Wed 18:** video assembled, timed (≤ 10:00), compressed (HandBrake "Very Fast 720p30") to ≤ 100 MB
- [ ] **Wed 18:** report final pass: page count ≤ 10, figures legible, all citations present, PDF ≤ 10 MB
- [ ] **Thu 19:** code cleanup, README, ZIP ≤ 25 MB, **test fresh-clone run instructions**
- [ ] **Thu 19:** dry-run submission checklist (§11)
- [ ] **Fri 20 by midday:** submit (6 hours buffer before the 18:00 deadline)

---

## 7. Report Plan (IEEE 2-column, ≤ 10 pages)

| Section | Pages | Content | Owner(s) |
|---|---|---|---|
| Abstract | — | 150–200 words: task, methods, best mAP, key finding | D |
| 1. Introduction | 0.75 | Ecological motivation, task definition, dataset summary + stats figure, contributions list | A |
| 2. Literature Review | 1.5 | Classical blob/threshold detection; HOG/LBP + SVM (Dalal & Triggs); CNN detectors (Faster R-CNN, YOLO, DETR); small-object detection (FPN, SAHI); bird detection datasets (spec references) | B, C, D |
| 3. Methods | 2.5 | One subsection per method with pipeline diagram; design choices justified from EDA; advanced-study design | B, C, D, E |
| 4. Experimental Results | 2.5 | Setup (splits, hardware, metrics, tuning protocol); main results table; ablations (features, classifiers, architectures, augmentation); robustness curves; data-efficiency curves; density breakdown | A, E |
| 5. Discussion | 1.5 | Why the best method wins; failure-case figure with explanations; trade-off of speed vs accuracy; what the advanced studies reveal | E, all |
| 6. Conclusion | 0.5 | What worked, what didn't, limitations, future work | E |
| References | extra | Spec papers, dataset, every library (OpenCV, scikit-image, scikit-learn, PyTorch, torchvision, Ultralytics, pycocotools) and repo | A |

**Key figures (aim for ~8):**

1. Dataset examples across density levels + birds-per-image histogram
2. Pipeline diagrams (one combined figure with 3 panels)
3. Main results bar chart (mAP@0.5 + inference time)
4. Precision–recall curves for the best variant of each method
5. mAP vs distortion severity (one line per method)
6. mAP vs training-data fraction / label noise
7. Per-density-bin mAP and counting error
8. Qualitative grid: successes and failures per method

---

## 8. Code Repository Structure

```
bird-detection/
├── README.md                 # setup, how to run each method, third-party code credits
├── requirements.txt          # pinned versions
├── configs/                  # YAML per method/experiment
├── data/                     # (gitignored) dataset goes here
├── src/
│   ├── data/                 # loaders, format conversion, EDA
│   ├── eval/                 # mAP, counting, timing, per-bin analysis
│   ├── classical/            # Method 1: blob/threshold detectors
│   ├── ml/                   # Method 2: features, classifiers, sliding window/proposals
│   ├── dl/                   # Method 3: YOLO + Faster R-CNN training/inference wrappers
│   ├── robustness/           # distortion generators + mitigations
│   ├── data_efficiency/      # subset sampling, label-noise injection
│   └── viz/                  # overlays, plots for report
├── scripts/                  # run_m1.py, run_m2.py, train_dl.py, evaluate.py, demo.py
├── notebooks/                # EDA, exploratory (keep light)
└── results/                  # (gitignored except CSVs) predictions, metrics
```

**Conventions:**

- Every script takes `--split {train,val,test}`. `test` only gets used for final runs.
- One `evaluate.py` entry point: `python scripts/evaluate.py --pred results/m1_val.json --split val`
- `demo.py`: runs a chosen method on a few images and shows/saves overlays (used for the video)
- Before zipping: exclude `data/`, weights (`*.pt`, `*.pth`, `*.pkl`), and result images

---

## 9. Video Plan (≤ 10:00)

| Time | Segment | Presenter |
|---|---|---|
| 0:00–1:15 | Problem, motivation, dataset + challenges (density, small objects, negatives) | A |
| 1:15–2:45 | Method 1: classical pipeline, design choices, results | B |
| 2:45–4:15 | Method 2: features + classifiers, hard negatives, results | C |
| 4:15–6:00 | Method 3: DL detectors, augmentation, tiling, results + overall comparison table | D |
| 6:00–7:45 | Robustness + data efficiency findings, failure analysis, future work | E |
| 7:45–9:45 | **Demo**: code running for each method on sample images (edited screen recording) | E (voice), others briefly |
| 9:45–10:00 | Close | — |

**Recording checklist:**

- Live webcam visible in a corner for each presenter (PowerPoint "Record" or OBS)
- Each person **states their name** at the start of their segment
- 720p/1080p MP4; check final length is **≤ 10:00**; compress to **≤ 100 MB**
- Keep a ~10 s safety margin. Content beyond 10:00 is ignored.

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Short timeline (~3 weeks) | High | Start baselines immediately; freeze configs Wed W9; strict milestones |
| GPU availability / Colab limits | Medium | Use small model variants, fewer epochs, or 50–75% of the data (allowed if reported); checkpoint often |
| Classical methods near-zero mAP on dense scenes | Medium | Expected and fine. Report honestly, analyse why, and use them as proposals for Method 2 |
| Max detections cap silently limits dense images | Medium | Set max detections ≥ 300 in all detectors and in the eval |
| Accidental test-set leakage | High | Separate test flag, single final run, code review of data paths |
| Uneven contribution | High (moderation) | Role table above; weekly check-ins; log contributions in the repo/commit history |
| Report over 10 pages | Medium | Page budget per section; draft early; move detail to compact tables |
| Video too long or too large | Medium | Time each segment in rehearsal; compress with HandBrake |
| Annotation format surprises | Low | Resolve in EDA on day 1 |

---

## 11. Final Submission Checklist

**Video**

- [ ] ≤ 10:00, MP4, 720p or 1080p, ≤ 100 MB
- [ ] Order: intro → methods → results → discussion/future → ~2 min demo
- [ ] All 5 members present, each visible on live webcam, each states their name
- [ ] Slides support the narrative (PowerPoint)

**Report**

- [ ] IEEE 2-column template, ≤ 10 pages main text, PDF ≤ 10 MB
- [ ] All 7 required sections present
- [ ] mAP@0.5 for every method, plus train and test times
- [ ] Success and failure examples with explanations
- [ ] Comparison and reasons why the best method is best
- [ ] Future work
- [ ] Every paper, dataset, library and repo cited

**Code**

- [ ] ZIP ≤ 25 MB, no models/images/results
- [ ] README: environment setup, data placement, commands per method, how to reproduce tables, third-party code clearly listed
- [ ] Inline comments, consistent structure
- [ ] Tested from a fresh clone

**Admin**

- [ ] Single submitter confirmed; submission done well before **Fri 20 Nov 2026, 18:00 AET**
- [ ] All members complete the post-deadline contribution survey

---

## 12. Starter Reading List

- Spec references: Rodriguez-Juan et al. 2025 (Visual WetlandBirds); Wilson et al. 2026 (Big Bird); Zou et al. 2023 (Object Detection in 20 Years)
- Dalal & Triggs, *Histograms of Oriented Gradients for Human Detection*, CVPR 2005
- Ojala et al., *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with LBP*, TPAMI 2002
- Lindeberg, *Feature Detection with Automatic Scale Selection* (LoG blob detection), IJCV 1998
- Ren et al., *Faster R-CNN*, NeurIPS 2015
- Lin et al., *Feature Pyramid Networks*, CVPR 2017
- Redmon et al., *YOLO*, CVPR 2016, plus Ultralytics YOLOv8/YOLO11 docs
- Carion et al., *DETR*, ECCV 2020; Zhao et al., *RT-DETR*, CVPR 2024
- Akyon et al., *Slicing Aided Hyper Inference (SAHI)*, ICIP 2022
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions*, ICLR 2019

# Slides: advanced studies, discussion and future work (Member E's share, presented by Johnnie)

Four slides for ~1:45. Same template as the rest of the deck. Webcam overlay in a corner.
Every number comes from a committed CSV (named under each slide).

---

## Slide 1: How robust are the detectors?

**Figure:** `docs/report/figures/e_robustness.png` (top row only, crop the bottom row)

- 5 corruptions × 3 severities on the test set, all 4 methods, fixed seed
- CNNs stay far ahead, but keep only ~63% of their score on average (classical: 71–86%)
- Occlusion hurts everyone most; Faster R-CNN tolerates heavy noise better than YOLO (0.42 vs 0.22)

*Source: results/robustness_test.csv*

---

## Slide 2: Fixing it: preprocessing vs training

**Figure:** `docs/report/figures/e_robustness.png` (bottom row)

- Denoising for M1/M2, chosen on val (CLAHE was rejected, it hurts M1): no real gain
- YOLO trained with the same corruptions: clean score unchanged (0.875), distorted mean 0.56 → 0.77 (best case)
- Caution: it learned "grey patch = bird", firing on 96% of grey squares pasted away from birds (val)

*Source: results/robustness_test.csv, results/occluder_shortcut_val.csv*

---

## Slide 3: How much data and how clean?

**Figure:** `docs/report/figures/e_data_efficiency.png`

- 10% → 100% of training images: 0.757 → 0.876 test mAP@0.5; flat beyond ~50%
- 20% of boxes deleted: no harm (0.881); every box edge jittered ±20%: 0.845
- Stronger augmentation with 10% of data: no clear gain (0.775 vs 0.757, worse box precision)

*Source: results/data_efficiency_test.csv*

---

## Slide 4: Where they still fail, and what next

**Figure:** `docs/report/figures/e_failures.png` + one gallery image (`docs/report/figures/gallery/yolo_worst_1_beach_bird_1790.jpg`)

- Most misses are birds inside flocks; small birds found 61% vs large 89% (YOLO)
- Future work: tiled/high-resolution inference for small birds; density-map counting for flocks;
  occluders with random texture and placement; test set with >100-bird flocks

*Source: results/failure_categories_test.csv, analysis/size_bins.csv*

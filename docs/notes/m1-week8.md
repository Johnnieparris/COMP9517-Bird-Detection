# Method 1, Week 8 notes (Member B)

All numbers are from `val`, using `src/classical/quick_eval.py`. A's shared evaluator hasn't landed, so they haven't been cross-checked yet. `test` has not been run. Configs freeze Wed 11 Nov.

## Comparison

From `results/m1_comparison.csv`, plus search times. Inference timings are serial, on CPU. P, R, F1 and Count MAE are all at each variant's best-F1 score threshold. "Max R" is recall@0.5 using every detection. mAP@0.5:0.95 waits for A's evaluator.

| Method | Variant | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 | Max R | Count MAE | Train time | Search time | Infer (ms/img) | Hardware |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | LoG blobs, Week 7 | 0.0011 | | 0.009 | 0.029 | 0.013 | 0.135 | 19.50 | 0 | none | 230 | Apple M5 Pro, CPU |
| M1 | LoG blobs + contrast score | 0.0033 | | 0.048 | 0.030 | 0.037 | 0.127 | 4.93 | 0 | none | 235 | Apple M5 Pro, CPU |
| M1 | LoG blobs + contrast, tuned | 0.0211 | | 0.097 | 0.075 | 0.085 | 0.225 | 5.51 | 0 | 547 s | 113 | Apple M5 Pro, CPU |
| M1 | Adaptive threshold, tuned | 0.0977 | | 0.309 | 0.169 | 0.218 | 0.206 | 4.90 | 0 | 206 s | 2.9 | Apple M5 Pro, CPU |
| M1 | **Lab colour distance + Otsu, tuned** | **0.1024** | | 0.311 | 0.181 | 0.229 | 0.246 | 4.78 | 0 | 535 s | 4.9 | Apple M5 Pro, CPU |

Search times cover two rounds of 200, 100 and 40 trials, on 12 workers. Round 2 shifted lists where round 1's best sat on an edge, adding values at that end and removing some at the other. Several round 2 values are on an edge again (`min_fill` 0.1, for example). I stopped there as planned, because more searching would only fit `val` harder.

## Why segmentation beats blobs

- **Colour and adaptive are effectively tied; both clearly beat blobs.** A paired bootstrap on val in the final code review (300 resamples, not in the repo) gave colour minus adaptive AP of +0.005, 95% CI [-0.012, +0.023], and train agrees (0.0416 against 0.0402). I kept colour as the main config for its higher recall, not because it is proven better. Adaptive is faster.
- **Segmentation rejects clutter.** Tuned colour gives 8.1 detections per image against 120 for tuned blobs. Ranking and precision drive AP: tuned blobs have similar maximum recall (0.225 against 0.246) at a fifth of the AP.
- **Its boxes take their shape from the bird.** LoG boxes are square.
- **Contrast scoring** tripled blob AP (0.0011 to 0.0033).
- **`min_area` is what controls false positives, not the Otsu floor.** On the tuned colour config:

| Change | Val AP@0.5 | Detections per negative image |
|---|---|---|
| None (`min_area` 256, floor 12) | 0.1024 | 5.2 |
| `min_area` 64 | 0.0657 | 17.8 |
| `min_area` 16 | 0.0194 | 75.6 |
| Floor 0 | 0.1024 | 5.2 |

The floor has no measurable effect on val, because Otsu's own threshold almost always sits above it. It only protects near-featureless images, such as the synthetic test. My Week 8 plan expected it to be the next gain, and that was wrong.

- **That trade costs small birds.** `min_area` 256 means birds under about 16×16 px can't be found at all (see below). Small boxes are few (154 of 1,843), so this costs little AP here, but a study of distant flocks would want a different setting.
- **Tuned parameters.** Colour: `downscale` 8, `bg_kernel` 31, `min_distance` 12, `morph_kernel` 1, `min_area` 256, `max_aspect` 2.5, `min_fill` 0.1, `nms_iou` 0.7. Adaptive: `block` 201, `c` 25, `morph_kernel` 3, `min_area` 256, `max_aspect` 2.5, `min_fill` 0.1. Blob: sigmas 13.08–37.49, `threshold` 0.03, no CLAHE, `nms_iou` 0.5, `min_area` 196.

**Before the Wed 11 Nov freeze:** record each variant's `score_threshold` from the val table as final. P, R, F1 and MAE on `test` must use that value, because choosing it again on `test` would be leakage.

## How far to trust these numbers

The tuned colour config scores AP 0.0416 on `train` against 0.1024 on `val`. This misses the plan's "within about 30%" check, but it doesn't look like search overfitting:

- The untuned starting config, which never saw a search, shows a similar gap: 0.0065 on train, 0.0172 on val.
- The top `val` configs also rank at the top on `train`, and tuning raised train AP by about 6×.
- Average GT per image is the same on both splits (6.33). Why `val` is easier isn't known.

Expect `test` to land anywhere between these two. **In the report, give train AP beside val AP**, and don't present 0.10 as the expected performance.

| Variant | Val AP@0.5 | Train AP@0.5 | Train recall@0.5 |
|---|---|---|---|
| Blob, tuned | 0.0211 | 0.0095 | 0.196 |
| Adaptive, tuned | 0.0977 | 0.0402 | 0.161 |
| Colour, tuned | 0.1024 | 0.0416 | 0.191 |

## Where it fails

Tuned colour config on `val`. Here a GT counts as found if any detection reaches IoU 0.5 with it, so this is slightly more generous than the one-to-one figures above. "Overlapped" means best IoU from 0.1 up to 0.5.

| Birds in image | Images | GT boxes | Found at IoU 0.5 | Overlapped | Detections per image |
|---|---|---|---|---|---|
| 0 | 33 | 0 | | | 5.2 |
| 1–5 | 162 | 396 | 0.432 | 0.316 | 7.3 |
| 6–20 | 77 | 748 | 0.294 | 0.421 | 10.1 |
| 21–50 | 16 | 488 | 0.117 | 0.275 | 10.6 |
| 51+ | 3 | 211 | 0.066 | 0.256 | 17.0 |

| GT area (px²) | GT boxes | Found at IoU 0.5 |
|---|---|---|
| under 256 | 154 | 0.000 |
| 256–1,024 | 497 | 0.125 |
| 1,024–9,216 | 833 | 0.337 |
| over 9,216 | 359 | 0.331 |

Of 1,721 false positives on images with birds, 70% (1,213) overlap no bird, 18% (311) are fragments smaller than the bird, and 11% (197) are larger than it.

From a 3×3 grid of random val images, in order of how much each seems to cost:

1. **False positives on background objects.** Rocks, seaweed, reflections, shoreline buildings and the far bank all differ from the median background. Bird reflections on wet sand often get their own box. This is the 70% above, and birds' colour alone can't separate them.
2. **Fragmented birds.** Only the part unlike the background is found: a body without the head, or a bright belly on a dark bird. Large close-up birds suffer most, because a big bird can dominate the median background.
3. **Thin parts missing from the box.** Legs and wing tips are in the GT box but too thin or faint for the mask, so a correct detection can still miss IoU 0.5. In-flight birds with long wings show this clearly.
4. **Merged birds in flocks.** Recall falls sharply with density, and one box often covers several overlapping birds. Note `51+` has only 3 val images.
5. **Small birds.** Excluded outright by `min_area`.

**Silent failure on dense scenes.** Nothing flags these, and `n_images_capped` stays 0. In the code review's synthetic tests on 156 separate discs, tuned colour and tuned adaptive both returned nothing for radius 4 and 8 at 24 px spacing. Tuned adaptive also returned nothing at radius 10 at 28 px spacing, where the flock pulls down the local mean and the mask merges into one rejected component. Above roughly 50% coverage, the median background becomes the birds, and colour finds about 90 of 156. The proposal config kept all 156 for radius 4 and 8. Global Otsu can also hide faint birds when strong ones are in the same image.

It does well on compact birds with clear colour contrast on plain water or sand.

## Tried and dropped

- **Hessian edge-ratio test on LoG peaks** (Week 7's first candidate). In a throwaway prototype with hand-set config, it barely moved AP (0.0033 to 0.0034) and added about 1 s per image. The prototype isn't in the repo, so this isn't a reportable result without rerunning it.
- **Watershed splitting of touching birds.** Val AP fell at every setting (`split_distance` 8, 15, 25, 40 gave 0.0792, 0.0851, 0.0885, 0.0855, against 0.1024) and inference took 12 ms/img. The likely reason, not yet checked on overlays, is that it splits single birds with irregular outlines while few images are dense. Code not kept; measured on the tuned colour config at commit f7d6291 plus the discarded change.

## Tutor consult, Thu 5 Nov

Not yet held. Questions to ask, and record the answers here:

1. Is it enough for Method 1 to compare a blob detector against colour segmentation at AP around 0.04–0.10, if the failure analysis is strong?
2. Does a search on `val` with a check on `train` satisfy the no-leakage rule?
3. What should be added to the advanced-study plan for the >27 band?

## Handoff

**Member A**
- The file to evaluate is `results/m1_val.json`, with its `.meta.json` sidecar: `python scripts/run_m1.py --config configs/m1_seg_colour_tuned.yaml --split val` regenerates it. It is the tuned colour config.
- `scripts/evaluate.py` isn't on `origin/main` yet, so the Sunday milestone's "through the shared eval script" isn't met until it lands. Please cross-check its mAP@0.5 against `quick_eval.ap50`, expecting agreement within about 0.002. Please pass on when it lands so `interim_data.py` can be swapped out.
- The team still needs to choose density bins. This note uses the README's.
- `results/m1_comparison.csv` is ready to copy into the master table. Use `recall_at_best_f1` for the R column. `recall50` is the maximum recall.

**Member C**
- Use `load_config("configs/m1_seg_proposals.yaml")` then `detect(image, cfg)`.
- Unlike the tuned detectors, this config also keeps dense small birds.
- These are much better than the blob proposals (`results/m1_proposal_recall.csv`):

| Proposals | Recall@0.5 | Recall@0.3 | Candidates/img | ms/img |
|---|---|---|---|---|
| Blob | 0.170 | 0.523 | 1,116 | 244 |
| Colour segmentation | 0.300 | 0.515 | 124 | 6.6 |

- Recall@0.5 of 30% still caps Method 2, so **keep the sliding-window recommendation**, or regress or refine boxes from these candidates. Recall@0.3 shows many birds are found but badly boxed.

**Member D**
- Where a learned detector should beat this: rocks and reflections, which need more than colour; birds with thin parts; and dense flocks. These are also good contrasts for the Discussion section.

**Member E**
- Use `load_config(path)` then `detect(image, cfg)`, on any uint8 BGR array, at about 5 ms/img for the colour config. `BlobConfig.from_yaml` still works for blob configs.
- Median background and Otsu are likely to react strongly to low contrast, and the adaptive config's fixed `c` likewise. The proposal config gave 1,379 candidates in a synthetic test at noise sigma 8, so expect noise to raise its candidate count. All worth watching in the robustness curves.

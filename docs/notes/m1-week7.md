# Method 1, Week 7 notes (Member B)

Week 8 results and the current handoff are in `docs/notes/m1-week8.md`.

All numbers are from `val`, using `src/classical/quick_eval.py`. A's shared evaluator hasn't landed, so they haven't been cross-checked yet. `test` has not been run.

## Baseline

| Method | Variant | mAP@0.5 | mAP@0.5:0.95 | P | R | Count MAE | Train time | Infer (ms/img) | Hardware |
|---|---|---|---|---|---|---|---|---|---|
| M1 | LoG blobs, `configs/m1_blob.yaml` | 0.0011 | | | 0.135 | | 0 | 232 | Apple M5 Pro, CPU |

The score is close to zero, as §10 of the project plan anticipated. It's a real result rather than a bug: the overlays show GT boxes correctly aligned, and the loader matches the README's counts exactly (8,663 train and 1,843 val boxes).

**The 300-box cap is active:** 176 of 291 val images hit it. Before the half-pixel fix, removing the cap raised recall@0.5 from 0.131 to 0.139 and left AP about the same (0.0010), so it costs a little recall, mostly on low-contrast birds. Each run's sidecar now records `n_images_capped`.

Polarity on all of val:

| Polarity | AP@0.5 | Recall@0.5 |
|---|---|---|
| dark | 0.0008 | 0.087 |
| bright | 0.0004 | 0.054 |
| both | 0.0011 | 0.135 |

I kept `both`, which is best on both measures. A first check on 50 images put `bright` slightly ahead on AP, but that didn't hold up on the full split.

## What the dataset changes

- All images are 512×512, YOLO txt, with empty label files for negatives. `max_side` never triggers.
- Birds are bigger than the project plan assumed: on train, the median box is about 45×42 px, and the 5th–95th percentile width is 13–206 px. Width/height spans 0.54–2.18 (5th–95th percentile).
- Sigmas come from train boxes: 4.56 to 63.47.
- The densest val image has exactly 100 birds, so §2's `>100` bin is empty on val. The README's own bins are `0, 1–5, 6–20, 21–50, 51+`, and `51+` has only 17 images (3 in val). The team should pick one set of bins.

## What the overlays show

From 20 random val images, in order of how much each seems to cost:

1. **Texture and edge false positives.** About 257 detections per image on average after the cap. Reeds, ripples, wave edges, rocks and shoreline each produce dozens, and they often score as high as real birds. LoG also responds along edges: one clean dark disc gets a ring of extra boxes around it (44 extra at r=30 in the code review, scoring about 0.1–0.2 against 0.5–0.6 for the main peak), and boxes trace the aeroplane's fuselage. How much this contributes on real images hasn't been measured. This is why AP is so low even where birds are found.
2. **Fragmented large birds.** LoG fires on a body patch or a head instead of the whole bird, so a large gull or goose gets several small boxes and none reaches IoU 0.5.
3. **Square boxes.** Small compact birds on plain sand or water are often found, but the square shape misses IoU 0.5. One-to-one recall is 41% at IoU 0.3 versus 13.5% at IoU 0.5, so many birds are found but badly boxed.
4. **Non-bird objects.** One negative image of an aeroplane gets a large number of detections along the fuselage.

## Week 8 candidates, ranked

1. Better scoring and rejection, since ranking is the main AP problem: reject edge responses with a Hessian ratio test, score by local contrast against a surrounding ring, filter out detections in textured regions, and tune `threshold`. Fewer false boxes also takes pressure off the cap.
2. Second approach: adaptive thresholding, morphology and connected components. Its boxes take their shape from the segment, which addresses items 2 and 3 directly. Colour-distance background estimation may suit plain sand and water.
3. Refine each blob's box from a local segmentation.
4. Tune sigmas and `min_area`, and try CLAHE.

## Handoff

**Member A**
- Predictions are a JSON list of `{"image_id", "x1", "y1", "x2", "y2", "score"}` in pixels. `image_id` is the file stem, such as `beach_bird_1390`. Each file has a `.meta.json` sidecar with config, ms/img, cap hits, platform and git SHA.
- Images with no detections have no rows, so the evaluator must take its image list from the split, not the predictions.
- Set max detections ≥ 300: the `pycocotools` default of 100 would cap dense images. Method 1 baseline files can have up to 300 boxes per image.
- `src/classical/quick_eval.py` is available as the hand-written check. Replace `interim_data.py` with your loader when it lands, and check the dataset folder's path, which contains spaces.

**Member C**
- **The blob candidates are not good enough to be Method 2's only proposals.** With `configs/m1_blob_proposals.yaml`, recall on val is 17.0% at IoU 0.5 and 52.3% at IoU 0.3, at about 1,116 candidates per image. That 17.0% would cap Method 2's recall, so plan on a multi-scale sliding window, or on regressing or refining boxes from these candidates.
- To use them anyway: `detect(image, BlobConfig.from_yaml("configs/m1_blob_proposals.yaml"))`.

**Member E**
- `detect` takes any uint8 BGR array, so distorted images need no file writes. It takes about 230 ms per image on CPU.

**Member D**
- Birds are mostly not tiny at 512×512, so tiling may matter less than the project plan expected. Check this against the small-box tail (5th percentile width is 13 px).

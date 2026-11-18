# Method 2, Week 9 notes (Member C)

Unless a row says otherwise, numbers are from `val`, scored with the shared
`src/evaluation.py` and the splits in `splits/official.json`. The final `test`
run has now been done, **once**, for the frozen configuration (see below).

Hardware for every row: Apple M1 Pro (10 core), 16 GB, CPU only, serial inference.

Method 2 turns detection into patch classification: candidate boxes, a handcrafted
descriptor per box, a trained classifier, then NMS. Where Method 1 sets its
thresholds by hand, the accept/reject boundary here is learned, which is the
comparison the two methods are meant to isolate.

## Comparison

From `results/m2_comparison.csv`. Precision, recall, F1 and count MAE are at the
score threshold that maximises F1 on val, and that threshold is in the `thr`
column; see "A note on score thresholds" below for why a single fixed cut-off is
not usable here. mAP@0.5 is threshold-free and is the number to compare.

| Features | Classifier | HN rounds | Proposals | mAP@0.5 | P | R | F1 | Count MAE | thr | Train s | Infer ms/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HOG+LBP+colour | **GB** | 1 | **M1 seg + box reg** | **0.1848** | 0.442 | 0.237 | 0.308 | 4.09 | 0.715 | 103 | 79 |
| HOG+LBP+colour | GB | 1 | M1 seg + box reg (0.4) | 0.1819 | 0.343 | 0.275 | 0.305 | 4.29 | 0.576 | 102 | 75 |
| HOG+LBP+colour | GB | 1 | M1 seg + box reg (0.5) | 0.1761 | 0.337 | 0.271 | 0.300 | 4.29 | 0.576 | 100 | 75 |
| HOG+LBP+colour | GB | 1 | M1 seg | 0.1472 | 0.388 | 0.208 | 0.271 | 4.09 | 0.714 | 84 | **74** |
| HOG+LBP+colour | GB | 1 | window | 0.0754 | 0.173 | 0.139 | 0.154 | 4.87 | 0.772 | 308 | 907 |
| HOG+LBP+colour | RF | 1 | window | 0.0481 | 0.135 | 0.111 | 0.122 | 6.19 | 0.655 | 370 | 923 |
| HOG | RF | 1 | window | 0.0470 | 0.121 | 0.101 | 0.110 | 4.79 | 0.590 | 177 | 332 |
| HOG+LBP+colour | SVM | 1 | window | 0.0227 | 0.072 | 0.115 | 0.088 | 9.87 | 0.682 | 320 | 880 |
| HOG | SVM | 2 | window | 0.0221 | 0.091 | 0.073 | 0.081 | 5.71 | 0.655 | 202 | 457 |
| HOG | SVM | 1 | window | 0.0205 | 0.064 | 0.103 | 0.079 | 8.63 | 0.665 | 102 | 299 |
| colour | RF | 1 | window | 0.0148 | 0.041 | 0.068 | 0.051 | 9.83 | 0.590 | 125 | 376 |
| HOG | SVM | 0 | window | 0.0146 | 0.058 | 0.093 | 0.072 | 9.27 | 0.750 | 23 | 300 |
| HOG+LBP+colour | GB | 1 | blob | 0.0144 | 0.068 | 0.052 | 0.059 | 5.49 | 0.670 | 741 | 911 |
| LBP | RF | 1 | window | 0.0051 | 0.017 | 0.071 | 0.027 | 24.21 | 0.530 | 139 | 418 |
| colour | SVM | 1 | window | 0.0049 | 0.013 | 0.246 | 0.024 | 118.05 | 0.584 | 108 | 355 |
| LBP | SVM | 1 | window | 0.0017 | 0.006 | 0.236 | 0.012 | 237.81 | 0.492 | 202 | 421 |

For reference, Method 1's best on val is **0.1024** (`results/m1_comparison.csv`,
tuned Lab colour distance + Otsu).

## The main result: whose candidates, not whose classifier

The single largest factor is not the descriptor or the classifier. It is where the
candidate boxes come from. The same classifier on member B's Method 1 candidates
scores **0.1472** against **0.0754** on our own sliding window, and runs **12x
faster** (74 against 907 ms/image) and **3.7x faster to train**. Adding box
regression on top takes it to **0.1848** (next section).

This is the opposite of what the Week 8 hand-off predicted. `docs/notes/m1-week8.md`
says "Recall@0.5 of 30% still caps Method 2, so keep the sliding-window
recommendation". The recall ceiling is real and we confirm it, but it is not what
limits this method:

From `results/m2_proposal_recall.csv` (val, 1,843 ground-truth boxes):

| Proposals | Recall@0.5 | Recall@0.3 | Candidates/img | ms/img |
|---|---|---|---|---|
| Sliding window | **0.500** | **0.849** | 1,830 | 0.3 |
| Blob (DoG) | 0.169 | 0.550 | 1,266 | 302 |
| M1 colour segmentation | 0.304 | 0.563 | **124** | 11 |

The window has the *highest* ceiling (0.500) and still loses, because recall is not
the binding constraint: class imbalance is. A sliding window hands the classifier
roughly 1,830 boxes per image against about 6 birds, a ratio near 300:1. B's
segmentation hands it 124, a ratio near 20:1, and the boxes it does produce are
already shaped by the bird rather than being square. The effect shows up exactly
where that theory predicts, in precision: **0.388 against 0.173**, a 2.2x gain, for
a method whose recall ceiling is 40% lower.

So the ranking of proposal sources by recall (window > M1 > blob) is not the ranking
by final mAP (M1 > window > blob). Blob proposals are worst on both counts and also
the slowest to train (741 s), and should be dropped.

**Caveat, and it matters.** B tuned `configs/m1_seg_proposals.yaml` with a random
search on `val`. Method 2 on those candidates therefore inherits val-tuned
hyperparameters indirectly, so some of the 0.1472 is optimism, and it is not yet a
clean claim that Method 2 beats Method 1. The single `test` run decides it. Until
then this should be written up as "the hybrid is the strongest val configuration",
not as a win.

## Box regression: fixing the size, not finding more birds

The failure analysis below says the binding constraint is box *fit*, not
detection. With member B's candidates on val, 56% of birds are overlapped at
IoU 0.3 but only 30% reach IoU 0.5, and a quick oracle check found that **82% of
birds have a candidate whose centre lands inside them**. The centres are right
and the sizes are wrong.

That is the situation bounding-box regression was designed for. Hosang et al.
(PAMI 2015) put the general case plainly: "improving proposal localisation
accuracy is as important as improving recall", which is why their Average Recall
metric averages over IoU 0.5 to 0.95 rather than reporting a single threshold.

`src/ml/boxreg.py` follows R-CNN (Girshick et al., CVPR 2014): a **linear** ridge
regression from the patch features we already compute to four offsets, in the
standard scale-invariant parameterisation

    tx = (Gx - Px) / Pw      tw = log(Gw / Pw)
    ty = (Gy - Py) / Ph      th = log(Gh / Ph)

so that a 12 px bird and a 200 px bird pose the same problem. As in R-CNN it uses
a large regularisation constant, is trained only on candidates that already
overlap a bird (a linear correction is only locally valid), and is applied once
rather than iteratively. Because the classifier has already computed a feature
vector for every candidate, the regressor reuses that same pass and generates no
extra candidates.

| Config | mAP@0.5 | P | R | F1 | Train s | Infer ms/img |
|---|---|---|---|---|---|---|
| concat+GB on M1 candidates | 0.1472 | 0.388 | 0.208 | 0.271 | 84 | 73.6 |
| **+ box regression** | **0.1848** | **0.442** | **0.237** | **0.308** | 103 | **79.1** |

**+25.5% relative mAP for +7% inference cost**, trained on 4,225 candidate/bird
pairs. Precision, recall and F1 all move together, which is what a localisation
fix should do: the same detections simply land inside IoU 0.5 more often.

**The training-overlap threshold matters, and the R-CNN default is wrong here.**
`boxreg_min_iou` decides which candidates are close enough to learn from:

| `boxreg_min_iou` | Training pairs | mAP@0.5 | P | R |
|---|---|---|---|---|
| **0.3** | 4,225 | **0.1848** | 0.442 | 0.237 |
| 0.4 | 3,277 | 0.1819 | 0.343 | 0.275 |
| 0.5 (R-CNN's value) | 2,371 | 0.1761 | 0.337 | 0.271 |

Looser is better on this dataset. R-CNN used 0.5 with selective-search proposals,
which already fit their objects more tightly than ours do; with segmentation
candidates that mostly sit at IoU 0.3-0.5, a 0.5 cut-off discards 44% of the
pairs and trains the regressor only on the cases that least need correcting. The
0.4 and 0.5 settings also sit at a different operating point, trading precision
(0.34) for recall (0.27). mAP prefers 0.3.

Worth stating honestly in the report: the oracle ceiling was 0.816 recall and we
are at 0.237, so a linear correction on handcrafted features recovers only a
small part of what perfect sizing would give. A learned feature, or a non-linear
regressor, would likely do better, and that is a concrete piece of future work
rather than a vague one.

## Hard negative mining

Monotonic, on HOG + SVM with window proposals:

| Rounds | mAP@0.5 | Train s |
|---|---|---|
| 0 | 0.0146 | 23 |
| 1 | 0.0205 | 102 |
| 2 | 0.0221 | 202 |

+51% relative from 0 to 2 rounds, for about 9x the training time. This is the
Dalal and Triggs recipe behaving as advertised: one pass of random background
crops under-samples the hard cases (sand ripples, wet rock, wave foam, bird
reflections), and feeding the classifier its own confident false positives tightens
the boundary faster per added negative than more random crops would. The gain is
diminishing (+0.0059 then +0.0016), so more rounds were not run for the slower
configurations.

## Negatives sampled from the candidate boxes

Random background crops are not what the classifier meets at inference. There it
sees the proposal stream, so training on crops drawn uniformly from the detector's
own candidates (`proposal_neg_per_image`, default 30) matches the two
distributions. Measured on the best window variant, everything else held fixed:

| Negatives | mAP@0.5 | P | R | Count MAE | Train s |
|---|---|---|---|---|---|
| random crops + mining only | 0.0663 | 0.155 | 0.124 | 5.18 | 290 |
| **+ sampled from the candidates** | **0.0754** | 0.173 | 0.139 | 4.87 | 308 |

**+13.7% relative mAP for +6% training time.** The sampling is uniform over
background candidates, unlike mining, which keeps only the hardest and so biases
the negative set the other way; the two are complementary and both are on by
default.

This also serves as a check on the port: the "random crops only" row (0.0663)
reproduces the pre-port standalone result (0.0661) to within noise, so the whole
improvement is attributable to this change and nothing else drifted when Method 2
moved onto the shared loader and scorer.

## Classifiers and features

**Tree ensembles beat the linear SVM at every descriptor**, at equal mining and
proposals: HOG 0.0470 against 0.0205 (2.3x), colour 0.0148 against 0.0049 (3.0x),
LBP 0.0051 against 0.0017 (3.0x). A single hyperplane, even class-balanced, cannot
separate bird patches from beach clutter in any of these spaces. Inference cost is
essentially unchanged (332 against 299 ms/image for HOG), so there is no trade here.

**HOG carries nearly all the signal.** With the SVM, LBP alone reaches 0.0017 and
colour alone 0.0049, against HOG's 0.0205. Resizing a candidate to 32x32 destroys
most of the fine texture LBP needs, and bird plumage (white, grey, brown) overlaps
heavily with sand, foam and wet rock, so colour separates poorly on this dataset.

**Concatenation only pays off with boosting.** concat + RF (0.0481) is a tie with
HOG + RF (0.0470): the forest gains nothing from the extra 70 dimensions. concat +
GB reaches 0.0754, +60% over either. Boosted trees are the only classifier tried
that exploits interactions *across* the shape, texture and colour blocks. This is
worth more discussion in the report than "the big feature vector won", because the
same big vector does nothing for the other two classifiers.

## Where it fails

Two failure modes, visible in the overlays (`python scripts/run_m2.py --config
configs/m2_concat_gb_m1props.yaml --split val --viz 6`) and consistent with the
recall table above.

1. **Boxes that nearly fit but miss IoU 0.5.** Window recall is 0.500 at IoU 0.5 but
   **0.849 at IoU 0.3**. So for a third of all birds, a candidate lands on the bird
   and is still scored as a miss. The window grid is square and coarsely stepped,
   and birds are not square. This, not finding the birds at all, is the biggest
   single loss for the window variant, and it is the clearest thing to fix: regress
   or refine the box after classification instead of accepting the candidate as-is.
2. **Dense flocks.** M1-segmentation recall falls steadily with density (0.449 for
   1-5 birds, 0.328 for 6-20, 0.209 for 21-50, 0.166 for 51+): adjoining birds merge
   into one candidate, so at most one of them can be reported. The window source is
   flatter across density (0.381 / 0.591 / 0.469 / 0.469) but pays for it everywhere
   else. Note the 51+ bin is only 3 val images, so treat it as indicative.

By size, the window source is strongly biased to mid-sized birds (small 0.354,
medium 0.773, large 0.376) because three fixed scales cannot cover both tails;
the M1 source is flatter (0.234 / 0.345 / 0.332).

LBP and colour with an SVM are not merely weak, they are badly behaved rankers:
their best-F1 threshold sits near the bottom of the score range and the count MAE is
238 and 118 birds per image. They are reported for the feature ablation and should
not be used for anything else.

## A note on score thresholds, for the whole team

`src/evaluation.py` thresholds precision, recall, F1 and counting at a fixed
`SCORE_THR = 0.5`. That is not comparable across methods whose scores mean different
things. Method 2 trains with `class_weight="balanced"`, so its score is a
balanced-prior posterior, and most candidate boxes sit above 0.5: `evaluate()`
reports a count MAE near **290 birds/image** for predictions whose MAE at a
val-tuned threshold is **4.9**. mAP@0.5 is threshold-free and is unaffected.

Method 1 already works around this by recording a per-variant `score_threshold` in
`results/m1_comparison.csv`, and `scripts/compare_m2.py` now does the same. Worth
agreeing one convention before the final table, otherwise the report will carry
precision and recall numbers that are not measured at comparable operating points.

Platt scaling (`calibrate: true` in `src/ml/config.py`) was tried and does not fix
this, because `class_weight="balanced"` deliberately cancels the class imbalance the
calibration would need to see. It is left in, off by default, since it triples
training time for no gain on the required metric.

## Tried and dropped

- **Difference-of-Gaussians proposals.** Worst recall (0.169), worst mAP of any
  variant using the best classifier (0.0144), and by far the slowest to train
  (741 s, because the proposal pass runs over every training image at 302 ms each).
  Kept in the repo as `configs/m2_concat_gb_blobprops.yaml` for the comparison only.
- **Platt scaling.** See above.

## Hand-off

**Member A**
- `results/m2_comparison.csv` is ready for the master table. Use `mAP50` for the
  headline column and `score_threshold` alongside P/R/F1, not a fixed 0.5.
- Please look at the fixed `SCORE_THR` question above before the results table is
  frozen; it affects Method 1's numbers too.
- `pytest.ini` now has `pythonpath = . src` so tests can import the shared modules,
  and `requirements.txt` gained `scikit-learn` and `scikit-image` (plus the trailing
  newline it was missing). Nothing else of yours is touched.

**Member B**
- Your `configs/m1_seg_proposals.yaml` is the single biggest win in Method 2:
  0.1472 against 0.0754 for our own window, at a twelfth of the inference cost.
  Thank you for building it.
- The sliding-window recommendation in your Week 8 note does not hold: the window
  has a higher recall ceiling and still loses, because candidate precision matters
  more than coverage here. Worth a line in the Discussion from both sides.
- If you tune that config further, Method 2's best number moves with it, so please
  flag any change before the Wed 11 Nov freeze.

**Member D**
- Method 2 tops out around 0.15 on val and is capped by its candidate stage, not by
  the classifier. The cleanest contrast for your section: a learned detector
  regresses box coordinates, so it is not bound by a fixed candidate grid at all.
  Our IoU 0.3 against IoU 0.5 gap (0.849 against 0.500) is direct evidence of how
  much is lost to box fit alone.
- Nobody is assigned to Method 3 (issues #10, #16, #17, #18) and there is no
  deep-learning code on any branch yet. This is the group's biggest remaining risk.

**Member E**
- `BirdDetector.image_preproc` is a hook applied to every image before proposals and
  cropping, for both training and inference:
  `det.image_preproc = lambda im: denoise(im)`. It takes and returns RGB uint8.
- For robustness curves, expect the M1-segmentation source to react strongly to
  noise (it inherits B's median-background and Otsu behaviour), while the sliding
  window is unaffected by noise in the candidate stage and will degrade only through
  the descriptors. That contrast is probably the most interesting thing Method 2 can
  offer your study.
- `fraction` in the config takes 0.1 / 0.25 / 0.5 / 1.0 for the data-efficiency
  experiments, with `seed` for repeats.

## Tutor consult

Not yet held. Questions to ask, and record the answers here:

1. Is the hybrid (member B's candidates scored by our classifier) acceptable as
   Method 2, or does it blur the "two meaningfully different methods" requirement?
   Our reading is that it strengthens the comparison, because the candidate stage is
   held fixed and only the accept/reject rule changes, hand-set against learned.
2. Given B tuned those proposals on val, how should we report the hybrid's val
   number without overclaiming?
3. Is a fixed `SCORE_THR` or a per-method tuned threshold the right convention for
   the group's results table?

# Results index

| File/folder | Purpose |
| --- | --- |
| `model_comparison.csv` | Selected YOLO and Faster R-CNN on the official test split; M1 omitted because it has no published test evaluation |
| `ml_comparison.csv` | Test scores for all three imported ML runs, using each run's checkpoint selected on validation |
| `ml_test_comparison.csv` | Final held-out test scores for the selected YOLO and Faster R-CNN checkpoints, with checkpoint/prediction hashes and local CPU timing |
| `m1_comparison.csv` | Original historical comparison of M1 variants, preserved as imported |
| `m1_proposal_recall.csv` | Original M1 region-proposal recall results |
| `timelines/` | Compact per-epoch metrics, losses, learning rates and elapsed time; no repeated configuration blobs |
| `runs/` | One settings/provenance/selection summary per ML run |
| `tuning/yolo_trials.csv` | Imported YOLO search trials, settings, scores and checkpoint-availability flags |
| `tuning/yolo_tune_v1/` | Winning YOLO hyperparameters and final training configuration |
| `m1_tuning_*.csv` | Original M1 search CSVs, retained in the results root |
| `manifest.json` | Selected model locations and archive mapping |
| `evaluation/` | Generated predictions, metadata, optional scores and overlays from `scripts/run_ml.py` |

Model weights belong in `models/`, not this directory. Raw training folders,
per-epoch checkpoints, full histories, intermediate predictions and Windows path
lists were **moved intact** into `archive/training/`; nothing was deleted.
This organization improves navigation but does not reclaim the archive's disk
space. The archive is excluded from Git. It is not needed for inference.

## Selected validation results

All rows below use 291 images / 1,843 birds and the shared evaluator. Precision,
recall, F1 and count MAE use confidence 0.5; AP uses low-confidence predictions.

| Model | Best epoch | mAP50 | mAP50–95 | Precision | Recall | F1 | Count MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO11n tuned | 38 / 50 evaluated | 0.860749 | 0.445970 | 0.896226 | 0.773196 | 0.830178 | 1.219931 |
| Faster R-CNN baseline | 8 / 10 evaluated | 0.839655 | 0.445729 | 0.735704 | 0.830711 | 0.780326 | 1.642612 |
| M1 colour segmentation | — | 0.102450 | 0.029507 | 0.249833 | 0.202930 | 0.223952 | 5.161512 |

The selected ML scores were recomputed from full-precision saved predictions and
checked against the imported records. M1 colour was re-run locally through the
shared evaluator with a 200-detection cap. The historical `m1_comparison.csv`
uses each variant's best-F1 threshold and its earlier evaluator; do not mix its
F1/counting columns into the new shared comparison.

Historical validation timings in the run summaries are from Windows CUDA for
ML and local CPU for M1, so they are not a controlled hardware comparison.
The test comparison CSVs use local CPU timings for every ML row. Local
two-image smoke-test timings are not included.

## Limits of the imported evidence

- Selection is among available checkpoints. A YOLO tuning trial reports mAP50
  0.862825, but none of its trial checkpoints were transferred. The best available
  YOLO model is the final run's epoch 38, mAP50 0.860749.
- The Faster R-CNN baseline folder is named `baseline_20`, but it was interrupted
  after 10 evaluated epochs. It outperformed the tuned run's best mAP50 0.831901.
- All imported training runs record **1,369 training IDs**, with a training
  manifest hash matching the official original split. Their augmentation metadata
  describes 2,738 images. This computer's augmented export also has 2,738 training
  IDs. Do not describe these imported runs as having used all augmented copies;
  the dataset directory's name alone is not evidence of that.
- Original YOLO baseline history and Faster R-CNN tuning-trial history were not
  present in the transfer, so no results have been invented for them.
- The table above contains validation results. Final ML test evaluation was
  completed on 2026-11-18; see `ml_test_comparison.csv` and the details below.

## Final ML test results

Both selected checkpoints were evaluated on all 240 official test images
(1,518 birds) on 2026-11-18, using CPU at image size 640. Models were selected
using validation scores before this test run; no retraining was performed.

| Model | Source epoch | mAP50 | mAP50–95 | Precision | Recall | F1 | Count MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO11n tuned | 38 | 0.832150 | 0.445543 | 0.867339 | 0.736495 | 0.796580 | 1.304167 |
| Faster R-CNN baseline | 8 | 0.825055 | 0.443385 | 0.723630 | 0.808959 | 0.763919 | 1.595833 |

Scores use the shared evaluator on the published `predictions/yolo_test.csv`
and `predictions/faster_rcnn_test.csv` files. Precision/recall/F1 and counting
use confidence 0.5. Reproduce scoring with `python src/evaluation.py`.
`evaluation/` contains each model's raw JSON/NPZ predictions, `.meta.json`
provenance, `.scores.json` scores, and `test_evaluation.txt` evaluator output.
Published CSVs round coordinates to two decimals and confidence to five;
these scores correspond to those CSVs. Raw NPZ files retain full precision.
The generated prediction and evaluation folders are ignored by Git; the
compact comparison CSVs can be versioned. `ml_comparison.csv` also includes the
tuned Faster R-CNN run's epoch 7 checkpoint on the same test split.
`model_comparison.csv` includes only the two models selected on validation;
test performance never changes which checkpoint is selected. M1 was not run
on test, and its existing files and results remain untouched. Its validation
row is omitted from the test-only comparison.

Rebuild both tables from saved test evaluations without running inference:
`python scripts/compare_test_results.py`. This checks the split, evaluated
image count, checkpoint epoch, and prediction/checkpoint/manifest hashes.
Validation history remains in `timelines/` and `runs/`.

`python scripts/organize_results.py` rebuilds compact training reports from the
raw archive and refreshes the test tables only when all test evaluations are
available. It never runs test inference or changes M1 artifacts. It refuses
conflicting run names, mismatched model hashes and replacement of an
already-selected different exported model.

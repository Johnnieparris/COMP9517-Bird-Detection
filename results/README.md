# Results index

| File/folder | Purpose |
| --- | --- |
| `model_comparison.csv` | Selected YOLO, Faster R-CNN and M1 colour detector on the same official validation split and shared evaluator |
| `ml_comparison.csv` | All three imported ML training runs, with selected-model flags, actual evaluated epoch counts and best scores |
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

Timing columns identify their device and platform: ML timings are from the
Windows CUDA runs, while the refreshed M1 timing is local CPU. They are not a
controlled hardware speed comparison. Run all methods on the same machine for
that comparison. Local two-image model smoke-test timings are not included.

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
- These are validation results. The test split remains reserved for final
  evaluation. Use the instructions in `models/README.md` when ready.

`python scripts/organize_results.py` rebuilds compact tables from the raw archive
and refreshes M1 validation. It refuses conflicting run names, mismatched model
hashes and replacement of an already-selected different exported model.

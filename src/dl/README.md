# Data augmentation for YOLO and Faster R-CNN

## Using the selected trained models

The imported trained models are packaged in `models/yolo/best.pt` and
`models/faster_rcnn/best.pt`. See `models/README.md` for selection details and
`results/README.md` for the compact comparison tables and epoch timelines.

```bash
python scripts/run_ml.py --config configs/yolo_best.yaml --split val --score --publish
python scripts/run_ml.py --config configs/faster_rcnn_best.yaml --split val --score --publish
```

This follows M1's runner interface, including `--data-root`, `--viz`, `--limit`,
and the `--final` guard for test data. Add `--device 0` for CUDA or `--device cpu`.
These inference commands do not train or require the original Windows paths.
Raw imported runs now live in `archive/training/`. The training instructions below
apply to new experiments, which continue writing new run folders under `results/`.

## Preparing augmentation

From the repository root, activate `birds` and generate three additional copies
of every training image:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate birds
python -m src.dl.augment --copies 3 --seed 42
```

The default output is `data/augmented_birds/`, which is ignored by Git. With the
official dataset this produces **5,476 training images**, 291 validation images,
and 240 test images. Original images and labels are copied unchanged. Only images
in the official training split are augmented. Existing output folders are refused;
use a different `--output` for another experiment.

Each added copy combines:

| Operation | Default | CLI setting |
| --- | --- | --- |
| Gaussian noise | Standard deviation 8 on 0–255 pixels | `--noise-std` |
| Horizontal flip | 50% probability | `--horizontal-flip` |
| Vertical flip | Disabled; optional probability | `--vertical-flip` |
| Brightness, contrast, saturation | Each varies by up to 20% | `--brightness`, `--contrast`, `--saturation` |
| Hue | Up to ±10 degrees | `--hue` |
| Scale | 0.8–1.2 times, about the centre | `--scale-min`, `--scale-max` |
| Rotation | Up to ±15 degrees | `--rotation` |

Set noise/colour/rotation amounts or flip probabilities to zero to disable them;
set both scale bounds to 1 to disable scaling. Run `python -m src.dl.augment --help`
for all options. For example:

```bash
python -m src.dl.augment --output data/augmented_birds_strong --copies 5 \
  --noise-std 12 --rotation 25 --scale-min 0.7 --scale-max 1.3 \
  --brightness 0.3 --saturation 0.3 --seed 42
```

Scaling and rotation preserve the image dimensions, using grey padding. All four
box corners are transformed; polygons are clipped to the image and enclosed in
axis-aligned boxes. Labels with less than 25% visibility or width/height below two
pixels are dropped (`--min-visibility`, `--min-box-size`). Empty labels are supported.
Review these settings for very small birds: aggressive cropping can leave visible
fragments without labels, and rotated axis-aligned boxes include extra background.

The output contains `images/` and `labels/` for all three splits, `data.yaml` for
YOLO, `splits.json` for the shared loader, and `augmentation.json` with settings,
counts, filtered-box count, and each added image's original ID. The original
`splits/official.json` is never edited. Reusing the seed and settings with the same
inputs and library versions reproduces the augmented data.

## YOLO

For the tracked baseline training pipeline, run from the repository root:

```bash
python -m src.dl.train_yolo --config configs/yolo_baseline.yaml
# Another independent experiment, retaining all previous results:
python -m src.dl.train_yolo --config configs/yolo_baseline.yaml --epochs 20 --name yolo_20ep
```

The baseline fine-tunes COCO-pretrained YOLO11n for five epochs with AdamW, image
size 640, batch size 8, and nominal batch size 16 (gradient accumulation). It
automatically selects CUDA, Apple MPS, or CPU; `--device` overrides that choice.
The dataset `data/augmented_birds_baseline` is generated if missing, using one
additional augmented copy per official training image. The augmentation metadata
is preserved in the run. Existing data must match the requested copy count and
augmentation seed; use another `data_root` when changing those settings.

Every epoch saves and evaluates a checkpoint against **all official validation
images** using `src/evaluation.py`. No test evaluation is performed. The saved
scores use the same metric definitions as the classical and other methods.
Compare validation scores only with other validation scores; reserve the official
test split for the final comparison of selected models.

Each uniquely named `results/<run>/` directory contains:

| Artifact | Contents |
| --- | --- |
| `run.json` | Run status, requested settings, dataset/augmentation provenance, package versions, code hashes, device, and best epoch |
| `resolved_hyperparameters.json` | Full resolved Ultralytics configuration, including defaults |
| `history.csv`, `history.json` | One row/event per epoch with shared scores, timing, hyperparameters, current learning rates, and training losses |
| `checkpoints/epoch_0001.pt`, etc. | Immutable epoch checkpoints copied before Ultralytics strips optimizer state at completion |
| `checkpoints/best_shared.pt`, `best_shared.json` | Checkpoint and scores selected by highest shared validation mAP@0.5 |
| `milestones/epoch_0001/scores.json`, etc. | Complete per-epoch evaluation record, checkpoint hash, hyperparameters, and optimizer groups |
| `milestones/epoch_0001/predictions.npz` | Full-precision predictions for reproducing scores |
| `milestones/epoch_0001/yolo_val.csv` | Predictions in the team's standard CSV format |
| `trainer/` | Ultralytics logs and its own `best.pt`/`last.pt` checkpoints |

The shared-best model's predictions are also published as
`predictions/yolo_<run>_val.csv`. Other methods' files are untouched. Ultralytics'
own best checkpoint uses its internal fitness and may differ from `best_shared.pt`.
Both sets of metrics are retained and clearly separated in milestone JSON files.

Inference retains scores down to 0.001 and uses the shared prediction module's
maximum-detection limit (currently 200). Shared precision/recall/F1 and counting
use the team's fixed 0.5 score threshold; AP uses the low-confidence predictions.
Timing uses one image at a time, three warmups, and excludes image loading. Device,
versions, image size and timing settings are recorded so future comparisons can
be made on the same hardware. CSV coordinates/scores are rounded by the shared
writer; use the NPZ file when exactly reproducing the stored metrics.

The history uses **one-based epochs**. Checkpoints are saved before shared
evaluation, so a validation failure still leaves the epoch's weights. A failed or
interrupted run is marked in `run.json`; completed milestone files remain intact.
The CLI starts independent experiments and refuses reused run names; it does not
currently resume an interrupted run. For future grid searches, supply a config and
unique name per combination and use the recorded shared validation mAP@0.5 to rank.

For a standalone Ultralytics training call without the milestone recorder:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # downloads pretrained weights on first use
model.train(data="data/augmented_birds/data.yaml", epochs=50, imgsz=640,
            seed=42, hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
            scale=0, shear=0, perspective=0, flipud=0, fliplr=0,
            mosaic=0, mixup=0, cutmix=0, copy_paste=0, bgr=0)
```

This example disables YOLO's usual additional detection augmentations so the
comparison uses the exported data. Keep the offline dataset and seed the same for
both methods; document model-specific resizing and other training differences.

## Faster R-CNN

The training pipeline uses COCO-pretrained torchvision Faster R-CNN with a
ResNet-50 FPN backbone, replacing the detection head with background + bird.
It uses **the same `data/augmented_birds_baseline` export as YOLO** (one added
copy, seed 42). It does not generate or apply any additional augmentation.
Copy that existing export onto Windows before running. Both training and
validation paths are explicit in `configs/faster_rcnn_baseline.yaml`; the
validation images in the export are unchanged copies of the official split.
The existing YOLO leakage checks and official manifest select the images.

Run from the repository root in the same Python environment as YOLO. Compatible
PyTorch and torchvision must already be installed; this pipeline does not depend
on Ultralytics' tuner or require replacing your working CUDA installation.
The first pretrained run downloads the official torchvision COCO weights into
PyTorch's cache.

```powershell
python -m src.dl.train_faster_rcnn --device 0 --name faster_rcnn_baseline_50
```

Defaults: 50 epochs, SGD, learning rate 0.005, momentum 0.9, weight decay 0.0005,
one epoch of linear warmup followed by cosine decay, batch 1 with two-batch
gradient accumulation, two data-loader workers, and AMP disabled. The model
preserves aspect ratio and fits images within 640 x 640, then applies its standard
normalization and padding. This differs from YOLO's internal preprocessing;
training image IDs and offline augmentation remain identical. Use `--device cpu`
for CPU, or `--epochs 1` for a short initial check. `auto` chooses CUDA when
available, otherwise CPU; an explicit unavailable CUDA device raises an error.

Every epoch retains model, optimizer, scaler, and random-generator states, then
runs the **same shared evaluator** used by YOLO. Predictions use confidence 0.001,
NMS IoU 0.7, and at most 200 detections per image. F1/counting use the shared fixed
0.5 threshold; timing is batch 1 with three warmups, including transfer back to CPU
but excluding image loading. Test data is never evaluated during training/tuning.

Artifacts under `results/<name>/`:

- `run.json` and `resolved_hyperparameters.json`: settings, dataset provenance,
  code hashes, versions, device, completion status, and best epoch.
- `history.csv` / `history.json`: scores, losses, learning rates, and hyperparameters
  for each epoch, in the same timeline format as YOLO.
- `checkpoints/epoch_0001.pt`, etc.: resumable checkpoints saved before evaluation.
- `checkpoints/best_shared.pt` / `best_shared.json`: highest shared validation mAP50.
- `milestones/epoch_0001/`: scores, full-precision predictions NPZ, and team-format CSV.
- `best_shared_predictions.csv`: the selected checkpoint's predictions; also
  published to `predictions/faster_rcnn_<name>_val.csv` when a new best is found.

All epoch checkpoints are retained, so allow substantial disk space for a
ResNet-50 run and tuning trials. `patience: 0` disables early stopping. A positive
value stops after that many evaluated epochs without a new best shared mAP50.

Resume a failed/interrupted run to its **original** epoch budget:

```powershell
python -m src.dl.train_faster_rcnn --resume results/faster_rcnn_baseline_50 --device 0
```

Resume also recovers an epoch whose checkpoint was saved but evaluation did not
finish. Only load trusted checkpoints generated by this pipeline. Keep the dataset
paths and manifests unchanged. To change hyperparameters or start another budget,
use a new run name and config rather than resuming.

### Hyperparameter grid search

```powershell
python -m src.dl.tune_faster_rcnn --device 0 --name faster_rcnn_tune
```

`configs/faster_rcnn_tune.yaml` defines an exhaustive grid of **six trials, ten
epochs each**: learning rates 0.001/0.003/0.005 and weight decays 0.0001/0.0005.
Every trial starts from the same COCO weights and seed, reuses the same augmented
set, and saves its complete checkpoint/evaluation timeline. Early stopping is
disabled for trials. Change `trial_epochs` or grid lists to change the budget;
allowed search keys are lr0, momentum, weight_decay, lrf, and warmup_epochs.
Short trials are a screening step, not a guarantee of the best 50-epoch result.

The tuner ranks trials by their best shared validation mAP50 and stops after all
grid combinations. It writes `trials.csv`, `trials.json`, `best_shared_trial.json`,
`best_shared_hyperparameters.yaml`, and **`best_training_config.yaml`**. The final
config restores the baseline's full epoch budget and starting weights; it does
not continue from the trial's checkpoint. Completion is marked in `tuning.json`
and by the final configuration message (the best config may exist earlier).

Resume tuning using the same command/config/grid plus `--resume`:

```powershell
python -m src.dl.tune_faster_rcnn --device 0 --name faster_rcnn_tune --resume
```

Then train a fresh model with the selected settings:

```powershell
python -m src.dl.train_faster_rcnn --config results/faster_rcnn_tune/best_training_config.yaml --epochs 50 --device 0 --name faster_rcnn_tuned_50
```

Compare best validation checkpoints against YOLO and classical methods using the
shared evaluator. Use the untouched test set only after model/config selection.

The `BirdDetectionDataset` adapter returns RGB float tensors in [0, 1] and pixel
XYXY boxes. Exported YOLO class 0 becomes torchvision bird class 1; class 0 is
background. Empty targets and varying image sizes/counts are supported.

# Data augmentation for YOLO and Faster R-CNN

From the repository root, activate `birds` and generate three additional copies
of every training image:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate birds
python -m src.ml.augment --copies 3 --seed 42
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
set both scale bounds to 1 to disable scaling. Run `python -m src.ml.augment --help`
for all options. For example:

```bash
python -m src.ml.augment --output data/augmented_birds_strong --copies 5 \
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
python -m src.ml.train_yolo --config configs/yolo_baseline.yaml
# Another independent experiment, retaining all previous results:
python -m src.ml.train_yolo --config configs/yolo_baseline.yaml --epochs 20 --name yolo_20ep
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

```python
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.ml.dataset import BirdDetectionDataset, detection_collate

train = BirdDetectionDataset("data/augmented_birds", split="train")
loader = DataLoader(train, batch_size=2, shuffle=True,
                    collate_fn=detection_collate)
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes=2)

images, targets = next(iter(loader))
model.train()
losses = model(images, targets)
loss = sum(losses.values())  # then backward() and your optimizer step
```

The adapter uses the shared `src/data/dataset.py` loader. It returns RGB float
tensors in `[0, 1]` and targets with pixel-coordinate `boxes`, `labels`, `image_id`,
`area`, and `iscrowd`. YOLO's bird class **0** maps to Faster R-CNN class **1**;
torchvision reserves class 0 for background. The collate function supports varying
image sizes and bird counts, including no birds. No extra augmentation is applied.

Use `split="val"` during development. Reserve `split="test"` for final evaluation.
For both methods, convert predictions to `x1, y1, x2, y2, score` and use the shared
evaluation and prediction-saving modules described in the main README.

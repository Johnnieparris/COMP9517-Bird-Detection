# Selected models

| Method | Checkpoint | Source | Epoch | Shared validation mAP50 |
| --- | --- | --- | --- | --- |
| YOLO11n | `yolo/best.pt` | `yolo11n_tuned_50` | 38 | 0.860749 |
| Faster R-CNN ResNet-50 FPN | `faster_rcnn/best.pt` | `faster_rcnn_baseline_20` | 8 | 0.839655 |

Selection uses the highest shared validation mAP50 among **available imported
checkpoints**. The Faster R-CNN baseline beats the tuned run (0.831901), so the
baseline is selected. The baseline folder's name says 20, but only 10 evaluated
epochs were imported. The YOLO tuner reports 0.862825 for a trial whose weights
were not transferred; that unavailable checkpoint is not the exported model.

These are inference exports, approximately 5.5 MB (YOLO) and 165.7 MB (Faster
R-CNN), with optimizer state removed. They are not training-resume checkpoints.
Each `metadata.json` records the original checkpoint hash, exported hash,
validation scores, epoch, settings and provenance. Original checkpoints remain
under `archive/training/`.

Weights are ignored by Git. Transfer this entire `models/` directory separately
when sharing the project; metadata and inference configs can be committed.

From the project root, use the same style of command as `scripts/run_m1.py`:

```bash
python scripts/run_ml.py --config configs/yolo_best.yaml --split val --score --publish
python scripts/run_ml.py --config configs/faster_rcnn_best.yaml --split val --score --publish
```

`--device 0` selects CUDA; `--device cpu` forces CPU. The default automatically
selects CUDA when available, otherwise CPU (including on this Mac). Add `--viz 5`
to save overlays. For a quick check, use `--limit 5 --score` without `--publish`;
partial runs cannot publish an incomplete split into the shared evaluator.

For final test predictions, use `--split test --final --publish` (without
`--score` or `--limit`), then run `python src/evaluation.py` to score all published
test predictions together. This follows M1's test-set guard. The test split was
not evaluated during organization or smoke testing.

The configs `configs/yolo_best.yaml` and `configs/faster_rcnn_best.yaml` point to
the portable weights using repository-relative paths. Loading them never needs
the old Windows paths or the training archive. The data root defaults to the
original dataset; override it with `--data-root data/augmented_birds_baseline`
if only the exported dataset is available. Official split IDs still apply.

API usage for robustness experiments:

```python
from src.dl.inference import load_detector

predict, metadata = load_detector("configs/yolo_best.yaml", device="cpu")
boxes = predict(image_rgb)  # uint8 RGB in; N x 5 pixel XYXY + score out
```

See `results/README.md` for comparison caveats, particularly the imported
training-manifest/augmentation discrepancy.

# Member E Study B training runs

Produced by `bash scripts/train_e_runs.sh` from the configs in `configs/e/` (built by
`scripts/make_e_training_roots.py`). Run folders `e_<run>/` are local only (ignored by Git);
only this README and the CSVs derived from them are committed.

- Full-data epoch budget **E = 30**. Subset runs: `min(round(E / f), 100)` epochs
  (de_50 = 60, de_25 = 100, de_10 = de_10_strong = 100).
- Timing check (plan Task 4): de_100 epoch 1 took 54 s including per-epoch shared validation
  on Apple M5 Pro (MPS), while CPU-only evaluation jobs ran alongside. Projected total for
  all 8 runs ≈ 4.6 h, below the 5 h limit, so E was not reduced.
- Checkpoint per run: `checkpoints/best_shared.pt`, selected by shared-evaluator mAP50 on clean val.

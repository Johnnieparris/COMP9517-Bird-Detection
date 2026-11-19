#!/usr/bin/env bash
# Train Member E's Study B YOLO runs one after another (spec order). Re-running skips
# completed runs and continues past a failed one. Log: results/e_runs/train_e_runs.log
#   bash scripts/train_e_runs.sh
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
OUT=results/e_runs
mkdir -p "$OUT"
LOG="$OUT/train_e_runs.log"
for run in de_100 de_10 de_25 de_50 robust_aug ln_jitter20 ln_drop20 de_10_strong; do
  dir="$OUT/e_$run"
  if [ -f "$dir/run.json" ] && grep -q '"status": "completed"' "$dir/run.json"; then
    echo "$(date '+%F %T') skip $run (completed)" >> "$LOG"; continue
  fi
  if [ -d "$dir" ]; then
    echo "$(date '+%F %T') $run has an unfinished run folder; resuming" >> "$LOG"
    "$PY" -m src.dl.train_yolo --resume "$dir" >> "$OUT/e_$run.out" 2>&1
  else
    echo "$(date '+%F %T') start $run" >> "$LOG"
    "$PY" -m src.dl.train_yolo --config "configs/e/$run.yaml" --name "e_$run" --output-root "$OUT" \
      > "$OUT/e_$run.out" 2>&1
  fi
  echo "$(date '+%F %T') end $run (exit $?)" >> "$LOG"
done
echo "$(date '+%F %T') all runs finished" >> "$LOG"

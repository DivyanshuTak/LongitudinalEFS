#!/bin/bash
# Train every CV fold, then print the aggregate report.
# Usage: scripts/run_cv.sh [config.yml] [n_folds]
set -e
CONFIG=${1:-config.yml}
FOLDS=${2:-5}
mkdir -p logs
for k in $(seq 0 $((FOLDS-1))); do
  echo "=== fold $k ==="
  python src/train.py --config "$CONFIG" --fold "$k" 2>&1 | tee "logs/fold${k}.log"
done
python scripts/summarize_cv.py --config "$CONFIG" --folds "$FOLDS"

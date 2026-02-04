#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/tmp

# Write small mock datasets into outputs/ so we don't modify the tracked
# data/train_structeval_t.json / data/valid_structeval_t.json fixtures.
python -m src.data.make_mock_structeval_t \
  --out-train outputs/tmp/train_structeval_t.smoke.json \
  --out-valid outputs/tmp/valid_structeval_t.smoke.json \
  --n-train 50 --n-valid 10

python train.py score --input outputs/tmp/valid_structeval_t.smoke.json --output outputs/smoke_scored.json
echo "Smoke test done. See outputs/smoke_scored.json and outputs/smoke_scored.summary.json"

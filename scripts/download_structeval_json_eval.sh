#!/usr/bin/env bash
set -euo pipefail

# Download StructEval (HF) and export JSON-output, non-rendering tasks into a local eval JSON.
# Requires: pip install datasets

SPLIT="${1:-test}"
OUT="${2:-data/structeval_json_eval.json}"
# Keep backward-compatible default: JSON only.
OUTPUT_TYPES="${3:-JSON}"

python -m src.data.import_structeval \
  --split "$SPLIT" \
  --out "$OUT" \
  --output-types "$OUTPUT_TYPES"

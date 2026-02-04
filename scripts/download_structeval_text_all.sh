#!/usr/bin/env bash
set -euo pipefail

# Download StructEval (HF) and export TEXT, non-rendering tasks across multiple output formats.
# Output: JSON array (for traceability) containing tasks with output_type in JSON/YAML/TOML/XML/CSV.
# Requires: pip install datasets

SPLIT="${1:-test}"
OUT="${2:-data/structeval_text_all.json}"

# Default: export all supported formats.
OUTPUT_TYPES="${3:-JSON,YAML,TOML,XML,CSV}"

python -m src.data.import_structeval \
  --split "$SPLIT" \
  --out "$OUT" \
  --output-types "$OUTPUT_TYPES"

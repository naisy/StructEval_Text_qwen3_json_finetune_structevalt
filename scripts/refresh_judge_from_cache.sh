#!/usr/bin/env bash
set -euo pipefail

IN_PATH="${1:-outputs/eval/structeval_t_eval.json}"
OUT_PATH="${2:-outputs/eval/structeval_t_eval.json}"
PROVIDER="${3:-openai}"
MODEL="${4:-gpt-5.2}"
CACHE_DIR="${5:-outputs/cache/judge}"

python -m src.judge.refresh_from_cache --in "$IN_PATH" --out "$OUT_PATH" --provider "$PROVIDER" --model "$MODEL" --cache_dir "$CACHE_DIR"

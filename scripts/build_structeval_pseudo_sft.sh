#!/usr/bin/env bash
set -euo pipefail

# Build pseudo-SFT JSONL by generating structured outputs with a teacher model for StructEval tasks.
#
# Usage:
#   bash scripts/build_structeval_pseudo_sft.sh <input.json> <out_path> <teacher_model> <limit> <seed> [backend]
#
# - <out_path> can be either a JSONL file (single output), or a directory when splitting by output_type.
# - Defaults for variants/output_types/splitting are read from configs/dataset_sft.yaml:pseudo_sft.
# - backend: hf (default) | ollama
#
# For ollama backend, set OLLAMA_HOST (default http://127.0.0.1:11434) and optionally OLLAMA_NUM_CTX (default 16384).

INPUT="${1:-data/structeval_json_eval.json}"
# Default to a directory because configs/dataset_sft.yaml enables split_by_output_type.
OUT="${2:-data/pseudo_sft}"
TEACHER="${3:-Qwen/Qwen3-4B-Instruct-2507}"
LIMIT="${4:-0}"
SEED="${5:-42}"
BACKEND="${6:-hf}"

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OLLAMA_TIMEOUT_S="${OLLAMA_TIMEOUT_S:-600}"
OLLAMA_RETRIES="${OLLAMA_RETRIES:-2}"
# NOTE: Ollama format=json is applied only to JSON-target tasks; non-JSON targets omit the format field.
OLLAMA_FORMAT="${OLLAMA_FORMAT:-json}"

COMMON_ARGS=(
  --dataset-sft-config configs/dataset_sft.yaml
  --input "${INPUT}"
  --output "${OUT}"
  --teacher-model "${TEACHER}"
  --limit "${LIMIT}"
  --seed "${SEED}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ "${BACKEND}" == "ollama" ]]; then
  python -m src.data.build_pseudo_sft_from_structeval \
    --teacher-backend ollama \
    --ollama-host "${OLLAMA_HOST}" \
    --ollama-num-ctx "${OLLAMA_NUM_CTX}" \
    --ollama-timeout-s "${OLLAMA_TIMEOUT_S}" \
    --ollama-retries "${OLLAMA_RETRIES}" \
    --ollama-format "${OLLAMA_FORMAT}" \
    "${COMMON_ARGS[@]}"
else
  python -m src.data.build_pseudo_sft_from_structeval \
    --teacher-backend hf \
    "${COMMON_ARGS[@]}"
fi

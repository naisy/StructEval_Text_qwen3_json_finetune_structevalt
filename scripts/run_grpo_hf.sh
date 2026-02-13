#!/usr/bin/env bash
set -euo pipefail

# Convert allowed HF datasets into StructEval-T style tasks (requires ATTRIBUTES blocks).
# You can override DATASETS (space-separated) and SPLIT.
DATASETS="${HF_GRPO_DATASETS:-daichira/structured-3k-mix-sft daichira/structured-5k-mix-sft daichira/structured-hard-sft-4k}"
SPLIT="${HF_GRPO_SPLIT:-train}"

PYTHONPATH="$(pwd)" python -m src.data.import_hf_structured_sft \
  --datasets ${DATASETS} \
  --split "${SPLIT}" \
  --out-grpo-tasks data/hf_grpo_tasks.json \
  --write-grpo-tasks \
  --filter-invalid \
  --shuffle-seed 42

# Optional: balance by task_key to reduce output-format bias.
# Default: enabled (set HF_BALANCE_BY_TASK=0 to disable)
BALANCE_BY_TASK="${HF_BALANCE_BY_TASK:-1}"
BALANCE_STRATEGY="${HF_BALANCE_STRATEGY:-min}"   # min|max|fixed
BALANCE_FIXED_N="${HF_BALANCE_FIXED_N:-}"
BALANCE_SEED="${HF_BALANCE_SEED:-42}"
BALANCE_MIN_COUNT="${HF_BALANCE_MIN_COUNT:-8}"

GRPO_INPUT_JSON="data/hf_grpo_tasks.json"
if [ "${BALANCE_BY_TASK}" != "0" ]; then
  echo "INFO  Balancing GRPO tasks by task_key (strategy=${BALANCE_STRATEGY}) ..."
  # Step estimate context comes from configs/grpo_hf.yaml.
  PER_DEVICE_BS="${GRPO_PER_DEVICE_TRAIN_BS:-2}"
  GRAD_ACCUM="${GRPO_GRAD_ACCUM:-8}"
  MAX_STEPS="${GRPO_MAX_STEPS:-100}"
  FIXED_ARG=()
  if [ "${BALANCE_STRATEGY}" = "fixed" ] && [ -n "${BALANCE_FIXED_N}" ]; then
    FIXED_ARG=(--fixed-n "${BALANCE_FIXED_N}")
  fi
  PYTHONPATH="$(pwd)" python -m src.data.balance_by_task \
    --input "${GRPO_INPUT_JSON}" \
    --input-format json \
    --output data/hf_grpo_tasks_balanced.json \
    --strategy "${BALANCE_STRATEGY}" \
    --seed "${BALANCE_SEED}" \
    --min-count "${BALANCE_MIN_COUNT}" \
    --per-device-train-batch-size "${PER_DEVICE_BS}" \
    --grad-accum "${GRAD_ACCUM}" \
    --max-steps "${MAX_STEPS}" \
    "${FIXED_ARG[@]}"
  GRPO_INPUT_JSON="data/hf_grpo_tasks_balanced.json"
else
  echo "INFO  HF_BALANCE_BY_TASK=0 -> using unbalanced GRPO tasks."
fi

PYTHONPATH="$(pwd)" python -m src.data.prepare_structeval_split \
  --in-json "${GRPO_INPUT_JSON}" \
  --out-train data/train_hf_grpo_tasks.json \
  --out-valid data/valid_hf_grpo_tasks.json \
  --seed 42 \
  --valid-ratio 0.1

# Ensure StructEval-T multi-format eval tasks exist for post-training evaluation.
if [ ! -f data/structeval_text_all.json ]; then
  echo "INFO  Downloading StructEval-T eval tasks (JSON/YAML/TOML/XML/CSV) to data/structeval_text_all.json ..."
  bash scripts/download_structeval_text_all.sh test data/structeval_text_all.json "JSON,YAML,TOML,XML,CSV"
fi

PYTHONPATH="$(pwd)" python train.py grpo --config configs/grpo_hf.yaml

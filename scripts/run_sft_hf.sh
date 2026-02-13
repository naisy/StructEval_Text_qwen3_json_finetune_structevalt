#!/usr/bin/env bash
set -euo pipefail

# Convert allowed HF datasets to SFT JSONL and split into train/valid.
# You can override DATASETS (space-separated) and SPLIT.
DATASETS="${HF_SFT_DATASETS:-u-10bei/structured_data_with_cot_dataset_512_v5 daichira/structured-3k-mix-sft daichira/structured-5k-mix-sft daichira/structured-hard-sft-4k}"
SPLIT="${HF_SFT_SPLIT:-train}"

PYTHONPATH="$(pwd)" python -m src.data.import_hf_structured_sft \
  --datasets ${DATASETS} \
  --split "${SPLIT}" \
  --out-sft-jsonl data/hf_sft.jsonl \
  --filter-invalid \
  --shuffle-seed 42

# Optional: balance by task_key to reduce output-format bias.
# Default: enabled (set HF_BALANCE_BY_TASK=0 to disable)
BALANCE_BY_TASK="${HF_BALANCE_BY_TASK:-1}"
BALANCE_STRATEGY="${HF_BALANCE_STRATEGY:-min}"   # min|max|fixed
BALANCE_FIXED_N="${HF_BALANCE_FIXED_N:-}"
BALANCE_SEED="${HF_BALANCE_SEED:-42}"
BALANCE_MIN_COUNT="${HF_BALANCE_MIN_COUNT:-8}"

SFT_INPUT_JSONL="data/hf_sft.jsonl"
if [ "${BALANCE_BY_TASK}" != "0" ]; then
  echo "INFO  Balancing SFT dataset by task_key (strategy=${BALANCE_STRATEGY}) ..."
  # Step estimate context comes from configs/sft_hf.yaml (we keep this script standalone;
  # pass the key hyperparams via env vars if you want different settings).
  PER_DEVICE_BS="${SFT_PER_DEVICE_TRAIN_BS:-2}"
  GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
  EPOCHS="${SFT_EPOCHS:-2}"
  FIXED_ARG=()
  if [ "${BALANCE_STRATEGY}" = "fixed" ] && [ -n "${BALANCE_FIXED_N}" ]; then
    FIXED_ARG=(--fixed-n "${BALANCE_FIXED_N}")
  fi
  PYTHONPATH="$(pwd)" python -m src.data.balance_by_task \
    --input "${SFT_INPUT_JSONL}" \
    --input-format jsonl \
    --output data/hf_sft_balanced.jsonl \
    --strategy "${BALANCE_STRATEGY}" \
    --seed "${BALANCE_SEED}" \
    --min-count "${BALANCE_MIN_COUNT}" \
    --per-device-train-batch-size "${PER_DEVICE_BS}" \
    --grad-accum "${GRAD_ACCUM}" \
    --epochs "${EPOCHS}" \
    "${FIXED_ARG[@]}"
  SFT_INPUT_JSONL="data/hf_sft_balanced.jsonl"
else
  echo "INFO  HF_BALANCE_BY_TASK=0 -> using unbalanced SFT dataset."
fi

PYTHONPATH="$(pwd)" python -m src.data.prepare_sft_split \
  --input "${SFT_INPUT_JSONL}" \
  --train-out data/train_hf_sft.jsonl \
  --valid-out data/valid_hf_sft.jsonl \
  --seed 42

# Ensure StructEval-T multi-format eval tasks exist for post-training evaluation.
# (If you prefer offline / no-download runs, you can provide EVAL_TASKS_PATH or
# rely on the built-in mock fallback.)
if [ ! -f data/structeval_text_all.json ]; then
  echo "INFO  Downloading StructEval-T eval tasks (JSON/YAML/TOML/XML/CSV) to data/structeval_text_all.json ..."
  bash scripts/download_structeval_text_all.sh test data/structeval_text_all.json "JSON,YAML,TOML,XML,CSV"
fi

PYTHONPATH="$(pwd)" python train.py sft --config configs/sft_hf.yaml

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

# Subset selection (balance by task_key OR per-output-type targets)
# Config: configs/grpo_hf.yaml -> data.sampling
PER_DEVICE_BS="${GRPO_PER_DEVICE_TRAIN_BS:-1}"
GRAD_ACCUM="${GRPO_GRAD_ACCUM:-8}"
MAX_STEPS="${GRPO_MAX_STEPS:-100}"
GRPO_INPUT_JSON="data/hf_grpo_tasks.json"
PYTHONPATH="$(pwd)" python -m src.data.hf_select_subset \
  --stage grpo \
  --config configs/grpo_hf.yaml \
  --input "${GRPO_INPUT_JSON}" \
  --input-format json \
  --output data/hf_grpo_tasks_selected.json \
  --output-format json \
  --per-device-train-batch-size "${PER_DEVICE_BS}" \
  --grad-accum "${GRAD_ACCUM}" \
  --max-steps "${MAX_STEPS}"

GRPO_INPUT_JSON="data/hf_grpo_tasks_selected.json"

PYTHONPATH="$(pwd)" python -m src.data.prepare_structeval_split \
  --in-json "${GRPO_INPUT_JSON}" \
  --out-train data/train_hf_grpo_tasks.json \
  --out-valid data/valid_hf_grpo_tasks.json \
  --seed 42 \
  --valid-ratio 0.1

# --------------------------------------------------------------
# Optional: append user-provided local datasets AFTER HF balancing
#
# Configure in configs/grpo_hf.yaml:
#   data:
#     extra_datasets:
#       - use: true
#         format: structeval_json
#         train_path: data/my_x_train.json
#         valid_path: data/my_x_valid.json
# --------------------------------------------------------------
PYTHONPATH="$(pwd)" python -m src.data.append_extra_datasets \
  --stage grpo \
  --config configs/grpo_hf.yaml \
  --train data/train_hf_grpo_tasks.json \
  --valid data/valid_hf_grpo_tasks.json

# Ensure StructEval-T multi-format eval tasks exist for post-training evaluation.
if [ ! -f data/structeval_text_all.json ]; then
  echo "INFO  Downloading StructEval-T eval tasks (JSON/YAML/TOML/XML/CSV) to data/structeval_text_all.json ..."
  bash scripts/download_structeval_text_all.sh test data/structeval_text_all.json "JSON,YAML,TOML,XML,CSV"
fi

PYTHONPATH="$(pwd)" python train.py grpo --config configs/grpo_hf.yaml

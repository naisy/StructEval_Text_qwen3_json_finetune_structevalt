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
  --shuffle-seed 42

PYTHONPATH="$(pwd)" python -m src.data.prepare_structeval_split \
  --in-json data/hf_grpo_tasks.json \
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

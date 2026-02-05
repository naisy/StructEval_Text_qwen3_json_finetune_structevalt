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

PYTHONPATH="$(pwd)" python -m src.data.prepare_sft_split \
  --input data/hf_sft.jsonl \
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

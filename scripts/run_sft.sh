#!/usr/bin/env bash
set -euo pipefail

# If a pseudo-SFT file exists, automatically split it into the train/valid files
# expected by configs/dataset_sft.yaml.
SFT_SOURCE="${SFT_INPUT_JSONL:-}"
if [[ -z "${SFT_SOURCE}" ]]; then
  if [[ -f "data/structeval_pseudo_sft.jsonl" ]]; then
    SFT_SOURCE="data/structeval_pseudo_sft.jsonl"
  elif [[ -f "data/pseudo_sft" ]]; then
    # Common default used in examples (no extension)
    SFT_SOURCE="data/pseudo_sft"
  elif [[ -f "data/pseudo_sft.jsonl" ]]; then
    SFT_SOURCE="data/pseudo_sft.jsonl"
  elif [[ -d "data/pseudo_sft" ]]; then
    # If pseudo-SFT was generated with --split-by-output-type, merge files for splitting.
    MERGED="data/_merged_pseudo_sft.jsonl"
    shopt -s nullglob
    files=(data/pseudo_sft/*.jsonl)
    if (( ${#files[@]} > 0 )); then
      cat "${files[@]}" > "${MERGED}"
      SFT_SOURCE="${MERGED}"
    fi
    shopt -u nullglob
  elif [[ -f "data/_debug_pseudo_sft.jsonl" ]]; then
    SFT_SOURCE="data/_debug_pseudo_sft.jsonl"
  fi
fi

if [[ -n "${SFT_SOURCE}" ]]; then
  echo "== Preparing SFT split from ${SFT_SOURCE}"
  python -m src.data.prepare_sft_split --input "${SFT_SOURCE}" --train-out data/train_sft.jsonl --valid-out data/valid_sft.jsonl --seed 42
fi

python train.py sft --config configs/sft.yaml

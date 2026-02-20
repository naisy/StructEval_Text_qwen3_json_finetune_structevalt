# Dataset sources and policy

## Goals

- Train on multi-format tasks (JSON/YAML/TOML/XML/CSV) without letting JSON dominate.
- Keep training data **syntax-clean** (lint/strict checks) and (for TOML) canonicalize to reduce mixed-style outputs.

## Sources

### Online datasets (Hugging Face)

- Enabled by `data.online_dataset.use: true` in `configs/sft_hf.yaml` / `configs/grpo_hf.yaml`.
- Dataset list is provided via env vars:
  - `HF_SFT_DATASETS`, `HF_GRPO_DATASETS`
  - `HF_SFT_SPLIT`, `HF_GRPO_SPLIT`
- Import scripts:
  - `src/data/import_hf_structured_sft.py` (creates SFT JSONL and GRPO task JSON)

Online flow **filters invalid** and **canonicalizes TOML**, but does not change the semantic content.

### Offline datasets (local)

- Enabled by `data.offline_dataset.use: true`.
- Inputs are local files/directories:
  - SFT: `*.jsonl`
  - GRPO: `*.json` (StructEval task arrays)

Offline-only build uses `src/data/build_offline_dataset.py` to normalize different local schemas into the repo format.

## Policy: no mixing when training on official HF data

When training on the officially provided HF datasets, do not mix in synthetic datasets.

- `scripts/run_*_hf.sh` only appends offline datasets when `online_dataset.use=true`.
- If `online_dataset.use=false`, offline datasets are treated as the primary data source.

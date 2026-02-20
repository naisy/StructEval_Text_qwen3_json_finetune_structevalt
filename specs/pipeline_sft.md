# SFT pipeline

Entry script:

- `scripts/run_sft_hf.sh`

Steps:

1. **Build training JSONL**
   - online: `import_hf_structured_sft` → `data/hf_sft.jsonl`
   - offline: `build_offline_dataset --stage sft` → `data/offline_sft.jsonl`

2. **Sampling / balancing**
   - `hf_select_subset --stage sft --dataset-source online|offline`

3. **Train/valid split**
   - `prepare_sft_split`

4. **(Optional) append extra datasets**
   - `append_extra_datasets` runs only when online mode is enabled.

5. **Training**
   - `train.py sft --config configs/sft_hf.yaml`

SFT uses `disable_cache: true` to avoid memory blow-ups from returning `past_key_values`.

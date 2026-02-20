# GRPO pipeline

Entry script:

- `scripts/run_grpo_hf.sh`

Steps:

1. **Build GRPO tasks JSON**
   - online: `import_hf_structured_sft` → `data/hf_grpo_tasks.json`
   - offline: `build_offline_dataset --stage grpo` → `data/offline_grpo_tasks.json`

2. **Sampling / balancing**
   - `hf_select_subset --stage grpo --dataset-source online|offline`

3. **Train/valid split**
   - `prepare_structeval_split`

4. **(Optional) append extra datasets**
   - `append_extra_datasets` runs only when online mode is enabled.

5. **Training**
   - `train.py grpo --config configs/grpo_hf.yaml`

GRPO keeps cache enabled by default for better generation speed.

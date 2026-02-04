# API: SFT (Stage A)

## Entry point
- `src/train_sft.py::run_sft(config_path, run_name=None)`

## Behavior
1. Loads config YAML (model, LoRA, training, dataset)
2. Loads dataset (JSONL with an `output` field by default)
3. Builds JSON-only prompts
4. Runs supervised fine-tuning with `transformers.Trainer`
5. Saves checkpoints under `runs/sft/<run_id>/checkpoints/`

## Notes
- SFT requires a dataset with an explicit target string (`output`).
- StructEval(-T) JSON tasks do not provide a unique gold output, so they are not suitable for SFT.
- For strict JSON tasks, prefer low temperature during inference.

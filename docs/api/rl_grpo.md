# API: GRPO RL (Stage B)

## Entry point
- `src/train_grpo.py::run_grpo(config_path, run_name=None)`

## Reward
Priority:
1. If the batch includes `raw_output_metric` (StructEval-style), reward uses **StructEval‑T** scoring:
   - `reward = final_eval_score = 0.2*syntax + 0.8*key_validation_score`
2. Otherwise fallback to the component reward:
   - Format-aware parse + "only" checks (JSON/YAML/TOML/XML/CSV)
   - Optional *gold matching* when the dataset provides `reference_output`

Implementation:
- `src/structeval_t/scorer.py::eval_structeval_t_json`
- `src/train_grpo.py::reward_fn(...)` 

Notes:
- TRL's GRPO passes dataset columns as keyword arguments to `reward_fn`.
  In particular, Hugging Face converted tasks (see `src/data/import_hf_structured_sft.py`) provide
  `output_type` and `reference_output` as columns; `reward_fn` consumes them directly.

## Outputs
- Run directory under `runs/grpo/<run_id>/`
- Checkpoints under `.../checkpoints/`

# API: GRPO RL (Stage B)

## Entry point
- `src/train_grpo.py::run_grpo(config_path, run_name=None)`

## Reward
Priority:
1. If the batch includes `raw_output_metric` (StructEval-style), reward uses **StructEval‑T** scoring:
   - `reward = final_eval_score = 0.2*syntax + 0.8*key_validation_score`
2. Otherwise fallback to the component reward:
   - JSON parse, schema validation, and task-specific constraints

Implementation:
- `src/structeval_t/scorer.py::eval_structeval_t_json`
- `src/train_grpo.py::reward_fn(...)` 

## Outputs
- Run directory under `runs/grpo/<run_id>/`
- Checkpoints under `.../checkpoints/`

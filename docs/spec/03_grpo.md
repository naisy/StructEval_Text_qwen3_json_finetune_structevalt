# GRPO

## Goal
Reinforcement learning (GRPO) to:
- reduce syntax errors (parse failures)
- reduce wrapper text (format-only)
- converge on canonical style

## Entry points
- `scripts/run_grpo_hf.sh`
- `src/train_grpo.py`

## Reward sources
1) **StructEval-T reward** when tasks include `raw_output_metric` (ATTRIBUTES):
   - uses StructEval-T scoring
   - adds extra shaping for YAML indentation / TOML canonical similarity

2) **Deterministic component reward** when ATTRIBUTES are absent (common in HF datasets):
   - parse / only / extraneous / match / match_soft
   - implemented in `src/rl/rewards.py`

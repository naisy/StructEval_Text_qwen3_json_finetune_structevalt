# SFT

## Goal
Supervised fine-tuning to teach the base model to:
- output *only* the requested structured format (JSON/YAML/TOML/XML/CSV)
- follow canonical style constraints (where defined)

## Entry points
- `scripts/run_sft_hf.sh`
- `src/train_sft.py`

## Notes
- `use_cache=False` is used in SFT to avoid memory growth from `past_key_values`.

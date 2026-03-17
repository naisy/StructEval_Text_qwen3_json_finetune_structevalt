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

## Loss masking
- `training.assistant_only_loss: true` masks prompt/user tokens with `labels=-100`.
- Only the assistant completion (`output + EOS`) contributes to SFT loss.
- Implementation: `src/train_sft.py`, `src/data/sft_collator.py`.

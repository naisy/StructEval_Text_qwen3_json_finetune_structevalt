# src.models — model loading & LoRA

## `src/models/load.py`
### `load_tokenizer(model_name, trust_remote_code=True)`
Loads tokenizer and configures padding defaults.

Notes:
- Uses `use_fast=True`.
- Applies the recommended `fix_mistral_regex=True` compatibility shim when supported by the installed
  `transformers` version, to avoid known fast-tokenizer regex issues for Mistral-family/derivative
  tokenizers.

### `load_model(model_name, dtype, trust_remote_code=True)`
Loads base model for CausalLM.

## `src/models/lora.py`
### `build_lora_config(cfg) -> peft.LoraConfig | None`
Builds LoRA config from YAML.

### `guess_target_modules(model) -> list[str]`
Heuristic for common target modules.

### `apply_lora(model, lora_cfg) -> model`
Applies PEFT adapters.

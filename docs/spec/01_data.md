# Data pipeline

## Inputs
- Hugging Face datasets (u-10bei/*, daichira/*)
- Optional local StructEval-T tasks JSON (for GRPO extension)

## HF import -> SFT/GRPO artifacts
The project converts HF chat-style datasets into two local files:
- `data/train_hf_sft.jsonl` / `data/valid_hf_sft.jsonl`
- `data/train_hf_grpo_tasks.json` / `data/valid_hf_grpo_tasks.json`

Conversion is implemented in `src/data/import_hf_structured_sft.py`.

### Key policies
- **Strict syntax filtering**: examples are dropped if the declared `output_type` cannot be parsed strictly.
- **Format-only filtering**: examples with wrapper text (markdown fences, extra prose) are dropped.
- **TOML canonicalization (output side)**: parsed TOML outputs are re-rendered into the repo's canonical TOML form to avoid style mixing across sources.

## Canonical TOML definition
Canonical TOML is defined by `src/data/toml_canonical.py`:
- stable key ordering
- deterministic `[table]` and `[[array_of_tables]]` emission
- no inline-table based nesting (repo style)

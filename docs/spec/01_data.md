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
- **Wrapper-text tolerant**: wrapper text (markdown fences, "Output:" preambles, etc.) is tolerated as long as the extracted structured payload parses strictly.
- **TOML canonicalization (output side)**: parsed TOML outputs are re-rendered into the repo's canonical TOML form to avoid style mixing across sources.

### u-10bei prompt/output policy
For `u-10bei/structured_data_with_cot_dataset*` datasets:
- Prefer `metadata.prompt` as the training `query` when present.
- Prefer `metadata.output` (or equivalent metadata keys) as the reference output when present.
- Fall back to the chat-style `messages` (system/user/assistant) only when metadata fields are missing.

Rationale: for text-only StructEval-T learning, `metadata.prompt`/`metadata.output` provides a clean single-turn representation without mixing system/user contexts or injecting extra metadata into the query.

## Canonical TOML definition
Canonical TOML is defined by `src/data/toml_canonical.py`:
- stable key ordering
- deterministic `[table]` and `[[array_of_tables]]` emission
- no inline-table based nesting (repo style)

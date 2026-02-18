# Query metadata supplement

## Goal

Some HuggingFace structured datasets (notably **u-10bei/** generation tasks) contain
very short user prompts (e.g. "Produce a TOML document for an api specification.").
In those cases, the **only** disambiguating information may live in metadata fields
(`schema`, `type`, `complexity`, ...).

Because the SFT/GRPO training pipeline for this repository uses **user-only** prompts
(`prompting.mode: contest`), metadata that never appears in the prompt cannot be used
as conditioning signal during training.

To address this, we append a small, clearly-marked *supplementary metadata* block to
the end of the `query` string when importing HF datasets.

## Format

The importer appends:

```
---
Supplementary metadata (for disambiguation only; do NOT include this block in your output):
FORMAT: TOML
TASK_KIND: generation
SCHEMA: api_specification
COMPLEXITY: medium
...
```

- The block is **English** and explicitly instructs the model not to copy it.
- Only available keys are emitted.
- This does **not** modify the upstream HF dataset. It only changes the derived
  JSONL files produced by this project.

## Ordering guarantees

For TOML examples, JSON-like payload conversion and canonicalization happen on the
**output** side. The metadata supplement affects only `query` and is independent of
output parsing/validation.

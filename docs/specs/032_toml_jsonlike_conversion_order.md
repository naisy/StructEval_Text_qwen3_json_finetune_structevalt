# TOML JSON-like payload conversion order (HF import)

Some Hugging Face structured-data datasets (observed in `u-10bei/*`) sometimes
label an example as `TOML`, but the assistant output payload is actually a JSON
object/array (starts with `{` or `[` and is parseable as JSON).

If we run strict TOML parsing first, those examples are filtered out as invalid.
However, they can be *deterministically* converted into canonical TOML and then
safely included in training.

## Where the conversion happens

During `src/data/import_hf_structured_sft.py` import:

1. Extract the final output payload from the assistant message.
2. If `output_type == TOML` and the payload looks like a JSON object/array:
   - Convert JSON -> TOML (`src/data/toml_jsonlike.py`).
3. Run strict format validation / cleaning policy.
4. Canonicalize TOML to the project's single style (`toml_canonical`).

## Invariants

- Conversion is applied **before** strict parsing.
- After conversion, the result must pass TOML parsing (tomllib) and canonicalization.
- Canonicalization never produces invalid TOML; if parsing fails, the example is dropped.

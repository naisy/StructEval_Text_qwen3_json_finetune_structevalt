# API: Evaluation (StructEval‑T)

## Entry point
- `src/eval/run_eval.py::run_eval(config_path, override_model_path=None)`

Produces:
- `outputs/eval/structeval_t_eval.json`
- `outputs/eval/eval_report.json`

## StructEval‑T fields (text-only)
For each item, we attach:
- `render_score` (float): **syntax_score** (0 or 1 for parsing the requested output format)
- `key_validation_score` (float): fraction of satisfied key paths
- `raw_output_eval` (list[bool]): per-path result
- `raw_output_score` (float): mean of `raw_output_eval`
- `final_eval_score` (float): `0.2 * render_score + 0.8 * key_validation_score` 

Additionally, we attach deterministic format diagnostics (shared with GRPO reward code):

- `parse` (float): 1 if strict parsing succeeds, else 0
- `parse_best_effort` (float): 1 if best-effort parsing succeeds (mainly JSON), else 0
- `only` (float): 1 if the output is *only* the target format (no wrapper text), else 0
- `extraneous` (float): 1 if wrapper/extraneous text is detected around the extracted payload, else 0
- `match` (float): 1 if parsed output matches `reference_output` exactly, else 0 (only when `reference_output` exists)
- `match_soft` (float): 0..1 string similarity on canonicalized structured output (only when `reference_output` exists)

This mirrors StructEval‑T's text-only scoring scheme. 

## Optional JSON Schema checks
If your dataset config enables the `schema` section, the evaluator can also run
JSON Schema validation (JSON output_type only) for debugging (not part of StructEval‑T).

- `schema.schema_path` must point to a **JSON file containing a schema object**.
- Implementation: `src/data/validators.py::build_schema_validator()` accepts either
  a schema dict or a file path.

## Offline scoring (smoke test)
- `src/eval/score_structeval.py::score_structeval_dataset(...)`

Usage:
- `python train.py score --input data/valid_structeval_t.json --output outputs/smoke_scored.json`

This does **not** run the model. It scores the `generation` field as-is (empty string if missing).

## "Untrained" evaluation
Running `train.py eval` without any fine-tuning is supported. It will load the
**base model** specified in `configs/eval.yaml` (e.g., `Qwen/Qwen3-4B-Instruct-2507`)
and run inference. If you only want to validate the scoring pipeline without
downloading a model, use `train.py score` instead.

## Debug counters (optional)
The template also computes:
- format-only rate (per output_type)
- format parse rate (per output_type)
- JSON Schema validation rate (JSON only; if enabled)
- toy task constraint rate (`authors` length == 2)

These are **not** part of StructEval‑T, but help debug training regressions.



## Notes
- Internal counters use `EvalCounts.total` (alias: `n`) for backward compatibility.


## Compatibility notes
- `EvalCounts` uses `total` as canonical count. Aliases: `n`, `constraints_ok`.


## Evaluation sampling
To keep evaluation fast on large datasets, evaluation defaults to **10 items**.

Configuration (in `configs/eval.yaml`):
- `eval.limit`: number of items to evaluate (`0` = evaluate all)
- `eval.seed`: if set, deterministically shuffle items with this seed before taking the first `limit`.

CLI overrides:
- `train.py eval --limit N --seed S`

Dataset overrides:
- `EVAL_TASKS_PATH=/path/to/tasks.json python train.py eval --config configs/eval.yaml`
  will override `dataset.valid_path` at runtime (useful for scripts and post-train eval).


## Optional: LLM-as-a-judge fields

When `configs/judge.yaml` enables a judge, each per-sample record includes `judge`:

- `judge.providers (priority list) / judge.provider (legacy)` / `judge.model`
- `judge.score` / `judge.passed`
- `judge.details` (raw JSON returned by the judge)
- `judge.cache_key` / `judge.cache_file` / `judge.cache_hit`
- `judge.reason` (compact summary: notes + failed_checks + breakdown)

You can refresh these fields from existing cache files with:

```bash
bash scripts/refresh_judge_from_cache.sh outputs/eval/structeval_t_eval.json outputs/eval/structeval_t_eval.json <provider> <model>
```

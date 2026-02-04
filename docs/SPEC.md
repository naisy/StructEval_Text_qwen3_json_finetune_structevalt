# Specification: Structured-Output Fine-tuning (Qwen3-4B-Instruct-2507 + StructEval‑T)

## Goal
Train a model that outputs **the requested structured format** (JSON / YAML / TOML / XML / CSV)
and meets task-defined constraints (e.g. required key paths / fields / shapes), using:

1. **SFT (Stage A)** to establish baseline format compliance
2. **GRPO RL (Stage B)** to optimize rule compliance with an automatic evaluator


### GRPO generation length settings

`configs/grpo.yaml` controls GRPO sampling lengths:

- `training.max_prompt_len`: prompt truncation length passed to TRL
- `training.max_completion_len`: max tokens sampled for the completion

Note: TRL versions differ in whether they accept these as top-level `GRPOTrainer(...)` kwargs or only inside `args`/`GRPOConfig`. This project injects them into `args` to ensure the configured values take effect across TRL versions.


Evaluation (primary): **StructEval‑T** style scoring for text-only structured outputs (JSON, YAML, ...). 

## Why StructEval‑T for this project?
StructEval organizes structured-output evaluation into:
- **StructEval‑T**: text-only outputs (JSON / YAML / CSV / TOML / …)
- **StructEval‑V**: renderable outputs (HTML / SVG / …)

This project targets **text-to-structure** tasks, so we adopt **StructEval‑T** scoring:
- `syntax_score` = 1 if the requested format parses, else 0
- `key_validation_score` = fraction of required key-paths satisfied
- `final_eval_score = 0.2 * syntax_score + 0.8 * key_validation_score` 

> Note: the official StructEval framework also supports rendering pipelines for StructEval‑V; we only implement the **text-only** subset here.

### Key-path syntax for `raw_output_metric`
This repo supports the key-path subset used by StructEval text tasks:
- Dot keys: `a.b.c`
- List index: `a.items[0].name`
- Wildcards over arrays or objects:
  - `a.items[*].name`
  - `a.items.*.name` (equivalent)
- Quoted keys: `a["weird.key"].b` / `a['weird.key'].b`

Wildcard semantics: **any** element/property value that satisfies the remainder path counts as present.
 

## Data contract

### Recommended (future) dataset format
Your future dataset can be kept in the **StructEval input format** (JSON array). Each item contains:
- `task_id` (string)
- `query` (string prompt)
- `raw_output_metric` (list of key paths to validate)
- `rendering` (false for text-only tasks)
- plus optional fields (task_name, etc.) 

Note:
- The public StructEval(-T) dataset does **not** provide `gold_output` for these text-to-structure tasks.
- In this repo, StructEval-style text tasks are treated as **prompt-only** (for eval/RL).

### Also supported (legacy) dataset format
JSONL with:
- `instruction` (string)
- `requirements` (list or string)
- `output` (string): target structured text (JSON/YAML/TOML/XML/CSV)

Choose via:
- `configs/dataset_eval.yaml` (eval/RL; StructEval JSON array, multi-format)
- `configs/dataset_sft.yaml` (SFT; JSONL with `output`)

## Training stages

### Stage A — SFT
Objective:
- Learn format-appropriate output (JSON/YAML/TOML/XML/CSV) and basic constraint adherence.

Method:
- LoRA on Qwen3-4B-Instruct-2507
- Teacher-forcing on a JSONL dataset that provides an `output` field (target text)

Deliverables:
- checkpoint under `runs/sft/<run_id>/checkpoints/`
- optional post-train eval via `train.py eval`

### Stage B — GRPO RL (reward shaping)
Objective:
- Optimize strict adherence to syntax (per output_type) + required key paths.

Reward:
- If `raw_output_metric` exists (StructEval-style): reward = `final_eval_score`
- Else: fallback reward = weighted mix of (JSON parse, schema validation, constraints)

Deliverables:
- checkpoint under `runs/grpo/<run_id>/checkpoints/`
- evaluation output JSON + summary JSON

## Reproducibility & experiment tracking
- Each `sft`/`grpo` run creates a run directory under `runs/<stage>/<run_id>/`
- `train.py plot` aggregates `*.summary.json` and produces a comparison plot (`runs/plots/score_by_run.png`)
- `train.py tune` runs a lightweight tuning loop (dry-run default) and records best overrides

## Smoke tests (no model required)
Because your real dataset is not available yet, you can validate the whole evaluation chain with mock data:

1. Generate mock StructEval-T dataset:
   - `python -m src.data.make_mock_structeval_t`
2. Score it offline (no model inference):
   - `python train.py score --input data/valid_structeval_t.json --output outputs/smoke_scored.json`
3. Plot summaries:
   - `python train.py plot`

## Files to update when requirements change
- Dataset format/fields → `docs/api/data.md`, `configs/dataset_eval.yaml`
- Evaluation math / metrics → `docs/api/eval.md`, `src/structeval_t/*`
- RL reward shaping → `docs/api/rl_grpo.md`, `src/train_grpo.py`, `src/structeval_t/scorer.py`


## Optional: LLM-as-a-judge evaluation
Enable `configs/judge.yaml` to add an LLM-based check for instruction/constraint following.
### Judge cache and analysis workflow

- Judge calls are cached under `outputs/cache/judge/<provider>/<model>/<hash>.json`.
- Cache filenames remain hash-based for reuse, but cache JSON now includes `task_id` (and `task_ids`) so you can identify
  which dataset items produced the cache entry.
- The main analysis artifact is `outputs/eval/structeval_t_eval.json`, which now includes:
  - `judge.cache_key` / `judge.cache_file` / `judge.cache_hit`
  - `judge.reason` (notes + failed_checks + breakdown)

If you want to re-hydrate / update `structeval_t_eval.json` from cache files (e.g., after switching judge provider/model),
run:

```bash
bash scripts/refresh_judge_from_cache.sh
```



## Scripts permissions
All `scripts/*.sh` are expected to be executable (chmod 755).


## Evaluation sampling
Default evaluation runs on 10 items for speed. Set `eval.limit: 0` to evaluate all items. If `eval.seed` is set, items are shuffled deterministically before slicing.


## Troubleshooting

### Warning: "incorrect regex pattern" when loading a tokenizer

If you see a warning like:

> The tokenizer you are loading ... with an incorrect regex pattern ... You should set the `fix_mistral_regex=True` flag ...

this refers to a known issue with some fast tokenizers (notably Mistral-family / derivative tokenizers) whose
pre-tokenizer regex shipped incorrectly and may lead to incorrect tokenization. Recent `transformers` versions
support a `fix_mistral_regex=True` flag on tokenizer loading.

This repo's `src.models.load.load_tokenizer()` applies this flag when available and falls back gracefully on
older `transformers` versions.


## Public StructEval import

- Eval dataset can be built from `TIGER-Lab/StructEval` on Hugging Face.
- JSON-output, non-rendering tasks are exported into a JSON array with fields compatible with this repo's eval.
- Use `scripts/download_structeval_json_eval.sh`.

## Pseudo-SFT dataset generation

Since public StructEval does not ship `gold_output`, SFT data can be generated by running a teacher model to produce *structured* targets (e.g. JSON/YAML/TOML/XML/CSV) and filtering for syntactic validity. Use `scripts/build_structeval_pseudo_sft.sh`.

The pseudo-SFT generator reads each task's `output_type` (or infers it from the query) and applies format-specific parsing. By default it writes only valid rows (`meta.parse_ok=true`) with non-empty `output`. For debugging, pass `--keep-invalid` to keep invalid/empty attempts; in that mode `output` may be empty, but fields under `meta` such as `meta.raw_generation`, `meta.raw_used`, `meta.output_type`, `meta.parse_ok`, `meta.syntax_strict_ok`, `meta.variant_id` (and for Ollama, `meta.ollama_api` / `meta.error`) are preserved for inspection.

Tip: `configs/dataset_sft.yaml` contains a `pseudo_sft:` block (e.g. `variants_per_task`, `output_types`, `split_by_output_type`). You can apply it by passing `--dataset-sft-config configs/dataset_sft.yaml` to the generator.

Note: `variants_per_task` is treated as *N variants per task* (data augmentation).

For multi-format exports (JSON/YAML/TOML/XML/CSV), the generator runs **output-type major**:

```
for output_type in OUTPUT_TYPES:
  for variant in 1..N:
    for task in tasks[output_type]:
      teacher_generate(...)
```

If you want diversity, set `temperature > 0` (and/or change `--seed`); with `temperature=0` some backends may return identical outputs.

### Pseudo-SFT progress logs

Progress is reported **within each variant pass** (StructEval exports `base_tasks` tasks; currently 50):

- `output_type variant i/N`: which output format and pass is running
- `base_tasks=T`: how many base tasks exist
- `progress X/T tasks`: task progress within this variant
- `attempts K/(T*N)`: global attempt count across all output types and variants
At the end of each output type, the generator prints both overall totals and a per-variant summary like:

```text
Ollama pseudo-SFT per-variant summary [json] (base_tasks=50):
  json variant 1/4 | success=33 invalid=1 empty=16 errors=0
  json variant 2/4 | success=...
  json variant 3/4 | success=...
  json variant 4/4 | success=...
```

## Eval scripts input selection

- `scripts/run_eval.sh` and `scripts/run_eval_with_judge.sh` automatically select the eval task JSON.
 - Preference order:
   1) `data/structeval_text_all.json` (public StructEval import; JSON/YAML/TOML/XML/CSV)
   2) `data/structeval_json_eval.json` (JSON-only export)
   3) `data/valid_structeval_t.json` (mock)
 - Override via `EVAL_TASKS_PATH` environment variable.

The same selection rule is applied to post-training evaluation when `run_eval_after_train: true` (SFT/GRPO). This avoids accidentally evaluating on the mock dataset (which contains placeholder queries).


### OpenAI judge (gpt-5.2) note

If you use `model: gpt-5.2`, do **not** send `temperature`. Set `judge.temperature: null` (or omit it) in `configs/judge.yaml`.
If you want sampling controls, use a model that supports them (e.g., `gpt-4.1`) and set `judge.temperature` to a float.

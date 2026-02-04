# Qwen3 JSON Finetune Template (SFT + GRPO RL + StructEval‑T)

This repository is a **project template** to fine-tune **Qwen3-4B-Instruct-2507** for **strict JSON output**.
It is structured for:
- **Stage A: SFT (supervised fine-tuning)** to stabilize *JSON-only* output and reduce syntax errors.
- **Stage B: RL (GRPO)** to optimize *schema compliance* and task-specific constraints with an automatic evaluator.

> This template focuses on **clarity**: `train.py` shows the full flow with comments, while detailed logic lives in `src/`.

---

## 1) Quick start

### Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### Prepare dataset
Place your training data (e.g. JSONL) under `data/` and adjust:
- `configs/dataset.yaml`
- prompt formatting in `src/data/dataset.py`

Expected JSONL fields (example):
- `instruction`: string
- `input`: string (optional)
- `output`: string (must be **JSON only**, no code fences)

### Run SFT
```bash
bash scripts/run_sft.sh
```

### Run GRPO (RL)
```bash
bash scripts/run_grpo.sh
```

### Evaluate
```bash
bash scripts/run_eval.sh
```

---

## 2) Project structure

- `train.py` — **main entry** with readable flow + explanations
- `src/cli.py` — command line interface (subcommands: `sft`, `grpo`, `eval`)
- `src/train_sft.py` — SFT orchestration
- `src/train_grpo.py` — GRPO orchestration
- `src/rl/rewards.py` — reward functions (JSON parse, JSON-only, schema compliance, etc.)
- `src/data/validators.py` — JSON parsing + schema checks
- `docs/` — specs & API docs (module-level references)

---

## 3) Notes on JSON-only enforcement

Many prompts say “Please output JSON code.” which may trigger Markdown code fences.
This repo enforces **JSON-only** in:
- training data formatting (`src/data/dataset.py`)
- evaluator / reward (`src/data/validators.py`, `src/rl/rewards.py`)

---

## 4) Docs

- Overall spec: `docs/SPEC.md`
- API-style docs: `docs/api/*.md`

---

## 5) License

MIT (see `LICENSE`).


---

## 5) Smoke test (no model required)

Because the real dataset will be released later, you can validate the pipeline now:

```bash
# 1) Generate mock StructEval‑T JSON tasks (no gold; eval-only)
python -m src.data.make_mock_structeval_t --n-train 2000 --n-valid 500

# 2) Offline scoring (no model inference)
python train.py score --input data/valid_structeval_t.json --output outputs/smoke_scored.json

# 3) Plot run summaries (if you ran sft/grpo/eval/tune already)
python train.py plot
```

## LLM-as-a-judge evaluation (optional)

You can optionally use an external evaluator LLM (OpenAI or Gemini) to score
**instruction/constraint following** in addition to StructEval-T.

1) Install optional deps:

```bash
pip install -r requirements-judge.txt
```

2) Configure and enable `configs/judge.yaml`:

To use an **Ollama** judge (local model), set in `configs/judge.yaml`:

```yaml
judge:
  enabled: true
  provider: ollama
  model: gpt-oss:20b
  ollama:
    host: http://127.0.0.1:11434
    use_chat: true
    num_ctx: 16384
```


- OpenAI: set `OPENAI_API_KEY`
- Gemini: set `GOOGLE_API_KEY` (or use Vertex AI auth with `GOOGLE_GENAI_USE_VERTEXAI=true`)

3) Run eval as usual:

```bash
bash scripts/run_eval.sh
```

When enabled, per-sample results include a `judge` field, and the eval summary includes `judge_score_avg`.
Judge results are cached under `outputs/cache/judge/<provider>/<model>/<hash>.json` (provider = openai | gemini | ollama` to avoid repeated paid calls.
Each cache entry now includes `task_id` (and `task_ids`) for easier debugging, and the per-sample eval file
`outputs/eval/structeval_t_eval.json` includes:

- `judge.cache_key` / `judge.cache_file` / `judge.cache_hit`
- `judge.reason` (a compact summary from cached judge fields)

If you switch judge providers/models or want to re-hydrate `structeval_t_eval.json` from existing cache files:

```bash
bash scripts/refresh_judge_from_cache.sh outputs/eval/structeval_t_eval.json outputs/eval/structeval_t_eval.json openai gpt-5.2
```



> Note: `OPENAI_API_KEY` must be **non-empty**. `export OPENAI_API_KEY=""` will fail.

## Using the public StructEval dataset (optional)

If you want to evaluate on more tasks before the official training dataset is released,
you can download StructEval from Hugging Face and export **text-only** tasks.

```bash
python -m src.data.import_structeval --out data/structeval_json_tasks.json

# Or export multiple output formats (JSON/YAML/TOML/XML/CSV)
python -m src.data.import_structeval \
  --split test \
  --out data/structeval_text_all.json \
  --output-types JSON,YAML,TOML,XML,CSV
```

Then point `configs/dataset.yaml` to that file.


### Evaluation item limit
By default, `train.py eval` evaluates **10 items** for speed. Configure in `configs/eval.yaml`:
- `eval.limit: 0` to evaluate all items
- `eval.seed` to deterministically shuffle before taking the first `limit`
You can override on the CLI: `train.py eval --limit N --seed S`.


## Import StructEval JSON eval tasks (optional)

You can download the public StructEval dataset (HF) and extract JSON-output, non-rendering tasks to evaluate at scale.

```bash
# requires: pip install datasets
bash scripts/download_structeval_json_eval.sh test data/structeval_json_eval.json

# evaluate (default 10 items; set eval.limit=0 for all)
python train.py eval --config configs/eval.yaml

```

## Import StructEval text tasks for multiple output formats (optional)

If you want to build pseudo-SFT data for multiple target formats, export all supported output types:

```bash
bash scripts/download_structeval_text_all.sh test data/structeval_text_all.json
```

## Build pseudo-SFT data from StructEval (optional)

StructEval tasks do not provide gold outputs. If you want SFT data, you can build **pseudo-labeled** examples using a teacher model:

```bash
bash scripts/build_structeval_pseudo_sft.sh data/structeval_json_eval.json data/structeval_pseudo_sft.jsonl Qwen/Qwen3-4B-Instruct-2507 200 42
```

You can also use an **Ollama** teacher (e.g. `gpt-oss:20b`) running on `http://127.0.0.1:11434`:

```bash
# backend=ollama is the 6th arg
bash scripts/build_structeval_pseudo_sft.sh data/structeval_json_eval.json data/structeval_pseudo_sft.jsonl gpt-oss:20b 200 42 ollama
```

You can control how many **variants per task** are generated (data augmentation) via:

- `configs/dataset_sft.yaml` → `pseudo_sft.variants_per_task` (default: 4)
- or env override: `VARIANTS_PER_TASK=...`

Example:

```bash
VARIANTS_PER_TASK=8 bash scripts/build_structeval_pseudo_sft.sh data/structeval_json_eval.json data/structeval_pseudo_sft.jsonl gpt-oss:20b 0 42 ollama
```

This will write JSONL rows with `instruction` and `output` (valid structured output for each task's target format).


#### What does `variants_per_task` mean?

`variants_per_task=4` means: for each **base StructEval task**, ask the teacher to generate **4 different outputs** (variant 1..4).

For multi-format exports (JSON/YAML/TOML/XML/CSV), the generator runs **output-type major**:

- take all CSV tasks → generate variant 1..N
- then all JSON tasks → generate variant 1..N
- ... and so on

So the maximum number of SFT rows is:

- single-format (e.g. 50 JSON tasks): `50 * variants_per_task`
- multi-format (e.g. 250 tasks = 50×5): `250 * variants_per_task`

Progress logs are reported per output type + variant (window=10 tasks):

```text
[ollama] csv variant 1/4 | base_tasks=50 | progress 10/50 tasks | attempts 10/1000 | window(10 tasks): success=6 invalid=0 empty=4 errors=0 | fallback_to_generate=0
...
[ollama] finished csv variant 4/4 | base_tasks=50 | success=...
Ollama pseudo-SFT stats [csv]: base_tasks=50 variants_per_task=4 max_rows=200
...
[ollama] json variant 1/4 | base_tasks=50 | ...
```

If `data/structeval_pseudo_sft.jsonl` (or `data/_debug_pseudo_sft.jsonl`) exists, `bash scripts/run_sft.sh` will automatically split it into:
- `data/train_sft.jsonl`
- `data/valid_sft.jsonl`

so you don't need to manually copy/rename files. You can also override the source path with `SFT_INPUT_JSONL=...`.

For debugging, you can run the generator with `--keep-invalid` (CLI option) to keep **invalid/empty** generations in the JSONL.
In that mode, `output` may be empty, but fields like `meta.raw_generation`, `meta.raw_used`, `meta.output_type`, `meta.parse_ok`, `meta.syntax_strict_ok`, `meta.output_only_ok`, `meta.variant_id`, and (for Ollama) `meta.ollama_api` / `meta.error` are preserved for analysis.

### Evaluation input selection

`bash scripts/run_eval.sh` and `bash scripts/run_eval_with_judge.sh` prefer `data/structeval_json_eval.json` if it exists; otherwise they fall back to `data/valid_structeval_t.json` (mock).

SFT/GRPO "run_eval_after_train" uses the same selection rule: if `data/structeval_json_eval.json` exists it will be used automatically; otherwise it falls back to the eval config's dataset (typically the mock file).

Override with:

```bash
EVAL_TASKS_JSON=data/your_tasks.json bash scripts/run_eval.sh
```


### OpenAI judge (gpt-5.2) note

If you use `model: gpt-5.2`, do **not** send `temperature`. Set `judge.temperature: null` (or omit it) in `configs/judge.yaml`.
If you want sampling controls, use a model that supports them (e.g., `gpt-4.1`) and set `judge.temperature` to a float.

## Tips

- Pseudo-SFT generation supports multiple samples per task. Use `VARIANTS_PER_TASK=4` (default) in `scripts/build_structeval_pseudo_sft.sh` to generate more diverse SFT rows.
- SFT saves a merged (LoRA-applied) model in `runs/.../final` by default so GRPO can start from it without stacking adapters.

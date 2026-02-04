Qwen/Qwen3-4B-Instruct-2507 をベースに、StructEval_T の Text-to-構造化変換（JSON/YAML/TOML/XML/CSV）を SFT + GRPO で強化するための作業メモです。

## 1) StructEval（Text）を全フォーマットでダウンロード

```bash
# test split の Text-to-{JSON,YAML,TOML,XML,CSV} を 1つのJSONにまとめて保存
bash scripts/download_structeval_text_all.sh test data/structeval_text_all.json
# -> Wrote 250 tasks (output_types=['JSON','YAML','TOML','XML','CSV']) ...
```

> 注: StructEval の HuggingFace 公開データは split が `test` のみです（train はありません）。


## 2) SFT 用の合成データ（pseudo-SFT）を作成（Ollama teacher）

```bash
time python -m src.data.build_pseudo_sft_from_structeval \
  --teacher-backend ollama \
  --teacher-model gpt-oss:20b \
  --ollama-host http://127.0.0.1:11434 \
  --ollama-num-ctx 16384 \
  --ollama-timeout-s 600 \
  --ollama-retries 2 \
  --ollama-format json \
  --input data/structeval_text_all.json \
  --output data/structeval_pseudo_sft.jsonl \
  --output-types json,yaml,toml,xml,csv \
  --limit 0 \
  --variants-per-task 4 \
  --temperature 0.2
```

- `variants-per-task=4` は「**各 output_type の base task（各50件）に対して 4 回ずつ生成**」です。
- 生成順は **output_type ごと**（例: `csv variant 1/4` → ... → `csv variant 4/4` → `json variant 1/4` → ...）です。
- `--ollama-format json` は **JSON タスクにだけ適用**され、YAML/TOML/XML/CSV では `format` フィールドを省略します。


## 3) SFT

```bash
time scripts/run_sft.sh
```

`SFT_INPUT_JSONL` を指定しなければ、デフォルトで `data/structeval_pseudo_sft.jsonl` を使います（内部で train/valid 分割します）。


## 4) GRPO

```bash
time scripts/run_grpo.sh
```


## 5) 評価（SFT/GRPO のあと）

評価は **学習に使っていない StructEval（download した元データ）** をそのまま使います。
（SFT は合成データを学習しているため、download した StructEval は未学習＝評価に利用可能）

```bash
# 推奨: multi-format (JSON/YAML/TOML/XML/CSV) の一括データ
bash scripts/run_eval.sh

# judge を有効にして評価したい場合
bash scripts/run_eval_with_judge.sh
```

デフォルトの評価サンプル数は **各フォーマット 10 件ずつ**（合計 50 件）です。
（`configs/eval.yaml` の `eval.limit_per_output_type` で変更できます）

データの優先順位（自動選択）:
1) `data/structeval_text_all.json`（multi-format）
2) `data/structeval_json_eval.json`（JSON-only）
3) `data/valid_structeval_t.json`（mock; 無ければ自動生成）

明示的に評価データを切り替える場合は環境変数を使います:

```bash
EVAL_TASKS_PATH=/path/to/tasks.json bash scripts/run_eval.sh
```

将来的に配布される「学習データ」「評価データ」は、
`configs/dataset_sft.yaml`（SFT）や `configs/dataset_structeval_json.yaml`（GRPO）内の
`dataset.*` / `eval_dataset.*` で指定でき、さらに CLI / 環境変数で上書きできます。


## Hugging Face datasets (allowed)

The following datasets are allowed for training (do **not** generate synthetic data from them):
- u-10bei/structured_data_with_cot_dataset_512_v2
- u-10bei/structured_data_with_cot_dataset_512_v4
- u-10bei/structured_data_with_cot_dataset_512_v5
- u-10bei/structured_data_with_cot_dataset_512
- u-10bei/structured_data_with_cot_dataset_v2
- u-10bei/structured_data_with_cot_dataset
- daichira/structured-3k-mix-sft
- daichira/structured-5k-mix-sft
- daichira/structured-hard-sft-4k

This repo includes a converter that turns these datasets into the local training formats:

- SFT JSONL: each line has `{ "query": "...", "output": "..." }`
- GRPO tasks JSON (StructEval-T style): JSON array with `raw_output_metric` extracted from `ATTRIBUTES:` blocks.

### SFT (Stage A) from HF datasets

```bash
# Convert + split + train (uses configs/sft_hf.yaml and configs/dataset_hf_sft.yaml)
bash scripts/run_sft_hf.sh
```

To customize which HF datasets to include:

```bash
export HF_SFT_DATASETS="u-10bei/structured_data_with_cot_dataset_512_v5 daichira/structured-3k-mix-sft"
export HF_SFT_SPLIT="train"
bash scripts/run_sft_hf.sh
```

### GRPO (Stage B) from HF datasets

GRPO requires `raw_output_metric` (key paths). The `daichira/*` datasets include an `ATTRIBUTES:` section in prompts, which this repo extracts automatically.

```bash
# Convert + split + train (uses configs/grpo_hf.yaml and configs/dataset_hf_grpo.yaml)
bash scripts/run_grpo_hf.sh
```

Customize datasets:

```bash
export HF_GRPO_DATASETS="daichira/structured-hard-sft-4k"
export HF_GRPO_SPLIT="train"
bash scripts/run_grpo_hf.sh
```

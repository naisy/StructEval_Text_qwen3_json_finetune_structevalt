# 05. フォーマット検証（非LLM）

## 目的

「JSON かどうか」「YAML として解釈できるか」などのフォーマット検証は、LLM judge に依存せずに決定的に行う。

## 実装

- `src/data/validators.py`
  - JSON: `json.loads`
  - YAML: `yaml.safe_load`
  - TOML: `tomllib.loads`
  - XML: `xml.etree.ElementTree.fromstring`
  - CSV: `csv.reader`

## プロンプト側の形式ルール

決定的パーサで採点する以上、モデル出力が構文的に壊れていると `syntax_score=0` になり、
GRPO reward も崩壊する。

そのため、`src.data.dataset.build_prompt()` が `output_type` を見て
system prompt に **形式別ルール**を注入する。

- ルール定義: `src/data/format_rules.py`（JSON/YAML/TOML/XML/CSV）
- 目的: 形式間の癖の混入（例: TOML に `:`、配列末尾カンマ、`[table]` を繰り返すなど）を抑制

## 追加 CLI

評価結果ファイル（`outputs/eval/structeval_t_eval.json`）を再チェックして、どのタスクがどの理由でパース失敗しているかを一覧表示する。

```bash
python -m src.tools.validate_eval_outputs \
  --input outputs/eval/structeval_t_eval.json \
  --show-failures 20
```
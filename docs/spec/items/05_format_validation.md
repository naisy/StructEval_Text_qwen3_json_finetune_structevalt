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

## 追加 CLI

評価結果ファイル（`outputs/eval/structeval_t_eval.json`）を再チェックして、どのタスクがどの理由でパース失敗しているかを一覧表示する。

```bash
python -m src.tools.validate_eval_outputs \
  --input outputs/eval/structeval_t_eval.json \
  --show-failures 20
```
# 05. フォーマット検証（非LLM）

## 目的

「JSON かどうか」「YAML として解釈できるか」などのフォーマット検証は、LLM judge に依存せずに決定的に行う。

## 実装

- `src/data/validators.py`
  - JSON: `json.loads`
  - YAML: `yaml.safe_load`
## 重要: markdown code fence は不合格

本プロジェクトでは、出力に `\`\`\`json ... \`\`\`` のような **markdown code fence** が含まれる場合は、
「余計な wrapper を含む」とみなし **不合格**とする。

- 評価（StructEval_T）: fenced 出力は `syntax_score=0`
- GRPO: fenced 出力は `only=0` かつ `extraneous=1` となり reward で不利

実装: `src/data/validators.py`
  - `contains_code_fence()`
  - 各 `parse_*()` は strict parse 時に fenced を検出したら失敗扱い

  - TOML: `tomllib.loads`
  - XML: `xml.etree.ElementTree.fromstring`
  - CSV: `csv.reader`


## 厳密性（YAML/CSV の落とし穴）

YAML と CSV はパーサが非常に寛容です。
例えば `Sure! Here is ...` のような自然文でも YAML としては「スカラー」として parse 成功してしまい、CSV も 1行1列として parse 成功します。

本リポジトリでは「構造データとして成立していること」を重視し、以下を **syntax/only 判定の前提**にしています。

- YAML: parse 結果が dict または list であること（スカラーは不合格）
- CSV: どこかの行に 2列以上が存在すること（1列のみは不合格）
  - さらにヘッダが 2列以上ある場合は、以降の行の列数がヘッダ幅と一致すること（空行は許容）

## YAML のスタイル（indent / flow-style）

YAML は parse できても、次のような「見た目の崩れ」が残ることがあります。

- **インデントが +2 ずれる**（過剰インデント）
- **JSON風の flow-style**（`{...}` / `[...]`）が混入する

これらは YAML の文法上は許容される場合があり、単純な `yaml.safe_load` だけでは検出できません。

本プロジェクトでは GRPO の報酬として、次の **決定的 lint** を追加で使用します。

- 2スペース単位のインデント（tab 禁止）
- ブロック開始行（末尾 `:` / `- ...:`）の次のネストは **ちょうど +2**
- flow-style（`{` / `[` を使ったコレクション）を避ける

実装: `src/data/validators.py`

- `yaml_indent_is_canonical()`
- `yaml_uses_flow_style()`

報酬側: `src/rl/rewards.py` / `src/train_grpo.py`

- `w_yaml_indent_canonical`, `p_yaml_indent_canonical_fail`
- `w_yaml_block_style`, `p_yaml_block_style_fail`

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
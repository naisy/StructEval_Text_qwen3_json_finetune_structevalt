# 04. LLM Judge（任意）

## 役割

StructEval_T のスコアとは別に、LLM による補助的な判定（例: キーのカバレッジ、制約順守）を付与する。

## 適用範囲

現行の judge プロンプトは JSON 指向（JSON 構文・キー検証が前提）である。
そのため、デフォルトでは **JSON タスクにのみ** judge を適用する。

設定:

- `configs/judge.yaml` の `judge.output_types: [JSON]`

## Judge 出力の安定化

OpenAI Responses API の JSON mode/structured output を有効にし、judge 自身の返答が JSON にならない失敗を減らす。

- 実装: `src/judge/providers/openai_judge.py` で `text.format.type: json_object` を設定

## 使い分け

- フォーマット検証だけが必要な場合は judge を無効化してよい（非 LLM）。
- judge の導入は「追加の品質信号」として扱う。

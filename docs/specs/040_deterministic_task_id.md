# HFインポート: GRPO/eval用タスクのtask_idを決定的にする

## 目的

Hugging Face の学習データを GRPO 用の StructEval-T 形式タスクへ変換する際、
`task_id` が実行ごとに変わると以下の問題が起きる。

- 既存の judge / eval キャッシュが task_id をキーにしている場合、
  同一タスクが別IDとして重複しやすい
- 再学習・再評価で「同じタスク」を追跡しづらい

そのため、HF インポートで生成する GRPO/eval タスクは **内容から決まる task_id** を使う。


## 仕様

### 対象

- `src/data/import_hf_structured_sft.py` の `--write-grpo-tasks` が生成するタスク
  - 出力: `data/hf_grpo_tasks.json`（JSON配列）
  - 各要素は StructEval-T 互換の `{task_id, query, output_type, raw_output_metric, reference_output, ...}` を含む


### task_id の生成規則

以下の文字列を連結し、`md5` でハッシュ化して `task_id` とする。

- `output_type`（例: `JSON` / `YAML` / `TOML` / `XML` / `CSV`）
- `query`（学習に使うプロンプト）
- `reference_output`（教師データ。CoT は含めない Output-only）

```
task_id = "hf_" + md5(output_type + "\n" + query + "\n" + reference_output).hexdigest()
```

これにより、

- 同一内容のタスクは実行順序や shuffle に関わらず同じ task_id になる
- 逆に、`reference_output` が違う場合は task_id も変わる


## 実装

- `src/data/import_hf_structured_sft.py`
  - `--write-grpo-tasks` 時に `hf_{md5}` 形式の task_id を生成する
  - 旧実装のような連番（`hf_00000001`）は使わない

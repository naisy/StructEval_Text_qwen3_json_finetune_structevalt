# 06. 既知の落とし穴とトラブルシュート

## 1) 「JSON なのに JSON じゃない」

主な原因は次の 2 つ。

### A. 生成が途中で切れている（max_new_tokens）

- 深いネストの JSON タスクで、末尾が欠けたまま出力が停止する。
- 対策: `configs/eval.yaml` の `generation.max_new_tokens` を 1024 以上へ。

### B. LLM judge の JSON 出力が壊れている

- `judge_output_not_json` が多い場合は judge 側の返答が JSON になっていない。
- 対策: OpenAI Responses API の JSON mode（`text.format.type=json_object`）を有効化。

## 2) YAML/TOML/XML/CSV が judge で失敗する

- 現行 judge テンプレートは JSON タスク向け。
- デフォルトで JSON のみに適用する（`configs/judge.yaml` の `output_types`）。

## 3) パース失敗の詳細を確認したい

- `python -m src.tools.validate_eval_outputs --input outputs/eval/structeval_t_eval.json`

## 4) SFT/GRPO 後の eval が異常に遅い（VRAM が共有メモリへスピルする）

学習直後のプロセスは、optimizer state / grad buffer などの CUDA メモリを保持したままになりやすい。
その状態で eval を同一プロセスで実行すると、VRAM が足りず共有 GPU メモリへ移動し推論が遅くなる。

対策:
- `configs/sft*.yaml` / `configs/grpo*.yaml` の `eval.post_train_eval_mode: exec`（推奨）
  - 学習プロセスを eval プロセスに置き換えるので、学習時の CUDA メモリが残らない。
- どうしても同一プロセスで走らせたい場合は `inprocess` にしても良いが、環境によっては残留する。

補足:
- `post_train_eval_mode: exec/subprocess` は内部的に `python -m src.cli eval ...` を実行します。
  そのため `src/cli.py` に `__main__` エントリが無いと、学習後に eval が起動しません。
  （このリポジトリでは修正済み）

## GRPO reward が JSON 前提に見える／多形式で学習できない

- `src/rl/rewards.py` が JSON のみを前提にしていると、YAML/TOML/XML/CSV は reward が正しく付かず学習が進みにくい。
- 本プロジェクトでは `output_type` を見て、形式別に `parse_*` / `*_only` を評価する実装にする。

## 5) SFT 中に GPU メモリがじわじわ増える

よくある原因は、CUDA のキャッシュアロケータが『より大きい一時領域』を要求するバッチ/カーネルに遭遇するたびに予約メモリを増やし、縮めないことです。
（リークではなく、予約領域が増える見え方になるケース）

対策（`configs/sft*.yaml`）:

- `training.pad_to_max_length: true`（推奨）: 常に `max_seq_len` までパディングして入力長を固定
- `training.disable_cache: true`（推奨）: 学習時の KV-cache を無効化
- `training.torch_empty_cache_steps: 50` など: 一定 step ごとに CUDA キャッシュを開放（速度は少し落ちる）
- `training.gradient_checkpointing: true`: 活性化メモリ削減（学習は遅くなる）

補足:
- `group_by_length: true` は padding を減らせますが、長いバッチが集中するとピークVRAMが上がることがあります。

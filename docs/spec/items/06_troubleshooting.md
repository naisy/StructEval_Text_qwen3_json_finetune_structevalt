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

## 2b) SFT/GRPO 後に TOML の syntax_score が 0 になる

学習データ自体が TOML として正しくても、**プロンプトに TOML ルールが無い**と、
モデルが JSON/YAML 由来の癖を混ぜて構文を壊しやすい。

典型例:
- 配列末尾カンマ: `["a",]`
- `key: value` の混入（TOML は `key = value`）
- `museum = ...` と `[[museum]]` を併用（immutable namespace エラー）
- `[artifact]` を繰り返す（本来は `[[artifact]]`）

対策:
- `src/data/format_rules.py` に形式別ルールを定義し、
  `src.data.dataset.build_prompt()` で `output_type` に応じて system prompt に注入する。

確認:
- `python -m src.tools.validate_eval_outputs --input outputs/eval/structeval_t_eval.json`

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

## 6) GRPO の loss が 0.0 のまま／completions が常に max_length で切れる

症状（TRL のログ例）:
- `completions/mean_length == max_completion_length`
- `completions/clipped_ratio == 1.0`
- `completions/*terminated_length == 0.0`（EOS が出ていない）

このとき **学習データが GRPO に向かない**というより、まず疑うべきは
**プロンプトがモデルの chat template と合っていない**ことです。

Qwen3 Instruct は `tokenizer.chat_template` が `<|im_start|>role ... <|im_end|>` を前提にしています。
誤って `<|system|>/<|user|>/<|assistant|>` のような別系列の sentinel を埋め込むと、
モデルはそれらをただの文字列として扱い、会話境界を認識できず生成が暴走しやすくなります。

対策:
- `src.data.dataset.build_prompt()` は Qwen3 互換の `<|im_start|>...<|im_end|>` 形式で組み立てる。

### 6b) GRPO の loss が 0.0 のまま（grad_norm=0.0 で reward がほぼ一定）

症状（今回のログ例に近い）:
- `loss: 0.0` が最初から続く
- `grad_norm: 0.0` が続く
- `rewards/reward_fn/std: 0.0` が続く（※per_device_train_batch_size=1 だと標準偏差表示は常に 0 になりやすい）

このパターンは **RL の advantage が 0 に潰れている**可能性が高い。
典型原因は「reward が各生成で同じ値になってしまう」こと。

例:
- reward が `parse + only` だけで、モデルがすでに形式だけ出せている
- `w_match`（厳密一致）が入っているが、全生成が一律に不一致で reward が一定

対策:
- HF データセットの `reference_output` を使って、reward に **dense な差**を入れる
  - `w_match_soft`（ソフト一致）を有効化（`configs/grpo_hf.yaml` がデフォルト）
  - さらに厳密一致の `w_match` も併用すると、完全一致を押し上げられる

確認:
- `data/train_hf_grpo_tasks.json` の数件を見て `reference_output` が入っているか
- `python - <<'PY'\nimport json\nfrom itertools import islice\nfor p in ['data/train_hf_grpo_tasks.json','data/valid_hf_grpo_tasks.json']:\n    try:\n        d=json.load(open(p,'r',encoding='utf-8'))\n    except FileNotFoundError:\n        continue\n    print(p,'n=',len(d))\n    for ex in islice(d,3):\n        print(' output_type=',ex.get('output_type'),'ref_len=',len((ex.get('reference_output')or'')))\nPY`

### 6c) GRPO 後に出力が fenced block（`\`\`\`...\`\`\``）に寄って評価が悪化する

症状:
- GRPO の後に、YAML/TOML/XML/CSV（場合によっては JSON も）が fenced block で出る比率が上がる

原因:
- 過去の実装では、YAML/TOML/XML/CSV の「形式のみ」判定が **fenced block を “only” として許容**していた。
  - その結果、GRPO が fenced block を強化してしまい、コンテスト推論（wrapper 禁止）の評価で悪化する。

対策（本リポジトリの現行実装）:
- `src/data/validators.py` で fenced block を **常に不合格**として扱う。
  - `parse_*()` の strict parse は fenced を検出したら失敗
  - `is_*_only()` は fenced を含む場合は必ず False
  - `extract_payload_and_extraneous()` は fenced がある場合 `extraneous=1` とする

関連メモ:
- 詳細の観察ログは `structeval_grpo_degradation_analysis.md` を参照

## 7) CSV→XML で snake_case が分解される（<some><thing> になる）

原因候補:
- XML のタグ名を推測する際に、モデルが `_` を区切りとして扱ってしまう。

対策:
- `src/data/format_rules.py` の XML ルールで「`some_thing` は `<some_thing>` のまま」と明示し、GRPO で矯正する。

補足:
- この症状は構文としては XML が成立するため、キー検証（raw_output_metric）で初めて落ちます。
  まずは eval レポートで該当タスクの required path を確認し、タグ名の期待が `_` を含むかを確認してください。


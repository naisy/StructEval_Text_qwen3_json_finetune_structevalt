# 02. データと学習（SFT/GRPO）

## データソース

### A. StructEval_T 由来の合成データ（初期フェーズ）

- StructEval_T の各形式（JSON/YAML/TOML/XML/CSV）を取得し、Ollama の `gpt-oss:20b` で SFT 用データを合成できる。
- `configs/dataset_sft.yaml` の `variants_per_task` で 1 タスクあたりの合成バリエーション数を指定する。

### B. Hugging Face 提供データ（現行ルール）

Hugging Face の提供データセットを使用する場合は、合成データの利用やデータ変更は行わない。
データの取得は `scripts/run_sft_hf.sh` / `scripts/run_grpo_hf.sh` が担当する。

## 学習スクリプト

- SFT: `scripts/run_sft.sh` / `scripts/run_sft_hf.sh`
- GRPO: `scripts/run_grpo.sh` / `scripts/run_grpo_hf.sh`

## 主な設定ファイル

- データ: `configs/dataset_sft.yaml`, `configs/dataset_grpo.yaml`
- 学習: `configs/sft.yaml`, `configs/grpo.yaml`

## 注意点

- `max_seq_len` は出力の欠損（途中切れ）に影響するため、極端に小さくしない。

## 学習時プロンプト（形式ルール）

本プロジェクトは、SFT/GRPO/eval のいずれも `src.data.dataset.build_prompt()` で
プロンプトを組み立てる。

- **SFT**: prompt + target（教師）を連結して teacher-forcing
- **GRPO**: prompt から生成した completion を決定的パーサで採点（外部LLM judge なし）
- **Eval**: prompt から生成し、StructEval_T で採点

そのため、学習データが構文的に正しくても、プロンプトに形式ルールが無いと
モデルが形式間の癖（例: TOML に `:` を混ぜる、配列末尾カンマ）を出して
`syntax_score` が崩壊することがある。

対策として、`example.output_type`（JSON/YAML/TOML/XML/CSV）に応じて
**形式別のルールを system prompt に注入**する（`src/data/format_rules.py`）。


## GRPO reward（形式別）

- GRPO の reward は **output_type（JSON/YAML/TOML/XML/CSV）ごとに決定的パーサ**で構文判定する。
- `configs/grpo.yaml` / `configs/grpo_hf.yaml` の `reward` では、`w_parse_<format>` と `w_only_<format>` を形式別に持つ。
  - 例: `w_parse_json`, `w_only_yaml`
- `p_only` は **「形式のみ」(fenced block 以外の余計な文字がない) の失敗**に対するペナルティ。
  - YAML/TOML/XML/CSV は自然文が構文的に成立してしまう場合があるため、fenced block を出した場合は **外側が空であること**を必須にしている。
  - さらに、解析時に「構造部分だけ」を抽出できた場合でも、外側に余計な文字があるケース（"Sure! ...", ```xml など）を強く減らしたい場合は `p_extraneous` を使う。
    - `p_extraneous_json` / `_yaml` / `_toml` / `_xml` / `_csv` で形式別に上書き可能。

## SFT 時のVRAM安定化（じわじわ増える対策）

Hugging Face Trainer は入力長やカーネル選択の揺らぎで CUDA の予約メモリ（cache）が段階的に増えることがあります。
以下の設定で挙動を安定させられます（`configs/sft*.yaml`）。

- `training.pad_to_max_length: true` で常に `max_seq_len` にパディングし、バッチごとの最大長の揺れをなくす
- `training.disable_cache: true` で学習時の KV-cache（past_key_values）を無効化
- まだ増える場合は `training.torch_empty_cache_steps: 50` などを設定し、一定stepごとに `torch.cuda.empty_cache()` を呼ぶ
- それでも厳しい場合は `training.gradient_checkpointing: true`（遅くなる代わりにVRAM削減）

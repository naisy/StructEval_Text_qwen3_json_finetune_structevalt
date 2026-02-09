# 02. データと学習（SFT/GRPO）

## データソース

### A. StructEval_T 由来の合成データ（初期フェーズ）

- StructEval_T の各形式（JSON/YAML/TOML/XML/CSV）を取得し、Ollama の `gpt-oss:20b` で SFT 用データを合成できる。
- `configs/dataset_sft.yaml` の `variants_per_task` で 1 タスクあたりの合成バリエーション数を指定する。

### B. Hugging Face 提供データ（現行ルール）

Hugging Face の提供データセットを使用する場合は、合成データの利用やデータ変更は行わない。
データの取得は `scripts/run_sft_hf.sh` / `scripts/run_grpo_hf.sh` が担当する。

#### HF データの「排除（cleaning）」

Hugging Face データはそのまま学習に流すと、
（a）出力の前後に余計な文章が混ざる、（b）prompt の指示形式と output_type がズレる、
（c）変換タスクなのに入力と出力が整合していない、などの理由で学習が不安定になることがある。

本プロジェクトでは `structeval_dataset_check.ipynb` で確定した排除手法を
HF 変換時に適用する（データ内容の改変はしない。サンプルを落とすだけ）。

- 実装: `src/data/hf_dataset_cleaning.py`（決定器）
- 適用箇所: `src/data/import_hf_structured_sft.py` の `--filter-invalid`
  - `scripts/run_sft_hf.sh` / `scripts/run_grpo_hf.sh` から呼ばれる

排除ポリシー（要点）:

- **drop（削除）対象 = 明らかに FAILED になるものだけ**
  - 抽出した structured payload が strict parse できない
  - prompt から推定した target format と output_type が一致しない
  - 変換タスク（input payload あり）で、入力/出力の意味的整合が **明確に FAIL**
    - **UNKNOWN は残す**（曖昧なら落とさない）
  - XML は containment F1 が高い soft match を許容する（デフォルト閾値 0.97）

※ 出力の前後に余計な文章（"Sure!" / "Output:" / fenced block の外側など）があるだけでは
FAILED と断定できないため、この理由だけでの削除は行わない。

## 学習スクリプト

- SFT: `scripts/run_sft.sh` / `scripts/run_sft_hf.sh`
- GRPO: `scripts/run_grpo.sh` / `scripts/run_grpo_hf.sh`

## 主な設定ファイル

- データ: `configs/dataset_sft.yaml`, `configs/dataset_grpo.yaml`
- 学習: `configs/sft.yaml`, `configs/grpo.yaml`

## 注意点

- `max_seq_len` は出力の欠損（途中切れ）に影響するため、極端に小さくしない。
- GRPO はデフォルトでは最後まで学習するが、短時間で挙動を確認したい場合は
  `configs/grpo*.yaml` の `training.max_steps` を正の整数に設定すると、その step で終了する。
  - `0` は「最後まで」（デフォルト）

## 学習時プロンプト（形式ルール）

本プロジェクトは、SFT/GRPO/eval のいずれも `src.data.dataset.build_prompt()` で
プロンプトを組み立てる（内部で tokenizer の `apply_chat_template()` を使用）。

### コンテスト推論との整合（最重要）

コンテストの推論は **user-only**（systemなし）で `query` をそのまま渡すため、
学習・評価も同じ形式に合わせないと、オフラインでは良く見えても本番で崩れる。

- `configs/dataset_hf_sft.yaml` / `configs/dataset_hf_grpo.yaml` / `configs/dataset_eval.yaml` は
  `prompting.mode: contest` を指定し、
  `messages=[{"role":"user","content": query}]` と同等になるようにする。

- **SFT**: prompt + target（教師）を連結して teacher-forcing
  - `build_prompt(..., add_generation_prompt=True)` は assistant turn を「開いた状態」で返す。
  - **SFT では target の末尾に EOS（`tokenizer.eos_token`）を付与し、終了境界を学習させる。**
- **GRPO**: prompt から生成した completion を決定的パーサで採点（外部LLM judge なし）
- **Eval**: prompt から生成し、StructEval_T で採点

そのため、学習データが構文的に正しくても、プロンプトに形式ルールが無いと
モデルが形式間の癖（例: TOML に `:` を混ぜる、配列末尾カンマ）を出して
`syntax_score` が崩壊することがある。

対策として、`prompting.mode: default` の場合は `example.output_type` に応じて
**形式別ルールを system prompt に注入**できる（`src/data/format_rules.py`）。

ただし `prompting.mode: contest` では system が無い前提のため、ルール注入は行わない。


## GRPO reward（形式別）

- GRPO の reward は **output_type（JSON/YAML/TOML/XML/CSV）ごとに決定的パーサ**で構文判定する。
- `configs/grpo.yaml` / `configs/grpo_hf.yaml` の `reward` では、`w_parse_<format>` と `w_only_<format>` を形式別に持つ。
  - 例: `w_parse_json`, `w_only_yaml`
- `p_only` は **「形式のみ」(余計な文章 / markdown wrapper なし) の失敗**に対するペナルティ。
  - 本プロジェクトのルールでは **markdown の code fence（```...```）は不合格**（「形式のみ」ではない）として扱う。
  - 解析時に「構造部分だけ」を抽出できた場合でも、外側に余計な文字があるケース（"Sure! ...", "Output:", fenced など）を強く減らしたい場合は `p_extraneous` を使う。
    - `p_extraneous_json` / `_yaml` / `_toml` / `_xml` / `_csv` で形式別に上書き可能。

### gold 出力がある場合（HF データセット）

Hugging Face の提供データセットは多くの場合 `reference_output`（正解の構造文字列）を含む。
このとき、構文（parse/only）だけを reward にすると、モデルが早期に「形式だけ出せる」状態に到達し、
**reward がバッチ内でほぼ一定になって GRPO の勾配が 0 に潰れる**ことがある。

本プロジェクトでは `reference_output` がある場合に以下の 2 種類を使える:

- `w_match`: **厳密一致**（パース結果が完全に同一なら 1）
- `w_match_soft`: **ソフト一致**（パース済み構造を正規化して文字列類似度で 0〜1）

HF データセットでのデフォルト設定は `configs/grpo_hf.yaml` を参照。

## SFT 時のVRAM安定化（じわじわ増える対策）

Hugging Face Trainer は入力長やカーネル選択の揺らぎで CUDA の予約メモリ（cache）が段階的に増えることがあります。
以下の設定で挙動を安定させられます（`configs/sft*.yaml`）。

- `training.pad_to_max_length: true` で常に `max_seq_len` にパディングし、バッチごとの最大長の揺れをなくす
- `training.disable_cache: true` で学習時の KV-cache（past_key_values）を無効化
- まだ増える場合は `training.torch_empty_cache_steps: 50` などを設定し、一定stepごとに `torch.cuda.empty_cache()` を呼ぶ
- それでも厳しい場合は `training.gradient_checkpointing: true`（遅くなる代わりにVRAM削減）

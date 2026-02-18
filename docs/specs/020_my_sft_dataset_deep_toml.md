# my_sft_dataset.jsonl: 深い階層のTOML（dotted tables / dotted keys）を追加学習する

## 背景
TOML の学習データにおいて、深い階層（depth > 1）の表現が希少な場合、
- `[a.b]` / `[a.b.c]` / `[[a.b]]` / `[[a.b.c]]` のような **ドット区切りのテーブル / 配列テーブル**
- `a.b = ...` / `a.b.c = ...` のような **ドット区切りキー代入**

の出現頻度が低くなり、推論時にこれらの構文が崩れるリスクがある。

本プロジェクトでは Hugging Face データセットの改変は禁止のため、追加学習用の少量データを
`data/my_sft_dataset.jsonl` に分離して保持し、必要に応じて SFT に追加する。

## 目的
- 深い階層の TOML 表現（dotted tables / dotted keys）を **確実に学習させる**
- 既存の HF train/valid 由来のデータと **重複しない** よう、追加データはローカル JSONL に分離

## 対象ファイル
- `data/my_sft_dataset.jsonl`

## データ形式（JSONL）
1行1JSON。最低限必要なフィールド：
- `task_id`: 一意な文字列（重複防止に利用）
- `query`: 学習時のユーザ入力（contest mode 互換の user-only を想定）
- `output`: 正解出力（TOML）
- `output_type`: `"TOML"`

推奨フィールド：
- `task_family`, `task_kind`, `task_schema`, `task_key`（分析・集計用）

## 追加ロジック
`scripts/insert_deep_toml_into_my_sft.py` が以下を行う。

- `data/my_sft_dataset.jsonl` が無ければ新規作成
- 既存ファイルがある場合、`task_id` の集合を取り、候補と重複しない行だけを追記
- 追加する候補は `is_deep_toml()`（dotted tables / dotted keys 検出）を満たすことを必須とする

## SFTへの取り込み
- `configs/sft_hf.yaml` の `data.use_extra_datasets: true` で有効化
- `data.extra_datasets` に `data/my_sft_dataset.jsonl` を指定

デフォルトは OFF（A/B テスト容易化）。

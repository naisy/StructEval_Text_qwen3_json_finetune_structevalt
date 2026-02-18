# 追加データセット: TOMLの深い階層（dotted paths）を最小深さでフィルタする

## 目的

`data/my_sft_dataset.jsonl` のような追加データセットには、
TOML の深い階層（dotted tables / dotted keys）を重点的に入れることがある。

一方で、

- 「2層以上（depth>=2）」を学習させたい
- 「3層以上（depth>=3）」のみをさらに強く学習させたい

など、投入する深さの閾値を実験的に切り替えたい。

そこで、**HF train/valid を改変せず**に、
`use_extra_datasets` で追加するローカルデータセットのみを対象に、
TOML の階層深さでフィルタできるようにする。

> 注意: このフィルタは **append 時に通すだけ** で、
> `data/my_sft_dataset.jsonl` 自体を書き換えたり、設定を「移し替える」ことはしない。


## 対象とする「深い階層」の定義

以下の dotted paths を検出し、`a.b.c` のように `.` で分割したセグメント数を depth とする。

- テーブルヘッダ
  - `[a.b.c]`
  - `[[a.b.c]]`
- dotted-key 代入
  - `a.b.c = ...`

上記の depth の最大値が `toml_min_depth` 以上なら「採用」する。


## 設定

`configs/sft_hf.yaml`（推奨）:

```yaml
data:
  use_extra_datasets: true
  extra_datasets:
    - data/my_sft_dataset.jsonl

  extra_filters:
    toml_min_depth: 3   # 2 または 3 を想定（null/未設定で無効）
```

- `data.use_extra_datasets=false` の場合は、追加データセット自体が読み込まれない（=フィルタも実行されない）。
- `data.extra_filters.toml_min_depth=null`（またはキー未設定）の場合は、深さフィルタは無効。

`configs/grpo_hf.yaml` にも同名キーを持たせるが、現状は **stage=sft のときのみ適用**される。


## 実装

- `src/data/toml_depth_check.py`
  - dotted paths を抽出し depth を計算する
- `src/data/append_extra_datasets.py`
  - `data.extra_filters.toml_min_depth` が設定されている場合、
    `output_type == "TOML"` の追加データのみ depth>=toml_min_depth を満たすものを残す
  - フィルタは `data/train_hf_sft.jsonl` / `data/valid_hf_sft.jsonl` に append する直前に適用される

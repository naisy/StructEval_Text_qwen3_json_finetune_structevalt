# HF Deep TOML Extraction (my_sft_dataset.jsonl)

## 目的

Hugging Face の公式学習データを使う場合でも、TOML の「深い階層（dotted depth が大きい）」例は希少で、
`hf_select_subset` のバランシング／間引きで落ちやすい。

そこで **深い TOML 例だけを抽出した JSONL を別ファイルとして作り、HF のバランシング後に append する**ことで、
「希少な深い TOML を確実に学習へ入れる」ことを可能にする。

## 設定

`configs/sft_hf.yaml`:

```yaml
data:
  use_extra_datasets: true
  extra_datasets:
    - data/my_sft_dataset.jsonl

  extra_filters:
    toml_min_depth: 2   # 2 以上（null/未設定で無効）
```

## 動作

`scripts/run_sft_hf.sh` は、HF の `data/hf_sft.jsonl` を生成した直後に、
以下の条件を満たす場合に `data/my_sft_dataset.jsonl` へ **追記（dedup付き）** する。

- `data.use_extra_datasets == true`
- `data.extra_datasets` に `data/my_sft_dataset.jsonl` が含まれる
- `data.extra_filters.toml_min_depth` が `2` 以上の int

※ `data/my_sft_dataset.jsonl` は他の希少データ（例: TOML-jsonlike修正）でも使うため、
ファイルの有無ではなく **--append により追記** する。

追記は `src/data/extract_deep_toml_from_sft_jsonl.py` が担当する（`--append`）。

### 抽出条件

- `output_type == "TOML"`
- `toml_depth_check.is_toml_deep(output, min_depth=toml_min_depth) == True`
- 重複排除（`output_type + query + output` の md5）

## 注意

- これは **合成データの生成ではなく**、HF データからの「抽出（コピー）」のみ。
- append は HF バランシング後に行うため、HF のサンプリング挙動を壊さずに希少例を追加できる。

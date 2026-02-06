# 03. 評価（StructEval_T）

## 目的

学習後モデルの出力を StructEval_T の指標で評価する。
本プロジェクトでは「指定フォーマットの厳密パース」と「キー検証（key validation）」を主に用いる。

## 実行

```bash
python -m src.eval.run_eval --config configs/eval.yaml
```

SFT/GRPO の学習完了直後に自動で評価を走らせる場合は、
`configs/sft*.yaml` / `configs/grpo*.yaml` の `eval.post_train_eval_mode` を使う。

- `exec`（推奨）: 学習プロセスを eval プロセスに置き換える。学習時の CUDA メモリが残らない。
- `subprocess`: 学習後に CUDA メモリを可能な限り解放した上で、別プロセスで eval を実行。
- `inprocess`: 学習後に CUDA メモリを解放してから同一プロセスで eval を実行（環境差で残留する場合あり）。

生成物

- `outputs/eval/structeval_t_eval.json`: サンプルごとの結果
- `outputs/eval/eval_report.json`: 集計レポート

`structeval_t_eval.json` の各サンプルには、StructEval‑T のスコアに加えて
GRPO の reward と同じ決定的ロジックで計算した診断値（`parse/only/extraneous` など）も含める。
これにより、YAML/TOML/XML/CSV で「構文は通るが余計な説明文が混ざる」といった失敗を
評価段階で定量化できる。

## 重要パラメータ

- `configs/eval.yaml` の `generation.max_new_tokens`
  - 512 だと深いネストの JSON が途中で切れて「JSON じゃない」と判定されるケースが実際に起きうる。
  - 1024 以上を推奨。

## 出力形式パース

`src.data.validators` の strict parser で JSON/YAML/TOML/XML/CSV のパース可否を判定する。
（この部分は LLM を使わない）
> 注意: YAML/CSV は parse 自体が寛容なため、本リポジトリでは「構造として成立」を追加条件にしています（YAML は dict/list、CSV は2列以上など）。


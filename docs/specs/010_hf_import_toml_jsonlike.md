# HFデータインポート: TOML出力に混入するJSON形式の検出と修正

## 背景
- u-10bei系データセットでは、`metadata.format == toml`（=出力種別がTOML）にもかかわらず、assistant出力がJSONオブジェクト/配列（`{...}` / `[...]`）になっているサンプルが混入することがある。
- これをSFT/GRPOで学習すると、TOML出力の自由度が不必要に上がり、厳密なTOML文法に収束しづらくなる。

## 目的
- Hugging Faceデータのインポート時に、TOMLとして扱う出力のうち「JSON形式に見えるもの」を検出し、TOMLに変換してから後段のTOML canonicalize処理へ流す。

## 適用箇所
- `src/data/import_hf_structured_sft.py`
  - HFデータをSFT JSONL / GRPO tasksへ変換する処理の中で、TOML出力の canonicalize の直前に適用する。

## 仕様
### 検出条件（保守的）
- `output_type == TOML` のときのみ対象。
- 抽出済みの参照出力 `out_text` の先頭（lstrip後）が `{` または `[` で始まる場合、JSONとしてparseを試みる。
  - JSONとしてparse成功した場合のみ「混入」とみなす。

### 変換ルール
- JSONの `null` はTOMLに存在しないため、空文字 `""` に写像する（学習データの安定化を優先）。
- JSON root が dict の場合:
  - TOML root dict としてダンプする。
- JSON root が list / scalar の場合:
  - `root` キー配下にラップして TOML としてダンプする。

### 変換後の整形
- 変換結果は既存のTOML canonicalize（決定的整形）処理により最終形式を統一する。

## 例
- 入力（誤混入）:
  - `{"a": 1, "b": true, "c": null}`
- 出力（修正後）:
  - `a = 1\n`
  - `b = true\n`
  - `c = ""\n`

## 影響範囲
- daichira系は基本的に該当しない想定だが、処理はTOML出力共通の前処理として適用されても副作用は小さい（JSON parse 成功時のみ変換される）。

# StructEval Text タスク評価：学習後にスコアが悪化する要因の整理（暫定）

作成日: 2026-02-09  
対象: Qwen/Qwen3-4B-Instruct-2507 をベースに SFT → GRPO を実施したモデルの推論ログ（`inference_3.json`, `inference_4.json`）と、検証用データ（`public_150.json`, `valid_hf_sft.jsonl`, `valid_hf_grpo_tasks.json`）からの観測に基づくまとめ。

---

## 1. どのモデルを使ったか

- ベースモデル: **Qwen/Qwen3-4B-Instruct-2507**
- 学習後モデル: 上記をベースに **SFT** と **GRPO** を適用した2系統
  - `inference_3.json`: SFT後 + GRPOを約1時間
  - `inference_4.json`: SFT後 + GRPOを約42時間

### モデルの特徴（一般的な性質）
- **Instruct（対話指向）**モデルであり、タスク文から「丁寧な前置き」「説明」「Markdown整形（```json```など）」を自然に出しやすい。
- 構造化出力（JSON/YAML/XML/TOML/CSV）タスクでは、**“出力全体が厳密にパース可能”**という評価条件がある場合、  
  こうした「余計な前置き／コードフェンス」が**致命的にスコアを落とす**原因になり得る（=能力というより“出力規律”の問題）。

---

## 2. どのデータセットを使ったか

- 公開データセット（ユーザー提示）:
  - Hugging Face: `u-10bei/structured_data_with_cot_dataset_512_v2`

- 手元検証用（添付）:
  - `valid_hf_sft.jsonl`（SFT用の検証セット）
  - `valid_hf_grpo_tasks.json`（GRPO用タスク形式の検証セット）
  - `public_150.json`（StructEval Textの150タスク）

### データセットの特徴（添付ファイルから確認できる範囲）
- **入出力変換タスク中心**（例: XML→JSON / テキスト→JSON / CSV→JSON 等）
- `output_type` は少なくとも **JSON / YAML / TOML / XML / CSV** を含む（`public_150.json`より）
- 参照出力（`reference_output` / `output`）は基本的に **“コード本体のみ”**で、Markdownフェンス（```）は含まれない（`valid_hf_grpo_tasks.json`で確認）。

---

## 3. どのような学習をおこなったか

### 学習手法
- **SFT（Supervised Fine-Tuning）**
  - 教師データ（query → output）での通常の教師あり微調整。
- **GRPO（Group Relative Policy Optimization）**
  - 強化学習系の最適化（相対報酬/グループ比較に基づくポリシー更新）。

### 学習パラメータ
- 学習率、KL係数、バッチサイズ、生成長、温度などの詳細は **未提供（本メモでは未記載）**。

---

## 4. どのようなデータセット変換をおこなったか（推定）

添付の `valid_hf_sft.jsonl` と `valid_hf_grpo_tasks.json` の中身から、少なくとも以下の形式への変換が行われていると判断できます。

### 4.1 SFT用データ（`valid_hf_sft.jsonl`）
各行が以下キーを持つ JSON:
- `query`: 入力（例: “Convert the following XML to JSON: ...”）
- `output`: 参照出力（例: JSON本体）
- `output_type`: 参照出力の形式（例: JSON）

**特徴**
- 1サンプル=1ペアのシンプルな教師あり形式。
- 参照出力は基本的に「構造コードのみ」。

### 4.2 GRPO用データ（`valid_hf_grpo_tasks.json`）
各要素が以下のキーを持つタスク辞書:
- `task_id`, `task_name`, `input_type`, `output_type`
- `query`（入力）
- `reference_output`（参照出力）
- `raw_output_metric`（生の評価指標らしきフィールド）
- `feature_requirements`（この検証ファイルでは空文字のものが多い）
- `query_example`, `VQA`, `rendering` など

**特徴**
- RLで報酬計算・評価を行うためのメタ情報が付加されている形式。
- 参照出力は「構造コードのみ」で、Markdownフェンスは含まれない。

---

## 5. 学習後の評価（推論結果）をみて、何が言えるか？

ここでは、StructEval Textの150タスク（`public_150.json`）に対する推論ログ2種を、  
**「出力全体が output_type として厳密にパース可能か」**で集計した観測結果です。

### 5.1 タスク構成（`public_150.json`）
- JSON: 50
- YAML: 35
- TOML: 25
- XML: 20
- CSV: 20

### 5.2 “出力全体の厳密パース成功率” （観測）
| output_type | inference_3 (SFT+GRPO≈1h) | inference_4 (SFT+GRPO≈42h) |
|---|---:|---:|
| JSON | 13/50 = **26%** | 11/50 = **22%** |
| YAML | 4/35 = **11.4%** | 0/35 = **0%** |
| TOML | 0/25 = **0%** | 0/25 = **0%** |
| XML | 2/20 = **10%** | 2/20 = **10%** |
| CSV | 19/20 = **95%** | 20/20 = **100%** |

### 5.3 典型的な失点パターン（観測）
- **コードフェンス（```json / ```yaml など）を付ける**
- **前置き文（“Sure!”, “Here’s …” など）を付ける**
- **長文化して末尾が切れ、括弧が閉じない**
- XMLでも `<` 以外の文字で始まってしまう（前置きが入る）

特に **YAML** は `inference_4` で、ほぼ全件がコードフェンス付きになり「出力全体パース」が成立しなくなっています。

### 5.4 フォーマット汚染（コードフェンス）率（観測）
| output_type | inference_3 | inference_4 |
|---|---:|---:|
| JSON | 74% | 78% |
| YAML | 88.6% | **100%** |
| TOML | **100%** | **100%** |
| XML | 90% | 90% |
| CSV | 0% | 0% |

**示唆**
- 42時間GRPO後（`inference_4`）は、**“純粋な構造コードのみを返す”規律がさらに崩れる**方向に進んでいる。
- 一方でCSVはほぼ問題なく、**形式によって崩れ方が偏っている**（TOML/YAML/XMLが特に厳しい）。

---

## 6. 何が課題か？

### 課題1: 「能力」ではなく「出力規律（フォーマット遵守）」が崩れている
- 参照出力は基本的に “コード本体のみ” なのに、生成では
  - 説明文
  - Markdownコードフェンス
  - 前置き
  が混入し、評価器が **失格扱い（パース不可）**にする可能性が高い。

### 課題2: GRPOが“コードフェンス/説明”を強化している可能性
- `inference_4` でYAMLが **100%コードフェンス化**しており、長時間GRPOで悪化。
- 報酬設計が
  - 「どこかに正しい構造が含まれていればOK」
  - 「部分抽出して評価」
  - 「フォーマット汚染に十分な罰がない」
  のようになっていると、モデルは“丁寧な説明＋コードブロック”へ最適化しやすい。

### 課題3: TOMLの失敗が恒常的
- TOMLは両ログで「全体パース成功 0%」かつ「フェンス率 100%」。
- 形式自体の生成品質（キー=値、クォート、配列/テーブル表現）も含めて、SFT/RLでの規律付けが不足している可能性。

---

## 7. 次の打ち手（プロンプトが使えない前提での方向性）

1. **報酬関数を“出力全体の厳密パース”にする（最優先）**
   - JSON/YAML/TOML/XML/CSV それぞれで、出力全体が strict parser を通らなければ報酬を0（または大減点）。
   - ``` を含む場合も強い減点。
2. **長さペナルティの導入**
   - 説明文が付くほど長くなるため、適度なlength penaltyが効きやすい。
3. **KL強化・学習率低下・早期停止**
   - 42hで崩壊している可能性があるため、ピーク性能のチェックポイント選択が重要。
4. **SFTデータの正規化**
   - 参照出力は「構造のみ」に統一し、フェンス/説明混在サンプルは除去・修正。
5. **推論側の制約（可能なら）**
   - Grammar decoding / constrained decoding（特にJSONは有効）。
   - プロンプトを変えずに“出力可能文字列”や“構文”を制約できる。

---

## 付録: 使用した手元ファイル
- `public_150.json`（150タスク、output_type分布の集計に使用）
- `inference_3.json`（推論ログ）
- `inference_4.json`（推論ログ）
- `valid_hf_sft.jsonl`（SFT用検証データの形式確認）
- `valid_hf_grpo_tasks.json`（GRPO用タスク形式の確認）

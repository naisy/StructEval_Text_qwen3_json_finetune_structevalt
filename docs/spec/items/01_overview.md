# 01. プロジェクト概要

## 目的

`Qwen/Qwen3-4B-Instruct-2507` をベースに、StructEval_T のタスク（JSON / YAML / TOML / XML / CSV）に対する推論能力を強化する。

## 想定する入出力

- 入力: StructEval_T 形式の task（`query` と `answer` など）
- 出力: 指定された形式（JSON/YAML/TOML/XML/CSV）のみ

## 主要コマンド

- SFT: `scripts/run_sft.sh` / `scripts/run_sft_hf.sh`
- GRPO: `scripts/run_grpo.sh` / `scripts/run_grpo_hf.sh`
- 評価: `python -m src.eval.run_eval --config configs/eval.yaml`

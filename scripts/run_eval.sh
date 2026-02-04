#!/usr/bin/env bash
set -euo pipefail

# Run evaluation.
#
# By default, this script prefers the downloaded public StructEval Text tasks
# (JSON/YAML/TOML/XML/CSV) at:
#   data/structeval_text_all.json
# If that file does not exist, it falls back to:
#   data/structeval_json_eval.json (JSON-only export)
#   data/valid_structeval_t.json   (local mock tasks, created by scripts/smoke_test.sh)
#
# You can override the eval input via:
#   EVAL_TASKS_PATH=/path/to/tasks.json bash scripts/run_eval.sh

EVAL_TASKS_PATH="${EVAL_TASKS_PATH:-}"

PREFERRED="data/structeval_text_all.json"
SECONDARY="data/structeval_json_eval.json"
FALLBACK="data/valid_structeval_t.json"

if [[ -z "${EVAL_TASKS_PATH}" ]]; then
  if [[ -f "${PREFERRED}" ]]; then
    EVAL_TASKS_PATH="${PREFERRED}"
  elif [[ -f "${SECONDARY}" ]]; then
    EVAL_TASKS_PATH="${SECONDARY}"
  else
    EVAL_TASKS_PATH="${FALLBACK}"
  fi
fi

if [[ ! -f "${EVAL_TASKS_PATH}" ]]; then
  echo "ERROR: eval input JSON not found: ${EVAL_TASKS_PATH}"
  echo ""
  echo "To download public StructEval tasks:"
  echo "  pip install datasets"
  echo "  bash scripts/download_structeval_text_all.sh test data/structeval_text_all.json"
  echo "  # or JSON-only"
  echo "  bash scripts/download_structeval_json_eval.sh test data/structeval_json_eval.json"
  echo ""
  echo "Or to generate mock tasks:"
  echo "  bash scripts/smoke_test.sh"
  exit 1
fi

mkdir -p outputs/tmp

# Create temporary dataset config pointing to the desired valid_path (and train_path for completeness).
TMP_DATASET_CFG="outputs/tmp/dataset.eval.yaml"
python - <<PY
import yaml, pathlib
p = pathlib.Path("configs/dataset_eval.yaml")
if not p.exists():
    p = pathlib.Path("configs/dataset.yaml")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
cfg.setdefault("dataset", {})
cfg["dataset"]["valid_path"] = "${EVAL_TASKS_PATH}"
cfg["dataset"]["train_path"] = "${EVAL_TASKS_PATH}"
pathlib.Path("${TMP_DATASET_CFG}").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("Wrote", "${TMP_DATASET_CFG}", "valid_path=", cfg["dataset"]["valid_path"])
PY

# Create temporary eval config pointing to the temporary dataset config.
TMP_EVAL_CFG="outputs/tmp/eval.eval.yaml"
python - <<PY
import yaml, pathlib
p = pathlib.Path("configs/eval.yaml")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
cfg["data"]["dataset_config"] = "${TMP_DATASET_CFG}"
pathlib.Path("${TMP_EVAL_CFG}").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("Wrote", "${TMP_EVAL_CFG}", "dataset_config=", cfg["data"]["dataset_config"])
PY

python train.py eval --config "${TMP_EVAL_CFG}"

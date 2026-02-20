#!/usr/bin/env bash
set -euo pipefail

CFG_PATH="configs/grpo_hf.yaml"

# Decide dataset source from config (online vs offline)
read -r USE_ONLINE USE_OFFLINE CFG_DATASETS CFG_SPLIT SHUFFLE_SEED <<<"$(python - <<'PY'
import yaml
cfg=yaml.safe_load(open('configs/grpo_hf.yaml','r',encoding='utf-8')) or {}
d=(cfg.get('data') or {})
on=(d.get('online_dataset') or {})
off=(d.get('offline_dataset') or {})
use_on = bool(on.get('use', True))
use_off = bool(off.get('use', False))
datasets = on.get('datasets', None)
split = on.get('split', 'train')
imp = on.get('import', {}) if isinstance(on.get('import', {}), dict) else {}
shuffle_seed = imp.get('shuffle_seed', 42)
if isinstance(datasets, list):
    datasets = ' '.join(str(x) for x in datasets)
elif datasets is None:
    datasets = ''
else:
    datasets = str(datasets)
print('1' if use_on else '0', '1' if use_off else '0', datasets, split, int(shuffle_seed))
PY
)"

GRPO_INPUT_JSON=""

if [ "${USE_ONLINE}" = "1" ] && [ "${USE_OFFLINE}" = "1" ]; then
  echo "WARN  Both online_dataset.use and offline_dataset.use are true. This usually violates contest rules (mixing sources)."
fi

if [ "${USE_ONLINE}" = "1" ]; then
  # Convert allowed HF datasets into StructEval-T style tasks.
  # You can override via env vars HF_GRPO_DATASETS / HF_GRPO_SPLIT.
  if [ -n "${CFG_DATASETS}" ]; then
    DEFAULT_DATASETS="${CFG_DATASETS}"
  else
    DEFAULT_DATASETS="daichira/structured-3k-mix-sft daichira/structured-5k-mix-sft daichira/structured-hard-sft-4k"
  fi
  DATASETS="${HF_GRPO_DATASETS:-${DEFAULT_DATASETS}}"
  SPLIT="${HF_GRPO_SPLIT:-${CFG_SPLIT}}"

  PYTHONPATH="$(pwd)" python -m src.data.import_hf_structured_sft \
    --datasets ${DATASETS} \
    --split "${SPLIT}" \
    --out-grpo-tasks data/hf_grpo_tasks.json \
    --write-grpo-tasks \
    --filter-invalid \
    --shuffle-seed "${SHUFFLE_SEED}"

  GRPO_INPUT_JSON="data/hf_grpo_tasks.json"
fi

if [ -z "${GRPO_INPUT_JSON}" ] && [ "${USE_OFFLINE}" = "1" ]; then
  PYTHONPATH="$(pwd)" python -m src.data.build_offline_dataset \
    --stage grpo \
    --config "${CFG_PATH}" \
    --out data/offline_grpo_tasks.json
  GRPO_INPUT_JSON="data/offline_grpo_tasks.json"
fi

if [ -z "${GRPO_INPUT_JSON}" ]; then
  echo "ERROR  No dataset source enabled. Set data.online_dataset.use or data.offline_dataset.use to true in ${CFG_PATH}." >&2
  exit 1
fi

# Subset selection (balance by task_key OR per-output-type targets)
# Config: configs/grpo_hf.yaml -> data.sampling
PER_DEVICE_BS="${GRPO_PER_DEVICE_TRAIN_BS:-1}"
GRAD_ACCUM="${GRPO_GRAD_ACCUM:-8}"
MAX_STEPS="${GRPO_MAX_STEPS:-100}"
PYTHONPATH="$(pwd)" python -m src.data.hf_select_subset \
  --stage grpo \
  --config configs/grpo_hf.yaml \
  --input "${GRPO_INPUT_JSON}" \
  --input-format json \
  --output data/hf_grpo_tasks_selected.json \
  --output-format json \
  --per-device-train-batch-size "${PER_DEVICE_BS}" \
  --grad-accum "${GRAD_ACCUM}" \
  --max-steps "${MAX_STEPS}"

GRPO_INPUT_JSON="data/hf_grpo_tasks_selected.json"

PYTHONPATH="$(pwd)" python -m src.data.prepare_structeval_split \
  --in-json "${GRPO_INPUT_JSON}" \
  --out-train data/train_hf_grpo_tasks.json \
  --out-valid data/valid_hf_grpo_tasks.json \
  --seed 42 \
  --valid-ratio 0.1

echo "INFO  Extra datasets feature removed; using only selected dataset source(s)."

# Ensure StructEval-T multi-format eval tasks exist ONLY when post-train eval is enabled.
#
# Hugging Face GRPO training itself does not require StructEval-T tasks.
NEED_POST_EVAL="$(python - <<'PY'
import yaml
cfg = yaml.safe_load(open('configs/grpo_hf.yaml', 'r', encoding='utf-8'))
print('1' if bool(((cfg.get('eval') or {}).get('run_eval_after_train', False))) else '0')
PY
)"

if [ "${NEED_POST_EVAL}" = "1" ]; then
  if [ ! -f data/structeval_text_all.json ]; then
    echo "INFO  Downloading StructEval-T eval tasks (JSON/YAML/TOML/XML/CSV) to data/structeval_text_all.json ..."
    bash scripts/download_structeval_text_all.sh test data/structeval_text_all.json "JSON,YAML,TOML,XML,CSV"
  fi
else
  echo "INFO  Post-train eval disabled (configs/grpo_hf.yaml: eval.run_eval_after_train=false). Skip StructEval-T download."
fi

PYTHONPATH="$(pwd)" python train.py grpo --config configs/grpo_hf.yaml

#!/usr/bin/env bash
set -euo pipefail

CFG_PATH="configs/sft_hf.yaml"

# Decide dataset source from config (online vs offline)
read -r USE_ONLINE USE_OFFLINE CFG_DATASETS CFG_SPLIT <<<"$(python - <<'PY'
import yaml
cfg=yaml.safe_load(open('configs/sft_hf.yaml','r',encoding='utf-8')) or {}
d=(cfg.get('data') or {})
on=(d.get('online_dataset') or {})
off=(d.get('offline_dataset') or {})
use_on = bool(on.get('use', True))
use_off = bool(off.get('use', False))
datasets = on.get('datasets', None)
split = on.get('split', 'train')
if isinstance(datasets, list):
    datasets = ' '.join(str(x) for x in datasets)
elif datasets is None:
    datasets = ''
else:
    datasets = str(datasets)
print('1' if use_on else '0', '1' if use_off else '0', datasets, split)
PY
)"

SFT_INPUT_JSONL=""

if [ "${USE_ONLINE}" = "1" ] && [ "${USE_OFFLINE}" = "1" ]; then
  echo "WARN  Both online_dataset.use and offline_dataset.use are true. This usually violates contest rules (mixing sources)."
fi

if [ "${USE_ONLINE}" = "1" ]; then
  # Convert allowed HF datasets to SFT JSONL and split into train/valid.
  # You can override via env vars HF_SFT_DATASETS / HF_SFT_SPLIT.
  if [ -n "${CFG_DATASETS}" ]; then
    DEFAULT_DATASETS="${CFG_DATASETS}"
  else
    DEFAULT_DATASETS="u-10bei/structured_data_with_cot_dataset_512_v5 daichira/structured-3k-mix-sft daichira/structured-5k-mix-sft daichira/structured-hard-sft-4k"
  fi
  DATASETS="${HF_SFT_DATASETS:-${DEFAULT_DATASETS}}"
  SPLIT="${HF_SFT_SPLIT:-${CFG_SPLIT}}"

  PYTHONPATH="$(pwd)" python -m src.data.import_hf_structured_sft \
    --datasets ${DATASETS} \
    --split "${SPLIT}" \
    --out-sft-jsonl data/hf_sft.jsonl \
    --filter-invalid \
    --shuffle-seed 42

  SFT_INPUT_JSONL="data/hf_sft.jsonl"
fi

if [ -z "${SFT_INPUT_JSONL}" ] && [ "${USE_OFFLINE}" = "1" ]; then
  # Build a single offline jsonl from local files/dirs.
  PYTHONPATH="$(pwd)" python -m src.data.build_offline_dataset \
    --stage sft \
    --config "${CFG_PATH}" \
    --out data/offline_sft.jsonl
  SFT_INPUT_JSONL="data/offline_sft.jsonl"
fi

if [ -z "${SFT_INPUT_JSONL}" ]; then
  echo "ERROR  No dataset source enabled. Set data.online_dataset.use or data.offline_dataset.use to true in ${CFG_PATH}." >&2
  exit 1
fi

# --------------------------------------------------------------
# Legacy: extra local datasets / deep TOML extraction
#
# This repo no longer supports the legacy "append extra datasets after balancing" mechanism.
# Keeping it around causes confusion and makes contest rule compliance harder.
# --------------------------------------------------------------

# Subset selection (balance by task_key OR per-output-type targets)
# Config: configs/sft_hf.yaml -> data.sampling
PER_DEVICE_BS="${SFT_PER_DEVICE_TRAIN_BS:-2}"
GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
EPOCHS="${SFT_EPOCHS:-2}"
PYTHONPATH="$(pwd)" python -m src.data.hf_select_subset \
  --stage sft \
  --config configs/sft_hf.yaml \
  --input "${SFT_INPUT_JSONL}" \
  --input-format jsonl \
  --output data/hf_sft_selected.jsonl \
  --output-format jsonl \
  --per-device-train-batch-size "${PER_DEVICE_BS}" \
  --grad-accum "${GRAD_ACCUM}" \
  --epochs "${EPOCHS}"

SFT_INPUT_JSONL="data/hf_sft_selected.jsonl"

PYTHONPATH="$(pwd)" python -m src.data.prepare_sft_split \
  --input "${SFT_INPUT_JSONL}" \
  --train-out data/train_hf_sft.jsonl \
  --valid-out data/valid_hf_sft.jsonl \
  --seed 42

# --------------------------------------------------------------
# Optional: append user-provided local datasets AFTER HF balancing
#
# Configure in configs/sft_hf.yaml:
#   data:
#     extra_datasets:
#       - use: true
#         format: jsonl
#         train_path: data/my_x_train.jsonl
#         valid_path: data/my_x_valid.jsonl
# --------------------------------------------------------------
echo "INFO  Extra datasets feature removed; using only selected dataset source(s)."

# Ensure StructEval-T multi-format eval tasks exist ONLY when post-train eval is enabled.
#
# Hugging Face training itself does not require StructEval-T tasks. Downloading them
# here is wasted work if you are not running post-training evaluation.
NEED_POST_EVAL="$(python - <<'PY'
import yaml
cfg = yaml.safe_load(open('configs/sft_hf.yaml', 'r', encoding='utf-8'))
print('1' if bool(((cfg.get('eval') or {}).get('run_eval_after_train', False))) else '0')
PY
)"

if [ "${NEED_POST_EVAL}" = "1" ]; then
  if [ ! -f data/structeval_text_all.json ]; then
    echo "INFO  Downloading StructEval-T eval tasks (JSON/YAML/TOML/XML/CSV) to data/structeval_text_all.json ..."
    bash scripts/download_structeval_text_all.sh test data/structeval_text_all.json "JSON,YAML,TOML,XML,CSV"
  fi
else
  echo "INFO  Post-train eval disabled (configs/sft_hf.yaml: eval.run_eval_after_train=false). Skip StructEval-T download."
fi

PYTHONPATH="$(pwd)" python train.py sft --config configs/sft_hf.yaml

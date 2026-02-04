"""train.py — Main entrypoint (readable orchestration)

This file is intentionally **simple and explanatory**:
- Only imports top-level functions from `src/`
- Describes *what each step does* (data loading, model setup, training, evaluation)
- Delegates detailed logic to modules under `src/`

Usage examples:
    # Stage A: supervised fine-tuning
    python train.py sft   --config configs/sft.yaml  --run-name baseline_sft

    # Stage B: GRPO RL fine-tuning
    python train.py grpo  --config configs/grpo.yaml --run-name baseline_grpo

    # Evaluation (model inference + StructEval-T scoring)
    python train.py eval  --config configs/eval.yaml

    # Smoke test (no model): offline StructEval-T scoring of a dataset file
    python train.py score --input data/valid_structeval_t.json --output outputs/smoke_scored.json

    # Hyperparameter tuning harness (dry-run default)
    python train.py tune  --base-config configs/sft.yaml --space configs/search_space.yaml --trials 20

    # Compare run summaries and generate plots
    python train.py plot
"""

from __future__ import annotations

import sys
from pathlib import Path

# 1) CLI: parse args and dispatch to subcommands.
#    We keep this in `src/cli.py` so that:
#    - `train.py` stays readable
#    - unit tests can call `src.cli.main()` directly
from src.cli import main as cli_main


def _ensure_repo_root() -> None:
    """Ensure current working dir is repo root for relative paths.

    This helps when running from IDEs or notebooks.
    """
    root = Path(__file__).resolve().parent
    if Path.cwd() != root:
        # Not changing cwd automatically to avoid surprises; just warn.
        # You can `cd` to repo root before running.
        pass


def main(argv: list[str] | None = None) -> int:
    """Program entry.

    What happens at runtime (high-level):
    - Parse command line args (subcommand: sft/grpo/eval/score/tune/plot)
    - Load YAML config
    - Create model+tokenizer (Qwen3-4B-Instruct-2507) + LoRA adapters
    - Load dataset and format prompts for JSON-only output
    - Run training (SFT or GRPO RL)
    - Save checkpoints and write evaluation metrics
    """
    _ensure_repo_root()
    return cli_main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line interface for training, tuning, and evaluation.

Subcommands:
- sft   : Supervised fine-tuning (Stage A)
- grpo  : RL fine-tuning with GRPO (Stage B)
- eval  : Run evaluation only (model inference + scoring)
- score : Offline StructEval-T scoring (smoke test without model)
- tune  : Hyperparameter tuning harness (dry-run by default)
- plot  : Plot and compare run summaries

Design note:
This file keeps imports **lazy** so that smoke tests (e.g. `score`) can run
without installing GPU-heavy deps such as `datasets`, `torch`, etc.
"""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sft = sub.add_parser("sft", help="Run supervised fine-tuning")
    p_sft.add_argument("--config", required=True, help="Path to SFT YAML config")
    p_sft.add_argument("--run-name", default=None, help="Optional run name (saved under runs/)")
    p_sft.add_argument("--train-path", default=None, help="Override dataset.train_path (SFT JSONL)")
    p_sft.add_argument("--valid-path", default=None, help="Override dataset.valid_path (SFT JSONL)")
    p_sft.add_argument(
        "--eval-tasks-path",
        default=None,
        help="Override eval task file used for post-train evaluation (if enabled in config)",
    )

    p_grpo = sub.add_parser("grpo", help="Run GRPO RL fine-tuning")
    p_grpo.add_argument("--config", required=True, help="Path to GRPO YAML config")
    p_grpo.add_argument("--run-name", default=None, help="Optional run name (saved under runs/)")
    p_grpo.add_argument("--train-path", default=None, help="Override dataset.train_path (StructEval JSON array)")
    p_grpo.add_argument("--valid-path", default=None, help="Override dataset.valid_path (optional; reserved)")
    p_grpo.add_argument(
        "--eval-tasks-path",
        default=None,
        help="Override eval task file used for post-train evaluation (if enabled in config)",
    )

    p_eval = sub.add_parser("eval", help="Run evaluation only")
    p_eval.add_argument("--config", required=True, help="Path to eval YAML config")
    p_eval.add_argument(
        "--model-path",
        default=None,
        help="Override model path for evaluation (useful for post-train eval of a saved checkpoint)",
    )
    p_eval.add_argument("--limit", type=int, default=None, help="Max number of eval items (default from config; 0=all)")
    p_eval.add_argument("--seed", type=int, default=None, help="Shuffle seed for eval item sampling")
    p_eval.add_argument(
        "--eval-tasks-path",
        default=None,
        help="Override eval task file (sets EVAL_TASKS_PATH under the hood)",
    )

    p_score = sub.add_parser("score", help="Offline StructEval-T scoring (no model)")
    p_score.add_argument("--input", required=True, help="StructEval-style dataset JSON")
    p_score.add_argument("--output", required=True, help="Path to write scored JSON")
    p_score.add_argument("--generation-field", default="generation", help="Field containing generations")
    p_score.add_argument("--limit", type=int, default=10, help="Max number of items to score (0=all)")
    p_score.add_argument("--seed", type=int, default=None, help="Shuffle seed for scoring item sampling")

    p_tune = sub.add_parser("tune", help="Hyperparameter tuning harness (dry-run default)")
    p_tune.add_argument("--base-config", required=True, help="Base YAML config (sft/grpo)")
    p_tune.add_argument("--space", required=True, help="Search space YAML")
    p_tune.add_argument("--trials", type=int, default=10)
    p_tune.add_argument("--stage", choices=["sft", "grpo"], default="sft")
    p_tune.add_argument("--no-dry-run", action="store_true")

    p_plot = sub.add_parser("plot", help="Plot and compare run summaries")
    p_plot.add_argument("--runs-dir", default="runs")
    p_plot.add_argument("--out-dir", default="runs/plots")

    return p


def main(argv: list[str]) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "sft":
        from src.train_sft import run_sft
        run_sft(
            args.config,
            run_name=args.run_name,
            train_path_override=args.train_path,
            valid_path_override=args.valid_path,
            eval_tasks_path_override=args.eval_tasks_path,
        )
        return 0
    if args.cmd == "grpo":
        from src.train_grpo import run_grpo
        run_grpo(
            args.config,
            run_name=args.run_name,
            train_path_override=args.train_path,
            valid_path_override=args.valid_path,
            eval_tasks_path_override=args.eval_tasks_path,
        )
        return 0
    if args.cmd == "eval":
        from src.eval.run_eval import run_eval
        if args.eval_tasks_path:
            os.environ["EVAL_TASKS_PATH"] = str(args.eval_tasks_path)
        run_eval(args.config, override_model_path=args.model_path, limit=args.limit, seed=args.seed)
        return 0
    if args.cmd == "score":
        from src.eval.score_structeval import score_structeval_dataset
        score_structeval_dataset(
            input_path=args.input,
            output_path=args.output,
            generation_field=args.generation_field,
            limit=args.limit,
            seed=args.seed,
        )
        return 0
    if args.cmd == "tune":
        from src.experiments.tune import run_tuning
        run_tuning(
            base_config_path=args.base_config,
            search_space_path=args.space,
            trials=args.trials,
            stage=args.stage,
            dry_run=not args.no_dry_run,
        )
        return 0
    if args.cmd == "plot":
        from src.experiments.plot import plot_run_summaries
        plot_run_summaries(runs_dir=args.runs_dir, out_dir=args.out_dir)
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.run_manager import start_run, save_json
from src.eval.score_structeval import score_structeval_dataset
from src.utils.config import load_yaml, save_yaml, apply_overrides


def _sample(space: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Randomly sample one configuration from a simple search space.

    Search space YAML format example:
      sampling:
        seed: 123
      params:
        trainer.learning_rate: {type: log_uniform, low: 1e-6, high: 5e-4}
        trainer.num_train_epochs: {type: choice, values: [1, 2, 3]}
        prompting.json_only: {type: choice, values: [true, false]}
    """
    out: dict[str, Any] = {}
    params = space.get("params", {})
    for k, spec in params.items():
        t = spec.get("type")
        if t == "choice":
            out[k] = rng.choice(spec["values"])
        elif t == "uniform":
            out[k] = rng.uniform(float(spec["low"]), float(spec["high"]))
        elif t == "log_uniform":
            lo = float(spec["low"])
            hi = float(spec["high"])
            # sample in log-space
            import math
            x = rng.uniform(math.log(lo), math.log(hi))
            out[k] = float(math.exp(x))
        elif t == "int":
            out[k] = rng.randint(int(spec["low"]), int(spec["high"]))
        else:
            raise ValueError(f"Unknown param type: {t} for {k}")
    return out


def run_tuning(
    base_config_path: str,
    search_space_path: str,
    *,
    trials: int = 10,
    stage: str = "sft",
    dry_run: bool = True,
) -> str:
    """Hyperparameter tuning harness.

    Why dry_run default True?
    - You may not have the real dataset yet.
    - This harness can still validate: config overrides, run directories,
      offline StructEval-T scoring, and plotting.

    Set dry_run=False once GPU training is available, by calling your training
    entrypoint inside this loop.
    """
    base_cfg = load_yaml(base_config_path)
    space = load_yaml(search_space_path)
    seed = int(space.get("sampling", {}).get("seed", 123))
    rng = random.Random(seed)

    best = None
    best_score = -1.0

    for t in range(trials):
        overrides = _sample(space, rng)

        rc = start_run(stage=f"tune_{stage}", run_name=f"trial{t:02d}")
        cfg_t = copy.deepcopy(base_cfg)
        cfg_t = apply_overrides(cfg_t, overrides)

        save_yaml(rc.run_dir / "config_resolved.yaml", cfg_t)
        save_json(rc.run_dir / "overrides.json", overrides)

        # Dry-run: offline scoring of a dataset file without running a model
        # Real run: train -> generate -> evaluate
        eval_out = rc.run_dir / "eval_scored.json"
        summary = score_structeval_dataset(
            input_path=cfg_t["dataset"]["valid_path"],
            output_path=str(eval_out),
            generation_field="generation",
        )

        score = float(summary.get("avg_final_eval_score", 0.0))
        if score > best_score:
            best_score = score
            best = {"run_dir": str(rc.run_dir), "overrides": overrides, "score": score}

    if best is None:
        best = {"score": 0.0}
    Path("runs/tuning_best.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(Path("runs/tuning_best.json"))

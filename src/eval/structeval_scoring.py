"""StructEval‑T scoring helpers.

Kept in a small module so it can be imported without heavyweight deps
(`transformers`, `torch`, ...). This enables light unit tests for scoring logic
and ensures `run_eval` and GRPO reward shaping share the same implementation.
"""

from __future__ import annotations

from typing import Any, Dict

from src.structeval_t.scorer import eval_structeval_t
from src.rl.rewards import compute_reward_components


def structeval_t_score(task: Dict[str, Any], generation: str) -> Dict[str, Any]:
    """Compute StructEval‑T scores for a single task.

    The task is expected to provide:
      - raw_output_metric: list[str]
      - output_type: str (JSON/YAML/TOML/XML/CSV)

    Returns a dict with the fields documented in `docs/api/eval.md`.
    """

    paths = task.get("raw_output_metric") or []
    if not isinstance(paths, list):
        paths = []
    paths = [p for p in paths if isinstance(p, str)]

    output_type = str(task.get("output_type") or "JSON").strip().upper()

    r = eval_structeval_t(generation, paths, output_type=output_type)

    # Extra deterministic diagnostics shared with GRPO reward code.
    # - parse / only / extraneous: helps debug format-only behavior for YAML/TOML/XML/CSV.
    # - match / match_soft: available when task includes `reference_output` (e.g., HF-imported tasks).
    comps = compute_reward_components(
        generation,
        output_type=output_type,
        reference_output=(task.get("reference_output") if isinstance(task, dict) else None),
    )

    return {
        "render_score": float(r.syntax_score),
        "key_validation_score": float(r.key_validation_score),
        "raw_output_score": float(r.raw_output_score),
        "raw_output_eval": list(r.raw_output_eval),
        "final_eval_score": float(r.final_eval_score),
        "output_type": output_type,
        # Deterministic format diagnostics (shared with GRPO reward).
        "parse": float(comps.get("parse", 0.0)),
        "parse_best_effort": float(comps.get("parse_best_effort", 0.0)),
        "only": float(comps.get("only", 0.0)),
        "extraneous": float(comps.get("extraneous", 0.0)),
        # Optional gold matching (HF tasks).
        "match": float(comps.get("match", 0.0)),
        "match_soft": float(comps.get("match_soft", 0.0)),
    }

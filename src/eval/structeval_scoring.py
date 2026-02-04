"""StructEval‑T scoring helpers.

Kept in a small module so it can be imported without heavyweight deps
(`transformers`, `torch`, ...). This enables light unit tests for scoring logic
and ensures `run_eval` and GRPO reward shaping share the same implementation.
"""

from __future__ import annotations

from typing import Any, Dict

from src.structeval_t.scorer import eval_structeval_t


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

    return {
        "render_score": float(r.syntax_score),
        "key_validation_score": float(r.key_validation_score),
        "raw_output_score": float(r.raw_output_score),
        "raw_output_eval": list(r.raw_output_eval),
        "final_eval_score": float(r.final_eval_score),
        "output_type": output_type,
    }

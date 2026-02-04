from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.structeval_t.scorer import eval_structeval_t


def score_structeval_dataset(
    input_path: str,
    output_path: str,
    *,
    generation_field: str = "generation",
    limit: int = 10,
    seed: int | None = None,
) -> dict[str, Any]:
    """Score a StructEval-style dataset file offline.

    This is a smoke-test utility: it does NOT run a model.
    It reads a StructEval-style dataset JSON (list of task objects), takes
    `generation_field` as model output, and computes StructEval-T scores (JSON/YAML/TOML/XML/CSV).

    Note:
        The public StructEval(-T) dataset does **not** include `gold_output` for
        JSON tasks. Therefore this scorer does not use any gold fallback.
    """

    tasks = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("StructEval dataset must be a JSON list")
    # Apply scoring sampling (shuffle with seed, then take `limit`).
    if seed is not None:
        rng = random.Random(int(seed))
        idxs = list(range(len(tasks)))
        rng.shuffle(idxs)
        tasks = [tasks[i] for i in idxs]

    if int(limit) > 0:
        tasks = tasks[: int(limit)]



    scored: list[dict[str, Any]] = []
    final_scores: list[float] = []

    for t in tasks:
        raw = t.get("raw_output_metric") or []
        gen = t.get(generation_field) or ""

        if not isinstance(raw, list):
            raw = []

        output_type = str(t.get("output_type") or "JSON").strip().upper()
        res = eval_structeval_t(str(gen), [str(x) for x in raw], output_type=output_type)

        out = dict(t)
        out["render_score"] = res.syntax_score
        out["key_validation_score"] = res.key_validation_score
        out["raw_output_eval"] = res.raw_output_eval
        out["raw_output_score"] = res.raw_output_score
        out["final_eval_score"] = res.final_eval_score
        out["output_type"] = output_type

        scored.append(out)
        final_scores.append(res.final_eval_score)

    summary = {
        "n": len(scored),
        "avg_final_eval_score": (sum(final_scores) / len(final_scores)) if final_scores else 0.0,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(Path(output_path).with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

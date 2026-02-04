from __future__ import annotations

from typing import Any, Dict


JUDGE_SYSTEM = """You are a strict evaluator for JSON-generation tasks.

You will be given:
- a task prompt (query)
- optional feature requirements text
- optional key-path list (`raw_output_metric`) which indicates required paths
- a model-generated output (candidate)

Your job: evaluate whether the candidate follows the instruction and satisfies
the structural constraints described in the prompt.

Return ONLY valid JSON matching this schema:
{
  "score": number,              # 0..1
  "passed": boolean,
  "breakdown": {
    "json_parse": 0|1,
    "path_coverage": number,    # 0..1 (fraction of required paths present)
    "constraint_following": number  # 0..1 (type/length rules, etc.)
  },
  "failed_checks": [string],
  "notes": string
}

Rules:
- score must be 0..1.
- If candidate is not parseable as JSON, json_parse=0 and score MUST be 0.
- Do NOT compare against any reference "gold" output.
"""


def build_judge_input(task: Dict[str, Any], generation: str) -> str:
    query = task.get("query", "")
    req = task.get("feature_requirements", "")
    raw_metric = task.get("raw_output_metric", [])
    return (
        "TASK PROMPT:\n"
        f"{query}\n\n"
        "FEATURE REQUIREMENTS (if provided):\n"
        f"{req}\n\n"
        "EXPECTED KEY PATHS (raw_output_metric, if provided):\n"
        f"{raw_metric}\n\n"
        "CANDIDATE OUTPUT (model generation):\n"
        f"{generation}\n"
    )

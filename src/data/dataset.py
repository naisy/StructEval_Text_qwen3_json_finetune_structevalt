from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logging import warn
from src.data.format_rules import FORMAT_RULES


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                warn(f"JSON decode error in {path}:{i}: {e}")
                raise
    return items


def load_structeval_json(path: str | Path) -> list[dict[str, Any]]:
    """Load StructEval-style dataset JSON (array of task objects)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"StructEval JSON must be a list, got {type(data)}")
    return data


def load_dataset_any(path: str | Path, fmt: str) -> list[dict[str, Any]]:
    """Load dataset in either JSONL or StructEval JSON array format."""
    fmt = fmt.lower().strip()
    if fmt == "jsonl":
        return load_jsonl(path)
    if fmt in {"structeval_json", "structeval-t", "structeval_t"}:
        return load_structeval_json(path)
    raise ValueError(f"Unknown dataset format: {fmt}")


def build_prompt(example: dict[str, Any], cfg: dict) -> str:
    """Build an instruction prompt.

    - For StructEval-style data, use `example['query']` as-is (already composed).
    - Otherwise use {instruction, requirements} style fields.

    This function also injects lightweight, format-specific constraints into the
    system message based on `example['output_type']` (JSON/YAML/TOML/XML/CSV).
    This is important for evaluation and GRPO where rewards depend on
    deterministic parsers.
    """
    sys_msg = cfg["prompting"]["system"].strip()

    # Inject format-specific rules (JSON/YAML/TOML/XML/CSV) to prevent
    # cross-format leakage (e.g., JSON-style ':' in TOML, trailing commas).
    # This is important for BOTH eval and GRPO: deterministic parsers drive
    # the syntax score / reward.
    ot = str(example.get("output_type") or "").strip().upper()
    extra = FORMAT_RULES.get(ot)
    if extra:
        sys_msg = f"{sys_msg}\n\n{extra}".strip()

    user_prefix = cfg["prompting"].get("user_prefix", "")
    schema_prefix = cfg["prompting"].get("schema_prefix", "")
    out_prefix = cfg["prompting"].get("output_prefix", "")

    if isinstance(example.get("query"), str):
        instruction = example["query"].strip()
        req_text = (example.get("feature_requirements") or "").strip()
    else:
        instruction = (example.get("instruction") or "").strip()
        req = example.get("requirements")
        if req is None and cfg["prompting"].get("include_requirements_in_prompt", True):
            req = example.get("feature_requirements") or example.get("constraints")

        if isinstance(req, list):
            req_text = "\n".join([f"{i+1}. {x}" for i, x in enumerate(req)])
        else:
            req_text = (req or "").strip()

    parts: list[str] = []
    parts.append(f"<|system|>\n{sys_msg}")

    if req_text and schema_prefix:
        user_block = f"{user_prefix}{instruction}\n\n{schema_prefix}{req_text}\n\n{out_prefix}"
    elif req_text:
        user_block = f"{user_prefix}{instruction}\n\n{req_text}\n\n{out_prefix}"
    else:
        user_block = f"{user_prefix}{instruction}\n\n{out_prefix}"

    parts.append(f"<|user|>\n{user_block.strip()}")

    parts.append("<|assistant|>\n")
    return "\n".join(parts)

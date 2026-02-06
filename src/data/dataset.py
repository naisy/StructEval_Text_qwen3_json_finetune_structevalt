from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

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


def build_messages(example: dict[str, Any], cfg: dict) -> list[dict[str, str]]:
    """Build chat messages for one example.

    - For StructEval-style data, use `example['query']` as-is (already composed).
    - Otherwise use {instruction, requirements} style fields.

    IMPORTANT (contest alignment)
    -----------------------------
    The contest inference code constructs prompts as:

        messages = [{"role": "user", "content": query}]
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    If you train with a different message structure (extra system rules, extra
    prefixes, different sentinels), the model can look great in offline eval
    but fail badly in the contest setting.

    Therefore this function supports two prompting modes:
      - prompting.mode == "contest": user-only (query as-is), NO system message
      - default: include system message + optional format rules
    """

    prompting = cfg.get("prompting", {}) or {}
    mode = str(prompting.get("mode") or "default").strip().lower()

    sys_msg = str(prompting.get("system") or "").strip()

    if mode != "contest":
        # Inject format-specific rules (JSON/YAML/TOML/XML/CSV) to prevent
        # cross-format leakage (e.g., JSON-style ':' in TOML, trailing commas).
        # This is important for BOTH eval and GRPO: deterministic parsers drive
        # the syntax score / reward.
        ot = str(example.get("output_type") or "").strip().upper()
        extra = FORMAT_RULES.get(ot)
        if extra:
            sys_msg = f"{sys_msg}\n\n{extra}".strip()

    user_prefix = str(prompting.get("user_prefix", ""))
    schema_prefix = str(prompting.get("schema_prefix", ""))
    out_prefix = str(prompting.get("output_prefix", ""))

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

    # Contest mode: user-only, query as-is.
    if mode == "contest":
        return [{"role": "user", "content": instruction}]

    # Default mode: system + (optional) requirement blocks + output prefix.
    if req_text and schema_prefix:
        user_block = f"{user_prefix}{instruction}\n\n{schema_prefix}{req_text}\n\n{out_prefix}"
    elif req_text:
        user_block = f"{user_prefix}{instruction}\n\n{req_text}\n\n{out_prefix}"
    else:
        user_block = f"{user_prefix}{instruction}\n\n{out_prefix}"

    msgs: list[dict[str, str]] = []
    if sys_msg:
        msgs.append({"role": "system", "content": sys_msg})
    msgs.append({"role": "user", "content": user_block.strip()})
    return msgs


def build_prompt(example: dict[str, Any], cfg: dict, *, tokenizer: Optional[Any] = None) -> str:
    """Build the final prompt string.

    If prompting.use_chat_template is true, we use tokenizer.apply_chat_template
    so training/eval prompts match contest inference.
    Otherwise we fall back to the legacy manual sentinel framing.
    """
    prompting = cfg.get("prompting", {}) or {}
    use_chat = bool(prompting.get("use_chat_template", True))
    msgs = build_messages(example, cfg)

    if use_chat:
        if tokenizer is None:
            raise ValueError("build_prompt requires tokenizer when use_chat_template=true")
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Legacy manual framing (kept for backward compatibility / debugging).
    parts: list[str] = []
    for m in msgs:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)

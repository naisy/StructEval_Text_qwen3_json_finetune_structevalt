from __future__ import annotations

"""
JSON-like TOML fixer.

Motivation
----------
Some HF datasets (notably u-10bei/*) label outputs as TOML, but a small portion
of examples contain a JSON object/array as the assistant "Output:" payload.
Those examples increase format freedom and harm strict TOML accuracy.

This module detects such "JSON-like TOML" payloads and deterministically
converts them into TOML (then callers can canonicalize via toml_canonical).

Design
------
- Detection is intentionally conservative:
  - Only treat as JSON-like when the trimmed payload starts with '{' or '['
    AND we can parse it as JSON (best-effort extraction is handled upstream).
- Conversion policy:
  - JSON null has no TOML equivalent; we map it to empty string "".
  - If the JSON root is a dict => emit as TOML root table.
  - Otherwise wrap into {"root": <value>} to keep TOML valid.

This is meant for *training-data normalization*, not for perfect semantic
preservation of every exotic edge case.
"""

import json
from typing import Any, Tuple

from src.data.toml_canonical import dumps_toml_canonical


def _normalize_json(obj: Any) -> Any:
    # TOML has no null; map to empty string.
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {k: _normalize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_json(v) for v in obj]
    return obj


def looks_like_json_payload(text: str) -> bool:
    """Heuristically detect JSON payloads embedded in TOML-labeled examples.

    Important: TOML tables/arrays-of-tables also start with '[' (e.g. "[public]",
    "[[items]]"). Those are *valid TOML* and must NOT be treated as JSON.

    This function is intentionally conservative and is only used as a fast
    pre-filter. Callers should still require a successful JSON->TOML conversion
    before acting on the result.
    """

    t = (text or "").lstrip()
    if not t:
        return False

    # TOML table headers: [table] / [[array_of_tables]] (possibly with dotted paths).
    if t[0] == "[":
        first_line = t.splitlines()[0].strip()
        if first_line.startswith("[[") and first_line.endswith("]]"):
            inner = first_line[2:-2].strip()
            # JSON arrays almost always contain ',', '{', '}', ':', or quotes near the top.
            if inner and not any(ch in inner for ch in ",:{}\"'"):
                return False
        if first_line.startswith("[") and first_line.endswith("]"):
            inner = first_line[1:-1].strip()
            if inner and not any(ch in inner for ch in ",:{}\"'"):
                return False
        return True

    if t[0] == "{":
        # TOML inline tables are '{ key = value }' and may appear in some datasets.
        # If the first line looks like TOML (has '=' but no ':'), do not treat as JSON.
        first_line = t.splitlines()[0]
        if "=" in first_line and ":" not in first_line:
            return False
        return True

    return False


def convert_json_payload_to_toml(text: str) -> Tuple[bool, str, str | None]:
    """
    Convert a JSON object/array/scalar string into TOML text.

    Returns (ok, toml_text, err).
    """
    try:
        obj = json.loads((text or "").strip())
    except Exception as e:
        return False, "", str(e)

    obj = _normalize_json(obj)

    if isinstance(obj, dict):
        try:
            return True, dumps_toml_canonical(obj), None
        except Exception as e:
            return False, "", str(e)

    # Non-dict root: wrap so the output is valid TOML.
    try:
        return True, dumps_toml_canonical({"root": obj}), None
    except Exception as e:
        return False, "", str(e)

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
    t = (text or "").lstrip()
    return bool(t) and t[0] in "{["


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

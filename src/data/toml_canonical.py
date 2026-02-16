from __future__ import annotations

"""TOML canonicalization utilities.

Motivation
----------
Some training datasets contain TOML outputs that are syntactically valid but
use inconsistent ordering / table emission patterns. Mixing those styles can
cause the model to learn a "hybrid" TOML that drifts away from the project's
preferred canonical form.

This module provides:
- Deterministic TOML dumping (canonical form)
- A predicate to check whether a TOML string is already canonical

Notes
-----
- We intentionally *sort keys* at every mapping level so canonicalization is
  stable regardless of the original key order.
- The canonical form is designed to be readable and deterministic; it is not
  meant to preserve comments or original formatting.
"""

import math
from typing import Any, Iterable

import tomllib


def _escape_basic_str(s: str) -> str:
    # Minimal TOML basic-string escaping.
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\b", "\\b")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\f", "\\f")
        .replace("\r", "\\r")
    )


def _format_value(v: Any) -> str:
    if v is None:
        # TOML has no null; represent as empty string to avoid crashes.
        # Callers should avoid producing None.
        return '""'

    if isinstance(v, bool):
        return "true" if v else "false"

    if isinstance(v, int):
        return str(v)

    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        # Use repr for a stable round-trippable float string.
        return repr(v)

    if isinstance(v, str):
        return f'"{_escape_basic_str(v)}"'

    # tomllib returns datetime/date/time objects for those TOML types.
    # They already have TOML-compatible ISO string repr.
    if hasattr(v, "isoformat"):
        return v.isoformat()

    if isinstance(v, list):
        inner = ", ".join(_format_value(x) for x in v)
        return f"[{inner}]"

    # Fallback: string
    return f'"{_escape_basic_str(str(v))}"'


def dumps_toml_canonical(obj: Any) -> str:
    """Dump parsed TOML object to a deterministic canonical TOML string."""

    if not isinstance(obj, dict):
        raise TypeError(f"TOML root must be a dict, got: {type(obj)}")

    out_lines: list[str] = []

    # NOTE: When we emit a `[table]` or `[[array-of-tables]]` header, keys that
    # follow MUST be written *relative* to that table (e.g., `height = 1` under
    # `[dimensions]`), not as dotted keys like `dimensions.height = 1`.
    #
    # Earlier versions of this module mistakenly prefixed scalar keys with the
    # full table path while also emitting the table header, which changes the
    # parsed structure (it nests tables twice) and defeats the purpose of
    # canonicalization.
    def emit_kv(key: str, value: Any) -> None:
        out_lines.append(f"{key} = {_format_value(value)}")

    def emit_table_header(path: Iterable[str]) -> None:
        p = ".".join(path)
        out_lines.append(f"[{p}]")

    def emit_array_table_header(path: Iterable[str]) -> None:
        p = ".".join(path)
        out_lines.append(f"[[{p}]]")

    def is_array_of_tables(v: Any) -> bool:
        return isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v)

    def walk(prefix: list[str], d: dict[str, Any]) -> None:
        # Emit scalar keys first, then tables/array-of-tables.
        scalar_keys: list[str] = []
        table_keys: list[str] = []
        aot_keys: list[str] = []

        for k in d.keys():
            v = d[k]
            if isinstance(v, dict):
                table_keys.append(k)
            elif is_array_of_tables(v):
                aot_keys.append(k)
            else:
                scalar_keys.append(k)

        for k in sorted(scalar_keys):
            # Under a table header we emit keys relative to the table.
            # At the root (no table header) we also emit bare keys.
            emit_kv(k, d[k])

        # Blank line between scalar section and tables when both exist.
        if scalar_keys and (table_keys or aot_keys):
            out_lines.append("")

        for k in sorted(table_keys):
            sub = d[k]
            assert isinstance(sub, dict)
            path = prefix + [k]
            emit_table_header(path)
            walk(path, sub)
            out_lines.append("")

        for k in sorted(aot_keys):
            arr = d[k]
            assert isinstance(arr, list)
            path = prefix + [k]
            for idx, item in enumerate(arr):
                assert isinstance(item, dict)
                emit_array_table_header(path)
                walk(path, item)
                # blank line between array table items
                if idx != len(arr) - 1:
                    out_lines.append("")
            out_lines.append("")

        # Trim trailing blank lines inserted by callers.
        while out_lines and out_lines[-1] == "":
            out_lines.pop()

    walk([], obj)

    # Ensure trailing newline for nicer diffs.
    return "\n".join(out_lines).rstrip() + "\n"


def canonicalize_toml_text(text: str) -> tuple[bool, bool, str, str | None]:
    """Return (ok, already_canonical, canonical_text, err)."""
    try:
        obj = tomllib.loads((text or "").strip())
        canonical = dumps_toml_canonical(obj)
        already = canonical.strip() == (text or "").strip()
        return True, already, canonical, None
    except Exception as e:
        return False, False, "", str(e)


def toml_is_canonical(text: str) -> bool:
    ok, already, _canon, _err = canonicalize_toml_text(text)
    return bool(ok and already)

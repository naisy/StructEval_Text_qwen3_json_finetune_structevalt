from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import xml.etree.ElementTree as ET

from src.data.validators import (
    parse_json,
    parse_json_best_effort,
    parse_yaml,
    parse_toml,
    parse_xml,
    parse_csv,
)
from src.structeval_t.paths import exists_path


@dataclass
class StructEvalTResult:
    """StructEval‑T result container."""

    syntax_score: float  # 0/1
    key_validation_score: float  # [0,1]
    raw_output_eval: list[bool]
    raw_output_score: float  # mean(raw_output_eval)
    final_eval_score: float  # 0.2*syntax + 0.8*key_score


def _parse_json_maybe(text: str) -> tuple[bool, Any | None, bool]:
    """Parse JSON; fall back to best-effort extraction.

    Returns: (ok_any, obj, ok_strict)
    """
    ok, obj, _ = parse_json(text)
    if ok:
        return True, obj, True
    ok2, obj2, _, _ = parse_json_best_effort(text)
    return ok2, obj2, False


def _xml_elem_to_obj(elem: ET.Element) -> Any:
    """Convert an XML Element into a JSON-like Python object.

    Design goal:
    - Work with StructEval's `raw_output_metric` key-path validator which expects
      dict/list primitives.

    Representation:
    - Each element becomes a dict.
    - Attributes are mapped under keys prefixed with "@".
    - Text content (trimmed) is stored under "#text" if non-empty.
    - Children are grouped by tag; repeated tags become lists.

    Note: This is a heuristic. StructEval's raw_output_metric for XML tasks is
    typically key-path oriented, so this grouped form is the most compatible.
    """
    node: dict[str, Any] = {}

    # attributes
    for k, v in (elem.attrib or {}).items():
        node[f"@{k}"] = v

    # children
    by_tag: dict[str, list[Any]] = {}
    for c in list(elem):
        by_tag.setdefault(c.tag, []).append(_xml_elem_to_obj(c))

    for tag, vals in by_tag.items():
        if len(vals) == 1:
            node[tag] = vals[0]
        else:
            node[tag] = vals

    # text
    txt = (elem.text or "").strip()
    if txt:
        if node:
            node["#text"] = txt
        else:
            # leaf text element: use raw string for simplicity
            return txt

    return node


def _csv_rows_to_obj(rows: list[list[str]]) -> Any:
    """Convert CSV rows to a JSON-like object.

    Heuristic:
    - If the first row looks like a header, return {"rows": [ {col: val, ...}, ... ]}
    - Otherwise return {"rows": rows}

    Header detection is intentionally permissive (any non-empty header cell).
    """
    if not rows:
        return {"rows": []}

    header = rows[0]
    if any(str(h).strip() for h in header) and len(rows) >= 2:
        cols = [str(h).strip() for h in header]
        out_rows: list[dict[str, str]] = []
        for r in rows[1:]:
            # pad or truncate
            rr = list(r) + [""] * max(0, len(cols) - len(r))
            rr = rr[: len(cols)]
            out_rows.append({cols[i] if cols[i] else f"col_{i}": str(rr[i]) for i in range(len(cols))})
        return {"rows": out_rows}

    return {"rows": rows}


def _parse_by_output_type(generation: str, output_type: str) -> tuple[bool, Any | None, bool]:
    """Return (ok_any, obj, ok_strict).

    For JSON, ok_any may be True even if only best-effort parsing succeeds.
    For other formats, ok_any == ok_strict.
    """
    ot = (output_type or "JSON").strip().upper()

    if ot == "JSON":
        return _parse_json_maybe(generation)

    if ot == "YAML":
        ok, obj, _ = parse_yaml(generation)
        # YAML is extremely permissive: plain prose parses as a scalar string.
        # For our structured-output tasks, treat non-(dict/list) as a syntax failure.
        if ok and not isinstance(obj, (dict, list)):
            return False, None, False
        return ok, obj, ok

    if ot == "TOML":
        ok, obj, _ = parse_toml(generation)
        return ok, obj, ok

    if ot == "XML":
        ok, root, _ = parse_xml(generation)
        if not ok or root is None:
            return False, None, False
        # Wrap root tag so key paths can start from it.
        obj = {root.tag: _xml_elem_to_obj(root)}
        return True, obj, True

    if ot == "CSV":
        ok, rows, _ = parse_csv(generation)
        if not ok or rows is None:
            return False, None, False
        # CSV is also permissive: a single prose line becomes a 1x1 table.
        # Require at least 2 columns somewhere, and consistent column counts.
        max_cols = max((len(r) for r in rows), default=0)
        if max_cols < 2:
            return False, None, False
        # Strictness: all non-empty rows must match header width when a header exists.
        header_w = len(rows[0]) if rows else 0
        if header_w >= 2:
            for r in rows[1:]:
                if len(r) not in (0, header_w):
                    return False, None, False
        obj = _csv_rows_to_obj(rows)
        return True, obj, True

    # Unknown output_type: treat as JSON for backward compatibility
    return _parse_json_maybe(generation)


def eval_structeval_t(generation: str, raw_output_metric: list[str], output_type: str = "JSON") -> StructEvalTResult:
    """Evaluate a generation in the spirit of StructEval‑T.

    For StructEval‑T (text-only) tasks, the score is:
      final = 0.2 * syntax_score + 0.8 * key_validation_score

    - syntax_score is 1 if the output strictly parses as the requested format.
      (For JSON we also compute best-effort parsing for key checks.)
    - key_validation_score is the fraction of required key paths satisfied.

    Args:
        generation: Model output string.
        raw_output_metric: List of StructEval key paths.
        output_type: One of JSON/YAML/TOML/XML/CSV.
    """
    ok_any, obj, ok_strict = _parse_by_output_type(generation, output_type)
    syntax = 1.0 if ok_strict else 0.0

    raw_eval: list[bool] = []
    if ok_any and raw_output_metric:
        for m in raw_output_metric:
            try:
                raw_eval.append(exists_path(obj, m))
            except Exception:
                raw_eval.append(False)
    else:
        raw_eval = [False for _ in raw_output_metric]

    if raw_output_metric:
        raw_score = sum(raw_eval) / float(len(raw_output_metric))
    else:
        raw_score = 0.0

    key_score = raw_score
    final_score = 0.2 * syntax + 0.8 * key_score

    return StructEvalTResult(
        syntax_score=syntax,
        key_validation_score=key_score,
        raw_output_eval=raw_eval,
        raw_output_score=raw_score,
        final_eval_score=final_score,
    )


def eval_structeval_t_json(generation: str, raw_output_metric: list[str]) -> StructEvalTResult:
    """Backward-compatible JSON-only entry point."""
    return eval_structeval_t(generation, raw_output_metric, output_type="JSON")

from __future__ import annotations

from typing import Any, Callable, Tuple

from src.data import validators as V


def _norm_output_type(output_type: str | None) -> str:
    """Normalize output_type labels to a small canonical set."""
    t = (output_type or "JSON").strip().upper()
    if t in {"YML"}:
        return "YAML"
    if t in {"TSV"}:
        return "CSV"
    return t


def _get_parsers(output_type: str) -> tuple[
    Callable[[str], Tuple[bool, Any | None, str | None]],
    Callable[[str], Tuple[bool, Any | None, str | None]] | None,
    Callable[[str], bool],
]:
    """Return (strict_parser, best_effort_parser, only_checker) for a format."""
    t = _norm_output_type(output_type)

    if t == "JSON":
        return V.parse_json, V.parse_json_best_effort, V.is_json_only
    if t == "YAML":
        # YAML already tolerates whitespace/comments; we treat strict==best_effort.
        return V.parse_yaml, None, V.is_yaml_only
    if t == "TOML":
        return V.parse_toml, None, V.is_toml_only
    if t == "XML":
        return V.parse_xml, None, V.is_xml_only
    if t == "CSV":
        return V.parse_csv, None, V.is_csv_only

    # Default to JSON behavior for unknown labels.
    return V.parse_json, V.parse_json_best_effort, V.is_json_only


def compute_reward_components(
    completion: str,
    output_type: str | None = None,
    schema_validator=None,
    allowed_top_level: set[str] | None = None,
) -> dict[str, float]:
    """Compute reward components for a single model completion.

    Notes
    -----
    - GRPO is trained across multiple output formats (JSON/YAML/TOML/XML/CSV).
      This function picks the right deterministic parser based on `output_type`.
    - We keep legacy JSON-named fields (parse_json/json_only/...) for backward
      compatibility with older report code and configs.
    """
    t = _norm_output_type(output_type)
    strict_parse, best_effort_parse, is_only = _get_parsers(t)

    ok, obj, _ = strict_parse(completion)

    # Best-effort parsing is currently implemented only for JSON.
    ok_be, obj_be = False, None
    if best_effort_parse is not None:
        ok_be, obj_be, _err_be, _used = best_effort_parse(completion)
    else:
        ok_be, obj_be = ok, obj

    out: dict[str, float] = {}
    # canonical, type-agnostic components
    out["parse"] = 1.0 if ok else 0.0
    out["parse_best_effort"] = 1.0 if ok_be else 0.0
    out["only"] = 1.0 if is_only(completion) else 0.0

    # legacy/type-specific mirrors (useful for logging/config back-compat)
    tl = t.lower()
    out[f"parse_{tl}"] = out["parse"]
    out[f"{tl}_only"] = out["only"]
    # keep original JSON keys as aliases when t==JSON
    out["parse_json"] = out["parse"] if t == "JSON" else 0.0
    out["parse_json_best_effort"] = out["parse_best_effort"] if t == "JSON" else 0.0
    out["json_only"] = out["only"] if t == "JSON" else 0.0

    # Schema-based scoring currently applies only to JSON objects, and only when
    # a schema validator is provided.
    if t == "JSON" and ok and schema_validator is not None:
        schema_ok, _errs = V.validate_schema(obj, schema_validator)
        out["schema_valid"] = 1.0 if schema_ok else 0.0
    else:
        out["schema_valid"] = 0.0

    # Task-specific constraints example (JSON article_meta)
    # For strict parse failures but best-effort successes, evaluate constraints
    # on extracted JSON.
    obj_for_checks = obj if ok else (obj_be if ok_be else None)
    if t == "JSON" and obj_for_checks is not None:
        checks = V.check_task_constraints_article_meta(obj_for_checks)
        out["title_is_string"] = 1.0 if checks["title_is_string"] else 0.0
        out["authors_len_is_2"] = 1.0 if checks["authors_len_is_2"] else 0.0
        out["authors_items_ok"] = 1.0 if checks["authors_items_ok"] else 0.0
        out["publication_year_is_int"] = 1.0 if checks["publication_year_is_int"] else 0.0
        out["keywords_is_list_str"] = 1.0 if checks["keywords_is_list_str"] else 0.0
    else:
        out.update({
            "title_is_string": 0.0,
            "authors_len_is_2": 0.0,
            "authors_items_ok": 0.0,
            "publication_year_is_int": 0.0,
            "keywords_is_list_str": 0.0,
        })

    if t == "JSON" and obj_for_checks is not None and allowed_top_level is not None:
        extra = V.count_extra_keys(obj_for_checks, allowed_top_level)
        out["extra_keys"] = float(extra)
    else:
        out["extra_keys"] = 0.0

    return out


def _cfg_get_typed(w: dict[str, Any], t: str, base_key: str, default: float) -> float:
    """Look up config weights with format-specific override.

    Example:
      base_key='w_parse' => tries w_parse_json / w_parse_yaml ... then w_parse
    """
    tl = t.lower()
    for k in (f"{base_key}_{tl}", base_key):
        if k in w:
            return float(w[k])
    return float(default)


def _cfg_get_typed_penalty(w: dict[str, Any], t: str, base_key: str, default: float) -> float:
    tl = t.lower()
    # legacy pattern: p_json_only_fail, p_yaml_only_fail, ...
    legacy = f"{base_key}_{tl}_fail" if base_key.endswith("_only") else None
    if legacy and legacy in w:
        return float(w[legacy])
    for k in (f"{base_key}_{tl}", base_key):
        if k in w:
            return float(w[k])
    return float(default)


def combine_reward(components: dict[str, float], cfg: dict[str, Any], output_type: str | None = None) -> float:
    """Combine components into a scalar reward using weights."""
    t = _norm_output_type(output_type)
    w = cfg["reward"]
    r = 0.0

    # If both strict and best-effort parsing fail: hard penalty.
    if components.get("parse", 0.0) < 1.0 and components.get("parse_best_effort", 0.0) < 1.0:
        return float(w.get("p_parse_fail", -3.0))

    # If only best-effort succeeds: mild penalty (mainly for JSON).
    if components.get("parse", 0.0) < 1.0 and components.get("parse_best_effort", 0.0) >= 1.0:
        r += float(w.get("p_parse_best_effort", -1.5))

    # base positives (format-specific)
    r += _cfg_get_typed(w, t, "w_parse", 1.0) * components.get("parse", 0.0)
    r += _cfg_get_typed(w, t, "w_only", 0.5) * components.get("only", 0.0)

    # JSON-only optional schema / constraints
    r += float(w.get("w_schema_valid", 2.0)) * components.get("schema_valid", 0.0)
    r += float(w.get("w_exact_authors_len_2", 1.0)) * components.get("authors_len_is_2", 0.0)

    # mild penalties / constraints (JSON only)
    r += float(w.get("w_no_extra_keys", 0.0)) * (1.0 if components.get("extra_keys", 0.0) == 0 else 0.0)
    r += -0.1 * components.get("extra_keys", 0.0)

    # If format-only failed, apply penalty (format-specific override supported)
    if components.get("only", 0.0) < 1.0:
        r += _cfg_get_typed_penalty(w, t, "p_only", -1.0)

    return float(r)

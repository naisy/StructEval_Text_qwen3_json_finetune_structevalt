from __future__ import annotations

import difflib
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
        # YAML itself is permissive, but our tasks expect structured outputs.
        # Best-effort helps when the model appends chatter after a valid YAML block.
        return V.parse_yaml, V.parse_yaml_best_effort, V.is_yaml_only
    if t == "TOML":
        # TOML is strict. "Best-effort" prefix parsing can mask real syntax errors
        # (e.g., an unterminated array or inline-table) by truncating the output to
        # a parseable prefix. That weakens the reward signal and lets invalid TOML
        # survive GRPO when tasks don't provide ATTRIBUTES.
        #
        # Policy: require *strict* parse for TOML rewards.
        return V.parse_toml, None, V.is_toml_only
    if t == "XML":
        return V.parse_xml, V.parse_xml_best_effort, V.is_xml_only
    if t == "CSV":
        return V.parse_csv, V.parse_csv_best_effort, V.is_csv_only

    # Default to JSON behavior for unknown labels.
    return V.parse_json, V.parse_json_best_effort, V.is_json_only


def compute_reward_components(
    completion: str,
    output_type: str | None = None,
    schema_validator=None,
    allowed_top_level: set[str] | None = None,
    reference_output: str | None = None,
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

    # Extract the structured payload for scoring, but keep a signal about
    # whether the model emitted any extraneous wrapper text ("Sure!", markdown,
    # code fences, etc.).
    payload, has_extraneous = V.extract_payload_and_extraneous(completion, t)

    ok, obj, _ = strict_parse(payload)

    # Best-effort parsing: recover a parseable prefix when the model appends
    # extra chatter after a structured block (common during early GRPO).
    ok_be, obj_be = False, None
    if best_effort_parse is not None:
        ok_be, obj_be, _err_be, _used = best_effort_parse(payload)
    else:
        ok_be, obj_be = ok, obj

    out: dict[str, float] = {}
    # canonical, type-agnostic components
    out["parse"] = 1.0 if ok else 0.0
    out["parse_best_effort"] = 1.0 if ok_be else 0.0
    out["only"] = 1.0 if is_only(completion) else 0.0
    out["extraneous"] = 1.0 if has_extraneous else 0.0

    # --------------------------------------------------------------
    # Format-specific style signals (beyond syntax)
    #
    # Motivation:
    # - YAML is syntactically permissive; structurally correct outputs can still
    #   violate the project's canonical indentation/style rules.
    # - These are used as *additional* shaping signals during GRPO. They should
    #   not be relied on as the sole correctness criterion.
    # --------------------------------------------------------------
    if t == "YAML":
        out["yaml_indent_canonical"] = 1.0 if V.yaml_indent_is_canonical(payload, indent=2) else 0.0
        out["yaml_block_style"] = 1.0 if (not V.yaml_uses_flow_style(payload)) else 0.0
    else:
        out["yaml_indent_canonical"] = 0.0
        out["yaml_block_style"] = 0.0

    if t == "TOML":
        out["toml_canonical"] = 1.0 if V.toml_is_canonical(payload) else 0.0
        out["toml_canonical_soft"] = float(V.toml_canonical_similarity(payload))
    else:
        out["toml_canonical"] = 0.0
        out["toml_canonical_soft"] = 0.0

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
    # For strict formats (TOML/XML/CSV), do not allow best-effort parsing to
    # stand in for correctness in downstream checks (e.g., match_soft). This
    # prevents invalid outputs from earning reward via a truncated prefix.
    if t in {"TOML", "XML", "CSV"}:
        obj_for_checks = obj if ok else None
    else:
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

    # Optional gold matching (format-aware). This is especially useful for HF datasets
    # that don't include StructEval-T ATTRIBUTES blocks (raw_output_metric=[]).
    if reference_output is not None and str(reference_output).strip():
        ref_payload, _ = V.extract_payload_and_extraneous(str(reference_output), t)
        ok_ref, obj_ref, _ = strict_parse(ref_payload)
        # Strict match (exact object equality)
        if ok and ok_ref and obj is not None and obj_ref is not None:
            out["match"] = 1.0 if obj == obj_ref else 0.0
        else:
            out["match"] = 0.0

        # Soft match: normalized similarity between canonicalized structured outputs.
        #
        # Why:
        # - With only parse/only rewards, GRPO quickly becomes constant once the model
        #   learns to emit syntactically valid, format-only outputs.
        # - Even with strict match, many tasks have multiple valid answers or the
        #   model may be close-but-not-equal, resulting in sparse rewards.
        #
        # Soft matching provides a *dense* learning signal as long as both sides are
        # parseable (strict or best-effort). This is especially important for HF
        # datasets where the prompt may not contain StructEval-T ATTRIBUTES blocks.
        obj_a = obj_for_checks
        # Prefer strict ref object, but fall back to best-effort when needed.
        obj_b = obj_ref
        if obj_a is None or obj_b is None:
            out["match_soft"] = 0.0
        else:
            try:
                s_a = V.canonicalize_structured(obj_a, t)
                s_b = V.canonicalize_structured(obj_b, t)
                out["match_soft"] = float(difflib.SequenceMatcher(a=s_a, b=s_b).ratio())
            except Exception:
                out["match_soft"] = 0.0
    else:
        out["match"] = 0.0
        out["match_soft"] = 0.0

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

    # Optional gold matching (format-aware). If the dataset provides reference_output,
    # this can provide a strong learning signal even when StructEval ATTRIBUTES are absent.
    r += _cfg_get_typed(w, t, "w_match", 0.0) * components.get("match", 0.0)

    # Dense, format-aware soft matching against reference_output.
    r += _cfg_get_typed(w, t, "w_match_soft", 0.0) * components.get("match_soft", 0.0)

    # JSON-only optional schema / constraints
    r += float(w.get("w_schema_valid", 2.0)) * components.get("schema_valid", 0.0)
    r += float(w.get("w_exact_authors_len_2", 1.0)) * components.get("authors_len_is_2", 0.0)

    # mild penalties / constraints (JSON only)
    r += float(w.get("w_no_extra_keys", 0.0)) * (1.0 if components.get("extra_keys", 0.0) == 0 else 0.0)
    r += -0.1 * components.get("extra_keys", 0.0)

    # If format-only failed, apply penalty (format-specific override supported)
    if components.get("only", 0.0) < 1.0:
        r += _cfg_get_typed_penalty(w, t, "p_only", -1.0)

    # --------------------------------------------------------------
    # YAML style penalties (indentation / block-style)
    #
    # YAML can parse successfully even when it violates the repo's
    # canonical formatting rules. This provides an extra shaping signal
    # so GRPO does not plateau with "valid but ugly" YAML.
    # --------------------------------------------------------------
    if t == "YAML":
        # positive bonuses
        r += float(w.get("w_yaml_indent_canonical", 0.0)) * components.get("yaml_indent_canonical", 0.0)
        r += float(w.get("w_yaml_block_style", 0.0)) * components.get("yaml_block_style", 0.0)
        # penalties when violated
        if components.get("yaml_indent_canonical", 0.0) < 1.0:
            r += float(w.get("p_yaml_indent_canonical_fail", 0.0))
        if components.get("yaml_block_style", 0.0) < 1.0:
            r += float(w.get("p_yaml_block_style_fail", 0.0))
    # --------------------------------------------------------------
    # TOML style shaping (canonical formatting)
    #
    # TOML is less permissive than YAML, but datasets may contain multiple
    # valid-but-inconsistent emission styles (key order, table ordering, etc.).
    # This shaping signal encourages the model to converge to the repo's canonical TOML form.
    # --------------------------------------------------------------
    if t == "TOML":
        r += float(w.get("w_toml_canonical", 0.0)) * components.get("toml_canonical", 0.0)
        # Dense shaping toward canonical TOML. This should be preferred over
        # cliff-like binary penalties.
        r += float(w.get("w_toml_canonical_soft", 0.0)) * components.get("toml_canonical_soft", 0.0)

        # Optional proportional penalty: penalize (1 - similarity).
        # This avoids the all-or-nothing behavior of `p_toml_canonical_fail`.
        p_scale = float(w.get("p_toml_canonical_soft_scale", 0.0))
        if p_scale != 0.0:
            r += p_scale * (1.0 - float(components.get("toml_canonical_soft", 0.0)))

        # Legacy binary penalty (kept for backward compatibility).
        if components.get("toml_canonical", 0.0) < 1.0:
            r += float(w.get("p_toml_canonical_fail", 0.0))


    # Penalize any wrapper text outside the structured payload (format-specific override supported).
    # This targets outputs like "Sure! ... ```xml" ... and pushes the model toward emitting
    # structure-only text.
    if components.get("extraneous", 0.0) >= 1.0:
        r += _cfg_get_typed_penalty(w, t, "p_extraneous", 0.0)

    return float(r)

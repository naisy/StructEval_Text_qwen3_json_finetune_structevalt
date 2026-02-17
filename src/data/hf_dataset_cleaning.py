from __future__ import annotations

"""Deterministic cleaning rules for HF SFT/GRPO datasets.

This module is a code version of the *final* exclusion policy that was
validated in `structeval_dataset_check.ipynb`.

Design goals
------------
- Deterministic (no external judge).
- Safe: never edits dataset contents; only decides keep/drop.
- Fast enough to run at HF import time.

Key policy (summary)
--------------------
1) base_ok:
   - strict parse (lint) succeeds for the declared output_type
   - prompt-inferred target format matches output_type
   - no extra text around the structured output

2) additionally drop *semantic FAIL* cases for conversion-like tasks:
   - task_kind == clean_strict (has input payload + target format inferred)
   - base_ok is True
   - equivalence check returns FAIL

We do *not* drop UNKNOWN equivalence because the checker can be ambiguous.
"""

import csv
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import yaml

from src.data import validators as V


FMT_ALIASES = {
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "xml": "xml",
    "csv": "csv",
}


def norm_fmt(s: str | None) -> str | None:
    if not s:
        return None
    return FMT_ALIASES.get(str(s).strip().lower())


def norm_output_type(s: str | None) -> str:
    t = (str(s) if s is not None else "").strip().upper()
    if t in {"YML"}:
        return "YAML"
    if t.startswith("C_"):
        t = t[2:]
    return t


_FMT_WORD_RE = re.compile(r"(?i)\b(json|yaml|yml|toml|xml|csv)\b")


def infer_target_format_from_prompt(prompt: str) -> str | None:
    """Infer target/output format from the prompt.

    Matches common patterns like:
    - "from JSON to YAML"
    - "convert ... to TOML"
    - "output XML"
    - "in CSV format"
    """
    if not prompt:
        return None
    q = (prompt or "").lower()

    # from A to B
    m = re.search(r"\bfrom\s+(json|yaml|yml|toml|xml|csv)\s+to\s+(json|yaml|yml|toml|xml|csv)\b", q)
    if m:
        return norm_fmt(m.group(2))

    # "A to B" when conversion verbs exist
    m = re.search(r"\b(json|yaml|yml|toml|xml|csv)\s+to\s+(json|yaml|yml|toml|xml|csv)\b", q)
    if m and any(w in q for w in ["convert", "transform", "reformat"]):
        return norm_fmt(m.group(2))

    # "into B"
    m = re.search(r"\binto\s+(json|yaml|yml|toml|xml|csv)\b", q)
    if m and any(w in q for w in ["convert", "transform", "reformat", "output", "produce", "generate", "create", "return", "respond"]):
        return norm_fmt(m.group(1))

    # "as B"
    m = re.search(r"\bas\s+(json|yaml|yml|toml|xml|csv)\b", q)
    if m and any(w in q for w in ["output", "produce", "generate", "create", "return", "respond"]):
        return norm_fmt(m.group(1))

    # "in B format"
    m = re.search(r"\bin\s+(json|yaml|yml|toml|xml|csv)\s+format\b", q)
    if m and any(w in q for w in ["output", "produce", "generate", "create", "transform", "convert", "return", "respond"]):
        return norm_fmt(m.group(1))

    # "output JSON" etc.
    m = re.search(r"\b(output|produce|generate|create|return|respond)\s+(?:a\s+)?(json|yaml|yml|toml|xml|csv)\b", q)
    if m:
        return norm_fmt(m.group(2))

    # If only one format word appears, treat it as target (better than nothing).
    hits = _FMT_WORD_RE.findall(q)
    if len(hits) == 1:
        return norm_fmt(hits[0])

    return None


def infer_src_tgt(prompt: str) -> tuple[str | None, str | None]:
    if not prompt:
        return None, None
    q = (prompt or "").lower()
    m = re.search(r"\bfrom\s+(json|yaml|yml|toml|xml|csv)\s+to\s+(json|yaml|yml|toml|xml|csv)\b", q)
    if m:
        return norm_fmt(m.group(1)), norm_fmt(m.group(2))
    m = re.search(r"\b(json|yaml|yml|toml|xml|csv)\s+to\s+(json|yaml|yml|toml|xml|csv)\b", q)
    if m and any(w in q for w in ["convert", "transform", "reformat"]):
        return norm_fmt(m.group(1)), norm_fmt(m.group(2))
    return None, infer_target_format_from_prompt(prompt)


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]+)?\n(.*?)\n```", re.DOTALL)


def _extract_first_fenced_block(text: str) -> str:
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    return (m.group(1) if m else "").strip()


def extract_input_payload_from_prompt(prompt: str) -> str:
    """Best-effort extraction of inline input payload from a prompt."""
    if not prompt:
        return ""

    block = _extract_first_fenced_block(prompt)
    if block:
        return block

    # After a line that ends with ':' (common patterns).
    lines = (prompt or "").splitlines()
    for i, line in enumerate(lines[:10]):
        if line.strip().endswith(":"):
            tail = "\n".join(lines[i + 1 :]).strip()
            if tail:
                return tail

    txt = (prompt or "").strip()

    def _find_balanced(start_char: str, end_char: str) -> str:
        start = txt.find(start_char)
        if start == -1:
            return ""
        depth = 0
        for j in range(start, len(txt)):
            ch = txt[j]
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return txt[start : j + 1]
        return ""

    obj = _find_balanced("{", "}")
    arr = _find_balanced("[", "]")
    if obj and (not arr or len(obj) >= len(arr)):
        return obj.strip()
    if arr:
        return arr.strip()

    # Fallback: last half
    return "\n".join(lines[max(0, len(lines) // 2) :]).strip()


def has_input_payload_from_text(text: str) -> bool:
    """Heuristic: does the text appear to contain an inline source payload?"""
    if not text:
        return False

    # Code fences
    if "```" in text:
        parts = re.split(r"```", text)
        for b in parts[1::2]:
            bb = b.strip()
            bb = re.sub(r"^[a-zA-Z0-9_+\-]+\n", "", bb)
            if bb.startswith(("{", "[", "<")):
                return True
            if "\n" in bb and bb.count(",") >= 2:
                return True
            if bb.count("=") >= 2 or bb.count(":") >= 3:
                return True

    # After a colon
    m = re.search(r":\s*(.*)$", text, flags=re.DOTALL)
    if m:
        tail = m.group(1).strip()
        if tail:
            if tail.startswith(("{", "[", "<")):
                return True
            if "\n" in tail and tail.count(",") >= 2:
                return True
            if tail.count("=") >= 2 or tail.count(":") >= 3:
                return True

    if "<?xml" in text.lower():
        return True
    return False


def classify_task_text(text: str) -> tuple[str, str | None, str | None, bool]:
    has_payload = has_input_payload_from_text(text)
    src, tgt = infer_src_tgt(text)
    if (tgt is not None) and has_payload:
        kind = "clean_strict"
    elif (tgt is not None) and (not has_payload):
        kind = "generation_like"
    else:
        kind = "clean_relaxed"
    return kind, src, tgt, has_payload


def _parse_by_fmt(fmt: str | None, s: str) -> tuple[bool, Any, str | None]:
    f = (fmt or "").lower()
    try:
        if f == "json":
            return True, json.loads(s), None
        if f == "yaml":
            return True, yaml.safe_load(s), None
        if f == "toml":
            # Prefer stdlib tomllib; fall back to tomllib-backed V.parse_toml if needed.
            ok, obj, err = V.parse_toml(s)
            return bool(ok), obj, err
        if f == "xml":
            root = ET.fromstring((s or "").strip())
            return True, root, None
        if f == "csv":
            fobj = io.StringIO(s)
            rows = list(csv.reader(fobj))
            return True, rows, None
        return False, None, f"unknown format: {fmt}"
    except Exception as e:
        return False, None, str(e)


def _xml_to_obj(elem: ET.Element) -> Any:
    # attributes
    out: Dict[str, Any] = {}
    if elem.attrib:
        out["@"] = dict(elem.attrib)

    children = list(elem)
    text = (elem.text or "").strip()

    if children:
        groups: Dict[str, list[Any]] = {}
        for c in children:
            groups.setdefault(c.tag, []).append(_xml_to_obj(c))
        for k, vs in groups.items():
            out[k] = vs[0] if len(vs) == 1 else vs
        if text:
            out["#text"] = text
        return {elem.tag: out}

    # leaf
    if out:
        if text:
            out["#text"] = text
        return {elem.tag: out}
    return {elem.tag: text}


def _canonicalize(x: Any) -> Any:
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)
        if isinstance(x, float) and x.is_integer():
            return int(x)
        return x
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canonicalize(v) for v in x]
    if isinstance(x, ET.Element):
        return _canonicalize(_xml_to_obj(x))
    return str(x)


def _leaf_items(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Collect (path, value) for leaf nodes."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_leaf_items(v, p))
        return out
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.extend(_leaf_items(v, p))
        return out
    out.append((prefix or "$", str(obj)))
    return out


def _containment_f1(a: Any, b: Any) -> float:
    """F1 of leaf-set containment between a and b."""
    sa = set(_leaf_items(a))
    sb = set(_leaf_items(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    p = inter / len(sb) if sb else 0.0
    r = inter / len(sa) if sa else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


@dataclass
class EquivalenceResult:
    equivalence: str  # PASS | FAIL | UNKNOWN
    score: float | None
    reason: str


def equivalence_check(prompt: str, output_fmt: str, output_structured: str) -> EquivalenceResult:
    """Semantic check for conversion-like tasks.

    - If input parse fails or src is unknown -> UNKNOWN
    - If output parse fails -> FAIL
    - Otherwise compare canonical structures.

    Special case: XML containment soft-match returns UNKNOWN with a score.
    """
    src_fmt, tgt_fmt = infer_src_tgt(prompt)
    input_payload = extract_input_payload_from_prompt(prompt)

    inferred_src = src_fmt
    if inferred_src is None:
        t = (input_payload or "").lstrip()
        if t.startswith(("{", "[")):
            inferred_src = "json"
        elif t.startswith("<"):
            inferred_src = "xml"
        elif "\n" in t and t.count(",") >= 2:
            inferred_src = "csv"
        elif t.count("=") >= 2:
            inferred_src = "toml"
        elif t.count(":") >= 3:
            inferred_src = "yaml"

    in_ok, in_obj, in_err = _parse_by_fmt(inferred_src, input_payload)
    out_ok, out_obj, out_err = _parse_by_fmt(output_fmt, output_structured)

    if not out_ok:
        return EquivalenceResult("FAIL", 0.0, f"output_parse_failed: {out_err}")
    if not in_ok:
        return EquivalenceResult("UNKNOWN", None, f"input_parse_failed_or_unknown: {in_err}")

    in_can = _canonicalize(in_obj)
    out_can = _canonicalize(out_obj)

    if in_can == out_can:
        return EquivalenceResult("PASS", 1.0, "")

    # XML soft match: containment F1
    if (inferred_src == "xml") or (output_fmt == "xml"):
        f1 = _containment_f1(in_can, out_can)
        return EquivalenceResult("UNKNOWN", f1, f"containment_f1:{f1:.4f}")

    return EquivalenceResult("FAIL", 0.0, "semantic_mismatch")


@dataclass
class CleaningDecision:
    keep: bool
    reason: str


def decide_keep_example(
    *,
    prompt: str,
    output_type: str,
    extracted_output: str,
    raw_answer_text: str | None,
    relax_xml_f1: float = 0.97,
) -> CleaningDecision:
    """Return keep/drop decision following the *final* dataset policy.

    IMPORTANT:
    - We must only drop examples that are clearly unusable (would become FAILED).
    - Anything that is ambiguous (UNKNOWN) should be kept for training.

    Concretely, we drop only when:
    - the extracted structured output does not lint/parse (strict)
    - the prompt-inferred target format contradicts the annotated output_type
    - for conversion-like tasks, the input/output semantic check is a clear FAIL

    We do *not* drop merely because the original answer contains extra
    explanation text or markers (e.g., "Output:"). Those often correspond to
    UNKNOWN/benign cases and should be retained.
    """

    out_t = norm_output_type(output_type)

    # 1) lint_ok
    lint_ok = False
    if out_t == "JSON":
        lint_ok = bool(V.parse_json(extracted_output)[0])
    elif out_t == "YAML":
        lint_ok = bool(V.parse_yaml(extracted_output)[0])
    elif out_t == "TOML":
        lint_ok = bool(V.parse_toml(extracted_output)[0])
    elif out_t == "XML":
        lint_ok = bool(V.parse_xml(extracted_output)[0])
    elif out_t == "CSV":
        lint_ok = bool(V.parse_csv(extracted_output)[0])
    else:
        lint_ok = bool(V.parse_json(extracted_output)[0])

    if not lint_ok:
        # Clearly unusable: structured payload itself is invalid.
        return CleaningDecision(False, "lint_fail")

    # 2) format mismatch
    tgt = infer_target_format_from_prompt(prompt)
    if tgt is not None and tgt.upper() != out_t:
        return CleaningDecision(False, "format_mismatch")

    # NOTE:
    # We intentionally do NOT drop "has extra text" cases.
    # They are not necessarily FAILED; many are UNKNOWN/benign and should remain.

    # base_ok satisfied here
    kind, src_u, tgt_u, has_payload = classify_task_text(prompt)

    # 4) semantic FAIL drop for *conversion* tasks only (clean_strict with explicit src->tgt)
    #
    # daichira/* datasets include many "extract" tasks where the input is free-form TEXT and
    # the prompt only specifies the *target* format (e.g. "output TOML"). For such tasks,
    # equivalence_check is not meaningful and tends to produce false FAILs (semantic_mismatch),
    # wiping out entire formats (observed: TOML/YAML nearly empty).
    #
    # Therefore we require an explicit source format (src_u) before running equivalence_check.
    if kind == "clean_strict" and has_payload and (tgt_u is not None) and (src_u is not None):
        eq = equivalence_check(prompt, out_t.lower(), extracted_output)
        if eq.equivalence == "FAIL":
            return CleaningDecision(False, f"equiv_fail:{eq.reason}")
        if (eq.equivalence == "UNKNOWN") and out_t == "XML" and (eq.score is not None) and (eq.score >= relax_xml_f1):
            # accept soft XML match
            return CleaningDecision(True, f"keep_xml_soft:{eq.reason}")
        return CleaningDecision(True, "keep")

    return CleaningDecision(True, "keep")

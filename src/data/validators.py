from __future__ import annotations

import csv
import io
import json
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Tuple

import yaml
from jsonschema import Draft202012Validator


# Strip Markdown fences like:
# ```json
# ...
# ```
# ```yaml
# ...
# ```
_CODE_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_\-]*\s*|\s*```\s*$", re.MULTILINE)


def contains_code_fence(text: str) -> bool:
    """Return True if the text contains Markdown triple-backtick fences.

    Notes
    -----
    This repo's evaluation rule treats fenced outputs as *invalid* because
    StructEval-T style tasks expect the model to emit the structured payload
    only (no markdown wrappers).
    """
    return "```" in (text or "")


def strip_code_fences(text: str) -> str:
    """Remove Markdown triple-backtick fences if present."""
    return re.sub(_CODE_FENCE_RE, "", text).strip()


# ---------------------------------------------------------------------------
# Structured payload extraction / extraneous-text detection
# ---------------------------------------------------------------------------


_YAML_START_RE = re.compile(
    r"^\s*(---\s*$|\-|\?|\{|\[|[A-Za-z0-9_\-\.]+\s*:)"
)
_TOML_START_RE = re.compile(r"^\s*(\[.*\]|[A-Za-z0-9_\-\.]+\s*=)")


# ---------------------------------------------------------------------------
# YAML style / indentation checks (project-specific strictness)
# ---------------------------------------------------------------------------


_YAML_COMMENT_RE = re.compile(r"^\s*#")


def yaml_uses_flow_style(text: str) -> bool:
    """Return True if YAML appears to use flow-style collections.

    Motivation
    ----------
    YAML allows JSON-like flow style (e.g., `{a: 1}`, `[1, 2]`). For this
    project we want models to learn the *block style* consistently to avoid
    drifting into JSON-ish formatting.

    This is a conservative heuristic and may flag some edge cases.
    """
    t = (text or "").lstrip()
    if not t:
        return False
    # Leading flow collection is a strong signal.
    if t[0] in "{[":
        return True
    # Inline flow collections are also treated as flow-style usage.
    # We intentionally do not attempt to parse YAML tokens here; the rule is
    # meant as a simple, deterministic shaping signal for RL.
    return ("{" in t) or ("[" in t) or ("}" in t) or ("]" in t)


def yaml_indent_is_canonical(text: str, *, indent: int = 2) -> bool:
    """Return True if YAML indentation follows a strict 2-space-per-level style.

    What we enforce (heuristic, deterministic):
    - No tabs for indentation
    - Indent width must be a multiple of `indent`
    - When a line opens a block (ends with ':'), the next non-empty/non-comment
      line that is more-indented must be exactly +`indent` deeper (not +4, +6, ...)
    - For sequence items (`-`), nested content must also increase by exactly
      +`indent` relative to the dash indentation.

    This intentionally goes beyond YAML syntax validity and is used to
    discourage "JSON-ish" YAML where indentation drifts.
    """
    if not text:
        return False

    lines = (text or "").splitlines()

    def _is_ignorable(ln: str) -> bool:
        s = ln.strip()
        return (s == "") or bool(_YAML_COMMENT_RE.match(ln))

    # Quick tab check
    for ln in lines:
        if ln.startswith("\t") or "\t" in (re.match(r"^[ \t]*", ln).group(0) if ln else ""):
            return False

    # Indent multiple check
    indents: list[int] = []
    for ln in lines:
        if _is_ignorable(ln):
            continue
        m = re.match(r"^[ ]*", ln)
        n = len(m.group(0)) if m else 0
        if n % indent != 0:
            return False
        indents.append(n)

    if not indents:
        return False

    # Block-opening strictness: when the next significant line is deeper, it must be +indent.
    for i, ln in enumerate(lines):
        if _is_ignorable(ln):
            continue
        m = re.match(r"^[ ]*", ln)
        cur = len(m.group(0)) if m else 0
        stripped = ln.strip()
        is_seq = stripped.startswith("-")
        opens_block = stripped.endswith(":")
        if not (is_seq or opens_block):
            continue

        # Find next significant line
        j = i + 1
        while j < len(lines) and _is_ignorable(lines[j]):
            j += 1
        if j >= len(lines):
            continue
        m2 = re.match(r"^[ ]*", lines[j])
        nxt = len(m2.group(0)) if m2 else 0
        if nxt > cur and (nxt - cur) != indent:
            return False

    return True


def _first_matching_line_start(text: str, pattern: re.Pattern[str]) -> int | None:
    """Return character index of the first line that matches pattern, else None."""
    if not text:
        return None
    idx = 0
    for line in text.splitlines(True):
        if pattern.search(line):
            return idx
        idx += len(line)
    return None


def extract_payload_and_extraneous(text: str, output_type: str) -> tuple[str, bool]:
    """Extract the most-likely structured payload and detect extraneous text.

    This is used for GRPO reward shaping.

    Returns
    -------
    payload: str
        The extracted structured content (code fences removed when applicable).
    has_extraneous: bool
        True if non-whitespace text exists outside the extracted payload.

    Notes
    -----
    - If a triple-backtick fenced block exists, we treat the first fenced block
      as the payload and mark any outside text as extraneous.
    - JSON: we also support best-effort extraction of the first object/array.
    - YAML/CSV are permissive syntactically, so we use simple "start line"
      heuristics to avoid treating a natural-language prefix as valid.
    """
    t = (output_type or "JSON").strip().upper()
    raw = text or ""

    inside, outside = extract_first_fenced_block(raw)
    if inside is not None:
        payload = strip_code_fences(inside).strip()
        # Even if the fenced block is the only content, the fences themselves
        # are wrapper text outside the payload. Treat them as extraneous so
        # GRPO reward can penalize markdown-wrapped outputs.
        has_extraneous = True
        if outside.strip() != "":
            has_extraneous = True
        return payload, has_extraneous

    # No fenced block: apply type-specific heuristics.
    s = strip_code_fences(raw)

    if t == "JSON":
        ok, _obj, _err, used = parse_json_best_effort(s)
        if ok and used is not None:
            has_extra = s.strip() != used.strip()
            return used.strip(), has_extra
        return s.strip(), False

    if t == "XML":
        st = s.lstrip()
        first_lt = st.find("<")
        if first_lt == -1:
            return st, False
        # Check if anything non-ws existed before the first '<'
        prefix = st[:first_lt]
        payload = st[first_lt:].strip()
        return payload, prefix.strip() != ""

    if t == "YAML":
        start = _first_matching_line_start(s, _YAML_START_RE)
        if start is None:
            return s.strip(), False
        prefix = s[:start]
        payload = s[start:].strip()
        return payload, prefix.strip() != ""

    if t == "TOML":
        start = _first_matching_line_start(s, _TOML_START_RE)
        if start is None:
            return s.strip(), False
        prefix = s[:start]
        payload = s[start:].strip()
        return payload, prefix.strip() != ""

    if t == "CSV":
        # Heuristic: payload likely starts at first line containing a comma.
        idx = 0
        start = None
        for line in s.splitlines(True):
            if "," in line:
                start = idx
                break
            idx += len(line)
        if start is None:
            # No obvious CSV structure found.
            return s.strip(), False
        prefix = s[:start]
        payload = s[start:].strip()
        return payload, prefix.strip() != ""

    return s.strip(), False




# ---------------------------------------------------------------------------
# "Only" heuristics for non-JSON formats
# ---------------------------------------------------------------------------

_FENCE_BLOCK_RE = re.compile(r"""```[a-zA-Z0-9_\-]*\s*\n(.*?)\n```""", re.DOTALL)


def extract_first_fenced_block(text: str) -> tuple[str | None, str]:
    """Return (inside_block, outside_text).

    - If a triple-backtick fenced block exists, returns its inner content and
      the remaining text with the whole fenced block removed.
    - Otherwise returns (None, original_text).
    """
    m = _FENCE_BLOCK_RE.search(text)
    if not m:
        return None, text
    inside = m.group(1)
    outside = text[:m.start()] + text[m.end():]
    return inside, outside


def is_fenced_block_only(text: str) -> bool:
    """True if the output is *only* a single fenced block (plus whitespace)."""
    inside, outside = extract_first_fenced_block(text)
    if inside is None:
        return False
    return outside.strip() == ""


def parse_json(text: str) -> Tuple[bool, Any | None, str | None]:
    """Parse JSON from raw model output (expects JSON-only)."""
    if contains_code_fence(text):
        return False, None, "markdown_fence"
    t = strip_code_fences(text).strip()
    try:
        obj = json.loads(t)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def parse_json_best_effort(text: str) -> Tuple[bool, Any | None, str | None, str | None]:
    """Best-effort JSON extraction.

    Many instruction-tuned models sometimes emit extra tokens before/after JSON,
    e.g. "Sure!" or trailing commentary. For RL reward shaping we want a
    non-binary signal:

    - Strict parse_json(): only succeeds if the whole output is JSON.
    - Best-effort: attempts to extract the first JSON object/array substring.

    Returns: (ok, obj, err, used_substring)
    """
    t = strip_code_fences(text).strip()

    # Fast path: already valid JSON.
    ok, obj, err = parse_json(t)
    if ok:
        return True, obj, None, t

    # Heuristic: find the first JSON object/array and parse until brackets match.
    # This is deliberately simple and deterministic.
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        return False, None, err, None
    start = min(starts)

    opener = t[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False

    for j in range(start, len(t)):
        ch = t[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                candidate = t[start : j + 1]
                try:
                    obj2 = json.loads(candidate)
                    return True, obj2, None, candidate
                except Exception as e2:
                    return False, None, str(e2), candidate

    return False, None, err, None


def is_json_only(text: str) -> bool:
    """Heuristic: succeeds if entire stripped output parses as JSON."""
    if contains_code_fence(text):
        return False
    ok, _, _ = parse_json(text)
    return ok


def parse_yaml(text: str) -> Tuple[bool, Any | None, str | None]:
    """Parse YAML from raw model output (expects YAML-only)."""
    if contains_code_fence(text):
        return False, None, "markdown_fence"
    t = strip_code_fences(text).strip()
    try:
        obj = yaml.safe_load(t)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def parse_yaml_best_effort(text: str) -> Tuple[bool, Any | None, str | None, str | None]:
    """Best-effort YAML extraction.

    YAML is extremely permissive (plain prose is a valid scalar), so for our
    structured-data tasks we require the parsed object to be a mapping or a
    sequence.

    Strategy (deterministic):
    - If a fenced block exists, only consider that block.
    - Otherwise, try to parse increasing line prefixes and keep the last
      successful structured (dict/list) parse.

    Returns: (ok, obj, err, used_substring)
    """
    inside, _outside = extract_first_fenced_block(text)
    base = inside if inside is not None else text
    t = strip_code_fences(base).strip()

    # Fast path.
    ok, obj, err = parse_yaml(t)
    if ok and isinstance(obj, (dict, list)):
        return True, obj, None, t

    # Incremental prefix scan.
    lines = t.splitlines()
    last_ok: tuple[bool, Any | None, str | None, str | None] = (False, None, err, None)
    max_lines = min(len(lines), 400)
    for i in range(1, max_lines + 1):
        cand = "\n".join(lines[:i]).strip()
        if not cand:
            continue
        ok2, obj2, err2 = parse_yaml(cand)
        if ok2 and isinstance(obj2, (dict, list)):
            last_ok = (True, obj2, None, cand)
        else:
            # keep scanning; YAML often needs a later line to become structured
            last_ok = last_ok if last_ok[0] else (False, None, err2, None)
    return last_ok


def is_yaml_only(text: str) -> bool:
    """Heuristic: prefer fenced-block-only outputs when fences are used.

    For formats like YAML/TOML/XML/CSV, extra natural-language text can still be
    syntactically valid (e.g., YAML treats it as a scalar/mapping). If the model
    outputs a fenced block, we require that *nothing else* is present outside it.
    Otherwise we fall back to parsing the full (stripped) output.
    """
    def _is_structured_yaml(t: str) -> bool:
        if contains_code_fence(t):
            return False
        ok, obj, _ = parse_yaml(t)
        if not ok:
            return False
        # YAML is very permissive (any prose is a valid scalar). For our tasks,
        # we expect a mapping or sequence.
        return isinstance(obj, (dict, list))

    inside, outside = extract_first_fenced_block(text)
    if inside is not None:
        # fenced outputs are considered invalid (no markdown wrappers allowed)
        return False
    return _is_structured_yaml(text)


def parse_toml(text: str) -> Tuple[bool, Any | None, str | None]:
    """Parse TOML from raw model output (expects TOML-only)."""
    if contains_code_fence(text):
        return False, None, "markdown_fence"
    t = strip_code_fences(text).strip()
    try:
        obj = tomllib.loads(t)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def parse_toml_best_effort(text: str) -> Tuple[bool, Any | None, str | None, str | None]:
    """Best-effort TOML extraction.

    TOML parsing is strict and fails on trailing junk. We try to find the
    longest line-prefix that parses.

    Returns: (ok, obj, err, used_substring)
    """
    inside, _outside = extract_first_fenced_block(text)
    base = inside if inside is not None else text
    t = strip_code_fences(base).strip()

    ok, obj, err = parse_toml(t)
    if ok:
        return True, obj, None, t

    lines = t.splitlines()
    last_ok: tuple[bool, Any | None, str | None, str | None] = (False, None, err, None)
    max_lines = min(len(lines), 400)
    for i in range(1, max_lines + 1):
        cand = "\n".join(lines[:i]).strip()
        if not cand:
            continue
        ok2, obj2, err2 = parse_toml(cand)
        if ok2:
            last_ok = (True, obj2, None, cand)
        else:
            last_ok = last_ok if last_ok[0] else (False, None, err2, None)
    return last_ok


def is_toml_only(text: str) -> bool:
    """Heuristic: prefer fenced-block-only outputs when fences are used.

    For formats like YAML/TOML/XML/CSV, extra natural-language text can still be
    syntactically valid (e.g., YAML treats it as a scalar/mapping). If the model
    outputs a fenced block, we require that *nothing else* is present outside it.
    Otherwise we fall back to parsing the full (stripped) output.
    """
    inside, outside = extract_first_fenced_block(text)
    if inside is not None:
        return False
    ok, _, _ = parse_toml(text)
    return ok


def parse_xml(text: str) -> Tuple[bool, Any | None, str | None]:
    """Parse XML from raw model output (expects XML-only).

    Returns the root Element.
    """
    if contains_code_fence(text):
        return False, None, "markdown_fence"
    t = strip_code_fences(text).strip()
    try:
        root = ET.fromstring(t)
        return True, root, None
    except Exception as e:
        return False, None, str(e)


def parse_xml_best_effort(text: str) -> Tuple[bool, Any | None, str | None, str | None]:
    """Best-effort XML extraction.

    XML is strict and fails on trailing junk. We try to find the longest
    line-prefix that forms a well-formed XML document.

    Returns: (ok, root, err, used_substring)
    """
    inside, _outside = extract_first_fenced_block(text)
    base = inside if inside is not None else text
    t = strip_code_fences(base).strip()
    # Trim leading non-XML characters but keep an extraneous signal in the caller.
    lt = t.find("<")
    if lt != -1:
        t = t[lt:]

    ok, root, err = parse_xml(t)
    if ok:
        return True, root, None, t

    lines = t.splitlines()
    last_ok: tuple[bool, Any | None, str | None, str | None] = (False, None, err, None)
    max_lines = min(len(lines), 600)
    for i in range(1, max_lines + 1):
        cand = "\n".join(lines[:i]).strip()
        if not cand:
            continue
        ok2, root2, err2 = parse_xml(cand)
        if ok2:
            last_ok = (True, root2, None, cand)
        else:
            last_ok = last_ok if last_ok[0] else (False, None, err2, None)
    return last_ok


def is_xml_only(text: str) -> bool:
    """Heuristic: prefer fenced-block-only outputs when fences are used.

    For formats like YAML/TOML/XML/CSV, extra natural-language text can still be
    syntactically valid (e.g., YAML treats it as a scalar/mapping). If the model
    outputs a fenced block, we require that *nothing else* is present outside it.
    Otherwise we fall back to parsing the full (stripped) output.
    """
    inside, outside = extract_first_fenced_block(text)
    if inside is not None:
        return False
    ok, _, _ = parse_xml(text)
    return ok


def parse_csv(text: str) -> Tuple[bool, Any | None, str | None]:
    """Parse CSV from raw model output (expects CSV-only).

    Returns a list of rows (each row is a list[str]).
    """
    if contains_code_fence(text):
        return False, None, "markdown_fence"
    t = strip_code_fences(text).strip()
    try:
        f = io.StringIO(t)
        rows = list(csv.reader(f))
        # Consider empty / whitespace-only outputs invalid CSV for our purposes.
        if len(rows) == 0:
            return False, None, "empty csv"
        return True, rows, None
    except Exception as e:
        return False, None, str(e)


def parse_csv_best_effort(text: str) -> Tuple[bool, Any | None, str | None, str | None]:
    """Best-effort CSV extraction.

    CSV parsing is permissive, but trailing junk lines (e.g., assistant chatter)
    will usually break the "same number of columns" structure. We try to find a
    longest prefix that:
      - parses as CSV
      - has at least one row with >=2 columns (to avoid prose-as-one-cell)

    Returns: (ok, rows, err, used_substring)
    """
    inside, _outside = extract_first_fenced_block(text)
    base = inside if inside is not None else text
    t = strip_code_fences(base).strip()

    ok, rows, err = parse_csv(t)
    if ok and rows is not None and max((len(r) for r in rows), default=0) >= 2:
        return True, rows, None, t

    lines = t.splitlines()
    last_ok: tuple[bool, Any | None, str | None, str | None] = (False, None, err, None)
    max_lines = min(len(lines), 600)
    for i in range(1, max_lines + 1):
        cand = "\n".join(lines[:i]).strip()
        if not cand:
            continue
        ok2, rows2, err2 = parse_csv(cand)
        if ok2 and rows2 is not None and max((len(r) for r in rows2), default=0) >= 2:
            last_ok = (True, rows2, None, cand)
        else:
            last_ok = last_ok if last_ok[0] else (False, None, err2, None)
    return last_ok


def is_csv_only(text: str) -> bool:
    """Heuristic: prefer fenced-block-only outputs when fences are used.

    For formats like YAML/TOML/XML/CSV, extra natural-language text can still be
    syntactically valid (e.g., YAML treats it as a scalar/mapping). If the model
    outputs a fenced block, we require that *nothing else* is present outside it.
    Otherwise we fall back to parsing the full (stripped) output.
    """
    def _is_structured_csv(t: str) -> bool:
        if contains_code_fence(t):
            return False
        ok, rows, _ = parse_csv(t)
        if not ok or rows is None:
            return False
        # CSV is also permissive; require at least 2 columns somewhere.
        max_cols = max((len(r) for r in rows), default=0)
        return max_cols >= 2

    inside, outside = extract_first_fenced_block(text)
    if inside is not None:
        return False
    return _is_structured_csv(text)


# ---------------------------------------------------------------------------
# Canonicalization (for soft matching)
# ---------------------------------------------------------------------------


def canonicalize_structured(obj: Any, output_type: str) -> str:
    """Return a stable string representation of a parsed structured output.

    This is used to compute a dense "soft match" reward in GRPO when a
    `reference_output` is available.

    The representation is *not* intended to round-trip to the original format.
    It is only intended to be stable enough that string similarity reflects
    structural similarity.
    """
    t = (output_type or "JSON").strip().upper()
    try:
        if t in {"JSON", "YAML", "TOML"}:
            # Normalize to JSON with sorted keys for stability.
            return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if t == "XML":
            if isinstance(obj, ET.Element):
                return ET.tostring(obj, encoding="unicode")
            return str(obj)
        if t == "CSV":
            # Expect list[list[str]]
            if isinstance(obj, list):
                rows = []
                for r in obj:
                    if isinstance(r, (list, tuple)):
                        rows.append(",".join(str(x) for x in r))
                    else:
                        rows.append(str(r))
                return "\n".join(rows)
            return str(obj)
    except Exception:
        return str(obj)
    return str(obj)


def build_schema_validator(schema: dict | str | Path) -> Draft202012Validator:
    """Build a JSON Schema validator.

    Accepts either:
    - a schema dict
    - a path (str / Path) to a JSON schema file

    This makes call sites simpler (configs typically provide a file path).
    """
    if isinstance(schema, (str, Path)):
        p = Path(schema)
        schema = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise TypeError(f"schema must be a dict or path to json, got: {type(schema)}")
    return Draft202012Validator(schema)


def validate_schema(obj: Any, validator: Draft202012Validator) -> Tuple[bool, list[str]]:
    errors = []
    for e in sorted(validator.iter_errors(obj), key=str):
        errors.append(e.message)
    return (len(errors) == 0), errors


def count_extra_keys(obj: Any, allowed_top_level: set[str]) -> int:
    if not isinstance(obj, dict):
        return 0
    return sum(1 for k in obj.keys() if k not in allowed_top_level)


def check_task_constraints_article_meta(obj: Any) -> dict[str, bool]:
    """Example constraint checks for the provided task:
    - title: string
    - authors: list of exactly two items, each has name+affiliation (strings)
    - publication.year: integer
    - keywords: list[str]
    """
    out = {
        "title_is_string": False,
        "authors_is_list": False,
        "authors_len_is_2": False,
        "authors_items_ok": False,
        "publication_year_is_int": False,
        "keywords_is_list_str": False,
    }
    if not isinstance(obj, dict):
        return out

    out["title_is_string"] = isinstance(obj.get("title"), str)

    authors = obj.get("authors")
    out["authors_is_list"] = isinstance(authors, list)
    if isinstance(authors, list):
        out["authors_len_is_2"] = (len(authors) == 2)
        items_ok = True
        for a in authors:
            if not isinstance(a, dict):
                items_ok = False
                break
            if not isinstance(a.get("name"), str) or not isinstance(a.get("affiliation"), str):
                items_ok = False
                break
        out["authors_items_ok"] = items_ok

    publication = obj.get("publication")
    if isinstance(publication, dict):
        out["publication_year_is_int"] = isinstance(publication.get("year"), int)

    keywords = obj.get("keywords")
    if isinstance(keywords, list) and all(isinstance(x, str) for x in keywords):
        out["keywords_is_list_str"] = True

    return out

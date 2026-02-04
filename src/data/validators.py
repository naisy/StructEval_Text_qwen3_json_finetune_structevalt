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


def strip_code_fences(text: str) -> str:
    """Remove Markdown triple-backtick fences if present."""
    return re.sub(_CODE_FENCE_RE, "", text).strip()




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
    ok, _, _ = parse_json(text)
    return ok


def parse_yaml(text: str) -> Tuple[bool, Any | None, str | None]:
    """Parse YAML from raw model output (expects YAML-only)."""
    t = strip_code_fences(text).strip()
    try:
        obj = yaml.safe_load(t)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def is_yaml_only(text: str) -> bool:
    """Heuristic: prefer fenced-block-only outputs when fences are used.

    For formats like YAML/TOML/XML/CSV, extra natural-language text can still be
    syntactically valid (e.g., YAML treats it as a scalar/mapping). If the model
    outputs a fenced block, we require that *nothing else* is present outside it.
    Otherwise we fall back to parsing the full (stripped) output.
    """
    inside, outside = extract_first_fenced_block(text)
    if inside is not None:
        if outside.strip() != "":
            return False
        ok, _, _ = parse_yaml(inside)
        return ok
    ok, _, _ = parse_yaml(text)
    return ok


def parse_toml(text: str) -> Tuple[bool, Any | None, str | None]:
    """Parse TOML from raw model output (expects TOML-only)."""
    t = strip_code_fences(text).strip()
    try:
        obj = tomllib.loads(t)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def is_toml_only(text: str) -> bool:
    """Heuristic: prefer fenced-block-only outputs when fences are used.

    For formats like YAML/TOML/XML/CSV, extra natural-language text can still be
    syntactically valid (e.g., YAML treats it as a scalar/mapping). If the model
    outputs a fenced block, we require that *nothing else* is present outside it.
    Otherwise we fall back to parsing the full (stripped) output.
    """
    inside, outside = extract_first_fenced_block(text)
    if inside is not None:
        if outside.strip() != "":
            return False
        ok, _, _ = parse_toml(inside)
        return ok
    ok, _, _ = parse_toml(text)
    return ok


def parse_xml(text: str) -> Tuple[bool, Any | None, str | None]:
    """Parse XML from raw model output (expects XML-only).

    Returns the root Element.
    """
    t = strip_code_fences(text).strip()
    try:
        root = ET.fromstring(t)
        return True, root, None
    except Exception as e:
        return False, None, str(e)


def is_xml_only(text: str) -> bool:
    """Heuristic: prefer fenced-block-only outputs when fences are used.

    For formats like YAML/TOML/XML/CSV, extra natural-language text can still be
    syntactically valid (e.g., YAML treats it as a scalar/mapping). If the model
    outputs a fenced block, we require that *nothing else* is present outside it.
    Otherwise we fall back to parsing the full (stripped) output.
    """
    inside, outside = extract_first_fenced_block(text)
    if inside is not None:
        if outside.strip() != "":
            return False
        ok, _, _ = parse_xml(inside)
        return ok
    ok, _, _ = parse_xml(text)
    return ok


def parse_csv(text: str) -> Tuple[bool, Any | None, str | None]:
    """Parse CSV from raw model output (expects CSV-only).

    Returns a list of rows (each row is a list[str]).
    """
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


def is_csv_only(text: str) -> bool:
    """Heuristic: prefer fenced-block-only outputs when fences are used.

    For formats like YAML/TOML/XML/CSV, extra natural-language text can still be
    syntactically valid (e.g., YAML treats it as a scalar/mapping). If the model
    outputs a fenced block, we require that *nothing else* is present outside it.
    Otherwise we fall back to parsing the full (stripped) output.
    """
    inside, outside = extract_first_fenced_block(text)
    if inside is not None:
        if outside.strip() != "":
            return False
        ok, _, _ = parse_csv(inside)
        return ok
    ok, _, _ = parse_csv(text)
    return ok


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

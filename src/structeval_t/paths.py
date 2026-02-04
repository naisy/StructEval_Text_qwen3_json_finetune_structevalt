from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


_TOKEN_RE = re.compile(r'''
    (?P<dot>\.)|
    (?P<star>\*)|
    (?P<bracket>\[\s*(?P<idx>\d+|\*|"[^"]+"|'[^']+')\s*\])|
    (?P<name>[A-Za-z_][A-Za-z0-9_\-]*)
''', re.VERBOSE)


@dataclass(frozen=True)
class PathToken:
    kind: str  # "key" | "index" | "wildcard"
    value: str | int | None


def parse_path(expr: str) -> list[PathToken]:
    """Parse a StructEval-style key path.

    Supported subset (sufficient for JSON tasks):
    - dot keys:      a.b.c
    - list index:    a.b[0].c
    - wildcard:      a.b[*].c  (any element satisfies the rest)
    - quoted keys:   a["weird.key"].b  or a['weird.key'].b
    """
    s = expr.strip()
    if not s:
        return []
    tokens: list[PathToken] = []
    i = 0
    last_was_dot = False
    while i < len(s):
        m = _TOKEN_RE.match(s, i)
        if not m:
            raise ValueError(f"Invalid path near: {s[i:i+20]!r} in {expr!r}")
        if m.group("dot"):
            last_was_dot = True
        elif m.group("star"):
            # dot-wildcard form: a.*.b
            tokens.append(PathToken("wildcard", None))
            last_was_dot = False
        elif m.group("name"):
            tokens.append(PathToken("key", m.group("name")))
            last_was_dot = False
        elif m.group("bracket"):
            raw = m.group("idx")
            if raw is None:
                raise ValueError(f"Invalid bracket token in {expr!r}")
            raw = raw.strip()
            if raw == "*":
                tokens.append(PathToken("wildcard", None))
            elif raw.isdigit():
                tokens.append(PathToken("index", int(raw)))
            else:
                # quoted key
                if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                    key = raw[1:-1]
                    tokens.append(PathToken("key", key))
                else:
                    raise ValueError(f"Unsupported bracket token: {raw!r} in {expr!r}")
            last_was_dot = False
        i = m.end()
    return tokens


def _traverse_once(obj: Any, tok: PathToken) -> Iterable[Any]:
    if tok.kind == "key":
        if isinstance(obj, dict) and tok.value in obj:
            yield obj[tok.value]  # type: ignore[index]
        return
    if tok.kind == "index":
        if isinstance(obj, list) and isinstance(tok.value, int) and 0 <= tok.value < len(obj):
            yield obj[tok.value]
        return
    if tok.kind == "wildcard":
        if isinstance(obj, list):
            for x in obj:
                yield x
        elif isinstance(obj, dict):
            # wildcard over object properties: a.*.b
            for v in obj.values():
                yield v
        return
    raise ValueError(f"Unknown token kind: {tok.kind}")


def exists_path(obj: Any, expr: str) -> bool:
    """Return True if the path exists in the JSON object.

    Wildcard semantics: any element must satisfy the remainder path.
    """
    toks = parse_path(expr)
    if not toks:
        return False

    frontier = [obj]
    for j, tok in enumerate(toks):
        next_frontier: list[Any] = []
        for node in frontier:
            for nxt in _traverse_once(node, tok):
                next_frontier.append(nxt)
        if not next_frontier:
            return False
        frontier = next_frontier
    return True

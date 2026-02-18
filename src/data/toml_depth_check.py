"""Detect deep hierarchy in TOML strings.

This is a lightweight heuristic used for dataset selection / distribution checks.

What counts as "deep"?
---------------------
We look for dotted paths that imply nesting:
  - Table headers: [a.b.c] or [[a.b.c]]
  - Dotted-key assignments: a.b.c = ...

Depth is defined as the number of dot-separated segments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


# [a.b.c]   or   [[a.b.c]]
RE_TABLE_HEADER = re.compile(r"^\s*(\[\[|\[)\s*([^\[\]]+?)\s*(\]\]|\])\s*$", re.M)

# a.b.c = ... (requires at least one dot)
# - keys limited to bare keys [A-Za-z0-9_-] for simplicity (fits most synthetic data)
RE_DOTTED_ASSIGN = re.compile(r"^\s*([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+)\s*=", re.M)

# Very simple comment stripper (not TOML-perfect; good enough for distribution checks)
RE_LINE_COMMENT = re.compile(r"(^|[^\\])#.*$", re.M)


@dataclass
class TomlDepthReport:
    max_depth: int
    deep_paths: List[Tuple[str, int]]  # (path, depth) with depth >= threshold
    all_paths: List[Tuple[str, int]]   # (path, depth) for inspection


def _path_depth(path: str) -> int:
    path = path.strip()
    path = re.sub(r"\s*\.\s*", ".", path)  # collapse spaces around dots
    parts = [p for p in path.split(".") if p]
    return len(parts)


def detect_toml_deep_hierarchy(toml_text: str, *, threshold: int = 3) -> TomlDepthReport:
    """Detect whether TOML text contains deep hierarchy (depth >= threshold)."""
    if threshold < 2:
        raise ValueError("threshold must be >= 2")

    text = RE_LINE_COMMENT.sub(r"\1", toml_text)
    paths: List[Tuple[str, int]] = []

    # Table headers
    for m in RE_TABLE_HEADER.finditer(text):
        raw_path = m.group(2)
        # Skip quoted keys / bracket-keys etc. (keep it explicit)
        if '"' in raw_path or "'" in raw_path:
            continue
        d = _path_depth(raw_path)
        paths.append((raw_path.strip(), d))

    # Dotted assignments
    for m in RE_DOTTED_ASSIGN.finditer(text):
        raw_path = m.group(1)
        d = _path_depth(raw_path)
        paths.append((raw_path.strip(), d))

    max_depth = max([d for _, d in paths], default=0)
    deep_paths = [(p, d) for p, d in paths if d >= threshold]

    return TomlDepthReport(max_depth=max_depth, deep_paths=deep_paths, all_paths=paths)


def is_toml_deep(toml_text: str, *, min_depth: int) -> bool:
    """Convenience helper for filtering."""
    rep = detect_toml_deep_hierarchy(toml_text, threshold=min_depth)
    return rep.max_depth >= min_depth

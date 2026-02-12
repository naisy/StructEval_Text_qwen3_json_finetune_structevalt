#!/usr/bin/env python3
"""Preview JSONL rows as *decoded* strings.

Why this exists
--------------
JSONL stores fields like `output` as JSON strings, so quotes inside the
structured text are escaped in the file (e.g. `\"`).

If you open the file in an editor, it can look like TOML/YAML is "broken",
but once you decode the JSON (`json.loads`), the content is the correct
structured text (no backslashes).

Usage:
  python scripts/preview_jsonl.py data/valid_hf_sft.jsonl --n 3
  python scripts/preview_jsonl.py data/valid_hf_sft.jsonl --idx 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="JSONL path")
    ap.add_argument("--n", type=int, default=1, help="print first N rows")
    ap.add_argument("--idx", type=int, default=-1, help="print only this 0-based row index")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    def _iter_rows():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    if args.idx >= 0:
        for i, row in enumerate(_iter_rows()):
            if i == args.idx:
                print(f"[row {i}] keys={list(row.keys())}")
                print("query:")
                print(row.get("query", ""))
                print("\noutput:")
                print(row.get("output", ""))
                return 0
        raise SystemExit(f"Index out of range: {args.idx}")

    for i, row in enumerate(_iter_rows()):
        if i >= args.n:
            break
        print(f"\n[row {i}] keys={list(row.keys())}")
        print("query:")
        print(row.get("query", ""))
        print("\noutput:")
        print(row.get("output", ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

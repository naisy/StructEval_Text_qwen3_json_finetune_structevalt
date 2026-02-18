from __future__ import annotations

"""Extract deep TOML examples from an SFT JSONL file.

This is used to "guarantee inclusion" of rare deep-hierarchy TOML examples when
training from Hugging Face datasets.

Workflow (HF SFT):
  1) import_hf_structured_sft -> data/hf_sft.jsonl
  2) (optional) extract deep TOML -> data/my_sft_dataset.jsonl
  3) hf_select_subset + prepare_sft_split
  4) append_extra_datasets appends data/my_sft_dataset.jsonl after balancing

The output JSONL keeps the same schema as the input SFT JSONL.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set

from src.data.toml_depth_check import is_toml_deep
from src.utils.logging import info


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            yield json.loads(s)


def _fingerprint(item: Dict[str, Any]) -> str:
    """Deterministic fingerprint for de-duplication."""
    query = str(item.get("query", ""))
    out = str(item.get("output", ""))
    ot = str(item.get("output_type", ""))
    payload = (ot + "\n" + query + "\n" + out).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def extract_deep_toml_items(
    items: Iterable[Dict[str, Any]], *, min_depth: int
) -> Iterable[Dict[str, Any]]:
    seen: Set[str] = set()
    for it in items:
        if str(it.get("output_type", "")).upper() != "TOML":
            continue
        out = str(it.get("output", ""))
        if not out:
            continue
        if not is_toml_deep(out, min_depth=min_depth):
            continue
        fp = _fingerprint(it)
        if fp in seen:
            continue
        seen.add(fp)
        yield it


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input SFT JSONL (e.g., data/hf_sft.jsonl)")
    ap.add_argument("--output", required=True, help="Output JSONL (e.g., data/my_sft_dataset.jsonl)")
    ap.add_argument("--min-depth", type=int, required=True, help="Keep TOML examples with depth >= this")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to output if exists (dedup by fingerprint). If neither --overwrite nor --append is set, existing output is kept and extraction is skipped.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if args.min_depth < 2:
        raise SystemExit("--min-depth must be >= 2")
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    if out_path.exists() and (not args.overwrite) and (not args.append):
        info(f"Deep TOML output already exists, skip: {out_path}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected = list(extract_deep_toml_items(_iter_jsonl(in_path), min_depth=args.min_depth))
    if args.append and out_path.exists() and (not args.overwrite):
        # Dedup against existing file
        existing_fps: Set[str] = set()
        for it in _iter_jsonl(out_path):
            existing_fps.add(_fingerprint(it))
        selected = [it for it in selected if _fingerprint(it) not in existing_fps]
        mode = "a"
    else:
        mode = "w"

    with out_path.open(mode, encoding="utf-8") as f:
        for it in selected:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    info(
        f"Wrote deep TOML extras: {out_path} added={len(selected)} (min_depth={args.min_depth})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

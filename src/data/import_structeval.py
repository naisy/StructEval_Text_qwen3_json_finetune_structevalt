from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def _ensure_deps() -> None:
    try:
        import datasets  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from e


def load_structeval(split: str = "test"):
    _ensure_deps()
    from datasets import get_dataset_split_names, load_dataset

    # NOTE:
    # As of Feb 2026, the Hugging Face dataset "TIGER-Lab/StructEval" publishes
    # only a single split named "test" (the parquet file is named train-*.parquet
    # but is still registered as split="test").
    # Users often expect "train" to exist, so we provide a backward-compatible
    # mapping train -> test when only test is available.
    try:
        split_names = set(get_dataset_split_names("TIGER-Lab/StructEval"))
    except Exception:
        # If split name lookup fails (offline, older datasets lib, etc.), fall
        # back to a direct load which will raise a clear error.
        split_names = set()

    requested = str(split).strip()
    if split_names:
        if requested not in split_names:
            if requested.lower() == "train" and "test" in split_names:
                print("[import_structeval] WARN: split 'train' not found; using 'test' instead.")
                requested = "test"
            else:
                raise ValueError(f"Unknown split '{split}'. Available splits: {sorted(split_names)}")

    return load_dataset("TIGER-Lab/StructEval", split=requested)


def _normalize_output_type(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().upper()


def filter_tasks(
    ds,
    *,
    allow_output_types: Optional[List[str]] = None,
    require_text_only: bool = True,
    require_non_rendering: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allowed = None
    if allow_output_types:
        allowed = {_normalize_output_type(x) for x in allow_output_types if str(x).strip()}
    for x in ds:
        ot = _normalize_output_type(x.get("output_type"))
        if allowed is not None and ot not in allowed:
            continue
        if require_text_only and x.get("input_type") not in ("Text", "text", None):
            continue
        if require_non_rendering and bool(x.get("rendering", False)):
            continue

        out.append(
            {
                # keep original fields for traceability
                "task_id": str(x.get("task_id", "")),
                "task_name": x.get("task_name", "StructEval"),
                "query": x.get("query", ""),
                "feature_requirements": x.get("feature_requirements", ""),
                "input_type": x.get("input_type", "Text"),
                "output_type": x.get("output_type", "JSON"),
                "query_example": x.get("query_example", ""),
                "VQA": x.get("VQA", []),
                "raw_output_metric": x.get("raw_output_metric", []) or [],
                "rendering": bool(x.get("rendering", False)),
            }
        )
    return out


def save_json(tasks: List[Dict[str, Any]], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Download StructEval and export tasks into a local JSON file (JSON array). "
            "By default, exports only JSON-output tasks to preserve backward compatibility."
        )
    )
    ap.add_argument("--split", default="test", help="Dataset split (default: test)")
    ap.add_argument("--out", default="data/structeval_json_eval.json", help="Output JSON path (JSON array).")
    ap.add_argument(
        "--output-types",
        default="JSON",
        help=(
            "Comma-separated list of allowed output types (e.g. JSON,YAML,TOML,XML,CSV). "
            "Default: JSON"
        ),
    )
    ap.add_argument("--text-only", action="store_true", default=True, help="Require input_type=Text (default: true)")
    ap.add_argument("--allow-non-text", action="store_true", help="Allow non-text input_type tasks.")
    ap.add_argument("--non-rendering", action="store_true", default=True, help="Require rendering=false (default: true)")
    ap.add_argument("--allow-rendering", action="store_true", help="Allow rendering=true tasks.")
    args = ap.parse_args(argv)

    require_text_only = not args.allow_non_text
    require_non_rendering = not args.allow_rendering

    output_types = [x.strip() for x in str(args.output_types).split(",") if x.strip()]

    ds = load_structeval(split=args.split)
    tasks = filter_tasks(
        ds,
        allow_output_types=output_types,
        require_text_only=require_text_only,
        require_non_rendering=require_non_rendering,
    )
    save_json(tasks, args.out)
    print(f"Wrote {len(tasks)} tasks (output_types={output_types}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

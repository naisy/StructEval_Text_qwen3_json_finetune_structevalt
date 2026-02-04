from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache import load_cache, save_cache, cache_path


def _reason_from_details(details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "notes": details.get("notes"),
        "failed_checks": details.get("failed_checks"),
        "breakdown": details.get("breakdown"),
    }


def refresh_structeval_eval_json(
    *,
    in_path: Path,
    out_path: Path,
    cache_dir: Path,
    provider: str,
    model: str,
) -> None:
    records: List[Dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array: {in_path}")

    updated = 0
    missing = 0

    for rec in records:
        judge = rec.get("judge")
        if not isinstance(judge, dict):
            continue

        cache_key = judge.get("cache_key") or (judge.get("details") or {}).get("_cache_key")
        if not cache_key:
            continue

        # Rebuild a minimal payload is not possible here; we load by direct key+path.
        cache_file = cache_dir / provider / model / f"{cache_key}.json"
        if not cache_file.exists():
            missing += 1
            continue

        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        # Backfill task_id into cache if needed
        task_id = str(rec.get("task_id", "")) if rec.get("task_id") is not None else ""
        if task_id and ("task_id" not in cached) and ("task_ids" not in cached):
            cached2 = dict(cached)
            cached2["task_id"] = task_id
            # We cannot recompute payload here (no payload), so just overwrite file directly.
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cached2, ensure_ascii=False, indent=2), encoding="utf-8")
            cached = cached2

        # Update judge fields
        score = float(cached.get("score", judge.get("score", 0.0) or 0.0))
        passed = bool(cached.get("passed", judge.get("passed", False) or False))
        details = {k: v for k, v in cached.items() if k != "_raw_text"}
        details["_cache_key"] = cache_key
        details["_cache_file"] = str(cache_file)
        judge.update(
            {
                "provider": provider,
                "model": model,
                "score": score,
                "passed": passed,
                "details": details,
                "cache_key": cache_key,
                "cache_file": str(cache_file),
                "cache_hit": True,
                "reason": _reason_from_details(details),
            }
        )
        updated += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"updated": updated, "missing_cache": missing, "out": str(out_path)}, ensure_ascii=False))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh outputs/eval/structeval_t_eval.json judge fields from cache files.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input structeval_t_eval.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Output path (can overwrite input)")
    ap.add_argument("--cache_dir", default="outputs/cache/judge", help="Judge cache root directory")
    ap.add_argument("--provider", default="openai", choices=["openai", "gemini"], help="Judge provider")
    ap.add_argument("--model", required=True, help="Judge model name (folder under cache/provider/)")
    args = ap.parse_args(argv)

    refresh_structeval_eval_json(
        in_path=Path(args.in_path),
        out_path=Path(args.out_path),
        cache_dir=Path(args.cache_dir),
        provider=args.provider,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

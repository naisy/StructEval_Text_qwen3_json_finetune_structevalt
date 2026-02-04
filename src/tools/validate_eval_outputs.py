"""CLI: validate per-sample generations in outputs/eval/structeval_t_eval.json.

This is a deterministic alternative to LLM judging for *format validity*.
It reuses src.data.validators for strict parsing of JSON/YAML/TOML/XML/CSV.

Usage:
  python -m src.tools.validate_eval_outputs \
    --input outputs/eval/structeval_t_eval.json \
    --show-failures 20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..data.validators import validate_output_format


def _load(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return [x for x in obj if isinstance(x, dict)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="outputs/eval/structeval_t_eval.json")
    ap.add_argument("--show-failures", type=int, default=10)
    ap.add_argument("--write", type=str, default="", help="Optional: write summary JSON")
    args = ap.parse_args()

    in_path = Path(args.input)
    records = _load(in_path)

    by_type: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "ok": 0, "fails": []})

    for r in records:
        out_t = str(r.get("output_type") or "").upper() or "UNKNOWN"
        gen = str(r.get("generation") or "")
        ok, _parsed, err = validate_output_format(out_t, gen)

        bucket = by_type[out_t]
        bucket["n"] += 1
        if ok:
            bucket["ok"] += 1
        else:
            bucket["fails"].append({
                "task_id": r.get("task_id"),
                "error": err,
                "snippet": gen[:200],
            })

    summary = {
        "input": str(in_path),
        "by_output_type": {
            k: {
                "n": v["n"],
                "parse_rate": (float(v["ok"]) / float(v["n"]) if v["n"] else 0.0),
                "fails": v["fails"],
            }
            for k, v in sorted(by_type.items())
        },
    }

    # Console output (compact)
    for k, v in sorted(by_type.items()):
        rate = (float(v["ok"]) / float(v["n"]) if v["n"] else 0.0)
        print(f"[{k}] n={v['n']} parse_ok={v['ok']} parse_rate={rate:.3f}")
        for f in v["fails"][: max(0, int(args.show_failures))]:
            tid = f.get("task_id")
            err = f.get("error")
            print(f"  - {tid}: {err}")

    if args.write:
        out_path = Path(args.write)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL at {path}:{ln}: {e}") from e


def _is_usable_row(row: dict, require_parse_ok: bool) -> bool:
    out = row.get("output")
    if not isinstance(out, str) or not out.strip():
        return False
    if not require_parse_ok:
        return True
    meta = row.get("meta") or {}
    ok = meta.get("parse_ok")
    if ok is None:
        ok = meta.get("json_parse_ok")
    # If the field is absent (e.g., human curated), accept.
    if ok is None:
        return True
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description="Split pseudo-SFT JSONL into train/valid JSONL.")
    ap.add_argument("--input", required=True, help="Input pseudo-SFT JSONL")
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--valid-out", required=True)
    ap.add_argument("--valid-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-invalid", action="store_true", help="Do not filter by meta.parse_ok (or meta.json_parse_ok)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    rows = [r for r in _iter_jsonl(inp) if _is_usable_row(r, require_parse_ok=(not args.allow_invalid))]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n = len(rows)
    n_valid = int(round(n * args.valid_ratio))
    n_valid = max(1, n_valid) if n > 1 else n
    valid = rows[:n_valid]
    train = rows[n_valid:]

    out_train = Path(args.train_out)
    out_valid = Path(args.valid_out)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_valid.parent.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, items: list[dict]):
        with path.open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(out_train, train)
    dump(out_valid, valid)

    print(f"Prepared SFT split: total={n} train={len(train)} valid={len(valid)}")
    print(f" train_out={out_train}")
    print(f" valid_out={out_valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

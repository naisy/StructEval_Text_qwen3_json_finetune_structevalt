from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, List, Dict

from src.utils.logging import info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", required=True, help="Input StructEval-T JSON (array)")
    p.add_argument("--out-train", required=True)
    p.add_argument("--out-valid", required=True)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array of tasks")
    rng = random.Random(args.seed)
    rng.shuffle(data)
    n = len(data)
    n_valid = max(1, int(n * args.valid_ratio)) if n > 1 else 1
    valid = data[:n_valid]
    train = data[n_valid:]
    Path(args.out_train).write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")
    Path(args.out_valid).write_text(json.dumps(valid, ensure_ascii=False), encoding="utf-8")
    info(f"Prepared StructEval split: total={n} train={len(train)} valid={len(valid)}")
    info(f" train_out={args.out_train}")
    info(f" valid_out={args.out_valid}")


if __name__ == "__main__":
    main()

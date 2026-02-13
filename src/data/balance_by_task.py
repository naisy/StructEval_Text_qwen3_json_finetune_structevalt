"""Balance datasets by task_key (to reduce output-format bias).

This script is intentionally small and dependency-light:
- Reads either JSONL (SFT) or JSON array (StructEval-style tasks for GRPO)
- Groups by `task_key` (produced by `import_hf_structured_sft.py`)
- Samples a balanced subset using a deterministic seed

Why:
HF mixtures often over-represent JSON-like tasks. That causes "format leakage":
- YAML indentation drift toward JSON-ish spacing / braces
- TOML sparsity -> higher syntax error rate

We keep the original examples unchanged; we only *select* a subset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL at {path}:{ln}: {e}") from e


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected JSON array: {path}")
    out: List[Dict[str, Any]] = []
    for i, x in enumerate(data):
        if not isinstance(x, dict):
            raise RuntimeError(f"Item {i} is not an object: {path}")
        out.append(x)
    return out


def _task_key(ex: Dict[str, Any]) -> str:
    # Prefer precomputed key.
    k = ex.get("task_key")
    if isinstance(k, str) and k.strip():
        return k.strip()
    # Fallback: at least separate by output_type.
    ot = ex.get("output_type") or "UNKNOWN"
    return f"unknown|{str(ot).strip().upper()}|unknown|unknown"


def _group_by_task(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in items:
        groups[_task_key(ex)].append(ex)
    return groups


def _filter_rare_tasks(
    groups: Dict[str, List[Dict[str, Any]]], *, min_count: int
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    if min_count <= 1:
        return groups, {}
    kept: Dict[str, List[Dict[str, Any]]] = {}
    dropped: Dict[str, int] = {}
    for k, v in groups.items():
        if len(v) >= min_count:
            kept[k] = v
        else:
            dropped[k] = len(v)
    return kept, dropped


def _balanced_sample(
    groups: Dict[str, List[Dict[str, Any]]],
    *,
    strategy: str,
    fixed_n: int | None,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Return (sampled_items, per_task_n_used)."""
    rng = random.Random(seed)
    keys = sorted(groups.keys())
    counts = {k: len(groups[k]) for k in keys}
    if not counts:
        return [], {}
    min_n = min(counts.values())
    max_n = max(counts.values())

    if strategy == "min":
        n_each = min_n
    elif strategy == "max":
        n_each = max_n
    elif strategy.startswith("fixed"):
        if fixed_n is None or fixed_n <= 0:
            raise ValueError("fixed strategy requires --fixed-n > 0")
        n_each = fixed_n
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    sampled: List[Dict[str, Any]] = []
    per_task_used: Dict[str, int] = {}
    for k in keys:
        pool = list(groups[k])
        rng.shuffle(pool)
        if n_each <= len(pool):
            take = pool[:n_each]
        else:
            # Repeat with replacement when asked for more than available.
            take = [pool[i % len(pool)] for i in range(n_each)]
        sampled.extend(take)
        per_task_used[k] = len(take)

    rng.shuffle(sampled)
    return sampled, per_task_used


def _estimate_steps(num_examples: int, *, per_device_bs: int, grad_accum: int, epochs: float | None) -> Dict[str, int]:
    eff = max(1, int(per_device_bs) * max(1, int(grad_accum)))
    steps_per_epoch = int(math.ceil(num_examples / eff))
    out = {"effective_batch": eff, "steps_per_epoch": steps_per_epoch}
    if epochs is not None:
        out["total_steps"] = int(math.ceil(steps_per_epoch * float(epochs)))
    return out


def _print_report(
    groups: Dict[str, List[Dict[str, Any]]],
    *,
    label: str,
    per_device_bs: int | None,
    grad_accum: int | None,
    epochs: float | None,
    max_steps: int | None,
) -> None:
    counts = {k: len(v) for k, v in groups.items()}
    if not counts:
        print(f"[{label}] No items.")
        return
    min_n = min(counts.values())
    max_n = max(counts.values())
    n_tasks = len(counts)
    total = sum(counts.values())

    print(f"[{label}] tasks={n_tasks} total_examples={total}")
    print(f"[{label}] min_per_task={min_n} max_per_task={max_n}")

    def _steps_for(n_each: int) -> int | None:
        if per_device_bs is None or grad_accum is None:
            return None
        n = n_each * n_tasks
        st = _estimate_steps(n, per_device_bs=per_device_bs, grad_accum=grad_accum, epochs=epochs)
        return st.get("total_steps") if epochs is not None else st.get("steps_per_epoch")

    if per_device_bs is not None and grad_accum is not None:
        eff = per_device_bs * grad_accum
        print(f"[{label}] effective_batch={eff} (per_device_train_batch_size={per_device_bs} * grad_accum={grad_accum})")
        if epochs is not None:
            st_min = _estimate_steps(min_n * n_tasks, per_device_bs=per_device_bs, grad_accum=grad_accum, epochs=epochs)
            st_max = _estimate_steps(max_n * n_tasks, per_device_bs=per_device_bs, grad_accum=grad_accum, epochs=epochs)
            print(
                f"[{label}] steps_if_min_strategy: examples={min_n*n_tasks} steps/epoch={st_min['steps_per_epoch']} total_steps(epochs={epochs})={st_min['total_steps']}"
            )
            print(
                f"[{label}] steps_if_max_strategy: examples={max_n*n_tasks} steps/epoch={st_max['steps_per_epoch']} total_steps(epochs={epochs})={st_max['total_steps']}"
            )
        else:
            st_min = _estimate_steps(min_n * n_tasks, per_device_bs=per_device_bs, grad_accum=grad_accum, epochs=None)
            st_max = _estimate_steps(max_n * n_tasks, per_device_bs=per_device_bs, grad_accum=grad_accum, epochs=None)
            print(
                f"[{label}] steps_per_epoch_if_min_strategy: examples={min_n*n_tasks} steps/epoch={st_min['steps_per_epoch']}"
            )
            print(
                f"[{label}] steps_per_epoch_if_max_strategy: examples={max_n*n_tasks} steps/epoch={st_max['steps_per_epoch']}"
            )

    if max_steps is not None:
        print(f"[{label}] config_max_steps={max_steps}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Balance a dataset by task_key.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--input-format", choices=["jsonl", "json"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strategy", default="min", choices=["min", "max", "fixed"], help="min|max|fixed")
    ap.add_argument("--fixed-n", type=int, default=None, help="Used when --strategy=fixed")
    ap.add_argument("--seed", type=int, default=42)
    # Optional training-context numbers for step estimation (report only)
    ap.add_argument("--per-device-train-batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--min-count",
        type=int,
        default=8,
        help="Drop task_key groups with fewer than this many examples before balancing (default: 8)",
    )
    ap.add_argument("--report-only", action="store_true", help="Only print stats; do not write output")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    if args.input_format == "jsonl":
        items = list(_iter_jsonl(inp))
    else:
        items = _load_json_array(inp)

    groups = _group_by_task(items)
    _print_report(
        groups,
        label=inp.name,
        per_device_bs=args.per_device_train_batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )

    groups_kept, dropped = _filter_rare_tasks(groups, min_count=args.min_count)
    if dropped:
        dropped_total = sum(dropped.values())
        print(
            f"[{inp.name}] dropping_rare_task_groups: groups_dropped={len(dropped)} examples_dropped={dropped_total} min_count={args.min_count}"
        )

    if not groups_kept:
        raise SystemExit(
            f"After dropping rare task groups (min_count={args.min_count}), no data remains."
        )

    if args.report_only:
        return 0

    sampled, per_task_used = _balanced_sample(
        groups_kept, strategy=args.strategy, fixed_n=args.fixed_n, seed=args.seed
    )
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if args.input_format == "jsonl":
        with outp.open("w", encoding="utf-8") as f:
            for r in sampled:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        outp.write_text(json.dumps(sampled, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote balanced dataset: {outp} items={len(sampled)} strategy={args.strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

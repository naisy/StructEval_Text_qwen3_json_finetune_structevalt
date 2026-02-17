"""Select / balance a subset of Hugging Face-derived datasets for SFT/GRPO.

Motivation
- HF mixtures can be heavily skewed toward JSON-like tasks.
- We sometimes want the original "balanced by task_key" subset.
- Other times we want to *intentionally* upweight a specific output format
  (e.g., train on *all* TOML to reduce syntax failures).

This tool reads sampling settings from configs/sft_hf.yaml or configs/grpo_hf.yaml
and writes a selected dataset file.

Design constraints
- We NEVER modify or synthesize HF examples here. We only *select* items.
- Works for both:
  - SFT JSONL (one example per line)
  - GRPO StructEval-style tasks JSON (a JSON array)

Config
data:
  sampling:
    mode: balance_by_task | per_output_type | none

    balance_by_task:
      enabled: true
      strategy: min|max|fixed
      fixed_n: null
      seed: 42
      min_count: 8

    per_output_type:
      seed: 42
      shuffle: true
      oversample_with_replacement: false
      default_target: null   # null => keep all for unspecified formats
      targets:
        TOML: all            # or an integer
        JSON: 1000

Env overrides (backward compatible)
- HF_BALANCE_BY_TASK, HF_BALANCE_STRATEGY, HF_BALANCE_FIXED_N, HF_BALANCE_SEED, HF_BALANCE_MIN_COUNT
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from src.data import balance_by_task as bbt


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _norm_output_type(x: Any) -> str:
    if not isinstance(x, str):
        return "UNKNOWN"
    s = x.strip().upper()
    return s if s else "UNKNOWN"


def _get_sampling_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    samp = data.get("sampling") if isinstance(data.get("sampling"), dict) else {}
    return samp


def _read_items(input_path: Path, *, input_format: str) -> List[Dict[str, Any]]:
    if input_format == "jsonl":
        return list(bbt._iter_jsonl(input_path))  # noqa: SLF001 (intentional reuse)
    return bbt._load_json_array(input_path)  # noqa: SLF001


def _write_items(output_path: Path, *, items: List[Dict[str, Any]], output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        output_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def _group_by_output_type(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in items:
        ot = _norm_output_type(ex.get("output_type"))
        # Fallback: infer from task_key if available ("...|TOML|...")
        if ot == "UNKNOWN":
            tk = ex.get("task_key")
            if isinstance(tk, str) and "|" in tk:
                parts = [p.strip() for p in tk.split("|")]
                if len(parts) >= 2:
                    ot = _norm_output_type(parts[1])
        groups[ot].append(ex)
    return groups


def _parse_target(v: Any) -> Tuple[str, int | None]:
    """Return (mode, n).

    mode:
      - "all": keep all
      - "n": keep n
    """
    if v is None:
        return "all", None
    if isinstance(v, str) and v.strip().lower() in {"all", "*", "-1"}:
        return "all", None
    if isinstance(v, bool):
        return "all", None
    if isinstance(v, (int, float)):
        n = int(v)
        if n < 0:
            return "all", None
        return "n", n
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return "n", int(s)
    raise ValueError(f"Invalid target value: {v!r} (use 'all' or an integer)")


def _sample_per_output_type(
    items: List[Dict[str, Any]],
    *,
    targets: Dict[str, Any],
    default_target: Any,
    seed: int,
    shuffle: bool,
    oversample_with_replacement: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    groups = _group_by_output_type(items)

    used: Dict[str, int] = {}
    sampled: List[Dict[str, Any]] = []

    for ot in sorted(groups.keys()):
        pool = list(groups[ot])
        rng.shuffle(pool)

        tgt_v = targets.get(ot) if isinstance(targets, dict) else None
        if tgt_v is None:
            tgt_v = default_target

        mode, n = _parse_target(tgt_v)
        if mode == "all":
            take = pool
        else:
            assert n is not None
            if n <= len(pool):
                take = pool[:n]
            else:
                if oversample_with_replacement and len(pool) > 0:
                    take = [pool[i % len(pool)] for i in range(n)]
                else:
                    take = pool
        sampled.extend(take)
        used[ot] = len(take)

    if shuffle:
        rng.shuffle(sampled)
    return sampled, used


def _print_output_type_report(items: List[Dict[str, Any]], *, label: str) -> None:
    groups = _group_by_output_type(items)
    cnt = {k: len(v) for k, v in groups.items()}
    total = sum(cnt.values())
    print(f"[{label}] total_examples={total}")
    for k in sorted(cnt.keys()):
        print(f"[{label}] {k}: {cnt[k]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Select subset for HF SFT/GRPO datasets based on config.")
    ap.add_argument("--stage", choices=["sft", "grpo"], required=True)
    ap.add_argument("--config", required=True, help="Path to configs/sft_hf.yaml or configs/grpo_hf.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--input-format", choices=["jsonl", "json"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output-format", choices=["jsonl", "json"], required=True)

    # Context numbers for balance_by_task report only.
    ap.add_argument("--per-device-train-batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)

    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    samp = _get_sampling_cfg(cfg)
    mode = str(samp.get("mode", "balance_by_task")).strip().lower()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    items = _read_items(inp, input_format=args.input_format)

    # Report before selection.
    _print_output_type_report(items, label=f"{inp.name}:before")

    if mode in {"none", "off", "disabled"}:
        selected = items
        print(f"INFO  sampling.mode={mode} -> no subset selection")

    elif mode in {"balance_by_task", "balanced", "task_key"}:
        # Read from config, but allow old env vars to override.
        bcfg = samp.get("balance_by_task") if isinstance(samp.get("balance_by_task"), dict) else {}

        enabled_cfg = bcfg.get("enabled", True)
        enabled_env = os.getenv("HF_BALANCE_BY_TASK")
        enabled = enabled_cfg
        if enabled_env is not None:
            enabled = enabled_env != "0"

        if not enabled:
            selected = items
            print("INFO  balance_by_task disabled -> using unbalanced dataset")
        else:
            strategy = str(bcfg.get("strategy", "min")).strip().lower()
            strategy = os.getenv("HF_BALANCE_STRATEGY", strategy)

            fixed_n = bcfg.get("fixed_n")
            fixed_env = os.getenv("HF_BALANCE_FIXED_N")
            if fixed_env is not None and fixed_env.strip() != "":
                fixed_n = int(fixed_env)

            seed = int(os.getenv("HF_BALANCE_SEED", str(bcfg.get("seed", 42))))
            min_count = int(os.getenv("HF_BALANCE_MIN_COUNT", str(bcfg.get("min_count", 8))))

            groups = bbt._group_by_task(items)  # noqa: SLF001
            bbt._print_report(  # noqa: SLF001
                groups,
                label=f"{inp.name}:task_key",
                per_device_bs=args.per_device_train_batch_size,
                grad_accum=args.grad_accum,
                epochs=args.epochs,
                max_steps=args.max_steps,
            )

            groups_kept, dropped = bbt._filter_rare_tasks(groups, min_count=min_count)  # noqa: SLF001
            if dropped:
                print(
                    f"[{inp.name}:task_key] dropping_rare_task_groups: groups_dropped={len(dropped)} examples_dropped={sum(dropped.values())} min_count={min_count}"
                )

            selected, _per_task_used = bbt._balanced_sample(  # noqa: SLF001
                groups_kept,
                strategy=strategy,
                fixed_n=fixed_n,
                seed=seed,
            )
            print(
                f"INFO  sampling.mode=balance_by_task enabled=1 strategy={strategy} fixed_n={fixed_n} seed={seed} min_count={min_count} -> selected={len(selected)}"
            )

    elif mode in {"per_output_type", "output_type", "format"}:
        pcfg = samp.get("per_output_type") if isinstance(samp.get("per_output_type"), dict) else {}
        seed = int(pcfg.get("seed", 42))
        shuffle = bool(pcfg.get("shuffle", True))
        oversample = bool(pcfg.get("oversample_with_replacement", False))
        default_target = pcfg.get("default_target", None)
        targets = pcfg.get("targets") if isinstance(pcfg.get("targets"), dict) else {}
        # Normalize keys to upper.
        targets_norm = {str(k).strip().upper(): v for k, v in targets.items()}

        selected, used = _sample_per_output_type(
            items,
            targets=targets_norm,
            default_target=default_target,
            seed=seed,
            shuffle=shuffle,
            oversample_with_replacement=oversample,
        )
        print(
            f"INFO  sampling.mode=per_output_type seed={seed} shuffle={int(shuffle)} oversample_with_replacement={int(oversample)} -> selected={len(selected)}"
        )
        print("INFO  per_output_type used_counts=" + json.dumps(used, ensure_ascii=False))

    else:
        raise SystemExit(f"Unknown sampling.mode: {mode}")

    _print_output_type_report(selected, label=f"{Path(args.output).name}:after")

    _write_items(Path(args.output), items=selected, output_format=args.output_format)
    print(f"Wrote selected dataset: {args.output} items={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

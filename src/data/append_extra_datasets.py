"""Append user-provided local datasets to the prepared HF train/valid splits.

Why this exists
---------------
The HF training scripts (`scripts/run_sft_hf.sh`, `scripts/run_grpo_hf.sh`) build
`data/train_hf_*.{jsonl,json}` and `data/valid_hf_*.{jsonl,json}`.

The project also supports adding *local* datasets after the HF balancing step
so that:
  - HF sampling/balancing behavior remains unchanged.
  - User datasets can focus on missing patterns (e.g. TOML inline tables,
    XML escaping like '&' -> '&amp;').

Configuration
-------------
Enable and list extras under the top-level `data.*` in:
  - `configs/sft_hf.yaml`  (stage='sft')
  - `configs/grpo_hf.yaml` (stage='grpo')

Minimal (recommended) form:

  data:
    use_extra_datasets: true
    extra_datasets:
      - data/my_extra_1.jsonl
      - data/my_extra_2.jsonl
    extra_split:
      valid_ratio: 0.1
      seed: 42

Each file is split into train/valid and appended to the prepared HF splits.

Each entry may be either:

1) Pre-split (recommended):

  - use: true
    format: jsonl | structeval_json
    train_path: data/my_xxx_train.jsonl
    valid_path: data/my_xxx_valid.jsonl

2) Single file + split (convenience):

  - use: true
    format: jsonl | structeval_json
    path: data/my_xxx.jsonl
    split:
      valid_ratio: 0.1
      seed: 42

This module is intentionally format-preserving:
  - jsonl stays jsonl (SFT)
  - StructEval task arrays stay JSON arrays (GRPO)
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data.dataset import load_dataset_any
from src.utils.io import load_yaml
from src.utils.logging import info, warn


@dataclass
class ExtraSpec:
    fmt: str
    train_path: str | None
    valid_path: str | None
    path: str | None
    split_valid_ratio: float
    split_seed: int


def _infer_fmt_from_path(p: str) -> str:
    s = str(p).lower()
    if s.endswith(".jsonl"):
        return "jsonl"
    if s.endswith(".json"):
        return "structeval_json"
    # Default: treat as jsonl (common for SFT); stage validation will catch mismatches.
    return "jsonl"


def _parse_extra_specs(cfg: dict, *, stage: str) -> List[ExtraSpec]:
    data = (cfg.get("data") or {})
    # Simple on/off switch for quick experiments.
    # If omitted, default is False (disabled).
    if not bool(data.get("use_extra_datasets", False)):
        return []

    default_split = data.get("extra_split") or {}
    if not isinstance(default_split, dict):
        raise ValueError("data.extra_split must be a mapping if provided")
    default_valid_ratio = float(default_split.get("valid_ratio", 0.1))
    default_seed = int(default_split.get("seed", 42))

    extras = data.get("extra_datasets")
    if not extras:
        return []
    if not isinstance(extras, list):
        raise ValueError("data.extra_datasets must be a list")

    # Convenience: if the whole list is exactly two string paths and looks like
    # a (train, valid) pair, treat it as a single pre-split entry.
    if (
        len(extras) == 2
        and all(isinstance(x, str) for x in extras)
        and ("train" in str(extras[0]).lower())
        and ("valid" in str(extras[1]).lower() or "val" in str(extras[1]).lower())
    ):
        p0, p1 = str(extras[0]), str(extras[1])
        fmt0 = _infer_fmt_from_path(p0)
        fmt1 = _infer_fmt_from_path(p1)
        if fmt0 != fmt1:
            raise ValueError("extra_datasets train/valid pair must have the same format")
        if stage == "sft" and fmt0 != "jsonl":
            raise ValueError("SFT extra datasets must be .jsonl")
        if stage == "grpo" and fmt0 != "structeval_json":
            raise ValueError("GRPO extra datasets must be JSON arrays (.json)")
        return [
            ExtraSpec(
                fmt=fmt0,
                train_path=p0,
                valid_path=p1,
                path=None,
                split_valid_ratio=0.1,
                split_seed=42,
            )
        ]

    out: List[ExtraSpec] = []
    for i, e in enumerate(extras):
        # Allow a shorthand string entry:
        #   extra_datasets: ["data/my_extra.jsonl"]
        # In this case we always split the file into train/valid.
        if isinstance(e, str):
            path = e
            fmt = _infer_fmt_from_path(path)
            if stage == "sft" and fmt != "jsonl":
                raise ValueError(
                    f"SFT extra dataset must be a .jsonl file, got {path}"
                )
            if stage == "grpo" and fmt != "structeval_json":
                raise ValueError(
                    f"GRPO extra dataset must be a JSON array file (.json), got {path}"
                )

            # Default split params
            out.append(
                ExtraSpec(
                    fmt=fmt,
                    train_path=None,
                    valid_path=None,
                    path=str(path),
                    split_valid_ratio=default_valid_ratio,
                    split_seed=default_seed,
                )
            )
            continue

        if not isinstance(e, dict):
            raise ValueError(f"data.extra_datasets[{i}] must be a mapping or a string path")
        if not bool(e.get("use", True)):
            continue

        fmt = str(e.get("format") or "").strip().lower()
        if not fmt:
            # If user omits format, infer from the provided path(s).
            fmt = _infer_fmt_from_path(e.get("path") or e.get("train_path") or "")
            if not fmt:
                raise ValueError(f"data.extra_datasets[{i}].format is required")
        if stage == "sft" and fmt != "jsonl":
            raise ValueError(
                f"SFT extra dataset must be format=jsonl, got {fmt} at index {i}"
            )
        if stage == "grpo" and fmt not in {"structeval_json", "structeval-t", "structeval_t"}:
            raise ValueError(
                f"GRPO extra dataset must be format=structeval_json, got {fmt} at index {i}"
            )

        train_path = e.get("train_path")
        valid_path = e.get("valid_path")
        path = e.get("path")

        split_cfg = e.get("split") or {}
        if not isinstance(split_cfg, dict):
            raise ValueError(f"data.extra_datasets[{i}].split must be a mapping")
        valid_ratio = float(split_cfg.get("valid_ratio", 0.1))
        seed = int(split_cfg.get("seed", 42))

        out.append(
            ExtraSpec(
                fmt=fmt,
                train_path=str(train_path) if train_path else None,
                valid_path=str(valid_path) if valid_path else None,
                path=str(path) if path else None,
                split_valid_ratio=valid_ratio,
                split_seed=seed,
            )
        )
    return out


def _split_items(items: List[Dict[str, Any]], *, valid_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be in (0,1), got {valid_ratio}")
    idx = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_valid = max(1, int(round(len(items) * valid_ratio)))
    valid_idx = set(idx[:n_valid])
    train = [items[i] for i in range(len(items)) if i not in valid_idx]
    valid = [items[i] for i in range(len(items)) if i in valid_idx]
    return train, valid


def _append_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _append_json_array(path: Path, items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    base: List[Dict[str, Any]] = []
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array at {path}")
        base = data
    base.extend(items)
    path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")


def append_extra_datasets(
    *,
    stage: str,
    config_path: str,
    train_path: str,
    valid_path: str,
) -> Dict[str, Any]:
    """Append extra datasets (configured in YAML) into the prepared split files."""
    stage = str(stage).strip().lower()
    if stage not in {"sft", "grpo"}:
        raise ValueError("stage must be 'sft' or 'grpo'")

    cfg = load_yaml(config_path)
    data_cfg = (cfg.get("data") or {})
    if not bool(data_cfg.get("use_extra_datasets", False)):
        info("Extra datasets disabled (data.use_extra_datasets=false).")
        return {"appended": 0, "entries": 0, "disabled": True}

    specs = _parse_extra_specs(cfg, stage=stage)
    if not specs:
        info("No extra datasets configured (data.extra_datasets empty).")
        return {"appended": 0, "entries": 0}

    appended_total = 0
    entry_reports: List[Dict[str, Any]] = []

    for s in specs:
        # Load items
        if s.train_path and s.valid_path:
            train_items = load_dataset_any(s.train_path, "jsonl" if stage == "sft" else "structeval_json")
            valid_items = load_dataset_any(s.valid_path, "jsonl" if stage == "sft" else "structeval_json")
            src = {"train_path": s.train_path, "valid_path": s.valid_path, "mode": "presplit"}
        elif s.path:
            all_items = load_dataset_any(s.path, "jsonl" if stage == "sft" else "structeval_json")
            train_items, valid_items = _split_items(
                all_items, valid_ratio=s.split_valid_ratio, seed=s.split_seed
            )
            src = {
                "path": s.path,
                "mode": "split",
                "valid_ratio": s.split_valid_ratio,
                "seed": s.split_seed,
            }
        else:
            raise ValueError("Each extra dataset entry must provide (train_path+valid_path) or path")

        # Append
        if stage == "sft":
            _append_jsonl(Path(train_path), train_items)
            _append_jsonl(Path(valid_path), valid_items)
        else:
            _append_json_array(Path(train_path), train_items)
            _append_json_array(Path(valid_path), valid_items)

        appended_total += len(train_items) + len(valid_items)
        entry_reports.append(
            {
                "source": src,
                "train_items": len(train_items),
                "valid_items": len(valid_items),
            }
        )

    info(
        f"Appended extra datasets: stage={stage} entries={len(entry_reports)} total_items_appended={appended_total}"
    )
    return {"appended": appended_total, "entries": len(entry_reports), "detail": entry_reports}


def main() -> int:
    ap = argparse.ArgumentParser(description="Append local extra datasets to prepared HF splits.")
    ap.add_argument("--stage", required=True, choices=["sft", "grpo"])
    ap.add_argument("--config", required=True, help="YAML config path (configs/sft_hf.yaml or configs/grpo_hf.yaml)")
    ap.add_argument("--train", required=True, help="Prepared train split path to append into")
    ap.add_argument("--valid", required=True, help="Prepared valid split path to append into")
    args = ap.parse_args()

    try:
        rep = append_extra_datasets(stage=args.stage, config_path=args.config, train_path=args.train, valid_path=args.valid)
    except Exception as e:
        warn(f"Failed to append extra datasets: {e}")
        raise
    # Print a compact machine-readable summary
    print(json.dumps(rep, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

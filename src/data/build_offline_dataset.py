"""Build a single offline dataset file from local files/directories.

This is used when:
  - configs/*_hf.yaml -> data.offline_dataset.use=true
  - configs/*_hf.yaml -> data.online_dataset.use=false

We intentionally do NOT modify examples beyond:
  - concatenating multiple files
  - optional filtering (e.g., toml_min_depth)

Stage formats
-------------
SFT:
  - input: one or more *.jsonl files
  - output: a single jsonl file (same schema as HF-imported jsonl)

GRPO:
  - input: one or more *.json files (each is a JSON array of tasks)
  - output: a single *.json file (JSON array)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from src.data.balance_by_task import _iter_jsonl, _load_json_array  # noqa: SLF001
from src.data.toml_depth_check import is_toml_deep


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _collect_files(inputs: List[str], *, suffix: str) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() == suffix:
                out.append(p)
            continue
        if p.is_dir():
            out.extend(sorted(p.rglob(f"*{suffix}")))
    # de-dup (stable)
    seen = set()
    uniq: List[Path] = []
    for p in out:
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def _offline_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    off = data.get("offline_dataset") if isinstance(data.get("offline_dataset"), dict) else {}
    return off


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json_array(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def _filter_sft_rows(rows: Iterable[Dict[str, Any]], *, toml_min_depth: int | None) -> List[Dict[str, Any]]:
    if toml_min_depth is None:
        return list(rows)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for r in rows:
        ot = str(r.get("output_type", "")).strip().upper()
        if ot == "TOML":
            out = r.get("output")
            if isinstance(out, str) and not is_toml_deep(out, min_depth=int(toml_min_depth)):
                dropped += 1
                continue
        kept.append(r)
    print(f"INFO  offline SFT filter: toml_min_depth={toml_min_depth} dropped={dropped} kept={len(kept)}")
    return kept


def build_sft(cfg_path: Path, out_path: Path) -> None:
    cfg = _load_yaml(cfg_path)
    off = _offline_cfg(cfg)
    datasets = off.get("datasets") if isinstance(off.get("datasets"), list) else []
    datasets = [str(x) for x in datasets if isinstance(x, str) and x.strip()]
    files = _collect_files(datasets, suffix=".jsonl")
    if not files:
        raise SystemExit("No offline SFT jsonl files found. Check data.offline_dataset.datasets")

    filt = off.get("filters") if isinstance(off.get("filters"), dict) else {}
    tmd = filt.get("toml_min_depth", None)
    tmd_int = None if tmd is None else int(tmd)

    rows: List[Dict[str, Any]] = []
    for p in files:
        rows.extend(list(_iter_jsonl(p)))
    rows = _filter_sft_rows(rows, toml_min_depth=tmd_int)
    _write_jsonl(out_path, rows)
    print(f"INFO  built offline SFT jsonl: {out_path} rows={len(rows)} sources={len(files)}")


def build_grpo(cfg_path: Path, out_path: Path) -> None:
    cfg = _load_yaml(cfg_path)
    off = _offline_cfg(cfg)
    datasets = off.get("datasets") if isinstance(off.get("datasets"), list) else []
    datasets = [str(x) for x in datasets if isinstance(x, str) and x.strip()]
    files = _collect_files(datasets, suffix=".json")
    if not files:
        raise SystemExit("No offline GRPO json files found. Check data.offline_dataset.datasets")

    tasks: List[Dict[str, Any]] = []
    for p in files:
        arr = _load_json_array(p)
        if not isinstance(arr, list):
            raise ValueError(f"Expected JSON array in {p}")
        tasks.extend(arr)

    _write_json_array(out_path, tasks)
    print(f"INFO  built offline GRPO tasks: {out_path} tasks={len(tasks)} sources={len(files)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build offline dataset files from local datasets.")
    ap.add_argument("--stage", choices=["sft", "grpo"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    out_path = Path(args.out)
    if args.stage == "sft":
        build_sft(cfg_path, out_path)
    else:
        build_grpo(cfg_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

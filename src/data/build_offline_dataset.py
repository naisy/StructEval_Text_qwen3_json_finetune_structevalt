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
from src.data.import_hf_structured_sft import (
    _extract_messages,
    _infer_output_type,
    _task_meta,
    _u10bei_output_from_metadata,
    _u10bei_prompt_from_metadata,
    build_query_text,
    extract_final_output,
    extract_attributes_from_prompt,
)


def _norm_output_type(x: Any) -> str:
    s = str(x or "").strip()
    return s.upper() if s else ""


def _is_u10bei(dataset_name: str) -> bool:
    return str(dataset_name or "").startswith("u-10bei/")


def _is_daichira(dataset_name: str) -> bool:
    return str(dataset_name or "").startswith("daichira/")


def _extract_prompt_and_output(row: Dict[str, Any]) -> tuple[str, str, str]:
    """Normalize one offline row into (query, output, output_type).

    Offline datasets in this project may be provided in either:
      A) "HF-imported SFT JSONL" schema (already normalized):
         {query, output, output_type, task_*...}
      B) "filtered_datasets" schema (raw-ish):
         {dataset, output_format, task_kind, task_name, messages, metadata, evaluation, ...}

    For (B), we apply the same *prompt selection policy* as HF import:
      - u-10bei/*: use metadata.prompt + metadata.output (messages are treated as logs)
      - others (daichira/*): use chat messages (user/assistant)
    """

    # Case A: already normalized.
    if isinstance(row.get("query"), str) and isinstance(row.get("output"), str):
        ot = _norm_output_type(row.get("output_type"))
        return row["query"].strip(), row["output"].strip(), ot

    dataset_name = str(row.get("dataset") or "offline")

    # Try to infer output type first.
    ot = _norm_output_type(row.get("output_type"))
    if not ot:
        ot = _norm_output_type(row.get("output_format"))
    if not ot:
        ot = _norm_output_type(_infer_output_type(row))

    sys_msg, user_msg, asst_msg = _extract_messages(row)

    # Prompt selection policy.
    prompt: str | None = None
    if _is_u10bei(dataset_name):
        prompt = _u10bei_prompt_from_metadata(row)
        if not prompt:
            # As in HF import: if metadata.prompt is missing, skip.
            return "", "", ot
    else:
        if not user_msg:
            return "", "", ot
        prompt = build_query_text(dataset_name, sys_msg, user_msg)

    # Output selection policy.
    out_text: str | None = None
    if _is_u10bei(dataset_name):
        out_text = _u10bei_output_from_metadata(row)
        if not out_text:
            return "", "", ot
    else:
        if not asst_msg:
            return "", "", ot
        out_text = asst_msg

    # Extract the pure structured payload.
    extracted = extract_final_output(out_text, ot) if out_text else ""
    return (prompt or "").strip(), extracted.strip(), ot


def _normalize_sft_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    query, output, out_type = _extract_prompt_and_output(row)
    if not query or not output:
        return None

    dataset_name = str(row.get("dataset") or "offline")

    # Build task classification fields (task_key, task_family, ...)
    # Reuse the same logic as HF import for consistency.
    user_msg_for_meta = query
    ex_for_meta: Dict[str, Any] = {}
    if isinstance(row.get("metadata"), dict):
        ex_for_meta["metadata"] = row.get("metadata")
    if _is_daichira(dataset_name):
        # HF import expects daichira fields (task, subcategory).
        ex_for_meta["task"] = str(row.get("task_kind") or "unknown")
        ex_for_meta["subcategory"] = str(row.get("task_name") or "unknown")
    tmeta = _task_meta(dataset_name, ex_for_meta, user_msg=user_msg_for_meta, output_type=out_type)

    meta: Dict[str, Any] = {}
    ev = row.get("evaluation")
    if isinstance(ev, dict):
        g = ev.get("grammar")
        if isinstance(g, dict) and "ok" in g:
            meta["parse_ok"] = bool(g.get("ok"))
        if "usable" in ev:
            meta["usable"] = bool(ev.get("usable"))

    out: Dict[str, Any] = {
        "query": query,
        "output": output,
        **tmeta,
    }
    if out_type:
        out["output_type"] = out_type
    if meta:
        out["meta"] = meta
    return out


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

    raw_rows: List[Dict[str, Any]] = []
    for p in files:
        raw_rows.extend(list(_iter_jsonl(p)))

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for r in raw_rows:
        nr = _normalize_sft_row(r)
        if nr is None:
            skipped += 1
            continue
        rows.append(nr)

    if skipped:
        print(f"INFO  offline SFT normalize: skipped={skipped} kept={len(rows)}")

    rows = _filter_sft_rows(rows, toml_min_depth=tmd_int)
    _write_jsonl(out_path, rows)
    print(f"INFO  built offline SFT jsonl: {out_path} rows={len(rows)} sources={len(files)}")


def build_grpo(cfg_path: Path, out_path: Path) -> None:
    cfg = _load_yaml(cfg_path)
    off = _offline_cfg(cfg)
    datasets = off.get("datasets") if isinstance(off.get("datasets"), list) else []
    datasets = [str(x) for x in datasets if isinstance(x, str) and x.strip()]
    json_files = _collect_files(datasets, suffix=".json")
    jsonl_files = _collect_files(datasets, suffix=".jsonl")
    if not json_files and not jsonl_files:
        raise SystemExit("No offline GRPO files found. Check data.offline_dataset.datasets")

    tasks: List[Dict[str, Any]] = []

    # Case A: already prepared StructEval-style task arrays.
    for p in json_files:
        arr = _load_json_array(p)
        if not isinstance(arr, list):
            raise ValueError(f"Expected JSON array in {p}")
        tasks.extend(arr)

    # Case B: local JSONL (same source as offline SFT). Convert to StructEval tasks.
    if jsonl_files:
        import hashlib

        def _stable_task_id(*, query: str, output_type: str, reference_output: str) -> str:
            base = f"{output_type}\n{query}\n{reference_output}".encode("utf-8")
            h = hashlib.md5(base).hexdigest()
            return f"offline_{h}"

        converted = 0
        for p in jsonl_files:
            for r in _iter_jsonl(p):
                nr = _normalize_sft_row(r)
                if nr is None:
                    continue
                q = str(nr.get("query") or "").strip()
                ref = str(nr.get("output") or "").strip()
                ot = _norm_output_type(nr.get("output_type") or "JSON") or "JSON"
                attrs = extract_attributes_from_prompt(q)
                tasks.append(
                    {
                        "task_id": _stable_task_id(query=q, output_type=ot, reference_output=ref),
                        "query": q,
                        "output_type": ot,
                        "reference_output": ref,
                        "raw_output_metric": attrs,
                        # Keep meta fields if present (helps analysis/debugging)
                        "task_key": nr.get("task_key"),
                        "task_family": nr.get("task_family"),
                        "task_kind": nr.get("task_kind"),
                        "task_name": nr.get("task_name"),
                        "schema": nr.get("schema"),
                        "complexity": nr.get("complexity"),
                    }
                )
                converted += 1
        if converted:
            print(f"INFO  converted offline JSONL -> GRPO tasks: {converted}")

    _write_json_array(out_path, tasks)
    print(
        f"INFO  built offline GRPO tasks: {out_path} tasks={len(tasks)} sources={len(json_files)} json + {len(jsonl_files)} jsonl"
    )


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

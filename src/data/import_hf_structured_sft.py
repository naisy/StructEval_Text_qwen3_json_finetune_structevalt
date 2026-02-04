from __future__ import annotations

"""Import Hugging Face SFT-style structured-data datasets into this project's formats.

Supported source formats (based on dataset viewer inspection):
- u-10bei/structured_data_with_cot_dataset* : example has `messages` (system/user/assistant) and `metadata` dict.
- daichira/structured-*-sft : example has `messages` (user/assistant) plus `category` like C_JSON/C_YAML/C_TOML/C_XML/C_CSV.

Outputs:
- JSONL for SFT training: each line has at minimum {query, output}. Optional fields: output_type.
- StructEval-T-style task JSON (array) for GRPO/eval: fields include {task_id, query, output_type, raw_output_metric, ...}.
  raw_output_metric is extracted from "ATTRIBUTES:" blocks in the user prompt when present.

This script does NOT generate any synthetic data. It only converts existing datasets.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.logging import info, warn


def _ensure_deps() -> None:
    try:
        import datasets  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'datasets'. Install with: pip install datasets") from e


def _normalize_output_type(x: Any) -> str:
    s = (str(x) if x is not None else "").strip().upper()
    # common variations
    if s in {"YML"}:
        return "YAML"
    if s.startswith("C_"):
        s = s[2:]
    return s


_OUTPUT_RE = re.compile(r"\bOutput\s*:\s*\n", re.IGNORECASE)


def extract_final_output(text: str) -> str:
    """Extract the final structured output from an assistant message.

    Many datasets include CoT or explanations like:
        Approach: ...
        Output:
        { ... }

    We take the substring after the last 'Output:' marker if present,
    otherwise return the whole string.
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""
    # Take content after the last Output: marker
    matches = list(_OUTPUT_RE.finditer(t))
    if matches:
        t = t[matches[-1].end():].strip()
    # Strip common code fences if any
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`\n").strip()
    return t


def _infer_output_type(ex: Dict[str, Any]) -> str:
    # u-10bei: metadata.format
    md = ex.get("metadata")
    if isinstance(md, dict) and "format" in md:
        return _normalize_output_type(md.get("format"))
    # daichira: category like C_TOML
    if "category" in ex:
        return _normalize_output_type(ex.get("category"))
    # try top-level fields
    if "format" in ex:
        return _normalize_output_type(ex.get("format"))
    if "output_type" in ex:
        return _normalize_output_type(ex.get("output_type"))
    return ""


def _extract_messages(ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (system, user, assistant) contents from a messages list."""
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None, None, None
    sys_msg = None
    user_msg = None
    asst_msg = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if role == "system":
            sys_msg = content
        elif role == "user":
            user_msg = content
        elif role == "assistant":
            asst_msg = content
    return sys_msg, user_msg, asst_msg


_ATTR_BLOCK_RE = re.compile(r"\bATTRIBUTES\s*:\s*(.*)", re.IGNORECASE)


def extract_attributes_from_prompt(prompt: str) -> List[str]:
    """Extract StructEval-T style key paths from an ATTRIBUTES: block in the prompt."""
    if not isinstance(prompt, str):
        return []
    lines = prompt.splitlines()
    attrs: List[str] = []
    i = 0
    while i < len(lines):
        m = _ATTR_BLOCK_RE.search(lines[i])
        if not m:
            i += 1
            continue
        # Case 1: same-line attributes
        rest = (m.group(1) or "").strip()
        collected: List[str] = []
        if rest:
            collected.append(rest)
        # collect subsequent lines until blank or a section marker
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                break
            if re.match(r"^(TEXT|JSON|XML|YAML|TOML|CSV)\s*:\s*", s, re.IGNORECASE):
                break
            collected.append(s)
            j += 1
        blob = " ".join(collected)
        # split by comma
        parts = [p.strip() for p in blob.split(",") if p.strip()]
        attrs.extend(parts)
        break
    # de-dup while keeping order
    seen = set()
    out = []
    for a in attrs:
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out


def load_hf(dataset_name: str, split: str) -> Iterable[Dict[str, Any]]:
    _ensure_deps()
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    return ds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True, help="HF dataset names (e.g. u-10bei/..., daichira/...)")
    p.add_argument("--split", default="train")
    p.add_argument("--out-sft-jsonl", default="data/hf_sft.jsonl")
    p.add_argument("--out-grpo-tasks", default="data/hf_grpo_tasks.json")
    p.add_argument("--write-grpo-tasks", action="store_true", help="Also write StructEval-T style tasks for GRPO/eval")
    p.add_argument("--max-rows-per-dataset", type=int, default=0, help="0 means no limit")
    p.add_argument("--shuffle-seed", type=int, default=0)
    args = p.parse_args()

    sft_out = Path(args.out_sft_jsonl)
    sft_out.parent.mkdir(parents=True, exist_ok=True)
    grpo_out = Path(args.out_grpo_tasks)
    grpo_out.parent.mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(args.shuffle_seed or 0)

    sft_rows: List[Dict[str, Any]] = []
    grpo_tasks: List[Dict[str, Any]] = []

    task_id = 1
    for name in args.datasets:
        info(f"[import_hf_structured_sft] Loading {name} split={args.split}...")
        ds = list(load_hf(name, args.split))
        if args.max_rows_per_dataset and args.max_rows_per_dataset > 0:
            ds = ds[: args.max_rows_per_dataset]
        if args.shuffle_seed:
            rng.shuffle(ds)

        for ex in ds:
            if not isinstance(ex, dict):
                continue
            sys_msg, user_msg, asst_msg = _extract_messages(ex)
            if not user_msg or not asst_msg:
                continue
            out_type = _infer_output_type(ex)

            out_text = extract_final_output(asst_msg)
            if not out_text:
                continue

            row = {
                "query": user_msg.strip(),
                "output": out_text,
            }
            if out_type:
                row["output_type"] = out_type
            sft_rows.append(row)

            if args.write_grpo_tasks:
                attrs = extract_attributes_from_prompt(user_msg)
                if attrs:
                    grpo_tasks.append(
                        {
                            "task_id": f"hf_{task_id:08d}",
                            "query": user_msg.strip(),
                            "feature_requirements": "",
                            "task_name": name,
                            "input_type": "Text",
                            "output_type": out_type or "JSON",
                            "query_example": "",
                            "VQA": [],
                            "raw_output_metric": attrs,
                            "rendering": False,
                        }
                    )
                    task_id += 1

    info(f"[import_hf_structured_sft] Writing SFT JSONL: {sft_out} rows={len(sft_rows)}")
    with sft_out.open("w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.write_grpo_tasks:
        info(f"[import_hf_structured_sft] Writing GRPO tasks JSON: {grpo_out} tasks={len(grpo_tasks)}")
        grpo_out.write_text(json.dumps(grpo_tasks, ensure_ascii=False), encoding="utf-8")

    info("Done.")


if __name__ == "__main__":
    main()

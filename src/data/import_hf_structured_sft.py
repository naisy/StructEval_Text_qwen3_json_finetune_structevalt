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

from src.data import validators as V
from src.data.hf_dataset_cleaning import decide_keep_example
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



_FENCE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)\n```", re.DOTALL)

def _strip_markdown_fences_anywhere(text: str) -> str:
    """
    If the text contains a fenced code block, return the *last* fenced block content.
    Otherwise return the original text.
    This helps remove preambles like:
        Sure! ... 
        ```xml
        <a/>
        ```
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""
    blocks = list(_FENCE_BLOCK_RE.finditer(t))
    if blocks:
        return blocks[-1].group(1).strip()
    return t


def _extract_json_substring(t: str) -> str:
    # Take the first JSON object/array by bracket matching.
    starts = [(t.find("{"), "{"), (t.find("["), "[")]
    starts = [(i, ch) for i, ch in starts if i != -1]
    if not starts:
        return ""
    i0, ch0 = min(starts, key=lambda x: x[0])
    end_ch = "}" if ch0 == "{" else "]"
    stack = []
    in_str = False
    esc = False
    for i in range(i0, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":  # escape
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    continue
                open_ch = stack.pop()
                if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                    # mismatch; bail
                    return ""
                if not stack:
                    return t[i0:i+1].strip()
    return ""


def _extract_xml_substring(t: str) -> str:
    # Basic heuristic: from first "<" to last ">".
    i0 = t.find("<")
    i1 = t.rfind(">")
    if i0 == -1 or i1 == -1 or i1 <= i0:
        return ""
    return t[i0:i1+1].strip()


def _strip_leading_explanations_lines(t: str, *, kind: str) -> str:
    lines = [ln.rstrip() for ln in t.splitlines()]
    if not lines:
        return ""
    def is_yaml_like(ln: str) -> bool:
        return (":" in ln and not ln.lstrip().startswith("#")) or ln.lstrip().startswith("- ")
    def is_toml_like(ln: str) -> bool:
        s = ln.lstrip()
        return s.startswith("[") and s.endswith("]") or ("=" in s and not s.startswith("#"))
    def is_csv_like(ln: str) -> bool:
        return "," in ln or "\t" in ln
    keep_from = 0
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if kind == "YAML" and is_yaml_like(s):
            keep_from = idx
            break
        if kind == "TOML" and is_toml_like(s):
            keep_from = idx
            break
        if kind == "CSV" and is_csv_like(s):
            keep_from = idx
            break
    return "\n".join(lines[keep_from:]).strip()


def extract_final_output(text: str, output_type: str | None = None) -> str:
    """Extract the final *structured* output from a dataset assistant message.

    HF datasets may include preambles / explanations / code fences. We try, in order:
    1) Take substring after the last 'Output:' marker.
    2) If code fences exist anywhere, take the last fenced block body.
    3) If output_type is known, extract the first well-formed blob (JSON/XML) or
       strip leading explanation lines (YAML/TOML/CSV).
    4) Fallback: return stripped text.
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""
    # 1) After Output:
    matches = list(_OUTPUT_RE.finditer(t))
    if matches:
        t = t[matches[-1].end():].strip()
    # 2) code fences anywhere
    t = _strip_markdown_fences_anywhere(t)

    ot = _normalize_output_type(output_type) if output_type else ""
    if ot == "JSON":
        j = _extract_json_substring(t)
        return j if j else t.strip()
    if ot == "XML":
        x = _extract_xml_substring(t)
        return x if x else t.strip()
    if ot in {"YAML", "TOML", "CSV"}:
        return _strip_leading_explanations_lines(t, kind=ot)
    return t.strip()
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


def _extract_reference_output(ex: Dict[str, Any], asst_msg: str | None, output_type: str | None) -> str:
    """Get the final structured output (gold) from an HF example.

    Priority:
    1) Some datasets include a clean gold output in metadata (e.g., v5 variants).
    2) Otherwise, parse the assistant message and extract the `Output:` section.
    """
    md = ex.get("metadata")
    if isinstance(md, dict):
        for k in ("output", "final_output", "answer", "gold", "target"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return extract_final_output(v, output_type)
    if isinstance(ex.get("output"), str) and str(ex.get("output") or "").strip():
        return extract_final_output(str(ex.get("output")), output_type)
    if asst_msg:
        return extract_final_output(asst_msg, output_type)
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



def _is_valid_strict_structured_output(text: str, output_type: str) -> bool:
    """Return True iff the structured payload can be parsed.

    IMPORTANT:
    - For *training data filtering*, we only want to drop examples that are
      clearly unusable (would become FAILED), i.e. the payload itself is
      syntactically invalid.
    - We must NOT drop merely because the raw answer contains extra text
      (preambles, "Output:", etc.). Those cases are often UNKNOWN/benign and
      should remain in the SFT pool.

    Therefore we parse the extracted payload and intentionally ignore whether
    there is text outside that payload.
    """
    t = _normalize_output_type(output_type) or "JSON"
    payload, _has_extraneous = V.extract_payload_and_extraneous(text, t)
    if t == "JSON":
        ok, _obj, _err = V.parse_json(payload)
        return bool(ok)
    if t == "YAML":
        ok, _obj, _err = V.parse_yaml(payload)
        return bool(ok)
    if t == "TOML":
        ok, _obj, _err = V.parse_toml(payload)
        return bool(ok)
    if t == "XML":
        ok, _obj, _err = V.parse_xml(payload)
        return bool(ok)
    if t == "CSV":
        ok, _obj, _err = V.parse_csv(payload)
        return bool(ok)
    # Unknown label: fall back to JSON strictness.
    ok, _obj, _err = V.parse_json(payload)
    return bool(ok)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True, help="HF dataset names (e.g. u-10bei/..., daichira/...)")
    p.add_argument("--split", default="train")
    p.add_argument("--out-sft-jsonl", default="data/hf_sft.jsonl")
    p.add_argument("--out-grpo-tasks", default="data/hf_grpo_tasks.json")
    p.add_argument("--write-grpo-tasks", action="store_true", help="Also write StructEval-T style tasks for GRPO/eval")
    p.add_argument(
        "--filter-invalid",
        action="store_true",
        help="Filter out examples whose extracted reference output fails strict parsing (recommended for SFT).",
    )
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

    filtered_invalid = 0
    filtered_invalid_by_type: dict[str, int] = {}
    filtered_invalid_by_reason: dict[str, int] = {}

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
            if not user_msg:
                continue
            out_type = _infer_output_type(ex)

            out_text = _extract_reference_output(ex, asst_msg, out_type)
            if not out_text:
                continue

            if args.filter_invalid and out_type:
                t = _normalize_output_type(out_type)
                # Stage 1: strict parsing (legacy behavior)
                if not _is_valid_strict_structured_output(out_text, t):
                    filtered_invalid += 1
                    filtered_invalid_by_type[t] = filtered_invalid_by_type.get(t, 0) + 1
                    filtered_invalid_by_reason["strict_parse_fail"] = filtered_invalid_by_reason.get("strict_parse_fail", 0) + 1
                    continue

                # Stage 2: deterministic cleaning policy (based on structeval_dataset_check.ipynb)
                dec = decide_keep_example(
                    prompt=user_msg.strip(),
                    output_type=t,
                    extracted_output=out_text,
                    raw_answer_text=asst_msg,
                )
                if not dec.keep:
                    filtered_invalid += 1
                    filtered_invalid_by_type[t] = filtered_invalid_by_type.get(t, 0) + 1
                    filtered_invalid_by_reason[dec.reason] = filtered_invalid_by_reason.get(dec.reason, 0) + 1
                    continue

            row = {
                "query": user_msg.strip(),
                "output": out_text,
            }
            if out_type:
                row["output_type"] = out_type
            sft_rows.append(row)

            if args.write_grpo_tasks:
                # Many allowed HF datasets (notably u-10bei/*) do NOT include StructEval-T
                # style "ATTRIBUTES:" blocks in the prompt.
                #
                # We still want GRPO tasks to be generated for those datasets; in that case
                # raw_output_metric becomes an empty list and the GRPO reward must fall back
                # to deterministic component rewards (e.g., parse/only) and/or gold matching.
                attrs = extract_attributes_from_prompt(user_msg)
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
                        # Gold output (Output-only; no CoT). Used for match-based reward.
                        "reference_output": out_text,
                        "rendering": False,
                    }
                )
                task_id += 1

    info(f"[import_hf_structured_sft] Writing SFT JSONL: {sft_out} rows={len(sft_rows)}")
    if args.filter_invalid and filtered_invalid:
        info(
            "[import_hf_structured_sft] Filtered invalid (strict-parse) examples: "
            f"{filtered_invalid} by_type={filtered_invalid_by_type}"
        )
        info(
            "[import_hf_structured_sft] Filter reasons (top): "
            f"{dict(sorted(filtered_invalid_by_reason.items(), key=lambda kv: kv[1], reverse=True)[:20])}"
        )
    with sft_out.open("w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.write_grpo_tasks:
        info(f"[import_hf_structured_sft] Writing GRPO tasks JSON: {grpo_out} tasks={len(grpo_tasks)}")
        grpo_out.write_text(json.dumps(grpo_tasks, ensure_ascii=False), encoding="utf-8")

    info("Done.")


if __name__ == "__main__":
    main()

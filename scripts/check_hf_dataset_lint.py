#!/usr/bin/env python3
"""Check Hugging Face structured-output datasets for syntax (lint) errors.

This script downloads a dataset via `datasets` and verifies that the *gold*
assistant output can be parsed as the claimed format (JSON/YAML/TOML/XML/CSV).

Why this exists
--------------
Some datasets include preambles ("Sure! ..."), code fences, or an "Output:"
section. For training/evaluation, we usually want the final structured blob.
We therefore:

1) infer output_type from each example
2) extract the final structured output using `extract_final_output()`
3) run the project's strict parser checks (StructEval-T syntax strictness)

The report shows how many examples fail strict parsing per format and prints
sample failing items.

Example
-------
python scripts/check_hf_dataset_lint.py \
  --datasets u-10bei/structured_data_with_cot_dataset_512_v2 \
  --split train --max-examples 200 --show 5

Notes
-----
- This script requires internet access at runtime to download HF datasets.
- It does not modify datasets; it only inspects and reports.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.data.import_hf_structured_sft import (
    _extract_messages,
    _infer_output_type,
    _normalize_output_type,
    _extract_reference_output,
)
from src.structeval_t.scorer import eval_structeval_t


def _ensure_deps() -> None:
    try:
        import datasets  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'datasets'. Install with: pip install datasets") from e


@dataclass
class FailCase:
    dataset: str
    idx: int
    output_type: str
    reason: str
    extracted_output_preview: str
    raw_assistant_preview: str


def _preview(s: str, n: int = 240) -> str:
    s = (s or "").strip().replace("\r\n", "\n")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _iter_examples(ds) -> Iterable[Tuple[int, Dict[str, Any]]]:
    # datasets.Dataset is indexable, but iterating yields dicts.
    for i, ex in enumerate(ds):
        yield i, ex


def _get_gold_and_type(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (output_type, raw_assistant, extracted_gold)."""
    output_type = _normalize_output_type(_infer_output_type(ex))
    sys_msg, user_msg, asst_msg = _extract_messages(ex)

    # Some datasets store output in metadata or top-level fields.
    extracted = _extract_reference_output(ex, asst_msg, output_type)

    raw_asst = asst_msg or ""
    return output_type or "", raw_asst, extracted or ""


def _strict_syntax_ok(text: str, output_type: str) -> bool:
    # eval_structeval_t returns syntax_score=1 only when strict parsing succeeds.
    r = eval_structeval_t(text, raw_output_metric=[], output_type=output_type)
    return bool(r.syntax_score >= 1.0)


def main() -> int:
    _ensure_deps()
    import datasets  # type: ignore

    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated HF datasets (e.g. u-10bei/..._v2,daichira/structured-3k-mix-sft)",
    )
    p.add_argument("--split", default="train", help="Split name (default: train)")
    p.add_argument("--max-examples", type=int, default=0, help="Limit examples per dataset (0 = all)")
    p.add_argument("--show", type=int, default=10, help="How many failing examples to print per dataset")
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset (only if needed).",
    )
    args = p.parse_args()

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    if not names:
        raise SystemExit("No datasets provided")

    total_by_type: Counter[str] = Counter()
    fail_by_type: Counter[str] = Counter()
    fail_cases_by_ds: defaultdict[str, List[FailCase]] = defaultdict(list)
    unknown_type_by_ds: defaultdict[str, int] = defaultdict(int)

    for name in names:
        ds = datasets.load_dataset(name, split=args.split, trust_remote_code=bool(args.trust_remote_code))
        n_seen = 0

        for idx, ex in _iter_examples(ds):
            if args.max_examples and n_seen >= args.max_examples:
                break
            n_seen += 1

            output_type, raw_asst, extracted = _get_gold_and_type(ex)
            if not output_type:
                unknown_type_by_ds[name] += 1
                continue

            total_by_type[output_type] += 1
            if not extracted.strip():
                fail_by_type[output_type] += 1
                fail_cases_by_ds[name].append(
                    FailCase(
                        dataset=name,
                        idx=idx,
                        output_type=output_type,
                        reason="empty_extracted_output",
                        extracted_output_preview="",
                        raw_assistant_preview=_preview(raw_asst),
                    )
                )
                continue

            ok = _strict_syntax_ok(extracted, output_type)
            if not ok:
                fail_by_type[output_type] += 1
                fail_cases_by_ds[name].append(
                    FailCase(
                        dataset=name,
                        idx=idx,
                        output_type=output_type,
                        reason="strict_parse_failed",
                        extracted_output_preview=_preview(extracted),
                        raw_assistant_preview=_preview(raw_asst),
                    )
                )

        print(f"\n== Dataset: {name} (split={args.split}, checked={n_seen}) ==")
        if unknown_type_by_ds[name]:
            print(f"- Unknown output_type examples: {unknown_type_by_ds[name]}")
        fails = fail_cases_by_ds.get(name, [])
        if not fails:
            print("- No strict-parse failures found.")
        else:
            # summarize per type in this dataset
            per_type = Counter([f.output_type for f in fails])
            print("- Failures by output_type:")
            for k, v in per_type.most_common():
                print(f"  - {k}: {v}")
            print("- Sample failing items:")
            for f in fails[: max(0, int(args.show))]:
                print(f"\n  [{f.idx}] type={f.output_type} reason={f.reason}")
                print(f"  extracted: {f.extracted_output_preview}")
                if f.raw_assistant_preview and f.raw_assistant_preview != f.extracted_output_preview:
                    print(f"  raw_asst : {f.raw_assistant_preview}")

    print("\n== Overall summary ==")
    if total_by_type:
        for t in sorted(total_by_type.keys()):
            tot = total_by_type[t]
            bad = fail_by_type.get(t, 0)
            rate = (bad / tot * 100.0) if tot else 0.0
            print(f"- {t}: {bad}/{tot} failed ({rate:.2f}%)")
    else:
        print("No typed examples were found (could not infer output_type).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

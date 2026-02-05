#!/usr/bin/env python3
"""Check Hugging Face structured-output datasets for syntax (lint) errors.

This script downloads a dataset via `datasets` and verifies that the *gold*
assistant output can be parsed as the claimed format (JSON/YAML/TOML/XML/CSV).

Enhancements:
- Adds --dump-failures <path> to dump ALL failing items as JSONL.
- Adds repo root to sys.path so `import src...` works without PYTHONPATH=.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Make `src` importable when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.import_hf_structured_sft import (  # noqa: E402
    _extract_messages,
    _infer_output_type,
    _normalize_output_type,
    _extract_reference_output,
)
from src.structeval_t.scorer import eval_structeval_t  # noqa: E402


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


def _json_sanitize(obj: Any) -> Any:
    """Make obj JSON-serializable for JSONL dumps (best-effort)."""
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False))
    except Exception:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


def _iter_examples(ds) -> Iterable[Tuple[int, Dict[str, Any]]]:
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
    p.add_argument(
        "--dump-failures",
        default="",
        help="If set, dump ALL failing items as JSONL to this path (includes raw_item).",
    )
    args = p.parse_args()

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    if not names:
        raise SystemExit("No datasets provided")

    total_by_type: Counter[str] = Counter()
    fail_by_type: Counter[str] = Counter()
    fail_cases_by_ds: defaultdict[str, List[FailCase]] = defaultdict(list)
    unknown_type_by_ds: defaultdict[str, int] = defaultdict(int)

    # JSONL dump buffer
    dump_records: List[Dict[str, Any]] = []

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
                fc = FailCase(
                    dataset=name,
                    idx=idx,
                    output_type=output_type,
                    reason="empty_extracted_output",
                    extracted_output_preview="",
                    raw_assistant_preview=_preview(raw_asst),
                )
                fail_cases_by_ds[name].append(fc)

                if args.dump_failures:
                    dump_records.append(
                        {
                            "dataset": name,
                            "split": args.split,
                            "idx": idx,
                            "output_type": output_type,
                            "reason": "empty_extracted_output",
                            "extracted": extracted,
                            "raw_asst": raw_asst,
                            "raw_item": _json_sanitize(ex),
                            "failcase": asdict(fc),
                        }
                    )
                continue

            ok = _strict_syntax_ok(extracted, output_type)
            if not ok:
                fail_by_type[output_type] += 1
                fc = FailCase(
                    dataset=name,
                    idx=idx,
                    output_type=output_type,
                    reason="strict_parse_failed",
                    extracted_output_preview=_preview(extracted),
                    raw_assistant_preview=_preview(raw_asst),
                )
                fail_cases_by_ds[name].append(fc)

                if args.dump_failures:
                    dump_records.append(
                        {
                            "dataset": name,
                            "split": args.split,
                            "idx": idx,
                            "output_type": output_type,
                            "reason": "strict_parse_failed",
                            "extracted": extracted,
                            "raw_asst": raw_asst,
                            "raw_item": _json_sanitize(ex),
                            "failcase": asdict(fc),
                        }
                    )

        print(f"\n== Dataset: {name} (split={args.split}, checked={n_seen}) ==")
        if unknown_type_by_ds[name]:
            print(f"- Unknown output_type examples: {unknown_type_by_ds[name]}")
        fails = fail_cases_by_ds.get(name, [])
        if not fails:
            print("- No strict-parse failures found.")
        else:
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

    if args.dump_failures:
        out_path = Path(args.dump_failures)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in dump_records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"[dump] wrote failures: {len(dump_records)} -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


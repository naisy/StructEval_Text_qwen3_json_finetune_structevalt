#!/usr/bin/env python3
"""Insert deep-hierarchy TOML SFT examples into data/my_sft_dataset.jsonl.

"Deep" means TOML uses at least one of:
  - dotted table / array-of-table headers like [a.b.c] or [[a.b.c]]
  - dotted key assignment like a.b.c = ...

This file is intended as an *optional* extra dataset (see configs/sft_hf.yaml).
We keep it separate from Hugging Face train/valid to avoid duplication.

Behavior:
- If data/my_sft_dataset.jsonl does not exist, create it.
- If it exists, append only examples whose `task_id` is not already present.

JSONL schema matches this repo's SFT loader:
- query: str
- output: str
- output_type: "TOML"
- task_id: unique str
- task_family/task_kind/task_schema/task_key: lightweight classification (optional but useful)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


# "Deep" here means depth > 1 (i.e., at least one dot).
# Examples:
#   [a.b.c], [[a.b]], a.b.c = 1
_DEEP_HEADER_RE = re.compile(
    r"^\s*\[\[?\s*([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+){1,})\s*\]\]?\s*$",
    re.MULTILINE,
)
_DEEP_DOTTED_KEY_RE = re.compile(
    r"^\s*[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+\s*=",
    re.MULTILINE,
)


def is_deep_toml(toml_text: str) -> bool:
    if not isinstance(toml_text, str):
        return False
    return bool(_DEEP_HEADER_RE.search(toml_text) or _DEEP_DOTTED_KEY_RE.search(toml_text))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_deep_examples() -> List[Dict[str, Any]]:
    """Hand-crafted deep TOML examples to force learning of dotted tables/keys."""

    examples: List[Dict[str, Any]] = []

    def add(idx: int, query: str, output: str, schema: str) -> None:
        if not is_deep_toml(output):
            raise ValueError(f"output is not deep TOML (idx={idx})")
        task_id = f"my_deep_toml_{idx:04d}"
        examples.append(
            {
                "task_id": task_id,
                "query": query.strip(),
                "output": output.strip() + "\n",
                "output_type": "TOML",
                "task_family": "extra",
                "task_kind": "extract",
                "task_schema": schema,
                "task_key": f"extra|TOML|extract|{schema}",
            }
        )

    # 1) dotted table headers ([a.b.c])
    add(
        1,
        """Extract the following attributes from text and output TOML.\n\nReturn ONLY TOML.\n\nATTRIBUTES:\nuser.profile.name, user.profile.contact.email, user.profile.contact.phone\n\nTEXT:\nuser.profile.name: Aya Tanaka | user.profile.contact.email: aya@example.com | user.profile.contact.phone: +81-90-1234-5678""",
        """[user.profile]\nname = \"Aya Tanaka\"\n\n[user.profile.contact]\nemail = \"aya@example.com\"\nphone = \"+81-90-1234-5678\"\n""",
        "deep_profile",
    )

    # 2) array-of-tables with dotted path ([[a.b.c]])
    add(
        2,
        """Convert the given order list into TOML.\nRules:\n- Use [[orders.items]] for the items list.\n- Return ONLY TOML.\n\nDATA:\norder_id=ORD-1001; currency=JPY; items=[(sku=AA-1, qty=2),(sku=BB-9, qty=1)]""",
        """[orders]\norder_id = \"ORD-1001\"\ncurrency = \"JPY\"\n\n[[orders.items]]\nsku = \"AA-1\"\nqty = 2\n\n[[orders.items]]\nsku = \"BB-9\"\nqty = 1\n""",
        "deep_orders",
    )

    # 3) dotted key assignment (a.b.c = ...)
    add(
        3,
        """Output TOML representing the config described below.\nReturn ONLY TOML.\n\nSPEC:\n- db.host = \"localhost\"\n- db.port = 5432\n- db.pool.size = 20\n- db.pool.timeout_sec = 5""",
        """db.host = \"localhost\"\ndb.port = 5432\ndb.pool.size = 20\ndb.pool.timeout_sec = 5\n""",
        "deep_dotted_keys",
    )

    # 4) combine dotted header + deeper nesting
    add(
        4,
        """Extract to TOML. Use dotted tables for nested objects. Return ONLY TOML.\n\nATTRIBUTES:\napp.features.auth.enabled, app.features.auth.providers[0], app.features.auth.providers[1], app.logging.level\n\nTEXT:\napp.features.auth.enabled: true | app.features.auth.providers[0]: google | app.features.auth.providers[1]: github | app.logging.level: info""",
        """[app.features.auth]\nenabled = true\nproviders = [\"google\", \"github\"]\n\n[app.logging]\nlevel = \"info\"\n""",
        "deep_features",
    )

    # 5) deep arrays-of-tables under dotted tables
    add(
        5,
        """Transform this inventory summary into TOML.\nRules: use [[warehouse.sections.bins]] for bins. Return ONLY TOML.\n\nDATA:\nwarehouse_id=W-7; section=A; bins=[(bin=1, sku=X1, count=3),(bin=2, sku=Y9, count=8)]""",
        """[warehouse]\nwarehouse_id = \"W-7\"\n\n[warehouse.sections]\nname = \"A\"\n\n[[warehouse.sections.bins]]\nbin = 1\nsku = \"X1\"\ncount = 3\n\n[[warehouse.sections.bins]]\nbin = 2\nsku = \"Y9\"\ncount = 8\n""",
        "deep_bins",
    )

    # Additional variety to reinforce depth (6..20)
    for i in range(6, 21):
        # alternate between dotted keys and dotted headers
        if i % 2 == 0:
            add(
                i,
                f"""Extract to TOML (Return ONLY TOML).\n\nATTRIBUTES:\nmetrics.run{i}.loss.train, metrics.run{i}.loss.valid, metrics.run{i}.epoch\n\nTEXT:\nmetrics.run{i}.loss.train: {0.1*i:.3f} | metrics.run{i}.loss.valid: {0.12*i:.3f} | metrics.run{i}.epoch: {i}""",
                f"""metrics.run{i}.loss.train = {0.1*i:.3f}\nmetrics.run{i}.loss.valid = {0.12*i:.3f}\nmetrics.run{i}.epoch = {i}\n""",
                "deep_metrics",
            )
        else:
            add(
                i,
                f"""Convert to TOML (Return ONLY TOML).\n\nDATA:\nproject=demo{i}; owner.name=User{i}; owner.contact.email=user{i}@example.com""",
                f"""[owner]\nname = \"User{i}\"\n\n[owner.contact]\nemail = \"user{i}@example.com\"\n\n[project]\nname = \"demo{i}\"\n""",
                "deep_owner",
            )

    return examples


def main() -> None:
    out_path = Path("data/my_sft_dataset.jsonl")
    existing = _read_jsonl(out_path)

    existing_ids: Set[str] = set()
    for r in existing:
        tid = r.get("task_id")
        if isinstance(tid, str) and tid.strip():
            existing_ids.add(tid.strip())

    candidates = build_deep_examples()

    added = 0
    for ex in candidates:
        tid = ex["task_id"]
        if tid in existing_ids:
            continue
        existing.append(ex)
        existing_ids.add(tid)
        added += 1

    _write_jsonl(out_path, existing)
    print(f"Wrote {out_path} (added {added} new examples, total {len(existing)})")


if __name__ == "__main__":
    main()

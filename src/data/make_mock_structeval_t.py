from __future__ import annotations

"""Generate a small, local StructEval-T style dataset for development.

This is used only as a fallback when real evaluation data is not available.
The goal is *not* to be a faithful replica of StructEval-T, but to:

- exercise JSON/YAML/TOML/XML/CSV parsing paths,
- allow stratified sampling by output_type (10 each by default),
- keep the repo runnable offline.
"""

import argparse
import json
import random
from pathlib import Path


FIRST = ["Alice", "Bob", "Carol", "David", "Eve", "Fay", "Grace", "Heidi", "Ivan", "Judy"]
LAST = ["Smith", "Tanaka", "Suzuki", "Chen", "Khan", "Garcia", "Miller", "Wang", "Sato", "Kim"]
AFFIL = ["University of Example", "Example Institute of Technology", "GClue Research", "Fictional Labs"]
KEYWORDS = ["LLM", "JSON", "Evaluation", "Schema", "Fine-tuning", "Reinforcement Learning", "StructEval-T"]


def _rand_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _rand_keywords(rng: random.Random) -> list[str]:
    k = rng.randint(3, 6)
    return rng.sample(KEYWORDS, k=k)


def _task_template(output_type: str) -> tuple[str, list[str]]:
    ot = output_type.strip().upper()

    if ot in {"JSON", "YAML", "TOML"}:
        query = (
            f"Please output {ot} code.\n\n"
            "Task:\n"
            "Summarize metadata about a fictional scientific article.\n\n"
            "Feature Requirements:\n"
            "1. Top-level field \"title\" is a string\n"
            "2. Field \"authors\" is a list of exactly two items\n"
            "3. Each author has \"name\" and \"affiliation\"\n"
            "4. Field \"publication.year\" is an integer\n"
            "5. Field \"keywords\" is a list of strings\n"
        )
        raw = [
            "title",
            "authors[0].name",
            "authors[0].affiliation",
            "authors[1].name",
            "authors[1].affiliation",
            "publication.year",
            "keywords[0]",
        ]
        return query, raw

    if ot == "XML":
        query = (
            "Please output XML code.\n\n"
            "Task:\n"
            "Summarize metadata about a fictional scientific article.\n\n"
            "Feature Requirements:\n"
            "- Root tag is <article>\n"
            "- Must contain <title> and <publication><year>\n"
        )
        raw = [
            "article.title",
            "article.publication.year",
        ]
        return query, raw

    if ot == "CSV":
        query = (
            "Please output CSV (comma-separated) code.\n\n"
            "Task:\n"
            "Create a single-row CSV with a header.\n\n"
            "Feature Requirements:\n"
            "- Header columns: title, year\n"
            "- Exactly one data row\n"
        )
        raw = [
            "rows[0].title",
            "rows[0].year",
        ]
        return query, raw

    # Unknown output type: treat as JSON.
    return _task_template("JSON")


def _build_task(task_id: int, rng: random.Random, output_type: str) -> dict:
    query, raw = _task_template(output_type)
    # Fill some fields (not used by the scorer directly, but keeps tasks realistic).
    _ = {
        "title": f"On {_rand_name(rng)}'s Findings in {rng.choice(KEYWORDS)}",
        "authors": [
            {"name": _rand_name(rng), "affiliation": rng.choice(AFFIL)},
            {"name": _rand_name(rng), "affiliation": rng.choice(AFFIL)},
        ],
        "publication": {"year": rng.randint(1990, 2026)},
        "keywords": _rand_keywords(rng),
    }

    ot = output_type.strip().upper()
    return {
        "task_id": f"{task_id:06d}",
        "query": query,
        "feature_requirements": "",
        "task_name": f"Text to {ot} (mock)",
        "input_type": "Text",
        "output_type": ot,
        "query_example": "",
        "VQA": [],
        "raw_output_metric": raw,
        "rendering": False,
    }


def make_dataset(out_path: str | Path, n: int, seed: int = 42, output_types: list[str] | None = None) -> None:
    rng = random.Random(seed)
    ots = [o.strip().upper() for o in (output_types or ["JSON"])]
    ots = [o for o in ots if o]
    if not ots:
        ots = ["JSON"]

    tasks: list[dict] = []
    for i in range(n):
        ot = ots[i % len(ots)]
        tasks.append(_build_task(500 + i, rng, ot))

    Path(out_path).write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-train", default="data/train_structeval_t.json")
    p.add_argument("--out-valid", default="data/valid_structeval_t.json")
    # Defaults are intentionally larger so you can run more extensive evals
    # before the official dataset is released.
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-valid", type=int, default=500)
    p.add_argument(
        "--output-types",
        default="JSON",
        help="Comma-separated list, e.g. JSON,YAML,TOML,XML,CSV",
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    ots = [x.strip() for x in str(a.output_types).split(",") if x.strip()]

    make_dataset(a.out_train, a.n_train, seed=a.seed, output_types=ots)
    make_dataset(a.out_valid, a.n_valid, seed=a.seed + 1, output_types=ots)
    print(f"Wrote {a.out_train} and {a.out_valid}")


if __name__ == "__main__":
    main()

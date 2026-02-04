from __future__ import annotations

"""Generate a small JSONL dataset suitable for SFT smoke-testing.

StructEval-T JSON tasks do not provide a unique gold output, so they are not
suitable as supervised targets. This script creates synthetic pairs:
- instruction: JSON formatting task
- output: a JSON string that satisfies the instruction

Usage:
  python -m src.data.make_mock_sft_jsonl --out-train data/train_sft.jsonl --out-valid data/valid_sft.jsonl
"""

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


def _build_instruction() -> str:
    return (
        "Please output JSON code.\n\n"
        "Task:\n"
        "Summarize metadata about a fictional scientific article.\n\n"
        "Feature Requirements:\n"
        "1. Top-level field \"title\" is a string\n"
        "2. Field \"authors\" is a list of exactly two items\n"
        "3. Each author has \"name\" and \"affiliation\"\n"
        "4. Field \"publication.year\" is an integer\n"
        "5. Field \"keywords\" is a list of strings\n"
    )


def _build_output(rng: random.Random) -> str:
    obj = {
        "title": f"On {rng.choice(['Robust', 'Reliable', 'Strict'])} JSON Generation with Small LLMs",
        "authors": [
            {"name": _rand_name(rng), "affiliation": rng.choice(AFFIL)},
            {"name": _rand_name(rng), "affiliation": rng.choice(AFFIL)},
        ],
        "publication": {"year": rng.randint(2015, 2026)},
        "keywords": _rand_keywords(rng),
    }
    # Compact JSON is usually better for strict parsers.
    return json.dumps(obj, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_dataset(out_path: str | Path, n: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    rows = []
    instr = _build_instruction()
    for _ in range(n):
        rows.append(
            {
                "instruction": instr,
                "requirements": "Output must be a single JSON object and nothing else.",
                "output": _build_output(rng),
            }
        )
    write_jsonl(out_path, rows)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out-train", default="data/train_sft.jsonl")
    p.add_argument("--out-valid", default="data/valid_sft.jsonl")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-valid", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    make_dataset(a.out_train, a.n_train, seed=a.seed)
    make_dataset(a.out_valid, a.n_valid, seed=a.seed + 1)
    print(f"Wrote {a.out_train} and {a.out_valid}")


if __name__ == "__main__":
    main()

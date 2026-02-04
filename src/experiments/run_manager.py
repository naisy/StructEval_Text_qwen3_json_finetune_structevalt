from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _git_rev_short() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


@dataclass
class RunContext:
    run_dir: Path
    run_id: str
    stage: str


def start_run(stage: str, run_name: str | None = None, base_dir: str = "runs") -> RunContext:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gitrev = _git_rev_short()
    safe_name = (run_name or "").strip().replace(" ", "_")
    run_id = f"{ts}_{stage}_{gitrev}" + (f"_{safe_name}" if safe_name else "")
    run_dir = Path(base_dir) / stage / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunContext(run_dir=run_dir, run_id=run_id, stage=stage)


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

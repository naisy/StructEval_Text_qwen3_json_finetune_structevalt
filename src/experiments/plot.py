from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _collect_summaries(runs_dir: str = "runs") -> pd.DataFrame:
    rows = []
    for p in Path(runs_dir).glob("**/*.summary.json"):
        try:
            summ = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_dir = p.parent
        rows.append({
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "stage": run_dir.parent.name,
            **summ,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_run_summaries(runs_dir: str = "runs", out_dir: str = "runs/plots") -> str:
    df = _collect_summaries(runs_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Create an empty placeholder plot.
        fig = plt.figure()
        plt.title("No run summaries found")
        out_path = out / "no_runs.png"
        fig.savefig(out_path)
        plt.close(fig)
        return str(out_path)

    # Sort by time inferred from run_id prefix (YYYYMMDD_HHMMSS)
    df = df.sort_values("run_id")
    fig = plt.figure()
    # default matplotlib colors (do not set specific colors)
    for stage, g in df.groupby("stage"):
        plt.plot(g["run_id"], g.get("avg_final_eval_score", 0.0), marker="o", label=stage)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("avg_final_eval_score")
    plt.title("StructEval-T score by run")
    plt.legend()
    plt.tight_layout()
    out_path = out / "score_by_run.png"
    fig.savefig(out_path)
    plt.close(fig)

    df.to_csv(out / "run_summaries.csv", index=False)
    return str(out_path)

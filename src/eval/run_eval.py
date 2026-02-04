from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM

from src.models.load import load_tokenizer

from src.data.dataset import load_jsonl, build_prompt
from src.data.validators import (
    parse_json,
    is_json_only,
    parse_yaml,
    is_yaml_only,
    parse_toml,
    is_toml_only,
    parse_xml,
    is_xml_only,
    parse_csv,
    is_csv_only,
    build_schema_validator,
    validate_schema,
    check_task_constraints_article_meta,
)
from src.eval.metrics import EvalCounts, safe_div
from src.utils.io import load_yaml, ensure_dir
from src.utils.logging import info, warn

# Lightweight helper (no `transformers`/`torch` dependencies) so scoring logic is shared & testable.
from src.eval.structeval_scoring import structeval_t_score


def _load_structeval_tasks(path: str | Path) -> List[Dict[str, Any]]:
    """Load StructEval-T style tasks.

    Expected file format: JSON array, each item includes:
      - query: str
      - raw_output_metric: list[str]  (paths to check)
      - (no gold output; StructEval-T JSON tasks are non-unique)
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"StructEval-T dataset must be a JSON array: {p}")
    return data


def _structeval_t_score(task: Dict[str, Any], generation: str) -> Dict[str, Any]:
    # Backward-compatible wrapper name.
    return structeval_t_score(task, generation)


def _parse_for_output_type(output_type: str, generation: str) -> tuple[bool, bool, Any | None]:
    """Return (strict_parse_ok, only_ok, parsed_obj_or_none).

    - strict_parse_ok: the requested format parses after stripping code fences.
    - only_ok: heuristic that the output is *only* the requested format.
      For JSON this is identical to strict_parse_ok (via `is_json_only`).

    Note: for YAML/TOML/XML/CSV, `only_ok` currently mirrors strict parsing.
    """
    ot = (output_type or "JSON").strip().upper()

    if ot == "JSON":
        ok, obj, _ = parse_json(generation)
        return ok, is_json_only(generation), obj if ok else None

    if ot == "YAML":
        ok, obj, _ = parse_yaml(generation)
        return ok, is_yaml_only(generation), obj if ok else None

    if ot == "TOML":
        ok, obj, _ = parse_toml(generation)
        return ok, is_toml_only(generation), obj if ok else None

    if ot == "XML":
        ok, obj, _ = parse_xml(generation)
        return ok, is_xml_only(generation), obj if ok else None

    if ot == "CSV":
        ok, obj, _ = parse_csv(generation)
        return ok, is_csv_only(generation), obj if ok else None

    # Unknown: default to JSON
    ok, obj, _ = parse_json(generation)
    return ok, is_json_only(generation), obj if ok else None


def _select_stratified_by_output_type(
    tasks: List[Dict[str, Any]],
    prompts: List[str],
    *,
    per_type_limit: int,
    output_types: List[str],
    seed: int | None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Select up to `per_type_limit` items for each output type.

    This is used to keep evaluation fast while still covering multiple
    serialization formats (JSON/YAML/TOML/XML/CSV).

    Behavior:
    - If `seed` is provided, items are deterministically shuffled *within*
      each output type bucket before selecting.
    - The returned order follows `output_types`.
    """
    # Bucket indices by normalized output type.
    buckets: Dict[str, List[int]] = {ot: [] for ot in output_types}
    for i, t in enumerate(tasks):
        ot = str((t or {}).get("output_type") or "JSON").strip().upper()
        if ot in buckets:
            buckets[ot].append(i)

    rng = random.Random(int(seed)) if seed is not None else None

    selected_idxs: List[int] = []
    for ot in output_types:
        idxs = buckets.get(ot, [])
        if rng is not None and len(idxs) > 1:
            rng.shuffle(idxs)
        selected_idxs.extend(idxs[: int(per_type_limit)])

    # Preserve the overall order implied by output_types (already), but also
    # preserve intra-bucket ordering after optional shuffle.
    sel_tasks = [tasks[i] for i in selected_idxs]
    sel_prompts = [prompts[i] for i in selected_idxs]
    return sel_tasks, sel_prompts




def run_eval(config_path: str, override_model_path: str | None = None, *, limit: int | None = None, seed: int | None = None) -> dict:
    """Run evaluation.

    This evaluates:
    - JSON structural metrics (parse/json-only/schema) on generated output
    - StructEval-T structural score (render + key paths)
    - Optional LLM-as-a-judge semantic score (candidate vs gold)

    Outputs:
    - outputs/eval/structeval_t_eval.json (per-sample records)
    - outputs/eval/eval_report.json (summary)
    """
    cfg = load_yaml(config_path)

    # Evaluation sampling (to keep eval fast on large datasets)
    eval_cfg = cfg.get("eval", {}) or {}

    # Track whether the caller explicitly provided overrides.
    limit_override = limit is not None
    seed_override = seed is not None
    # Default: evaluate 10 examples in total (legacy behavior) unless a
    # per-output-type limit is configured.
    cfg_limit_raw = eval_cfg.get("limit", None)
    cfg_limit = int(cfg_limit_raw) if cfg_limit_raw is not None else 10

    # New: stratified sampling across formats.
    # Example: JSON 10 + YAML 10 + TOML 10 + XML 10 + CSV 10.
    cfg_per_type_raw = eval_cfg.get("limit_per_output_type", eval_cfg.get("per_type_limit", None))
    cfg_per_type = int(cfg_per_type_raw) if cfg_per_type_raw is not None else None
    cfg_output_types = eval_cfg.get("output_types", None)

    cfg_seed = eval_cfg.get("seed", None)
    if not limit_override:
        limit = cfg_limit
    if (not seed_override) and (cfg_seed is not None):
        seed = int(cfg_seed)

    # Judge config (optional)
    judge_cfg_path = cfg.get("judge_config", "configs/judge.yaml")
    judge_cfg_all = load_yaml(judge_cfg_path) if Path(judge_cfg_path).exists() else {}
    judge_cfg = (judge_cfg_all.get("judge") or {}) if isinstance(judge_cfg_all, dict) else {}
    judge = None
    # Which output types the judge should be applied to.
    # Default is JSON-only because the current prompt/template is JSON-focused.
    judge_output_types = [str(x).upper() for x in (judge_cfg.get("output_types") or ["JSON"])]
    # For reporting convenience.
    _providers = judge_cfg.get("providers") or []
    if isinstance(_providers, list) and _providers:
        _p0 = _providers[0] if isinstance(_providers[0], dict) else {}
        judge_model_name = f"{_p0.get('provider','multi')}:{_p0.get('model','')}".rstrip(":")
    else:
        judge_model_name = judge_cfg.get("provider") or "multi"
    if judge_cfg.get("enabled", False):
        from src.judge import build_judge
        judge = build_judge(judge_cfg)

    # Dataset config
    dcfg = load_yaml(cfg["data"]["dataset_config"])

    # Choose dataset loader: StructEval JSON array or JSONL pairs.
    fmt = (dcfg.get("dataset") or {}).get("format", "jsonl")
    valid_path = (dcfg.get("dataset") or {}).get("valid_path")

    # Optional override via environment variable.
    # This is useful for scripts / post-train eval without rewriting YAMLs.
    env_tasks = os.getenv("EVAL_TASKS_PATH", "").strip()
    if env_tasks:
        valid_path = env_tasks
        # Heuristic: StructEval exports are JSON arrays (.json). If the dataset
        # config was JSONL, switch loader accordingly.
        if str(env_tasks).lower().endswith(".json") and fmt in ("jsonl", "jsonl_pairs"):
            fmt = "structeval_json"

    if not valid_path:
        raise ValueError("dataset.valid_path is required")

    # If the desired eval dataset is missing, fall back to mock tasks.
    # This keeps the repo runnable even before you download public StructEval.
    if valid_path and not Path(valid_path).exists():
        warn(f"Eval dataset not found: {valid_path}. Falling back to mock tasks.")
        # If mock tasks do not exist, generate them.
        if not Path("data/valid_structeval_t.json").exists():
            import subprocess
            # Keep fallback aligned with multi-format defaults:
            #   JSON 10 + YAML 10 + TOML 10 + XML 10 + CSV 10 = 50
            per_type = int(cfg_per_type) if cfg_per_type is not None else 10
            ots = cfg_output_types or ["JSON", "YAML", "TOML", "XML", "CSV"]
            n_valid = max(10, per_type * max(1, len(ots)))
            subprocess.check_call([
                "python",
                "-m",
                "src.data.make_mock_structeval_t",
                "--n-train",
                str(n_valid * 5),
                "--n-valid",
                str(n_valid),
                "--output-types",
                ",".join(ots),
            ])
        valid_path = "data/valid_structeval_t.json"
        # StructEval-T mock dataset is a JSON array.
        if fmt in ("jsonl", "jsonl_pairs"):
            fmt = "structeval_json"

    if fmt in ("structeval", "structeval_json", "structeval_t"):
        tasks = _load_structeval_tasks(valid_path)
        prompts = [build_prompt(t, dcfg) for t in tasks]
    else:
        valid_items = load_jsonl(valid_path)
        tasks = valid_items
        prompts = [build_prompt(ex, dcfg) for ex in valid_items]

    # Apply evaluation sampling.
    # Priority:
    # 1) If `limit` was explicitly passed to run_eval(), it wins (total-limit).
    # 2) Else if config sets `limit_per_output_type`, use stratified selection.
    # 3) Else use legacy behavior: optional shuffle + total `limit`.
    if limit_override and limit is not None and int(limit) > 0:
        # Total-limit override path.
        if seed is not None:
            rng = random.Random(int(seed))
            idxs = list(range(len(tasks)))
            rng.shuffle(idxs)
            tasks = [tasks[i] for i in idxs]
            prompts = [prompts[i] for i in idxs]

        tasks = tasks[: int(limit)]
        prompts = prompts[: int(limit)]
        info(f"Eval items: {len(tasks)} (limit={limit}, seed={seed})")

    elif cfg_per_type is not None and int(cfg_per_type) > 0:
        # Stratified per-format sampling.
        if cfg_output_types is None:
            output_types = ["JSON", "YAML", "TOML", "XML", "CSV"]
        else:
            output_types = [str(x).strip().upper() for x in list(cfg_output_types)]

        tasks, prompts = _select_stratified_by_output_type(
            tasks,
            prompts,
            per_type_limit=int(cfg_per_type),
            output_types=output_types,
            seed=seed,
        )

        info(
            f"Eval items: {len(tasks)} (limit_per_output_type={int(cfg_per_type)}, "
            f"output_types={','.join(output_types)}, seed={seed})"
        )

    else:
        # Legacy default path: shuffle then take total `limit`.
        if seed is not None:
            rng = random.Random(int(seed))
            idxs = list(range(len(tasks)))
            rng.shuffle(idxs)
            tasks = [tasks[i] for i in idxs]
            prompts = [prompts[i] for i in idxs]

        if limit is not None and int(limit) > 0:
            tasks = tasks[: int(limit)]
            prompts = prompts[: int(limit)]

        info(f"Eval items: {len(tasks)} (limit={limit}, seed={seed})")

    model_name = override_model_path or cfg["model"]["base_model"]
    info(f"Loading model: {model_name}")

    # Load tokenizer with a safe compatibility shim for the known Mistral regex issue.
    tok = load_tokenizer(model_name, trust_remote_code=cfg["model"].get("trust_remote_code", True))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        device_map="auto",
    )

    gen_cfg = cfg.get("generation", {}) or {}
    temperature = float(gen_cfg.get("temperature", 0.0))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))

    # Schema (optional)
    schema_validator = None
    if (dcfg.get("schema") or {}).get("enabled", False):
        schema_path = Path((dcfg.get("schema") or {}).get("schema_path", ""))
        if schema_path.exists():
            schema_validator = build_schema_validator(schema_path)
        else:
            warn(f"Schema enabled but not found: {schema_path}")

    counts = EvalCounts()
    records: List[Dict[str, Any]] = []
    final_sum = 0.0
    judge_scores: List[float] = []
    parse_by_type: dict[str, dict[str, int]] = {}

    for task, prompt in zip(tasks, prompts):
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            out = model.generate(**inputs, **gen_kwargs)

        generation = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        counts.total += 1

        # Basic format checks (by task.output_type)
        output_type = str((task or {}).get("output_type") or "JSON").strip().upper()
        strict_ok, only_ok, parsed_obj = _parse_for_output_type(output_type, generation)

        d = parse_by_type.setdefault(output_type, {"n": 0, "parse_ok": 0, "only_ok": 0})
        d["n"] += 1
        if strict_ok:
            d["parse_ok"] += 1
        if only_ok:
            d["only_ok"] += 1

        if strict_ok:
            counts.format_parse_ok += 1
        if only_ok:
            counts.format_only_ok += 1

        # Backward-compatible JSON counters (only meaningful on JSON tasks)
        if output_type == "JSON":
            if strict_ok:
                counts.json_parse_ok += 1
            if only_ok:
                counts.json_only_ok += 1

            # Optional schema
            if strict_ok and schema_validator is not None:
                s_ok, _errs = validate_schema(parsed_obj, schema_validator)
                if s_ok:
                    counts.schema_ok += 1

            # Example constraint (authors length)
            checks = check_task_constraints_article_meta(parsed_obj) if parsed_obj is not None else {}
            if checks.get("authors_len_is_2", False):
                counts.authors_len_ok += 1

        # StructEval-T score (computed for any StructEval-style task)
        struct = _structeval_t_score(task if isinstance(task, dict) else {}, generation)
        final_sum += struct["final_eval_score"]

        record: Dict[str, Any] = dict(task) if isinstance(task, dict) else {"task": task}
        record.update({
            "generation": generation,
            "structeval_t": struct,
        })

        # Optional judge score
        if judge is not None and output_type in judge_output_types:
            try:
                jr = judge.judge(
                    task=task if isinstance(task, dict) else {"query": ""},
                    generation=generation,
                )

                # Extract a human-friendly "reason" summary from judge details.
                details = jr.details or {}
                reason = {
                    "notes": details.get("notes"),
                    "failed_checks": details.get("failed_checks"),
                    "breakdown": details.get("breakdown"),
                }

                record["judge"] = {
                    "provider": details.get("judge_provider") or judge_cfg.get("provider") or "multi",
                    "model": judge_model_name,
                    "score": jr.score,
                    "passed": jr.passed,
                    "details": details,
                    # cache metadata (set by judge providers)
                    "cache_key": details.get("_cache_key"),
                    "cache_file": details.get("_cache_file"),
                    "cache_hit": details.get("_cache_hit"),
                    # reason summary for quick analysis
                    "reason": reason,
                }
                judge_scores.append(jr.score)

            except Exception as e:
                record["judge"] = {
                    "provider": judge_cfg.get("provider") or "multi",
                    "model": judge_model_name,
                    "error": str(e),
                }

        records.append(record)

    out_dir = ensure_dir("outputs/eval")
    per_path = out_dir / "structeval_t_eval.json"
    per_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "total": counts.total,
        "format_parse_rate": safe_div(counts.format_parse_ok, counts.total),
        "format_only_rate": safe_div(counts.format_only_ok, counts.total),
        "by_output_type": {
            k: {
                "n": v.get("n", 0),
                "parse_rate": safe_div(v.get("parse_ok", 0), v.get("n", 0)),
                "only_rate": safe_div(v.get("only_ok", 0), v.get("n", 0)),
            }
            for k, v in sorted(parse_by_type.items())
        },
        # Legacy counters were JSON-only. Keep them for backward compatibility,
        # but also report rates normalized by the number of JSON tasks.
        "json_parse_rate": safe_div(counts.json_parse_ok, counts.total),
        "json_only_rate": safe_div(counts.json_only_ok, counts.total),
        "schema_rate": safe_div(counts.schema_ok, counts.total),
        "json_tasks_n": parse_by_type.get("JSON", {}).get("n", 0),
        "json_parse_rate_on_json_tasks": safe_div(
            counts.json_parse_ok, parse_by_type.get("JSON", {}).get("n", 0)
        ),
        "json_only_rate_on_json_tasks": safe_div(
            counts.json_only_ok, parse_by_type.get("JSON", {}).get("n", 0)
        ),
        "schema_rate_on_json_tasks": safe_div(
            counts.schema_ok, parse_by_type.get("JSON", {}).get("n", 0)
        ),
        "authors_len_2_rate": safe_div(counts.authors_len_ok, counts.total),
        "final_eval_score_avg": (final_sum / counts.total) if counts.total else 0.0,
        "judge_score_avg": (sum(judge_scores) / len(judge_scores)) if judge_scores else None,
        "model": model_name,
        "judge_enabled": bool(judge is not None),
        "judge_output_types": judge_output_types if (judge is not None) else [],
        "judge_model": judge_model_name if (judge is not None) else None,
    }

    rep_path = out_dir / "eval_report.json"
    rep_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    info(f"Wrote per-sample eval: {per_path}")
    info(f"Wrote summary report: {rep_path}")
    return summary
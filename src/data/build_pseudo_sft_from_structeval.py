from __future__ import annotations

import argparse
import json
import re
import random
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from src.data.validators import (
    is_csv_only,
    is_json_only,
    is_toml_only,
    is_xml_only,
    is_yaml_only,
    parse_csv,
    parse_json,
    parse_json_best_effort,
    parse_toml,
    parse_xml,
    parse_yaml,
    strip_code_fences,
)
from src.utils.ollama import ollama_chat, ollama_generate


def _ensure_deps() -> None:
    """Guard for optional HF teacher backend."""
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing deps. Install transformers + torch for pseudo-SFT generation (HF teacher backend)."
        ) from e


def load_eval_json(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array.")
    return data


def build_student_instruction(task: Dict[str, Any]) -> str:
    # Use the task's query verbatim; it already contains constraints.
    return (task.get("query", "") or "").strip()


def _normalize_output_type(s: str | None) -> str:
    """Normalize output types to one of: json/yaml/toml/xml/csv.

    StructEval exports typically include an `output_type` field, but some local
    or mock datasets may omit it.
    """
    if not s:
        return 'json'
    s2 = str(s).strip().lower()
    aliases = {
        'json': 'json',
        'yaml': 'yaml',
        'yml': 'yaml',
        'toml': 'toml',
        'xml': 'xml',
        'csv': 'csv',
    }
    return aliases.get(s2, 'json')


def infer_output_type(task: Dict[str, Any]) -> str:
    """Infer desired output type for a task.

    Priority:
      1) explicit `output_type` field
      2) heuristics from the query text
      3) default: json
    """
    ot = task.get('output_type')
    if isinstance(ot, str) and ot.strip():
        return _normalize_output_type(ot)

    q = (task.get('query') or '').lower()
    for k in ['yaml', 'yml', 'toml', 'xml', 'csv', 'json']:
        if k in q:
            return _normalize_output_type(k)
    return 'json'


def build_teacher_system_msg(output_type: str) -> str:
    ot = _normalize_output_type(output_type)
    if ot == 'json':
        return 'Return ONLY valid JSON. Do not include code fences or commentary.'
    if ot == 'yaml':
        return 'Return ONLY valid YAML. Do not include code fences or commentary.'
    if ot == 'toml':
        return 'Return ONLY valid TOML. Do not include code fences or commentary.'
    if ot == 'xml':
        return 'Return ONLY valid XML. Do not include code fences or commentary.'
    if ot == 'csv':
        return 'Return ONLY valid CSV text. Do not include code fences or commentary.'
    return 'Return ONLY the requested structured output. Do not include code fences or commentary.'


def build_teacher_prompt(task: Dict[str, Any], *, variant_id: int, variants_per_task: int, output_type: str) -> str:
    """Prompt used for teacher generation.

    NOTE: We do *not* store this as the student's `instruction`.
    The student's instruction should stay as close as possible to the original StructEval query.
    """
    q = build_student_instruction(task)
    ot = _normalize_output_type(output_type)
    suffix = f"\n\nIMPORTANT: Output ONLY valid {ot.upper()}. Do not include code fences, markdown, or any extra text."
    prompt = q + suffix

    if variants_per_task > 1:
        prompt += (
            "\n\n[Variation] Produce a different fictional instance than previous attempts "
            "(different names/values), while still satisfying all Feature Requirements. "
            f"Return {ot.upper()} only."
        )
        prompt += f"\n[Variation ID] {variant_id + 1} / {variants_per_task}"

    return prompt



def _parse_structured_best_effort(text: str, output_type: str) -> Tuple[bool, Optional[Any], str, bool, str]:
    """Parse structured output.

    Returns:
      (parse_ok, parsed_obj, used_text, strict_syntax_ok, err)

    For JSON we allow best-effort extraction (used for debugging/analysis) and
    distinguish strict vs best-effort parsing.

    For other formats we do strict parsing on the stripped output.
    """
    raw = text.strip() if isinstance(text, str) else ''
    ot = _normalize_output_type(output_type)

    if ot == 'json':
        ok_strict, obj, err = parse_json(raw)
        if ok_strict:
            return True, obj, strip_code_fences(raw).strip(), True, ''
        ok_any, obj2, err2, used = parse_json_best_effort(raw)
        used2 = (used or strip_code_fences(raw).strip()).strip()
        return bool(ok_any), obj2 if ok_any else None, used2, False, (err2 or err or '')

    cleaned = strip_code_fences(raw).strip()
    if ot == 'yaml':
        ok, obj, err = parse_yaml(cleaned)
        return bool(ok), obj if ok else None, cleaned, bool(ok), (err or '')
    if ot == 'toml':
        ok, obj, err = parse_toml(cleaned)
        return bool(ok), obj if ok else None, cleaned, bool(ok), (err or '')
    if ot == 'xml':
        ok, obj, err = parse_xml(cleaned)
        return bool(ok), obj if ok else None, cleaned, bool(ok), (err or '')
    if ot == 'csv':
        ok, obj, err = parse_csv(cleaned)
        return bool(ok), obj if ok else None, cleaned, bool(ok), (err or '')

    # Fallback: accept non-empty text but mark as not-parseable.
    return False, None, cleaned, False, f'unsupported output_type: {output_type}'


def _is_output_only(text: str, output_type: str) -> bool:
    ot = _normalize_output_type(output_type)
    if ot == 'json':
        return is_json_only(text)
    if ot == 'yaml':
        return is_yaml_only(text)
    if ot == 'toml':
        return is_toml_only(text)
    if ot == 'xml':
        return is_xml_only(text)
    if ot == 'csv':
        return is_csv_only(text)
    return bool(text.strip())


def generate_teacher_outputs_hf(
    tasks: List[Dict[str, Any]],
    *,
    teacher_model: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    variants_per_task: int = 4,
    limit: int = 0,
    seed: Optional[int] = None,
    log_every: int = 10,
    keep_invalid: bool = False,
) -> List[Dict[str, Any]]:
    """Generate teacher outputs with a local HuggingFace model.

    Stats are reported **per task**:
      - ok: at least one variant parsed as JSON
      - empty: all variants returned empty text
      - invalid: at least one non-empty output but none parsed as JSON

    Returned rows are **per generation** (task x variant).
    """
    _ensure_deps()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if limit and limit > 0:
        tasks = tasks[:limit]

    rng = random.Random(seed)
    if seed is not None:
        rng.shuffle(tasks)

    variants_per_task = max(1, int(variants_per_task))

    if variants_per_task > 1 and float(temperature) == 0.0:
        print(
            "[warn] variants_per_task > 1 but temperature=0.0. "
            "Depending on the teacher backend/model, outputs may be identical. "
            "Consider --temperature 0.2 (or similar) for diversity."
        )

    model = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Some tokenizers require a compatibility shim for the known Mistral regex issue.
    try:
        tok = AutoTokenizer.from_pretrained(teacher_model, use_fast=True, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(teacher_model, use_fast=True)

    def _variant_bucket(*, ok: bool, text: str, err: str) -> str:
        """Classify a single (task, variant) generation into one of four buckets."""
        if err and not text:
            return "errors"
        if not text:
            return "empty"
        return "ok" if ok else "invalid"

    out_rows: List[Dict[str, Any]] = []
    attempt_stats = {"ok": 0, "invalid": 0, "empty": 0, "errors": 0}
    task_stats = {"ok": 0, "invalid": 0, "empty": 0, "errors": 0}

    # Per-variant *task* stats (each task contributes exactly 1 bucket per variant).
    per_variant_task_stats: List[Dict[str, int]] = [
        {"ok": 0, "invalid": 0, "empty": 0, "errors": 0} for _ in range(variants_per_task)
    ]
    per_variant_task_stats_last: List[Dict[str, int]] = [
        {"ok": 0, "invalid": 0, "empty": 0, "errors": 0} for _ in range(variants_per_task)
    ]

    total = len(tasks)

    for ti, t in enumerate(tasks, start=1):
        task_id = t.get("task_id", "")
        student_instruction = build_student_instruction(t)

        output_type = infer_output_type(t)

        ok_variants = 0
        nonempty_variants = 0

        for v in range(variants_per_task):
            teacher_prompt = build_teacher_prompt(t, variant_id=v, variants_per_task=variants_per_task, output_type=output_type)

            messages = [{"role": "user", "content": teacher_prompt}]
            try:
                input_ids = tok.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                inputs = {"input_ids": input_ids}
            except Exception:
                inputs = tok(teacher_prompt, return_tensors="pt").to(model.device)

            text = ""
            err = ""
            try:
                with torch.no_grad():
                    gen_kwargs: Dict[str, Any] = {
                        "max_new_tokens": int(max_new_tokens),
                        "do_sample": (float(temperature) > 0.0),
                    }
                    if float(temperature) > 0.0:
                        gen_kwargs["temperature"] = float(temperature)
                    out = model.generate(**inputs, **gen_kwargs)

                prompt_len = inputs["input_ids"].shape[-1]
                text = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

                # Cut on chat special tokens if the model starts a new turn mid-completion.
                for stop in ("<|assistant|>", "<|user|>", "<|system|>"):
                    if stop in text:
                        text = text.split(stop, 1)[0].strip()
            except Exception as e:
                err = f"hf_generate_failed={type(e).__name__}: {e}"
                attempt_stats["errors"] += 1

            text = (text or "").strip()
            if text:
                nonempty_variants += 1
            else:
                attempt_stats["empty"] += 1

            ok, obj, used, strict_ok, parse_err = _parse_structured_best_effort(text, output_type)

            # Variant-per-task bucket (used for progress logs and per-variant totals).
            bucket = _variant_bucket(ok=bool(ok), text=text, err=err)
            per_variant_task_stats[v][bucket] += 1

            row: Dict[str, Any] = {
                "instruction": student_instruction,
                "output": (
                    json.dumps(obj, ensure_ascii=False)
                    if (ok and obj is not None and _normalize_output_type(output_type) == "json")
                    else (used if ok else "")
                ),
                "meta": {
                    "source": "TIGER-Lab/StructEval",
                    "task_id": task_id,
                    "output_type": _normalize_output_type(output_type),
                    "variant_id": v,
                    "teacher_model": teacher_model,
                    "teacher_backend": "hf",
                    "teacher_prompt": teacher_prompt,
                    "raw_output_metric": t.get("raw_output_metric", []),
                    "raw_generation": text,
                    "raw_used": used,
                    "parse_ok": bool(ok),
                    "syntax_strict_ok": bool(strict_ok),
                    "output_only_ok": bool(ok and _is_output_only(used, output_type)),
                    "parse_error": parse_err,
                    "error": err,
                },
            }

            if ok:
                attempt_stats["ok"] += 1
                ok_variants += 1
                out_rows.append(row)
            else:
                if text:
                    attempt_stats["invalid"] += 1
                if keep_invalid:
                    out_rows.append(row)

        # per-task stats
        if ok_variants > 0:
            task_stats["ok"] += 1
        elif nonempty_variants == 0:
            task_stats["empty"] += 1
        else:
            task_stats["invalid"] += 1

        if log_every and (ti % int(log_every) == 0 or ti == total):
            window = ti % int(log_every) if (ti % int(log_every)) else int(log_every)

            # Progress is reported at the **task** granularity (StructEval has 50 tasks).
            # Each task produces `variants_per_task` generations, which is conceptually equivalent
            # to running the same 50-task set `variants_per_task` times.
            d_by_variant: List[Dict[str, int]] = []
            sum_d = {"ok": 0, "invalid": 0, "empty": 0, "errors": 0}
            for v in range(variants_per_task):
                cur = per_variant_task_stats[v]
                last = per_variant_task_stats_last[v]
                d = {k: cur[k] - last[k] for k in cur}
                d_by_variant.append(d)
                for k in sum_d:
                    sum_d[k] += int(d[k])

            variants_done = ti * variants_per_task
            variants_total = total * variants_per_task
            if variants_per_task == 1:
                print(
                    f"[hf] progress tasks {ti}/{total} | variants {variants_done}/{variants_total} | "
                    f"window({window} tasks): success={sum_d['ok']} invalid={sum_d['invalid']} "
                    f"empty={sum_d['empty']} errors={sum_d['errors']}"
                )
            else:
                per_v_success = " ".join(
                    [f"v{v+1}={d_by_variant[v]['ok']}/{window}" for v in range(variants_per_task)]
                )
                print(
                    f"[hf] progress tasks {ti}/{total} | variants {variants_done}/{variants_total} | "
                    f"window({window} tasks): success={sum_d['ok']} invalid={sum_d['invalid']} "
                    f"empty={sum_d['empty']} errors={sum_d['errors']} | per-variant success: {per_v_success}"
                )

            # Update last snapshot
            for v in range(variants_per_task):
                per_variant_task_stats_last[v] = dict(per_variant_task_stats[v])

    print(
        "HF pseudo-SFT stats (attempts): "
        f"ok={attempt_stats['ok']} invalid={attempt_stats['invalid']} empty={attempt_stats['empty']} errors={attempt_stats['errors']}"
    )
    print(
        "HF pseudo-SFT stats (per-task): "
        f"ok={task_stats['ok']} invalid={task_stats['invalid']} empty={task_stats['empty']} errors={task_stats['errors']}"
    )
    if variants_per_task > 1:
        print(f"HF pseudo-SFT per-variant summary (tasks={total}):")
        for v in range(variants_per_task):
            s = per_variant_task_stats[v]
            print(
                f"  variant {v+1}/{variants_per_task} | success={s['ok']} invalid={s['invalid']} "
                f"empty={s['empty']} errors={s['errors']}"
            )
    return out_rows


def generate_teacher_outputs_ollama(
        tasks: List[Dict[str, Any]],
        *,
        output_type_order: Optional[List[str]] = None,
        teacher_model: str,
        host: str,
        use_chat: bool,
        num_ctx: int,
        max_new_tokens: int,
        temperature: float,
        variants_per_task: int,
        limit: int,
        seed: Optional[int],
        timeout_s: int,
        retries: int,
        format: Optional[str],
        keep_invalid: bool = False,
        log_every: int = 10,
    ) -> List[Dict[str, Any]]:
        """Generate pseudo-SFT rows from StructEval tasks using an Ollama-served teacher.

        Semantics:
          - `tasks` refers to the **base StructEval tasks**.
          - `variants_per_task` means: generate *N variants per task* (data augmentation).

        Generation order is **output-type major** so progress is easy to interpret for
        multi-format runs:

            for output_type in OUTPUT_TYPES:
                for variant in 1..N:
                    for task in tasks[output_type]:
                        teacher_generate(...)

        This matches the expectation: e.g., `csv variant 1/4`, then `json variant 1/4`, etc.
        """

        rng = random.Random(seed)

        # Keep a short, human-readable error sample for debugging.
        first_error: Optional[str] = None

        # Apply sampling limit (0 means "all")
        if limit and limit > 0:
            tasks = tasks[:limit]

        variants_per_task = max(1, int(variants_per_task))

        if variants_per_task > 1 and float(temperature) == 0.0:
            print(
                "[warn] variants_per_task > 1 but temperature=0.0. "
                "Depending on the teacher backend/model, outputs may be identical. "
                "Consider --temperature 0.2 (or similar) for diversity."
            )

        def _variant_bucket(*, ok: bool, text: str, err: str) -> str:
            """Classify a single (task, variant) generation into one of four buckets."""
            if err and not text:
                return "errors"
            if not text:
                return "empty"
            return "ok" if ok else "invalid"

        # Group tasks by output_type (normalized).
        tasks_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for t in tasks:
            ot = _normalize_output_type(infer_output_type(t))
            tasks_by_type.setdefault(ot, []).append(t)

        # Determine execution order of output types.
        canonical = ["json", "yaml", "toml", "xml", "csv"]
        if output_type_order:
            order = [_normalize_output_type(x) for x in output_type_order]
        else:
            order = canonical[:]
        # Append any unexpected types (shouldn't happen, but keep stable behavior).
        for ot in sorted(tasks_by_type.keys()):
            if ot not in order:
                order.append(ot)

        # Shuffle tasks deterministically *within each output type* (if seed is provided).
        if seed is not None:
            for i, ot in enumerate(order):
                group = tasks_by_type.get(ot, [])
                if not group:
                    continue
                rr = random.Random(int(seed) + (i + 1) * 1000003)
                rr.shuffle(group)

        out_rows: List[Dict[str, Any]] = []

        # Global attempt stats (generation-call granularity).
        attempt_stats = {"ok": 0, "invalid": 0, "empty": 0, "errors": 0, "fallback_to_generate": 0}

        # Per-type stats
        per_type_attempt_stats: Dict[str, Dict[str, int]] = {}
        per_type_task_bucket_counts: Dict[str, List[Dict[str, int]]] = {}
        per_type_per_variant_task_stats: Dict[str, List[Dict[str, int]]] = {}
        per_type_ok_variants_dist: Dict[str, Counter] = {}

        total_base_tasks = sum(len(tasks_by_type.get(ot, [])) for ot in order)
        total_attempts = total_base_tasks * variants_per_task
        global_attempt = 0

        # If user asked for Ollama JSON format, we'll apply it ONLY to json-target tasks.
        if format == "json":
            non_json_present = any(ot != "json" and len(tasks_by_type.get(ot, [])) > 0 for ot in order)
            if non_json_present:
                print(
                    "[info] --ollama-format=json is applied only to JSON-target tasks; "
                    "non-JSON tasks omit the format field."
                )

        # Execute by output type.
        for ot in order:
            group_tasks = tasks_by_type.get(ot, [])
            if not group_tasks:
                continue

            base_tasks = len(group_tasks)

            # Per-type init
            per_type_attempt_stats.setdefault(
                ot, {"ok": 0, "invalid": 0, "empty": 0, "errors": 0, "fallback_to_generate": 0}
            )
            per_type_task_bucket_counts[ot] = [
                {"ok": 0, "invalid": 0, "empty": 0, "errors": 0} for _ in range(base_tasks)
            ]
            per_type_per_variant_task_stats[ot] = [
                {"ok": 0, "invalid": 0, "empty": 0, "errors": 0} for _ in range(variants_per_task)
            ]
            per_type_ok_variants_dist[ot] = Counter()

            # Per-variant fallback counters (within this output type).
            per_variant_fallback: List[int] = [0 for _ in range(variants_per_task)]

            for v in range(variants_per_task):
                # Window stats are reported within a single variant pass (for this output type).
                w = {"ok": 0, "invalid": 0, "empty": 0, "errors": 0}
                fallback_w = 0

                for ti, t in enumerate(group_tasks, start=1):
                    global_attempt += 1

                    task_id = t.get("task_id", "")
                    student_instruction = build_student_instruction(t)
                    output_type = infer_output_type(t)

                    # Per-task system message / output type handling.
                    system_msg = build_teacher_system_msg(output_type)
                    teacher_prompt = build_teacher_prompt(
                        t,
                        variant_id=v,
                        variants_per_task=variants_per_task,
                        output_type=output_type,
                    )

                    # Ollama `format` is ONLY valid for JSON-mode. For non-JSON targets, omit it.
                    format_used = (
                        "json" if (_normalize_output_type(output_type) == "json" and format == "json") else None
                    )

                    last_err = ""
                    api_path = "chat" if use_chat else "generate"

                    try:
                        # Ollama request options are passed via a single `options` dict.
                        options: Dict[str, Any] = {
                            "num_ctx": int(num_ctx),
                            "num_predict": int(max_new_tokens),
                            "temperature": float(temperature),
                        }

                        # If a seed is provided, derive a deterministic but distinct seed per attempt.
                        if seed is not None:
                            options["seed"] = int(seed) + (global_attempt - 1)

                        if use_chat:
                            text = ollama_chat(
                                host=host,
                                model=teacher_model,
                                messages=[
                                    {"role": "system", "content": system_msg},
                                    {"role": "user", "content": teacher_prompt},
                                ],
                                options=options,
                                format=format_used,
                                timeout_s=timeout_s,
                                retries=retries,
                            )
                        else:
                            prompt_with_system = f"{system_msg}\n\n{teacher_prompt}"
                            text = ollama_generate(
                                host=host,
                                model=teacher_model,
                                prompt=prompt_with_system,
                                options=options,
                                timeout_s=timeout_s,
                                retries=retries,
                                format=format_used,
                            )
                    except Exception as e:
                        # If chat fails and we're in chat mode, try /api/generate as a fallback.
                        if use_chat:
                            try:
                                attempt_stats["fallback_to_generate"] += 1
                                per_type_attempt_stats[ot]["fallback_to_generate"] += 1
                                per_variant_fallback[v] += 1
                                fallback_w += 1
                                api_path = "generate"
                                options = {
                                    "num_ctx": int(num_ctx),
                                    "num_predict": int(max_new_tokens),
                                    "temperature": float(temperature),
                                }
                                if seed is not None:
                                    options["seed"] = int(seed) + (global_attempt - 1)
                                prompt_with_system = f"{system_msg}\n\n{teacher_prompt}"
                                text = ollama_generate(
                                    host=host,
                                    model=teacher_model,
                                    prompt=prompt_with_system,
                                    options=options,
                                    timeout_s=timeout_s,
                                    retries=retries,
                                    format=format_used,
                                )
                                last_err = f"chat_failed={type(e).__name__}: {e}"
                                if first_error is None:
                                    first_error = last_err
                            except Exception as e2:
                                text = ""
                                last_err = (
                                    f"chat_failed={type(e).__name__}: {e} "
                                    f"generate_failed={type(e2).__name__}: {e2}"
                                )
                                if first_error is None:
                                    first_error = last_err
                        else:
                            text = ""
                            last_err = f"generate_failed={type(e).__name__}: {e}"
                            if first_error is None:
                                first_error = last_err

                    text = (text or "").strip()
                    ok, obj, used, strict_ok, parse_err = _parse_structured_best_effort(text, output_type)

                    # Attempt stats (generation-call granularity)
                    if ok:
                        attempt_stats["ok"] += 1
                        per_type_attempt_stats[ot]["ok"] += 1
                    else:
                        if not text and not last_err:
                            attempt_stats["empty"] += 1
                            per_type_attempt_stats[ot]["empty"] += 1
                        elif not text and last_err:
                            attempt_stats["errors"] += 1
                            per_type_attempt_stats[ot]["errors"] += 1
                        else:
                            attempt_stats["invalid"] += 1
                            per_type_attempt_stats[ot]["invalid"] += 1

                    # Variant-per-task bucket (task granularity for this variant)
                    bucket = _variant_bucket(ok=bool(ok), text=text, err=last_err)
                    per_type_per_variant_task_stats[ot][v][bucket] += 1
                    w[bucket] += 1

                    # Per-task accumulation across variants (within output type)
                    per_type_task_bucket_counts[ot][ti - 1][bucket] += 1

                    row: Dict[str, Any] = {
                        "instruction": student_instruction,
                        "output": (
                            json.dumps(obj, ensure_ascii=False)
                            if (ok and obj is not None and _normalize_output_type(output_type) == "json")
                            else (used if ok else "")
                        ),
                        "meta": {
                            "source": "TIGER-Lab/StructEval",
                            "task_id": task_id,
                            "output_type": _normalize_output_type(output_type),
                            "variant_id": v,
                            "teacher_model": teacher_model,
                            "teacher_backend": "ollama",
                            "ollama_host": host,
                            "ollama_api": api_path,
                            "ollama_format": format_used,
                            "teacher_prompt": teacher_prompt,
                            "raw_output_metric": t.get("raw_output_metric", []),
                            "raw_generation": text,
                            "raw_used": used,
                            "parse_ok": bool(ok),
                            "syntax_strict_ok": bool(strict_ok),
                            "output_only_ok": bool(ok and _is_output_only(used, output_type)),
                            "parse_error": parse_err,
                            "error": last_err,
                        },
                    }

                    if ok:
                        out_rows.append(row)
                    elif keep_invalid:
                        out_rows.append(row)

                    # Progress logs (within a single variant pass for this output type).
                    if log_every and log_every > 0 and (ti % int(log_every) == 0 or ti == base_tasks):
                        window = ti % int(log_every) if (ti % int(log_every)) else int(log_every)
                        print(
                            f"[ollama] {ot} variant {v+1}/{variants_per_task} | base_tasks={base_tasks} | "
                            f"progress {ti}/{base_tasks} tasks | attempts {global_attempt}/{total_attempts} | "
                            f"window({window} tasks): success={w['ok']} invalid={w['invalid']} "
                            f"empty={w['empty']} errors={w['errors']} | fallback_to_generate={fallback_w}"
                        )
                        w = {"ok": 0, "invalid": 0, "empty": 0, "errors": 0}
                        fallback_w = 0

                # End-of-variant summary for this output type
                s = per_type_per_variant_task_stats[ot][v]
                print(
                    f"[ollama] finished {ot} variant {v+1}/{variants_per_task} | base_tasks={base_tasks} | "
                    f"success={s['ok']} invalid={s['invalid']} empty={s['empty']} errors={s['errors']} | "
                    f"fallback_to_generate={per_variant_fallback[v]}"
                )

            # Per-task stats for this output type
            task_stats = {"ok": 0, "invalid": 0, "empty": 0, "errors": 0}
            for c in per_type_task_bucket_counts[ot]:
                ok_n = int(c.get("ok", 0))
                if ok_n > 0:
                    task_stats["ok"] += 1
                else:
                    if int(c.get("errors", 0)) > 0 and (
                        int(c.get("empty", 0)) + int(c.get("invalid", 0)) == 0
                    ):
                        task_stats["errors"] += 1
                    elif int(c.get("empty", 0)) == variants_per_task:
                        task_stats["empty"] += 1
                    else:
                        task_stats["invalid"] += 1
                per_type_ok_variants_dist[ot][ok_n] += 1

            print(
                f"Ollama pseudo-SFT stats [{ot}]: "
                f"base_tasks={base_tasks} variants_per_task={variants_per_task} max_rows={base_tasks * variants_per_task}"
            )
            print(
                f"Ollama pseudo-SFT stats (attempts) [{ot}]: "
                f"ok={per_type_attempt_stats[ot]['ok']} invalid={per_type_attempt_stats[ot]['invalid']} "
                f"empty={per_type_attempt_stats[ot]['empty']} errors={per_type_attempt_stats[ot]['errors']} "
                f"fallback_to_generate={per_type_attempt_stats[ot]['fallback_to_generate']}"
            )
            print(
                f"Ollama pseudo-SFT stats (per-task) [{ot}]: "
                f"ok={task_stats['ok']} invalid={task_stats['invalid']} "
                f"empty={task_stats['empty']} errors={task_stats['errors']}"
            )

            if variants_per_task > 1:
                print(
                    f"Ollama pseudo-SFT ok variants per task [{ot}]:",
                    dict(sorted(per_type_ok_variants_dist[ot].items())),
                )
                print(f"Ollama pseudo-SFT per-variant summary [{ot}] (base_tasks={base_tasks}):")
                for vv in range(variants_per_task):
                    ss = per_type_per_variant_task_stats[ot][vv]
                    print(
                        f"  {ot} variant {vv+1}/{variants_per_task} | success={ss['ok']} invalid={ss['invalid']} "
                        f"empty={ss['empty']} errors={ss['errors']}"
                    )

        # Global summary
        print(
            "Ollama pseudo-SFT stats (global): "
            f"base_tasks={total_base_tasks} variants_per_task={variants_per_task} max_rows={total_attempts}"
        )
        print(
            "Ollama pseudo-SFT stats (attempts, global): "
            f"ok={attempt_stats['ok']} invalid={attempt_stats['invalid']} "
            f"empty={attempt_stats['empty']} errors={attempt_stats['errors']} "
            f"fallback_to_generate={attempt_stats['fallback_to_generate']}"
        )

        if first_error and attempt_stats.get("errors", 0) > 0:
            print("First Ollama error (sample):", first_error)
            if attempt_stats.get("ok", 0) == 0:
                print(
                    "Hint: if you see only errors/empty, try forcing the other Ollama API path: "
                    "--ollama-use-chat or --ollama-use-generate (depending on your server/model)."
                )

        return out_rows


def save_jsonl(rows: List[Dict[str, Any]], out_path: str | Path, *, require_valid: bool = True) -> int:
    """Write pseudo-SFT rows to JSONL.

    If require_valid=True (default), only rows with meta.parse_ok=True are written
    and empty outputs are skipped. If require_valid=False (used with --keep-invalid),
    rows are written even when output is empty so that downstream debugging can see
    every task.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open('w', encoding='utf-8') as f:
        for r in rows:
            if require_valid and not r.get('meta', {}).get('parse_ok', False):
                continue
            if require_valid and not r.get('output'):
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build pseudo-SFT JSONL from StructEval tasks using a teacher model. "
            "Supports multiple output types (json/yaml/toml/xml/csv) when present in the input."
        )
    )
    ap.add_argument(
        "--input",
        default="data/structeval_json_eval.json",
        help="Input eval JSON created by import_structeval.",
    )
    ap.add_argument(
        "--output",
        dest="output",
        default="data/structeval_pseudo_sft.jsonl",
        help="Output JSONL for SFT (instruction/output). If --split-by-output-type is set, this is treated as an output directory.",
    )
    ap.add_argument("--out", dest="output", help="(deprecated) same as --output")

    # Optional dataset config (to avoid duplicating variants_per_task, etc.)
    ap.add_argument(
        "--dataset-sft-config",
        default=None,
        help="Optional path to configs/dataset_sft.yaml. When provided, pseudo_sft.* values can be used as defaults.",
    )

    ap.add_argument("--teacher-model", default="Qwen/Qwen3-4B-Instruct-2507", help="Teacher model name/path.")
    ap.add_argument(
        "--teacher-backend",
        choices=["hf", "ollama"],
        default="hf",
        help="Teacher backend: hf (transformers) or ollama (HTTP).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--variants-per-task",
        type=int,
        default=4,
        help="Number of teacher generations per task (data augmentation).",
    )
    ap.add_argument("--limit", type=int, default=0, help="0=all, else first N tasks")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Shuffle tasks deterministically and (when supported) seed teacher sampling.",
    )
    ap.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep invalid/empty generations (not recommended for training).",
    )
    ap.add_argument("--log-every", type=int, default=10, help="Print progress every N tasks (0 disables).")

    ap.add_argument(
        "--output-types",
        default=None,
        help=(
            "Optional comma-separated list of output types to generate (json,yaml,toml,xml,csv). "
            "If omitted, all types present in --input are generated."
        ),
    )
    ap.add_argument(
        "--split-by-output-type",
        action="store_true",
        help="Write one JSONL per output type under --output (treated as a directory).",
    )

    # Ollama options
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434", help="Ollama base URL.")
    ap.add_argument("--ollama-num-ctx", type=int, default=16384, help="Ollama num_ctx (context length).")
    ap.add_argument("--ollama-timeout-s", type=int, default=300, help="HTTP timeout seconds.")
    ap.add_argument("--ollama-retries", type=int, default=2, help="Retry count for transient HTTP errors.")
    ap.add_argument(
        "--ollama-format",
        choices=["json", "none", "text"],
        default="json",
        help=(
            "If json, ask Ollama to return JSON for JSON-target tasks. "
            "For non-JSON tasks, the format field is omitted automatically. "
            "Use none to omit format even for JSON tasks. (text is deprecated; treated as none.)"
        ),
    )

    # NOTE: gpt-oss:20b on Ollama is typically served via /api/generate.
    # Use /api/chat only when the model & server support chat.
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--ollama-use-chat", action="store_true", help="Use /api/chat (chat-style messages).")
    grp.add_argument("--ollama-use-generate", action="store_true", help="Use /api/generate (prompt-only).")

    argv0 = argv if argv is not None else sys.argv[1:]
    args = ap.parse_args(argv0)

    # Apply defaults from dataset_sft config (only if the CLI did not explicitly set them).
    if args.dataset_sft_config:
        cfg_path = Path(args.dataset_sft_config)
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            pseudo = (cfg.get("pseudo_sft") or {})

            def _cli_has(flag: str) -> bool:
                return flag in argv0

            if (not _cli_has("--variants-per-task")) and pseudo.get("variants_per_task"):
                args.variants_per_task = int(pseudo["variants_per_task"])

            if (not _cli_has("--output-types")) and pseudo.get("output_types"):
                args.output_types = ",".join([str(x) for x in pseudo["output_types"]])

            if (not _cli_has("--split-by-output-type")) and pseudo.get("split_by_output_type"):
                args.split_by_output_type = True
        else:
            print(f"[warn] --dataset-sft-config not found: {cfg_path}")

    tasks = load_eval_json(args.input)

    # Optional filtering by output type.
    desired_types = None
    output_type_order: Optional[List[str]] = None
    if args.output_types:
        output_type_order = [_normalize_output_type(x.strip()) for x in str(args.output_types).split(",") if x.strip()]
        # Preserve order for progress reporting; also build a set for filtering.
        desired_types = set(output_type_order)

    if desired_types:
        tasks = [t for t in tasks if _normalize_output_type(infer_output_type(t)) in desired_types]

    # Print a quick summary of the input tasks by output_type.
    from collections import Counter

    cnt = Counter([_normalize_output_type(infer_output_type(t)) for t in tasks])
    if cnt:
        print("Input tasks by output_type:", dict(sorted(cnt.items())))

    if args.teacher_backend == "ollama":
        # Default to /api/chat for Ollama because many instruct/chat models behave
        # better there. You can force /api/generate via --ollama-use-generate.
        use_chat = True
        if getattr(args, "ollama_use_generate", False):
            use_chat = False
        elif getattr(args, "ollama_use_chat", False):
            use_chat = True

        rows = generate_teacher_outputs_ollama(
            tasks,
            output_type_order=output_type_order,
            teacher_model=args.teacher_model,
            host=args.ollama_host,
            use_chat=use_chat,
            num_ctx=args.ollama_num_ctx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            variants_per_task=args.variants_per_task,
            limit=args.limit,
            seed=args.seed,
            timeout_s=args.ollama_timeout_s,
            retries=args.ollama_retries,
            format=(None if (args.ollama_format in ("none", "text")) else args.ollama_format),
            keep_invalid=bool(args.keep_invalid),
            log_every=int(args.log_every),
        )
    else:
        rows = generate_teacher_outputs_hf(
            tasks,
            teacher_model=args.teacher_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            variants_per_task=args.variants_per_task,
            limit=args.limit,
            seed=args.seed,
            log_every=int(args.log_every),
            keep_invalid=bool(args.keep_invalid),
        )

    # Write outputs (single file or split).
    if args.split_by_output_type:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        by_type = {}
        for r in rows:
            ot = (r.get("meta") or {}).get("output_type", "json")
            by_type.setdefault(ot, []).append(r)
        total_written = 0
        for ot, rws in sorted(by_type.items()):
            out_path = out_dir / f"structeval_pseudo_sft_{ot}.jsonl"
            n = save_jsonl(rws, out_path, require_valid=(not args.keep_invalid))
            total_written += n
            print(f"Wrote {n} rows to {out_path}")
        print(f"Total written: {total_written} rows under {out_dir}")
    else:
        written = save_jsonl(rows, args.output, require_valid=(not args.keep_invalid))
        print(f"Wrote {written} SFT rows to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

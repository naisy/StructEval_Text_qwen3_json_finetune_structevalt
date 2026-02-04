from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseJudge
from ..cache import load_cache, save_cache, cache_path
from ..prompt import JUDGE_SYSTEM, build_judge_input
from ..types import JudgeResult
from ...data.validators import parse_json
from ...utils.logging import warn


def _model_rejects_temperature(model: str) -> bool:
    # Observed: gpt-5.* rejects `temperature` (400 Unsupported parameter).
    return model.startswith("gpt-5")


class OpenAIJudge(BaseJudge):
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: int = 256,
        timeout_s: int = 60,
        reasoning_effort: Optional[str] = None,
        cache_enabled: bool = True,
        cache_dir: str = "outputs/cache/judge",
    ) -> None:
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(f"OpenAI judge enabled but env var {api_key_env} is not set.")
        self.base_url = base_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)

    def judge(self, *, task: Dict[str, Any], generation: str) -> JudgeResult:
        task_id = str(task.get("task_id", "")) if isinstance(task, dict) else ""

        payload: Dict[str, Any] = {
            "system": JUDGE_SYSTEM,
            "input": build_judge_input(task, generation),
            "max_output_tokens": self.max_output_tokens,
            "model": self.model,
        }
        # Only include temperature if explicitly set and model supports it.
        if (self.temperature is not None) and (not _model_rejects_temperature(self.model)):
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        cache_k, cache_p = cache_path(self.cache_dir, "openai", self.model, payload)

        # Cache hit
        if self.cache_enabled:
            cached = load_cache(self.cache_dir, "openai", self.model, payload)
            if cached is not None:
                # Backfill task_id into existing cache entry for easier debugging.
                if task_id and ("task_id" not in cached) and ("task_ids" not in cached):
                    cached2 = dict(cached)
                    cached2["task_id"] = task_id
                    save_cache(self.cache_dir, "openai", self.model, payload, cached2)
                    cached = cached2

                cached["_cache_key"] = cache_k
                cached["_cache_file"] = str(cache_p)
                cached["_cache_hit"] = True
                if task_id:
                    cached.setdefault("task_id", task_id)
                return _to_result(cached)

        # Call OpenAI
        try:
            from openai import OpenAI  # optional dependency
        except Exception as e:
            value = {
                "score": 0.0,
                "passed": False,
                "breakdown": {"json_parse": 0, "path_coverage": 0.0, "constraint_following": 0.0},
                "failed_checks": [f"openai_import_error: {type(e).__name__}: {e}"],
                "notes": "OpenAI SDK not installed or failed to import.",
                "task_id": task_id,
            }
            if self.cache_enabled:
                save_cache(self.cache_dir, "openai", self.model, payload, value)
            value["_cache_key"] = cache_k
            value["_cache_file"] = str(cache_p)
            value["_cache_hit"] = False
            return _to_result(value)

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        req: Dict[str, Any] = {
            "model": self.model,
            "instructions": payload["system"],
            "input": payload["input"],
            "max_output_tokens": self.max_output_tokens,
        }

        # Force JSON output from the judge. This dramatically reduces
        # "judge_output_not_json" failures when the candidate output contains
        # quotes, newlines, or long nested structures.
        # (Uses Responses API text.format JSON mode / structured output.)
        req["text"] = {"format": {"type": "json_object"}}

        if (self.temperature is not None) and (not _model_rejects_temperature(self.model)):
            req["temperature"] = self.temperature
        elif (self.temperature is not None) and _model_rejects_temperature(self.model):
            warn(f"OpenAI model '{self.model}' rejects temperature; omitting it.")

        if self.reasoning_effort:
            # Responses API
            req["reasoning"] = {"effort": self.reasoning_effort}

        try:
            resp = client.responses.create(**req)
            text = getattr(resp, "output_text", None) or ""
            if not text:
                try:
                    text = resp.output[0].content[0].text  # type: ignore[attr-defined]
                except Exception:
                    text = ""
        except Exception as e:
            value = {
                "score": 0.0,
                "passed": False,
                "breakdown": {"json_parse": 0, "path_coverage": 0.0, "constraint_following": 0.0},
                "failed_checks": [f"openai_api_error: {type(e).__name__}: {e}"],
                "notes": "OpenAI judge request failed.",
                "task_id": task_id,
            }
            if self.cache_enabled:
                save_cache(self.cache_dir, "openai", self.model, payload, value)
            value["_cache_key"] = cache_k
            value["_cache_file"] = str(cache_p)
            value["_cache_hit"] = False
            return _to_result(value)

        ok, obj, err = parse_json(text)
        if not ok:
            value: Dict[str, Any] = {
                "score": 0.0,
                "passed": False,
                "breakdown": {"json_parse": 0, "path_coverage": 0.0, "constraint_following": 0.0},
                "failed_checks": [f"judge_output_not_json: {err}"],
                "notes": "Judge model did not return valid JSON.",
                "_raw_text": text,
                "task_id": task_id,
            }
        else:
            value = dict(obj)
            value["_raw_text"] = text
            if task_id:
                value.setdefault("task_id", task_id)

        # Persist cache with task_id for debugging; keep filename (hash) unchanged.
        if self.cache_enabled:
            save_cache(self.cache_dir, "openai", self.model, payload, dict(value))

        # Add non-persisted metadata for downstream analysis.
        value["_cache_key"] = cache_k
        value["_cache_file"] = str(cache_p)
        value["_cache_hit"] = False
        return _to_result(value)


def _to_result(value: Dict[str, Any]) -> JudgeResult:
    score = float(value.get("score", 0.0))
    passed = bool(value.get("passed", False))
    raw_text = value.get("_raw_text")
    details = {k: v for k, v in value.items() if k != "_raw_text"}
    return JudgeResult(score=score, passed=passed, details=details, raw_text=raw_text)

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


class GeminiJudge(BaseJudge):
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "GOOGLE_API_KEY",
        temperature: Optional[float] = None,
        max_output_tokens: int = 256,
        timeout_s: int = 60,
        cache_enabled: bool = True,
        cache_dir: str = "outputs/cache/judge",
        use_vertexai_env: str = "GOOGLE_GENAI_USE_VERTEXAI",
    ) -> None:
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        self.use_vertexai = os.environ.get(use_vertexai_env, "false").lower() in ("1", "true", "yes")
        if (not self.api_key) and (not self.use_vertexai):
            raise RuntimeError(
                f"Gemini judge enabled but env var {api_key_env} is not set (or set {use_vertexai_env}=true and use ADC)."
            )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
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
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        cache_k, cache_p = cache_path(self.cache_dir, "gemini", self.model, payload)

        if self.cache_enabled:
            cached = load_cache(self.cache_dir, "gemini", self.model, payload)
            if cached is not None:
                if task_id and ("task_id" not in cached) and ("task_ids" not in cached):
                    cached2 = dict(cached)
                    cached2["task_id"] = task_id
                    save_cache(self.cache_dir, "gemini", self.model, payload, cached2)
                    cached = cached2
                cached["_cache_key"] = cache_k
                cached["_cache_file"] = str(cache_p)
                cached["_cache_hit"] = True
                if task_id:
                    cached.setdefault("task_id", task_id)
                return _to_result(cached)

        # Call Gemini (google-genai)
        try:
            from google import genai  # type: ignore
        except Exception as e:
            value = {
                "score": 0.0,
                "passed": False,
                "breakdown": {"json_parse": 0, "path_coverage": 0.0, "constraint_following": 0.0},
                "failed_checks": [f"gemini_import_error: {type(e).__name__}: {e}"],
                "notes": "google-genai SDK not installed or failed to import.",
                "task_id": task_id,
            }
            if self.cache_enabled:
                save_cache(self.cache_dir, "gemini", self.model, payload, value)
            value["_cache_key"] = cache_k
            value["_cache_file"] = str(cache_p)
            value["_cache_hit"] = False
            return _to_result(value)

        try:
            if self.use_vertexai:
                client = genai.Client(vertexai=True)
            else:
                client = genai.Client(api_key=self.api_key)
        except Exception as e:
            value = {
                "score": 0.0,
                "passed": False,
                "breakdown": {"json_parse": 0, "path_coverage": 0.0, "constraint_following": 0.0},
                "failed_checks": [f"gemini_client_error: {type(e).__name__}: {e}"],
                "notes": "Gemini client init failed.",
                "task_id": task_id,
            }
            if self.cache_enabled:
                save_cache(self.cache_dir, "gemini", self.model, payload, value)
            value["_cache_key"] = cache_k
            value["_cache_file"] = str(cache_p)
            value["_cache_hit"] = False
            return _to_result(value)

        contents = payload["system"] + "\n\n" + payload["input"]

        gen_cfg: Dict[str, Any] = {"max_output_tokens": self.max_output_tokens}
        if self.temperature is not None:
            gen_cfg["temperature"] = self.temperature

        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=gen_cfg,
            )
            text = getattr(resp, "text", None) or ""
        except Exception as e:
            value = {
                "score": 0.0,
                "passed": False,
                "breakdown": {"json_parse": 0, "path_coverage": 0.0, "constraint_following": 0.0},
                "failed_checks": [f"gemini_api_error: {type(e).__name__}: {e}"],
                "notes": "Gemini judge request failed.",
                "task_id": task_id,
            }
            if self.cache_enabled:
                save_cache(self.cache_dir, "gemini", self.model, payload, value)
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

        if self.cache_enabled:
            save_cache(self.cache_dir, "gemini", self.model, payload, dict(value))

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

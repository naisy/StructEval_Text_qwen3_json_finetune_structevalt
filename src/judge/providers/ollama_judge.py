from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseJudge
from ..cache import load_cache, save_cache, cache_path
from ..prompt import JUDGE_SYSTEM, build_judge_input
from ..types import JudgeResult
from ...data.validators import parse_json
from ...utils.ollama import ollama_chat, ollama_generate, OllamaError
from ...utils.logging import warn


def _to_result(value: Dict[str, Any]) -> JudgeResult:
    score = float(value.get("score", 0.0))
    passed = bool(value.get("passed", False))
    details = dict(value)
    raw_text = value.get("raw_text")
    return JudgeResult(score=score, passed=passed, details=details, raw_text=raw_text if isinstance(raw_text, str) else None)


class OllamaJudge(BaseJudge):
    def __init__(
        self,
        *,
        model: str,
        host: str = "http://127.0.0.1:11434",
        use_chat: bool = True,
        temperature: Optional[float] = None,
        max_output_tokens: int = 256,
        timeout_s: int = 120,
        num_ctx: Optional[int] = 16384,
        cache_enabled: bool = True,
        cache_dir: str = "outputs/cache/judge",
    ) -> None:
        self.model = model
        self.host = host
        self.use_chat = use_chat
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)

    def judge(self, *, task: Dict[str, Any], generation: str) -> JudgeResult:
        task_id = str(task.get("task_id", "")) if isinstance(task, dict) else ""

        options: Dict[str, Any] = {"num_predict": int(self.max_output_tokens)}
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)
        if self.temperature is not None:
            options["temperature"] = float(self.temperature)

        payload: Dict[str, Any] = {
            "system": JUDGE_SYSTEM,
            "input": build_judge_input(task, generation),
            "model": self.model,
            "options": options,
            "use_chat": self.use_chat,
        }

        cache_k, cache_p = cache_path(self.cache_dir, "ollama", self.model, payload)

        if self.cache_enabled:
            cached = load_cache(self.cache_dir, "ollama", self.model, payload)
            if cached is not None:
                if task_id:
                    cached.setdefault("task_id", task_id)
                cached["_cache_key"] = cache_k
                cached["_cache_file"] = str(cache_p)
                cached["_cache_hit"] = True
                return _to_result(cached)

        # Call Ollama
        try:
            if self.use_chat:
                text = ollama_chat(
                    host=self.host,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": payload["input"]},
                    ],
                    options=options,
                    timeout_s=self.timeout_s,
                )
            else:
                # Fallback to generate endpoint
                text = ollama_generate(
                    host=self.host,
                    model=self.model,
                    prompt=JUDGE_SYSTEM + "\n\n" + payload["input"],
                    options=options,
                    timeout_s=self.timeout_s,
                )
        except OllamaError as e:
            value = {
                "score": 0.0,
                "passed": False,
                "breakdown": {"json_parse": 0, "path_coverage": 0.0, "constraint_following": 0.0},
                "failed_checks": [f"ollama_error: {type(e).__name__}: {e}"],
                "notes": "Failed to call Ollama judge.",
                "task_id": task_id,
                "raw_text": None,
            }
            if self.cache_enabled:
                save_cache(self.cache_dir, "ollama", self.model, payload, value)
            value["_cache_key"] = cache_k
            value["_cache_file"] = str(cache_p)
            value["_cache_hit"] = False
            return _to_result(value)

        ok, obj, _ = parse_json(text)
        if not ok or not isinstance(obj, dict):
            value = {
                "score": 0.0,
                "passed": False,
                "breakdown": {"json_parse": 0, "path_coverage": 0.0, "constraint_following": 0.0},
                "failed_checks": ["judge_output_not_json"],
                "notes": "Judge did not return valid JSON.",
                "task_id": task_id,
                "raw_text": text,
            }
        else:
            value = dict(obj)
            value.setdefault("task_id", task_id)
            value["raw_text"] = text

        if self.cache_enabled:
            save_cache(self.cache_dir, "ollama", self.model, payload, value)

        value["_cache_key"] = cache_k
        value["_cache_file"] = str(cache_p)
        value["_cache_hit"] = False
        return _to_result(value)

"""Ollama HTTP client helpers.

We call Ollama's REST API directly (no optional python client dependency) so the
behavior is stable and matches the documented endpoints.

Relevant API endpoints (non-streaming):
  - POST /api/chat     {model, messages, stream, format?, options?, keep_alive?}
  - POST /api/generate {model, prompt,   stream, format?, options?, keep_alive?}

See: Ollama "API" documentation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests


class OllamaError(RuntimeError):
    pass


def _join_url(host: str, path: str) -> str:
    host = host.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return host + path


def _post_json_with_retries(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout_s: int,
    retries: int,
) -> Dict[str, Any]:
    last_err: Optional[BaseException] = None
    for attempt in range(max(0, retries) + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout_s)
            # Prefer a helpful message if Ollama returns JSON errors.
            if resp.status_code >= 400:
                try:
                    j = resp.json()
                except Exception:
                    j = {"error": resp.text}
                raise OllamaError(f"HTTP {resp.status_code}: {j}")
            return resp.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= retries:
                break
            # Basic backoff: 0.5s, 1s, 2s, ...
            time.sleep(0.5 * (2**attempt))

    raise OllamaError(f"POST {url} failed after {retries+1} attempts: {last_err}")


def ollama_chat(
    host: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    format: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    keep_alive: Optional[str] = None,
    timeout_s: int = 600,
    retries: int = 2,
) -> str:
    """Call POST /api/chat and return the assistant message content."""
    url = _join_url(host, "/api/chat")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if format is not None:
        payload["format"] = format
    if options:
        payload["options"] = options
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    data = _post_json_with_retries(url, payload, timeout_s=timeout_s, retries=retries)
    msg = (data.get("message") or {})
    content = msg.get("content")
    if not isinstance(content, str):
        raise OllamaError(f"Unexpected /api/chat response shape: {data}")
    return content


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    format: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    keep_alive: Optional[str] = None,
    timeout_s: int = 600,
    retries: int = 2,
) -> str:
    """Call POST /api/generate and return the completion text.

    Note: /api/generate does not have a separate "system" field. If you provide
    `system`, we prefix it to the prompt.
    """
    url = _join_url(host, "/api/generate")

    final_prompt = prompt
    if system:
        final_prompt = f"{system}\n\n{prompt}".strip()

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": final_prompt,
        "stream": False,
    }
    if format is not None:
        payload["format"] = format
    if options:
        payload["options"] = options
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    data = _post_json_with_retries(url, payload, timeout_s=timeout_s, retries=retries)
    text = data.get("response")
    if not isinstance(text, str):
        raise OllamaError(f"Unexpected /api/generate response shape: {data}")
    return text

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import BaseJudge
from .composite import CompositeJudge
from .providers.openai_judge import OpenAIJudge
from .providers.gemini_judge import GeminiJudge
from .providers.ollama_judge import OllamaJudge


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    # allow empty string -> None
    if isinstance(v, str) and v.strip() == "":
        return None
    return float(v)


def _merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge b over a (dict values only)."""
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = v
    return out


def _build_single_judge(judge_cfg: Dict[str, Any]) -> BaseJudge:
    provider = judge_cfg.get("provider", "openai")
    model = judge_cfg.get("model")
    if not model:
        raise ValueError("judge.model is required when judge.enabled=true")

    temperature = _opt_float(judge_cfg.get("temperature", None))
    max_output_tokens = int(judge_cfg.get("max_output_tokens", 256))
    timeout_s = int(judge_cfg.get("timeout_s", 60))
    cache_cfg = judge_cfg.get("cache", {}) or {}
    cache_enabled = bool(cache_cfg.get("enabled", True))
    cache_dir = str(cache_cfg.get("dir", "outputs/cache/judge"))

    if provider == "openai":
        ocfg = judge_cfg.get("openai", {}) or {}
        return OpenAIJudge(
            model=model,
            api_key_env=str(ocfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url=ocfg.get("base_url"),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            reasoning_effort=ocfg.get("reasoning_effort"),
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

    if provider == "gemini":
        gcfg = judge_cfg.get("gemini", {}) or {}
        return GeminiJudge(
            model=model,
            api_key_env=str(gcfg.get("api_key_env", "GOOGLE_API_KEY")),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
            use_vertexai_env=str(gcfg.get("use_vertexai_env", "GOOGLE_GENAI_USE_VERTEXAI")),
        )

    if provider == "ollama":
        ocfg = judge_cfg.get("ollama", {}) or {}
        return OllamaJudge(
            model=model,
            host=str(ocfg.get("host", "http://127.0.0.1:11434")),
            use_chat=bool(ocfg.get("use_chat", True)),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            num_ctx=int(ocfg.get("num_ctx", 16384)),
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

    raise ValueError(f"Unknown judge.provider: {provider}")


def build_judge(judge_cfg: Dict[str, Any]) -> BaseJudge:
    """
    Build judge from config.

    Backward compatible:
      - Old style: {provider: openai, model: gpt-5.2, ...}
      - New style: {providers: [{provider: openai, model: ...}, {provider: ollama, model: ...}], ...}

    New style enables provider fallback (e.g., when OPENAI_API_KEY is not set).
    """
    providers = judge_cfg.get("providers")
    if isinstance(providers, list) and providers:
        # Base config acts as defaults for each provider entry.
        base = dict(judge_cfg)
        base.pop("providers", None)

        built: List[Tuple[str, BaseJudge]] = []
        for i, spec in enumerate(providers):
            if not isinstance(spec, dict):
                raise ValueError(f"judge.providers[{i}] must be a mapping")
            merged = dict(base)

            # spec overrides base (including provider-specific blocks if provided)
            merged = _merge_dict(merged, spec)

            # provider-specific nested blocks: allow partial override
            for nested in ("openai", "gemini", "ollama"):
                if nested in base or nested in spec:
                    merged[nested] = _merge_dict(base.get(nested, {}) or {}, spec.get(nested, {}) or {})

            prov = str(merged.get("provider", "openai"))
            mdl = str(merged.get("model", ""))
            name = f"{prov}:{mdl}" if mdl else prov
            try:
                built.append((name, _build_single_judge(merged)))
            except Exception:
                # Skip providers that cannot be initialized (e.g., missing API key).
                # The CompositeJudge will use the next provider.
                continue

        if not built:
            raise RuntimeError("No judge providers could be initialized. Check API keys / configs.")

        return CompositeJudge(built)

    # old single-provider config
    return _build_single_judge(judge_cfg)

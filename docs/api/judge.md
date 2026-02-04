# Judge (LLM-as-a-judge)

This module adds an optional, pluggable evaluator LLM that scores **instruction/constraint following**
based on the task prompt (`query` / `feature_requirements` / `raw_output_metric`) and the candidate output.
It complements StructEval-T, which primarily checks parseability and required key paths.

## Config

See `configs/judge.yaml`.

### Provider priority / fallback

`configs/judge.yaml` supports **multiple providers in priority order**:

- `judge.providers`: list of provider specs (first match wins)
- If a provider fails (e.g., missing API key, temporary outage), the next provider is tried automatically.

This is useful when you sometimes have `OPENAI_API_KEY` set (OpenAI), sometimes have `GOOGLE_API_KEY` set (Gemini),
and want a **local fallback** (Ollama) when neither is available.

### Key fields

- `judge.enabled`: enable/disable
- `judge.providers[*].provider`: `openai` | `gemini` | `ollama`
- `judge.providers[*].model`: model id / string
- `judge.cache`: cache settings (recommended)

Backward compatibility: the older single-provider form (`judge.provider` / `judge.model`) is still supported.

## Credentials

- OpenAI: set `OPENAI_API_KEY`
- Gemini API: set `GOOGLE_API_KEY`
- Vertex AI: set `GOOGLE_GENAI_USE_VERTEXAI=true` and authenticate via ADC.
- Ollama: no key required (configure `host`)

## Output

When enabled, `train.py eval` adds a per-sample section:

```json
"judge": {
  "provider": "openai:gpt-5.2",
  "model": "gpt-5.2",
  "score": 0.83,
  "passed": true,
  "details": { "...": "..." }
}
```

`provider` indicates which provider was actually used (after fallback).
The summary report also includes `judge_score_avg`.

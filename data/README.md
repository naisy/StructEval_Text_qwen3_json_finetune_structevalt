# Data folder

Place your dataset files here, e.g.:
- `train.jsonl`
- `valid.jsonl`
- `schema.json` (optional)

Example JSONL line:
```json
{"instruction":"Summarize metadata about a fictional scientific article.","requirements":["Top-level field \"title\" is a string", "Field \"authors\" is a list of exactly two items"],"output":"{\"title\":\"...\",\"authors\":[{\"name\":\"...\",\"affiliation\":\"...\"},{\"name\":\"...\",\"affiliation\":\"...\"}],\"publication\":{\"year\":2024},\"keywords\":[\"...\" ]}"}
```

**Important:** `output` must be JSON only (no Markdown fences).

from __future__ import annotations

"""Format-specific output rules injected into the system prompt.

Why this exists
--------------
Even when the training targets are syntactically valid, models often leak
JSON/YAML habits into TOML, repeat table headers instead of using
array-of-tables, or emit trailing commas.

During GRPO training we do not rely on an external LLM judge; rewards are
computed by deterministic parsers (StructEval-T / local validators). If the
model output is not parseable, reward collapses.

Therefore we inject lightweight, format-specific rules into the system prompt
based on `output_type`.
"""

from typing import Dict


FORMAT_RULES: Dict[str, str] = {
    "JSON": """\
JSON rules (must follow):
- Output MUST be valid JSON (RFC 8259).
- Use double quotes for all strings and object keys.
- Do NOT use trailing commas.
- Do NOT include comments.
- Output ONLY the JSON value (no prose, no markdown, no code fences).
""",
    "YAML": """\
YAML rules (must follow):
- Output MUST be valid YAML (YAML 1.2).
- Use spaces for indentation (no tabs).
- Output a single YAML document (do not add '---' / '...').
- Do NOT include comments.
- Output ONLY the YAML content (no prose, no markdown, no code fences).
""",
    "TOML": """\
TOML rules (must follow):
- Use '=' for assignments. Never use ':' (JSON/YAML style).
- Do NOT use trailing commas in arrays: [a, b] OK, [a, b,] NG.
- Represent objects as tables:
  - Use [table] for an object.
  - Use [[table]] for an array of objects (array-of-tables).
- Never assign a table name as a value if you also use it as a table/array-table:
  - NG: museum = "x" then [museum] / [[museum]]
  - OK: [museum] name = "x"
- If you need multiple objects of the same type, use [[...]] (not repeating [..] blocks).
- Output ONLY TOML (no prose, no markdown, no code fences).
""",
    "XML": """\
XML rules (must follow):
- Output MUST be well-formed XML.
- Use exactly ONE root element that contains the entire document.
- Close all tags properly and quote all attribute values.
- Escape special characters in text/attributes (&, <, >, \").
- Preserve field/tag names exactly as given (including underscores `_`).
  - Example: `some_thing` must stay as `<some_thing>...` (do NOT split into `<some><thing>`).
- Output ONLY the XML (no prose, no markdown, no code fences).
""",
    "CSV": """\
CSV rules (must follow):
- Output MUST be valid CSV (RFC 4180 style).
- First row MUST be a header.
- Every row MUST have the same number of columns as the header.
- If a field contains a comma, quote, or newline, wrap it in double quotes and
  escape internal quotes by doubling them.
- Output ONLY the CSV text (no prose, no markdown, no code fences).
""",
}

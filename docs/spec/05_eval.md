# Evaluation

## StructEval-T
Evaluation uses StructEval-T across output types:
- JSON / YAML / TOML / XML / CSV
- `configs/eval.yaml` controls the per-format evaluation count (e.g. `limit_per_output_type`).

## Syntax checks
The repo uses deterministic parsers and "format-only" checks per type:
- JSON: `json.loads`
- YAML: `yaml.safe_load`
- TOML: `tomllib.loads`
- XML: `xml.etree.ElementTree`
- CSV: `csv.reader`

Canonical style checks:
- YAML: 2-space indentation + block style
- TOML: canonicalized rendering similarity

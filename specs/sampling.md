# Sampling and balancing

Training quality is sensitive to task distribution.

- If JSON-heavy, the model may output JSON-like indentation in YAML.
- If TOML is underrepresented, TOML tasks may have more syntax errors.

`configs/*_hf.yaml` supports two sampling strategies.

## 1) balance_by_task

- Groups by `task_key`.
- Makes counts balanced across groups.

Config:

```yaml
data:
  online_dataset:
    sampling:
      mode: balance_by_task
      balance_by_task:
        enabled: true
        strategy: min   # min|max|fixed
        fixed_n: null
        seed: 42
        min_count: 8
```

## 2) per_output_type

- Sets targets for each output type.
- Optional overrides per `task_family`.

```yaml
data:
  online_dataset:
    sampling:
      mode: per_output_type
      per_output_type:
        targets:
          JSON: all
          YAML: 500
          TOML: 500
          XML: 200
          CSV: 200
        by_task_family:
          enabled: true
          families:
            u10bei: { targets: { TOML: all, JSON: all, YAML: all, XML: all, CSV: all } }
            daichira: { targets: { TOML: 100, JSON: 0, YAML: 0, XML: 0, CSV: 0 } }
```

## Selector implementation

- Entry point: `src/data/hf_select_subset.py`
- For the new config schema, run scripts pass `--dataset-source online|offline`.

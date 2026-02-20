# Eval and lint

Evaluation is performed with StructEval_T per output type.

- Default eval limit is configured in `configs/eval.yaml`:
  - `limit_per_output_type: 10`

The repo also includes format checks:

- Strict parsing / linting per format
- Checks that the output has no extra wrapper text before/after the structured content
- Removes invalid examples from training data
- Canonicalizes TOML to avoid mixed-style training targets

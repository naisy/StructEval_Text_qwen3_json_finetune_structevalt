# Reward design

## Component extraction
`src/rl/rewards.py:compute_reward_components()` extracts:
- `parse`: strict parse success
- `only`: output is format-only (no wrapper prose / fences)
- `extraneous`: wrapper-text detection
- `match` / `match_soft`: comparison against `reference_output` when provided

## TOML strictness policy
TOML is strict and small syntax mistakes are common (unterminated arrays, malformed inline tables, etc.).

**Important:** For TOML, the reward must not accept "best-effort" prefix parsing.
Prefix parsing can incorrectly treat an incomplete output as valid by truncating to a parseable prefix, which:
- avoids the hard parse-fail penalty
- still allows `match_soft` to award reward based on the truncated object

Policy implemented in `src/rl/rewards.py`:
- TOML uses **strict parse only** (no best-effort parser)
- `match` / `match_soft` are computed only when strict TOML parsing succeeds

This ensures invalid TOML receives a strong negative reward signal.

#!/usr/bin/env python
"""Quick sanity check for TOML reward strictness.

Usage:
  python scripts/dev_check_toml_reward.py

Expected:
  - invalid TOML should NOT receive best-effort parse credit
  - match/match_soft should be 0 when strict parse fails
"""

from __future__ import annotations

import os
import sys

# Allow running as a standalone script from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.rl.rewards import compute_reward_components

BAD = """star = \"HD 12345\"\nplanets = [\n  {\n    name = \"Kepler-1\"\n  }\n]"""

GOOD = """star = \"HD 12345\"\n\n[[planets]]\nname = \"Kepler-1\"\n"""


def main() -> None:
    comps_bad = compute_reward_components(
        completion=BAD,
        output_type="TOML",
        reference_output=GOOD,
    )
    comps_good = compute_reward_components(
        completion=GOOD,
        output_type="TOML",
        reference_output=GOOD,
    )

    print("[BAD]", comps_bad)
    print("[GOOD]", comps_good)


if __name__ == "__main__":
    main()

"""Test configuration.

This project is intentionally runnable without installation (we invoke modules via
`python -m src...`). Some environments run pytest with an import mode that does
not automatically add the repository root to `sys.path`, which would make
`import src` fail.

To keep tests robust and aligned with how the project is executed, we
explicitly prepend the repo root to `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

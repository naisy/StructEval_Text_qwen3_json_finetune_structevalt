from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class JudgeResult:
    score: float
    passed: bool
    details: Dict[str, Any]
    raw_text: Optional[str] = None

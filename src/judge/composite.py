from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import BaseJudge
from .types import JudgeResult


class CompositeJudge(BaseJudge):
    """
    Try multiple judges in priority order.

    This enables graceful fallback when an API key is missing (e.g., OPENAI_API_KEY),
    a provider is temporarily unavailable, or a model rejects parameters.

    Notes:
    - The first judge that returns a JudgeResult is used.
    - If all judges fail, the last exception is raised.
    """

    def __init__(self, judges: Sequence[Tuple[str, BaseJudge]]) -> None:
        self.judges: List[Tuple[str, BaseJudge]] = list(judges)
        if not self.judges:
            raise ValueError("CompositeJudge requires at least one judge.")

    def judge(self, *, task: Dict[str, Any], generation: str) -> JudgeResult:
        last_exc: Optional[BaseException] = None
        for name, j in self.judges:
            try:
                res = j.judge(task=task, generation=generation)
                # annotate for downstream analysis
                res.details.setdefault("judge_provider", name)
                return res
            except Exception as e:  # noqa: BLE001
                last_exc = e
                continue
        raise RuntimeError("All judge providers failed.") from last_exc

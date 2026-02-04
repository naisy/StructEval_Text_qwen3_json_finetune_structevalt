from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from .types import JudgeResult

class BaseJudge(ABC):
    @abstractmethod
    def judge(self, *, task: Dict[str, Any], generation: str) -> JudgeResult:
        raise NotImplementedError

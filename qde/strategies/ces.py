# qde/strategies/ces.py
from __future__ import annotations
from typing import Any
from ..core.base import SelectionResult
from base_strategy import BaseFilteringStrategy

class CES(BaseFilteringStrategy):
    name = "ces"

    def select(
        self,
        *,
        estimator,
        **kw: Any
    ) -> SelectionResult:
        pass

# qde/strategies/oes.py
from __future__ import annotations
from typing import Any, Literal
from ..core.base import SelectionResult
from base_strategy import BaseFilteringStrategy

distance_modes = Literal["euclidean", "cosine"]

class OES(BaseFilteringStrategy):
    name = "oes"

    def select(
        self,
        *,
        estimator,
        k_neighbors: int = 5,
        distance_mode: distance_modes = "euclidean",
        **kw: Any
    ) -> SelectionResult:
        pass

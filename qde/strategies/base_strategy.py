# qde/core/base_strategy.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder

from ..core.base import FilterStrategy, SelectionResult, DatasetViews


class BaseFilteringStrategy(FilterStrategy, ABC):
    name: str = "base"

    def fit(
        self,
        datasets: DatasetViews,
        *,
        encode_labels: bool = True,
        **kw: Any,) -> "BaseFilteringStrategy":
        self.train_X, self.train_y = datasets.get("train")
        self.synth_X, self.synth_y = datasets.get("sample") 
        self.test_X, self.test_y = datasets.get("test")

        self.train_y = LabelEncoder().fit(self.train_y) if encode_labels else self.train_y
        self.synth_y = LabelEncoder().fit(self.synth_y) if encode_labels else self.synth_y
        self.test_y = LabelEncoder().fit(self.test_y) if encode_labels else self.test_y

        return self

    @abstractmethod
    def select(
        self,
        **kw: Any,
    ) -> SelectionResult:
        ...

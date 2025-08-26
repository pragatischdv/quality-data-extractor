# qde/strategies/oes.py
from __future__ import annotations
import numpy as np
from typing import Any, Literal, List
from ..core.base import SelectionResult
from .base_strategy import BaseFilteringStrategy
from sklearn.neighbors import NearestNeighbors
from qde.metrics.scorers import accuracy_for_view, predictions_for_view

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
        yhat_synth = predictions_for_view(self.views, "sample", estimator=estimator)
        augmented_accuracy = accuracy_for_view(self.views, "train+synth", estimator=estimator)
        original_accuracy = accuracy_for_view(self.views, "train", estimator=estimator)

        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric=distance_mode)
        nbrs.fit(self.synth_X, self.synth_y)

        indices = set()

        for i in range(self.test_size):
            if yhat_synth[i] == self.test_y[i]:
                distances, nearest_indices = nbrs.kneighbors([self.test_X[i]])
                indices.update(nearest_indices.flatten().tolist())

        return SelectionResult(
            indices=np.asarray(list(indices)),
            meta={"strategy": self.name,
                "selected-samples": len(indices),
                "original-accuracy": original_accuracy, 
                "augmented-accuracy": augmented_accuracy}
        )


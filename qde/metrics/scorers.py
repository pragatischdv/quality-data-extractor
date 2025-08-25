from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from typing import Tuple
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from core.base import DatasetViews, View

def _fit_predict_for_view(
    views, view: str, *, estimator: BaseEstimator | ClassifierMixin
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = views.get(view)
    if y is None:
        raise ValueError("Labels are required to compute predictions/accuracy.")
    est = clone(estimator)
    est.fit(X, y)
    yhat = est.predict(X)
    return y, yhat

def accuracy_for_view(
    views, view: str, *, estimator: BaseEstimator | ClassifierMixin
) -> float:
    y, yhat = _fit_predict_for_view(views, view, estimator=estimator)
    return float(accuracy_score(y, yhat))

def predictions_for_view(
    views, view: str, *, estimator: BaseEstimator | ClassifierMixin
) -> np.ndarray:
    _, yhat = _fit_predict_for_view(views, view, estimator=estimator)
    return yhat

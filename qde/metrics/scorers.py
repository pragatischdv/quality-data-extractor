from __future__ import annotations
from typing import Callable, Optional
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from core.base import DatasetViews, View

def accuracy_for_view(
    views: DatasetViews,
    view: View = "train",
    *,
    estimator: BaseEstimator | ClassifierMixin = GaussianNB(),
) -> float:
    X, y = views.get(view)
    est = clone(estimator)  
    est.fit(X, y)
    yhat = est.predict(X)
    return float(accuracy_score(y, yhat))

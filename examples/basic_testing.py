from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from qde.qde import QDE
RANDOM_STATE = 42

def build_dataframes(n_samples: int = 1200, n_features: int = 20):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=2,
        n_repeated=0,
        n_classes=3,
        class_sep=1.2,
        flip_y=0.02,
        random_state=RANDOM_STATE,
    )

    # Split into (train + synth) vs test first
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # Then split tmp into train and synth
    X_train, X_synth, y_train, y_synth = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE
    )

    # Try pandas to exercise your iloc path
    cols = [f"f{i}" for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=cols)
    X_synth_df = pd.DataFrame(X_synth, columns=cols)
    X_test_df  = pd.DataFrame(X_test,  columns=cols)

    y_train_s = pd.Series(y_train, name="y")
    y_synth_s = pd.Series(y_synth, name="y")
    y_test_s  = pd.Series(y_test,  name="y")

    return (X_train_df, y_train_s), (X_synth_df, y_synth_s), (X_test_df, y_test_s)

def run_strategy(name: str, estimator):
    print(f"\n=== Strategy: {name} | Estimator: {estimator.__class__.__name__} ===")
    (X_tr, y_tr), (X_sy, y_sy), (X_te, y_te) = build_dataframes()

    qde = QDE(default_strategy=name)

    # Prepare (attaches datasets to the selected strategy; enables label encoding)
    qde.fit(
        train_X=X_tr, train_y=y_tr,
        syn_X=X_sy, syn_y=y_sy,
        test_X=X_te, test_y=y_te,
        strategy=name,
        estimator=estimator,         
        encode_labels=True,
    )

    # Extract (computes filtered-accuracy by default per your QDE.extract)
    result, X_sel, y_sel = qde.extract(
        estimator=estimator,
        compute_filtered_accuracy=True,
        k_neighbors=7,
        distance_mode="cosine",
    )

    # Report
    print(f"Selected indices (first 20): {result.indices[:20].tolist()}")
    print(f"Accepted count: {len(result.indices)} / synth_size={len(X_sy)}")
    if hasattr(result, "meta") and result.meta:
        for k, v in result.meta.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    # Try GaussianNB and LogisticRegression to exercise cloning, etc.
    run_strategy("ces", GaussianNB())
    run_strategy("oes", LogisticRegression(max_iter=200))

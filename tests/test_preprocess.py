"""
Unit tests for Module 7: Preprocessing Pipeline (core/preprocess.py)
"""

import pandas as pd
import numpy as np
from core.preprocess import build_preprocessor, prepare_train_test_data


def test_prepare_train_test_data_split_ordering():
    np.random.seed(42)
    df = pd.DataFrame({
        "num_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "cat_b": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    res = prepare_train_test_data(
        df=df,
        feature_cols=["num_a", "cat_b"],
        target_col="target",
        problem_type="binary_classification",
        test_size=0.3,
        num_scaler="standard",
        num_impute_strategy="mean",
        cat_encoder="onehot",
    )

    assert "X_train_proc" in res
    assert "X_test_proc" in res
    assert len(res["X_train_proc"]) == 7
    assert len(res["X_test_proc"]) == 3
    assert not np.isnan(res["X_train_proc"]).any()
    assert not np.isnan(res["X_test_proc"]).any()


def test_prepare_unsupervised_data():
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0],
        "x2": [10.0, 20.0, 30.0, 40.0],
    })
    res = prepare_train_test_data(
        df=df,
        feature_cols=["x1", "x2"],
        problem_type="unsupervised",
    )
    assert res["X_proc"].shape == (4, 2)

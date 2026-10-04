"""
Unit tests for Modules 4 & 5: Problem Detection, Leakage, and Imbalance (core/quality.py)
"""

import pandas as pd
import numpy as np
from core.quality import detect_problem_type, detect_data_leakage, detect_class_imbalance


def test_detect_problem_type():
    # 1. Binary classification
    df_binary = pd.DataFrame({"target": ["yes", "no", "yes", "no"], "x": [1, 2, 3, 4]})
    res = detect_problem_type(df_binary, "target")
    assert res["problem_type"] == "binary_classification"

    # 2. Multiclass classification
    df_multi = pd.DataFrame({"target": ["cat", "dog", "bird", "cat"], "x": [1, 2, 3, 4]})
    res = detect_problem_type(df_multi, "target")
    assert res["problem_type"] == "multiclass_classification"

    # 3. Regression
    np.random.seed(42)
    df_reg = pd.DataFrame({"price": np.random.uniform(100, 500, size=50)})
    res = detect_problem_type(df_reg, "price")
    assert res["problem_type"] == "regression"

    # 4. Unsupervised
    res = detect_problem_type(df_reg, None)
    assert res["problem_type"] == "unsupervised"


def test_detect_data_leakage():
    df = pd.DataFrame({
        "churn": [1, 0, 1, 0, 1, 0] * 10,
        "churn_status": [1, 0, 1, 0, 1, 0] * 10,  # Name similarity & perfect correlation
        "age": np.random.randint(20, 60, size=60),
    })
    warnings = detect_data_leakage(df, target_col="churn", feature_cols=["churn_status", "age"])
    assert len(warnings) > 0
    warn_cols = [w["column"] for w in warnings]
    assert "churn_status" in warn_cols


def test_detect_class_imbalance():
    # 90% Class 0, 10% Class 1 -> Imbalanced
    y_imbalanced = pd.Series([0] * 90 + [1] * 10)
    res = detect_class_imbalance(y_imbalanced)
    assert res["is_imbalanced"] is True
    assert res["minority_pct"] == 10.0

    # 50-50 -> Balanced
    y_balanced = pd.Series([0] * 50 + [1] * 50)
    res_b = detect_class_imbalance(y_balanced)
    assert res_b["is_imbalanced"] is False

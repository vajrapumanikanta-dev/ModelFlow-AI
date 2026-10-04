"""
Unit tests for Module 2: Profiler (core/profiler.py)
"""

import pandas as pd
import numpy as np
from core.profiler import profile_dataset, get_column_summary_df


def test_profile_dataset():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25.0, 30.0, np.nan, 35.0, 100.0],  # 100 is outlier
        "city": ["NY", "LA", "NY", "SF", "LA"],
        "constant_col": [1, 1, 1, 1, 1],
    })

    profile = profile_dataset(df)
    summary = profile["summary"]

    assert summary["total_rows"] == 5
    assert summary["total_cols"] == 4
    assert summary["total_missing_cells"] == 1
    assert summary["duplicate_rows"] == 0

    cols_meta = {c["name"]: c for c in profile["columns"]}
    assert cols_meta["id"]["is_id_candidate"] is True
    assert cols_meta["constant_col"]["is_constant"] is True
    assert cols_meta["age"]["type_category"] == "numeric"
    assert cols_meta["city"]["type_category"] == "categorical"
    assert cols_meta["city"]["top_value"] in ["NY", "LA"]


def test_column_summary_df():
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": ["a", "b", "c"],
    })
    profile = profile_dataset(df)
    sum_df = get_column_summary_df(profile)
    assert len(sum_df) == 2
    assert "Column" in sum_df.columns
    assert "Missing" in sum_df.columns


def test_health_score_and_insights():
    from core.profiler import calculate_health_score, generate_dataset_insights
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0],
        "cat": ["A", "B", "A", "B"],
    })
    health = calculate_health_score(df)
    assert health["score"] == 100
    assert health["grade"] == "Excellent"

    insights = generate_dataset_insights(df)
    assert len(insights) >= 2


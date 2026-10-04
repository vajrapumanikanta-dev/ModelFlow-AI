"""
Unit tests for Module 3: Cleaner (core/cleaner.py)
"""

import pandas as pd
import numpy as np
from core.cleaner import DataCleaner


def test_cleaner_missing_values_and_undo():
    df = pd.DataFrame({
        "num": [10.0, np.nan, 30.0],
        "cat": ["a", np.nan, "a"],
    })
    cleaner = DataCleaner(df)

    # 1. Impute mean
    res = cleaner.handle_missing(["num"], strategy="mean")
    assert res["columns"] == ["num"]
    assert cleaner.current_df["num"].isna().sum() == 0
    assert cleaner.current_df["num"].iloc[1] == 20.0

    # 2. Undo
    assert cleaner.undo() is True
    assert cleaner.current_df["num"].isna().sum() == 1


def test_cleaner_duplicates():
    df = pd.DataFrame({
        "a": [1, 2, 2, 3],
        "b": ["x", "y", "y", "z"],
    })
    cleaner = DataCleaner(df)
    dups = cleaner.handle_duplicates()
    assert dups == 1
    assert len(cleaner.current_df) == 3


def test_cleaner_outlier_capping():
    df = pd.DataFrame({
        "val": [10, 12, 11, 13, 12, 11, 100],  # 100 is outlier
    })
    cleaner = DataCleaner(df)
    count = cleaner.handle_outliers("val", method="iqr", threshold=1.5, action="cap")
    assert count >= 1
    assert cleaner.current_df["val"].max() < 100


def test_cleaner_log_and_replay():
    df = pd.DataFrame({
        "a": [1, 1, 2, 3],
        "b": [10.0, 10.0, 30.0, 40.0],
        "id": [101, 101, 103, 104],
    })
    cleaner = DataCleaner(df)
    cleaner.handle_duplicates()
    cleaner.handle_missing(["b"], strategy="median")
    cleaner.drop_columns(["id"])

    log = cleaner.get_cleaning_log()
    assert len(log) == 3

    # Replay on fresh dataframe
    replayed_df = DataCleaner.replay(df, log)
    assert replayed_df.shape == cleaner.current_df.shape
    assert "id" not in replayed_df.columns
    assert replayed_df["b"].isna().sum() == 0

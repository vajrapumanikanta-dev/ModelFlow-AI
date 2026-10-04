"""
Unit tests for Module 1: Ingestion (core/loader.py)
"""

import os
import io
import pytest
import pandas as pd
from core.loader import detect_file_type, load_dataset


def test_detect_file_type_extensions():
    assert detect_file_type("dataset.csv") == "csv"
    assert detect_file_type("data.xlsx") == "excel"
    assert detect_file_type("records.json") == "json"


def test_detect_file_type_buffer():
    buf_json = io.BytesIO(b'{"a": [1, 2, 3]}')
    assert detect_file_type(buf_json) == "json"

    buf_csv = io.BytesIO(b"a,b,c\n1,2,3\n4,5,6")
    assert detect_file_type(buf_csv) == "csv"


def test_load_csv_dataset():
    csv_content = "id,feature_a,feature_b,target\n1,10.5,red,1\n2,20.2,blue,0\n3,15.1,green,1"
    buf = io.StringIO(csv_content)
    df, meta = load_dataset(buf, file_type="csv")

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 4)
    assert meta["rows"] == 3
    assert meta["cols"] == 4
    assert "target" in df.columns


def test_load_csv_encoding_fallback():
    # Latin-1 encoded CSV string with special characters
    csv_bytes = "name,val\nRen\xe9,42\n".encode("latin1")
    buf = io.BytesIO(csv_bytes)
    df, meta = load_dataset(buf, file_type="csv")

    assert len(df) == 1
    assert df.iloc[0]["name"] == "René"


def test_load_json_dataset():
    json_content = '[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]'
    buf = io.StringIO(json_content)
    df, meta = load_dataset(buf, file_type="json")

    assert len(df) == 2
    assert "a" in df.columns


def test_load_dataset_size_limit():
    buf = io.StringIO("a,b\n1,2")
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        load_dataset(buf, file_type="csv", max_size_mb=0.0000001)


def test_load_empty_dataset_raises():
    buf = io.StringIO("")
    with pytest.raises(ValueError):
        load_dataset(buf, file_type="csv")

"""
Module 1: Ingestion (core/loader.py)
Handles dataset ingestion for CSV, Excel, and JSON files with automatic
file-type detection, encoding fallbacks, delimiter detection, and size limits.
"""

import io
import os
import json
from typing import Tuple, Dict, Any, Union
import pandas as pd


def detect_file_type(filename_or_buffer: Union[str, io.BytesIO, io.StringIO, Any]) -> str:
    """
    Detects file type based on file name or header signature.
    Returns: 'csv', 'excel', or 'json'.
    """
    filename = ""
    if isinstance(filename_or_buffer, str):
        filename = filename_or_buffer
    elif hasattr(filename_or_buffer, "name"):
        filename = getattr(filename_or_buffer, "name", "")

    ext = os.path.splitext(filename)[1].lower() if filename else ""

    if ext in [".csv", ".tsv", ".txt"]:
        return "csv"
    elif ext in [".xlsx", ".xls", ".xlsm"]:
        return "excel"
    elif ext in [".json", ".jsonl"]:
        return "json"

    # Inspect buffer header if extension is missing or unknown
    if hasattr(filename_or_buffer, "seek") and hasattr(filename_or_buffer, "read"):
        try:
            pos = filename_or_buffer.tell()
            filename_or_buffer.seek(0)
            sample = filename_or_buffer.read(1024)
            filename_or_buffer.seek(pos)

            if isinstance(sample, bytes):
                # Check for zip / excel signature
                if sample.startswith(b"PK\x03\x04") or sample.startswith(b"\xd0\xcf\x11\xe0"):
                    return "excel"
                try:
                    text_sample = sample.decode("utf-8", errors="ignore").strip()
                except Exception:
                    text_sample = ""
            else:
                text_sample = str(sample).strip()

            if text_sample.startswith("{") or text_sample.startswith("["):
                return "json"
            if "," in text_sample or ";" in text_sample or "\t" in text_sample:
                return "csv"
        except Exception:
            pass

    return "csv"  # Default fallback


def load_dataset(
    file_source: Union[str, io.BytesIO, io.StringIO, Any],
    file_type: str = None,
    max_size_mb: float = 100.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads dataset from a path, buffer, or Streamlit UploadedFile.
    
    Args:
        file_source: File path or file-like buffer.
        file_type: Optional explicitly specified type ('csv', 'excel', 'json').
        max_size_mb: Maximum allowed file size in Megabytes.
        
    Returns:
        Tuple of (pd.DataFrame, metadata_dict).
        
    Raises:
        ValueError: If file is empty, corrupted, or exceeds size limit.
    """
    filename = getattr(file_source, "name", "uploaded_dataset")
    size_bytes = 0

    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise FileNotFoundError(f"Dataset file not found: {file_source}")
        size_bytes = os.path.getsize(file_source)
        filename = os.path.basename(file_source)
    elif hasattr(file_source, "size"):
        size_bytes = getattr(file_source, "size", 0)
    elif hasattr(file_source, "getbuffer"):
        size_bytes = len(file_source.getbuffer())
    elif hasattr(file_source, "getvalue"):
        val = file_source.getvalue()
        size_bytes = len(val.encode("utf-8")) if isinstance(val, str) else len(val)

    if max_size_mb > 0 and size_bytes > (max_size_mb * 1024 * 1024):
        raise ValueError(
            f"File size ({size_bytes / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {max_size_mb:.2f} MB."
        )

    if not file_type:
        file_type = detect_file_type(file_source)

    df = None
    encoding_used = "utf-8"

    if file_type == "csv":
        encodings = ["utf-8", "utf-8-sig", "latin1", "iso-8859-1", "cp1252", "utf-16"]
        delimiters = [None, ",", ";", "\t", "|"]
        last_error = None

        for enc in encodings:
            for delim in delimiters:
                try:
                    if hasattr(file_source, "seek"):
                        file_source.seek(0)
                    
                    if delim is None:
                        # Auto-detect separator
                        df = pd.read_csv(file_source, encoding=enc, sep=None, engine="python")
                    else:
                        df = pd.read_csv(file_source, encoding=enc, sep=delim)
                    
                    if df is not None and df.shape[1] > 0 and len(df) >= 0:
                        encoding_used = enc
                        break
                except Exception as e:
                    last_error = e
                    continue
            if df is not None and df.shape[1] > 0:
                break

        if df is None:
            raise ValueError(f"Failed to read CSV dataset. Error: {last_error}")

    elif file_type == "excel":
        try:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            df = pd.read_excel(file_source, engine="openpyxl")
            encoding_used = "binary"
        except Exception as e:
            try:
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                df = pd.read_excel(file_source)
                encoding_used = "binary"
            except Exception as e2:
                raise ValueError(f"Failed to read Excel dataset: {e2}")

    elif file_type == "json":
        try:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            df = pd.read_json(file_source)
            encoding_used = "utf-8"
        except Exception:
            try:
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                df = pd.read_json(file_source, lines=True)
                encoding_used = "utf-8"
            except Exception as e:
                raise ValueError(f"Failed to parse JSON dataset: {e}")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    if df is None or df.empty:
        raise ValueError("Uploaded dataset is empty (0 rows or 0 columns).")

    # Clean column names (strip leading/trailing whitespace, keep non-empty)
    df.columns = [str(c).strip() if str(c).strip() != "" else f"Unnamed_{i}" for i, c in enumerate(df.columns)]

    metadata = {
        "filename": filename,
        "file_type": file_type,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3) if size_bytes else 0.0,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "encoding": encoding_used,
        "column_names": list(df.columns),
    }

    return df, metadata

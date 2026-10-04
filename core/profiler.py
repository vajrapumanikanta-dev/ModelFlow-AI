"""
Module 2: Profiling (core/profiler.py)
Generates comprehensive dataset and per-column statistical profiling.
"""

import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def format_memory_size(bytes_val: int) -> str:
    """Formats bytes into human-readable string (KB, MB, GB)."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def detect_type_category(series: pd.Series) -> str:
    """Classifies a pandas Series into numeric, categorical, boolean, or datetime."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif pd.api.types.is_numeric_dtype(series):
        return "numeric"
    else:
        # Check if text column can be parsed as datetime
        if series.dropna().count() > 0 and series.dtype == "object":
            sample = series.dropna().iloc[:30].astype(str)
            has_date_pattern = sample.str.contains(r"\d{4}[-/]\d{1,2}|[-/]\d{1,2}[-/]\d{2,4}", regex=True).any()
            if has_date_pattern:
                try:
                    pd.to_datetime(sample, errors="raise", format="mixed")
                    return "datetime"
                except Exception:
                    pass
        return "categorical"


def is_potential_id_column(col_name: str, unique_count: int, total_rows: int, type_cat: str) -> bool:
    """Detects if a column is likely an ID or primary key index."""
    name_lower = str(col_name).lower().strip()
    id_patterns = [r"^id$", r"_id$", r"id_", r"guid", r"uuid", r"key", r"index", r"customer_?id", r"passenger_?id"]
    name_matches = any(re.search(pat, name_lower) for pat in id_patterns)
    
    high_cardinality = (total_rows > 10 and (unique_count / total_rows) >= 0.95)
    
    if name_matches:
        return True
    if high_cardinality and type_cat in ["categorical", "numeric"] and total_rows > 30:
        return True
    return False


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes DataFrame and produces detailed dataset-level and column-level profiling.
    """
    total_rows, total_cols = df.shape
    memory_bytes = int(df.memory_usage(deep=True).sum())
    
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0, 2)
    
    total_cells = total_rows * total_cols
    total_missing_cells = int(df.isna().sum().sum())
    total_missing_pct = round((total_missing_cells / total_cells * 100) if total_cells > 0 else 0.0, 2)

    columns_profile: List[Dict[str, Any]] = []
    type_counts = {"numeric": 0, "categorical": 0, "boolean": 0, "datetime": 0}

    for col in df.columns:
        series = df[col]
        type_cat = detect_type_category(series)
        type_counts[type_cat] = type_counts.get(type_cat, 0) + 1

        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / total_rows * 100) if total_rows > 0 else 0.0, 2)
        unique_count = int(series.nunique(dropna=True))
        unique_pct = round((unique_count / total_rows * 100) if total_rows > 0 else 0.0, 2)
        is_const = (unique_count <= 1)
        is_id = is_potential_id_column(col, unique_count, total_rows, type_cat)

        col_meta: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "type_category": type_cat,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
            "is_constant": is_const,
            "is_id_candidate": is_id,
        }

        if type_cat == "numeric":
            clean_s = series.dropna()
            if not clean_s.empty:
                q25 = float(clean_s.quantile(0.25))
                q75 = float(clean_s.quantile(0.75))
                iqr = q75 - q25
                iqr_outliers = int(((clean_s < (q25 - 1.5 * iqr)) | (clean_s > (q75 + 1.5 * iqr))).sum())

                col_meta.update({
                    "min": float(clean_s.min()),
                    "max": float(clean_s.max()),
                    "mean": round(float(clean_s.mean()), 4),
                    "median": round(float(clean_s.median()), 4),
                    "std": round(float(clean_s.std()), 4) if len(clean_s) > 1 else 0.0,
                    "q25": round(q25, 4),
                    "q75": round(q75, 4),
                    "skewness": round(float(clean_s.skew()), 4) if len(clean_s) > 2 else 0.0,
                    "zeros_count": int((clean_s == 0).sum()),
                    "negatives_count": int((clean_s < 0).sum()),
                    "outliers_iqr_count": iqr_outliers,
                })
            else:
                col_meta.update({
                    "min": None, "max": None, "mean": None, "median": None,
                    "std": None, "q25": None, "q75": None, "skewness": None,
                    "zeros_count": 0, "negatives_count": 0, "outliers_iqr_count": 0,
                })
        else:
            clean_s = series.dropna()
            if not clean_s.empty:
                val_counts = clean_s.value_counts()
                top_val = str(val_counts.index[0])
                top_freq = int(val_counts.iloc[0])
                sample_vals = [str(x) for x in clean_s.unique()[:5]]
                col_meta.update({
                    "top_value": top_val,
                    "top_freq": top_freq,
                    "sample_values": sample_vals,
                })
            else:
                col_meta.update({
                    "top_value": None,
                    "top_freq": 0,
                    "sample_values": [],
                })

        columns_profile.append(col_meta)

    return {
        "summary": {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "memory_usage_bytes": memory_bytes,
            "memory_usage_str": format_memory_size(memory_bytes),
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": duplicate_pct,
            "total_missing_cells": total_missing_cells,
            "total_missing_pct": total_missing_pct,
            "type_counts": type_counts,
        },
        "columns": columns_profile,
    }


def get_column_summary_df(profile: Dict[str, Any]) -> pd.DataFrame:
    """Converts the columns profile list into a clean tabular DataFrame for UI display."""
    rows = []
    for c in profile.get("columns", []):
        row = {
            "Column": c["name"],
            "Data Type": c["dtype"],
            "Category": c["type_category"].capitalize(),
            "Missing": f"{c['missing_count']} ({c['missing_pct']}%)",
            "Unique": f"{c['unique_count']} ({c['unique_pct']}%)",
            "Constant": "Yes" if c["is_constant"] else "No",
            "ID Candidate": "Yes" if c["is_id_candidate"] else "No",
        }
        if c["type_category"] == "numeric" and c.get("mean") is not None:
            row["Min / Max"] = f"{c.get('min')} / {c.get('max')}"
            row["Mean ± Std"] = f"{c.get('mean')} ± {c.get('std')}"
            row["Outliers (IQR)"] = str(c.get("outliers_iqr_count", 0))
        elif c.get("top_value") is not None:
            row["Min / Max"] = "-"
            row["Mean ± Std"] = f"Top: {c.get('top_value')} ({c.get('top_freq')})"
            row["Outliers (IQR)"] = "-"
        else:
            row["Min / Max"] = "-"
            row["Mean ± Std"] = "-"
            row["Outliers (IQR)"] = "-"

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_health_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive 0-100% Data Quality & Health Score based on
    missing values, duplicate rows, constant columns, and outlier prevalence.
    """
    if df.empty:
        return {"score": 0, "grade": "Empty", "deductions": ["Dataset is empty."]}

    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    deductions = []
    score = 100.0

    # 1. Missing data penalty
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    if missing_pct > 0:
        missing_penalty = min(30.0, missing_pct * 1.5)
        score -= missing_penalty
        deductions.append(f"Missing data ({missing_pct:.1f}% of cells) — deducted {missing_penalty:.1f} pts")

    # 2. Duplicate rows penalty
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / total_rows) * 100 if total_rows > 0 else 0
    if dup_pct > 0:
        dup_penalty = min(20.0, dup_pct * 2.0)
        score -= dup_penalty
        deductions.append(f"Duplicate rows ({dup_pct:.1f}%) — deducted {dup_penalty:.1f} pts")

    # 3. Constant columns penalty
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        const_penalty = min(15.0, len(constant_cols) * 5.0)
        score -= const_penalty
        deductions.append(f"{len(constant_cols)} constant column(s) — deducted {const_penalty:.1f} pts")

    # 4. Outlier penalties in numeric features
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    outlier_cols_count = 0
    for col in numeric_cols:
        clean = df[col].dropna()
        if len(clean) > 10:
            q25, q75 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                n_out = ((clean < (q25 - 1.5 * iqr)) | (clean > (q75 + 1.5 * iqr))).sum()
                if (n_out / len(clean)) > 0.05:
                    outlier_cols_count += 1
    if outlier_cols_count > 0:
        outlier_penalty = min(15.0, outlier_cols_count * 3.0)
        score -= outlier_penalty
        deductions.append(f"{outlier_cols_count} numeric column(s) with high outlier density — deducted {outlier_penalty:.1f} pts")

    final_score = max(0, min(100, int(round(score))))
    if final_score >= 90:
        grade = "Excellent"
        color = "#10b981"
    elif final_score >= 75:
        grade = "Good"
        color = "#3b82f6"
    elif final_score >= 60:
        grade = "Fair"
        color = "#f59e0b"
    else:
        grade = "Needs Attention"
        color = "#ef4444"

    return {
        "score": final_score,
        "grade": grade,
        "color": color,
        "deductions": deductions if deductions else ["No significant data quality issues detected."],
    }


def generate_dataset_insights(df: pd.DataFrame, target_col: Optional[str] = None) -> List[str]:
    """
    Extracts factual, statistical key insights and recommendations from the dataset.
    """
    insights = []
    total_rows, total_cols = df.shape
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    insights.append(
        f"Dataset contains **{total_rows:,}** rows and **{total_cols}** columns ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)."
    )

    # Missing insights
    missing_by_col = df.isna().sum()
    missing_cols = missing_by_col[missing_by_col > 0]
    if not missing_cols.empty:
        top_missing = missing_cols.sort_values(ascending=False).head(3)
        missing_desc = ", ".join([f"`{col}` ({cnt} missing, {cnt/total_rows*100:.1f}%)" for col, cnt in top_missing.items()])
        insights.append(f"Missing values detected in {len(missing_cols)} column(s): {missing_desc}.")
    else:
        insights.append("Data integrity: 100% complete dataset with zero missing values.")

    # Duplicate insights
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        insights.append(f"Found **{dup_count:,}** duplicate rows ({dup_count/total_rows*100:.1f}%). Cleaning recommended before training.")

    # Correlation insights
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            corr_unstacked = corr_matrix.unstack()
            diff_idx = [i for i in corr_unstacked.index if i[0] != i[1]]
            if diff_idx:
                corr_pairs = corr_unstacked.loc[diff_idx].dropna().sort_values(ascending=False)
                if not corr_pairs.empty:
                    (col1, col2), val = corr_pairs.index[0], corr_pairs.iloc[0]
                    if val >= 0.70:
                        insights.append(f"Strong collinearity detected between `{col1}` and `{col2}` (Pearson |r| = {val:.2f}).")
        except Exception:
            pass

    # Target insights
    if target_col and target_col in df.columns:
        target_s = df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(target_s) and target_s.nunique() > 10:
            insights.append(f"Target `{target_col}` is continuous with mean {target_s.mean():.2f} (median {target_s.median():.2f}, range [{target_s.min():.2f}, {target_s.max():.2f}]).")
        else:
            class_counts = target_s.value_counts()
            top_class = str(class_counts.index[0])
            top_pct = (class_counts.iloc[0] / len(target_s)) * 100
            insights.append(f"Target `{target_col}` has {len(class_counts)} classes. Majority class `{top_class}` accounts for {top_pct:.1f}% of samples.")

    return insights


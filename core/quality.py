"""
Module 4 & 5: Problem Detection, Leakage Detection, and Class Imbalance Checks (core/quality.py)
Implements problem type auto-detection, heuristic data leakage detection, and class imbalance checks.
"""

import re
import difflib
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def detect_problem_type(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Detects ML problem type based on target column characteristics.
    Returns: 'binary_classification', 'multiclass_classification', 'regression', or 'unsupervised'.
    """
    if not target_col or target_col not in df.columns:
        return {
            "problem_type": "unsupervised",
            "task_type": "unsupervised",
            "target_col": None,
            "num_classes": 0,
            "classes": [],
            "reason": "No target column selected. Unsupervised learning (Clustering / PCA) enabled.",
        }

    y = df[target_col].dropna()
    total_samples = len(y)
    if total_samples == 0:
        return {
            "problem_type": "unsupervised",
            "task_type": "unsupervised",
            "target_col": target_col,
            "num_classes": 0,
            "classes": [],
            "reason": "Target column has 0 non-null values.",
        }

    unique_vals = y.unique()
    num_unique = len(unique_vals)
    is_numeric = pd.api.types.is_numeric_dtype(y)
    is_bool = pd.api.types.is_bool_dtype(y)

    # Boolean or text column -> classification
    if is_bool:
        return {
            "problem_type": "binary_classification",
            "task_type": "classification",
            "target_col": target_col,
            "num_classes": 2,
            "classes": [str(c) for c in sorted(unique_vals)],
            "reason": f"Target '{target_col}' is boolean with 2 classes.",
        }

    if not is_numeric:
        if num_unique == 2:
            return {
                "problem_type": "binary_classification",
                "task_type": "classification",
                "target_col": target_col,
                "num_classes": 2,
                "classes": [str(c) for c in sorted(unique_vals, key=str)],
                "reason": f"Target '{target_col}' is categorical/text with 2 distinct classes.",
            }
        else:
            return {
                "problem_type": "multiclass_classification",
                "task_type": "classification",
                "target_col": target_col,
                "num_classes": num_unique,
                "classes": [str(c) for c in sorted(unique_vals, key=str)],
                "reason": f"Target '{target_col}' is categorical/text with {num_unique} distinct classes.",
            }

    # Numeric target
    if num_unique == 2:
        return {
            "problem_type": "binary_classification",
            "task_type": "classification",
            "target_col": target_col,
            "num_classes": 2,
            "classes": [str(c) for c in sorted(unique_vals)],
            "reason": f"Target '{target_col}' is numeric with only 2 distinct values (binary).",
        }

    # Low unique cardinality numeric
    if num_unique <= 10 or (total_samples > 30 and (num_unique / total_samples) < 0.05 and num_unique <= 20):
        return {
            "problem_type": "multiclass_classification",
            "task_type": "classification",
            "target_col": target_col,
            "num_classes": num_unique,
            "classes": [str(c) for c in sorted(unique_vals)],
            "reason": f"Target '{target_col}' is integer/numeric with few unique values ({num_unique} classes).",
        }

    return {
        "problem_type": "regression",
        "task_type": "regression",
        "target_col": target_col,
        "num_classes": 0,
        "classes": [],
        "reason": f"Target '{target_col}' is continuous numeric with {num_unique} distinct values.",
    }


def detect_data_leakage(
    df: pd.DataFrame, target_col: str, feature_cols: List[str]
) -> List[Dict[str, Any]]:
    """
    Detects potential data leakage heuristics:
    1. Feature name similarity to target name.
    2. Suspicious keywords in feature names (outcome, post_, future_, target, etc.).
    3. Extreme correlation with target (> 0.95 or < -0.95).
    4. Single-feature near-perfect prediction (> 0.98 score).
    """
    if not target_col or target_col not in df.columns or not feature_cols:
        return []

    warnings: List[Dict[str, Any]] = []
    target_clean_name = re.sub(r"[_\W]+", "", target_col.lower())
    suspicious_patterns = [
        r"final", r"outcome", r"result", r"post_", r"future_", r"target",
        r"label", r"ground_?truth", r"output", r"score_?after", r"y_true"
    ]

    # Prepare data for numerical / correlation checks
    y_raw = df[target_col]
    y_numeric = None
    if pd.api.types.is_numeric_dtype(y_raw):
        y_numeric = y_raw
    elif pd.api.types.is_bool_dtype(y_raw):
        y_numeric = y_raw.astype(float)
    else:
        # Convert categories to factorized codes for correlation
        y_numeric = pd.Series(pd.factorize(y_raw)[0], index=df.index, dtype=float)

    for col in feature_cols:
        if col == target_col or col not in df.columns:
            continue

        col_lower = col.lower()
        col_clean_name = re.sub(r"[_\W]+", "", col_lower)

        # 1. Name Similarity
        similarity = difflib.SequenceMatcher(None, target_clean_name, col_clean_name).ratio()
        is_sub = (target_clean_name in col_clean_name or col_clean_name in target_clean_name) and len(col_clean_name) > 3
        if similarity >= 0.80 or is_sub:
            warnings.append({
                "column": col,
                "risk_level": "High",
                "reason": f"Column name '{col}' is very similar to target '{target_col}' (similarity: {similarity:.2f}).",
                "recommendation": "Exclude this feature if it represents target metadata or future outcome.",
            })
            continue

        # 2. Suspicious keyword patterns
        if any(re.search(pat, col_lower) for pat in suspicious_patterns):
            warnings.append({
                "column": col,
                "risk_level": "Medium",
                "reason": f"Column name '{col}' contains a post-event keyword suggesting future leakage.",
                "recommendation": "Verify whether this feature is available prior to the prediction event.",
            })

        # 3. Correlation check
        feat_series = df[col]
        if pd.api.types.is_numeric_dtype(feat_series):
            valid_mask = feat_series.notna() & y_numeric.notna()
            if valid_mask.sum() > 10:
                corr = float(feat_series[valid_mask].corr(y_numeric[valid_mask]))
                if not np.isnan(corr) and abs(corr) >= 0.95:
                    warnings.append({
                        "column": col,
                        "risk_level": "High",
                        "reason": f"Extreme correlation (|r| = {abs(corr):.3f}) with target '{target_col}'.",
                        "recommendation": "High risk of target leakage or duplicate column. Consider excluding.",
                    })
                    continue

        # 4. Single-feature predictive power
        try:
            valid_data = df[[col, target_col]].dropna()
            if len(valid_data) >= 30:
                x_single = valid_data[[col]]
                y_single = valid_data[target_col]

                # Convert categorical single feature
                if not pd.api.types.is_numeric_dtype(x_single[col]):
                    x_single = pd.get_dummies(x_single, drop_first=True)

                if pd.api.types.is_numeric_dtype(y_single) and y_single.nunique() > 10:
                    dt = DecisionTreeRegressor(max_depth=2, random_state=42)
                    dt.fit(x_single, y_single)
                    r2_score = dt.score(x_single, y_single)
                    if r2_score > 0.98:
                        warnings.append({
                            "column": col,
                            "risk_level": "High",
                            "reason": f"Single feature '{col}' explains {r2_score*100:.1f}% of target variance.",
                            "recommendation": "Feature may be a direct proxy for the target.",
                        })
                else:
                    dt = DecisionTreeClassifier(max_depth=2, random_state=42)
                    dt.fit(x_single, y_single)
                    acc = dt.score(x_single, y_single)
                    if acc > 0.99 and y_single.nunique() > 1:
                        warnings.append({
                            "column": col,
                            "risk_level": "High",
                            "reason": f"Single feature '{col}' alone achieves {acc*100:.1f}% accuracy on the target.",
                            "recommendation": "Feature is nearly identical to target outcome. Consider excluding.",
                        })
        except Exception:
            pass

    return warnings


def detect_class_imbalance(y: pd.Series) -> Dict[str, Any]:
    """
    Detects class imbalance in target column.
    Flags imbalance if minority class is below 20% or imbalance ratio exceeds 4:1.
    """
    y_clean = y.dropna()
    total = len(y_clean)
    if total == 0:
        return {"is_imbalanced": False, "reason": "No target data available."}

    counts = y_clean.value_counts()
    num_classes = len(counts)
    if num_classes <= 1:
        return {"is_imbalanced": False, "reason": "Target has only 1 unique class."}

    minority_class = counts.index[-1]
    minority_count = int(counts.iloc[-1])
    minority_pct = round((minority_count / total) * 100, 2)
    majority_count = int(counts.iloc[0])
    imbalance_ratio = round(majority_count / max(minority_count, 1), 2)

    is_imbalanced = (minority_pct < 20.0) or (imbalance_ratio >= 4.0)

    class_distribution = [
        {"class": str(cls), "count": int(cnt), "percentage": round((cnt / total) * 100, 2)}
        for cls, cnt in counts.items()
    ]

    return {
        "is_imbalanced": is_imbalanced,
        "minority_class": str(minority_class),
        "minority_count": minority_count,
        "minority_pct": minority_pct,
        "imbalance_ratio": imbalance_ratio,
        "class_distribution": class_distribution,
        "recommended_strategies": ["class_weights", "smote", "random_oversampling", "random_undersampling"] if is_imbalanced else [],
        "reason": f"Minority class '{minority_class}' makes up {minority_pct}% of the data (ratio {imbalance_ratio}:1)."
        if is_imbalanced
        else "Class distribution is reasonably balanced.",
    }

"""
Unit tests for Module 11: Explainability (core/explain.py)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from core.explain import (
    extract_feature_importance,
    compute_shap_explanations,
    compute_lime_explanation,
    generate_plain_english_explanation,
)


def test_plain_english_explanation():
    imp_df = pd.DataFrame({
        "feature": ["MonthlyCharges", "Contract", "Tenure"],
        "importance": [0.5, 0.3, 0.2],
        "abs_importance": [0.5, 0.3, 0.2],
        "importance_pct": [50.0, 30.0, 20.0],
    })

    res = generate_plain_english_explanation(imp_df, prediction="Churn", probabilities={"Churn": 0.85, "No Churn": 0.15})
    assert "headline" in res
    assert "Confidence: 85.0%" in res["headline"]
    assert len(res["top_drivers"]) == 3
    assert res["top_drivers"][0]["impact_level"] == "High Impact"


def test_feature_importance_extraction():
    X = np.random.randn(50, 3)
    y = np.random.choice([0, 1], size=50)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    df_imp = extract_feature_importance(rf, ["f1", "f2", "f3"])
    assert len(df_imp) == 3
    assert "importance_pct" in df_imp.columns
    assert df_imp["abs_importance"].sum() > 0

"""
Module 11: Explainability (core/explain.py)
Provides model interpretability via Feature Importance, SHAP, and LIME.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Extracts feature importance from model (tree feature_importances_, linear coef_, or permutation).
    """
    importances = None

    # 1. Tree-based feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # 2. Linear models coefficients
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
    # 3. Permutation importance fallback
    elif X_test is not None and y_test is not None and len(X_test) > 0:
        try:
            perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            importances = perm.importances_mean
        except Exception:
            importances = None

    if importances is None or len(importances) == 0:
        # Uniform fallback
        importances = np.zeros(len(feature_names))

    # Match length
    if len(importances) != len(feature_names):
        # Truncate or pad
        n = min(len(importances), len(feature_names))
        feature_names = feature_names[:n]
        importances = importances[:n]

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "abs_importance": np.abs(importances),
    })

    # Normalize to percentage sum=100% if sum > 0
    total = df_imp["abs_importance"].sum()
    if total > 0:
        df_imp["importance_pct"] = (df_imp["abs_importance"] / total) * 100
    else:
        df_imp["importance_pct"] = 0.0

    df_imp = df_imp.sort_values(by="abs_importance", ascending=False).reset_index(drop=True)
    return df_imp


def compute_shap_explanations(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    sample_size: int = 60,
) -> Dict[str, Any]:
    """
    Computes SHAP values and global summary for model explanations.
    """
    if not HAS_SHAP:
        return {"available": False, "error": "SHAP library is not installed."}

    try:
        # Subsample for speed and responsiveness
        bg_size = min(len(X_train), sample_size)
        fg_size = min(len(X_test), sample_size)

        np.random.seed(42)
        bg_indices = np.random.choice(len(X_train), bg_size, replace=False)
        fg_indices = np.random.choice(len(X_test), fg_size, replace=False)

        bg_sample = X_train[bg_indices]
        fg_sample = X_test[fg_indices]

        explainer = None
        model_name = type(model).__name__.lower()

        if "tree" in model_name or "forest" in model_name or "gradient" in model_name:
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = shap.Explainer(model, bg_sample)
        elif "linear" in model_name or "logistic" in model_name or "ridge" in model_name or "lasso" in model_name:
            try:
                explainer = shap.LinearExplainer(model, bg_sample)
            except Exception:
                explainer = shap.Explainer(model, bg_sample)
        else:
            # General model explainer
            try:
                predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
                explainer = shap.KernelExplainer(predict_fn, bg_sample)
            except Exception:
                explainer = shap.Explainer(model, bg_sample)

        shap_values_raw = explainer(fg_sample) if hasattr(explainer, "__call__") else explainer.shap_values(fg_sample)

        # Handle different SHAP output formats (Explanation object vs ndarray vs list)
        if hasattr(shap_values_raw, "values"):
            shap_vals = shap_values_raw.values
            base_vals = shap_values_raw.base_values
        elif isinstance(shap_values_raw, list):
            # Binary/Multiclass list of arrays: pick positive class or average
            shap_vals = shap_values_raw[1] if len(shap_values_raw) == 2 else shap_values_raw[0]
            base_vals = getattr(explainer, "expected_value", [0.0])[1] if isinstance(getattr(explainer, "expected_value", 0.0), list) else getattr(explainer, "expected_value", 0.0)
        else:
            shap_vals = shap_values_raw
            base_vals = getattr(explainer, "expected_value", 0.0)

        # Calculate mean absolute SHAP value per feature
        if shap_vals.ndim == 3:
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=(0, 2))
        else:
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        shap_summary_df = pd.DataFrame({
            "feature": feature_names[: len(mean_abs_shap)],
            "mean_abs_shap": mean_abs_shap[: len(feature_names)],
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return {
            "available": True,
            "shap_summary_df": shap_summary_df,
            "shap_values": shap_vals,
            "base_values": base_vals,
            "fg_sample": fg_sample,
            "feature_names": feature_names,
        }
    except Exception as e:
        return {"available": False, "error": f"SHAP computation encountered an issue: {str(e)}"}


def compute_lime_explanation(
    model: Any,
    X_train: np.ndarray,
    instance: np.ndarray,
    feature_names: List[str],
    problem_type: str = "classification",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes local sample explanation using LIME (LimeTabularExplainer).
    """
    if not HAS_LIME:
        return {"available": False, "error": "LIME library is not installed."}

    try:
        mode = "classification" if problem_type in ["binary_classification", "multiclass_classification"] else "regression"
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            verbose=False,
            random_state=42,
        )

        predict_fn = model.predict_proba if mode == "classification" and hasattr(model, "predict_proba") else model.predict
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=min(10, len(feature_names)),
        )

        explanation_list = exp.as_list()
        return {
            "available": True,
            "explanation_list": explanation_list,
            "predicted_value": float(exp.predicted_value) if hasattr(exp, "predicted_value") else None,
            "score": float(exp.score) if hasattr(exp, "score") else None,
            "intercept": float(exp.intercept[1]) if hasattr(exp, "intercept") and isinstance(exp.intercept, list) else None,
        }
    except Exception as e:
        return {"available": False, "error": f"LIME computation error: {str(e)}"}


def generate_plain_english_explanation(
    importance_df: pd.DataFrame,
    prediction: Any = None,
    probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Translates mathematical importance scores and probabilities into clear,
    human-readable non-technical summaries for stakeholders and faculty.
    """
    if importance_df.empty:
        return {
            "headline": "No feature importance data available.",
            "top_drivers": [],
            "narrative": "Train a model to generate feature attribution summaries.",
        }

    top_drivers = []
    top_n = importance_df.head(5)

    for _, row in top_n.iterrows():
        pct = row.get("importance_pct", 0.0)
        if pct >= 25.0:
            impact_level = "High Impact"
            color = "#ef4444"
        elif pct >= 10.0:
            impact_level = "Moderate Impact"
            color = "#f59e0b"
        else:
            impact_level = "Supporting Factor"
            color = "#3b82f6"

        top_drivers.append({
            "feature": str(row["feature"]),
            "importance_pct": round(pct, 1),
            "impact_level": impact_level,
            "color": color,
        })

    top_feat_names = [d["feature"] for d in top_drivers[:3]]
    feat_str = " and ".join([f"**{f}**" for f in top_feat_names]) if len(top_feat_names) <= 2 else f"**{top_feat_names[0]}**, **{top_feat_names[1]}**, and **{top_feat_names[2]}**"

    if prediction is not None:
        headline = f"Prediction: **{prediction}**"
        if probabilities and len(probabilities) > 1:
            best_prob = max(probabilities.values()) * 100
            headline += f" (Confidence: {best_prob:.1f}%)"
        narrative = f"The model's decision is primarily driven by {feat_str}. Variations in these attributes have the strongest statistical weight on the outcome."
    else:
        headline = f"Primary Drivers: {feat_str}"
        narrative = f"Across all trained samples, {feat_str} account for the majority of predictive power."

    return {
        "headline": headline,
        "top_drivers": top_drivers,
        "narrative": narrative,
    }


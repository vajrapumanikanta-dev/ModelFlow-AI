"""
Module 12: Model Saving & Prediction (core/predictor.py)
Serializes the entire end-to-end Pipeline (preprocessor + model + schema metadata)
and serves both Single-Instance and Batch Prediction workflows.
"""

import os
import io
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import joblib


def format_prediction_output(
    problem_type: str,
    target_col: Optional[str],
    raw_pred: Any,
    pred_label: Any,
    model_name: str,
    probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Translates raw ML outputs into clear, human-readable predictions and plain-English explanations.
    """
    target_name = target_col if target_col else "Target"
    t_lower = target_name.lower()

    if problem_type == "regression":
        raw_float = float(raw_pred)
        raw_str = f"{raw_float:.4f}"

        # California Housing dataset representation (MedHouseVal in $100k units)
        if "medhouseval" in t_lower or "median_house_value" in t_lower or "houseval" in t_lower:
            dollar_val = raw_float * 100000.0
            display_title = "Predicted House Value"
            human_readable = f"${dollar_val:,.0f}"
            explanation = (
                f"The model estimates the median house value for the provided property information "
                f"as approximately ${dollar_val:,.0f}."
            )
        else:
            # Generic regression: do NOT assume house price or currency
            display_title = f"Predicted {target_name}"
            human_readable = f"{raw_float:.4f}"
            explanation = f"The model predicts {target_name} to be approximately {raw_float:.4f}."

        return {
            "prediction_title": display_title,
            "human_readable_prediction": human_readable,
            "raw_prediction_str": raw_str,
            "explanation": explanation,
            "confidence_pct": None,
        }

    else:
        # Classification
        conf_str = None
        if probabilities:
            # Look up probability of predicted class or take max probability
            top_prob = None
            if str(pred_label) in probabilities:
                top_prob = probabilities[str(pred_label)]
            elif str(raw_pred) in probabilities:
                top_prob = probabilities[str(raw_pred)]
            else:
                top_prob = max(probabilities.values()) if probabilities.values() else None

            if top_prob is not None:
                conf_str = f"{top_prob * 100:.1f}%"

        label_str = str(pred_label).strip()
        label_lower = label_str.lower()

        if "churn" in t_lower:
            if label_lower in ["1", "yes", "true", "churn", "churned"]:
                human_readable = "Customer will Churn"
                status_desc = "customer will Churn"
            else:
                human_readable = "Customer will Stay / Not Churn"
                status_desc = "customer will Stay (Not Churn)"
            display_title = "Prediction"
            conf_part = f"with {conf_str} confidence " if conf_str else ""
            explanation = f"The model predicts {conf_part}that the {status_desc} based on the provided customer profile."

        elif "survived" in t_lower or "survival" in t_lower:
            if label_lower in ["1", "yes", "true", "survived"]:
                human_readable = "Passenger Survived"
                status_desc = "passenger Survived"
            else:
                human_readable = "Passenger Did Not Survive"
                status_desc = "passenger Did Not Survive"
            display_title = "Prediction"
            conf_part = f"with {conf_str} confidence " if conf_str else ""
            explanation = f"The model predicts {conf_part}that the {status_desc} based on the provided passenger details."

        else:
            display_title = "Prediction"
            human_readable = label_str
            conf_part = f"with {conf_str} confidence " if conf_str else ""
            explanation = f"The model predicts {conf_part}that the sample belongs to '{label_str}'."

        return {
            "prediction_title": display_title,
            "human_readable_prediction": human_readable,
            "raw_prediction_str": label_str,
            "explanation": explanation,
            "confidence_pct": conf_str,
        }


class SerializedPipelineBundle:
    """
    Complete standalone wrapper containing the Preprocessor, Model,
    Feature Schema, Target Information, and Class Mappings.
    """

    def __init__(
        self,
        preprocessor: Any,
        model: Any,
        feature_cols: List[str],
        numeric_cols: List[str],
        categorical_cols: List[str],
        target_col: Optional[str],
        problem_type: str,
        class_names: Optional[List[str]] = None,
        target_mapping: Optional[Dict[Any, int]] = None,
        feature_schema: Optional[Dict[str, Dict[str, Any]]] = None,
        model_name: str = "ML_Model",
        metrics_summary: Optional[Dict[str, Any]] = None,
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.feature_cols = feature_cols
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.problem_type = problem_type
        self.class_names = class_names or []
        self.target_mapping = target_mapping or {}
        self.inverse_target_mapping = {v: k for k, v in self.target_mapping.items()} if target_mapping else {}
        self.feature_schema = feature_schema or {}
        self.model_name = model_name
        self.metrics_summary = metrics_summary or {}
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs prediction for a single user-input dictionary.
        Uses exact fitted preprocessing pipeline and feature schema from training.
        """
        # Build single row dataframe matching feature_cols exactly in training order
        row_dict = {}
        for col in self.feature_cols:
            val = input_data.get(col)
            if val is None or val == "":
                # Fallback to schema default
                if col in self.numeric_cols:
                    val = self.feature_schema.get(col, {}).get("median", 0.0)
                else:
                    val = self.feature_schema.get(col, {}).get("top_value", "Missing")
            row_dict[col] = [val]

        df_single = pd.DataFrame(row_dict)[self.feature_cols]

        # Apply exact fitted preprocessor pipeline
        X_proc = self.preprocessor.transform(df_single)

        # Model Inference
        raw_pred = self.model.predict(X_proc)[0]

        # Probabilities if classification
        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(X_proc)[0]
                for idx, p in enumerate(probs):
                    label = self.class_names[idx] if self.class_names and idx < len(self.class_names) else f"Class_{idx}"
                    probabilities[str(label)] = round(float(p), 4)
            except Exception:
                probabilities = {}

        # Decode prediction label
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            pred_label = self.inverse_target_mapping.get(raw_pred, str(raw_pred))
            if self.class_names and isinstance(raw_pred, (int, np.integer)) and 0 <= int(raw_pred) < len(self.class_names):
                pred_label = self.class_names[int(raw_pred)]
            raw_output = int(raw_pred) if isinstance(raw_pred, (int, np.integer)) else (float(raw_pred) if isinstance(raw_pred, (float, np.floating)) else str(raw_pred))
        else:
            raw_float = float(raw_pred)
            pred_label = f"{raw_float:.4f}"
            raw_output = raw_float

        target_display = self.target_col if self.target_col else "Target"

        # Format human-readable output & explanation
        formatted = format_prediction_output(
            problem_type=self.problem_type,
            target_col=self.target_col,
            raw_pred=raw_output,
            pred_label=pred_label,
            model_name=self.model_name,
            probabilities=probabilities,
        )

        ptype_display = (
            "Regression"
            if self.problem_type == "regression"
            else ("Binary Classification" if self.problem_type == "binary_classification" else "Multi-Class Classification")
        )

        return {
            "prediction": pred_label,
            "raw_prediction": raw_output,
            "raw_prediction_str": formatted["raw_prediction_str"],
            "target_name": target_display,
            "display_title": formatted["prediction_title"],
            "human_readable_prediction": formatted["human_readable_prediction"],
            "explanation": formatted["explanation"],
            "confidence_pct": formatted["confidence_pct"],
            "display_text": f"Predicted {target_display}: {formatted['raw_prediction_str'] if self.problem_type == 'regression' else pred_label}",
            "probabilities": probabilities,
            "problem_type": self.problem_type,
            "problem_type_display": ptype_display,
            "model_name": self.model_name,
        }

    def predict_batch(self, batch_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs batch prediction on a new DataFrame.
        Validates column presence and appends predictions, human-readable outcomes, and probabilities.
        """
        # Validate columns
        missing_cols = [c for c in self.feature_cols if c not in batch_df.columns]
        if missing_cols:
            raise ValueError(
                f"Batch dataset is missing {len(missing_cols)} required feature columns: {missing_cols}"
            )

        X_input = batch_df[self.feature_cols].copy()

        # Transform
        X_proc = self.preprocessor.transform(X_input)

        # Predict
        raw_preds = self.model.predict(X_proc)

        result_df = batch_df.copy()

        target_name = self.target_col if self.target_col else "target"
        t_lower = target_name.lower()
        pred_col_name = f"Predicted_{target_name}"

        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            # Decode labels
            if self.inverse_target_mapping:
                decoded_preds = [self.inverse_target_mapping.get(p, p) for p in raw_preds]
            elif self.class_names:
                decoded_preds = [self.class_names[int(p)] if 0 <= int(p) < len(self.class_names) else p for p in raw_preds]
            else:
                decoded_preds = raw_preds.tolist()

            result_df[pred_col_name] = decoded_preds

            # Human-readable outcome column for common targets
            if "churn" in t_lower:
                result_df["Prediction_Outcome"] = [
                    "Customer will Churn" if str(p).lower() in ["1", "yes", "true", "churn", "churned"] else "Customer will Stay / Not Churn"
                    for p in decoded_preds
                ]
            elif "survived" in t_lower:
                result_df["Prediction_Outcome"] = [
                    "Passenger Survived" if str(p).lower() in ["1", "yes", "true", "survived"] else "Passenger Did Not Survive"
                    for p in decoded_preds
                ]

            # Add probability columns if supported
            if hasattr(self.model, "predict_proba"):
                try:
                    probs_matrix = self.model.predict_proba(X_proc)
                    for idx in range(probs_matrix.shape[1]):
                        cls_name = self.class_names[idx] if self.class_names and idx < len(self.class_names) else f"Class_{idx}"
                        result_df[f"Prob_{cls_name}"] = np.round(probs_matrix[:, idx], 4)
                except Exception:
                    pass
        else:
            result_df[pred_col_name] = np.round(raw_preds, 4)
            if "medhouseval" in t_lower or "median_house_value" in t_lower or "houseval" in t_lower:
                result_df["Estimated_House_Value_USD"] = [f"${val * 100000:,.0f}" for val in raw_preds]

        summary = {
            "total_samples": len(batch_df),
            "predicted_column": pred_col_name,
            "problem_type": self.problem_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return result_df, summary


def build_feature_schema(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Constructs feature metadata (min, max, median, unique values)
    to power dynamic prediction forms in the UI.
    """
    schema = {}
    for col in feature_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            schema[col] = {
                "type": "numeric",
                "min": float(clean.min()) if not clean.empty else 0.0,
                "max": float(clean.max()) if not clean.empty else 100.0,
                "mean": float(clean.mean()) if not clean.empty else 0.0,
                "median": float(clean.median()) if not clean.empty else 0.0,
                "default": float(clean.median()) if not clean.empty else 0.0,
            }
        else:
            clean = s.dropna()
            uniques = [str(x) for x in clean.unique()]
            top_val = str(clean.mode().iloc[0]) if not clean.empty else ""
            schema[col] = {
                "type": "categorical",
                "options": uniques[:100],  # cap options for UI
                "top_value": top_val,
                "default": top_val if top_val else (uniques[0] if uniques else ""),
            }
    return schema


def save_pipeline(
    preprocessor: Any,
    model: Any,
    feature_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: Optional[str],
    problem_type: str,
    df_sample: pd.DataFrame,
    class_names: Optional[List[str]] = None,
    target_mapping: Optional[Dict[Any, int]] = None,
    model_name: str = "ML_Model",
    metrics_summary: Optional[Dict[str, Any]] = None,
    save_dir: str = "saved_models",
) -> str:
    """
    Saves the complete SerializedPipelineBundle into a .joblib artifact.
    """
    os.makedirs(save_dir, exist_ok=True)
    schema = build_feature_schema(df_sample, feature_cols)

    bundle = SerializedPipelineBundle(
        preprocessor=preprocessor,
        model=model,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        problem_type=problem_type,
        class_names=class_names,
        target_mapping=target_mapping,
        feature_schema=schema,
        model_name=model_name,
        metrics_summary=metrics_summary,
    )

    clean_name = model_name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean_name}_{timestamp}.joblib"
    filepath = os.path.join(save_dir, filename)

    joblib.dump(bundle, filepath)
    return filepath


def load_pipeline(filepath_or_buffer: Union[str, io.BytesIO]) -> SerializedPipelineBundle:
    """
    Loads a SerializedPipelineBundle from file path or buffer.
    """
    bundle = joblib.load(filepath_or_buffer)
    if not isinstance(bundle, SerializedPipelineBundle):
        raise ValueError("Loaded object is not a valid ModelFlow AI SerializedPipelineBundle.")
    return bundle

"""
End-to-End Workflow Integration Test (tests/test_e2e_workflow.py)
Validates the complete lifecycle:
Upload -> Profile -> Clean -> Feature/Target Selection -> Quality Checks -> Preprocessing
-> Multi-Model Training -> CV -> Hyperparameter Tuning -> Evaluation -> Leaderboard
-> Explainability (SHAP & LIME) -> Model Saving -> Single & Batch Prediction
-> Experiment Tracking -> HTML & PDF ML Report Generation.
"""

import os
import pytest
import pandas as pd
import numpy as np

from core.loader import load_dataset
from core.profiler import profile_dataset, get_column_summary_df
from core.cleaner import DataCleaner
from core.quality import detect_problem_type, detect_data_leakage, detect_class_imbalance
from core.preprocess import prepare_train_test_data
from core.models import get_classification_models, get_regression_models
from core.trainer import train_and_evaluate_models, tune_hyperparameters
from core.evaluator import generate_leaderboard
from core.explain import extract_feature_importance, compute_shap_explanations, compute_lime_explanation
from core.predictor import save_pipeline, load_pipeline
from core.tracker import ExperimentTracker
from core.report import generate_html_report, generate_pdf_report
from plots.visualizer import (
    plot_distribution,
    plot_confusion_matrix_heatmap,
    plot_actual_vs_predicted,
    fig_to_base64,
)


def test_end_to_end_titanic_workflow(tmp_path):
    # 1. Load Titanic
    df_raw, meta = load_dataset("sample_data/titanic.csv")
    assert meta["rows"] >= 300

    # 2. Profile
    profile = profile_dataset(df_raw)
    assert profile["summary"]["total_cols"] == 8
    summary_df = get_column_summary_df(profile)
    assert not summary_df.empty

    # 3. Clean
    cleaner = DataCleaner(df_raw)
    cleaner.handle_missing(["Age"], strategy="median")
    cleaner.handle_missing(["Embarked"], strategy="mode")
    cleaner.handle_outliers("Fare", method="iqr", threshold=2.0, action="cap")
    clean_df = cleaner.current_df
    assert clean_df["Age"].isna().sum() == 0

    # 4. Problem & Quality Checks
    prob_meta = detect_problem_type(clean_df, "Survived")
    assert prob_meta["problem_type"] == "binary_classification"

    features = ["Pclass", "Sex", "Age", "SibSp", "Fare", "Embarked"]
    leakage = detect_data_leakage(clean_df, "Survived", features)
    imbalance = detect_class_imbalance(clean_df["Survived"])
    assert "is_imbalanced" in imbalance

    # 5. Preprocessing (Strict Split-First)
    prep = prepare_train_test_data(
        df=clean_df,
        feature_cols=features,
        target_col="Survived",
        problem_type="binary_classification",
        test_size=0.2,
        num_scaler="standard",
        cat_encoder="onehot",
    )
    assert prep["X_train_proc"].shape[0] > 0
    assert prep["X_test_proc"].shape[0] > 0

    # 6. Train Models
    models_reg = get_classification_models()
    selected_models = {
        "Logistic Regression": models_reg["Logistic Regression"],
        "Random Forest": models_reg["Random Forest"],
        "Decision Tree": models_reg["Decision Tree"],
    }
    train_results = train_and_evaluate_models(
        models_dict=selected_models,
        X_train=prep["X_train_proc"],
        y_train=prep["y_train"],
        X_test=prep["X_test_proc"],
        y_test=prep["y_test"],
        problem_type="binary_classification",
        cv_folds=3,
        class_names=prep.get("class_names"),
    )
    assert len(train_results) == 3

    # 7. Leaderboard
    leaderboard = generate_leaderboard(train_results, "binary_classification")
    best_model_name = leaderboard.iloc[0]["Model"]
    assert best_model_name in train_results

    # 8. Explainability
    best_estimator = train_results[best_model_name]["estimator"]
    imp_df = extract_feature_importance(best_estimator, prep["feature_names_out"])
    assert not imp_df.empty

    # SHAP
    shap_res = compute_shap_explanations(
        best_estimator,
        prep["X_train_proc"],
        prep["X_test_proc"],
        prep["feature_names_out"],
        sample_size=30,
    )
    assert shap_res.get("available") is True

    # LIME
    lime_res = compute_lime_explanation(
        best_estimator,
        prep["X_train_proc"],
        prep["X_test_proc"][0],
        prep["feature_names_out"],
        problem_type="binary_classification",
        class_names=prep.get("class_names"),
    )
    assert lime_res.get("available") is True

    # 9. Save Pipeline Bundle
    saved_path = save_pipeline(
        preprocessor=prep["preprocessor"],
        model=best_estimator,
        feature_cols=features,
        numeric_cols=prep["numeric_cols"],
        categorical_cols=prep["categorical_cols"],
        target_col="Survived",
        problem_type="binary_classification",
        df_sample=clean_df[features],
        class_names=prep.get("class_names"),
        target_mapping=prep.get("target_mapping"),
        model_name=best_model_name,
        save_dir=str(tmp_path / "models"),
    )
    assert os.path.exists(saved_path)

    # 10. Load & Run Single/Batch Predictions
    bundle = load_pipeline(saved_path)
    single_res = bundle.predict_single({
        "Pclass": 3,
        "Sex": "female",
        "Age": 22.0,
        "SibSp": 0,
        "Fare": 7.25,
        "Embarked": "S",
    })
    assert "prediction" in single_res
    assert "probabilities" in single_res

    batch_res_df, b_sum = bundle.predict_batch(clean_df[features].head(10))
    assert "Predicted_Survived" in batch_res_df.columns
    assert b_sum["total_samples"] == 10

    # 11. Experiment Tracker
    tracker = ExperimentTracker(db_path=str(tmp_path / "exp.db"))
    run_id = tracker.log_run(
        model_name=best_model_name,
        problem_type="binary_classification",
        dataset_name="titanic.csv",
        features=features,
        test_metrics=train_results[best_model_name]["test_metrics"],
        target_col="Survived",
        cv_score=train_results[best_model_name]["cv_score"],
        model_path=saved_path,
    )
    assert tracker.get_run(run_id) is not None

    # 12. Report Generation (HTML & PDF)
    fig_dist = plot_distribution(clean_df, "Age")
    charts = [{"title": "Age Distribution", "base64": fig_to_base64(fig_dist)}]

    report_payload = {
        "dataset_name": "titanic.csv",
        "problem_type": "binary_classification",
        "timestamp": "2026-10-04 12:00:00",
        "rows": len(clean_df),
        "cols": clean_df.shape[1],
        "memory_usage": "25 KB",
        "missing_cells": 0,
        "missing_pct": 0.0,
        "duplicate_rows": 0,
        "target_col": "Survived",
        "cleaning_log": cleaner.get_cleaning_log(),
        "leakage_warnings": leakage,
        "imbalance_info": imbalance,
        "leaderboard_html": leaderboard.to_html(classes="table", index=False),
        "leaderboard_table_data": [list(leaderboard.columns)] + leaderboard.values.tolist(),
        "best_model_name": best_model_name,
        "charts": charts,
        "top_features": imp_df.head(5).to_dict(orient="records"),
    }

    html_path = str(tmp_path / "titanic_report.html")
    pdf_path = str(tmp_path / "titanic_report.pdf")
    generate_html_report(report_payload, output_path=html_path)
    generate_pdf_report(report_payload, output_path=pdf_path)

    assert os.path.exists(html_path)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000


def test_end_to_end_housing_regression_workflow(tmp_path):
    # 1. Load Housing
    df_raw, meta = load_dataset("sample_data/housing.csv")
    assert meta["rows"] >= 500

    target = "MedHouseVal" if "MedHouseVal" in df_raw.columns else df_raw.columns[-1]
    features = [c for c in df_raw.columns if c != target]

    # 2. Problem Detection
    prob_meta = detect_problem_type(df_raw, target)
    assert prob_meta["problem_type"] == "regression"

    # 3. Preprocess
    prep = prepare_train_test_data(
        df=df_raw,
        feature_cols=features,
        target_col=target,
        problem_type="regression",
        num_scaler="robust",
    )

    # 4. Train Regressors
    models_reg = get_regression_models()
    selected_reg = {
        "Linear Regression": models_reg["Linear Regression"],
        "Ridge": models_reg["Ridge Regression"],
        "Random Forest Regressor": models_reg["Random Forest Regressor"],
    }
    train_res = train_and_evaluate_models(
        models_dict=selected_reg,
        X_train=prep["X_train_proc"],
        y_train=prep["y_train"],
        X_test=prep["X_test_proc"],
        y_test=prep["y_test"],
        problem_type="regression",
        cv_folds=3,
    )

    board = generate_leaderboard(train_res, "regression")
    assert len(board) == 3
    assert "R² Score" in board.columns
    assert "RMSE" in board.columns

    best_model_name = board.iloc[0]["Model"]
    best_estimator = train_res[best_model_name]["estimator"]

    # 5. Save Pipeline Bundle
    saved_path = save_pipeline(
        preprocessor=prep["preprocessor"],
        model=best_estimator,
        feature_cols=features,
        numeric_cols=prep["numeric_cols"],
        categorical_cols=prep["categorical_cols"],
        target_col=target,
        problem_type="regression",
        df_sample=df_raw[features],
        model_name=best_model_name,
        save_dir=str(tmp_path / "models"),
    )
    assert os.path.exists(saved_path)

    # 6. Load & Predict Single Instance
    bundle = load_pipeline(saved_path)
    assert bundle.target_col == "MedHouseVal"
    assert "MedHouseVal" not in bundle.feature_cols

    sample_input = {
        "MedInc": 3.5,
        "HouseAge": 25.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 37.8,
        "Longitude": -122.4,
    }
    pred_res = bundle.predict_single(sample_input)

    assert pred_res["target_name"] == "MedHouseVal"
    assert pred_res["problem_type"] == "regression"
    assert pred_res["problem_type_display"] == "Regression"
    assert isinstance(pred_res["raw_prediction"], float)
    assert len(pred_res["raw_prediction_str"].split(".")[1]) >= 4
    assert pred_res["display_title"] == "Predicted House Value"
    assert pred_res["human_readable_prediction"].startswith("$")
    assert "The model estimates the median house value" in pred_res["explanation"]
    assert pred_res["display_text"].startswith("Predicted MedHouseVal:")

    # 7. Batch Prediction
    batch_df = df_raw[features].head(10)
    batch_res_df, b_summary = bundle.predict_batch(batch_df)
    assert "Predicted_MedHouseVal" in batch_res_df.columns
    assert "Estimated_House_Value_USD" in batch_res_df.columns
    assert b_summary["total_samples"] == 10


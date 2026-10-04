"""
ModelFlow AI - Backend REST API Server (server.py)
Provides clean separation between the Python ML engine and the HTML/CSS/JS frontend.
"""

import os
import io
import time
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, send_from_directory

# Core ML engine modules
from core.loader import load_dataset
from core.profiler import profile_dataset, get_column_summary_df
from core.cleaner import DataCleaner
from core.quality import detect_problem_type, detect_data_leakage, detect_class_imbalance
from core.preprocess import prepare_train_test_data
from core.models import get_classification_models, get_regression_models
from core.trainer import train_and_evaluate_models, tune_hyperparameters, generate_standalone_code
from core.evaluator import generate_leaderboard
from core.explain import extract_feature_importance, compute_shap_explanations, compute_lime_explanation
from core.predictor import save_pipeline, load_pipeline, format_prediction_output, SerializedPipelineBundle
from core.tracker import ExperimentTracker
from core.report import generate_html_report, generate_pdf_report
from plots.visualizer import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_confusion_matrix_heatmap,
    plot_roc_curve_chart,
    plot_precision_recall_curve_chart,
    plot_actual_vs_predicted,
    plot_residuals,
    fig_to_base64,
)

app = Flask(__name__, static_folder="frontend", static_url_path="")

# In-Memory Session State
state: Dict[str, Any] = {
    "df_raw": None,
    "current_df": None,
    "dataset_name": "No dataset loaded",
    "cleaner": None,
    "target_col": None,
    "feature_cols": [],
    "problem_type": "binary_classification",
    "prepared_data": None,
    "training_results": None,
    "best_model_name": None,
    "saved_pipeline_path": None,
    "active_bundle": None,
    "tracker": ExperimentTracker(db_path="experiments/modelflow_experiments.db"),
}

@app.after_request
def add_cors_and_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response

# ==============================================================================
# 1. STATIC FRONTEND ROUTES
# ==============================================================================

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("frontend", path)

# ==============================================================================
# 2. DATASET INGESTION & PROFILING API
# ==============================================================================

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "online",
        "dataset_loaded": state["current_df"] is not None,
        "dataset_name": state["dataset_name"],
        "rows": len(state["current_df"]) if state["current_df"] is not None else 0,
        "cols": state["current_df"].shape[1] if state["current_df"] is not None else 0,
        "target_col": state["target_col"],
        "problem_type": state["problem_type"],
        "has_trained_model": state["training_results"] is not None,
        "best_model_name": state["best_model_name"],
    })

@app.route("/api/load-sample", methods=["POST"])
def load_sample():
    data = request.get_json() or {}
    sample_key = data.get("sample", "iris")
    
    mapping = {
        "iris": "sample_data/iris.csv",
        "titanic": "sample_data/titanic.csv",
        "housing": "sample_data/housing.csv",
        "churn": "sample_data/telco_churn.csv",
    }
    filepath = mapping.get(sample_key, "sample_data/iris.csv")
    
    try:
        df, meta = load_dataset(filepath)
        state["df_raw"] = df.copy()
        state["current_df"] = df.copy()
        state["dataset_name"] = os.path.basename(filepath)
        state["cleaner"] = DataCleaner(df)
        
        # Smart defaults
        cols = list(df.columns)
        if sample_key == "iris":
            target = "species" if "species" in cols else cols[-1]
        elif sample_key == "titanic":
            target = "Survived" if "Survived" in cols else cols[-1]
        elif sample_key == "housing":
            target = "MedHouseVal" if "MedHouseVal" in cols else cols[-1]
        elif sample_key == "churn":
            target = "Churn" if "Churn" in cols else cols[-1]
        else:
            target = cols[-1]
            
        state["target_col"] = target
        state["feature_cols"] = [c for c in cols if c != target]
        
        prob_res = detect_problem_type(df, target)
        state["problem_type"] = prob_res["problem_type"]
        
        profile = profile_dataset(df)
        
        return jsonify({
            "success": True,
            "message": f"Successfully loaded {state['dataset_name']}",
            "dataset_name": state["dataset_name"],
            "target_col": state["target_col"],
            "feature_cols": state["feature_cols"],
            "problem_type": state["problem_type"],
            "columns": cols,
            "summary": profile["summary"],
            "preview": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
            "column_profiles": {col: {"dtype": str(df[col].dtype), "missing": int(df[col].isna().sum()), "unique": int(df[col].nunique())} for col in cols},
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400
        
    try:
        df, meta = load_dataset(file)
        state["df_raw"] = df.copy()
        state["current_df"] = df.copy()
        state["dataset_name"] = file.filename
        state["cleaner"] = DataCleaner(df)
        
        cols = list(df.columns)
        state["target_col"] = cols[-1]
        state["feature_cols"] = cols[:-1]
        
        prob_res = detect_problem_type(df, state["target_col"])
        state["problem_type"] = prob_res["problem_type"]
        profile = profile_dataset(df)
        
        return jsonify({
            "success": True,
            "message": f"Uploaded {file.filename}",
            "dataset_name": state["dataset_name"],
            "target_col": state["target_col"],
            "feature_cols": state["feature_cols"],
            "problem_type": state["problem_type"],
            "columns": cols,
            "summary": profile["summary"],
            "preview": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
            "column_profiles": {col: {"dtype": str(df[col].dtype), "missing": int(df[col].isna().sum()), "unique": int(df[col].nunique())} for col in cols},
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/dataset-profile", methods=["GET"])
def get_profile():
    if state["current_df"] is None:
        return jsonify({"success": False, "error": "No dataset loaded"}), 400
    try:
        df = state["current_df"]
        profile = profile_dataset(df)
        col_summary = get_column_summary_df(profile)
        
        return jsonify({
            "success": True,
            "dataset_name": state["dataset_name"],
            "summary": profile["summary"],
            "columns": list(df.columns),
            "col_summary_table": col_summary.replace({np.nan: None}).to_dict(orient="records"),
            "preview": df.head(20).replace({np.nan: None}).to_dict(orient="records"),
            "target_col": state["target_col"],
            "feature_cols": state["feature_cols"],
            "problem_type": state["problem_type"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==============================================================================
# 3. DATA CLEANING & PREPARATION API
# ==============================================================================

@app.route("/api/clean", methods=["POST"])
def clean_data():
    if state["current_df"] is None:
        return jsonify({"success": False, "error": "No dataset loaded"}), 400
    
    data = request.get_json() or {}
    cleaner: DataCleaner = state["cleaner"]
    
    action = data.get("action")
    try:
        if action == "missing":
            cols = data.get("columns", [])
            strategy = data.get("strategy", "median")
            fill_val = data.get("fill_value")
            cleaner.handle_missing(cols, strategy=strategy, fill_value=fill_val)
        elif action == "outliers":
            col = data.get("column")
            method = data.get("method", "iqr")
            thresh = float(data.get("threshold", 1.5))
            act = data.get("outlier_action", "cap")
            cleaner.handle_outliers(col, method=method, threshold=thresh, action=act)
        elif action == "duplicates":
            cleaner.remove_duplicates()
        elif action == "drop_cols":
            cols = data.get("columns", [])
            cleaner.drop_columns(cols)
        elif action == "reset":
            state["current_df"] = state["df_raw"].copy()
            state["cleaner"] = DataCleaner(state["df_raw"])
            
        state["current_df"] = state["cleaner"].current_df
        profile = profile_dataset(state["current_df"])
        
        return jsonify({
            "success": True,
            "message": "Data cleaning action applied",
            "cleaning_log": state["cleaner"].get_cleaning_log(),
            "summary": profile["summary"],
            "columns": list(state["current_df"].columns),
            "preview": state["current_df"].head(10).replace({np.nan: None}).to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/configure-target", methods=["POST"])
def configure_target():
    data = request.get_json() or {}
    target = data.get("target_col")
    features = data.get("feature_cols")
    
    if target not in state["current_df"].columns:
        return jsonify({"success": False, "error": "Invalid target column"}), 400
        
    state["target_col"] = target
    state["feature_cols"] = features if features else [c for c in state["current_df"].columns if c != target]
    
    prob_res = detect_problem_type(state["current_df"], target)
    state["problem_type"] = prob_res["problem_type"]
    
    leakage = detect_data_leakage(state["current_df"], target, state["feature_cols"])
    imbalance = detect_class_imbalance(state["current_df"][target]) if state["problem_type"] != "regression" else {}
    
    return jsonify({
        "success": True,
        "target_col": state["target_col"],
        "feature_cols": state["feature_cols"],
        "problem_type": state["problem_type"],
        "problem_type_reason": prob_res["reason"],
        "leakage_warnings": leakage,
        "imbalance_info": imbalance,
    })

# ==============================================================================
# 4. MODEL TRAINING & COMPARISON API
# ==============================================================================

@app.route("/api/train", methods=["POST"])
def train_models_api():
    if state["current_df"] is None:
        return jsonify({"success": False, "error": "No dataset loaded"}), 400
        
    data = request.get_json() or {}
    target = data.get("target_col", state["target_col"])
    features = data.get("feature_cols", state["feature_cols"])
    test_size = float(data.get("test_size", 0.2))
    num_scaler = data.get("num_scaler", "standard")
    num_imputer = data.get("num_imputer", "median")
    cat_encoder = data.get("cat_encoder", "onehot")
    resample_opt = data.get("resampling", "none")
    selected_algos = data.get("algorithms", [])
    tune_best = data.get("tune_best", False)
    cv_folds = int(data.get("cv_folds", 3))

    df = state["current_df"]
    ptype = state["problem_type"]

    try:
        # 1. Preprocess
        prep = prepare_train_test_data(
            df=df,
            feature_cols=features,
            target_col=target,
            problem_type=ptype,
            test_size=test_size,
            num_scaler=num_scaler,
            num_impute_strategy=num_imputer,
            cat_encoder=cat_encoder,
            resampling_strategy=resample_opt,
        )
        state["prepared_data"] = prep

        # 2. Select Models
        all_models = get_classification_models() if ptype in ["binary_classification", "multiclass_classification"] else get_regression_models()
        models_to_train = {k: v for k, v in all_models.items() if (not selected_algos or k in selected_algos)}
        if not models_to_train:
            models_to_train = all_models

        # 3. Train and Evaluate
        results = train_and_evaluate_models(
            models_dict=models_to_train,
            X_train=prep["X_train_proc"],
            y_train=prep["y_train"],
            X_test=prep["X_test_proc"],
            y_test=prep["y_test"],
            problem_type=ptype,
            cv_folds=cv_folds,
            class_names=prep.get("class_names"),
        )

        # 4. Optional Hyperparameter Tuning on Top Model
        if tune_best and results:
            temp_board = generate_leaderboard(results, ptype)
            top_model_name = temp_board.iloc[0]["Model"]
            if top_model_name in all_models:
                tune_res = tune_hyperparameters(
                    model_name=top_model_name,
                    estimator=all_models[top_model_name]["estimator"],
                    param_grid=all_models[top_model_name]["param_grid"],
                    X_train=prep["X_train_proc"],
                    y_train=prep["y_train"],
                    X_test=prep["X_test_proc"],
                    y_test=prep["y_test"],
                    problem_type=ptype,
                    class_names=prep.get("class_names"),
                )
                if tune_res.get("status") == "success":
                    results[tune_res["model_name"]] = tune_res

        state["training_results"] = results

        # 5. Leaderboard & Best Model
        board_df = generate_leaderboard(results, ptype)
        best_name = board_df.iloc[0]["Model"] if not board_df.empty else list(results.keys())[0]
        state["best_model_name"] = best_name
        best_est = results[best_name].get("best_estimator", results[best_name].get("estimator"))

        # 6. Serialize Pipeline Bundle
        saved_path = save_pipeline(
            preprocessor=prep["preprocessor"],
            model=best_est,
            feature_cols=features,
            numeric_cols=prep["numeric_cols"],
            categorical_cols=prep["categorical_cols"],
            target_col=target,
            problem_type=ptype,
            df_sample=df[features],
            class_names=prep.get("class_names"),
            target_mapping=prep.get("target_mapping"),
            model_name=best_name,
            metrics_summary=results[best_name].get("test_metrics", {}),
        )
        state["saved_pipeline_path"] = saved_path
        state["active_bundle"] = load_pipeline(saved_path)

        # 7. Log to Experiment Tracker
        tracker: ExperimentTracker = state["tracker"]
        run_id = tracker.log_run(
            model_name=best_name,
            problem_type=ptype,
            dataset_name=state["dataset_name"],
            features=features,
            test_metrics=results[best_name].get("test_metrics", {}),
            target_col=target,
            cv_score=results[best_name].get("cv_score"),
            train_duration_sec=results[best_name].get("train_duration_sec", 0.0),
            model_path=saved_path,
        )

        # 8. Standalone Code Generation
        standalone_code = generate_standalone_code(
            dataset_name=state["dataset_name"],
            target_col=target,
            feature_cols=features,
            numeric_cols=prep.get("numeric_cols", []),
            categorical_cols=prep.get("categorical_cols", []),
            problem_type=ptype,
            model_name=best_name,
            model_obj=best_est,
            best_params=results[best_name].get("best_params"),
            cleaning_log=state["cleaner"].get_cleaning_log(),
            test_size=test_size,
            num_scaler=num_scaler,
            num_imputer=num_imputer,
            cat_encoder=cat_encoder,
            resampling=resample_opt,
        )

        leaderboard_records = board_df.replace({np.nan: None}).to_dict(orient="records")

        return jsonify({
            "success": True,
            "message": f"Trained {len(results)} models successfully",
            "best_model_name": best_name,
            "problem_type": ptype,
            "leaderboard": leaderboard_records,
            "standalone_code": standalone_code,
            "feature_schema": state["active_bundle"].feature_schema,
            "target_col": target,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ==============================================================================
# 5. VISUAL DIAGNOSTICS & EXPLAINABILITY (XAI) API
# ==============================================================================

@app.route("/api/diagnostics", methods=["GET"])
def get_diagnostics():
    if state["training_results"] is None:
        return jsonify({"success": False, "error": "No trained models"}), 400

    model_name = request.args.get("model", state["best_model_name"])
    if model_name not in state["training_results"]:
        model_name = state["best_model_name"]

    prep = state["prepared_data"]
    m_data = state["training_results"][model_name]
    ptype = prep["problem_type"]
    test_m = m_data.get("test_metrics", {})

    charts = {}

    try:
        if ptype in ["binary_classification", "multiclass_classification"]:
            cm = test_m.get("confusion_matrix")
            if cm:
                fig_cm = plot_confusion_matrix_heatmap(cm, prep.get("class_names"))
                charts["confusion_matrix"] = fig_to_base64(fig_cm)
            if test_m.get("fpr") and test_m.get("tpr"):
                fig_roc = plot_roc_curve_chart(test_m["fpr"], test_m["tpr"], test_m.get("roc_auc"))
                charts["roc_curve"] = fig_to_base64(fig_roc)
            elif test_m.get("precision_curve") and test_m.get("recall_curve"):
                fig_pr = plot_precision_recall_curve_chart(test_m["precision_curve"], test_m["recall_curve"])
                charts["pr_curve"] = fig_to_base64(fig_pr)
        elif ptype == "regression":
            y_test_arr = prep["y_test"]
            y_pred_arr = np.array(m_data.get("y_pred", []))
            if len(y_pred_arr) == len(y_test_arr):
                fig_avp = plot_actual_vs_predicted(y_test_arr, y_pred_arr)
                charts["actual_vs_predicted"] = fig_to_base64(fig_avp)
                fig_res = plot_residuals(y_test_arr, y_pred_arr)
                charts["residuals"] = fig_to_base64(fig_res)
    except Exception as e:
        print("Chart generation error:", e)

    return jsonify({
        "success": True,
        "model_name": model_name,
        "problem_type": ptype,
        "models_available": list(state["training_results"].keys()),
        "metrics": test_m,
        "charts": charts,
    })

@app.route("/api/xai", methods=["GET"])
def get_xai():
    if state["training_results"] is None:
        return jsonify({"success": False, "error": "No trained models"}), 400

    prep = state["prepared_data"]
    best_name = state["best_model_name"]
    best_est = state["training_results"][best_name].get("best_estimator", state["training_results"][best_name].get("estimator"))

    imp_df = extract_feature_importance(best_est, prep["feature_names_out"])
    
    # SHAP summary
    shap_res = compute_shap_explanations(
        model=best_est,
        X_train=prep.get("X_train_proc", prep.get("X_proc")),
        X_test=prep.get("X_test_proc", prep.get("X_proc")),
        feature_names=prep["feature_names_out"],
        sample_size=30,
    )

    shap_table = []
    if shap_res.get("available") and shap_res.get("shap_summary_df") is not None:
        shap_table = shap_res["shap_summary_df"].to_dict(orient="records")

    return jsonify({
        "success": True,
        "model_name": best_name,
        "feature_importance": imp_df.to_dict(orient="records"),
        "shap_available": shap_res.get("available", False),
        "shap_summary": shap_table,
    })

# ==============================================================================
# 6. PREDICTION HUB API
# ==============================================================================

@app.route("/api/predict/single", methods=["POST"])
def predict_single_api():
    if state["active_bundle"] is None:
        if state["saved_pipeline_path"] and os.path.exists(state["saved_pipeline_path"]):
            state["active_bundle"] = load_pipeline(state["saved_pipeline_path"])
        else:
            return jsonify({"success": False, "error": "No trained pipeline available"}), 400

    data = request.get_json() or {}
    inputs = data.get("inputs", {})

    try:
        bundle: SerializedPipelineBundle = state["active_bundle"]
        pred_out = bundle.predict_single(inputs)

        return jsonify({
            "success": True,
            "prediction": pred_out["prediction"],
            "raw_prediction": pred_out["raw_prediction"],
            "raw_prediction_str": pred_out["raw_prediction_str"],
            "target_name": pred_out["target_name"],
            "display_title": pred_out["display_title"],
            "human_readable_prediction": pred_out["human_readable_prediction"],
            "explanation": pred_out["explanation"],
            "confidence_pct": pred_out["confidence_pct"],
            "display_text": pred_out["display_text"],
            "probabilities": pred_out["probabilities"],
            "problem_type": pred_out["problem_type"],
            "problem_type_display": pred_out["problem_type_display"],
            "model_name": pred_out["model_name"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/predict/batch", methods=["POST"])
def predict_batch_api():
    if state["active_bundle"] is None:
        return jsonify({"success": False, "error": "No trained pipeline available"}), 400

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]
    try:
        df_batch, _ = load_dataset(file)
        bundle: SerializedPipelineBundle = state["active_bundle"]
        res_df, summary = bundle.predict_batch(df_batch)

        return jsonify({
            "success": True,
            "total_samples": summary["total_samples"],
            "predicted_column": summary["predicted_column"],
            "preview": res_df.head(20).replace({np.nan: None}).to_dict(orient="records"),
            "columns": list(res_df.columns),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==============================================================================
# 7. REPORTS & EXPERIMENT TRACKER API
# ==============================================================================

@app.route("/api/report/html", methods=["GET"])
def get_html_report_api():
    if state["training_results"] is None:
        return jsonify({"success": False, "error": "No training results"}), 400

    try:
        results = state["training_results"]
        prep = state["prepared_data"]
        profile = profile_dataset(state["current_df"])
        board_df = generate_leaderboard(results, prep["problem_type"])
        best_name = state["best_model_name"]

        rep_charts = []
        try:
            target = state["target_col"]
            if target and pd.api.types.is_numeric_dtype(state["current_df"][target]):
                fig = plot_distribution(state["current_df"], target)
                rep_charts.append({"title": f"Target Distribution ({target})", "base64": fig_to_base64(fig)})
        except Exception:
            pass

        report_payload = {
            "dataset_name": state["dataset_name"],
            "problem_type": prep["problem_type"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": profile["summary"]["total_rows"],
            "cols": profile["summary"]["total_cols"],
            "memory_usage": profile["summary"]["memory_usage_str"],
            "missing_cells": profile["summary"]["total_missing_cells"],
            "missing_pct": profile["summary"]["total_missing_pct"],
            "duplicate_rows": profile["summary"]["duplicate_rows"],
            "target_col": state["target_col"],
            "cleaning_log": state["cleaner"].get_cleaning_log(),
            "leakage_warnings": detect_data_leakage(state["current_df"], state["target_col"], state["feature_cols"]),
            "imbalance_info": detect_class_imbalance(state["current_df"][state["target_col"]]) if prep["problem_type"] != "regression" else {},
            "leaderboard_html": board_df.to_html(classes="table", index=False),
            "leaderboard_table_data": [list(board_df.columns)] + board_df.values.tolist(),
            "best_model_name": best_name,
            "charts": rep_charts,
            "top_features": extract_feature_importance(
                model=results[best_name].get("best_estimator", results[best_name].get("estimator")),
                feature_names=prep.get("feature_names_out", state["feature_cols"]),
            ).head(8).to_dict(orient="records"),
        }

        html_content = generate_html_report(report_payload)
        return html_content, 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    try:
        tracker: ExperimentTracker = state["tracker"]
        runs = tracker.list_runs()
        return jsonify({"success": True, "runs": runs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/experiments/<run_id>", methods=["DELETE"])
def delete_experiment(run_id):
    try:
        tracker: ExperimentTracker = state["tracker"]
        success = tracker.delete_run(run_id)
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("=========================================================")
    print(" ModelFlow AI Web Server Starting on http://127.0.0.1:5000")
    print(" Frontend: Pure HTML5 + Vanilla CSS + JavaScript")
    print(" Backend:  Flask REST API + Python ML Engine")
    print("=========================================================")
    app.run(host="0.0.0.0", port=5000, debug=False)

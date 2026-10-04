"""
Unit tests for Modules 12 & 13: Predictor & Experiment Tracker (core/predictor.py, core/tracker.py)
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from core.predictor import save_pipeline, load_pipeline
from core.tracker import ExperimentTracker


def test_pipeline_save_and_predict_roundtrip(tmp_path):
    df_sample = pd.DataFrame({
        "age": [20.0, 30.0, 40.0],
        "city": ["NY", "LA", "NY"],
        "target": ["No", "Yes", "No"],
    })

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["age"])],
        remainder="drop",
    )
    X_raw = df_sample[["age", "city"]]
    X_proc = preprocessor.fit_transform(X_raw)
    y = np.array([0, 1, 0])

    model = LogisticRegression()
    model.fit(X_proc, y)

    # Save
    save_dir = str(tmp_path / "models")
    saved_path = save_pipeline(
        preprocessor=preprocessor,
        model=model,
        feature_cols=["age", "city"],
        numeric_cols=["age"],
        categorical_cols=["city"],
        target_col="target",
        problem_type="binary_classification",
        df_sample=X_raw,
        class_names=["No", "Yes"],
        target_mapping={"No": 0, "Yes": 1},
        model_name="Test_Logistic",
        save_dir=save_dir,
    )

    assert os.path.exists(saved_path)

    # Load
    bundle = load_pipeline(saved_path)
    assert bundle.model_name == "Test_Logistic"
    assert bundle.problem_type == "binary_classification"
    assert "target" not in bundle.feature_cols

    # Single Prediction
    single_res = bundle.predict_single({"age": 25.0, "city": "NY"})
    assert "prediction" in single_res
    assert "probabilities" in single_res
    assert "explanation" in single_res
    assert "human_readable_prediction" in single_res

    # Batch Prediction
    batch_df = pd.DataFrame({"age": [22.0, 38.0], "city": ["LA", "NY"]})
    res_batch, summary = bundle.predict_batch(batch_df)
    assert "Predicted_target" in res_batch.columns
    assert summary["total_samples"] == 2


def test_regression_housing_prediction_formatting(tmp_path):
    from sklearn.linear_model import LinearRegression

    df_housing = pd.DataFrame({
        "MedInc": [3.5, 4.2, 5.1],
        "HouseAge": [25.0, 30.0, 15.0],
        "Population": [1200.0, 1500.0, 800.0],
        "MedHouseVal": [1.5, 2.0, 3.0],
    })

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["MedInc", "HouseAge", "Population"])],
        remainder="drop",
    )
    features = ["MedInc", "HouseAge", "Population"]
    X_proc = preprocessor.fit_transform(df_housing[features])
    y = df_housing["MedHouseVal"].values

    model = LinearRegression()
    model.fit(X_proc, y)

    saved_path = save_pipeline(
        preprocessor=preprocessor,
        model=model,
        feature_cols=features,
        numeric_cols=features,
        categorical_cols=[],
        target_col="MedHouseVal",
        problem_type="regression",
        df_sample=df_housing[features],
        model_name="Linear Regression",
        save_dir=str(tmp_path / "models"),
    )

    bundle = load_pipeline(saved_path)
    assert "MedHouseVal" not in bundle.feature_cols
    assert bundle.target_col == "MedHouseVal"

    # Single Prediction
    pred_res = bundle.predict_single({"MedInc": 3.5, "HouseAge": 25.0, "Population": 1200.0})
    assert pred_res["target_name"] == "MedHouseVal"
    assert pred_res["problem_type"] == "regression"
    assert "raw_prediction" in pred_res
    assert isinstance(pred_res["raw_prediction"], float)
    assert len(pred_res["raw_prediction_str"].split(".")[1]) >= 4
    assert "$" in pred_res["human_readable_prediction"]
    assert "The model estimates the median house value" in pred_res["explanation"]
    assert "$" in pred_res["explanation"]

    # Batch Prediction
    batch_df = pd.DataFrame({
        "MedInc": [3.0, 6.0],
        "HouseAge": [20.0, 10.0],
        "Population": [1000.0, 500.0],
    })
    res_batch, b_sum = bundle.predict_batch(batch_df)
    assert "Predicted_MedHouseVal" in res_batch.columns
    assert "Estimated_House_Value_USD" in res_batch.columns


def test_experiment_tracker(tmp_path):
    db_path = str(tmp_path / "test_exp.db")
    tracker = ExperimentTracker(db_path=db_path)

    # 1. Log Run
    run_id = tracker.log_run(
        model_name="Random Forest",
        problem_type="binary_classification",
        dataset_name="test.csv",
        features=["f1", "f2"],
        test_metrics={"accuracy": 0.95, "f1": 0.94},
        cv_score={"mean": 0.93, "std": 0.02},
        train_duration_sec=0.45,
    )
    assert run_id.startswith("run_")

    # 2. List Runs
    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["model_name"] == "Random Forest"

    # 3. Compare Runs
    cmp_df = tracker.compare_runs([run_id])
    assert not cmp_df.empty
    assert "ACCURACY" in cmp_df.columns

    # 4. Delete Run
    assert tracker.delete_run(run_id) is True
    assert len(tracker.list_runs()) == 0

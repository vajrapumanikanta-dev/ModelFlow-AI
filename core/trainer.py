"""
Module 8 & 9: Trainer, Cross-Validation & Hyperparameter Tuning (core/trainer.py)
Orchestrates training of multiple models, cross-validation, hyperparameter tuning,
unsupervised clustering, and dimensionality reduction.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from core.evaluator import (
    evaluate_classification,
    evaluate_regression,
    evaluate_clustering,
)
from core.models import get_pca_model


def train_and_evaluate_models(
    models_dict: Dict[str, Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    cv_folds: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Trains multiple models, performs K-Fold Cross-Validation, and evaluates on test set.
    """
    results: Dict[str, Dict[str, Any]] = {}
    is_classification = problem_type in ["binary_classification", "multiclass_classification"]
    is_regression = problem_type == "regression"

    cv_splitter = None
    if cv_folds > 1:
        if is_classification:
            # Check if class counts support requested cv folds
            min_class = pd.Series(y_train).value_counts().min()
            actual_folds = min(cv_folds, max(2, min_class))
            cv_splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        elif is_regression:
            cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scoring_metric = "f1_weighted" if is_classification else "r2"

    for model_name, config in models_dict.items():
        estimator = config["estimator"]
        start_time = time.time()

        # 1. Cross-Validation on Training set
        cv_res = {"mean": 0.0, "std": 0.0, "scores": []}
        if cv_splitter is not None:
            try:
                scores = cross_val_score(estimator, X_train, y_train, cv=cv_splitter, scoring=scoring_metric)
                cv_res = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "scores": scores.tolist(),
                }
            except Exception:
                cv_res = {"mean": 0.0, "std": 0.0, "scores": []}

        # 2. Fit model on full training set
        estimator.fit(X_train, y_train)
        duration = time.time() - start_time

        # 3. Predict on Test Set
        y_pred = estimator.predict(X_test)
        y_prob = None
        if is_classification and hasattr(estimator, "predict_proba"):
            try:
                y_prob = estimator.predict_proba(X_test)
            except Exception:
                y_prob = None

        # 4. Compute Test Metrics
        if is_classification:
            test_metrics = evaluate_classification(y_test, y_pred, y_prob=y_prob, class_names=class_names)
        else:
            test_metrics = evaluate_regression(y_test, y_pred, n_features=X_train.shape[1])

        results[model_name] = {
            "model_name": model_name,
            "key": config.get("key", model_name.lower().replace(" ", "_")),
            "estimator": estimator,
            "supports_proba": config.get("supports_proba", False),
            "supports_importance": config.get("supports_importance", False),
            "train_duration_sec": duration,
            "cv_score": cv_res,
            "test_metrics": test_metrics,
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist() if y_prob is not None else None,
        }

    return results


def tune_hyperparameters(
    model_name: str,
    estimator: Any,
    param_grid: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    search_type: str = "grid",
    n_iter: int = 10,
    cv_folds: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes GridSearchCV or RandomizedSearchCV for a specified model.
    """
    is_classification = problem_type in ["binary_classification", "multiclass_classification"]
    scoring = "f1_weighted" if is_classification else "r2"

    if is_classification:
        min_class = pd.Series(y_train).value_counts().min()
        actual_folds = min(cv_folds, max(2, min_class))
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    start_time = time.time()
    if search_type == "random":
        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=42,
        )
    else:
        search = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
        )

    search.fit(X_train, y_train)
    duration = time.time() - start_time
    best_estimator = search.best_estimator_

    # Evaluate tuned model on test set
    y_pred = best_estimator.predict(X_test)
    y_prob = None
    if is_classification and hasattr(best_estimator, "predict_proba"):
        try:
            y_prob = best_estimator.predict_proba(X_test)
        except Exception:
            pass

    if is_classification:
        test_metrics = evaluate_classification(y_test, y_pred, y_prob=y_prob, class_names=class_names)
    else:
        test_metrics = evaluate_regression(y_test, y_pred, n_features=X_train.shape[1])

    return {
        "model_name": f"{model_name} (Tuned)",
        "best_estimator": best_estimator,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "train_duration_sec": duration,
        "test_metrics": test_metrics,
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist() if y_prob is not None else None,
    }


def compute_clustering_elbow_and_silhouette(
    X: np.ndarray, k_range: range = range(2, 9)
) -> Dict[str, Any]:
    """
    Calculates K-Means inertia (elbow method) and silhouette scores across k range.
    """
    inertias = []
    sil_scores = []
    ks = list(k_range)

    for k in ks:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertias.append(float(km.inertia_))
        try:
            sil = float(silhouette_score(X, km.labels_))
            sil_scores.append(sil)
        except Exception:
            sil_scores.append(0.0)

    # Determine recommended k (highest silhouette)
    best_k = ks[int(np.argmax(sil_scores))] if sil_scores else 3

    return {
        "k_range": ks[: len(inertias)],
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "recommended_k": best_k,
    }


def train_clustering_models(
    models_dict: Dict[str, Dict[str, Any]],
    X: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Fits unsupervised clustering algorithms (KMeans, DBSCAN, Hierarchical).
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name, config in models_dict.items():
        est = config["estimator"]
        start_time = time.time()
        labels = est.fit_predict(X)
        duration = time.time() - start_time
        metrics = evaluate_clustering(X, labels)

        results[name] = {
            "model_name": name,
            "estimator": est,
            "labels": labels.tolist(),
            "test_metrics": metrics,
            "train_duration_sec": duration,
        }
    return results


def run_pca_decomposition(X: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
    """
    Performs PCA dimensionality reduction and computes explained variance ratios.
    """
    n_comp = min(n_components, X.shape[1], X.shape[0])
    pca = get_pca_model(n_components=n_comp)
    X_pca = pca.fit_transform(X)

    return {
        "pca_estimator": pca,
        "X_pca": X_pca,
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def generate_standalone_code(
    dataset_name: str,
    target_col: Optional[str],
    feature_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    problem_type: str,
    model_name: str,
    model_obj: Any,
    best_params: Optional[Dict[str, Any]] = None,
    cleaning_log: Optional[List[Dict[str, Any]]] = None,
    test_size: float = 0.2,
    num_scaler: str = "standard",
    num_imputer: str = "median",
    cat_encoder: str = "onehot",
    cat_imputer: str = "most_frequent",
    resampling: str = "none",
) -> str:
    """
    Generates a complete, standalone, production-ready Python script
    implementing every pipeline step for the best-performing model.
    """
    model_class_name = type(model_obj).__name__
    model_module = type(model_obj).__module__

    # Format parameters
    params = best_params if best_params else getattr(model_obj, "get_params", lambda: {})()
    # Filter only relevant non-default / simple params
    filtered_params = {k: v for k, v in params.items() if not k.startswith("_") and not callable(v) and k in ["n_estimators", "max_depth", "min_samples_split", "C", "alpha", "kernel", "learning_rate", "n_neighbors", "var_smoothing", "criterion"]}
    params_str = ", ".join([f"{k}={repr(v)}" for k, v in filtered_params.items()])
    if "random_state" not in filtered_params and hasattr(model_obj, "random_state"):
        params_str += ", random_state=42" if params_str else "random_state=42"

    scaler_cls = {
        "standard": "StandardScaler()",
        "minmax": "MinMaxScaler()",
        "robust": "RobustScaler()",
        "none": "'passthrough'",
    }.get(num_scaler, "StandardScaler()")

    encoder_cls = (
        "OneHotEncoder(handle_unknown='ignore', sparse_output=False)"
        if cat_encoder == "onehot"
        else "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)"
    )

    is_classif = problem_type in ["binary_classification", "multiclass_classification"]

    code = f'''"""
================================================================================
ModelFlow AI — Production Training Pipeline
Auto-generated standalone Python script for: {model_name}
Dataset: {dataset_name} | Problem Type: {problem_type}
================================================================================
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from {model_module} import {model_class_name}
'''

    if is_classif:
        code += 'from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score\n'
    else:
        code += 'from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n'

    if resampling != "none":
        code += 'from imblearn.over_sampling import SMOTE, RandomOverSampler\nfrom imblearn.under_sampling import RandomUnderSampler\n'

    code += f'''

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Data Cleaning (Applied deterministically based on ModelFlow AI audit log)
    """
    df_clean = df.copy()
'''

    if cleaning_log:
        for step in cleaning_log:
            act = step.get("action")
            cols = step.get("columns", [])
            prms = step.get("params", {})
            if act == "remove_duplicates":
                code += "    # Remove duplicates\n    df_clean = df_clean.drop_duplicates().reset_index(drop=True)\n"
            elif act == "handle_missing":
                strat = prms.get("strategy", "mean")
                if strat in ["mean", "median"]:
                    for c in cols:
                        code += f"    df_clean['{c}'] = df_clean['{c}'].fillna(df_clean['{c}'].{strat}())\n"
                elif strat == "mode":
                    for c in cols:
                        code += f"    df_clean['{c}'] = df_clean['{c}'].fillna(df_clean['{c}'].mode()[0])\n"
                elif strat == "drop_rows":
                    code += f"    df_clean = df_clean.dropna(subset={cols}).reset_index(drop=True)\n"
            elif act == "drop_columns":
                code += f"    # Drop unneeded columns\n    df_clean = df_clean.drop(columns={cols}, errors='ignore')\n"
            elif act == "handle_outliers":
                for c in cols:
                    code += f"    # Outlier capping on {c}\n"
                    code += f"    q25, q75 = df_clean['{c}'].quantile(0.25), df_clean['{c}'].quantile(0.75)\n"
                    code += f"    iqr = q75 - q25\n"
                    code += f"    df_clean['{c}'] = df_clean['{c}'].clip(lower=q25 - 1.5*iqr, upper=q75 + 1.5*iqr)\n"
    else:
        code += "    # No special cleaning steps required\n    pass\n"

    code += f'''    return df_clean


def build_pipeline() -> Pipeline:
    """
    Step 2: Preprocessing Pipeline Construction (Leakage-Free ColumnTransformer)
    """
    numeric_features = {repr(numeric_cols)}
    categorical_features = {repr(categorical_cols)}

    # Numeric sub-pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='{num_imputer}')),
        ('scaler', {scaler_cls})
    ])

    # Categorical sub-pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='{cat_imputer}')),
        ('encoder', {encoder_cls})
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Instantiate Model with Optimal Hyperparameters
    model = {model_class_name}({params_str})

    # Assemble End-to-End Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def main():
    print("=" * 70)
    print("Executing ModelFlow AI Standalone Training Pipeline")
    print("=" * 70)

    # 1. Load Dataset
    data_path = "{dataset_name}"
    print(f"\\n1. Loading dataset from '{{data_path}}'...")
    df = pd.read_csv(data_path)
    print(f"   Raw shape: {{df.shape[0]}} rows, {{df.shape[1]}} columns")

    # 2. Clean Data
    print("\\n2. Applying data cleaning steps...")
    df_clean = clean_dataset(df)

    # 3. Features and Target
    target_col = "{target_col if target_col else ''}"
    feature_cols = {repr(feature_cols)}

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # 4. Train/Test Split (Strictly before preprocessor fitting)
    print(f"\\n3. Splitting dataset (Test size: {test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size={test_size}, random_state=42{', stratify=y' if is_classif else ''}
    )
    print(f"   Train samples: {{len(X_train)}}, Test samples: {{len(X_test)}}")

    # 5. Build and Fit Full Pipeline
    print(f"\\n4. Fitting end-to-end pipeline ({model_name})...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("   ✓ Pipeline fitting completed successfully.")

    # 6. Evaluate Model on Test Set
    print("\\n5. Evaluating model performance on unseen test data...")
    y_pred = pipeline.predict(X_test)

'''

    if is_classif:
        code += '''    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Weighted F1-Score: {f1:.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
'''
    else:
        code += '''    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE:     {rmse:.4f}")
    print(f"   MAE:      {mae:.4f}")
'''

    code += f'''
    # 7. Save Pipeline Package
    export_path = "best_{model_name.lower().replace(' ', '_')}_pipeline.joblib"
    joblib.dump(pipeline, export_path)
    print(f"\\n6. Saved complete serialized pipeline to '{{export_path}}'")

    # 8. Demonstration: Load and Predict on Sample
    print("\\n7. Testing inference roundtrip...")
    loaded_pipe = joblib.load(export_path)
    sample_input = X_test.head(2)
    sample_preds = loaded_pipe.predict(sample_input)
    print(f"   Sample Inputs:\\n{{sample_input}}")
    print(f"   Predictions: {{sample_preds}}")
    print("\\n" + "=" * 70)
    print("Pipeline Execution Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''
    return code


"""
Module 10: Evaluator & Leaderboard (core/evaluator.py)
Computes comprehensive performance metrics and constructs model leaderboards.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes classification metrics including Accuracy, Precision, Recall, F1, ROC-AUC,
    Confusion Matrix, and ROC/PR curve data.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    roc_auc_val = None
    fpr_list, tpr_list = [], []
    precision_curve_list, recall_curve_list = [], []

    if y_prob is not None:
        try:
            num_classes = len(np.unique(y_true))
            if num_classes == 2:
                # Binary classification
                proba_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob.ravel()
                roc_auc_val = float(roc_auc_score(y_true, proba_pos))
                fpr, tpr, _ = roc_curve(y_true, proba_pos)
                fpr_list = fpr.tolist()
                tpr_list = tpr.tolist()
                p_c, r_c, _ = precision_recall_curve(y_true, proba_pos)
                precision_curve_list = p_c.tolist()
                recall_curve_list = r_c.tolist()
            elif num_classes > 2:
                # Multiclass classification
                roc_auc_val = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc_val = None

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "roc_auc": round(roc_auc_val, 4) if roc_auc_val is not None else None,
        "confusion_matrix": cm,
        "classification_report": report,
        "fpr": fpr_list,
        "tpr": tpr_list,
        "precision_curve": precision_curve_list,
        "recall_curve": recall_curve_list,
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes regression metrics: MAE, MSE, RMSE, R², Adjusted R², and Residuals.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    adj_r2 = r2
    if n_features is not None and n > (n_features + 1):
        adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))

    residuals = (y_true - y_pred).tolist()

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "adjusted_r2": round(adj_r2, 4),
        "residuals": residuals[:200],  # Sample for serialization
    }


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Computes clustering quality metrics: Silhouette Score, Calinski-Harabasz, Davies-Bouldin.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    # Ignore noise cluster (-1) in DBSCAN for silhouette calculation if present
    non_noise_mask = labels != -1
    valid_labels = labels[non_noise_mask]
    valid_X = X[non_noise_mask] if isinstance(X, np.ndarray) else np.asarray(X)[non_noise_mask]

    n_clusters = len(np.unique(valid_labels))

    if n_clusters < 2 or len(valid_labels) < (n_clusters + 1):
        return {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            "n_clusters": int(n_clusters),
            "cluster_sizes": {str(k): int((labels == k).sum()) for k in unique_labels},
        }

    try:
        sil = float(silhouette_score(valid_X, valid_labels))
    except Exception:
        sil = None

    try:
        ch = float(calinski_harabasz_score(valid_X, valid_labels))
    except Exception:
        ch = None

    try:
        db = float(davies_bouldin_score(valid_X, valid_labels))
    except Exception:
        db = None

    return {
        "silhouette": round(sil, 4) if sil is not None else None,
        "calinski_harabasz": round(ch, 2) if ch is not None else None,
        "davies_bouldin": round(db, 4) if db is not None else None,
        "n_clusters": int(n_clusters),
        "cluster_sizes": {str(k): int((labels == k).sum()) for k in unique_labels},
    }


def generate_leaderboard(
    training_results: Dict[str, Dict[str, Any]],
    problem_type: str,
) -> pd.DataFrame:
    """
    Generates a structured, ranked Leaderboard DataFrame from model training outputs.
    """
    rows = []
    is_classification = problem_type in ["binary_classification", "multiclass_classification"]
    is_regression = problem_type == "regression"

    for model_name, res in training_results.items():
        test_metrics = res.get("test_metrics", {})
        cv_score = res.get("cv_score", {})
        duration = res.get("train_duration_sec", 0.0)

        row = {
            "Model": model_name,
            "CV Score (Mean ± Std)": f"{cv_score.get('mean', 0.0):.4f} ± {cv_score.get('std', 0.0):.4f}" if cv_score else "-",
            "Train Time (s)": round(duration, 3),
        }

        if is_classification:
            row["Accuracy"] = test_metrics.get("accuracy", 0.0)
            row["F1 (Weighted)"] = test_metrics.get("f1_weighted", 0.0)
            row["Precision (Weighted)"] = test_metrics.get("precision_weighted", 0.0)
            row["Recall (Weighted)"] = test_metrics.get("recall_weighted", 0.0)
            row["ROC-AUC"] = test_metrics.get("roc_auc") if test_metrics.get("roc_auc") is not None else "-"
        elif is_regression:
            row["R² Score"] = test_metrics.get("r2", 0.0)
            row["RMSE"] = test_metrics.get("rmse", 0.0)
            row["MAE"] = test_metrics.get("mae", 0.0)
            row["Adj R²"] = test_metrics.get("adjusted_r2", 0.0)
        else:
            # Clustering
            row["Silhouette"] = test_metrics.get("silhouette", "-")
            row["Calinski-Harabasz"] = test_metrics.get("calinski_harabasz", "-")
            row["Davies-Bouldin"] = test_metrics.get("davies_bouldin", "-")

        rows.append(row)

    df_board = pd.DataFrame(rows)
    if df_board.empty:
        return df_board

    # Rank leaderboard based on primary metric
    if is_classification:
        sort_col = "F1 (Weighted)" if "F1 (Weighted)" in df_board.columns else "Accuracy"
        df_board = df_board.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    elif is_regression:
        sort_col = "R² Score" if "R² Score" in df_board.columns else "RMSE"
        ascending = False if sort_col == "R² Score" else True
        df_board = df_board.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)

    df_board.insert(0, "Rank", [f"#{i+1}" for i in range(len(df_board))])
    return df_board

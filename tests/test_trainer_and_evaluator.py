"""
Unit tests for Modules 8, 9, and 10: Trainer, Tuning, and Evaluator (core/trainer.py, core/evaluator.py)
"""

import numpy as np
import pandas as pd
from core.models import get_classification_models, get_regression_models, get_clustering_models
from core.trainer import (
    train_and_evaluate_models,
    tune_hyperparameters,
    train_clustering_models,
    run_pca_decomposition,
)
from core.evaluator import (
    evaluate_classification,
    evaluate_regression,
    evaluate_clustering,
    generate_leaderboard,
)


def test_classification_training_and_evaluation():
    np.random.seed(42)
    X_train = np.random.randn(60, 4)
    y_train = np.random.choice([0, 1], size=60)
    X_test = np.random.randn(20, 4)
    y_test = np.random.choice([0, 1], size=20)

    models_reg = get_classification_models()
    selected = {
        "Logistic Regression": models_reg["Logistic Regression"],
        "Decision Tree": models_reg["Decision Tree"],
    }

    results = train_and_evaluate_models(
        selected, X_train, y_train, X_test, y_test, problem_type="binary_classification", cv_folds=3
    )

    assert "Logistic Regression" in results
    assert "Decision Tree" in results
    assert "test_metrics" in results["Logistic Regression"]
    assert "accuracy" in results["Logistic Regression"]["test_metrics"]
    assert results["Logistic Regression"]["cv_score"]["mean"] >= 0.0

    board = generate_leaderboard(results, "binary_classification")
    assert len(board) == 2
    assert "F1 (Weighted)" in board.columns


def test_regression_training_and_tuning():
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] * 0.5 + np.random.randn(50) * 0.1
    X_test = np.random.randn(15, 3)
    y_test = X_test[:, 0] * 2 + X_test[:, 1] * 0.5

    models_reg = get_regression_models()
    selected = {"Linear Regression": models_reg["Linear Regression"], "Ridge": models_reg["Ridge Regression"]}

    results = train_and_evaluate_models(
        selected, X_train, y_train, X_test, y_test, problem_type="regression", cv_folds=3
    )
    assert "Linear Regression" in results
    assert "r2" in results["Linear Regression"]["test_metrics"]

    # Test tuning
    tune_res = tune_hyperparameters(
        model_name="Ridge",
        estimator=models_reg["Ridge Regression"]["estimator"],
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        problem_type="regression",
        cv_folds=3,
    )
    assert "best_params" in tune_res
    assert "alpha" in tune_res["best_params"]


def test_clustering_and_pca():
    np.random.seed(42)
    X = np.random.randn(40, 4)

    # 1. Clustering
    clust_models = get_clustering_models(n_clusters=3)
    c_res = train_clustering_models(clust_models, X)
    assert "K-Means" in c_res
    assert len(c_res["K-Means"]["labels"]) == 40

    # 2. PCA
    pca_res = run_pca_decomposition(X, n_components=2)
    assert pca_res["X_pca"].shape == (40, 2)
    assert len(pca_res["explained_variance_ratio"]) == 2

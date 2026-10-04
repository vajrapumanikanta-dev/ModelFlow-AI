"""
Module 8: Model Registry (core/models.py)
Provides model factories, default configurations, and hyperparameter grids
for Classification, Regression, Clustering, and Dimensionality Reduction.
"""

from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA


def get_classification_models() -> Dict[str, Dict[str, Any]]:
    """Returns classification model registry with default estimators and tuning grids."""
    return {
        "Logistic Regression": {
            "key": "logistic_regression",
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "supports_proba": True,
            "supports_importance": True,
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
            },
        },
        "Decision Tree": {
            "key": "decision_tree",
            "estimator": DecisionTreeClassifier(random_state=42),
            "supports_proba": True,
            "supports_importance": True,
            "param_grid": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        },
        "Random Forest": {
            "key": "random_forest",
            "estimator": RandomForestClassifier(n_estimators=100, random_state=42),
            "supports_proba": True,
            "supports_importance": True,
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            },
        },
        "KNN": {
            "key": "knn",
            "estimator": KNeighborsClassifier(),
            "supports_proba": True,
            "supports_importance": False,
            "param_grid": {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
        "Naive Bayes": {
            "key": "naive_bayes",
            "estimator": GaussianNB(),
            "supports_proba": True,
            "supports_importance": False,
            "param_grid": {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            },
        },
        "SVM": {
            "key": "svm",
            "estimator": SVC(probability=True, random_state=42),
            "supports_proba": True,
            "supports_importance": False,
            "param_grid": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            },
        },
        "Gradient Boosting": {
            "key": "gradient_boosting",
            "estimator": GradientBoostingClassifier(random_state=42),
            "supports_proba": True,
            "supports_importance": True,
            "param_grid": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5],
            },
        },
    }


def get_regression_models() -> Dict[str, Dict[str, Any]]:
    """Returns regression model registry with default estimators and tuning grids."""
    return {
        "Linear Regression": {
            "key": "linear_regression",
            "estimator": LinearRegression(),
            "supports_proba": False,
            "supports_importance": True,
            "param_grid": {
                "fit_intercept": [True, False],
            },
        },
        "Ridge Regression": {
            "key": "ridge",
            "estimator": Ridge(random_state=42),
            "supports_proba": False,
            "supports_importance": True,
            "param_grid": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["auto", "svd", "cholesky"],
            },
        },
        "Lasso Regression": {
            "key": "lasso",
            "estimator": Lasso(random_state=42),
            "supports_proba": False,
            "supports_importance": True,
            "param_grid": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            },
        },
        "Decision Tree Regressor": {
            "key": "decision_tree_regressor",
            "estimator": DecisionTreeRegressor(random_state=42),
            "supports_proba": False,
            "supports_importance": True,
            "param_grid": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
            },
        },
        "Random Forest Regressor": {
            "key": "random_forest_regressor",
            "estimator": RandomForestRegressor(n_estimators=100, random_state=42),
            "supports_proba": False,
            "supports_importance": True,
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            },
        },
        "SVR": {
            "key": "svr",
            "estimator": SVR(),
            "supports_proba": False,
            "supports_importance": False,
            "param_grid": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
                "epsilon": [0.01, 0.1, 0.2],
            },
        },
        "Gradient Boosting Regressor": {
            "key": "gradient_boosting_regressor",
            "estimator": GradientBoostingRegressor(random_state=42),
            "supports_proba": False,
            "supports_importance": True,
            "param_grid": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5],
            },
        },
    }


def get_clustering_models(n_clusters: int = 3) -> Dict[str, Dict[str, Any]]:
    """Returns unsupervised clustering models."""
    return {
        "K-Means": {
            "key": "kmeans",
            "estimator": KMeans(n_clusters=n_clusters, random_state=42, n_init="auto"),
        },
        "DBSCAN": {
            "key": "dbscan",
            "estimator": DBSCAN(eps=0.5, min_samples=5),
        },
        "Hierarchical": {
            "key": "hierarchical",
            "estimator": AgglomerativeClustering(n_clusters=n_clusters),
        },
    }


def get_pca_model(n_components: int = 2) -> PCA:
    """Returns PCA dimensionality reduction estimator."""
    return PCA(n_components=n_components, random_state=42)
